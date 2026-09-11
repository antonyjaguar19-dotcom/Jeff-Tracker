"""JeffTracker = LocoTrack + cross-track attention, initialised to be LocoTrack exactly.

LocoTrack's refinement transformer is per-track by construction: locotrack_model.py:891
folds the track axis into the batch before calling the mixer. This module rebuilds that
mixer so the vendor's temporal blocks stay exactly as they are and a cross-track block is
interleaved between them, then subclasses LocoTrack to use it.

Two properties are load-bearing and both are checked by ../../check_identity.py:

  1. The vendor's parameters keep their names. `input_proj`, `transformer.*` and
     `output_proj` are the SAME module objects, re-parented, so a LocoTrack checkpoint
     loads into JeffTracker with the new `cross.*` keys as the only additions -- not a rename
     in sight, and no weight is transformed on the way in.
  2. With the cross blocks zero-initialised the model's output is bit-for-bit identical to
     LocoTrack's. Not close: identical. That is what makes a baseline-vs-treatment
     measurement on identical footage possible from one binary, the property
     the same property check_identity.py enforces exists to preserve on the TAPNext path.

Scope limit worth knowing before reading any result: LocoTrack chunks queries
(`query_chunk_size`, default 64), so cross-track attention sees the tracks within one
chunk, not every track in the shot. CoTracker attends across all tracks in its window.
Raise query_chunk_size to widen it, at the usual VRAM cost.
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn

from jefftrack.paths import add_vendor_to_path

add_vendor_to_path()

from models.locotrack_model import LocoTrack, PIPSTransformer  # noqa: E402

from jefftrack.model.cross_track import CrossTrackAttention  # noqa: E402


class CrossTrackPIPSTransformer(nn.Module):
    """The vendor's PIPSTransformer with cross-track blocks interleaved.

    Input is (B*N, F, C) -- the shape refine_pips hands over -- so the module has to be
    told the batch size to recover N. It defaults to 1 and JeffTracker.forward sets it.
    """

    def __init__(self, base: PIPSTransformer, batch_size: int = 1,
                 cross_layers: Optional[Sequence[int]] = None,
                 num_proxies: int = 16, num_heads: int = 4, zero_init: bool = True,
                 carry_state: bool = False):
        super().__init__()
        # Re-parent, do not rebuild: these are the vendor's own modules, so their parameter
        # names under torch_pips_mixer.* are unchanged and the published checkpoint fits.
        self.input_proj = base.input_proj
        self.transformer = base.transformer
        self.output_proj = base.output_proj
        self.dim = base.dim
        self.batch_size = batch_size

        n_layers = len(self.transformer.layers)
        if cross_layers is None:
            cross_layers = tuple(range(n_layers))
        self.cross_layers = tuple(int(i) for i in cross_layers)
        for i in self.cross_layers:
            if not 0 <= i < n_layers:
                raise ValueError(
                    "cross layer {} outside the mixer's {} layers".format(i, n_layers))
        self.carry_state = bool(carry_state)
        self.cross = nn.ModuleDict({
            str(i): CrossTrackAttention(self.dim, num_heads=num_heads,
                                        num_proxies=num_proxies, zero_init=zero_init,
                                        carry_state=self.carry_state)
            for i in self.cross_layers
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        bn, f, d = x.shape
        b = max(1, int(self.batch_size))
        if bn % b:
            raise ValueError("mixer got {} rows, not divisible by batch {}".format(bn, b))
        n = bn // b

        # The vendor's Transformer.forward, unrolled so a block can be inserted between
        # layers. Its dropout_rate is 0, so the dropout calls are omitted rather than
        # reproduced -- F.dropout(p=0) is the identity and leaving it out keeps this
        # readable. Everything else is line for line.
        #
        # With carry_state the proxy tokens are put through the vendor's temporal block as
        # well, so a proxy arriving at the next cross block has seen the same temporal
        # attention the tracks have and carries the group's motion over time rather than a
        # snapshot of one frame.
        #
        # They go through as a SEPARATE call on the same layer, not as extra rows appended
        # to the real tracks. Mathematically the two are the same thing -- the block
        # attends over the frame axis within each row and never across rows -- but they are
        # not the same in floating point: adding rows changes the batch size handed to
        # `scaled_dot_product_attention`, which changes its kernel and its reduction order.
        # Measured: appending the rows moved the mixer output by 1.9e-4, which the four
        # refinement iterations amplified into **0.51 px** on `tracks`, and check_identity
        # failed. Cost of keeping them apart is one extra attention call over k rows per
        # layer, against k=16 proxies and n up to 600 tracks.
        state = None
        if self.carry_state and self.cross:
            first = self.cross[str(self.cross_layers[0])]
            state = first.init_state(b, f, x.device, x.dtype)

        def time_block(layer, t):
            t_norm = layer["layer_norm1"](t)
            t = t + layer["attn"](t_norm, t_norm, t_norm, mask=None)
            t_norm = layer["layer_norm2"](t)
            return t + layer["dense"](t_norm)

        h = x
        for i, layer in enumerate(self.transformer.layers):
            h = time_block(layer, h)
            if state is not None:
                k = state.shape[1]
                state = time_block(
                    layer, state.reshape(b * k, f, d)).reshape(b, k, f, d)
            # nn.ModuleDict has no .get(); membership is the supported test.
            if str(i) in self.cross:
                if state is None:
                    h = self.cross[str(i)](h.reshape(b, n, f, d)).reshape(bn, f, d)
                else:
                    hh, state = self.cross[str(i)](h.reshape(b, n, f, d), state)
                    h = hh.reshape(bn, f, d)
        h = self.transformer.ln_out(h)
        return self.output_proj(h)


class JeffTracker(LocoTrack):
    """LocoTrack whose refinement mixer can see across tracks."""

    def __init__(self, *args, cross_layers: Optional[Sequence[int]] = None,
                 num_proxies: int = 16, cross_heads: int = 4, zero_init: bool = True,
                 carry_state: bool = False, pyramid_level: int = 0, **kwargs):
        super().__init__(*args, pyramid_level=pyramid_level, **kwargs)
        # NOTE the model is built here at the VENDOR's width, three correlation levels, even
        # when a fourth is asked for. Widening is `_add_pyramid_levels`, and load_jefftrack
        # calls it AFTER the checkpoint has landed -- see the note there. Doing it in
        # __init__ splices a random input_proj and copies a random CMDTop, and since the
        # widened tensors are then skipped by the loader as shape-mismatched, the model
        # stays part-random and every number off it is fiction. That is what happened on
        # the first attempt and the identity check is what caught it: 202 px.
        self.extra_pyramid = int(pyramid_level)
        self.torch_pips_mixer = CrossTrackPIPSTransformer(
            self.torch_pips_mixer, cross_layers=cross_layers, num_proxies=num_proxies,
            num_heads=cross_heads, zero_init=zero_init, carry_state=carry_state)

    def _add_pyramid_levels(self, extra: int, zero_init: bool = True) -> None:
        """Give the correlation pyramid `extra` more levels. CoTracker3 runs four.

        The vendor already accepts `pyramid_level` and already builds the extra grids --
        an `avg_pool3d` of the level below, re-using the same query and support features
        (locotrack_model.py:753-768). It just cannot RUN with it: `self.cmdtop` is built as
        `range(3)`, so a fourth level indexes past the end of the list, and the mixer's
        `input_proj` is sized for three levels' worth of correlation. Both are fixed here.

        Two decisions worth stating.

        **The new CMDTop is a copy of the coarsest existing one, not a fresh init.** The
        level it processes is that level avg-pooled, so its input statistics are the closest
        thing in the model to what the copy was trained on. A random init would have to
        relearn from nothing while attached to a network that is already converged.

        **The new correlation channels enter the mixer with ZERO weight.** They are not
        appended: `mlp_input` is `[occ(1), expd(1), corrs(levels x C), rel_pos(84)]`, so a
        fourth level's channels land in the MIDDLE, and the existing `rel_pos` weights have
        to keep the columns they were trained on. The block is spliced in at the right
        offset with a zero-filled weight, which makes the widened model bit-for-bit the old
        one until training moves it -- the same property `zero_init` gives the cross blocks,
        and for the same reason: without it a widened model is part-random and every number
        measured from it is a mixture of "the fourth level helped" and "the base was
        disturbed".
        """
        import copy  # noqa: PLC0415  (only needed on this path)

        for _ in range(extra):
            self.cmdtop.append(copy.deepcopy(self.cmdtop[-1]))

        old = self.torch_pips_mixer.input_proj
        in_old = old.in_features
        # [occ, expd] + corrs + rel_pos(84). Three levels ship, so one level is this wide.
        per_level = (in_old - 2 - 84) // 3
        cut = 2 + per_level * 3            # first channel after the existing corrs
        new = nn.Linear(in_old + per_level * extra, old.out_features)
        with torch.no_grad():
            w = torch.zeros_like(new.weight)
            w[:, :cut] = old.weight[:, :cut]
            w[:, cut + per_level * extra:] = old.weight[:, cut:]
            if zero_init:
                pass                       # the spliced block stays zero
            else:
                nn.init.normal_(w[:, cut:cut + per_level * extra], std=0.002)
            new.weight.copy_(w)
            new.bias.copy_(old.bias)
        self.torch_pips_mixer.input_proj = new

    def forward(self, video: torch.Tensor, *args, **kwargs):
        # refine_pips folds the track axis into the batch; the mixer needs the batch size
        # back to undo that. video is [batch, time, height, width, 3].
        self.torch_pips_mixer.batch_size = int(video.shape[0])
        return super().forward(video, *args, **kwargs)

    def refine_pips(self, target_feature, support_feature, frame_features, pyramid,
                    *args, **kwargs):
        """Pad the SUPPORT features to the pyramid's depth before the vendor zips them.

        This is the other half of the vendor's unusable `pyramid_level`. Its
        estimate_trajectories extends `queries` and `pyramid` for each extra level
        (locotrack_model.py:753, :760) but never `supports` -- and refine_pips consumes the
        three with `zip`, which stops at the shortest. So an extra level was silently
        dropped: the model was built expecting four levels' worth of correlation channels
        and handed three, and the failure surfaced as a shape mismatch in the mixer rather
        than anywhere near the cause.

        Padded the same way the vendor pads `queries`: repeat the last level. The extra
        grid IS that level average-pooled, so its support patch is the right one to reuse.
        """
        if getattr(self, "extra_pyramid", 0) and len(support_feature) < len(pyramid):
            support_feature = list(support_feature) + \
                [support_feature[-1]] * (len(pyramid) - len(support_feature))
        return super().refine_pips(target_feature, support_feature, frame_features,
                                   pyramid, *args, **kwargs)

    def cross_parameters(self):
        """Just the new blocks -- what a fine-tune trains while the rest stays frozen."""
        return self.torch_pips_mixer.cross.parameters()

    def freeze_base(self, unfreeze_mixer: bool = False) -> None:
        """Everything except the cross-track blocks stops learning.

        On a 16 GB A4000 this is not a preference, it is the difference between a
        fine-tune that fits and one that does not.

        `unfreeze_mixer` additionally releases the vendor's TEMPORAL transformer -- the
        part that moves a track through time -- while the image encoder stays frozen.

        Measured reason for the option. Cross-track attention runs per frame
        (cross_track.py reshapes to (B*F, N, D)); it has no temporal path of its own, so
        anything it infers from a point's neighbours can only be carried across an
        occlusion by the vendor's temporal blocks -- which are frozen, and which were
        trained under an objective that zeroed the position loss on occluded frames. The
        block is not short of information: raising query_chunk_size from 64 to 256, four
        times the neighbours, moved nothing past the third decimal at either resolution.
        It is short of authority.

        The cost is real and is why this is a flag rather than the default: with the
        vendor's parameters moving, check_identity.py no longer separates "cross-track
        attention helped" from "the base drifted", and a later result is a mixture of the
        two. Keep a frozen run alongside if that attribution is wanted.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.cross_parameters():
            p.requires_grad_(True)
        # The EXTRA correlation levels' processors train with them. Not a second knob: a
        # pyramid level and the CMDTop that reads it are one unit, and the three shipped
        # levels stay frozen, so the control is still "no fourth level" against "a fourth
        # level". Leaving it frozen would ask the mixer to extract signal from a processor
        # tuned for the level ABOVE, applied to a pooled grid it never saw -- a null result
        # there would say nothing about whether a fourth level helps.
        for i in range(3, len(self.cmdtop)):
            for p in self.cmdtop[i].parameters():
                p.requires_grad_(True)
        if unfreeze_mixer:
            mixer = self.torch_pips_mixer
            for module in (mixer.input_proj, mixer.transformer, mixer.output_proj):
                for p in module.parameters():
                    p.requires_grad_(True)

    def mixer_parameters(self):
        """The vendor's temporal mixer only -- what `unfreeze_mixer` releases."""
        mixer = self.torch_pips_mixer
        for module in (mixer.input_proj, mixer.transformer, mixer.output_proj):
            for p in module.parameters():
                yield p


def load_jefftrack(ckpt_path: str, model_size: str = "base",
                 cross_layers: Optional[Sequence[int]] = None, num_proxies: int = 16,
                 cross_heads: int = 4, zero_init: bool = True,
                 carry_state: bool = False, pyramid_level: int = 0,
                 device: str = "cpu") -> JeffTracker:
    """Build JeffTracker and load LocoTrack weights into it.

    The published checkpoints are Lightning checkpoints: weights live under 'state_dict'
    with every key prefixed 'model.'. Anything missing must be a cross-track parameter and
    nothing may be unexpected -- if a vendor key fails to land, the model is silently part
    random and every number measured from it is fiction.

    A checkpoint written by train_cross.py carries its own architecture under 'jefftrack',
    and that wins over the arguments here. A trained checkpoint loaded into a
    differently-shaped model would either throw a shape error or, worse, land its proxies
    in a block of the wrong width, so the shape travels with the weights rather than
    depending on the caller passing the same flags months later.
    """
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Every checkpoint in weights/ was written before the rename and carries its
    # architecture under "dbtrack". Reading only the new key would make all of them
    # unloadable, so both are accepted and only the new one is ever written.
    saved = None
    if isinstance(blob, dict):
        saved = blob.get("jefftrack")
        if saved is None:
            saved = blob.get("dbtrack")
    if saved:
        model_size = saved.get("model_size", model_size)
        cross_layers = saved.get("cross_layers", cross_layers)
        num_proxies = int(saved.get("num_proxies", num_proxies))
        cross_heads = int(saved.get("cross_heads", cross_heads))
        carry_state = bool(saved.get("carry_state", carry_state))
        pyramid_level = int(saved.get("pyramid_level", pyramid_level))
    model = JeffTracker(model_size=model_size, cross_layers=cross_layers,
                    num_proxies=num_proxies, cross_heads=cross_heads, zero_init=zero_init,
                    carry_state=carry_state, pyramid_level=pyramid_level)
    state = blob["state_dict"] if "state_dict" in blob else blob
    state = {k.replace("model.", "", 1): v for k, v in state.items()}

    # Order matters, and it is the opposite of the obvious one. The model is built at the
    # vendor's three-level width so a LocoTrack checkpoint lands cleanly on every tensor;
    # only THEN is the pyramid widened, splicing the fourth level's channels in at zero
    # against weights that are now the trained ones. Widen first and the splice copies
    # random values, the loader skips the mismatched tensors, and the result is a
    # part-random model that still produces plausible tracks.
    #
    # A checkpoint TRAINED at four levels already carries the wide tensors, so there the
    # model has to be widened before loading instead. Which case applies is read off the
    # checkpoint rather than assumed.
    widened = "torch_pips_mixer.input_proj.weight"
    ckpt_w = state.get(widened)
    narrow = model.torch_pips_mixer.input_proj.in_features
    pre_widened = bool(pyramid_level) and ckpt_w is not None and ckpt_w.shape[1] > narrow
    if pre_widened:
        model._add_pyramid_levels(pyramid_level, zero_init=zero_init)

    missing, unexpected = model.load_state_dict(state, strict=False)
    stray = [k for k in missing if ".cross." not in k]
    if stray or unexpected:
        raise SystemExit(
            "[ERROR] LocoTrack weights do not fit JeffTracker: {} non-cross keys missing, "
            "{} unexpected. missing={} unexpected={}".format(
                len(stray), len(unexpected), stray[:5], list(unexpected)[:5]))
    if pyramid_level and not pre_widened:
        model._add_pyramid_levels(pyramid_level, zero_init=zero_init)
    return model.to(device).eval()
