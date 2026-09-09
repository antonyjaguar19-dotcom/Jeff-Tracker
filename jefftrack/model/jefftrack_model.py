"""Jeff-Tracker = LocoTrack + cross-track attention, initialised to be LocoTrack exactly.

LocoTrack's refinement transformer is per-track by construction: locotrack_model.py:891
folds the track axis into the batch before calling the mixer. This module rebuilds that
mixer so the vendor's temporal blocks stay exactly as they are and a cross-track block is
interleaved between them, then subclasses LocoTrack to use it.

Two properties are load-bearing and both are checked by ../../check_identity.py:

  1. The vendor's parameters keep their names. `input_proj`, `transformer.*` and
     `output_proj` are the SAME module objects, re-parented, so a LocoTrack checkpoint
     loads into Jeff-Tracker with the new `cross.*` keys as the only additions -- not a rename
     in sight, and no weight is transformed on the way in.
  2. With the cross blocks zero-initialised the model's output is bit-for-bit identical to
     LocoTrack's. Not close: identical. That is what makes a baseline-vs-treatment
     measurement on identical footage possible from one binary, the property
     a per-track config view exists to preserve in a host pipeline.

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
                 num_proxies: int = 16, num_heads: int = 4, zero_init: bool = True):
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
        self.cross = nn.ModuleDict({
            str(i): CrossTrackAttention(self.dim, num_heads=num_heads,
                                        num_proxies=num_proxies, zero_init=zero_init)
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
        h = x
        for i, layer in enumerate(self.transformer.layers):
            h_norm = layer["layer_norm1"](h)
            h = h + layer["attn"](h_norm, h_norm, h_norm, mask=None)
            h_norm = layer["layer_norm2"](h)
            h = h + layer["dense"](h_norm)
            # nn.ModuleDict has no .get(); membership is the supported test.
            if str(i) in self.cross:
                h = self.cross[str(i)](h.reshape(b, n, f, d)).reshape(bn, f, d)
        h = self.transformer.ln_out(h)
        return self.output_proj(h)


class JeffTracker(LocoTrack):
    """LocoTrack whose refinement mixer can see across tracks."""

    def __init__(self, *args, cross_layers: Optional[Sequence[int]] = None,
                 num_proxies: int = 16, cross_heads: int = 4, zero_init: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.torch_pips_mixer = CrossTrackPIPSTransformer(
            self.torch_pips_mixer, cross_layers=cross_layers, num_proxies=num_proxies,
            num_heads=cross_heads, zero_init=zero_init)

    def forward(self, video: torch.Tensor, *args, **kwargs):
        # refine_pips folds the track axis into the batch; the mixer needs the batch size
        # back to undo that. video is [batch, time, height, width, 3].
        self.torch_pips_mixer.batch_size = int(video.shape[0])
        return super().forward(video, *args, **kwargs)

    def cross_parameters(self):
        """Just the new blocks -- what a fine-tune trains while the rest stays frozen."""
        return self.torch_pips_mixer.cross.parameters()

    def freeze_base(self) -> None:
        """Everything except the cross-track blocks stops learning.

        On a 16 GB A4000 this is not a preference, it is the difference between a
        fine-tune that fits and one that does not.
        """
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.cross_parameters():
            p.requires_grad_(True)


def load_jefftrack(ckpt_path: str, model_size: str = "base",
                 cross_layers: Optional[Sequence[int]] = None, num_proxies: int = 16,
                 cross_heads: int = 4, zero_init: bool = True,
                 device: str = "cpu") -> JeffTracker:
    """Build Jeff-Tracker and load LocoTrack weights into it.

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
    # Checkpoints written before the rename carry their architecture under "dbtrack".
    # Reading only the new key would make every already-published weight file
    # unloadable by its own repository, so both are accepted and the old one is
    # never written again.
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
    model = JeffTracker(model_size=model_size, cross_layers=cross_layers,
                    num_proxies=num_proxies, cross_heads=cross_heads, zero_init=zero_init)
    state = blob["state_dict"] if "state_dict" in blob else blob
    state = {k.replace("model.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    stray = [k for k in missing if ".cross." not in k]
    if stray or unexpected:
        raise SystemExit(
            "[ERROR] LocoTrack weights do not fit Jeff-Tracker: {} non-cross keys missing, "
            "{} unexpected. missing={} unexpected={}".format(
                len(stray), len(unexpected), stray[:5], list(unexpected)[:5]))
    return model.to(device).eval()
