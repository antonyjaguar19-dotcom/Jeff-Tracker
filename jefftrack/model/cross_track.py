"""Cross-track attention: let tracks see each other.

LocoTrack refines every track in complete isolation. The vendor makes this explicit at
locotrack_model.py:891 --

    x = rearrange(mlp_input, 'b n f c -> (b n) f c')
    res = self.torch_pips_mixer(x)

-- tracks are folded into the batch dimension, so the temporal transformer never sees more
than one trajectory at a time. That is the structural difference from CoTracker, and it is
what the occlusion measurement in FINDINGS.md reports from the other end: only 5.9% of
ground-truth-occluded frames get a position at all, because a point with nothing to look at
has nothing to go on. A point that is hidden RIGHT NOW is not unknowable if fifty of its
neighbours are visible and moving rigidly with it.

Written from the description in the CoTracker3 paper (arXiv 2410.11831), which reports
cross-track attention as worth +5.1 delta_avg on occluded points against +1.6 on visible
ones (Table 3). No code, weights, or tensors here derive from the CoTracker repository,
which is CC-BY-NC-4.0 -- see ../../LICENSES.md. Architecture is not copyrightable; source
code is.

Shape of the thing:

  * attention runs over the TRACK axis, independently per frame. Tracks are a set, not a
    sequence, so there is no positional encoding and no causal mask -- adding either would
    be asserting an order between track 7 and track 8 that does not exist.
  * it is mediated by K learnable proxy tokens rather than run as N x N. Tracks write into
    the proxies, the proxies are read back by every track. Cost is O(N*K) instead of
    O(N^2), which is what makes a few hundred tracks affordable; the paper uses the same
    device for the same reason.
  * the output projection is ZERO-INITIALISED. A freshly built model therefore adds
    exactly nothing and reproduces LocoTrack bit for bit. That property is not a nicety:
    it is what lets one binary produce both the baseline and the treatment on identical
    footage, the same reason the same property check_identity.py enforces returns the shot config object
    itself when there are no overrides. check_identity.py enforces it.

Two kinds of proxy, and the difference is the whole point of `carry_state`
--------------------------------------------------------------------------

**Static (carry_state=False, the original).** The K proxies are rebuilt from the same
learned parameter for every frame, independently. They summarise the tracks *at that
instant* and are thrown away. Nothing links the proxies at frame t to the proxies at
frame t-1.

**Carried (carry_state=True).** The caller owns the tokens, passes them through the
vendor's TEMPORAL blocks alongside the real tracks, and hands them back to the next cross
block. The tokens therefore accumulate state along the time axis: what the group was
doing, integrated over frames.

That distinction is not cosmetic, and it was measured. Widening how many tracks the static
block can see does nothing at all -- on lab02_occ with the temporal mixer UNFROZEN, the
occluded mean is 3.186 px at query_chunk_size 64, 3.185 at 256 and 3.184 at 600, where 600
puts every track in one chunk and the block sees the entire shot's worth of neighbours. Four
times, then nine times the information, for 0.002 px. (VRAM 2.79 -> 11.29 GB, 6x the time.)

The block is not short of neighbours. It is short of a way to remember them. A point that
is hidden at frame t has an uninformative token at frame t, and a per-frame summary of its
neighbours is built from that same frame -- so the one thing that could place it, *where
the group has been heading over the last twenty frames*, is exactly what a static proxy
cannot hold. CoTracker3 gets this from virtual track tokens that ride through its time
attention with the real ones; carry_state is that, on LocoTrack's mixer.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossTrackAttention(nn.Module):
    """One block: gather across tracks into proxies, broadcast back, add as a residual.

    Input and output are both (B, N, F, D) -- batch, tracks, frames, channels.
    """

    def __init__(self, dim: int, num_heads: int = 4, num_proxies: int = 16,
                 mlp_ratio: float = 2.0, zero_init: bool = True,
                 carry_state: bool = False):
        super().__init__()
        self.dim = dim
        self.num_proxies = num_proxies
        # See the module docstring's "Two kinds of proxy". False keeps the original static
        # basis; True makes the caller own the tokens and pass them through time attention.
        self.carry_state = bool(carry_state)

        # The proxies are a learned summary basis, not per-track state: the same K tokens
        # are used for every frame and every batch item, and they carry no identity of
        # their own until training gives them one. With carry_state they are only the
        # INITIAL value -- the caller carries the updated tokens between layers.
        self.proxies = nn.Parameter(torch.randn(num_proxies, dim) * 0.02)

        self.norm_in = nn.LayerNorm(dim)
        self.gather = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm_proxy = nn.LayerNorm(dim)
        self.broadcast = nn.MultiheadAttention(dim, num_heads, batch_first=True)

        hidden = int(dim * mlp_ratio)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        # The single gate on the whole block. Everything above can be arbitrary; as long as
        # out_proj is zero the block contributes exactly zero and the model is LocoTrack.
        # It gates only the REAL tracks: the proxies may update freely, since nothing reads
        # them but this block, and gating them too would make their gradient identically
        # zero and freeze them at their init for the whole of training.
        self.out_proj = nn.Linear(dim, dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def init_state(self, batch: int, frames: int, device, dtype) -> torch.Tensor:
        """The proxy tokens a carry_state caller starts with: (B, K, F, D)."""
        p = self.proxies.to(device=device, dtype=dtype)
        return p[None, :, None, :].expand(batch, -1, frames, -1).contiguous()

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        """(B, N, F, D) -> (B, N, F, D), or (x, state) -> (x, state) when carrying.

        `state` is (B, K, F, D). When it is given the block does NOT rebuild the proxies
        from `self.proxies`; it reads the ones handed in and returns them updated, so the
        caller can put them through its own temporal attention between layers. That is the
        entire difference between a per-frame summary and a token with a memory.
        """
        B, N, F, D = x.shape
        # One attention problem per (batch, frame): the tracks at that instant.
        h = x.permute(0, 2, 1, 3).reshape(B * F, N, D)
        h = self.norm_in(h)

        if state is None:
            p = self.proxies.unsqueeze(0).expand(B * F, -1, -1)
        else:
            # (B, K, F, D) -> (B*F, K, D), matching h's flattening exactly.
            p = state.permute(0, 2, 1, 3).reshape(B * F, state.shape[1], D)

        p_new, _ = self.gather(p, h, h, need_weights=False)   # proxies read the tracks
        p_new = self.norm_proxy(p_new)

        h, _ = self.broadcast(h, p_new, p_new, need_weights=False)  # tracks read proxies
        h = h + self.mlp(self.norm_mlp(h))
        h = self.out_proj(h)

        h = h.reshape(B, F, N, D).permute(0, 2, 1, 3)
        if state is None:
            return x + h
        # Residual on the proxies as well, so a carried token is refined layer by layer
        # rather than overwritten -- overwriting would discard everything the time blocks
        # put into it since the last cross block, which is the point of carrying it.
        K = state.shape[1]
        p_out = state + p_new.reshape(B, F, K, D).permute(0, 2, 1, 3)
        return x + h, p_out
