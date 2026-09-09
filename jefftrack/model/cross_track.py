"""Cross-track attention: let tracks see each other.

LocoTrack refines every track in complete isolation. The vendor makes this explicit at
locotrack_model.py:891 --

    x = rearrange(mlp_input, 'b n f c -> (b n) f c')
    res = self.torch_pips_mixer(x)

-- tracks are folded into the batch dimension, so the temporal transformer never sees more
than one trajectory at a time. That limitation is what the occlusion measurement in
METHOD.md reports from the other end: only 5.9% of
ground-truth-occluded frames get a position at all, because a point with nothing to look at
has nothing to go on. A point that is hidden RIGHT NOW is not unknowable if fifty of its
neighbours are visible and moving rigidly with it.

ATTRIBUTION. Cross-track attention is not an idea of ours: it is implemented from the
description in Karaev et al., arXiv:2410.11831, which reports it as worth +5.1 delta_avg
on occluded points against +1.6 on visible ones (Table 3). This file is written from that
description alone -- no code, weights or tensors here derive from any implementation of
the paper, which is CC-BY-NC-4.0. Method and architecture are not copyrightable; source
code is. See ../../docs/LICENSES.md.

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
    footage, the same reason a per-track config view returns the shot config object
    itself when there are no overrides. check_identity.py enforces it.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class CrossTrackAttention(nn.Module):
    """One block: gather across tracks into proxies, broadcast back, add as a residual.

    Input and output are both (B, N, F, D) -- batch, tracks, frames, channels.
    """

    def __init__(self, dim: int, num_heads: int = 4, num_proxies: int = 16,
                 mlp_ratio: float = 2.0, zero_init: bool = True):
        super().__init__()
        self.dim = dim
        self.num_proxies = num_proxies

        # The proxies are a learned summary basis, not per-track state: the same K tokens
        # are used for every frame and every batch item, and they carry no identity of
        # their own until training gives them one.
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
        self.out_proj = nn.Linear(dim, dim)
        if zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, F, D) -> (B, N, F, D)."""
        B, N, F, D = x.shape
        # One attention problem per (batch, frame): the tracks at that instant.
        h = x.permute(0, 2, 1, 3).reshape(B * F, N, D)
        h = self.norm_in(h)

        p = self.proxies.unsqueeze(0).expand(B * F, -1, -1)
        p, _ = self.gather(p, h, h, need_weights=False)      # proxies read the tracks
        p = self.norm_proxy(p)

        h, _ = self.broadcast(h, p, p, need_weights=False)   # tracks read the proxies
        h = h + self.mlp(self.norm_mlp(h))
        h = self.out_proj(h)

        h = h.reshape(B, F, N, D).permute(0, 2, 1, 3)
        return x + h
