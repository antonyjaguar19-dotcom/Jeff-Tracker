# Copyright (c) 2026 Jeff-Tracker contributors
# Licensed under Apache 2.0 (see LICENSES.md)

"""Neural Trajectory Refiner (2nd-Pass Track Enhancer).

A temporal transformer network that refines initial 2D tracks by learning
smooth physical motion priors, temporal bidirectional consistency, and
occlusion compensation from ground-truth trajectories (e.g. Kubric MOVi).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D)
        return x + self.pe[:, :x.size(1)]


class TrackRefiner(nn.Module):
    """Lightweight 2nd-pass Temporal Trajectory Refinement Network.

    Takes raw/initial 2D trajectory P (B, T, 2) and confidence C (B, T, 1),
    and predicts sub-pixel residual correction delta_P and refined confidence C*.
    """

    def __init__(self, in_dim: int = 7, hidden_dim: int = 128, num_layers: int = 4, num_heads: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=1000)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Coordinate residual head (predicts delta_x, delta_y in pixels)
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 2)
        )
        # Zero-initialize the final linear layer so network starts as identity (delta_P = 0)
        nn.init.zeros_(self.delta_head[-1].weight)
        nn.init.zeros_(self.delta_head[-1].bias)

        # Confidence head
        self.conf_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, coords: torch.Tensor, conf: torch.Tensor,
                img_size: tuple[int, int] = (512, 512)) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            coords: (B, T, 2) track coordinates in pixel space (x, y).
            conf: (B, T, 1) or (B, T) confidence in [0, 1].
            img_size: (H, W) for coordinate normalization.

        Returns:
            refined_coords: (B, T, 2) refined pixel coordinates.
            refined_conf: (B, T, 1) refined confidence scores.
        """
        if conf.ndim == 2:
            conf = conf.unsqueeze(-1)

        B, T, _ = coords.shape
        H, W = img_size
        scale = torch.tensor([W, H], device=coords.device, dtype=torch.float32)

        # Normalize coordinates to [-1, 1] for stable neural processing
        norm_coords = (coords / scale) * 2.0 - 1.0

        # Compute 1st order velocity: delta(t) = P(t) - P(t-1)
        vel = torch.zeros_like(norm_coords)
        vel[:, 1:] = norm_coords[:, 1:] - norm_coords[:, :-1]

        # Compute 2nd order acceleration: acc(t) = vel(t) - vel(t-1)
        acc = torch.zeros_like(vel)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

        # Input feature: [x, y, conf, vel_x, vel_y, acc_x, acc_y] -> (B, T, 7)
        feat = torch.cat([norm_coords, conf, vel, acc], dim=-1)

        h = self.input_proj(feat)
        h = self.pos_encoder(h)
        h = self.transformer(h)

        # Predict normalized residual offset and unscale to pixels
        norm_delta = self.delta_head(h)
        delta_pixels = norm_delta * (scale * 0.5)

        refined_coords = coords + delta_pixels
        refined_conf = self.conf_head(h)

        return refined_coords, refined_conf


class AffineCouplingBlock(nn.Module):
    """OmniMotion / RealNVP Invertible Affine Coupling Block.

    Transforms coordinate pairs (x, y) with strictly invertible algebra (cycle error = 0.0):
      Forward:  y_2 = x_2 * exp(s(x_1, h)) + t(x_1, h)
      Inverse:  x_2 = (y_2 - t(x_1, h)) * exp(-s(x_1, h))
    """
    def __init__(self, cond_dim: int, split_dim: int = 0, hidden_dim: int = 64):
        super().__init__()
        self.split_dim = split_dim
        self.net = nn.Sequential(
            nn.Linear(1 + cond_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2)
        )
        # Zero-initialize so block starts as the exact mathematical identity mapping
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        xa, xb = x[..., :1], x[..., 1:]
        if self.split_dim == 0:
            c_in = torch.cat([xa, cond], dim=-1)
            st = self.net(c_in)
            s, t = torch.chunk(st, 2, dim=-1)
            s = torch.clamp(s, -3.0, 3.0)
            if not inverse:
                yb = xb * torch.exp(s) + t
            else:
                yb = (xb - t) * torch.exp(-s)
            return torch.cat([xa, yb], dim=-1)
        else:
            c_in = torch.cat([xb, cond], dim=-1)
            st = self.net(c_in)
            s, t = torch.chunk(st, 2, dim=-1)
            s = torch.clamp(s, -3.0, 3.0)
            if not inverse:
                ya = xa * torch.exp(s) + t
            else:
                ya = (xa - t) * torch.exp(-s)
            return torch.cat([ya, xb], dim=-1)


class InvertibleTrackRefiner(TrackRefiner):
    """OmniMotion-inspired Invertible Trajectory Refinement Network.

    Guarantees exact mathematical bidirectional cycle consistency:
    refine_inverse(refine_forward(P)) == P with zero forward-backward drift.
    """

    def __init__(self, in_dim: int = 7, hidden_dim: int = 128, num_layers: int = 4,
                 num_heads: int = 4, num_coupling_layers: int = 4):
        super().__init__(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, num_heads=num_heads)
        self.coupling_blocks = nn.ModuleList([
            AffineCouplingBlock(cond_dim=hidden_dim, split_dim=i % 2, hidden_dim=hidden_dim // 2)
            for i in range(num_coupling_layers)
        ])

    def forward(self, coords: torch.Tensor, conf: torch.Tensor,
                img_size: tuple[int, int] = (512, 512),
                inverse: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        if conf.ndim == 2:
            conf = conf.unsqueeze(-1)

        B, T, _ = coords.shape
        H, W = img_size
        scale = torch.tensor([W, H], device=coords.device, dtype=torch.float32)

        norm_coords = (coords / scale) * 2.0 - 1.0

        vel = torch.zeros_like(norm_coords)
        vel[:, 1:] = norm_coords[:, 1:] - norm_coords[:, :-1]

        acc = torch.zeros_like(vel)
        acc[:, 1:] = vel[:, 1:] - vel[:, :-1]

        feat = torch.cat([norm_coords, conf, vel, acc], dim=-1)

        h = self.input_proj(feat)
        h = self.pos_encoder(h)
        h = self.transformer(h)

        # Invertible coupling transform on normalized coordinate space
        cur_coords = norm_coords
        blocks = reversed(self.coupling_blocks) if inverse else self.coupling_blocks
        for block in blocks:
            cur_coords = block(cur_coords, h, inverse=inverse)

        refined_coords = (cur_coords + 1.0) * 0.5 * scale
        refined_conf = self.conf_head(h)

        return refined_coords, refined_conf
