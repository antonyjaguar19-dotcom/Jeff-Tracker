"""Bidirectional Forward-Backward Tracking & Temporal Fusion for Matchmove.

In visual effects post-production, the entire shot is available on disk.
Unlike real-time causal systems, an offline tracker can track both forward (0 -> T)
and backward (T -> 0).

At occlusion boundaries, the forward pass loses visibility as an object enters,
while the backward pass maintains visibility right up to that boundary from the
opposite direction. Confidence-weighted fusion of both passes eliminates boundary
hesitation and provides clean, continuous tracks with reduced variance.
"""
from __future__ import annotations

import numpy as np
import torch

from jefftrack.engine import JeffTrackEngine


def track_bidirectional(engine: JeffTrackEngine, frames: np.ndarray,
                        query_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run bidirectional forward-backward tracking with confidence fusion.

    Args:
        engine: initialized JeffTrackEngine.
        frames: (T, H, W, 3) video frames.
        query_points: (N, 3) query points [t, y, x] in pixels.

    Returns:
        fused_tracks: (T, N, 2) fused coordinates [x, y].
        fused_vis: (T, N) bool visibility mask.
        fused_conf: (T, N) float confidence scores in [0, 1].
    """
    T = len(frames)
    N = len(query_points)

    # 1. Forward Pass
    tracks_fwd, vis_fwd, conf_fwd = engine.track_queries_conf(frames, query_points)

    # 2. Backward Pass
    # Invert query frame indices: t_rev = (T - 1) - t_orig
    frames_rev = frames[::-1].copy()
    q_rev = query_points.copy()

    # If queries were seeded at frame 0 and clip spans multiple windows,
    # seed the backward pass from the predicted final positions at frame T-1 (local frame 0 in frames_rev)
    # to avoid window-boundary anchor clipping.
    if np.all(query_points[:, 0] == 0) and T > 120:
        q_rev[:, 0] = 0.0
        q_rev[:, 1] = tracks_fwd[-1, :, 1]  # y
        q_rev[:, 2] = tracks_fwd[-1, :, 0]  # x
    else:
        q_rev[:, 0] = (T - 1) - query_points[:, 0]

    tracks_rev, vis_rev, conf_rev = engine.track_queries_conf(frames_rev, q_rev)

    # Un-reverse backward outputs
    tracks_bwd = tracks_rev[::-1].copy()
    vis_bwd = vis_rev[::-1].copy()
    conf_bwd = conf_rev[::-1].copy()

    # 3. Cycle Consistency Gating & Confidence-weighted fusion
    # Compute spatial difference between forward and backward tracks
    diff = np.hypot(tracks_fwd[..., 0] - tracks_bwd[..., 0],
                    tracks_fwd[..., 1] - tracks_bwd[..., 1])  # (T, N)

    # If the backward pass deviates from the forward pass (> 10 px), decay its fusion weight to zero
    consistency_weight = np.exp(-0.5 * (np.maximum(0.0, diff - 2.0) / 6.0) ** 2)

    eps = 1e-6
    w_fwd = np.maximum(conf_fwd, 0.0)
    w_bwd = np.maximum(conf_bwd, 0.0) * consistency_weight
    total_w = w_fwd + w_bwd + eps

    # (T, N, 2)
    fused_tracks = (w_fwd[..., None] * tracks_fwd + w_bwd[..., None] * tracks_bwd) / total_w[..., None]
    fused_conf = np.maximum(conf_fwd, conf_bwd * consistency_weight)
    fused_vis = vis_fwd | (vis_bwd & (diff < 12.0))

    return fused_tracks.astype(np.float32), fused_vis, fused_conf.astype(np.float32)
