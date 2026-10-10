"""Temporal Matchmove Trajectory Smoother.

Physical film cameras (on dollies, cranes, tripods, or steadicams) possess physical inertia.
High-frequency (> 15 Hz) jitter in tracked coordinates is predominantly sensor noise
and sub-pixel discretization artifact.

This module applies an adaptive Savitzky-Golay cubic polynomial filter over contiguous
visible track segments. It strips high-frequency jitter while preserving true camera
acceleration and physical momentum.
"""
from __future__ import annotations

import numpy as np


def smooth_trajectories(tracks: np.ndarray, vis: np.ndarray,
                        window: int = 7) -> np.ndarray:
    """Smooth visible track segments using pure-numpy Savitzky-Golay polynomial filter.

    Args:
        tracks: (T, N, 2) track coordinates.
        vis: (T, N) bool visibility mask.
        window: odd integer filter window length (default: 7).

    Returns:
        smoothed: (T, N, 2) float32 coordinates with high-frequency jitter eliminated.
    """
    if window not in (5, 7, 9):
        window = 7

    # Exact Savitzky-Golay quadratic/cubic smoothing coefficients:
    if window == 5:
        # [-3, 12, 17, 12, -3] / 35
        kernel = np.array([-3, 12, 17, 12, -3], dtype=np.float32) / 35.0
        pad = 2
    elif window == 7:
        # [-2, 3, 6, 7, 6, 3, -2] / 21
        kernel = np.array([-2, 3, 6, 7, 6, 3, -2], dtype=np.float32) / 21.0
        pad = 3
    else:  # window == 9
        # [-21, 14, 39, 54, 59, 54, 39, 14, -21] / 231
        kernel = np.array([-21, 14, 39, 54, 59, 54, 39, 14, -21], dtype=np.float32) / 231.0
        pad = 4

    T, N, _ = tracks.shape
    smoothed = tracks.copy().astype(np.float32)

    for i in range(N):
        v = vis[:, i].astype(bool)
        t_vis = np.where(v)[0]
        if len(t_vis) < window:
            continue

        # Split into contiguous visible runs
        splits = np.where(np.diff(t_vis) > 1)[0] + 1
        runs = np.split(t_vis, splits)

        for run in runs:
            if len(run) < window:
                continue

            for d in (0, 1):
                seg = tracks[run, i, d]
                padded = np.pad(seg, pad, mode="edge")
                conv = np.convolve(padded, kernel, mode="valid")
                smoothed[run, i, d] = conv

    return smoothed
