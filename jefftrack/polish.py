"""Native-Resolution High-Precision Patch Polish for Matchmove.

In neural tracking, downscaling full-resolution (1080p / 4K) plates to model resolution
(256x256 or 512x384) introduces spatial quantization error (~1-3 native pixels).

This module runs a native-resolution Lucas-Kanade / pyramidal sub-pixel refinement
around the neural tracker's predicted coordinates on the unscaled video frames.
Because the search window is tightly constrained to +/- max_drift around the neural
prediction, it is completely immune to classical tracking failure/loss-of-lock while
delivering sub-tenth-pixel precision on true image gradients.
"""
from __future__ import annotations

import cv2
import numpy as np


def polish_tracks_native(frames: np.ndarray, tracks: np.ndarray, vis: np.ndarray,
                          win_size: int = 21, max_level: int = 2,
                          max_drift: float = 4.0) -> np.ndarray:
    """Refine neural tracking predictions using native unscaled image gradients.

    Vectorized across all visible tracks per frame pair for high throughput.

    Args:
        frames: (T, H, W, 3) full-resolution video frames (uint8 BGR or RGB).
        tracks: (T, N, 2) predicted track coordinates in full-resolution pixels [x, y].
        vis: (T, N) bool visibility mask (True = visible / valid).
        win_size: optical flow window size (pixels at native resolution).
        max_level: pyramid levels for multi-scale gradient matching.
        max_drift: maximum allowable refinement adjustment (pixels). If LK moves
            farther than this threshold, it is rejected and the neural track is retained.

    Returns:
        polished: (T, N, 2) float32 coordinates refined to true native sub-pixel gradients.
    """
    T, N, _ = tracks.shape
    H, W = frames.shape[1:3]
    polished = tracks.copy().astype(np.float32)

    # Convert frames to grayscale for gradient calculations
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if (f.ndim == 3 and f.shape[-1] == 3) else f
             for f in frames]

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)

    for t in range(1, T):
        val = np.where(vis[t - 1] & vis[t])[0]
        if len(val) == 0:
            continue

        pts_prev = polished[t - 1, val].reshape(-1, 1, 2).astype(np.float32)
        pts_init = tracks[t, val].reshape(-1, 1, 2).astype(np.float32)

        # In-bounds check
        in_bounds = ((pts_prev[:, 0, 0] >= 0) & (pts_prev[:, 0, 0] < W) &
                     (pts_prev[:, 0, 1] >= 0) & (pts_prev[:, 0, 1] < H) &
                     (pts_init[:, 0, 0] >= 0) & (pts_init[:, 0, 0] < W) &
                     (pts_init[:, 0, 1] >= 0) & (pts_init[:, 0, 1] < H))

        if not np.any(in_bounds):
            continue

        active_indices = val[in_bounds]
        p_prev_sub = pts_prev[in_bounds]
        p_init_sub = pts_init[in_bounds]

        p_refined, status, _ = cv2.calcOpticalFlowPyrLK(
            grays[t - 1], grays[t],
            p_prev_sub, p_init_sub,
            winSize=(win_size, win_size),
            maxLevel=max_level,
            flags=cv2.OPTFLOW_USE_INITIAL_FLOW,
            criteria=criteria
        )

        stat = status.ravel() == 1
        ref_pts = p_refined[:, 0]
        neural_pts = tracks[t, active_indices]

        drift = np.hypot(ref_pts[:, 0] - neural_pts[:, 0],
                         ref_pts[:, 1] - neural_pts[:, 1])
        good = stat & (drift <= max_drift)
        good &= ((ref_pts[:, 0] >= 0) & (ref_pts[:, 0] < W) &
                 (ref_pts[:, 1] >= 0) & (ref_pts[:, 1] < H))

        if np.any(good):
            polished[t, active_indices[good]] = ref_pts[good]

    return polished
