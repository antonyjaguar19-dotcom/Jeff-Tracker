"""Rigid Motion Propagator & Occlusion Inpainter for Matchmove.

In visual effects camera tracking (3DEqualizer, SynthEyes, Blender), features are
assumed to belong to rigid 3D scene geometry under camera motion.

When a tracked point is occluded by a foreground actor or object, neural appearance
correlation either freezes at the boundary or drifts with the occluder. This module
inpaints and propagates occluded intervals by estimating the local rigid/affine motion
field from surrounding visible background inliers, with velocity consistency filtering
to discard moving foreground outliers.

For interior occlusion intervals [t_start, t_end], it performs bilateral (forward-backward)
rigid propagation with smooth bridge fusion, guaranteeing seamless continuity with the
pre-occlusion track and the post-occlusion re-acquisition.
"""
from __future__ import annotations

import cv2
import numpy as np


def apply_affine(pt: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply 2x3 affine transformation to a 2D point [x, y]."""
    x, y = float(pt[0]), float(pt[1])
    nx = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    ny = M[1, 0] * x + M[1, 1] * y + M[1, 2]
    return np.array([nx, ny], dtype=np.float32)


def compute_global_rigid_motions(tracks: np.ndarray, vis: np.ndarray, ransac_thresh: float = 2.0) -> dict[int, np.ndarray]:
    """Compute the dominant global rigid background transformation between consecutive frame pairs."""
    T, N, _ = tracks.shape
    M_globals = {}
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    for t in range(1, T):
        val = np.where(vis[t - 1] & vis[t])[0]
        if len(val) >= 4:
            src = tracks[t - 1, val].astype(np.float32)
            dst = tracks[t, val].astype(np.float32)
            M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                                     ransacReprojThreshold=ransac_thresh,
                                                     maxIters=1000, confidence=0.99)
            M_globals[t] = M.astype(np.float32) if (M is not None and inliers is not None and inliers.sum() >= 4) else eye
        else:
            M_globals[t] = eye
    return M_globals


def estimate_consistent_motion(pts_prev: np.ndarray, pts_curr: np.ndarray,
                               query_pt: np.ndarray, query_vel: np.ndarray | None = None,
                               k_neighbors: int = 16, ransac_thresh: float = 2.0,
                               M_fallback: np.ndarray | None = None) -> np.ndarray:
    """Estimate a 2x3 affine matrix prioritizing neighbors with consistent velocity.
    
    Strictly filters out moving foreground actors (e.g. pedestrians, dancers) by testing
    deviation from the dominant camera background motion.
    """
    eye = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    fallback = M_fallback if M_fallback is not None else eye
    if len(pts_prev) < 3:
        return fallback

    disps = pts_curr - pts_prev

    # Filter out foreground objects: if candidate displacement diverges from background camera motion
    if M_fallback is not None and len(pts_prev) >= 4:
        cam_p_curr = (M_fallback[:, :2] @ pts_prev.T).T + M_fallback[:, 2]
        cam_disp = cam_p_curr - pts_prev
        disp_err = np.hypot(disps[:, 0] - cam_disp[:, 0], disps[:, 1] - cam_disp[:, 1])
        # Background scene points agree with camera motion; dynamic actors diverge sharply
        bg_mask = disp_err < 2.5
        if bg_mask.sum() >= 4:
            pts_prev = pts_prev[bg_mask]
            pts_curr = pts_curr[bg_mask]
            disps = disps[bg_mask]

    if len(pts_prev) < 3:
        return fallback

    dists = np.hypot(pts_prev[:, 0] - query_pt[0], pts_prev[:, 1] - query_pt[1])
    scores = dists.copy()
    if query_vel is not None and np.hypot(query_vel[0], query_vel[1]) > 0.5:
        # Penalize candidate neighbors moving in the opposite or sharply divergent direction
        v_diff = np.hypot(disps[:, 0] - query_vel[0], disps[:, 1] - query_vel[1])
        scores = scores + v_diff * 8.0

    order = np.argsort(scores)
    k = min(len(order), max(4, k_neighbors))
    sel = order[:k]

    src = pts_prev[sel].astype(np.float32)
    dst = pts_curr[sel].astype(np.float32)

    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                             ransacReprojThreshold=ransac_thresh,
                                             maxIters=1000, confidence=0.99)
    if M is not None and inliers is not None and inliers.sum() >= 3:
        return M.astype(np.float32)
    return fallback


def inpaint_rigid_occlusions(tracks: np.ndarray, vis: np.ndarray,
                             conf: np.ndarray | None = None,
                             k_neighbors: int = 16,
                             ransac_thresh: float = 2.0,
                             max_gap: int = 40,
                             require_reacquisition: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Propagate and inpaint occluded intervals using local rigid/affine transformations.

    Args:
        tracks: (T, N, 2) float coordinates [x, y].
        vis: (T, N) bool visibility / confidence mask (True = valid/visible).
        conf: (T, N) optional float confidence scores in [0, 1].
        k_neighbors: number of spatial neighbors used for local affine fit.
        ransac_thresh: RANSAC reprojection threshold in pixels.
        max_gap: maximum occluded frames to bridge (longer gaps are left gapped).
        require_reacquisition: if True, only inpaint interior gaps that actually re-emerge
            (never extrapolate dead tracks into empty air).

    Returns:
        filled_tracks: (T, N, 2) coordinates with occlusions inpainted.
        is_inpainted: (T, N) bool mask indicating frames that were inpainted.
    """
    T, N, _ = tracks.shape
    filled = tracks.copy().astype(np.float32)
    is_inpainted = np.zeros((T, N), dtype=bool)

    M_globals = compute_global_rigid_motions(tracks, vis, ransac_thresh=ransac_thresh)

    for i in range(N):
        v = vis[:, i].astype(bool)
        if np.all(v) or not np.any(v):
            continue

        spans = []
        in_span = False
        s_start = 0
        for t in range(T):
            if not v[t] and not in_span:
                in_span = True
                s_start = t
            elif v[t] and in_span:
                in_span = False
                spans.append((s_start, t - 1))
        if in_span:
            spans.append((s_start, T - 1))

        for (t0, t1) in spans:
            span_len = t1 - t0 + 1
            if span_len > max_gap:
                continue

            has_prev = t0 > 0 and v[t0 - 1]
            has_next = t1 < T - 1 and v[t1 + 1]

            if require_reacquisition and not (has_prev and has_next):
                # Do not extrapolate tracks into empty air if they never re-emerge
                continue

            # Forward propagation
            fwd_pts = []
            if has_prev:
                cur_pt = filled[t0 - 1, i].copy()
                cur_vel = (cur_pt - filled[t0 - 2, i]) if (t0 >= 2 and v[t0 - 2]) else np.zeros(2, dtype=np.float32)
                for t in range(t0, t1 + 1):
                    val = np.where(vis[t - 1] & vis[t])[0]
                    val = val[val != i]
                    if len(val) >= 3:
                        prev_pts = filled[t - 1, val]
                        curr_pts = filled[t, val]
                        M = estimate_consistent_motion(prev_pts, curr_pts, cur_pt,
                                                       query_vel=cur_vel,
                                                       k_neighbors=k_neighbors,
                                                       ransac_thresh=ransac_thresh,
                                                       M_fallback=M_globals[t])
                    else:
                        M = M_globals[t]
                    nxt = apply_affine(cur_pt, M)
                    cur_vel = nxt - cur_pt
                    cur_pt = nxt
                    fwd_pts.append(cur_pt.copy())

            # Backward propagation
            bwd_pts = []
            if has_next:
                cur_pt = filled[t1 + 1, i].copy()
                cur_vel = (cur_pt - filled[t1 + 2, i]) if (t1 + 2 < T and v[t1 + 2]) else np.zeros(2, dtype=np.float32)
                for t in range(t1, t0 - 1, -1):
                    val = np.where(vis[t + 1] & vis[t])[0]
                    val = val[val != i]
                    M_bwd_fb = cv2.invertAffineTransform(M_globals[t + 1]) if M_globals[t + 1] is not None else np.array([[1., 0., 0.], [0., 1., 0.]], dtype=np.float32)
                    if len(val) >= 3:
                        next_pts = filled[t + 1, val]
                        curr_pts = filled[t, val]
                        M = estimate_consistent_motion(next_pts, curr_pts, cur_pt,
                                                       query_vel=cur_vel,
                                                       k_neighbors=k_neighbors,
                                                       ransac_thresh=ransac_thresh,
                                                       M_fallback=M_bwd_fb)
                    else:
                        M = M_bwd_fb
                    nxt = apply_affine(cur_pt, M)
                    cur_vel = nxt - cur_pt
                    cur_pt = nxt
                    bwd_pts.append(cur_pt.copy())
                bwd_pts.reverse()

            for k in range(span_len):
                frame_idx = t0 + k
                is_inpainted[frame_idx, i] = True
                if has_prev and has_next:
                    alpha = float(k + 1) / float(span_len + 1)
                    filled[frame_idx, i] = (1.0 - alpha) * fwd_pts[k] + alpha * bwd_pts[k]
                elif has_prev:
                    filled[frame_idx, i] = fwd_pts[k]
                elif has_next:
                    filled[frame_idx, i] = bwd_pts[k]

    return filled, is_inpainted
