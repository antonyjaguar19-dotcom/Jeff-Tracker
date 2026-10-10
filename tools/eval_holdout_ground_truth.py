"""Evaluate Jeff-Tracker Reacq checkpoint against Baseline on held-out Kubric Ground Truth.

Evaluates shard 0007.npz (which was explicitly held out during training via --holdout 1):
  - Visible position error
  - Occluded position error
  - Re-acquisition error (first frame of emergence post-occlusion)
  - Precision thresholds (< 1px, < 2px, < 4px)
  - Trajectory acceleration jitter
"""
from __future__ import annotations

import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import torch

from jefftrack.engine import JeffTrackEngine
from jefftrack.rigid_fill import inpaint_rigid_occlusions


def evaluate_model(name: str, ckpt: str, video: np.ndarray, query_points: np.ndarray,
                   gt_coords: np.ndarray, gt_occ: np.ndarray, device: str = "cuda"):
    T, H, W = video.shape[:3]
    N = len(query_points)

    print(f"\nEvaluating {name} ({ckpt})...")
    eng = JeffTrackEngine(device=device, model_size="base", ckpt=ckpt,
                          model_res=(256, 256), arch="jefftrack")
    t0 = time.time()
    pred_tracks, vis, conf = eng.track_queries_conf(video, query_points)
    dt = time.time() - t0
    fps = T / dt

    diff = pred_tracks - gt_coords
    dist = np.hypot(diff[..., 0], diff[..., 1])  # (T, N)

    # Masks
    # gt_occ: (N, T) bool -> transpose to (T, N)
    occ_t = gt_occ.T
    vis_t = ~occ_t

    # Re-acquisition mask: frame t is visible, frame t-1 was occluded
    reacq_mask = np.zeros_like(occ_t, dtype=bool)
    reacq_mask[1:] = occ_t[:-1] & vis_t[1:]

    # Jitter
    acc = pred_tracks[2:] - 2.0 * pred_tracks[1:-1] + pred_tracks[:-2]
    jitter = float(np.mean(np.hypot(acc[..., 0], acc[..., 1])))

    def _stats(arr):
        if len(arr) == 0:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "<1px": 0.0, "<2px": 0.0, "<4px": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "<1px": float(np.mean(arr < 1.0) * 100.0),
            "<2px": float(np.mean(arr < 2.0) * 100.0),
            "<4px": float(np.mean(arr < 4.0) * 100.0),
        }

    all_stats = _stats(dist.ravel())
    vis_stats = _stats(dist[vis_t])
    occ_stats = _stats(dist[occ_t])
    reacq_stats = _stats(dist[reacq_mask])

    # Also evaluate Rigid Inpaint on top of pred_tracks
    rigid_tracks, _ = inpaint_rigid_occlusions(pred_tracks, vis, conf)
    diff_rigid = rigid_tracks - gt_coords
    dist_rigid = np.hypot(diff_rigid[..., 0], diff_rigid[..., 1])
    acc_rigid = rigid_tracks[2:] - 2.0 * rigid_tracks[1:-1] + rigid_tracks[:-2]
    jitter_rigid = float(np.mean(np.hypot(acc_rigid[..., 0], acc_rigid[..., 1])))
    rigid_reacq = _stats(dist_rigid[reacq_mask])
    rigid_occ = _stats(dist_rigid[occ_t])

    return {
        "name": name,
        "ckpt": ckpt,
        "fps": float(fps),
        "dt": float(dt),
        "jitter_raw": jitter,
        "jitter_rigid": jitter_rigid,
        "all": all_stats,
        "visible": vis_stats,
        "occluded": occ_stats,
        "occluded_rigid": rigid_occ,
        "reacquisition": reacq_stats,
        "reacquisition_rigid": rigid_reacq,
        "n_samples": int(dist.size),
        "n_visible": int(vis_t.sum()),
        "n_occluded": int(occ_t.sum()),
        "n_reacq_events": int(reacq_mask.sum()),
    }


def main():
    holdout_path = os.path.join(ROOT, "datasets", "kubric_cache", "0007.npz")
    if not os.path.isfile(holdout_path):
        print(f"[ERROR] holdout file {holdout_path} not found")
        return 1

    print(f"[data] loading held-out Kubric shard: {holdout_path}...")
    with np.load(holdout_path) as z:
        offs = z["frames_off"]
        blob = z["frames_jpg"]
        coords = z["coords"]       # (N, T, 2)
        occluded = z["occluded"]   # (N, T) bool
        n_frames = len(offs) - 1

        # Decode frames
        frames = []
        for f in range(n_frames):
            img = cv2.imdecode(blob[offs[f]:offs[f + 1]], cv2.IMREAD_COLOR)
            frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        video = np.stack(frames)  # (T, H, W, 3)

    # Sample 256 tracks for clean evaluation
    rng = np.random.default_rng(42)
    # Pick tracks that have at least one occlusion and re-acquisition
    has_occ = occluded.any(axis=1)
    has_vis_after = np.zeros(len(occluded), dtype=bool)
    for i in range(len(occluded)):
        if has_occ[i]:
            for t in range(1, n_frames):
                if occluded[i, t - 1] and not occluded[i, t]:
                    has_vis_after[i] = True
                    break

    cand = np.where(has_vis_after)[0]
    print(f"[data] total tracks: {len(coords)}, tracks with emergence events: {len(cand)}")
    sel = rng.choice(cand, size=min(256, len(cand)), replace=False)
    sel = np.sort(sel)

    sel_coords = coords[sel]       # (N, T, 2)
    sel_occ = occluded[sel]        # (N, T) bool

    # Pick query frame for each track (first visible frame)
    q_list = []
    for i in range(len(sel)):
        vis_frames = np.flatnonzero(~sel_occ[i])
        qf = vis_frames[0] if len(vis_frames) > 0 else 0
        xy = sel_coords[i, qf]
        q_list.append([float(qf), float(xy[0]), float(xy[1])])  # (f, x, y)
    query_points = np.asarray(q_list, dtype=np.float32)

    # Transpose gt_coords to (T, N, 2) to match engine output
    gt_coords = np.transpose(sel_coords, (1, 0, 2))

    models = [
        ("LocoTrack Baseline", os.path.join(ROOT, "weights", "locotrack_base.ckpt")),
        ("Jeff-Tracker Trained", os.path.join(ROOT, "weights", "jefftrack_reacq.ckpt")),
        ("Jeff-Tracker Trained (EMA)", os.path.join(ROOT, "weights", "jefftrack_reacq_ema.ckpt")),
    ]

    results = []
    for name, ckpt in models:
        r = evaluate_model(name, ckpt, video, query_points, gt_coords, sel_occ)
        results.append(r)

    out_file = os.path.join(ROOT, "out", "benchmark_holdout_ground_truth.json")
    with open(out_file, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 92)
    print(" HELD-OUT SYNTHETIC GROUND TRUTH BENCHMARK (KUBRIC SHARD 0007)")
    print("=" * 92)
    print(f"Tracks: {len(sel)} | Frames: {n_frames} | Total Samples: {results[0]['n_samples']} | Emergence Events: {results[0]['n_reacq_events']}")
    print("-" * 92)
    hdr = f"{'METRIC':<28} | {'BASELINE':<18} | {'JEFF-TRACK RAW':<18} | {'JEFF-TRACK EMA':<18}"
    print(hdr)
    print("-" * 92)

    b, raw, ema = results[0], results[1], results[2]

    rows = [
        ("Visible Mean Error (px)", f"{b['visible']['mean']:.3f} px", f"{raw['visible']['mean']:.3f} px", f"{ema['visible']['mean']:.3f} px"),
        ("Visible Median Error (px)", f"{b['visible']['median']:.3f} px", f"{raw['visible']['median']:.3f} px", f"{ema['visible']['median']:.3f} px"),
        ("Visible < 1px Accuracy (%)", f"{b['visible']['<1px']:.1f} %", f"{raw['visible']['<1px']:.1f} %", f"{ema['visible']['<1px']:.1f} %"),
        ("Visible < 2px Accuracy (%)", f"{b['visible']['<2px']:.1f} %", f"{raw['visible']['<2px']:.1f} %", f"{ema['visible']['<2px']:.1f} %"),
        ("RE-ACQUISITION Mean (px)", f"{b['reacquisition']['mean']:.3f} px", f"{raw['reacquisition']['mean']:.3f} px", f"{ema['reacquisition']['mean']:.3f} px"),
        ("RE-ACQUISITION Median (px)", f"{b['reacquisition']['median']:.3f} px", f"{raw['reacquisition']['median']:.3f} px", f"{ema['reacquisition']['median']:.3f} px"),
        ("RE-ACQUISITION < 2px (%)", f"{b['reacquisition']['<2px']:.1f} %", f"{raw['reacquisition']['<2px']:.1f} %", f"{ema['reacquisition']['<2px']:.1f} %"),
        ("RE-ACQUISITION < 4px (%)", f"{b['reacquisition']['<4px']:.1f} %", f"{raw['reacquisition']['<4px']:.1f} %", f"{ema['reacquisition']['<4px']:.1f} %"),
        ("Occluded Mean Error (px)", f"{b['occluded']['mean']:.3f} px", f"{raw['occluded']['mean']:.3f} px", f"{ema['occluded']['mean']:.3f} px"),
        ("Occluded Median (px)", f"{b['occluded']['median']:.3f} px", f"{raw['occluded']['median']:.3f} px", f"{ema['occluded']['median']:.3f} px"),
        ("Jitter / Accel (px/f^2)", f"{b['jitter_raw']:.3f}", f"{raw['jitter_raw']:.3f}", f"{ema['jitter_raw']:.3f}"),
        ("Jitter Rigid (px/f^2)", f"{b['jitter_rigid']:.3f}", f"{raw['jitter_rigid']:.3f}", f"{ema['jitter_rigid']:.3f}"),
        ("Throughput (fps)", f"{b['fps']:.1f} fps", f"{raw['fps']:.1f} fps", f"{ema['fps']:.1f} fps"),
    ]

    for label, v0, v1, v2 in rows:
        print(f"{label:<28} | {v0:>16} | {v1:>16} | {v2:>16}")
    print("=" * 92)
    print(f"Full benchmark data written to: {out_file}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
