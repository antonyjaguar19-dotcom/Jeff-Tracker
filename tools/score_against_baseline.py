"""Score Jeff-Tracker against an external baseline tracker's output on a real shot.

Real footage has no ground truth, so the baseline's tracks stand in as a REFERENCE: the
numbers below measure agreement with that tracker, not error against truth. The baseline is
loaded through JT_BASELINE_ADAPTER (see tools/external_baseline.py).

Evaluates:
  1. Jeff-Tracker Raw (declines occluded points or leaves them held)
  2. Jeff-Tracker + Matchmove Rigid Inpainting (bilateral RANSAC affine propagation)

Both are evaluated against the baseline on:
  - All frames
  - Visible frames only
  - Occluded frames only (where JT detected occlusion)
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

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

from jefftrack.engine import JeffTrackEngine, DEFAULT_CKPT
from jefftrack.io import load_plate_frames, seed_corners
from jefftrack.rigid_fill import inpaint_rigid_occlusions
from external_baseline import label as baseline_label, run_baseline


def compute_metrics(pred_tracks: np.ndarray, gt_tracks: np.ndarray, mask: np.ndarray | None = None):
    """Compute comprehensive error metrics between predicted and ground truth tracks."""
    diff = pred_tracks - gt_tracks
    dist = np.hypot(diff[..., 0], diff[..., 1])  # (T, N)

    if mask is not None:
        valid_dist = dist[mask]
    else:
        valid_dist = dist.ravel()

    if len(valid_dist) == 0:
        return {}

    return {
        "count": int(len(valid_dist)),
        "mean": float(np.mean(valid_dist)),
        "median": float(np.median(valid_dist)),
        "p75": float(np.percentile(valid_dist, 75)),
        "p90": float(np.percentile(valid_dist, 90)),
        "p95": float(np.percentile(valid_dist, 95)),
        "p99": float(np.percentile(valid_dist, 99)),
        "max": float(np.max(valid_dist)),
        "within_1px": float(np.mean(valid_dist < 1.0) * 100.0),
        "within_2px": float(np.mean(valid_dist < 2.0) * 100.0),
        "within_4px": float(np.mean(valid_dist < 4.0) * 100.0),
        "within_8px": float(np.mean(valid_dist < 8.0) * 100.0),
        "within_16px": float(np.mean(valid_dist < 16.0) * 100.0),
    }


def compute_acceleration_jitter(tracks: np.ndarray):
    """Mean 2nd-order temporal difference ||p_{t+1} - 2*p_t + p_{t-1}|| (pixels/frame^2)."""
    if len(tracks) < 3:
        return 0.0
    acc = tracks[2:] - 2.0 * tracks[1:-1] + tracks[:-2]
    return float(np.mean(np.hypot(acc[..., 0], acc[..., 1])))


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Score Jeff-Tracker against a baseline tracker")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT, help="checkpoint path to evaluate")
    ap.add_argument("--video", required=True, help="path to input video file")
    ap.add_argument("--cached-baseline", action="store_true", help="reuse cached baseline output if available")
    ap.add_argument("--out-suffix", default="", help="suffix for output JSON score sheet")
    a = ap.parse_args()

    video_path = a.video
    out_dir = os.path.join(ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[input] loading {video_path}...")
    frames, plate_wh, first_frame = load_plate_frames(video_path, start=1, end=0, work_w=0)
    T, Hw, Ww = frames.shape[:3]
    print(f"[plate] {T} frames, {Ww}x{Hw}")

    pts = seed_corners(frames[0], 350)
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], axis=1)
    N = len(pts)
    print(f"[seed] {N} points")

    bl_label = baseline_label()
    ct_cache_file = os.path.join(out_dir, "tracks_baseline.npz")
    if a.cached_baseline and os.path.isfile(ct_cache_file):
        print(f"\n[1/3] Loading cached {bl_label} results from {ct_cache_file}...")
        with np.load(ct_cache_file) as z:
            gt_tracks = z["tracks"]
            ct_vis = z["vis"]
            ct_conf = z["conf"]
    else:
        # 1. Run the baseline (reference, not ground truth)
        print(f"\n[1/3] Running {bl_label} (acting as reference)...")
        gt_tracks, ct_vis, ct_conf, _, _, _ = run_baseline(frames, pts)
        np.savez_compressed(ct_cache_file, tracks=gt_tracks, vis=ct_vis, conf=ct_conf)

    # 2. Run Jeff-Tracker Raw
    print(f"[2/3] Running Jeff-Tracker Raw with ckpt: {a.ckpt}...")
    model_res = (512, 384)
    eng = JeffTrackEngine(device="cuda", model_size="base", ckpt=a.ckpt,
                          model_res=model_res, arch="jefftrack")
    jt_raw_tracks, jt_vis, jt_conf = eng.track_queries_conf(frames, q)

    # 3. Run Jeff-Tracker Rigid Inpainting
    print("[3/3] Inpainting occlusions rigidly...")
    jt_rigid_tracks, is_inpainted = inpaint_rigid_occlusions(jt_raw_tracks, jt_vis, jt_conf)

    suffix = f"_{a.out_suffix}" if a.out_suffix else ""

    # Save npz files
    np.savez_compressed(os.path.join(out_dir, "tracks_baseline.npz"),
                        tracks=gt_tracks, vis=ct_vis, conf=ct_conf)
    np.savez_compressed(os.path.join(out_dir, "tracks_jefftrack_raw.npz"),
                        tracks=jt_raw_tracks, vis=jt_vis, conf=jt_conf)
    np.savez_compressed(os.path.join(out_dir, "tracks_jefftrack_rigid.npz"),
                        tracks=jt_rigid_tracks, vis=jt_vis, conf=jt_conf,
                        is_inpainted=is_inpainted)

    occ_mask = ~jt_vis
    vis_mask = jt_vis

    # Compute metrics
    m_all_raw = compute_metrics(jt_raw_tracks, gt_tracks)
    m_all_rigid = compute_metrics(jt_rigid_tracks, gt_tracks)

    m_vis_raw = compute_metrics(jt_raw_tracks, gt_tracks, mask=vis_mask)
    m_vis_rigid = compute_metrics(jt_rigid_tracks, gt_tracks, mask=vis_mask)

    m_occ_raw = compute_metrics(jt_raw_tracks, gt_tracks, mask=occ_mask)
    m_occ_rigid = compute_metrics(jt_rigid_tracks, gt_tracks, mask=occ_mask)

    jitter_ct = compute_acceleration_jitter(gt_tracks)
    jitter_raw = compute_acceleration_jitter(jt_raw_tracks)
    jitter_rigid = compute_acceleration_jitter(jt_rigid_tracks)

    results = {
        "total_points": N,
        "total_frames": T,
        "total_eval_samples": N * T,
        "visible_samples": int(vis_mask.sum()),
        "occluded_samples": int(occ_mask.sum()),
        "all_frames": {
            "raw": m_all_raw,
            "rigid": m_all_rigid,
        },
        "visible_frames": {
            "raw": m_vis_raw,
            "rigid": m_vis_rigid,
        },
        "occluded_frames": {
            "raw": m_occ_raw,
            "rigid": m_occ_rigid,
        },
        "jitter_acceleration": {
            "baseline": jitter_ct,
            "jefftrack_raw": jitter_raw,
            "jefftrack_rigid": jitter_rigid,
        }
    }

    results_json = os.path.join(out_dir, f"score_against_baseline{suffix}.json")
    with open(results_json, "w") as fh:
        json.dump(results, fh, indent=2)

    print("\n" + "=" * 78)
    print(f" SCORE SHEET: JEFF-TRACKER vs {bl_label.upper()} (REFERENCE, NOT GROUND TRUTH)")
    print("=" * 78)
    print(f"Total samples: {N * T} ({N} tracks x {T} frames) | Visible: {vis_mask.sum()} ({100 * vis_mask.mean():.1f}%) | Occluded: {occ_mask.sum()} ({100 * occ_mask.mean():.1f}%)")
    print("-" * 78)
    print(f"{'POPULATION':<20} | {'METRIC':<18} | {'RAW JEFF-TRACK':<16} | {'RIGID INPAINT':<16}")
    print("-" * 78)

    for pop_name, raw_m, rig_m in [
        ("ALL FRAMES", m_all_raw, m_all_rigid),
        ("VISIBLE ONLY", m_vis_raw, m_vis_rigid),
        ("OCCLUDED ONLY", m_occ_raw, m_occ_rigid),
    ]:
        print(f"{pop_name:<20} | Mean Error (px)    | {raw_m['mean']:>12.3f} px   | {rig_m['mean']:>12.3f} px")
        print(f"{'':<20} | Median Error (px)  | {raw_m['median']:>12.3f} px   | {rig_m['median']:>12.3f} px")
        print(f"{'':<20} | p95 Error (px)     | {raw_m['p95']:>12.3f} px   | {rig_m['p95']:>12.3f} px")
        print(f"{'':<20} | Max Error (px)     | {raw_m['max']:>12.3f} px   | {rig_m['max']:>12.3f} px")
        print(f"{'':<20} | < 1 px accuracy    | {raw_m['within_1px']:>11.1f} %    | {rig_m['within_1px']:>11.1f} %")
        print(f"{'':<20} | < 2 px accuracy    | {raw_m['within_2px']:>11.1f} %    | {rig_m['within_2px']:>11.1f} %")
        print(f"{'':<20} | < 4 px accuracy    | {raw_m['within_4px']:>11.1f} %    | {rig_m['within_4px']:>11.1f} %")
        print(f"{'':<20} | < 8 px accuracy    | {raw_m['within_8px']:>11.1f} %    | {rig_m['within_8px']:>11.1f} %")
        print("-" * 78)

    print(f"{'JITTER / ACCEL':<20} | {bl_label}: {jitter_ct:.3f} px/f^2 | Raw: {jitter_raw:.3f} px/f^2 | Rigid: {jitter_rigid:.3f} px/f^2")
    print("=" * 78)
    print(f"Results saved to: {results_json}")


if __name__ == "__main__":
    main()
