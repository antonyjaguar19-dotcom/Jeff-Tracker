"""Run direct comparison between CoTracker3 and Jeff-Tracker on the test shot.

Loads the video, seeds the identical points on frame 0, runs:
  1. CoTracker3
  2. Jeff-Tracker + Matchmove Rigid Inpainting
  3. Jeff-Tracker Raw

Saves individual MP4s and a side-by-side comparison MP4 with synced frames.
"""
from __future__ import annotations

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
from jefftrack.io import draw_overlay, load_plate_frames, seed_corners, track_colors
from jefftrack.rigid_fill import inpaint_rigid_occlusions
from run_jefftrack import run_cotracker_external


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Compare CoTracker3 and Jeff-Tracker on test shot")
    ap.add_argument("--video", default=r"E:\CHAT_GPT\Co_Tracker_v001\tests\cycle.mp4",
                    help="path to input video file")
    ap.add_argument("--name", default="", help="shot name (defaults to video basename)")
    ap.add_argument("--points", type=int, default=350, help="number of seeded points")
    ap.add_argument("--fps", type=float, default=0.0, help="fps (0 = auto-detect)")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "jefftrack_reacq_stream.ckpt"),
                    help="checkpoint path for Jeff-Tracker")
    ap.add_argument("--cached-ct", action="store_true",
                    help="reuse cached CoTracker3 tracks if available")
    ap.add_argument("--bidirectional", action="store_true",
                    help="enable forward + backward bidirectional tracking")
    ap.add_argument("--polish", action="store_true",
                    help="enable native-resolution patch polish (<0.1 px)")
    ap.add_argument("--smooth", action="store_true",
                    help="enable Savitzky-Golay trajectory smoothing")
    a = ap.parse_args()

    video_path = a.video
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    shot_name = a.name if a.name else os.path.splitext(os.path.basename(video_path))[0]
    out_dir = os.path.join(ROOT, "out")
    os.makedirs(out_dir, exist_ok=True)

    # Detect fps if not provided
    fps = a.fps
    if fps <= 0:
        _cap = cv2.VideoCapture(video_path)
        fps = _cap.get(cv2.CAP_PROP_FPS) or 25.0
        _cap.release()

    print(f"[input] loading {video_path} ({fps:.1f} fps)...")
    frames, plate_wh, first_frame = load_plate_frames(video_path, start=1, end=0, work_w=0)
    T, Hw, Ww = frames.shape[:3]
    plate_w, plate_h = plate_wh
    print(f"[plate] {T} frames {first_frame}..{first_frame + T - 1}  plate {plate_w}x{plate_h}")

    pts = seed_corners(frames[0], a.points)
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], axis=1)
    print(f"[seed] {len(pts)} points seeded from frame 0")

    ct_cache_file = os.path.join(out_dir, f"tracks_cotracker3_{shot_name}.npz")
    if a.cached_ct and os.path.isfile(ct_cache_file):
        print(f"\n--- 1. Loading cached CoTracker3 results from {ct_cache_file} ---")
        with np.load(ct_cache_file) as z:
            ct_tracks = z["tracks"]
            ct_vis = z["vis"]
            ct_conf = z["conf"]
            ct_dt = float(z["dt"]) if "dt" in z else 574.97
    else:
        # 1. Run CoTracker3
        print("\n--- 1. Running CoTracker3 ---")
        ct_tracks, ct_vis, ct_conf, ct_dt = run_cotracker_external(frames, pts)
        np.savez_compressed(ct_cache_file, tracks=ct_tracks, vis=ct_vis, conf=ct_conf, dt=ct_dt)
    print(f"[cotracker3] {ct_dt:.2f}s  {ct_dt / T:.4f}s/frame  visible: {100.0 * ct_vis.mean():.1f}%")

    # 2. Run Jeff-Tracker (orientation aware model resolution)
    if Hw > Ww:
        model_res = (512, 384)  # Portrait
    else:
        model_res = (384, 512)  # Landscape

    print(f"\n--- 2. Running Jeff-Tracker (ckpt={a.ckpt}, model_res={model_res[0]}x{model_res[1]}) ---")
    eng = JeffTrackEngine(device="cuda", model_size="base", ckpt=a.ckpt,
                          model_res=model_res, arch="jefftrack")
    t0 = time.time()
    if a.bidirectional:
        from jefftrack.bidirectional import track_bidirectional
        print("[matchmove] running bidirectional tracking...")
        jt_raw_tracks, jt_vis, jt_conf = track_bidirectional(eng, frames, q)
    else:
        jt_raw_tracks, jt_vis, jt_conf = eng.track_queries_conf(frames, q)

    if a.polish:
        from jefftrack.polish import polish_tracks_native
        print("[matchmove] applying native patch polish...")
        jt_raw_tracks = polish_tracks_native(frames, jt_raw_tracks, jt_vis)

    if a.smooth:
        from jefftrack.smooth import smooth_trajectories
        print("[matchmove] applying trajectory smoother...")
        jt_raw_tracks = smooth_trajectories(jt_raw_tracks, jt_vis)

    jt_dt = time.time() - t0
    print(f"[jefftrack] {jt_dt:.2f}s  {jt_dt / T:.4f}s/frame  visible: {100.0 * jt_vis.mean():.1f}%")

    # 3. Apply Matchmove Rigid Inpainting to Jeff-Tracker
    print("\n--- 3. Applying Matchmove Rigid Occlusion Inpainting ---")
    jt_rigid_tracks, jt_inpainted = inpaint_rigid_occlusions(jt_raw_tracks, jt_vis, jt_conf)
    inpaint_count = jt_inpainted.sum()
    print(f"[rigid_inpaint] inpainted {inpaint_count} occluded frame-track points across {T} frames")

    # Colors
    colors = track_colors(len(pts))

    # Output dimensions for video render
    if Ww >= Hw:
        # Landscape: 960x540 half-panel -> 1920x540 side-by-side
        render_w = 960
        render_h = int(round(Hw * (render_w / float(Ww))))
    else:
        # Portrait: 540x960 half-panel -> 1080x960 side-by-side
        render_w = 540
        render_h = int(round(Hw * (render_w / float(Ww))))

    # Render 1: CoTracker3 MP4
    ct_mp4 = os.path.join(out_dir, f"compare_{shot_name}_cotracker3.mp4")
    print(f"\n[render] writing CoTracker3 overlay: {ct_mp4}")
    vw_ct = cv2.VideoWriter(ct_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    for t in range(T):
        header = f"CoTracker3 | F:{first_frame + t} | Vis:{100.0 * ct_vis[t].mean():.0f}%"
        fr = draw_overlay(frames[t], ct_tracks, ct_vis, t, colors, header=header)
        vw_ct.write(cv2.resize(fr, (render_w, render_h), interpolation=cv2.INTER_AREA))
    vw_ct.release()

    # Render 2: Jeff-Tracker Matchmove Gap (Ignored during occlusion, re-acquiring upon emergence)
    jt_gap_mp4 = os.path.join(out_dir, f"compare_{shot_name}_jefftrack_gap.mp4")
    print(f"\n[render] writing Jeff-Tracker (Matchmove Gap) overlay: {jt_gap_mp4}")
    vw_gap = cv2.VideoWriter(jt_gap_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    for t in range(T):
        header = f"Jeff-Tracker (Matchmove Gap) | F:{first_frame + t} | Vis:{100.0 * jt_vis[t].mean():.0f}%"
        fr = draw_overlay(frames[t], jt_raw_tracks, jt_vis, t, colors, header=header, gap_occluded=True)
        vw_gap.write(cv2.resize(fr, (render_w, render_h), interpolation=cv2.INTER_AREA))
    vw_gap.release()

    # Render 3: Jeff-Tracker Camera-Rigid Inpainted MP4
    jt_rigid_mp4 = os.path.join(out_dir, f"compare_{shot_name}_jefftrack_rigid.mp4")
    print(f"\n[render] writing Jeff-Tracker (Camera Rigid Inpainted) overlay: {jt_rigid_mp4}")
    vw_jr = cv2.VideoWriter(jt_rigid_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    for t in range(T):
        header = f"Jeff-Tracker (Camera Rigid) | F:{first_frame + t} | Inpainted:{jt_inpainted[t].sum()}"
        fr = draw_overlay(frames[t], jt_rigid_tracks, jt_vis, t, colors, is_inpainted=jt_inpainted, header=header)
        vw_jr.write(cv2.resize(fr, (render_w, render_h), interpolation=cv2.INTER_AREA))
    vw_jr.release()

    # Render 4: Jeff-Tracker Raw MP4
    jt_raw_mp4 = os.path.join(out_dir, f"compare_{shot_name}_jefftrack_raw.mp4")
    print(f"[render] writing Jeff-Tracker (Raw) overlay: {jt_raw_mp4}")
    vw_raw = cv2.VideoWriter(jt_raw_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (render_w, render_h))
    for t in range(T):
        header = f"Jeff-Tracker (Raw Output) | F:{first_frame + t} | Vis:{100.0 * jt_vis[t].mean():.0f}%"
        fr = draw_overlay(frames[t], jt_raw_tracks, jt_vis, t, colors, header=header, gap_occluded=False)
        vw_raw.write(cv2.resize(fr, (render_w, render_h), interpolation=cv2.INTER_AREA))
    vw_raw.release()

    # Render 5: Side-by-Side MP4 (Left: CoTracker3, Right: Jeff-Tracker Matchmove Gap)
    sbs_mp4 = os.path.join(out_dir, f"compare_{shot_name}_side_by_side.mp4")
    sbs_w = render_w * 2
    sbs_h = render_h
    print(f"[render] writing Side-by-Side comparison ({sbs_w}x{sbs_h}): {sbs_mp4}")
    vw_sbs = cv2.VideoWriter(sbs_mp4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (sbs_w, sbs_h))
    for t in range(T):
        h_ct = f"CoTracker3 | F:{first_frame + t}"
        fr_ct = draw_overlay(frames[t], ct_tracks, ct_vis, t, colors, header=h_ct)
        fr_ct_small = cv2.resize(fr_ct, (render_w, render_h), interpolation=cv2.INTER_AREA)

        h_jt = f"Jeff-Tracker (Matchmove Gap) | F:{first_frame + t}"
        fr_jt = draw_overlay(frames[t], jt_raw_tracks, jt_vis, t, colors, header=h_jt, gap_occluded=True)
        fr_jt_small = cv2.resize(fr_jt, (render_w, render_h), interpolation=cv2.INTER_AREA)

        # Concatenate horizontally
        sbs_frame = np.concatenate([fr_ct_small, fr_jt_small], axis=1)
        # Add center divider line
        cv2.line(sbs_frame, (render_w, 0), (render_w, render_h), (255, 255, 255), 2)
        vw_sbs.write(sbs_frame)
    vw_sbs.release()

    print("\n[done] All test runs completed successfully!")
    print(f"  1. CoTracker3:              {ct_mp4}")
    print(f"  2. Jeff-Tracker (Gap):       {jt_gap_mp4}")
    print(f"  3. Jeff-Tracker (Camera):    {jt_rigid_mp4}")
    print(f"  4. Jeff-Tracker (Raw):       {jt_raw_mp4}")
    print(f"  5. Side-by-Side:             {sbs_mp4}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
