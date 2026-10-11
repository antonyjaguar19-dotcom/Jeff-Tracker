"""Jeff-Tracker on a plate directory or video file -> 3DE 2D-track ASCII, npz, overlay, stats.

    python tools/run_jefftrack.py ^
        --plate E:\\path\\to\\shot.mp4 --name myshot ^
        --model-res 512x384 --occlusion-mode rigid_fill

Engines:
  --engine jefftrack   (default: the Apache-2.0 model)
  --engine baseline    (optional external tracker, via JT_BASELINE_ADAPTER;
                        see tools/external_baseline.py)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.engine import JeffTrackEngine, DEFAULT_CKPT  # noqa: E402
from jefftrack.io import (  # noqa: E402
    draw_hud, draw_overlay, list_frames, read_frame, load_plate_frames, seed_corners, seed_grid,
    track_colors, write_3de,
)
from jefftrack.rigid_fill import inpaint_rigid_occlusions  # noqa: E402

if HERE not in sys.path:
    sys.path.insert(0, HERE)
from external_baseline import label as baseline_label, run_baseline  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Jeff-Tracker on a plate or video")
    ap.add_argument("--plate", required=True, help="Video file or directory of frames")
    ap.add_argument("--name", default="shot")
    ap.add_argument("--engine", default="jefftrack", choices=["jefftrack", "baseline"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--work-width", type=int, default=0, help="0 = keep native width")
    ap.add_argument("--model-res", default="384x512",
                    help="Model input resolution HxW (or snap to aspect)")
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--arch", default="jefftrack", choices=["locotrack", "jefftrack"])
    ap.add_argument("--window", type=int, default=0, help="time window; 0 = auto")
    ap.add_argument("--seed", default="corners", choices=["grid", "corners"])
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--points", type=int, default=400)
    ap.add_argument("--render-width", type=int, default=1080)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--hide-occluded", action="store_true",
                    help="drop occluded points from the render instead of drawing them "
                         "hollow. On a hard plate half the points can be hollow, both "
                         "populations move together, and the usable track set becomes "
                         "unreadable. Affects the mp4 only -- the .txt and .npz are "
                         "unchanged, and the 3DE export already omits occluded frames, so "
                         "this shows what the exported file actually contains")
    ap.add_argument("--label", action="store_true",
                    help="write each track's number beside it, matching the name in the "
                         "3DE export (JT_%%04d), so a point on screen can be named")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--occlusion-mode", default="gap",
                    choices=["rigid_fill", "raw", "gap"],
                    help="rigid_fill: inpaint occluded frames from rigid visible neighbors")
    ap.add_argument("--bidirectional", action="store_true",
                    help="run forward + backward two-pass tracking with confidence fusion")
    ap.add_argument("--polish", action="store_true",
                    help="run native-resolution Lucas-Kanade patch polish (< 0.1 px precision)")
    ap.add_argument("--smooth", action="store_true",
                    help="apply temporal Savitzky-Golay trajectory smoothing to eliminate jitter")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    frames, plate_wh, first_frame = load_plate_frames(a.plate, a.start, a.end, a.work_width)
    T, Hw, Ww = frames.shape[:3]
    plate_w, plate_h = plate_wh
    print(f"[plate] {T} frames {first_frame}..{first_frame + T - 1}  plate {plate_w}x{plate_h}  fed at {Ww}x{Hw}")

    pts = (seed_grid(a.grid, Ww, Hw) if a.seed == "grid"
           else seed_corners(frames[0], a.points))
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)
    print(f"[seed] {len(pts)} points ({a.seed})")

    # Adapt model_res orientation to frame orientation (portrait vs landscape)
    mr_parts = [int(v) for v in a.model_res.lower().split("x")]
    if Hw > Ww and mr_parts[0] < mr_parts[1]:
        # Vertical video: swap to H > W
        model_res = (mr_parts[1], mr_parts[0])
    else:
        model_res = (mr_parts[0], mr_parts[1])

    is_inpainted = None
    if a.engine == "baseline":
        engine_label = baseline_label()
        print(f"[engine] running external {engine_label}...")
        tracks, vis, conf, dt, peak_gb, _ = run_baseline(frames, pts)
    else:
        print(f"[engine] running Jeff-Tracker ({a.arch}) at {model_res[0]}x{model_res[1]}...")
        eng = JeffTrackEngine(device="cuda", model_size=a.model_size, ckpt=a.ckpt,
                              model_res=model_res, window=a.window, arch=a.arch)
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        if a.bidirectional:
            from jefftrack.bidirectional import track_bidirectional
            print("[matchmove] running bidirectional forward-backward tracking pass...")
            tracks, vis, conf = track_bidirectional(eng, frames, q)
        else:
            tracks, vis, conf = eng.track_queries_conf(frames, q)
        dt = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9

        if a.occlusion_mode == "rigid_fill":
            print("[matchmove] applying camera rigid occlusion inpainting...")
            tracks, is_inpainted = inpaint_rigid_occlusions(tracks, vis, conf)
            engine_label = "Jeff-Tracker + Camera Rigid Inpaint"
        elif a.occlusion_mode == "gap":
            engine_label = "Jeff-Tracker (Matchmove Gap)"
        else:
            engine_label = f"Jeff-Tracker ({a.arch}) Raw"

        if a.polish:
            from jefftrack.polish import polish_tracks_native
            print("[matchmove] applying native-resolution sub-pixel patch polish (< 0.1 px)...")
            tracks = polish_tracks_native(frames, tracks, vis)

        if a.smooth:
            from jefftrack.smooth import smooth_trajectories
            print("[matchmove] applying temporal trajectory smoother...")
            tracks = smooth_trajectories(tracks, vis)

    print(f"[track] {dt:.2f}s  {dt / T:.4f}s/frame  peak VRAM {peak_gb:.2f} GB  visible {100.0 * vis.mean():.1f}%")

    # Export to plate space
    sx, sy = plate_w / float(Ww), plate_h / float(Hw)
    tracks_plate = tracks.copy()
    tracks_plate[..., 0] *= sx
    tracks_plate[..., 1] *= sy

    base = os.path.join(a.out, f"{a.name}__{a.engine}_{a.occlusion_mode}")
    write_3de(base + ".txt", tracks_plate, vis, first_frame, plate_h)
    np.savez_compressed(base + ".npz", tracks=tracks_plate, visibility=vis,
                        confidence=conf, first_frame=first_frame,
                        plate_size=np.array([plate_w, plate_h]))

    if not a.no_render:
        rw = min(a.render_width, Ww)
        rh = int(round(Hw * rw / float(Ww)))
        out_mp4 = base + "_overlay.mp4"
        vw = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (rw, rh))
        colors = track_colors(tracks.shape[1])
        for t in range(T):
            header = f"{engine_label} | F:{first_frame + t} | Vis:{100.0 * vis[t].mean():.0f}%"
            frame = draw_overlay(frames[t], tracks, vis, t, colors, is_inpainted=is_inpainted, header=header,
                                 hide_occluded=a.hide_occluded or (a.occlusion_mode == "gap"),
                                 label=a.label, scale=Ww / float(rw))
            vw.write(cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA) if rw != Ww else frame)
        vw.release()
        print(f"[render] saved overlay video -> {out_mp4}")

    stats = {
        "name": a.name, "engine": a.engine, "occlusion_mode": a.occlusion_mode,
        "frames": int(T), "first_frame": int(first_frame),
        "plate": [int(plate_w), int(plate_h)], "fed": [int(Ww), int(Hw)],
        "seconds": round(dt, 2), "seconds_per_frame": round(dt / T, 4),
        "peak_vram_gb": round(peak_gb, 3),
        "visible_frac": round(float(vis.mean()), 4),
    }
    with open(base + "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print(f"[out] {base}.txt  {base}.npz  {base}_stats.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
