"""Jeff-Tracker on a plate directory -> 3DE 2D-track ASCII, npz, overlay, stats.

    python run_jefftrack.py ^
        --plate C:\\path\\to\\plate --name myshot

It vendors LocoTrack under vendor/ (Apache-2.0, see
LICENSES.md) and reads weights from weights/.

Two resolutions matter and they are not the same knob:

  --work-width   what the frames are decoded and held at. Only saves host RAM; the model
                 never sees this resolution.
  --model-res    what LocoTrack actually runs at, and therefore what bounds precision.
                 The model was trained at 256x256, and feeding it more makes
                 get_feature_grids() build a ladder of refinement resolutions rather than
                 just upscaling. On a 2560-wide plate one model pixel at 256x256 is TEN
                 plate pixels, so sub-pixel plate accuracy is simply not reachable there
                 no matter how good the tracker is -- 384x512 is the setting the paper
                 reports as its comparable setting, and the one to quote numbers from.
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
    draw_hud, draw_overlay, list_frames, read_frame, seed_corners, seed_grid,
    track_colors,
    write_3de,
)


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="Jeff-Tracker (LocoTrack) on a plate directory")
    ap.add_argument("--plate", required=True)
    ap.add_argument("--name", default="shot")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--model-res", default="256x256",
                    help="LocoTrack input resolution HxW; this is what bounds precision")
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--arch", default="locotrack", choices=["locotrack", "jefftrack"],
                    help="jefftrack adds cross-track attention; zero-initialised it is "
                         "LocoTrack bit for bit, so the two agree until it is trained")
    ap.add_argument("--window", type=int, default=0, help="time window; 0 = auto")
    ap.add_argument("--seed", default="corners", choices=["grid", "corners"])
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--points", type=int, default=400)
    ap.add_argument("--render-width", type=int, default=1920)
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
    ap.add_argument("--fps", type=float, default=24.0)
    a = ap.parse_args()

    model_res = tuple(int(v) for v in a.model_res.lower().split("x"))
    os.makedirs(a.out, exist_ok=True)
    files, first_frame = list_frames(a.plate, a.start, a.end)

    t_load = time.time()
    frames, plate_wh = [], None
    for f in files:
        img, wh = read_frame(f, a.work_width)
        plate_wh = wh
        frames.append(img)
    frames = np.stack(frames)
    T, Hw, Ww = frames.shape[:3]
    plate_w, plate_h = plate_wh
    print("[plate] {} frames {}..{}  plate {}x{}  fed at {}x{}  ({:.1f}s)".format(
        T, first_frame, first_frame + T - 1, plate_w, plate_h, Ww, Hw,
        time.time() - t_load))

    pts = (seed_grid(a.grid, Ww, Hw) if a.seed == "grid"
           else seed_corners(frames[0], a.points))
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)
    print("[seed] {} points ({})".format(len(pts), a.seed))

    eng = JeffTrackEngine(device="cuda", model_size=a.model_size, ckpt=a.ckpt,
                        model_res=model_res, window=a.window, arch=a.arch)
    if eng.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tracks, vis, conf = eng.track_queries_conf(frames, q)
    dt = time.time() - t0
    peak_gb = (torch.cuda.max_memory_allocated() / 1e9) if eng.device == "cuda" else 0.0
    print("[track] {:.1f}s  {:.3f}s/frame  peak VRAM {:.2f} GB  visible {:.1f}%".format(
        dt, dt / T, peak_gb, 100.0 * vis.mean()))

    # Everything above ran in the fed frame's pixel space; the export is plate space.
    sx, sy = plate_w / float(Ww), plate_h / float(Hw)
    tracks_plate = tracks.copy()
    tracks_plate[..., 0] *= sx
    tracks_plate[..., 1] *= sy

    base = os.path.join(a.out, a.name + "__jefftrack")
    write_3de(base + ".txt", tracks_plate, vis, first_frame, plate_h)
    np.savez_compressed(base + ".npz", tracks=tracks_plate, visibility=vis,
                        confidence=conf, first_frame=first_frame,
                        plate_size=np.array([plate_w, plate_h]))

    if not a.no_render:
        rw = min(a.render_width, Ww)
        rh = int(round(Hw * rw / float(Ww)))
        vw = cv2.VideoWriter(base + "_overlay.mp4",
                             cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (rw, rh))
        colors = track_colors(tracks.shape[1])
        for t in range(T):
            frame = draw_overlay(frames[t], tracks, vis, t, colors,
                                 hide_occluded=a.hide_occluded, label=a.label,
                                 scale=Ww / float(rw))
            # The SHOT's frame number, not the clip index: --start offsets the run and a
            # number that disagrees with 3DE is worse than none.
            draw_hud(frame, "frame {}   {} tracks".format(
                first_frame + t, int(vis[t].sum())), scale=Ww / float(rw))
            vw.write(cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_AREA)
                     if rw != Ww else frame)
        vw.release()

    stats = {
        "name": a.name, "frames": int(T), "first_frame": int(first_frame),
        "plate": [int(plate_w), int(plate_h)], "fed": [int(Ww), int(Hw)],
        "model": {"arch": a.arch, "size": a.model_size, "res": list(model_res),
                  "window": int(eng._auto_window(T)), "ckpt": os.path.basename(a.ckpt)},
        "seed": {"mode": a.seed, "points": int(len(pts))},
        "seconds": round(dt, 2), "seconds_per_frame": round(dt / T, 4),
        "peak_vram_gb": round(peak_gb, 3),
        "visible_frac": round(float(vis.mean()), 4),
        "mean_confidence": round(float(conf.mean()), 4),
    }
    with open(base + "_stats.json", "w") as fh:
        json.dump(stats, fh, indent=2)
    print("[out] {}.txt  .npz  {}_stats.json".format(base, base))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
