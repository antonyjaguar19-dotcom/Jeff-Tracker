"""DBtracker on a plate directory -> 3DE 2D-track ASCII, npz, overlay, stats.

    python run_dbtrack.py ^
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
                 reports as comparable to CoTracker, and the one to quote numbers from.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from dbtrack_engine import DBTrackEngine, DEFAULT_CKPT  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".dpx")


# --------------------------------------------------------------------------- plate io
def list_frames(plate_dir: str, start: int, end: int):
    files = sorted(f for f in glob.glob(os.path.join(plate_dir, "*"))
                   if os.path.splitext(f)[1].lower() in IMG_EXT)
    if not files:
        raise SystemExit("[ERROR] no frames in {}".format(plate_dir))
    lo = max(1, start) - 1
    hi = len(files) if end in (0, None) else min(end, len(files))
    if hi <= lo:
        raise SystemExit("[ERROR] empty frame range {}..{}".format(start, end))
    return files[lo:hi], lo + 1


def read_frame(path: str, work_w: int):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise SystemExit("[ERROR] cannot read {}".format(path))
    # In-process with torch loaded, cv2 has been seen to return a trailing channel axis
    # where it would not standalone (CLAUDE.md records this for IMREAD_GRAYSCALE); be
    # explicit about the shape rather than trusting it.
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    H, W = img.shape[:2]
    if work_w and work_w < W:
        work_h = int(round(H * work_w / float(W)))
        img = cv2.resize(img, (work_w, work_h), interpolation=cv2.INTER_AREA)
    return img, (W, H)


# --------------------------------------------------------------------------- seeding
def seed_grid(n_side: int, w: int, h: int):
    """Uniform grid, inset from the border so no query sits on the frame edge."""
    ys = np.linspace(0, h - 1, n_side + 2)[1:-1]
    xs = np.linspace(0, w - 1, n_side + 2)[1:-1]
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)


def seed_corners(frame_bgr, n: int):
    """Shi-Tomasi corners -- the seeding the rest of this repo's tooling uses, so the
    resulting tracks sit on the same kind of feature a classical tracker would have chosen."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    pts = cv2.goodFeaturesToTrack(gray, maxCorners=n, qualityLevel=0.01,
                                  minDistance=max(6, min(gray.shape) // 40), blockSize=7)
    if pts is None:
        raise SystemExit("[ERROR] no corners found; try --seed grid")
    return pts.reshape(-1, 2).astype(np.float32)


# --------------------------------------------------------------------------- overlay
def track_colors(n: int):
    hsv = np.zeros((n, 1, 3), np.uint8)
    hsv[:, 0, 0] = (np.arange(n) * 180 // max(1, n)).astype(np.uint8)
    hsv[:, 0, 1] = 255
    hsv[:, 0, 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(n, 3)
    return [tuple(int(c) for c in row) for row in bgr]


def draw_overlay(frame_bgr, tracks, vis, t, colors, tail=12):
    """Points at frame t with a short tail. Occluded points are drawn hollow rather than
    dropped -- a track surviving an occlusion with a hole is the behaviour that matters
    here, so it has to be visible in the render."""
    out = frame_bgr.copy()
    for i in range(tracks.shape[1]):
        col = colors[i]
        t0 = max(0, t - tail)
        seg, sv = tracks[t0:t + 1, i], vis[t0:t + 1, i]
        for k in range(1, len(seg)):
            if sv[k] and sv[k - 1]:
                cv2.line(out, tuple(np.int32(seg[k - 1])), tuple(np.int32(seg[k])),
                         col, 1, cv2.LINE_AA)
        p = tuple(np.int32(tracks[t, i]))
        cv2.circle(out, p, 3, col, -1 if vis[t, i] else 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- 3DE export
def write_3de(path, tracks, vis, first_frame, plate_h, prefix="DBT"):
    """Classic 3DE 2D-track ASCII. Y is flipped: 3DE's origin is bottom-left, OpenCV's is
    top-left. Occluded frames are omitted, which is legal here -- gaps are how this repo
    represents an occlusion rather than deleting the track."""
    T, N, _ = tracks.shape
    with open(path, "w") as fh:
        fh.write("{}\n".format(N))
        for i in range(N):
            frames = [t for t in range(T) if vis[t, i]]
            fh.write("{}_{:04d}\n0\n{}\n".format(prefix, i, len(frames)))
            for t in frames:
                x, y = tracks[t, i]
                fh.write("{} {:.6f} {:.6f}\n".format(first_frame + t, x, plate_h - y))


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description="DBtracker (LocoTrack) on a plate directory")
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
    ap.add_argument("--arch", default="locotrack", choices=["locotrack", "dbtrack"],
                    help="dbtrack adds cross-track attention; zero-initialised it is "
                         "LocoTrack bit for bit, so the two agree until it is trained")
    ap.add_argument("--window", type=int, default=0, help="time window; 0 = auto")
    ap.add_argument("--seed", default="corners", choices=["grid", "corners"])
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--points", type=int, default=400)
    ap.add_argument("--render-width", type=int, default=1920)
    ap.add_argument("--no-render", action="store_true")
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

    eng = DBTrackEngine(device="cuda", model_size=a.model_size, ckpt=a.ckpt,
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

    base = os.path.join(a.out, a.name + "__dbtrack")
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
            frame = draw_overlay(frames[t], tracks, vis, t, colors)
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
