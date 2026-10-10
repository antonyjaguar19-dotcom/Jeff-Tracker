"""Plate I/O, seeding, overlay drawing and 3DE 2D-track export.

Split out of the CLI so the same helpers back `tools/run_jefftrack.py`, the evaluation
scripts and the demo renderer, rather than three copies drifting apart.
"""
from __future__ import annotations

import glob
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


IMG_EXT = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".dpx")
VIDEO_EXT = (".mp4", ".mov", ".avi", ".mkv", ".webm")


# --------------------------------------------------------------------------- plate io
def load_plate_frames(plate_path: str, start: int = 1, end: int = 0, work_w: int = 0):
    """Load frames from either a directory of image files or a video file (.mp4, .mov, etc.).

    Returns (frames (T,H,W,3) uint8 BGR, (plate_w, plate_h), first_frame).
    """
    if os.path.isfile(plate_path) and os.path.splitext(plate_path)[1].lower() in VIDEO_EXT:
        cap = cv2.VideoCapture(plate_path)
        if not cap.isOpened():
            raise SystemExit("[ERROR] cannot open video: {}".format(plate_path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        plate_wh = (orig_w, orig_h)
        lo = max(1, start) - 1
        hi = total if end in (0, None) else min(end, total)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
        for _ in range(hi - lo):
            ret, fr = cap.read()
            if not ret or fr is None:
                break
            if work_w and work_w < orig_w:
                work_h = int(round(orig_h * work_w / float(orig_w)))
                fr = cv2.resize(fr, (work_w, work_h), interpolation=cv2.INTER_AREA)
            frames.append(fr)
        cap.release()
        if not frames:
            raise SystemExit("[ERROR] no frames decoded from {}".format(plate_path))
        return np.stack(frames), plate_wh, lo + 1

    # Directory of frames
    files, first_frame = list_frames(plate_path, start, end)
    frames, plate_wh = [], None
    for f in files:
        img, wh = read_frame(f, work_w)
        plate_wh = wh
        frames.append(img)
    return np.stack(frames), plate_wh, first_frame


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


def draw_hud(out, text, scale=1.0):
    """Burn the frame number into the corner.

    Without it a track number is only half an address: a fault is "track 26 drifts", which
    cannot be looked up, rather than "track 26 drifts at frame 145", which can. Drawn on a
    filled box because a plate corner is as likely to be white sky as black shadow.
    """
    fs = 0.7 * scale
    th = max(1, int(round(2 * scale)))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = int(round(8 * scale))
    cv2.rectangle(out, (pad, pad), (pad * 2 + tw, pad * 2 + tht), (0, 0, 0), -1)
    cv2.putText(out, text, (pad + pad // 2, pad + tht + pad // 4),
                cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), th, cv2.LINE_AA)
    return out


def draw_overlay(frame_bgr, tracks, vis, t, colors, tail=12,
                 is_inpainted=None, header="", gap_occluded=False,
                 hide_occluded=False, label=False, prefix="JT", scale=1.0):
    """Points at frame t with a short tail.

    - Solid circle: visible and confident.
    - Hollow ring: raw occluded point (when gap_occluded=False and hide_occluded=False).
    - Double ring with crosshair: rigidly inpainted occlusion (when is_inpainted is set).
    - If gap_occluded=True or hide_occluded=True: occluded points are ignored/hidden completely.
    """
    out = frame_bgr.copy()
    h, w = out.shape[:2]
    fs = 0.34 * scale

    for i in range(tracks.shape[1]):
        shown = bool(vis[t, i])
        if (hide_occluded or gap_occluded) and not shown:
            continue
        col = colors[i]
        t0 = max(0, t - tail)
        seg, sv = tracks[t0:t + 1, i], vis[t0:t + 1, i]
        for k in range(1, len(seg)):
            if (gap_occluded or hide_occluded) and (not sv[k - 1] or not sv[k]):
                continue
            cv2.line(out, tuple(np.int32(seg[k - 1])), tuple(np.int32(seg[k])),
                     col, 1, cv2.LINE_AA)
        p = tuple(np.int32(tracks[t, i]))
        if 0 <= p[0] < w and 0 <= p[1] < h:
            if is_inpainted is not None and is_inpainted[t, i]:
                # Inpainted rigid track: double ring
                cv2.circle(out, p, 5, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.circle(out, p, 2, col, -1, cv2.LINE_AA)
            elif vis[t, i]:
                cv2.circle(out, p, 3, col, -1, cv2.LINE_AA)
            elif not (gap_occluded or hide_occluded):
                cv2.circle(out, p, 4, col, 1, cv2.LINE_AA)

            if label and shown:
                org = (p[0] + 5, p[1] - 5)
                cv2.putText(out, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0),
                            3, cv2.LINE_AA)
                cv2.putText(out, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, fs, col,
                            1, cv2.LINE_AA)

    if header:
        # Semi-transparent banner
        banner_h = 36
        overlay_banner = out[:banner_h, :].copy()
        cv2.rectangle(out, (0, 0), (w, banner_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay_banner, 0.35, out[:banner_h, :], 0.65, 0, out[:banner_h, :])
        cv2.putText(out, header, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (240, 240, 240), 2, cv2.LINE_AA)

    return out


# --------------------------------------------------------------------------- 3DE export
def write_3de(path, tracks, vis, first_frame, plate_h, prefix="JT"):
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


def read_3de(path):
    """Read back what write_3de wrote: {track_name: {frame: (x, y)}} in 3DE coordinates.

    The mirror of write_3de, and the reason it exists here rather than in a tool: two
    tools needed it and both reached into the studio application this model was extracted
    from (`from app.compare_tracks import load_tracks`), which is not part of this
    repository. On a clean clone they raised ModuleNotFoundError, so the documented
    verification steps could not run at all.

    Y is NOT flipped on the way back -- callers that need raster coordinates flip with the
    plate height, the same way write_3de flipped on the way out. Gaps are expected: a track
    occluded for part of the shot simply has no entry for those frames.
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        tok = fh.read().split()
    i = 0
    n = int(tok[i]); i += 1
    out = {}
    for _ in range(n):
        name = tok[i]; i += 1
        i += 1                                   # colour id, unused
        count = int(tok[i]); i += 1
        pts = {}
        for _ in range(count):
            pts[int(tok[i])] = (float(tok[i + 1]), float(tok[i + 2]))
            i += 3
        out[name] = pts
    return out
