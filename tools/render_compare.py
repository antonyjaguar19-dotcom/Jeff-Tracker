"""Side-by-side overlay of two runs on the SAME plate, from the npz they already wrote.

For real footage, where there is no ground truth. It cannot colour a track by how wrong it
is -- nothing here knows that -- so it shows what each engine committed to and lets the eye
do the judging:

    filled dot   the engine calls this frame visible
    hollow dot   it calls the point occluded; in the 3DE export this becomes a hole
    trail        the last N frames of the track

A track that slides off its feature, or a dot that pops back on the wrong corner after
something passes in front of it, is obvious in motion and invisible in a table. That is the
whole reason this exists alongside the numbers.

    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\render_compare.py ^
        --plate experiments\\blender_track\\out\\SH006\\plate ^
        --left out\\SH006_256x256__jefftrack.npz  --left-label "256x256" ^
        --right out\\SH006_384x512__jefftrack.npz --right-label "384x512" ^
        --out out\\SH006_res_compare.mp4
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
REPO = ROOT
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from jefftrack.io import list_frames, read_frame, track_colors  # noqa: E402


def panel(frame, tr, vis, t, colors, label, scale, tail):
    out = frame.copy()
    for i in range(tr.shape[1]):
        col = colors[i]
        t0 = max(0, t - tail)
        seg = tr[t0:t + 1, i] * scale
        v = vis[t0:t + 1, i]
        for k in range(1, len(seg)):
            if v[k] and v[k - 1]:
                cv2.line(out, tuple(np.int32(seg[k - 1])), tuple(np.int32(seg[k])),
                         col, 1, cv2.LINE_AA)
        p = tuple(np.int32(tr[t, i] * scale))
        cv2.circle(out, p, 4, col, -1 if vis[t, i] else 1, cv2.LINE_AA)
    n_vis = int(vis[t].sum())
    cv2.rectangle(out, (0, 0), (out.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(out, "{}   {}/{} committed".format(label, n_vis, tr.shape[1]),
                (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="side-by-side overlay of two runs")
    ap.add_argument("--plate", required=True)
    ap.add_argument("--left", required=True)
    ap.add_argument("--right", required=True)
    ap.add_argument("--left-label", default="A")
    ap.add_argument("--right-label", default="B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--width", type=int, default=940, help="per-panel width")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--tail", type=int, default=14)
    ap.add_argument("--frames", type=int, default=0, help="cap, 0 = all")
    a = ap.parse_args()

    L, R = np.load(a.left), np.load(a.right)
    trL, viL = L["tracks"], L["visibility"].astype(bool)
    trR, viR = R["tracks"], R["visibility"].astype(bool)

    files, _ = list_frames(a.plate, 1, 0)
    T = min(len(files), trL.shape[0], trR.shape[0])
    if a.frames:
        T = min(T, a.frames)

    img0, (pw, ph) = read_frame(files[0], 0)
    W = min(a.width, pw)
    H = int(round(ph * W / float(pw)))
    # Tracks are stored in PLATE pixels; the panel is a scaled plate, so one factor.
    scale = np.array([W / float(pw), H / float(ph)])

    colors = track_colors(trL.shape[1])
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W * 2, H))
    for t in range(T):
        img, _ = read_frame(files[t], 0)
        small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        left = panel(small, trL, viL, t, colors, a.left_label, scale, a.tail)
        right = panel(small, trR, viR, t, colors, a.right_label, scale, a.tail)
        strip = np.hstack([left, right])
        cv2.line(strip, (W, 0), (W, H), (255, 255, 255), 1)
        vw.write(strip)
    vw.release()
    print("[mp4] {}  ({} frames, {}x{})".format(a.out, T, W * 2, H))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
