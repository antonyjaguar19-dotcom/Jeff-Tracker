"""Side-by-side GIF of two or three trackers on the same DAVIS clip, same seeds.

    python tools/make_compare.py --clip dance-twirl --out assets/compare.gif \
        --tapnext-root ... --tapnext-engine ... --cotracker ... --cotracker-ckpt ...

Every panel is handed the identical frames and the identical query points, and every panel
is drawn by the same overlay code, so a visible difference is a difference between the
trackers rather than between two renderers.

The overlay follows the same rule as tools/make_demo.py: a point is only ever drawn at a
position its model has committed to. On a frame a model calls occluded, the position it
emits is unconstrained, and drawing that produces markers flying across the picture --
noise presented as output. Occluded points are held at their last committed position and
retired after `--drop-after` frames.

**CoTracker3 is CC-BY-NC-4.0** and is not vendored here; it is only used if you point this
at your own copy. TAPNext++ is Apache-2.0.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# HERE too: this reuses make_demo and benchmark, which are siblings, not a package.
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from jefftrack.io import seed_grid, track_colors  # noqa: E402
from make_demo import load_clip, save_gif  # noqa: E402
from benchmark import (  # noqa: E402
    CoTracker3Adapter, JeffTrackerAdapter, TapNextAdapter)

TITLES = {
    "jefftracker": "Jeff-Tracker  (Apache-2.0)",
    "tapnext": "TAPNext++  (Apache-2.0)",
    "cotracker3": "CoTracker3  (CC-BY-NC)",
}


def panel(frame_bgr, tracks, vis, t, colors, title, tail=8, drop_after=8,
          max_step_frac=0.2):
    out = frame_bgr.copy()
    W = out.shape[1]
    max_step = max_step_frac * W
    for i in range(tracks.shape[1]):
        seen = vis[:t + 1, i]
        if not seen.any():
            continue
        last = int(np.nonzero(seen)[0][-1])
        gap = t - last
        if gap > drop_after:
            continue
        col = colors[i]
        for k in range(max(1, t - tail + 1), t + 1):
            if vis[k, i] and vis[k - 1, i]:
                a, b = tracks[k - 1, i], tracks[k, i]
                if float(np.hypot(b[0] - a[0], b[1] - a[1])) <= max_step:
                    cv2.line(out, tuple(np.int32(a)), tuple(np.int32(b)), col, 1,
                             cv2.LINE_AA)
        p = tuple(np.int32(tracks[last, i]))
        if gap == 0:
            cv2.circle(out, p, 3, col, -1, cv2.LINE_AA)
        else:
            cv2.circle(out, p, 4, (70, 70, 235), 1, cv2.LINE_AA)

    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(out, title, (7, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="side-by-side tracker comparison GIF")
    ap.add_argument("--clip", default="dance-twirl")
    ap.add_argument("--pkl", default=os.path.join(ROOT, "data", "tapvid_davis",
                                                  "tapvid_davis.pkl"))
    ap.add_argument("--models", default="jefftracker,tapnext,cotracker3")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "inf_s4000.ckpt"))
    ap.add_argument("--tapnext-root", default="")
    ap.add_argument("--tapnext-engine", default="")
    ap.add_argument("--cotracker", default="")
    ap.add_argument("--cotracker-ckpt", default="")
    ap.add_argument("--grid", type=int, default=12)
    ap.add_argument("--frames", type=int, default=50)
    ap.add_argument("--tail", type=int, default=6)
    ap.add_argument("--drop-after", type=int, default=8)
    ap.add_argument("--panel-width", type=int, default=340)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--colors", type=int, default=64)
    a = ap.parse_args()

    rgb = load_clip(a.pkl, a.clip)
    if a.frames:
        rgb = rgb[:a.frames]
    frames = np.ascontiguousarray(rgb[:, :, :, ::-1])          # RGB -> BGR
    T, H, W = frames.shape[:3]
    print("[cmp] {} {} frames {}x{}".format(a.clip, T, W, H))

    pts = seed_grid(a.grid, W, H)
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1).astype(np.float32)
    colors = track_colors(len(pts))
    print("[cmp] {} shared seeds".format(len(pts)))

    results = []
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        if m == "jefftracker":
            ad = JeffTrackerAdapter(a.ckpt)
        elif m == "tapnext":
            ad = TapNextAdapter(a.tapnext_root, a.tapnext_engine or None)
        elif m == "cotracker3":
            ad = CoTracker3Adapter(a.cotracker, a.cotracker_ckpt)
        else:
            raise SystemExit("[ERROR] unknown model {!r}".format(m))
        tr, vs = ad.track(frames, q)
        print("[cmp] {:<11} {:.1f}% visible".format(m, 100.0 * vs.mean()))
        results.append((m, tr, vs))
        del ad
        try:
            import torch                                       # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:                                      # noqa: BLE001
            pass

    pw = a.panel_width
    ph = int(round(H * pw / float(W)))
    out = []
    for t in range(T):
        cells = []
        for m, tr, vs in results:
            img = panel(frames[t], tr, vs, t, colors, TITLES.get(m, m),
                        tail=a.tail, drop_after=a.drop_after)
            cells.append(cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA))
        strip = np.hstack(cells)
        for i in range(1, len(cells)):
            cv2.line(strip, (i * pw, 0), (i * pw, ph), (255, 255, 255), 1)
        out.append(strip[:, :, ::-1])
    save_gif(a.out, out, a.fps, a.colors)
    print("[cmp] wrote {} ({:.1f} MB)".format(a.out, os.path.getsize(a.out) / 1048576.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
