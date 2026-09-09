"""Render what bench3 measured, so the numbers can be checked by eye.

Two outputs, and they answer different questions.

**The mp4** -- three panels, one per engine, same frame, same seeds. Each track is drawn
against its OWN ground truth position, which the synthetic bench knows exactly:

    filled dot   the engine calls this frame visible
    hollow dot   the engine calls it occluded (in 3DE this becomes a hole)
    dot colour   GREEN under 2 px from truth, AMBER 2-5 px, RED over 5 px
    thin line    the tail of the track, last N frames
    white cross  where the point actually is

A number says the worst visible error is 389 px. This says what that looks like: a dot
that leaves and never comes back, while its neighbours keep tracking. That is the failure
mode the whole Phase-1 argument turns on, and it should be watched rather than read.

**The contact sheet** -- the worst tracks in the run, cropped around the moment they go
wrong, one row each. The mp4 shows a shot; this shows the specific frames to look at, so a
600-track run does not have to be scrubbed by hand to find the eight that matter.

    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\render_bench3.py ^
        --shot bench\\synth\\lab02_occ
    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\render_bench3.py --all

Reads the npz that bench3.py already wrote, so it never re-runs a model and costs no GPU.
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

from jefftrack.io import list_frames, read_frame  # noqa: E402
from score_occlusion import load_shot, warp, is_occluded  # noqa: E402
from bench3 import LOCKED_BENCHES  # noqa: E402  (tools/ is on sys.path)

ENGINE_ORDER = ["jefftrack", "tapnext", "cotracker3"]   # overridden by --engines
GREEN, AMBER, RED, WHITE = (90, 220, 90), (60, 190, 250), (60, 60, 255), (255, 255, 255)


def err_colour(e):
    return GREEN if e < 2.0 else (AMBER if e < 5.0 else RED)


def gt_for(tracks, Hs, T):
    """Ground truth for each track, anchored on its own frame-0 position.

    Same anchoring the scorers use: on a plane, a point 1 px from the intended corner is
    still a valid scene point, so what is measured is how well the tracker FOLLOWS it.
    """
    src = warp(np.linalg.inv(Hs[0]), tracks[0].astype(np.float64))
    g = np.zeros((T, len(src), 2))
    for t in range(T):
        g[t] = warp(Hs[t], src)
    return g


def load_engine(out_dir, name, eng, T):
    p = os.path.join(out_dir, "{}__{}.npz".format(name, eng))
    if not os.path.isfile(p):
        return None
    d = np.load(p)
    tr, vis = d["tracks"], d["visibility"].astype(bool)
    n = min(T, tr.shape[0])
    return {"tracks": tr[:n], "vis": vis[:n]}


def panel(frame, tr, vis, gt, t, label, scale, tail):
    out = frame.copy()
    N = tr.shape[1]
    for i in range(N):
        e = float(np.hypot(tr[t, i, 0] - gt[t, i, 0], tr[t, i, 1] - gt[t, i, 1]))
        col = err_colour(e)
        t0 = max(0, t - tail)
        seg = tr[t0:t + 1, i] * scale
        for k in range(1, len(seg)):
            cv2.line(out, tuple(np.int32(seg[k - 1])), tuple(np.int32(seg[k])),
                     col, 1, cv2.LINE_AA)
        p = tuple(np.int32(tr[t, i] * scale))
        g = tuple(np.int32(gt[t, i] * scale))
        # The truth cross is drawn only where the engine has drifted far enough that the
        # two would otherwise overlap into one blob.
        if e >= 2.0:
            cv2.drawMarker(out, g, WHITE, cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
        cv2.circle(out, p, 4, col, -1 if vis[t, i] else 1, cv2.LINE_AA)

    bad = int(sum(1 for i in range(N)
                  if np.hypot(tr[t, i, 0] - gt[t, i, 0], tr[t, i, 1] - gt[t, i, 1]) > 5.0))
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), (0, 0, 0), -1)
    cv2.putText(out, "{}   {} over 5px".format(label, bad), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2, cv2.LINE_AA)
    return out


def render_mp4(path, frames, engines, gts, width, fps, tail):
    T, H, W = frames.shape[:3]
    pw = min(width, W)
    ph = int(round(H * pw / float(W)))
    names = [e for e in ENGINE_ORDER if engines.get(e)]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (pw * len(names), ph))
    # Tracks are stored in plate pixels; the frame here is the fed/rendered size.
    scale = np.array([pw / float(W) * (W / float(W)), ph / float(H) * (H / float(H))])
    for t in range(T):
        cols = []
        for e in names:
            img = panel(frames[t], engines[e]["tracks"], engines[e]["vis"], gts[e], t,
                        e, np.array([1.0, 1.0]), tail)
            if (pw, ph) != (W, H):
                img = cv2.resize(img, (pw, ph), interpolation=cv2.INTER_AREA)
            cols.append(img)
        strip = np.hstack(cols)
        for k in range(1, len(cols)):
            cv2.line(strip, (pw * k, 0), (pw * k, ph), WHITE, 1)
        vw.write(strip)
    vw.release()
    _ = scale


def contact_sheet(path, frames, engines, gts, occ, Hs, n_worst, crop, cell):
    """The worst tracks, cropped at the frame each one goes wrong.

    Ranked by peak error on a frame ground truth says was VISIBLE -- an error inside an
    occlusion is expected and uninteresting, an error in clear view is the flyer.
    """
    T, H, W = frames.shape[:3]
    rows = []
    for eng in [e for e in ENGINE_ORDER if engines.get(e)]:
        tr, vis, gt = engines[eng]["tracks"], engines[eng]["vis"], gts[eng]
        err = np.hypot(tr[..., 0] - gt[..., 0], tr[..., 1] - gt[..., 1])
        occl = np.zeros(err.shape, bool)
        for t in range(T):
            occl[t] = is_occluded(occ, t, gt[t])
        # Rank on clear-view failures only, and only where the truth is still ON the
        # plate. Without the second restriction the worst rows are all points that walked
        # out of frame -- they were 1.2% of samples and carried the entire tail (389.30 px
        # worst visible with them, 16.86 px without), so they would fill this sheet with
        # the one failure that is not the tracker's.
        H_, W_ = frames.shape[1], frames.shape[2]
        on_plate = ((gt[..., 0] >= 0) & (gt[..., 0] < W_) &
                    (gt[..., 1] >= 0) & (gt[..., 1] < H_))
        clear = err.copy()
        clear[occl | ~on_plate] = 0.0
        peak_t = clear.argmax(axis=0)
        peak_e = clear.max(axis=0)
        order = np.argsort(-peak_e)[:n_worst]
        for i in order:
            rows.append((eng, int(i), int(peak_t[i]), float(peak_e[i])))

    if not rows:
        return False
    sheet = np.zeros((cell * len(rows), cell * 3 + 260, 3), np.uint8)
    for r, (eng, i, t, e) in enumerate(rows):
        tr, gt = engines[eng]["tracks"], gts[eng]
        # The crop has to hold BOTH the truth and wherever the tracker actually went, or a
        # 389 px flyer renders as an empty patch captioned "off crop" -- technically true
        # and visually useless. Size it from the row's own peak error, per row, and say
        # what it is so two rows are not read as the same magnification.
        span = int(max(crop, 2.4 * e + 60))
        span = min(span, min(W, H))
        for c, tt in enumerate([max(0, t - 6), t, min(T - 1, t + 6)]):
            cx = 0.5 * (gt[tt, i, 0] + tr[tt, i, 0])
            cy = 0.5 * (gt[tt, i, 1] + tr[tt, i, 1])
            x0, y0 = int(cx - span // 2), int(cy - span // 2)
            x0 = max(0, min(W - span, x0))
            y0 = max(0, min(H - span, y0))
            patch = frames[tt][y0:y0 + span, x0:x0 + span].copy()
            patch = cv2.resize(patch, (cell, cell), interpolation=cv2.INTER_NEAREST)
            k = cell / float(span)
            gp = (int((gt[tt, i, 0] - x0) * k), int((gt[tt, i, 1] - y0) * k))
            pp = (int((tr[tt, i, 0] - x0) * k), int((tr[tt, i, 1] - y0) * k))
            ee = float(np.hypot(tr[tt, i, 0] - gt[tt, i, 0], tr[tt, i, 1] - gt[tt, i, 1]))
            # Black halo under every mark: the plate is pale here and a 1 px white cross on
            # pale skin texture is invisible, which is what the first render did.
            cv2.drawMarker(patch, gp, (0, 0, 0), cv2.MARKER_CROSS, 17, 3, cv2.LINE_AA)
            cv2.drawMarker(patch, gp, WHITE, cv2.MARKER_CROSS, 15, 1, cv2.LINE_AA)
            if 0 <= pp[0] < cell and 0 <= pp[1] < cell:
                if ee >= 2.0:
                    cv2.line(patch, gp, pp, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.line(patch, gp, pp, err_colour(ee), 1, cv2.LINE_AA)
                cv2.circle(patch, pp, 6, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.circle(patch, pp, 6, err_colour(ee), 2, cv2.LINE_AA)
            else:
                cv2.putText(patch, "off crop {:.0f}px".format(ee), (6, cell - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, RED, 1, cv2.LINE_AA)
            cv2.putText(patch, "f{}  {:.0f}px".format(tt, ee), (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(patch, "f{}  {:.0f}px".format(tt, ee), (6, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
            sheet[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = patch
        cv2.putText(sheet, "{}  track {}".format(eng, i),
                    (cell * 3 + 12, r * cell + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    WHITE, 1, cv2.LINE_AA)
        cv2.putText(sheet, "peak {:.1f}px in clear view".format(e),
                    (cell * 3 + 12, r * cell + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    err_colour(e), 1, cv2.LINE_AA)
        span_r = min(int(max(crop, 2.4 * e + 60)), min(W, H))
        cv2.putText(sheet, "at frame {}   crop {}px".format(t, span_r),
                    (cell * 3 + 12, r * cell + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                    (170, 170, 170), 1, cv2.LINE_AA)
        cv2.putText(sheet, "cross = truth   circle = tracker",
                    (cell * 3 + 12, r * cell + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    (130, 130, 130), 1, cv2.LINE_AA)
    cv2.imwrite(path, sheet)
    return True


def do_shot(shot_dir, out_dir, width, fps, tail, n_worst, crop, cell, no_mp4, tag=""):
    name = os.path.basename(shot_dir.rstrip("/\\"))
    gt_json, Hs, occ = load_shot(shot_dir)
    files, _ = list_frames(os.path.join(shot_dir, "plate"), 1, 0)

    # Render at plate resolution scaled to the panel width, so the drawn positions and the
    # stored plate-pixel tracks share one coordinate system and nothing needs rescaling.
    frames = []
    for f in files:
        img, _ = read_frame(f, 0)
        frames.append(img)
    frames = np.stack(frames)
    T = len(frames)

    engines, gts = {}, {}
    for eng in ENGINE_ORDER:
        d = load_engine(out_dir, name, eng, T)
        if d is None:
            print("[{}] no npz for {} -- run bench3.py first".format(name, eng))
            continue
        engines[eng] = d
        gts[eng] = gt_for(d["tracks"], Hs, min(T, d["tracks"].shape[0]))
    if not engines:
        return

    T = min(T, min(e["tracks"].shape[0] for e in engines.values()))
    frames = frames[:T]

    suffix = ("_" + tag) if tag else ""
    sheet = os.path.join(out_dir, "{}__worst{}.png".format(name, suffix))
    if contact_sheet(sheet, frames, engines, gts, occ, Hs, n_worst, crop, cell):
        print("[{}] {}".format(name, sheet))

    if not no_mp4:
        mp4 = os.path.join(out_dir, "{}__3way{}.mp4".format(name, suffix))
        render_mp4(mp4, frames, engines, gts, width, fps, tail)
        print("[{}] {}".format(name, mp4))


def main() -> int:
    global ENGINE_ORDER
    ap = argparse.ArgumentParser(description="render what bench3 measured")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--all", action="store_true", help="the locked bench set")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "bench3"))
    ap.add_argument("--width", type=int, default=860, help="per-panel width")
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--tail", type=int, default=12)
    ap.add_argument("--worst", type=int, default=3, help="worst tracks per engine")
    ap.add_argument("--crop", type=int, default=200, help="crop size in plate pixels")
    ap.add_argument("--cell", type=int, default=190)
    ap.add_argument("--no-mp4", action="store_true", help="contact sheet only")
    ap.add_argument("--tag", default="",
                    help="suffix for the output names. Without it a later phase silently "
                         "overwrites an earlier phase's sheets -- they share a filename.")
    ap.add_argument("--engines", default="jefftrack,tapnext,cotracker3",
                    help="comma list of npz suffixes to render, in panel order. A config "
                         "variant (jefftrack_r384x512) is its own entry.")
    a = ap.parse_args()
    ENGINE_ORDER = [e.strip() for e in a.engines.split(",") if e.strip()]

    if a.all:
        shots = [os.path.join(REPO, "bench", "synth", b) for b in LOCKED_BENCHES]
    elif a.shot:
        shots = [a.shot if os.path.isabs(a.shot) else os.path.join(REPO, a.shot)]
    else:
        ap.error("--shot or --all is required")

    for s in shots:
        do_shot(s, a.out, a.width, a.fps, a.tail, a.worst, a.crop, a.cell,
                a.no_mp4, a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
