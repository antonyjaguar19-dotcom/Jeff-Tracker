"""Score DBtracker against a reference by seeding it ON the reference points.

proximity-paired scorers pair tracker output to reference tracks by proximity, which is the right
thing when scoring a real tracker export -- the tracker picked its own features and nobody told it
about the references. It is the wrong thing for measuring THIS model: a corner seeded 8 px
from the reference feature is a different feature, and the deviation then reports that
offset rather than any tracking error. Measured on a real plate, that hid the real number
completely -- pushing the model from 256x256 to 384x682 moved the paired deviation by
0.6 px, which is not what a 2.7x resolution change does to a tracker.

So this seeds each reference's own first position, tracks it, and compares the two
trajectories frame by frame. One row per reference, labelled by feature kind, same shape
as eval_refs -- an averaged single number hides a change that helps corners and hurts
blobs.

    python eval_vs_manual.py ^
        --ref refs\\myshot_lk --plate C:\\path\\to\\plate

Read the reference's own precision before believing any row: refs/*_lk are pyramidal
Lucas-Kanade round trips, not artist hand tracks, and reference.json records the closure
per track. A deviation below the closure is not an improvement, it is noise.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
REPO = ROOT
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from app.compare_tracks import load_tracks  # noqa: E402
from dbtrack.engine import DBTrackEngine, DEFAULT_CKPT  # noqa: E402
from dbtrack.io import list_frames, read_frame  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="DBtracker vs a reference, seeded on it")
    ap.add_argument("--ref", required=True, help="reference folder (manual.txt, refs.json)")
    ap.add_argument("--plate", required=True)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--model-res", default="256x256")
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--window", type=int, default=0)
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--end", type=int, default=0)
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()

    model_res = tuple(int(v) for v in a.model_res.lower().split("x"))
    manual = load_tracks(os.path.join(a.ref, "manual.txt"))
    kinds = {}
    kj = os.path.join(a.ref, "refs.json")
    if os.path.isfile(kj):
        with open(kj) as fh:
            kinds = json.load(fh)
    closure = {}
    rj = os.path.join(a.ref, "reference.json")
    if os.path.isfile(rj):
        with open(rj) as fh:
            closure = json.load(fh).get("closure_px", {})

    files, first_frame = list_frames(a.plate, a.start, a.end)
    frames, plate_wh = [], None
    for f in files:
        img, wh = read_frame(f, a.work_width)
        plate_wh = wh
        frames.append(img)
    frames = np.stack(frames)
    T, Hw, Ww = frames.shape[:3]
    plate_w, plate_h = plate_wh
    sx, sy = Ww / float(plate_w), Hw / float(plate_h)

    # Seed each reference at its own first frame. 3DE ASCII is bottom-left origin, the
    # tracker is top-left: y must be flipped going in, exactly as write_3de flips it
    # coming out. Getting this backwards produces tracks that look fine and score
    # nonsense -- a y-flip probe against a control caught exactly this bug once.
    names, seeds, starts = [], [], []
    for name, tr in sorted(manual.items()):
        fr = sorted(tr)
        f0 = fr[0]
        idx = f0 - first_frame
        if idx < 0 or idx >= T:
            print("[skip] {} starts at frame {}, outside {}..{}".format(
                name, f0, first_frame, first_frame + T - 1))
            continue
        x3, y3 = tr[f0]
        names.append(name)
        starts.append(idx)
        seeds.append([float(idx), x3 * sx, (plate_h - y3) * sy])
    if not names:
        raise SystemExit("[ERROR] no reference track starts inside the frame range")
    q = np.asarray(seeds, np.float32)

    eng = DBTrackEngine(device="cuda", model_size=a.model_size, ckpt=a.ckpt,
                        model_res=model_res, window=a.window)
    if eng.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tracks, vis, conf = eng.track_queries_conf(frames, q)
    dt = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9) if eng.device == "cuda" else 0.0

    print("reference {}   plate {}x{} fed at {}x{}   model {} {}   {} tracks".format(
        a.ref, plate_w, plate_h, Ww, Hw, a.model_size, model_res, len(names)))
    print("{:.1f}s  {:.3f}s/frame  peak VRAM {:.2f} GB".format(dt, dt / T, peak))
    print()
    print("{:<8}{:<9}{:>9}{:>9}{:>9}{:>9}{:>9}".format(
        "track", "class", "mean_px", "med_px", "p95_px", "max_px", "closure"))
    print("-" * 62)

    rows = []
    for i, name in enumerate(names):
        tr = manual[name]
        errs = []
        for f, (x3, y3) in tr.items():
            idx = f - first_frame
            if idx < 0 or idx >= T or not vis[idx, i]:
                continue
            gx, gy = x3 * sx, (plate_h - y3) * sy
            dx = tracks[idx, i, 0] - gx
            dy = tracks[idx, i, 1] - gy
            errs.append((dx * dx + dy * dy) ** 0.5)
        if not errs:
            print("{:<8}{:<9}{:>9}".format(name, kinds.get(name, "?"), "NO OVERLAP"))
            continue
        e = np.asarray(errs)
        # Errors are measured in FED pixels; report in plate pixels, which is the space
        # the artist and 3DE both work in.
        e = e / sx
        rows.append((name, kinds.get(name, "?"), e))
        print("{:<8}{:<9}{:>9.2f}{:>9.2f}{:>9.2f}{:>9.2f}{:>9}".format(
            name, kinds.get(name, "?"), e.mean(), np.median(e),
            np.percentile(e, 95), e.max(),
            "{:.2f}".format(closure[name]) if name in closure else "-"))

    if rows:
        allm = np.array([r[2].mean() for r in rows])
        print()
        print("scored {}/{}   mean {:.2f}px   median-of-means {:.2f}px".format(
            len(rows), len(names), allm.mean(), np.median(allm)))
        by = {}
        for n, k, e in rows:
            by.setdefault(k, []).append(e.mean())
        for k in sorted(by):
            print("  {:<8} n={}  mean {:.2f}px".format(k, len(by[k]), np.mean(by[k])))
        if closure:
            c = np.mean([closure[n] for n, _, _ in rows if n in closure])
            print("  reference closure mean {:.2f}px -- deviations near this are noise, "
                  "not accuracy".format(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
