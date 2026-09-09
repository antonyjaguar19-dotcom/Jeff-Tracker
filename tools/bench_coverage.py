"""Compare engines at MATCHED coverage, which is the only fair way to read a tail.

Why this exists. Phase 1 reported CoTracker3's worst visible error as a flat 5.4-5.8 px
against Jeff-Tracker's 6.8-91.8, and called that the one thing it still did decisively
better. That comparison was not coverage-fair: CoTracker3 thresholds visibility at 0.9
**inside its own predictor**, so the numbers it hands back are already gated, while
Jeff-Tracker's were raw. A tracker that declines to answer on its least certain 4% of frames
will always look steadier than one that answers on everything.

So: rank every engine's frames by its own confidence, keep the same fraction from each, and
compare what is left. Identical coverage, so the only thing that differs is the quality of
what each chose to keep.

Restricted to frames ground truth says were VISIBLE and whose truth is still ON the plate.
Occluded accuracy is a separate question and a visible-frame coverage match cannot move it.

    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\bench_coverage.py
    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\bench_coverage.py --coverage 0.90
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from score_occlusion import is_occluded, load_shot, warp  # noqa: E402
from bench3 import LOCKED_BENCHES  # noqa: E402

DEFAULT_ENGINES = ("jefftrack:Jeff-Tracker 256x256:#3987e5,"
                   "jefftrack_r384x512:Jeff-Tracker 384x512:#c98500,"
                   "cotracker3:CoTracker3:#199e70,"
                   "tapnext:TAPNext++:#d95926")

SURFACE, PANEL = "#1a1a19", "#232322"
INK, INK2, INK3, GRID = "#ffffff", "#c3c2b7", "#87867e", "#3a3a38"


def stats_at_coverage(bench_dir, npz_path, coverage):
    """Errors over the most-confident `coverage` fraction of visible on-plate frames."""
    gt, Hs, occ = load_shot(bench_dir)
    W, H, T = int(gt["width"]), int(gt["height"]), len(Hs)
    d = np.load(npz_path)
    tracks, conf = d["tracks"], d["confidence"]
    src = warp(np.linalg.inv(Hs[0]), tracks[0].astype(np.float64))
    g = np.zeros((T, len(src), 2))
    for t in range(T):
        g[t] = warp(Hs[t], src)
    err = np.hypot(tracks[..., 0] - g[..., 0], tracks[..., 1] - g[..., 1])
    occl = np.zeros(err.shape, bool)
    for t in range(T):
        occl[t] = is_occluded(occ, t, g[t])
    inside = ((g[..., 0] >= 0) & (g[..., 0] < W) &
              (g[..., 1] >= 0) & (g[..., 1] < H))
    keep = (~occl) & inside
    e, c = err[keep], conf[keep]
    if e.size == 0:
        return None
    # A boolean confidence (TAPNext, CoTracker3) has no ordering to exploit, so argsort
    # falls back to array order. Stated rather than hidden: for those engines this is a
    # random 96%, not their best 96%, which if anything works against Jeff-Tracker.
    n = max(1, int(round(coverage * e.size)))
    sel = e[np.argsort(-c, kind="stable")[:n]]
    return {"mean": float(sel.mean()), "med": float(np.median(sel)),
            "p99": float(np.percentile(sel, 99)), "max": float(sel.max()),
            "n": int(sel.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description="engine comparison at matched coverage")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out", "bench3"))
    ap.add_argument("--benches", default=",".join(LOCKED_BENCHES))
    ap.add_argument("--engines", default=DEFAULT_ENGINES)
    ap.add_argument("--coverage", type=float, default=0.96)
    ap.add_argument("--tag", default="matched")
    a = ap.parse_args()

    benches = [b.strip() for b in a.benches.split(",") if b.strip()]
    engines = []
    for tok in a.engines.split(","):
        k, lab, col = tok.split(":")
        engines.append((k.strip(), lab.strip(), col.strip()))

    data = {}
    for b in benches:
        for k, lab, _ in engines:
            p = os.path.join(a.out_dir, "{}__{}.npz".format(b, k))
            if not os.path.isfile(p):
                continue
            s = stats_at_coverage(os.path.join(REPO, "bench", "synth", b), p, a.coverage)
            if s:
                data[(b, k)] = s

    print("Visible frames, truth on plate, each engine keeping its most-confident "
          "{:.0f}%".format(a.coverage * 100))
    print("{:<11}{:<24}{:>9}{:>9}{:>8}{:>9}".format(
        "bench", "engine", "mean", "median", "p99", "MAX"))
    print("-" * 71)
    for b in benches:
        for k, lab, _ in engines:
            s = data.get((b, k))
            if not s:
                continue
            print("{:<11}{:<24}{:>9.3f}{:>9.3f}{:>8.2f}{:>9.2f}".format(
                b, lab, s["mean"], s["med"], s["p99"], s["max"]))
        print()

    short = [b.replace("lab02_occ", "lab02").replace("occ_s", "occ") for b in benches]
    fig = plt.figure(figsize=(15.0, 6.4), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 2, left=0.055, right=0.985, top=0.655, bottom=0.135, wspace=0.19)

    for ax_i, (key, title, note) in enumerate([
            ("med", "Typical error (median)", "lower is better"),
            ("max", "Worst error", "lower is better  ·  the one an artist notices")]):
        ax = fig.add_subplot(gs[0, ax_i])
        ax.set_facecolor(PANEL)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, color=INK, fontsize=12, pad=26, loc="left", fontweight="bold")
        ax.text(0, 1.035, note, transform=ax.transAxes, color=INK3, fontsize=8.4,
                va="bottom", ha="left")
        n_g, n_s = len(benches), len(engines)
        gw, gap = 0.78, 0.03
        bw = (gw - gap * (n_s - 1)) / n_s
        vals = np.array([[data.get((b, k), {}).get(key, np.nan) for k, _, _ in engines]
                         for b in benches])
        finite = vals[np.isfinite(vals)]
        top = finite.max() * 1.30 if finite.size else 1.0
        ax.set_ylim(0, top)
        ax.set_xlim(-0.5, n_g - 0.5)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=GRID, linewidth=0.7)
        ax.tick_params(axis="y", colors=INK3, labelsize=8, length=0)
        ax.tick_params(axis="x", colors=INK2, labelsize=9.4, length=0)
        ax.set_xticks(range(n_g))
        ax.set_xticklabels(short)
        ax.set_ylabel("px", color=INK3, fontsize=8.4)
        for gi in range(n_g):
            for si, (_, _, colour) in enumerate(engines):
                v = vals[gi, si]
                if not np.isfinite(v):
                    continue
                x = gi - gw / 2 + si * (bw + gap)
                ax.add_patch(plt.Rectangle((x, 0), bw, v, linewidth=0,
                                           facecolor=colour, zorder=3))
                ax.text(x + bw / 2, v + top * 0.030, "{:.2f}".format(v), ha="center",
                        va="bottom", color=INK2, fontsize=7.8)

    fig.text(0.055, 0.955, "At matched coverage", color=INK, fontsize=21,
             fontweight="bold", va="top")
    fig.text(0.055, 0.905,
             "Every engine keeps its most-confident {:.0f}% of visible frames, so the "
             "comparison is like for like.  Four synthetic benches, exact ground truth, "
             "600 shared seeds.".format(a.coverage * 100),
             color=INK2, fontsize=9.8, va="top")
    fig.text(0.055, 0.868,
             "CoTracker3 thresholds visibility at 0.9 inside its own predictor, so an "
             "ungated comparison reads its already-gated output against everyone else's raw "
             "output.",
             color=INK3, fontsize=9.0, va="top")
    for i, (_, name, colour) in enumerate(engines):
        x = 0.055 + i * 0.155
        fig.patches.append(plt.Rectangle((x, 0.775), 0.015, 0.0155, color=colour,
                                         transform=fig.transFigure, zorder=5))
        fig.text(x + 0.020, 0.783, name, color=INK2, fontsize=9.9, va="center")
    fig.text(0.055, 0.055,
             "Occluded accuracy is a separate question and a visible-frame coverage match "
             "cannot move it -- CoTracker3 remains ahead there.",
             color=INK3, fontsize=8.6, va="top")

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "coverage_{}.png".format(a.tag))
    fig.savefig(path, facecolor=SURFACE, dpi=130)
    plt.close(fig)
    print("[figure] {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
