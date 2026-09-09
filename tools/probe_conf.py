"""Is LocoTrack's own confidence able to find its own bad frames?

The occlusion bench turned up a split personality at 384x680: median visible error 0.70 px
and mean 40.6 px, i.e. most samples are better than the 256x256 run and a few are hundreds
of pixels out. That tail is only a problem if it cannot be identified. LocoTrack emits a
per-frame confidence (1 - P(occluded or uncertain), see jefftrack.engine._infer), so the
question has an answer that can be measured rather than argued.

This reads the .npz that run_jefftrack writes -- which keeps the continuous confidence, the
thing a visibility-logit threshold throws away -- and scores it against the bench homography.

    python probe_conf.py ^
        --npz out\\lab02_occ_384x680__jefftrack.npz --shot bench\\synth\\lab02_occ

Reported: error by confidence bucket, the AUC of confidence as a detector of a bad frame,
and what a threshold actually buys -- how much of the tail it removes and how much good
data it costs. A gate that removes the tail by removing everything is not a gate.
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

import numpy as np  # noqa: E402

from score_occlusion import load_shot, warp, is_occluded  # noqa: E402


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(score of a positive > score of a negative), by rank. 0.5 = coin toss."""
    if labels.sum() == 0 or (~labels).sum() == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    npos = int(labels.sum())
    return float((ranks[labels].sum() - npos * (npos + 1) / 2.0) /
                 (npos * (~labels).sum()))


def main() -> int:
    ap = argparse.ArgumentParser(description="confidence vs error on a bench shot")
    ap.add_argument("--npz", required=True)
    ap.add_argument("--shot", required=True)
    ap.add_argument("--bad-px", type=float, default=5.0,
                    help="a sample is 'bad' above this error, in plate pixels")
    a = ap.parse_args()

    d = np.load(a.npz)
    tracks, conf = d["tracks"], d["confidence"]
    first_frame = int(d["first_frame"])
    gt, Hs, occ = load_shot(a.shot)
    T, Hp = len(Hs), int(gt["height"])
    Tn, N = tracks.shape[0], tracks.shape[1]
    T = min(T, Tn)

    # Anchor each track's ground truth on its own frame-0 position, as the bench scorers do.
    src = warp(np.linalg.inv(Hs[0]), tracks[0].astype(np.float64))
    err = np.zeros((T, N))
    occl = np.zeros((T, N), bool)
    for t in range(T):
        g = warp(Hs[t], src)
        err[t] = np.hypot(tracks[t, :, 0] - g[:, 0], tracks[t, :, 1] - g[:, 1])
        occl[t] = is_occluded(occ, t, g)

    c, e, o = conf[:T].ravel(), err.ravel(), occl.ravel()
    clear = ~o
    print("{}  {} frames x {} tracks   {} samples ({} occluded by GT)".format(
        os.path.basename(a.npz), T, N, c.size, int(o.sum())))
    print()
    print("{:<14}{:>9}{:>10}{:>10}{:>10}{:>12}".format(
        "conf bucket", "n", "med_px", "mean_px", "p95_px", "%>{}px".format(a.bad_px)))
    print("-" * 65)
    edges = [0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 1.001]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (c >= lo) & (c < hi)
        if not m.any():
            continue
        print("{:<14}{:>9}{:>10.2f}{:>10.2f}{:>10.2f}{:>11.1f}%".format(
            "[{:.2f},{:.2f})".format(lo, hi), int(m.sum()), np.median(e[m]),
            e[m].mean(), np.percentile(e[m], 95), 100.0 * (e[m] > a.bad_px).mean()))

    print()
    bad = e > a.bad_px
    print("AUC of confidence detecting a >{}px frame : {:.3f}   (all samples)".format(
        a.bad_px, auc(-c, bad)))
    print("AUC on GT-visible samples only            : {:.3f}".format(
        auc(-c[clear], bad[clear])))
    print("  1.0 = confidence ranks every bad frame below every good one, 0.5 = useless.")
    print()
    print("{:<10}{:>10}{:>10}{:>10}{:>11}{:>10}".format(
        "keep >=", "kept%", "med_px", "mean_px", "p99_px", "max_px"))
    print("-" * 62)
    for thr in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        m = c >= thr
        if not m.any():
            continue
        print("{:<10.2f}{:>9.1f}%{:>10.2f}{:>10.2f}{:>11.2f}{:>10.1f}".format(
            thr, 100.0 * m.mean(), np.median(e[m]), e[m].mean(),
            np.percentile(e[m], 99), e[m].max()))
    print()
    print("A threshold is only worth having if the mean falls a lot further than the kept")
    print("fraction does -- otherwise it is discarding good tracks to flatter the average.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
