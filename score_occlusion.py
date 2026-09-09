"""Score a 3DE export on an occlusion bench: visible vs occluded, and re-acquisition.

Three numbers, kept apart on purpose. CoTracker3's own ablation for cross-track attention
reports visible and occluded separately (71.3 -> 72.9 visible, 35.9 -> 41.0 occluded); an
average across both would have shown a modest gain and hidden the entire effect.

  VISIBLE     mean error on frames where ground truth says the point was in clear view.
              This is the localisation number, and it must not get worse.
  OCCLUDED    mean error on frames where an occluder covered the point, counted only where
              the tracker actually emitted a position. A tracker that drops the point
              instead scores no error here at all, so this number is meaningless without
              the coverage figure printed beside it.
  RE-ACQUIRE  mean error over the first --reacq frames AFTER an occlusion ends. This is the
              one that matters for matchmove: a track that coasts through the occluder and
              lands back on the right pixel is usable, and one that comes back onto the
              neighbouring feature is worse than a gap, because it looks fine.

    python score_occlusion.py --control
    python score_occlusion.py ^
        --shot bench\\synth\\lab02_occ --bot out\\lab02_occ__dbtrack.txt

--control feeds the scorer ground truth as if it were a tracker's export. Every error must
come back ~0. Both metric defects found in 2026-08 were metrics that looked plausible and
were measuring the plate instead of the tracker, and each was caught exactly this way;
nothing below the control line is believable until the control line passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = HERE
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from app.compare_tracks import load_tracks  # noqa: E402


def load_shot(shot: str):
    with open(os.path.join(shot, "gt.json")) as fh:
        gt = json.load(fh)
    occ_path = os.path.join(shot, "occluders.json")
    occ = None
    if os.path.isfile(occ_path):
        with open(occ_path) as fh:
            occ = json.load(fh)
    H = [np.asarray(m, np.float64) for m in gt["H"]]
    return gt, H, occ


def warp(Hm: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Apply a 3x3 homography to (N,2) points."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1)
    q = p @ Hm.T
    return q[:, :2] / q[:, 2:3]


def is_occluded(occ: dict, t: int, pts: np.ndarray) -> np.ndarray:
    """(N,) bool: does an occluder box cover each point on frame t."""
    hit = np.zeros(len(pts), bool)
    if occ is None:
        return hit
    for o in occ["occluders"]:
        b = o["boxes"][t]
        if b is None:
            continue
        x, y, bw, bh = b
        hit |= ((pts[:, 0] >= x) & (pts[:, 0] < x + bw) &
                (pts[:, 1] >= y) & (pts[:, 1] < y + bh))
    return hit


def score(shot: str, bot_path: str, reacq: int, first_frame: int, control: bool):
    gt, Hs, occ = load_shot(shot)
    T, Wp, Hp = len(Hs), int(gt["width"]), int(gt["height"])

    tracks = load_tracks(bot_path)
    # 3DE ASCII is bottom-left origin; the homography works in raster coords. Flip on the
    # way in, the exact mirror of run_dbtrack.write_3de.
    seq = {}
    for name, tr in tracks.items():
        fr = sorted(tr)
        pts = {f: (tr[f][0], Hp - tr[f][1]) for f in fr}
        seq[name] = pts

    vis_e, occ_e, reacq_e = [], [], []
    occ_seen = occ_emitted = 0
    n_reacq_events = 0

    for name, pts in seq.items():
        fr = sorted(pts)
        t0 = fr[0] - first_frame
        if t0 < 0 or t0 >= T:
            continue
        # Anchor ground truth on this track's own seed, exactly as bench/score_synth does:
        # on a plane the point 1 px from the intended corner is still a valid scene point,
        # so what is being measured is how well the tracker FOLLOWS it, not where it began.
        src = warp(np.linalg.inv(Hs[t0]), np.asarray([pts[fr[0]]], np.float64))

        gtp = np.zeros((T, 2))
        for t in range(T):
            gtp[t] = warp(Hs[t], src)[0]
        occl = np.array([is_occluded(occ, t, gtp[t:t + 1])[0] for t in range(T)])

        for f in fr:
            t = f - first_frame
            if t < 0 or t >= T:
                continue
            e = float(np.hypot(pts[f][0] - gtp[t, 0], pts[f][1] - gtp[t, 1]))
            if occl[t]:
                occ_e.append(e)
            else:
                vis_e.append(e)

        have = set(f - first_frame for f in fr)
        occ_seen += int(occl.sum())
        occ_emitted += int(sum(1 for t in range(T) if occl[t] and t in have))

        # Re-acquisition: every falling edge of the occlusion mask that has clear frames
        # after it. A track occluded at the very end of the shot has nothing to re-acquire.
        for t in range(1, T):
            if occl[t - 1] and not occl[t]:
                n_reacq_events += 1
                for k in range(t, min(t + reacq, T)):
                    if occl[k] or k not in have:
                        continue
                    f = k + first_frame
                    reacq_e.append(float(np.hypot(pts[f][0] - gtp[k, 0],
                                                  pts[f][1] - gtp[k, 1])))

    def stat(name, arr, extra=""):
        if not arr:
            print("  {:<12} {:>8}".format(name, "no samples"))
            return
        a = np.asarray(arr)
        print("  {:<12} mean {:7.3f}  med {:7.3f}  p95 {:7.3f}  max {:8.3f}  n={}{}".format(
            name, a.mean(), np.median(a), np.percentile(a, 95), a.max(), len(a), extra))

    print("shot {}   {} frames {}x{}   {} tracks   {}".format(
        shot, T, Wp, Hp, len(seq), "CONTROL (ground truth as input)" if control else ""))
    stat("VISIBLE", vis_e)
    cov = (100.0 * occ_emitted / occ_seen) if occ_seen else 0.0
    stat("OCCLUDED", occ_e, "  coverage {:.1f}% of occluded frames emitted".format(cov))
    stat("RE-ACQUIRE", reacq_e, "  {} occlusion exits".format(n_reacq_events))

    if control:
        worst = max([max(vis_e) if vis_e else 0.0,
                     max(occ_e) if occ_e else 0.0,
                     max(reacq_e) if reacq_e else 0.0])
        ok = worst < 0.01
        print("  {}  control worst error {:.5f}px (must be ~0)".format(
            "PASS" if ok else "FAIL", worst))
        return 0 if ok else 1
    return 0


def write_control_export(shot: str, path: str, n: int, first_frame: int):
    """Emit ground-truth tracks in the export format, to feed the scorer its own answer."""
    gt, Hs, _ = load_shot(shot)
    T, Wp, Hp = len(Hs), int(gt["width"]), int(gt["height"])
    rng = np.random.default_rng(3)
    src = np.stack([rng.uniform(0.15, 0.85, n) * Wp,
                    rng.uniform(0.15, 0.85, n) * Hp], 1)
    # Ground truth is anchored on frame 0 positions, so build source points by pulling
    # frame-0 raster positions back through H[0] -- the same inverse the scorer applies.
    src = warp(np.linalg.inv(Hs[0]), src)
    with open(path, "w") as fh:
        fh.write("{}\n".format(n))
        for i in range(n):
            fh.write("GT_{:04d}\n0\n{}\n".format(i, T))
            for t in range(T):
                p = warp(Hs[t], src[i:i + 1])[0]
                fh.write("{} {:.6f} {:.6f}\n".format(first_frame + t, p[0], Hp - p[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description="score an export on an occlusion bench")
    ap.add_argument("--shot", default=None)
    ap.add_argument("--bot", default=None)
    ap.add_argument("--reacq", type=int, default=10,
                    help="frames after an occlusion ends that count as re-acquisition")
    ap.add_argument("--first-frame", type=int, default=1)
    ap.add_argument("--control", action="store_true",
                    help="feed ground truth to the scorer; every error must come back ~0")
    a = ap.parse_args()

    shot = a.shot or os.path.join(REPO, "bench", "synth", "lab02_occ")
    if a.control:
        tmp = os.path.join(HERE, "out", "_control_gt.txt")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        write_control_export(shot, tmp, 40, a.first_frame)
        return score(shot, tmp, a.reacq, a.first_frame, control=True)
    if not a.bot:
        ap.error("--bot is required unless --control")
    return score(shot, a.bot, a.reacq, a.first_frame, control=False)


if __name__ == "__main__":
    raise SystemExit(main())
