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

    python tools/score_occlusion.py --control
    python tools/score_occlusion.py \
        --shot bench\\synth\\lab02_occ --bot out/lab02_occ__jefftrack.txt

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
REPO = os.path.dirname(HERE)          # the repo root, one level up
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

from jefftrack.io import read_3de as load_tracks  # noqa: E402


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
        if "polys" in o:
            # Depth occluders carry a convex silhouette per frame instead of a box. Convex,
            # so a point is inside exactly when it is on the same side of every edge; the
            # sign of the whole polygon is taken from its own winding rather than assumed,
            # because the vertices are warped per frame and can change orientation.
            poly = o["polys"][t]
            if poly is None:
                continue
            v = np.asarray(poly, float)
            e = np.roll(v, -1, axis=0) - v
            d = pts[:, None, :] - v[None, :, :]
            cross = e[None, :, 0] * d[:, :, 1] - e[None, :, 1] * d[:, :, 0]
            hit |= (cross >= 0).all(1) | (cross <= 0).all(1)
            continue
        b = o["boxes"][t]
        if b is None:
            continue
        x, y, bw, bh = b
        hit |= ((pts[:, 0] >= x) & (pts[:, 0] < x + bw) &
                (pts[:, 1] >= y) & (pts[:, 1] < y + bh))
    return hit


def seq_from_export(bot_path: str, Hp: int):
    """The 3DE export, which is what an artist actually receives.

    3DE ASCII is bottom-left origin; the homography works in raster coords. Flip on the
    way in, the exact mirror of run_jefftrack.write_3de.

    Note what this format cannot represent: write_3de drops any frame the model was
    unsure about, so a track that comes back from an occlusion WRONG AND UNSURE is not in
    here at all. Scoring re-acquisition on this alone counts only the frames the model
    was willing to commit to -- see seq_from_npz.
    """
    tracks = load_tracks(bot_path)
    seq = {}
    for name, tr in tracks.items():
        fr = sorted(tr)
        seq[name] = {f: (tr[f][0], Hp - tr[f][1]) for f in fr}
    return seq


def seq_from_npz(npz_path: str, gate: float):
    """The raw model output, before the export's confidence gate.

    This is the honest input for RE-ACQUIRE. The export gate makes that metric
    survivorship-biased: any change that makes a model more willing to commit lets
    previously-dropped bad frames into the measurement, so the score can regress on a
    model that improved, with no way to separate the two afterwards. Same defect class as
    the two metrics found in 2026-08 that were measuring the plate instead of the tracker.

    `gate` reproduces any threshold on demand -- 0.0 keeps every frame (ungated, the
    default), 0.5 reproduces the exporter exactly. Sweeping it is how two models get
    compared at MATCHED coverage rather than at whatever coverage each happens to pick.

    npz is already raster (top-left) in plate pixels, so no flip. Returns the sequence and
    the npz's own first_frame, which wins over the CLI default.
    """
    d = np.load(npz_path)
    tracks, conf = d["tracks"], d["confidence"]
    first_frame = int(d["first_frame"])
    vis = d["visibility"] if "visibility" in d.files else np.ones(conf.shape, bool)
    T, N = tracks.shape[0], tracks.shape[1]
    seq = {}
    for i in range(N):
        pts = {}
        for t in range(T):
            # Visibility is the model's own occlusion call; confidence is its certainty.
            # The exporter requires both, so reproducing it needs both.
            if gate > 0.0 and (float(conf[t, i]) < gate or not bool(vis[t, i])):
                continue
            pts[first_frame + t] = (float(tracks[t, i, 0]), float(tracks[t, i, 1]))
        if pts:
            seq["T_{:04d}".format(i)] = pts
    return seq, first_frame


def score(shot: str, seq: dict, reacq: int, first_frame: int, control: bool,
          source: str = "", margin: float = 0.0):
    gt, Hs, occ = load_shot(shot)
    T, Wp, Hp = len(Hs), int(gt["width"]), int(gt["height"])

    vis_e, occ_e, reacq_e = [], [], []
    occ_seen = occ_emitted = 0
    n_reacq_events = 0
    n_reacq_dropped = 0
    n_offframe = 0

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

        # A point whose TRUTH has left the picture cannot be tracked, and the error
        # against it measures nothing about the tracker. Measured on lab02_occ these are
        # 1.2% of visible samples and they carried the ENTIRE tail: worst visible error
        # 389.30 px with them, 16.86 px without, on the same run. Excluding them is the
        # same judgement losses.py already applies to occluded supervision -- "predicting
        # where an off-screen point went is not what track through an occlusion means".
        inside = ((gtp[:, 0] >= -margin) & (gtp[:, 0] < Wp + margin) &
                  (gtp[:, 1] >= -margin) & (gtp[:, 1] < Hp + margin))

        for f in fr:
            t = f - first_frame
            if t < 0 or t >= T:
                continue
            if not inside[t]:
                n_offframe += 1
                continue
            e = float(np.hypot(pts[f][0] - gtp[t, 0], pts[f][1] - gtp[t, 1]))
            if occl[t]:
                occ_e.append(e)
            else:
                vis_e.append(e)

        have = set(f - first_frame for f in fr)
        occ_seen += int((occl & inside).sum())
        occ_emitted += int(sum(1 for t in range(T)
                               if occl[t] and inside[t] and t in have))

        # Re-acquisition: every falling edge of the occlusion mask that has clear frames
        # after it. A track occluded at the very end of the shot has nothing to re-acquire.
        for t in range(1, T):
            if occl[t - 1] and not occl[t]:
                n_reacq_events += 1
                for k in range(t, min(t + reacq, T)):
                    if occl[k] or not inside[k]:
                        continue
                    if k not in have:
                        # Not scored because the tracker emitted nothing here. Counted, so
                        # the survivorship in this metric is visible instead of silent: a
                        # track that returns wrong and unsure lands in THIS number, not in
                        # the error above it.
                        n_reacq_dropped += 1
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

    print("shot {}   {} frames {}x{}   {} tracks   {}{}".format(
        shot, T, Wp, Hp, len(seq),
        source + "  " if source else "",
        "CONTROL (ground truth as input)" if control else ""))
    stat("VISIBLE", vis_e, "  {} samples excluded: truth outside the plate".format(
        n_offframe) if n_offframe else "")
    cov = (100.0 * occ_emitted / occ_seen) if occ_seen else 0.0
    stat("OCCLUDED", occ_e, "  coverage {:.1f}% of occluded frames emitted".format(cov))
    tot_reacq = len(reacq_e) + n_reacq_dropped
    drop_pct = (100.0 * n_reacq_dropped / tot_reacq) if tot_reacq else 0.0
    stat("RE-ACQUIRE", reacq_e, "  {} occlusion exits, {} post-occlusion frames not "
         "emitted ({:.1f}% unscored)".format(n_reacq_events, n_reacq_dropped, drop_pct))
    if n_reacq_dropped and not control:
        print("  {:<12} {} post-occlusion frames carry no position and are therefore "
              "absent from".format("", n_reacq_dropped))
        print("  {:<12} the error above. --npz scores the model ungated; a track that "
              "returns wrong".format(""))
        print("  {:<12} AND unsure lands there, not here.".format(""))

    def pack(arr):
        if not arr:
            return None
        a = np.asarray(arr)
        return {"mean": float(a.mean()), "med": float(np.median(a)),
                "p95": float(np.percentile(a, 95)), "max": float(a.max()),
                "n": int(a.size)}

    out = {"shot": os.path.basename(shot.rstrip("/\\")), "source": source,
           "tracks": len(seq), "visible": pack(vis_e), "occluded": pack(occ_e),
           "reacquire": pack(reacq_e), "coverage_pct": cov,
           "reacq_events": n_reacq_events, "reacq_dropped": n_reacq_dropped,
           "reacq_dropped_pct": drop_pct, "offframe_excluded": n_offframe}

    if control:
        worst = max([max(vis_e) if vis_e else 0.0,
                     max(occ_e) if occ_e else 0.0,
                     max(reacq_e) if reacq_e else 0.0])
        ok = worst < 0.01
        print("  {}  control worst error {:.5f}px (must be ~0)".format(
            "PASS" if ok else "FAIL", worst))
        out["control_ok"] = bool(ok)
        out["control_worst"] = float(worst)
    return out


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
    ap.add_argument("--bot", default=None, help="a 3DE export -- what the artist receives")
    ap.add_argument("--npz", default=None,
                    help="raw model output instead of the export. RE-ACQUIRE read from the "
                         "export is survivorship-biased: a track that returns wrong AND "
                         "unsure is dropped before it is ever scored, so a model that "
                         "commits more can look worse while being better. Score from here.")
    ap.add_argument("--gate", type=float, default=0.0,
                    help="confidence threshold applied to --npz. 0.0 = ungated (default), "
                         "0.5 = reproduce the exporter. Sweep it to compare two models at "
                         "MATCHED coverage rather than at whatever each one picks.")
    ap.add_argument("--reacq", type=int, default=10,
                    help="frames after an occlusion ends that count as re-acquisition")
    ap.add_argument("--first-frame", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="how far outside the plate a truth position may sit and still be "
                         "scored. 0 = exclude anything off-plate (default). These were "
                         "1.2%% of visible samples on lab02_occ and carried the whole "
                         "tail: worst visible error 389.30 px with them, 16.86 px without.")
    ap.add_argument("--control", action="store_true",
                    help="feed ground truth to the scorer; every error must come back ~0")
    a = ap.parse_args()

    shot = a.shot or os.path.join(REPO, "bench", "synth", "lab02_occ")
    if a.control:
        tmp = os.path.join(HERE, "out", "_control_gt.txt")
        os.makedirs(os.path.dirname(tmp), exist_ok=True)
        write_control_export(shot, tmp, 40, a.first_frame)
        gt, Hs, _ = load_shot(shot)
        r = score(shot, seq_from_export(tmp, int(gt["height"])), a.reacq,
                  a.first_frame, control=True, source="[export]", margin=a.margin)
        return 0 if r.get("control_ok") else 1
    if a.npz:
        seq, ff = seq_from_npz(a.npz, a.gate)
        label = "[npz ungated]" if a.gate <= 0.0 else "[npz gate {:.2f}]".format(a.gate)
        score(shot, seq, a.reacq, ff, control=False, source=label, margin=a.margin)
        return 0
    if not a.bot:
        ap.error("--bot, --npz or --control is required")
    gt, Hs, _ = load_shot(shot)
    score(shot, seq_from_export(a.bot, int(gt["height"])), a.reacq,
          a.first_frame, control=False, source="[export]", margin=a.margin)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
