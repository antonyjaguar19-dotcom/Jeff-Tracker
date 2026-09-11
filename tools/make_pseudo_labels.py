"""Stage 2: label real plates with an ensemble of teachers, the way CoTracker3 scales.

Its stage 1 is Kubric with exact truth (that is `train_cross.py`). Stage 2 is 15k unlabelled
internet videos pseudo-labelled by four teachers, and the paper's own claim is that this
second stage, not the architecture, is where its accuracy comes from.

Four things are copied from that recipe because each one is doing work, and they are copied
from the DESCRIPTION of it -- see LICENSES.md. CoTracker is CC-BY-NC-4.0 and is not a
teacher here; every teacher below is Apache-2.0 or MIT.

  * **One teacher per sample, drawn at random -- not an average.** Averaging four trackers
    produces a label with all four systematic biases blended into it, and a student trained
    on it learns the blend. Drawing one per sample means the student sees four different
    error distributions over the run and can only fit what they agree on, which is the
    signal. This is the most transferable idea in the recipe and it costs nothing.
  * **Support points.** Extra queries -- a grid plus extra detector points -- are handed to
    the teacher, tracked jointly, and then thrown away. Every one of these trackers mixes
    information across the points in a window, so the extras stabilise the points that are
    kept. Free label quality, paid for in teacher time only.
  * **Detector-sampled queries.** Queries come from SIFT keypoints rather than a uniform
    grid, so the student is only ever asked to learn points a classical detector would call
    trackable. A grid point in the middle of clear sky has no correct answer and teaching a
    model to produce one confidently is worse than not teaching it at all.
  * **Hard visibility threshold.** Teacher visibility is thresholded at 0.9 and stored as a
    boolean, not kept as a soft target. A teacher's uncertainty is not calibrated truth.

What is deliberately NOT copied: the keyword filter. Meta's stopword list throws out water,
sky, fire, fast motion and CG -- which is a hand-written statement of what pseudo-labelling
fails on, and is also a fair description of a VFX shot. Those plates are the target domain
here, so they stay in, and the closure check below is what guards label quality instead.

Labels are stored as a WINDOW SPEC plus the tracks, not as pixels: a shard names the plate,
the frame range, the stride and the crop, and the trainer re-decodes. A thousand samples of
decoded 384x512x80 frames would be ~450 GB; the specs are a few MB.

    python tools/make_pseudo_labels.py \
        --plates out/<plate> --out data\\pseudo\\v1 \
        --samples 200
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # the repo root, one level up
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from run_jefftrack import list_frames, read_frame  # noqa: E402

IMG_EXT = (".png", ".jpg", ".jpeg", ".exr", ".tif", ".tiff", ".dpx")


# ------------------------------------------------------------------------------ teachers
def _free():
    import gc  # noqa: PLC0415
    import torch  # noqa: PLC0415
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def teach_jefftrack(frames, pts, ckpt, arch, res=(256, 256), qframes=None):
    from jefftrack.engine import JeffTrackEngine  # noqa: PLC0415
    eng = JeffTrackEngine(device="cuda", model_size="base", ckpt=ckpt,
                          model_res=res, arch=arch, query_chunk_size=64)
    qf = (np.zeros((len(pts), 1), np.float32) if qframes is None
          else np.asarray(qframes, np.float32).reshape(-1, 1))
    q = np.concatenate([qf, pts], 1)
    tr, vis, conf = eng.track_queries_conf(frames, q)
    del eng
    _free()
    return tr, vis, conf


def teach_tapnext(frames, pts):
    from tapnext_engine import TapNextEngine  # noqa: PLC0415
    eng = TapNextEngine(tool_root=REPO, device="cuda")
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts], 1)[None]
    tr, vis = eng.track_queries(frames, q)
    del eng
    _free()
    vis = np.asarray(vis).astype(bool)
    return tr, vis, vis.astype(np.float32)


TEACHERS = {
    "locotrack": dict(kind="jt", ckpt=os.path.join(HERE, "weights", "locotrack_base.ckpt"),
                      arch="locotrack", licence="Apache-2.0"),
    "jefftrack_c3": dict(kind="jt", ckpt=os.path.join(HERE, "weights", "c3_occnorm.ckpt"),
                         arch="jefftrack", licence="ours"),
    "tapnext": dict(kind="tapnext", licence="Apache-2.0"),
}


def run_teacher(name, frames, pts, qframes=None):
    """`qframes` gives each track its OWN query frame. Only the LocoTrack-family teachers
    support it; TAPNext is causal and seeds at the block's frame 0, so it ignores it and the
    caller has to fall back -- see closure_of."""
    spec = TEACHERS[name]
    if spec["kind"] == "jt":
        return teach_jefftrack(frames, pts, spec["ckpt"], spec["arch"], qframes=qframes)
    if spec["kind"] == "tapnext":
        return teach_tapnext(frames, pts)
    raise SystemExit("[ERROR] unknown teacher {}".format(name))


def per_track_qframes(name):
    return TEACHERS[name]["kind"] == "jt"


# ------------------------------------------------------------------------------ sampling
def sift_queries(img, n, rng, border=12):
    """SIFT keypoints, strongest first, spatially thinned.

    Returns fewer than `n` when the frame has nothing to offer, and the caller drops the
    sample rather than padding it out with grid points -- which is the whole point. A frame
    with no structure is not a training example with fewer labels, it is not a training
    example.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    try:
        sift = cv2.SIFT_create(nfeatures=n * 6)
    except AttributeError:                       # not built with the feature2d module
        return np.zeros((0, 2), np.float32)
    kp = sift.detect(g, None)
    if not kp:
        return np.zeros((0, 2), np.float32)
    kp = sorted(kp, key=lambda k: -k.response)
    H, W = g.shape[:2]
    keep, min_d = [], max(6.0, 0.5 * np.sqrt(W * H / float(max(1, n))))
    for k in kp:
        x, y = k.pt
        if x < border or y < border or x >= W - border or y >= H - border:
            continue
        if all((x - u) ** 2 + (y - v) ** 2 >= min_d ** 2 for u, v in keep):
            keep.append((x, y))
        if len(keep) >= n:
            break
    return np.array(keep, np.float32).reshape(-1, 2)


def support_points(W, H, grid, rng, inset=0.06):
    """The regular grid handed to the teacher and then discarded."""
    xs = np.linspace(inset * W, (1 - inset) * W, grid)
    ys = np.linspace(inset * H, (1 - inset) * H, grid)
    gx, gy = np.meshgrid(xs, ys)
    return np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32)


def closure_of(frames, pts, name, tracks, vis, min_span=0.5):
    """Track the teacher's own endpoint backwards and measure the round trip.

    There is no ground truth on a plate, so nothing here can say a label is right. What it
    CAN say is that a label is self-inconsistent: re-query on the position the teacher
    itself emitted, run the clip backwards, and see how far from the original seed it lands.
    Same technique as tools/make_lk_reference.py, and the same caveat -- a teacher that
    drifts the same way in both directions closes perfectly and is still wrong -- so it is a
    lower bound on error, and a BAD closure is the conclusive half.

    Measured from each track's OWN last confident frame, not the clip's. The first version
    used the clip's last frame and dropped every sample on a shot where the subject is
    hidden at the end: on a 32-frame window of a diver, 11 of 30 tracks were confident on
    the final frame, and windows landing mid-dive had fewer than the 8 needed to keep the
    sample at all -- 6 of 6 discarded. A track that goes behind something and stays there is
    not a bad label, it is a good label for the frames it covers, and `min_span` is what
    keeps the round trip long enough to be worth measuring.

    Teachers that cannot be queried per-track (TAPNext is causal, it seeds at frame 0 of the
    block) get the clip-end form instead, so their labels are checked over a common span.
    """
    T = len(frames)
    seen = vis.any(0)
    last = np.where(seen, T - 1 - np.argmax(vis[::-1], 0), 0)
    ok = vis[0] & seen & (last >= max(1, int(min_span * (T - 1))))
    if not ok.any():
        return np.full(len(pts), np.inf, np.float32)

    if per_track_qframes(name):
        end = tracks[last, np.arange(len(pts))].astype(np.float32)
        # In the reversed clip, original frame f is at index T-1-f.
        qf = (T - 1 - last).astype(np.float32)
    else:
        ok = ok & vis[T - 1]
        if not ok.any():
            return np.full(len(pts), np.inf, np.float32)
        end = tracks[T - 1].astype(np.float32)
        qf = None

    back_tr, _, _ = run_teacher(name, frames[::-1], end, qframes=qf)
    closed = back_tr[T - 1]                      # reversed last == original frame 0
    d = np.hypot(closed[:, 0] - pts[:, 0], closed[:, 1] - pts[:, 1]).astype(np.float32)
    d[~ok] = np.inf
    return d


def main() -> int:
    ap = argparse.ArgumentParser(description="pseudo-label real plates with an ensemble")
    ap.add_argument("--plates", nargs="+", required=True,
                    help="plate directories, or roots to search for them")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--seq-len", type=int, default=48, help="frames per sample (max)")
    ap.add_argument("--min-seq", type=int, default=24)
    ap.add_argument("--tracks", type=int, default=96, help="KEPT queries per sample")
    ap.add_argument("--support-grid", type=int, default=6,
                    help="NxN extra points given to the teacher and then discarded")
    ap.add_argument("--support-sift", type=int, default=64)
    ap.add_argument("--work-width", type=int, default=960)
    ap.add_argument("--max-stride", type=int, default=4,
                    help="random frame-rate augmentation, 1..N")
    ap.add_argument("--vis-thresh", type=float, default=0.9)
    ap.add_argument("--min-tracks", type=int, default=8,
                    help="a sample with fewer surviving tracks than this is discarded")
    ap.add_argument("--max-closure", type=float, default=2.0,
                    help="drop a track whose forward-backward round trip misses its own "
                         "seed by more than this, in work pixels. inf disables the check.")
    ap.add_argument("--teachers", default="locotrack,jefftrack_c3,tapnext")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    names = [t.strip() for t in a.teachers.split(",") if t.strip()]
    for t in names:
        if t not in TEACHERS:
            raise SystemExit("[ERROR] unknown teacher {!r}; have {}".format(
                t, ", ".join(sorted(TEACHERS))))
        spec = TEACHERS[t]
        if spec["kind"] == "jt" and not os.path.isfile(spec["ckpt"]):
            raise SystemExit("[ERROR] teacher {} needs {}".format(t, spec["ckpt"]))

    # Every directory that actually holds frames, so --plates can be a shot or a tree.
    plates = []
    for root in a.plates:
        if any(glob.glob(os.path.join(root, "*" + e)) for e in IMG_EXT):
            plates.append(root)
            continue
        for d, _, _ in os.walk(root):
            if any(glob.glob(os.path.join(d, "*" + e)) for e in IMG_EXT):
                plates.append(d)
    plates = sorted(set(plates))
    if not plates:
        raise SystemExit("[ERROR] no plate directories under {}".format(a.plates))

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    print("[stage2] {} plates, {} samples, teachers: {}".format(
        len(plates), a.samples, ", ".join(names)))

    made = dropped = 0
    # A generator that discards most of what it is given is not obviously broken and not
    # obviously fine -- on a shot whose subject is hidden for eighty frames a high drop rate
    # is CORRECT. The only way to tell the two apart is to say which gate did it, so the
    # reasons are counted and printed rather than inferred from the total.
    why = {"short_clip": 0, "no_features": 0, "closure": 0}
    closures = []
    per_teacher = {t: 0 for t in names}
    t_start = time.time()
    for s in range(a.samples):
        plate = plates[int(rng.integers(len(plates)))]
        files, first = list_frames(plate, 1, 0)
        stride = int(rng.integers(1, a.max_stride + 1))
        want = int(rng.integers(a.min_seq, a.seq_len + 1))
        span = want * stride
        if span > len(files):
            stride = max(1, len(files) // max(1, want))
            span = want * stride
        if span > len(files):
            want = len(files) // stride
            span = want * stride
        if want < a.min_seq:
            dropped += 1
            why["short_clip"] += 1
            continue
        start = int(rng.integers(0, len(files) - span + 1))
        sel = list(range(start, start + span, stride))[:want]

        frames = np.stack([read_frame(files[i], a.work_width)[0] for i in sel])
        Hw, Ww = frames.shape[1], frames.shape[2]

        keep_q = sift_queries(frames[0], a.tracks, rng)
        if len(keep_q) < max(8, a.tracks // 4):
            dropped += 1
            why["no_features"] += 1
            continue
        # Support points are appended AFTER the kept ones, so slicing them off afterwards is
        # a fixed prefix and no bookkeeping can drift.
        sup = support_points(Ww, Hw, a.support_grid, rng)
        sup2 = sift_queries(frames[0], a.support_sift, rng, border=4)
        pts = np.concatenate([keep_q, sup, sup2], 0).astype(np.float32)

        name = names[int(rng.integers(len(names)))]
        tr, vis, conf = run_teacher(name, frames, pts)

        n_keep = len(keep_q)
        tr, vis, conf = tr[:, :n_keep], vis[:, :n_keep], conf[:, :n_keep]
        hard_vis = (conf >= a.vis_thresh) & vis

        good = np.ones(n_keep, bool)
        if np.isfinite(a.max_closure):
            d = closure_of(frames, keep_q, name, tr, hard_vis)
            good = d <= a.max_closure
            fin = d[np.isfinite(d)]
            if len(fin):
                closures.append(float(np.median(fin)))
        if good.sum() < a.min_tracks:
            dropped += 1
            why["closure"] += 1
            continue

        np.savez_compressed(
            os.path.join(a.out, "s{:06d}.npz".format(s)),
            plate=plate, frame_files=np.array([os.path.basename(files[i]) for i in sel]),
            work_width=a.work_width, teacher=name,
            queries=keep_q[good], tracks=tr[:, good].astype(np.float32),
            visible=hard_vis[:, good], size=np.array([Ww, Hw]))
        made += 1
        per_teacher[name] += 1
        if made % 10 == 0 or s == a.samples - 1:
            el = time.time() - t_start
            print("[stage2] {}/{} kept  {} dropped  {:.1f}s  ({:.1f}s/sample)".format(
                made, a.samples, dropped, el, el / max(1, made)), flush=True)

    with open(os.path.join(a.out, "index.json"), "w") as fh:
        json.dump({"samples": made, "dropped": dropped, "drops_by_gate": why,
                   "teachers": per_teacher,
                   "plates": plates, "vis_thresh": a.vis_thresh,
                   "max_closure": a.max_closure, "support_grid": a.support_grid,
                   "support_sift": a.support_sift, "seed": a.seed}, fh, indent=2)
    print("[stage2] {} samples written to {}  (dropped {})".format(made, a.out, dropped))
    print("[stage2] per teacher: {}".format(per_teacher))
    print("[stage2] drops by gate: {}".format(why))
    if closures:
        print("[stage2] per-sample median closure: median {:.2f} px  p90 {:.2f} px  "
              "(--max-closure {})".format(float(np.median(closures)),
                                          float(np.percentile(closures, 90)),
                                          a.max_closure))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
