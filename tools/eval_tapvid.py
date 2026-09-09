"""TAP-Vid DAVIS: AJ / delta_avg / OA, so DBtracker can be put next to published numbers.

This is how the CoTracker comparison gets made without CoTracker ever being run here.
CoTracker3's DAVIS figures are published in its paper; TAP-Vid is the standard protocol
that produced them; so evaluating DBtracker under the same protocol puts the two on one
axis using nothing but a public fact. See LICENSES.md for why that matters.

The metric is the vendor's own `compute_tapvid_metrics` and the query sampling is the
vendor's `sample_queries_first` / `sample_queries_strided` -- both are the official TAP-Vid
implementations, carried in locotrack_pytorch/data/evaluation_datasets.py. Reimplementing
either would make the number incomparable to every published table, which is the entire
point of computing it.

    python eval_tapvid.py --mode first
    python eval_tapvid.py --mode strided ^
        --arch dbtrack --ckpt weights\\dbtrack_cross.ckpt

Data: tapvid_davis.pkl from https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip,
kept in the gitignored data/. Evaluation only -- never trained on, never shipped.

Reading the result: `average_jaccard` (AJ) is the headline, `average_pts_within_thresh`
is delta_avg, `occlusion_accuracy` is OA. All three are computed at 256x256 raster
coordinates, which is what the paper metrics assume.
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dbtrack.paths import add_vendor_to_path  # noqa: E402

add_vendor_to_path()

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from data.evaluation_datasets import (  # noqa: E402  (vendor, official TAP-Vid code)
    compute_tapvid_metrics, sample_queries_first, sample_queries_strided)

from dbtrack.engine import DBTrackEngine, DEFAULT_CKPT  # noqa: E402

DEFAULT_PKL = os.path.join(ROOT, "data", "tapvid_davis", "tapvid_davis.pkl")


def resize_video(video: np.ndarray, size: int) -> np.ndarray:
    """(T,H,W,3) uint8 -> (T,size,size,3) uint8, matching the TAP-Vid protocol."""
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_LINEAR)
                     for f in video])


def main() -> int:
    ap = argparse.ArgumentParser(description="TAP-Vid DAVIS evaluation")
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--mode", default="first", choices=["first", "strided"])
    ap.add_argument("--arch", default="locotrack", choices=["locotrack", "dbtrack"])
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--model-res", default="256x256")
    ap.add_argument("--limit", type=int, default=0, help="only N videos (a smoke run)")
    a = ap.parse_args()

    if not os.path.isfile(a.pkl):
        raise SystemExit(
            "[ERROR] {} not found.\n"
            "        curl -L -o data/tapvid_davis.zip "
            "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip && unzip it"
            .format(a.pkl))
    with open(a.pkl, "rb") as fh:
        davis = pickle.load(fh)

    model_res = tuple(int(v) for v in a.model_res.lower().split("x"))
    eng = DBTrackEngine(device="cuda", model_size=a.model_size, ckpt=a.ckpt,
                        model_res=model_res, arch=a.arch)

    names = sorted(davis.keys())
    if a.limit:
        names = names[:a.limit]
    print("TAP-Vid DAVIS   {} videos   mode={}   arch={} {} {}".format(
        len(names), a.mode, a.arch, a.model_size, eng.model_res))
    print()

    rows, t0 = [], time.time()
    for vi, name in enumerate(names):
        rec = davis[name]
        frames = rec["video"]
        if frames.dtype != np.uint8:  # some copies store float
            frames = np.round(frames * 255).astype(np.uint8)
        frames = resize_video(frames, 256)
        # 'points' are normalised 0..1 (x, y); the protocol works at 256x256 raster.
        target_points = rec["points"] * 256.0
        target_occ = rec["occluded"]

        sampler = sample_queries_first if a.mode == "first" else sample_queries_strided
        q = sampler(target_occ, target_points, frames)

        # The sampler returns query_points as (1, N, 3) in 't y x'. The engine takes
        # [frame, x, y] in the pixel space of the frames it is handed, so swap here --
        # this is exactly the convention crossing probe_data.py checks for on the
        # training side, and it silently halves a score if it is wrong.
        qp = q["query_points"][0]
        eq = np.stack([qp[:, 0], qp[:, 2], qp[:, 1]], 1).astype(np.float32)

        bgr = frames[:, :, :, ::-1].copy()
        tracks, vis, _ = eng.track_queries_conf(bgr, eq)

        # Back to the metric's layout: (1, N, T, 2) xy and (1, N, T) occluded.
        pred_tracks = np.transpose(tracks, (1, 0, 2))[None]
        pred_occ = np.transpose(~vis, (1, 0))[None]

        m = compute_tapvid_metrics(
            q["query_points"], q["occluded"], q["target_points"],
            pred_occ, pred_tracks, query_mode=a.mode)
        rows.append({k: float(np.mean(v)) for k, v in m.items()})
        print("  {:<28} AJ {:5.1f}  d_avg {:5.1f}  OA {:5.1f}   ({}/{})".format(
            name, 100 * rows[-1]["average_jaccard"],
            100 * rows[-1]["average_pts_within_thresh"],
            100 * rows[-1]["occlusion_accuracy"], vi + 1, len(names)), flush=True)

    print()
    keys = ("average_jaccard", "average_pts_within_thresh", "occlusion_accuracy")
    agg = {k: 100.0 * float(np.mean([r[k] for r in rows])) for k in keys}
    print("{:<22}{:>8}".format("metric", "value"))
    print("-" * 30)
    print("{:<22}{:>8.1f}".format("AJ", agg["average_jaccard"]))
    print("{:<22}{:>8.1f}".format("delta_avg", agg["average_pts_within_thresh"]))
    print("{:<22}{:>8.1f}".format("OA", agg["occlusion_accuracy"]))
    print()
    print("{} videos in {:.1f}s. Compare against published DAVIS tables -- these are the "
          "standard TAP-Vid definitions, computed by the vendor's own implementation."
          .format(len(rows), time.time() - t0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
