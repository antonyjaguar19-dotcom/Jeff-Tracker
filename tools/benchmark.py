"""Jeff-Tracker on TAP-Vid DAVIS.

    python tools/benchmark.py --models jefftracker --whole-clip --out out/bench.json

Reports AJ / delta_avg / OA and cost, using the official TAP-Vid metric implementation
unmodified. `--arch locotrack` runs the unmodified base the model is fine-tuned from, which
is the ablation behind the left panel of assets/results.png.

TAPNext++ (Apache-2.0) is available as an optional second adapter if you point
`--tapnext-root` at your own checkout; nothing of it is vendored here either.

## Protocol, and why it is the same for all three

TAP-Vid DAVIS, **query-first**, 256x256 -- the standard published setting. The metric
(`compute_tapvid_metrics`) is the reference implementation, called unmodified.

In `first` mode the metric builds its evaluation mask as `cumsum(eye) - eye`, so only
frames strictly **after** a point's query frame are scored. That single fact is what makes
this comparison fair: TAPNext is causal -- it streams forward from a seed and cannot look
back -- while Jeff-Tracker sees the whole clip at once. So every model here is
run the same way: queries are grouped by their query frame `k`, the model is given
`frames[k:]` with the queries at local frame 0, and its output is scattered back into the
full timeline. Frames before `k` are left at zero and are never read by the metric.

This handicaps the two offline models relative to what they could do with the whole clip
and backward tracking. That is deliberate. Comparing a causal tracker against an offline
one that was allowed to peek backwards would measure the protocol, not the models.

Numbers from this script are therefore *first-mode, forward-only*, and are not
interchangeable with the strided-mode figures in docs/METHOD.md.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jefftrack.paths import add_vendor_to_path  # noqa: E402

add_vendor_to_path()

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from data.evaluation_datasets import (  # noqa: E402  (vendor, official TAP-Vid code)
    compute_tapvid_metrics, sample_queries_first)

DEFAULT_PKL = os.path.join(ROOT, "data", "tapvid_davis", "tapvid_davis.pkl")
RES = 256


# ------------------------------------------------------------------ model adapters
class Adapter:
    """One method: track `queries` (N,3 as [t,x,y]) through `frames` (T,H,W,3 BGR).

    Returns (tracks (T,N,2) xy, visible (T,N) bool) in the frames' own pixel space.
    Every adapter is handed identical frames and identical queries.
    """

    name = "?"

    def track(self, frames_bgr, queries):
        raise NotImplementedError


class JeffTrackerAdapter(Adapter):
    name = "jefftracker"

    def __init__(self, ckpt, arch="jefftrack", res=(RES, RES)):
        from jefftrack.engine import JeffTrackEngine                      # noqa: PLC0415
        self.eng = JeffTrackEngine(device="cuda", model_size="base", ckpt=ckpt,
                                 model_res=res, arch=arch)

    def track(self, frames_bgr, queries):
        tracks, vis, _ = self.eng.track_queries_conf(frames_bgr, queries)
        return tracks, vis


class TapNextAdapter(Adapter):
    name = "tapnext"

    def __init__(self, tool_root, engine_dir=None):
        # tool_root is the tree that CONTAINS the tapnet checkout (pipeline/tapnext-main
        # or thirdparty/), not the checkout itself -- the engine resolves it from there.
        for p in (engine_dir or tool_root, tool_root):
            if p and p not in sys.path:
                sys.path.insert(0, p)
        from tapnext_engine import TapNextEngine                      # noqa: PLC0415
        self.eng = TapNextEngine(tool_root=tool_root, device="cuda")

    def track(self, frames_bgr, queries):
        return self.eng.track_queries(frames_bgr, np.asarray(queries, np.float32))


# ------------------------------------------------------------------ the run
def run_clip_whole(adapter, frames_bgr, query_points):
    """Give an OFFLINE model what it is actually built for: the whole clip at once, its
    queries at their true frame index, and backward tracking on where the model has it.

    This exists because the forward-only protocol below is not neutral. It is neutral
    between a causal model and an offline one only in the sense that both are handed the
    same frames; what it actually does is strip an offline model of the thing that makes it
    offline. Both protocols are therefore reported rather than one being chosen.
    """
    q = np.stack([query_points[:, 0].astype(np.float32),
                  query_points[:, 2].astype(np.float32),
                  query_points[:, 1].astype(np.float32)], axis=1)
    try:
        tr, vs = adapter.track(frames_bgr, q, backward=True)
    except TypeError:
        tr, vs = adapter.track(frames_bgr, q)
    return tr, vs


def run_clip(adapter, frames_bgr, query_points):
    """query_points: (N,3) as [t, y, x] in 256-raster, the sampler's own layout.

    Grouped by query frame, each group streamed from that frame. See the module docstring
    for why every model is run this way.
    """
    T = frames_bgr.shape[0]
    N = query_points.shape[0]
    tracks = np.zeros((T, N, 2), np.float32)
    visible = np.zeros((T, N), bool)

    qframe = np.round(query_points[:, 0]).astype(np.int32)
    for k in np.unique(qframe):
        idx = np.nonzero(qframe == k)[0]
        # The sampler emits [t, y, x]; every adapter here takes [frame, x, y]. This is the
        # same convention crossing probe_data.py checks on the training side, and getting
        # it wrong silently halves a score rather than raising.
        q = np.stack([np.zeros(len(idx), np.float32),
                      query_points[idx, 2].astype(np.float32),
                      query_points[idx, 1].astype(np.float32)], axis=1)
        tr, vs = adapter.track(np.ascontiguousarray(frames_bgr[k:]), q)
        tracks[k:, idx] = tr
        visible[k:, idx] = vs
    return tracks, visible


def evaluate(adapter, data, limit=0, verbose=True, whole_clip=False):
    names = sorted(data)
    if limit:
        names = names[:limit]
    rows, totals, seconds, frames_done = [], {}, 0.0, 0

    for n, name in enumerate(names, 1):
        rec = data[name]
        frames = rec["video"]
        if frames.dtype != np.uint8:                  # some copies store float
            frames = np.round(frames * 255).astype(np.uint8)
        frames = np.stack([cv2.resize(f, (RES, RES), interpolation=cv2.INTER_LINEAR)
                           for f in frames])
        # 'points' are normalised 0..1 (x, y); the protocol works at 256x256 raster.
        q = sample_queries_first(rec["occluded"], rec["points"] * float(RES), frames)
        bgr = np.ascontiguousarray(frames[:, :, :, ::-1])

        t0 = time.time()
        runner = run_clip_whole if whole_clip else run_clip
        tracks, visible = runner(adapter, bgr, q["query_points"][0])
        dt = time.time() - t0
        seconds += dt
        frames_done += frames.shape[0]

        m = compute_tapvid_metrics(
            q["query_points"], q["occluded"], q["target_points"],
            np.transpose(~visible, (1, 0))[None],
            np.transpose(tracks, (1, 0, 2))[None],
            query_mode="first")
        row = {k: float(np.mean(v)) for k, v in m.items()}
        row["clip"] = name
        rows.append(row)
        for k, v in row.items():
            if k != "clip":
                totals.setdefault(k, []).append(v)
        if verbose:
            print("  [{:>2}/{}] {:<20} AJ {:5.1f}  d_avg {:5.1f}  OA {:5.1f}  ({:.1f}s)"
                  .format(n, len(names), name, 100 * row["average_jaccard"],
                          100 * row["average_pts_within_thresh"],
                          100 * row["occlusion_accuracy"], dt), flush=True)

    summary = {
        "model": adapter.name,
        "protocol": "whole-clip" if whole_clip else "forward-only",
        "clips": len(names),
        "AJ": round(100 * float(np.mean(totals["average_jaccard"])), 1),
        "delta_avg": round(100 * float(np.mean(totals["average_pts_within_thresh"])), 1),
        "OA": round(100 * float(np.mean(totals["occlusion_accuracy"])), 1),
        "seconds": round(seconds, 1),
        "seconds_per_frame": round(seconds / max(1, frames_done), 4),
    }
    return summary, rows


def main() -> int:
    ap = argparse.ArgumentParser(description="three-way TAP-Vid DAVIS benchmark")
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--models", default="jefftracker")
    ap.add_argument("--limit", type=int, default=0, help="only N clips (a smoke run)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "benchmark.json"))
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "inf_s4000.ckpt"))
    ap.add_argument("--arch", default="jefftrack", choices=["locotrack", "jefftrack"])
    ap.add_argument("--tapnext-root", default=os.environ.get("BTR_TAPNEXT_ROOT", ""))
    ap.add_argument("--whole-clip", action="store_true",
                    help="let an OFFLINE model (jefftracker) use the whole "
                         "clip and backward tracking, which is what they are built for. "
                         "TAPNext is causal and is unaffected by this flag.")
    ap.add_argument("--tapnext-engine", default="",
                    help="dir holding tapnext_engine.py, if not tapnext-root")
    a = ap.parse_args()

    if not os.path.isfile(a.pkl):
        raise SystemExit(
            "[ERROR] {} not found.\n"
            "        curl -L -o tapvid_davis.zip "
            "https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip".format(a.pkl))
    with open(a.pkl, "rb") as fh:
        data = pickle.load(fh)

    want = [m.strip() for m in a.models.split(",") if m.strip()]
    results, per_clip = [], {}
    for m in want:
        print("\n=== {} ===".format(m), flush=True)
        if m == "jefftracker":
            ad = JeffTrackerAdapter(a.ckpt, arch=a.arch)
        elif m == "tapnext":
            if not a.tapnext_root:
                raise SystemExit("[ERROR] --tapnext-root is required for tapnext")
            ad = TapNextAdapter(a.tapnext_root, a.tapnext_engine or None)
        else:
            raise SystemExit("[ERROR] unknown model {!r}".format(m))

        whole = a.whole_clip and m != "tapnext"   # TAPNext is causal; it has no whole-clip mode
        summary, rows = evaluate(ad, data, limit=a.limit, whole_clip=whole)
        print("  -> AJ {AJ}  delta_avg {delta_avg}  OA {OA}  "
              "({seconds_per_frame} s/frame)".format(**summary), flush=True)
        results.append(summary)
        per_clip[m] = rows
        del ad
        try:
            import torch                                              # noqa: PLC0415
            torch.cuda.empty_cache()
        except Exception:                                             # noqa: BLE001
            pass

    print("\n{:<12} {:>6} {:>10} {:>6} {:>12}".format(
        "model", "AJ", "delta_avg", "OA", "s/frame"))
    for r in results:
        print("{:<12} {:>6.1f} {:>10.1f} {:>6.1f} {:>12.4f}".format(
            r["model"], r["AJ"], r["delta_avg"], r["OA"], r["seconds_per_frame"]))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"protocol": "tapvid_davis first-mode forward-only, 256x256",
                   "clips": results[0]["clips"] if results else 0,
                   "summary": results, "per_clip": per_clip}, fh, indent=2)
    print("\n[out] {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
