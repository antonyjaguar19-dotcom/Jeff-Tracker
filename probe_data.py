"""Control pass on the training data, before a single training step is run.

A data pipeline that silently emits the wrong thing produces a training run that looks
healthy -- the loss goes down, the GPU is busy, hours pass -- and a model that has learned
something else. Everything measurable about it is checked here first, against what the
model's forward pass actually expects:

  * shapes and dtypes match LocoTrack's signature (video [B,T,H,W,3], query 'tyx',
    targets 'xy');
  * the video is really in [-1, 1], because dbtrack_engine feeds inference in that range
    and a training set in [0, 255] would teach the model a different input distribution
    than it is ever shown afterwards;
  * query points land ON their own track: target_points at the query frame must equal the
    query's own xy. If this is off, the tyx/xy convention has been crossed somewhere and
    every position loss is being computed against a shifted target;
  * a real fraction of samples are labelled occluded, and it is neither ~0 nor ~1. This is
    the entire reason for choosing this dataset over a homography bench -- if the occlusion
    labels are degenerate there is nothing here for cross-track attention to learn.

    python probe_data.py --batches 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="control pass on the MOVi-E pipeline")
    ap.add_argument("--data-dir", default="gs://kubric-public/tfds")
    ap.add_argument("--dataset", default="movi_e/256x256")
    ap.add_argument("--split", default="train")
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--tracks", type=int, default=256)
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args()

    from dbtrack.data.movi import batches  # noqa: E402

    stream = batches(device="cpu", data_dir=a.data_dir, name=a.dataset, split=a.split,
                     train_size=(a.res, a.res), batch_size=1, tracks_to_sample=a.tracks,
                     shuffle_buffer_size=None, color_augmentation=False)

    ok = True
    for i in range(a.batches):
        t0 = time.time()
        b = next(stream)
        dt = time.time() - t0
        v, q = b["video"], b["query_points"]
        tp, oc = b["target_points"], b["occluded"]
        B, T = v.shape[0], v.shape[1]
        N = q.shape[1]

        print("batch {}  {:.1f}s".format(i, dt))
        print("  video         {}  {}  range [{:.2f}, {:.2f}]".format(
            tuple(v.shape), v.dtype, float(v.min()), float(v.max())))
        print("  query_points  {}  target_points {}  occluded {}".format(
            tuple(q.shape), tuple(tp.shape), tuple(oc.shape)))

        shape_ok = (v.shape[-1] == 3 and q.shape[-1] == 3 and
                    tp.shape == (B, N, T, 2) and oc.shape == (B, N, T))
        range_ok = float(v.min()) >= -1.01 and float(v.max()) <= 1.01

        # query is 't y x'; target_points is 'x y'. At the query frame they must agree.
        qt = q[0, :, 0].long().clamp(0, T - 1)
        at_q = tp[0, torch.arange(N), qt]                      # (N, 2) xy
        want = torch.stack([q[0, :, 2], q[0, :, 1]], -1)       # xy from tyx
        d = (at_q - want).abs().max().item()
        anchor_ok = d < 0.01

        occ_frac = float(oc.mean())
        occ_ok = 0.01 < occ_frac < 0.9
        # A track that is occluded at some point but not always is what the new block has
        # to exploit: neighbours visible while this one is hidden.
        per_track = oc[0].mean(-1)
        mixed = float(((per_track > 0.02) & (per_track < 0.98)).float().mean())

        print("  shapes {}   video range {}   query anchored {} (max {:.4f}px)".format(
            "ok" if shape_ok else "BAD", "ok" if range_ok else "BAD",
            "ok" if anchor_ok else "BAD", d))
        print("  occluded {:.1%} of samples {}   tracks with a partial occlusion {:.1%}"
              .format(occ_frac, "ok" if occ_ok else "DEGENERATE", mixed))
        ok &= shape_ok and range_ok and anchor_ok and occ_ok

    print()
    print("{}  the pipeline emits what the model expects".format("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
