"""How long a clip fits in one window, measured rather than guessed.

`jefftrack_engine._auto_window` caps a window at 250 frames for 256x256. That number was a
headroom guess, and on SH006 (312 frames) it cost more than half the tracks in the last
quarter of the shot for **0.26 GB** -- 6.87 GB windowed against 7.13 GB in one window. Any
plate longer than the cap pays a coverage penalty at the seam, because a later window
re-finds every track from its appearance on the QUERY frame, which on a long clip is a
jump of hundreds of frames in one step with no intermediate context.

So the cap wants a measurement. This runs ONE configuration and prints a JSON row:

    python tools/measure_window.py --frames 400 --tracks 400 --res 256x256

One config per process on purpose. An out-of-memory error leaves the CUDA allocator
fragmented and the next measurement in the same process reads low, which is precisely the
direction that would produce an over-confident default. The driver loop is in the
docstring of `main`; the table it produced is docs/verdicts/window_budget.txt.

Frames are random noise. Content does not change the allocation -- the tensors are the same
shape whatever the pixels are -- and this way the sweep needs no plate on disk.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.engine import JeffTrackEngine  # noqa: E402


def measure(frames: int, tracks: int, res, ckpt: str, work=(960, 540)) -> dict:
    W, H = work
    rng = np.random.default_rng(0)
    clip = rng.integers(0, 255, (frames, H, W, 3), dtype=np.uint8)
    # Queries on frame 0, spread over the image the way `--seed corners` would land them.
    n = int(tracks)
    gx = int(np.ceil(np.sqrt(n * W / float(H))))
    gy = int(np.ceil(n / float(gx)))
    xs = np.linspace(8, W - 8, gx)
    ys = np.linspace(8, H - 8, gy)
    pts = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)[:n]
    q = np.concatenate([np.zeros((len(pts), 1), np.float32), pts.astype(np.float32)], 1)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    eng = JeffTrackEngine(ckpt=ckpt, model_res=res, arch="jefftrack",
                          window=frames)          # force ONE window
    import time
    t0 = time.time()
    eng.track_queries_conf(clip, q)
    dt = time.time() - t0
    peak = torch.cuda.max_memory_reserved() / 1e9
    return {"frames": frames, "tracks": n, "res": "{}x{}".format(*res),
            "peak_gb": round(peak, 2), "seconds": round(dt, 1),
            "s_per_frame": round(dt / frames, 4), "ok": True}


def main() -> int:
    """Driver used to produce the table:

        for f in 250 312 400 500 600 700 800 1000; do
          for n in 400 800; do
            python tools/measure_window.py --frames $f --tracks $n
          done
        done
    """
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--frames", type=int, required=True)
    ap.add_argument("--tracks", type=int, default=400)
    ap.add_argument("--res", default="256x256")
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "weights", "c8_kubric.ckpt"))
    a = ap.parse_args()
    w, h = (int(v) for v in a.res.lower().split("x"))

    try:
        row = measure(a.frames, a.tracks, (w, h), a.ckpt)
    except torch.cuda.OutOfMemoryError:
        row = {"frames": a.frames, "tracks": a.tracks, "res": a.res,
               "peak_gb": None, "ok": False, "why": "CUDA out of memory"}
    except RuntimeError as exc:
        oom = "out of memory" in str(exc).lower()
        row = {"frames": a.frames, "tracks": a.tracks, "res": a.res, "peak_gb": None,
               "ok": False, "why": "CUDA out of memory" if oom else str(exc)[:120]}
    print("ROW " + json.dumps(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
