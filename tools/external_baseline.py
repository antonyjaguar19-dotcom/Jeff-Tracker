"""Run an external comparison tracker through an adapter file the user supplies.

No third-party tracker is vendored, downloaded or imported by this repository. To put one
in a comparison, write a small adapter file and point JT_BASELINE_ADAPTER at it. The
adapter defines:

    def track(frames, points):
        '''frames: (T, H, W, 3) uint8 BGR, exactly as jefftrack.io loads them.
        points: (N, 2) float32 [x, y] in plate pixels, queried on frame 0.
        Returns (tracks, vis): tracks (T, N, 2) [x, y] in plate pixels, vis (T, N) bool.'''

and may optionally define:

    LABEL = "Baseline"           # name used in overlays and score sheets
    INTERNAL_RES = "384x512"     # what the tracker resizes to internally, for the record

Without JT_BASELINE_ADAPTER set, the baseline engine reports itself unavailable and every
other engine still runs.
"""
from __future__ import annotations

import importlib.util
import os
import time

import numpy as np

ENV = "JT_BASELINE_ADAPTER"
DEFAULT_LABEL = "Baseline"


def load_adapter(path: str | None = None):
    path = path or os.environ.get(ENV, "")
    if not path:
        raise RuntimeError(
            "No baseline tracker is set up here, by design -- none is vendored. To include "
            "one in the comparison, set {} to an adapter file that defines "
            "track(frames, points); see tools/external_baseline.py.".format(ENV))
    if not os.path.isfile(path):
        raise RuntimeError("{} does not point at a file: {}".format(ENV, path))
    spec = importlib.util.spec_from_file_location("jt_baseline_adapter", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, "track", None)):
        raise RuntimeError("baseline adapter {} defines no track(frames, points)".format(path))
    return mod


def label(path: str | None = None) -> str:
    try:
        return getattr(load_adapter(path), "LABEL", DEFAULT_LABEL)
    except RuntimeError:
        return DEFAULT_LABEL


def run_baseline(frames, pts, path: str | None = None):
    """Returns (tracks, vis, conf, dt, peak_gb, internal_res).

    Baseline visibility is taken as a boolean, so conf is that boolean stated as a float --
    there is no continuous confidence to rank by.
    """
    mod = load_adapter(path)
    try:
        import torch  # noqa: PLC0415
        cuda = torch.cuda.is_available()
    except ImportError:
        cuda = False
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tracks, vis = mod.track(frames, np.asarray(pts, np.float32))
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0
    tracks = np.asarray(tracks, np.float32)
    vis = np.asarray(vis).astype(bool)
    return tracks, vis, vis.astype(np.float32), dt, peak, getattr(mod, "INTERNAL_RES", "native")
