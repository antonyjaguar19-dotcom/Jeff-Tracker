"""Put pydeps/ and the vendored LocoTrack on sys.path, from inside the process.

The embeddable interpreter runs in isolated mode, so PYTHONPATH is ignored and this is the
only way these land on the path. pydeps must precede vendor for the same reason
experiments/track_on/run_trackon.py:36-43 orders them that way: a shim in pydeps has to be
what an `import` inside the vendor finds.
"""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PYDEPS = os.path.join(ROOT, "pydeps")
VENDOR_PT = os.path.join(ROOT, "vendor", "locotrack", "locotrack_pytorch")


def add_vendor_to_path() -> None:
    for p in (PYDEPS, VENDOR_PT):
        if p not in sys.path:
            sys.path.insert(0, p)
