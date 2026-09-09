"""Prove an untrained Jeff-Tracker is LocoTrack -- bit for bit, not approximately.

This is the gate that makes every Stage-3 number meaningful. If Jeff-Tracker at
zero-initialisation differs from LocoTrack at all, then a measured change after training
is a mixture of "cross-track attention helped" and "the port moved something", and there
is no way to tell the two apart afterwards. Held exactly, one binary produces both the
baseline and the treatment on identical footage, which is the same property
the per-track config view in a downstream integration preserves for per-track policy.

Three things are checked, and all three have to pass:

  1. every vendor parameter is byte-identical between the two models -- no rename, no
     silent re-initialisation, no dtype change;
  2. the cross-track blocks' output projections are exactly zero, which is what makes the
     block contribute nothing;
  3. tracks, occlusion and expected_dist agree to 0.0 on a real forward pass.

Bit-for-bit is achievable here because the added path contributes `x + out_proj(...)` with
out_proj identically zero: the addition of a hard zero is exact in floating point. If this
ever starts reporting a tiny non-zero difference, something has been reordered in the
vendor's own arithmetic and that is worth knowing about, not worth loosening a tolerance
for.

    python check_identity.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from jefftrack.paths import add_vendor_to_path  # noqa: E402

add_vendor_to_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.model.jefftrack_model import load_jefftrack  # noqa: E402
from jefftrack.engine import DEFAULT_CKPT, _load_locotrack  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="untrained Jeff-Tracker must equal LocoTrack")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--res", type=int, default=256)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = _load_locotrack(a.ckpt, a.model_size, device)
    db = load_jefftrack(a.ckpt, a.model_size, device=device)

    # ---------------------------------------------------------------- 1. parameters
    bsd, dsd = base.state_dict(), db.state_dict()
    extra = [k for k in dsd if k not in bsd]
    lost = [k for k in bsd if k not in dsd]
    bad = []
    for k in bsd:
        if k in dsd and not torch.equal(bsd[k].cpu(), dsd[k].cpu()):
            bad.append(k)
    print("vendor parameters : {} shared, {} differ, {} lost, {} added by Jeff-Tracker"
          .format(len(bsd), len(bad), len(lost), len(extra)))
    if bad[:3]:
        print("  differing: {}".format(bad[:3]))
    if lost[:3]:
        print("  lost:      {}".format(lost[:3]))

    # ---------------------------------------------------------------- 2. zero gate
    zero_ok = True
    for name, blk in db.torch_pips_mixer.cross.items():
        w = blk.out_proj.weight.detach()
        b = blk.out_proj.bias.detach()
        ok = bool((w == 0).all() and (b == 0).all())
        zero_ok &= ok
        print("cross block {:<3}  out_proj zero: {:<5}  params {}".format(
            name, str(ok), sum(p.numel() for p in blk.parameters())))

    # ---------------------------------------------------------------- 3. forward pass
    rng = np.random.default_rng(0)
    vid = torch.from_numpy(
        rng.integers(0, 255, (1, a.frames, a.res, a.res, 3), dtype=np.uint8)).to(device)
    vid = vid.float() / 255.0 * 2 - 1
    q = np.stack([np.zeros(a.points),
                  rng.uniform(20, a.res - 20, a.points),
                  rng.uniform(20, a.res - 20, a.points)], 1)
    q = torch.from_numpy(q[None]).float().to(device)

    with torch.no_grad():
        ob = base(vid, q, query_chunk_size=64)
        od = db(vid, q, query_chunk_size=64)

    print()
    print("{:<16}{:>14}{:>14}".format("output", "max |diff|", "identical"))
    print("-" * 44)
    exact = True
    for k in ("tracks", "occlusion", "expected_dist"):
        d = (ob[k] - od[k]).abs().max().item()
        same = torch.equal(ob[k], od[k])
        exact &= same
        print("{:<16}{:>14.3e}{:>14}".format(k, d, str(same)))

    ok = (not bad) and (not lost) and zero_ok and exact
    print()
    print("{}  untrained Jeff-Tracker {} LocoTrack".format(
        "PASS" if ok else "FAIL", "is" if ok else "is NOT"))
    if ok:
        n_new = sum(p.numel() for p in db.cross_parameters())
        n_all = sum(p.numel() for p in db.parameters())
        print("  {} cross-track parameters added to {} total ({:.1f}%) -- this is what a "
              "fine-tune trains with freeze_base().".format(
                  n_new, n_all, 100.0 * n_new / n_all))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
