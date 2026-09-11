"""Prove an untrained JeffTracker is LocoTrack -- bit for bit, not approximately.

This is the gate that makes every Stage-3 number meaningful. If JeffTracker at
zero-initialisation differs from LocoTrack at all, then a measured change after training
is a mixture of "cross-track attention helped" and "the port moved something", and there
is no way to tell the two apart afterwards. Held exactly, one binary produces both the
baseline and the treatment on identical footage, which is the same property
the same property check_identity.py enforces preserves for per-track policy.

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

    python tools/check_identity.py
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jefftrack.paths import add_vendor_to_path  # noqa: E402

add_vendor_to_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.model.jefftrack_model import load_jefftrack  # noqa: E402
from jefftrack.engine import DEFAULT_CKPT, _load_locotrack  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="untrained JeffTracker must equal LocoTrack")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--points", type=int, default=24)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--pyramid-level", type=int, default=0,
                    help="extra correlation levels (CoTracker3 runs four; LocoTrack ships "
                         "three). The new channels enter the mixer at ZERO weight, so they "
                         "contribute nothing -- but they WIDEN the input_proj GEMM, which "
                         "changes its accumulation order. Measured: 3.4e-07 relative on "
                         "the projection itself, which the four refinement iterations "
                         "amplify to ~0.04 px on CPU and ~0.21 px on CUDA. So this path "
                         "is checked against a TOLERANCE, not against zero, and that is a "
                         "property of float32, not a defect to be fixed.")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="max |diff| accepted on tracks. 0 means bit-identical, which is "
                         "the right gate for every path except --pyramid-level.")
    ap.add_argument("--carry-state", action="store_true",
                    help="build the cross blocks with proxy tokens carried through the "
                         "vendor's temporal attention. That path adds ROWS to the tensor "
                         "the time blocks see, so it has to be proved not to disturb the "
                         "real tracks -- which is exactly what this script tests.")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = _load_locotrack(a.ckpt, a.model_size, device)
    db = load_jefftrack(a.ckpt, a.model_size, device=device,
                        carry_state=a.carry_state, pyramid_level=a.pyramid_level)

    # ---------------------------------------------------------------- 1. parameters
    bsd, dsd = base.state_dict(), db.state_dict()
    extra = [k for k in dsd if k not in bsd]
    lost = [k for k in bsd if k not in dsd]
    bad = []
    # A widened correlation pyramid deliberately replaces input_proj with a spliced, wider
    # tensor, so it cannot match the vendor's and is not evidence of a fault. Nothing else
    # is exempt.
    widened = "torch_pips_mixer.input_proj.weight"
    for k in bsd:
        if k == widened and a.pyramid_level:
            continue
        if k in dsd and not torch.equal(bsd[k].cpu(), dsd[k].cpu()):
            bad.append(k)
    print("vendor parameters : {} shared, {} differ, {} lost, {} added by JeffTracker"
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
    worst = 0.0
    for k in ("tracks", "occlusion", "expected_dist"):
        d = (ob[k] - od[k]).abs().max().item()
        same = torch.equal(ob[k], od[k])
        exact &= same
        worst = max(worst, d)
        print("{:<16}{:>14.3e}{:>14}".format(k, d, str(same)))

    # --tol exists for ONE case: a widened correlation pyramid, where the new channels are
    # spliced in at zero and so contribute nothing arithmetically, but the wider GEMM
    # accumulates in a different order. Every other path must be bit-identical, and the
    # default tol of 0 keeps it that way -- a tolerance that applied everywhere would let a
    # real leak hide under it.
    ok = (not bad) and (not lost) and zero_ok and (exact or worst <= a.tol)
    print()
    if ok and not exact:
        print("within tolerance: worst {:.3e} <= --tol {:.3e} (not bit-identical; expected "
              "only with --pyramid-level)".format(worst, a.tol))
    print("{}  untrained JeffTracker {} LocoTrack".format(
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
