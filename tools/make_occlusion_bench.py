"""Composite moving occluders over a bench/synth shot, keeping its exact ground truth.

bench/README.md is explicit that the synthetic bench does NOT measure occlusion: "one
plane, nothing moves relative to anything else". That is the one thing this whole
experiment is about -- cross-track attention is the component with the
occluded-point number attached to it (paper Table 3: occluded 35.9 -> 41.0, visible only
71.3 -> 72.9), so a bench that cannot see occlusion cannot decide whether to build it.

The plate stays a rigid plane warped by a known homography, so the true position of every
seed on every frame is still exact arithmetic. Occluders are drawn ON TOP, and their
geometry is written out per frame, so the scorer can label each (track, frame) sample as
visible or occluded from the ground truth rather than from the tracker's own opinion --
asking the tracker whether it was occluded is how a metric ends up measuring itself.

Occluders are textured, not flat: a solid black box is easier to survive than a busy one,
because a tracker with any photometric check simply fails to match and coasts. Real
occluders in a plate carry detail, and detail is what pulls a tracker off its feature.

    python make_occlusion_bench.py ^
        --src bench\\synth\\lab02 --out bench\\synth\\lab02_occ
"""
from __future__ import annotations

import argparse
import glob
import json
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def build_occluders(W: int, H: int, T: int, n: int, seed: int):
    """n rigid rectangles sweeping across the frame on straight paths.

    Each is described per frame as an axis-aligned box, which is what the scorer tests
    against. Rotation is deliberately left out: a rotated box would need the scorer to
    carry the same polygon test, and an axis-aligned box already occludes honestly.
    """
    rng = np.random.default_rng(seed)
    occ = []
    for i in range(n):
        # Alternate horizontal and vertical sweeps so tracks are crossed from both axes,
        # and size them against the frame so the crossing lasts a real number of frames.
        vertical = (i % 2 == 1)
        bw = int(rng.uniform(0.06, 0.14) * W)
        bh = int(rng.uniform(0.30, 0.75) * H)
        if vertical:
            bw, bh = int(rng.uniform(0.30, 0.75) * W), int(rng.uniform(0.06, 0.14) * H)
        # Start fully outside and end fully outside, so every track it meets sees a clean
        # enter/exit rather than an occluder that is simply present the whole shot.
        if vertical:
            y0, y1 = -bh - 10, H + 10
            x0 = x1 = rng.uniform(0.1, 0.9) * W - bw / 2
        else:
            x0, x1 = -bw - 10, W + 10
            y0 = y1 = rng.uniform(0.1, 0.9) * H - bh / 2
        if rng.random() < 0.5:
            x0, x1, y0, y1 = x1, x0, y1, y0
        # Stagger so the crossings do not all happen at once, and slow them to about
        # 60% of the shot so each occluder is off-frame at both ends of its span.
        lead = rng.uniform(0.0, 0.35)
        span = rng.uniform(0.45, 0.65)
        tex = rng.integers(0, 255, (max(8, bh // 8), max(8, bw // 8), 3), dtype=np.uint8)
        tex = cv2.GaussianBlur(cv2.resize(tex, (bw, bh), interpolation=cv2.INTER_LINEAR),
                               (5, 5), 0)
        boxes = []
        for t in range(T):
            u = (t / max(1, T - 1) - lead) / span
            if u < 0.0 or u > 1.0:
                boxes.append(None)
                continue
            x = x0 + (x1 - x0) * u
            y = y0 + (y1 - y0) * u
            boxes.append([float(x), float(y), float(bw), float(bh)])
        occ.append({"id": i, "w": bw, "h": bh, "boxes": boxes, "_tex": tex})
    return occ


def main() -> int:
    ap = argparse.ArgumentParser(description="add occluders to a bench/synth shot")
    ap.add_argument("--src", required=True, help="source bench shot (needs gt.json, plate/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--occluders", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    gt_path = os.path.join(a.src, "gt.json")
    if not os.path.isfile(gt_path):
        raise SystemExit("[ERROR] {} has no gt.json -- not a bench shot".format(a.src))
    with open(gt_path) as fh:
        gt = json.load(fh)
    files = sorted(glob.glob(os.path.join(a.src, "plate", "*.png")))
    if not files:
        raise SystemExit("[ERROR] no frames in {}/plate".format(a.src))

    W, H, T = int(gt["width"]), int(gt["height"]), len(files)
    occ = build_occluders(W, H, T, a.occluders, a.seed)

    os.makedirs(os.path.join(a.out, "plate"), exist_ok=True)
    covered = 0
    for t, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        for o in occ:
            b = o["boxes"][t]
            if b is None:
                continue
            x, y, bw, bh = int(round(b[0])), int(round(b[1])), o["w"], o["h"]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(W, x + bw), min(H, y + bh)
            if x1 <= x0 or y1 <= y0:
                continue
            img[y0:y1, x0:x1] = o["_tex"][y0 - y:y1 - y, x0 - x:x1 - x]
            covered += (x1 - x0) * (y1 - y0)
        cv2.imwrite(os.path.join(a.out, "plate", os.path.basename(f)), img)

    # The ground truth is unchanged -- the plane still moves by the same homography, the
    # occluders merely hide parts of it. Copy it verbatim so the same scorers apply.
    with open(os.path.join(a.out, "gt.json"), "w") as fh:
        json.dump(gt, fh)
    with open(os.path.join(a.out, "occluders.json"), "w") as fh:
        json.dump({"width": W, "height": H, "frames": T, "source": a.src,
                   "occluders": [{k: v for k, v in o.items() if k != "_tex"}
                                 for o in occ]}, fh)

    print("[out] {}  {} frames  {} occluders  mean cover {:.1f}% of frame".format(
        a.out, T, len(occ), 100.0 * covered / float(T * W * H)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
