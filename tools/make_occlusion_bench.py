"""Composite moving occluders over a bench/synth shot, keeping its exact ground truth.

bench/README.md is explicit that the synthetic bench does NOT measure occlusion: "one
plane, nothing moves relative to anything else". That is the one thing this whole
experiment is about -- cross-track attention is the CoTracker3 component with the
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

    python tools/make_occlusion_bench.py \
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


def _apply_h(H, pts):
    """Map (N,2) points through a 3x3 homography."""
    p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ np.asarray(H, float).T
    return p[:, :2] / p[:, 2:3]


def build_depth_occluders(W, H, T, n, seed, Hs, depth=2.2):
    """Occluders on a plane at a DIFFERENT DEPTH, under the same camera motion.

    What the box occluders above cannot test. They translate across the frame on a straight
    line at constant speed, which is a card slid over a photo: the occluder's motion is
    unrelated to the camera's, so its edge carries no parallax and nothing ever goes
    *behind* anything. Real occlusion in a plate is a depth discontinuity -- the background
    slides past the foreground because they are at different distances, and the edge where
    they meet is the hardest place in the shot for a tracker precisely because the pixels on
    either side of it are moving differently.

    Here each occluder is a convex polygon on its own fronto-parallel plane. The background
    plane's induced displacement for frame t is known exactly -- it is the bench's own
    homography `Hs[t]` -- so a plane at `depth` times the disparity moves by that
    displacement scaled, which is the first-order parallax relation for a translating
    camera. `depth > 1` puts the occluder NEARER than the background, which is the case that
    matters: a foreground object sweeping past a background the camera is tracking.

    Exactness is preserved, which is the whole reason the bench is worth having. Every
    vertex is arithmetic on `Hs[t]`, so the silhouette is known per frame and the scorer can
    label a (track, frame) sample occluded from geometry rather than from the tracker's
    opinion.
    """
    rng = np.random.default_rng(seed)
    occ = []
    for i in range(n):
        # An irregular convex silhouette, not a rectangle: a straight vertical edge is an
        # unrealistically easy thing to survive, because the frames where a track is
        # half-covered are the ones that pull it off its feature and a box makes those
        # frames identical for every track it crosses.
        # Elongated ACROSS the direction of travel, like the box occluders: a compact blob
        # sweeping past covers each track for two or three frames, which is not an occlusion
        # so much as a flicker. The box bench gets 6898 occluded samples out of 100 frames
        # and a silhouette has to be sized to match, or the two benches are not comparable
        # and this one has no statistical power at all.
        vertical = (i % 2 == 1)
        k = int(rng.integers(5, 8))
        rad = rng.uniform(0.16, 0.28) * min(W, H)
        ang = np.sort(rng.uniform(0, 2 * np.pi, k))
        r = rad * rng.uniform(0.70, 1.30, k)
        long_ax = rng.uniform(2.2, 4.0)
        sx, sy = (long_ax, 1.0) if vertical else (1.0, long_ax)
        base = np.stack([np.cos(ang) * r * sx, np.sin(ang) * r * sy], 1)

        # Start off-frame, cross, end off-frame, so every track it meets sees a clean
        # enter and exit rather than an occluder that is simply present all shot.
        pad = rad * long_ax * 1.6
        if vertical:
            cx = rng.uniform(0.15, 0.85) * W
            p0 = np.array([cx, -pad])
            p1 = np.array([cx, H + pad])
        else:
            cy = rng.uniform(0.15, 0.85) * H
            p0 = np.array([-pad, cy])
            p1 = np.array([W + pad, cy])
        if rng.random() < 0.5:
            p0, p1 = p1, p0
        lead = rng.uniform(0.0, 0.35)
        span = rng.uniform(0.45, 0.65)

        tex = rng.integers(0, 255, (24, 24, 3), dtype=np.uint8)
        tex = cv2.GaussianBlur(
            cv2.resize(tex, (max(16, int(rad)), max(16, int(rad))),
                       interpolation=cv2.INTER_LINEAR), (5, 5), 0)

        polys = []
        for t in range(T):
            u = (t / max(1, T - 1) - lead) / span
            if u < 0.0 or u > 1.0:
                polys.append(None)
                continue
            centre = p0 + (p1 - p0) * u              # the occluder's own object motion
            verts = base + centre
            # ...plus the camera's effect on a plane at this depth: the background's own
            # displacement for these pixels, scaled by the depth ratio.
            moved = _apply_h(Hs[t], verts)
            verts = verts + depth * (moved - verts)
            polys.append([[float(x), float(y)] for x, y in verts])
        occ.append({"id": i, "depth": float(depth), "polys": polys, "_tex": tex,
                    "_rad": float(rad)})
    return occ


def draw_poly_occluder(img, o, t):
    """Composite one polygonal occluder onto frame t. Returns pixels covered."""
    poly = o["polys"][t]
    if poly is None:
        return 0
    pts = np.array(poly, np.float32)
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.fillConvexPoly(mask, cv2.convexHull(np.int32(np.round(pts))), 255)
    tex, (h, w) = o["_tex"], img.shape[:2]
    tiled = np.tile(tex, (h // tex.shape[0] + 1, w // tex.shape[1] + 1, 1))[:h, :w]
    # Shift the texture with the occluder so it reads as one moving object rather than a
    # window onto a static pattern -- a static texture inside a moving hole is a cue no real
    # occluder gives, and a tracker can exploit it.
    c = pts.mean(0)
    M = np.float32([[1, 0, c[0] % tex.shape[1]], [0, 1, c[1] % tex.shape[0]]])
    tiled = cv2.warpAffine(tiled, M, (w, h), borderMode=cv2.BORDER_WRAP)
    img[mask > 0] = tiled[mask > 0]
    return int((mask > 0).sum())


def main() -> int:
    ap = argparse.ArgumentParser(description="add occluders to a bench/synth shot")
    ap.add_argument("--src", required=True, help="source bench shot (needs gt.json, plate/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--occluders", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--depth", type=float, default=0.0,
                    help="build occluders on a plane at this depth ratio relative to the "
                         "background instead of sliding axis-aligned boxes. >1 puts them "
                         "NEARER the camera, so the background parallaxes past their edge "
                         "-- the thing a pasted card cannot test. 0 keeps the box "
                         "occluders, so every existing bench reproduces exactly.")
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
    depth_mode = a.depth > 0.0
    if depth_mode:
        Hs = gt.get("H")
        if not Hs or len(Hs) < T:
            raise SystemExit(
                "[ERROR] --depth needs the per-frame homography in gt.json ('H'); "
                "{} has {}".format(gt_path, "none" if not Hs else len(Hs)))
        occ = build_depth_occluders(W, H, T, a.occluders, a.seed, Hs, a.depth)
    else:
        occ = build_occluders(W, H, T, a.occluders, a.seed)

    os.makedirs(os.path.join(a.out, "plate"), exist_ok=True)
    covered = 0
    for t, f in enumerate(files):
        img = cv2.imread(f, cv2.IMREAD_COLOR)
        if depth_mode:
            for o in occ:
                covered += draw_poly_occluder(img, o, t)
            cv2.imwrite(os.path.join(a.out, "plate", os.path.basename(f)), img)
            continue
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

    print("[out] {}  {} frames  {} {} occluders  mean cover {:.1f}% of frame".format(
        a.out, T, len(occ), "depth-{:g}".format(a.depth) if depth_mode else "box",
        100.0 * covered / float(T * W * H)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
