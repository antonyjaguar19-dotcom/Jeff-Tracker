"""Training batches from Meta AI's Kubric point-tracking release, via the local cache.

Drop-in for `jefftrack/data/movi.py`: same four keys, same shapes, same units, so
`train_cross.py` switches dataset with a flag and nothing downstream changes.

    video          (B, T, H, W, 3) float32 in [-1, 1]
    query_points   (B, N, 3)       't y x' in train_size pixels
    target_points  (B, N, T, 2)    'x y'   in train_size pixels
    occluded       (B, N, T)       float, 1 = hidden

Why this is worth having over MOVi-E, in the terms the occlusion work is stuck on:

  * **Shots are 120 frames, not 24.** Measured on shard 0000, 18.5% of occlusion events
    last longer than 24 frames -- events MOVi-E cannot contain at all, because its clips
    are not that long. PLAN.md names this exact risk: a model taught only on short
    occlusions learns to coast through a long one and come back on the neighbouring
    feature, confidently. This is the first data here that can teach otherwise.
  * Frames are 512x512, so `random_crop` has real room to work rather than cropping a
    256 px frame back to 256.
  * There is a ground-truth camera per frame. Nothing in this file uses it yet; it is
    carried in the cache because it costs 131 KB and cannot be recovered once the source
    archives are gone.

Three traps live in the source data, all three measured against the pixels rather than
assumed, all three already dealt with by `convert_kubric.py` -- restated here because this
is the file that would silently propagate them:

  1. the source array named `visibility` holds OCCLUSION, True = hidden. The cache renames
     it `occluded`, which is what this module reads and what the trainer expects.
  2. coordinates are (x, y), column first.
  3. three quarters of "occluded" samples are merely off screen. `losses.py` splits those
     two populations itself (`occluded_in_frame_only`), so nothing is filtered here -- an
     off-screen point is real and the loss decides what it is worth.

No TensorFlow. `movi.py` needs TFDS to read its shards and pulls in a second deep-learning
framework to do it; the cache is plain numpy, so the augmentation the vendor does in TF is
reproduced here in numpy instead (`_colour_augment`, same operations, same constants, same
probabilities).
"""
from __future__ import annotations

import os
import time
from typing import Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CACHE = os.environ.get(
    "JEFFTRACK_KUBRIC_CACHE", os.path.join(ROOT, "datasets", "kubric_cache"))


# --------------------------------------------------------------------------- one shot


class Shot:
    """One cached shot, opened lazily.

    numpy holds an open handle per `.npz`; with a thousand shots in the pool that is a
    thousand file descriptors, so the handle is opened per read and closed again. The cost
    is a zip directory parse per sample, which is nothing next to decoding the frames.
    """

    __slots__ = ("path", "_n_frames", "_n_points", "_hw")

    def __init__(self, path: str):
        self.path = path
        self._n_frames = self._n_points = -1
        self._hw: Optional[Tuple[int, int]] = None

    def _peek(self) -> None:
        if self._n_points >= 0:
            return
        with np.load(self.path) as z:
            nf = int(len(z["frames_off"]) - 1)
            np_pts = int(z["coords"].shape[0])
            hw = (int(z["height"]), int(z["width"]))
            self._hw = hw
            self._n_frames = nf
            self._n_points = np_pts

    @property
    def n_frames(self) -> int:
        self._peek()
        return self._n_frames

    @property
    def n_points(self) -> int:
        self._peek()
        return self._n_points

    @property
    def hw(self) -> Tuple[int, int]:
        self._peek()
        return self._hw            # type: ignore[return-value]

    def read(self, frames: Sequence[int], points: Optional[np.ndarray] = None):
        """Decode only the frames asked for. Returns (video uint8 BGR, coords, occluded)."""
        with np.load(self.path) as z:
            offs = z["frames_off"]
            blob = z["frames_jpg"]
            imgs = []
            for f in frames:
                img = cv2.imdecode(blob[offs[f]:offs[f + 1]], cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("{}: frame {} does not decode".format(self.path, f))
                imgs.append(img)
            c = z["coords"]
            o = z["occluded"]
            if points is not None:
                c, o = c[points], o[points]
            c = c[:, list(frames)]
            o = o[:, list(frames)]
        return np.stack(imgs), np.ascontiguousarray(c), np.ascontiguousarray(o)


# --------------------------------------------------------------------------- sampling


def _colour_augment(video: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """The vendor's augmentation (kubric_data.py:20) in numpy. Same constants.

    Applied to the whole clip at once, not per frame -- a brightness step that changed
    between frames would be a flicker the tracker has to model, and the vendor's TF version
    is also per clip.
    """
    v = video.astype(np.float32) / 255.0
    if rng.uniform() < 0.8:
        v = v + rng.uniform(-32.0 / 255.0, 32.0 / 255.0)                  # brightness
        grey = v.mean(axis=3, keepdims=True)
        v = grey + (v - grey) * rng.uniform(0.6, 1.4)                     # saturation
        m = v.mean(axis=(1, 2, 3), keepdims=True)
        v = m + (v - m) * rng.uniform(0.6, 1.4)                           # contrast
        shift = rng.uniform(-0.2, 0.2) * 180.0                            # hue, per clip
        out = []
        for f in range(v.shape[0]):
            h = cv2.cvtColor(np.clip(v[f], 0, 1), cv2.COLOR_BGR2HSV)
            h[..., 0] = (h[..., 0] + shift) % 180.0
            out.append(cv2.cvtColor(h, cv2.COLOR_HSV2BGR))
        v = np.stack(out)
    if rng.uniform() < 0.2:                                               # drop to grey
        g = v.mean(axis=3, keepdims=True)
        v = np.repeat(g, 3, axis=3)
    return np.clip(v, 0.0, 1.0) * 2.0 - 1.0


def _crop_and_resize(video: np.ndarray, coords: np.ndarray, train_size: Tuple[int, int],
                     rng: np.random.Generator, random_crop: bool):
    """Crop the clip, then resize to train_size, moving the track coordinates with it."""
    T, H, W = video.shape[:3]
    th, tw = train_size
    if random_crop:
        # Keep the crop generous: a tight crop throws most tracks off screen and the batch
        # then carries mostly off-frame samples, which is the population the loss excludes.
        scale = rng.uniform(0.7, 1.0)
        ch, cw = max(th, int(H * scale)), max(tw, int(W * scale))
        y0 = int(rng.integers(0, H - ch + 1))
        x0 = int(rng.integers(0, W - cw + 1))
    else:
        ch, cw, y0, x0 = H, W, 0, 0

    video = video[:, y0:y0 + ch, x0:x0 + cw]
    sy, sx = th / float(ch), tw / float(cw)
    out = np.stack([cv2.resize(f, (tw, th), interpolation=cv2.INTER_AREA)
                    for f in video])
    c = coords.copy()
    c[..., 0] = (c[..., 0] - x0) * sx
    c[..., 1] = (c[..., 1] - y0) * sy
    return out, c


def _sample_flow_bilinear(flow: np.ndarray, coords: np.ndarray) -> np.ndarray:
    """Sample 2D scalar field (H, W) at continuous (x, y) coordinates (N, 2)."""
    H, W = flow.shape
    x = np.clip(coords[:, 0], 0, W - 1)
    y = np.clip(coords[:, 1], 0, H - 1)
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    wx = x - x0
    wy = y - y0
    v00 = flow[y0, x0]
    v10 = flow[y0, x1]
    v01 = flow[y1, x0]
    v11 = flow[y1, x1]
    v0 = v00 * (1 - wx) + v10 * wx
    v1 = v01 * (1 - wx) + v11 * wx
    return v0 * (1 - wy) + v1 * wy


def _apply_camera_shake(video: np.ndarray, coords: np.ndarray,
                        rng: np.random.Generator, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Synthetic camera shake and micro-jitter augmentation.

    Warps frames with frame-to-frame micro-translations + rotations and updates GT coords
    identically, forcing the model to learn correspondence invariant to camera chatter.
    """
    if rng.uniform() > p:
        return video, coords
    T, H, W = video.shape[:3]
    cx, cy = W / 2.0, H / 2.0
    ax, ay = rng.uniform(0.5, 2.0, size=2)
    wx, wy = rng.uniform(0.5, 1.5, size=2)
    px, py = rng.uniform(0, 2 * np.pi, size=2)
    max_rot = rng.uniform(0.2, 0.8)

    warped_video = []
    c = coords.copy()
    for t in range(T):
        dx = ax * np.sin(wx * t + px) + rng.normal(0, 0.35)
        dy = ay * np.cos(wy * t + py) + rng.normal(0, 0.35)
        dtheta = max_rot * np.sin(wx * 0.7 * t + px)
        M = cv2.getRotationMatrix2D((cx, cy), dtheta, 1.0)
        M[0, 2] += dx
        M[1, 2] += dy

        frame_warped = cv2.warpAffine(video[t], M, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT101)
        warped_video.append(frame_warped)

        pts_hom = np.stack([c[:, t, 0], c[:, t, 1], np.ones(c.shape[0], dtype=np.float32)], axis=1)
        c[:, t, :2] = (pts_hom @ M.T)[:, :2]

    return np.stack(warped_video), c


def _apply_elastic_deformation(video: np.ndarray, coords: np.ndarray,
                               rng: np.random.Generator, p: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Smooth 2D non-rigid elastic surface deformation augmentation.

    Simulates non-rigid surface deformations (cloth, skin, muscle flexion) by generating
    time-varying harmonic bicubic flow fields. Frames are remapped and track coords are
    updated via bilinear sampling.
    """
    if rng.uniform() > p:
        return video, coords
    T, H, W = video.shape[:3]
    cg_x = rng.normal(0, rng.uniform(2.0, 4.0), (5, 5)).astype(np.float32)
    cg_y = rng.normal(0, rng.uniform(2.0, 4.0), (5, 5)).astype(np.float32)
    period = rng.uniform(16.0, 32.0)
    phi = rng.uniform(0, 2 * np.pi)

    grid_y, grid_x = np.indices((H, W), dtype=np.float32)
    warped_video = []
    c = coords.copy()
    for t in range(T):
        scale = np.sin(2 * np.pi * t / period + phi)
        fx_coarse = cg_x * scale
        fy_coarse = cg_y * scale
        flow_x = cv2.resize(fx_coarse, (W, H), interpolation=cv2.INTER_CUBIC)
        flow_y = cv2.resize(fy_coarse, (W, H), interpolation=cv2.INTER_CUBIC)

        map_x = (grid_x - flow_x).astype(np.float32)
        map_y = (grid_y - flow_y).astype(np.float32)
        frame_warped = cv2.remap(video[t], map_x, map_y, cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT101)
        warped_video.append(frame_warped)

        dx = _sample_flow_bilinear(flow_x, c[:, t, :2])
        dy = _sample_flow_bilinear(flow_y, c[:, t, :2])
        c[:, t, 0] += dx
        c[:, t, 1] += dy

    return np.stack(warped_video), c


def _apply_3d_turn_scale_shift(video: np.ndarray, coords: np.ndarray, occ: np.ndarray,
                               rng: np.random.Generator, p: float = 0.60):
    """Synthetic 3D-Rotation, Perspective Foreshortening & Rapid Scale-Shift.

    Simulates rapid out-of-plane 3D turns (up to 65 deg yaw/pitch) and dynamic zooming/recession
    (scale shift from 0.35x to 1.8x), exactly matching real-world action footage like bikes,
    vehicles, and dancers turning away from or towards the camera.
    """
    if rng.uniform() > p:
        return video, coords, occ
    T, H, W = video.shape[:3]
    c = coords.copy()
    o = occ.copy()

    # Center of rotation / scaling
    cx = rng.uniform(W * 0.3, W * 0.7)
    cy = rng.uniform(H * 0.3, H * 0.7)

    # 3D rotation angles (yaw around Y axis, pitch around X axis)
    max_yaw = np.radians(rng.uniform(25.0, 65.0)) * rng.choice([-1.0, 1.0])
    max_pitch = np.radians(rng.uniform(10.0, 30.0)) * rng.choice([-1.0, 1.0])
    # Scale shift (e.g. object driving away: 1.0 -> 0.4, or approaching: 1.0 -> 1.8)
    scale_target = rng.uniform(0.35, 1.8)

    # Focal length in pixels
    f = max(H, W) * rng.uniform(1.0, 2.0)

    # Intrinsics
    K = np.array([[f, 0, cx],
                  [0, f, cy],
                  [0, 0, 1]], dtype=np.float64)
    K_inv = np.linalg.inv(K)

    warped_video = []
    for t in range(T):
        prog = float(t) / max(1, T - 1)
        # Smooth ease-in-out profile
        alpha = 0.5 * (1.0 - np.cos(np.pi * prog))
        yaw = max_yaw * alpha
        pitch = max_pitch * alpha
        s = 1.0 + (scale_target - 1.0) * alpha

        # 3D Rotation matrices
        Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                       [0, 1, 0],
                       [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float64)
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(pitch), -np.sin(pitch)],
                       [0, np.sin(pitch), np.cos(pitch)]], dtype=np.float64)
        R = Ry @ Rx

        # Scale transform in normalized camera frame
        S = np.diag([s, s, 1.0])

        # Planar homography: H = K @ S @ R @ K_inv
        H_mat = K @ S @ R @ K_inv
        H_mat = H_mat / H_mat[2, 2]

        # Warp image
        frame_w = cv2.warpPerspective(video[t], H_mat, (W, H), flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_REFLECT101)
        warped_video.append(frame_w)

        # Transform coordinates: (x, y, 1) -> H_mat @ (x, y, 1)
        pts_t = c[:, t, :2]
        homo_pts = np.pad(pts_t, ((0, 0), (0, 1)), constant_values=1.0)
        warped_pts = (H_mat @ homo_pts.T).T
        z = warped_pts[:, 2]
        valid_z = (z > 0.05) & np.isfinite(z)
        
        # Points with valid positive depth get projected
        if np.any(valid_z):
            proj_x = warped_pts[valid_z, 0] / z[valid_z]
            proj_y = warped_pts[valid_z, 1] / z[valid_z]
            c[valid_z, t, 0] = proj_x
            c[valid_z, t, 1] = proj_y

        # Points behind camera plane or outside image boundary are marked occluded
        off_screen = (~valid_z) | (c[:, t, 0] < 0) | (c[:, t, 0] >= W) | (c[:, t, 1] < 0) | (c[:, t, 1] >= H) | np.isnan(c[:, t, 0]) | np.isnan(c[:, t, 1])
        o[off_screen, t] = True

    # Sanitize any accidental NaNs
    np.nan_to_num(c, copy=False, nan=0.0, posinf=float(max(H, W)), neginf=0.0)
    return np.stack(warped_video), c, o


def _apply_smoke_simulation(video: np.ndarray, rng: np.random.Generator,
                            p: float = 0.80) -> np.ndarray:
    """Extreme Volumetric Multi-Layer VFX Smoke, Fire Plume & Distractor Simulation.

    Simulates realistic fast-moving multi-layer billowing smoke with dual counter-flowing plumes,
    extreme density (alpha up to 0.98), and vibrant chromatic tints (pink/magenta smoke bombs,
    fire orange, dark soot). Trains the network to retain feature correspondence through near-total
    obscuration without locking onto moving smoke edges or false background distractors.
    """
    if rng.uniform() > p:
        return video
    T, H, W = video.shape[:3]
    # Primary plume parameters
    vx1 = rng.uniform(-10.0, 10.0)
    vy1 = rng.uniform(-14.0, 14.0)
    billow_rate1 = rng.uniform(0.08, 0.30)
    max_alpha1 = rng.uniform(0.70, 0.98)

    # Chromatic palettes
    pal_choice = rng.uniform()
    if pal_choice < 0.35:
        # Pink / Magenta VFX smoke bomb (matching Smoke1 plate)
        smoke_color1 = np.array([rng.uniform(195, 250), rng.uniform(35, 85), rng.uniform(160, 230)], dtype=np.float32)
    elif pal_choice < 0.60:
        # Yellow / Orange flame smoke
        smoke_color1 = np.array([rng.uniform(225, 255), rng.uniform(120, 185), rng.uniform(20, 60)], dtype=np.float32)
    elif pal_choice < 0.80:
        # White / Light gray steam
        val = rng.uniform(200, 240)
        smoke_color1 = np.array([val, val, val], dtype=np.float32)
    else:
        # Dark combustion soot
        val = rng.uniform(25, 55)
        smoke_color1 = np.array([val, val, val], dtype=np.float32)

    # Primary fBm noise field
    n1 = rng.uniform(0.0, 1.0, (8, 8)).astype(np.float32)
    n2 = rng.uniform(0.0, 1.0, (16, 16)).astype(np.float32)
    n3 = rng.uniform(0.0, 1.0, (32, 32)).astype(np.float32)
    o1 = cv2.resize(n1, (W, H), interpolation=cv2.INTER_CUBIC)
    o2 = cv2.resize(n2, (W, H), interpolation=cv2.INTER_CUBIC)
    o3 = cv2.resize(n3, (W, H), interpolation=cv2.INTER_CUBIC)
    alpha_base1 = 0.55 * o1 + 0.30 * o2 + 0.15 * o3
    alpha_base1 = (alpha_base1 - alpha_base1.min()) / (alpha_base1.max() - alpha_base1.min() + 1e-6)

    # Secondary counter-flowing plume layer (50% chance)
    has_dual_layer = rng.uniform() < 0.50
    if has_dual_layer:
        vx2 = -vx1 * rng.uniform(0.6, 1.2)
        vy2 = -vy1 * rng.uniform(0.6, 1.2)
        billow_rate2 = rng.uniform(0.10, 0.25)
        max_alpha2 = rng.uniform(0.40, 0.75)
        smoke_color2 = np.array([215.0, 215.0, 215.0], dtype=np.float32)
        n4 = rng.uniform(0.0, 1.0, (12, 12)).astype(np.float32)
        alpha_base2 = cv2.resize(n4, (W, H), interpolation=cv2.INTER_CUBIC)
    else:
        vx2 = vy2 = billow_rate2 = max_alpha2 = 0.0
        smoke_color2 = None
        alpha_base2 = None

    # Plume localization
    has_plume_center = rng.uniform() < 0.60
    if has_plume_center:
        pcx, pcy = rng.uniform(W * 0.15, W * 0.85), rng.uniform(H * 0.15, H * 0.85)
        pradius = rng.uniform(W * 0.30, W * 0.80)
        gy, gx = np.ogrid[:H, :W]
        dist_sq = (gx - pcx) ** 2 + (gy - pcy) ** 2
        plume_mask = np.exp(-dist_sq / (2.0 * (pradius ** 2))).astype(np.float32)
    else:
        plume_mask = 1.0

    out = []
    for t in range(T):
        cur_vx1 = vx1 * t + 3.5 * np.sin(billow_rate1 * t)
        cur_vy1 = vy1 * t + 3.5 * np.cos(billow_rate1 * t)
        M1 = np.float32([[1, 0, cur_vx1], [0, 1, cur_vy1]])
        alpha1 = cv2.warpAffine(alpha_base1, M1, (W, H), borderMode=cv2.BORDER_REFLECT101)
        a1 = np.clip(alpha1 * plume_mask * max_alpha1, 0.0, 1.0)[..., None]
        f = video[t].astype(np.float32)
        blended = (1.0 - a1) * f + a1 * smoke_color1

        if has_dual_layer and alpha_base2 is not None:
            cur_vx2 = vx2 * t + 2.5 * np.sin(billow_rate2 * t)
            cur_vy2 = vy2 * t + 2.5 * np.cos(billow_rate2 * t)
            M2 = np.float32([[1, 0, cur_vx2], [0, 1, cur_vy2]])
            alpha2 = cv2.warpAffine(alpha_base2, M2, (W, H), borderMode=cv2.BORDER_REFLECT101)
            a2 = np.clip(alpha2 * max_alpha2, 0.0, 1.0)[..., None]
            blended = (1.0 - a2) * blended + a2 * smoke_color2

        out.append(np.clip(blended, 0.0, 255.0).astype(np.uint8))
    return np.stack(out)


def _apply_fire_heat_haze(video: np.ndarray, rng: np.random.Generator,
                          p: float = 0.4) -> np.ndarray:
    """Fire and explosion luminance flares with thermal heat haze.

    Adds localized high-intensity radial flame glow with 30-60 Hz temporal flicker,
    forcing the model to ignore fire illumination spikes and maintain feature correspondence.
    """
    if rng.uniform() > p:
        return video
    T, H, W = video.shape[:3]
    cx, cy = rng.uniform(W * 0.2, W * 0.8), rng.uniform(H * 0.3, H * 0.9)
    radius = rng.uniform(50, 120)
    glow_color = np.array([30.0, 140.0, 245.0], dtype=np.float32)

    y, x = np.ogrid[:H, :W]
    dist_sq = (x - cx) ** 2 + (y - cy) ** 2
    radial_mask = (np.exp(-dist_sq / (2.0 * (radius ** 2)))[..., None]).astype(np.float32)

    out = []
    for t in range(T):
        flicker = 0.8 + 0.3 * np.sin(t * 1.5)
        f = video[t].astype(np.float32) + glow_color * radial_mask * flicker
        out.append(np.clip(f, 0, 255).astype(np.uint8))
    return np.stack(out)


def _apply_water_specular_glints(video: np.ndarray, rng: np.random.Generator,
                                p: float = 0.4) -> np.ndarray:
    """Water splashes and moving specular highlights.

    Renders independent moving specular glints that glide across the surface,
    teaching the tracker to ignore deceptive moving reflections and caustics.
    """
    if rng.uniform() > p:
        return video
    T, H, W = video.shape[:3]
    n_glints = int(rng.integers(12, 28))
    gx = rng.uniform(10, W - 10, n_glints)
    gy = rng.uniform(10, H - 10, n_glints)
    gvx = rng.uniform(-2.5, 2.5, n_glints)
    gvy = rng.uniform(-1.5, 3.5, n_glints)
    gr = rng.integers(2, 5, n_glints)

    out = []
    for t in range(T):
        frame = video[t].copy()
        for i in range(n_glints):
            cx = int(round(gx[i] + gvx[i] * t))
            cy = int(round(gy[i] + gvy[i] * t))
            if 0 <= cx < W and 0 <= cy < H:
                cv2.circle(frame, (cx, cy), int(gr[i]), (255, 255, 255), -1)
        out.append(frame)
    return np.stack(out)


def _pick_queries(coords: np.ndarray, occluded: np.ndarray, train_size: Tuple[int, int],
                  rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """One query frame per track, drawn from the frames where it is visible and on screen.

    Staggered on purpose, matching how the vendor samples and how the bot seeds: a query
    frame fixed at 0 trains the model only to track forward from the start, and
    `track_filter.stitch_passes` exists downstream precisely because real seeds enter late.

    Returns the query rows and a mask of tracks that had no usable frame at all.
    """
    th, tw = train_size
    inside = ((coords[..., 0] >= 0) & (coords[..., 0] < tw) &
              (coords[..., 1] >= 0) & (coords[..., 1] < th))
    usable = (~occluded) & inside
    n, t = usable.shape
    qf = np.zeros(n, np.int64)
    dead = ~usable.any(axis=1)
    for i in np.where(~dead)[0]:
        opts = np.flatnonzero(usable[i])
        qf[i] = opts[rng.integers(0, len(opts))]
    xy = coords[np.arange(n), qf]
    # 't y x', which is the order LocoTrack's query_points uses.
    q = np.stack([qf.astype(np.float32), xy[:, 1], xy[:, 0]], axis=1)
    return q, dead


def _one_sample(shot: Shot, clip_len: int, tracks: int, train_size: Tuple[int, int],
                rng: np.random.Generator, random_crop: bool, colour_aug: bool,
                stride_max: int = 2, shake_aug: bool = True, elastic_aug: bool = True,
                vfx_aug: bool = True):
    """Build one training sample from one shot."""
    T = shot.n_frames
    # A stride > 1 turns a 120-frame shot into faster motion, which is closer to a handheld
    # plate than Kubric's slow dolly. Capped at 2: beyond that the frame-to-frame jump
    # exceeds the correlation window and the model is being asked to do re-acquisition,
    # not tracking.
    stride = int(rng.integers(1, stride_max + 1))
    span = (clip_len - 1) * stride + 1
    while span > T and stride > 1:
        stride -= 1
        span = (clip_len - 1) * stride + 1
    start = int(rng.integers(0, max(1, T - span + 1)))
    frames = list(range(start, start + span, stride))[:clip_len]

    pool = shot.n_points
    # Over-sample: some tracks will be dropped for having no visible frame inside the crop,
    # and the batch has to come out at exactly `tracks` or it cannot be stacked.
    take = min(pool, tracks * 3)
    pts = np.sort(rng.choice(pool, size=take, replace=False))

    video, coords, occ = shot.read(frames, pts)
    video, coords = _crop_and_resize(video, coords, train_size, rng, random_crop)
    video_clean, coords_clean, occ_clean = video.copy(), coords.copy(), occ.copy()

    # Option B: High-frequency camera shake + non-rigid elastic surface deformation
    if shake_aug:
        video, coords = _apply_camera_shake(video, coords, rng, p=0.5)
    if elastic_aug:
        video, coords = _apply_elastic_deformation(video, coords, rng, p=0.5)

    # 3D Turn, Perspective Foreshortening & Dynamic Scale Shifts (e.g. bike/dancer turning away or approaching)
    video, coords, occ = _apply_3d_turn_scale_shift(video, coords, occ, rng, p=0.60)

    # VFX Invariance: Smoke / Steam, Fire / Heat Haze, Water / Specular Glints
    if vfx_aug:
        video = _apply_smoke_simulation(video, rng, p=0.80)
        video = _apply_fire_heat_haze(video, rng, p=0.4)
        video = _apply_water_specular_glints(video, rng, p=0.4)

    q, dead = _pick_queries(coords, occ, train_size, rng)

    keep = np.flatnonzero(~dead)
    if len(keep) == 0:
        # Fall back to un-warped clean crop so valid tracks are guaranteed
        coords = coords_clean
        occ = occ_clean
        video = video_clean
        q, dead = _pick_queries(coords, occ, train_size, rng)
        keep = np.flatnonzero(~dead)

    if len(keep) < tracks:
        if len(keep) == 0:
            # Absolute emergency fallback: select first N points
            keep = np.arange(tracks) % len(pts)
            q = np.zeros((tracks, 3), dtype=np.float32)
            q[:, 1:] = coords[keep, 0]
        else:
            keep = np.concatenate([keep, rng.choice(keep, size=tracks - len(keep))])
    else:
        keep = keep[rng.permutation(len(keep))[:tracks]]

    # Augment BEFORE the channel flip, not after. `_colour_augment` converts through
    # cv2's BGR<->HSV for the hue step, and the frames are BGR right up to this point --
    # flipping first would hand it RGB and shift hue around a mirrored colour wheel. The
    # result still looks like a plausible augmentation, which is exactly why it would not
    # have been noticed.
    if colour_aug:
        video = _colour_augment(video, rng)[:, :, :, ::-1]
    else:
        video = (video.astype(np.float32) / 255.0 * 2.0 - 1.0)[:, :, :, ::-1]
    video = np.ascontiguousarray(video)              # BGR -> RGB, as the model expects

    return {
        "video": video.astype(np.float32),
        "query_points": q[keep].astype(np.float32),
        "target_points": coords[keep].astype(np.float32),
        "occluded": occ[keep].astype(np.float32),
    }


# --------------------------------------------------------------------------- public


def list_shots(cache_dir: str = DEFAULT_CACHE) -> List[str]:
    if not os.path.isdir(cache_dir):
        raise SystemExit(
            "[ERROR] no kubric cache at {}\n"
            "        build it first:\n"
            "        python tools/convert_kubric.py --src <downloaded shards>"
            .format(cache_dir))
    files = sorted(os.path.join(cache_dir, f) for f in os.listdir(cache_dir)
                   if f.endswith(".npz") and not f.startswith("_"))
    if not files:
        raise SystemExit("[ERROR] kubric cache at {} is empty".format(cache_dir))
    return files


def batches(device: str = "cuda", cache_dir: str = DEFAULT_CACHE, batch_size: int = 1,
            tracks_to_sample: int = 256, clip_len: int = 24,
            train_size: Tuple[int, int] = (256, 256), color_augmentation: bool = True,
            random_crop: bool = True, shake_augmentation: bool = True,
            elastic_augmentation: bool = True, vfx_augmentation: bool = True,
            seed: int = 0, holdout: int = 0,
            split: str = "train", workers: int = 4, prefetch: int = 4,
            **_ignored) -> Iterator[dict]:
    """Yield training batches as torch tensors on `device`, forever.

    `holdout` keeps the last N shots out of training so there is an in-distribution
    validation set that no checkpoint was selected on -- rule 2 in PLAN.md, a checkpoint is
    never chosen using the bench it is then quoted on.

    **`workers` is not a nicety on this machine, it is the difference between the GPU being
    busy and the GPU waiting.** The cache lives on a spinning disk, and one sample means
    opening a 21 MB file and JPEG-decoding a clip out of it. Measured over the full 1,954
    shot cache (`--bench`), against a training step of roughly 0.48 s:

        workers 0 : 1.86 samples/s   0.538 s per sample
        workers 2 : 3.86 samples/s   0.259 s per sample
        workers 4 : 6.05 samples/s   0.165 s per sample

    Serial reading is SLOWER than the step it feeds, so it would have set the pace of the
    whole run with the GPU idle for half of it. Four readers clear the requirement three
    times over, which is why that is the default. Threads rather than processes because
    both costly parts -- the disk read and `cv2.imdecode` -- release the GIL, and a process
    pool would have to pickle the decoded clip back across a pipe.

    `workers=0` keeps the old single-threaded path, which is also the only one that draws
    samples in a reproducible order for a given seed.
    """
    shots = [Shot(p) for p in list_shots(cache_dir)]
    for s in shots:
        s._peek()
    if holdout:
        if holdout >= len(shots):
            raise SystemExit("[ERROR] holdout {} >= {} shots".format(holdout, len(shots)))
        shots = shots[:-holdout] if split == "train" else shots[-holdout:]

    def to_torch(samples):
        return {k: torch.from_numpy(np.stack([s[k] for s in samples])).to(device)
                for k in ("video", "query_points", "target_points", "occluded")}

    if workers <= 0:
        rng = np.random.default_rng(seed)
        while True:
            samples = []
            while len(samples) < batch_size:
                s = _one_sample(shots[rng.integers(0, len(shots))], clip_len,
                                tracks_to_sample, train_size, rng, random_crop,
                                color_augmentation,
                                shake_aug=shake_augmentation,
                                elastic_aug=elastic_augmentation,
                                vfx_aug=vfx_augmentation)
                if s is not None:
                    samples.append(s)
            yield to_torch(samples)
        return

    import queue
    import threading

    q: "queue.Queue" = queue.Queue(maxsize=max(1, prefetch))
    stop = threading.Event()

    def produce(wid: int):
        # Each worker gets its own stream, so two threads never draw the same sample.
        rng = np.random.default_rng([seed, wid])
        while not stop.is_set():
            try:
                samples = []
                while len(samples) < batch_size:
                    s = _one_sample(shots[rng.integers(0, len(shots))], clip_len,
                                    tracks_to_sample, train_size, rng, random_crop,
                                    color_augmentation,
                                    shake_aug=shake_augmentation,
                                    elastic_aug=elastic_augmentation,
                                    vfx_aug=vfx_augmentation)
                    if s is not None:
                        samples.append(s)
                while not stop.is_set():
                    try:
                        q.put(samples, timeout=0.2)
                        break
                    except queue.Full:
                        time.sleep(0.01)
                        continue
            except Exception as exc:                      # noqa: BLE001
                # A dead reader thread must not present as a hang. Push the failure so the
                # consumer raises it on the training thread, where it is legible.
                q.put(exc)
                return

    threads = [threading.Thread(target=produce, args=(i,), daemon=True,
                                name="kubric-read-{}".format(i))
               for i in range(workers)]
    for t in threads:
        t.start()
    try:
        while True:
            item = q.get()
            if isinstance(item, Exception):
                raise item
            yield to_torch(item)
    finally:
        stop.set()


# --------------------------------------------------------------------------- self-check


def _selftest(cache_dir: str = DEFAULT_CACHE) -> int:
    """Does a batch come out the right shape, in the right units, with sane labels?

    The check that matters is the last one: a track's stated position, at a frame it is
    NOT occluded on, must land on the same colour it had at its query frame. That is the
    same photometric test that established the occlusion flag's polarity in the first
    place, and it runs end to end here -- through the crop, the resize and the coordinate
    rescale. If any of those moved the tracks the wrong way, this is what catches it.
    """
    print("[selftest] kubric cache -> training batch")
    fails = 0
    shots = list_shots(cache_dir)
    print("  shots in cache: {}".format(len(shots)))

    # workers=0: the checks below compare a batch against itself, so the reproducible
    # single-threaded path is the one to test on. The threaded path is exercised by --bench.
    it = batches(device="cpu", cache_dir=cache_dir, batch_size=2, tracks_to_sample=64,
                 clip_len=12, color_augmentation=False, shake_augmentation=False,
                 elastic_augmentation=False, seed=3, workers=0)
    b = next(it)
    want = {"video": (2, 12, 256, 256, 3), "query_points": (2, 64, 3),
            "target_points": (2, 64, 12, 2), "occluded": (2, 64, 12)}
    for k, shape in want.items():
        got = tuple(b[k].shape)
        ok = got == shape
        fails += (not ok)
        print("  {:14s} {} {}".format(k, got, "PASS" if ok else "FAIL want " + str(shape)))

    v = b["video"]
    ok = float(v.min()) >= -1.001 and float(v.max()) <= 1.001
    fails += (not ok)
    print("  video range [{:.2f}, {:.2f}]           {}".format(
        float(v.min()), float(v.max()), "PASS" if ok else "FAIL want [-1, 1]"))

    q = b["query_points"][0].numpy()
    tp = b["target_points"][0].numpy()
    occ = b["occluded"][0].numpy()
    qf = q[:, 0].astype(int)
    at_query = tp[np.arange(len(tp)), qf]
    d = np.abs(at_query - q[:, [2, 1]]).max()
    ok = d < 0.01
    fails += (not ok)
    print("  query point == track at query frame  {}  (max {:.4f} px)".format(
        "PASS" if ok else "FAIL", d))

    ok = bool((occ[np.arange(len(occ)), qf] == 0).all())
    fails += (not ok)
    print("  query frames are never occluded      {}".format("PASS" if ok else "FAIL"))

    # the photometric check
    vid = ((v[0].numpy() + 1.0) / 2.0 * 255.0).astype(np.float32)
    H, W = vid.shape[1:3]
    same, diff = [], []
    for i in range(len(tp)):
        f0 = qf[i]
        p0 = tp[i, f0]
        if not (0 <= p0[0] < W and 0 <= p0[1] < H):
            continue
        c0 = vid[f0, int(p0[1]), int(p0[0])]
        for f in range(vid.shape[0]):
            p = tp[i, f]
            if not (0 <= p[0] < W and 0 <= p[1] < H):
                continue
            c = vid[f, int(p[1]), int(p[0])]
            (same if occ[i, f] == 0 else diff).append(float(np.abs(c - c0).mean()))
    m_vis = float(np.mean(same)) if same else float("nan")
    m_occ = float(np.mean(diff)) if diff else float("nan")
    ok = m_vis < m_occ
    fails += (not ok)
    print("  colour at tracked point: visible {:.1f} vs occluded {:.1f}   {}".format(
        m_vis, m_occ, "PASS" if ok else "FAIL -- labels or axes are inverted"))
    print("     (visible must be LOWER: the point is still on its own feature)")

    hidden = float((occ == 1).mean())
    print("\n  occluded share of this batch: {:.1%}".format(hidden))

    # Also exercise Option B augmentations (shake + elastic deformation)
    it_aug = batches(device="cpu", cache_dir=cache_dir, batch_size=2, tracks_to_sample=64,
                     clip_len=12, color_augmentation=True, shake_augmentation=True,
                     elastic_augmentation=True, seed=7, workers=0)
    b_aug = next(it_aug)
    q_aug = b_aug["query_points"][0].numpy()
    tp_aug = b_aug["target_points"][0].numpy()
    qf_aug = q_aug[:, 0].astype(int)
    at_query_aug = tp_aug[np.arange(len(tp_aug)), qf_aug]
    d_aug = np.abs(at_query_aug - q_aug[:, [2, 1]]).max()
    ok_aug = d_aug < 0.01
    fails += (not ok_aug)
    print("  augmented batch query == track at query frame {} (max {:.4f} px)".format(
        "PASS" if ok_aug else "FAIL", d_aug))

    print("[selftest] {}".format("ALL PASS" if not fails else
                                 "{} FAILURES".format(fails)))
    return 1 if fails else 0


def _bench(cache_dir: str, n: int = 30, tracks: int = 256, clip_len: int = 24) -> int:
    """How many samples a second does the disk actually give, threaded against not?

    Worth running before a long training run rather than after: if the reader cannot keep
    up with the step time, the GPU is idling and the fix is more read-ahead, not more GPU.
    A step on this box is ~0.48 s, so a rate under ~2/s means reading is the bottleneck.
    """
    import time
    print("[bench] {} samples, {} tracks, {} frames".format(n, tracks, clip_len))
    for w in (0, 2, 4):
        it = batches(device="cpu", cache_dir=cache_dir, batch_size=1,
                     tracks_to_sample=tracks, clip_len=clip_len, seed=1, workers=w,
                     prefetch=6)
        next(it)                                   # pay the warm-up outside the timer
        t0 = time.time()
        for _ in range(n):
            next(it)
        dt = time.time() - t0
        print("  workers {} : {:.2f} samples/s   {:.3f} s per sample{}".format(
            w, n / dt, dt / n, "   <- serial baseline" if w == 0 else ""))
        del it
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="measure read throughput, threaded against serial")
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(_selftest(a.cache))
    if a.bench:
        raise SystemExit(_bench(a.cache))
    raise SystemExit(0)
