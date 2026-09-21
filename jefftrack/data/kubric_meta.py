"""Training batches from Meta's CoTracker3_Kubric, via the local cache.

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
        if self._n_frames >= 0:
            return
        with np.load(self.path) as z:
            self._n_frames = int(len(z["frames_off"]) - 1)
            self._n_points = int(z["coords"].shape[0])
            self._hw = (int(z["height"]), int(z["width"]))

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
                stride_max: int = 2):
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
    q, dead = _pick_queries(coords, occ, train_size, rng)

    keep = np.flatnonzero(~dead)
    if len(keep) < tracks:
        if len(keep) == 0:
            return None
        # Pad by repeating -- rare, and better than a short batch that cannot stack.
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
            random_crop: bool = True, seed: int = 0, holdout: int = 0,
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
                                color_augmentation)
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
                                    color_augmentation)
                    if s is not None:
                        samples.append(s)
                while not stop.is_set():
                    try:
                        q.put(samples, timeout=0.5)
                        break
                    except queue.Full:
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
                 clip_len=12, color_augmentation=False, seed=3, workers=0)
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
