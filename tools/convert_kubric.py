"""Meta's CoTracker3_Kubric shards -> a compact training cache, 585 GB -> ~45 GB.

Source: `facebook/CoTracker3_Kubric` on Hugging Face, **Apache-2.0** (verified on the
dataset page and the HF API `cardData.license`, 2026-09-21). Worth restating because it is
the whole reason this folder exists: CoTracker3's *code and weights* are CC-BY-NC and
cannot ship, but the *shots Meta trained it on* are Apache-2.0 and can. See ../../LICENSES.md.

One `NNNN.tar.gz` is one shot: 120 frames at 512x512, 32,768 tracked points, a depth pass
and a ground-truth camera. Unpacked that is ~460 MB per shot, and 1,954 shots do not fit on
a disk with 207 GB free. Most of the bulk is redundant, and this script keeps only what
training reads.

What is dropped, and why it is safe -- every claim here was checked on shard 0000 before a
byte was deleted:

  * `depths/*.npy` -- 120 x 2.1 MB of float64, **252 MB per shot, 55% of the archive**. The
    occlusion labels are already in the visibility array, so training never opens these.
  * `NNNN.npy` -- a 145 MB bundle holding the depth again as uint16, the segmentations, and
    copies of the two arrays below. Verified byte-identical: `bundle["coords"]` equals
    `NNNN_trajs_2d.npy` and `bundle["visibility"]` equals `NNNN_visibility.npy`
    (`np.array_equal` -> True on both). So this script reads the small sidecars and never
    opens the 145 MB file at all, which is also what makes it fast.
  * points that are never trackable -- a track with fewer than `--min-visible` frames
    visible *and* on screen can never be a query, so storing 120 frames of it is dead weight.

Three properties of this data are counter-intuitive and each one would silently produce a
broken model. All three were measured against the pixels, not inferred:

  1. **The array named `visibility` actually stores OCCLUSION.** `True` means HIDDEN. Read
     the obvious way it inverts the labels and trains the model inside out. The proof is
     photometric: sampling the frames at each track's position over 12 frames, tracks with
     the flag False hold a near-constant colour (spread 2.80) and tracks with it True jump
     around (23.69). This script writes the field out under the name `occluded`, matching
     the trainer's own vocabulary, so the trap is not re-exported.
  2. **Coordinates are (x, y), column first.** Same test: 2.80 spread read as (x, y)
     against 17.61 read as (y, x). Not ambiguous.
  3. **75% of "occluded" samples are simply off screen.** Only 7.06% of all samples are the
     thing the occlusion work actually needs -- hidden behind something while still inside
     the frame. `losses.py:occluded_in_frame_only` already separates those two populations,
     and the per-shot counts written into the cache let it be checked per batch.

JPEG at quality 98 with **4:4:4 chroma** (no subsampling), which is the setting the frames
are stored under. Measured against the source PNG: mean absolute error 0.65 of one grey
level, max 8, at 58% of the PNG size. The default 4:2:0 subsampling was rejected -- it costs
2.0 levels even at quality 100, and this model is trained to sub-pixel accuracy. For scale,
the vendor's own colour augmentation moves brightness by up to 32 levels on purpose, so
0.65 is far below the noise training already adds.

Frames are kept at the full 512x512, NOT pre-resized to the 256x256 the model trains at, so
random-crop augmentation still has something to crop from. That decision is the reason the
cache is 45 GB rather than 12, and it is what stops the source archives being load-bearing.

    hf download facebook/CoTracker3_Kubric --repo-type dataset \
        --local-dir datasets/CoTracker3_Kubric
    python tools/convert_kubric.py --selftest   # no dataset needed, seconds
    python tools/convert_kubric.py --limit 4    # try four shots
    python tools/convert_kubric.py              # the real run, ~4 s per shot

Deleting the source archives is a SEPARATE run of this script (`--delete-source`), never
part of a conversion, and it re-verifies every cache file first. See `_delete_sources`.
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
DEFAULT_SRC = os.path.join(ROOT, "datasets", "CoTracker3_Kubric")
DEFAULT_OUT = os.environ.get(
    "JEFFTRACK_KUBRIC_CACHE", os.path.join(ROOT, "datasets", "kubric_cache"))

JPEG_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, 98,
               cv2.IMWRITE_JPEG_SAMPLING_FACTOR, int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444)]

CACHE_VERSION = 1


# --------------------------------------------------------------------------- reading


def _read_shard(tar_path: str) -> dict:
    """Pull the four things worth keeping out of one archive.

    Streamed (`r|gz`), so the archive is decompressed once, in order, and never written to
    disk. The 145 MB bundle and the 252 MB of float64 depth are skipped without being
    materialised -- they still pass through the gzip decoder, which is why this is IO bound
    rather than free.
    """
    frames: dict = {}
    trajs = vis = None
    cam: dict = {}

    with tarfile.open(tar_path, "r|gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            name = m.name
            base = os.path.basename(name)
            if "/frames/" in name and base.endswith(".png"):
                f = tf.extractfile(m)
                if f is None:
                    continue
                buf = np.frombuffer(f.read(), np.uint8)
                frames[int(base[:-4])] = buf
            elif base.endswith("_trajs_2d.npy"):
                trajs = np.load(io.BytesIO(tf.extractfile(m).read()))
            elif base.endswith("_visibility.npy"):
                vis = np.load(io.BytesIO(tf.extractfile(m).read()))
            elif base.endswith("_with_rank.npz"):
                z = np.load(io.BytesIO(tf.extractfile(m).read()))
                cam = {k: z[k] for k in z.files}
            # every other member -- the bundle, depths/ -- is deliberately ignored

    if trajs is None or vis is None:
        raise ValueError("shard is missing its track arrays")
    if not frames:
        raise ValueError("shard has no frames")
    return {"frames": frames, "coords": trajs, "occluded": vis, "cam": cam}


def _pick_points(coords: np.ndarray, occluded: np.ndarray, n_keep: int,
                 min_visible: int, width: int, height: int,
                 seed: int) -> np.ndarray:
    """Which of the 32,768 tracks to keep.

    A track is a candidate when it is visible AND on screen for at least `min_visible`
    frames -- fewer than that and it can never supply a query point, so it would occupy
    120 frames of storage and never be trained on.

    The subset is then drawn UNIFORMLY at random from the candidates. Deliberately not
    biased toward tracks that carry an occlusion: `losses.py:position_loss_normalised`
    already fixes the occluded population's share of the gradient explicitly (`occ_share`),
    and skewing the stored data as well would apply that correction twice, with the second
    one invisible and unrecorded. The seed makes the choice reproducible.
    """
    inside = ((coords[..., 0] >= 0) & (coords[..., 0] < width) &
              (coords[..., 1] >= 0) & (coords[..., 1] < height))
    usable = (~occluded) & inside                      # visible and on screen
    cand = np.where(usable.sum(axis=1) >= min_visible)[0]
    if len(cand) <= n_keep:
        return cand
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(cand, size=n_keep, replace=False))


def _encode_frames(frames: dict) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """PNG bytes in, one concatenated JPEG blob plus offsets out.

    Stored as a flat uint8 array with an offsets table rather than 120 separate members, so
    the loader can pull a single clip's frames out by slicing without unpacking all 120 --
    and so the cache file needs no pickle to read.
    """
    idx = sorted(frames)
    blobs: List[bytes] = []
    h = w = 0
    for i in idx:
        img = cv2.imdecode(frames[i], cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("frame {} failed to decode".format(i))
        h, w = img.shape[:2]
        ok, enc = cv2.imencode(".jpg", img, JPEG_PARAMS)
        if not ok:
            raise ValueError("frame {} failed to encode".format(i))
        blobs.append(enc.tobytes())
    offs = np.zeros(len(blobs) + 1, np.int64)
    offs[1:] = np.cumsum([len(b) for b in blobs])
    return np.frombuffer(b"".join(blobs), np.uint8), offs, h, w


def convert_one(tar_path: str, out_dir: str, n_keep: int, min_visible: int,
                seed: int, overwrite: bool) -> dict:
    """One archive -> one `.npz`. Returns a row for the summary table."""
    stem = os.path.basename(tar_path).split(".")[0]
    out_path = os.path.join(out_dir, stem + ".npz")
    t0 = time.time()

    if os.path.exists(out_path) and not overwrite:
        return {"shot": stem, "status": "skipped", "out": out_path,
                "bytes": os.path.getsize(out_path), "secs": 0.0}

    src = _read_shard(tar_path)
    coords, occluded = src["coords"], src["occluded"]
    blob, offs, h, w = _encode_frames(src["frames"])

    n_frames = len(offs) - 1
    if coords.shape[1] != n_frames:
        raise ValueError("{} tracks span {} frames but {} frames were read"
                         .format(stem, coords.shape[1], n_frames))

    keep = _pick_points(coords, occluded, n_keep, min_visible, w, h, seed)
    if len(keep) == 0:
        raise ValueError("no trackable points survived the visibility filter")

    c = np.ascontiguousarray(coords[keep], dtype=np.float32)
    o = np.ascontiguousarray(occluded[keep])

    inside = ((c[..., 0] >= 0) & (c[..., 0] < w) & (c[..., 1] >= 0) & (c[..., 1] < h))
    hidden_onscreen = float((o & inside).mean())

    payload = {
        "version": np.int32(CACHE_VERSION),
        "frames_jpg": blob,
        "frames_off": offs,
        "height": np.int32(h),
        "width": np.int32(w),
        # (N, T, 2) float32, (x, y) in 512-space pixels. Off-screen values are kept as they
        # are: a point that leaves the frame is a real event and the loss decides what to
        # do with it, so clipping here would hide it.
        "coords": c,
        # (N, T) bool, True = HIDDEN. Renamed from the source's misleading "visibility".
        "occluded": o,
        "src_points": np.int32(coords.shape[0]),
        "kept_points": np.int32(len(keep)),
        "hidden_onscreen_frac": np.float32(hidden_onscreen),
    }
    for k, v in src["cam"].items():
        payload["cam_" + k] = v

    # Write to a temp name and rename, so an interrupted run never leaves a short file
    # that verify_one would have to catch later. np.savez APPENDS ".npz" to a path that
    # lacks it, so the temp name is handed over as an open file object instead -- passing
    # "0001.npz.part" produces "0001.npz.part.npz" and the rename then fails.
    tmp = out_path + ".part"
    with open(tmp, "wb") as fh:
        np.savez(fh, **payload)
    os.replace(tmp, out_path)

    return {"shot": stem, "status": "ok", "out": out_path,
            "bytes": os.path.getsize(out_path), "secs": time.time() - t0,
            "kept": int(len(keep)), "src": int(coords.shape[0]),
            "hidden": hidden_onscreen, "frames": n_frames}


# --------------------------------------------------------------------------- verifying


def verify_one(npz_path: str, deep: bool = True) -> Tuple[bool, str]:
    """Is this cache file complete and readable? Run before the source is ever deleted.

    `deep` also decodes the first, middle and last frame. A truncated JPEG blob is the one
    corruption that survives a shape check, and it is exactly what a half-written file looks
    like.
    """
    try:
        with np.load(npz_path) as z:
            need = ("version", "frames_jpg", "frames_off", "coords", "occluded",
                    "height", "width")
            missing = [k for k in need if k not in z.files]
            if missing:
                return False, "missing arrays: {}".format(missing)
            if int(z["version"]) != CACHE_VERSION:
                return False, "cache version {} != {}".format(int(z["version"]),
                                                              CACHE_VERSION)
            offs, blob = z["frames_off"], z["frames_jpg"]
            c, o = z["coords"], z["occluded"]
            n_frames = len(offs) - 1
            if n_frames < 2:
                return False, "only {} frames".format(n_frames)
            if int(offs[-1]) != blob.size:
                return False, "frame blob is {} bytes, offsets end at {}".format(
                    blob.size, int(offs[-1]))
            if c.shape[:2] != o.shape or c.shape[1] != n_frames:
                return False, "coords {} / occluded {} disagree with {} frames".format(
                    c.shape, o.shape, n_frames)
            if not np.isfinite(c).all():
                return False, "coords contain nan or inf"
            if deep:
                h, w = int(z["height"]), int(z["width"])
                for f in (0, n_frames // 2, n_frames - 1):
                    img = cv2.imdecode(blob[offs[f]:offs[f + 1]], cv2.IMREAD_COLOR)
                    if img is None:
                        return False, "frame {} does not decode".format(f)
                    if img.shape[:2] != (h, w):
                        return False, "frame {} is {} not {}".format(
                            f, img.shape[:2], (h, w))
        return True, "ok"
    except Exception as exc:
        return False, "{}: {}".format(type(exc).__name__, exc)


def verify_against_source(tar_path: str, npz_path: str) -> Tuple[bool, str]:
    """The strict check: do the cached tracks still equal the ones in the archive?

    Re-reads the source and compares the kept tracks exactly. Slow -- it decompresses the
    whole archive -- so it runs on a sample rather than on all 1,954, but it is the only
    check that would catch a wrong-index bug in `_pick_points`, which would otherwise
    produce a cache that passes every structural test and trains on mismatched labels.
    """
    try:
        src = _read_shard(tar_path)
        with np.load(npz_path) as z:
            c, o = z["coords"], z["occluded"]
            h, w = int(z["height"]), int(z["width"])
        sc, so = src["coords"], src["occluded"]
        # find where each cached track came from, by exact match on the full trajectory
        hits = 0
        rng = np.random.default_rng(0)
        for i in rng.choice(len(c), size=min(16, len(c)), replace=False):
            eq = np.all(np.isclose(sc, c[i][None], rtol=0, atol=0), axis=(1, 2))
            j = np.where(eq)[0]
            if len(j) == 0:
                return False, "cached track {} is not in the source".format(i)
            if not np.array_equal(so[j[0]], o[i]):
                return False, "track {} has different occlusion flags".format(i)
            hits += 1
        n_src_frames = len(src["frames"])
        if c.shape[1] != n_src_frames:
            return False, "cached {} frames, source has {}".format(c.shape[1],
                                                                   n_src_frames)
        return True, "{} sampled tracks match the source exactly".format(hits)
    except Exception as exc:
        return False, "{}: {}".format(type(exc).__name__, exc)


# --------------------------------------------------------------------------- deleting


def _delete_sources(src_dir: str, out_dir: str, yes: bool, deep: bool = True) -> int:
    """Delete an archive only when its cache file passes verification. Never automatic.

    This is separated from conversion on purpose. A converter that deletes as it goes will
    happily destroy 585 GB on the strength of a bug it is itself carrying, and the source is
    a ~1.9 TB re-download over the internet. So: convert, look at the summary, then run this.
    """
    tars = sorted(f for f in os.listdir(src_dir) if f.endswith(".tar.gz"))
    ready, skip = [], []
    for t in tars:
        stem = t.split(".")[0]
        npz = os.path.join(out_dir, stem + ".npz")
        if not os.path.exists(npz):
            skip.append((t, "no cache file"))
            continue
        ok, why = verify_one(npz, deep=deep)
        (ready if ok else skip).append((t, why) if not ok else t)

    freed = sum(os.path.getsize(os.path.join(src_dir, t)) for t in ready)
    print("\n  verified and safe to delete : {} archives, {:.1f} GB".format(
        len(ready), freed / 1e9))
    print("  KEPT (no valid cache)       : {} archives".format(len(skip)))
    for t, why in skip[:10]:
        print("      {}  <- {}".format(t, why))
    if len(skip) > 10:
        print("      ... and {} more".format(len(skip) - 10))

    if not ready:
        print("\n  nothing to delete.")
        return 0
    if not yes:
        print("\n  DRY RUN -- nothing was deleted.")
        print("  Re-run with --yes to delete those {} archives.".format(len(ready)))
        return 0

    n = 0
    for t in ready:
        try:
            os.remove(os.path.join(src_dir, t))
            n += 1
        except OSError as exc:
            print("  could not delete {}: {}".format(t, exc))
    print("\n  deleted {} archives, {:.1f} GB freed.".format(n, freed / 1e9))
    return n


# --------------------------------------------------------------------------- self-test


def _selftest(out_dir: str) -> int:
    """Round-trip the cache format on synthetic data. No dataset needed, runs in seconds.

    Checks the two things that would corrupt a run silently: that a JPEG blob sliced by its
    offsets gives back the frame it went in as, and that `_pick_points` returns indices into
    the ORIGINAL array -- an off-by-one there pairs every track with its neighbour's labels
    and still produces a cache that looks perfectly well formed.
    """
    print("[selftest] cache format round trip")
    rng = np.random.default_rng(0)
    T, H, W, N = 6, 64, 64, 50
    frames = {}
    for f in range(T):
        img = rng.integers(0, 255, (H, W, 3), dtype=np.uint8)
        ok, enc = cv2.imencode(".png", img)
        frames[f] = np.frombuffer(enc.tobytes(), np.uint8)
    blob, offs, h, w = _encode_frames(frames)
    fails = 0
    if (h, w) != (H, W):
        print("  FAIL size {} != {}".format((h, w), (H, W))); fails += 1
    for f in range(T):
        a = cv2.imdecode(frames[f], cv2.IMREAD_COLOR)
        b = cv2.imdecode(blob[offs[f]:offs[f + 1]], cv2.IMREAD_COLOR)
        if b is None or b.shape != a.shape:
            print("  FAIL frame {} shape".format(f)); fails += 1; continue
        err = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
        if err > 12.0:                      # pure noise is the worst case for jpeg
            print("  FAIL frame {} error {:.2f}".format(f, err)); fails += 1
    print("  frame blob round trip            {}".format("PASS" if not fails else "FAIL"))

    coords = rng.uniform(0, W, (N, T, 2)).astype(np.float32)
    occ = np.zeros((N, T), bool)
    occ[:25] = True                          # first 25 never visible -> never candidates
    keep = _pick_points(coords, occ, n_keep=10, min_visible=T, width=W, height=H, seed=0)
    bad = [int(i) for i in keep if i < 25]
    print("  point filter drops dead tracks   {}".format("PASS" if not bad else
                                                         "FAIL {}".format(bad)))
    fails += bool(bad)

    # the off-by-one that matters: indices must address the source array
    mark = coords.copy()
    mark[:, 0, 0] = np.arange(N)
    sel = mark[keep][:, 0, 0].astype(int)
    ident = np.array_equal(sel, np.asarray(keep))
    print("  kept indices address the source  {}".format("PASS" if ident else "FAIL"))
    fails += (not ident)

    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, "_selftest.npz")
    np.savez(p, version=np.int32(CACHE_VERSION), frames_jpg=blob, frames_off=offs,
             height=np.int32(h), width=np.int32(w), coords=coords[keep],
             occluded=occ[keep])
    ok, why = verify_one(p)
    print("  verify_one on a good file        {}".format("PASS" if ok else
                                                         "FAIL " + why))
    fails += (not ok)
    with open(p, "r+b") as fh:               # truncate it and make sure verify notices
        fh.truncate(os.path.getsize(p) - 2048)
    ok2, _ = verify_one(p)
    print("  verify_one catches a truncation  {}".format("PASS" if not ok2 else "FAIL"))
    fails += bool(ok2)
    os.remove(p)

    print("\n[selftest] {}".format("ALL PASS" if not fails else
                                   "{} FAILURES".format(fails)))
    return 1 if fails else 0


# --------------------------------------------------------------------------- driver


def _job(args):
    tar, out_dir, n_keep, min_vis, seed, overwrite = args
    try:
        return convert_one(tar, out_dir, n_keep, min_vis, seed, overwrite)
    except Exception as exc:
        return {"shot": os.path.basename(tar).split(".")[0], "status": "failed",
                "error": "{}: {}".format(type(exc).__name__, exc),
                "trace": traceback.format_exc(limit=3)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--points", type=int, default=4096,
                    help="tracks kept per shot out of the source's 32,768. 4096 is ~4.4 MB "
                         "per shot against 31 MB for all of them, and training samples 256 "
                         "per step, so the cap costs variety it cannot use")
    ap.add_argument("--min-visible", type=int, default=8,
                    help="a track needs this many frames visible AND on screen to be worth "
                         "storing -- below it, it can never supply a query point")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="only the first N archives")
    ap.add_argument("--offset", type=int, default=0,
                    help="skip the first N archives. Exists for honest timing runs: "
                         "re-reading archives the OS still has cached reports a throughput "
                         "the disk cannot actually sustain, and this source is on a "
                         "spinning disk where that difference is a factor of ten")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--selftest", action="store_true",
                    help="round-trip the cache format on synthetic data, no dataset needed")
    ap.add_argument("--verify", action="store_true",
                    help="check existing cache files and stop")
    ap.add_argument("--verify-deep", type=int, default=8,
                    help="additionally re-read this many archives and compare tracks "
                         "exactly against the cache (slow, catches index bugs)")
    ap.add_argument("--delete-source", action="store_true",
                    help="delete archives whose cache file verifies. DRY RUN unless --yes")
    ap.add_argument("--yes", action="store_true", help="actually delete")
    a = ap.parse_args()

    if a.selftest:
        return _selftest(a.out)

    if not os.path.isdir(a.src):
        print("[ERROR] no such source dir: {}".format(a.src))
        return 2
    os.makedirs(a.out, exist_ok=True)

    if a.delete_source:
        print("=== delete source archives ===")
        print("source : {}".format(a.src))
        print("cache  : {}".format(a.out))
        _delete_sources(a.src, a.out, yes=a.yes)
        return 0

    if a.verify:
        files = sorted(f for f in os.listdir(a.out) if f.endswith(".npz"))
        bad = []
        for i, f in enumerate(files):
            ok, why = verify_one(os.path.join(a.out, f))
            if not ok:
                bad.append((f, why))
            if (i + 1) % 200 == 0:
                print("  checked {}/{}".format(i + 1, len(files)))
        print("\n  {} cache files, {} bad".format(len(files), len(bad)))
        for f, why in bad[:20]:
            print("     {}  <- {}".format(f, why))
        if a.verify_deep and files:
            print("\n  deep check against the source archives:")
            rng = np.random.default_rng(0)
            for f in rng.choice(files, size=min(a.verify_deep, len(files)),
                                replace=False):
                tar = os.path.join(a.src, f.replace(".npz", ".tar.gz"))
                if not os.path.exists(tar):
                    print("     {}  source already deleted, skipped".format(f))
                    continue
                ok, why = verify_against_source(tar, os.path.join(a.out, f))
                print("     {}  {}  {}".format(f, "PASS" if ok else "FAIL", why))
        return 1 if bad else 0

    tars = sorted(f for f in os.listdir(a.src) if f.endswith(".tar.gz"))
    if a.offset:
        tars = tars[a.offset:]
    if a.limit:
        tars = tars[:a.limit]
    if not tars:
        print("[ERROR] no .tar.gz archives in {}".format(a.src))
        return 2

    src_bytes = sum(os.path.getsize(os.path.join(a.src, t)) for t in tars)
    print("=== CoTracker3_Kubric -> training cache ===")
    print("archives   : {}  ({:.1f} GB)".format(len(tars), src_bytes / 1e9))
    print("out        : {}".format(a.out))
    print("points/shot: {}   min visible frames: {}".format(a.points, a.min_visible))
    print("jpeg       : quality 98, 4:4:4, frames kept at source resolution")
    print("workers    : {}\n".format(a.workers))

    jobs = [(os.path.join(a.src, t), a.out, a.points, a.min_visible, a.seed,
             a.overwrite) for t in tars]
    rows, failed = [], []
    t0 = time.time()

    def _report(r, i):
        if r["status"] == "failed":
            failed.append(r)
            print("  [{}/{}] {} FAILED  {}".format(i, len(jobs), r["shot"], r["error"]))
            return
        rows.append(r)
        done = len(rows)
        if r["status"] == "skipped":
            return
        if done % 10 == 0 or done <= 3:
            mb = r["bytes"] / 1e6
            el = time.time() - t0
            rate = el / max(done, 1)
            left = rate * (len(jobs) - i) / 60.0
            print("  [{}/{}] {}  {:.1f} MB  {} pts  hidden-onscreen {:.1%}  "
                  "{:.1f}s  (~{:.0f} min left)".format(
                      i, len(jobs), r["shot"], mb, r.get("kept", 0),
                      r.get("hidden", 0.0), r["secs"], left))

    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(_job, j): n for n, j in enumerate(jobs, 1)}
            for n, fut in enumerate(as_completed(futs), 1):
                _report(fut.result(), n)
    else:
        for n, j in enumerate(jobs, 1):
            _report(_job(j), n)

    out_bytes = sum(r["bytes"] for r in rows)
    conv = [r for r in rows if r["status"] == "ok"]
    print("\n=== summary ===")
    print("converted  : {}   skipped (already done): {}   failed: {}".format(
        len(conv), len(rows) - len(conv), len(failed)))
    print("cache size : {:.1f} GB   from {:.1f} GB of archives  ({:.1f}x smaller)".format(
        out_bytes / 1e9, src_bytes / 1e9, src_bytes / max(out_bytes, 1)))
    if conv:
        hid = np.mean([r["hidden"] for r in conv])
        print("mean hidden-but-on-screen samples: {:.2%}  "
              "(the occlusion signal training needs)".format(hid))
        print("elapsed    : {:.1f} min".format((time.time() - t0) / 60))
    for r in failed[:10]:
        print("\n{} failed:\n{}".format(r["shot"], r.get("trace", r["error"])))

    if conv and not failed:
        print("\nNext: verify, then free the source archives.")
        print("  --verify           check every cache file and spot-check against source")
        print("  --delete-source    dry run, lists what would go")
        print("  --delete-source --yes")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
