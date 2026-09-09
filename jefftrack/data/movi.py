"""MOVi-E point-tracking batches from Kubric's public TFDS bucket.

Why this dataset and not the one LocoTrack actually trained on: the panning-MOVi-E upload
on Hugging Face has **no declared licence**, and the whole reason this experiment exists is
that the tracker has to be shippable. `movi_e` in `gs://kubric-public/tfds` is Google's own
publication of the Kubric data and is Apache-2.0, so training on it leaves the provenance
chain intact end to end. See ../../LICENSES.md.

Kubric cannot be *rendered* on this box -- rendering needs Docker and there is neither
Docker nor WSL here -- but the pre-rendered TFDS shards need only tensorflow to read, and
`add_tracks` (vendored into pydeps/kubric/, Apache-2.0) turns a raw record into point
tracks. That function derives occlusion from **depth and segmentation**, which is the
supervision cross-track attention actually needs: a point labelled occluded because
something is genuinely in front of it, not because a tracker lost confidence.

This mirrors the vendor's own `data/kubric_data.py` rather than improving on it -- same
`add_tracks` call, same augmentation -- with the dataset name made configurable and the
torch.distributed scatter dropped, since this trains on one GPU.

Emits, per batch:
    video          (B, T, H, W, 3) float32 in [-1, 1]   -- already normalised by add_tracks
    query_points   (B, N, 3)       't y x' in train_size pixels
    target_points  (B, N, T, 2)    'x y'   in train_size pixels
    occluded       (B, N, T)       bool, True = hidden
"""
from __future__ import annotations

from typing import Iterator, Mapping, Optional, Tuple

import numpy as np
import torch

from jefftrack.paths import add_vendor_to_path

add_vendor_to_path()

PUBLIC_BUCKET = "gs://kubric-public/tfds"


def _import_tf():
    """Import tensorflow with a legible failure, and keep it off the GPU.

    TF is installed into pydeps/ only -- the inference runtime is not
    touched. It is here to read TFRecord shards and nothing else, so it must never take
    VRAM the model needs; the vendor does the same at data/kubric_data.py:14.
    """
    try:
        import tensorflow as tf  # type: ignore
        import tensorflow_datasets as tfds  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "[ERROR] tensorflow / tensorflow_datasets not importable: {!r}\n"
            "        install them into this project's own pydeps, never a shared "
            "runtime:\n"
            "        python -m pip install --target "
            "pydeps tensorflow-cpu tensorflow_datasets"
            .format(exc))
    tf.config.set_visible_devices([], "GPU")
    return tf, tfds


def _color_augment(tf, inputs: Mapping):
    """The vendor's standard colour augmentation (data/kubric_data.py:20), unchanged."""
    frames = inputs["video"]
    if frames.dtype != tf.float32:
        raise ValueError("`frames` should be in float32.")

    def augment(video):
        video = (video + 1.0) / 2.0                     # [-1,1] -> [0,1]
        video = tf.image.random_brightness(video, max_delta=32.0 / 255.0)
        video = tf.image.random_saturation(video, lower=0.6, upper=1.4)
        video = tf.image.random_contrast(video, lower=0.6, upper=1.4)
        video = tf.image.random_hue(video, max_delta=0.2)
        video = tf.clip_by_value(video, 0.0, 1.0)
        return video * 2.0 - 1.0

    def drop(video):
        video = tf.image.rgb_to_grayscale((video + 1.0) / 2.0)
        video = tf.tile(video, [1, 1, 1, 1, 3])
        return video * 2.0 - 1.0

    coin = tf.random.uniform([], 0, 1, dtype=tf.float32)
    frames = tf.cond(coin < 0.8, lambda: augment(frames), lambda: frames)
    coin = tf.random.uniform([], 0, 1, dtype=tf.float32)
    frames = tf.cond(coin < 0.2, lambda: drop(frames), lambda: frames)
    out = dict(inputs)
    out["video"] = frames
    return out


def make_dataset(
    data_dir: str = PUBLIC_BUCKET,
    name: str = "movi_e/256x256",
    split: str = "train",
    train_size: Tuple[int, int] = (256, 256),
    batch_size: int = 1,
    tracks_to_sample: int = 256,
    shuffle_buffer_size: Optional[int] = 64,
    color_augmentation: bool = True,
    random_crop: bool = True,
    sampling_stride: int = 4,
    max_seg_id: int = 25,
    max_sampled_frac: float = 0.1,
    repeat: bool = True,
    num_parallel_calls: int = 4,
):
    """Build the numpy-yielding TFDS pipeline. Mirrors the vendor's own construction."""
    tf, tfds = _import_tf()
    import functools

    from kubric.challenges.point_tracking.dataset import add_tracks  # type: ignore

    ds = tfds.load(name, data_dir=data_dir,
                   shuffle_files=shuffle_buffer_size is not None)
    ds = ds[split]
    if repeat:
        ds = ds.repeat()
    ds = ds.map(
        functools.partial(
            add_tracks,
            train_size=train_size,
            vflip=False,
            random_crop=random_crop,
            tracks_to_sample=tracks_to_sample,
            sampling_stride=sampling_stride,
            max_seg_id=max_seg_id,
            max_sampled_frac=max_sampled_frac),
        num_parallel_calls=num_parallel_calls)
    if shuffle_buffer_size is not None:
        ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.batch(batch_size)
    if color_augmentation:
        ds = ds.map(functools.partial(_color_augment, tf))
    ds = ds.prefetch(2)
    return tfds.as_numpy(ds)


def batches(device: str = "cuda", **kw) -> Iterator[dict]:
    """Yield training batches as torch tensors on `device`."""
    ds = make_dataset(**kw)
    for rec in ds:
        yield {
            "video": torch.from_numpy(np.asarray(rec["video"], np.float32)).to(device),
            "query_points": torch.from_numpy(
                np.asarray(rec["query_points"], np.float32)).to(device),
            "target_points": torch.from_numpy(
                np.asarray(rec["target_points"], np.float32)).to(device),
            "occluded": torch.from_numpy(
                np.asarray(rec["occluded"])).to(device).float(),
        }
