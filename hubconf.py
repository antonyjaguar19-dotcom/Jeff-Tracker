"""torch.hub entry points.

    import torch
    tracker = torch.hub.load("antonyjaguar19-dotcom/Jeff-Tracker", "jefftracker").cuda()
    tracks, vis, conf = tracker.track_queries_conf(frames_bgr, queries)

`frames_bgr` is (T, H, W, 3) uint8 BGR; `queries` is (N, 3) as [frame, x, y] in those
frames' pixel space. Returns positions in that same space, y DOWN.

Default model weights are `jefftrack_reacq_stream.ckpt`, the winning streamed checkpoint
trained on Apache-2.0 Kubric with Re-acquisition Surge Loss.
Weights are fetched automatically from the repository / Git LFS or Hugging Face.
Pass `pretrained=False` to build the architecture without downloading anything.
"""
from __future__ import annotations

import os
import torch

dependencies = ["torch", "einops", "numpy", "cv2"]

HERE = os.path.dirname(os.path.abspath(__file__))
GIT_LFS_RAW_BASE = "https://github.com/antonyjaguar19-dotcom/Jeff-Tracker/raw/main/weights/"


def _resolve_checkpoint(filename: str = "jefftrack_reacq_stream.ckpt") -> str:
    """Resolve a checkpoint file.

    Search order:
      1. Repo-local weights/ folder (if valid binary > 1 MB)
      2. Torch hub checkpoints cache folder
      3. Environment variable JEFFTRACK_CKPT
      4. Direct download from GitHub Git LFS media storage
      5. Gated Hugging Face Hub fallback
    """
    # 1. Local weights/ in repo
    local_weights = os.path.join(HERE, "weights", filename)
    if os.path.isfile(local_weights) and os.path.getsize(local_weights) > 1024 * 1024:
        return local_weights

    # 2. Torch hub checkpoints dir
    hub_dir = torch.hub.get_dir()
    cached_ckpt = os.path.join(hub_dir, "checkpoints", filename)
    if os.path.isfile(cached_ckpt) and os.path.getsize(cached_ckpt) > 1024 * 1024:
        return cached_ckpt

    # 3. Explicit environment variable
    env_ckpt = os.environ.get("JEFFTRACK_CKPT")
    if env_ckpt and os.path.isfile(env_ckpt):
        return env_ckpt

    # 4. Download from Git repository (Git LFS)
    os.makedirs(os.path.join(hub_dir, "checkpoints"), exist_ok=True)
    git_url = GIT_LFS_RAW_BASE + filename
    try:
        torch.hub.download_url_to_file(git_url, cached_ckpt, progress=True)
        if os.path.isfile(cached_ckpt) and os.path.getsize(cached_ckpt) > 1024 * 1024:
            return cached_ckpt
    except Exception:
        pass

    # 5. Hugging Face fallback
    return _ckpt(filename)


def _ckpt(filename, repo_id=None, repo_type="model"):
    """Resolve a checkpoint from the Hugging Face Hub (gated fallback).

    The weights repo is gated, so this needs a logged-in account that has been granted
    access (`hf auth login`). An unauthorised request comes back as 'not found', which is
    a misleading thing to hand a user, so it is re-raised as what it actually is.
    """
    from huggingface_hub import hf_hub_download          # noqa: PLC0415
    repo_id = repo_id or os.environ.get("JEFFTRACK_HF_REPO", "JeffyAntony/Jeff-Tracker")
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)
    except Exception as exc:                             # noqa: BLE001
        raise RuntimeError(
            "could not fetch {} from {}: {}. The weights are gated -- request access at "
            "https://huggingface.co/{} and run `hf auth login`. To use a local file "
            "instead, pass ckpt=/path/to.ckpt.".format(filename, repo_id, exc, repo_id))


def jefftracker(pretrained: bool = True, model_res=(256, 256), device: str = "cuda",
              ckpt: str = None, **kw):
    """Jeff-Tracker -- LocoTrack-B plus cross-track attention.

    The default weights are `jefftrack_reacq_stream.ckpt`: the winning streamed checkpoint
    trained on Meta's Apache-2.0 Kubric with Re-acquisition Surge Loss.
    """
    from jefftrack.engine import JeffTrackEngine              # noqa: PLC0415
    path = ckpt or (_resolve_checkpoint("jefftrack_reacq_stream.ckpt") if pretrained else None)
    return JeffTrackEngine(device=device, model_size="base", ckpt=path,
                         model_res=model_res, arch="jefftrack", **kw)


def jefftracker_reacq_stream(pretrained: bool = True, model_res=(256, 256), device: str = "cuda",
                            ckpt: str = None, **kw):
    """Explicit entry point for the winning streamed checkpoint."""
    return jefftracker(pretrained=pretrained, model_res=model_res, device=device, ckpt=ckpt, **kw)


def locotrack(pretrained: bool = True, model_res=(256, 256), device: str = "cuda",
              ckpt: str = None, **kw):
    """The unmodified LocoTrack-B baseline, for reproducing the baseline rows.

    With cross-track attention zero-initialised, `jefftracker(pretrained=False)` and this
    are the same function to 0.000e+00 -- check_identity.py asserts it.
    """
    from jefftrack.engine import JeffTrackEngine              # noqa: PLC0415
    path = ckpt
    if not path and pretrained:
        local_base = os.path.join(HERE, "weights", "locotrack_base.ckpt")
        hub_base = os.path.join(torch.hub.get_dir(), "checkpoints", "locotrack_base.ckpt")
        if os.path.isfile(local_base) and os.path.getsize(local_base) > 1024 * 1024:
            path = local_base
        elif os.path.isfile(hub_base) and os.path.getsize(hub_base) > 1024 * 1024:
            path = hub_base
        else:
            path = _ckpt("locotrack_base.ckpt",
                         repo_id="hamacojr/LocoTrack-pytorch-weights",
                         repo_type="dataset")
    return JeffTrackEngine(device=device, model_size="base", ckpt=path,
                         model_res=model_res, arch="locotrack", **kw)
