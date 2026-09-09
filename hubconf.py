"""torch.hub entry points.

    import torch
    tracker = torch.hub.load("antonyjaguar19-dotcom/DBtracker", "dbtracker").cuda()
    tracks, vis, conf = tracker.track_queries_conf(frames_bgr, queries)

`frames_bgr` is (T, H, W, 3) uint8 BGR; `queries` is (N, 3) as [frame, x, y] in those
frames' pixel space. Returns positions in that same space, y DOWN.

Weights come from the Hugging Face Hub, not from this repository -- see fetch_weights.py
for why. Pass `pretrained=False` to build the architecture without downloading anything.
"""
from __future__ import annotations

import os

dependencies = ["torch", "einops", "numpy", "cv2"]


def _ckpt(filename, repo_id=None, repo_type="model"):
    from huggingface_hub import hf_hub_download          # noqa: PLC0415
    repo_id = repo_id or os.environ.get("BTR_HF_REPO", "antonyjaguar19-dotcom/DBtracker")
    return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type)


def dbtracker(pretrained: bool = True, model_res=(256, 256), device: str = "cuda",
              ckpt: str = None, **kw):
    """DBtracker -- LocoTrack-B plus cross-track attention, trained on MOVi-E.

    256x256 is the default deliberately. At higher model resolutions the median improves
    and a tail of confident, badly wrong tracks appears with it; METHOD.md has the
    numbers. Raise it only if you are gating on the returned confidence.
    """
    from dbtrack_engine import DBTrackEngine              # noqa: PLC0415
    path = ckpt or (_ckpt("inf_s4000.ckpt") if pretrained else None)
    return DBTrackEngine(device=device, model_size="base", ckpt=path,
                         model_res=model_res, arch="dbtrack", **kw)


def locotrack(pretrained: bool = True, model_res=(256, 256), device: str = "cuda",
              ckpt: str = None, **kw):
    """The unmodified LocoTrack-B baseline, for reproducing the baseline rows.

    With cross-track attention zero-initialised, `dbtracker(pretrained=False)` and this
    are the same function to 0.000e+00 -- check_identity.py asserts it.
    """
    from dbtrack_engine import DBTrackEngine              # noqa: PLC0415
    path = ckpt or (_ckpt("locotrack_base.ckpt",
                          repo_id="hamacojr/LocoTrack-pytorch-weights",
                          repo_type="dataset") if pretrained else None)
    return DBTrackEngine(device=device, model_size="base", ckpt=path,
                         model_res=model_res, arch="locotrack", **kw)
