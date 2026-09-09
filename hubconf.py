"""torch.hub entry points.

    import torch
    tracker = torch.hub.load("antonyjaguar19-dotcom/Jeff-Tracker", "jefftracker").cuda()
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
    """Resolve a checkpoint from the Hub.

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
    """Jeff-Tracker -- LocoTrack-B plus cross-track attention, trained on MOVi-E.

    256x256 is the default deliberately. At higher model resolutions the median improves
    and a tail of confident, badly wrong tracks appears with it; METHOD.md has the
    numbers. Raise it only if you are gating on the returned confidence.
    """
    from jefftrack.engine import JeffTrackEngine              # noqa: PLC0415
    path = ckpt or (_ckpt("inf_s4000.ckpt") if pretrained else None)
    return JeffTrackEngine(device=device, model_size="base", ckpt=path,
                         model_res=model_res, arch="jefftrack", **kw)


def locotrack(pretrained: bool = True, model_res=(256, 256), device: str = "cuda",
              ckpt: str = None, **kw):
    """The unmodified LocoTrack-B baseline, for reproducing the baseline rows.

    With cross-track attention zero-initialised, `jefftracker(pretrained=False)` and this
    are the same function to 0.000e+00 -- check_identity.py asserts it.
    """
    from jefftrack.engine import JeffTrackEngine              # noqa: PLC0415
    path = ckpt or (_ckpt("locotrack_base.ckpt",
                          repo_id="hamacojr/LocoTrack-pytorch-weights",
                          repo_type="dataset") if pretrained else None)
    return JeffTrackEngine(device=device, model_size="base", ckpt=path,
                         model_res=model_res, arch="locotrack", **kw)
