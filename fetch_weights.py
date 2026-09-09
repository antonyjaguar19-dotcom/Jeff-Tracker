"""Download checkpoints from the Hugging Face Hub into weights/.

    python fetch_weights.py                 # the shipping DBtracker checkpoint
    python fetch_weights.py --all           # plus the LocoTrack baseline

Weights are hosted on the Hub rather than committed here, which is what the models this
one descends from do: LocoTrack publishes to `hamacojr/LocoTrack-pytorch-weights`, and a
66 MB blob per checkpoint would sit in git history forever for no benefit.

The LocoTrack baseline is what `--arch locotrack` loads, and reproducing the baseline rows
in METHOD.md needs it. Everything else is a DBtracker checkpoint.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.path.join(HERE, "weights")

# The DBtracker checkpoints. Set BTR_HF_REPO to point at a fork or a private mirror.
HF_REPO = os.environ.get("BTR_HF_REPO", "antonyjaguar19-dotcom/DBtracker")

# LocoTrack's own release, Apache-2.0, published by the LocoTrack authors as a dataset
# repo rather than a model repo.
LOCO_REPO = "hamacojr/LocoTrack-pytorch-weights"
LOCO_TYPE = "dataset"

DBTRACK_FILES = ["inf_s4000.ckpt"]
LOCO_FILES = ["locotrack_base.ckpt", "locotrack_small.ckpt"]


def grab(repo_id: str, filename: str, repo_type: str = "model") -> str:
    from huggingface_hub import hf_hub_download          # noqa: PLC0415
    from huggingface_hub.utils import (                  # noqa: PLC0415
        EntryNotFoundError, RepositoryNotFoundError,
    )
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type,
                               local_dir=WEIGHTS)
    except (RepositoryNotFoundError, EntryNotFoundError) as exc:
        raise SystemExit(
            "[ERROR] {}/{} is not on the Hub ({}).\n"
            "        If you are publishing your own checkpoints, upload them with\n"
            "          hf auth login\n"
            "          hf upload {} weights/{} {}\n"
            "        or point this script elsewhere with BTR_HF_REPO."
            .format(repo_id, filename, type(exc).__name__, repo_id, filename, filename))


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch DBtracker checkpoints")
    ap.add_argument("--all", action="store_true",
                    help="also fetch the LocoTrack baseline weights")
    ap.add_argument("--baseline-only", action="store_true",
                    help="fetch ONLY the LocoTrack baseline (works without the "
                         "DBtracker repo existing)")
    a = ap.parse_args()

    os.makedirs(WEIGHTS, exist_ok=True)
    got = []
    if not a.baseline_only:
        for f in DBTRACK_FILES:
            got.append(grab(HF_REPO, f))
    if a.all or a.baseline_only:
        for f in LOCO_FILES:
            got.append(grab(LOCO_REPO, f, LOCO_TYPE))

    for p in got:
        print("[weights] {}  ({:.1f} MB)".format(p, os.path.getsize(p) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
