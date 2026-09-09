"""Download checkpoints from the Hugging Face Hub into weights/.

    python tools/fetch_weights.py                 # the shipping Jeff-Tracker checkpoint
    python tools/fetch_weights.py --all           # plus the LocoTrack baseline
    python tools/fetch_weights.py --baseline-only # only LocoTrack (ungated, no login)

Weights are hosted on the Hub rather than committed here, which is what the models this one
descends from do: LocoTrack publishes to `hamacojr/LocoTrack-pytorch-weights`, and a 66 MB
blob per checkpoint would sit in git history forever for no benefit.

The Jeff-Tracker repo is **gated** — access is requested on the Hub and granted per person,
so you need `hf auth login` with an approved account. The LocoTrack baseline is not gated
and `--baseline-only` needs no login at all.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
WEIGHTS = os.path.join(ROOT, "weights")

# The Jeff-Tracker checkpoints. Set JEFFTRACK_HF_REPO to use a fork or a private mirror.
HF_REPO = os.environ.get("JEFFTRACK_HF_REPO", "antonyjaguar19-dotcom/Jeff-Tracker")

# LocoTrack's own release, Apache-2.0, published by the LocoTrack authors as a dataset repo
# rather than a model repo.
LOCO_REPO = "hamacojr/LocoTrack-pytorch-weights"
LOCO_TYPE = "dataset"

JEFFTRACK_FILES = ["inf_s4000.ckpt"]
LOCO_FILES = ["locotrack_base.ckpt", "locotrack_small.ckpt"]


def grab(repo_id: str, filename: str, repo_type: str = "model") -> str:
    """Download one file, and turn every failure into an instruction.

    A gated repo answers an unauthenticated request the same way it answers a request for
    something that does not exist. A bare 404 would send people hunting for a typo when the
    actual answer is "ask for access", so the cases are separated and each is told what to
    do next.
    """
    from huggingface_hub import hf_hub_download                # noqa: PLC0415
    from huggingface_hub.utils import (                        # noqa: PLC0415
        EntryNotFoundError, RepositoryNotFoundError,
    )
    try:
        from huggingface_hub.utils import GatedRepoError       # noqa: PLC0415
    except ImportError:                                        # older hub versions
        GatedRepoError = ()                                    # noqa: N806

    url = "https://huggingface.co/" + repo_id
    steps = [
        "[ERROR] {} is gated and this account has not been granted access.".format(repo_id),
        "        1. open " + url,
        "        2. click 'Request access' and complete the form",
        "        3. once the maintainer approves:  hf auth login",
        "        then re-run this script.",
    ]
    missing = [
        "[ERROR] {} could not be read.".format(repo_id),
        "        If it exists it is gated or private and you are not logged in -- the Hub",
        "        returns 'not found' for both. Try:  hf auth login",
        "        and request access at " + url,
        "        Publishing your own checkpoints? Point this elsewhere with JEFFTRACK_HF_REPO.",
    ]
    try:
        return hf_hub_download(repo_id=repo_id, filename=filename, repo_type=repo_type,
                               local_dir=WEIGHTS)
    except GatedRepoError:
        raise SystemExit("\n".join(steps))
    except RepositoryNotFoundError:
        raise SystemExit("\n".join(missing))
    except EntryNotFoundError:
        raise SystemExit(
            "[ERROR] {} has no file named {!r}.\n"
            "        Upload it with:  hf upload {} weights/{} {}"
            .format(repo_id, filename, repo_id, filename, filename))


def main() -> int:
    ap = argparse.ArgumentParser(description="fetch Jeff-Tracker checkpoints")
    ap.add_argument("--all", action="store_true",
                    help="also fetch the LocoTrack baseline weights")
    ap.add_argument("--baseline-only", action="store_true",
                    help="fetch ONLY the ungated LocoTrack baseline, which needs no login "
                         "and no access request")
    a = ap.parse_args()

    os.makedirs(WEIGHTS, exist_ok=True)
    got = []
    if not a.baseline_only:
        for f in JEFFTRACK_FILES:
            got.append(grab(HF_REPO, f))
    if a.all or a.baseline_only:
        for f in LOCO_FILES:
            got.append(grab(LOCO_REPO, f, LOCO_TYPE))

    for p in got:
        print("[weights] {}  ({:.1f} MB)".format(p, os.path.getsize(p) / 1048576.0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
