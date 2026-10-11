"""Download a batch of Kubric training shots from Hugging Face into datasets/kubric_meta.

Source: Meta AI's Kubric point-tracking release on Hugging Face (Apache-2.0, ungated).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DEST = os.path.join(ROOT, "datasets", "kubric_meta")


def main():
    ap = argparse.ArgumentParser(description="Download Kubric training shards")
    ap.add_argument("--count", type=int, default=6, help="number of shots to download")
    ap.add_argument("--start", type=int, default=0, help="starting shard index")
    ap.add_argument("--out-dir", default=DEST, help="destination folder")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    repo_id = "facebook/CoTracker3_Kubric"

    print(f"[fetch] downloading {a.count} shots from {repo_id} to {a.out_dir}...")
    t0 = time.time()
    downloaded = []
    for idx in range(a.start, a.start + a.count):
        filename = f"{idx:04d}.tar.gz"
        dest_file = os.path.join(a.out_dir, filename)
        if os.path.isfile(dest_file):
            sz_mb = os.path.getsize(dest_file) / (1024 * 1024)
            print(f"  [cached] {filename} ({sz_mb:.1f} MB)")
            downloaded.append(dest_file)
            continue

        print(f"  [downloading] {filename}...", flush=True)
        t_dl = time.time()
        p = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            local_dir=a.out_dir,
            local_dir_use_symlinks=False,
        )
        sz_mb = os.path.getsize(p) / (1024 * 1024)
        print(f"  [done] {filename} ({sz_mb:.1f} MB, {time.time() - t_dl:.1f}s)")
        downloaded.append(p)

    print(f"\n[fetch] successfully fetched {len(downloaded)} shots in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
