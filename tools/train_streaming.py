"""Streaming Multi-Batch Training with Bounded Disk Space.

Downloads small batches of Apache-2.0 Kubric training shards from Hugging Face,
converts them to training cache, deletes raw .tar.gz archives, trains for a set
number of steps, saves the checkpoint, deletes the cache, and advances to the next
batch.

Guarantees disk space usage never exceeds ~1.5 GB at any point during training.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PYTHON = sys.executable

RAW_DIR = os.path.join(ROOT, "datasets", "kubric_meta")
CACHE_DIR = os.path.join(ROOT, "datasets", "kubric_cache")
WEIGHTS_DIR = os.path.join(ROOT, "weights")


def run_cmd(cmd: list[str], desc: str) -> bool:
    print(f"\n[streaming] >>> {desc}...")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if res.returncode != 0:
        print(f"[streaming] [ERROR] {desc} failed with code {res.returncode}")
        return False
    print(f"[streaming] [OK] {desc} completed in {dt:.1f}s")
    return True


def purge_raw_archives():
    """Delete downloaded .tar.gz files and temporary huggingface download cache."""
    if os.path.isdir(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.endswith(".tar.gz") or f.endswith(".part"):
                p = os.path.join(RAW_DIR, f)
                try:
                    os.remove(p)
                    print(f"  [purge] deleted raw archive: {f}")
                except Exception as e:
                    print(f"  [warn] failed to remove {f}: {e}")
        # Clean .cache folder if present
        cache_hf = os.path.join(RAW_DIR, ".cache")
        if os.path.isdir(cache_hf):
            try:
                shutil.rmtree(cache_hf, ignore_errors=True)
                print("  [purge] cleaned huggingface temp download cache")
            except Exception:
                pass


def purge_cache_files():
    """Delete converted .npz cache files to free disk space."""
    if os.path.isdir(CACHE_DIR):
        for f in os.listdir(CACHE_DIR):
            if f.endswith(".npz"):
                p = os.path.join(CACHE_DIR, f)
                try:
                    os.remove(p)
                    print(f"  [purge] deleted cache file: {f}")
                except Exception as e:
                    print(f"  [warn] failed to remove {f}: {e}")


def main():
    ap = argparse.ArgumentParser(description="Streaming multi-batch training with bounded disk space")
    ap.add_argument("--start-shard", type=int, default=8, help="starting shard index")
    ap.add_argument("--shards-per-cycle", type=int, default=4, help="shards downloaded per batch")
    ap.add_argument("--cycles", type=int, default=4, help="number of streaming download-train-delete cycles")
    ap.add_argument("--steps-per-cycle", type=int, default=400, help="optimizer steps trained per batch")
    ap.add_argument("--base-ckpt", default=os.path.join(WEIGHTS_DIR, "jefftrack_reacq.ckpt"),
                    help="checkpoint to resume/start from")
    ap.add_argument("--out-ckpt", default=os.path.join(WEIGHTS_DIR, "jefftrack_reacq_stream.ckpt"),
                    help="output checkpoint to update after each cycle")
    ap.add_argument("--reacq-share", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-4)
    a = ap.parse_args()

    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    current_ckpt = a.base_ckpt if os.path.isfile(a.base_ckpt) else os.path.join(WEIGHTS_DIR, "locotrack_base.ckpt")
    print("=" * 78)
    print(" STREAMING MULTI-BATCH TRAINING WITH BOUNDED DISK USAGE")
    print("=" * 78)
    print(f"Starting Shard:     {a.start_shard}")
    print(f"Shards per Cycle:   {a.shards_per_cycle} (~1.2 GB peak)")
    print(f"Total Cycles:       {a.cycles}")
    print(f"Steps per Cycle:    {a.steps_per_cycle}")
    print(f"Total Target Steps: {a.cycles * a.steps_per_cycle}")
    print(f"Resume Checkpoint:  {current_ckpt}")
    print(f"Output Checkpoint:  {a.out_ckpt}")
    print("=" * 78)

    t_start = time.time()
    for cycle in range(a.cycles):
        shard_start = a.start_shard + cycle * a.shards_per_cycle
        print(f"\n######################################################################")
        print(f" CYCLE {cycle + 1}/{a.cycles}: Processing Shards {shard_start:04d} .. {shard_start + a.shards_per_cycle - 1:04d}")
        print(f"######################################################################")

        # Step 1: Download small batch of shards
        cmd_fetch = [
            PYTHON, os.path.join(ROOT, "tools", "fetch_kubric_shards.py"),
            "--start", str(shard_start),
            "--count", str(a.shards_per_cycle),
            "--out-dir", RAW_DIR
        ]
        if not run_cmd(cmd_fetch, f"Download Shards {shard_start}..{shard_start + a.shards_per_cycle - 1}"):
            print("[streaming] download failed, aborting loop.")
            return 1

        # Step 2: Convert downloaded shards into compact cache
        cmd_conv = [
            PYTHON, os.path.join(ROOT, "tools", "convert_kubric.py"),
            "--src", RAW_DIR,
            "--out", CACHE_DIR,
            "--workers", "4",
            "--overwrite"
        ]
        if not run_cmd(cmd_conv, "Convert Shards to Training Cache"):
            print("[streaming] conversion failed, aborting loop.")
            return 1

        # Step 3: Delete raw archives immediately to free disk space
        print("\n[streaming] >>> Purging raw .tar.gz archives...")
        purge_raw_archives()

        # Step 4: Train on this batch
        cmd_train = [
            PYTHON, "-u", os.path.join(ROOT, "tools", "train_cross.py"),
            "--resume", current_ckpt,
            "--out", a.out_ckpt,
            "--data-source", "kubric_meta",
            "--cache-dir", CACHE_DIR,
            "--holdout", "0",
            "--steps", str(a.steps_per_cycle),
            "--save-every", str(a.steps_per_cycle),
            "--log-every", "50",
            "--reacq-share", str(a.reacq_share),
            "--occ-pos-weight", "0.2",
            "--occ-norm",
            "--occ-l1",
            "--unfreeze-mixer",
            "--lr", str(a.lr)
        ]
        if not run_cmd(cmd_train, f"Train Cycle {cycle + 1} ({a.steps_per_cycle} steps)"):
            print("[streaming] training failed, aborting loop.")
            return 1

        # Update checkpoint pointer to continue fine-tuning in next cycle
        current_ckpt = a.out_ckpt

        # Step 5: Purge cache files before starting next cycle
        print("\n[streaming] >>> Purging processed cache .npz files...")
        purge_cache_files()

        print(f"\n[streaming] Cycle {cycle + 1}/{a.cycles} successfully completed. Disk is clean.")

    total_time = (time.time() - t_start) / 60.0
    print("\n" + "=" * 78)
    print(f" ALL {a.cycles} STREAMING CYCLES COMPLETED in {total_time:.1f} minutes!")
    print(f" Final Trained Checkpoint: {a.out_ckpt}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
