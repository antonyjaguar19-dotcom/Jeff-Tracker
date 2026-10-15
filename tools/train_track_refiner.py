# Copyright (c) 2026 Jeff-Tracker contributors
# Licensed under Apache 2.0 (see LICENSES.md)

"""Train the 2nd-Pass Neural Trajectory Refiner on Kubric GT tracks.

Trains TrackRefiner to learn:
1. Physical momentum & trajectory smoothness.
2. In-filling occlusions from unoccluded endpoint anchors.
3. Sub-pixel noise rejection and confidence calibration.
"""
from __future__ import annotations

import argparse
import glob
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from jefftrack.model.refiner import TrackRefiner


class KubricTrackDataset(Dataset):
    """Loads 2D trajectories and occlusion masks from converted Kubric cache."""

    def __init__(self, cache_dir: str, clip_len: int = 64, split: str = "train", holdout: int = 1):
        super().__init__()
        self.clip_len = clip_len
        npz_files = sorted(glob.glob(os.path.join(cache_dir, "*.npz")))
        if not npz_files:
            raise RuntimeError(f"No .npz files found in {cache_dir}")

        if split == "train":
            self.files = npz_files[:-holdout] if len(npz_files) > holdout else npz_files
        else:
            self.files = npz_files[-holdout:] if len(npz_files) > holdout else npz_files

        print(f"[Dataset {split}] loaded {len(self.files)} shots from {cache_dir}")
        self.shots = []
        for f in self.files:
            data = np.load(f)
            self.shots.append({
                "coords": data["coords"],       # (4096, 120, 2)
                "occluded": data["occluded"],   # (4096, 120)
                "H": int(data["height"]),
                "W": int(data["width"])
            })
        # Pre-filter points with meaningful occlusion events (>= 6 occluded frames)
        self.occl_point_indices = []
        for s in self.shots:
            occ_counts = np.sum(s["occluded"], axis=1)
            candidates = np.where(occ_counts >= 6)[0]
            self.occl_point_indices.append(candidates if len(candidates) > 0 else np.arange(len(occ_counts)))

    def __len__(self) -> int:
        return len(self.shots) * 1000  # Virtual epoch length

    def __getitem__(self, idx: int):
        shot_idx = random.randint(0, len(self.shots) - 1)
        shot = self.shots[shot_idx]
        H, W = shot["H"], shot["W"]

        # 70% chance to sample a track that experiences occlusion
        if random.random() < 0.70 and len(self.occl_point_indices[shot_idx]) > 0:
            pt_idx = random.choice(self.occl_point_indices[shot_idx])
        else:
            pt_idx = random.randint(0, shot["coords"].shape[0] - 1)

        T_total = shot["coords"].shape[1]
        start_t = random.randint(0, max(0, T_total - self.clip_len))
        end_t = start_t + self.clip_len

        gt_coords = shot["coords"][pt_idx, start_t:end_t].copy()  # (T, 2)
        gt_occl = shot["occluded"][pt_idx, start_t:end_t].copy()   # (T,)

        # Synthesize realistic Pass 1 tracking imperfections:
        noisy_coords = gt_coords.copy()
        noisy_conf = np.ones((self.clip_len,), dtype=np.float32)

        # 1. Subtle sensor jitter on visible frames
        vis_mask = ~gt_occl
        noisy_coords[vis_mask] += np.random.normal(0.0, 0.40, size=noisy_coords[vis_mask].shape)
        noisy_conf[vis_mask] = np.random.uniform(0.80, 1.0, size=np.sum(vis_mask))

        # 2. Tracking drift & uncertainty during occlusion
        if np.any(gt_occl):
            occl_indices = np.where(gt_occl)[0]
            noisy_conf[occl_indices] = np.random.uniform(0.05, 0.35, size=len(occl_indices))
            drift_vec = np.random.normal(0.0, 14.0, size=2)
            drift_curve = np.linspace(0.0, 1.0, len(occl_indices))[:, None]
            noisy_coords[occl_indices] += drift_curve * drift_vec

        return (
            torch.from_numpy(noisy_coords).float(),
            torch.from_numpy(noisy_conf).float(),
            torch.from_numpy(gt_coords).float(),
            torch.from_numpy(~gt_occl).float(),  # GT visibility
            torch.tensor([H, W], dtype=torch.float32)
        )


def main():
    parser = argparse.ArgumentParser(description="Train Neural Trajectory Refiner")
    parser.add_argument("--cache-dir", default=os.path.join(ROOT, "datasets", "kubric_cache"))
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--clip-len", type=int, default=64)
    parser.add_argument("--log-file", default=os.path.join(ROOT, "weights", "refiner_training.log"))
    parser.add_argument("--out", default=os.path.join(ROOT, "weights", "track_refiner_20k.pt"))
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train Refiner] device: {device}, target steps: {args.steps}")

    train_ds = KubricTrackDataset(args.cache_dir, clip_len=args.clip_len, split="train", holdout=2)
    val_ds = KubricTrackDataset(args.cache_dir, clip_len=args.clip_len, split="val", holdout=2)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = TrackRefiner(in_dim=7, hidden_dim=128, num_layers=4, num_heads=4).to(device)
    # Initialize delta head with small random weights so gradients flow immediately
    nn.init.normal_(model.delta_head[-1].weight, std=1e-3)
    nn.init.zeros_(model.delta_head[-1].bias)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.05)

    pos_criterion = nn.SmoothL1Loss(beta=1.0)
    conf_criterion = nn.BCELoss()

    step = 0
    t0 = time.time()

    print("[Train Refiner] starting training loop...")
    while step < args.steps:
        for batch in train_loader:
            step += 1
            if step > args.steps:
                break

            noisy_p, noisy_c, gt_p, gt_vis, img_sz = [x.to(device) for x in batch]
            img_hw = (int(img_sz[0, 0].item()), int(img_sz[0, 1].item()))

            optimizer.zero_grad()
            pred_p, pred_c = model(noisy_p, noisy_c, img_size=img_hw)

            # 1. Position error against GT
            l_pos = pos_criterion(pred_p, gt_p)
            # Extra penalty on occluded frames where drift occurs
            occl_mask = (gt_vis == 0)
            if torch.any(occl_mask):
                l_pos_occl = pos_criterion(pred_p[occl_mask], gt_p[occl_mask]) * 3.0
            else:
                l_pos_occl = 0.0

            # 2. Smoothness regularization (acceleration penalty)
            acc = pred_p[:, 2:] - 2.0 * pred_p[:, 1:-1] + pred_p[:, :-2]
            l_smooth = torch.mean(torch.sum(acc ** 2, dim=-1)) * 0.05

            # 3. Confidence calibration BCE
            l_conf = conf_criterion(pred_c.squeeze(-1), gt_vis)

            total_loss = l_pos + l_pos_occl + l_smooth + 0.1 * l_conf
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if step % 250 == 0 or step == 1:
                err_raw = torch.mean(torch.norm(noisy_p - gt_p, dim=-1)).item()
                err_refined = torch.mean(torch.norm(pred_p - gt_p, dim=-1)).item()
                dt = time.time() - t0
                log_line = (f"Step {step:05d}/{args.steps} | Loss: {total_loss.item():.4f} (pos: {l_pos.item():.4f}) | "
                            f"Raw Err: {err_raw:.2f} px -> Refined Err: {err_refined:.2f} px | {dt:.1f}s")
                print(log_line)
                with open(args.log_file, "a") as log_f:
                    log_f.write(f"{time.time()},{step},{total_loss.item():.4f},{l_pos.item():.4f},{err_raw:.4f},{err_refined:.4f}\n")

            if step % args.save_every == 0 or step == args.steps:
                model.eval()
                val_err_raw = []
                val_err_ref = []
                with torch.no_grad():
                    for v_batch in val_loader:
                        vn_p, vn_c, v_gt_p, v_gt_vis, v_img_sz = [x.to(device) for x in v_batch]
                        v_hw = (int(v_img_sz[0, 0].item()), int(v_img_sz[0, 1].item()))
                        vp_p, vp_c = model(vn_p, vn_c, img_size=v_hw)
                        val_err_raw.append(torch.mean(torch.norm(vn_p - v_gt_p, dim=-1)).item())
                        val_err_ref.append(torch.mean(torch.norm(vp_p - v_gt_p, dim=-1)).item())
                        if len(val_err_raw) >= 15:
                            break
                m_raw = np.mean(val_err_raw)
                m_ref = np.mean(val_err_ref)
                val_str = f"  [Validation @ {step}] Mean Raw Error: {m_raw:.2f} px -> Refined Error: {m_ref:.2f} px (Gain: {m_raw - m_ref:.2f} px)"
                print(val_str)
                with open(args.log_file, "a") as log_f:
                    log_f.write(f"# {val_str}\n")

                ckpt_data = {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "val_err_raw": m_raw,
                    "val_err_refined": m_ref
                }
                torch.save(ckpt_data, args.out)
                rolling_path = args.out.replace(".pt", f"_step{step}.pt")
                torch.save(ckpt_data, rolling_path)
                print(f"  [Saved Checkpoint] {args.out} and {rolling_path}")
                model.train()

    print(f"[Train Refiner] training completed in {time.time() - t0:.1f}s!")


if __name__ == "__main__":
    main()
