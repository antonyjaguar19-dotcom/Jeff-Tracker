# Copyright (c) 2026 Jeff-Tracker contributors
# Licensed under Apache 2.0 (see LICENSES.md)

"""Train the lightweight Smart Neural Feature Router in RAM.

Trains SmartRouter (< 65k parameters) in GPU memory to classify visual patches
and temporal dynamics into 'agile' vs 'smoke'.
"""

from __future__ import annotations

import glob
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jefftrack.router import SmartRouter, ROUTER_WEIGHTS


def extract_crops_from_plate(flist, label, num_crops_per_pair=20, max_pairs=10, patch_size=64):
    tensors = []
    labels = []
    r = patch_size // 2

    # Step through plate
    step = max(1, len(flist) // max_pairs)
    pairs = [(flist[i], flist[min(len(flist)-1, i + 1)]) for i in range(0, min(len(flist) - 1, step * max_pairs), step)]

    for f0_p, f1_p in pairs:
        im0 = cv2.imread(f0_p)
        im1 = cv2.imread(f1_p)
        if im0 is None or im1 is None:
            continue
        H, W = im0.shape[:2]

        if label == 0:
            # Agile: corners / high-gradient spots
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            corners = cv2.goodFeaturesToTrack(gray, maxCorners=60, qualityLevel=0.01, minDistance=20)
            pts = [(int(pt[0, 0]), int(pt[0, 1])) for pt in corners] if corners is not None else []
            for _ in range(num_crops_per_pair):
                pts.append((random.randint(r + 10, W - r - 10), random.randint(r + 10, H - r - 10)))
        else:
            # Smoke / fluid: diffuse and middle regions
            pts = [(random.randint(r + 10, W - r - 10), random.randint(r + 10, H - r - 10))
                   for _ in range(num_crops_per_pair * 2)]

        random.shuffle(pts)
        for cx, cy in pts[:num_crops_per_pair]:
            cx = int(np.clip(cx, r, W - r - 1))
            cy = int(np.clip(cy, r, H - r - 1))
            c0 = im0[cy - r:cy + r, cx - r:cx + r]
            c1 = im1[cy - r:cy + r, cx - r:cx + r]
            if c0.shape[:2] != (patch_size, patch_size):
                continue
            rgb0 = c0[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
            rgb1 = c1[:, :, ::-1].astype(np.float32) / 127.5 - 1.0
            comb = np.concatenate([rgb0, rgb1], axis=-1)  # (64, 64, 6)
            tensors.append(torch.from_numpy(comb).permute(2, 0, 1).float())
            labels.append(label)

    return tensors, labels


def main():
    print("=== Training Smart Neural Feature Router ===", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base_in = "E:/CHAT_GPT/Co_Tracker_v001/tests/IN"

    agile_plates = ["Jocker_plate", "Cycle", "Dance1_plate", "Dance2_plate"]
    smoke_plates = ["Smoke1", "water"]

    all_tensors = []
    all_labels = []

    print("Harvesting and caching crops in memory...", flush=True)
    t0 = time.time()
    for p in agile_plates:
        flist = sorted(glob.glob(os.path.join(base_in, p, "*.*")))
        if flist:
            t, l = extract_crops_from_plate(flist, label=0, num_crops_per_pair=25, max_pairs=8)
            all_tensors.extend(t)
            all_labels.extend(l)

    for p in smoke_plates:
        flist = sorted(glob.glob(os.path.join(base_in, p, "*.*")))
        if flist:
            t, l = extract_crops_from_plate(flist, label=1, num_crops_per_pair=30, max_pairs=10)
            all_tensors.extend(t)
            all_labels.extend(l)

    X = torch.stack(all_tensors, dim=0)  # (N, 6, 64, 64)
    Y = torch.tensor(all_labels, dtype=torch.long)
    print(f"Harvested {len(Y)} crops in {time.time()-t0:.2f}s ({sum(1 for y in all_labels if y==0)} Agile, {sum(1 for y in all_labels if y==1)} Smoke)", flush=True)

    # Train / Val Split
    indices = torch.randperm(len(Y))
    val_sz = int(len(Y) * 0.15)
    train_idx = indices[val_sz:]
    val_idx = indices[:val_sz]

    train_ds = TensorDataset(X[train_idx], Y[train_idx])
    val_ds = TensorDataset(X[val_idx], Y[val_idx])

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = SmartRouter().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    epochs = 25
    best_acc = 0.0

    print(f"Training on {device}...", flush=True)
    t_train = time.time()
    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = torch.argmax(logits, dim=-1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        train_acc = correct / max(1, total)

        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for vx, vy in val_loader:
                vx, vy = vx.to(device), vy.to(device)
                v_logits = model(vx)
                v_preds = torch.argmax(v_logits, dim=-1)
                v_correct += (v_preds == vy).sum().item()
                v_total += vx.size(0)
        val_acc = v_correct / max(1, v_total)

        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            print(f"Epoch {ep+1:02d}/{epochs:02d}: Loss={total_loss/total:.4f}, Train Acc={train_acc*100:.1f}%, Val Acc={val_acc*100:.1f}%", flush=True)

        if val_acc > best_acc or ep == epochs - 1:
            best_acc = val_acc
            os.makedirs(os.path.dirname(ROUTER_WEIGHTS), exist_ok=True)
            torch.save(model.state_dict(), ROUTER_WEIGHTS)

    print(f"\n[SmartRouter] Done in {time.time()-t_train:.2f}s! Best Val Acc: {best_acc*100:.2f}%", flush=True)
    print(f"[SmartRouter] Saved weights to {ROUTER_WEIGHTS} ({os.path.getsize(ROUTER_WEIGHTS)/1024:.1f} KB)", flush=True)


if __name__ == "__main__":
    main()
