# Copyright (c) 2026 Jeff-Tracker contributors
# Licensed under Apache 2.0 (see LICENSES.md)

"""Dual-Expert Mixture-of-Experts (MoE) Tracking Engine.

Maintains two specialized model checkpoints concurrently in GPU memory:
  1. Agile Kinetic Expert (jefftrack_reacq_stream / locotrack_base):
     Optimized for fast camera whip, rapid saccadic motion (eyeballs, fingers),
     high-gradient micro-features, and unfiltered sharp movements.
  2. VFX & Disturbance Invariant Expert (jefftrack_large_v3):
     Optimized for dense atmospheric smoke, fire, lighting flares, occlusions,
     and non-rigid surface mesh deformation.

Provides automatic per-seed routing using photometric patch verification (NCC)
and neural confidence, ensuring each tracked feature uses its optimal expert.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

from .engine import JeffTrackEngine, DEFAULT_MODEL_RES, ROOT


def _calc_patch_ncc(gray_img: np.ndarray, pt: Tuple[float, float],
                    ref_patch: np.ndarray, r: int = 9) -> float:
    """Compute Normalized Cross Correlation of patch at pt against ref_patch."""
    H, W = gray_img.shape[:2]
    cx, cy = int(round(pt[0])), int(round(pt[1]))
    if cx - r < 0 or cx + r + 1 > W or cy - r < 0 or cy + r + 1 > H:
        return 0.0
    patch = gray_img[cy - r:cy + r + 1, cx - r:cx + r + 1].astype(np.float32)
    p_std = np.std(patch)
    ref_std = np.std(ref_patch)
    if p_std < 1e-4 or ref_std < 1e-4:
        return 0.5  # Low texture region
    p_norm = (patch - np.mean(patch)) / p_std
    ref_norm = (ref_patch - np.mean(ref_patch)) / ref_std
    return float(np.clip(np.mean(p_norm * ref_norm), -1.0, 1.0))


class DualExpertEngine:
    """Dual-Expert Tracking Engine with smart per-point routing."""

    def __init__(
        self,
        agile_ckpt: Optional[str] = None,
        smoke_ckpt: Optional[str] = None,
        ckpt: Optional[str] = None,
        device: str = "cuda",
        model_size: str = "base",
        model_res: Tuple[int, int] = DEFAULT_MODEL_RES,
        window: int = 0,
        window_overlap: int = 32,
        **kw,
    ):
        if ckpt:
            smoke_ckpt = smoke_ckpt or ckpt
        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = device
        self.model_res = model_res
        self.window = window
        self.window_overlap = window_overlap

        # Resolve agile checkpoint (favors streaming reacquisition or base)
        weights_dir = os.path.join(ROOT, "weights")
        if agile_ckpt is None:
            candidates_agile = [
                os.path.join(weights_dir, "jefftrack_reacq_stream.ckpt"),
                os.path.join(weights_dir, "locotrack_base.ckpt"),
                os.path.join(weights_dir, "backup_v2_smooth", "jefftrack_reacq_v2_smooth.ckpt"),
            ]
            for c in candidates_agile:
                if os.path.isfile(c):
                    agile_ckpt = c
                    break
        if agile_ckpt is None:
            agile_ckpt = os.path.join(weights_dir, "locotrack_base.ckpt")

        # Resolve smoke & disturbance checkpoint (favors large v3 widened model)
        if smoke_ckpt is None:
            candidates_smoke = [
                os.path.join(weights_dir, "jefftrack_large_v3.ckpt"),
                os.path.join(weights_dir, "backup_v2_smooth", "jefftrack_reacq_v2_smooth.ckpt"),
                os.path.join(weights_dir, "jefftrack_reacq_stream.ckpt"),
            ]
            for c in candidates_smoke:
                if os.path.isfile(c):
                    smoke_ckpt = c
                    break
        if smoke_ckpt is None:
            smoke_ckpt = agile_ckpt

        self.agile_ckpt = agile_ckpt
        self.smoke_ckpt = smoke_ckpt

        arch_agile = "locotrack" if "locotrack" in os.path.basename(agile_ckpt).lower() else "jefftrack"
        arch_smoke = "locotrack" if "locotrack" in os.path.basename(smoke_ckpt).lower() else "jefftrack"

        print(f"[DualExpert] Initializing Dual-Expert Engine on {device}:")
        print(f"  Expert 1 (Agile Kinetic): {os.path.basename(agile_ckpt)} ({arch_agile})")
        print(f"  Expert 2 (Smoke & VFX Invariant): {os.path.basename(smoke_ckpt)} ({arch_smoke})")

        self.expert_agile = JeffTrackEngine(
            ckpt=agile_ckpt, arch=arch_agile, device=device,
            model_size=model_size, model_res=model_res, window=window,
            window_overlap=window_overlap
        )

        if os.path.abspath(smoke_ckpt) != os.path.abspath(agile_ckpt):
            self.expert_smoke = JeffTrackEngine(
                ckpt=smoke_ckpt, arch=arch_smoke, device=device,
                model_size=model_size, model_res=model_res, window=window,
                window_overlap=window_overlap
            )
        else:
            self.expert_smoke = self.expert_agile

        self.last_routing_info: Dict[str, object] = {}

    def track_queries_conf(
        self,
        frames_bgr: np.ndarray | List[np.ndarray],
        queries: np.ndarray,
        expert_mode: str = "auto",
        fp16: bool = False,
        is_bgr: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track queries using selected or automatically routed expert model.

        Parameters:
            frames_bgr: (T, H, W, 3) video frames in BGR or RGB uint8
            queries: (N, 3) [frame, x, y] query coordinates
            expert_mode: 'auto' (smart per-track routing), 'agile', or 'smoke'
            is_bgr: True if frames are BGR, False if RGB (avoids 6.5 GB array copies!)
        Returns:
            tracks: (T, N, 2)
            vis: (T, N) bool
            conf: (T, N) float 0..1
        """
        expert_mode = str(expert_mode).strip().lower()

        # If forced to a specific expert or models are identical:
        if expert_mode == "agile" or self.expert_smoke is self.expert_agile:
            tr, vis, conf = self.expert_agile.track_queries_conf(frames_bgr, queries, fp16=fp16, is_bgr=is_bgr)
            self.last_routing_info = {"mode": expert_mode, "selected": ["agile"] * tr.shape[1]}
            return tr, vis, conf

        if expert_mode in ("smoke", "large", "large_v3"):
            tr, vis, conf = self.expert_smoke.track_queries_conf(frames_bgr, queries, fp16=fp16, is_bgr=is_bgr)
            self.last_routing_info = {"mode": expert_mode, "selected": ["smoke"] * tr.shape[1]}
            return tr, vis, conf

        # --- Smart Neural Routing Mode ('auto' or 'smart') ---
        # 1. Inspect each seed point in < 1 millisecond using the trained SmartRouter
        from .router import predict_expert_for_seed
        q = np.asarray(queries, np.float32).reshape(-1, 3)
        N = q.shape[0]
        T = len(frames_bgr)
        selected_experts = []
        conf_preds = []

        for i in range(N):
            q_fr = int(np.clip(q[i, 0], 0, T - 1))
            f_curr0 = frames_bgr[q_fr]
            f_curr1 = frames_bgr[min(T - 1, q_fr + 1)]
            # If input is RGB, convert the tiny 64x64 patch to BGR for the router
            if not is_bgr:
                f_curr0 = f_curr0[..., ::-1]
                f_curr1 = f_curr1[..., ::-1]
            exp_name, conf = predict_expert_for_seed(f_curr0, f_curr1, (q[i, 1], q[i, 2]), device=self.device)
            selected_experts.append(exp_name)
            conf_preds.append(conf)

        # 2. If all queries agree (e.g. single point or all agile / all smoke):
        # Run ONLY the winning expert model at full 1x speed!
        unique_experts = set(selected_experts)
        if len(unique_experts) == 1:
            chosen = selected_experts[0]
            print(f"[SmartRouter] Auto-routed {N} query(ies) -> Expert: '{chosen}' ({conf_preds[0]*100:.1f}% confidence)")
            eng = self.expert_smoke if chosen in ("smoke", "large", "large_v3") else self.expert_agile
            tr, vis, conf = eng.track_queries_conf(frames_bgr, queries, fp16=fp16, is_bgr=is_bgr)
            self.last_routing_info = {"mode": "smart", "selected": selected_experts, "conf": conf_preds}
            return tr, vis, conf

        # 3. If heterogeneous queries (e.g. point 1 is eyeball, point 2 is smoke):
        # Track each query with its specialized expert and assemble the result
        print(f"[SmartRouter] Heterogeneous queries detected: {selected_experts.count('agile')} Agile, {selected_experts.count('smoke')} Smoke")
        out_tr = np.zeros((T, N, 2), dtype=np.float32)
        out_vis = np.zeros((T, N), dtype=bool)
        out_conf = np.zeros((T, N), dtype=np.float32)

        agile_indices = [i for i, exp in enumerate(selected_experts) if exp == "agile"]
        smoke_indices = [i for i, exp in enumerate(selected_experts) if exp != "agile"]

        if agile_indices:
            tr_a, v_a, c_a = self.expert_agile.track_queries_conf(frames_bgr, queries[agile_indices], fp16=fp16, is_bgr=is_bgr)
            out_tr[:, agile_indices] = tr_a
            out_vis[:, agile_indices] = v_a
            out_conf[:, agile_indices] = c_a

        if smoke_indices:
            tr_s, v_s, c_s = self.expert_smoke.track_queries_conf(frames_bgr, queries[smoke_indices], fp16=fp16, is_bgr=is_bgr)
            out_tr[:, smoke_indices] = tr_s
            out_vis[:, smoke_indices] = v_s
            out_conf[:, smoke_indices] = c_s

        self.last_routing_info = {"mode": "smart", "selected": selected_experts, "conf": conf_preds}
        return out_tr, out_vis, out_conf
