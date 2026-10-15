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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Track queries using selected or automatically routed expert model.

        Parameters:
            frames_bgr: (T, H, W, 3) video frames in BGR uint8
            queries: (N, 3) [frame, x, y] query coordinates
            expert_mode: 'auto' (smart per-track routing), 'agile', or 'smoke'
        Returns:
            tracks: (T, N, 2)
            vis: (T, N) bool
            conf: (T, N) float 0..1
        """
        expert_mode = str(expert_mode).strip().lower()

        # If forced to a specific expert or models are identical:
        if expert_mode == "agile" or self.expert_smoke is self.expert_agile:
            tr, vis, conf = self.expert_agile.track_queries_conf(frames_bgr, queries, fp16=fp16)
            self.last_routing_info = {"mode": expert_mode, "selected": ["agile"] * tr.shape[1]}
            return tr, vis, conf

        if expert_mode in ("smoke", "large", "large_v3"):
            tr, vis, conf = self.expert_smoke.track_queries_conf(frames_bgr, queries, fp16=fp16)
            self.last_routing_info = {"mode": expert_mode, "selected": ["smoke"] * tr.shape[1]}
            return tr, vis, conf

        # --- Automatic Multi-Weight Routing Mode ('auto') ---
        # 1. Run inference on both experts
        tr_a, vis_a, conf_a = self.expert_agile.track_queries_conf(frames_bgr, queries, fp16=fp16)
        tr_s, vis_s, conf_s = self.expert_smoke.track_queries_conf(frames_bgr, queries, fp16=fp16)

        T, N, _ = tr_a.shape
        q = np.asarray(queries, np.float32).reshape(-1, 3)

        out_tr = np.zeros_like(tr_a)
        out_vis = np.zeros_like(vis_a)
        out_conf = np.zeros_like(conf_a)
        selected_experts = []
        scores_breakdown = {}

        # Convert sample frames to grayscale for rapid photometric patch check
        sample_indices = np.linspace(0, T - 1, min(T, 12), dtype=int)
        grays = {}
        for s_idx in sample_indices:
            fr = frames_bgr[s_idx]
            grays[s_idx] = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY) if fr.ndim == 3 else fr

        r = 9  # patch radius (19x19 patch)

        for i in range(N):
            q_fr = int(np.clip(q[i, 0], 0, T - 1))
            seed_x, seed_y = float(q[i, 1]), float(q[i, 2])

            # Extract seed reference patch
            if q_fr not in grays:
                fr_seed = frames_bgr[q_fr]
                grays[q_fr] = cv2.cvtColor(fr_seed, cv2.COLOR_BGR2GRAY) if fr_seed.ndim == 3 else fr_seed

            g_seed = grays[q_fr]
            H_g, W_g = g_seed.shape[:2]
            sx_i, sy_i = int(round(seed_x)), int(round(seed_y))
            if (sx_i - r >= 0 and sx_i + r + 1 <= W_g and sy_i - r >= 0 and sy_i + r + 1 <= H_g):
                ref_patch = g_seed[sy_i - r:sy_i + r + 1, sx_i - r:sx_i + r + 1].astype(np.float32)
            else:
                ref_patch = np.zeros((2 * r + 1, 2 * r + 1), dtype=np.float32)

            # Evaluate Expert A (Agile)
            ncc_a_list = []
            for s_idx in sample_indices:
                if s_idx == q_fr:
                    continue
                pt_a = tr_a[s_idx, i]
                score_ncc = _calc_patch_ncc(grays[s_idx], (pt_a[0], pt_a[1]), ref_patch, r=r)
                ncc_a_list.append(score_ncc)
            mean_ncc_a = float(np.mean(ncc_a_list)) if ncc_a_list else 0.5
            mean_conf_a = float(np.mean(conf_a[:, i]))

            # Acceleration / jitter penalty for Expert A
            acc_a = np.linalg.norm(tr_a[2:, i] - 2 * tr_a[1:-1, i] + tr_a[:-2, i], axis=-1) if T >= 3 else [0.0]
            jitter_a = float(np.mean(acc_a))

            # Evaluate Expert S (Smoke / Large-v3)
            ncc_s_list = []
            for s_idx in sample_indices:
                if s_idx == q_fr:
                    continue
                pt_s = tr_s[s_idx, i]
                score_ncc = _calc_patch_ncc(grays[s_idx], (pt_s[0], pt_s[1]), ref_patch, r=r)
                ncc_s_list.append(score_ncc)
            mean_ncc_s = float(np.mean(ncc_s_list)) if ncc_s_list else 0.5
            mean_conf_s = float(np.mean(conf_s[:, i]))

            # Acceleration / jitter penalty for Expert S
            acc_s = np.linalg.norm(tr_s[2:, i] - 2 * tr_s[1:-1, i] + tr_s[:-2, i], axis=-1) if T >= 3 else [0.0]
            jitter_s = float(np.mean(acc_s))

            # Composite Quality Metric:
            # 50% Photometric Patch Fidelity (NCC)
            # 40% Continuous Neural Confidence
            # 10% Temporal Stability
            q_score_a = 0.50 * max(0.0, mean_ncc_a) + 0.40 * mean_conf_a + 0.10 / (1.0 + 0.05 * jitter_a)
            q_score_s = 0.50 * max(0.0, mean_ncc_s) + 0.40 * mean_conf_s + 0.10 / (1.0 + 0.05 * jitter_s)

            # Decision rule:
            # Large-v3 (Smoke expert) handles heavy smoke and occlusions where NCC drops for both.
            # When NCC is low (< 0.25) because of visual degradation (smoke/shadow), higher neural confidence wins!
            # When NCC is strong (clear feature, eyes, high contrast), patch fidelity takes top priority!
            if (mean_ncc_a < 0.25 and mean_ncc_s < 0.25):
                use_expert = "smoke" if mean_conf_s >= mean_conf_a else "agile"
            else:
                use_expert = "agile" if q_score_a >= q_score_s else "smoke"

            selected_experts.append(use_expert)
            scores_breakdown[i] = {
                "expert": use_expert,
                "agile": {"ncc": mean_ncc_a, "conf": mean_conf_a, "jitter": jitter_a, "score": q_score_a},
                "smoke": {"ncc": mean_ncc_s, "conf": mean_conf_s, "jitter": jitter_s, "score": q_score_s},
            }

            if use_expert == "agile":
                out_tr[:, i] = tr_a[:, i]
                out_vis[:, i] = vis_a[:, i]
                out_conf[:, i] = conf_a[:, i]
            else:
                out_tr[:, i] = tr_s[:, i]
                out_vis[:, i] = vis_s[:, i]
                out_conf[:, i] = conf_s[:, i]

        self.last_routing_info = {
            "mode": "auto",
            "selected": selected_experts,
            "breakdown": scores_breakdown,
        }

        print(f"[DualExpert] Routed {N} track(s): {selected_experts.count('agile')} Agile, {selected_experts.count('smoke')} Smoke/Large")
        return out_tr, out_vis, out_conf
