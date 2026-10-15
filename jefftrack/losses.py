"""The TAPIR loss, with one change: occluded points get a position gradient.

Measured reason for existing. The vendor's `huber_loss` (model_utils.py:15) does

    loss_huber = loss_huber * (1.0 - occluded.float())

so the position loss is **zeroed** on every frame where the ground truth says the point is
hidden. LocoTrack is therefore never trained to place a point it cannot see, and no amount
of training under that objective can teach it to cross an occlusion -- which is exactly
what the occlusion bench reported: fine-tuning improved visible-frame error 1.20 -> 1.16 px
median while occluded-frame error got *worse*, 3.54 -> 3.66 px, and the fraction of
occluded frames landing within 5 px fell from 71.2% to 69.4%. With no gradient there, the
model is free to drift wherever suits the visible frames.

The published recipe (Karaev et al. 2024, see NOTICE) weights occluded points at one fifth instead of zero -- the `(1_occ/5 + 1_vis)`
term in its loss -- which is the difference between "predict where it went" and "ignore it
until it comes back". That weighting is a formula stated in the paper;
this file is written from that description, and no code here derives from any
implementation of it. See ../LICENSES.md.

Everything else is the vendor's arithmetic, deliberately: same Huber with the same delta,
same coordinate conversion to 256x256, same occlusion BCE, same uncertainty BCE summed over
the refinement iterations. Only the occluded position weight moves, so a difference between
two runs is attributable to it.

Note `prob_loss` keeps its hard mask. That head answers "is this prediction within
`expected_dist_thresh` of the truth", and it is read at inference as confidence; teaching
it to be confident about a point it cannot see would corrupt the one signal this model has
that the TAPNext path lacks (measured AUC 0.955, see ../FINDINGS.md).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from jefftrack.paths import add_vendor_to_path

add_vendor_to_path()

from models.utils import convert_grid_coordinates  # noqa: E402


def prob_loss(tracks, expd, target_points, occluded, expected_dist_thresh=8.0, reduction_axes=(1, 2)):
    """Loss for classifying if a point is within pixel threshold of its target.

    Mirrors vendor model_utils.prob_loss without pulling in data.evaluation_datasets
    (which requires mediapy and tensorflow).
    """
    err = torch.sum((tracks - target_points) ** 2, dim=-1)
    invalid = (err > expected_dist_thresh ** 2).float()
    logprob = F.binary_cross_entropy_with_logits(expd, invalid, reduction="none")
    logprob = logprob * (1.0 - occluded.float())
    if reduction_axes:
        logprob = torch.mean(logprob, dim=reduction_axes)
    return logprob



def huber_loss_weighted(tracks, target_points, occluded, delta=4.0,
                        occluded_weight=0.2, occluded_in_frame_only=True,
                        frame_size=256.0, margin=0.0):
    """Vendor huber_loss with the hard occlusion mask replaced by a weight.

    occluded_weight=0.0 reproduces the vendor's behaviour exactly, which is what makes an
    A/B between the two objectives a one-flag change.

    `occluded_in_frame_only` matters more than it sounds. MOVi-E marks a point occluded
    both when something moves in front of it AND when it leaves the image, and measured on
    this data **49.3% of occluded samples are out of frame**, by up to 218 px (visible
    samples: 0%). Huber is linear past `delta`, so one such sample contributes ~860 while a
    genuinely hidden point contributes single digits -- the off-frame half drowns the half
    worth learning. That is visible in the training loss: 2 to 18 with spikes, against 1.2
    to 1.8 for the masked objective, and an occluded-error gain at 3,000 steps that had
    evaporated by 4,000.

    Predicting where an off-screen point went is not what "track through an occlusion"
    means, and it is not recoverable from the frame. So the occluded weight applies only
    where the target is still inside the image.
    """
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)  # eps prevents a nan gradient at zero
    loss = torch.where(dist < delta, distsqr / 2, delta * (torch.abs(dist) - delta / 2))
    occ = occluded.float()
    if occluded_in_frame_only:
        lo, hi = -margin, frame_size + margin
        inside = ((target_points[..., 0] >= lo) & (target_points[..., 0] < hi) &
                  (target_points[..., 1] >= lo) & (target_points[..., 1] < hi)).float()
        occ = occ * inside
    return loss * ((1.0 - occluded.float()) + occluded_weight * occ)


def cauchy_loss(pred, gt, c=4.0):
    """OmniMotion Cauchy (Lorentzian) robust loss: c^2 * log(1 + ||pred - gt||^2 / c^2).

    Acts quadratically (like L2) for small sub-pixel errors, but has bounded sub-linear logarithmic
    growth with a derivative approaching 0 for gross outliers (severe occlusions, smoke, camera jumps).
    """
    diff = pred - gt
    distsqr = torch.sum(diff ** 2, dim=-1)
    return (c ** 2) * torch.log(1.0 + distsqr / (c ** 2))


def trimmed_mean(loss_tensor, mask, quantile=0.90):
    """OmniMotion quantile trimming: trims top (1 - quantile) worst outlier errors from gradient computation."""
    if mask is None or mask.sum() == 0:
        return loss_tensor.mean()
    valid = loss_tensor[mask > 0.5]
    if valid.numel() == 0:
        return torch.tensor(0.0, device=loss_tensor.device)
    if quantile >= 1.0 or valid.numel() < 10:
        return valid.mean()
    cutoff = torch.quantile(valid.detach(), quantile)
    keep = (valid <= cutoff)
    if keep.sum() == 0:
        return valid.mean()
    return valid[keep].mean()


def position_loss_normalised(tracks, target_points, occluded, delta=4.0,
                             occ_share=0.2, occluded_in_frame_only=True,
                             frame_size=256.0, margin=0.0, occ_l1=False,
                             reacq_share=0.0, use_cauchy=False, loss_quantile=1.0):
    """Position loss whose occluded share is a STATED fraction, not an accident.

    Optionally integrates OmniMotion Cauchy robust loss (use_cauchy=True) and quantile
    trimming (loss_quantile < 1.0) to eliminate outlier gradient spikes from severe smoke/occlusion.
    """
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)

    if use_cauchy:
        loss = cauchy_loss(tracks, target_points, c=delta)
    else:
        loss = torch.where(dist < delta, distsqr / 2, delta * (torch.abs(dist) - delta / 2))

    loss_occ_pop = dist if occ_l1 else loss

    occ = occluded.float()
    if occluded_in_frame_only:
        lo, hi = -margin, frame_size + margin
        inside = ((target_points[..., 0] >= lo) & (target_points[..., 0] < hi) &
                  (target_points[..., 1] >= lo) & (target_points[..., 1] < hi)).float()
        occ = occ * inside
    vis = 1.0 - occluded.float()

    if loss_quantile < 1.0:
        mean_vis = trimmed_mean(loss, vis, quantile=loss_quantile)
        mean_occ = trimmed_mean(loss_occ_pop, occ, quantile=loss_quantile) if float(occ.sum()) > 0.0 else torch.tensor(0.0, device=tracks.device)
    else:
        n_vis, n_occ = vis.sum(), occ.sum()
        mean_vis = (loss * vis).sum() / torch.clamp(n_vis, min=1.0)
        if float(n_occ) <= 0.0:
            mean_occ = torch.tensor(0.0, device=tracks.device)
        else:
            mean_occ = (loss_occ_pop * occ).sum() / n_occ

    # Re-acquisition surge: frames transitioning from occluded (t-1) to visible (t)
    loss_reacq = torch.tensor(0.0, device=tracks.device)
    if reacq_share > 0.0 and occluded.shape[-1] > 1:
        is_reacq = (occluded[..., :-1] > 0.5) & (occluded[..., 1:] < 0.5)
        reacq_mask = torch.cat([torch.zeros_like(is_reacq[..., :1]), is_reacq], dim=-1).float()
        n_reacq = reacq_mask.sum()
        if float(n_reacq) > 0.0:
            loss_reacq = (dist * reacq_mask).sum() / n_reacq

    if float(occ.sum()) <= 0.0:
        if reacq_share > 0.0:
            return (1.0 - reacq_share) * mean_vis + reacq_share * loss_reacq
        return mean_vis

    w_vis = max(0.0, 1.0 - occ_share - reacq_share)
    return w_vis * mean_vis + occ_share * mean_occ + reacq_share * loss_reacq



def occluded_share_of(tracks, target_points, occluded, delta=4.0,
                      occluded_weight=0.2, occluded_in_frame_only=True,
                      frame_size=256.0, margin=0.0):
    """What fraction of the position loss actually comes from occluded frames.

    Exists because the number this file is built around -- "the occluded term is ~1% of
    the position loss" -- was an inference from the data statistics, and an inference is
    not a measurement. This reports it directly for a real batch, so the claim can be
    checked and so `occ_share` can be verified to mean what it says.
    """
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)
    loss = torch.where(dist < delta, distsqr / 2, delta * (torch.abs(dist) - delta / 2))
    occ = occluded.float()
    if occluded_in_frame_only:
        lo, hi = -margin, frame_size + margin
        inside = ((target_points[..., 0] >= lo) & (target_points[..., 0] < hi) &
                  (target_points[..., 1] >= lo) & (target_points[..., 1] < hi)).float()
        occ = occ * inside
    vis = 1.0 - occluded.float()
    occ_part = float((loss * occ).sum()) * occluded_weight
    vis_part = float((loss * vis).sum())
    total = occ_part + vis_part
    return (occ_part / total) if total > 0 else 0.0


def temporal_acceleration_loss(points: torch.Tensor, target_points: torch.Tensor,
                               target_occ: torch.Tensor, huber_delta: float = 2.0) -> torch.Tensor:
    """Anti-Jitter Loss: Enforces physical acceleration and smooth temporal continuity.

    Points shape: [B, N, T, 2] in (256, 256) grid.
    Target_points: [B, N, T, 2].
    Target_occ: [B, N, T] bool/float (1.0 = occluded).

    Computes second temporal differences (acceleration) on consecutive 3-frame windows
    and penalizes divergence from ground-truth acceleration to eliminate high-frequency chatter.
    """
    T = points.shape[2]
    if T < 3:
        return torch.tensor(0.0, device=points.device)

    # Second differences: p_{t+1} - 2*p_t + p_{t-1}
    accel_pred = points[:, :, 2:] - 2.0 * points[:, :, 1:-1] + points[:, :, :-2]
    accel_gt = target_points[:, :, 2:] - 2.0 * target_points[:, :, 1:-1] + target_points[:, :, :-2]

    # Only supervise where point was visible in all 3 frames
    vis = 1.0 - target_occ.float()
    vis3 = vis[:, :, 2:] * vis[:, :, 1:-1] * vis[:, :, :-2]

    diff = accel_pred - accel_gt
    dist = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)

    # Smooth Huber penalty on acceleration error
    huber = torch.where(dist < huber_delta, 0.5 * (dist ** 2), huber_delta * (dist - 0.5 * huber_delta))

    num_valid = torch.sum(vis3)
    if num_valid > 0:
        return torch.sum(huber * vis3) / num_valid
    return torch.tensor(0.0, device=points.device)


def surface_deformation_loss(points: torch.Tensor, target_points: torch.Tensor,
                             target_occ: torch.Tensor, max_neighbors: int = 8,
                             radius_px: float = 40.0) -> torch.Tensor:
    """Non-Rigid Surface / Mesh Deformation Consistency Loss.

    Enforces that pairs of tracks that start as spatial neighbors preserve their relative
    distances dynamically, teaching cross-track attention to maintain mesh surface geometry.

    Points shape: [B, N, T, 2] in (256, 256) grid.
    """
    B, N, T, _ = points.shape
    if N < 2:
        return torch.tensor(0.0, device=points.device)

    # Sample reference frame distances (frame 0)
    p0_gt = target_points[:, :, 0]  # [B, N, 2]
    # Pairwise distance matrix at frame 0
    diff0 = p0_gt.unsqueeze(2) - p0_gt.unsqueeze(1)  # [B, N, N, 2]
    dist0 = torch.sqrt(torch.sum(diff0 ** 2, dim=-1) + 1e-12)  # [B, N, N]

    # Neighbors mask: distance < radius_px and not self
    mask_neighbor = (dist0 < radius_px) & (dist0 > 1.0)
    if not torch.any(mask_neighbor):
        return torch.tensor(0.0, device=points.device)

    # Check pairwise visibility across frames.
    # When both are visible: weight 1.0.
    # When one is occluded and neighbor is visible: weight 0.5 (penalizes occluded point drifting away from visible anchor).
    # When both are occluded: weight 0.1.
    occ_i = target_occ.float().unsqueeze(2)  # [B, N, 1, T]
    occ_j = target_occ.float().unsqueeze(1)  # [B, 1, N, T]
    vis_weight = ((1.0 - occ_i) * (1.0 - occ_j) +
                  0.5 * ((1.0 - occ_i) * occ_j + occ_i * (1.0 - occ_j)) +
                  0.1 * (occ_i * occ_j))

    # Temporal pairwise distances: [B, N, N, T]
    diff_pred = points.unsqueeze(2) - points.unsqueeze(1)  # [B, N, N, T, 2]
    dist_pred = torch.sqrt(torch.sum(diff_pred ** 2, dim=-1) + 1e-12)

    diff_gt = target_points.unsqueeze(2) - target_points.unsqueeze(1)
    dist_gt = torch.sqrt(torch.sum(diff_gt ** 2, dim=-1) + 1e-12)

    dist_err = torch.abs(dist_pred - dist_gt)

    # Combine neighbor mask and visibility
    full_mask = mask_neighbor.unsqueeze(-1) * vis_weight  # [B, N, N, T]
    num_valid = torch.sum(full_mask)
    if num_valid > 0:
        return torch.sum(dist_err * full_mask) / num_valid
    return torch.tensor(0.0, device=points.device)


def _one_iter(points, occlusion, target_points, target_occ, shape, expected_dist,
              position_loss_weight, expected_dist_thresh, huber_delta, occluded_weight,
              occluded_in_frame_only, occ_norm=False, occ_l1=False, reacq_share=0.0,
              jitter_weight=0.0, deform_weight=0.0, use_cauchy=False, loss_quantile=1.0):
    points = convert_grid_coordinates(points, shape[3:1:-1], (256, 256),
                                      coordinate_format="xy")
    target_points = convert_grid_coordinates(target_points, shape[3:1:-1], (256, 256),
                                             coordinate_format="xy")
    if occ_norm:
        # `occluded_weight` is read as a SHARE here, not a per-sample multiplier.
        loss_huber = position_loss_normalised(
            points, target_points, target_occ, delta=huber_delta,
            occ_share=occluded_weight, occ_l1=occ_l1,
            occluded_in_frame_only=occluded_in_frame_only,
            reacq_share=reacq_share,
            use_cauchy=use_cauchy,
            loss_quantile=loss_quantile) * position_loss_weight
    else:
        loss_huber = huber_loss_weighted(points, target_points, target_occ,
                                         delta=huber_delta,
                                         occluded_weight=occluded_weight,
                                         occluded_in_frame_only=occluded_in_frame_only)
        loss_huber = torch.mean(loss_huber) * position_loss_weight

    # Anti-jitter acceleration loss
    loss_jitter = torch.tensor(0.0, device=points.device)
    if jitter_weight > 0.0:
        loss_jitter = temporal_acceleration_loss(points, target_points, target_occ) * jitter_weight

    # Non-rigid surface deformation loss
    loss_deform = torch.tensor(0.0, device=points.device)
    if deform_weight > 0.0:
        loss_deform = surface_deformation_loss(points, target_points, target_occ) * deform_weight

    loss_pos_total = loss_huber + loss_jitter + loss_deform

    if expected_dist is None:
        loss_prob = torch.tensor(0.0, device=points.device)
    else:
        loss_prob = torch.mean(prob_loss(points.detach(), expected_dist, target_points,
                                         target_occ, expected_dist_thresh,
                                         reduction_axes=None))

    loss_occ = torch.mean(F.binary_cross_entropy_with_logits(
        occlusion, target_occ.to(dtype=occlusion.dtype), reduction="none"))
    return loss_pos_total, loss_occ, loss_prob, loss_jitter, loss_deform


def tapir_loss_weighted(batch, output, position_loss_weight=0.05,
                        expected_dist_thresh=6.0, huber_delta=4.0,
                        occluded_weight=0.2, occluded_in_frame_only=True,
                        occ_norm=False, occ_l1=False, reacq_share=0.0,
                        jitter_weight=0.0, deform_weight=0.0,
                        use_cauchy=False, loss_quantile=1.0):
    """Same total as the vendor's tapir_loss, refinement iterations included.

    `occ_norm=False` is the historical path and, at occluded_weight=0.0, reproduces the
    vendor's arithmetic exactly -- that equality is the control this file is checked by and
    it must keep holding. `occ_norm=True` switches to the per-population normalisation in
    position_loss_normalised, where occluded_weight is read as a SHARE of the position
    loss rather than a per-sample multiplier.
    Optionally supports OmniMotion Cauchy loss and Quantile Trimming for outlier rejection.
    """
    shape = batch["video"].shape
    tgt_p, tgt_o = batch["target_points"], batch["occluded"]

    loss_pos, loss_occ, loss_prob, l_jit, l_def = _one_iter(
        output["tracks"], output["occlusion"], tgt_p, tgt_o, shape,
        output.get("expected_dist"), position_loss_weight, expected_dist_thresh,
        huber_delta, occluded_weight, occluded_in_frame_only, occ_norm, occ_l1, reacq_share,
        jitter_weight=jitter_weight, deform_weight=deform_weight,
        use_cauchy=use_cauchy, loss_quantile=loss_quantile)
    loss = loss_pos + loss_occ + loss_prob
    scalars = {"position_loss": loss_pos, "occlusion_loss": loss_occ,
               "prob_loss": loss_prob, "jitter_loss": l_jit, "deform_loss": l_def}

    for l in range(len(output.get("unrefined_tracks", []))):
        ed = output["unrefined_expected_dist"][l] \
            if "unrefined_expected_dist" in output else None
        h, o, p, _, _ = _one_iter(
            output["unrefined_tracks"][l], output["unrefined_occlusion"][l],
            tgt_p, tgt_o, shape, ed, position_loss_weight, expected_dist_thresh,
            huber_delta, occluded_weight, occluded_in_frame_only, occ_norm, occ_l1, reacq_share,
            jitter_weight=jitter_weight, deform_weight=deform_weight,
            use_cauchy=use_cauchy, loss_quantile=loss_quantile)
        loss = loss + h + o + p

    scalars["loss"] = loss
    return loss, scalars
