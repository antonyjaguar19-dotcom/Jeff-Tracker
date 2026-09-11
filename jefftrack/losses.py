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

CoTracker3 weights occluded points at one fifth instead of zero -- the `(1_occ/5 + 1_vis)`
term in its loss -- which is the difference between "predict where it went" and "ignore it
until it comes back". That weighting is a formula stated in the paper (arXiv 2410.11831);
this file is written from that description, and no code here derives from the CoTracker
repository. See ../LICENSES.md.

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

from model_utils import prob_loss  # noqa: E402  (vendor, unchanged)
from models.utils import convert_grid_coordinates  # noqa: E402


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


def position_loss_normalised(tracks, target_points, occluded, delta=4.0,
                             occ_share=0.2, occluded_in_frame_only=True,
                             frame_size=256.0, margin=0.0, occ_l1=False):
    """Position loss whose occluded share is a STATED fraction, not an accident.

    The defect this fixes, measured: `huber_loss_weighted` takes one mean over every
    sample, so the occluded contribution is (occluded count / total count) * weight. On
    MOVi-E that is ~11% occluded, roughly half of it in frame, times 0.2 -- about **1% of
    the position loss**. Too weak to hold a solution against the visible objective, which
    is why the occluded result oscillates between +3.4 points and zero across checkpoints
    instead of converging. Step 4000 was a good draw from that wandering, not a plateau.

    Here the two populations are averaged separately, each by its OWN count, and then
    combined:

        L = (1 - occ_share) * mean(visible) + occ_share * mean(occluded in frame)

    so `occ_share` is exactly the fraction of the position gradient that comes from
    occluded frames, whatever proportion of the clip happens to be occluded. That is the
    difference between a weight and a share, and only the second is a quantity you can
    reason about.

    Reduces to nothing: with no occluded-and-in-frame samples in the batch the occluded
    term is dropped and the visible mean carries the whole loss, rather than producing a
    nan. That case is common at small batch, which is the regime this is aimed at.

    `occ_l1` switches the OCCLUDED population from Huber to plain L1, which is what
    CoTracker3's recipe does (Huber on visible, L1 on invisible; the weights are stated in
    its training script, and this is written from that description -- see ../LICENSES.md).
    The reason it is not a detail: Huber is quadratic below `delta`, so an occluded point
    sitting 2-3 px out -- exactly the range this model loses in, measured occluded mean
    2.837 px against CoTracker3's 1.69-1.99 -- contributes only dist^2/2 and is barely
    corrected. L1 keeps a constant gradient all the way to zero, so the moderate misses
    that make up the gap are pushed on as hard as the large ones. The visible population
    keeps Huber, because there the quadratic region is doing the right thing: it is what
    stops sub-pixel noise dominating a converged track.
    """
    error = tracks - target_points
    distsqr = torch.sum(error ** 2, dim=-1)
    dist = torch.sqrt(distsqr + 1e-12)
    loss = torch.where(dist < delta, distsqr / 2, delta * (torch.abs(dist) - delta / 2))
    loss_occ_pop = dist if occ_l1 else loss

    occ = occluded.float()
    if occluded_in_frame_only:
        lo, hi = -margin, frame_size + margin
        inside = ((target_points[..., 0] >= lo) & (target_points[..., 0] < hi) &
                  (target_points[..., 1] >= lo) & (target_points[..., 1] < hi)).float()
        occ = occ * inside
    vis = 1.0 - occluded.float()

    n_vis, n_occ = vis.sum(), occ.sum()
    mean_vis = (loss * vis).sum() / torch.clamp(n_vis, min=1.0)
    if float(n_occ) <= 0.0:
        return mean_vis
    mean_occ = (loss_occ_pop * occ).sum() / n_occ
    return (1.0 - occ_share) * mean_vis + occ_share * mean_occ


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


def _one_iter(points, occlusion, target_points, target_occ, shape, expected_dist,
              position_loss_weight, expected_dist_thresh, huber_delta, occluded_weight,
              occluded_in_frame_only, occ_norm=False, occ_l1=False):
    points = convert_grid_coordinates(points, shape[3:1:-1], (256, 256),
                                      coordinate_format="xy")
    target_points = convert_grid_coordinates(target_points, shape[3:1:-1], (256, 256),
                                             coordinate_format="xy")
    if occ_norm:
        # `occluded_weight` is read as a SHARE here, not a per-sample multiplier.
        loss_huber = position_loss_normalised(
            points, target_points, target_occ, delta=huber_delta,
            occ_share=occluded_weight, occ_l1=occ_l1,
            occluded_in_frame_only=occluded_in_frame_only) * position_loss_weight
    else:
        loss_huber = huber_loss_weighted(points, target_points, target_occ,
                                         delta=huber_delta,
                                         occluded_weight=occluded_weight,
                                         occluded_in_frame_only=occluded_in_frame_only)
        loss_huber = torch.mean(loss_huber) * position_loss_weight

    if expected_dist is None:
        loss_prob = torch.tensor(0.0, device=points.device)
    else:
        loss_prob = torch.mean(prob_loss(points.detach(), expected_dist, target_points,
                                         target_occ, expected_dist_thresh,
                                         reduction_axes=None))

    loss_occ = torch.mean(F.binary_cross_entropy_with_logits(
        occlusion, target_occ.to(dtype=occlusion.dtype), reduction="none"))
    return loss_huber, loss_occ, loss_prob


def tapir_loss_weighted(batch, output, position_loss_weight=0.05,
                        expected_dist_thresh=6.0, huber_delta=4.0,
                        occluded_weight=0.2, occluded_in_frame_only=True,
                        occ_norm=False, occ_l1=False):
    """Same total as the vendor's tapir_loss, refinement iterations included.

    `occ_norm=False` is the historical path and, at occluded_weight=0.0, reproduces the
    vendor's arithmetic exactly -- that equality is the control this file is checked by and
    it must keep holding. `occ_norm=True` switches to the per-population normalisation in
    position_loss_normalised, where occluded_weight is read as a SHARE of the position
    loss rather than a per-sample multiplier.
    """
    shape = batch["video"].shape
    tgt_p, tgt_o = batch["target_points"], batch["occluded"]

    loss_huber, loss_occ, loss_prob = _one_iter(
        output["tracks"], output["occlusion"], tgt_p, tgt_o, shape,
        output.get("expected_dist"), position_loss_weight, expected_dist_thresh,
        huber_delta, occluded_weight, occluded_in_frame_only, occ_norm, occ_l1)
    loss = loss_huber + loss_occ + loss_prob
    scalars = {"position_loss": loss_huber, "occlusion_loss": loss_occ,
               "prob_loss": loss_prob}

    for l in range(len(output.get("unrefined_tracks", []))):
        ed = output["unrefined_expected_dist"][l] \
            if "unrefined_expected_dist" in output else None
        h, o, p = _one_iter(
            output["unrefined_tracks"][l], output["unrefined_occlusion"][l],
            tgt_p, tgt_o, shape, ed, position_loss_weight, expected_dist_thresh,
            huber_delta, occluded_weight, occluded_in_frame_only, occ_norm, occ_l1)
        loss = loss + h + o + p

    scalars["loss"] = loss
    return loss, scalars
