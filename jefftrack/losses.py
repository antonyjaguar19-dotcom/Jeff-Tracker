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
that the TAPNext path lacks (measured AUC 0.955, see METHOD.md).
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


def _one_iter(points, occlusion, target_points, target_occ, shape, expected_dist,
              position_loss_weight, expected_dist_thresh, huber_delta, occluded_weight,
              occluded_in_frame_only):
    points = convert_grid_coordinates(points, shape[3:1:-1], (256, 256),
                                      coordinate_format="xy")
    target_points = convert_grid_coordinates(target_points, shape[3:1:-1], (256, 256),
                                             coordinate_format="xy")
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
                        occluded_weight=0.2, occluded_in_frame_only=True):
    """Same total as the vendor's tapir_loss, refinement iterations included."""
    shape = batch["video"].shape
    tgt_p, tgt_o = batch["target_points"], batch["occluded"]

    loss_huber, loss_occ, loss_prob = _one_iter(
        output["tracks"], output["occlusion"], tgt_p, tgt_o, shape,
        output.get("expected_dist"), position_loss_weight, expected_dist_thresh,
        huber_delta, occluded_weight, occluded_in_frame_only)
    loss = loss_huber + loss_occ + loss_prob
    scalars = {"position_loss": loss_huber, "occlusion_loss": loss_occ,
               "prob_loss": loss_prob}

    for l in range(len(output.get("unrefined_tracks", []))):
        ed = output["unrefined_expected_dist"][l] \
            if "unrefined_expected_dist" in output else None
        h, o, p = _one_iter(
            output["unrefined_tracks"][l], output["unrefined_occlusion"][l],
            tgt_p, tgt_o, shape, ed, position_loss_weight, expected_dist_thresh,
            huber_delta, occluded_weight, occluded_in_frame_only)
        loss = loss + h + o + p

    scalars["loss"] = loss
    return loss, scalars
