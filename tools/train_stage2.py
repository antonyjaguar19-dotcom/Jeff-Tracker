"""Stage 2: distil the pseudo-labels from make_pseudo_labels.py into the student.

Companion to `train_cross.py`, which is stage 1 on Kubric with exact ground truth. This is
the second stage of CoTracker3's recipe, written from its description (LICENSES.md), run on
our own plates with commercially-clean teachers.

**What this stage does NOT supervise, and why that is the whole point.** CoTracker3's
real-data loss carries the position terms only:

    coords, visible      0.05      Huber
    coords, invisible    0.01      plain L1, and dropped entirely for one of its teachers
    visibility BCE       ABSENT
    confidence BCE       ABSENT

The vis and conf heads are frozen by omission -- they keep whatever stage 1 taught them.
That is not an oversight to be tidied up. A pseudo-label's *position* is a teacher's best
estimate and is worth regressing toward; a pseudo-label's *visibility* is a teacher's
opinion about its own reliability, and training a student to reproduce it teaches the
student to be confident exactly where the teacher was confident, including where the
teacher was confidently wrong. Our confidence head is the one signal this model has that the
TAPNext path lacks (measured AUC 0.955); corrupting it to chase a stage-2 number would be a
bad trade even if it worked.

Read this before expecting it to close the occlusion gap: **it will not, and it is not
supposed to.** The occluded-position term here is weighted 0.01 against 0.05 visible, the
teachers are least reliable exactly where a point is hidden, and the closure gate in
make_pseudo_labels.py drops the tracks whose hidden stretches it cannot verify. Stage 2 buys
GENERALISATION to real footage -- grain, motion blur, defocus, real camera motion, the
things Kubric does not have. The occlusion gap is stage 1's problem, and `--occ-l1` and
`--carry-state` in train_cross.py are what aim at it.

    python tools/train_stage2.py \
        --data data\\pseudo\\v1 --resume weights/c3_occnorm.ckpt \
        --out weights/s2_plates.ckpt --steps 8000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)          # the repo root, one level up
for _p in (HERE, REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.model.jefftrack_model import load_jefftrack  # noqa: E402
from jefftrack.paths import add_vendor_to_path  # noqa: E402
from run_jefftrack import read_frame  # noqa: E402

add_vendor_to_path()

from models.utils import convert_grid_coordinates  # noqa: E402


def load_sample(path, model_res):
    """One shard -> (video (1,T,h,w,3) float, queries (1,N,3) tyx, tracks, visible).

    The shard stores a WINDOW SPEC, not pixels, so the frames are decoded here. Coordinates
    are in the work-resolution space the teacher ran at and are rescaled to the model
    resolution, exactly as the engine does at inference -- a mismatch here would train the
    student against targets in the wrong space and still converge, to the wrong thing.
    """
    d = np.load(path, allow_pickle=True)
    plate = str(d["plate"])
    files = [os.path.join(plate, str(f)) for f in d["frame_files"]]
    work_w = int(d["work_width"])
    imgs = np.stack([read_frame(f, work_w)[0] for f in files])
    T, Hw, Ww = imgs.shape[:3]

    h, w = model_res
    vid = np.empty((T, h, w, 3), np.uint8)
    import cv2  # noqa: PLC0415
    for t in range(T):
        vid[t] = cv2.resize(imgs[t], (w, h), interpolation=cv2.INTER_AREA)[:, :, ::-1]

    sx, sy = w / float(Ww), h / float(Hw)
    q = d["queries"].astype(np.float32).copy()
    tr = d["tracks"].astype(np.float32).copy()
    q[:, 0] *= sx
    q[:, 1] *= sy
    tr[..., 0] *= sx
    tr[..., 1] *= sy

    video = torch.from_numpy(vid)[None].float() / 255.0 * 2 - 1
    # Engine convention: queries are (t, y, x) in model pixels.
    qq = np.stack([np.zeros(len(q), np.float32), q[:, 1], q[:, 0]], 1)
    # The shard stores tracks TIME-major, (T, N, 2), because that is what the engine hands
    # back; the model emits POINT-major, (B, N, T, 2). Transpose here rather than in the
    # loss, so the loss only ever sees one convention -- broadcasting a (T,N) against an
    # (N,T) silently succeeds whenever T == N and trains against a transposed target.
    tr = np.transpose(tr, (1, 0, 2))
    visible = np.transpose(np.asarray(d["visible"]), (1, 0))
    return (video, torch.from_numpy(qq)[None].float(),
            torch.from_numpy(np.ascontiguousarray(tr))[None].float(),
            torch.from_numpy(np.ascontiguousarray(visible))[None].bool(),
            str(d["teacher"]))


def position_loss(pred, target, visible, shape, huber_delta=6.0,
                  w_vis=0.05, w_occ=0.01):
    """CoTracker3's real-data position loss: Huber where visible, L1 where not.

    The 5:1 split is theirs and is deliberate -- occluded positions are supervised, but
    weakly, so a hallucinated position can never dominate the gradient on a label that was
    itself a guess.
    """
    pred = convert_grid_coordinates(pred, shape[3:1:-1], (256, 256), coordinate_format="xy")
    target = convert_grid_coordinates(target, shape[3:1:-1], (256, 256),
                                      coordinate_format="xy")
    err = pred - target
    dist = torch.sqrt(torch.sum(err ** 2, dim=-1) + 1e-12)
    hub = torch.where(dist < huber_delta, dist ** 2 / 2,
                      huber_delta * (dist - huber_delta / 2))
    vis = visible.float()
    n_v, n_o = vis.sum(), (1.0 - vis).sum()
    out = pred.new_zeros(())
    if float(n_v) > 0:
        out = out + w_vis * (hub * vis).sum() / n_v
    if float(n_o) > 0:
        out = out + w_occ * (dist * (1.0 - vis)).sum() / n_o
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="stage 2: distil pseudo-labelled plates")
    ap.add_argument("--data", required=True, help="a make_pseudo_labels.py output dir")
    ap.add_argument("--resume", required=True, help="the stage-1 checkpoint to start from")
    ap.add_argument("--out", default=os.path.join(HERE, "weights", "stage2.ckpt"))
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--accum", type=int, default=8)
    # CoTracker3 drops the learning rate by 10x for stage 2 and restores from the stage-1
    # baseline. Distillation on labels that are themselves estimates is a refinement of a
    # converged model, not a second training run, and a stage-1 learning rate here walks a
    # good model into the teachers' error distribution.
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--huber-delta", type=float, default=6.0)
    ap.add_argument("--w-vis", type=float, default=0.05)
    ap.add_argument("--w-occ", type=float, default=0.01)
    ap.add_argument("--freeze-base", action="store_true",
                    help="train only the cross-track blocks")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(a.data, "s*.npz")))
    if not shards:
        raise SystemExit("[ERROR] no shards in {} -- run make_pseudo_labels.py".format(
            a.data))
    idx_path = os.path.join(a.data, "index.json")
    idx = json.load(open(idx_path)) if os.path.isfile(idx_path) else {}
    print("[stage2] {} shards from {}  teachers={}".format(
        len(shards), a.data, idx.get("teachers", "?")))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # The architecture travels with the weights, and it has to travel THROUGH this stage as
    # well: a stage-2 checkpoint that records only "stage 2" reloads as a default-shaped
    # model, which for a carry_state or 4-level checkpoint is either a shape error or, worse,
    # proxies landing in a block of the wrong width. Read the stage-1 block and carry it.
    blob = torch.load(a.resume, map_location="cpu", weights_only=False)
    arch = {}
    if isinstance(blob, dict):
        arch = blob.get("jefftrack") or blob.get("dbtrack") or {}
    del blob
    model = load_jefftrack(a.resume, model_size=a.model_size, device=device)
    model.train()
    if a.freeze_base:
        model.freeze_base(unfreeze_mixer=False)
    params = [p for p in model.parameters() if p.requires_grad]
    print("[stage2] training {} tensors ({:.1f}M params)".format(
        len(params), sum(p.numel() for p in params) / 1e6))

    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)
    warm = torch.optim.lr_scheduler.LinearLR(
        opt, start_factor=0.1, total_iters=max(1, a.warmup // max(1, a.accum)))
    rng = np.random.default_rng(a.seed)
    model_res = (a.res, a.res)

    running, seen, t0 = 0.0, 0, time.time()
    opt.zero_grad(set_to_none=True)
    for step in range(1, a.steps + 1):
        shard = shards[int(rng.integers(len(shards)))]
        try:
            video, q, tgt, vis, teacher = load_sample(shard, model_res)
        except Exception as exc:                                  # noqa: BLE001
            print("[stage2] skip {}: {!r}".format(os.path.basename(shard), exc))
            continue
        video, q = video.to(device), q.to(device)
        tgt, vis = tgt.to(device), vis.to(device)

        out = model(video, q, query_chunk_size=max(64, tgt.shape[2]))
        loss = position_loss(out["tracks"], tgt, vis, video.shape,
                             a.huber_delta, a.w_vis, a.w_occ)
        # Every refinement iteration is supervised, as the vendor's own loss does -- the
        # unrefined outputs are what the later iterations are correcting, and leaving them
        # unsupervised trains only the last step of a four-step process.
        for l in range(len(out.get("unrefined_tracks", []))):
            loss = loss + position_loss(out["unrefined_tracks"][l], tgt, vis, video.shape,
                                        a.huber_delta, a.w_vis, a.w_occ)

        (loss / a.accum).backward()
        running += float(loss.detach())
        seen += 1
        if step % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            warm.step()
        if step % a.log_every == 0:
            el = time.time() - t0
            print("[stage2] {}/{}  loss {:.4f}  lr {:.2e}  {:.1f}s  eta {:.1f}h".format(
                step, a.steps, running / max(1, seen), warm.get_last_lr()[0], el,
                el / step * (a.steps - step) / 3600.0), flush=True)
            running, seen = 0.0, 0
        if step % a.save_every == 0 or step == a.steps:
            cfg = dict(arch)
            cfg.update({"model_size": a.model_size, "stage": 2, "data": a.data,
                        "resume": a.resume, "lr": a.lr, "w_vis": a.w_vis,
                        "w_occ": a.w_occ, "huber_delta": a.huber_delta,
                        "teachers": idx.get("teachers"),
                        "vis_conf_supervised": False})
            os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "jefftrack": cfg, "step": step},
                       a.out)
            print("[stage2] saved {} at step {}".format(a.out, step), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
