"""Fine-tune the cross-track blocks on MOVi-E. Everything else stays frozen.

The base is LocoTrack-B, and it is left exactly as published: `freeze_base()` trains only
the 5.8M cross-track parameters (33% of the model) and leaves the other 11.5M alone. This
is not modesty about compute -- though on a 16 GB A4000 it is also that -- it is what keeps
the comparison clean. The baseline and the treatment share every vendor weight bit for
bit, so a change in the numbers is attributable to the new blocks and to nothing else.
check_identity.py proves the starting point; this script is the only thing that moves it.

Loss is the vendor's arithmetic with exactly one thing changed, in jefftrack/losses.py: the
position term is weighted on occluded frames rather than zeroed. The vendor multiplies it
by `(1.0 - occluded)` (model_utils.py:15), so LocoTrack is never trained to place a point
it cannot see -- measured consequence, fine-tuning under that objective improved visible
error 1.20 -> 1.16 px median while occluded error got WORSE, 3.54 -> 3.66 px. Nothing that
optimises it can learn to cross an occlusion. `--occ-pos-weight 0.0` reproduces the vendor
objective exactly, so the two are one flag apart.

    python tools/train_cross.py \
        --steps 20000 --save-every 1000 --out weights/jefftrack_cross.ckpt

Checkpoints are written in the same shape load_jefftrack() reads, and carry the architecture
config with them, so a checkpoint can never be loaded into a differently-shaped model
without the loader noticing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jefftrack.paths import add_vendor_to_path  # noqa: E402

add_vendor_to_path()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from jefftrack.losses import tapir_loss_weighted  # noqa: E402
from jefftrack.model.jefftrack_model import load_jefftrack  # noqa: E402
from jefftrack.engine import DEFAULT_CKPT  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="fine-tune Jeff-Tracker's cross-track blocks")
    ap.add_argument("--base-ckpt", default=DEFAULT_CKPT,
                    help="LocoTrack weights to start from (stays frozen)")
    ap.add_argument("--resume", default=None, help="a previous jefftrack checkpoint")
    ap.add_argument("--out", default=os.path.join(HERE, "weights", "jefftrack_cross.ckpt"))
    ap.add_argument("--model-size", default="base", choices=["small", "base"])
    ap.add_argument("--data-dir", default="gs://kubric-public/tfds")
    ap.add_argument("--dataset", default="movi_e/256x256")
    ap.add_argument("--split", default="train")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--accum", type=int, default=4, help="gradient accumulation steps")
    ap.add_argument("--tracks", type=int, default=256)
    ap.add_argument("--res", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500,
                    help="warmup length in TRAINING STEPS, converted to optimiser updates "
                         "internally. Note the change: the scheduler steps once per update, "
                         "so the previous code treated this number as updates and warmed up "
                         "for warmup*accum steps. Runs before 2026-09-09 therefore warmed "
                         "up 4x longer than the flag implied.")
    ap.add_argument("--occ-pos-weight", type=float, default=0.2,
                    help="position-loss weight on GT-occluded frames. The vendor uses a "
                         "hard 0, which is why LocoTrack never learns to cross an "
                         "occlusion; 0.2 is CoTracker3's 1/5. Pass 0.0 to reproduce the "
                         "vendor objective exactly.")
    ap.add_argument("--occ-norm", action="store_true",
                    help="normalise the occluded position term by its OWN sample count, "
                         "so --occ-pos-weight becomes a stated SHARE of the position loss "
                         "instead of a per-sample multiplier. Without this the occluded "
                         "term is ~1%% of the position loss on MOVi-E -- an accident of "
                         "how much occlusion a clip happens to contain, and the reason "
                         "the occluded result oscillates instead of converging.")
    ap.add_argument("--unfreeze-mixer", action="store_true",
                    help="also train the vendor's TEMPORAL transformer, keeping the image "
                         "encoder frozen. Cross-track attention runs per frame and has no "
                         "temporal path of its own, so carrying a point across an "
                         "occlusion happens in the frozen blocks. Breaks the "
                         "check_identity.py attribution -- run a frozen job alongside if "
                         "that matters.")
    ap.add_argument("--lr-decay", default="cosine", choices=["cosine", "none"],
                    help="cosine decays to --lr-min-frac by the end. 'none' reproduces the "
                         "old warmup-then-flat schedule, which never lets the model settle "
                         "-- the measured symptom is results jumping between checkpoints "
                         "with no trend.")
    ap.add_argument("--lr-min-frac", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.999,
                    help="EMA decay on the trainable weights. 0 disables. The EMA copy is "
                         "saved beside the raw one; averaging is the cheap half of the fix "
                         "for a wandering solution.")
    ap.add_argument("--num-proxies", type=int, default=16)
    ap.add_argument("--cross-heads", type=int, default=4)
    ap.add_argument("--pyramid-level", type=int, default=0,
                    help="Route 3: extra correlation levels. CoTracker3 runs FOUR (1024 "
                         "corr dims); LocoTrack ships three (768). The vendor accepts this "
                         "argument but cannot run with it -- it builds the extra grid and "
                         "then drops it, because cmdtop is range(3) and `supports` is never "
                         "extended. Both are fixed in jefftrack_model.py. New channels "
                         "enter at zero weight, so this starts as the 3-level model to "
                         "within float noise (~0.2 px) and has to be TRAINED to pay.")
    ap.add_argument("--occ-l1", action="store_true",
                    help="Route 2: plain L1 for the OCCLUDED population instead of Huber, "
                         "which is what CoTracker3's recipe uses. Huber is quadratic below "
                         "--huber-delta, so a hidden point 2-3 px out -- the range this "
                         "model actually loses in -- is barely corrected. Visible keeps "
                         "Huber. Only meaningful with --occ-norm.")
    ap.add_argument("--carry-state", action="store_true",
                    help="carry the proxy tokens through the vendor's TEMPORAL attention "
                         "between cross blocks, instead of rebuilding them per frame from "
                         "a static parameter. Route 1: measured, widening how many tracks "
                         "a static block sees does nothing (3.186 / 3.185 / 3.184 px "
                         "occluded at chunk 64 / 256 / 600), because a per-frame summary "
                         "cannot hold where the group has been heading. This gives the "
                         "proxies a memory. Still bit-identical at init -- "
                         "check_identity.py --carry-state.")
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--bf16", action="store_true", default=True)
    ap.add_argument("--fp32", dest="bf16", action="store_false")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("[ERROR] training needs a GPU")

    model = load_jefftrack(a.resume or a.base_ckpt, model_size=a.model_size,
                         num_proxies=a.num_proxies, cross_heads=a.cross_heads,
                         carry_state=a.carry_state, pyramid_level=a.pyramid_level,
                         zero_init=a.resume is None, device=device)
    model.freeze_base(unfreeze_mixer=a.unfreeze_mixer)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    n_mixer = sum(p.numel() for p in model.mixer_parameters() if p.requires_grad)
    print("[model] {} trainable of {} ({:.1f}%)   zero-init={}".format(
        n_train, n_all, 100.0 * n_train / n_all, a.resume is None))
    # Report each bucket from its OWN parameters, not by subtraction. With a widened
    # pyramid the extra CMDTop levels are trainable too, and a subtracted "cross" figure
    # silently absorbs them -- the log then claims 168k more cross-track parameters than
    # exist.
    n_cross = sum(p.numel() for p in model.cross_parameters() if p.requires_grad)
    n_cmdtop = sum(p.numel() for i in range(3, len(model.cmdtop))
                   for p in model.cmdtop[i].parameters() if p.requires_grad)
    print("[model] cross {}   temporal mixer {}   extra cmdtop {}   image encoder frozen"
          .format(n_cross, n_mixer if a.unfreeze_mixer else "FROZEN", n_cmdtop))
    leftover = n_train - n_cross - n_mixer - n_cmdtop
    if leftover:
        print("[model] WARNING {} trainable parameters in neither bucket".format(leftover))
    print("[loss]  occluded position {} {} ({})".format(
        "SHARE" if a.occ_norm else "weight", a.occ_pos_weight,
        "vendor objective: occluded frames get NO position gradient"
        if a.occ_pos_weight == 0 else
        ("a stated fraction of the position loss" if a.occ_norm
         else "per-sample multiplier -- actual share depends on how much occlusion the "
              "clip contains")))

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)

    # sched.step() runs once per OPTIMISER step, not once per sample, so the horizon is
    # steps//accum. Getting that wrong decays over the wrong length and looks like a
    # learning-rate bug much later.
    total_updates = max(1, a.steps // max(1, a.accum))
    warmup_updates = max(1, a.warmup // max(1, a.accum))

    def _lr(u):
        if u < warmup_updates:
            return (u + 1) / warmup_updates
        if a.lr_decay == "none":
            return 1.0
        prog = (u - warmup_updates) / max(1, total_updates - warmup_updates)
        prog = min(1.0, max(0.0, prog))
        return a.lr_min_frac + (1.0 - a.lr_min_frac) * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr)
    print("[sched] {} decay over {} updates ({} steps / accum {}), warmup {}".format(
        a.lr_decay, total_updates, a.steps, a.accum, warmup_updates))

    # EMA over the trainable weights only -- the frozen ones never move, so shadowing them
    # would just double the memory for a copy of themselves.
    ema = None
    if a.ema and a.ema > 0:
        ema = {n: p.detach().clone() for n, p in model.named_parameters()
               if p.requires_grad}
        print("[ema]   decay {} over {} tensors".format(a.ema, len(ema)))

    # Imported here, not at module scope: tensorflow is a heavy import that only the
    # training path needs, and a missing pydeps install should fail with the message in
    # movi._import_tf rather than at the top of this file.
    from jefftrack.data.movi import batches  # noqa: E402

    print("[data] {} {} from {}".format(a.dataset, a.split, a.data_dir))
    stream = batches(device=device, data_dir=a.data_dir, name=a.dataset, split=a.split,
                     train_size=(a.res, a.res), batch_size=1, tracks_to_sample=a.tracks)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    cfg = {"model_size": a.model_size, "num_proxies": a.num_proxies,
           "cross_heads": a.cross_heads, "cross_layers": None,
           "carry_state": a.carry_state, "pyramid_level": a.pyramid_level,
           "base_ckpt": os.path.basename(a.base_ckpt), "res": a.res,
           "tracks": a.tracks, "dataset": a.dataset, "data_dir": a.data_dir,
           "occ_pos_weight": a.occ_pos_weight, "occ_norm": bool(a.occ_norm),
           "occ_l1": bool(a.occ_l1),
           "unfreeze_mixer": bool(a.unfreeze_mixer), "lr_decay": a.lr_decay,
           "ema": a.ema, "accum": a.accum, "lr": a.lr}

    def save(step: int, running: float):
        blob = {"state_dict": model.state_dict(), "jefftrack": cfg,
                "step": step, "loss": running}
        torch.save(blob, a.out)
        # Also keep a step-tagged copy. Writing only to --out overwrites the previous save,
        # so a finished run left exactly one checkpoint and the across-checkpoint stability
        # question -- the one Tier 1 is actually asking, since the occluded result has
        # oscillated between +3.4 and 0 before -- could not be answered at all.
        torch.save(blob, "{}_s{}.ckpt".format(os.path.splitext(a.out)[0], step))
        if ema is not None:
            # The EMA copy is a full, loadable checkpoint -- frozen weights from the live
            # model, trainable ones from the shadow -- so it can be scored by exactly the
            # same tooling with no special case.
            sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
            sd.update({k: v.clone() for k, v in ema.items()})
            torch.save({"state_dict": sd, "jefftrack": cfg, "step": step,
                        "loss": running, "ema": a.ema},
                       os.path.splitext(a.out)[0] + "_ema.ckpt")
        with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
            json.dump({"step": step, "loss": running, **cfg}, fh, indent=2)

    t0 = time.time()
    running, seen = 0.0, 0
    gn_sum, gn_n, gn_clipped = 0.0, 0, 0
    opt.zero_grad(set_to_none=True)
    for step in range(1, a.steps + 1):
        batch = next(stream)
        ctx = (torch.autocast("cuda", dtype=torch.bfloat16) if a.bf16
               else torch.autocast("cuda", enabled=False))
        with ctx:
            out = model(batch["video"], batch["query_points"],
                        is_training=True,
                        # The block can only attend across tracks that share a chunk, so
                        # the chunk has to hold every track or the training signal is
                        # quietly narrower than the architecture claims.
                        query_chunk_size=a.tracks)
            loss, _ = tapir_loss_weighted(batch, out,
                                          occluded_weight=a.occ_pos_weight,
                                          occ_norm=a.occ_norm, occ_l1=a.occ_l1)
        (loss / a.accum).backward()

        running += float(loss.detach())
        seen += 1
        if step % a.accum == 0:
            # clip_grad_norm_ returns the norm BEFORE clipping. Worth logging: raising the
            # occluded share raises the gradient, and if that just saturates the clip then
            # the visible term's effective step shrinks instead -- the change would look
            # applied and be partly cancelled. A clip rate near 100% means the ceiling,
            # not the loss, is deciding the step.
            gnorm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
            gn_sum += gnorm
            gn_n += 1
            gn_clipped += int(gnorm > 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            if ema is not None:
                with torch.no_grad():
                    for n_, p_ in model.named_parameters():
                        if n_ in ema:
                            ema[n_].mul_(a.ema).add_(p_.detach(), alpha=1.0 - a.ema)

        if step % a.log_every == 0:
            el = time.time() - t0
            print("[{:>6}/{}] loss {:.4f}  lr {:.2e}  gnorm {:.2f} clip {:.0f}%  "
                  "{:.2f}s/step  vram {:.1f}GB  eta {:.1f}h".format(
                      step, a.steps, running / max(1, seen), sched.get_last_lr()[0],
                      gn_sum / max(1, gn_n), 100.0 * gn_clipped / max(1, gn_n),
                      el / step, torch.cuda.max_memory_allocated() / 1e9,
                      el / step * (a.steps - step) / 3600.0), flush=True)
            running, seen = 0.0, 0
            gn_sum, gn_n, gn_clipped = 0.0, 0, 0
        if step % a.save_every == 0 or step == a.steps:
            save(step, running / max(1, seen))
            print("[save] {} at step {}".format(a.out, step), flush=True)

    print("[done] {:.1f}h".format((time.time() - t0) / 3600.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
