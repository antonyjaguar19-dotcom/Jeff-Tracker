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

    python train_cross.py ^
        --steps 20000 --save-every 1000 --out weights\\jefftrack_cross.ckpt

Checkpoints are written in the same shape load_jefftrack() reads, and carry the architecture
config with them, so a checkpoint can never be loaded into a differently-shaped model
without the loader noticing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

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
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--occ-pos-weight", type=float, default=0.2,
                    help="position-loss weight on GT-occluded frames. The vendor uses a "
                         "hard 0, which is why LocoTrack never learns to cross an "
                         "occlusion; 0.2 weights an occluded point at one fifth of a "
                         "visible one. Pass 0.0 to reproduce the "
                         "vendor objective exactly.")
    ap.add_argument("--num-proxies", type=int, default=16)
    ap.add_argument("--cross-heads", type=int, default=4)
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
                         zero_init=a.resume is None, device=device)
    model.freeze_base()
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    n_all = sum(p.numel() for p in model.parameters())
    print("[model] {} trainable of {} ({:.1f}%)   base frozen, zero-init={}".format(
        n_train, n_all, 100.0 * n_train / n_all, a.resume is None))
    print("[loss]  occluded position weight {} ({})".format(
        a.occ_pos_weight,
        "vendor objective: occluded frames get NO position gradient"
        if a.occ_pos_weight == 0 else "occluded frames are supervised"))

    opt = torch.optim.AdamW(trainable, lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, a.warmup)))

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
           "base_ckpt": os.path.basename(a.base_ckpt), "res": a.res,
           "tracks": a.tracks, "dataset": a.dataset, "data_dir": a.data_dir,
           "occ_pos_weight": a.occ_pos_weight}

    def save(step: int, running: float):
        blob = {"state_dict": model.state_dict(), "jefftrack": cfg,
                "step": step, "loss": running}
        torch.save(blob, a.out)
        with open(os.path.splitext(a.out)[0] + ".json", "w") as fh:
            json.dump({"step": step, "loss": running, **cfg}, fh, indent=2)

    t0 = time.time()
    running, seen = 0.0, 0
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
                                          occluded_weight=a.occ_pos_weight)
        (loss / a.accum).backward()

        running += float(loss.detach())
        seen += 1
        if step % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)

        if step % a.log_every == 0:
            el = time.time() - t0
            print("[{:>6}/{}] loss {:.4f}  lr {:.2e}  {:.2f}s/step  vram {:.1f}GB  "
                  "eta {:.1f}h".format(
                      step, a.steps, running / max(1, seen), sched.get_last_lr()[0],
                      el / step, torch.cuda.max_memory_allocated() / 1e9,
                      el / step * (a.steps - step) / 3600.0), flush=True)
            running, seen = 0.0, 0
        if step % a.save_every == 0 or step == a.steps:
            save(step, running / max(1, seen))
            print("[save] {} at step {}".format(a.out, step), flush=True)

    print("[done] {:.1f}h".format((time.time() - t0) / 3600.0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
