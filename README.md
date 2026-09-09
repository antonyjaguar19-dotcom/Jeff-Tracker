# DBtracker

A point tracker in the CoTracker class, built so it can ship commercially.

Every piece of it is **Apache-2.0**: the model code, the base weights, and the data it was
trained on. That is the whole reason the project exists. The strongest open point trackers
— CoTracker, CoTracker3, MFT, SpatialTracker — are CC-BY-NC on code *and* weights, and
NonCommercial restricts **use**, not just redistribution, so "we only run it in-house"
does not make them safe in a commercial pipeline. DBtracker is what is left when you refuse
to take that risk and still want the capability.

**Foundation:** [LocoTrack-B](https://github.com/cvlab-kaist/locotrack) (ECCV 2024,
Apache-2.0), plus cross-track attention written from the CoTracker3 paper
([arXiv:2410.11831](https://arxiv.org/abs/2410.11831)) and fine-tuned on MOVi-E. No code,
weights, or tensors here derive from the CoTracker repository — architecture is not
copyrightable, source code is. Full provenance in [LICENSES.md](LICENSES.md), every
measurement and every negative result in [METHOD.md](METHOD.md).

## Results

| | DBtracker |
|---|---|
| Localisation, exact synthetic ground truth | **1.03 px** at 256×256, **0.54 px** at 384×680 |
| Pixel locking | **+0.0005 px** — none, on an NCC-free path |
| TAP-Vid DAVIS strided (AJ / δ_avg / OA) | **67.7 / 79.5 / 89.8** |
| Confidence as a bad-frame detector | **AUC 0.955** at 256×256 |
| Speed | **0.030 s/frame**, 312 frames of 4K, 6.85 GB peak |
| Occluded within 5 px, vs the LocoTrack base | **+3.4 points**, held out across three benches |
| Visible accuracy vs base | **−0.03 px** at every checkpoint |

The DAVIS row is the one that matters most: LocoTrack-B as published is 67.8 / 79.6 / 89.9,
so this port is within **0.1 on all three metrics**. Everything else is only believable
because that is true.

### What it does not do

**It gaps an occlusion rather than crossing one.** Only ~6% of ground-truth-occluded
frames get a position at all. Cross-track attention improves the frames it does emit, but
the vendor loss zeroes the position term on occluded frames, so the base model was never
taught to place a point it cannot see. [METHOD.md](METHOD.md) has the arithmetic on why the
occluded gain oscillates between +3.4 and 0 across checkpoints, and what the next run
should change. Do not read +3.4 as a level.

**It is a seed-and-track stage, not a finished pipeline.** The 1.03 px figure is raw
neural output over every seed with no refinement. A classical sub-pixel refinement pass
downstream (NCC + affine, or similar) still buys most of an order of magnitude.

## Install

```bash
git clone https://github.com/antonyjaguar19-dotcom/DBtracker
cd DBtracker
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
python fetch_weights.py --all
```

LocoTrack is vendored under `vendor/locotrack/` (Apache-2.0, LICENSE and PROVENANCE
retained), so there is nothing else to clone. Weights live on the Hugging Face Hub rather
than in git — the same arrangement LocoTrack and CoTracker both use.

> **Checkpoint hosting is not live yet.** `fetch_weights.py --baseline-only` works today
> and pulls the Apache-2.0 LocoTrack weights from the authors' own Hub repo. The DBtracker
> checkpoint needs uploading before the plain `fetch_weights.py` path resolves; the script
> prints the exact `hf upload` command if it 404s.

## Use

```python
import torch
tracker = torch.hub.load("antonyjaguar19-dotcom/DBtracker", "dbtracker").cuda()

# frames: (T, H, W, 3) uint8 BGR      queries: (N, 3) as [frame, x, y]
tracks, vis, conf = tracker.track_queries_conf(frames, queries)
# tracks (T, N, 2) xy in the input frames' pixel space, y DOWN
# vis    (T, N) bool   -- conf >= 0.5, the vendor's own threshold
# conf   (T, N) float  -- 1 - P(occluded or uncertain), continuous
```

Or on a directory of frames, straight to 3DE-style 2D-track ASCII plus an overlay mp4:

```bash
python run_dbtrack.py --plate /path/to/frames --name myshot --seed corners --points 600
```

**Keep `conf`.** It is the reason to prefer this model over a TAPNext-style path, which
thresholds visibility logits at zero and throws the continuous signal away. At 256×256 it
detects the model's own >5 px frames at AUC 0.955, and gating at 0.5 capped the worst
sample in a whole run at 9 px.

### The two resolution knobs are not the same knob

- `--work-width` — what frames are decoded and held at. Saves host RAM. The model never
  sees it.
- `--model-res` — what the model actually runs at, and therefore what bounds precision. It
  was trained at 256×256. On a 2560-wide plate one model pixel at 256×256 is **ten plate
  pixels**, which is where the ~1 px synthetic floor comes from.

Raising `--model-res` lowers the median and raises the tail. At 384×680 the median visible
error is better (0.697 px vs 1.185) while the mean is 40.6 px, and **no confidence
threshold recovers it** — at conf ≥ 0.99 the mean is still 39.3 px. Confident and wrong is
the worst failure mode there is. Use 256×256 unless you have measured otherwise on your
own footage.

## Verify it before believing it

Nothing in METHOD.md is quoted without a control pass that returns a number which *should*
be zero. Both metric defects found while building this were metrics measuring the input
instead of the tracker, and each was caught exactly this way.

```bash
python dbtrack_engine.py --selftest      # rigid translation, exact GT -> 0.100 px
python check_identity.py                 # untrained DBtrack IS LocoTrack, 0.000e+00
python score_occlusion.py --control      # scorer against truth -> 0.00000 px
python eval_tapvid.py --mode strided     # the port, against published numbers
```

`check_identity.py` is the load-bearing one. With cross-track attention zero-initialised
the model is LocoTrack **bit for bit**, so any change measured after training is
attributable to the new blocks and not to the port having moved something. It asserts
`max |diff| == 0.000e+00` on tracks, occlusion and expected_dist.

## Train

```bash
python probe_data.py --batches 2         # data control pass -- run this first
python train_cross.py --occ-pos-weight 0.2 --steps 4000
python score_ckpt.py --ckpt weights/step4000.ckpt --tag s4000
```

`freeze_base()` trains the 5.8M cross-track parameters and leaves the 11.5M vendor weights
bit-identical. `--occ-pos-weight` is the flag the whole Stage-3 result turns on: `0.0` is
the vendor's objective, which zeroes the position loss on occluded frames and therefore
cannot teach a model to cross an occlusion no matter what attention you bolt on. `0.2` is
CoTracker3's `(𝟙_occ/5 + 𝟙_vis)` weighting, implemented from the paper's formula.

~2 s/step at 2.3 GB on an RTX A4000.

## Licence

Apache-2.0. See [LICENSE](LICENSE), [NOTICE](NOTICE), and [LICENSES.md](LICENSES.md) for
the artifact-by-artifact ledger including the one open item — the base LocoTrack weights
rest on the LocoTrack authors' own Apache-2.0 declaration, which is theirs to make.

## Citing

DBtracker is a derivative of LocoTrack and reimplements an idea from CoTracker3. Cite both:

```bibtex
@inproceedings{cho2024locotrack,
  title     = {Local All-Pair Correspondence for Point Tracking},
  author    = {Cho, Seokju and Huang, Jiahui and Nam, Seungryong and
               Min, Dongbo and Lee, Joon-Young},
  booktitle = {ECCV},
  year      = {2024}
}
@inproceedings{karaev2024cotracker3,
  title     = {CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling
               Real Videos},
  author    = {Karaev, Nikita and Makarov, Iurii and Wang, Jianyuan and
               Rocco, Ignacio and Graham, Benjamin and Neverova, Natalia and
               Vedaldi, Andrea and Rupprecht, Christian},
  booktitle = {arXiv:2410.11831},
  year      = {2024}
}
```
