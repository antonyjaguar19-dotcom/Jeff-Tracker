<div align="center">

# DBtracker

**A point tracker in the CoTracker class, licensed so it can actually ship.**

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Weights](https://img.shields.io/badge/%F0%9F%A4%97%20weights-Hugging%20Face-yellow)](https://huggingface.co/antonyjaguar19-dotcom/DBtracker)
[![Base](https://img.shields.io/badge/base-LocoTrack--B-green.svg)](https://github.com/cvlab-kaist/locotrack)
[![DAVIS](https://img.shields.io/badge/TAP--Vid%20DAVIS-67.7%20AJ-orange.svg)](docs/METHOD.md)

<img src="assets/demo_grid.gif" width="90%" alt="DBtracker tracking a dense grid through a DAVIS clip">

*A 14×14 grid carried through `breakdance`. [Footage: DAVIS, CC BY 4.0](assets/README.md).*

</div>

---

Every part of DBtracker is **Apache-2.0** — model code, base weights, and training data.
That is the entire reason it exists. The strongest open point trackers (CoTracker,
CoTracker3, MFT, SpatialTracker) are CC-BY-NC on code *and* weights, and NonCommercial
restricts **use**, not just redistribution, so "we only run it in-house" does not make them
safe in a commercial pipeline.

**Foundation:** [LocoTrack-B](https://github.com/cvlab-kaist/locotrack) (ECCV 2024,
Apache-2.0), plus cross-track attention written from the CoTracker3 paper
([arXiv:2410.11831](https://arxiv.org/abs/2410.11831)) and fine-tuned on Apache-2.0 MOVi-E.
No code, weights or tensors here derive from the CoTracker repository — architecture is not
copyrightable, source code is. Ledger: [docs/LICENSES.md](docs/LICENSES.md).

## Results

| | DBtracker |
|---|---|
| Localisation, exact synthetic ground truth | **1.03 px** @ 256×256 · **0.54 px** @ 384×680 |
| Pixel locking | **+0.0005 px** — none, on an NCC-free path |
| TAP-Vid DAVIS strided (AJ / δ_avg / OA) | **67.7 / 79.5 / 89.8** |
| Confidence as a bad-frame detector | **AUC 0.955** @ 256×256 |
| Speed | **0.030 s/frame** · 312 frames of 4K · 6.85 GB peak |
| Occluded within 5 px, vs the LocoTrack base | **+3.4 points**, held out over three benches |
| Visible accuracy vs base | **−0.03 px** at every checkpoint |

The DAVIS row is the load-bearing one. LocoTrack-B **as published** is 67.8 / 79.6 / 89.9,
so this port is within **0.1 on all three metrics** — which is what makes every other number
here worth reading. Full measurements, controls and negative results:
**[docs/METHOD.md](docs/METHOD.md)**.

## Two things it does not do

<table>
<tr>
<td width="50%"><img src="assets/demo_confidence.gif" alt="points coloured by the model's own confidence"></td>
<td width="50%"><img src="assets/demo_occlusion.gif" alt="hollow rings where the model declines to place a point"></td>
</tr>
<tr>
<td><b>It tells you when it is unsure.</b> Points coloured by the model's own confidence,
green through red. That signal detects its own &gt;5 px frames at AUC 0.955 and costs
nothing — it is already computed. Keep it.</td>
<td><b>It gaps an occlusion instead of crossing one.</b> Hollow rings are points held at
their last committed position — only ~6% of ground-truth-occluded frames get a position at
all, and after 8 frames the track retires. The limitation, drawn rather than described.</td>
</tr>
</table>

**Use 256×256 unless you have measured otherwise.** Raising the model resolution lowers the
median and raises the tail. At 384×680 the median visible error improves to 0.697 px while
the *mean* is 40.6 px — and **no confidence threshold recovers it**: at conf ≥ 0.99 the mean
is still 39.3 px. Confident and wrong is the worst failure mode there is.

**It is a seed-and-track stage, not a finished pipeline.** The 1.03 px figure is raw neural
output over every seed with no refinement. A classical sub-pixel pass downstream still buys
most of an order of magnitude.

## Install

```bash
git clone https://github.com/antonyjaguar19-dotcom/DBtracker
cd DBtracker
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match your CUDA
pip install -r requirements.txt
python tools/fetch_weights.py --all
```

LocoTrack is vendored under `vendor/locotrack/` (Apache-2.0, LICENSE and PROVENANCE
retained), so there is nothing else to clone. Weights live on the Hugging Face Hub rather
than in git — the arrangement LocoTrack and CoTracker both use.

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

On a directory of frames, straight to 3DE-style 2D-track ASCII plus an overlay mp4:

```bash
python tools/run_dbtrack.py --plate /path/to/frames --name myshot --seed corners --points 600
```

### The two resolution knobs are not the same knob

- `--work-width` — what frames are decoded and held at. Saves host RAM. The model never
  sees it.
- `--model-res` — what the model actually runs at, and therefore what bounds precision. It
  was trained at 256×256. On a 2560-wide plate one model pixel at 256×256 is **ten plate
  pixels**, which is where the ~1 px synthetic floor comes from.

## Verify it before believing it

No number in [docs/METHOD.md](docs/METHOD.md) is quoted without a control pass that returns
a value which *should* be zero. Both metric defects found while building this were metrics
measuring the input instead of the tracker, and each was caught exactly this way.

```bash
python -m dbtrack.engine --selftest     # rigid translation, exact GT -> 0.100 px
python tools/check_identity.py          # untrained DBtrack IS LocoTrack, 0.000e+00
python tools/score_occlusion.py --control   # scorer against truth -> 0.00000 px
python tools/eval_tapvid.py --mode strided  # the port, against published numbers
```

`check_identity.py` is the load-bearing one. With cross-track attention zero-initialised the
model is LocoTrack **bit for bit**, so any change measured after training is attributable to
the new blocks and not to the port having moved something.

## Train

```bash
python tools/probe_data.py --batches 2      # data control pass -- run this first
python tools/train_cross.py --occ-pos-weight 0.2 --steps 4000
python tools/score_ckpt.py --ckpt weights/step4000.ckpt --tag s4000
```

`freeze_base()` trains the 5.8M cross-track parameters and leaves the 11.5M vendor weights
bit-identical. `--occ-pos-weight` is the flag the whole result turns on: `0.0` is the
vendor's objective, which zeroes the position loss on occluded frames and therefore
*cannot* teach a model to cross an occlusion whatever attention you bolt on. `0.2` is
CoTracker3's `(𝟙_occ/5 + 𝟙_vis)` weighting, implemented from the paper's formula.

~2 s/step at 2.3 GB on an RTX A4000.

## Layout

```
dbtrack/            the package
  engine.py           DBTrackEngine -- the inference surface
  io.py               plate I/O, seeding, overlay, 3DE export
  losses.py           tapir_loss with the occluded-position weighting
  model/              cross_track.py, dbtrack_model.py
  data/               MOVi-E reader
tools/              every CLI: run, train, eval, the control passes, make_demo
docs/               METHOD.md (measurements), LICENSES.md (provenance)
assets/             README GIFs + their source attribution
vendor/locotrack/   LocoTrack, redistributed under Apache-2.0
```

## Licence

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).
[docs/LICENSES.md](docs/LICENSES.md) is the artifact-by-artifact ledger, including the one
open item: the base LocoTrack weights rest on the LocoTrack authors' own Apache-2.0
declaration, which is theirs to make.

Demo footage is DAVIS (CC BY 4.0) — attribution and the changes made are recorded in
[assets/README.md](assets/README.md).

## Citing

DBtracker derives from LocoTrack and reimplements an idea from CoTracker3. Cite both:

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
