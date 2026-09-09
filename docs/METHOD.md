# Method and measurements

Everything DBtracker claims, how it was measured, and the things that did not work. The
negative results are here on purpose: a page that only lists wins is not evidence, and two
of the most useful findings in this project were metrics that looked plausible and were
measuring the input instead of the tracker.

Rules this document follows:

- an accuracy claim needs a number against ground truth, not a viewport impression
- a metric is not believed until it has been fed known-good input and returned the number
  it should have returned — usually zero
- ranges across checkpoints, not the best row. Quoting a peak as a level was a mistake
  made twice here and corrected twice

## Control passes, first

Nothing below is believable without these.

```
python -m dbtrack.engine --selftest
  median 0.100 px on a known rigid translation with exact GT      PASS

python tools/score_occlusion.py --control
  VISIBLE / OCCLUDED / RE-ACQUIRE worst error 0.00000 px          PASS

python tools/check_identity.py
  vendor parameters : 147 shared, 0 differ, 0 lost, 63 added by DBTrack
  cross block 0/1/2   out_proj zero: True
  tracks / occlusion / expected_dist   max |diff| 0.000e+00   identical True
  PASS  untrained DBTrack is LocoTrack
  5,789,952 cross-track parameters added to 17,309,669 total (33.4%)
```

The identity property is the one that makes every later number interpretable. Without it,
a change measured after training is a mixture of "cross-track attention helped" and "the
port moved something", with no way to separate the two afterwards. It holds at the pipeline
level too: the same occlusion bench through `--arch dbtrack` produces a **byte-identical
export** to `--arch locotrack`.

## The port, against the authors' own published numbers

```
python tools/eval_tapvid.py --mode strided
```

| TAP-Vid DAVIS, strided, 256×256 | AJ | δ_avg | OA |
|---|---|---|---|
| LocoTrack-B, **as published** (arXiv 2407.15420, Table 1) | 67.8 | 79.6 | 89.9 |
| **measured here** | **67.7** | **79.5** | **89.8** |

Within **0.1 on all three**. This is the check that matters most: it says the port, the
coordinate conventions and the evaluation path are all correct, so every accuracy and
occlusion number below is measuring the tracker rather than a broken wrapper. 30 videos in
72 s.

Query-first mode, for completeness: **AJ 62.6, δ_avg 74.7, OA 86.9** (26 s). The paper
publishes no first-mode DAVIS row, so there is nothing to check that against.

### On comparing against CoTracker3's published table

CoTracker3 Table 1 reports δ_avg^**vis** — averaged over visible points only — while the
standard `compute_tapvid_metrics` averages over all evaluation points. Those columns are
**not** directly comparable and must not be quoted side by side. What is meaningful is the
gap *within* that table, all Kubric-only, all DAVIS:

| CoTracker3 Table 1 (δ_avg^vis) | AJ | δ_avg^vis | OA |
|---|---|---|---|
| CoTracker3 offline | 74.0 | 84.9 | 90.5 |
| CoTracker3 online | 71.1 | 81.9 | 90.3 |
| LocoTrack | 69.7 | 83.2 | 89.5 |
| CoTracker | 67.4 | 78.9 | 85.2 |
| TAPIR | 55.5 | 69.7 | 88.0 |

**CoTracker3 offline is +4.3 AJ over LocoTrack**, at +1.7 δ and +1.0 OA — a gap
concentrated somewhere other than plain localisation, which is what the occlusion bench
below says independently. That is the gap cross-track attention is aimed at, and
establishing it needs no CoTracker code, weights or outputs at all.

## Localisation — synthetic bench, exact homography ground truth, 600 tracks

A single textured plane under a known homography, so every seed has exact truth. It
measures localisation only: one plane means no parallax and no occlusion, and every track
on it comes out good.

| model res | mean_err | tracks over 1 px | worst track | s/frame | peak VRAM |
|---|---|---|---|---|---|
| 256×256 | 1.03 px | 324/600 | 2.87 px | 0.057 | 2.77 GB |
| 288×512 | 0.61 px | 30/600 | 1.35 px | 0.064 | 8.14 GB |
| 384×680 | **0.54 px** | 18/600 | **29.14 px** | 0.082 | 9.95 GB |

Per feature class at 384×680: corner 0.56, blob 0.55, edge 0.51, flat 0.53, dense-corner
0.72, dense-edge 0.67. **Flat across classes**, which is worth saying plainly: this model
has no feature type it is bad at. Trackers that do need per-track policy machinery to work
around it.

Peak locking amplitude **+0.0005 px** over 117k samples. The exported-position phase
histogram is flat and matches the ground-truth control histogram. There is no pixel
locking to correct — notable on a path with no NCC anywhere in it.

Note the `worst track` column moving the wrong way as resolution rises. That is the same
tail the occlusion bench and the README warn about, visible here before occlusion is even
introduced.

### This is not comparable to a refined pipeline

The figure above is raw neural output over all 600 seeds with no refinement stage. A
classical pipeline — moving-tile re-track at native resolution, then NCC + affine sub-pixel
polish — over the handful of seeds that survive its quality gates reaches 0.06 px on the
same bench. The honest reading is that DBtracker is a seed-and-track stage that would still
feed such a refinement downstream, not a replacement for it.

## Real footage

Scored against a Lucas-Kanade reference built by round-tripping each track forward and all
the way back, on a 160-frame 2560×1440 plate. LK is a different algorithm family from
neural tracking, so the two do not share a pixel-locking bias, but it is **not ground
truth**: a fault both methods share is invisible to it. The reference's own round-trip
closure averages 0.60 px, so the deviations below are real.

| model res | mean | median-of-means | per track |
|---|---|---|---|
| 256×256 | 3.37 px | 3.52 | 4.17 / 2.02 / 3.52 / 5.13 / 2.01 |
| 288×512 | 3.67 px | 1.95 | 9.51 / 1.73 / 3.34 / 1.83 / 1.95 |
| 384×680 | 3.24 px | 3.07 | 3.07 / 1.79 / 6.75 / 1.12 / 3.47 |

n=5 corners, spread is large and non-monotonic in resolution. Read this as a sanity check
that the model works on real footage; take the accuracy conclusion from the 600-track
synthetic bench instead.

**A scoring lesson worth repeating.** A proximity-pairing scorer — one that matches tracker
output to references within 25 px, which is correct when the tracker chose its own features
and was never told about the references — reported 6.84 / 6.63 / 6.28 px for these same
three runs. It was scoring corners seeded up to 25 px away from the reference feature, so
the "error" was mostly that offset. The tell was that a 2.7× resolution change moved the
number by 0.6 px, which is not what a resolution change does to a tracker.
`tools/eval_vs_manual.py`, which seeds **on** the reference points, exists because of this.

## Occlusion

`tools/make_occlusion_bench.py` composites moving textured occluders over the synthetic bench,
keeping exact ground truth. Rebuilding from the same `--seed` produces an identical
`occluders.json` and **pixel-identical frames**, so these numbers can be re-measured
elsewhere rather than merely quoted.

| model res | VISIBLE mean | VISIBLE med | OCCLUDED mean | occluded frames emitted | RE-ACQUIRE mean |
|---|---|---|---|---|---|
| 256×256 | 1.346 | 1.185 | 2.602 | **5.9%** | **1.622** |
| 384×680 | 40.553 | **0.697** | 213.4 | 4.2% | **35.777** |

Two findings, and the second is the dangerous one.

**1. It gaps an occlusion rather than crossing it.** Only ~6% of the frames ground truth
says were covered got a position at all; on the rest the model marked the point unusable
and the exporter wrote a hole. That is honest behaviour, and a gap is a legal representation
of an occlusion in 3DE ASCII — but it makes the occluded-error column nearly vacuous, so
the coverage figure is printed beside it and **that number must never be quoted alone**.

**2. At 384×680 it does not survive an occlusion, and is confident about it.** Median
visible error 0.697 px — better than 256×256 — while the mean is 40.6 px and the worst
sample is 1730 px. Re-acquisition mean 35.8 px says what happened: after an occluder
passes, the track does not come back, and every later frame counts as visible-and-wrong. On
the occluder-free bench the same config's worst track was 29 px, so this is
occlusion-triggered, not general instability.

### Can the model's own confidence catch that tail?

| model res | AUC (conf detects a >5 px frame) | at conf ≥ 0.5: kept | mean | p99 | max |
|---|---|---|---|---|---|
| 256×256 | **0.955** | 87.9% | 1.36 px | 4.08 | **9.0** |
| 384×680 | 0.818 | 81.8% | 41.57 px | 1244 | 1730 |

At 256×256 the confidence is a genuinely good detector and gating at its own 0.5 threshold
caps the worst sample in the entire run at 9 px. **At 384×680 no threshold helps** — at
conf ≥ 0.99 the mean is still 39.3 px and p99 still 1204 px. Trading a 1.17 px median for
0.69 px is not worth buying silent 1000 px errors. This is why 256×256 is the default.

This native confidence (AUC 0.955) also beats a photometric confidence metric
reconstructed from pixels (AUC 0.934 at 5 px) and costs nothing — the model already
computes it.

## Speed and memory

| frames | plate | fed | model res | s/frame | peak VRAM |
|---|---|---|---|---|---|
| 312 | 3840×2160 | 960×540 | 256×256 | **0.030** | 6.85 GB |
| 160 | 2560×1440 | 960×540 | 256×256 | 0.020 | 4.40 GB |
| 160 | 2560×1440 | 960×540 | 384×680 | 0.045 | 9.96 GB |

On an RTX A4000 (16 GB).

## Two bugs found while building this

**The window seam cost 72 px.** LocoTrack is not causal — one query produces the whole clip
— so windowing exists only to bound VRAM. The first version re-queried each track at the
new window's frame 0 using the position from the *previous* window's last frame: an
`overlap`-frame mismatch between the frame index and the position it referred to, which
compounded at every seam. Selftest median went 0.100 px → **72.588 px**. The fix is to
re-query at frame `s` with the position the previous window predicted *for frame s*, which
is what the overlap is for. It also has to be clamped: `window_overlap=8` with `window=8`
collapsed the step to 1 and turned the clip into one re-query per frame. Post-fix the seam
cost is graceful — 0.100 px (1 window) → 0.173 (3) → 0.248 (5).

This is exactly the failure class the selftest was written to catch: every one of those
runs produced a full set of plausible-looking tracks.

**Model resolutions must be multiples of 8.** The ResNet strides down by 8; a side that is
not a multiple emits `output size is not a multiple of 8. Final layer will round size
down.` and the feature grid stops corresponding to the pixels fed in. `384x682` was
silently in this state. The engine now snaps and says so.

## Cross-track attention

`dbtrack/model/cross_track.py`. Attention over the **track** axis, run independently per
frame, mediated by 16 learned proxy tokens so cost is O(N·K) rather than O(N²). Tracks are
a set, not a sequence, so there is no positional encoding and no causal mask — either would
assert an order between track 7 and track 8 that does not exist.

Written from the description in the CoTracker3 paper (arXiv 2410.11831). No file, function,
or weight tensor derives from the CoTracker repository.

`dbtrack/model/dbtrack_model.py` re-parents LocoTrack's own `input_proj`, `transformer` and
`output_proj` — the same module objects, not rebuilt — and unrolls the vendor's layer loop
so a cross-track block can sit between layers. Parameter names are therefore unchanged and
the published checkpoint loads with the new `cross.*` keys as the only additions.

### Training data, and why not the obvious one

| | licence | verdict |
|---|---|---|
| `LocoTrack-panning-MOVi-E`, 293 GB — what LocoTrack actually trained on | **none declared** | rejected. Zero distribution shift, but the entire point of this project is a tracker that ships |
| `movi_e/256x256` from `gs://kubric-public/tfds` | **Apache-2.0**, published by Google | chosen. Provenance holds end to end: Apache code, Apache base weights, Apache data |

The price is a distribution shift — LocoTrack-B was fine-tuned on *panning* MOVi-E and this
is plain MOVi-E. That is measurable and recoverable. An unclear licence is neither.

Rendering fresh Kubric data is not an option without Docker, so only the pre-rendered TFDS
shards are used, which need nothing but tensorflow to read. `add_tracks` comes from
Kubric's own `challenges/point_tracking/dataset.py` (Apache-2.0), vendored as that single
file rather than pip-installing `kubric`, which on Windows drags in pybullet and OpenEXR
for a renderer that cannot run.

### Data control pass — before a single training step

```
python tools/probe_data.py --batches 2

  video (1,24,256,256,3) float32 in [-1.00, 0.80]        range ok
  query_points (1,256,3)  target_points (1,256,24,2)  occluded (1,256,24)
  query anchored to its own track: max 0.0034 px         ok
  occluded 9.8-11.2% of samples                          ok
  tracks with a PARTIAL occlusion 25.4-58.6%
  PASS
```

The last row is why this dataset was chosen over extending the homography bench: a track
hidden *while its neighbours stay visible* is precisely the case cross-track attention
exists to exploit, and one moving plane cannot produce it. The query-anchoring check exists
because `query_points` is `tyx` and `target_points` is `xy`; crossing those conventions
computes every position loss against a shifted target and still trains happily.

## The objective was the blocker, not the architecture

The first fine-tune ran 3,600 steps before its own numbers said it could not work. Scored
on **all** frames from the `.npz` rather than the export, so the export's visibility gate
is out of the picture:

| | occluded-frame error | within 5 px | visible median |
|---|---|---|---|
| LocoTrack | median **3.54 px** | 71.2% | 1.20 px |
| after 3,000 steps | median **3.66 px** | 69.4% | **1.16 px** |

Visible error improved and occluded error got **worse**. The cause is one line in the
vendor's loss:

```python
loss_huber = loss_huber * (1.0 - occluded.float())
```

The position term is **zeroed** on every frame ground truth calls occluded. LocoTrack is
never trained to place a point it cannot see, so with no gradient there the model is free
to drift wherever suits the visible frames — which is exactly what it did. **No amount of
training under that objective can teach a model to cross an occlusion**, regardless of what
attention is bolted on.

CoTracker3 weights occluded points at one fifth instead of zero — the `(𝟙_occ/5 + 𝟙_vis)`
term in its loss. `dbtrack/losses.py` implements that weighting from the paper's formula
and is otherwise the vendor's arithmetic line for line. Verified to reduce to it exactly:

```
vendor tapir_loss      61.05917740
ours, occ weight 0.0   61.05917740   diff 0.000e+00
ours, occ weight 0.2   65.46984863
```

So the two objectives are one flag apart, and a difference between two runs is attributable
to that flag — the same discipline `tools/check_identity.py` enforces for the weights.

## Result

The third run — occluded points weighted at 1/5 **and** restricted to targets still inside
the frame — clears every clause of the stop condition by step 1,000, a third of the steps
the earlier runs spent going nowhere. The condition was written down before the result
existed.

| clause | required | @1000 | across 5 checkpoints |
|---|---|---|---|
| re-acquire | beat 1.622 px | 1.514 ✓ | 1.512–1.559, all better ✓ |
| occluded localisation | beat 3.540 px | 3.392 ✓ | 3.207–3.543, one row at baseline |
| occluded within 5 px | beat 71.2% | 74.1% ✓ | 72.1–76.0%, all better ✓ |
| visible not lost | hold 1.346 px | 1.298 ✓ | 1.298–1.316, all better ✓ |
| confidence AUC | hold 0.955 | 0.959 ✓ | 0.959 |

Per checkpoint, so the trend can be read rather than a single flattering row:

| | visible | occ med | within 5px | re-acquire | AJ | δ_avg | OA |
|---|---|---|---|---|---|---|---|
| LocoTrack baseline | 1.346 | 3.540 | 71.2% | 1.622 | 67.7 | 79.5 | 89.8 |
| step 1000 | 1.298 | 3.392 | 74.1% | 1.514 | 67.8 | 79.6 | 89.7 |
| step 2000 | 1.316 | 3.411 | 73.3% | 1.559 | 67.7 | 79.6 | 89.5 |
| step 3000 | 1.308 | 3.331 | 75.5% | 1.543 | 67.9 | 79.7 | 89.9 |
| step 4000 | 1.303 | **3.207** | **76.0%** | **1.512** | 67.9 | 79.7 | 89.8 |
| step 5000 | 1.310 | 3.543 | 72.1% | 1.553 | 67.8 | 79.6 | 89.7 |

**Read this as a band, not a trend.** Within-5px goes 74.1 → 73.3 → 75.5 → 76.0 → 72.1, and
the occluded median at step 5000 (3.543) is indistinguishable from baseline (3.540). An
earlier revision of this document called the improvement monotonic on four checkpoints; the
fifth falsified it. What survives the correction: **every** checkpoint beats baseline on
occluded within-5px (worst 72.1% vs 71.2%), and none loses visible accuracy, AJ, δ_avg or
OA. The direction is real; the magnitude is smaller and noisier than any single row
suggests.

### The magnitude, measured properly — three independent occluder layouts

The first bench had 6,905 occluded samples from 600 tracks crossing only **4** occluders —
heavily correlated, with an effective sample size far below the raw count. Three more
benches were built, each with **8** occluders and its own seed, each control pass passing
at 0.00000 px. Baseline LocoTrack against the step-4000 checkpoint:

| bench (frame cover) | within-5px | occluded median | visible mean | re-acquire | coverage |
|---|---|---|---|---|---|
| occ11 (17.8%) | 70.0 → **73.9** | 3.538 → 3.278 | 1.424 → 1.380 | 1.700 → 1.626 | 7.6 → 8.7 |
| occ12 (21.5%) | 59.4 → **63.0** | 4.234 → 3.940 | 1.442 → 1.411 | 1.928 → 1.890 | 6.3 → 7.2 |
| occ13 (24.4%) | 59.3 → **62.1** | 4.286 → 4.058 | 1.453 → 1.431 | 1.867 → 1.830 | 6.0 → 7.0 |

**Same sign on every metric on every bench.** Mean effect: occluded within-5px **+3.4
points**, occluded median **−6%**, visible mean **−0.03 px** (better, not traded away),
re-acquire **−0.05 px**, coverage **+1.0 point**.

That is the number to quote — **+3.4**, not the +4.8 that came from the best row of a
single bench. It is also **held out**: step 4000 was chosen on the first bench before
occ11/12/13 existed. It is the one occlusion figure here obtained without looking at the
bench it is quoted on.

### The occluded gain is unstable, and the arithmetic says why

Every checkpoint, averaged over the same three benches, as a delta against baseline:

| checkpoint | Δ within-5px | Δ occ median | Δ visible | Δ re-acquire |
|---|---|---|---|---|
| step 3000 | +3.03 | −0.223 | −0.031 | −0.047 |
| step 4000 | **+3.43** | −0.261 | −0.032 | −0.050 |
| step 5000 | −0.13 | +0.007 | −0.031 | −0.035 |
| step 6000 | +1.63 | −0.126 | −0.013 | −0.010 |
| step 7000 | −0.13 | −0.013 | −0.058 | −0.074 |

Two behaviours, and they must not be reported as one:

- **Visible accuracy improves reliably** — −0.03 px at every single checkpoint. Solid.
- **Occluded accuracy oscillates between +3.4 and 0** with no trend in either direction. An
  earlier revision called this a decay after seeing step 7000; with 3000–6000 filled in it
  is plainly oscillation, and that reading was as wrong as the "monotone" one before it.

The cause is arithmetic that should have been checked before the run, not after. Occluded
samples are ~11% of the data; the in-frame restriction keeps about half of them; they carry
weight 0.2. So the occluded term is roughly **1% of the position loss**. At batch size 1
that is too weak to hold a solution against the visible objective, and the occluded
behaviour wanders instead of converging. Step 4000 is a good draw from that wandering, not
a plateau the model climbed to.

**What to do about it** — stated as a hypothesis, not a result: normalise the occluded term
by the *count of occluded samples* rather than letting it be a mean over all samples, so its
share of the gradient is a stated quantity instead of an accident of how much occlusion a
clip happens to contain. That makes the occluded weight mean what it appears to mean, and
is the obvious next run.

**Shipping checkpoint: `inf_s4000.ckpt`**, selected on the first bench and validated on
three built afterwards. Picking the best of five checkpoints *using* occ11/12/13 would be
selection on the test set, and no number here is obtained that way.
