# Three-way benchmark: Jeff-Tracker, TAPNext++, CoTracker3

TAP-Vid DAVIS, 30 clips, 256×256, query-first. The metric is `compute_tapvid_metrics` from
the official TAP-Vid code, called unmodified — reimplementing it would make the numbers
incomparable to every published table, which is the only reason to compute them.

Reproduce with `tools/benchmark.py`. Only Jeff-Tracker is built in; the other two are loaded
from paths you supply. **CoTracker3 is CC-BY-NC-4.0** and is not vendored, mirrored or
redistributed here. TAPNext++ is Apache-2.0, like Jeff-Tracker.

## Result

Offline models given the whole clip and backward tracking — what they are built for.
TAPNext is causal and has no such mode.

| model | licence | AJ | δ_avg | OA | s/frame |
|---|---|---|---|---|---|
| **TAPNext++** | Apache-2.0 | **66.2** | **79.4** | **92.1** | 0.2440 |
| Jeff-Tracker | Apache-2.0 | 62.6 | 74.9 | 86.7 | **0.0065** |
| CoTracker3 | CC-BY-NC | 61.9 | 76.8 | 87.5 | 0.0264 |

**TAPNext++ is the most accurate model here, and it is also Apache-2.0.** That is not the
result this project set out to find, and it is stated first because burying it would make
everything else in this repository less trustworthy. If accuracy is all that matters and
the licence must be clean, TAPNext++ is the better choice today.

Jeff-Tracker's case is cost. It is **37× faster** than TAPNext++ here at 3.6 AJ behind, and
it returns a calibrated per-frame confidence that neither of the others provides
(AUC 0.955 at detecting its own >5 px frames — see [METHOD.md](METHOD.md)). Against
CoTracker3 — the model whose licence motivated this whole project — Jeff-Tracker is +0.7 AJ,
−1.9 δ_avg, −0.8 OA, at a quarter of the cost.

The speed gap is workload-dependent and both figures are honest:

| | Jeff-Tracker | TAPNext++ | ratio |
|---|---|---|---|
| each in its natural mode | 0.0065 | 0.2440 | **37×** |
| both forced to forward-only grouping | 0.0186 | 0.2458 | **13×** |

TAPNext is causal, so a clip whose queries start on many different frames costs it one
streamed pass per distinct query frame. Jeff-Tracker answers all of them in a single pass.
That is a genuine property of the two designs, not a measurement artifact — but it is a
property of *this* workload, and a single-query-frame workload would narrow it.

## The protocol, and the check that it is not doing the work

In `first` mode the metric builds its evaluation mask as `cumsum(eye) - eye`, so only
frames strictly **after** a point's query frame are scored. Two protocols are therefore
available, and the ranking is reported under both because choosing one and not showing the
other would leave the result resting on a choice nobody could audit:

- **forward-only** — every model, causal or not, is given `frames[k:]` with its queries at
  local frame 0. Identical treatment, but it strips the offline models of the thing that
  makes them offline.
- **whole-clip** — the offline models get the entire clip, their queries at their true
  frame index, and backward tracking. TAPNext is unaffected.

| model | forward-only AJ | whole-clip AJ | Δ |
|---|---|---|---|
| Jeff-Tracker | 62.8 | 62.6 | −0.2 |
| TAPNext++ | 66.2 | 66.2 | 0.0 |
| CoTracker3 | 61.4 | 61.9 | +0.5 |

**The ranking is identical under both, and no model moves by more than 0.5 AJ.** So the
protocol is not what produces the result. This check was run precisely because the
forward-only protocol looked like it might be quietly handicapping the two offline models,
and it turns out not to be.

## Why the CoTracker3 row is below its published figure

CoTracker3's paper reports AJ 74.0 on DAVIS. It measures 61.9 here, and that gap needs an
account rather than a shrug.

**The harness is validated.** The same code path reproduces LocoTrack-B's published DAVIS
strided figures to within 0.1 on all three metrics (67.7 / 79.5 / 89.8 against a published
67.8 / 79.6 / 89.9). A harness that reproduces one published number and not another is
saying something about the difference between them, not about itself.

**Input resolution is not the cause.** CoTracker3's predictor rescales internally to
384×512, so feeding it 256×256 frames per the TAP-Vid protocol could in principle cost it.
Measured over the first 10 clips:

| CoTracker3 input | AJ | δ_avg | OA |
|---|---|---|---|
| 256×256 (the protocol) | 68.0 | 82.8 | 87.9 |
| native 854×480 | 68.6 | 82.6 | 88.5 |

**+0.6 AJ.** Not the explanation.

**The most likely cause is the metric definition.** CoTracker3 Table 1 reports
δ_avg^**vis** — averaged over visible points only — while `compute_tapvid_metrics` averages
over all evaluation points. Those columns are not directly comparable and this repository
has never quoted them side by side (see METHOD.md). Their checkpoints also differ: the
scaled model here is one of several the paper reports.

**So read the CoTracker3 row as "as configured here", not as a refutation of its paper.**
What the table supports is a like-for-like statement — three models, one metric
implementation, one protocol, one machine — and not a claim about anyone's published
numbers.

## Limits

- 30 DAVIS clips, one seed, one machine (RTX A4000, 16 GB). No error bars: inference is
  deterministic, but 30 clips is a small sample and per-clip AJ ranges from 19 to 97.
- One checkpoint per model. CoTracker3 has several; only `scaled_offline` was run.
- DAVIS only. It is the standard, and it is also 30 short clips of mostly-centred subjects
  — the synthetic and real-plate measurements in [METHOD.md](METHOD.md) exist because
  DAVIS alone does not describe how a tracker behaves on a 4K plate.
- Timings include the model's own forward pass only, on frames already decoded and resident.

## Reproduce

```bash
# whole-clip (the headline table)
python tools/benchmark.py --models jefftracker,tapnext,cotracker3 --whole-clip \
    --tapnext-root /path/to/tapnet-tree --tapnext-engine /path/to/tapnext_engine \
    --cotracker /path/to/co-tracker --cotracker-ckpt /path/to/scaled_offline.pth \
    --out out/benchmark_whole.json

# forward-only (the robustness check)
python tools/benchmark.py --models jefftracker,tapnext,cotracker3 ... --out out/benchmark.json

python tools/plot_benchmark.py --json out/benchmark_whole.json --out assets/benchmark.png
python tools/make_compare.py --clip breakdance --out assets/compare_three.gif ...
```
