# TAP-Vid DAVIS: results, protocol and controls

Jeff-Tracker on TAP-Vid DAVIS, 30 clips, 256×256, query-first. The metric is
`compute_tapvid_metrics` from the official TAP-Vid code, called unmodified — reimplementing
it would make the numbers incomparable to every published table, which is the only reason
to compute them.

```bash
python tools/benchmark.py --models jefftracker --whole-clip --out out/benchmark_whole.json
python tools/plot_results.py --json docs/results.json --out assets/results.png
```

## Result

| | AJ | δ_avg | OA | s/frame |
|---|---|---|---|---|
| **Jeff-Tracker** | **62.6** | **74.9** | **86.7** | **0.0065** |
| LocoTrack-B, the base it is fine-tuned from | 62.6 | 74.7 | 86.9 | 0.0063 |

**On DAVIS the fine-tune is level with its base.** That is the honest result and it is
reported first, because the checkpoint was trained for occlusion and the question a reader
should ask is whether that cost anything general. It did not — and it did not buy anything
general either. What it bought is in [METHOD.md](METHOD.md) and in the right-hand panel of
`assets/results.png`: occluded frames placed within 5 px, **+3.4 points** averaged over
three occlusion benches built after the checkpoint was chosen.

Run the base row yourself with `--arch locotrack --ckpt weights/locotrack_base.ckpt`; those
weights are ungated.

## The port is correct, and this is how that is known

The same code path reproduces LocoTrack-B's **published** DAVIS strided figures:

| DAVIS, strided, 256×256 | AJ | δ_avg | OA |
|---|---|---|---|
| LocoTrack-B, as published (arXiv 2407.15420, Table 1) | 67.8 | 79.6 | 89.9 |
| measured here | **67.7** | **79.5** | **89.8** |

Within **0.1 on all three**. This is the load-bearing check for everything else in this
repository: it says the port, the coordinate conventions and the evaluation path are all
correct, so the accuracy and occlusion numbers are measuring the tracker rather than a
broken wrapper. Reproduce with `python tools/eval_tapvid.py --mode strided`.

Note that the strided figures (67.7) and the query-first figures (62.6) are different
protocols and are not interchangeable. Both are standard; neither is "the" DAVIS number.

## Two protocols, and the check that the choice is not doing the work

In `first` mode the metric builds its evaluation mask as `cumsum(eye) - eye`, so only frames
strictly **after** a point's query frame are scored. That leaves two defensible ways to run
a model that can see the whole clip:

- **forward-only** — the model is given `frames[k:]` with its queries at local frame 0. This
  is what a causal tracker is limited to, and it strips an offline model of the thing that
  makes it offline.
- **whole-clip** — the model gets the entire clip, queries at their true frame index, and
  backward tracking.

| | forward-only AJ | whole-clip AJ | Δ |
|---|---|---|---|
| Jeff-Tracker | 62.8 | 62.6 | −0.2 |

**0.2 AJ between them.** The protocol is not what produces the number. Both are reported
because choosing one and not showing the other would leave the result resting on a decision
nobody could audit.

## Limits

- 30 DAVIS clips, one seed, one machine (RTX A4000, 16 GB). No error bars: inference is
  deterministic, but 30 clips is a small sample and per-clip AJ ranges from 19 to 97.
- DAVIS is the standard, and it is also 30 short clips of mostly-centred subjects. The
  synthetic and real-plate measurements in [METHOD.md](METHOD.md) exist because DAVIS alone
  does not describe how a tracker behaves on a 4K plate.
- Timings cover the model's forward pass only, on frames already decoded and resident.
- `tools/benchmark.py` also carries an optional TAPNext++ adapter (Apache-2.0) for anyone
  who wants a second reference point; nothing of it is vendored here.
