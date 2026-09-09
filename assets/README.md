# Demo footage — source and attribution

The three GIFs in this folder are rendered from **DAVIS 2017**, reached through the
**TAP-Vid DAVIS** evaluation pack. DAVIS is the footage the whole tracking-any-point family
demos on, which is the point: the pictures are comparable by
eye against theirs, and unlike our own footage it can be redistributed.

| clip | file | licence |
|---|---|---|
| `breakdance` | `demo_grid.gif` | DAVIS video: **CC BY 4.0** |
| `dance-twirl` | `demo_confidence.gif` | DAVIS video: **CC BY 4.0** |
| `horsejump-high` | `demo_occlusion.gif` | DAVIS video: **CC BY 4.0** |

**DAVIS** — *A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation*,
Perazzi et al., CVPR 2016, and *The 2017 DAVIS Challenge on Video Object Segmentation*,
Pont-Tuset et al. https://davischallenge.org/ — videos licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

**TAP-Vid** — *TAP-Vid: A Benchmark for Tracking Any Point in a Video*, Doersch et al.,
NeurIPS 2022. https://github.com/google-deepmind/tapnet — point annotations Apache-2.0,
Copyright DeepMind Technologies Limited. The annotations are used for the evaluation
numbers in [`../docs/METHOD.md`](../docs/METHOD.md); the GIFs use only the video frames.

CC BY 4.0 requires attribution and an indication of changes. **Changes made:** frames were
resized, cropped in time, quantised to a 64-colour palette, and had tracking overlays drawn
on them. The underlying footage is otherwise unmodified.

## Reproducing them

Nothing here is hand-made. `tools/make_demo.py` regenerates all three from the TAP-Vid pack:

```bash
curl -L -o tapvid_davis.zip https://storage.googleapis.com/dm-tapnet/tapvid_davis.zip
unzip tapvid_davis.zip -d data/

python tools/make_demo.py --clip breakdance  --mode grid --grid 14 --tail 12 \
    --frames 55 --width 460 --fps 12 --colors 64 --out assets/demo_grid.gif
python tools/make_demo.py --clip dance-twirl --mode conf --grid 12 --tail 8 \
    --frames 55 --width 460 --fps 12 --colors 64 --out assets/demo_confidence.gif
python tools/make_demo.py --clip horsejump-high --mode occl --grid 12 --tail 4 \
    --drop-after 8 --frames 50 --width 460 --fps 12 --colors 64 \
    --out assets/demo_occlusion.gif
```

## Why these three clips

Picked on measured coverage across ten DAVIS clips, not on how they look.

| clip | frames | visible | mean conf |
|---|---|---|---|
| breakdance | 84 | 78.4% | 0.784 |
| dog | 60 | 66.7% | 0.670 |
| car-shadow | 40 | 66.5% | 0.662 |
| goat | 90 | 65.8% | 0.662 |
| horsejump-high | 50 | 65.4% | 0.649 |
| dance-twirl | 90 | 59.8% | 0.610 |
| paragliding-launch | 80 | 51.9% | 0.519 |
| libby | 49 | 30.6% | 0.296 |
| bmx-trees | 80 | 22.3% | 0.222 |
| motocross-jump | 40 | 18.1% | 0.182 |

`breakdance` has a locked-off camera, so only genuinely moving points draw a trail and a
dense grid stays legible.

`horsejump-high` is the occlusion panel. It pans, so its tail is cut to 4 frames — at 12
the background streaks across the whole frame, which is correct tracking and an unreadable
picture. It was chosen over `libby` (30.6% visible) on what the panel actually shows:

| clip | filled | hollow — declined, held | retired |
|---|---|---|---|
| horsejump-high | 65.4% | **12.6%** | 21.9% |
| goat | 72.9% | 8.8% | 18.3% |

`libby` is the most literal occlusion in DAVIS — a dog walks behind a tree — but by the
time it is behind the trunk almost every track has retired, so the panel reads as "the
overlay vanished" rather than showing the model declining to place points. A demo that
communicates nothing is not more honest for being dramatic.

## How occluded frames are drawn, and why

The overlay never draws a position the model has not committed to. On a frame it calls
occluded, the position it emits is unconstrained — nothing in the loss pins it there — so
it wanders, and an earlier version of these GIFs drew those wandering points as rings
flying across the frame. That was noise being presented as output.

A point that goes occluded is now **held at its last committed position**, drawn as a
hollow ring, and **retired entirely after 8 frames** (`--drop-after`). Tail segments longer
than 20% of the frame width are skipped too: a jump that large between two frames the model
both calls visible is a re-acquisition landing elsewhere, not motion, and drawing it as a
line implies a path that was never travelled.

This is a display choice, not a claim about the model. It still emits those positions; the
coverage figures in [`../docs/METHOD.md`](../docs/METHOD.md) are what quantify how often it
declines to commit.

## The comparison score sheets

`comparison_default_settings.png` and `comparison_matched_resolution.png` are generated,
not drawn: `tools/score_sheet.py` re-scores every `.npz` a benchmark run produced and
renders the page from those numbers, so the sheets cannot drift from the tables in
[`../docs/COMPARISON.md`](../docs/COMPARISON.md).

```bash
python tools/score_sheet.py --tag phase0
python tools/score_sheet.py --tag phase1 \
    --engines "jefftrack:Jeff-Tracker 256x256:#3987e5,jefftrack_r384x512:Jeff-Tracker 384x512:#c98500,cotracker3:CoTracker3:#199e70,tapnext:TAPNext++:#d95926" \
    --verdict docs/verdicts/phase1.txt
```

They contain no footage - only measurements. The four benches behind them are synthetic and
built by `tools/make_occlusion_bench.py`, so nothing in these images is derived from
licensed video.

Series colours are fixed per engine so two sheets can be laid side by side, and the set was
checked for colour-blind separation before use rather than by eye (worst adjacent CVD delta-E
9.4, worst normal-vision delta-E 20.9, all four at or above 3:1 contrast on the sheet's
surface).

## The logo

`logo.png` is the project's own mark, supplied by the repository owner. It is cropped from
the original render and downscaled to 1100 px wide; nothing else about it is changed. It is
not part of the Apache-2.0 grant that covers the code — the licence in `LICENSE` applies to
software, not to a trademark or logo.
