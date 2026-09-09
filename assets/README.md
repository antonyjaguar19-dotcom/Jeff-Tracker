# Demo footage — source and attribution

The three GIFs in this folder are rendered from **DAVIS 2017**, reached through the
**TAP-Vid DAVIS** evaluation pack. DAVIS is the footage the whole tracking-any-point family
demos on — TAPIR, LocoTrack, CoTracker — which is the point: the pictures are comparable by
eye against theirs, and unlike our own footage it can be redistributed.

| clip | file | licence |
|---|---|---|
| `breakdance` | `demo_grid.gif` | DAVIS video: **CC BY 4.0** |
| `dance-twirl` | `demo_confidence.gif` | DAVIS video: **CC BY 4.0** |
| `libby` | `demo_occlusion.gif` | DAVIS video: **CC BY 4.0** |

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
python tools/make_demo.py --clip libby       --mode occl --grid 12 --tail 8 \
    --frames 49 --width 460 --fps 12 --colors 64 --out assets/demo_occlusion.gif
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
dense grid stays legible. On a panning clip such as `horsejump-high` every background point
streaks across the frame — correct tracking, unreadable picture.

`libby` sits near the bottom of that table at 30.6%, and it is in here **because** of that.
It is the occlusion demo: the hollow rings are frames the model declines to place. Choosing
only the clips at the top of this table would be a brochure.
