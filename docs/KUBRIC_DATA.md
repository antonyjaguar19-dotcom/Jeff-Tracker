# Training on Meta's Kubric release: what it bought, and what it did not

`facebook/CoTracker3_Kubric` is the synthetic set Meta released for CoTracker3 stage 1, and
it is **Apache-2.0** — verified on the dataset page and on the Hub API's `cardData.license`,
not assumed. CoTracker3's own code and weights are CC-BY-NC and are neither used nor derived
from anywhere in this repository; only the openly-licensed data is.

```bash
hf download facebook/CoTracker3_Kubric --repo-type dataset \
    --local-dir datasets/CoTracker3_Kubric
python tools/convert_kubric.py --selftest        # no dataset needed, runs in seconds
python tools/convert_kubric.py                   # ~4 s per shot, 15x smaller on disk
python tools/train_cross.py --data-source kubric_meta --steps 20000 --accum 16 \
    --occ-norm --occ-l1 --unfreeze-mixer --out weights/c8_kubric.ckpt
```

## Why this data

Against the MOVi-E the earlier checkpoints trained on: 120 frames instead of 24, 512×512
instead of 256×256, 32,768 tracked points per shot, a depth pass and a ground-truth camera.

The reason to want it was **occlusion length**, not volume. MOVi-E clips are 24 frames, so an
occlusion lasting longer than that cannot exist in them; measured on this data, 18.5% of
occlusion events run past 24 frames, median 4, mean 12.7, longest 117.

The volume argument does not hold and is recorded here so nobody re-derives it: 28.45% of
samples are flagged occluded, but **75.2% of those are simply off screen**. Hidden while
still inside the frame is 7.06%, against MOVi-E's ~5.5%. Marginal.

## Three traps in the source data

Each will silently invert a training run, and none is discoverable by reading a field name.
All three were settled photometrically — sampling the frames along each track and measuring
colour stability — rather than assumed.

| trap | the check |
|---|---|
| the array named `visibility` holds **occlusion**; `True` means HIDDEN | colour spread 2.80 for flag-False against 23.69 for flag-True: False is the point still sitting on its own feature |
| coordinates are **(x, y)**, column first | 2.80 read as (x, y) against 17.61 read as (y, x) |
| 75% of "occluded" samples are **off screen**, not hidden | `losses.py` already separates those two populations (`occluded_in_frame_only`) |

`tools/convert_kubric.py` writes the field out as `occluded`, so trap 1 is not re-exported
into the trainer. `jefftrack/data/kubric_meta.py --selftest` runs the same photometric check
end to end, through the crop, resize and coordinate rescale, so a sampling change that moved
the tracks the wrong way fails a test instead of producing plausible numbers.

## The cache

585 GB of archives become ~41 GB, 15.2× smaller, by dropping what training never reads: 252 MB
per shot of float64 depth, and a 145 MB bundle holding that depth again plus **byte-identical
copies** of the two track arrays stored beside it (verified with `np.array_equal` before
anything was discarded). Frames are re-encoded at JPEG quality 98 with **4:4:4 chroma** —
mean error 0.65 of one grey level, against 2.0 at quality 100 with the default 4:2:0 — and
kept at 512×512 so random-crop augmentation still has room.

`--delete-source` is a separate run with its own verification and a dry run by default. The
source is a ~1.9 TB re-download; a converter that deletes as it goes would destroy it on the
strength of a bug it is itself carrying.

## Results: a trade, not an upgrade

`c8_kubric` against `c5_occl1` (the previous best), three late checkpoints per arm, five
synthetic occlusion benches with exact ground truth, 20 comparisons:

| metric | separated better | separated worse | overlap |
|---|---|---|---|
| **visible mean** | **5 of 5** | 0 | 0 |
| **re-acquire mean** | **5 of 5** | 0 | 0 |
| occluded mean | 1 | 1 | 3 |
| occluded median | 1 | 1 | 3 |

Every treatment checkpoint beats every control checkpoint on visible accuracy (~3%) and on
re-acquisition (~4.5%), on every bench. Re-acquisition is the column that matters most for
matchmove: a point returning on the *neighbouring* feature looks correct in the viewport and
quietly poisons a camera solve.

**It costs about a point on the public benchmark:**

| TAP-Vid DAVIS strided | AJ | δ_avg | OA |
|---|---|---|---|
| `c5_occl1` | **68.3** | **79.8** | **89.9** |
| `c8_kubric` | 67.2 | 79.3 | 88.8 |

DAVIS is internet video; the benches are camera-move footage with per-pixel truth. Take
`c8_kubric` for matchmove, and not for a DAVIS number.

## What it did not do, and the mechanism

Accuracy **while a point is hidden** did not move: 2.90 px against CoTracker3's 1.69 on the
same bench. The coverage figure explains why, and it is the opposite of the intended
direction:

| willingness to commit a position while hidden, gate 0.5 | |
|---|---|
| `c5_occl1` | ~5.5% |
| `c8_kubric` | ~3.7% |
| `c9_clip48` (48-frame windows) | ~3.6% |

The model became **more cautious, not more capable** — it withholds about a third of the
hidden-frame positions it used to emit. That accounts for the whole shape of the result:
fewer committed guesses give tighter visible tracks and better re-acquisition, and no
progress through an occlusion.

Doubling the training window to 48 frames — the experiment this dataset was chosen for —
changed that coverage figure by nothing and lost to the 24-frame run on 6 of 20 comparisons
while winning 1. Rejected; see `docs/verdicts/kubric_run2.txt`.

## Five attempts, five refusals

| attempt | parameters added | verdict |
|---|---|---|
| more neighbours per chunk | 0 | no effect (0.002 px at 9×) |
| proxies carrying temporal state | 0 | wash |
| a fourth correlation level | 266,752 | worse on 3 benches |
| better training data | 0 | localisation yes, occlusion no |
| doubling the training window | 0 | no effect, worse overall |

Three change the architecture, two change what it is fed, and the occluded column has moved
for none of them. Treat it as a structural property of per-track refinement rather than a
tuning problem: a hidden point's position is computed without the model ever being *required*
to produce one, and that is the thing all five attempts left alone.
