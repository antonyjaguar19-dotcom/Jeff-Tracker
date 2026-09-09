"""Chart Jeff-Tracker's own measurements from docs/results.json.

    python tools/plot_results.py --json docs/results.json --out assets/results.png

Two panels, and neither is a league table:

  left   TAP-Vid DAVIS AJ / delta_avg / OA, with LocoTrack-B -- the base this model is
         fine-tuned from -- as a hollow reference bar. That is an ablation, not a
         comparison: the question it answers is whether the fine-tune cost anything on the
         general benchmark, and the honest answer is that DAVIS is flat.
  right  what the fine-tune actually bought: occluded frames placed within 5 px, on three
         synthetic occlusion benches with exact ground truth, base against trained.

The right panel is the point of the model and the left panel is the control that stops the
right one being quoted alone.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BASE = "#9AA6B2"      # the LocoTrack base: present, deliberately recessive
MODEL = "#0072B2"     # Jeff-Tracker


def main() -> int:
    ap = argparse.ArgumentParser(description="chart Jeff-Tracker's own results")
    ap.add_argument("--json", default="docs/results.json")
    ap.add_argument("--out", default="assets/results.png")
    a = ap.parse_args()

    with open(a.json) as fh:
        r = json.load(fh)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    fig.patch.set_facecolor("white")

    # ---- left: DAVIS, with the base as a reference ------------------------------------
    d = r["davis"]
    ref = d["reference"]
    keys = [("AJ", "AJ"), ("delta_avg", "$\\delta_{avg}$"), ("OA", "OA")]
    xs = range(len(keys))
    w = 0.36
    axL.bar([x - w / 2 for x in xs], [ref[k] for k, _ in keys], w,
            color="white", edgecolor=BASE, linewidth=1.4, label="LocoTrack-B (base)")
    axL.bar([x + w / 2 for x in xs], [d[k] for k, _ in keys], w,
            color=MODEL, edgecolor="white", linewidth=0.6, label="Jeff-Tracker")
    for x, (k, _) in zip(xs, keys):
        axL.text(x - w / 2, ref[k] + 1.0, "{:.1f}".format(ref[k]), ha="center",
                 va="bottom", fontsize=8, color="#5b6670")
        axL.text(x + w / 2, d[k] + 1.0, "{:.1f}".format(d[k]), ha="center",
                 va="bottom", fontsize=8)
    axL.set_xticks(list(xs))
    axL.set_xticklabels([lab for _, lab in keys])
    axL.set_ylim(0, 100)
    axL.set_ylabel("higher is better")
    axL.set_title("TAP-Vid DAVIS, 30 clips, 256$\\times$256", fontsize=10)
    axL.legend(fontsize=8, frameon=False, loc="upper left")

    # ---- right: what the fine-tune bought ---------------------------------------------
    benches = r["occlusion"]["benches"]
    names = ["{}\n{:.0f}% cover".format(b["name"], b["frame_cover_pct"]) for b in benches]
    xs = range(len(benches))
    axR.bar([x - w / 2 for x in xs], [b["base"] for b in benches], w,
            color="white", edgecolor=BASE, linewidth=1.4, label="LocoTrack-B (base)")
    axR.bar([x + w / 2 for x in xs], [b["jefftracker"] for b in benches], w,
            color=MODEL, edgecolor="white", linewidth=0.6, label="Jeff-Tracker")
    for x, b in zip(xs, benches):
        axR.text(x - w / 2, b["base"] + 0.8, "{:.1f}".format(b["base"]), ha="center",
                 va="bottom", fontsize=8, color="#5b6670")
        axR.text(x + w / 2, b["jefftracker"] + 0.8, "{:.1f}".format(b["jefftracker"]),
                 ha="center", va="bottom", fontsize=8)
    axR.set_xticks(list(xs))
    axR.set_xticklabels(names, fontsize=8)
    axR.set_ylim(0, 100)
    axR.set_ylabel("occluded frames within 5 px  (%)")
    axR.set_title("Occlusion benches, exact ground truth  (mean +{} pts)".format(
        r["occlusion"]["mean_gain_points"]), fontsize=10)
    axR.legend(fontsize=8, frameon=False, loc="upper right")

    for ax in (axL, axR):
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("[plot] wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
