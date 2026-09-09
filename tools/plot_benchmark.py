"""Chart the benchmark JSON: accuracy bars, and accuracy against cost.

    python tools/plot_benchmark.py --json out/benchmark.json --out assets/benchmark.png

Two panels, because one of them alone misleads:

  left   AJ / delta_avg / OA side by side. The headline comparison.
  right  AJ against seconds-per-frame, log x. A tracker that wins by 2 AJ at 6x the cost
         has not simply won, and a bar chart cannot say that.

Every bar is annotated with its value -- a reader should not have to measure a pixel height
against an axis to quote a number.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Colour-blind-safe, and deliberately not red/green: the point is to compare heights, not
# to code one model as good and another as bad.
COLORS = {
    "jefftracker": "#0072B2",
    "tapnext": "#E69F00",
}
LABEL = {
    "jefftracker": "Jeff-Tracker\n(Apache-2.0)",
    "tapnext": "TAPNext++\n(Apache-2.0)",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="chart the benchmark JSON")
    ap.add_argument("--json", default="out/benchmark.json")
    ap.add_argument("--out", default="assets/benchmark.png")
    ap.add_argument("--title", default="TAP-Vid DAVIS, query-first, 256x256")
    a = ap.parse_args()

    with open(a.json) as fh:
        blob = json.load(fh)
    rows = blob["summary"]
    n_clips = blob.get("clips", "?")

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.4))
    fig.patch.set_facecolor("white")

    metrics = [("AJ", "AJ"), ("delta_avg", "$\\delta_{avg}$"), ("OA", "OA")]
    n = len(rows)
    width = 0.8 / max(1, n)
    for i, r in enumerate(rows):
        xs = [j + (i - (n - 1) / 2.0) * width for j in range(len(metrics))]
        ys = [r[k] for k, _ in metrics]
        axL.bar(xs, ys, width * 0.92, label=LABEL.get(r["model"], r["model"]),
                color=COLORS.get(r["model"], "#666666"), edgecolor="white", linewidth=0.6)
        for x, y in zip(xs, ys):
            axL.text(x, y + 0.7, "{:.1f}".format(y), ha="center", va="bottom", fontsize=8)

    axL.set_xticks(range(len(metrics)))
    axL.set_xticklabels([lab for _, lab in metrics])
    axL.set_ylabel("higher is better")
    axL.set_ylim(0, 100)
    axL.set_title("Accuracy  ({} DAVIS clips)".format(n_clips), fontsize=10)
    axL.legend(fontsize=7.5, frameon=False, ncol=1, loc="upper left")
    axL.grid(axis="y", alpha=0.25, linewidth=0.6)
    axL.set_axisbelow(True)
    for s in ("top", "right"):
        axL.spines[s].set_visible(False)

    for r in rows:
        axR.scatter(r["seconds_per_frame"], r["AJ"], s=120,
                    color=COLORS.get(r["model"], "#666666"), zorder=3,
                    edgecolor="white", linewidth=1.2)
        axR.annotate("{}  ({:.1f} AJ, {:.3f} s/f)".format(
                         r["model"], r["AJ"], r["seconds_per_frame"]),
                     (r["seconds_per_frame"], r["AJ"]), textcoords="offset points",
                     xytext=(9, -3), fontsize=8)
    axR.set_xscale("log")
    axR.set_xlabel("seconds per frame  (log, lower is better)")
    axR.set_ylabel("AJ")
    axR.set_title("Accuracy against cost", fontsize=10)
    axR.grid(alpha=0.25, linewidth=0.6)
    axR.set_axisbelow(True)
    for s in ("top", "right"):
        axR.spines[s].set_visible(False)
    lo = min(r["AJ"] for r in rows)
    hi = max(r["AJ"] for r in rows)
    pad = max(2.0, (hi - lo) * 0.6)
    axR.set_ylim(lo - pad, hi + pad)
    # log x with the label drawn to the RIGHT of each point, so the rightmost annotation
    # needs room or it runs off the axes.
    xs = [r["seconds_per_frame"] for r in rows]
    axR.set_xlim(min(xs) / 2.2, max(xs) * 9.0)

    fig.suptitle(a.title, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print("[plot] wrote {}".format(a.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
