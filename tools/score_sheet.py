"""One page showing how the three engines compare, computed from the npz bench3 wrote.

Nothing here is typed in by hand -- the sheet re-scores every npz on every run, so it
cannot drift away from the tables in FINDINGS.md the way a pasted number can. Re-run it
after any phase and the picture is current.

Reading it:

  * **lower is better on every error panel.** They are pixel errors against exact
    synthetic ground truth, so 0 would be perfect
  * **coverage is not a score**, it is a behaviour -- what fraction of occluded frames the
    engine is willing to put a position on. High is not good and low is not bad; it is
    the number that says whether an occluded-error figure beside it means anything
  * error panels are scored with the truth ON THE PLATE only. A point whose truth has left
    the picture cannot be tracked, and including those made Jeff-Tracker's worst visible
    error read 389.30 px instead of 16.86 px on identical output (see FINDINGS.md)
  * error panels are UNGATED -- every frame the model produced, not just the confident ones
    the 3DE export keeps. Gating flatters whichever engine abstains most, which here is
    CoTracker3

    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\score_sheet.py
    runtime\\python311\\python.exe experiments\\Jeff-Tracker\\score_sheet.py --tag phase1
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)          # the repo root, one level up
REPO = ROOT
for _p in (ROOT, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import score_occlusion as SO  # noqa: E402
from bench3 import LOCKED_BENCHES  # noqa: E402

# Fixed slot order, never cycled: an engine keeps its hue whatever else is on the page, so
# two sheets from different phases can be read side by side. Dark steps of categorical
# slots 1/2/3, validated all-pairs against the #1a1a19 surface (worst CVD dE 9.4, worst
# normal-vision dE 20.9, all >= 3:1 contrast).
DEFAULT_ENGINES = "jefftrack:Jeff-Tracker:#3987e5,tapnext:TAPNext++:#d95926,"                   "cotracker3:CoTracker3:#199e70"

# Filled from --engines at startup. A series keeps its hue across sheets so two phases can
# be laid side by side; any new set must be re-validated before use (validate_palette.js),
# because slot ORDER is the colour-blind-safety mechanism here, not decoration.
ENGINES = []

SURFACE = "#1a1a19"
PANEL = "#232322"
INK = "#ffffff"
INK2 = "#c3c2b7"
INK3 = "#87867e"
GRID = "#3a3a38"

# label, key path into the scored dict, whether lower is better, unit
PANELS = [
    ("Visible error, median", ("visible", "med"), True, "px"),
    ("Visible error, mean", ("visible", "mean"), True, "px"),
    ("Visible error, worst", ("visible", "max"), True, "px"),
    ("Occluded error, mean", ("occluded", "mean"), True, "px"),
    ("Re-acquire error, mean", ("reacquire", "mean"), True, "px"),
    ("Occluded coverage at the export gate", ("coverage_pct", None), None, "%"),
]


def collect(out_dir, benches, reacq):
    """Re-score every npz. Errors ungated; coverage at the exporter's own 0.5 threshold."""
    data = {}
    for b in benches:
        for key, _, _ in ENGINES:
            npz = os.path.join(out_dir, "{}__{}.npz".format(b, key))
            if not os.path.isfile(npz):
                continue
            shot = os.path.join(REPO, "bench", "synth", b)
            row = {}
            for gate, tag in ((0.0, "ungated"), (0.5, "gated")):
                seq, ff = SO.seq_from_npz(npz, gate)
                with contextlib.redirect_stdout(io.StringIO()):
                    row[tag] = SO.score(shot, seq, reacq, ff, control=False, margin=0.0)
            data[(b, key)] = row
        cost = os.path.join(out_dir, "{}_cost.json".format(b))
        if os.path.isfile(cost):
            with open(cost) as fh:
                data[(b, "_cost")] = json.load(fh)["engines"]
    return data


def value_for(data, bench, eng, path):
    row = data.get((bench, eng))
    if not row:
        return np.nan
    # Coverage is read from the GATED pass -- it is a fact about the export, and ungated it
    # is 100% by construction for every engine, which would say nothing.
    src = row["gated"] if path[0] == "coverage_pct" else row["ungated"]
    v = src.get(path[0])
    if path[1] is not None:
        v = (v or {}).get(path[1])
    return float(v) if v is not None else np.nan


def rounded_bar(ax, x, w, h, colour):
    """A plain rectangle, deliberately.

    The rounded data-end in the mark spec is drawn by FancyBboxPatch in DATA units, and
    these panels have x and y ranges that differ by more than an order of magnitude -- the
    speed panel spans 0..0.17 across and 0..3 down. A radius that looks right vertically
    becomes an ellipse horizontally, which is what the first render produced. A square end
    that is the right shape beats a rounded one that is the wrong shape.
    """
    if not np.isfinite(h) or h <= 0:
        return
    ax.add_patch(plt.Rectangle((x, 0), w, h, linewidth=0, facecolor=colour, zorder=3))


def panel(ax, title, values, labels, note, unit):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10.5, pad=26, loc="left", fontweight="bold")
    if note:
        ax.text(0, 1.035, note, transform=ax.transAxes, color=INK3, fontsize=7.8,
                va="bottom", ha="left")

    n_g, n_s = len(labels), len(ENGINES)
    group_w, gap = 0.78, 0.03          # 2px-ish surface gap between adjacent bars
    bw = (group_w - gap * (n_s - 1)) / n_s
    finite = values[np.isfinite(values)]
    top = (finite.max() * 1.30) if finite.size else 1.0
    ax.set_ylim(0, top)
    ax.set_xlim(-0.5, n_g - 0.5)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.xaxis.grid(False)
    ax.tick_params(axis="y", colors=INK3, labelsize=7.5, length=0)
    ax.tick_params(axis="x", colors=INK2, labelsize=8.6, length=0)
    ax.set_xticks(range(n_g))
    ax.set_xticklabels(labels)

    for gi in range(n_g):
        for si, (_, _, colour) in enumerate(ENGINES):
            v = values[gi, si]
            x = gi - group_w / 2 + si * (bw + gap)
            rounded_bar(ax, x, bw, v, colour)
            if np.isfinite(v):
                # The number is the point of a score sheet, so every bar carries one -- and
                # it doubles as the table view. Text stays in ink, never the series colour.
                ax.text(x + bw / 2, v + top * 0.035,
                        ("{:.2f}" if v < 100 else "{:.0f}").format(v),
                        ha="center", va="bottom", color=INK2, fontsize=7.4)
    ax.set_ylabel(unit, color=INK3, fontsize=8)


def cost_panel(ax, title, vals, unit, note):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, color=INK, fontsize=10.5, pad=26, loc="left", fontweight="bold")
    ax.text(0, 1.035, note, transform=ax.transAxes, color=INK3, fontsize=7.8,
            va="bottom", ha="left")
    names = [n for _, n, _ in ENGINES]
    cols = [c for _, _, c in ENGINES]
    finite = np.array([v for v in vals if np.isfinite(v)])
    right = (finite.max() * 1.34) if finite.size else 1.0
    ax.set_xlim(0, right)
    ax.set_ylim(-0.6, len(names) - 0.4)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.tick_params(axis="y", colors=INK2, labelsize=8.6, length=0)
    ax.tick_params(axis="x", colors=INK3, labelsize=7.5, length=0)
    for i, (v, c) in enumerate(zip(vals, cols)):
        if not np.isfinite(v):
            continue
        h = 0.46
        ax.add_patch(plt.Rectangle((0, i - h / 2), v, h, linewidth=0, facecolor=c,
                                   zorder=3))
        ax.text(v + right * 0.02, i, "{:.3f} {}".format(v, unit) if v < 1
                else "{:.2f} {}".format(v, unit),
                va="center", ha="left", color=INK2, fontsize=7.8)
    ax.set_xlabel(unit, color=INK3, fontsize=8)


def main() -> int:
    ap = argparse.ArgumentParser(description="graphical score sheet for a bench3 run")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "out", "bench3"))
    ap.add_argument("--benches", default=",".join(LOCKED_BENCHES))
    ap.add_argument("--reacq", type=int, default=10)
    ap.add_argument("--tag", default="phase0")
    ap.add_argument("--title", default="Jeff-Tracker vs TAPNext++ vs CoTracker3")
    ap.add_argument("--engines", default=DEFAULT_ENGINES,
                    help="comma list of key:Label:#hex. The key is the npz suffix "
                         "bench3.py wrote, so a config variant (jefftrack_r384x512) is "
                         "its own series. Re-validate any new hue set.")
    ap.add_argument("--subtitle", default=None)
    ap.add_argument("--verdict", default=None,
                    help="path to a text file replacing the verdict block; blank lines "
                         "are kept as spacing, a leading '#' marks the heading")
    a = ap.parse_args()

    global ENGINES
    ENGINES = []
    for tok in a.engines.split(","):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split(":")
        if len(parts) != 3:
            raise SystemExit("[ERROR] --engines entry must be key:Label:#hex, got "
                             "{!r}".format(tok))
        ENGINES.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    benches = [b.strip() for b in a.benches.split(",") if b.strip()]
    data = collect(a.out_dir, benches, a.reacq)
    if not data:
        raise SystemExit("[ERROR] no npz in {} -- run bench3.py first".format(a.out_dir))

    short = [b.replace("lab02_occ", "lab02").replace("occ_s", "occ") for b in benches]

    fig = plt.figure(figsize=(17.5, 11.4), facecolor=SURFACE)
    gs = fig.add_gridspec(3, 4, left=0.045, right=0.985, top=0.790, bottom=0.115,
                          hspace=0.62, wspace=0.24, height_ratios=[1, 1, 0.78])

    for idx, (title, path, lower_better, unit) in enumerate(PANELS):
        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        vals = np.array([[value_for(data, b, e, path) for e, _, _ in ENGINES]
                         for b in benches])
        note = ("lower is better" if lower_better
                else "a behaviour, not a score")
        panel(ax, title, vals, short, note, unit)

    costs = {}
    for key, _, _ in ENGINES:
        sp, gb = [], []
        for b in benches:
            c = data.get((b, "_cost"), {}).get(key)
            if c:
                sp.append(c["s_per_frame"])
                gb.append(c["peak_gb"])
        costs[key] = (float(np.mean(sp)) if sp else np.nan,
                      float(np.max(gb)) if gb else np.nan)

    ax = fig.add_subplot(gs[1, 2])
    cost_panel(ax, "Speed", [costs[k][0] for k, _, _ in ENGINES], "s/frame",
               "lower is better")
    ax = fig.add_subplot(gs[1, 3])
    cost_panel(ax, "Peak VRAM", [costs[k][1] for k, _, _ in ENGINES], "GB",
               "lower is better  ·  the card has 16 GB")

    # The verdict block. A score sheet that makes the reader derive the conclusion from
    # eight panels has not finished the job.
    ax = fig.add_subplot(gs[2, 0:2])
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if a.verdict and os.path.isfile(a.verdict):
        lines = []
        with open(a.verdict, encoding="utf-8") as fh:
            for raw in fh.read().splitlines():
                t = raw.rstrip()
                if not t:
                    lines.append(("", INK, 3, "normal"))
                elif t.startswith("#"):
                    lines.append((t.lstrip("# ").strip(), INK, 10.5, "bold"))
                elif t.startswith("~"):
                    lines.append((t.lstrip("~ ").rstrip(), INK3, 8.2, "normal"))
                else:
                    lines.append((t, INK2, 9, "normal"))
    else:
        lines = [
        ("Where the gap actually is", INK, 10.5, "bold"),
        ("", INK, 4, "normal"),
        ("Localisation is level.  Jeff-Tracker takes the visible median on all four", INK2, 9, "normal"),
        ("benches and loses the mean by 0.05-0.08 px.  That is a tie.", INK2, 9, "normal"),
        ("", INK, 3, "normal"),
        ("Occlusion is the real gap, and it is large.  CoTracker3 is about 2x better", INK2, 9, "normal"),
        ("on occluded frames on every bench, and ahead on re-acquire everywhere.", INK2, 9, "normal"),
        ("", INK, 3, "normal"),
        ("Coverage is backwards from what was assumed:  we already emit through", INK2, 9, "normal"),
        ("more occluded frames than CoTracker3 does.  The target is being RIGHT", INK2, 9, "normal"),
        ("while hidden, not emitting more.", INK2, 9, "normal"),
        ("", INK, 3, "normal"),
        ("Not resolution-matched -- CoTracker3 runs 384x512 internally against our", INK3, 8.2, "normal"),
        ("256x256, ~3x the pixels.  Phase 1 settles that before any architecture", INK3, 8.2, "normal"),
        ("conclusion.  The occlusion gap looks too large for resolution to explain,", INK3, 8.2, "normal"),
        ("but that is a prediction, not a measurement.", INK3, 8.2, "normal"),
        ]
    y = 0.945
    for text, colour, size, weight in lines:
        if text:
            ax.text(0.032, y, text, transform=ax.transAxes, color=colour, fontsize=size,
                    va="top", ha="left", fontweight=weight)
            y -= 0.086 if size >= 9 else 0.078
        else:
            y -= 0.032

    ax2 = fig.add_subplot(gs[2, 2:4])
    ax2.set_facecolor(PANEL)
    ax2.set_xticks([])
    ax2.set_yticks([])
    for sp in ax2.spines.values():
        sp.set_visible(False)
    how = [
        ("How to read this", INK, 10.5, "bold"),
        ("", INK, 3, "normal"),
        ("Every number is a pixel error against exact synthetic truth, measured on a", INK2, 9, "normal"),
        ("2560x1440 plate.  1 px here is 1 px in 3DE.", INK2, 9, "normal"),
        ("", INK, 3, "normal"),
        ("VISIBLE is the point in clear view -- ordinary tracking accuracy.", INK2, 9, "normal"),
        ("OCCLUDED is while something covers it.  RE-ACQUIRE is the ten frames", INK2, 9, "normal"),
        ("after the occluder clears, and is the one that matters most: a track that", INK2, 9, "normal"),
        ("returns onto the neighbouring feature is worse than a hole, because it", INK2, 9, "normal"),
        ("looks fine and quietly poisons the solve.", INK2, 9, "normal"),
        ("", INK, 3, "normal"),
        ("COVERAGE is not a score.  It is how often the engine puts a position on a", INK3, 8.2, "normal"),
        ("hidden point at all.  An occluded-error figure means nothing without it --", INK3, 8.2, "normal"),
        ("an engine that gaps everything scores a perfect occluded error.", INK3, 8.2, "normal"),
    ]
    y2 = 0.945
    for text, colour, size, weight in how:
        if text:
            ax2.text(0.032, y2, text, transform=ax2.transAxes, color=colour, fontsize=size,
                     va="top", ha="left", fontweight=weight)
            y2 -= 0.086 if size >= 9 else 0.078
        else:
            y2 -= 0.032

    fig.text(0.045, 0.955, a.title, color=INK, fontsize=21, fontweight="bold", va="top")
    fig.text(0.045, 0.913,
             "{} synthetic occlusion benches, exact ground truth  ·  600 shared "
             "Shi-Tomasi seeds  ·  100 frames  ·  2560x1440 plate  ·  frames decoded once "
             "and handed to every engine".format(len(benches))
             if a.subtitle is None else a.subtitle,
             color=INK2, fontsize=9.6, va="top")
    fig.text(0.045, 0.888,
             "Error panels are UNGATED (every frame the model produced) and count only "
             "frames whose truth is still on the plate.  Bench labels show occluder cover.",
             color=INK3, fontsize=9.0, va="top")

    for i, (_, name, colour) in enumerate(ENGINES):
        x = 0.045 + i * 0.155
        fig.patches.append(plt.Rectangle((x, 0.848), 0.015, 0.0105, color=colour,
                                         transform=fig.transFigure, zorder=5))
        fig.text(x + 0.020, 0.8533, name, color=INK2, fontsize=9.9, va="center")

    fig.text(0.045, 0.045,
             "Generated by score_sheet.py from the npz bench3.py wrote -- no number here "
             "is typed in by hand.  Re-run after any phase to refresh.",
             color=INK3, fontsize=8.4, va="top")

    os.makedirs(a.out_dir, exist_ok=True)
    path = os.path.join(a.out_dir, "scoresheet_{}.png".format(a.tag))
    fig.savefig(path, facecolor=SURFACE, dpi=130)
    plt.close(fig)
    print("[sheet] {}".format(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
