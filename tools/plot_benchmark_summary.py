"""Render the README benchmark figures from docs/benchmark_summary.json.

Every number comes from the JSON, which names the document or result file each panel is
taken from; per-bench arrays are averaged here rather than by hand. Each panel states its
winner, including the panels Jeff-Tracker does not win.

    python tools/plot_benchmark_summary.py
    python tools/plot_benchmark_summary.py --json docs/benchmark_summary.json --only assets/benchmark_synthetic.png

Series colours keep the repository's validated slot order (Jeff-Tracker, LocoTrack-B,
Baseline, TAPNext++), checked with the dataviz palette validator against the dark panel
surface: worst adjacent CVD delta-E 8.4, normal-vision 19.8, all slots >= 3:1 contrast.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SURFACE = "#1a1a19"
PANEL = "#232322"
INK = "#ffffff"
INK2 = "#c3c2b7"
INK3 = "#87867e"
GRID = "#3a3a38"

BAR_H = 0.56          # share of a row a bar takes; the rest is air between rows
DPI = 100
ROW_IN = 0.34         # inches per bar row, the same in every panel of a figure
HEAD_IN = 1.25        # figure title + subtitle
PANEL_HEAD_IN = 1.05  # panel title + winner line + note
FOOT_IN = 0.75        # x tick labels + axis label


def fmt(v: float) -> str:
    if v >= 100:
        return "{:.0f}".format(v)
    if v >= 10:
        return "{:.1f}".format(v)
    if v >= 1:
        return "{:.2f}".format(v)
    return "{:.3f}".format(v)


def ratio_text(best: float, other: float, lower_is_better: bool) -> str:
    r = other / best if lower_is_better else best / other
    return "{:.1f}x {}".format(r, "lower" if lower_is_better else "higher")


def style_axes(ax, title, note, winner_line, winner_is_jt):
    ax.set_facecolor(PANEL)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", colors=INK2, labelsize=10.5, length=0, pad=8)
    ax.tick_params(axis="x", colors=INK3, labelsize=9, length=0)
    # Header lines are placed in points above the axes, so they never depend on panel height.
    for dy, text, colour, size, weight in ((52, title, INK, 13, "bold"),
                                           (30, winner_line, INK if winner_is_jt else INK2, 11,
                                            "bold" if winner_is_jt else "normal"),
                                           (10, note, INK3, 9, "normal")):
        if text:
            ax.annotate(text, xy=(0, 1), xycoords="axes fraction", xytext=(0, dy),
                        textcoords="offset points", color=colour, fontsize=size,
                        fontweight=weight, va="bottom", ha="left")


def draw_bars(ax, rows, xmax, unit):
    """rows: list of (y, value, colour, label_text, is_winner)."""
    for y, v, colour, text, win in rows:
        ax.barh(y, v, height=BAR_H, color=colour, zorder=3, linewidth=0)
        ax.text(v + xmax * 0.012, y, text + ("   ✓ best" if win else ""),
                va="center", ha="left", color=INK if win else INK2, fontsize=10,
                fontweight="bold" if win else "normal")
    ax.set_xlim(0, xmax * 1.42)
    ax.set_xlabel(unit, color=INK3, fontsize=9)


def panel_speedup(ax, p, engines):
    base, jt = p["series"]["baseline"], p["series"]["jefftrack"]
    speed = [b / j for b, j in zip(base, jt)]
    n = len(speed)
    gap = 2.0                          # two-line plate labels need two rows of air
    rows = [((n - 1 - i) * gap, s, engines["jefftrack"]["color"], "{:.1f}x".format(s), False)
            for i, s in enumerate(speed)]
    xmax = max(speed)
    draw_bars(ax, rows, xmax, "times faster than the baseline (baseline = 1x)")
    ax.axvline(1.0, color=INK3, linewidth=1.2, zorder=4)
    ax.set_yticks([i * gap for i in range(n)])
    ax.set_yticklabels(list(reversed(p["groups"])))
    ax.extent = (-0.5, (n - 1) * gap + 0.5)
    wins = sum(s > 1.0 for s in speed)
    style_axes(ax, p["title"], p.get("note", "seconds per shot, baseline runtime / Jeff-Tracker runtime"),
               "Winner: Jeff-Tracker, faster on {} of {} plates".format(wins, n), wins == n)


def winner_of(values: dict, lower: bool):
    order = sorted(values.items(), key=lambda kv: kv[1], reverse=not lower)
    return order[0], order[1]


def panel_grouped(ax, p, engines):
    """Several groups (plates), two or more engines per group."""
    lower = p["lower_is_better"]
    keys = list(p["series"])
    groups = p["groups"]
    k = len(keys)
    step = k + 0.9                     # rows per group plus a gap between groups
    rows, ticks, wins = [], [], {key: 0 for key in keys}
    xmax = max(max(v) for v in p["series"].values())
    for gi, g in enumerate(groups):
        vals = {key: p["series"][key][gi] for key in keys}
        (best, _), _ = winner_of(vals, lower)
        wins[best] += 1
        top = (len(groups) - 1 - gi) * step
        for si, key in enumerate(keys):
            y = top + (k - 1 - si) * (1.0 / BAR_H) * BAR_H
            rows.append((y, vals[key], engines[key]["color"],
                         "{}  {}".format(engines[key]["label"], fmt(vals[key])), key == best))
        ticks.append((top + (k - 1) / 2.0, g))
    draw_bars(ax, rows, xmax, p["unit"] + (", lower is better" if lower else ", higher is better"))
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([g for _, g in ticks])
    ax.extent = (-0.5, (len(groups) - 1) * step + k - 0.5)
    champ = max(wins, key=wins.get)
    if len(groups) == 1:
        vals = {key: p["series"][key][0] for key in keys}
        (best, bv), (_, rv) = winner_of(vals, lower)
        line = "Winner: {}, {}".format(engines[best]["label"], ratio_text(bv, rv, lower))
    else:
        line = "Winner: {}, best on {} of {} plates".format(
            engines[champ]["label"], wins[champ], len(groups))
    style_axes(ax, p["title"], p.get("note", ""), line, champ == "jefftrack")


def panel_single(ax, p, engines):
    """One value per engine (per-bench arrays averaged)."""
    lower = p["lower_is_better"]
    vals = {key: sum(v) / len(v) for key, v in p["series"].items()}
    (best, bv), (_, rv) = winner_of(vals, lower)
    keys = list(vals)
    n = len(keys)
    rows = [(n - 1 - i, vals[key], engines[key]["color"], fmt(vals[key]), key == best)
            for i, key in enumerate(keys)]
    draw_bars(ax, rows, max(vals.values()),
              p["unit"] + (", lower is better" if lower else ", higher is better"))
    ax.set_yticks(range(n))
    ax.set_yticklabels([engines[key]["label"] for key in reversed(keys)])
    ax.extent = (-0.5, n - 0.5)
    note = p.get("note", "mean of four benches" if p.get("per_bench") else "")
    style_axes(ax, p["title"], note,
               "Winner: {}, {}".format(engines[best]["label"], ratio_text(bv, rv, lower)),
               best == "jefftrack")


def render(fig_spec, engines, out_path):
    panels = fig_spec["panels"]
    n = len(panels)
    fig = plt.figure(dpi=DPI)
    fig.patch.set_facecolor(SURFACE)
    axes = [fig.add_subplot(1, n, i + 1) for i in range(n)]
    for ax, p in zip(axes, panels):
        if p.get("kind") == "speedup":
            panel_speedup(ax, p, engines)
        elif "groups" in p and not p.get("per_bench"):
            panel_grouped(ax, p, engines)
        else:
            panel_single(ax, p, engines)
    # One row scale for the whole figure, so a two-bar panel does not get bars twice as fat.
    span = max(ax.extent[1] - ax.extent[0] for ax in axes)
    for ax in axes:
        mid = (ax.extent[0] + ax.extent[1]) / 2.0
        ax.set_ylim(mid - span / 2.0, mid + span / 2.0)
    plot_in = ROW_IN * span
    height = HEAD_IN + PANEL_HEAD_IN + plot_in + FOOT_IN
    width = 5.6 * n
    fig.set_size_inches(width, height)
    fig.subplots_adjust(left=fig_spec.get("left_in", 1.55) / width, right=0.985,
                        top=1.0 - (HEAD_IN + PANEL_HEAD_IN) / height,
                        bottom=FOOT_IN / height, wspace=fig_spec.get("wspace", 0.62))
    fig.text(0.012, 1.0 - 0.3 / height, fig_spec["title"], color=INK, fontsize=17,
             fontweight="bold", va="top")
    fig.text(0.012, 1.0 - 0.72 / height, fig_spec["subtitle"], color=INK2, fontsize=10.5,
             va="top")
    fig.savefig(out_path, facecolor=SURFACE)
    plt.close(fig)
    print("wrote", out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", default=os.path.join(ROOT, "docs", "benchmark_summary.json"))
    ap.add_argument("--only", default="", help="render just this output file")
    a = ap.parse_args()
    with open(a.json, encoding="utf-8") as fh:
        spec = json.load(fh)
    for fig_spec in spec["figures"]:
        if a.only and os.path.normpath(a.only) != os.path.normpath(fig_spec["file"]):
            continue
        render(fig_spec, spec["engines"], os.path.join(ROOT, fig_spec["file"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
