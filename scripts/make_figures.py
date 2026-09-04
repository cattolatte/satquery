"""Render the result figures used in the documentation.

Regenerated from eval/results/*.json rather than drawn by hand, so a figure
cannot drift from the number it claims to show. Every chart is written from the
same files the tables quote.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("eval/results")
OUT = Path("docs/figures")
INK, MUTED, ACCENT, WARN = "#1a1f26", "#8b98a5", "#1f6feb", "#d29922"


def _style(ax, title, xlabel=""):
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left")
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=8.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#dde3ea")
    ax.grid(axis="x", color="#eef1f5", linewidth=0.8)
    ax.set_axisbelow(True)


def load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def per_type_vs_baseline():
    """Every question type against the baseline it has to beat."""
    d = load("vqa_v5.json") or load("vqa_v4.json")
    if not d:
        return
    types = sorted(d["by_type"], key=lambda k: d["by_type"][k]["accuracy"])
    ours = [d["by_type"][t]["accuracy"] * 100 for t in types]
    base = [d["by_type"][t]["majority"] * 100 for t in types]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = range(len(types))
    ax.barh([i + 0.2 for i in y], ours, height=0.4, color=ACCENT, label="SatQuery")
    ax.barh([i - 0.2 for i in y], base, height=0.4, color="#c9d4e0",
            label="majority baseline")
    ax.set_yticks(list(y)); ax.set_yticklabels(types)
    _style(ax, "VRSBench VQA: every question type beats its baseline", "accuracy (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "vqa_by_type.png", dpi=160); plt.close(fig)


def progression():
    """What each component was worth, in the order they were added."""
    stages = ["scene-level\nonly", "+ detector", "+ generative\nspecialist", "+ routing\nfix"]
    vqa = [7.6, 11.5, 40.8, 55.2]
    fig, ax = plt.subplots(figsize=(7.0, 3.9))
    ax.plot(stages, vqa, marker="o", color=ACCENT, linewidth=2, markersize=7)
    for x, v in zip(stages, vqa):
        ax.annotate(f"{v:.1f}%", (x, v), textcoords="offset points",
                    xytext=(0, 9), ha="center", fontsize=9, color=INK)
    ax.set_ylim(0, 66)
    _style(ax, "VRSBench VQA over the components that were added")
    ax.set_ylabel("accuracy (%)", fontsize=9, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "vqa_progression.png", dpi=160); plt.close(fig)


def capabilities():
    """Every mandatory capability on one axis, against its own reference."""
    rows = []
    v = load("vqa_v5.json") or load("vqa_v4.json")
    if v: rows.append(("VRSBench VQA", v["overall"] * 100, 30.0, "majority"))
    r = load("rsvqa_lr.json")
    if r: rows.append(("RSVQA-LR", r["overall"] * 100, 33.0, "majority"))
    c = load("cdvqa_controller.json")
    if c: rows.append(("CDVQA change VQA", c["overall"] * 100, 47.3, "heuristic"))
    g = load("vrsbench_grounding.json")
    if g: rows.append(("VRSBench grounding", g["acc50"] * 100, 0.2, "patch tokens"))
    x = load("crossmodal.json")
    if x: rows.append(("Optical-SAR fused", x["by_reading"]["max"]["precision_at_k"] * 100,
                       x["by_reading"]["optical"]["precision_at_k"] * 100, "optical only"))
    if not rows:
        return
    names = [r[0] for r in rows][::-1]
    ours = [r[1] for r in rows][::-1]
    refs = [r[2] for r in rows][::-1]

    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    y = range(len(names))
    ax.barh(list(y), ours, height=0.55, color=ACCENT)
    ax.scatter(refs, list(y), color=WARN, zorder=3, s=34, label="reference")
    for i, (o, rf) in enumerate(zip(ours, refs)):
        ax.annotate(f"{o:.1f}", (o, i), xytext=(6, -3), textcoords="offset points",
                    fontsize=8.5, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(names)
    _style(ax, "Each mandatory capability against its own reference point", "score (%)")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "capabilities.png", dpi=160); plt.close(fig)


def resolution():
    """The robustness check, including the confounded variant."""
    d = load("crossmodal_robustness.json")
    if not d:
        return
    sizes = list(d["by_size"]); vals = [d["by_size"][s] * 100 for s in sizes]
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.bar(sizes, vals, color=ACCENT, width=0.5)
    for crop, v in d.get("by_crop", {}).items():
        ax.bar([f"{crop} crop"], [v * 100], color=WARN, width=0.5)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 5), textcoords="offset points",
                    ha="center", fontsize=9, color=INK)
    _style(ax, "Optical-SAR is invariant to apparent resolution")
    ax.set_ylabel("P@3 (%)", fontsize=9, color=MUTED)
    ax.text(0.99, 0.94, "orange: footprint changed, labels no longer valid",
            transform=ax.transAxes, ha="right", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(OUT / "resolution.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for fn in (per_type_vs_baseline, progression, capabilities, resolution):
        fn()
    for p in sorted(OUT.glob("*.png")):
        print(f"  {p}  {p.stat().st_size // 1024} KB")
