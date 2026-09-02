"""Optical-SAR joint analysis on co-registered Sentinel-1 and Sentinel-2 pairs.

The problem statement makes this mandatory -- "The system must extract
complementary information from a co-registered optical/multispectral and SAR
image pair" -- and it is the one mandatory capability that had no measurement at
all, because BigEarthNet as we had it was optical-only.

"Complementary" is a testable claim, not a description, so the test is whether
fusing the two sensors beats either alone on the same task. Each patch carries
gold CORINE land-cover labels, and the same vocabulary is scored three ways:
through the optical image, through the SAR image, and fused. If fusion does not
win, the pair is not being used complementarily and saying so would be false.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import pandas as pd

from satquery.registry import build_registry
from satquery.schema import Evidence, ImageMeta, Modality
from satquery.tools.specialists import LAND_COVER


def build_index(root: Path) -> dict[str, Path]:
    """Map every patch stem to its file, once.

    Globbing per patch would rescan 27k files for each lookup. The archive also
    nests by sensor and split, so the flat index is what makes the join cheap.
    """
    return {path.stem: path for path in root.rglob("*.tif")}


def scores_at_k(pred: list[str], gold: set[str], k: int) -> tuple[float, float]:
    top = pred[:k]
    hit = len(set(top) & gold)
    return hit / max(len(top), 1), hit / max(len(gold), 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/bench/crossmodal/metadata.parquet")
    ap.add_argument("--root", default="data/bench/crossmodal")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/crossmodal.json")
    a = ap.parse_args()

    root = Path(a.root)
    index = build_index(root)
    if not index:
        raise SystemExit(f"no .tif files under {root}; is the archive extracted?")

    meta = pd.read_parquet(a.meta)
    if "split" in meta.columns:
        meta = meta[meta["split"] == "test"]
    # The archive holds a 14k subset of a 480k-row metadata table, and the
    # `s2v1_name` column is a collapsed form that does not match the files.
    # `patch_id` carries the full form the archive actually uses.
    meta = meta[meta["patch_id"].isin(index) & meta["s1_name"].isin(index)]
    rows = meta.to_dict("records")
    random.Random(a.seed).shuffle(rows)
    print(f"{len(rows)} co-registered test pairs available")
    tool = build_registry().get("rs_cross_modal")
    ok, why = tool.available()
    if not ok:
        raise SystemExit(f"rs_cross_modal unavailable: {why}")

    vocab_lower = {c.lower(): c for c in LAND_COVER}
    modes = ["optical", "sar", "mean", "max", "weighted"]
    prec: dict[str, list[float]] = collections.defaultdict(list)
    rec: dict[str, list[float]] = collections.defaultdict(list)
    used = 0

    for row in rows:
        if used >= a.limit:
            break
        # `labels` arrives as a numpy array, so truthiness on it raises.
        raw = row.get("labels")
        labels = [] if raw is None else list(raw)
        gold = {vocab_lower[str(l).lower()] for l in labels
                if str(l).lower() in vocab_lower}
        if not gold:
            continue
        s2 = index.get(str(row["patch_id"]))
        s1 = index.get(str(row["s1_name"]))
        if not s2 or not s1:
            continue

        metas = [
            ImageMeta(path=str(s2), fmt="GeoTIFF" if s2.suffix.startswith(".tif") else "PNG",
                      width=120, height=120, bands=12, modality=Modality.OPTICAL),
            ImageMeta(path=str(s1), fmt="GeoTIFF" if s1.suffix.startswith(".tif") else "PNG",
                      width=120, height=120, bands=2, modality=Modality.SAR),
        ]
        for mode in modes:
            _, _, ev, _ = tool.invoke(metas, "identify the land cover",
                                      {"vocab": LAND_COVER, "fusion": mode,
                                       "top_k": a.k})
            pred = [e.data for e in ev if isinstance(e, Evidence)
                    and e.kind == "label" and str(e.label).startswith("fused:")]
            p, r = scores_at_k(pred, gold, a.k)
            prec[mode].append(p)
            rec[mode].append(r)
        used += 1
        if used % 50 == 0:
            print(f"  {used}/{a.limit}  fused-mean P@{a.k} "
                  f"{sum(prec['mean'])/len(prec['mean']):.1%}")

    if not used:
        raise SystemExit("no usable pairs found; is the archive extracted?")

    print(f"\n{used} co-registered pairs, vocabulary of {len(LAND_COVER)} CORINE classes")
    print(f"\n{'reading':<16}{'P@' + str(a.k):>9}{'R@' + str(a.k):>9}")
    out = {}
    for mode in modes:
        p = sum(prec[mode]) / len(prec[mode])
        r = sum(rec[mode]) / len(rec[mode])
        out[mode] = {"precision_at_k": p, "recall_at_k": r}
        label = {"optical": "optical only", "sar": "SAR only",
                 "mean": "fused (mean)", "max": "fused (max)",
                 "weighted": "fused (weighted)"}[mode]
        print(f"{label:<16}{p:>8.1%}{r:>9.1%}")

    best_single = max(out["optical"]["precision_at_k"], out["sar"]["precision_at_k"])
    best_fused = max(out["mean"]["precision_at_k"], out["max"]["precision_at_k"],
                     out["weighted"]["precision_at_k"])
    gain = best_fused - best_single
    print(f"\nfusion gain over the better single sensor: {gain:+.1%}")
    print("complementary:" , "yes" if gain > 0 else "no — fusion does not beat the better sensor")

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"benchmark": "BigEarthNet co-registered S1/S2",
                                "n": used, "k": a.k, "by_reading": out,
                                "fusion_gain": gain}, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
