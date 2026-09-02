"""VRSBench referring-expression grounding.

The problem statement names VRSBench for single-image grounding. This measures
text-guided region localisation against real box supervision, using Acc@0.5 IoU
-- the standard referring-expression metric -- plus mean IoU, which shows
whether near-misses are close or nowhere.

A structural caveat belongs with the numbers rather than after them. Grounding
here comes from CLIP patch tokens on a 7x7 grid. Over a 512x512 image that is
73 pixels per cell, so the smallest box this method can express is roughly
73x73 -- and VRSBench refers to individual vehicles, which are far smaller than
one cell. The metric is therefore measuring a resolution limit as much as a
localisation ability, and the per-size breakdown makes that visible.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from satquery.registry import build_registry
from satquery.schema import ImageMeta, Modality

_PROMPT = re.compile(
    r"(?:<image>\s*)?(?:please\s+)?provide the bounding box coordinate of the region "
    r"this sentence describes:\s*", re.I)


def referring_sentence(problem: str) -> str:
    """Strip the instruction wrapper, leaving the expression itself."""
    return _PROMPT.sub("", str(problem)).replace("<image>", "").strip()


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def size_bucket(box: list[float], w: int, h: int) -> str:
    """How large the target is relative to one 7x7 patch cell."""
    cell = (w / 7.0) * (h / 7.0)
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    ratio = area / cell if cell else 0.0
    if ratio < 0.25:
        return "tiny (<1/4 cell)"
    if ratio < 1.0:
        return "sub-cell"
    if ratio < 4.0:
        return "1-4 cells"
    return "large (>4 cells)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bench/vrsbench_fs/val-0.parquet")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tiles", type=int, default=1,
                    help="T x T crop grid; 1 is a single pass")
    ap.add_argument("--min-area", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=0.6)
    ap.add_argument("--out", default="eval/results/vrsbench_grounding.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    if a.limit:
        df = df.head(a.limit)
    print(f"VRSBench referring grounding: {len(df)} expressions")

    tool = build_registry().get("rs_grounding")
    ok, why = tool.available()
    if not ok:
        raise SystemExit(f"rs_grounding unavailable: {why}")

    params = {"tiles": a.tiles, "min_area": a.min_area, "threshold": a.threshold}
    print(f"params: {params}")
    tmp = Path(tempfile.mkdtemp(prefix="vrsb-"))
    ious: list[float] = []
    by_size: dict[str, list[float]] = collections.defaultdict(list)
    misses = 0

    for i, row in enumerate(df.itertuples(), 1):
        blob = row.images[0]["bytes"]
        path = tmp / f"{i}.png"
        path.write_bytes(blob)
        from PIL import Image
        w, h = Image.open(io.BytesIO(blob)).size

        gold = [float(x) for x in np.asarray(row.answer).ravel()[:4]]
        sentence = referring_sentence(row.problem)
        meta = ImageMeta(path=str(path), fmt="PNG", width=w, height=h,
                         bands=3, modality=Modality.OPTICAL)

        _, _, evidence, _ = tool.invoke([meta], sentence, params)
        boxes = [e for e in evidence if e.kind == "bbox"]
        if not boxes:
            misses += 1
            score = 0.0
        else:
            # Normalised coordinates back to pixels for comparison.
            pred = [boxes[0].data[0] * w, boxes[0].data[1] * h,
                    boxes[0].data[2] * w, boxes[0].data[3] * h]
            score = iou(pred, gold)
        ious.append(score)
        by_size[size_bucket(gold, w, h)].append(score)
        path.unlink(missing_ok=True)

        if i % 200 == 0:
            print(f"  {i}/{len(df)}  mean IoU {np.mean(ious):.3f}")

    arr = np.array(ious)
    print(f"\n{'metric':<26}{'value':>10}")
    print(f"{'Acc@0.5 IoU':<26}{(arr >= 0.5).mean():>9.1%}")
    print(f"{'Acc@0.25 IoU':<26}{(arr >= 0.25).mean():>9.1%}")
    print(f"{'mean IoU':<26}{arr.mean():>10.3f}")
    print(f"{'no box produced':<26}{misses:>10}")

    print(f"\n{'target size':<22}{'n':>6}{'mean IoU':>10}{'Acc@0.5':>9}")
    rows = {}
    for bucket in sorted(by_size, key=lambda k: -len(by_size[k])):
        v = np.array(by_size[bucket])
        rows[bucket] = {"n": len(v), "mean_iou": float(v.mean()),
                        "acc50": float((v >= 0.5).mean())}
        print(f"{bucket:<22}{len(v):>6}{v.mean():>10.3f}{(v >= 0.5).mean():>8.1%}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"benchmark": "VRSBench referring grounding (omlab/VRSBench-FS val shard 0)",
         "n": len(arr), "acc50": float((arr >= 0.5).mean()),
         "acc25": float((arr >= 0.25).mean()), "mean_iou": float(arr.mean()),
         "no_box": misses, "params": params, "by_target_size": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
