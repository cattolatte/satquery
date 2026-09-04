"""Is optical-SAR joint reading sensitive to ground resolution?

The graded ISRO/SAC set is Cartosat-2S optical and RISAT SAR -- sub-metre
against the 10 m/px Sentinel-2 everything here was adapted on. That is the
distribution shift this project has been caught by at every previous stage, so
it was worth testing before the judges did.

The answer is no, and getting there needed a correction.

The first version cropped the centre of a patch and rescaled it, which does
shrink the ground footprint the way a finer sensor would. P@3 fell from 33.1%
to 7.8% at 4x and the script called the model resolution-sensitive. That was
wrong. Cropping to a sixteenth of the area also invalidates the labels, which
describe the whole 1.2 km patch and not the 300 m crop -- so the experiment
measured its own artefact.

Rescaling the whole patch changes apparent resolution while keeping footprint
and labels exactly valid. Under that test the reading does not move at all:

    60 px   P@3 33.1%
    120 px  P@3 33.1%   (native)
    480 px  P@3 33.1%

Identical to three figures, which is what one should expect: the processor
resamples every input to 224x224, so apparent resolution is normalised away
before the encoder sees it. Resolution is not the ISRO risk.

What remains genuinely unknown is sensor character -- RISAT's speckle,
incidence angle and polarimetry against Sentinel-1's -- and scene content at a
much smaller footprint. Neither is testable without the data, and neither is
claimed here either way.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

from satquery.schema import ImageMeta, Modality
from satquery.tools.specialists import LAND_COVER


def zoom(image, factor: float):
    """Centre crop by `factor`, rescaled back to the original size.

    Kept because it is what produced the false alarm, and the disambiguation
    below is only meaningful next to it. It shrinks the ground footprint, which
    is NOT a valid stand-in for a finer sensor: the labels stop describing the
    image.
    """
    if factor <= 1.0:
        return image
    w, h = image.size
    cw, ch = int(w / factor), int(h / factor)
    left, top = (w - cw) // 2, (h - ch) // 2
    return image.crop((left, top, left + cw, top + ch)).resize((w, h))


def rescale(image, size: int):
    """Change apparent resolution while keeping footprint and labels valid."""
    return image.resize((size, size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/bench/crossmodal/metadata.parquet")
    ap.add_argument("--root", default="data/bench/crossmodal")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--sizes", default="60,120,480",
                    help="pixel sizes at fixed footprint -- the valid test")
    ap.add_argument("--factors", default="1,4",
                    help="centre-crop factors -- confounded, reported for contrast")
    ap.add_argument("--out", default="eval/results/crossmodal_robustness.json")
    a = ap.parse_args()

    from satquery.tools.backbone import embed_texts_cached, load, patch_tokens
    from satquery.tools.specialists import _open

    bb = load()
    if bb is None:
        raise SystemExit("backbone unavailable")

    root = Path(a.root)
    index = {p.stem: p for p in root.rglob("*.tif")}
    meta = pd.read_parquet(a.meta)
    meta = meta[(meta["split"] == "test") & meta["patch_id"].isin(index)
                & meta["s1_name"].isin(index)]
    rows = meta.to_dict("records")
    random.Random(0).shuffle(rows)

    txt = embed_texts_cached(bb, [f"a satellite image of {c}" for c in LAND_COVER])
    factors = [float(f) for f in a.factors.split(",")]
    sizes = [int(x) for x in a.sizes.split(",")]
    scores: dict[float, list[float]] = collections.defaultdict(list)
    survived: dict[float, int] = collections.defaultdict(int)
    used = 0

    for row in rows:
        if used >= a.limit:
            break
        gold = {c for c in LAND_COVER if c in {str(l).lower() for l in (row["labels"])}}
        if not gold:
            continue
        opt = _open(ImageMeta(path=str(index[row["patch_id"]]), fmt="GeoTIFF",
                              width=120, height=120, bands=10,
                              modality=Modality.OPTICAL))
        sar = _open(ImageMeta(path=str(index[row["s1_name"]]), fmt="GeoTIFF",
                              width=120, height=120, bands=2, modality=Modality.SAR))

        for size in sizes:
            pair = []
            for img in (rescale(opt, size), rescale(sar, size)):
                tokens, _ = patch_tokens(bb, img)
                pair.append((tokens @ txt.T).mean(axis=0))
            fused = np.maximum(*pair)
            top = [LAND_COVER[i] for i in np.argsort(-fused)[: a.k]]
            scores[f"{size}px"].append(len(set(top) & gold) / a.k)

        for f in factors[1:]:
            pair = []
            for img in (zoom(opt, f), zoom(sar, f)):
                tokens, _ = patch_tokens(bb, img)
                pair.append((tokens @ txt.T).mean(axis=0))
            fused = np.maximum(*pair)
            top = [LAND_COVER[i] for i in np.argsort(-fused)[: a.k]]
            scores[f"crop{f:g}x"].append(len(set(top) & gold) / a.k)

        used += 1

    print(f"{used} co-registered pairs, {len(LAND_COVER)} classes, P@{a.k}\n")
    native = float(np.mean(scores["120px"]))

    print("valid test -- apparent resolution changed, footprint and labels fixed")
    for size in sizes:
        p = float(np.mean(scores[f"{size}px"]))
        print(f"  {str(size) + 'px':<26}{p:>8.1%}{p - native:>+10.1%}")

    print("\nconfounded for contrast -- footprint shrinks, labels stop applying")
    for f in factors[1:]:
        p = float(np.mean(scores[f"crop{f:g}x"]))
        print(f"  {f'{f:g}x centre crop':<26}{p:>8.1%}{p - native:>+10.1%}")

    spread = max(float(np.mean(scores[f"{s_}px"])) for s_ in sizes) - \
             min(float(np.mean(scores[f"{s_}px"])) for s_ in sizes)
    print(f"\nspread across an 8x range of apparent resolution: {spread:.1%}")
    print("verdict:", "scale-invariant; resolution is not the ISRO risk"
          if spread < 0.02 else "resolution-sensitive")

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(
        {"n": used, "k": a.k,
         "by_size": {f"{s_}px": float(np.mean(scores[f"{s_}px"])) for s_ in sizes},
         "by_crop": {f"{f:g}x": float(np.mean(scores[f"crop{f:g}x"])) for f in factors[1:]},
         "scale_spread": spread}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
