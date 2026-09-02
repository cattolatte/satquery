"""Build a mixed optical + SAR adaptation set from co-registered pairs.

The problem statement names BigEarthNet.txt as the adaptation dataset and
describes it as "co-registered Sentinel-1 SAR, Sentinel-2 multispectral imagery,
and diverse text annotations". The first adaptation pass used only the optical
half, which left the backbone unable to read radar: SAR-only land-cover scored
12.4% against optical's 21.1%, and fusing the two came out *below* optical alone
because averaging in a weak reading dilutes a good one.

Training one encoder on both modalities against the same text puts SAR in the
same embedding space rather than bolting on a second model, which keeps the
registry's one-backbone-many-heads shape intact.

Patches are rendered to PNG here rather than at training time: the loader would
otherwise pay GeoTIFF decode and the SAR composite on every epoch, and the
rendering is deterministic.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pandas as pd

from satquery.schema import ImageMeta, Modality
from satquery.tools.specialists import _open

ROOT = Path("data/bench/crossmodal")
OUT = Path("data/adapt_crossmodal")


def caption(labels: list[str], modality: str) -> str:
    """A text description the encoder can align both sensors against.

    The sensor is named in the caption so the model can use it as context rather
    than having to explain away the appearance difference between a radar and an
    optical view of the same ground.
    """
    named = ", ".join(sorted(str(l).lower() for l in labels))
    sensor = ("a Sentinel-1 synthetic aperture radar image"
              if modality == "sar" else "a Sentinel-2 optical satellite image")
    return f"{sensor} of {named}"


def render(path: Path, modality: Modality, dest: Path) -> bool:
    bands = 2 if modality is Modality.SAR else 10
    meta = ImageMeta(path=str(path), fmt="GeoTIFF", width=120, height=120,
                     bands=bands, modality=modality)
    try:
        _open(meta).save(dest)
        return True
    except Exception:                                          # noqa: BLE001
        return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--limit", type=int, default=0, help="0 = all training pairs")
    a = ap.parse_args()

    root = Path(a.root)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    index = {p.stem: p for p in (root / "BEN_14k").rglob("*.tif")}
    meta = pd.read_parquet(root / "metadata.parquet")
    meta = meta[meta["patch_id"].isin(index) & meta["s1_name"].isin(index)]

    train = meta[meta["split"] == "train"]
    held = meta[meta["split"] == "validation"]
    if a.limit:
        train = train.head(a.limit)
    print(f"train pairs {len(train)}  held-out {len(held)}")

    for split, frame in (("train", train), ("test", held)):
        rows = []
        for r in frame.to_dict("records"):
            raw = r.get("labels")          # numpy array; truthiness raises
            labels = [] if raw is None else list(raw)
            if not labels:
                continue
            for modality, key in ((Modality.OPTICAL, "patch_id"), (Modality.SAR, "s1_name")):
                src = index.get(str(r[key]))
                if src is None:
                    continue
                tag = "s1" if modality is Modality.SAR else "s2"
                dest = out / "images" / f"{r['patch_id']}_{tag}.png"
                if not dest.exists() and not render(src, modality, dest):
                    continue
                rows.append({"key": f"{r['patch_id']}_{tag}",
                             "image": str(dest), "modality": tag,
                             "caption": caption(labels, tag)})
        random.Random(0).shuffle(rows)
        path = out / f"{split}.jsonl"
        path.write_text("\n".join(json.dumps(x) for x in rows))
        sar = sum(1 for x in rows if x["modality"] == "s1")
        print(f"  {split}: {len(rows)} samples ({sar} SAR, {len(rows)-sar} optical) -> {path}")


if __name__ == "__main__":
    main()
