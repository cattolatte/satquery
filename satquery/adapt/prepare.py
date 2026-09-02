"""
Build the remote-sensing adaptation set.

The problem statement names BigEarthNet.txt as "the primary dataset for
adapting image-text representations to multisensor remote-sensing data", so
that is what we adapt on.

Two sources have to be joined and they do not agree on patch identity:

    text  : S2A_MSIL2A_20170613T101031_N9999_R022_T33UUP_26_57
    image : S2A_MSIL2A_20170613T101031_26_57

The text carries the full BigEarthNet v1 product name; the image index drops
the orbit and tile tokens. Normalising to <sensor>_<level>_<datetime>_<row>_<col>
takes the join from 0 matches to 74,584 patches across the full index.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

# Sentinel-2 L2A band order in the BigEarthNet HDF5: B02,B03,B04,B05,B06,B07,
# B08,B8A,B11,B12,B01,B09. Natural colour is red,green,blue = B04,B03,B02.
RGB_BANDS = (2, 1, 0)


def normalise_patch_id(pid: str) -> str:
    parts = pid.split("_")
    return "_".join(parts[:3] + parts[-2:]) if len(parts) >= 5 else pid


def to_rgb(patch: np.ndarray) -> np.ndarray:
    """(12, H, W) float -> (H, W, 3) uint8, percentile-stretched.

    Reflectance has a long right tail; a raw min-max stretch renders almost
    everything dark. The 2-98 percentile stretch is what the visual products
    these captions describe were made with.
    """
    rgb = np.stack([patch[b] for b in RGB_BANDS], axis=-1).astype(np.float32)
    lo, hi = np.percentile(rgb, [2, 98])
    return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5", default="data/ben_val_p8.hdf5")
    ap.add_argument("--index", default="data/ben_val.csv")
    ap.add_argument("--text", default="data/bigearthnet_txt.parquet")
    ap.add_argument("--shard", default="bigearthnet_val_p8.hdf5")
    ap.add_argument("--out", default="data/adapt")
    ap.add_argument("--test-frac", type=float, default=0.15)
    a = ap.parse_args()

    out = Path(a.out); (out / "images").mkdir(parents=True, exist_ok=True)

    idx = pd.read_csv(a.index)
    shard_rows = idx[idx.s2_hdf5_file == a.shard]
    row_for = dict(zip(shard_rows.s2_folder, shard_rows["index"]))

    txt = pd.read_parquet(a.text, columns=["patch_id", "s1_name", "input", "output",
                                           "type", "category", "split"])
    txt["key"] = txt.patch_id.map(normalise_patch_id)
    txt = txt[txt.key.isin(row_for)]
    caps = txt[txt.type == "captioning"][["key", "s1_name", "output"]].drop_duplicates("key")
    print(f"patches with imagery and a caption: {len(caps):,}")

    from PIL import Image
    records = []
    with h5py.File(a.hdf5, "r") as f:
        imgs = f["images"]
        for i, r in enumerate(caps.itertuples()):
            row = row_for[r.key]
            if row >= len(imgs):
                continue
            png = out / "images" / f"{r.key}.png"
            if not png.exists():
                Image.fromarray(to_rgb(imgs[row])).save(png)
            records.append({"key": r.key, "image": str(png),
                            "caption": r.output, "s1_name": r.s1_name})
            if i and i % 500 == 0:
                print(f"  {i:,} written", flush=True)

    # Split by PATCH, never by annotation. The same scene appearing in both
    # splits would let the model memorise it and score well for the wrong
    # reason - the failure this project has already learned to distrust.
    rng = np.random.default_rng(0)
    order = rng.permutation(len(records))
    cut = int(len(records) * (1 - a.test_frac))
    train = [records[i] for i in order[:cut]]
    test = [records[i] for i in order[cut:]]

    for name, rows in (("train", train), ("test", test)):
        p = out / f"{name}.jsonl"
        p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
        print(f"{name}: {len(rows):,} -> {p}")

    # Keep the other annotation types for evaluating the downstream tools.
    for kind, fname in (("binary", "vqa_binary"), ("bounding box", "grounding")):
        sub = txt[txt.type == kind]
        sub = sub[sub.key.isin({r["key"] for r in test})]
        sub.to_parquet(out / f"eval_{fname}.parquet", index=False)
        print(f"eval_{fname}: {len(sub):,} annotations over held-out patches")


if __name__ == "__main__":
    main()
