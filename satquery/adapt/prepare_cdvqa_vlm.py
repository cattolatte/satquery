"""Build a supervised change-VQA set from CDVQA for the generative specialist.

Change analysis is one of the statement's mandatory capabilities, and until now
it was the only one answered entirely by a hand-written heuristic: patch areas
differenced between two dates, thresholded. That scores 47.3% while CDVQA ships
65,967 training examples of exactly this task, none of which had been used.

The statement is explicit that the final score combines normalised metrics
across tasks, so a mandatory capability left at heuristic level costs as much as
a strong one gains. This is the largest untouched lever in the project.

Answers are kept in CDVQA's own vocabulary -- "yes", "no", "NVG_surface",
"0_to_10" -- so the model learns the tokens the benchmark scores, and the tool
converts nothing at the boundary.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def parse(record: Path) -> dict | None:
    """One CDVQA sample as a two-image training row."""
    data = json.loads(record.read_text())
    turns = data.get("conversations") or []
    if len(turns) < 2:
        return None
    stem = record.name[: -len(".json")]
    pair = [record.parent / f"{stem}.0.img", record.parent / f"{stem}.1.img"]
    if not all(p.exists() for p in pair):
        return None

    question = str(turns[0]["value"]).split("\n")[-1].strip()
    answer = str(turns[1]["value"]).strip()
    if not question or not answer:
        return None
    return {
        "images": [str(p) for p in pair],
        "task": "change",
        "qtype": data.get("meta", {}).get("question_type", "unknown"),
        "image_id": data.get("meta", {}).get("image_id", stem),
        "question": question,
        "answer": answer,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bench/cdvqa/xtrain")
    ap.add_argument("--out", default="data/adapt_change")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = []
    for record in sorted(Path(a.src).glob("*.json")):
        parsed = parse(record)
        if parsed:
            rows.append(parsed)
    if not rows:
        raise SystemExit(f"no samples under {a.src}; is the archive extracted?")

    rng = random.Random(a.seed)
    rng.shuffle(rows)
    if a.limit:
        rows = rows[: a.limit]

    # Held out by scene, not by row: CDVQA asks several questions of the same
    # image pair, so splitting by row would put the same imagery on both sides.
    scenes = sorted({r["image_id"] for r in rows})
    rng.shuffle(scenes)
    held = set(scenes[: max(1, len(scenes) // 10)])
    train = [r for r in rows if r["image_id"] not in held]
    test = [r for r in rows if r["image_id"] in held]

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, group in (("train", train), ("test", test)):
        (out / f"{name}.jsonl").write_text("\n".join(json.dumps(r) for r in group))
        by = collections.Counter(r["qtype"] for r in group)
        print(f"  {name}: {len(group):>6} rows, {len({r['image_id'] for r in group})} scenes")
        print(f"          {dict(by.most_common(8))}")


if __name__ == "__main__":
    main()
