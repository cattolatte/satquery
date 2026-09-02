"""VRSBench visual question answering.

The problem statement names VRSBench for single-image VQA. Its questions are
overwhelmingly object-level -- the colour, count, position and shape of
individual vehicles and buildings -- which a scene-level land-cover backbone
cannot address. The scene-level types it does contain are where this system
has anything to say, so results are reported per type.

Every type is reported next to its own majority baseline, because several are
severely skewed: 86% of "object existence" answers are "Yes", so a model that
answers Yes to everything scores 86% there while knowing nothing. Quoting an
aggregate against those types without their baselines would be meaningless.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

from satquery.registry import build_registry
from satquery.schema import ImageMeta, Modality

IMAGES = Path("data/bench/vrsbench/Images_val")

# Types this architecture can address at all: they ask about the scene, not
# about an individual object within it.
SCENE_LEVEL = {"scene type", "rural or urban", "image"}


def normalise(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", str(text).strip().lower())


_TOP_CLASS = re.compile(r"most likely land cover:\s*([^(]+?)\s*\(", re.I)


def decode(text: str, gold_hint: str) -> str:
    """Reduce the tool's prose to the token a benchmark can compare.

    The land-cover listing needs its own case: the answer is the top-ranked
    class, and splitting on whitespace would return the literal word "most"
    from the sentence that introduces it -- which scored every open-ended
    question wrong regardless of whether the class was right.
    """
    low = text.strip().lower()
    if low.startswith("yes"):
        return "yes"
    if low.startswith("no"):
        return "no"
    ranked = _TOP_CLASS.search(low)
    if ranked:
        return normalise(ranked.group(1))
    # Either/or answers lead with the winning alternative.
    lead = re.split(r"[\s—\-,.]", low, maxsplit=1)[0]
    return normalise(lead)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bench/vrsbench/VRSBench_EVAL_vqa.json")
    ap.add_argument("--images", default=str(IMAGES))
    ap.add_argument("--per-type", type=int, default=250,
                    help="stratified sample per question type")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/vrsbench_vqa.json")
    a = ap.parse_args()

    rows = json.loads(Path(a.data).read_text())
    images = Path(a.images)
    if not images.is_dir():
        raise SystemExit(f"images not found at {images}; run scripts/fetch_vrsbench.sh")

    by_type: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        by_type[r["type"]].append(r)

    rng = random.Random(a.seed)
    sample = []
    for qtype, group in by_type.items():
        rng.shuffle(group)
        sample.extend(group[: a.per_type])
    print(f"VRSBench VQA: {len(sample)} questions "
          f"(stratified, up to {a.per_type} per type, from {len(rows)})")

    tool = build_registry().get("rs_vqa")
    ok, why = tool.available()
    if not ok:
        raise SystemExit(f"rs_vqa unavailable: {why}")

    def lenient(pred: str, gold: str) -> bool:
        """Partial credit for a semantically right answer worded differently.

        VRSBench gold answers are free-form English while this system emits
        CORINE class names, so "inland waters" is scored wrong against "Body of
        water" under exact match even though it identifies the same thing.
        Reported strictly as a secondary number: exact match is the benchmark's
        metric and stays the headline.
        """
        stop = {"a", "an", "the", "of", "is", "are", "in", "on", "and", "area", "image"}
        pw = {w for w in pred.split() if w not in stop and len(w) > 2}
        gw = {w for w in gold.split() if w not in stop and len(w) > 2}
        return bool(pw & gw)

    hits: dict[str, list[bool]] = collections.defaultdict(list)
    soft: dict[str, list[bool]] = collections.defaultdict(list)
    golds: dict[str, list[str]] = collections.defaultdict(list)
    missing = 0

    for i, r in enumerate(sample, 1):
        path = images / r["image_id"]
        if not path.exists():
            missing += 1
            continue
        meta = ImageMeta(path=str(path), fmt="PNG", width=512, height=512,
                         bands=3, modality=Modality.OPTICAL)
        _, text, _, _ = tool.invoke([meta], r["question"], {})
        gold = normalise(r["ground_truth"])
        pred = decode(text, gold)
        hits[r["type"]].append(pred == gold)
        soft[r["type"]].append(pred == gold or lenient(pred, gold))
        golds[r["type"]].append(gold)
        if i % 300 == 0:
            done = sum(len(v) for v in hits.values())
            print(f"  {i}/{len(sample)}  running {sum(sum(v) for v in hits.values())/max(done,1):.1%}")

    if missing:
        print(f"  ({missing} questions skipped: image not found)")

    print(f"\n{'question type':<20}{'n':>6}{'exact':>8}{'lenient':>9}{'majority':>10}")
    report, scene_hits, scene_n, scene_base = {}, 0, 0, 0
    for qtype in sorted(hits, key=lambda k: -len(hits[k])):
        v = hits[qtype]
        acc = sum(v) / len(v)
        base = collections.Counter(golds[qtype]).most_common(1)[0][1] / len(v)
        sacc = sum(soft[qtype]) / len(soft[qtype])
        report[qtype] = {"n": len(v), "accuracy": acc, "lenient": sacc,
                         "majority": base, "scene_level": qtype in SCENE_LEVEL}
        mark = "  *" if qtype in SCENE_LEVEL else ""
        print(f"{qtype:<20}{len(v):>6}{acc:>7.1%}{sacc:>9.1%}{base:>10.1%}{mark}")
        if qtype in SCENE_LEVEL:
            scene_hits += sum(v); scene_n += len(v)
            scene_base += collections.Counter(golds[qtype]).most_common(1)[0][1]

    total = sum(len(v) for v in hits.values())
    correct = sum(sum(v) for v in hits.values())
    soft_total = sum(sum(v) for v in soft.values())
    print(f"\n{'OVERALL':<20}{total:>6}{correct/total:>7.1%}{soft_total/total:>9.1%}")
    if scene_n:
        scene_soft = sum(sum(soft[q]) for q in SCENE_LEVEL if q in soft)
        print(f"{'scene-level (*)':<20}{scene_n:>6}{scene_hits/scene_n:>7.1%}"
              f"{scene_soft/scene_n:>9.1%}{scene_base/scene_n:>10.1%}")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"benchmark": "VRSBench VQA (stratified sample)", "n": total,
         "overall": correct / total, "overall_lenient": soft_total / total,
         "scene_level": (scene_hits / scene_n) if scene_n else None,
         "scene_level_majority": (scene_base / scene_n) if scene_n else None,
         "by_type": report}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
