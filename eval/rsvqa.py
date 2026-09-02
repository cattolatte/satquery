"""RSVQA-LR evaluation.

The problem statement names RSVQA among the benchmarks used to evaluate
single-image visual question answering, so this measures the real serving path
-- the registered VQATool through Tool.invoke -- rather than a bespoke scoring
routine that could quietly differ from what the API does.

Results are broken down by question type because the aggregate is misleading
here. RSVQA-LR is dominated by counting and comparison questions, which a
CLIP-similarity backbone cannot answer even in principle: there is no mechanism
in a global image-text similarity for "how many farmlands are there". Reporting
one number would hide that the system is competent on presence questions and
incapable on counting, which are very different findings.
"""
from __future__ import annotations

import argparse
import collections
import io
import json
import re
import tempfile
from pathlib import Path

import pandas as pd

from satquery.registry import build_controller
from satquery.schema import ImageMeta, Modality

# RSVQA-LR question families, in the order they should be tested.
_COUNT = re.compile(r"^\s*(how many|what is the number of|what is the amount of)", re.I)
_COMPARE = re.compile(r"\b(more|less|fewer|greater|larger|smaller)\b", re.I)
_RURAL = re.compile(r"\b(rural|urban)\b", re.I)
_PRESENCE = re.compile(r"^\s*(is|are|does|do|there)\b", re.I)

# RSVQA-LR presence questions split into two very different problems. Scene-level
# ones ("is there water") ask what the landscape contains and are answerable from
# land cover. Object-level ones ("is a circular building present") ask about an
# individual instance and its shape or size, which a global image-text similarity
# has no mechanism to resolve. Averaging the two hides that the system is doing
# real work on one and guessing on the other.
_OBJECT_LEVEL = re.compile(
    r"\b(circular|square|rectangular|large|small|medium|tall|short)\b", re.I)


def question_type(q: str) -> str:
    if _COUNT.search(q):
        return "count"
    if _RURAL.search(q):
        return "rural/urban"
    if _COMPARE.search(q):
        return "comparison"
    if _PRESENCE.search(q):
        return "presence-object" if _OBJECT_LEVEL.search(q) else "presence-scene"
    return "other"


def normalise(ans: str) -> str:
    return re.sub(r"[^a-z0-9/ ]", "", str(ans).strip().lower())


def predicted_answer(text: str, qtype: str) -> str:
    """Reduce a tool's prose to the token RSVQA scores against.

    The benchmark expects a bare 'yes'/'no'/'rural'/'urban'/count, so a verbose
    answer has to be reduced before it can be compared at all.
    """
    low = text.lower()
    if qtype.startswith("presence") or qtype == "comparison":
        if low.startswith("yes") or " yes " in low[:12]:
            return "yes"
        if low.startswith("no"):
            return "no"
        return "unparsed"
    if qtype == "rural/urban":
        # The tool answers either/or questions with the winning alternative, so
        # read that directly; fall back to the land-cover family it named.
        if low.startswith("urban"):
            return "urban"
        if low.startswith("rural"):
            return "rural"
        urban = ("urban fabric", "industrial or commercial units", "road network")
        return "urban" if any(c in low for c in urban) else "rural"
    if qtype == "count":
        # The detector answers with the count as the leading token. Only that
        # position is read: scraping any digit from the prose would pick up
        # thresholds and ranks and score accidental hits.
        m = re.match(r"\s*(\d+)\b", text)
        return m.group(1) if m else "unsupported"
    return "unparsed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bench/rsvqa/validation.parquet")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", default="eval/results/rsvqa_lr.json")
    a = ap.parse_args()

    df = pd.read_parquet(a.data)
    if a.limit:
        df = df.head(a.limit)
    print(f"RSVQA-LR: {len(df)} questions")

    # Through the controller, not a single tool: selecting between the
    # scene-level backbone and the detector is part of what is being measured,
    # and calling one tool directly would bypass exactly that step.
    controller = build_controller()

    per_type: dict[str, list[bool]] = collections.defaultdict(list)
    unparsed: dict[str, int] = collections.defaultdict(int)
    tmp = Path(tempfile.mkdtemp(prefix="rsvqa-"))

    for i, row in enumerate(df.itertuples(), 1):
        path = tmp / f"{i}.png"
        path.write_bytes(row.image["bytes"])
        qtype = question_type(row.question)
        text = controller.run(row.question, [str(path)]).text
        pred = predicted_answer(text, qtype)
        gold = normalise(row.answer)
        if pred in ("unparsed", "unsupported"):
            unparsed[qtype] += 1
        per_type[qtype].append(pred == gold)
        path.unlink(missing_ok=True)

        if i % 200 == 0:
            done = sum(len(v) for v in per_type.values())
            acc = sum(sum(v) for v in per_type.values()) / done
            print(f"  {i}/{len(df)}  running accuracy {acc:.1%}")

    total = sum(len(v) for v in per_type.values())
    correct = sum(sum(v) for v in per_type.values())
    print(f"\n{'type':<14}{'n':>6}{'accuracy':>11}{'unparsed':>10}")
    rows = {}
    for qtype in sorted(per_type, key=lambda k: -len(per_type[k])):
        hits = per_type[qtype]
        acc = sum(hits) / len(hits)
        rows[qtype] = {"n": len(hits), "accuracy": acc, "unparsed": unparsed[qtype]}
        print(f"{qtype:<14}{len(hits):>6}{acc:>10.1%}{unparsed[qtype]:>10}")
    print(f"{'OVERALL':<14}{total:>6}{correct/total:>10.1%}")

    # The honest headline: accuracy over the questions this architecture can
    # answer at all, reported next to the aggregate rather than instead of it.
    answerable = [q for q in per_type if q not in ("count", "presence-object")]
    n_ans = sum(len(per_type[q]) for q in answerable)
    c_ans = sum(sum(per_type[q]) for q in answerable)
    if n_ans:
        print(f"{'scene-level':<14}{n_ans:>6}{c_ans/n_ans:>10.1%}"
              "   <- what this architecture can actually address")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"benchmark": "RSVQA-LR (dmarsili/RSVQA-LR-2k validation)",
         "n": total, "overall": correct / total,
         "scene_level": (c_ans / n_ans) if n_ans else None,
         "by_type": rows}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
