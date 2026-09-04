"""Build a supervised set for the generative specialist from VRSBench train.

The statement permits this explicitly: adaptation may use "BigEarthNet.txt or
the any open source training data". VRSBench ships a 142,390-conversation
training split covering the three single-image tasks it is scored on, and until
now none of it was used -- every VRSBench number was zero-shot transfer from
BigEarthNet land-cover captions onto object-centric aerial questions.

Why a generative model at all: the fixed-vocabulary head has a measured ceiling
of 65.5% on VRSBench VQA, because a third of the gold answers ("Body of water",
"residential", "grayscale") are not in its output space at any confidence.
Captioning is worse -- references average 48 words against a five-class
template. Generation removes the ceiling; training on this split closes the gap
to it.

Boxes are kept in VRSBench's own <0-99> convention rather than converted here,
so the model learns the format its supervision uses and the tool converts once
at the boundary.
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import random
import re
from pathlib import Path

TASK = re.compile(r"\[(vqa|refer|caption)\]")

# The question taxonomy lives with the serving code and is imported here, so
# the vocabulary built from training data is keyed exactly the way inference
# looks it up.
from satquery.tools.generative import question_type  # noqa: E402

# Share of the benchmark each type accounts for, measured from its own eval
# split rather than guessed.
EVAL_SHARE = {
    "object existence": 0.208, "object quantity": 0.170, "object position": 0.156,
    "object category": 0.145, "object color": 0.095, "scene type": 0.085,
    "object shape": 0.038, "image": 0.030, "object size": 0.027,
    "reasoning": 0.024, "object direction": 0.013, "rural or urban": 0.008,
}


def parse(entry: dict) -> tuple[str, str, str] | None:
    """(task, question, answer) from one VRSBench conversation."""
    turns = entry["conversations"]
    if isinstance(turns, str):
        try:
            turns = ast.literal_eval(turns)
        except (ValueError, SyntaxError):
            return None
    if len(turns) < 2:
        return None
    human, gpt = str(turns[0]["value"]), str(turns[1]["value"])
    m = TASK.search(human)
    if not m:
        return None

    question = TASK.sub("", human).replace("<image>", "").strip()
    # The VQA prompts carry a trailing instruction that is not part of the
    # question; keeping it would teach the model to expect it at inference.
    question = re.sub(r"\.\s*A short answer to the question is\s*$", "", question).strip()
    question = re.sub(r"^\s*\n+", "", question).strip()
    return m.group(1), question, gpt.strip()


def _stratify(rows: list[dict], budget: int, rng: random.Random) -> list[dict]:
    """Draw `budget` rows in the proportions the benchmark actually asks.

    Sampling uniformly gives the corpus's own distribution, which is not the
    evaluation's: object quantity is 17% of the benchmark and was 3.6% of
    training. Types short of their quota take everything available rather than
    being padded with duplicates, and whatever budget that frees is given back
    to the types that still have rows.
    """
    pools: dict[str, list] = {}
    for row in rows:
        pools.setdefault(question_type(row["question"]), []).append(row)

    picked, shortfall = [], 0
    for qtype, share in EVAL_SHARE.items():
        want = int(budget * share)
        have = pools.get(qtype, [])
        take = min(want, len(have))
        picked.extend(have[:take])
        pools[qtype] = have[take:]
        shortfall += want - take

    # Redistribute what the thin types could not fill.
    leftovers = [r for pool in pools.values() for r in pool]
    rng.shuffle(leftovers)
    picked.extend(leftovers[:shortfall])
    rng.shuffle(picked)
    return picked


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/bench/vrsbench/VRSBench_train.json")
    ap.add_argument("--images", default="data/bench/vrsbench/Images_train")
    ap.add_argument("--out", default="data/adapt_vlm")
    ap.add_argument("--per-task", type=int, default=0,
                    help="cap per task, 0 = all")
    ap.add_argument("--vqa", type=int, default=0,
                    help="VQA rows to draw, stratified to the eval's type mix")
    ap.add_argument("--caption", type=int, default=0, help="caption rows to draw")
    ap.add_argument("--refer", type=int, default=0,
                    help="grounding rows; 0 is deliberate -- the detector serves "
                         "grounding at 19%% Acc@0.5 against this model's 0.7%%")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rows = json.loads(Path(a.src).read_text())
    images = Path(a.images)
    by_task: dict[str, list] = collections.defaultdict(list)
    missing = 0

    for entry in rows:
        parsed = parse(entry)
        if parsed is None:
            continue
        task, question, answer = parsed
        path = images / str(entry["image"])
        if not path.exists():
            missing += 1
            continue
        by_task[task].append({"image": str(path), "task": task,
                              "question": question, "answer": answer})

    rng = random.Random(a.seed)
    out_rows = []
    for task, group in by_task.items():
        rng.shuffle(group)
        if task == "vqa" and a.vqa:
            out_rows.extend(_stratify(group, a.vqa, rng))
        elif task == "caption" and a.caption:
            out_rows.extend(group[: a.caption])
        elif task == "refer":
            out_rows.extend(group[: a.refer])
        elif a.vqa or a.caption:
            continue
        else:
            out_rows.extend(group[: a.per_task] if a.per_task else group)
    rng.shuffle(out_rows)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # Held out by image, not by row: the same scene appears in a caption, a
    # grounding and several VQA rows, so splitting by row would leak it.
    scenes = sorted({r["image"] for r in out_rows})
    rng.shuffle(scenes)
    held = set(scenes[: max(1, len(scenes) // 20)])
    train = [r for r in out_rows if r["image"] not in held]
    test = [r for r in out_rows if r["image"] in held]

    for name, rows_ in (("train", train), ("test", test)):
        path = out / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(r) for r in rows_))
        counts = collections.Counter(r["task"] for r in rows_)
        print(f"  {name}: {len(rows_):>6}  {dict(counts)} -> {path}")
    if missing:
        print(f"  ({missing} rows skipped: image not downloaded)")


if __name__ == "__main__":
    main()
