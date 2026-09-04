"""Derive the per-question-type answer vocabulary from the training split.

Constrained decoding needs a candidate set, and where that set comes from
decides whether the technique is legitimate. Taken from the evaluation answers
it would be leakage; taken from the training answers it is simply a property of
the corpus the model was fitted to, and is available at training time.

The vocabularies are tight enough for this to work: object quantity has 33
distinct answers whose top eight cover 93% of the split, and `image` has 18
covering 97%.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from satquery.adapt.prepare_vlm import question_type

# Types whose answers come from a small closed set. Counting is included
# because its answers are small integers, not because they are categorical.
CLOSED = {"object existence", "object quantity", "object color", "object shape",
          "object size", "object direction", "scene type", "rural or urban",
          "image", "object position"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/adapt_vlm2/train.jsonl")
    ap.add_argument("--top", type=int, default=24,
                    help="candidates kept per type; the tail is long and rare")
    ap.add_argument("--min-count", type=int, default=3,
                    help="drop answers seen fewer times than this")
    ap.add_argument("--out", default="checkpoints/rs_vlm/answer_vocab.json")
    a = ap.parse_args()

    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for line in Path(a.train).read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("task") != "vqa":
            continue
        counts[question_type(row["question"])][str(row["answer"]).strip().lower()] += 1

    vocab, coverage = {}, {}
    for qtype, counter in counts.items():
        if qtype not in CLOSED:
            continue
        kept = [answer for answer, n in counter.most_common(a.top) if n >= a.min_count]
        if len(kept) < 2:
            continue
        vocab[qtype] = kept
        total = sum(counter.values())
        coverage[qtype] = sum(counter[k] for k in kept) / total

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"vocab": vocab, "coverage": coverage}, indent=1))
    print(f"{'type':<20}{'candidates':>12}{'train coverage':>16}")
    for qtype in sorted(vocab, key=lambda k: -coverage[k]):
        print(f"{qtype:<20}{len(vocab[qtype]):>12}{coverage[qtype]:>15.0%}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
