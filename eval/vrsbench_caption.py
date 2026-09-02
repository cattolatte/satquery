"""VRSBench captioning.

The problem statement names VRSBench for single-image captioning. Metrics are
implemented here rather than pulled from a package so the scoring is inspectable
and the run has no extra dependency.

Three numbers, because n-gram overlap alone would be misleading. VRSBench
references are long, fluent, human-written descriptions naming objects, colours
and spatial relations. This system emits a land-cover summary from a fixed
vocabulary. BLEU and ROUGE-L largely measure that difference in form, so a
content-word recall is reported alongside them: it asks whether the terms we do
emit appear in the reference at all, which is the part a land-cover captioner
could in principle get right.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import random
import re
from pathlib import Path

from satquery.registry import build_registry
from satquery.schema import ImageMeta, Modality

_STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "and", "or", "is", "are",
    "with", "this", "that", "it", "its", "from", "by", "as", "for", "there",
    "image", "images", "photo", "picture", "scene", "aerial", "satellite",
    "shows", "showing", "features", "featuring", "visible", "seen", "px",
}


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", str(text).lower())


def ngrams(seq: list[str], n: int) -> collections.Counter:
    return collections.Counter(tuple(seq[i:i + n]) for i in range(len(seq) - n + 1))


def bleu(candidate: list[str], reference: list[str], max_n: int = 4) -> float:
    """Sentence BLEU with add-one smoothing on higher orders."""
    if not candidate:
        return 0.0
    precisions = []
    for n in range(1, max_n + 1):
        cand_ng, ref_ng = ngrams(candidate, n), ngrams(reference, n)
        overlap = sum(min(c, ref_ng[g]) for g, c in cand_ng.items())
        total = max(sum(cand_ng.values()), 1)
        # Smoothing keeps a single missing 4-gram from zeroing the whole score.
        precisions.append((overlap + (1 if n > 1 else 0)) / (total + (1 if n > 1 else 0)))
    if min(precisions) <= 0:
        return 0.0
    score = math.exp(sum(math.log(p) for p in precisions) / max_n)
    brevity = min(1.0, math.exp(1 - len(reference) / max(len(candidate), 1)))
    return score * brevity


def rouge_l(candidate: list[str], reference: list[str]) -> float:
    """F-measure over the longest common subsequence."""
    if not candidate or not reference:
        return 0.0
    prev = [0] * (len(reference) + 1)
    for c in candidate:
        cur = [0]
        for j, r in enumerate(reference):
            cur.append(prev[j] + 1 if c == r else max(cur[j], prev[j + 1]))
        prev = cur
    lcs = prev[-1]
    p, r = lcs / len(candidate), lcs / len(reference)
    return 2 * p * r / (p + r) if p + r else 0.0


def content_recall(candidate: list[str], reference: list[str]) -> float:
    """Share of the content words we emit that the reference also uses."""
    cand = {w for w in candidate if w not in _STOP and len(w) > 3}
    ref = {w for w in reference if w not in _STOP and len(w) > 3}
    return len(cand & ref) / len(cand) if cand else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bench/vrsbench/VRSBench_EVAL_Cap.json")
    ap.add_argument("--images", default="data/bench/vrsbench/Images_val")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval/results/vrsbench_caption.json")
    a = ap.parse_args()

    rows = json.loads(Path(a.data).read_text())
    images = Path(a.images)
    if not images.is_dir():
        raise SystemExit(f"images not found at {images}; run scripts/fetch_vrsbench.sh")

    random.Random(a.seed).shuffle(rows)
    rows = rows[: a.limit] if a.limit else rows
    print(f"VRSBench captioning: {len(rows)} images")

    tool = build_registry().get("rs_caption")
    ok, why = tool.available()
    if not ok:
        raise SystemExit(f"rs_caption unavailable: {why}")

    b1, b4, rl, cr = [], [], [], []
    for i, r in enumerate(rows, 1):
        path = images / r["image_id"]
        if not path.exists():
            continue
        meta = ImageMeta(path=str(path), fmt="PNG", width=512, height=512,
                         bands=3, modality=Modality.OPTICAL)
        _, text, _, _ = tool.invoke([meta], r.get("question", "Describe this image."), {})
        cand, ref = tokens(text), tokens(r["ground_truth"])
        b1.append(bleu(cand, ref, 1))
        b4.append(bleu(cand, ref, 4))
        rl.append(rouge_l(cand, ref))
        cr.append(content_recall(cand, ref))
        if i % 200 == 0:
            print(f"  {i}/{len(rows)}  ROUGE-L {sum(rl)/len(rl):.3f}")

    n = len(rl)
    out = {"benchmark": "VRSBench captioning", "n": n,
           "bleu1": sum(b1) / n, "bleu4": sum(b4) / n,
           "rouge_l": sum(rl) / n, "content_recall": sum(cr) / n}
    print(f"\n{'metric':<20}{'value':>9}")
    for k in ("bleu1", "bleu4", "rouge_l", "content_recall"):
        print(f"{k:<20}{out[k]:>9.3f}")

    path = Path(a.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
