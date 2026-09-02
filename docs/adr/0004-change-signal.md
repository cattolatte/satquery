# ADR 0004 — Per-class area change, not scene-level score difference

**Status:** accepted

## Context

Change VQA is mandatory: "Change description or change-based visual question
answering from a bi-temporal image pair shall be mandatory." CDVQA is the named
benchmark.

The first implementation answered "did the area of X change?" from the
difference in scene-level CLIP similarity to X between the two dates. It was
never measured against anything, because BigEarthNet is single-date.

## What measurement showed

On CDVQA's polar change questions, tuning the threshold on shards 0–1 and
testing on a held-out shard 2:

| signal | tune | held-out test |
|---|---|---|
| scene-level score difference | 60.9% | 56.5% |
| **per-class area difference** | 67.8% | **58.1%** |
| majority baseline | — | 54.8% |

The scene-level signal beat the baseline by 1.7 points, which is not a result.
The area signal beats it by 3.3 and is stable across thresholds 0.08–0.20 —
all give 58.1% on the held-out shard — so it is not a knife-edge fit.

## Decision

Assign every patch to its nearest class in the joint embedding space on each
date, take the class's area as the fraction of patches assigned to it, and
difference the two. This measures what the question actually asks: whether the
*area* of a class changed, rather than whether the scene's overall description
drifted.

The threshold is 0.12, chosen from the middle of the stable plateau rather than
the tune-set argmax, which sat at its edge.

## Result on the full benchmark

300 CDVQA questions, 22.5% → 48.3% overall as the signal and answer decoding
were fixed:

| question type | n | accuracy |
|---|---|---|
| decrease_or_not | 35 | 77.1% |
| increase_or_not | 27 | 74.1% |
| change_or_not | 115 | 53.9% |
| change_ratio_types | 45 | 37.8% |
| change_ratio | 14 | 28.6% |
| largest / smallest_change | 42 | 23.8% |
| change_to_what | 22 | 22.7% |

Direction is where this works: knowing a class gained or lost area is exactly
what an area difference encodes, and 74–77% on those reflects it. Identifying
*which* class changed most is much weaker, because ranking six small area
deltas against each other asks far more precision of the patch assignment than
reading the sign of one.

No change-detection training was done at all — the capability is carried
entirely by the adapted backbone.

## Consequences

- `ChangeTool` emits `area_delta` evidence for **every** class, not just the
  movers, because a caller asking about one class needs its value even when
  others moved more.
- `max_classes` became a permitted parameter for the same reason.
- The scene-level delta is still emitted as `delta` evidence and still drives
  the prose description, where "what moved most" is the right summary.
