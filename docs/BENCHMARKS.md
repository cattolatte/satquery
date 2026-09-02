# Benchmark results

The problem statement names the benchmarks it will score against: VRSBench and
RSVQA for single-image tasks, CDVQA for multitemporal change VQA. These are the
measured numbers, with the baselines they should be read against.

Every number here comes from the real serving path — the registered tool
invoked through `Tool.invoke` — not a separate scoring routine that could
quietly differ from what the API does.

## CDVQA — change VQA (mandatory task)

300 questions, test shards 0–2 of `ljx620/CDVQA`.

| question type | n | accuracy |
|---|---|---|
| decrease_or_not | 35 | **77.1%** |
| increase_or_not | 27 | **74.1%** |
| change_or_not | 115 | 53.9% |
| change_ratio_types | 45 | 37.8% |
| change_ratio | 14 | 28.6% |
| largest_change | 21 | 23.8% |
| smallest_change | 21 | 23.8% |
| change_to_what | 22 | 22.7% |
| **overall** | **300** | **48.3%** |

No change-detection training was done. The capability comes entirely from the
adapted backbone plus per-class area differencing, so 48.3% is what the
adaptation buys on its own.

Direction questions are where it genuinely works — knowing whether a class
gained or lost area is exactly what an area difference encodes. Identifying
*which* of six classes changed most is much weaker, since that asks for
precision the patch assignment does not have.

See [ADR 0004](adr/0004-change-signal.md) for how the signal was chosen: the
threshold was tuned on shards 0–1 and reported on a held-out shard 2, and the
scene-level alternative was rejected on that held-out evidence.

## RSVQA-LR — single-image VQA

2,000 questions, `dmarsili/RSVQA-LR-2k` validation.

| question type | n | accuracy | note |
|---|---|---|---|
| rural/urban | 29 | **69.0%** | either/or scoring |
| comparison | 535 | 54.6% | |
| presence-scene | 427 | 52.7% | |
| presence-object | 406 | 50.5% | object-level, see below |
| count | 603 | 0.0% | **not attempted** |
| **overall** | **2000** | **37.1%** | |
| scene-level subset | 991 | 54.2% | what this architecture addresses |

Two honest qualifications, both structural rather than tuning problems.

**Counting is not attempted.** A global image–text similarity has no mechanism
for "how many farmlands are there". Scraping a digit out of the prose scored
12.5% by accident on an early run, which misrepresents the system as partially
capable when it is not capable at all, so counting now returns `unsupported`
and scores zero. 603 of 2,000 questions are counting.

**Object-level questions are out of reach.** "Is a *circular* building
present?" asks about an individual instance and its shape. A scene-level
land-cover classifier cannot resolve that, and 50.5% — exactly chance — says so.

Even on the scene-level subset the margin over a majority baseline is small.
A controlled comparison of five yes/no decision rules
([results](../eval/results/decision_rules.md)) found the adopted rule best at
55.7% against a 50.4% baseline, and — more informatively — found that rules
which answer "yes" often score *below* chance. The backbone's ranking is
anti-correlated with RSVQA's notion of presence.

That is the same distribution-shift finding that shaped the backbone choice:
remote-sensing adaptation is corpus-specific, not one transferable property.
BigEarthNet's CORINE land-cover vocabulary does not carry over to RSVQA's
object-centric annotation, even on the same sensor family. Closing this needs
adaptation on RSVQA itself, not a better decision rule, which is why no further
rule tuning was attempted.

## VRSBench — referring-expression grounding

1,078 expressions from `omlab/VRSBench-FS` val shard 0, which packages
VRSBench's referring task as parquet. Boxes are real pixel supervision, scored
with Acc@0.5 IoU, the standard referring metric.

| metric | value |
|---|---|
| Acc@0.5 IoU | **0.4%** |
| Acc@0.25 IoU | 4.3% |
| mean IoU | 0.035 |
| no box produced | 0 |

This is a clear negative result and the breakdown says why it is structural
rather than a tuning problem:

| target size | n | mean IoU |
|---|---|---|
| large (>4 patch cells) | 189 | 0.083 |
| 1–4 cells | 333 | 0.052 |
| sub-cell | 278 | 0.012 |
| tiny (<1/4 cell) | 278 | 0.004 |

Grounding here comes from CLIP patch tokens on a 7×7 grid. Over a 512×512 image
that is 73 px per cell, so the smallest box the method can express is about
73×73 — and VRSBench refers to individual vehicles, most of which are smaller
than a single cell. Accuracy falls monotonically with target size, exactly as
that limit predicts.

Tiling was implemented to test whether resolution alone was the cause. Running
the encoder over a T×T grid of overlapping crops gives 7T×7T cells:

| tiles | passes/image | mean IoU | Acc@0.25 |
|---|---|---|---|
| 1 | 1 | 0.006 | 0.0% |
| 3 | 9 | 0.019 | 1.0% |
| 5 | 25 | 0.025 | 3.0% |

Resolution is part of the story — the trend is monotonic — but four times the
mean IoU on a base of 0.006 is still nowhere near usable. CLIP patch tokens
were never supervised for object localisation, and no amount of tiling supplies
supervision that was never there. Closing this needs an open-vocabulary
detection head, which is an architectural addition rather than a parameter.

`tiles` is a permitted parameter, defaulting to 1. It is worth raising for
small targets and not worth the 25× cost for the scene-level regions the
problem statement's own grounding example asks about ("Highlight the water body
referred to in the query"), where the single-pass grid is already the right
scale.

### VRSBench VQA and captioning — not run

Their annotations are downloaded (`VRSBench_EVAL_vqa.json`, `_Cap.json`), but
they index into the 4 GB `Images_val.zip`, which did not finish downloading.
No VQA or captioning numbers are claimed for VRSBench.

## Reproducing

```bash
PYTHONPATH=. python3 eval/cdvqa.py
```

```bash
PYTHONPATH=. python3 eval/rsvqa.py
```

Results are written to `eval/results/*.json`.
