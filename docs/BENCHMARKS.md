# Benchmark results

All three benchmarks the problem statement names, plus the optical–SAR pairing
it makes mandatory. Every number comes from the real serving path — the
registered tool invoked through `Tool.invoke` — not a separate scoring routine
that could quietly differ from what the API does.

Baselines are reported beside every result. Several of these benchmarks are
severely skewed, and an accuracy quoted without its baseline says nothing.

## Optical–SAR joint analysis (mandatory)

600 co-registered Sentinel-1/Sentinel-2 pairs, gold CORINE labels, P@3 over 21
classes. "Complementary" is a testable claim, so the same vocabulary is scored
through each sensor alone and fused.

| reading | before SAR adaptation | after |
|---|---|---|
| optical only | 21.1% | 34.5% |
| SAR only | 12.4% | **36.2%** |
| fused (max) | 18.9% | **37.6%** |
| fusion vs. better single sensor | **−2.2%** | **+1.4%** |

Before adapting to radar, fusion scored *below* optical alone — averaging a
weak reading into a good one dilutes it. Weighting by sensor confidence did not
fix it, because confidence is not correctness and the model was confidently
wrong on data it had never seen. Training one encoder on both modalities
against the same text is what changed the sign.

The +1.4 margin is about eight samples in six hundred and is not presented as
more than that. What matters is the direction: fusion now beats the better
single sensor rather than losing to it.

Two bugs found by running on real sensor data, neither reachable by unit test:

- BigEarthNet's Sentinel-1 patches ship **already in decibels** (−35..0). The
  composite applied `10·log10` anyway, so every sample clamped to the floor and
  the channel flattened to a constant. SAR-only rose 4.9% → 12.4% on the fix
  alone. The input was being destroyed, not merely misread.
- Sentinel-2 red/green/blue sit at **different indices** depending on packing.
  A full product starts at B01 (indices 3/2/1); BigEarthNet starts at B02
  (2/1/0). The loader assumed the former and read B05 as red, returning a
  plausible false-colour image rather than failing. Optical rose 19.1% → 21.1%.

## CDVQA — change VQA (mandatory)

300 questions, test shards 0–2.

| question type | n | accuracy |
|---|---|---|
| increase_or_not | 27 | **74.1%** |
| decrease_or_not | 35 | **65.7%** |
| change_to_what | 22 | 45.5% |
| change_or_not | 115 | 45.2% |
| change_ratio_types | 45 | 44.4% |
| largest_change | 21 | 42.9% |
| change_ratio | 14 | 42.9% |
| smallest_change | 21 | 9.5% |
| **overall** | **300** | **47.3%** |

No change-detection training was done; the capability comes from the adapted
backbone plus per-class area differencing. Direction is where it works — the
sign of an area difference is exactly what "increase or decrease" asks.

See [ADR 0004](adr/0004-change-signal.md) for how the signal was chosen: tuned
on shards 0–1, reported on held-out shard 2, with the scene-level alternative
rejected on that held-out evidence.

## RSVQA-LR — single-image VQA

800 questions, same subset under both checkpoints.

| question type | n | cross-modal | optical-only |
|---|---|---|---|
| comparison | 209 | 54.5% | 56.9% |
| presence-object | 170 | 50.6% | 47.1% |
| presence-scene | 160 | 46.2% | 56.2% |
| rural/urban | 11 | 45.5% | 72.7% |
| count | 250 | 0.0% | 0.0% |
| **overall** | **800** | **34.9%** | **37.1%** |
| scene-level subset | 380 | 50.8% | 57.1% |

**Counting is not attempted.** A global image–text similarity has no mechanism
for "how many farmlands are there". An early version scored 12.5% by scraping
stray digits out of prose like "ranks 19 of 21" — that misrepresents the system
as partly capable when it is not capable at all. It now returns `unsupported`
and scores zero across 250 of 800 questions.

**Object-level questions are out of reach.** "Is a *circular* building present?"
scores at chance: a scene-level land-cover classifier cannot resolve an
individual instance and its shape.

A controlled comparison of five yes/no decision rules
([results](../eval/results/decision_rules.md)) found the adopted rule best, and
— more informatively — that rules answering "yes" often score *below* chance.
The backbone's ranking is anti-correlated with RSVQA's notion of presence.

## VRSBench VQA

1,440 questions, stratified across all 12 types.

| question type | n | exact | lenient | majority |
|---|---|---|---|---|
| rural or urban | 120 | 40.0% | 43.3% | 40.8% |
| object existence | 120 | 20.8% | 20.8% | 83.3% |
| reasoning | 120 | 12.5% | 12.5% | 37.5% |
| scene type | 120 | 5.0% | **14.2%** | 11.7% |
| image | 120 | 2.5% | 2.5% | 51.7% |
| the eight object-level types | 960 | ≤3.3% | ≤3.3% | 9–30% |
| **overall** | **1440** | **7.6%** | **8.7%** | |

This is a poor result and the table says why rather than hiding it. Eight of
twelve types ask about an individual object's colour, count, position, size,
shape, direction or category — none of which a scene-level backbone addresses.

The lenient column gives partial credit when the answer is semantically right
but worded differently ("inland waters" against gold "Body of water"). It
matters in exactly one place: **scene type**, which goes 5.0% → 14.2% and is
the only type to clear its own baseline. Everywhere else lenient ≈ exact,
confirming those are genuine failures rather than vocabulary mismatches.

Two decoding bugs were fixed here and both were ours, not the benchmark's: the
land-cover answer was being read as the literal word "most" from the sentence
introducing it, and the either/or pattern required an *is/are* opener so
"**Does** the image depict a rural or urban area?" fell through to yes/no.
Fixing them took rural/urban from 5.0% to 40.0%.

## VRSBench referring grounding

Referring expressions with real pixel boxes, Acc@0.5 IoU.

| metric | shipped (cross-modal) | optical-only |
|---|---|---|
| Acc@0.5 IoU | 0.2% | 0.4% |
| Acc@0.25 IoU | 1.8% | 4.3% |
| mean IoU | 0.018 | 0.035 |
| n | 400 | 1,078 |

The shipped checkpoint is the weaker of the two here, consistent with the
RSVQA trade-off. Both are effectively zero, so the choice between them does not
matter for this task — the numbers are given for the shipped configuration
rather than the more flattering one.

A clear negative result, and structural rather than a tuning problem:

| target size | n | mean IoU |
|---|---|---|
| large (>4 patch cells) | 19 | 0.089 |
| 1–4 cells | 150 | 0.035 |
| sub-cell | 112 | 0.002 |
| tiny (<¼ cell) | 119 | 0.001 |

Grounding comes from CLIP patch tokens on a 7×7 grid — 73 px per cell over a
512 px image — while VRSBench refers to individual vehicles, most smaller than
one cell. Accuracy falls monotonically with target size, exactly as that limit
predicts.

Tiling was implemented to test whether resolution alone explained it. A T×T
grid of overlapping crops gives 7T×7T cells:

| tiles | passes/image | mean IoU |
|---|---|---|
| 1 | 1 | 0.006 |
| 3 | 9 | 0.019 |
| 5 | 25 | 0.025 |

Monotonic, so resolution is part of the story — but four times a base of 0.006
is still unusable. CLIP patch tokens were never supervised for object
localisation, and tiling cannot supply supervision that was never there.
Closing this needs an open-vocabulary detection head.

`tiles` is a permitted parameter defaulting to 1: worth raising for small
targets, not worth 25 forward passes for the scene-level regions the
statement's own grounding example asks about.

## VRSBench captioning

400 images.

| metric | value |
|---|---|
| BLEU-1 | 0.011 |
| BLEU-4 | 0.008 |
| ROUGE-L | 0.026 |
| content-word recall | 0.030 |

VRSBench references are long human descriptions naming vehicles, colours and
spatial relations; this system emits a land-cover summary from a 21-class
vocabulary. The two barely share vocabulary, so these numbers measure a
difference in kind more than a difference in quality. Reported anyway, because
captioning is a named task and a missing number would be the wrong kind of
silence.

## Which checkpoint ships, and what it costs

The cross-modal checkpoint is the default. It is the only one under which the
mandatory optical–SAR capability works at all: without it SAR-only sits at
12.4% and fusion is actively harmful.

It costs 2.2 points overall on RSVQA-LR and 6.3 on that benchmark's scene-level
subset, measured on the same 800 questions. That trade is worth taking — a
mandatory capability that works beats a few points on a benchmark where the
system is near chance either way — but it is a real cost and is recorded here
rather than left out.

## Reproducing

```bash
PYTHONPATH=. python3 eval/crossmodal.py --limit 600
```

```bash
PYTHONPATH=. python3 eval/cdvqa.py
```

```bash
PYTHONPATH=. python3 eval/rsvqa.py
```

```bash
PYTHONPATH=. python3 eval/vrsbench_vqa.py --per-type 120
```

Results are written to `eval/results/*.json`.
