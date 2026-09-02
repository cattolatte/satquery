# ADR 0005 — Adapt one encoder to both modalities

**Status:** accepted

## Context

The statement makes optical–SAR joint analysis mandatory: "The system must
extract complementary information from a co-registered optical/multispectral
and SAR image pair." It also describes the adaptation dataset as "co-registered
Sentinel-1 SAR, Sentinel-2 multispectral imagery, and diverse text annotations".

The first adaptation pass used only the optical half.

## What that cost, measured

Optical–SAR was the one mandatory capability with no measurement at all,
because the shard we had was optical-only. Getting real co-registered pairs
made the cost visible:

| reading | P@3 |
|---|---|
| optical only | 21.1% |
| SAR only | 12.4% |
| fused (max) | 18.9% |

Fusion scored **below optical alone**. That is the expected result once stated
plainly: averaging a weak reading into a good one dilutes it, so "combining two
sensors" is not automatically an improvement and must be verified rather than
assumed.

Reliability weighting — weighting each sensor by the peakedness of its own
score distribution — did not fix it. Peakedness measures confidence, not
correctness, and the model was confidently wrong on data it had never seen. The
weighting is kept because it is the right shape for the problem, but it was not
the fix.

## Decision

Train **one** encoder on both modalities against the same text, continuing from
the optical checkpoint rather than restarting.

One encoder rather than two keeps the one-backbone-many-heads shape the
registry depends on: every specialist is a head, and a second encoder would
double the cost of every tool. Continuing from the optical checkpoint rather
than restarting preserves the optical adaptation instead of relearning it.

The sensor is named in each caption ("a Sentinel-1 synthetic aperture radar
image of…"), so the model can use it as context rather than having to explain
away why the same ground looks so different in radar.

14,360 samples from 7,180 co-registered pairs.

## Result

| reading | before | after |
|---|---|---|
| optical only | 21.1% | 34.5% |
| SAR only | 12.4% | **36.2%** |
| fused (max) | 18.9% | **37.6%** |
| fusion vs. better single | −2.2% | **+1.4%** |

SAR now slightly outperforms optical, and fusion beats the better single sensor
rather than losing to it. The margin is about eight samples in six hundred and
is not claimed as more; the sign changing is the result.

Zero-shot on the mixed held-out set went 4.4% → 81.3% against a 61.9% majority
baseline. Selection kept epoch 0 — zero-shot fell every epoch after while loss
kept dropping, the same lesson the first adaptation taught.

## Consequences

- **It costs accuracy elsewhere.** On the same 800 RSVQA-LR questions the
  cross-modal checkpoint scores 34.9% against the optical-only checkpoint's
  37.1%, and 50.8% against 57.1% on the scene-level subset. The trade is worth
  taking, because a mandatory capability that works beats a few points on a
  benchmark where the system is near chance either way — but it is a real cost
  and is recorded rather than omitted.
- Backbone provenance is now read from the checkpoint's own `adaptation.json`
  rather than hard-coded. The hard-coded stage names kept claiming the first
  checkpoint's history after the weights had been replaced, and a stale answer
  to "which weights are these" is worse than none.
- Two rendering bugs had to be fixed before any of this could be measured; see
  [BENCHMARKS.md](../BENCHMARKS.md). Both needed real sensor data to surface,
  which is the argument for running on the real benchmark early rather than
  trusting that a tool which returns plausible output is correct.
