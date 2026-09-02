# ADR 0006 — Add a detector, and choose between backbones per query

**Status:** accepted

## Context

The statement invites this directly: "The system may use multiple specialised
components, such as a remote-sensing VQA or captioning model, a grounding
model, a change-understanding or change-VQA model, and an optical–SAR fusion or
information-extraction model."

After the first round of benchmarking, every remaining weakness turned out to
be the same weakness. A global image–text similarity has no notion of an
*instance*, so it cannot count, cannot say where a thing is, cannot compare two
things' sizes, and cannot localise anything smaller than a patch cell:

- VRSBench referring grounding: 0.2% Acc@0.5, with accuracy falling
  monotonically as targets got smaller.
- RSVQA counting: not attempted at all, 250 of 800 questions.
- Eight of VRSBench VQA's twelve types, all near or below their baselines.

That is one missing capability, not four separate problems.

## Decision

Add OWLv2 as an open-vocabulary detection specialist, and make the controller
choose between the scene-level backbone and the detector **per query**.

Two stages, because a referring expression is not a detection query. The
detector finds *all* the vehicles; the sentence names *one* — "the large yellow
vehicle at the top-left". 95% of VRSBench's referring expressions state a
position, 40% a size and 12% a colour, so a second stage scores every candidate
against the constraints the sentence actually states.

## Results

| task | before | after |
|---|---|---|
| VRSBench referring grounding, Acc@0.5 | 0.2% | **25.1%** |
| VRSBench referring grounding, mean IoU | 0.018 | **0.216** |
| VRSBench VQA overall | 7.6% | **11.2%** |
| RSVQA counting | not attempted | **8.8%** |
| RSVQA overall | 34.9% | **37.6%** |

Constraint resolution is worth 3.3 points of Acc@0.5 over taking the
highest-scoring detection, which is the part that could not have been bought by
swapping models.

The size profile also **inverted**, which is the clearest evidence the
mechanism is the intended one: under patch-token grounding, tiny targets scored
worst (mean IoU 0.001) because they were smaller than a grid cell. Under
detection they score comparably to large ones.

## What it cost, and what was measured rather than assumed

Three routing policies were tried, because the subject noun does not settle
whether a question is instance-level — "road" and "building" name discrete
things in one corpus and ground cover in another:

| routing | RSVQA | VRSBench VQA |
|---|---|---|
| any head noun | 33.0% | 13.5% |
| discrete nouns, or an attribute in the question | 35.5% | 11.4% |
| **discrete nouns only** | **37.6%** | 11.2% |

The attribute test is the more principled rule and it lost. It is a superset of
the adopted routing, and the extra questions it sent to the detector cost 2.1
points on RSVQA to gain 0.2 on VRSBench. Kept out on that evidence.

Thresholds are split by what the question needs rather than shared:

- **Localisation, 0.03.** A weak box still ranks; discarding it guarantees a
  miss. At 0.10, 78 of 150 expressions produced nothing; at 0.03, 37 did.
- **Existence, 0.40.** Asserting "yes, there is one" on a weak detection is a
  false claim. At 0.10 it made presence accuracy *worse* than the scene-level
  backbone it replaced. Tuned on the first half of RSVQA's presence questions,
  reported on the second: 46% at 0.10, 60% at 0.40 — which is also the majority
  baseline there, so this stops harm rather than adding signal.

## Consequences

- `_plan` now takes the query, because selecting between two tools that serve
  the same task cannot be done from the task alone. This is the selection step
  the statement asks the controller to perform.
- Tools are chosen **by name**, not by taking the first candidate. The registry
  orders alphabetically, so `rs_detect` sorted ahead of `rs_vqa` and briefly
  sent every scene-level question to the detector — scene-level accuracy fell
  to 0.6% before the cause was found. Position in a sorted list is not a
  priority signal.
- Grounding splits by target: the detector localises objects, patch-token
  similarity localises land cover. "Highlight the water body" names a region,
  not an instance, and asking a detector for it returns whatever objects happen
  to be nearby.
