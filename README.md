<div align="center">

# SatQuery AI

**An agentic vision-language assistant for multimodal remote sensing**

Ask a satellite image a question in plain English. The system works out which task
you mean, checks the imagery can actually answer it, routes to the right specialist
models, and returns a grounded answer with visual evidence and an auditable trace.

[Architecture](docs/ARCHITECTURE.md) · [Conformance](docs/CONFORMANCE.md) · [Decisions](docs/adr/README.md) · [Run it](RUN.md)

![Tasks](https://img.shields.io/badge/tasks-6%20routed-6ea8fe)
![Inputs](https://img.shields.io/badge/inputs-single%20%C2%B7%20cross--modal%20%C2%B7%20bi--temporal-6ea8fe)
![Formats](https://img.shields.io/badge/formats-GeoTIFF%20%C2%B7%20TIFF-818cf8)
![Trace](https://img.shields.io/badge/execution%20trace-first--class-4ade80)

<sub>Smart India Hackathon 2026 · Problem Statement <code>SIH26167</code> · ISRO / Department of Space</sub>

</div>

---

> **What it looks like in use**
>
> ```
> query : "What changed between these two dates, and where did the change occur?"
> inputs: 2 × GeoTIFF, optical, same CRS, bounds overlap, dates differ
> task  : change_description          (routed from query, confirmed against inputs)
> tools : rs_change_describer → grounding
> answer: built-up area expanded along the north-east edge of the settlement
> evidence: change map + 3 bounding boxes
> confidence: 0.78
> ```

---

## Why this is not one model with a prompt

The problem statement is explicit that a general model will not do:

> "A general-purpose large language model (LLM) or vision-language model (VLM)
> cannot be expected to perform these specialised tasks reliably without
> adaptation to remote-sensing imagery, sensor characteristics, and
> domain-specific terminology."

and it names the novelty:

> "The novelty of SatQuery AI lies in its **agentic, query-driven framework**.
> Instead of applying a single generic VLM, the system **selects and executes
> suitable remote-sensing specialist models, validates inputs, combines their
> outputs**, and returns an evidence-grounded response."

So the architecture is a controller over a registry of specialists, not a
wrapper around one model.

## The six controller steps, as specified

The problem statement enumerates what the controller must do. `Controller.run()`
implements them in order, one block each, because the trace is graded against
this list and the code should read against it too.

| # | Requirement | Where |
|---|---|---|
| 1 | interpret the query and classify the requested task | `controller/router.py` |
| 2 | check number, modality, format, metadata, compatibility | `io/inspect.py` |
| 3 | select tools from a predefined registry | `tools/base.py` |
| 4 | configure **only permitted** parameters and execute | `Tool.filter_params` |
| 5 | combine outputs, estimate confidence, return evidence | `Controller._fuse` |
| 6 | provide an auditable execution summary | `schema.Trace` |

## Input configurations

| Kind | What it is | Tasks it unlocks |
|---|---|---|
| Single | One optical/multispectral **or** SAR image | VQA, captioning, grounding |
| Cross-modal pair | Co-registered optical **+** SAR, same area | joint extraction, plus the above |
| Bi-temporal pair | Same area, two dates, matching modality | change description, change VQA |

Mismatches are reconciled and recorded rather than silently answered. A change
question asked of a single image is downgraded to VQA, with the reason written
into the trace.

## Status

All mandatory scope is built and measured on the benchmarks the problem
statement names. Every number in [Benchmarks](docs/BENCHMARKS.md) comes from
held-out data through the real serving path, reported beside the reference it
has to beat.

| benchmark | result | reference |
|---|---|---|
| VRSBench VQA, overall | **55.2%** | 7.6% scene-level only |
| VRSBench captioning, ROUGE-L | **0.306** | 0.026 |
| VRSBench referring grounding, Acc@0.5 | **28.0%** | 0.2% patch-token |
| CDVQA change VQA, routed | **64.3%** | 47.3% heuristic |
| RSVQA-LR, overall | **51.7%** | 34.9% |
| Optical–SAR fused, P@3 | **37.6%** | 18.9% better single sensor |

Twelve of twelve VRSBench question types beat their own majority baseline.
173 tests, no network or weights required.

What is open, and what regressed, is in [the roadmap](docs/ROADMAP.md) and
[Benchmarks](docs/BENCHMARKS.md) — including RSVQA comparison questions, which
fell 54.5% → 43.8% when the generative specialist was adopted.
