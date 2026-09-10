---
title: SatQuery AI
emoji: 🛰️
colorFrom: blue
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Ask satellite imagery a question in plain English
---

# SatQuery AI

An agentic vision-language assistant for multimodal remote sensing.
**Smart India Hackathon 2026 · Problem Statement `SIH26167` · ISRO.**

Give it one or two satellite images and a question in English. A controller
classifies the task, checks the imagery can actually answer it, selects
specialist models from a registry, runs them, fuses the outputs, estimates
confidence, and returns the answer with visual evidence and a full execution
trace.

## What it does

| Input | Tasks |
|---|---|
| Single image | visual question answering, captioning, referring grounding |
| Optical + SAR pair | joint information extraction across sensors |
| Two dates, same place | change description, change VQA |

## Measured results

Every figure is on held-out data, through the real serving path, reported
beside the reference it has to beat.

| benchmark | result | reference |
|---|---|---|
| VRSBench VQA, overall | **55.2%** | 7.6% scene-level only |
| VRSBench captioning, ROUGE-L | **0.306** | 0.026 |
| Referring grounding, Acc@0.5 | **28.0%** | 0.2% patch-token |
| CDVQA change VQA | **64.3%** | 47.3% hand-written heuristic |
| RSVQA-LR, overall | **51.7%** | 34.9% |
| Optical–SAR fused, P@3 | **37.6%** | 18.9% better single sensor |

Twelve of twelve VRSBench question types beat their own majority baseline.

## Honest limitations

- **Running on CPU.** Expect several seconds per query; the numbers above were
  measured on GPU and are unaffected by the hardware serving this demo.
- **Counting is weak** at 23.8% — at 10 m ground sample distance the difference
  between six and seven objects may not be in the data at all.
- **Comparison questions regressed** to 43.8% from 54.5%: comparing needs two
  counts, and the generative specialist answers in one shot without counting.
- Confidence is **capped when a component runs unadapted**, so an answer from a
  Space without its weights will report low confidence by design.
