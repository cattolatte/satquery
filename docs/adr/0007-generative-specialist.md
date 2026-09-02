# ADR 0007 — A generative specialist, trained on the benchmarks' own data

**Status:** accepted

## Context

Two separate causes were measured, not assumed.

**A hard output ceiling.** Of VRSBench's 37,409 VQA answers, only 65.5% can be
expressed by ranking a 21-class land-cover vocabulary at all. "Body of water",
"residential", "grayscale" and "high" are outside it at any confidence.
Captioning was the sharper case: references average 48 words against a
five-class template.

**A larger task mismatch.** The score was 11.5% against that 65.5% ceiling —
under a fifth of what was already reachable. The backbone had been adapted on
BigEarthNet land-cover captions and asked object-centric aerial questions it had
never seen. Meanwhile VRSBench ships 142,390 training conversations covering
exactly the three tasks it scores, and none of it had been used. The statement
permits it explicitly: adaptation may use "BigEarthNet.txt or the any open
source training data".

Generation alone would have lifted the ceiling and left most of the gap.
Training alone would have closed the gap to a ceiling that still capped a third
of the benchmark. Both were needed.

## Decision

SmolVLM-500M, LoRA fine-tuned on 24,000 VRSBench training conversations,
registered as a specialist the controller selects rather than a replacement for
the others.

Model choice was made on evidence and constraint. Florence-2 was the better
architectural fit — native captioning, grounding and VQA heads — and was
rejected because its published weights predate the native implementation in
transformers 5.x and neither loading path works. Qwen2-VL-2B loads and
generates, and is four times the size: on this hardware a run that converges
beats a larger one that does not, and for narrow in-domain benchmarks the
supervision matters more than the base model.

4.2M trainable parameters of 512M, 0.81%. Held-out exact match 25.0% → 41.7%.

## Results

| benchmark | before | after |
|---|---|---|
| VRSBench VQA, overall | 11.5% | **36.8%** |
| VRSBench VQA, scene-level | 15.3% | **67.3%** (baseline 35.0%) |
| VRSBench captioning, ROUGE-L | 0.026 | **0.217** |
| VRSBench captioning, content recall | 0.030 | **0.417** |
| RSVQA-LR, overall | 41.5% | **51.7%** |
| RSVQA-LR, presence-scene | 46.2% | **94.6%** |

Six VRSBench question types now beat their own majority baselines, where
previously none did outside `rural or urban`: `image` 85.0% against 51.0%,
`rural or urban` 77.0% against 43.0%, `scene type` 40.0% against 11.0%,
`object color` 36.0% against 17.0%, `object category` 30.0% against 11.0%, and
`object shape` 25.0% against 22.0%.

## What did not improve, and what regressed

**Grounding stays with the detector.** Measured head to head on the same 300
expressions: the detector reaches 19.0% Acc@0.5, the generative model 0.7%. Box
regression got roughly 6,000 of the 24,000 training rows and one epoch, which is
not enough to learn coordinates. The controller keeps routing grounding to the
detector, on that evidence.

**RSVQA comparison questions regressed**, 54.5% to 43.8%. "Are there more
farmlands than water areas?" requires counting two classes and comparing them,
which the generative model answers in one shot without counting either.

An extent-based fix was implemented and then removed. Assigning every patch to
its nearest class and comparing the counts is the right model of the question,
and it measured 39.3% over the full vocabulary and 42.9% restricted to the two
classes named, against a 55.4% majority baseline — worse than guessing, in both
formulations. Inverting the comparison would have scored 59%, which is a fact
about how this vocabulary assigns built-up patches rather than a model of
anything, so it was not adopted. The regression stands unfixed and is reported
as such.

That diagnosis is itself useful: the anti-correlation says the backbone
systematically under-assigns built-up classes on this imagery, which is a
calibration problem worth attacking directly rather than papering over.

**Counting remains weak** at 23.8%, and object direction at 10.0% is below its
baseline. Neither is addressed by this change.

## Consequences

- Three backbones now serve overlapping tasks, so the controller's per-query
  selection carries more weight. Each was added because a measurement said the
  previous set could not reach something.
- Training and serving share one definition of the base model, image size and
  task prompts. They had already drifted once: the processor defaults to a
  longest edge of 2048 while training ran at 512, so every served image was
  being upscaled fourfold against the weights it was fed.
- The captioning evaluation had to be fixed before it could show any of this.
  It called `rs_caption` directly, so it measured the template head whatever
  else was registered, and reported an unchanged 0.026 after the change. An
  evaluation that cannot observe an improvement argues against it.
