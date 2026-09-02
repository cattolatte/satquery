# Architecture

Three backbones with disjoint competences, many cheap heads, and a controller
that picks between them per query and is allowed to say no.

```
query ─┐
       ├─> router.classify ──> reconcile(task, input_kind) ──> Task
image ─┴─> inspect_image ────> classify_inputs ─────────────> InputKind
                                                                │
                                    Registry.for_task ──────────┤
                                                                ▼
                                    Tool.invoke ──> filter_params ──> run
                                                                │
                                    fuse + confidence ──────────┤
                                                                ▼
                                                    Answer + Trace
```

## Why the shape is this shape

**The trace is an output, not a log.** The statement says only the observable
execution trace is evaluated, so `Trace` is a dataclass returned on every
request — including successful ones — and the web UI renders it beside the
answer rather than hiding it behind a debug flag.

**Parameters are filtered structurally.** The statement requires the controller
to "configure only permitted task parameters". Rather than trusting callers, a
tool declares `accepts` in its `ToolSpec` and `Tool.invoke()` filters before
dispatch, recording what it dropped. A tool cannot receive a parameter it never
declared even if the caller sends one.

**Reconciliation degrades explicitly.** A change question asked of a single
image becomes VQA *with the reason written into the trace*, rather than failing
outright or silently pretending. `test_router.py` asserts that reconciliation
can never return a task the inputs cannot support, for every task/input pair.

**Confidence is the minimum, not the mean.** A chain is only as trustworthy as
its weakest step; averaging lets one confident tool disguise an uncertain one.
Unverified input assumptions cost a fixed penalty on top, because an answer
computed over a pair we could not confirm covers the same ground is worth less
regardless of how sure the model was.

**Tools fail soft, the backbone fails hard.** A missing dependency makes one
tool unavailable and the controller routes around it — losing coverage beats
losing the query. The backbone is the exception: it refuses to load rather than
return partly-initialised weights, because a silently random backbone makes
every downstream number meaningless. See [ADR 0002](adr/0002-remote-sensing-backbone.md).

## Modules

| Module | Responsibility |
|---|---|
| `schema.py` | `Task`, `InputKind`, `ImageMeta`, `ToolCall`, `Evidence`, `Trace`, `Answer` |
| `io/inspect.py` | format, bands, modality, georeferencing; pair compatibility |
| `controller/router.py` | query → task, then task × inputs → executable task |
| `controller/agent.py` | the statement's six enumerated steps, in order |
| `tools/base.py` | `ToolSpec`, the `Tool` contract, `Registry` |
| `tools/backbone.py` | the adapted CLIP; embeddings and dense patch tokens |
| `tools/specialists.py` | land-cover VQA, captioning, patch-token grounding |
| `tools/detector.py` | open-vocabulary detection, referring resolution |
| `tools/generative.py` | free-form VQA, captioning and grounded boxes |
| `tools/multi_image.py` | change detection, optical–SAR analysis |
| `adapt/` | BigEarthNet pair preparation, conversion, fine-tuning |
| `server.py` / `web/` | HTTP API and the interactive application |
| `report.py` | self-contained downloadable reports |

## Why three backbones

They fail in different directions, and each was added because a measurement
said the previous set could not reach something.

**CLIP over a class vocabulary** ranks land cover well and is well calibrated on
the vocabulary it was adapted to. It has no notion of an instance, so it cannot
count, locate, or compare two objects — and it can only ever emit a word from
its vocabulary, which caps VRSBench VQA at 65.5% before perception is
considered.

**The detector** supplies instances: boxes, counts, positions and sizes. It took
referring grounding from 0.2% to 25.1% and made counting possible at all. It has
no notion of land cover, so "highlight the water body" goes elsewhere.

**The generative model** removes the vocabulary ceiling and is the only
component that can write a caption: references average 48 words against a
five-class template. It is the least calibrated of the three, which is why
confidence is capped when its adapter is absent.

The controller chooses between them from the query, not the task, because two
tools may serve the same task and answer disjoint question sets. Selection is
by name rather than by taking the first candidate — the registry sorts
alphabetically, and relying on that position once sent every scene-level
question to the detector.

## Grounding without box supervision

CLIP aligns only the pooled CLS token with text, so patch tokens are not
directly comparable to text embeddings. Projecting them through the same visual
projection puts them in the joint space, which makes text-driven localisation
possible with no box labels at all — connected components over the similarity
map become boxes.

It is approximate, and the confidence function says so: it measures how far the
selected patches separate from the rest and decays as the mask spreads across
the scene, because a mask covering most of the image is not an answer to
"where is it" however peaked it looks.
