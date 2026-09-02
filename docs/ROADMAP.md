# Roadmap

What is done, what is open, and what each open item is worth. Ordered by what
the problem statement actually scores.

## Done

- Agentic controller implementing the statement's six enumerated steps, with
  permitted-parameter filtering enforced structurally.
- Five specialist tools: VQA, captioning, grounding, change, optical–SAR.
- Remote-sensing adaptation on BigEarthNet.txt, measured on held-out patches
  (zero-shot land-cover 2.1% → 45.8%; i2t R@5 1.8% → 28.5%).
- Web application returning answer, confidence, visual evidence, and the full
  execution trace on every request.
- Self-contained downloadable reports, HTML and JSON.
- 49 tests, no network or weights required.

## Open, in priority order

### 1. Benchmark evaluation — highest value

The statement names the benchmarks it will score against: VRSBench and RSVQA
for single-image captioning, grounding, and VQA; CDVQA for change VQA. All
three are available and ungated on HuggingFace.

No benchmark numbers are claimed anywhere in this repo until this is run. It is
the single largest gap between "works" and "demonstrably works".

### 2. Bi-temporal data

The current BigEarthNet shard is single-date optical, so the change and
cross-modal tools are unit tested but unmeasured on real pairs. CDVQA supplies
genuine bi-temporal pairs and closes this at the same time as (1).

### 3. Zero-shot land-cover below the majority baseline

45.8% against a 49.0% majority-class baseline on a 96-patch subset dominated by
broad-leaved forest. The baseline is degenerate — it names one class always —
but the gap is real. Worth attacking with class-balanced prompts, a larger
adaptation shard, or prompt ensembling.

### 4. Longer captions than CLIP can read

BigEarthNet captions run past the 77-token limit and their discriminative
content sits at the end, which is why retrieval on them is near chance for any
model. A long-context text encoder, or sentence-level chunking with pooled
supervision, would make retrieval a usable metric rather than a reported one.

### 5. SAR-specific handling

`_open()` takes bands 4/3/2 as natural colour, which is right for Sentinel-2
and wrong for SAR. Sentinel-1 needs its own composite (VV/VH/ratio) before the
cross-modal tool can be trusted on real RISAT-style inputs.

## Explicitly out of scope

- An LLM planner in the controller — see [ADR 0001](adr/0001-agentic-controller.md).
- Internal reasoning text, which the statement says is neither required nor
  evaluated.
