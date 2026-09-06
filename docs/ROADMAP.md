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
- 173 tests, no network or weights required.

## Open, in priority order

### ~~1. Benchmark evaluation~~ — done

All three named benchmarks are now run and reported in
[BENCHMARKS.md](BENCHMARKS.md): VRSBench (VQA, captioning, referring
grounding), RSVQA-LR, and CDVQA for change VQA, plus the optical–SAR pairing
the statement makes mandatory. Every number comes from the real serving path.

What this surfaced, and what is now open in its place:

- **RSVQA comparison questions regressed**, 54.5% → 43.8%. Counting two classes
  and comparing them is compositional; the generative model answers in one shot
  without counting either. Needs a count-then-compare path, not more training.
- **Counting is weak** at 23.8% on RSVQA-LR. Partly a resolution limit at 10 m
  GSD, partly a model limit, and the two have not been separated.
- **RSVQA presence adaptation does not transfer.** The adopted decision rule
  beats a 50.4% majority baseline by five points, and two candidate rules scored
  *below* chance — the backbone's ranking is anti-correlated with RSVQA's notion
  of presence. BigEarthNet's CORINE land-cover vocabulary does not carry to
  RSVQA-LR's object-centric annotation, despite the same sensor family.

### ~~2. Bi-temporal data~~ — done

CDVQA supplied genuine bi-temporal pairs. Change VQA is measured at 64.3%
routed, against 47.3% for the hand-written heuristic that remains in the system
as the reference it has to beat.

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
