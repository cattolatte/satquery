# ADR 0002 — Backbone: converted RemoteCLIP, fine-tuned on BigEarthNet.txt

**Status:** accepted

## Context

The problem statement makes adaptation mandatory and says so twice, ending with
"A generic LLM or VLM without remote-sensing adaptation will not satisfy the
requirements." So the backbone choice is a conformance decision, not only a
quality one.

## The failure that shaped this decision

`CLIPModel.from_pretrained("MVRL/remote-clip-vit-base-patch32")` returns a
**randomly initialised model**. It raises nothing. RemoteCLIP — and GeoRSCLIP,
and SkyCLIP — ship in `open_clip` layout, whose tensor names share nothing with
HuggingFace's: all 398 expected tensors are reported missing and all 302
checkpoint tensors unexpected, and transformers silently initialises the lot.

Fine-tuning would have run, loss would have fallen, and every number would have
been meaningless.

`adapt/convert_openclip.py` does the rename explicitly and raises rather than
returning a partly-populated model.

## Verifying the conversion

Retrieval could not verify it: both models sat at chance, because BigEarthNet
captions are templated and their discriminative content falls past CLIP's
77-token limit. A test that cannot fail is not a test.

The check that worked was numerical equivalence against `open_clip` itself —
**cosine 1.000000** on both towers.

Getting there required a correction. The first comparison gave 0.91, which
looks like a conversion bug. A control on *identical stock OpenAI weights*
showed the same 0.92 gap, proving the discrepancy was in the reference, not the
conversion: `open_clip` 3.x splits `ViT-B-32` from `ViT-B-32-quickgelu`, and
OpenAI-lineage CLIP is the latter. Against the right variant, exact.

The lesson generalises: when a comparison disagrees, control on a case where
the answer is already known before concluding which side is wrong.

## Which base

Measured, not assumed, since RemoteCLIP is adapted to high-resolution aerial
imagery while BigEarthNet is 120×120 Sentinel-2 at 10 m/px.

| | RemoteCLIP | stock CLIP |
|---|---|---|
| zero-shot before | 2.1% | 8.3% |
| zero-shot after | **45.8%** | 42.7% |
| i2t R@5 after | **28.5%** | 25.6% |
| t2i R@10 after | **44.0%** | 39.6% |

RemoteCLIP starts *worse* and finishes *better*. Its zero-shot prompt alignment
transfers badly to Sentinel-2, but its features are the better starting point
once adapted.

The zero-shot margin is 3 samples in 96 and is not decisive alone. All six
retrieval metrics agreeing in the same direction is the stronger signal.

## Consequences

- The backbone is one model with many cheap heads, so each specialist is a head
  rather than a model, which is what makes the registry affordable.
- Selection during training is on zero-shot land-cover, not retrieval, because
  retrieval on templated captions cannot separate a good model from a bad one.
- Early stopping matters: zero-shot peaks at epoch 3 and declines to 38.5% by
  epoch 5 while loss keeps falling. Loss is not the metric.
- The converter is tested by round-tripping a tiny CLIP through `open_clip`
  layout and back — offline, one second, and stronger than eyeballing one real
  checkpoint.
