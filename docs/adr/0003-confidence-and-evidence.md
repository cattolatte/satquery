# ADR 0003 — Confidence is the minimum across the chain

**Status:** accepted

## Context

The statement asks the system to "combine textual and spatial outputs, estimate
confidence, and return visual evidence". A confidence number that does not
track correctness is worse than none: it invites trust it has not earned.

## Decision

Aggregate confidence is the **minimum** across the executed chain, not the mean,
with a fixed penalty for unverified input assumptions.

A chain is only as trustworthy as its weakest step. Averaging lets one confident
tool disguise an uncertain one — exactly the case where a caller most needs the
warning.

## The bug that motivated the rule

Grounding confidence was originally the maximum of its heat map. The heat map is
min-max normalised, so its maximum is **1.0 by construction**. Every successful
localisation reported perfect certainty, no matter how weak the evidence.

It now measures two independent ways a localisation can be worthless:

- **No separation** — selected patches barely outscore the rest, caught by the
  z-score of their mean against the map.
- **No localisation** — a mask covering most of the scene is not an answer to
  "where is it", so confidence decays as coverage passes half the image.

A regression test asserts confidence never reaches 1.0 on realistic maps.

## Consequences

- Per-tool confidences use domain-appropriate signals: softmax margin over the
  class vocabulary for VQA and captioning, separation-and-focus for grounding.
- Binary VQA reports probability mass against the uniform prior, so "present"
  is distinguishable from "least unlikely of twenty-one".
- Any normalised quantity is now suspect as a confidence signal. Normalisation
  destroys the scale that made the number meaningful.
