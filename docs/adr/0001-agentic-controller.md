# ADR 0001 — An agentic controller over a predefined registry

**Status:** accepted

## Context

The statement locates the novelty precisely: "The novelty of SatQuery AI lies in
its agentic, query-driven framework. Instead of applying a single generic VLM,
the system selects and executes suitable remote-sensing specialist models,
validates inputs, combines their outputs, and returns an evidence-grounded
response."

It then enumerates six controller steps and adds that "only the observable
execution trace … will be evaluated. Internal reasoning text is neither required
nor evaluated."

## Decision

A deterministic controller over a declared registry, not an LLM planner.

`Controller.run()` implements the six steps in the order given. Tools declare
their tasks, accepted parameters, image count, and dependencies in a `ToolSpec`.
Selection is a registry lookup, not generation.

## Why not an LLM planner

It would be the obvious "agentic" choice and the wrong one here.

The trace is what gets marked, and a deterministic controller produces a trace
that is complete and reproducible by construction. An LLM planner would add a
failure mode — hallucinated tool names, invented parameters — directly on the
evaluated surface, in exchange for flexibility over a task space with six
members and a registry with five tools.

The statement explicitly does not ask for internal reasoning text. Generating
reasoning that is neither required nor evaluated would be cost with no score.

## Consequences

- Routing is regex over a task taxonomy, which is auditable and testable, and
  every representative query in the statement is a test case.
- Adding a tool is registering a `ToolSpec`; nothing else changes.
- The limitation is honest: a query phrased far outside the patterns falls back
  to VQA. `reconcile()` records the fallback in the trace rather than hiding it.
