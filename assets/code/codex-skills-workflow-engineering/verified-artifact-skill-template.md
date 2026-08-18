---
name: verified-artifact-workflow
description: Create or revise a structured artifact when semantic accuracy, deterministic generation, rendered inspection, and repeatable acceptance checks are required.
---

# Verified artifact workflow

## Input contract

Before generation, write a requirements ledger containing:

- the artifact's purpose and audience;
- required semantic elements and forbidden additions;
- output formats and editability requirements;
- hard acceptance checks;
- tool, version, and portability constraints.

If a missing choice changes the meaning of the result, ask for it. Otherwise,
record a conservative assumption and continue.

## Canonical source

Create a semantic source before creating the final rendered artifact. Prefer a
small structured representation whose identifiers remain stable across
iterations. Do not use rendered coordinates as the only source of truth.

## Deterministic generation

Run the task-specific generator from `scripts/` when available. Pin or record
the engine version. Change the semantic source or generator options when a
result is wrong; do not patch generated coordinates unless the output format
itself is the requested source.

## Verification loop

For each completed pass:

1. Validate source structure and required identifiers.
2. Generate the editable artifact and a rendered preview.
3. Inspect the complete render and any risk-heavy regions.
4. Check the render against the requirements ledger.
5. Ask an independent reviewer to return `PASS` or `REVISE` with hard issues
   separated from optional suggestions.
6. If revision is required, change the smallest responsible semantic rule or
   generator option, regenerate, and repeat.

Stop after three failed review cycles and report the remaining conflict rather
than continuing unbounded aesthetic tuning.

## Evidence bundle

Deliver:

- the requirements ledger;
- the canonical semantic source;
- the editable generated artifact;
- a rendered preview;
- engine and version metadata;
- the final acceptance result.

## Boundaries

- Keep domain interpretation in the model-facing instructions or references.
- Keep geometry, serialization, formatting, and other deterministic work in
  scripts or dedicated engines.
- Keep acceptance criteria explicit and observable.
- Preserve a previous working workflow as a control when changing the
  architecture materially.
