# Glossary

## Resource

An input consumed by a run, such as the raw dataset, `paths.yaml`, a task
profile, threshold YAML, or a snapshotted decision policy. Resource descriptors
record paths, hashes where available, and whether the resource is required.

## Artifact

A generated output with a typed envelope, run identity, provenance, generator,
configuration, software, and determinism metadata. The run directory is the
source of truth for a completed execution.

## Validated Run Context

The resolved contract created after configuration validation. It binds dataset,
task profile, paths, thresholds, calibration source, known case IDs, and
expected artifacts before evidence generation.

## Dataset Validation

Task-aware checks that raw resources and required segmentation files exist.
Validation failures are evidence available to routing and final policy.

## Task Profile

A YAML declaration of required components and segmentations, active QC domains,
hard-failure domains, and calibratable domains. `pancreas_only` also declares the
visible-pancreas task identity and target semantics.

## Deterministic Evidence

Per-case QC generated from the fixed thresholds in `configs/thresholds.yaml`.
It remains authoritative for hard failures.

## Calibrated Evidence

Per-case QC generated from calibrated thresholds tied to the run's summary and
task mode. Calibration adjusts only allowed domains and cannot override hard
failures.

## Evidence Comparison

The aligned difference between deterministic and calibrated evidence, including
recommendation, risk, score, and driving-domain changes.

## Hard Failure

A validation or deterministic condition that dominates routing and final
composition. Examples include invalid or unreadable data, critical active hard
domains, and required-target failures. Hard failures are not routed for override.

## Soft Domain

An active domain that is not declared hard for the selected task. Its warning,
uncertainty, risk, or calibration sensitivity may trigger deterministic review
routing.

## Nonbinding Reasoning

An evidence-grounded explanation of aligned artifacts. It cites evidence and
records conflicts and uncertainty but has no decision, action, routing, or
policy authority.

## Nonbinding Medical Critique

A grounding and internal-consistency audit of the reasoning artifact. It is
explicitly nonbinding, does not infer independent medical facts, and is not a
routing or policy input.

## Review Routing

Deterministic selection of uncertain non-hard-failure cases for human review.
Routing consumes validation and structured QC evidence, not reasoning, critique,
or LLM output.

## Policy Snapshot

An immutable policy resource copied into a run before final execution. The
current snapshot is `decision_policy.yaml`, copied from
`configs/policies/visible_pancreas_v1.yaml` for `pancreas_only`. It declares
first-match precedence, canonical actions, and a fail-closed no-match action.

## Canonical Action

The only vocabulary allowed in final decisions and golden expected actions:

- `keep`: suitable without a policy warning.
- `warning`: usable with a non-blocking curation concern.
- `review`: requires human adjudication.
- `reject`: unsuitable under deterministic policy.
- `insufficient_evidence`: required evidence cannot support another action.

Evidence-layer recommendations such as `exclude`, `omit_from_training`,
`keep_with_warning`, and `keep_with_metadata_warning` are internal inputs and are
normalized before final output.

## Final Decision

The source-of-truth case action produced by deterministic composition of
validation, deterministic/calibrated evidence, comparison, and routing. For
`pancreas_only`, it includes the exact policy ID, version, and matched-rule trace.

## Post-Hoc Evaluation

An audit of existing artifacts and final decisions. Evaluation may report
internal consistency and optional golden-action agreement, but it does not
change decisions and does not establish clinical or external validity.

## Golden Labels

Blinded human annotations of dataset-curation suitability. The expected action
uses the canonical vocabulary and is explicitly not a diagnosis or clinical
label.

## Interactive LLM Classifier

An optional intent classifier used only for otherwise unrecognized read-only
interactive queries. It has no access to hidden artifacts or policy authority;
deterministic parsing, command execution, and rendering remain authoritative.
