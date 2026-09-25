# QC System Agents

This document explains the agent scripts in `QC_System/agents/` and how they fit into the deterministic, artifact-based QC workflow.

## AgentQC Project Memory

This directory contains the executable implementation. Local clone paths and
private project-memory locations are machine-specific and must not be encoded in
public instructions. Use the checked-out repository root as the working
directory for commands in this file.

Preserve the core rule: deterministic QC, routing, and policy remain the source of truth. Reasoning and critique explain artifacts only and cannot control routing or final decisions. V2 learned-evidence and world-model concepts live under `11_Ideas/V2/` and are outside active V1 implementation.

For implementation approval, use the actual local clone path in `allowed_paths`.
Do not treat another machine's local path as equivalent without explicit
approval.

## Overview

The QC system is organized as a multi-step pipeline. Each agent reads explicit inputs, writes JSON artifacts, and records provenance so downstream steps can trace which data, thresholds, task mode, and prior artifacts were used.

The main entry point is `scripts/run_qc.py`, which delegates to `agents/orchestrator_agent.py`. It runs the full workflow and stores all run-specific artifacts under a task-specific `runs/<run_id>/` directory.

## Pipeline

Typical full workflow:

1. `scripts/validate_config.py` checks path and task-mode configuration.
2. `agents/validated_run_context_agent.py` records the resolved, validated run context.
3. `agents/validation_agent.py` validates raw dataset resources and required segmentation files.
4. `scripts/summarize_dataset.py` builds the per-case dataset summary.
5. `scripts/calibrate_thresholds.py` optionally creates calibrated thresholds from the summary.
6. `agents/qc_agent.py` runs deterministic QC.
7. `agents/qc_agent.py` runs calibrated QC.
8. `agents/qc_comparison_agent.py` compares deterministic and calibrated QC outputs.
9. `agents/reasoning_agent.py` explains aligned evidence without decision authority.
10. `agents/medical_critic_agent.py` audits explanation grounding without decision authority.
11. `agents/review_routing_agent.py` deterministically routes uncertain cases to manual review.
12. `agents/final_decision_agent.py` executes deterministic final policy.
13. `agents/evaluation_agent.py` performs post-hoc internal audit.

## OrchestratorAgent

File: `agents/orchestrator_agent.py`

Purpose: Runs the complete artifact-based QC workflow.

Primary inputs:

- `--paths-yaml`: path configuration, usually `configs/paths.yaml`.
- `--task-mode`: one of the task profiles discovered from `configs/task_profiles/*.yaml`.
- `--score-version`: scoring mode used by `qc_agent.py`.
- `--workers`: parallel workers for summary generation.
- `--run-id` or `--run-dir`: optional explicit run location.

Primary outputs:

- `dataset_validation.json`
- `validated_run_context.json`
- `summary.json`
- `threshold_calibration.json` when calibration report exists
- `qc_report_deterministic.json`
- `qc_report_calibrated.json`
- `qc_comparison.json`
- `reasoning_artifact.json`
- `medical_critique.json`
- `review_routing.json`
- `final_qc_decisions.json`
- `eval_report.json`
- `execution_graph.json`
- `decision_policy.yaml` for `pancreas_only`, an immutable snapshot used by final policy execution

Notes:

- The orchestrator wraps raw JSON outputs as provenance-aware artifacts.
- It runs both deterministic and calibrated QC so their decisions can be compared.

## ValidationAgent

File: `agents/validation_agent.py`

Purpose: Validates raw dataset resources before expensive processing begins.

What it checks:

- Raw dataset root exists.
- Images directory exists.
- Labels directory exists.
- Each image case has the segmentation files required by the selected task profile.
- Label case directories without matching images are reported.

Primary inputs:

- Resolved paths from `configs/paths.yaml`.
- Task profile loaded from `configs/task_profiles/<task_mode>.yaml`.

Primary output:

- `dataset_validation.json`

Important behavior:

- Required segmentations come from the task profile key `REQUIRED_SEGMENTATIONS`.
- Missing required segmentations mark a case invalid and set `omit_from_training` for that validation artifact.

## QCAgent

File: `agents/qc_agent.py`

Purpose: Main deterministic QC engine for per-case quality assessment.

What it does:

- Reads a dataset summary JSON.
- Computes QC flags, evidence, component scores, risk levels, and recommendations.
- Builds domain-level QC profiles such as `geometry_integrity`, `lesion_localization`, `pancreas_context`, `fov_integrity`, `region_consistency`, `attenuation_integrity`, and `metadata_completeness`.
- Writes human-readable, CSV, and JSON reports.

Primary inputs:

- Summary JSON from `scripts/summarize_dataset.py`.
- Deterministic thresholds from `configs/thresholds.yaml` or an explicit run-scoped calibrated threshold file.
- Selected `task_mode`.

Primary outputs:

- `qc_report.json`
- `qc_report.csv`
- `qc_report.txt`

Important behavior:

- The current QC logic is task-aware but still contains lesion-centric behavior for `pancreas_lesion` style workflows.
- Negative lesion cases can be treated as valid training samples depending on task mode and flags.
- Critical geometry or lesion-localization failures can lead to exclusion.

## QCComparisonAgent

File: `agents/qc_comparison_agent.py`

Purpose: Compares deterministic QC results with calibrated QC results.

What it compares:

- Recommendation changes.
- Risk-level changes.
- Score deltas.
- Driving-domain changes.

Primary inputs:

- Deterministic QC artifact.
- Calibrated QC artifact.

Primary outputs:

- `qc_comparison.json`
- `qc_comparison.csv`
- `calibration_impact_report.txt`

Use this agent to understand how calibration changed case-level decisions or severity assignments.

## ReviewRoutingAgent

File: `agents/review_routing_agent.py`

Purpose: Selects uncertain non-hard-failure cases for optional human review.

Routing criteria include:

- Deterministic and calibrated QC disagreement.
- Medium risk cases.
- Moderate or high warnings in soft domains.
- Possible partial FOV or truncation issues.

Hard-failure domains:

- `geometry_integrity`
- `lesion_localization`

Soft domains:

- `fov_integrity`
- `pancreas_context`
- `lesion_burden`
- `attenuation_integrity`
- `region_consistency`

Primary inputs:

- Deterministic QC artifact.
- Calibrated QC artifact.
- QC comparison artifact.

Primary outputs:

- `review_routing.json`
- `manual_review_queue.csv`

Important behavior:

- Critical hard failures are not routed for review; they remain deterministic hard failures.
- Soft, uncertain, or calibration-sensitive cases are routed for review.

## FinalDecisionAgent

File: `agents/final_decision_agent.py`

Purpose: Executes final deterministic policy from dataset validation, deterministic/calibrated QC, evidence comparison, and review routing.

Decision priority:

1. For `pancreas_only`, `configs/policies/visible_pancreas_v1.yaml` is evaluated in declared first-match precedence order.
2. Dataset-validation and deterministic hard failures remain `reject`.
3. Missing required evidence becomes `insufficient_evidence`.
4. Partial visible-target annotation with uncertain completeness becomes `review`.
5. Remaining cases follow deterministic routing and calibrated evidence under the YAML policy.

Primary inputs:

- Deterministic QC artifact.
- Calibrated QC artifact.
- Review routing artifact.
- Dataset validation artifact.
- Evidence comparison artifact.
- Deterministic policy YAML for `pancreas_only`.

Primary outputs:

- `final_qc_decisions.json`
- `final_qc_decisions.csv`

Final decision fields include:

- `final_decision`
- `decision_basis`
- `hard_failure`
- `requires_human_review`
- `deterministic_recommendation`
- `calibrated_recommendation`
- `risk_level`
- `primary_domain`
- `policy_id`, `policy_version`, and the exact matched-rule trace.

## EvaluationAgent

File: `agents/evaluation_agent.py`

Purpose: Performs post-hoc internal audit without changing decisions.

What it reports:

- Number of final cases.
- Final decision counts.
- Decision-basis counts.
- Number of cases requiring human review.
- Number of hard failures.
- Calibration impact summary.
- Review-routing summary.
- Artifact integrity, evidence stability, reasoning/critique validity, routing behavior, policy safety, and task consistency.
- Optional golden-case validation when labels are supplied.
- Golden metrics include coverage, overall/per-action agreement, confusion counts, unsafe-keep rate, and review coverage.

Primary inputs:

- Final decisions artifact.
- Optional QC comparison artifact.
- Optional review routing artifact.
- Optional validated context, validation, evidence, reasoning, critique, and golden-label artifacts.

Primary output:

- `eval_report.json`

## Shared Utilities

Directory: `agents/utils/`

Important files:

- `paths.py`: central path resolution, task-mode discovery, output directory construction.
- `task_profiles.py`: task profile loading and helper accessors.
- `deterministic_policy.py`: fail-closed first-match YAML policy evaluator.

Task modes are discovered from YAML files in `configs/task_profiles/`. The currently supported profile schema uses uppercase keys such as:

- `TASK_MODE`
- `REQUIRED_COMPONENTS`
- `REQUIRED_SEGMENTATIONS`
- `ACTIVE_QC_DOMAINS`
- `HARD_FAILURE_DOMAINS`
- `CALIBRATABLE_DOMAINS`

## Task Profiles

Existing profile files:

- `configs/task_profiles/pancreas_only.yaml`
- `configs/task_profiles/pancreas_lesion.yaml`
- `configs/task_profiles/pancreas_lesion_subregions.yaml`

Profiles primarily control:

- Which segmentation files are required during validation.
- Which QC domains are active for the task.
- Which domains are hard failures.
- Which domains can be calibrated.

The lower-case `task_profile` identity is used by validated run context. The executable visible-pancreas decision rules live in `configs/policies/visible_pancreas_v1.yaml`; unrelated lower-case profile prose remains descriptive.

## Running The Full Workflow

Example:

```bash
python QC_System/agents/orchestrator_agent.py --task-mode pancreas_lesion_subregions --workers 4 --overwrite
```

From inside `QC_System/`, the equivalent is:

```bash
python agents/orchestrator_agent.py --task-mode pancreas_lesion_subregions --workers 4 --overwrite
```

Use `--skip-summary` to reuse an existing summary. `--skip-calibration` requires an explicit compatible `--calibrated-thresholds` file.

## Golden Review

Create a deterministic blinded 60-case package from a completed run:

```bash
python scripts/build_golden_review_package.py --run-dir <run_dir> --output-dir <run_dir>/golden_review_package
```

The builder separates administrative and reviewer files:

```text
golden_review_package/
  system_reference.csv
  reviewer_package/
    review_manifest.csv
    REVIEW_PROTOCOL.md
```

Distribute only `reviewer_package/`. OS-level copying, permissions, identity management, and access control remain operational responsibilities.

After a qualified reviewer completes `review_manifest.csv`, import and evaluate it:

```bash
python scripts/import_golden_review.py --review-package <golden_review_package/reviewer_package> --system-reference <golden_review_package/system_reference.csv> --output <golden_labels.json>
python scripts/evaluate_qc_run.py --run-dir <run_dir> --golden-labels <golden_labels.json>
```

Never populate reviewer labels from `system_reference.csv`; it must remain blinded until review is locked.

## Interactive Review

Admin mode requires a completed run:

```bash
python agents/interactive_review_agent.py --mode admin --run-dir <run_dir>
```

Reviewer mode accepts only the sanitized package and rejects `--run-dir`:

```bash
python agents/interactive_review_agent.py --mode reviewer --review-package <reviewer_package>
```

Deterministic parsing and rendering are authoritative. The configured LLM is used only to classify otherwise unknown queries into a fixed read-only command schema; its prose is never displayed and it never receives artifacts, paths, evidence, policy results, or hidden reference data. OpenAI requires explicit `--allow-external-provider`. Reviewer logging is off by default and can be enabled with `--log`.
