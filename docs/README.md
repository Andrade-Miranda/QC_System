# QC System

The QC system is a task-aware, artifact-based framework for medical imaging
dataset reliability. Deterministic evidence, routing, and policy execution are
the source of truth. Explanatory artifacts and interactive tools cannot change
QC outcomes.

## Environment

Install the declared dependencies into the Python environment used for the
pipeline:

```bash
python -m pip install -r requirements.txt
```

The convenience launcher uses `python` by default. Set
`PYTHON=/path/to/python` to select another environment. Use one environment with
all packages from `requirements.txt`: admin interaction requires `jsonschema`,
while dataset summarization requires the listed imaging and tabular packages.

## Supported Workflow

The canonical entry point is `scripts/run_qc.py`. It executes:

```text
configuration validation
  -> validated run context
  -> dataset validation
  -> per-case summary
  -> run-scoped threshold calibration
  -> deterministic and calibrated QC evidence
  -> deterministic/calibrated evidence comparison
  -> nonbinding reasoning
  -> nonbinding medical critique
  -> deterministic review routing
  -> snapshotted deterministic policy decision (pancreas_only)
  -> post-hoc evaluation
```

For `pancreas_only`, the orchestrator copies
`configs/policies/visible_pancreas_v1.yaml` into the run as
`decision_policy.yaml` and executes that immutable snapshot. The policy is a
fail-closed, first-match ruleset. The lesion task modes currently use the
implemented deterministic V1 composition rather than a YAML policy snapshot.

Final decisions and golden labels use one canonical action vocabulary:

```text
keep | warning | review | reject | insufficient_evidence
```

Legacy evidence recommendations such as `exclude`, `omit_from_training`, or
`keep_with_metadata_warning` are normalized before becoming final actions.

## Run

From the `QC_System/` directory:

```bash
python scripts/run_qc.py --task-mode pancreas_only --overwrite
```

Run-scoped outputs are written to:

```text
outputs/DATASET/TASK_MODE/runs/RUN_ID/
```

See `USAGE.md` for supported commands, `ARTIFACTS.md` for artifact semantics,
and `OUTPUT_STRUCTURE.md` for the run layout.

## Human Review

The golden-review workflow builds a reproducible, system-action-blinded package,
imports completed curation labels, and optionally evaluates system actions
against those labels. Review labels are curation actions, not diagnoses or
clinical labels.

Interactive review has separate read-only surfaces:

```bash
python agents/interactive_review_agent.py --mode admin --run-dir "$RUN_DIR"
python agents/interactive_review_agent.py --mode reviewer --review-package "$REVIEWER_PACKAGE"
```

Admin startup validates available schemas, artifact alignment and hashes,
golden-package provenance, and the run-local policy snapshot. Admin interaction
currently supports `pancreas_only` runs only.
Reviewer mode accepts only a sanitized directory containing
`review_manifest.csv` and `REVIEW_PROTOCOL.md`.

The optional LLM only classifies queries that the deterministic parser does not
understand into a fixed read-only command schema. It does not receive artifact
contents or hidden administrative data, does not render prose, and has no QC,
routing, or policy authority. External OpenAI use requires
`--allow-external-provider`.

## Archive

Everything under `archive/` is historical, unsupported, and non-executable. It
is not an alternative command surface. Do not import archive code into the
active workflow or use archived command examples as current instructions.
