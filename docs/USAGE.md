# Usage

Commands below assume the current directory is `QC_System/`. Define completed
run and reviewer-package locations when using post-run tools:

```bash
RUN_DIR="outputs/DATASET/TASK_MODE/runs/RUN_ID"
REVIEW_PACKAGE="$RUN_DIR/golden_review_package/reviewer_package"
```

## Canonical Run

Run the complete supported workflow:

```bash
python scripts/run_qc.py \
  --task-mode pancreas_only \
  --run-id RUN_ID \
  --workers 8 \
  --overwrite
```

`run_qc.py` is the canonical workflow surface. It performs validated context,
dataset validation, summary, run-scoped calibration, both evidence passes,
comparison, nonbinding reasoning and critique, deterministic routing, final
policy execution, and post-hoc evaluation.

Use a custom path configuration when needed:

```bash
python scripts/run_qc.py \
  --paths-yaml configs/paths.yaml \
  --task-mode pancreas_lesion \
  --run-id RUN_ID
```

Tracked configs do not contain workstation-specific raw dataset paths. Configure
the local dataset root with either a gitignored override:

```bash
cp configs/paths.local.example.yaml configs/paths.local.yaml
# edit RAW_DATASET_ROOT in configs/paths.local.yaml
```

or environment variables, which have highest precedence:

```bash
export AGENTQC_RAW_DATASET_ROOT="/absolute/path/to/PantsMini"
```

If `RAW_DATASET_ROOT` or required subdirectories are missing, validation fails
closed before real-data runs. Do not commit `configs/paths.local.yaml` or raw
medical data.

Reuse is explicit and provenance-checked:

```bash
python scripts/run_qc.py \
  --task-mode pancreas_lesion \
  --run-id RUN_ID \
  --skip-summary \
  --skip-calibration \
  --calibrated-thresholds /path/to/compatible/thresholds.calibrated.yaml \
  --overwrite
```

`--skip-summary` requires the configured task summary to exist.
`--skip-calibration` requires an explicit `--calibrated-thresholds` file whose
metadata identifies the same summary and task mode. It is not a shortcut for
recomposing selected downstream stages of an old run.

Validate configuration only:

```bash
python scripts/validate_config.py --task-mode pancreas_only
```

## Post-Hoc Evaluation

Regenerate only `eval_report.json` for a completed run without changing final
decisions:

```bash
python scripts/evaluate_qc_run.py --run-dir "$RUN_DIR"
```

The helper discovers available run artifacts and submits them to
`EvaluationAgent`. Evaluation is an internal audit; it does not rerun comparison,
routing, or policy.

## Golden Review

Build a deterministic, blinded 60-case review package:

```bash
python scripts/build_golden_review_package.py \
  --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/golden_review_package"
```

Distribute only `$REVIEW_PACKAGE`. Keep
`$RUN_DIR/golden_review_package/system_reference.csv` administrative and blinded
until review is locked.

After a qualified reviewer completes `review_manifest.csv`, import it:

```bash
python scripts/import_golden_review.py \
  --review-package "$REVIEW_PACKAGE" \
  --system-reference "$RUN_DIR/golden_review_package/system_reference.csv" \
  --output "$RUN_DIR/golden_review_package/golden_labels.json"
```

Evaluate final actions against the imported curation labels:

```bash
python scripts/evaluate_qc_run.py \
  --run-dir "$RUN_DIR" \
  --golden-labels "$RUN_DIR/golden_review_package/golden_labels.json"
```

Golden `expected_action` values must be `keep`, `warning`, `review`, `reject`, or
`insufficient_evidence`. They assess dataset curation only and must not contain
medical diagnoses.

## Interactive Review

Admin mode reads and validates a completed `pancreas_only` run:

```bash
python agents/interactive_review_agent.py --mode admin --run-dir "$RUN_DIR"
```

Reviewer mode accepts only the sanitized reviewer package:

```bash
python agents/interactive_review_agent.py \
  --mode reviewer \
  --review-package "$REVIEW_PACKAGE"
```

Add `--query "help"` for a single non-interactive query. Deterministic parsing
and rendering remain authoritative. Reviewer logging is off by default; use
`--log` only under an approved operational policy. External OpenAI intent
classification requires `--allow-external-provider`.

## Launcher

`scripts/qc_commands.sh` exposes shortcuts for the same active surfaces:

```bash
bash scripts/qc_commands.sh qc --task-mode pancreas_only --run-id RUN_ID
bash scripts/qc_commands.sh evaluate --run-dir "$RUN_DIR"
bash scripts/qc_commands.sh golden-build --run-dir "$RUN_DIR" \
  --output-dir "$RUN_DIR/golden_review_package"
bash scripts/qc_commands.sh golden-import \
  --review-package "$REVIEW_PACKAGE" \
  --system-reference "$RUN_DIR/golden_review_package/system_reference.csv" \
  --output "$RUN_DIR/golden_review_package/golden_labels.json"
bash scripts/qc_commands.sh admin --run-dir "$RUN_DIR"
bash scripts/qc_commands.sh reviewer --review-package "$REVIEW_PACKAGE"
```

Archived scripts are historical and non-executable. They are not supported
alternatives to the commands above.
