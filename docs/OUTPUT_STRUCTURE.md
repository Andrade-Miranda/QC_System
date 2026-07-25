# Output Structure

The source of truth for one execution is its run directory:

```text
outputs/DATASET/TASK_MODE/runs/RUN_ID/
```

Canonical run layout:

```text
RUN_DIR/
  validated_run_context.json
  dataset_validation.json
  summary.json
  thresholds.calibrated.yaml
  thresholds.calibrated.calibration_report.json
  thresholds.calibrated.calibration_report.txt
  threshold_calibration.json
  raw_qc_deterministic/
    qc_report.json
    qc_report.csv
    qc_report.txt
  raw_qc_calibrated/
    qc_report.json
    qc_report.csv
    qc_report.txt
  qc_report_deterministic.json
  qc_report_deterministic.csv
  qc_report_deterministic.txt
  qc_report_calibrated.json
  qc_report_calibrated.csv
  qc_report_calibrated.txt
  qc_comparison.json
  qc_comparison.csv
  calibration_impact_report.txt
  reasoning_artifact.json
  medical_critique.json
  review_routing.json
  manual_review_queue.csv
  decision_policy.yaml                 # pancreas_only only
  final_qc_decisions.json
  final_qc_decisions.csv
  eval_report.json
  execution_graph.json
  run_manifest.json
```

`threshold_calibration.json` is present when a calibration report is available.
With validated threshold reuse, the exact calibration-side files depend on the
selected reusable resource. The `raw_qc_*` directories are implementation
intermediates; downstream consumers should use the provenance-wrapped
run-scoped artifacts at the run root.

Calibration and both QC passes consume the run-local `summary.json`. A shared
task summary may seed a run or validate reusable calibration by content hash,
but downstream run provenance does not depend on that shared path.

Optional golden-review and interactive-admin outputs may be placed under the run:

```text
RUN_DIR/
  golden_review_package/
    system_reference.csv
    golden_labels.json                 # after completed-review import, if chosen
    reviewer_package/
      review_manifest.csv
      REVIEW_PROTOCOL.md
  logs/
    interactive_review_interactions.jsonl  # configurable admin log
```

Only `golden_review_package/reviewer_package/` is suitable for reviewer
distribution. `system_reference.csv`, run artifacts, policy traces, and system
actions are administrative data and must remain blinded during review.

Do not treat shared summary folders, ad hoc report paths, or archived workflow
layouts as the source of truth for a completed run.

## Paper-Trackable Manifest And Aliases

Each completed orchestrated run writes `run_manifest.json` at the run root. The
manifest records the run ID, dataset, task mode, run directory, artifact paths,
artifact hashes where available, selected input resources, and the deterministic
authority boundary. It explicitly records that reasoning, critique, and LLM
outputs are explanatory and nonbinding. The manifest does not include a stale
self-entry for `run_manifest.json`; aliases point to it after it has been
written.

The task directory also receives lightweight JSON pointer aliases:

```text
outputs/DATASET/TASK_MODE/latest_run.json
outputs/DATASET/TASK_MODE/latest_eval_report.json
outputs/DATASET/TASK_MODE/latest_final_qc_decisions.json
```

Aliases point to the newest run artifacts and include target hashes. They do not
copy large artifacts and do not delete, move, or rewrite older generated runs.
