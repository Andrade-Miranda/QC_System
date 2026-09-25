# Running AgentQC

## Validate configuration

```bash
python scripts/validate_config.py --task-mode pancreas_only
```

## Run one task profile

```bash
python scripts/run_qc.py --task-mode pancreas_only --run-id RUN_ID --overwrite
python scripts/run_qc.py --task-mode pancreas_lesion --run-id RUN_ID --overwrite
python scripts/run_qc.py --task-mode pancreas_lesion_subregions --run-id RUN_ID --overwrite
```

Each run writes artifacts under `outputs/DATASET/TASK_MODE/runs/RUN_ID/`.

## Post-hoc evaluation

```bash
RUN_DIR="outputs/DATASET/TASK_MODE/runs/RUN_ID"
python scripts/evaluate_qc_run.py --run-dir "$RUN_DIR"
```

## Nonbinding LLM explanation artifact

```bash
python scripts/explain_qc_run.py --run-dir "$RUN_DIR" --case PanTS_00007001 --provider ollama
```

This writes `llm_explanation_artifact.json` and does not affect routing or final actions.

## Paper/poster reproduction

```bash
python reproducibility/run_all.py
```

This uses validated derived artifacts included under `reproducibility/artifacts/` and does not require raw medical data.
