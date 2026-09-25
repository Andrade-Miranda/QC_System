# Outputs

AgentQC writes run-scoped artifacts under:

```text
outputs/DATASET/TASK_MODE/runs/RUN_ID/
```

Authoritative action-path artifacts include dataset validation, deterministic/calibrated QC evidence, QC comparison, review routing, final decisions, and run manifest. Advisory artifacts include reasoning, medical critique, and optional LLM explanations. Advisory artifacts are useful for audit and explanation but are not consumed by deterministic policy.

See `docs/OUTPUT_STRUCTURE.md` for the detailed layout.
