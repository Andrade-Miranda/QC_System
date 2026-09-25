# Reproducibility

This folder contains public reproduction scripts for the current AgentQC paper and poster analyses. The scripts use validated derived artifacts under `reproducibility/artifacts/` and do not require raw medical images.

## Run everything

```bash
python reproducibility/run_all.py
```

Generated outputs are written to `reproducibility/outputs/` and are not tracked.

## Analyses

- `decision_stability/`: verifies 1,000-case action stability and pairwise task-profile transitions.
- `evidence_domains/`: verifies stable-vs-task-changing multi-label evidence-domain percentages.
- `representative_cases/`: exports validated representative evidence-to-policy traces.
- `reasoning_critic_audit/`: verifies nonbinding reasoning/critic audit counts.
- `figures/poster/`: regenerates poster result figures from the same validated values.

## Boundary

These scripts reproduce internal artifact analyses. They do not claim clinical correctness, expert agreement, clinical ground truth, downstream model improvement, per-case clinical utility, or causal effects.
