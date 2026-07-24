# Legacy Workflows

Everything in this directory is historical, unsupported, and non-executable.
The files are retained only to preserve implementation history. They may contain
stale imports, path assumptions, artifact contracts, action names, and workflow
ordering, and must not be imported or exposed by active commands.

Historical inventory:

- `generate_nnunet_dataset.py`: retired split-driven nnUNet export workflow.
- `finalize_qc_run.py`: retired partial-run recomposition workflow.
- `compare_qc.py`: retired ad hoc report-to-report comparison workflow.

Use `scripts/run_qc.py` for the complete supported QC workflow. The orchestrator
already creates the deterministic/calibrated comparison artifact. Use
`scripts/evaluate_qc_run.py` only for post-hoc evaluation of a completed run.
