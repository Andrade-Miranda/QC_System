# Reproducibility

The reproducibility layer is under `reproducibility/`. It separates full pipeline reproduction, which requires raw data, from paper/poster analysis reproduction, which uses validated derived artifacts included in the repository.

Run all public checks:

```bash
python reproducibility/run_all.py
```

Outputs are written to `reproducibility/outputs/`, which is gitignored. Scripts fail loudly if validated numerical assertions do not match `reproducibility/expected_results.json`.
