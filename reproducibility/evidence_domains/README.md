# Evidence domains

Script: `analyze_evidence_domains.py`

Input:

- `reproducibility/artifacts/evidence_domains/changed_vs_stable_evidence_domains.csv`

Command:

```bash
python reproducibility/evidence_domains/analyze_evidence_domains.py
```

The validated artifact reports case-level multi-label frequencies for stable (`n=507`) and task-changing (`n=493`) cases. Rows are not mutually exclusive and must not be summed to 100%. The public script verifies the validated aggregate artifact and exports JSON results; raw case-level medical artifacts are not distributed in this repository.
