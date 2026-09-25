# Decision stability

Script: `analyze_decision_stability.py`

Inputs:

- `reproducibility/artifacts/decision_stability/decision_stability_summary.json`
- `reproducibility/artifacts/decision_stability/decision_stability_case_transitions.csv`

Command:

```bash
python reproducibility/decision_stability/analyze_decision_stability.py
```

Expected validated results:

- 493/1000 cases changed final action under at least one task profile.
- 507/1000 cases were stable across all three profiles.
- Pairwise changed actions: `τP→τL` 350/1000, `τP→τS` 429/1000, `τL→τS` 226/1000.

The pairwise sets overlap and must not be summed as a global percentage.
