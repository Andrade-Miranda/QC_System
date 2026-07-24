# Archive

This directory contains retired implementations kept for historical reference
only. Archive code is unsupported, non-executable, and outside the active
command surface. Do not run it, import it from active code, or treat its paths,
schemas, actions, or examples as current behavior.

`legacy_workflows/` contains retired command-line workflows; see its README for
the historical inventory. Other files preserve earlier curation and LLM
experiments.

Current source of truth:

```text
ValidatedRunContextAgent -> ValidationAgent -> SummaryAgent -> CalibrationAgent
-> DeterministicQCAgent/CalibratedQCAgent -> QCComparisonAgent
-> ReasoningAgent/MedicalCriticAgent (nonbinding) -> ReviewRoutingAgent
-> FinalDecisionAgent -> EvaluationAgent (post-hoc)
```

The supported entry point is `scripts/run_qc.py`. Active documentation lives in
`docs/`.
