# Optional LLM Explanations

The Evidence Abstraction Adapter (`A_tau`) is an opt-in explanatory layer for
completed AgentQC runs. It is separate from the interactive review intent
classifier and does not replace deterministic rendering.

Pipeline:

```text
authoritative task-specific artifacts
  -> Evidence Abstraction Adapter A_tau
  -> compact validated evidence object Z_tau
  -> Ollama/Devstral advisory explanation
  -> strict output-grounding validator C_tau
  -> validated explanation or deterministic fallback
```

The deterministic policy (`pi_tau`) remains the exclusive source of final
curation actions. LLM output is never consumed by evidence extraction, scoring,
review routing, thresholds, task profiles, or final policy.

## Supported tasks

The shared adapter supports all current task profiles:

- pancreas-segmentation task, `tau_P` (`pancreas_only`);
- pancreatic-lesion segmentation task, `tau_L` (`pancreas_lesion`);
- pancreatic-lesion subregion task, `tau_S` (`pancreas_lesion_subregions`).

Task-specific fields that do not apply are encoded as `not_applicable` rather
than being silently omitted. Missing applicable values remain `null` or
`unknown` according to their source semantics.

## Prompt minimization

The LLM receives only compact structured evidence. It does not receive complete
QC reports, raw `supported_claim_checks`, filesystem paths, image bytes, masks,
or full provenance hashes. Stable artifact filenames, artifact identifiers, and
JSON pointers are retained for traceability outside the prompt.

## Output validation

The validator rejects or quarantines output when JSON is malformed, identifiers
change, final actions are contradicted, unsupported numbers or routing reasons
are introduced, filesystem paths or terminal control sequences appear, medical
correctness is claimed, or deterministic policy authority is omitted.

Invalid output is preserved for audit and replaced by a deterministic fallback
explanation. Fallback does not alter final actions.

## Usage

```bash
python scripts/explain_qc_run.py \
  --run-dir outputs/PantsMini/pancreas_only/runs/RUN_ID \
  --case PanTS_00007001 \
  --provider ollama
```

The generated artifact is `llm_explanation_artifact.json` in the run directory.
Live Ollama availability is optional for tests; provider failure triggers
deterministic fallback.

## Visualization status

The adapter does not synthesize visualization failures. If no canonical
validated visualization artifact is present, `visualization_status` is `unknown`,
`visualization_error` is `null`, and a limitation is recorded.
