# Confirmed Negative Lesion Manifests

Lesion-capable V1 segmentation profiles support confirmed negative cases only
with explicit provenance. This applies to:

- `pancreas_lesion`
- `pancreas_lesion_subregions`

It does not apply to `pancreas_only`, where lesion evidence is outside the task.
Supplying a confirmed-negative lesion manifest for a non-lesion task mode is
rejected fail-closed during validation.

## Contract

Absent or empty lesion supervision is accepted as a negative sample only when a
manifest supplies a matching `case_id`, active `task_mode`, and
`lesion_status: confirmed_absent` record.

Required fields per record:

- `case_id`
- `task_mode`
- `lesion_status`
- `confirmed_by`
- `confirmation_date`
- `confirmation_source`
- `confirmation_scope`

Unconfirmed absent lesion masks fail closed and are surfaced as deterministic QC
evidence. The manifest is a provenance artifact, not a medical-correctness claim
unless it comes from locked human/golden labels.

## CLI

```bash
python scripts/run_qc.py \
  --task-mode pancreas_lesion \
  --confirmed-negative-lesions confirmed_negative_lesions.json
```

The orchestrator records the manifest path and SHA-256 in the validated run
context, dataset validation artifact, QC report metadata, and run manifest.
