# Confirmed Negative Lesion Evidence

Lesion-capable V1 segmentation profiles support confirmed negative cases only
with explicit provenance. This applies to:

- `pancreas_lesion`
- `pancreas_lesion_subregions`

It does not apply to `pancreas_only`, where lesion evidence is outside the task.
Supplying a confirmed-negative lesion manifest for a non-lesion task mode is
rejected fail-closed during validation.

## Contract

Absent or empty lesion supervision is accepted as a negative sample only when
the separate lesion mask proves absence:

1. Lesion status is determined exclusively from
   `labelsTr/CASE_ID/segmentations/pancreatic_lesion.nii.gz`.
2. A readable, geometry-valid nonempty separate lesion mask means lesion present.
3. A readable, geometry-valid empty separate lesion mask means confirmed absent.
4. A missing, unreadable, incomplete, or geometry-invalid separate lesion mask
   means insufficient evidence and fails closed.

`combined_labels.nii.gz` and other shared multi-label annotations have no role in
lesion presence, lesion absence, contradiction detection, review routing,
scoring, or final decisions. They cannot confirm lesion absence and cannot create
separate-versus-shared lesion contradiction evidence.

A manifest with matching `case_id`, active `task_mode`, and
`lesion_status: confirmed_absent` may still be recorded as provenance, but it
does not replace the required separate lesion-mask evidence in current runs.

Required fields per record:

- `case_id`
- `task_mode`
- `lesion_status`
- `confirmed_by`
- `confirmation_date`
- `confirmation_source`
- `confirmation_scope`

Unconfirmed absent lesion masks fail closed and are surfaced as deterministic QC
evidence. Separate-mask-derived absence and manifest provenance are dataset
curation provenance, not external medical-correctness claims unless they come
from locked human/golden labels.

## CLI

```bash
python scripts/run_qc.py \
  --task-mode pancreas_lesion \
  --confirmed-negative-lesions confirmed_negative_lesions.json
```

The orchestrator records the manifest path and SHA-256 in the validated run
context, dataset validation artifact, QC report metadata, and run manifest.
