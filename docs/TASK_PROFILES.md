# Task Profiles

Task profiles live in `configs/task_profiles/` and are selected with
`--task-mode`.

V1 executable support is intentionally limited to exactly three CT segmentation
profiles. Extra YAML files do not become supported task families without an
approved implementation plan for their evidence, routing, and policy semantics.

| Task mode | Required segmentations | Hard domains | Calibratable domains |
| --- | --- | --- | --- |
| `pancreas_only` | pancreas | `geometry_integrity` | `fov_integrity`, `pancreas_context` |
| `pancreas_lesion` | pancreas, lesion | `geometry_integrity`, `lesion_localization` | `fov_integrity`, `pancreas_context`, `lesion_burden` |
| `pancreas_lesion_subregions` | pancreas, lesion, head, body, tail | `geometry_integrity`, `lesion_localization` | `fov_integrity`, `pancreas_context`, `lesion_burden` |

The executable uppercase contract includes:

```yaml
TASK_MODE: pancreas_lesion

REQUIRED_COMPONENTS:
  image: true
  pancreas_mask: true
  lesion_mask: true
  subregion_masks: false
  metadata: optional

REQUIRED_SEGMENTATIONS:
  pancreas_mask: pancreas.nii.gz
  lesion_mask: pancreatic_lesion.nii.gz

ACTIVE_QC_DOMAINS:
  geometry_integrity: true
  lesion_localization: true
  lesion_burden: true
  pancreas_context: true
  fov_integrity: true
  region_consistency: false
  attenuation_integrity: true
  metadata_completeness: true

HARD_FAILURE_DOMAINS:
  - geometry_integrity
  - lesion_localization

CALIBRATABLE_DOMAINS:
  - fov_integrity
  - pancreas_context
  - lesion_burden
```

Profiles determine validation requirements and which QC domains are active,
hard, or calibratable. They do not grant reasoning, critique, or LLM outputs any
decision authority.

## Confirmed Negative Lesion Cases

For lesion-capable segmentation profiles (`pancreas_lesion` and
`pancreas_lesion_subregions`), absent lesion supervision is allowed only when a
run supplies explicit confirmed-negative provenance. Unconfirmed absent or empty
lesion masks fail closed as `unconfirmed_negative_lesion`; they are not silently
converted into valid negative training samples.

Use the optional manifest argument in orchestrated runs:

```bash
python scripts/run_qc.py \
  --task-mode pancreas_lesion \
  --confirmed-negative-lesions path/to/confirmed_negative_lesions.json
```

Minimal manifest shape:

```json
{
  "confirmed_negative_lesions": [
    {
      "case_id": "case_001",
      "task_mode": "pancreas_lesion",
      "lesion_status": "confirmed_absent",
      "confirmed_by": "reviewer_or_source",
      "confirmation_date": "2026-07-24",
      "confirmation_source": "manual_review_or_dataset_metadata",
      "confirmation_scope": "Lesion absent for this segmentation task."
    }
  ]
}
```

The manifest is provenance for the dataset-curation task. It is not an external
medical-correctness claim unless the source is a locked human/golden label.

## Visible Pancreas Profile

`pancreas_only.yaml` also contains the additive lower-case `task_profile`
contract for visible-pancreas CT segmentation. It requires pancreas presence and
a non-empty mask, disallows negative cases, and permits partial FOV only when
geometry is valid, FOV is assessed, and all visible pancreas tissue is annotated.

Its executable final rules are in
`configs/policies/visible_pancreas_v1.yaml` (`visible_pancreas_v1`, version
`1.0.0`). Orchestrated `pancreas_only` runs copy that file to
`RUN_DIR/decision_policy.yaml` and execute the snapshot with strict first-match
precedence. The no-match action is `insufficient_evidence`; a CLI execution of
the final agent for this task requires an explicit `--policy` path.

The policy accepts only validation, deterministic evidence, calibrated evidence,
comparison, and routing roots. It cannot consume reasoning or critique.

Current first-match precedence:

| Rule | Condition | Action |
| --- | --- | --- |
| `VP-001` | blocking processing or validation failure | `reject` |
| `VP-002` | unreadable image | `reject` |
| `VP-003` | invalid or hard-failure geometry | `reject` |
| `VP-004` | required target absent | `reject` |
| `VP-005` | required mask missing or empty | `reject` |
| `VP-006` | visible-target annotation incomplete | `reject` |
| `VP-007` | required evidence missing | `insufficient_evidence` |
| `VP-008` | target presence uncertain | `insufficient_evidence` |
| `VP-009` | partial visible target with uncertain annotation | `review` |
| `VP-010` | explicit deterministic review route | `review` |
| `VP-011` | recommendation or risk instability | `review` |
| `VP-012` | allowed partial FOV with complete visible annotation | `warning` |
| `VP-013` | non-blocking evidence warning | `warning` |
| `VP-014` | valid visible-pancreas case | `keep` |

If none match, the policy returns `insufficient_evidence`; the final agent fails
closed if an evaluated case has no matched rule.

## Validation

Validate the selected profile and path configuration without running QC:

```bash
python scripts/validate_config.py --task-mode pancreas_only
```

New profiles require implementation support for their evidence and decision
semantics; adding a YAML file alone does not establish a validated task family.
