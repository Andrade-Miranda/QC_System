# Data setup

The public repository does not distribute raw medical images, masks, patient data, or private metadata. Use only datasets you are legally permitted to access.

## Expected structure

```text
DATASET_ROOT/
  imagesTr/
    CASE_ID_0000.nii.gz
  labelsTr/
    CASE_ID/
      segmentations/
        pancreas.nii.gz
        pancreatic_lesion.nii.gz
        pancreas_head.nii.gz
        pancreas_body.nii.gz
        pancreas_tail.nii.gz
  metadata.xlsx
```

Task profiles determine which segmentations are required for a run: `pancreas_only`, `pancreas_lesion`, or `pancreas_lesion_subregions`. Metadata is optional in the tracked task profiles, but missing metadata can still be recorded as QC evidence.

Use a gitignored local override or an environment variable:

```bash
cp configs/paths.local.example.yaml configs/paths.local.yaml
export AGENTQC_RAW_DATASET_ROOT="/path/to/DATASET_ROOT"
```

Do not commit local absolute paths or private data.
