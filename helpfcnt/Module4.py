"""
Module 4 — Image Cropping

Reads the bbox-enriched CSV produced by Module 3 and physically crops each
patient's image and label NIfTI to the bounding box region.

The affine matrix is updated so that the new volume's origin in world
coordinates (mm) stays consistent with the original scan — this is critical
for nnUNet and any tool that uses physical coordinates. Without this step,
the cropped file would still claim to start at the original scan origin
which would shift all anatomical coordinates.

Outputs go to a dedicated folder (default: cropped_dataset/) and the final
CSV is re-balanced 50/50 in case some crops failed.
"""

import os
import json

import numpy as np
import nibabel as nib
import pandas as pd

import Module0 as m0

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

# Output folder for cropped NIfTI files
OUTPUT_ROOT = "cropped_dataset"

# Random seed for the post-crop 50/50 re-balancing draw.
# If some patients fail during crop the cohort can become unbalanced —
# we re-draw here to keep it exactly 50/50.
RANDOM_SEED = 42


# =============================================================================
# SINGLE FILE CROP
# =============================================================================

def crop_nifti(input_path, output_path, bbox):
    """
    Crops a NIfTI volume to the given bounding box and updates the affine
    so that world coordinates remain correct in the cropped file.

    The key operation is:
        new_origin = old_affine @ [x_min, y_min, z_min, 1]
    This transforms the voxel origin of the crop back into mm-space using
    the original affine, then replaces the translation column of the affine.
    Without this, any downstream tool that reads RAS coordinates (nnUNet,
    ITK, 3D Slicer...) would get wrong anatomical positions.
    """
    img  = nib.load(input_path)
    data = img.get_fdata()

    # Slice the data — +1 on max because Python slicing is exclusive
    cropped = data[
        bbox["x_min"] : bbox["x_max"] + 1,
        bbox["y_min"] : bbox["y_max"] + 1,
        bbox["z_min"] : bbox["z_max"] + 1,
    ]

    # Recompute the origin in world space for the new top-left voxel
    old_affine   = img.affine.copy()
    origin_voxel = np.array([bbox["x_min"], bbox["y_min"], bbox["z_min"], 1.0])
    new_origin   = old_affine @ origin_voxel

    new_affine       = old_affine.copy()
    new_affine[:, 3] = new_origin   # replace translation column only

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    nib.save(nib.Nifti1Image(cropped, new_affine, img.header), output_path)


# =============================================================================
# SINGLE PATIENT CROP
# =============================================================================

def crop_patient(row, output_root):
    """
    Crops both image and label for one patient using the bbox stored in the
    row. Returns a dict with the output paths, or None if the bbox is missing
    or any file is absent.
    """
    case_id  = row["case_id"]
    bbox_cols = ["bbox_x_min", "bbox_x_max", "bbox_y_min",
                 "bbox_y_max", "bbox_z_min", "bbox_z_max"]

    if any(pd.isna(row.get(c)) for c in bbox_cols):
        print(f"  [WARNING] {case_id} — bbox missing, skipped.")
        return None

    bbox = {c: int(row[c]) for c in bbox_cols}
    # Strip the "bbox_" prefix to get the keys crop_nifti expects
    bbox = {k.replace("bbox_", ""): v for k, v in bbox.items()}

    patient_dir = os.path.join(output_root, case_id)

    image_out = os.path.join(patient_dir, "ct_cropped.nii.gz")
    label_out = os.path.join(patient_dir, "label_cropped.nii.gz")

    crop_nifti(row["image_path"], image_out, bbox)
    crop_nifti(row["label_path"], label_out, bbox)

    return {
        "case_id"           : case_id,
        "image_cropped_path": image_out,
        "label_cropped_path": label_out,
    }


# =============================================================================
# CROP METADATA — useful for traceability and potential uncrop later
# =============================================================================

def save_crop_metadata(df, output_root):
    """
    Saves a JSON file with the original shape and bbox for every patient.
    Useful if you ever need to map predictions back to the original space,
    or just to keep a trace of what was cropped and where.
    """
    metadata  = {}
    bbox_cols = ["bbox_x_min", "bbox_x_max", "bbox_y_min",
                 "bbox_y_max", "bbox_z_min", "bbox_z_max"]

    for _, row in df.iterrows():
        case_id = row["case_id"]
        if any(pd.isna(row.get(c)) for c in bbox_cols):
            continue
        try:
            original_shape = list(nib.load(row["image_path"]).shape)
        except Exception:
            original_shape = None

        metadata[case_id] = {
            "original_shape": original_shape,
            "bbox": {c.replace("bbox_", ""): int(row[c]) for c in bbox_cols},
            "image_path": row["image_path"],
            "label_path": row["label_path"],
        }

    out_path = os.path.join(output_root, "crop_metadata.json")
    with open(out_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\nCrop metadata saved: {out_path}")
    return out_path


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_crop(cohort_csv, output_root=OUTPUT_ROOT):
    """
    Full Module 4 pipeline:
      1. Load bbox-enriched CSV from Module 3
      2. Crop every patient (image + label)
      3. Save crop metadata JSON for traceability
      4. Re-balance 50/50 after crop (some patients may have failed)
      5. Export final cropped cohort CSV for Module 5 (nnUNet formatting)
    """
    df = pd.read_csv(cohort_csv)
    print(f"Loaded: {len(df)} patients from {cohort_csv}")

    os.makedirs(output_root, exist_ok=True)
    save_crop_metadata(df, output_root)

    results = []
    print("\n--- CROPPING PATIENTS ---")
    for i, (_, row) in enumerate(df.iterrows()):
        try:
            result = crop_patient(row, output_root)
        except Exception as e:
            print(f"  [ERROR] {row['case_id']} — {e}")
            result = None

        if result is not None:
            status = "SICK   " if row.get("has_tumor") else "HEALTHY"
            if i % 20 == 0:
                print(f"  [{status}] {row['case_id']} cropped.")
        results.append(result)

    # Merge crop paths back into the main dataframe
    valid_results = [r for r in results if r is not None]
    df_cropped    = pd.DataFrame(valid_results)
    df_out = df.merge(
        df_cropped[["case_id", "image_cropped_path", "label_cropped_path"]],
        on="case_id",
        how="inner",   # drop patients that failed crop entirely
    )

    # Post-crop 50/50 re-balancing — a few crops can fail on either side,
    # so we re-draw to guarantee the final cohort is exactly balanced.
    is_sick    = df_out["has_tumor"] == True
    sick_df    = df_out[is_sick]
    healthy_df = df_out[~is_sick]
    min_count  = min(len(sick_df), len(healthy_df))

    print(f"\n--- POST-CROP BALANCING ---")
    print(f"  Sick successfully cropped    : {len(sick_df)}")
    print(f"  Healthy successfully cropped : {len(healthy_df)}")

    if min_count > 0:
        sick_final    = sick_df.sample(n=min_count,    random_state=RANDOM_SEED)
        healthy_final = healthy_df.sample(n=min_count, random_state=RANDOM_SEED)
        df_final = (pd.concat([sick_final, healthy_final])
                    .sample(frac=1, random_state=RANDOM_SEED)
                    .reset_index(drop=True))
        print(f"  Final balanced cohort: {min_count} sick + {min_count} healthy "
              f"= {len(df_final)} patients.")
    else:
        print("  [ERROR] One class is empty after crop — check your data.")
        df_final = df_out

    out_csv = os.path.join(output_root, "cropped_cohort.csv")
    df_final.to_csv(out_csv, index=False)
    print(f"\nFinal cropped cohort saved: {out_csv}")
    print("=== MODULE 4 DONE — run Module5.py to build nnUNet datasets ===")

    return df_final, out_csv


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # Find the bbox-enriched CSV from Module 3
    candidates = [f for f in os.listdir(".")
                  if f.startswith("bbox_cohort_") and f.endswith(".csv")]
    if not candidates:
        raise SystemExit("[FATAL] No bbox_cohort_*.csv found. Run Module 3 first.")

    cohort_csv = sorted(candidates)[-1]
    print(f"Bbox cohort CSV: {cohort_csv}")

    run_crop(cohort_csv)