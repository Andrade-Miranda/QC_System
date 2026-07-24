"""
Module 3 — Bounding Box Computation

Reads the balanced cohort CSV produced by Module 2 and computes a tight
anatomical bounding box for each patient around a curated set of organs
(pancreas, kidneys, stomach, duodenum, liver, spleen, aorta + tumor).
The organ IDs are read from config.json via Module 0 — no hardcoded IDs.

The bbox coordinates are added as new columns to the cohort CSV and saved
as a new file ready for Module 5 (crop). No image is written here — this
module only computes coordinates.
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

# Margin added around the bounding box in mm on each side.
# 20 mm is a safe default — tight enough to reduce volume significantly,
# generous enough not to clip any organ boundary after resampling.
BBOX_MARGIN_MM = 20

# Names of the organs used to define the bounding box.
# These must match the label names in config.json exactly.
# We chose this specific set because it tightly wraps the pancreatic region
# without pulling the box too wide (e.g. excluding bowel loops or lungs).
# If you add "liver" make sure it doesn't push the box too high on your data.
BBOX_ORGAN_NAMES = [
    "pancreas",
    "pancreas_head",
    "pancreas_body",
    "pancreas_tail",
    "kidney_right",
    "kidney_left",
    "stomach",
    "duodenum",
    "liver",
    "spleen",
    "aorta",
]


# =============================================================================
# LABEL ID RESOLUTION
# =============================================================================

def resolve_bbox_organ_ids(config_file):
    """
    Reads all labels from config.json and returns a dict mapping organ name
    to label ID for every organ in BBOX_ORGAN_NAMES that is actually present
    in the config. Names that are missing are warned and skipped — this is
    fine if e.g. pancreas parts were not annotated for a given dataset.
    """
    with open(config_file, "r") as f:
        config_data = json.load(f)

    # config stores {id: name}, we need {name: id}
    name_to_id = {v.lower(): int(k)
                  for k, v in config_data["label_config"]["labels"].items()}

    resolved = {}
    for organ in BBOX_ORGAN_NAMES:
        if organ.lower() in name_to_id:
            resolved[organ] = name_to_id[organ.lower()]
        else:
            print(f"  [WARNING] '{organ}' not found in config.json — skipped for bbox.")

    print(f"\nOrgan IDs resolved for bbox ({len(resolved)}/{len(BBOX_ORGAN_NAMES)}):")
    for name, oid in resolved.items():
        print(f"  {name:<35}: {oid}")

    return resolved


# =============================================================================
# BOUNDING BOX COMPUTATION
# =============================================================================

def compute_bounding_box(label_data, spacing, organ_ids, tumor_id):
    """
    Builds a combined binary mask from all listed organ IDs plus the tumor,
    then computes the axis-aligned bounding box with a margin in mm.

    We use physical margin (mm) rather than voxels so the box is consistent
    across scanners with different resolutions.

    Returns a dict with x/y/z min/max, or None if no organ mask was found
    (which would mean the label volume is essentially empty).
    """
    shape    = label_data.shape
    combined = np.zeros(shape, dtype=bool)

    # Include all bbox organs that are actually present in this patient's label
    ids_to_use = list(organ_ids.values())
    if tumor_id is not None:
        ids_to_use.append(int(tumor_id))

    for oid in ids_to_use:
        combined |= (label_data == oid)

    coords = np.argwhere(combined)
    if len(coords) == 0:
        return None

    # Convert margin from mm to voxels along each axis
    m_x = int(np.ceil(BBOX_MARGIN_MM / spacing[0]))
    m_y = int(np.ceil(BBOX_MARGIN_MM / spacing[1]))
    m_z = int(np.ceil(BBOX_MARGIN_MM / spacing[2]))

    return {
        "bbox_x_min": max(0,          int(coords[:, 0].min()) - m_x),
        "bbox_x_max": min(shape[0]-1, int(coords[:, 0].max()) + m_x),
        "bbox_y_min": max(0,          int(coords[:, 1].min()) - m_y),
        "bbox_y_max": min(shape[1]-1, int(coords[:, 1].max()) + m_y),
        "bbox_z_min": max(0,          int(coords[:, 2].min()) - m_z),
        "bbox_z_max": min(shape[2]-1, int(coords[:, 2].max()) + m_z),
    }


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_bbox_computation(config_file, cohort_csv, organ_ids, tumor_label_id):
    """
    Loops over every patient in the cohort CSV, computes their bounding box,
    and appends the six bbox columns to the dataframe.
    Patients for whom bbox computation fails (empty label, missing file) get
    NaN in the bbox columns — Module 5 will skip them with a warning.
    """
    df = pd.read_csv(cohort_csv)
    print(f"\nLoaded cohort: {len(df)} patients from {cohort_csv}")

    bbox_records = []

    print("\n--- BOUNDING BOX COMPUTATION ---")
    for i, (_, row) in enumerate(df.iterrows()):
        case_id    = row["case_id"]
        label_path = row["label_path"]
        image_path = row["image_path"]

        if not os.path.exists(label_path) or not os.path.exists(image_path):
            print(f"  [WARNING] {case_id} — file missing, bbox set to NaN.")
            bbox_records.append({})
            continue

        try:
            scan       = nib.load(image_path)
            label      = nib.load(label_path)
            spacing    = tuple(float(x) for x in scan.header.get_zooms())
            # np.round to fix NIfTI float drift on label values
            label_data = np.round(label.get_fdata()).astype(np.int16)

            # Tumor ID is None for healthy patients — that's fine, we just
            # skip adding it to the mask and the box still covers the organs.
            patient_tumor_id = int(tumor_label_id) if row.get("has_tumor") else None

            bbox = compute_bounding_box(label_data, spacing, organ_ids, patient_tumor_id)

            if bbox is None:
                print(f"  [WARNING] {case_id} — no organ mask found, bbox set to NaN.")
                bbox_records.append({})
            else:
                bbox_records.append(bbox)
                if i % 20 == 0:
                    size = (
                        f"x={bbox['bbox_x_max']-bbox['bbox_x_min']} "
                        f"y={bbox['bbox_y_max']-bbox['bbox_y_min']} "
                        f"z={bbox['bbox_z_max']-bbox['bbox_z_min']} voxels"
                    )
                    status = "SICK   " if row.get("has_tumor") else "HEALTHY"
                    print(f"  [{status}] {case_id} — {size}")

        except Exception as e:
            print(f"  [ERROR] {case_id} — {e}")
            bbox_records.append({})

    # Merge bbox columns into the dataframe
    df_bbox = pd.DataFrame(bbox_records, index=df.index)
    df_out  = pd.concat([df, df_bbox], axis=1)

    # Save enriched CSV — Module 5 reads this file as input
    out_name = cohort_csv.replace("selected_cohort_", "bbox_cohort_")
    if out_name == cohort_csv:
        # Fallback if the filename doesn't match the expected pattern
        out_name = cohort_csv.replace(".csv", "_bbox.csv")
    df_out.to_csv(out_name, index=False)

    n_ok  = df_bbox["bbox_x_min"].notna().sum()
    n_fail = len(df) - n_ok
    print(f"\nBbox computed: {n_ok}/{len(df)} patients "
          f"({n_fail} failed — will be skipped in Module 5).")
    print(f"Enriched CSV saved: {out_name}")
    print("\n=== MODULE 3 DONE — run Module5.py to crop the images ===")

    return df_out, out_name


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    # --- Step 1: config + labels ---
    file, found_label = m0.find_config_file()
    if file is None:
        raise SystemExit("[FATAL] No valid config found.")

    found_tumors, _ = m0.separation(found_label)
    tumor_label_id  = int(list(found_tumors.values())[0])

    # --- Step 2: resolve organ IDs from config ---
    organ_ids = resolve_bbox_organ_ids(file)

    # --- Step 3: find the cohort CSV from Module 2 ---
    candidates = [f for f in os.listdir(".")
                  if f.startswith("selected_cohort_") and f.endswith(".csv")]
    if not candidates:
        raise SystemExit("[FATAL] No selected_cohort_*.csv found. Run Module 2 first.")
    cohort_csv = sorted(candidates)[-1]
    print(f"\nCohort CSV: {cohort_csv}")

    # --- Step 4: compute bboxes ---
    run_bbox_computation(file, cohort_csv, organ_ids, tumor_label_id)