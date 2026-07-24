""" Sélection de la cohorte finale. 
Localise chaque tumeur (tête/corps/queue) par comparaison de centroïdes, 
puis tire aléatoirement autant de patients sains que de malades pour constituer 
une cohorte 50/50 parfaitement équilibrée. 
Exporte le CSV selected_cohort_N.csv avec toutes les colonnes nécessaires à ModuleStats."""

import os
import json
import random

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import center_of_mass

import Module0 as m0
import Module1 as m1

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

# Random seed for the healthy patient draw — fix it so the cohort is
# reproducible across runs. Change it if you want a different draw.
RANDOM_SEED = 42


# =============================================================================
# TUMOR LOCALIZATION
# =============================================================================

def locate_tumor_in_pancreas(label_data, tumor_id, pancreas_part_ids):
    """
    Determines whether the tumor sits in the head, body, or tail of the
    pancreas by comparing the tumor centroid to the centroid of each
    available pancreas part.

    pancreas_part_ids: dict mapping part names to label IDs, e.g.
        {"pancreas_head": 17, "pancreas_body": 18, "pancreas_tail": 19}
    Parts that are absent from the config or empty in the label volume are
    skipped with a warning — this should not happen if Module 0 ran cleanly,
    but we handle it gracefully just in case.

    Returns one of: 'head', 'body', 'tail', 'unknown', 'no_tumor'.
    """
    label_rounded = np.round(label_data).astype(np.int16)
    tumor_mask    = label_rounded == int(tumor_id)

    if not np.any(tumor_mask):
        return "no_tumor"

    tumor_center = center_of_mass(tumor_mask)
    distances    = {}

    for part_name, part_id in pancreas_part_ids.items():
        if part_id is None:
            print(f"    [WARNING] {part_name} missing from config — skipped.")
            continue
        part_mask = label_rounded == int(part_id)
        if not np.any(part_mask):
            print(f"    [WARNING] {part_name} empty in label volume — skipped.")
            continue
        distances[part_name] = np.linalg.norm(
            np.array(tumor_center) - np.array(center_of_mass(part_mask))
        )

    if not distances:
        print("    [WARNING] No pancreas parts available — location unknown.")
        return "unknown"

    closest = min(distances, key=distances.get)
    # Strip the "pancreas_" prefix so ModuleStats gets "head" / "body" / "tail"
    return closest.replace("pancreas_", "")


# =============================================================================
# 50/50 COHORT SELECTION
# =============================================================================

def select_healthy_patients(healthy_patients, n_needed, seed=RANDOM_SEED):
    """
    Draws n_needed healthy controls at random from the available pool.
    We do a simple random draw here — all QC has already been done in
    Module 1, so every patient in this pool is equally valid.
    If fewer healthy patients are available than sick ones, we take all of
    them and print a warning so you know the cohort is unbalanced.
    """
    random.seed(seed)
    if len(healthy_patients) >= n_needed:
        selected = random.sample(healthy_patients, n_needed)
    else:
        print(f"  [WARNING] Only {len(healthy_patients)} healthy patients available "
              f"for {n_needed} sick — cohort will be unbalanced.")
        selected = list(healthy_patients)
    return selected


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_cohort_summary(sick_patients, healthy_selected):
    all_p = sick_patients + healthy_selected
    print("\n" + "=" * 55)
    print("  COHORT SUMMARY")
    print("=" * 55)
    print(f"  Sick patients    : {len(sick_patients)}")
    print(f"  Healthy patients : {len(healthy_selected)}")
    print(f"  Total            : {len(all_p)}")
    print("-" * 55)

    print("  Tumor volume (cc) :")
    vols = [p["tumor_volume_cc"] for p in sick_patients if p.get("tumor_volume_cc")]
    if vols:
        print(f"    Mean   : {round(sum(vols)/len(vols), 2)} cc")
        print(f"    Median : {round(sorted(vols)[len(vols)//2], 2)} cc")
        print(f"    Min    : {round(min(vols), 2)} cc")
        print(f"    Max    : {round(max(vols), 2)} cc")

    print("  Tumor location :")
    locs = sorted(set(p.get("tumor_location", "unknown") for p in sick_patients))
    for loc in locs:
        n = sum(1 for p in sick_patients if p.get("tumor_location") == loc)
        print(f"    {loc:<10}: {n}")

    print("  Tumor attenuation :")
    atts = sorted(set(p.get("tumor_attenuation", "Unknown") for p in sick_patients))
    for att in atts:
        n = sum(1 for p in sick_patients if p.get("tumor_attenuation") == att)
        print(f"    {att:<10}: {n}")

    print("=" * 55 + "\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_cohort_selection(config_file, tumor_label_id, pancreas_label_ids,
                         pancreas_part_ids):
    """
    Full Module 2 pipeline:
      1. Load QC-validated patients from Module 1
      2. Localize tumor in head / body / tail for sick patients
      3. Draw an equal number of healthy controls at random
      4. Export the final balanced cohort as a CSV for ModuleStats

    pancreas_part_ids: dict with keys "pancreas_head", "pancreas_body",
        "pancreas_tail" mapping to their label IDs (subset of pancreas_label_ids).
    """
    # --- Step 1: run Module 1 QC ---
    accepted, all_exclusions = m1.run_integrity_and_qc(
        config_file, tumor_label_id, pancreas_label_ids
    )
    m1.export_exclusion_log(all_exclusions)

    sick_patients    = [p for p in accepted if p.get("has_tumor")]
    healthy_patients = [p for p in accepted if not p.get("has_tumor")]

    print(f"\n{len(sick_patients)} sick / {len(healthy_patients)} healthy after QC.")

    # --- Step 2: localize tumor for each sick patient ---
    print("\n--- TUMOR LOCALIZATION ---")
    for patient in sick_patients:
        label_data = nib.load(patient["label_path"]).get_fdata()
        loc = locate_tumor_in_pancreas(label_data, tumor_label_id, pancreas_part_ids)
        patient["tumor_location"] = loc
        print(f"  {patient['case_id']} — {loc} | "
              f"{patient.get('tumor_volume_cc')} cc | "
              f"{patient.get('tumor_attenuation')}")

    # --- Step 3: 50/50 draw ---
    healthy_selected = select_healthy_patients(healthy_patients, n_needed=len(sick_patients))
    print(f"\n{len(healthy_selected)} healthy patients selected (random draw, seed={RANDOM_SEED}).")

    # --- Step 4: build and export the cohort CSV ---
    # Flatten spacing tuple into separate columns so ModuleStats can read them
    rows = []
    for p in sick_patients + healthy_selected:
        spacing = p.get("spacing", (None, None, None))
        rows.append({
            "case_id"          : p["case_id"],
            "image_path"       : p["image_path"],
            "label_path"       : p["label_path"],
            "has_tumor"        : p.get("has_tumor", False),
            "tumor_location"   : p.get("tumor_location", "N/A"),
            "tumor_volume_cc"  : p.get("tumor_volume_cc"),
            "volume_pancreas_cc": _compute_pancreas_volume(p, pancreas_label_ids),
            "tumor_pancreas_ratio": p.get("ratio"),
            "tumor_attenuation": p.get("tumor_attenuation"),
            "mean_hu_tumor"    : p.get("mean_hu_tumor"),
            "mean_hu_pancreas" : p.get("mean_hu_pancreas"),
            "spacing_x"        : spacing[0],
            "spacing_y"        : spacing[1],
            "spacing_z"        : spacing[2],
        })

    df_cohort = pd.DataFrame(rows)
    csv_name  = f"selected_cohort_{len(df_cohort)}.csv"
    df_cohort.to_csv(csv_name, index=False)
    print(f"\nCohort saved: {csv_name}")

    print_cohort_summary(sick_patients, healthy_selected)
    print("=== MODULE 2 DONE — run ModuleStats.py for the full report ===")

    return sick_patients, healthy_selected


def _compute_pancreas_volume(patient, pancreas_label_ids):
    """
    Reads the label file and returns the total pancreas volume in cc
    (pancreas-only, tumor excluded) for ModuleStats.
    Kept as a helper here to avoid loading the file twice in the main loop.
    """
    try:
        label      = nib.load(patient["label_path"])
        label_data = np.round(label.get_fdata()).astype(np.int16)
        scan       = nib.load(patient["image_path"])
        spacing    = tuple(float(x) for x in scan.header.get_zooms())
        voxel_vol  = spacing[0] * spacing[1] * spacing[2] / 1000.0

        pancreas_mask = np.zeros_like(label_data, dtype=bool)
        for pid in pancreas_label_ids:
            pancreas_mask |= (label_data == int(pid))

        return round(float(np.sum(pancreas_mask)) * voxel_vol, 3)
    except Exception:
        return None


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    file, found_label = m0.find_config_file()
    if file is None:
        raise SystemExit("[FATAL] No valid config found.")

    found_tumors, found_pancreas = m0.separation(found_label)
    tumor_label_id     = int(list(found_tumors.values())[0])
    pancreas_label_ids = [int(v) for v in found_pancreas.values()]

    # Build the head/body/tail sub-dict for localization
    # We rely on the label names from Module 0 containing "head", "body", "tail"
    pancreas_part_ids = {
        name: int(num)
        for name, num in found_pancreas.items()
        if any(k in name.lower() for k in ["head", "body", "tail"])
    }

    print(f"\nTumor label ID     : {tumor_label_id}")
    print(f"Pancreas label IDs : {pancreas_label_ids}")
    print(f"Pancreas parts     : {pancreas_part_ids}")

    with open(file, "r") as f:
        cfg = json.load(f)
    if not cfg.get("patients"):
        m0.generate_patient_list(file)

    run_cohort_selection(file, tumor_label_id, pancreas_label_ids, pancreas_part_ids)