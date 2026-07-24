import os
import json
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation

import Module0 as m0
from Module0 import (
    MIN_TUMOR_VOXELS,
    MIN_PANCREAS_CONTACT_VOXELS,
    MAX_TUMOR_PANCREAS_RATIO,
    DILATION_RADIUS,
    EXCL_MISSING,
    EXCL_CORRUPTED,
    EXCL_EMPTY_MASK,
    EXCL_SMALL_MASK,
    EXCL_NO_CONTACT,
    EXCL_HIGH_RATIO,
    _exclude_patient,
)


# =============================================================================
# STEP 1 — File integrity check
# =============================================================================

def check_patient_integrity(config_file):
    """
    Goes through every patient entry in config.json and verifies:
      - the image file exists on disk
      - the label file exists on disk
      - both files can be opened by nibabel without errors

    Patients that fail are moved to EXCLUDED/MISSING_FILES or
    EXCLUDED/CORRUPTED and removed from config.json so downstream
    modules never encounter broken data.

    Returns (valid_patients list, integrity_exclusions list of (id, reason)).
    """
    with open(config_file, "r") as f:
        config_data = json.load(f)

    base_dir = config_data.get("base_dir", "")
    patients = config_data.get("patients", [])

    valid_patients        = []
    integrity_exclusions  = []

    for patient in patients:
        case_id    = patient.get("case_id", "")
        image_path = os.path.join(base_dir, patient.get("image", ""))
        label_path = os.path.join(base_dir, patient.get("label", ""))

        # --- missing files ---
        if not os.path.exists(image_path) or not os.path.exists(label_path):
            missing = []
            if not os.path.exists(image_path): missing.append("image")
            if not os.path.exists(label_path): missing.append("label")
            _exclude_patient(base_dir, case_id, EXCL_MISSING,
                             f"missing {', '.join(missing)}")
            integrity_exclusions.append((case_id, "MISSING_FILES"))
            continue

        # --- corrupted files (nibabel can't open them) ---
        try:
            nib.load(image_path)
            nib.load(label_path)
        except Exception as e:
            _exclude_patient(base_dir, case_id, EXCL_CORRUPTED,
                             f"nibabel error: {e}")
            integrity_exclusions.append((case_id, "CORRUPTED"))
            continue

        valid_patients.append({
            "case_id"   : case_id,
            "image_path": image_path,
            "label_path": label_path,
        })

    # Remove excluded patients from config.json
    if integrity_exclusions:
        excl_ids = [e[0] for e in integrity_exclusions]
        config_data["patients"] = [
            p for p in config_data["patients"]
            if p.get("case_id") not in excl_ids
        ]
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"config.json updated — {len(integrity_exclusions)} patient(s) removed.")

    print(f"Integrity check done. Valid patients: {len(valid_patients)}")
    return valid_patients, integrity_exclusions


# =============================================================================
# STEP 2 — Tumor detection + quality checks
# =============================================================================

def analyze_patient(image_path, label_path, tumor_label_id, pancreas_label_ids):
    """
    Loads label and image volumes, then runs four QC checks on the tumor mask:

      1. Empty mask   — the tumor label ID exists in the volume, but the sum of
                        image intensities at those voxel positions is zero.
                        IMPORTANT: we check image intensity (HU values), NOT the
                        binary mask. The mask is by construction always "non-zero"
                        whenever the label ID is present — we need to look at
                        what's underneath in the actual image to detect truly
                        blank annotations.

      2. Too small    — fewer than MIN_TUMOR_VOXELS voxels. Almost certainly a
                        stray click rather than a real segmentation.

      3. No contact   — after dilating the tumor by DILATION_RADIUS voxels,
                        fewer than MIN_PANCREAS_CONTACT_VOXELS overlap with the
                        pancreas. Dilation is necessary because in PanTS adjacent
                        labels touch but never share the same voxel coordinate —
                        without it we'd incorrectly reject valid cases.

      4. High ratio   — tumor / (tumor + pancreas) volume > MAX_TUMOR_PANCREAS_RATIO.
                        We include the tumor in the denominator because annotators
                        sometimes replace pancreas voxels with the tumor label, so
                        a pancreas-only denominator would undercount organ volume.

    Returns a result dict. exclusion_reason is None if all checks pass.
    """
    scan       = nib.load(image_path)
    label      = nib.load(label_path)
    spacing    = tuple(float(x) for x in scan.header.get_zooms())

    # np.round corrects floating-point drift common in NIfTI files
    # (e.g. label value stored as 1.00019837 should just be 1)
    label_data = np.round(label.get_fdata()).astype(np.int16)
    image_data = scan.get_fdata()

    tumor_mask = label_data == int(tumor_label_id)
    has_tumor  = bool(np.any(tumor_mask))

    result = {
        "spacing"          : spacing,
        "has_tumor"        : has_tumor,
        "exclusion_reason" : None,
        "tumor_voxels"     : 0,
        "tumor_volume_cc"  : 0.0,
        "pancreas_voxels"  : 0,
        "ratio"            : None,
        "mean_hu_tumor"    : None,
        "mean_hu_pancreas" : None,
        "tumor_attenuation": None,
    }

    if not has_tumor:
        return result

    # --- 1. Empty mask — sum image intensities at tumor voxel positions ---
    intensity_sum = float(np.sum(image_data[tumor_mask]))
    if intensity_sum == 0.0:
        result["exclusion_reason"] = "EMPTY_MASK"
        return result

    # --- 2. Too small ---
    tumor_voxels = int(np.sum(tumor_mask))
    if tumor_voxels < MIN_TUMOR_VOXELS:
        result["exclusion_reason"] = "SMALL_MASK"
        result["tumor_voxels"]     = tumor_voxels
        return result

    # Build the pancreas mask (head + body + tail all merged — IDs from Module 0)
    pancreas_mask = np.zeros_like(label_data, dtype=bool)
    for pid in pancreas_label_ids:
        pancreas_mask |= (label_data == int(pid))

    # --- 3. Contact check with dilation ---
    tumor_dilated  = binary_dilation(tumor_mask, iterations=DILATION_RADIUS)
    contact_voxels = int(np.sum(tumor_dilated & pancreas_mask))
    if contact_voxels < MIN_PANCREAS_CONTACT_VOXELS:
        result["exclusion_reason"] = "NO_CONTACT"
        result["tumor_voxels"]     = tumor_voxels
        return result

    # --- 4. Ratio check ---
    voxel_vol_mm3       = spacing[0] * spacing[1] * spacing[2]
    tumor_vol_cc        = tumor_voxels * voxel_vol_mm3 / 1000.0
    pancreas_total_mask = pancreas_mask | tumor_mask
    pancreas_total_cc   = float(np.sum(pancreas_total_mask)) * voxel_vol_mm3 / 1000.0
    ratio               = round(tumor_vol_cc / pancreas_total_cc, 4) if pancreas_total_cc > 0 else 0.0

    # --- HU statistics — needed by ModuleStats for attenuation classification ---
    mean_hu_tumor    = round(float(np.mean(image_data[tumor_mask])), 2)
    mean_hu_pancreas = (round(float(np.mean(image_data[pancreas_mask])), 2)
                        if np.any(pancreas_mask) else None)

    # Classify tumor attenuation relative to surrounding healthy pancreas tissue
    tumor_attenuation = None
    if mean_hu_pancreas is not None:
        diff = mean_hu_tumor - mean_hu_pancreas
        if diff < -10:
            tumor_attenuation = "Hypo"
        elif diff > 10:
            tumor_attenuation = "Hyper"
        else:
            tumor_attenuation = "Iso"

    result.update({
        "tumor_voxels"     : tumor_voxels,
        "tumor_volume_cc"  : round(tumor_vol_cc, 3),
        "pancreas_voxels"  : int(np.sum(pancreas_mask)),
        "ratio"            : ratio,
        "mean_hu_tumor"    : mean_hu_tumor,
        "mean_hu_pancreas" : mean_hu_pancreas,
        "tumor_attenuation": tumor_attenuation,
    })

    if ratio > MAX_TUMOR_PANCREAS_RATIO:
        result["exclusion_reason"] = "HIGH_RATIO"

    return result


# =============================================================================
# STEP 3 — Full QC loop
# =============================================================================

def run_integrity_and_qc(config_file, tumor_label_id, pancreas_label_ids):
    """
    Runs integrity check then tumor QC in a single pass so each file is
    only loaded once. Returns (accepted, all_exclusions) where all_exclusions
    is a list of (case_id, reason) tuples covering both integrity failures
    and QC failures — ready to be exported as a CSV for auditing.
    """
    with open(config_file, "r") as f:
        config_data = json.load(f)
    base_dir = config_data.get("base_dir", "")

    valid_patients, integrity_exclusions = check_patient_integrity(config_file)

    accepted      = []
    qc_exclusions = []

    print("\n--- TUMOR QC ---")
    for i, patient in enumerate(valid_patients):
        case_id = patient["case_id"]
        res     = analyze_patient(
            patient["image_path"],
            patient["label_path"],
            tumor_label_id,
            pancreas_label_ids,
        )

        # Healthy controls pass QC automatically — no tumor mask to validate
        if not res["has_tumor"]:
            patient.update({
                "has_tumor"        : False,
                "spacing"          : res["spacing"],
                "tumor_volume_cc"  : None,
                "tumor_attenuation": None,
                "ratio"            : None,
                "mean_hu_tumor"    : None,
                "mean_hu_pancreas" : None,
            })
            accepted.append(patient)
            if i % 50 == 0:
                print(f"  [{i}/{len(valid_patients)}] {case_id} — healthy")
            continue

        reason = res["exclusion_reason"]

        if reason == "EMPTY_MASK":
            _exclude_patient(base_dir, case_id, EXCL_EMPTY_MASK,
                             "tumor label present but image intensity sums to zero")
            qc_exclusions.append((case_id, "EMPTY_MASK"))

        elif reason == "SMALL_MASK":
            _exclude_patient(base_dir, case_id, EXCL_SMALL_MASK,
                             f"only {res['tumor_voxels']} voxels "
                             f"(minimum: {MIN_TUMOR_VOXELS})")
            qc_exclusions.append((case_id, "SMALL_MASK"))

        elif reason == "NO_CONTACT":
            _exclude_patient(base_dir, case_id, EXCL_NO_CONTACT,
                             f"no pancreas contact within {DILATION_RADIUS}-voxel "
                             f"dilation (minimum: {MIN_PANCREAS_CONTACT_VOXELS} voxels)")
            qc_exclusions.append((case_id, "NO_CONTACT"))

        elif reason == "HIGH_RATIO":
            _exclude_patient(base_dir, case_id, EXCL_HIGH_RATIO,
                             f"ratio {res['ratio']:.1%} exceeds "
                             f"limit {MAX_TUMOR_PANCREAS_RATIO:.0%}")
            qc_exclusions.append((case_id, "HIGH_RATIO"))

        else:
            patient.update({
                "has_tumor"        : True,
                "spacing"          : res["spacing"],
                "tumor_voxels"     : res["tumor_voxels"],
                "tumor_volume_cc"  : res["tumor_volume_cc"],
                "ratio"            : res["ratio"],
                "mean_hu_tumor"    : res["mean_hu_tumor"],
                "mean_hu_pancreas" : res["mean_hu_pancreas"],
                "tumor_attenuation": res["tumor_attenuation"],
            })
            accepted.append(patient)
            print(f"  [OK] {case_id} — {res['tumor_volume_cc']} cc | "
                  f"ratio {res['ratio']:.1%} | {res['tumor_attenuation']}")

    # Remove QC-excluded patients from config.json
    all_exclusions = integrity_exclusions + qc_exclusions
    if qc_exclusions:
        with open(config_file, "r") as f:
            config_data = json.load(f)
        excl_ids = [e[0] for e in qc_exclusions]
        config_data["patients"] = [
            p for p in config_data["patients"]
            if p.get("case_id") not in excl_ids
        ]
        with open(config_file, "w") as f:
            json.dump(config_data, f, indent=2)
        print(f"\nconfig.json updated — {len(qc_exclusions)} patient(s) excluded by QC.")

    n_sick    = sum(1 for p in accepted if p.get("has_tumor"))
    n_healthy = sum(1 for p in accepted if not p.get("has_tumor"))
    print(f"\nQC complete — accepted: {len(accepted)} "
          f"({n_sick} sick / {n_healthy} healthy) | "
          f"excluded: {len(all_exclusions)}")

    return accepted, all_exclusions


# =============================================================================
# EXCLUSION LOG — CSV export for auditing / sending to supervisor
# =============================================================================

def export_exclusion_log(all_exclusions, output_path="exclusion_log.csv"):
    """
    Writes a CSV with one row per excluded patient.
    Columns: case_id, exclusion_reason
    Sorted by reason then by case_id so it's easy to scan.
    """
    if not all_exclusions:
        print("[INFO] No exclusions to export.")
        return

    df = pd.DataFrame(all_exclusions, columns=["case_id", "exclusion_reason"])
    df = df.sort_values(["exclusion_reason", "case_id"]).reset_index(drop=True)
    df.to_csv(output_path, index=False)
    print(f"\nExclusion log saved: {output_path} ({len(df)} patients)")

    for reason, group in df.groupby("exclusion_reason"):
        print(f"  {reason:<25}: {len(group)}")


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

    print(f"\nTumor label ID     : {tumor_label_id}")
    print(f"Pancreas label IDs : {pancreas_label_ids}")

    with open(file, "r") as f:
        cfg = json.load(f)
    if not cfg.get("patients"):
        m0.generate_patient_list(file)

    accepted, all_exclusions = run_integrity_and_qc(
        file, tumor_label_id, pancreas_label_ids
    )

    export_exclusion_log(all_exclusions)

    print(f"\nFirst 3 accepted patients:")
    for p in accepted[:3]:
        status = "SICK" if p.get("has_tumor") else "HEALTHY"
        print(f"  [{status}] {p['case_id']}")