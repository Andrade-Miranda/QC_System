import os
import json
import glob
import shutil

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.ndimage import binary_dilation

# =============================================================================
# TUNABLE PARAMETERS
# Adjust these thresholds without touching the rest of the code.
# =============================================================================

# Minimum number of tumor voxels — below this the mask is considered too small
# to be a real annotation (likely a stray click or segmentation artefact).
MIN_TUMOR_VOXELS = 50

# Minimum number of voxels the tumor mask must share with the pancreas mask
# (after dilation) to be considered anatomically plausible.
MIN_PANCREAS_CONTACT_VOXELS = 10

# Maximum allowed ratio of tumor volume to total pancreas volume (tumor included).
# A ratio above this usually means the tumor label has leaked outside the pancreas.
# Start at 0.20 and raise to e.g. 0.30 if you find it's too aggressive.
MAX_TUMOR_PANCREAS_RATIO = 0.20

# Dilation radius (in voxels) applied to the tumor mask before checking contact
# with the pancreas. Needed because in PanTS, tumor and pancreas labels are
# adjacent but never overlap — without dilation we'd miss real contacts.
DILATION_RADIUS = 3

# =============================================================================
# KEYWORD LISTS — used to detect relevant labels in config.json
# =============================================================================

KEYWORDS_TUMORS = [
    "tumor", "tumour", "lesion", "mass", "nodule", "cyst",
    "carcinoma", "adenoma", "sarcoma", "lymphoma", "melan", "cancer"
]

KEYWORDS_PANCREAS = [
    "pancreas", "pancreatic", "pancreatique", "pancréas", "pancréatique"
]

# Subfolder names inside the main EXCLUDED directory.
# Keeping them as constants makes it easy to rename them later.
EXCL_ROOT           = "EXCLUDED"
EXCL_MISSING        = os.path.join(EXCL_ROOT, "MISSING_FILES")
EXCL_CORRUPTED      = os.path.join(EXCL_ROOT, "CORRUPTED")
EXCL_EMPTY_MASK     = os.path.join(EXCL_ROOT, "EMPTY_MASK")
EXCL_SMALL_MASK     = os.path.join(EXCL_ROOT, "SMALL_MASK")
EXCL_NO_CONTACT     = os.path.join(EXCL_ROOT, "NO_PANCREAS_CONTACT")
EXCL_HIGH_RATIO     = os.path.join(EXCL_ROOT, "HIGH_RATIO")


# =============================================================================
# HELPER — move a patient folder to a given exclusion subfolder
# =============================================================================

def _exclude_patient(base_dir, case_id, destination, reason):
    """
    Moves the patient folder to the appropriate exclusion subfolder and
    prints a clear message so it's easy to audit the log afterwards.
    """
    os.makedirs(destination, exist_ok=True)
    src = os.path.join(base_dir, case_id)
    dst = os.path.join(destination, case_id)
    if os.path.exists(src):
        shutil.move(src, dst)
    print(f"  [EXCLUDED] {case_id} -> {destination}  ({reason})")


# =============================================================================
# STEP 1 — Find config.json and extract label IDs
# =============================================================================

def find_config_file():
    """
    Looks for a config.json in the current directory first, then falls back
    to any *.json file that contains a tumor keyword.
    Returns (config_path, found_labels_dict) where found_labels_dict maps
    label names to their numeric IDs — or (None, {}) on failure.
    A valid config MUST contain at least one tumor label; without it the
    pipeline has no target to segment and cannot continue.
    """
    found_label = {}
    found_tumor = False

    def _parse_labels(data, filepath):
        nonlocal found_tumor
        labels = data.get("label_config", {}).get("labels", {})
        for num, name in labels.items():
            matched_tumor = False
            for kw in KEYWORDS_TUMORS:
                if kw in name.lower():
                    print(f"  -> Tumor label  : '{name}' (ID {num})  [{kw}]")
                    found_label[name] = num
                    found_tumor = True
                    matched_tumor = True
                    break
            if matched_tumor:
                continue
            for kw in KEYWORDS_PANCREAS:
                if kw in name.lower():
                    print(f"  -> Pancreas label: '{name}' (ID {num})  [{kw}]")
                    found_label[name] = num
                    break

    # --- primary location ---
    if os.path.exists("config.json"):
        print("Config file found: config.json")
        try:
            with open("config.json", "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"[ERROR] Cannot read config.json: {e}")
            return None, {}
        _parse_labels(data, "config.json")
        if not found_tumor:
            print(f"[ERROR] config.json found but no tumor label detected. "
                  f"Expected keywords: {KEYWORDS_TUMORS}")
            return None, {}
        return "config.json", found_label

    # --- fallback: scan all *.json files in the current directory ---
    print("config.json not found — scanning for alternative JSON files...")
    for filepath in glob.glob("*.json"):
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARNING] Could not read {filepath}: {e}")
            continue
        if any(kw in json.dumps(data).lower() for kw in KEYWORDS_TUMORS):
            print(f"Config file found: {filepath}")
            _parse_labels(data, filepath)
            if not found_tumor:
                print(f"[ERROR] {filepath} contains no tumor label.")
                return None, {}
            return filepath, found_label

    print("[ERROR] No valid config file found. The pipeline cannot continue.")
    return None, {}


# =============================================================================
# STEP 2 — Separate tumor labels from pancreas labels
# =============================================================================

def separation(found_label):
    """
    Splits the flat label dict into two dicts: one for tumor labels and one
    for pancreas labels (including head/body/tail parts if present).
    This separation is done once here so every downstream module can just
    grab the right dict without re-parsing the config.
    """
    found_tumors   = {}
    found_pancreas = {}
    for name, num in found_label.items():
        if any(kw in name.lower() for kw in KEYWORDS_TUMORS):
            found_tumors[name] = num
        elif any(kw in name.lower() for kw in KEYWORDS_PANCREAS):
            found_pancreas[name] = num
    return found_tumors, found_pancreas


# =============================================================================
# STEP 3 — Build the patient list and write it into config.json
# =============================================================================

def generate_patient_list(config_file):
    """
    Scans base_dir for folders that start with patient_prefix and registers
    each one as a patient entry in config.json.
    No metadata enrichment here — we stripped ct_phase and manufacturer
    because they turned out not to be needed for the pipeline.
    """
    with open(config_file, "r") as f:
        config_data = json.load(f)

    base_dir       = config_data.get("base_dir", "")
    patient_prefix = config_data.get("patient_prefix", "")

    patients = []
    for folder in sorted(os.listdir(base_dir)):
        if folder.startswith(patient_prefix) and os.path.isdir(os.path.join(base_dir, folder)):
            patients.append({
                "case_id": folder,
                "image"  : os.path.join(folder, config_data.get("image", "")),
                "label"  : os.path.join(folder, config_data.get("label", "")),
            })

    config_data["patients"] = patients
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)

    print(f"\n{len(patients)} patients found and written to config.json.")
    return patients


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    file, found_label = find_config_file()
    if file is None:
        raise SystemExit("[FATAL] No valid config found.")

    found_tumors, found_pancreas = separation(found_label)
    print(f"\nTumor labels   : {found_tumors}")
    print(f"Pancreas labels: {found_pancreas}")

    patients = generate_patient_list(file)
    print(f"\nFirst 3 patients:")
    for p in patients[:3]:
        print(f"  {p['case_id']}")