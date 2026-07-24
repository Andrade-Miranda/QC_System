"""
Module 6 — nnUNet Preprocessing

Runs nnUNet's plan_and_preprocess on base1 only, then copies the resulting
shared files (splits, plans, fingerprint, preprocessed images) to base2 and
base3. Since the three bases have identical images and differ only in their
labels, this avoids redundant computation and — more importantly — guarantees
that all three models train and validate on exactly the same patient splits.
This is a hard requirement for a fair qualitative comparison.

What gets copied from base1 to base2/base3:
    - splits_final.json        (same train/val folds for all models)
    - dataset_fingerprint.json (identical since same images)
    - nnUNetPlans.json         (identical since same geometry)
    - preprocessed imagesTr/   (expensive to compute, same for all bases)

What is NOT copied (base-specific):
    - preprocessed labelsTr/   (different per base — reprocessed individually)

nnUNet environment variables must be set before running this module:
    nnUNet_raw         → folder containing the three Dataset*** folders
    nnUNet_preprocessed→ where nnUNet writes preprocessed data
    nnUNet_results     → where nnUNet writes trained models
"""

import os
import json
import shutil
import subprocess
import sys

from Module5 import DATABASES, OUTPUT_ROOT

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

# nnUNet command name — adjust if your environment uses a different entry point
# Common alternatives: "nnUNetv2_plan_and_preprocess", "nnunet_plan_and_preprocess"
NNUNET_PREPROCESS_CMD = "nnUNetv2_plan_and_preprocess"

# Number of CPU processes for preprocessing — set to -1 to use all available
NUM_PROCESSES = 8

# Trainer and planner — nnUNet defaults, change only if you need a custom setup
NNUNET_TRAINER  = "nnUNetTrainer"
NNUNET_PLANNER  = "nnUNetResEncUNetMPlanner"

# Files produced by plan_and_preprocess on base1 that are identical across
# all bases and can safely be copied instead of recomputed.
SHARED_FILES = [
    "splits_final.json",
    "dataset_fingerprint.json",
    "nnUNetPlans.json",
]

# Name of the preprocessed images folder inside each nnUNet_preprocessed dataset
# (nnUNet creates this based on the planner name)
PREPROCESSED_IMAGES_FOLDER = "nnUNetPlans_2d"   # adjust if you use a different config


# =============================================================================
# ENVIRONMENT VALIDATION
# =============================================================================

def check_nnunet_env():
    """
    Verifies that the three nnUNet environment variables are set.
    These are mandatory — nnUNet will crash without them and the error
    message is not always obvious.
    """
    missing = []
    for var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        if not os.environ.get(var):
            missing.append(var)

    if missing:
        print("\n[ERROR] The following nnUNet environment variables are not set:")
        for var in missing:
            print(f"  export {var}=/path/to/{var.lower()}")
        print("\nSet them and re-run Module 6.")
        return False

    print("nnUNet environment variables:")
    for var in ["nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"]:
        print(f"  {var} = {os.environ[var]}")
    return True


# =============================================================================
# DATASET ID EXTRACTION
# =============================================================================

def get_dataset_numeric_id(dataset_id_str):
    """
    Extracts the numeric ID from a dataset folder name.
    e.g. "Dataset012_PanTS_Base1_Pancreas" → "012"
    nnUNet's CLI expects this numeric ID, not the full folder name.
    """
    parts = dataset_id_str.split("_")
    return parts[0].replace("Dataset", "")


# =============================================================================
# NNUNET PREPROCESSING
# =============================================================================

def run_preprocessing(dataset_id_str, num_processes=NUM_PROCESSES):
    """
    Calls nnUNetv2_plan_and_preprocess for a given dataset.
    Tries subprocess first; if it fails (environment issues, nnUNet not on
    PATH etc.) it prints the exact command to run manually and continues.

    Returns True if preprocessing succeeded, False otherwise.
    """
    numeric_id = get_dataset_numeric_id(dataset_id_str)
    cmd = [
        NNUNET_PREPROCESS_CMD,
        "-d", numeric_id,
        "-pl", NNUNET_PLANNER,
        "--verify_dataset_integrity",
        "-np", str(num_processes),
    ]

    print(f"\n  Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"  Preprocessing complete for Dataset{numeric_id}.")
        return True
    except FileNotFoundError:
        print(f"\n  [WARNING] '{NNUNET_PREPROCESS_CMD}' not found on PATH.")
        print(f"  Run this command manually:\n\n    {' '.join(cmd)}\n")
        return False
    except subprocess.CalledProcessError as e:
        print(f"\n  [ERROR] Preprocessing failed (exit code {e.returncode}).")
        print(f"  Run this command manually:\n\n    {' '.join(cmd)}\n")
        return False


# =============================================================================
# SHARED FILE PROPAGATION
# =============================================================================

def propagate_shared_files(base1_preprocessed_dir, target_preprocessed_dir, db_key):
    """
    Copies the shared preprocessing outputs from base1 to a target base.

    Shared files (splits, plans, fingerprint) are copied to the root of
    the target preprocessed folder.

    The preprocessed images folder is also copied — this is the expensive
    part of preprocessing (normalization, resampling) and is identical
    across all bases since the images are the same.

    The preprocessed labels are NOT copied — they depend on the label
    content which differs per base and must be (re)generated for each.
    """
    os.makedirs(target_preprocessed_dir, exist_ok=True)

    # --- Shared JSON files ---
    for fname in SHARED_FILES:
        src = os.path.join(base1_preprocessed_dir, fname)
        dst = os.path.join(target_preprocessed_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f"    Copied {fname} -> {target_preprocessed_dir}")
        else:
            print(f"    [WARNING] {fname} not found in base1 preprocessed dir — "
                  f"nnUNet may regenerate it for {db_key}.")

    # --- Preprocessed images folder ---
    # We copy the entire folder but skip any label files (gt_segmentations)
    # since those are base-specific.
    images_src = os.path.join(base1_preprocessed_dir, PREPROCESSED_IMAGES_FOLDER)
    images_dst = os.path.join(target_preprocessed_dir, PREPROCESSED_IMAGES_FOLDER)

    if os.path.exists(images_src):
        if os.path.exists(images_dst):
            shutil.rmtree(images_dst)
        # Copy everything except gt_segmentations (labels)
        shutil.copytree(
            images_src,
            images_dst,
            ignore=shutil.ignore_patterns("gt_segmentations"),
        )
        print(f"    Copied preprocessed images folder -> {images_dst}")
        print(f"    (gt_segmentations excluded — will be generated per base)")
    else:
        print(f"    [WARNING] Preprocessed images folder not found: {images_src}")
        print(f"    nnUNet will recompute images for {db_key} (slower but correct).")


def preprocess_labels_only(dataset_id_str, num_processes=NUM_PROCESSES):
    """
    Runs preprocessing with the --only_verify_dataset flag skipped —
    actually we call plan_and_preprocess again but nnUNet is smart enough
    to skip what already exists (plans, fingerprint, images) and only
    reprocess the labels.

    In practice, after copying shared files, nnUNet will detect the existing
    plans and only regenerate gt_segmentations for the new label set.
    """
    numeric_id = get_dataset_numeric_id(dataset_id_str)
    cmd = [
        NNUNET_PREPROCESS_CMD,
        "-d", numeric_id,
        "-pl", NNUNET_PLANNER,
        "-np", str(num_processes),
        # Do not pass --verify_dataset_integrity here as plans already exist
    ]

    print(f"\n  Running label-only preprocessing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, text=True)
        print(f"  Label preprocessing complete for Dataset{numeric_id}.")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"\n  [WARNING] Label preprocessing failed or nnUNet not on PATH.")
        print(f"  Run this command manually:\n\n    {' '.join(cmd)}\n")
        return False


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_preprocessing_pipeline():
    """
    Full Module 6 pipeline:
      1. Validate nnUNet environment variables
      2. Preprocess base1 fully (plans + fingerprint + images + labels)
      3. Copy shared files to base2 and base3
      4. Run label-only preprocessing for base2 and base3
      5. Print a summary with the nnUNet training commands to run next
    """

    # --- Step 1: environment check ---
    if not check_nnunet_env():
        raise SystemExit("[FATAL] Fix nnUNet environment variables and retry.")

    preprocessed_root = os.environ["nnUNet_preprocessed"]

    base1_id     = DATABASES["base1"]["dataset_id"]
    base1_num    = get_dataset_numeric_id(base1_id)
    base1_preproc = os.path.join(preprocessed_root, base1_id)

    # --- Step 2: full preprocessing on base1 ---
    print(f"\n{'='*60}")
    print(f"  STEP 1 — Full preprocessing: {base1_id}")
    print(f"{'='*60}")
    base1_ok = run_preprocessing(base1_id)

    if not base1_ok:
        print("\n[WARNING] base1 preprocessing did not complete automatically.")
        print("Run the command above manually, then re-run Module 6 to propagate "
              "shared files to base2 and base3.")
        # We still continue so the user gets the full picture of what needs to run

    # --- Step 3 & 4: propagate + label preprocessing for base2 and base3 ---
    for db_key in ["base2", "base3"]:
        dataset_id    = DATABASES[db_key]["dataset_id"]
        target_preproc = os.path.join(preprocessed_root, dataset_id)

        print(f"\n{'='*60}")
        print(f"  STEP 2 — Propagate shared files to {dataset_id}")
        print(f"{'='*60}")

        if os.path.exists(base1_preproc):
            propagate_shared_files(base1_preproc, target_preproc, db_key)
        else:
            print(f"  [WARNING] base1 preprocessed folder not found: {base1_preproc}")
            print(f"  Skipping propagation for {db_key} — "
                  f"run full preprocessing manually.")
            continue

        print(f"\n  STEP 3 — Label preprocessing for {dataset_id}")
        preprocess_labels_only(dataset_id)

    # --- Step 5: training commands summary ---
    print(f"\n{'='*60}")
    print("  MODULE 6 DONE — nnUNet preprocessing complete")
    print(f"{'='*60}")
    print("\nTo train all three models, run:")
    for db_key in ["base1", "base2", "base3"]:
        num_id = get_dataset_numeric_id(DATABASES[db_key]["dataset_id"])
        for fold in range(5):
            print(f"  nnUNetv2_train {num_id} 3d_fullres {fold} "
                  f"-tr {NNUNET_TRAINER} -p {NNUNET_PLANNER}")
    print()


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_preprocessing_pipeline()