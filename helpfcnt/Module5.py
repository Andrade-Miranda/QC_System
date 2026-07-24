"""
Module 5 — nnUNet Dataset Generation

Reads the cropped cohort CSV produced by Module 4 and assembles it into
the nnUNet raw data format. Three dataset variants are created, each with
a different set of organs as context — this is the basis for the qualitative
comparison between models.

Label remapping is consistent across all bases:
    0 = background
    1 = pancreas  (all parts merged)
    2 = lesion    (tumor)
    3 = other organs (only in base2 and base3, absent in base1)

Images are identical across all three bases — only the labels differ.
This is intentional: Module 6 will preprocess base1 and copy the result
to base2 and base3 to ensure a perfectly fair comparison.

Organ IDs are resolved from config.json — no hardcoded numeric IDs.
A case_id_mapping.json and metadata_sync CSV are produced per dataset
so nnUNet predictions can be traced back to clinical IDs during evaluation.
"""

import os
import json
import shutil

import numpy as np
import nibabel as nib
import pandas as pd

import Module0 as m0

# =============================================================================
# TUNABLE PARAMETERS
# =============================================================================

# Root output directory — nnUNet expects its raw data here
OUTPUT_ROOT = "nnUNet_raw"

# Adjacent organs for base3
ORGANS_LIMITROPHES = [
    "duodenum",
    "stomach",
    "spleen",
    "kidney_left",
    "kidney_right",
]

# Full abdominal organs for base2
ORGANS_ABDOMINAUX = [
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "duodenum",
    "aorta",
    "portal_vein_and_splenic_vein",
    "inferior_vena_cava",
]

# Three dataset variants — base4 (duodenum only) was removed because it
# added no meaningful difference over base3 in preliminary experiments.
DATABASES = {
    "base1": {
        "dataset_id" : "Dataset012_PanTS_Base1_Pancreas",
        "description": "Pancreas + Lesion only — baseline model",
        "organs"     : [],
    },
    "base2": {
        "dataset_id" : "Dataset013_PanTS_Base2_Abdomen",
        "description": "Pancreas + Lesion + Full abdominal organs",
        "organs"     : ORGANS_ABDOMINAUX,
    },
    "base3": {
        "dataset_id" : "Dataset014_PanTS_Base3_Limitrophes",
        "description": "Pancreas + Lesion + Adjacent organs",
        "organs"     : ORGANS_LIMITROPHES,
    },
}


# =============================================================================
# LABEL REMAPPING
# =============================================================================

def build_label_remap(config_file, found_pancreas, tumor_label_id, organs_to_include):
    """
    Builds a dict mapping original label IDs → nnUNet label IDs:
        all pancreas parts → 1
        tumor              → 2
        requested organs   → 3  (empty for base1)

    Organ IDs are resolved by name from config.json so this works regardless
    of the numeric IDs used in the annotation tool.
    """
    with open(config_file, "r") as f:
        config_data = json.load(f)

    name_to_id = {v: int(k) for k, v in config_data["label_config"]["labels"].items()}

    remap = {}

    # All pancreas parts (head, body, tail, whole) → label 1
    for name, orig_id in found_pancreas.items():
        remap[int(orig_id)] = 1

    # Tumor → label 2
    remap[int(tumor_label_id)] = 2

    # Extra organs → label 3
    for organ_name in organs_to_include:
        if organ_name in name_to_id:
            remap[name_to_id[organ_name]] = 3
        else:
            print(f"    [WARNING] '{organ_name}' not found in config.json — skipped.")

    return remap


def remap_label_array(label_data, remap):
    """
    Applies the remap dict to a label array. Everything not in the remap
    becomes 0 (background). np.round fixes float drift in NIfTI label files.
    """
    label_rounded = np.round(label_data).astype(np.int16)
    remapped      = np.zeros_like(label_rounded, dtype=np.uint8)
    for orig_id, new_id in remap.items():
        remapped[label_rounded == orig_id] = new_id
    return remapped


# =============================================================================
# DATASET.JSON
# =============================================================================

def generate_dataset_json(output_dir, db_config, n_training, labels_dict):
    """
    Writes the dataset.json that nnUNet reads to understand the dataset
    structure. Must be present at the root of each dataset folder.
    """
    dataset = {
        "channel_names": {"0": "CT"},
        "labels"       : labels_dict,
        "numTraining"  : n_training,
        "file_ending"  : ".nii.gz",
        "name"         : db_config["dataset_id"],
        "description"  : db_config["description"],
        "reference"    : "PanTS Dataset — PINKCC Challenge",
        "licence"      : "Research only",
    }
    out_path = os.path.join(output_dir, "dataset.json")
    with open(out_path, "w") as f:
        json.dump(dataset, f, indent=2)
    print(f"    dataset.json written: {out_path}")


# =============================================================================
# SINGLE DATASET CREATION
# =============================================================================

def create_database(db_key, db_config, df_cohort, config_file,
                    found_pancreas, tumor_label_id, output_root):
    """
    Creates one nnUNet dataset variant:
      - copies image files as PANTS_NNN_0000.nii.gz (cropped if available)
      - remaps and saves label files as PANTS_NNN.nii.gz
      - writes case_id_mapping.json  (nnUNet ID ↔ clinical ID)
      - writes dataset.json
      - writes metadata_sync_{db_key}.csv for eval.py

    Images are identical across all bases — only labels differ.
    Module 6 will exploit this to avoid redundant preprocessing.
    """
    organs         = db_config["organs"]
    dataset_folder = os.path.join(output_root, db_config["dataset_id"])
    images_dir     = os.path.join(dataset_folder, "imagesTr")
    labels_dir     = os.path.join(dataset_folder, "labelsTr")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    print(f"\n  Building {db_config['dataset_id']}...")
    print(f"  Extra organs (label 3): {organs if organs else 'none — base1 only has pancreas + lesion'}")

    remap       = build_label_remap(config_file, found_pancreas, tumor_label_id, organs)
    labels_dict = {"background": 0, "pancreas": 1, "lesion": 2}
    if organs:
        labels_dict["other_organs"] = 3

    case_id_mapping = {}
    n_skipped       = 0

    for idx, (_, row) in enumerate(df_cohort.iterrows(), start=1):
        case_id = row["case_id"]
        prefix  = f"PANTS_{idx:03d}"

        # Prefer cropped files, fall back to originals if Module 4 was skipped
        image_src = row.get("image_cropped_path", None)
        label_src = row.get("label_cropped_path", None)

        if pd.isna(image_src) or not image_src:
            image_src = row["image_path"]
        if pd.isna(label_src) or not label_src:
            label_src = row["label_path"]

        if not os.path.exists(str(image_src)) or not os.path.exists(str(label_src)):
            print(f"    [WARNING] {case_id} — file missing, skipped.")
            n_skipped += 1
            continue

        # Copy image — no modification needed across bases
        image_dst = os.path.join(images_dir, f"{prefix}_0000.nii.gz")
        shutil.copy2(image_src, image_dst)

        # Remap and save label — this is what differs between bases
        label_img  = nib.load(label_src)
        remapped   = remap_label_array(label_img.get_fdata(), remap)
        label_dst  = os.path.join(labels_dir, f"{prefix}.nii.gz")
        nib.save(nib.Nifti1Image(remapped, label_img.affine, label_img.header), label_dst)

        case_id_mapping[prefix] = case_id

    # Save clinical ↔ nnUNet ID mapping
    mapping_path = os.path.join(dataset_folder, "case_id_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump(case_id_mapping, f, indent=2)

    n_training = len(case_id_mapping)
    generate_dataset_json(dataset_folder, db_config, n_training, labels_dict)

    # Metadata sync CSV — lets eval.py join predictions back to clinical info
    reverse_mapping          = {v: k for k, v in case_id_mapping.items()}
    df_sync                  = df_cohort.copy()
    df_sync["case_id_clean"] = df_sync["case_id"].astype(str).str.strip()
    df_sync["nnunet_id"]     = df_sync["case_id_clean"].map(reverse_mapping)
    df_sync                  = df_sync.dropna(subset=["nnunet_id"])
    df_sync["case_id"]       = df_sync["nnunet_id"]

    csv_out = os.path.join(dataset_folder, f"metadata_sync_{db_key}.csv")
    df_sync.to_csv(csv_out, index=False)
    print(f"    metadata_sync saved: {csv_out}")
    print(f"    -> {n_training} patients written, {n_skipped} skipped.")

    return n_training


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

def print_summary(results):
    print("\n" + "=" * 60)
    print("  MODULE 5 SUMMARY — NNUNET DATASETS")
    print("=" * 60)
    for db_key, n in results.items():
        print(f"  {DATABASES[db_key]['dataset_id']}")
        print(f"    -> {n} patients")
    print("=" * 60)
    print("\nGenerated structure:")
    print(f"  {OUTPUT_ROOT}/")
    for db_key in results:
        did = DATABASES[db_key]["dataset_id"]
        print(f"    {did}/")
        print(f"      ├── dataset.json")
        print(f"      ├── imagesTr/    PANTS_001_0000.nii.gz ...")
        print(f"      ├── labelsTr/    PANTS_001.nii.gz ...")
        print(f"      ├── case_id_mapping.json")
        print(f"      └── metadata_sync_{db_key}.csv")
    print("\n=== MODULE 5 DONE — run Module6.py to preprocess ===")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    file, found_label = m0.find_config_file()
    if file is None:
        raise SystemExit("[FATAL] No valid config found.")

    found_tumors, found_pancreas = m0.separation(found_label)
    tumor_label_id = int(list(found_tumors.values())[0])

    print(f"\nTumor label ID  : {tumor_label_id}")
    print(f"Pancreas labels : {found_pancreas}")

    cohort_csv = os.path.join("cropped_dataset", "cropped_cohort.csv")
    if not os.path.exists(cohort_csv):
        candidates = [f for f in os.listdir(".")
                      if f.startswith("selected_cohort_") and f.endswith(".csv")]
        if not candidates:
            raise SystemExit("[FATAL] No cohort CSV found. Run Module 2 or 4 first.")
        cohort_csv = sorted(candidates)[-1]
        print(f"[INFO] Using uncropped cohort: {cohort_csv}")
    else:
        print(f"Cropped cohort: {cohort_csv}")

    df_cohort = pd.read_csv(cohort_csv)
    print(f"{len(df_cohort)} patients in cohort.")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    results = {}
    for db_key in ["base1", "base2", "base3"]:
        results[db_key] = create_database(
            db_key         = db_key,
            db_config      = DATABASES[db_key],
            df_cohort      = df_cohort,
            config_file    = file,
            found_pancreas = found_pancreas,
            tumor_label_id = tumor_label_id,
            output_root    = OUTPUT_ROOT,
        )

    print_summary(results)