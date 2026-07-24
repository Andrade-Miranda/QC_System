#!/usr/bin/env python3
"""
nnUNet Dataset Generator
=========================
Converts the curated case lists into a properly structured nnUNet raw dataset
using **symlinks by default** (no data duplication).

The source dataset is determined by RAW_DATASET_ROOT in configs/paths.yaml.

Output location (from paths.yaml → OUTPUT_NNUNET_DIR):
    <OUTPUT_NNUNET_DIR>/Dataset<ID>_<name>/
    ├── dataset.json
    ├── imagesTr/          (symlinks → <RAW_DATASET_ROOT>/imagesTr/*)
    └── labelsTr/          (symlinks → <RAW_DATASET_ROOT>/labelsTr/*)

Usage:
    python scripts/generate_nnunet_dataset.py
    python scripts/generate_nnunet_dataset.py --dataset-id 42 --dataset-name MyPancreas
    python scripts/generate_nnunet_dataset.py --mode copy
    python scripts/generate_nnunet_dataset.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import (
    resolve_project_paths,
    ensure_output_directories,
    build_output_dirs,
    setup_file_logging,
    VALID_TASK_MODES,
)

logger = logging.getLogger("generate_nnunet_dataset")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _link_or_copy(src: Path, dst: Path, mode: str, dry_run: bool) -> None:
    """Create a symlink or copy src → dst, skipping if dst already exists."""
    if dst.exists() or dst.is_symlink():
        return
    if dry_run:
        print(f"  [dry] {'symlink' if mode == 'symlink' else 'copy'}: {src} → {dst}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        if src.is_dir():
            shutil.copytree(src, dst, symlinks=False)
        else:
            shutil.copy2(src, dst)


def _build_dataset_json(
    dataset_name: str,
    dataset_id: int,
    n_train: int,
    n_test: int,
) -> dict:
    """Return the dataset.json dict required by nnUNet v2."""
    return {
        "name":              dataset_name,
        "description":       f"{dataset_name} dataset",
        "tensorImageSize":   "4D",
        "reference":         dataset_name,
        "licence":           "see original dataset",
        "release":           "1.0",
        "channel_names":     {"0": "CT"},
        "labels": {
            "background": 0,
            "pancreas":   1,
            "lesion":     2,
        },
        "numTraining":       n_train,
        "numTest":           n_test,
        "file_ending":       ".nii.gz",
        "dataset_id":        dataset_id,
    }


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def generate_nnunet_dataset(
    nnunet_dir:    Path,
    raw_images:    Path,
    raw_labels:    Path,
    train_cases:   list[str],
    val_cases:     list[str],
    test_cases:    list[str],
    image_suffix:  str,
    dataset_id:    int,
    dataset_name:  str,
    mode:          str,
    dry_run:       bool,
) -> Path:
    """
    Build the nnUNet dataset directory.  Returns the dataset root path.
    """
    ds_root = nnunet_dir / f"Dataset{dataset_id:03d}_{dataset_name}"
    images_tr_dir = ds_root / "imagesTr"
    labels_tr_dir = ds_root / "labelsTr"
    images_ts_dir = ds_root / "imagesTs"
    labels_ts_dir = ds_root / "labelsTs"

    if not dry_run:
        for d in (images_tr_dir, labels_tr_dir, images_ts_dir, labels_ts_dir):
            d.mkdir(parents=True, exist_ok=True)

    # Training + validation → imagesTr / labelsTr
    all_train = train_cases + val_cases
    logger.info("Creating imagesTr / labelsTr for %d cases", len(all_train))
    for case_id in sorted(all_train):
        img_src  = raw_images / f"{case_id}{image_suffix}"
        lbl_src  = raw_labels / case_id
        img_dst  = images_tr_dir / f"{case_id}{image_suffix}"
        lbl_dst  = labels_tr_dir / case_id

        if img_src.exists():
            _link_or_copy(img_src, img_dst, mode, dry_run)
        else:
            logger.warning("Image not found: %s", img_src)

        if lbl_src.exists():
            _link_or_copy(lbl_src, lbl_dst, mode, dry_run)
        else:
            logger.warning("Label dir not found: %s", lbl_src)

    # Test cases → imagesTs / labelsTs
    logger.info("Creating imagesTs / labelsTs for %d cases", len(test_cases))
    for case_id in sorted(test_cases):
        img_src  = raw_images / f"{case_id}{image_suffix}"
        lbl_src  = raw_labels / case_id
        img_dst  = images_ts_dir / f"{case_id}{image_suffix}"
        lbl_dst  = labels_ts_dir / case_id

        if img_src.exists():
            _link_or_copy(img_src, img_dst, mode, dry_run)
        if lbl_src.exists():
            _link_or_copy(lbl_src, lbl_dst, mode, dry_run)

    # Write dataset.json
    ds_json = _build_dataset_json(
        dataset_name, dataset_id,
        n_train=len(all_train),
        n_test=len(test_cases),
    )
    ds_json_path = ds_root / "dataset.json"
    if not dry_run:
        ds_json_path.write_text(json.dumps(ds_json, indent=2), encoding="utf-8")
        logger.info("Wrote %s", ds_json_path)
    else:
        print(f"  [dry] would write dataset.json to {ds_json_path}")

    return ds_root


def _load_cases_txt(path: Path) -> list[str]:
    """Read a case-list .txt file (one case ID per line, ignoring comments/blanks)."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.startswith("#")]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a nnUNet-ready dataset from curated split files.")
    parser.add_argument("--paths-yaml",    default=None,
                        help="Path to paths.yaml (default: configs/paths.yaml)")
    parser.add_argument("--dataset-id",    type=int, default=100,
                        help="nnUNet dataset ID (padded to 3 digits, e.g. 100 → Dataset100)")
    parser.add_argument("--dataset-name",  default=None,
                        help="Dataset name appended after the ID (default: inferred from RAW_DATASET_ROOT)")
    parser.add_argument("--mode",          default="symlink",
                        choices=["symlink", "copy"],
                        help="symlink (default, no data duplication) or copy")
    parser.add_argument("--train-txt",     default=None,
                        help="Path to train_cases.txt (default: outputs/curation/train_cases.txt)")
    parser.add_argument("--val-txt",       default=None,
                        help="Path to val_cases.txt (default: outputs/curation/val_cases.txt)")
    parser.add_argument("--test-txt",      default=None,
                        help="Path to test_cases.txt (default: outputs/curation/test_cases.txt)")
    parser.add_argument("--dry-run",       action="store_true",
                        help="Show what would be done without creating any files")
    parser.add_argument("--task-mode",     default=None, dest="task_mode",
                        choices=sorted(VALID_TASK_MODES),
                        help="Task mode (default: read from thresholds.yaml) — determines "
                             "which curation subfolder to read split files from. "
                             + " | ".join(sorted(VALID_TASK_MODES)))
    args = parser.parse_args()

    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    ensure_output_directories(paths)  # creates shared logs_dir only

    # Resolve task_mode: CLI > thresholds.yaml > default
    _thr_task_mode = "pancreas_lesion_subregions"
    _thr_path = paths.thresholds_config
    if _thr_path.exists():
        try:
            import yaml as _yaml
            _thr_raw = _yaml.safe_load(_thr_path.read_text(encoding="utf-8")) or {}
            _v = str(_thr_raw.get("TASK_MODE", "")).strip().lower()
            if _v and _v != "auto" and _v in VALID_TASK_MODES:
                _thr_task_mode = _v
        except Exception:
            pass
    _task_mode = args.task_mode if args.task_mode else _thr_task_mode

    # Build task-mode-specific dirs to locate curation split files
    _base_output_dir = paths.summary_dir.parent  # outputs/<dataset_name>/
    _task_dirs = build_output_dirs(_base_output_dir, _task_mode, create=False)
    # Default dataset name: use explicit arg, else infer from RAW_DATASET_ROOT folder name
    if args.dataset_name is None:
        args.dataset_name = paths.dataset_name or "Dataset"
    lg = setup_file_logging(paths.logs_dir / "generate_nnunet_dataset.log",
                             logger_name="generate_nnunet_dataset")

    train_txt = Path(args.train_txt) if args.train_txt else _task_dirs["curation_dir"] / "train_cases.txt"
    val_txt   = Path(args.val_txt)   if args.val_txt   else _task_dirs["curation_dir"] / "val_cases.txt"
    test_txt  = Path(args.test_txt)  if args.test_txt  else _task_dirs["curation_dir"] / "test_cases.txt"

    train_cases = _load_cases_txt(train_txt)
    val_cases   = _load_cases_txt(val_txt)
    test_cases  = _load_cases_txt(test_txt)

    if not train_cases and not val_cases and not test_cases:
        print("[ERROR] No case list files found. Artifact-native curation/splitting is not implemented yet.")
        print("        Provide --train-txt/--val-txt/--test-txt explicitly, or add a future CurationAgent output.")
        print(f"  Expected: {train_txt}")
        raise SystemExit(1)

    print(f"Cases — train: {len(train_cases)}, val: {len(val_cases)}, test: {len(test_cases)}")
    print(f"Mode    : {args.mode}")
    print(f"Output  : {paths.nnunet_dir / f'Dataset{args.dataset_id:03d}_{args.dataset_name}'}")
    if args.dry_run:
        print("(dry-run — no files will be created)\n")

    lg.info("train=%d  val=%d  test=%d  mode=%s", len(train_cases), len(val_cases),
            len(test_cases), args.mode)

    ds_root = generate_nnunet_dataset(
        nnunet_dir   = paths.nnunet_dir,
        raw_images   = paths.raw_images_dir,
        raw_labels   = paths.raw_labels_dir,
        train_cases  = train_cases,
        val_cases    = val_cases,
        test_cases   = test_cases,
        image_suffix = paths.image_suffix,
        dataset_id   = args.dataset_id,
        dataset_name = args.dataset_name,
        mode         = args.mode,
        dry_run      = args.dry_run,
    )

    if not args.dry_run:
        print(f"\nDataset created at: {ds_root}")
        lg.info("Dataset written to %s", ds_root)
    else:
        print("\nDry-run complete — no files written.")


if __name__ == "__main__":
    main()
