#!/usr/bin/env python3
from __future__ import annotations
"""
Dataset Summarizer
==================
Generates a comprehensive JSON summary (<dataset_name>_summary.json) for every
training case in the dataset pointed to by RAW_DATASET_ROOT in paths.yaml.

Each entry contains:
  - Image information  (shape, spacing, paths)
  - Metadata           (from metadata.xlsx)
  - Pancreas           (volume, centroid, bounding box)
  - Regions            (head / body / tail statistics + overlap with pancreas)
  - Region consistency (sum vs. total, inter-region overlaps)
  - Lesions            (per connected component statistics)
  - QC flags           (empty masks, shape match, spacing validity …)

Usage:
    python scripts/summarize_dataset.py [--paths-yaml configs/paths.yaml]
"""

import argparse
import datetime
import json
import multiprocessing
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

try:
    import yaml as _yaml
    _HAS_YAML = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_YAML = False

# ---------------------------------------------------------------------------
# Bootstrap: make agents/ importable when running this script directly
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import (
    resolve_project_paths,
    ensure_output_directories,
    setup_file_logging,
    ProjectPaths,
    build_output_dirs,
    VALID_TASK_MODES,
)

# ===========================================================================
# CONFIGURATION  — loaded from configs/paths.yaml
# ===========================================================================

def _load_config(yaml_path: Path | None = None) -> ProjectPaths:
    """Resolve all project paths from paths.yaml."""
    paths = resolve_project_paths(yaml_path)
    ensure_output_directories(paths)
    return paths

# Deferred globals — populated in main() after loading config
_PATHS: ProjectPaths | None = None

def _paths() -> ProjectPaths:
    assert _PATHS is not None, "Call _load_config() before using path globals."
    return _PATHS

# Convenience aliases — referenced throughout the script
def _IMAGES_DIR()   -> Path: return _paths().raw_images_dir
def _LABELS_DIR()   -> Path: return _paths().raw_labels_dir
def _IMAGE_SUFFIX() -> str:  return _paths().image_suffix

# Backward-compatible module-level names (set in main, used by process_case)
ROOT          = None   # type: ignore[assignment]
METADATA_PATH = None   # type: ignore[assignment]
OUTPUT_JSON   = None   # type: ignore[assignment]
IMAGES_DIR    = None   # type: ignore[assignment]
IMAGE_SUFFIX  = "_0000.nii.gz"  # safe default until config is loaded

# ---------------------------------------------------------------------------
# Parallel processing support
# ---------------------------------------------------------------------------
# Populated by the pool initializer so workers do not receive meta_dict on
# every task (avoids serializing the full dict N times).
_worker_meta_dict: dict = {}


def _init_worker(paths_obj: "ProjectPaths", meta_dict_snapshot: dict) -> None:
    """Pool initializer — copy path config and metadata into each worker."""
    global _PATHS, ROOT, METADATA_PATH, IMAGES_DIR, IMAGE_SUFFIX
    global _worker_meta_dict
    _PATHS            = paths_obj
    ROOT              = paths_obj.raw_root
    METADATA_PATH     = paths_obj.raw_metadata
    IMAGES_DIR        = paths_obj.raw_images_dir
    IMAGE_SUFFIX      = paths_obj.image_suffix
    _worker_meta_dict = meta_dict_snapshot


def _parallel_worker(task: tuple) -> tuple:
    """Top-level worker function called by multiprocessing.Pool.

    Must be a module-level function for pickle compatibility.
    Receives (case_id, task_mode) and returns (case_id, result_dict).
    """
    case_id, task_mode = task
    return case_id, process_case(case_id, _worker_meta_dict, task_mode)


# Segmentation filenames (inside labelsTr/{case_id}/segmentations/)
SEG_PANCREAS = "pancreas.nii.gz"
SEG_HEAD     = "pancreas_head.nii.gz"
SEG_BODY     = "pancreas_body.nii.gz"
SEG_TAIL     = "pancreas_tail.nii.gz"
SEG_LESION   = "pancreatic_lesion.nii.gz"

# Mapping: metadata.xlsx column name → output JSON key
METADATA_COLUMNS = {
    "sex":                "sex",
    "age":                "age",
    "ct phase":           "ct_phase",
    "manufacturer":       "manufacturer",
    "manufacturer model": "manufacturer_model",
    "study type":         "study_type",
    "site nationality":   "site_nationality",
}

# ===========================================================================
# TASK MODE HELPERS
# ===========================================================================

def get_task_mode(thr: dict | None) -> str:
    """Return the effective TASK_MODE from a loaded thresholds dict.

    Defaults to ``"pancreas_lesion_subregions"`` for backward compatibility.
    ``"auto"`` (a qc_agent concept) resolves to the default here because the
    summarizer has no existing summary data to auto-detect from.
    """
    if thr:
        raw = thr.get("TASK_MODE", "")
        mode = str(raw).strip().lower()
        if mode and mode != "auto":
            return mode
    return "pancreas_lesion_subregions"


def should_compute_lesion(task_mode: str) -> bool:
    """Return True if this task mode requires lesion mask loading/analysis."""
    return task_mode in ("pancreas_lesion", "pancreas_lesion_subregions")


def should_compute_subregions(task_mode: str) -> bool:
    """Return True if this task mode requires head/body/tail sub-region analysis."""
    return task_mode == "pancreas_lesion_subregions"


def case_id_from_image(path: Path, suffix: str) -> str:
    name = path.name
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _load_thresholds(yaml_path: "Path | None" = None) -> dict:
    """Load thresholds.yaml and return as a plain dict. Returns {} on any failure."""
    if yaml_path is None:
        yaml_path = _PROJECT_ROOT / "configs" / "thresholds.yaml"
    if not yaml_path.exists():
        return {}
    if not _HAS_YAML:
        print("[WARN] PyYAML not available — TASK_MODE will use default.")
        return {}
    try:
        with open(yaml_path) as fh:
            return _yaml.safe_load(fh) or {}
    except Exception as exc:
        print(f"[WARN] Could not load thresholds.yaml: {exc}")
        return {}


# ===========================================================================
# UTILITIES
# ===========================================================================

def _to_python(val):
    """Convert any numpy / pandas scalar to a plain Python type (or None)."""
    if val is None:
        return None
    if isinstance(val, float) and (val != val):          # float NaN check
        return None
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        return None if np.isnan(val) else float(val)
    if isinstance(val, pd.Timestamp):
        return str(val)
    return val


def _r(val, decimals=4):
    """Round a numeric value to `decimals` places and return as Python float."""
    if val is None:
        return None
    return round(float(val), decimals)


def load_sitk_image(path: Path):
    """Load a NIfTI file with SimpleITK.  Returns None if the file is missing or unreadable.
    Used for segmentation masks only; CT image loading goes through _load_ct_image().
    """
    if path is None or not path.exists():
        return None
    try:
        return sitk.ReadImage(str(path))
    except Exception as exc:
        print(f"  [WARN] Cannot load {path}: {exc}")
        return None


def _valid_mask_geometry(mask_sitk, ref_sitk) -> bool:
    if mask_sitk is None or ref_sitk is None:
        return False
    try:
        return bool(compare_to_reference_geometry(ref_sitk, mask_sitk).get("affine_consistency_with_image"))
    except Exception:
        return False


def build_lesion_annotation_evidence(
    *,
    separate_path: Path,
    separate_sitk,
    separate_arr: np.ndarray | None,
    ref_sitk,
    task_mode: str,
) -> dict:
    """Classify deterministic lesion-annotation evidence for lesion tasks.

    Lesion evidence for the current AgentQC experiments comes exclusively from
    the separate binary ``segmentations/pancreatic_lesion.nii.gz`` mask.
    Missing, unreadable, or geometry-invalid masks do not prove absence.  Empty
    but readable geometry-valid separate masks prove absence for the configured
    lesion task.
    """
    if not should_compute_lesion(task_mode):
        return {
            "status": "not_applicable",
            "lesion_present": None,
            "absence_confirmed": False,
            "selected_source": None,
            "sources": [],
        }

    def _source_entry(name: str, path: Path, sitk_obj, arr: np.ndarray | None) -> dict:
        exists = path.exists()
        readable = sitk_obj is not None
        geometry_valid = _valid_mask_geometry(sitk_obj, ref_sitk) if readable else False
        positive_voxels = None
        if readable and geometry_valid and arr is not None:
            positive_voxels = int(np.count_nonzero(arr > 0))
        status = "valid"
        if not exists:
            status = "missing"
        elif not readable:
            status = "unreadable"
        elif not geometry_valid:
            status = "geometry_mismatch"
        return {
            "name": name,
            "path": str(path),
            "exists": bool(exists),
            "readable": bool(readable),
            "geometry_valid": bool(geometry_valid),
            "status": status,
            "label_value": None,
            "positive_voxels": positive_voxels,
            "lesion_present": (positive_voxels is not None and positive_voxels > 0),
        }

    separate_source = _source_entry("separate_lesion_mask", separate_path, separate_sitk, separate_arr)
    sources = [separate_source]

    if separate_source["status"] != "valid":
        status = "insufficient_evidence"
        selected = None
        lesion_present = None
    else:
        lesion_present = bool(separate_source["lesion_present"])
        status = "lesion_present" if lesion_present else "confirmed_absent"
        selected = "separate_lesion_mask"

    return {
        "status": status,
        "lesion_present": lesion_present,
        "absence_confirmed": status == "confirmed_absent",
        "selected_source": selected,
        "sources": sources,
    }


# ===========================================================================
# CASE-STATUS HELPERS
# ===========================================================================

def _case_status_success() -> dict:
    """Return the case_status block for a successfully processed case."""
    return {
        "included_in_summary"  : True,
        "processing_successful": True,
        "usable_for_training"  : True,
        "omit_from_training"   : False,
        "omit_category"        : None,
        "error_type"           : None,
        "omit_reason"          : None,
        "error_stage"          : None,
        "error_message"        : None,
        "blocking_issues"      : [],
    }


def _case_status_error(
    error_type: str,
    omit_category: str,
    omit_reason: str,
    error_stage: str,
    error_message: str,
    blocking_issues: list,
) -> dict:
    """Return the case_status block for a case that could not be processed."""
    return {
        "included_in_summary"  : True,
        "processing_successful": False,
        "usable_for_training"  : False,
        "omit_from_training"   : True,
        "omit_category"        : omit_category,
        "error_type"           : error_type,
        "omit_reason"          : omit_reason,
        "error_stage"          : error_stage,
        "error_message"        : error_message,
        "blocking_issues"      : blocking_issues,
    }


def _null_case_schema(case_id: str, meta_dict: dict, case_status: dict) -> dict:
    """Return a schema-compatible entry with null values for a case that failed to load."""
    metadata = dict(meta_dict.get(case_id, {}))
    for key in METADATA_COLUMNS.values():
        metadata.setdefault(key, None)
    return {
        "case_id"        : case_id,
        "image"          : {
            "shape_zyx"       : None,
            "spacing_xyz_mm"  : None,
            "voxel_volume_mm3": None,
        },
        "metadata"       : metadata,
        "geometry"       : {
            "reference_image": None,
            "pancreas_mask"  : None,
            "lesion_mask"    : None,
            "summary"        : {
                "pancreas_affine_mismatch": None,
                "lesion_affine_mismatch"  : None,
                "any_affine_mismatch"     : None,
            },
        },
        "pancreas"       : {
            "volume_mm3"            : None,
            "centroid_voxel_zyx"    : None,
            "centroid_world_xyz_mm" : None,
            "bounding_box_zyx"      : None,
            "regions"               : {"head": None, "body": None, "tail": None},
            "region_consistency"    : None,
        },
        "lesions"        : {
            "n_lesions"       : None,
            "total_volume_mm3": None,
            "per_lesion"      : [],
        },
        "quality_control": {
            "pancreas_mask_empty"               : None,
            "lesion_mask_empty"                 : None,
            "lesion_pancreas_overlap_exists"    : None,
            "total_overlap_lesion_pancreas_mm3" : None,
            "total_overlap_rate_vs_lesion"      : None,
            "total_overlap_rate_vs_pancreas"    : None,
            "pancreas_mask_affine_mismatch"     : None,
            "lesion_mask_affine_mismatch"       : None,
            "any_affine_mismatch"               : None,
            "lesion_annotation_evidence"        : None,
            "lesion_overlap_analysis"           : None,
        },
        "hu_statistics"  : {
            "ct_phase"                      : None,
            "hu_statistics_status"          : None,
            "tumor_mean_hu"                 : None,
            "tumor_median_hu"               : None,
            "tumor_std_hu"                  : None,
            "tumor_p05_hu"                  : None,
            "tumor_p25_hu"                  : None,
            "tumor_p75_hu"                  : None,
            "tumor_p95_hu"                  : None,
            "tumor_voxel_count"             : None,
            "pancreas_mean_hu"              : None,
            "pancreas_median_hu"            : None,
            "pancreas_std_hu"               : None,
            "pancreas_p05_hu"               : None,
            "pancreas_p25_hu"               : None,
            "pancreas_p75_hu"               : None,
            "pancreas_p95_hu"               : None,
            "pancreas_voxel_count"          : None,
            "delta_hu_tumor_vs_pancreas"    : None,
            "z_score_hu"                    : None,
            "tumor_attenuation"             : "Unknown",
            "attenuation_rule_used"         : "unknown",
            "delta_hu_attenuation"          : "Unknown",
            "attenuation_rule_disagreement" : False,
            "suspicious_hu_distribution"    : False,
        },
        "subregion_qc"   : None,
        "case_status"    : case_status,
    }


def sitk_to_uint8(sitk_img) -> np.ndarray:
    """Convert a SimpleITK image to a uint8 numpy array in (z, y, x) order."""
    return sitk.GetArrayFromImage(sitk_img).astype(np.uint8)


# ===========================================================================
# ROBUST HU STATISTICS
# ===========================================================================

def compute_hu_statistics(
    image_data: np.ndarray,
    tumor_mask: np.ndarray | None,
    pancreas_arr: np.ndarray | None,
    ct_phase: str | None,
    lesion_present: bool = True,
) -> dict:
    """Compute robust HU statistics for tumor and normal-pancreas reference.

    For positive cases (lesion_present=True):
      Normal pancreas reference = pancreas AND NOT tumor, eroded + percentile-
      clipped. Primary classification uses z_score_hu (fallback: delta_hu).

    For negative cases (lesion_present=False):
      Tumor fields are null. Pancreas stats are still computed when the pancreas
      mask is present — useful for normal-tissue characterisation.
      hu_statistics_status = "pancreas_only_negative_case" or
                             "no_pancreas_no_lesion_negative_case".

    Returns the full hu_statistics dict with all fields expected by the schema.
    """
    _MIN_TUMOR_VOXELS    = 10
    _MIN_PANCREAS_VOXELS = 50

    null: dict = {
        "ct_phase"                      : ct_phase,
        "hu_statistics_status"          : "unknown",
        # Tumor stats
        "tumor_mean_hu"                 : None,
        "tumor_median_hu"               : None,
        "tumor_std_hu"                  : None,
        "tumor_p05_hu"                  : None,
        "tumor_p25_hu"                  : None,
        "tumor_p75_hu"                  : None,
        "tumor_p95_hu"                  : None,
        "tumor_voxel_count"             : None,
        # Pancreas reference stats
        "pancreas_mean_hu"              : None,
        "pancreas_median_hu"            : None,
        "pancreas_std_hu"               : None,
        "pancreas_p05_hu"               : None,
        "pancreas_p25_hu"               : None,
        "pancreas_p75_hu"               : None,
        "pancreas_p95_hu"               : None,
        "pancreas_voxel_count"          : None,
        # Derived
        "delta_hu_tumor_vs_pancreas"    : None,
        "z_score_hu"                    : None,
        "tumor_attenuation"             : "Unknown",
        "attenuation_rule_used"         : "unknown",
        "delta_hu_attenuation"          : "Unknown",
        "attenuation_rule_disagreement" : False,
        "suspicious_hu_distribution"    : False,
    }

    def _pancreas_stats(exclude_tumor: bool) -> dict | None:
        """Return pancreas HU stats dict or None if insufficient voxels."""
        if pancreas_arr is None or not np.any(pancreas_arr > 0):
            return None
        pan_mask = (pancreas_arr > 0)
        if exclude_tumor and tumor_mask is not None and np.any(tumor_mask):
            pan_mask = pan_mask & ~tumor_mask
        eroded = ndimage.binary_erosion(pan_mask)
        core = eroded if eroded.sum() >= _MIN_PANCREAS_VOXELS else pan_mask
        if not np.any(core):
            return None
        raw = image_data[core].astype(np.float32)
        p05, p95 = float(np.percentile(raw, 5)), float(np.percentile(raw, 95))
        clipped = raw[(raw >= p05) & (raw <= p95)]
        pan_hu = clipped if len(clipped) >= _MIN_PANCREAS_VOXELS else raw
        return {
            "pancreas_voxel_count": int(len(pan_hu)),
            "pancreas_mean_hu"    : round(float(np.mean(pan_hu)),            2),
            "pancreas_median_hu"  : round(float(np.median(pan_hu)),          2),
            "pancreas_std_hu"     : round(float(np.std(pan_hu)),             2),
            "pancreas_p05_hu"     : round(float(np.percentile(pan_hu,  5)), 2),
            "pancreas_p25_hu"     : round(float(np.percentile(pan_hu, 25)), 2),
            "pancreas_p75_hu"     : round(float(np.percentile(pan_hu, 75)), 2),
            "pancreas_p95_hu"     : round(float(np.percentile(pan_hu, 95)), 2),
        }

    # ── Negative case (no lesion) ─────────────────────────────────────────
    if not lesion_present:
        pan_stats = _pancreas_stats(exclude_tumor=False)
        if pan_stats is None:
            return {**null,
                    "hu_statistics_status" : "no_pancreas_no_lesion_negative_case",
                    "tumor_attenuation"    : "Not applicable",
                    "attenuation_rule_used": "not_applicable_negative_case",
                    "delta_hu_attenuation" : "Not applicable"}
        result = dict(null)
        result.update(pan_stats)
        result["hu_statistics_status"]   = "pancreas_only_negative_case"
        result["tumor_attenuation"]      = "Not applicable"
        result["attenuation_rule_used"]  = "not_applicable_negative_case"
        result["delta_hu_attenuation"]   = "Not applicable"
        result["suspicious_hu_distribution"] = False
        return result

    # ── Positive case but no tumor voxels (edge case / segmentation failure) ──
    if tumor_mask is None or not np.any(tumor_mask):
        return null

    result = dict(null)
    result["hu_statistics_status"] = "full"

    # ── Tumor HU stats ────────────────────────────────────────────────────
    tumor_hu = image_data[tumor_mask].astype(np.float32)
    result["tumor_voxel_count"] = int(len(tumor_hu))
    result["tumor_mean_hu"]     = round(float(np.mean(tumor_hu)),   2)
    result["tumor_median_hu"]   = round(float(np.median(tumor_hu)), 2)
    result["tumor_std_hu"]      = round(float(np.std(tumor_hu)),    2)
    result["tumor_p05_hu"]      = round(float(np.percentile(tumor_hu,  5)), 2)
    result["tumor_p25_hu"]      = round(float(np.percentile(tumor_hu, 25)), 2)
    result["tumor_p75_hu"]      = round(float(np.percentile(tumor_hu, 75)), 2)
    result["tumor_p95_hu"]      = round(float(np.percentile(tumor_hu, 95)), 2)

    # ── Normal pancreas reference (exclude tumor voxels) ─────────────────
    pan_stats = _pancreas_stats(exclude_tumor=True)
    if pan_stats is None:
        result["suspicious_hu_distribution"] = True
        return result
    result.update(pan_stats)

    # ── Delta HU ─────────────────────────────────────────────────────────
    delta = round(result["tumor_median_hu"] - result["pancreas_median_hu"], 2)
    result["delta_hu_tumor_vs_pancreas"] = delta

    # Delta-based secondary label
    if delta < -10:
        delta_att = "Hypo"
    elif delta > 10:
        delta_att = "Hyper"
    else:
        delta_att = "Iso"
    result["delta_hu_attenuation"] = delta_att

    # ── Z-score (primary rule) ────────────────────────────────────────────
    pan_std = result["pancreas_std_hu"]
    z_valid = (pan_std is not None
               and not np.isnan(pan_std)
               and 1e-3 < pan_std < 200)

    if z_valid:
        z = round(float(delta / pan_std), 4)
        result["z_score_hu"] = z
        if z < -2:
            z_att = "Hypo"
        elif z > 2:
            z_att = "Hyper"
        else:
            z_att = "Iso"
        result["tumor_attenuation"]     = z_att
        result["attenuation_rule_used"] = "z_score"
        if z_att != delta_att:
            result["attenuation_rule_disagreement"] = True
    else:
        result["tumor_attenuation"]     = delta_att
        result["attenuation_rule_used"] = "delta_hu_fallback"

    # ── Suspicious distribution flag ──────────────────────────────────────
    suspicious = False
    if result["tumor_voxel_count"] < _MIN_TUMOR_VOXELS:
        suspicious = True
    if result["pancreas_voxel_count"] < _MIN_PANCREAS_VOXELS:
        suspicious = True
    if not z_valid:
        suspicious = True
    t_mean   = result["tumor_mean_hu"]
    t_median = result["tumor_median_hu"]
    if t_mean is not None and t_median is not None and abs(t_mean - t_median) > 50:
        suspicious = True
    result["suspicious_hu_distribution"] = suspicious

    return result


def sitk_to_float32(sitk_img) -> np.ndarray:
    """Return CT image as float32 numpy array, preserving original HU values."""
    return sitk.GetArrayFromImage(sitk_img).astype(np.float32)


def get_spacing_xyz(sitk_img) -> tuple:
    """
    Return voxel spacing (sx, sy, sz) in mm.
    SimpleITK stores spacing in x, y, z order — this is preserved here.
    """
    return sitk_img.GetSpacing()   # (sx, sy, sz)


def voxel_volume_mm3(spacing_xyz: tuple) -> float:
    """Scalar voxel volume in mm³."""
    return float(spacing_xyz[0] * spacing_xyz[1] * spacing_xyz[2])


def centroid_zyx_to_world_xyz(sitk_img, centroid_zyx: list) -> list:
    """
    Convert a (z, y, x) voxel centroid to physical world coordinates (x, y, z) in mm.
    SimpleITK's TransformContinuousIndexToPhysicalPoint expects (x, y, z) index order.
    """
    z, y, x = centroid_zyx
    world = sitk_img.TransformContinuousIndexToPhysicalPoint((float(x), float(y), float(z)))
    return [_r(w, 4) for w in world]


def safe_rate(numerator, denominator) -> float | None:
    """Return numerator / denominator, or None when denominator is zero or None."""
    if denominator is None or denominator == 0:
        return None
    return _r(float(numerator) / float(denominator), 6)


def overlap_mm3_fn(arr1, arr2, vox_vol: float) -> float:
    """Intersection volume of two binary masks in mm³ (0.0 if either is None)."""
    if arr1 is None or arr2 is None:
        return 0.0
    return _r(float(np.logical_and(arr1 > 0, arr2 > 0).sum()) * vox_vol, 4)


# ===========================================================================
# GEOMETRY CONSISTENCY HELPERS
# ===========================================================================

def compute_direction_metrics(direction) -> dict:
    """
    Compute orthogonality error, determinant, and column-axis norms for a
    9-element row-major direction tuple (as returned by SimpleITK.GetDirection()).

    Orthogonality thresholds:
      < 1e-6  → normal (floating-point noise only)
      1e-6–1e-3 → warning (minor numerical drift)
      > 1e-2  → suspicious (likely non-orthonormal cosines)
    """
    R = np.array(direction, dtype=float).reshape(3, 3)
    orth_error = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det        = float(np.linalg.det(R))
    axis_norms = [float(np.linalg.norm(R[:, i])) for i in range(3)]
    return {
        "orthogonality_error":             _r(orth_error, 8),
        "determinant":                     _r(det, 8),
        "axis_norms":                      [_r(n, 6) for n in axis_norms],
        "non_orthonormal_direction_cosines": orth_error > 1e-3,
    }


def extract_sitk_geometry(img) -> dict:
    """
    Extract full geometry metadata from a SimpleITK image object.

    Returns size_xyz, shape_zyx, spacing_xyz_mm, origin_xyz_mm,
    direction_matrix (3×3 list), and direction quality metrics.
    """
    size_xyz   = list(img.GetSize())           # (nx, ny, nz)
    spacing    = [_r(float(s), 6) for s in img.GetSpacing()]
    origin     = [_r(float(o), 6) for o in img.GetOrigin()]
    direction  = img.GetDirection()            # 9-element flat tuple
    dir_matrix = [list(direction[i * 3:(i + 1) * 3]) for i in range(3)]
    dm         = compute_direction_metrics(direction)
    shape_zyx  = list(reversed(size_xyz))      # (nz, ny, nx)
    return {
        "size_xyz":                          size_xyz,
        "shape_zyx":                         shape_zyx,
        "spacing_xyz_mm":                    spacing,
        "origin_xyz_mm":                     origin,
        "direction_matrix":                  dir_matrix,
        "orthogonality_error":               dm["orthogonality_error"],
        "determinant":                       dm["determinant"],
        "axis_norms":                        dm["axis_norms"],
        "non_orthonormal_direction_cosines": dm["non_orthonormal_direction_cosines"],
    }


def compare_to_reference_geometry(
    ref_img,
    mask_img,
    spacing_atol:   float = 1e-4,
    origin_atol:    float = 1e-2,
    direction_atol: float = 1e-4,
) -> dict:
    """
    Compare mask_img geometry against ref_img (the CT image).

    Returns per-field boolean match flags and an overall
    affine_consistency_with_image flag (True only when all four match).
    """
    size_match = list(mask_img.GetSize()) == list(ref_img.GetSize())

    sp_ref  = np.array(ref_img.GetSpacing(),   dtype=float)
    sp_mask = np.array(mask_img.GetSpacing(),  dtype=float)
    spacing_match = bool(np.allclose(sp_ref, sp_mask, atol=spacing_atol))

    or_ref  = np.array(ref_img.GetOrigin(),   dtype=float)
    or_mask = np.array(mask_img.GetOrigin(),  dtype=float)
    origin_match = bool(np.allclose(or_ref, or_mask, atol=origin_atol))

    di_ref  = np.array(ref_img.GetDirection(),  dtype=float)
    di_mask = np.array(mask_img.GetDirection(), dtype=float)
    direction_match = bool(np.allclose(di_ref, di_mask, atol=direction_atol))

    return {
        "size_match_with_image":         size_match,
        "spacing_match_with_image":      spacing_match,
        "origin_match_with_image":       origin_match,
        "direction_match_with_image":    direction_match,
        "affine_consistency_with_image": size_match and spacing_match and origin_match and direction_match,
    }


# ===========================================================================
# METADATA LOADING
# ===========================================================================

def load_metadata(xlsx_path: Path) -> dict:
    """
    Parse metadata.xlsx into a dict keyed by PanTS ID.
    Missing values are stored as None.  Gracefully handles missing columns.
    """
    meta: dict = {}
    if not xlsx_path.exists():
        print(f"[WARN] metadata.xlsx not found: {xlsx_path}")
        return meta
    try:
        df = pd.read_excel(xlsx_path)
    except Exception as exc:
        print(f"[ERROR] Cannot read metadata.xlsx: {exc}")
        return meta

    for _, row in df.iterrows():
        case_id = str(row.get("PanTS ID", "")).strip()
        if not case_id:
            continue
        entry: dict = {}
        for col, key in METADATA_COLUMNS.items():
            raw = row.get(col, None)
            entry[key] = _to_python(raw)
        meta[case_id] = entry
    return meta


# ===========================================================================
# MASK STATISTICS
# ===========================================================================

def compute_mask_stats(arr, sitk_img, vox_vol: float) -> dict:
    """
    Compute spatial statistics for a binary mask array (z, y, x).

    Returns a dict with:
      volume_mm3, centroid_voxel_zyx, centroid_world_xyz_mm,
      bbox_voxel_zyx, bbox_size_voxel_zyx, bbox_size_physical_mm_zyx, bbox_volume_mm3.

    All spatial fields are None when the mask is empty or missing.
    """
    stats = {
        "volume_mm3":            None,
        "centroid_voxel_zyx":    None,
        "centroid_world_xyz_mm": None,
    }

    if arr is None or arr.sum() == 0:
        stats["volume_mm3"] = 0.0
        return stats

    n_vox = int(arr.sum())
    stats["volume_mm3"] = _r(n_vox * vox_vol, 4)

    # Centroid in voxel space (z, y, x)
    idx = np.nonzero(arr)                   # tuple: (z_indices, y_indices, x_indices)
    cz = float(np.mean(idx[0]))
    cy = float(np.mean(idx[1]))
    cx = float(np.mean(idx[2]))
    centroid_zyx = [_r(cz, 4), _r(cy, 4), _r(cx, 4)]
    stats["centroid_voxel_zyx"] = centroid_zyx
    stats["centroid_world_xyz_mm"] = centroid_zyx_to_world_xyz(sitk_img, centroid_zyx)

    # Bounding box: [[z_min,y_min,x_min], [z_max,y_max,x_max]] in voxel space
    stats["bounding_box_zyx"] = [
        [int(idx[0].min()), int(idx[1].min()), int(idx[2].min())],
        [int(idx[0].max()), int(idx[1].max()), int(idx[2].max())],
    ]

    return stats


# ===========================================================================
# LESION ANALYSIS
# ===========================================================================

def compute_lesion_info(
    lesion_arr, pancreas_arr, head_arr, body_arr, tail_arr,
    sitk_img, vox_vol: float, pancreas_vol: float
) -> dict:
    """
    Identify connected lesion components and compute per-lesion statistics
    including overlaps with the total pancreas and anatomical sub-regions.
    """
    if lesion_arr is None or lesion_arr.sum() == 0:
        return {
            "n_lesions":        0,
            "total_volume_mm3": 0.0,
            "per_lesion":       [],
        }

    labeled_arr, n_lesions = ndimage.label(lesion_arr > 0)
    total_vol = _r(float((lesion_arr > 0).sum()) * vox_vol, 4)

    # Build extended pancreas reference: union of whole-pancreas + sub-regions.
    # Sub-region masks (especially head) can extend slightly beyond the whole-
    # pancreas mask due to annotation inconsistency.  Using the union avoids
    # falsely reporting 0% overlap for lesions that sit in that spill-over zone.
    _avail = [a for a in [pancreas_arr, head_arr, body_arr, tail_arr] if a is not None]
    if _avail:
        pancreas_ref = np.zeros_like(_avail[0], dtype=bool)
        for _a in _avail:
            pancreas_ref |= (_a > 0)
    else:
        pancreas_ref = pancreas_arr

    per_lesion = []
    for lid in range(1, n_lesions + 1):
        component = (labeled_arr == lid).astype(np.uint8)
        s = compute_mask_stats(component, sitk_img, vox_vol)
        l_vol = s["volume_mm3"] or 0.0

        # Overlap with full pancreas (using union of whole-pancreas + sub-regions)
        ov_pan  = overlap_mm3_fn(component, pancreas_ref, vox_vol)

        # Overlap with anatomical sub-regions
        ov_head = overlap_mm3_fn(component, head_arr, vox_vol)
        ov_body = overlap_mm3_fn(component, body_arr, vox_vol)
        ov_tail = overlap_mm3_fn(component, tail_arr, vox_vol)

        # Dominant region (largest overlap; None if no overlap at all)
        region_overlaps_raw = [("head", ov_head), ("body", ov_body), ("tail", ov_tail)]
        dominant = max(region_overlaps_raw, key=lambda t: t[1])[0]
        if all(v == 0.0 for _, v in region_overlaps_raw):
            dominant = None

        per_lesion.append({
            "lesion_id":                 lid,
            "volume_mm3":                s["volume_mm3"],
            "centroid_voxel_zyx":        s["centroid_voxel_zyx"],
            "centroid_world_xyz_mm":     s["centroid_world_xyz_mm"],
            "overlap_with_pancreas_mm3": ov_pan,
            "overlap_rate_vs_lesion":    safe_rate(ov_pan, l_vol),
            "overlap_rate_vs_pancreas":  safe_rate(ov_pan, pancreas_vol),
            "overlap_with_regions": {
                "head": {
                    "volume_mm3":     ov_head,
                    "rate_vs_lesion": safe_rate(ov_head, l_vol),
                },
                "body": {
                    "volume_mm3":     ov_body,
                    "rate_vs_lesion": safe_rate(ov_body, l_vol),
                },
                "tail": {
                    "volume_mm3":     ov_tail,
                    "rate_vs_lesion": safe_rate(ov_tail, l_vol),
                },
            },
            "dominant_region": dominant,
        })

    return {
        "n_lesions":        n_lesions,
        "total_volume_mm3": total_vol,
        "per_lesion":       per_lesion,
    }


# ===========================================================================
# PER-CASE PROCESSING
# ===========================================================================

def process_case(case_id: str, meta_dict: dict,
                 task_mode: str = "pancreas_lesion_subregions") -> dict:
    """
    Build the full statistics dictionary for a single PanTS case.
    Always returns a dict with the full schema.  Failed cases receive
    null field values and a ``case_status`` block describing the error.

    task_mode controls which segments are loaded and which fields are computed:
      "pancreas_only"              — pancreas only; lesion + sub-regions skipped
      "pancreas_lesion"            — pancreas + lesion; sub-regions skipped
      "pancreas_lesion_subregions" — full computation (default)
    """
    seg_dir    = _LABELS_DIR() / case_id / "segmentations"
    image_path = _IMAGES_DIR() / f"{case_id}{_IMAGE_SUFFIX()}"

    # ── Load CT image — distinguish file-missing vs read error ───────────────
    if not image_path.exists():
        print(f"  [OMIT] {case_id}: CT image file not found ({image_path})")
        return _null_case_schema(
            case_id, meta_dict,
            _case_status_error(
                error_type      = "file_not_found",
                omit_category   = "blocking_geometry_error",
                omit_reason     = "ct_image_missing",
                error_stage     = "read_ct_image",
                error_message   = f"CT image not found: {image_path}",
                blocking_issues = ["ct_image_file_not_found"],
            ),
        )

    try:
        sitk_img = sitk.ReadImage(str(image_path))
    except Exception as _exc:
        exc_str = str(_exc)
        _low = exc_str.lower()
        if "orthonormal" in _low or "no orthonormal definition" in _low:
            _etype   = "non_orthonormal_direction_cosines"
            _oreason = "invalid_ct_geometry"
            _issues  = ["non_orthonormal_direction_cosines",
                        "ct_image_unreadable_by_simpleitk"]
        else:
            _etype   = "image_read_error"
            _oreason = "ct_image_unreadable"
            _issues  = ["ct_image_unreadable_by_simpleitk"]
        print(f"  [OMIT] {case_id}: {exc_str[:100]}")
        return _null_case_schema(
            case_id, meta_dict,
            _case_status_error(
                error_type      = _etype,
                omit_category   = "blocking_geometry_error",
                omit_reason     = _oreason,
                error_stage     = "read_ct_image",
                error_message   = exc_str[:500],
                blocking_issues = _issues,
            ),
        )

    spacing_xyz = get_spacing_xyz(sitk_img)   # (sx, sy, sz) in mm
    img_arr     = sitk_to_uint8(sitk_img)
    image_data  = sitk_to_float32(sitk_img)   # float32 for HU statistics
    img_shape   = img_arr.shape               # (nz, ny, nx)
    vox_vol     = voxel_volume_mm3(spacing_xyz)

    # ── Helper: load a segmentation mask safely ──────────────────────────────
    def load_mask(seg_name: str):
        p = seg_dir / seg_name
        img = load_sitk_image(p)
        return (sitk_to_uint8(img) if img is not None else None), str(p)

    # Pancreas and lesion: keep SimpleITK objects for geometry checks BEFORE
    # converting to numpy.  Head/body/tail go through the normal helper path.
    _pancreas_path = seg_dir / SEG_PANCREAS
    pancreas_sitk  = load_sitk_image(_pancreas_path)
    pancreas_path  = str(_pancreas_path)

    _lesion_path = seg_dir / SEG_LESION
    lesion_path  = str(_lesion_path)
    if should_compute_lesion(task_mode):
        lesion_sitk = load_sitk_image(_lesion_path)
    else:
        lesion_sitk = None

    # ── Geometry checks (before numpy conversion) ─────────────────────────────
    ref_geom = extract_sitk_geometry(sitk_img)

    def _mask_geom_entry(mask_sitk, path_str: str) -> dict:
        if mask_sitk is None:
            return {"exists": False, "path": path_str}
        geom = extract_sitk_geometry(mask_sitk)
        comp = compare_to_reference_geometry(sitk_img, mask_sitk)
        return {"exists": True, "path": path_str, **geom, **comp}

    pan_geom    = _mask_geom_entry(pancreas_sitk, pancreas_path)

    # Convert pancreas and lesion to numpy now that geometry checks are done
    pancreas_arr = sitk_to_uint8(pancreas_sitk) if pancreas_sitk is not None else None
    separate_lesion_arr = sitk_to_uint8(lesion_sitk) if lesion_sitk is not None else None

    lesion_annotation_evidence = build_lesion_annotation_evidence(
        separate_path=_lesion_path,
        separate_sitk=lesion_sitk,
        separate_arr=separate_lesion_arr,
        ref_sitk=sitk_img,
        task_mode=task_mode,
    )
    if should_compute_lesion(task_mode):
        lesion_arr = separate_lesion_arr
        lesion_geom = _mask_geom_entry(lesion_sitk, lesion_path)
        lesion_geom["source"] = "separate_lesion_mask"
    else:
        lesion_arr = None
        lesion_geom = {"exists": False, "path": lesion_path,
                       "reason": f"not_applicable: TASK_MODE={task_mode}"}

    pan_mismatch    = pan_geom["exists"] and not pan_geom.get("affine_consistency_with_image", True)
    lesion_mismatch = lesion_geom["exists"] and not lesion_geom.get("affine_consistency_with_image", True)

    geometry = {
        "reference_image": ref_geom,
        "pancreas_mask":   pan_geom,
        "lesion_mask":     lesion_geom,
        "summary": {
            "pancreas_affine_mismatch": pan_mismatch,
            "lesion_affine_mismatch":   lesion_mismatch,
            "any_affine_mismatch":      pan_mismatch or lesion_mismatch,
        },
    }

    if should_compute_subregions(task_mode):
        head_arr, _ = load_mask(SEG_HEAD)
        body_arr, _ = load_mask(SEG_BODY)
        tail_arr, _ = load_mask(SEG_TAIL)
    else:
        head_arr = body_arr = tail_arr = None

    # ── HU statistics ─────────────────────────────────────────────────────────
    # Delegates to compute_hu_statistics() for robust, auditable HU analysis.
    tumor_mask     = (lesion_arr > 0) if lesion_arr is not None else None
    _lesion_present = tumor_mask is not None and bool(np.any(tumor_mask))
    _ct_phase      = meta_dict.get(case_id, {}).get("ct_phase")   # e.g. "Venous", "Arterial"
    hu_statistics = compute_hu_statistics(
        image_data, tumor_mask, pancreas_arr, _ct_phase,
        lesion_present=_lesion_present,
    )

    def is_empty(arr) -> bool:
        return arr is None or int(arr.sum()) == 0

    # ── Shape & spacing consistency ──────────────────────────────────────────

    # ── 1. Image information ─────────────────────────────────────────────────
    image_info = {
        "shape_zyx":        list(img_shape),
        "spacing_xyz_mm":   [_r(float(s), 6) for s in spacing_xyz],
        "voxel_volume_mm3": _r(vox_vol, 6),
    }

    # ── 2. Metadata ───────────────────────────────────────────────────────────
    metadata = dict(meta_dict.get(case_id, {}))
    for key in METADATA_COLUMNS.values():
        metadata.setdefault(key, None)

    # ── 3. Pancreas ───────────────────────────────────────────────────────────
    pancreas_stats = compute_mask_stats(pancreas_arr, sitk_img, vox_vol)
    pancreas_vol   = pancreas_stats["volume_mm3"] or 0.0

    # ── 4. Pancreas anatomical regions ───────────────────────────────────────
    def region_entry(arr) -> dict:
        s   = compute_mask_stats(arr, sitk_img, vox_vol)
        ov  = overlap_mm3_fn(arr, pancreas_arr, vox_vol)
        s["overlap_with_pancreas_mm3"] = ov
        s["rate_vs_pancreas"]          = safe_rate(ov, pancreas_vol)
        return s

    if should_compute_subregions(task_mode):
        regions = {
            "head": region_entry(head_arr),
            "body": region_entry(body_arr),
            "tail": region_entry(tail_arr),
        }

        # ── 5. Region consistency ─────────────────────────────────────────────
        head_vol   = regions["head"]["volume_mm3"] or 0.0
        body_vol   = regions["body"]["volume_mm3"] or 0.0
        tail_vol   = regions["tail"]["volume_mm3"] or 0.0
        region_sum = head_vol + body_vol + tail_vol

        ov_hb = overlap_mm3_fn(head_arr, body_arr, vox_vol)
        ov_ht = overlap_mm3_fn(head_arr, tail_arr, vox_vol)
        ov_bt = overlap_mm3_fn(body_arr, tail_arr, vox_vol)

        region_consistency = {
            "sum_regions_mm3":            _r(region_sum, 4),
            "relative_error_vs_pancreas": safe_rate(abs(region_sum - pancreas_vol), pancreas_vol),
            "head_body_overlap_mm3":      ov_hb,
            "head_tail_overlap_mm3":      ov_ht,
            "body_tail_overlap_mm3":      ov_bt,
        }

        _subregions_found = any(
            (regions[rn]["volume_mm3"] or 0.0) > 0
            for rn in ("head", "body", "tail")
        )
        if _subregions_found:
            subregion_qc = {
                "enabled":              True,
                "status":               "present",
                "region_coverage_ratio": safe_rate(region_sum, pancreas_vol),
                "region_relative_error": region_consistency["relative_error_vs_pancreas"],
            }
        else:
            subregion_qc = {
                "enabled": True,
                "status":  "warning",
                "warning": "TASK_MODE requires subregions, but no head/body/tail "
                           "annotations were found.",
                "region_coverage_ratio": None,
                "region_relative_error": None,
            }
    else:
        regions            = {"head": None, "body": None, "tail": None}
        region_consistency = None
        subregion_qc       = {
            "enabled": False,
            "reason":  f"TASK_MODE={task_mode}",
            "status":  "not_applicable",
        }

    # ── 6. Lesions ────────────────────────────────────────────────────────────
    if should_compute_lesion(task_mode):
        lesions_info = compute_lesion_info(
            lesion_arr, pancreas_arr, head_arr, body_arr, tail_arr,
            sitk_img, vox_vol, pancreas_vol,
        )
    else:
        lesions_info = {
            "n_lesions":        None,
            "total_volume_mm3": None,
            "per_lesion":       [],
            "task_mode_note":   f"not_applicable: TASK_MODE={task_mode}",
        }

    # ── 7. Quality control ────────────────────────────────────────────────────
    total_lesion_vol = lesions_info["total_volume_mm3"] or 0.0

    # ── Build 4 pancreas support masks ─────────────────────────────────────────
    # Mask 1: total pancreas only
    _pan_total = (pancreas_arr > 0) if pancreas_arr is not None else None

    # Mask 2: sub-regions union (head | body | tail)
    _reg_avail = [a for a in [head_arr, body_arr, tail_arr] if a is not None]
    _regions_union_available = len(_reg_avail) > 0
    if _reg_avail:
        _reg_union: np.ndarray | None = np.zeros_like(_reg_avail[0], dtype=bool)
        for _a in _reg_avail:
            _reg_union |= (_a > 0)
    else:
        _reg_union = None

    # Mask 3: combined = pancreas_total | regions_union  (backward-compatible)
    _all_avail = [a for a in [pancreas_arr, head_arr, body_arr, tail_arr] if a is not None]
    if _all_avail:
        _combined: np.ndarray | None = np.zeros_like(_all_avail[0], dtype=bool)
        for _a in _all_avail:
            _combined |= (_a > 0)
    else:
        _combined = _pan_total

    # Mask 4: combined_filled = binary_fill_holes(combined) — QC analysis only.
    # Fills fully-enclosed internal holes; handles the "lesion-as-hole" convention
    # where the pancreas mask explicitly excludes lesion voxels.
    # binary_fill_holes does NOT inflate external surface concavities.
    _combined_filled: np.ndarray | None = (
        ndimage.binary_fill_holes(_combined) if _combined is not None else None
    )

    # ── Compute overlaps against each support mask ──────────────────────────────
    ov_total    = overlap_mm3_fn(lesion_arr, _pan_total,       vox_vol)
    ov_regions  = overlap_mm3_fn(lesion_arr, _reg_union,       vox_vol)
    ov_combined = overlap_mm3_fn(lesion_arr, _combined,        vox_vol)
    ov_filled   = overlap_mm3_fn(lesion_arr, _combined_filled, vox_vol)

    # Backward-compatible aggregate uses combined (same as previous behaviour).
    ov_lesion_pan = ov_combined

    # ── Diagnostic flags (informational — >10 pp threshold) ────────────────────
    _rate_total    = safe_rate(ov_total,    total_lesion_vol) or 0.0
    _rate_combined = safe_rate(ov_combined, total_lesion_vol) or 0.0
    _rate_filled   = safe_rate(ov_filled,   total_lesion_vol) or 0.0
    _rate_regions  = safe_rate(ov_regions,  total_lesion_vol)

    _hole_filling_changed  = bool(
        _combined is not None
        and _combined_filled is not None
        and (_rate_filled - _rate_combined) > 0.10
    )
    _regions_added_overlap = bool(
        _reg_union is not None
        and _pan_total is not None
        and (_rate_combined - _rate_total) > 0.10
    )

    qc = {
        "pancreas_mask_empty":               bool(is_empty(pancreas_arr)),
        "lesion_mask_empty":                 bool(is_empty(lesion_arr)),
        "lesion_pancreas_overlap_exists":    ov_lesion_pan > 0,
        "total_overlap_lesion_pancreas_mm3": _r(ov_lesion_pan),
        "total_overlap_rate_vs_lesion":      safe_rate(ov_lesion_pan, total_lesion_vol),
        "total_overlap_rate_vs_pancreas":    safe_rate(ov_lesion_pan, pancreas_vol),
        "pancreas_mask_affine_mismatch":     pan_mismatch,
        "lesion_mask_affine_mismatch":       lesion_mismatch,
        "any_affine_mismatch":               pan_mismatch or lesion_mismatch,
        "lesion_annotation_evidence":        lesion_annotation_evidence,
        "lesion_overlap_analysis": {
            "overlap_reference_used_for_qc": (
                "regions_union"
                if should_compute_subregions(task_mode) and _regions_union_available
                else "combined_filled_pancreas"
            ),
            "pancreas_total": {
                "overlap_mm3":            _r(ov_total),
                "overlap_rate_vs_lesion": safe_rate(ov_total, total_lesion_vol),
            },
            "regions_union": {
                "available":              _regions_union_available,
                "overlap_mm3":            _r(ov_regions),
                "overlap_rate_vs_lesion": _rate_regions,
            },
            "combined_pancreas": {
                "overlap_mm3":            _r(ov_combined),
                "overlap_rate_vs_lesion": safe_rate(ov_combined, total_lesion_vol),
            },
            "combined_filled_pancreas": {
                "overlap_mm3":            _r(ov_filled),
                "overlap_rate_vs_lesion": safe_rate(ov_filled, total_lesion_vol),
            },
            "hole_filling_changed_overlap": _hole_filling_changed,
            "regions_added_overlap":        _regions_added_overlap,
        },
    }

    # ── Override lesion-related QC fields to None when task_mode skips lesions ─
    if not should_compute_lesion(task_mode):
        qc["lesion_mask_empty"]                = None
        qc["lesion_pancreas_overlap_exists"]   = None
        qc["total_overlap_lesion_pancreas_mm3"]= None
        qc["total_overlap_rate_vs_lesion"]     = None
        qc["total_overlap_rate_vs_pancreas"]   = None
        qc["lesion_mask_affine_mismatch"]      = None
        qc["any_affine_mismatch"]              = pan_mismatch
        qc["lesion_annotation_evidence"]       = None
        qc["lesion_overlap_analysis"]          = None

    return {
        "case_id":       case_id,
        "image":         image_info,
        "metadata":      metadata,
        "geometry":      geometry,
        "pancreas": {
            **pancreas_stats,
            "regions":            regions,
            "region_consistency": region_consistency,
        },
        "lesions":         lesions_info,
        "quality_control": qc,
        "hu_statistics":   hu_statistics,
        "subregion_qc":    subregion_qc,
        "case_status":     _case_status_success(),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    global _PATHS, ROOT, METADATA_PATH, OUTPUT_JSON, IMAGES_DIR, IMAGE_SUFFIX

    parser = argparse.ArgumentParser(
        description="Dataset Summarizer — generates <dataset_name>_summary.json")
    parser.add_argument(
        "--paths-yaml", default=None,
        help="Path to paths.yaml (default: configs/paths.yaml relative to project root)")
    parser.add_argument(
        "--thresholds-yaml", default=None,
        help="Path to thresholds.yaml (default: configs/thresholds.yaml). "
             "Controls TASK_MODE.")
    parser.add_argument(
        "--task-mode", default=None, dest="task_mode",
        choices=sorted(VALID_TASK_MODES),
        help="Override TASK_MODE (highest priority; overrides thresholds.yaml). "
             "Choices: pancreas_only | pancreas_lesion | pancreas_lesion_subregions.")
    parser.add_argument(
        "--output-dir", default=None, dest="output_dir",
        help="Dataset-level output root (default: outputs/<dataset_name>/ from "
             "paths.yaml). Task-specific sub-folders are created automatically: "
             "<output_dir>/<task_mode>/summary/ and <output_dir>/<task_mode>/qc/.")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing output files. "
             "Aborts with an error if the output already exists and this flag is absent.")
    parser.add_argument(
        "--workers", type=int, default=1, metavar="N", dest="workers",
        help="Number of parallel worker processes (default: 1 = serial). "
             "Workers use multiprocessing.Pool with fork on Linux. "
             "Try N = os.cpu_count() // 2 as a safe starting point. "
             "Progress output is in completion order, not case order, when N > 1.")
    args = parser.parse_args()

    # Load centralized path config
    _PATHS = _load_config(Path(args.paths_yaml) if args.paths_yaml else None)

    # Set backward-compat module globals so helper functions still work
    ROOT          = _PATHS.raw_root
    METADATA_PATH = _PATHS.raw_metadata
    OUTPUT_JSON   = _PATHS.summary_json
    IMAGES_DIR    = _PATHS.raw_images_dir
    IMAGE_SUFFIX  = _PATHS.image_suffix

    # Set up file logging
    _dataset = _PATHS.dataset_name or "dataset"
    log_path = _PATHS.logs_dir / f"summarize_{_dataset}.log"
    lg = setup_file_logging(log_path, logger_name=f"summarize_{_dataset}")
    lg.info("Starting dataset summarizer for: %s", _dataset)
    lg.info("Raw dataset root : %s", ROOT)
    lg.info("Output JSON      : %s", OUTPUT_JSON)

    print("=" * 60)
    print(f"  Dataset Summarizer  ({_dataset})")
    print("=" * 60)
    print(f"  Raw dataset : {ROOT}")
    print(f"  Output      : {OUTPUT_JSON}")

    # Load thresholds and resolve TASK_MODE
    _thr_path    = Path(args.thresholds_yaml) if args.thresholds_yaml else None
    _thresholds  = _load_thresholds(_thr_path)
    _task_mode   = get_task_mode(_thresholds)
    if args.task_mode:   # CLI override wins over thresholds.yaml
        _task_mode = args.task_mode
    if _task_mode not in VALID_TASK_MODES:
        raise ValueError(
            f"Unsupported TASK_MODE: '{_task_mode}'. "
            f"Valid values: {sorted(VALID_TASK_MODES)}"
        )
    lg.info("TASK_MODE: %s", _task_mode)
    print(f"  Task mode   : {_task_mode}")

    # Build task-specific output directories
    _base_output_dir = (
        Path(args.output_dir) if args.output_dir
        else _PATHS.summary_dir.parent   # outputs/<dataset_name>/
    )
    _task_dirs  = build_output_dirs(_base_output_dir, _task_mode)
    OUTPUT_JSON = _task_dirs["summary_dir"] / f"{_dataset}_summary.json"
    lg.info("Output (task-specific): %s", OUTPUT_JSON)
    print(f"  Output      : {OUTPUT_JSON}")

    # Overwrite guard — check before the expensive per-case loop
    if OUTPUT_JSON.exists() and not args.overwrite:
        raise SystemExit(
            f"[ERROR] Output already exists: {OUTPUT_JSON}\n"
            "        Use --overwrite to replace it."
        )

    # Load metadata lookup table
    print(f"\nLoading metadata: {METADATA_PATH}")
    meta_dict = load_metadata(METADATA_PATH)
    print(f"  Loaded {len(meta_dict)} metadata entries.")

    # Discover training cases from images and match labels by case_id. This is
    # dataset-agnostic and avoids PantsMini/PanTS naming assumptions.
    image_paths = sorted(IMAGES_DIR.glob(f"*{IMAGE_SUFFIX}"))
    case_ids = [case_id_from_image(p, IMAGE_SUFFIX) for p in image_paths]
    print(f"\nFound {len(case_ids)} image cases in {IMAGES_DIR}")

    summary: dict = {}
    n_processed = 0
    n_omitted   = 0
    _n_total    = len(case_ids)

    def _record(i: int, case_id: str, result: dict) -> None:
        """Store one case result and update counters — shared by both execution paths."""
        nonlocal n_processed, n_omitted
        if i == 0 or (i + 1) % 100 == 0:
            print(f"  [{i+1:4d}/{_n_total}] {case_id}")
        summary[case_id] = result
        if result.get("case_status", {}).get("processing_successful", True):
            n_processed += 1
        else:
            n_omitted += 1

    if args.workers > 1:
        _n_workers = min(args.workers, _n_total)
        print(f"  Parallel mode : {_n_workers} worker process(es)")
        lg.info("Parallel mode: %d workers", _n_workers)
        _tasks = [(case_id, _task_mode) for case_id in case_ids]
        with multiprocessing.Pool(
            processes=_n_workers,
            initializer=_init_worker,
            initargs=(_PATHS, meta_dict),
        ) as _pool:
            for i, (case_id, result) in enumerate(
                _pool.imap(_parallel_worker, _tasks, chunksize=4)
            ):
                _record(i, case_id, result)
    else:
        for i, case_id in enumerate(case_ids):
            _record(i, case_id,
                    process_case(case_id, meta_dict, task_mode=_task_mode))

    # Serialise — use a custom encoder to handle any residual numpy scalars
    class _Encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return None if np.isnan(obj) else float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    _file_metadata = {
        "dataset_name":   _dataset,
        "task_mode":      _task_mode,
        "output_dir":     str(OUTPUT_JSON.parent),
        "created_at":     datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "schema_version": "4",
        "n_cases":        n_processed + n_omitted,
        "n_processed":    n_processed,
        "n_omitted":      n_omitted,
    }

    print(f"\nWriting {OUTPUT_JSON} …")
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as fh:
        json.dump({"metadata": _file_metadata, "cases": summary}, fh, indent=2, cls=_Encoder)

    lg.info("Wrote %d processed + %d omitted cases to %s",
            n_processed, n_omitted, OUTPUT_JSON)
    print("\n" + "=" * 60)
    print("  DONE")
    print(f"  Processed : {n_processed}")
    print(f"  Omitted   : {n_omitted}  (case_status.omit_from_training=true)")
    print(f"  Total     : {n_processed + n_omitted}")
    print(f"  Output    : {OUTPUT_JSON}")
    print("=" * 60)


if __name__ == "__main__":
    main()
