#!/usr/bin/env python3
"""
Config Validator
================
Validates configs/thresholds.yaml and configs/paths.yaml before running the
QC pipeline. Checks for missing required keys, wrong types, out-of-range
values, and broken file paths.

All errors are collected and printed together; the script never fails fast on
the first error. Exit code: 0 = clean, 1 = one or more errors found.

Usage:
    python scripts/validate_config.py
    python scripts/validate_config.py --paths-yaml configs/paths.yaml \\
                                      --thresholds-yaml configs/thresholds.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Bootstrap: make agents/ importable when running this script directly
# ---------------------------------------------------------------------------
_SCRIPTS_DIR  = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    import yaml as _yaml
    _HAS_YAML = True
except ModuleNotFoundError:
    _HAS_YAML = False

# ── Valid task modes (single source of truth from paths.py) ─────────────────
try:
    from agents.utils.paths import VALID_TASK_MODES
except ImportError:
    VALID_TASK_MODES = frozenset({
        "pancreas_only", "pancreas_lesion", "pancreas_lesion_subregions",
    })

_VALID_TASK_MODES_EXTENDED = VALID_TASK_MODES | {"auto"}
_TASK_PROFILE_REQUIRED_KEYS = (
    "TASK_MODE",
    "REQUIRED_COMPONENTS",
    "REQUIRED_SEGMENTATIONS",
    "ACTIVE_QC_DOMAINS",
    "HARD_FAILURE_DOMAINS",
    "CALIBRATABLE_DOMAINS",
)

# ── Required domain keys for QC_PROFILE_THRESHOLDS (5-level severity) ────────
_PROFILE_DOMAINS = {
    "geometry_integrity",
    "lesion_localization",
    "lesion_burden",
    "pancreas_context",
    "fov_integrity",
    "region_consistency",
    "attenuation_integrity",
    "metadata_completeness",
}

_PROFILE_SEVERITY_LEVELS = ("low_warning", "moderate_warning", "high_warning", "critical")
_ALL_SEVERITY_LEVELS = ("not_applicable", "normal", "low_warning", "moderate_warning", "high_warning", "critical")

# ===========================================================================
# VALIDATION HELPERS
# ===========================================================================

class _Errors:
    """Collect validation errors by section."""

    def __init__(self) -> None:
        self._items: list[tuple[str, str]] = []  # (section, message)

    def add(self, section: str, msg: str) -> None:
        self._items.append((section, msg))

    def __bool__(self) -> bool:
        return len(self._items) > 0

    def __len__(self) -> int:
        return len(self._items)

    def print_report(self) -> None:
        from collections import defaultdict
        by_section: dict[str, list[str]] = defaultdict(list)
        for sec, msg in self._items:
            by_section[sec].append(msg)
        for sec, msgs in by_section.items():
            print(f"\n  [{sec}]")
            for m in msgs:
                print(f"    \u2717 {m}")


def _load_yaml(path: Path, errors: _Errors, label: str) -> dict | None:
    if not path.exists():
        errors.add(label, f"File not found: {path}")
        return None
    if not _HAS_YAML:
        errors.add(label, "PyYAML is not installed — cannot parse YAML files. "
                           "Install with: pip install pyyaml")
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            data = _yaml.safe_load(fh)
        if data is None:
            errors.add(label, f"File is empty or contains only whitespace: {path}")
            return {}
        if not isinstance(data, dict):
            errors.add(label, f"Top-level value must be a mapping (dict), got {type(data).__name__}")
            return None
        return data
    except Exception as exc:
        errors.add(label, f"Cannot parse YAML: {exc}")
        return None


def _check_type(data: dict, key: str, expected_type, label: str,
                errors: _Errors, required: bool = True) -> bool:
    """Check that a key exists and has the expected type. Returns True if OK."""
    if key not in data:
        if required:
            errors.add(label, f"Missing required key: {key!r}")
        return False
    val = data[key]
    if val is None:
        if required:
            errors.add(label, f"{key!r} must not be null")
        return False
    if not isinstance(val, expected_type):
        errors.add(label,
                   f"{key!r}: expected {expected_type.__name__}, "
                   f"got {type(val).__name__} ({val!r})")
        return False
    return True


def _check_range(data: dict, key: str, lo, hi, label: str, errors: _Errors,
                 lo_inclusive: bool = True, hi_inclusive: bool = True) -> None:
    """Check that data[key] is within [lo, hi] (or open variants)."""
    if key not in data or data[key] is None:
        return
    val = data[key]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return  # type error already caught elsewhere
    ok_lo = (v >= lo) if lo_inclusive else (v > lo)
    ok_hi = (v <= hi) if hi_inclusive else (v < hi)
    if not (ok_lo and ok_hi):
        lo_sym = "[" if lo_inclusive else "("
        hi_sym = "]" if hi_inclusive else ")"
        errors.add(label,
                   f"{key!r} = {v} is outside valid range {lo_sym}{lo}, {hi}{hi_sym}")


def _check_numeric(data: dict, key: str, label: str, errors: _Errors,
                   required: bool = True) -> bool:
    """Check that data[key] is a number (int or float). Returns True if OK."""
    if key not in data:
        if required:
            errors.add(label, f"Missing required key: {key!r}")
        return False
    val = data[key]
    if not isinstance(val, (int, float)):
        errors.add(label,
                   f"{key!r}: expected a number, got {type(val).__name__} ({val!r})")
        return False
    return True


# ===========================================================================
# THRESHOLDS.YAML VALIDATION
# ===========================================================================

def validate_thresholds(path: Path, errors: _Errors) -> None:
    label = "thresholds.yaml"
    data = _load_yaml(path, errors, label)
    if data is None:
        return

    # ── THRESHOLD_METADATA ─────────────────────────────────────────────────
    if "THRESHOLD_METADATA" not in data:
        errors.add(label, "Missing required key: 'THRESHOLD_METADATA'")
    elif not isinstance(data["THRESHOLD_METADATA"], dict):
        errors.add(label,
                   f"'THRESHOLD_METADATA' must be a mapping, got {type(data['THRESHOLD_METADATA']).__name__}")
    else:
        tm = data["THRESHOLD_METADATA"]
        method = tm.get("method")
        if method not in ("deterministic", "calibrated"):
            errors.add(label,
                       "THRESHOLD_METADATA.method must be 'deterministic' or 'calibrated'")
        if not isinstance(tm.get("source"), str):
            errors.add(label, "THRESHOLD_METADATA.source must be a string")
        if method == "calibrated":
            if not tm.get("calibrated_from_summary"):
                errors.add(label, "Calibrated thresholds require THRESHOLD_METADATA.calibrated_from_summary")
            for key in ("reference_clean_cases", "total_cases"):
                if not isinstance(tm.get(key), int) or int(tm.get(key)) < 0:
                    errors.add(label, f"Calibrated thresholds require non-negative integer THRESHOLD_METADATA.{key}")

    # ── TASK_MODE ─────────────────────────────────────────────────────────
    if _check_type(data, "TASK_MODE", str, label, errors):
        mode = str(data["TASK_MODE"]).strip().lower()
        if mode not in _VALID_TASK_MODES_EXTENDED:
            errors.add(label,
                       f"TASK_MODE {mode!r} is not valid. "
                       f"Allowed: {sorted(_VALID_TASK_MODES_EXTENDED)}")

    # ── Overlap / region thresholds ────────────────────────────────────────
    for key in ("MIN_LESION_PANCREAS_OVERLAP", "MAX_LESION_PANCREAS_RATIO"):
        if _check_numeric(data, key, label, errors):
            _check_range(data, key, 0.0, 1.0, label, errors)

    if _check_numeric(data, "MAX_REGION_RELATIVE_ERROR", label, errors):
        _check_range(data, "MAX_REGION_RELATIVE_ERROR", 0.0, float("inf"),
                     label, errors, lo_inclusive=False)

    # ── Volume bounds ──────────────────────────────────────────────────────
    for key in ("MIN_LESION_VOLUME_MM3", "MAX_LESION_VOLUME_MM3",
                "MIN_PANCREAS_VOLUME_MM3", "MAX_PANCREAS_VOLUME_MM3"):
        if _check_numeric(data, key, label, errors):
            _check_range(data, key, 0.0, float("inf"), label, errors,
                         lo_inclusive=False)

    for mn_k, mx_k in (
        ("MIN_LESION_VOLUME_MM3",  "MAX_LESION_VOLUME_MM3"),
        ("MIN_PANCREAS_VOLUME_MM3", "MAX_PANCREAS_VOLUME_MM3"),
    ):
        if mn_k in data and mx_k in data:
            try:
                if float(data[mn_k]) >= float(data[mx_k]):
                    errors.add(label,
                               f"{mn_k} ({data[mn_k]}) must be < {mx_k} ({data[mx_k]})")
            except (TypeError, ValueError):
                pass

    # ── QC score gates ──────────────────────────────────────────────────────
    for key in ("MAX_QC_SCORE_FOR_KEEP", "MAX_QC_SCORE_FOR_REVIEW"):
        if _check_numeric(data, key, label, errors):
            _check_range(data, key, 0, 100, label, errors)
    if "MAX_QC_SCORE_FOR_KEEP" in data and "MAX_QC_SCORE_FOR_REVIEW" in data:
        try:
            if float(data["MAX_QC_SCORE_FOR_KEEP"]) >= float(data["MAX_QC_SCORE_FOR_REVIEW"]):
                errors.add(label,
                           f"MAX_QC_SCORE_FOR_KEEP ({data['MAX_QC_SCORE_FOR_KEEP']}) "
                           f"must be < MAX_QC_SCORE_FOR_REVIEW ({data['MAX_QC_SCORE_FOR_REVIEW']})")
        except (TypeError, ValueError):
            pass

    # ── Hybrid scoring params ──────────────────────────────────────────────
    for key in ("HYBRID_ALPHA", "HYBRID_BETA"):
        if _check_numeric(data, key, label, errors):
            _check_range(data, key, 0.0, float("inf"), label, errors, lo_inclusive=False)

    weight_keys = (
        "HYBRID_WEIGHT_LESION_BURDEN",
        "HYBRID_WEIGHT_PANCREAS_CONTEXT",
        "HYBRID_WEIGHT_REGION_CONSISTENCY",
        "HYBRID_WEIGHT_METADATA",
        "HYBRID_WEIGHT_ATTENUATION_CONSISTENCY",
    )
    for key in weight_keys:
        if _check_numeric(data, key, label, errors):
            _check_range(data, key, 0.0, float("inf"), label, errors)

    # ── SCORE_WEIGHTS ──────────────────────────────────────────────────────
    _SCORE_WEIGHT_KEYS = {
        "geometry_integrity", "lesion_localization", "lesion_burden",
        "pancreas_context", "fov_integrity", "region_consistency",
        "metadata_completeness", "attenuation_integrity",
    }
    if "SCORE_WEIGHTS" in data:
        sw = data["SCORE_WEIGHTS"]
        if not isinstance(sw, dict):
            errors.add(label,
                       f"'SCORE_WEIGHTS' must be a mapping, got {type(sw).__name__}")
        else:
            for mode in VALID_TASK_MODES:
                if mode not in sw:
                    errors.add(label,
                               f"SCORE_WEIGHTS is missing mode: {mode!r}")
                    continue
                mode_entry = sw[mode]
                if not isinstance(mode_entry, dict):
                    errors.add(label,
                               f"SCORE_WEIGHTS.{mode}: must be a mapping, "
                               f"got {type(mode_entry).__name__}")
                    continue
                for comp_key, w_val in mode_entry.items():
                    if comp_key not in _SCORE_WEIGHT_KEYS:
                        errors.add(label,
                                   f"SCORE_WEIGHTS.{mode}: unknown component {comp_key!r}")
                    elif not isinstance(w_val, (int, float)) or float(w_val) < 0:
                        errors.add(label,
                                   f"SCORE_WEIGHTS.{mode}.{comp_key}: "
                                   f"must be a non-negative number, got {w_val!r}")

    # ── RISK_SCORE_CEILING ─────────────────────────────────────────────────
    if "RISK_SCORE_CEILING" in data:
        rsc = data["RISK_SCORE_CEILING"]
        if not isinstance(rsc, dict):
            errors.add(label,
                       f"'RISK_SCORE_CEILING' must be a mapping, got {type(rsc).__name__}")
        else:
            for ceil_key in ("low", "medium"):
                if ceil_key not in rsc:
                    errors.add(label,
                               f"RISK_SCORE_CEILING missing required key: {ceil_key!r}")
                elif not isinstance(rsc[ceil_key], (int, float)):
                    errors.add(label,
                               f"RISK_SCORE_CEILING.{ceil_key}: expected a number, "
                               f"got {type(rsc[ceil_key]).__name__}")
                else:
                    _check_range(rsc, ceil_key, 0, 100, label, errors)
            if "low" in rsc and "medium" in rsc:
                try:
                    if float(rsc["low"]) >= float(rsc["medium"]):
                        errors.add(label,
                                   f"RISK_SCORE_CEILING: low ({rsc['low']}) "
                                   f"must be < medium ({rsc['medium']})")
                except (TypeError, ValueError):
                    pass

    # ── Domain policy overrides ────────────────────────────────────────────
    if "METADATA_POLICY" in data:
        mp = data["METADATA_POLICY"]
        if not isinstance(mp, dict):
            errors.add(label, f"'METADATA_POLICY' must be a mapping, got {type(mp).__name__}")
        else:
            max_sev = mp.get("max_severity")
            if max_sev not in _ALL_SEVERITY_LEVELS:
                errors.add(label, f"METADATA_POLICY.max_severity must be one of {_ALL_SEVERITY_LEVELS}, got {max_sev!r}")
            if not isinstance(mp.get("recommendation"), str):
                errors.add(label, "METADATA_POLICY.recommendation must be a string")
            if not isinstance(mp.get("excluded_from_risk_escalation"), bool):
                errors.add(label, "METADATA_POLICY.excluded_from_risk_escalation must be a boolean")

    if "FOV_POLICY" in data:
        fp = data["FOV_POLICY"]
        if not isinstance(fp, dict):
            errors.add(label, f"'FOV_POLICY' must be a mapping, got {type(fp).__name__}")
        else:
            for key in ("border_touching_severity", "missing_region_severity",
                        "incomplete_coverage_severity", "truncation_severity"):
                if fp.get(key) not in _ALL_SEVERITY_LEVELS:
                    errors.add(label, f"FOV_POLICY.{key} must be one of {_ALL_SEVERITY_LEVELS}, got {fp.get(key)!r}")

    if "FOV_SCORE_POINTS" in data:
        fsp = data["FOV_SCORE_POINTS"]
        if not isinstance(fsp, dict):
            errors.add(label, f"'FOV_SCORE_POINTS' must be a mapping, got {type(fsp).__name__}")
        else:
            for key in ("border_touching", "partial_visibility_likely", "anatomical_truncation_suspected"):
                if key not in fsp or not isinstance(fsp[key], (int, float)) or float(fsp[key]) < 0:
                    errors.add(label, f"FOV_SCORE_POINTS.{key} must be a non-negative number")

    if "REGION_SCORE_POINTS" in data:
        rsp = data["REGION_SCORE_POINTS"]
        if not isinstance(rsp, dict):
            errors.add(label, f"'REGION_SCORE_POINTS' must be a mapping, got {type(rsp).__name__}")
        else:
            for key in ("incomplete_anatomical_coverage", "subregions_expected_but_missing", "missing_subregion"):
                if key not in rsp or not isinstance(rsp[key], (int, float)) or float(rsp[key]) < 0:
                    errors.add(label, f"REGION_SCORE_POINTS.{key} must be a non-negative number")

    # ── QC_PROFILE_THRESHOLDS ──────────────────────────────────────────────
    if "QC_PROFILE_THRESHOLDS" not in data:
        errors.add(label, "Missing required key: 'QC_PROFILE_THRESHOLDS'")
    elif not isinstance(data["QC_PROFILE_THRESHOLDS"], dict):
        errors.add(label,
                   f"'QC_PROFILE_THRESHOLDS' must be a mapping, "
                   f"got {type(data['QC_PROFILE_THRESHOLDS']).__name__}")
    else:
        qpt = data["QC_PROFILE_THRESHOLDS"]
        missing_pd = _PROFILE_DOMAINS - set(qpt.keys())
        if missing_pd:
            errors.add(label,
                       f"QC_PROFILE_THRESHOLDS is missing domains: "
                       f"{sorted(missing_pd)}")
        for domain, thrs in qpt.items():
            if domain not in _PROFILE_DOMAINS:
                continue  # unknown domain: skip
            if not isinstance(thrs, dict):
                errors.add(label,
                           f"QC_PROFILE_THRESHOLDS.{domain}: must be a mapping, "
                           f"got {type(thrs).__name__}")
                continue
            # All 4 level keys must be present
            for lvl in _PROFILE_SEVERITY_LEVELS:
                if lvl not in thrs:
                    errors.add(label,
                               f"QC_PROFILE_THRESHOLDS.{domain}: missing '{lvl}' key")
                elif not isinstance(thrs[lvl], (int, float)):
                    errors.add(label,
                               f"QC_PROFILE_THRESHOLDS.{domain}.{lvl}: "
                               f"expected a number, got {type(thrs[lvl]).__name__}")
            # Monotonicity: low < moderate < high < critical
            levels_present = [
                (lvl, thrs[lvl]) for lvl in _PROFILE_SEVERITY_LEVELS
                if lvl in thrs and isinstance(thrs[lvl], (int, float))
            ]
            for (la, va), (lb, vb) in zip(levels_present, levels_present[1:]):
                try:
                    if float(va) >= float(vb):
                        errors.add(label,
                                   f"QC_PROFILE_THRESHOLDS.{domain}: "
                                   f"{la} ({va}) must be < {lb} ({vb})")
                except (TypeError, ValueError):
                    pass


# ===========================================================================
# PATHS.YAML VALIDATION
# ===========================================================================

def validate_paths(path: Path, errors: _Errors) -> None:
    label = "paths.yaml"
    data = _load_yaml(path, errors, label)
    if data is None:
        return

    # ── Required string keys ───────────────────────────────────────────────
    string_keys = (
        "PROJECT_ROOT", "OUTPUT_ROOT",
        "RAW_IMAGES_SUBDIR", "RAW_LABELS_SUBDIR",
        "RAW_METADATA_FILE", "IMAGE_SUFFIX",
        "OUTPUT_QC_SUBDIR", "OUTPUT_SUMMARY_SUBDIR",
        "LOGS_DIR",
        "THRESHOLDS_CONFIG",
        "DEFAULT_THRESHOLD_METHOD",
    )
    for key in string_keys:
        _check_type(data, key, str, label, errors)

    # ── PROJECT_ROOT must exist on disk ────────────────────────────────────
    if "PROJECT_ROOT" in data and isinstance(data["PROJECT_ROOT"], str):
        pr = Path(data["PROJECT_ROOT"])
        if not pr.exists():
            errors.add(label, f"PROJECT_ROOT does not exist: {pr}")
        elif not pr.is_dir():
            errors.add(label, f"PROJECT_ROOT is not a directory: {pr}")
        else:
            # ── DATASET_CONFIG (relative to PROJECT_ROOT) ──────────────────
            if "DATASET_CONFIG" in data:
                dc = data["DATASET_CONFIG"]
                if isinstance(dc, str) and dc.strip():
                    dc_path = pr / dc
                    if not dc_path.exists():
                        errors.add(label,
                                   f"DATASET_CONFIG not found: {dc_path}\n"
                                   f"    (relative to PROJECT_ROOT={pr})")
                # empty string is allowed — means legacy manual mode

            # ── THRESHOLDS_CONFIG ──────────────────────────────────────────
            if "THRESHOLDS_CONFIG" in data and isinstance(data["THRESHOLDS_CONFIG"], str):
                tc_path = pr / data["THRESHOLDS_CONFIG"]
                if not tc_path.exists():
                    errors.add(label,
                               f"THRESHOLDS_CONFIG not found: {tc_path}")

            default_method = str(data.get("DEFAULT_THRESHOLD_METHOD", "deterministic")).strip().lower()
            if "DEFAULT_THRESHOLD_METHOD" in data and default_method not in {"deterministic", "calibrated"}:
                errors.add(label,
                           "DEFAULT_THRESHOLD_METHOD must be 'deterministic' or 'calibrated'")

            # ── Task profiles ──────────────────────────────────────────────
            profiles_dir = pr / "configs" / "task_profiles"
            if not profiles_dir.exists():
                errors.add(label, f"Task profile directory not found: {profiles_dir}")

    # ── IMAGE_SUFFIX sanity check ─────────────────────────────────────────
    if "IMAGE_SUFFIX" in data and isinstance(data["IMAGE_SUFFIX"], str):
        suf = data["IMAGE_SUFFIX"]
        if not suf.startswith("_"):
            errors.add(label,
                       f"IMAGE_SUFFIX {suf!r} does not start with '_' — "
                       "this is unusual; verify the naming convention")
        if not suf.endswith(".nii.gz") and not suf.endswith(".nii"):
            errors.add(label,
                       f"IMAGE_SUFFIX {suf!r} does not end in .nii.gz or .nii")


def validate_task_profile(path: Path, task_mode: str | None, errors: _Errors) -> None:
    label = "task_profile.yaml"
    data = _load_yaml(path, errors, label)
    if data is None:
        return
    for key in _TASK_PROFILE_REQUIRED_KEYS:
        if key not in data:
            errors.add(label, f"Missing required key: {key!r}")
    mode = str(data.get("TASK_MODE", "")).strip().lower()
    if mode not in VALID_TASK_MODES:
        errors.add(label, f"TASK_MODE {mode!r} is not valid. Allowed: {sorted(VALID_TASK_MODES)}")
    if task_mode and mode and mode != task_mode:
        errors.add(label, f"TASK_MODE mismatch: profile={mode!r}, requested={task_mode!r}")
    for key in ("REQUIRED_COMPONENTS", "REQUIRED_SEGMENTATIONS", "ACTIVE_QC_DOMAINS"):
        if key in data and not isinstance(data[key], dict):
            errors.add(label, f"{key!r} must be a mapping")
    if isinstance(data.get("REQUIRED_SEGMENTATIONS"), dict):
        for seg_key, seg_name in data["REQUIRED_SEGMENTATIONS"].items():
            if not isinstance(seg_name, str) or not seg_name.endswith((".nii", ".nii.gz")):
                errors.add(label, f"REQUIRED_SEGMENTATIONS.{seg_key}: expected a NIfTI filename string")
    for key in ("HARD_FAILURE_DOMAINS", "CALIBRATABLE_DOMAINS"):
        if key in data and not isinstance(data[key], list):
            errors.add(label, f"{key!r} must be a list")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate thresholds.yaml and paths.yaml before running the QC pipeline.")
    parser.add_argument("--paths-yaml", default=None, dest="paths_yaml",
                        help="Path to paths.yaml (default: configs/paths.yaml)")
    parser.add_argument("--thresholds-yaml", default=None, dest="thresholds_yaml",
                        help="Path to thresholds.yaml (default: configs/thresholds.yaml)")
    parser.add_argument("--strict", action="store_true",
                        help="Treat warnings as errors (exit 1 even for minor issues).")
    parser.add_argument("--task-mode", default=None, choices=sorted(VALID_TASK_MODES),
                        help="Validate configs/task_profiles/<task-mode>.yaml")
    parser.add_argument("--task-profile", default=None,
                        help="Explicit task profile YAML to validate.")
    args = parser.parse_args()

    thr_path  = (Path(args.thresholds_yaml) if args.thresholds_yaml
                 else _PROJECT_ROOT / "configs" / "thresholds.yaml")
    path_path = (Path(args.paths_yaml) if args.paths_yaml
                 else _PROJECT_ROOT / "configs" / "paths.yaml")
    profile_path = None
    if args.task_profile:
        profile_path = Path(args.task_profile)
    elif args.task_mode:
        profile_path = _PROJECT_ROOT / "configs" / "task_profiles" / f"{args.task_mode}.yaml"

    if not _HAS_YAML:
        print("[ERROR] PyYAML is not installed.")
        print("        Install with: pip install pyyaml")
        raise SystemExit(1)

    print(f"Validating: {thr_path}")
    print(f"         +  {path_path}\n")

    errors = _Errors()
    validate_thresholds(thr_path,  errors)
    validate_paths(path_path, errors)
    if profile_path:
        validate_task_profile(profile_path, args.task_mode, errors)

    if errors:
        print(f"{'='*60}")
        print(f"  VALIDATION FAILED  ({len(errors)} error(s) found)")
        print(f"{'='*60}")
        errors.print_report()
        print()
        raise SystemExit(1)
    else:
        print(f"{'='*60}")
        print("  VALIDATION PASSED  (no errors)")
        print(f"{'='*60}")
        print(f"  thresholds : {thr_path}")
        print(f"  paths      : {path_path}")
        if profile_path:
            print(f"  task profile : {profile_path}")


if __name__ == "__main__":
    main()
