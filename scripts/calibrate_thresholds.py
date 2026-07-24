#!/usr/bin/env python3
"""Calibrate QC thresholds from reference-clean cases.

The output is a same-schema thresholds YAML file that can be passed directly to
qc_agent.py via --thresholds-yaml. Calibration is intentionally conservative:
hard clinical/technical rules remain deterministic, and domains with too few
reference-clean cases keep the base threshold values.
"""
from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
import math
from pathlib import Path
import sys

try:
    import yaml as _yaml
except ModuleNotFoundError as exc:  # pragma: no cover - CLI dependency guard
    raise SystemExit("PyYAML is required: pip install pyyaml") from exc

try:
    from ruamel.yaml import YAML as _RuamelYAML
    _HAS_RUAMEL = True
except ModuleNotFoundError:  # pragma: no cover - optional comment-preserving writer
    _RuamelYAML = None
    _HAS_RUAMEL = False

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from artifacts.hashing import hash_json_payload


def _load_yaml(path: Path) -> dict:
    if _HAS_RUAMEL:
        yaml = _RuamelYAML()
        yaml.preserve_quotes = True
        with path.open(encoding="utf-8") as fh:
            data = yaml.load(fh) or {}
        if not isinstance(data, dict):
            raise SystemExit(f"YAML top-level must be a mapping: {path}")
        return data
    with path.open(encoding="utf-8") as fh:
        data = _yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise SystemExit(f"YAML top-level must be a mapping: {path}")
    return data


def _write_yaml(data: dict, path: Path) -> None:
    if _HAS_RUAMEL:
        yaml = _RuamelYAML()
        yaml.preserve_quotes = True
        yaml.indent(mapping=2, sequence=4, offset=2)
        with path.open("w", encoding="utf-8") as fh:
            yaml.dump(data, fh)
        return
    path.write_text(_yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _load_summary(path: Path) -> tuple[dict, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "data" in raw and "metadata" in raw:
        raw = raw.get("data") or {}
    if isinstance(raw, dict) and "cases" in raw:
        return raw.get("cases") or {}, raw.get("metadata") or {}
    return raw, {}


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires non-empty values")
    vals = sorted(values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def _clamp(value: float, limits: list[float] | tuple[float, float] | None) -> float:
    if not limits or len(limits) != 2:
        return value
    return max(float(limits[0]), min(float(value), float(limits[1])))


def _case_status(case: dict) -> dict:
    return case.get("case_status") or {}


def _quality(case: dict) -> dict:
    return case.get("quality_control") or {}


def _pancreas(case: dict) -> dict:
    return case.get("pancreas") or {}


def _lesions(case: dict) -> dict:
    return case.get("lesions") or {}


def _geometry_failed(case: dict) -> bool:
    geo = ((case.get("geometry") or {}).get("summary") or {})
    img = case.get("image") or {}
    spacing = img.get("spacing_xyz_mm") or []
    try:
        bad_spacing = any(float(s) <= 0 or float(s) > 10 for s in spacing)
    except (TypeError, ValueError):
        bad_spacing = True if spacing else False
    return bool(
        geo.get("pancreas_affine_mismatch")
        or geo.get("lesion_affine_mismatch")
        or _case_status(case).get("omit_from_training")
        or bad_spacing
    )


def _lesion_present(case: dict) -> bool:
    les = _lesions(case)
    return not bool((_quality(case)).get("lesion_mask_empty")) and (
        (les.get("n_lesions") or 0) > 0 or (les.get("total_volume_mm3") or 0) > 0
    )


def _pancreas_present(case: dict) -> bool:
    pan = _pancreas(case)
    return not bool(_quality(case).get("pancreas_mask_empty")) and (pan.get("volume_mm3") or 0) > 0


def _overlap(case: dict) -> float | None:
    qc = _quality(case)
    loa = qc.get("lesion_overlap_analysis") or {}
    ref_key = loa.get("overlap_reference_used_for_qc") or "combined_filled_pancreas"
    ref = loa.get(ref_key) or {}
    val = ref.get("overlap_rate_vs_lesion")
    return val if val is not None else qc.get("total_overlap_rate_vs_lesion")


def _reference_clean_cases(cases: dict, cfg: dict) -> tuple[list[str], dict[str, list[str]]]:
    policy = ((cfg.get("CALIBRATION_POLICY") or {}).get("reference_clean") or {})
    min_overlap = float(policy.get("min_lesion_overlap_for_reference", cfg.get("MIN_LESION_PANCREAS_OVERLAP", 0.80)))
    selected: list[str] = []
    excluded: dict[str, list[str]] = {}
    for cid, case in cases.items():
        reasons: list[str] = []
        if _case_status(case).get("omit_from_training"):
            reasons.append("omitted")
        if _geometry_failed(case):
            reasons.append("geometry_failure")
        if not _pancreas_present(case):
            reasons.append("pancreas_absent")
        if _lesion_present(case):
            ov = _overlap(case)
            if ov is None or float(ov) < min_overlap:
                reasons.append("low_lesion_overlap")
        if reasons:
            excluded[cid] = reasons
        else:
            selected.append(cid)
    return selected, excluded


def _fov_score(case: dict, cfg: dict) -> float:
    pts = cfg.get("FOV_SCORE_POINTS") or {}
    pan = _pancreas(case)
    img = case.get("image") or {}
    shape = img.get("shape_zyx")
    bbox = pan.get("bounding_box_zyx")
    border = False
    if bbox is not None and shape is not None:
        bmin, bmax = bbox[0], bbox[1]
        nz, ny, nx = shape
        margin = 2
        border = (
            bmin[0] <= margin or bmax[0] >= nz - 1 - margin
            or bmin[1] <= margin or bmax[1] >= ny - 1 - margin
            or bmin[2] <= margin or bmax[2] >= nx - 1 - margin
        )
    score = float(pts.get("border_touching", 2)) if border else 0.0
    # Calibration does not infer partial/truncation from absent subregions. Those
    # are evaluated in final QC; this reference score captures general FOV signal.
    return score


def _pancreas_context_score(case: dict, cfg: dict) -> float:
    pan = _pancreas(case)
    pvol = float(pan.get("volume_mm3") or 0.0)
    pancreas_present = _pancreas_present(case)
    lesion_present = _lesion_present(case)
    min_pan = float(cfg.get("MIN_PANCREAS_VOLUME_MM3", 20_000))
    max_pan = float(cfg.get("MAX_PANCREAS_VOLUME_MM3", 200_000))
    score = 0.0
    if pancreas_present:
        if 0 < pvol < min_pan * 0.5:
            score += 20
        elif 0 < pvol < min_pan:
            score += 10
        elif pvol > max_pan:
            score += 5
    elif lesion_present:
        score += 5
    return min(score, 20.0)


def _lesion_burden_score(case: dict, cfg: dict) -> float:
    pan = _pancreas(case)
    les = _lesions(case)
    pvol = float(pan.get("volume_mm3") or 0.0)
    total_lv = float(les.get("total_volume_mm3") or 0.0)
    if not _lesion_present(case) or total_lv <= 0:
        return 0.0

    min_les = float(cfg.get("MIN_LESION_VOLUME_MM3", 10))
    max_les = float(cfg.get("MAX_LESION_VOLUME_MM3", 200_000))
    max_ratio = float(cfg.get("MAX_LESION_PANCREAS_RATIO", 0.80))
    score = 0.0
    if total_lv < min_les:
        score += 15
    if total_lv > max_les:
        score += 15
    if _pancreas_present(case) and pvol > 0 and total_lv / pvol > max_ratio:
        score += 10
    return min(score, 20.0)


def _calibrate_profile_domain(values: list[float], base: dict, policy: dict) -> tuple[dict, dict]:
    pct = policy.get("percentiles") or {}
    guard = policy.get("guardrails") or {}
    out = dict(base)
    details: dict = {"n_values": len(values), "method": "percentile"}
    if not values:
        details["status"] = "skipped_no_values"
        details["changed"] = False
        return out, details
    for key in ("low_warning", "moderate_warning", "high_warning", "critical"):
        if key not in pct:
            continue
        raw = _percentile(values, float(pct[key]))
        val = round(_clamp(raw, guard.get(key)))
        out[key] = int(max(0, val))
        details[key] = {"percentile": pct[key], "raw": raw, "final": out[key], "guardrail": guard.get(key)}
    last = 0
    for key in ("low_warning", "moderate_warning", "high_warning", "critical"):
        out[key] = max(int(out[key]), last)
        last = int(out[key])
    # validate_config.py requires a strict severity ladder. Guardrails may clamp
    # adjacent percentiles to the same value, so repair from critical downward.
    order = ("low_warning", "moderate_warning", "high_warning", "critical")
    for prev, cur in zip(reversed(order[:-1]), reversed(order[1:])):
        out[prev] = min(int(out[prev]), int(out[cur]) - 1)
    for key in order:
        out[key] = max(0, int(out[key]))
    details["status"] = "calibrated"
    details["changed"] = out != base
    return out, details


def _calibrate_domain_from_scores(
    cfg: dict,
    cases: dict,
    ref: list[str],
    domain: str,
    scorer,
    report: dict,
) -> None:
    domain_policy = (cfg.get("CALIBRATION_POLICY") or {}).get(domain) or {}
    if not (
        domain_policy.get("calibrate", False)
        and domain_policy.get("calibrate_score_thresholds", True)
    ):
        report["domains"][domain] = {"status": "kept_deterministic", "changed": False}
        return

    values = [scorer(cases[cid], cfg) for cid in ref]
    values = [v for v in values if v > 0]
    base_domain = (cfg.get("QC_PROFILE_THRESHOLDS") or {}).get(domain, {})
    calibrated, details = _calibrate_profile_domain(values, base_domain, domain_policy)
    cfg.setdefault("QC_PROFILE_THRESHOLDS", {})[domain] = calibrated
    report["domains"][domain] = details


def calibrate(
    base: dict,
    cases: dict,
    summary_path: Path,
    *,
    task_mode: str | None = None,
    calibratable_domains: set[str] | None = None,
) -> tuple[dict, dict]:
    cfg = copy.deepcopy(base)
    if task_mode:
        cfg["TASK_MODE"] = task_mode
    policy = cfg.get("CALIBRATION_POLICY") or {}
    supported_domains = {"fov_integrity", "pancreas_context", "lesion_burden"}
    selected_domains = supported_domains if calibratable_domains is None else set(calibratable_domains)
    unknown_domains = selected_domains - supported_domains
    if unknown_domains:
        raise ValueError(f"Unsupported calibratable domains: {sorted(unknown_domains)}")
    minimums = policy.get("minimums") or {}
    min_ref = int(minimums.get("min_reference_cases", 100))
    min_frac = float(minimums.get("min_reference_fraction", 0.15))
    ref, excluded = _reference_clean_cases(cases, cfg)
    report = {
        "summary": str(summary_path),
        "task_mode": task_mode,
        "total_cases": len(cases),
        "reference_clean_cases": len(ref),
        "excluded_from_calibration": len(excluded),
        "exclusion_reasons": {},
        "calibratable_domains": sorted(selected_domains),
        "domains": {},
    }
    for reasons in excluded.values():
        for reason in reasons:
            report["exclusion_reasons"][reason] = report["exclusion_reasons"].get(reason, 0) + 1

    enough_ref = len(ref) >= min_ref and (len(ref) / max(len(cases), 1)) >= min_frac
    if not enough_ref:
        report["status"] = "insufficient_reference_clean_cases"
    else:
        report["status"] = "partial_or_full_calibration"

    fov_policy = policy.get("fov_integrity") or {}
    if "fov_integrity" not in selected_domains:
        report["domains"]["fov_integrity"] = {
            "status": "not_calibratable_for_task",
            "changed": False,
        }
    elif enough_ref and fov_policy.get("calibrate", False) and fov_policy.get("calibrate_score_thresholds", True):
        values = [_fov_score(cases[cid], cfg) for cid in ref]
        values = [v for v in values if v > 0]
        base_domain = (cfg.get("QC_PROFILE_THRESHOLDS") or {}).get("fov_integrity", {})
        calibrated, details = _calibrate_profile_domain(values, base_domain, fov_policy)
        cfg.setdefault("QC_PROFILE_THRESHOLDS", {})["fov_integrity"] = calibrated
        report["domains"]["fov_integrity"] = details
    else:
        report["domains"]["fov_integrity"] = {"status": "kept_deterministic", "changed": False}

    for domain, scorer in (
        ("pancreas_context", _pancreas_context_score),
        ("lesion_burden", _lesion_burden_score),
    ):
        if domain not in selected_domains:
            report["domains"][domain] = {
                "status": "not_calibratable_for_task",
                "changed": False,
            }
        elif enough_ref:
            _calibrate_domain_from_scores(cfg, cases, ref, domain, scorer, report)
        else:
            report["domains"][domain] = {"status": "kept_deterministic", "changed": False}

    report["effective_thresholds_changed"] = any(
        bool(info.get("changed")) for info in report["domains"].values()
    )
    if not report["effective_thresholds_changed"]:
        report["warning"] = "Calibration completed, but effective thresholds are unchanged from base config."

    meta = cfg.setdefault("THRESHOLD_METADATA", {})
    meta["method"] = "calibrated"
    meta["source"] = "reference_clean_calibration"
    meta["created_at"] = _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    meta["calibrated_from_summary"] = str(summary_path)
    meta["calibrated_from_summary_payload_sha256"] = hash_json_payload(cases)
    meta["task_mode"] = task_mode
    meta["calibration_policy_version"] = str(policy.get("version", "v1"))
    meta["reference_clean_cases"] = len(ref)
    meta["total_cases"] = len(cases)
    return cfg, report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate same-schema calibrated QC thresholds.")
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="Run-scoped output calibrated thresholds YAML.")
    parser.add_argument("--report-json", type=Path, default=None)
    parser.add_argument("--report-txt", type=Path, default=None)
    parser.add_argument("--task-mode", required=True)
    parser.add_argument("--task-profile", required=True, type=Path)
    args = parser.parse_args()

    base = _load_yaml(args.base_config)
    cases, summary_metadata = _load_summary(args.summary)
    task_mode = args.task_mode or summary_metadata.get("task_mode")
    task_profile = _load_yaml(args.task_profile)
    calibrated, report = calibrate(
        base,
        cases,
        args.summary,
        task_mode=task_mode,
        calibratable_domains=set(task_profile.get("CALIBRATABLE_DOMAINS") or []),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(calibrated, args.output)

    report_json = args.report_json or args.output.with_suffix(".calibration_report.json")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report_txt = args.report_txt or args.output.with_suffix(".calibration_report.txt")
    lines = [
        "QC Threshold Calibration Report",
        "================================",
        f"Summary: {report['summary']}",
        f"Total cases: {report['total_cases']}",
        f"Reference-clean cases: {report['reference_clean_cases']}",
        f"Status: {report['status']}",
        "",
        "Domains:",
    ]
    for domain, info in report["domains"].items():
        changed = "changed" if info.get("changed") else "unchanged"
        lines.append(f"  {domain}: {info.get('status')} ({changed})")
    if report.get("warning"):
        lines.extend(["", f"Warning: {report['warning']}"])
    report_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote calibrated thresholds: {args.output}")
    print(f"Wrote calibration report: {report_json}")


if __name__ == "__main__":
    main()
