"""Shared deterministic evidence and policy helpers."""

from __future__ import annotations

from pathlib import Path

from agents.utils.task_profiles import (
    active_qc_domains,
    hard_failure_domains,
    load_task_profile,
)
from artifacts.io import load_artifact
from artifacts.loaders import load_qc_cases

DEFAULT_HARD_DOMAINS = {"geometry_integrity", "lesion_localization"}
DEFAULT_SOFT_DOMAINS = {
    "fov_integrity",
    "pancreas_context",
    "lesion_burden",
    "attenuation_integrity",
    "region_consistency",
}
REVIEW_SEVERITIES = {"moderate_warning", "high_warning", "critical"}
REVIEW_RISKS = {"medium", "high", "critical"}


def load_qc_case_map(path: Path) -> dict:
    cases, _, _ = load_qc_cases(path)
    return cases


def load_artifact_data(path: Path) -> dict:
    _, data = load_artifact(path)
    return data if isinstance(data, dict) else {}


def triage(case: dict) -> dict:
    case_triage = case.get("triage") or {}
    return {
        "recommendation": case_triage.get("recommendation", case.get("recommendation")),
        "risk_level": case_triage.get("risk_level", case.get("risk_level")),
        "score": case_triage.get("score", case.get("qc_score", case.get("score"))),
    }


def primary_domain(case: dict) -> str | None:
    return ((case.get("primary_issue") or {}).get("domain") or case.get("driving_domain"))


def domain_severity(case: dict, domain: str | None) -> str:
    if not domain:
        return "normal"
    return ((case.get("qc_domains") or {}).get(domain) or {}).get("severity", "normal")


def policy_domains(task_mode: str | None, profile: dict | None) -> tuple[set[str], set[str]]:
    selected = profile
    if selected is None and task_mode:
        try:
            selected = load_task_profile(task_mode)
        except (FileNotFoundError, ValueError):
            selected = None
    if selected is None:
        return set(DEFAULT_HARD_DOMAINS), set(DEFAULT_SOFT_DOMAINS)

    active_config = active_qc_domains(selected)
    active = {domain for domain, enabled in active_config.items() if enabled}
    hard = hard_failure_domains(selected)
    if active_config:
        hard &= active
    return hard, active - hard


def comparison_case_map(comparison: dict) -> dict[str, dict]:
    records: dict[str, dict] = {}
    cases = comparison.get("cases") or {}
    if isinstance(cases, dict):
        records.update({str(case_id): row for case_id, row in cases.items() if isinstance(row, dict)})
    for key in ("all_cases", "changed_cases"):
        for row in comparison.get(key, []) or []:
            if isinstance(row, dict) and row.get("case_id") is not None:
                records[str(row["case_id"])] = row
    return records


def deterministic_hard_failure_rules(case: dict, hard_domains: set[str]) -> list[str]:
    rules: list[str] = []
    omitted = (
        triage(case).get("recommendation") == "omit_from_training"
        or (case.get("case_status") or {}).get("omit_from_training") is True
    )
    if omitted:
        rules.append("deterministic_omit_from_training")
    for domain in sorted(hard_domains):
        if domain_severity(case, domain) == "critical":
            rules.append(f"deterministic_critical_hard_domain:{domain}")
    return rules


def dataset_validation_hard_failure_rules(case: dict) -> list[str]:
    rules: list[str] = []
    if case.get("omit_from_training") is True:
        rules.append("dataset_validation_omit_from_training")
    if str(case.get("status", "")).strip().lower() == "invalid":
        rules.append("dataset_validation_status_invalid")
    return rules


def has_signal(case: dict, *names: str) -> bool:
    case_triage = case.get("triage") or {}
    for name in names:
        value = case.get(name, case_triage.get(name))
        if value is True or (isinstance(value, str) and value.lower() in {"true", "yes", "uncertain"}):
            return True
    return False


def evidence_disagrees(deterministic: dict, calibrated: dict) -> bool:
    det_triage = triage(deterministic)
    cal_triage = triage(calibrated)
    pairs = (
        (det_triage.get("recommendation"), cal_triage.get("recommendation")),
        (det_triage.get("risk_level"), cal_triage.get("risk_level")),
        (primary_domain(deterministic), primary_domain(calibrated)),
    )
    if any(left is not None and right is not None and left != right for left, right in pairs):
        return True
    try:
        det_score = det_triage.get("score")
        cal_score = cal_triage.get("score")
        return (
            det_score is not None
            and cal_score is not None
            and abs(float(cal_score) - float(det_score)) >= 1.0
        )
    except (TypeError, ValueError):
        return False
