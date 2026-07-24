"""Shared helpers for strictly explanatory, artifact-grounded agents."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

from artifacts.io import artifact_descriptor, load_artifact
from artifacts.loaders import load_qc_cases


class CaseBackend(Protocol):
    """Provider-neutral interface for frozen per-case structured backends."""

    name: str
    frozen: bool

    def generate_case(self, case_input: Mapping[str, Any]) -> dict[str, Any]: ...


def load_mapping_artifact(path: Path) -> tuple[dict, dict]:
    metadata, data = load_artifact(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected object payload in artifact: {path}")
    return metadata, data


def load_case_artifact(path: Path) -> tuple[dict[str, dict], dict]:
    cases, _, metadata = load_qc_cases(path)
    normalized = {
        str(case_id): case
        for case_id, case in cases.items()
        if isinstance(case, dict)
    }
    return normalized, metadata


def json_pointer_token(value: str) -> str:
    return str(value).replace("~", "~0").replace("/", "~1")


def case_reference(path: Path, artifact_type: str, case_id: str, *, index: int | None = None) -> dict:
    descriptor = artifact_descriptor(path, artifact_type)
    if index is None:
        descriptor["case_pointer"] = f"/data/cases/{json_pointer_token(case_id)}"
    else:
        descriptor["case_pointer"] = f"/data/all_cases/{index}"
    return descriptor


def comparison_index(data: Mapping[str, Any]) -> tuple[dict[str, dict], dict[str, int], list[str]]:
    rows = data.get("all_cases") or []
    if not isinstance(rows, list):
        return {}, {}, ["comparison_all_cases_not_a_list"]
    indexed: dict[str, dict] = {}
    positions: dict[str, int] = {}
    issues: list[str] = []
    for position, row in enumerate(rows):
        if not isinstance(row, dict):
            issues.append(f"comparison_row_{position}_not_an_object")
            continue
        case_id = row.get("case_id")
        if case_id is None or str(case_id).strip() == "":
            issues.append(f"comparison_row_{position}_missing_case_id")
            continue
        case_id = str(case_id)
        if case_id in indexed:
            issues.append(f"comparison_duplicate_case_id:{case_id}")
            continue
        indexed[case_id] = row
        positions[case_id] = position
    return indexed, positions, issues


def expected_context_case_ids(context: Mapping[str, Any]) -> set[str]:
    for key in ("expected_case_ids", "case_ids"):
        values = context.get(key)
        if isinstance(values, list):
            return {str(value) for value in values}
    cases = context.get("cases")
    if isinstance(cases, dict):
        return {str(case_id) for case_id in cases}
    return set()


def context_status(context: Mapping[str, Any]) -> str:
    value = context.get("validation_status", context.get("status", "unknown"))
    if isinstance(value, bool):
        return "passed" if value else "failed"
    return str(value).lower()


def identity_value(metadata: Mapping[str, Any], data: Mapping[str, Any], field: str) -> Any:
    value = metadata.get(field)
    if value is not None:
        return value
    return data.get(field)


def metadata_alignment_issues(
    context_metadata: Mapping[str, Any],
    context_data: Mapping[str, Any],
    inputs: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> list[str]:
    issues: list[str] = []
    for field in ("run_id", "dataset_name", "task_mode"):
        expected = identity_value(context_metadata, context_data, field)
        if expected is None:
            continue
        for source, (metadata, data) in inputs.items():
            actual = identity_value(metadata, data, field)
            if actual is not None and actual != expected:
                issues.append(f"{source}_{field}_mismatch:{actual!s}!={expected!s}")
    return sorted(issues)


def qc_snapshot(case: Mapping[str, Any] | None) -> dict[str, Any]:
    if case is None:
        return {"availability": "missing"}
    triage = case.get("triage") or {}
    primary = case.get("primary_issue") or {}
    domains = case.get("qc_domains") or {}
    abnormal_domains = []
    if isinstance(domains, dict):
        for domain, details in sorted(domains.items()):
            if not isinstance(details, dict):
                continue
            severity = details.get("severity", "unknown")
            if severity not in {"normal", "not_applicable"}:
                abnormal_domains.append({
                    "domain": domain,
                    "severity": severity,
                    "component_score": details.get("component_score", details.get("score")),
                })
    return {
        "availability": "available",
        "reported_recommendation": triage.get("recommendation", case.get("recommendation", "unknown")),
        "risk_level": triage.get("risk_level", case.get("risk_level", "unknown")),
        "score": triage.get("score", case.get("qc_score", case.get("score"))),
        "primary_domain": primary.get("domain") or case.get("driving_domain"),
        "abnormal_domains": abnormal_domains,
    }


def comparison_snapshot(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {"availability": "missing"}
    return {
        "availability": "available",
        "changed": bool(row.get("changed", False)),
        "recommendation_change": row.get("recommendation_change", "unknown"),
        "risk_change": row.get("risk_change", "unknown"),
        "score_delta": row.get("score_delta"),
        "domain_deterministic": row.get("domain_deterministic"),
        "domain_calibrated": row.get("domain_calibrated"),
    }


def source_reference_map(
    case_id: str,
    deterministic_path: Path,
    calibrated_path: Path,
    comparison_path: Path,
    comparison_position: int | None,
) -> dict[str, dict]:
    return {
        "deterministic": case_reference(
            deterministic_path, "qc_report_deterministic", case_id
        ),
        "calibrated": case_reference(
            calibrated_path, "qc_report_calibrated", case_id
        ),
        "comparison": case_reference(
            comparison_path,
            "qc_comparison",
            case_id,
            index=comparison_position,
        ) if comparison_position is not None else {
            **artifact_descriptor(comparison_path, "qc_comparison"),
            "case_pointer": None,
        },
    }


def assert_no_forbidden_keys(
    payload: Any,
    *,
    exact: set[str],
    forbidden_tokens: set[str] | None = None,
    location: str = "data",
) -> None:
    if isinstance(payload, dict):
        for key, value in payload.items():
            normalized = str(key).lower()
            tokens = set(normalized.split("_"))
            has_forbidden_token = forbidden_tokens and any(
                token.startswith(forbidden)
                for token in tokens
                for forbidden in forbidden_tokens
            )
            if normalized in exact or has_forbidden_token:
                raise ValueError(f"Forbidden explanatory output field at {location}.{key}")
            assert_no_forbidden_keys(
                value,
                exact=exact,
                forbidden_tokens=forbidden_tokens,
                location=f"{location}.{key}",
            )
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            assert_no_forbidden_keys(
                value,
                exact=exact,
                forbidden_tokens=forbidden_tokens,
                location=f"{location}[{index}]",
            )


def require_fields(payload: Mapping[str, Any], fields: set[str], *, label: str) -> None:
    missing = sorted(fields - set(payload))
    if missing:
        raise ValueError(f"{label} backend output missing required fields: {', '.join(missing)}")
