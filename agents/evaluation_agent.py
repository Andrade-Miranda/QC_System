#!/usr/bin/env python3
"""EvaluationAgent: post-hoc internal audit of QC pipeline artifacts."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import resolve_project_paths, VALID_TASK_MODES
from artifacts.io import artifact_descriptor, load_artifact, write_artifact


HARD_DOMAINS = {"geometry_integrity", "lesion_localization"}
KEEP_ACTIONS = {"keep", "keep_with_metadata_warning"}
CANONICAL_ACTIONS = ("keep", "warning", "review", "reject", "insufficient_evidence")
FINAL_REQUIRED_FIELDS = {
    "final_decision",
    "decision_basis",
    "hard_failure",
    "requires_human_review",
    "deterministic_recommendation",
    "calibrated_recommendation",
}
ARTIFACT_ALIASES = {
    "validated_context": "validated_context",
    "dataset_validation": "dataset_validation",
    "deterministic": "deterministic_evidence",
    "deterministic_evidence": "deterministic_evidence",
    "qc_report_deterministic": "deterministic_evidence",
    "calibrated": "calibrated_evidence",
    "calibrated_evidence": "calibrated_evidence",
    "qc_report_calibrated": "calibrated_evidence",
    "comparison": "comparison",
    "qc_comparison": "comparison",
    "reasoning": "reasoning",
    "critique": "critique",
    "routing": "routing",
    "review_routing": "routing",
    "final": "final_decisions",
    "final_decisions": "final_decisions",
    "final_qc_decisions": "final_decisions",
    "golden_labels": "golden_labels",
}


def _metric(
    numerator: int | float,
    denominator: int,
    *,
    eligible_n: int | None = None,
    missing_n: int = 0,
    excluded_n: int = 0,
    value: int | float | None = None,
    status: str = "available",
    definition: str | None = None,
) -> dict:
    """Build the common metric envelope, including safe zero-denominator handling."""
    if denominator == 0:
        value = None
        status = "not_applicable"
    elif value is None:
        value = numerator / denominator
    result = {
        "value": value,
        "numerator": numerator,
        "denominator": denominator,
        "status": status,
        "eligible_n": denominator + missing_n if eligible_n is None else eligible_n,
        "missing_n": missing_n,
        "excluded_n": excluded_n,
    }
    if definition:
        result["definition"] = definition
    return result


def _rule_metric(numerator: int, denominator: int, **kwargs: Any) -> dict:
    status = "passed" if denominator and numerator == denominator else "failed"
    return _metric(numerator, denominator, status=status, **kwargs)


def _load(path: Path | None) -> dict:
    if path is None:
        return {"path": None, "status": "not_provided", "metadata": {}, "data": {}}
    try:
        metadata, data = load_artifact(path)
    except (OSError, ValueError, TypeError) as exc:
        return {
            "path": str(path),
            "status": "load_error",
            "error": str(exc),
            "metadata": {},
            "data": {},
        }
    return {
        "path": str(path),
        "status": "available",
        "metadata": metadata if isinstance(metadata, dict) else {},
        "data": data if isinstance(data, dict) else {},
    }


def _records(data: dict, *keys: str) -> dict[str, dict]:
    for key in keys:
        value = data.get(key)
        if isinstance(value, dict):
            return {str(case_id): row if isinstance(row, dict) else {"value": row} for case_id, row in value.items()}
        if isinstance(value, list):
            rows = {}
            for row in value:
                if not isinstance(row, dict):
                    continue
                case_id = row.get("case_id") or row.get("id")
                if case_id is not None:
                    rows[str(case_id)] = row
            return rows
    return {}


def _artifact_cases(name: str, data: dict) -> dict[str, dict]:
    if name == "final_decisions":
        records = _records(data, "decisions")
        if records:
            return records
        return {
            str(case_id): row for case_id, row in data.items()
            if isinstance(row, dict) and "final_decision" in row
        }
    if name == "routing":
        rows = {}
        rows.update(_records(data, "not_routed_cases"))
        rows.update(_records(data, "routed_cases"))
        return rows
    if name == "comparison":
        return _records(data, "all_cases", "changed_cases")
    if name == "golden_labels":
        records = _records(data, "golden_labels", "labels", "cases")
        if records:
            return records
        reserved = {"metadata", "summary", "status", "task_mode", "dataset_name"}
        return {
            str(case_id): row if isinstance(row, dict) else {"value": row}
            for case_id, row in data.items()
            if case_id not in reserved
        }
    if name == "reasoning":
        return _records(data, "cases", "reasoning", "results", "explanations")
    if name == "critique":
        return _records(data, "cases", "critiques", "critique", "results")
    records = _records(data, "cases", "validated_cases", "context")
    if records:
        return records
    if name in {"deterministic_evidence", "calibrated_evidence"}:
        return {
            str(case_id): row for case_id, row in data.items()
            if isinstance(row, dict)
            and any(field in row for field in ("triage", "qc_domains", "recommendation", "qc_score"))
        }
    if name == "dataset_validation":
        return {
            str(case_id): row for case_id, row in data.items()
            if isinstance(row, dict) and any(field in row for field in ("status", "resources", "omit_from_training"))
        }
    return {}


def _triage(case: dict) -> dict:
    triage = case.get("triage") or {}
    return {
        "recommendation": triage.get("recommendation", case.get("recommendation")),
        "risk_level": triage.get("risk_level", case.get("risk_level")),
        "score": triage.get("score", case.get("qc_score", case.get("score"))),
        "domain": case.get("driving_domain") or (case.get("primary_issue") or {}).get("domain"),
    }


def _is_hard_failure(case: dict) -> bool:
    if case.get("hard_failure") or case.get("omit_from_training"):
        return True
    triage = _triage(case)
    if triage["recommendation"] == "omit_from_training":
        return True
    domain = triage["domain"] or case.get("primary_domain")
    severity = ((case.get("qc_domains") or {}).get(domain) or {}).get("severity") if domain else None
    return domain in HARD_DOMAINS and severity == "critical"


def _reference_case_ids(artifacts: dict[str, dict]) -> tuple[str | None, set[str]]:
    for name in (
        "validated_context",
        "dataset_validation",
        "deterministic_evidence",
        "calibrated_evidence",
        "final_decisions",
        "comparison",
        "routing",
    ):
        cases = _artifact_cases(name, artifacts[name]["data"])
        if cases:
            return name, set(cases)
    return None, set()


def _required_artifact_names(context: dict) -> list[str]:
    required = context.get("required_artifacts")
    if isinstance(required, dict):
        required = [name for name, is_required in required.items() if is_required]
    if not isinstance(required, list):
        return ["final_decisions"]
    names = []
    for name in required:
        normalized = ARTIFACT_ALIASES.get(str(name))
        if normalized and normalized != "golden_labels" and normalized not in names:
            names.append(normalized)
    return names or ["final_decisions"]


def _artifact_integrity(artifacts: dict[str, dict]) -> dict:
    availability = {
        name: {
            "status": artifact["status"],
            "path": artifact["path"],
            **({"error": artifact["error"]} if artifact.get("error") else {}),
        }
        for name, artifact in artifacts.items()
        if name != "golden_labels"
    }
    required_names = _required_artifact_names(artifacts["validated_context"]["data"])
    required_present = sum(artifacts[name]["status"] == "available" for name in required_names)

    provided = [artifact for artifact in artifacts.values() if artifact["path"] is not None]
    loaded = sum(artifact["status"] == "available" for artifact in provided)

    reference_name, reference_ids = _reference_case_ids(artifacts)
    case_sets = {}
    for name, artifact in artifacts.items():
        if name == "golden_labels" or artifact["status"] != "available":
            continue
        ids = set(_artifact_cases(name, artifact["data"]))
        if ids:
            case_sets[name] = ids
    comparisons = {name: ids for name, ids in case_sets.items() if name != reference_name}
    matching_sets = sum(ids == reference_ids for ids in comparisons.values())
    mismatches = {
        name: {
            "missing_case_ids": sorted(reference_ids - ids),
            "unexpected_case_ids": sorted(ids - reference_ids),
        }
        for name, ids in comparisons.items()
        if ids != reference_ids
    }

    final_cases = _artifact_cases("final_decisions", artifacts["final_decisions"]["data"])
    field_denominator = len(final_cases) * len(FINAL_REQUIRED_FIELDS)
    fields_present = sum(field in row for row in final_cases.values() for field in FINAL_REQUIRED_FIELDS)

    metadata_artifacts = {
        name: artifact["metadata"]
        for name, artifact in artifacts.items()
        if name != "golden_labels" and artifact["status"] == "available" and artifact["metadata"]
    }
    metadata_checks = []
    metadata_mismatches = []
    if metadata_artifacts:
        metadata_reference_name = next(iter(metadata_artifacts))
        metadata_reference = metadata_artifacts[metadata_reference_name]
        for name, metadata in metadata_artifacts.items():
            if name == metadata_reference_name:
                continue
            for field in ("run_id", "dataset_name", "task_mode"):
                left, right = metadata_reference.get(field), metadata.get(field)
                if left is None or right is None:
                    continue
                matches = left == right
                metadata_checks.append(matches)
                if not matches:
                    metadata_mismatches.append({
                        "artifact": name,
                        "field": field,
                        "expected": left,
                        "observed": right,
                    })
    return {
        "artifact_availability": availability,
        "required_artifacts": required_names,
        "required_artifacts_present": _rule_metric(
            required_present,
            len(required_names),
            eligible_n=len(required_names),
            missing_n=len(required_names) - required_present,
        ),
        "provided_artifacts_load_rate": _rule_metric(
            loaded,
            len(provided),
            eligible_n=len(provided),
            missing_n=len(provided) - loaded,
        ),
        "case_id_set_match": {
            **_rule_metric(matching_sets, len(comparisons)),
            "reference_artifact": reference_name,
            "reference_case_n": len(reference_ids),
            "mismatches": mismatches,
        },
        "required_final_fields_present": _rule_metric(
            fields_present,
            field_denominator,
            eligible_n=field_denominator,
            missing_n=field_denominator - fields_present,
        ),
        "artifact_metadata_alignment": {
            **_rule_metric(sum(metadata_checks), len(metadata_checks)),
            "fields_checked": ["run_id", "dataset_name", "task_mode"],
            "mismatches": metadata_mismatches,
        },
    }


def _comparison_rows(comparison: dict, deterministic: dict, calibrated: dict) -> list[dict]:
    rows = comparison.get("all_cases")
    if isinstance(rows, list):
        return [row for row in rows if isinstance(row, dict)]
    if not deterministic or not calibrated:
        return []
    result = []
    for case_id in sorted(set(deterministic) | set(calibrated)):
        det = _triage(deterministic.get(case_id) or {})
        cal = _triage(calibrated.get(case_id) or {})
        result.append({
            "case_id": case_id,
            "recommendation_deterministic": det["recommendation"],
            "recommendation_calibrated": cal["recommendation"],
            "score_deterministic": det["score"],
            "score_calibrated": cal["score"],
            "changed": det != cal,
        })
    return result


def _evidence_stability(artifacts: dict[str, dict]) -> dict:
    comparison = artifacts["comparison"]["data"]
    deterministic = _artifact_cases("deterministic_evidence", artifacts["deterministic_evidence"]["data"])
    calibrated = _artifact_cases("calibrated_evidence", artifacts["calibrated_evidence"]["data"])
    rows = _comparison_rows(comparison, deterministic, calibrated)

    if rows:
        paired_rows = [
            row for row in rows
            if row.get("recommendation_deterministic") is not None
            and row.get("recommendation_calibrated") is not None
        ]
        changed = sum(bool(row.get("changed")) for row in paired_rows)
        agreements = sum(
            row["recommendation_deterministic"] == row["recommendation_calibrated"]
            for row in paired_rows
        )
    else:
        summary = comparison.get("summary") or {}
        compared = int(summary.get("n_cases_compared") or 0)
        changed = int(summary.get("n_changed_cases") or 0)
        recommendation_changes = int(summary.get("n_recommendation_changes") or 0)
        rows = [{} for _ in range(compared)]
        paired_rows = rows
        agreements = max(compared - recommendation_changes, 0)

    score_shifts = []
    missing_scores = 0
    for row in rows:
        det_score = row.get("score_deterministic")
        cal_score = row.get("score_calibrated")
        if det_score is None or cal_score is None:
            missing_scores += 1
            continue
        try:
            score_shifts.append(abs(float(cal_score) - float(det_score)))
        except (TypeError, ValueError):
            missing_scores += 1
    return {
        "calibration_sensitivity_rate": _metric(
            changed,
            len(paired_rows),
            eligible_n=len(rows),
            missing_n=len(rows) - len(paired_rows),
            definition="Fraction of compared cases with any recorded calibration-induced change.",
        ),
        "agreement_rate": _metric(
            agreements,
            len(paired_rows),
            eligible_n=len(rows),
            missing_n=len(rows) - len(paired_rows),
            definition="Deterministic/calibrated recommendation agreement; not medical-label agreement.",
        ),
        "mean_absolute_score_shift": _metric(
            sum(score_shifts),
            len(score_shifts),
            eligible_n=len(rows),
            missing_n=missing_scores,
            value=(sum(score_shifts) / len(score_shifts)) if score_shifts else None,
            definition="Mean absolute calibrated-minus-deterministic QC score shift.",
        ),
    }


def _contains_override(value: Any) -> bool:
    forbidden_tokens = {"decision", "route", "routing", "action", "block", "blocking"}
    if isinstance(value, dict):
        for key, child in value.items():
            tokens = str(key).lower().split("_")
            if any(
                token.startswith(forbidden)
                for token in tokens
                for forbidden in forbidden_tokens
            ):
                return True
            if _contains_override(child):
                return True
    elif isinstance(value, list):
        return any(_contains_override(item) for item in value)
    return False


def _structured_evidence_references(value: Any) -> list[Any]:
    references = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "evidence_reference":
                references.append(child)
            elif key == "evidence_references":
                if isinstance(child, dict):
                    references.extend(child.values())
                elif isinstance(child, list):
                    references.extend(child)
                else:
                    references.append(child)
            else:
                references.extend(_structured_evidence_references(child))
    elif isinstance(value, list):
        for item in value:
            references.extend(_structured_evidence_references(item))
    return references


def _valid_evidence_reference(reference: Any, case_id: str) -> bool:
    if not isinstance(reference, dict):
        return False
    if not isinstance(reference.get("artifact_type"), str) or not reference["artifact_type"]:
        return False
    pointer = reference.get("case_pointer")
    if pointer is None:
        return True
    if not isinstance(pointer, str):
        return False
    if pointer.startswith("/data/cases/"):
        encoded_case_id = case_id.replace("~", "~0").replace("/", "~1")
        return pointer == f"/data/cases/{encoded_case_id}"
    if pointer.startswith("/data/all_cases/"):
        return pointer.removeprefix("/data/all_cases/").isdigit()
    return False


def _explanation_validity(artifact: dict, reference_ids: set[str], kind: str) -> dict:
    records = _artifact_cases(kind, artifact["data"])
    if artifact["status"] != "available":
        return {
            "status": "unavailable",
            "reason": f"{kind}_artifact_not_provided" if artifact["status"] == "not_provided" else artifact["status"],
            "case_coverage": _metric(0, 0, eligible_n=len(reference_ids), missing_n=len(reference_ids)),
            "case_id_validity": _metric(0, 0),
            "decision_non_interference": _metric(0, 0),
            "evidence_reference_validity": _metric(0, 0),
        }

    record_ids = set(records)
    covered = len(reference_ids & record_ids)
    valid_ids = len(record_ids & reference_ids) if reference_ids else 0
    reference_checks = []
    records_without_references = 0
    for case_id, row in records.items():
        references = _structured_evidence_references(row)
        if not references:
            records_without_references += 1
            continue
        reference_checks.extend(
            _valid_evidence_reference(reference, case_id) for reference in references
        )
    return {
        "status": "available",
        "case_coverage": _metric(
            covered,
            len(reference_ids),
            eligible_n=len(reference_ids),
            missing_n=len(reference_ids - record_ids),
            excluded_n=len(record_ids - reference_ids),
        ),
        "case_id_validity": _rule_metric(
            valid_ids,
            len(record_ids),
            excluded_n=len(record_ids - reference_ids),
        ),
        "decision_non_interference": _rule_metric(
            sum(not _contains_override(row) for row in records.values()),
            len(records),
        ),
        "evidence_reference_validity": _rule_metric(
            sum(reference_checks),
            len(reference_checks),
            excluded_n=records_without_references,
        ),
    }


def _routing_behavior(artifacts: dict[str, dict], final_cases: dict, hard_ids: set[str]) -> dict:
    if artifacts["routing"]["status"] != "available":
        return {
            "status": "unavailable",
            "scope": "Internal rule and queue consistency only; medical recall is not evaluated.",
            "review_action_queue_consistency": _metric(0, 0),
            "review_queue_action_consistency": _metric(0, 0),
            "routing_record_consistency": _metric(0, 0),
            "hard_failure_queue_exclusion": _metric(0, 0),
        }
    routing = artifacts["routing"]["data"]
    routed_rows = _records(routing, "routed_cases")
    not_routed_rows = _records(routing, "not_routed_cases")
    routed_ids = set(routed_rows)
    review_ids = {
        case_id for case_id, row in final_cases.items()
        if row.get("final_decision") == "review" or row.get("requires_human_review") is True
    }
    queued_review_ids = routed_ids & review_ids
    queue_action_ids = {
        case_id for case_id in routed_ids
        if (final_cases.get(case_id) or {}).get("final_decision") == "review"
        or (final_cases.get(case_id) or {}).get("requires_human_review") is True
    }
    routing_rows = list(routed_rows.values()) + list(not_routed_rows.values())
    internally_consistent = sum(
        row.get("route_to_review", True) is not False for row in routed_rows.values()
    ) + sum(
        row.get("route_to_review", False) is not True for row in not_routed_rows.values()
    )
    hard_queued = routed_ids & hard_ids
    return {
        "status": "available",
        "scope": "Internal rule and queue consistency only; medical recall is not evaluated.",
        "review_action_queue_consistency": {
            **_rule_metric(
                len(queued_review_ids),
                len(review_ids),
                eligible_n=len(review_ids),
                missing_n=len(review_ids - routed_ids),
            ),
            "review_actions_missing_from_queue": sorted(review_ids - routed_ids),
        },
        "review_queue_action_consistency": {
            **_rule_metric(
                len(queue_action_ids),
                len(routed_ids),
                eligible_n=len(routed_ids),
                missing_n=len(routed_ids - set(final_cases)),
            ),
            "queued_cases_without_review_action": sorted(routed_ids - queue_action_ids),
        },
        "routing_record_consistency": _rule_metric(internally_consistent, len(routing_rows)),
        "hard_failure_queue_exclusion": {
            **_rule_metric(len(hard_ids) - len(hard_queued), len(hard_ids)),
            "hard_failures_in_review_queue": sorted(hard_queued),
        },
    }


def _hard_failure_ids(artifacts: dict[str, dict], final_cases: dict) -> set[str]:
    hard_ids = {case_id for case_id, row in final_cases.items() if _is_hard_failure(row)}
    deterministic = _artifact_cases("deterministic_evidence", artifacts["deterministic_evidence"]["data"])
    hard_ids.update(case_id for case_id, row in deterministic.items() if _is_hard_failure(row))
    validation = _artifact_cases("dataset_validation", artifacts["dataset_validation"]["data"])
    hard_ids.update(
        case_id for case_id, row in validation.items()
        if row.get("omit_from_training") or row.get("status") == "invalid"
    )
    return hard_ids


def _policy_safety(final_cases: dict, hard_ids: set[str], reasoning: dict, critique: dict) -> dict:
    observed_hard = hard_ids & set(final_cases)
    preserved = {
        case_id for case_id in observed_hard
        if final_cases[case_id].get("final_decision") not in KEEP_ACTIONS
    }
    hard_basis = {
        case_id for case_id in observed_hard
        if final_cases[case_id].get("decision_basis") == "deterministic_hard_failure"
    }
    return {
        "hard_failure_preservation": {
            **_rule_metric(
                len(preserved),
                len(observed_hard),
                eligible_n=len(hard_ids),
                missing_n=len(hard_ids - set(final_cases)),
            ),
            "hard_failures_inappropriately_kept": sorted(observed_hard - preserved),
        },
        "hard_failure_basis_consistency": {
            **_rule_metric(
                len(hard_basis),
                len(observed_hard),
                eligible_n=len(hard_ids),
                missing_n=len(hard_ids - set(final_cases)),
            ),
            "hard_failures_without_deterministic_basis": sorted(observed_hard - hard_basis),
        },
        "reasoning_decision_non_interference": reasoning["decision_non_interference"],
        "critique_decision_non_interference": critique["decision_non_interference"],
    }


def _task_consistency(context: dict) -> dict:
    pairs = context.get("task_pairs") or context.get("case_task_pairs")
    if not isinstance(pairs, list) or not pairs:
        return {
            "status": "unavailable",
            "reason": "task_pairs_not_provided",
            "pair_consistency": _metric(0, 0),
        }
    evaluations = []
    missing = 0
    for pair in pairs:
        if not isinstance(pair, dict):
            missing += 1
            continue
        if isinstance(pair.get("consistent"), bool):
            evaluations.append(pair["consistent"])
        elif "expected_decision" in pair and "observed_decision" in pair:
            evaluations.append(pair["expected_decision"] == pair["observed_decision"])
        else:
            missing += 1
    return {
        "status": "available" if evaluations else "unavailable",
        "pair_consistency": _metric(
            sum(evaluations),
            len(evaluations),
            eligible_n=len(pairs),
            missing_n=missing,
        ),
    }


def _golden_validation(artifact: dict, final_cases: dict) -> dict:
    if artifact["status"] != "available":
        unavailable_metric = _metric(0, 0)
        return {
            "status": "unavailable",
            "reason": artifact["status"],
            "label_coverage": unavailable_metric,
            "overall_agreement": unavailable_metric,
            "decision_agreement_rate": unavailable_metric,
            "per_action_agreement": {
                action: _metric(0, 0) for action in CANONICAL_ACTIONS
            },
            "confusion_counts": {
                action: {observed: 0 for observed in CANONICAL_ACTIONS}
                for action in CANONICAL_ACTIONS
            },
            "unsafe_keep_count": 0,
            "unsafe_keep_rate": unavailable_metric,
            "review_coverage": unavailable_metric,
        }
    labels = _artifact_cases("golden_labels", artifact["data"])
    eligible_labels = {}
    excluded = 0
    for case_id, row in labels.items():
        if row.get("excluded") or row.get("eligible") is False:
            excluded += 1
            continue
        expected = row.get(
            "expected_action",
            row.get("golden_label", row.get("expected_decision", row.get("label", row.get("value")))),
        )
        if expected is not None:
            eligible_labels[case_id] = str(expected)

    covered_ids = set(eligible_labels) & set(final_cases)
    labels_without_final = set(eligible_labels) - set(final_cases)
    compared = []
    action_rows: dict[str, list[str]] = {action: [] for action in CANONICAL_ACTIONS}
    confusion = {
        action: {observed: 0 for observed in CANONICAL_ACTIONS}
        for action in CANONICAL_ACTIONS
    }
    compared_ids = set()
    for case_id in sorted(covered_ids):
        expected = eligible_labels[case_id]
        observed = (final_cases.get(case_id) or {}).get("final_decision")
        if observed is None:
            continue
        compared_ids.add(case_id)
        compared.append(expected == observed)
        if expected in action_rows:
            action_rows[expected].append(str(observed))
            if observed not in confusion[expected]:
                confusion[expected][str(observed)] = 0
            confusion[expected][str(observed)] += 1

    overall = _metric(
        sum(compared),
        len(compared),
        eligible_n=len(eligible_labels),
        missing_n=len(eligible_labels) - len(compared),
        excluded_n=excluded,
        definition="Exact agreement with supplied expected curation actions; not clinical diagnostic accuracy.",
    )
    per_action = {}
    for action in CANONICAL_ACTIONS:
        observed_actions = action_rows[action]
        missing_action = sum(
            expected == action and case_id not in compared_ids
            for case_id, expected in eligible_labels.items()
        )
        per_action[action] = _metric(
            sum(observed == action for observed in observed_actions),
            len(observed_actions),
            eligible_n=len(observed_actions) + missing_action,
            missing_n=missing_action,
            definition=f"Exact agreement among cases with expected action {action!r}.",
        )

    expected_reject_observed = action_rows["reject"]
    unsafe_keep_count = sum(
        observed in {"keep", "warning"} for observed in expected_reject_observed
    )
    unsafe_keep_rate = _metric(
        unsafe_keep_count,
        len(expected_reject_observed),
        eligible_n=sum(expected == "reject" for expected in eligible_labels.values()),
        missing_n=sum(
            expected == "reject" and case_id not in compared_ids
            for case_id, expected in eligible_labels.items()
        ),
        definition="System keep or warning among cases whose expected action is reject.",
    )
    expected_review_observed = action_rows["review"]
    review_coverage = _metric(
        sum(observed == "review" for observed in expected_review_observed),
        len(expected_review_observed),
        eligible_n=sum(expected == "review" for expected in eligible_labels.values()),
        missing_n=sum(
            expected == "review" and case_id not in compared_ids
            for case_id, expected in eligible_labels.items()
        ),
        definition="System review among cases whose expected action is review.",
    )
    return {
        "status": "available",
        "label_coverage": _metric(
            len(covered_ids),
            len(final_cases),
            eligible_n=len(final_cases),
            missing_n=len(set(final_cases) - covered_ids),
            excluded_n=excluded + len(labels_without_final),
            definition="Fraction of final-decision cases with an eligible expected action.",
        ),
        "overall_agreement": overall,
        # Compatibility alias retained for existing evaluation consumers.
        "decision_agreement_rate": overall,
        "per_action_agreement": per_action,
        "confusion_counts": confusion,
        "unsafe_keep_count": unsafe_keep_count,
        "unsafe_keep_rate": unsafe_keep_rate,
        "review_coverage": review_coverage,
    }


def evaluate(
    final_path: Path | None = None,
    comparison_path: Path | None = None,
    routing_path: Path | None = None,
    *,
    validated_context_path: Path | None = None,
    dataset_validation_path: Path | None = None,
    deterministic_path: Path | None = None,
    calibrated_path: Path | None = None,
    reasoning_path: Path | None = None,
    critique_path: Path | None = None,
    golden_labels_path: Path | None = None,
) -> dict:
    """Audit supplied artifacts without changing or re-deriving final decisions."""
    artifacts = {
        "validated_context": _load(validated_context_path),
        "dataset_validation": _load(dataset_validation_path),
        "deterministic_evidence": _load(deterministic_path),
        "calibrated_evidence": _load(calibrated_path),
        "comparison": _load(comparison_path),
        "reasoning": _load(reasoning_path),
        "critique": _load(critique_path),
        "routing": _load(routing_path),
        "final_decisions": _load(final_path),
        "golden_labels": _load(golden_labels_path),
    }
    final = artifacts["final_decisions"]["data"]
    decisions = _artifact_cases("final_decisions", final)
    comparison = artifacts["comparison"]["data"]
    routing = artifacts["routing"]["data"]
    counts = Counter(row.get("final_decision", "unknown") for row in decisions.values())
    basis = Counter(row.get("decision_basis", "unknown") for row in decisions.values())
    _, reference_ids = _reference_case_ids(artifacts)
    reasoning_validity = _explanation_validity(artifacts["reasoning"], reference_ids, "reasoning")
    critique_validity = _explanation_validity(artifacts["critique"], reference_ids, "critique")
    hard_ids = _hard_failure_ids(artifacts, decisions)

    result = {
        # Compatibility fields retained for existing consumers.
        "dataset_statistics": {
            "n_cases": len(decisions),
            "decision_counts": dict(sorted(counts.items())),
            "decision_basis_counts": dict(sorted(basis.items())),
            "human_review_required": sum(1 for row in decisions.values() if row.get("requires_human_review")),
            "hard_failures": sum(1 for row in decisions.values() if row.get("hard_failure")),
        },
        "calibration_impact": comparison.get("summary", {}),
        "review_routing": routing.get("summary", {}),
        "artifact_consistency": {
            "final_cases": len(decisions),
            "comparison_cases": (comparison.get("summary") or {}).get("n_cases_compared"),
            "routed_cases": (routing.get("summary") or {}).get("n_routed"),
        },
        "artifact_integrity": _artifact_integrity(artifacts),
        "evidence_stability": _evidence_stability(artifacts),
        "reasoning_validity": reasoning_validity,
        "critique_validity": critique_validity,
        "routing_behavior": _routing_behavior(artifacts, decisions, hard_ids),
        "policy_safety": _policy_safety(decisions, hard_ids, reasoning_validity, critique_validity),
        "task_consistency": _task_consistency(artifacts["validated_context"]["data"]),
        "external_validation_limitations": {
            "status": "not_externally_validated",
            "limitations": [
                "This report measures internal artifact and policy consistency only.",
                "Routing metrics do not estimate medical recall, sensitivity, or specificity.",
                "Clinical and cross-dataset validity require independent external evaluation.",
            ],
        },
    }
    if golden_labels_path is not None:
        result["golden_case_validation"] = _golden_validation(artifacts["golden_labels"], decisions)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit QC artifact outputs without changing decisions.")
    parser.add_argument("--validated-context", type=Path, default=None)
    parser.add_argument("--dataset-validation", type=Path, default=None)
    parser.add_argument("--deterministic-evidence", "--deterministic", dest="deterministic_evidence", type=Path, default=None)
    parser.add_argument("--calibrated-evidence", "--calibrated", dest="calibrated_evidence", type=Path, default=None)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--reasoning", type=Path, default=None)
    parser.add_argument("--critique", type=Path, default=None)
    parser.add_argument("--routing", type=Path, default=None)
    parser.add_argument("--final-decisions", type=Path, default=None)
    parser.add_argument("--golden-labels", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    data = evaluate(
        args.final_decisions,
        args.comparison,
        args.routing,
        validated_context_path=args.validated_context,
        dataset_validation_path=args.dataset_validation,
        deterministic_path=args.deterministic_evidence,
        calibrated_path=args.calibrated_evidence,
        reasoning_path=args.reasoning,
        critique_path=args.critique,
        golden_labels_path=args.golden_labels,
    )
    inputs = []
    for path in (
        args.validated_context,
        args.dataset_validation,
        args.deterministic_evidence,
        args.calibrated_evidence,
        args.comparison,
        args.reasoning,
        args.critique,
        args.routing,
        args.final_decisions,
        args.golden_labels,
    ):
        if path is not None:
            inputs.append(artifact_descriptor(path))
    write_artifact(
        args.output,
        artifact_type="eval_report",
        generator="EvaluationAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        input_artifacts=inputs,
        run_id=args.run_id,
        project_root=paths.project_root,
    )
    print(f"Wrote evaluation artifact: {args.output}")


if __name__ == "__main__":
    main()
