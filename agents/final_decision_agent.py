#!/usr/bin/env python3
"""FinalDecisionAgent: deterministic final QC decision composition."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.deterministic_policy import FirstMatchPolicy, PolicyError, load_first_match_policy
from agents.utils.deterministic_evidence import (
    REVIEW_RISKS,
    REVIEW_SEVERITIES,
    comparison_case_map,
    dataset_validation_hard_failure_rules,
    deterministic_hard_failure_rules,
    domain_severity,
    evidence_disagrees,
    has_signal,
    load_artifact_data,
    load_qc_case_map,
    policy_domains,
    primary_domain,
    triage,
)
from agents.utils.paths import resolve_project_paths, VALID_TASK_MODES
from agents.utils.task_profiles import load_task_profile
from artifacts import resource_descriptor
from artifacts.io import artifact_descriptor, write_artifact

POLICY_VERSION = "v1.0"
VISIBLE_PANCREAS_POLICY_PATH = _PROJECT_ROOT / "configs" / "policies" / "visible_pancreas_v1.yaml"
VISIBLE_PANCREAS_POLICY_INPUTS = {"validation", "deterministic", "calibrated", "comparison", "routing"}
_ACTION_MAP = {
    "keep": "keep",
    "keep_with_metadata_warning": "warning",
    "keep_with_warning": "warning",
    "warning": "warning",
    "review": "review",
    "exclude": "reject",
    "omit_from_training": "reject",
    "reject": "reject",
    "insufficient_evidence": "insufficient_evidence",
}


def _normalize_action(recommendation: str | None) -> str:
    return _ACTION_MAP.get(str(recommendation).lower(), "insufficient_evidence") if recommendation is not None else "insufficient_evidence"


def _routing_cases(routing: dict) -> dict[str, dict]:
    cases = routing.get("cases")
    if isinstance(cases, dict):
        return {str(case_id): row for case_id, row in cases.items() if isinstance(row, dict)}
    records: dict[str, dict] = {}
    for key in ("routed_cases", "not_routed_cases"):
        for row in routing.get(key, []) or []:
            if isinstance(row, dict) and row.get("case_id") is not None:
                records[str(row["case_id"])] = row
    return records


def _review_rules(det: dict, cal: dict, soft_domains: set[str]) -> list[str]:
    rules: list[str] = []
    for source, case in (("deterministic", det), ("calibrated", cal)):
        case_triage = triage(case)
        if _normalize_action(case_triage.get("recommendation")) == "review":
            rules.append(f"{source}_recommendation_review")
        risk = case_triage.get("risk_level")
        if risk in REVIEW_RISKS:
            rules.append(f"{source}_{risk}_risk")
        for domain in sorted(soft_domains):
            severity = domain_severity(case, domain)
            if severity in REVIEW_SEVERITIES:
                rules.append(f"{source}_soft_domain:{domain}:{severity}")
    if evidence_disagrees(det, cal):
        rules.append("deterministic_calibrated_disagreement")
    if has_signal(det, "uncertainty", "uncertain", "evidence_uncertainty"):
        rules.append("deterministic_uncertainty")
    if has_signal(cal, "uncertainty", "uncertain", "evidence_uncertainty"):
        rules.append("calibrated_uncertainty")
    if has_signal(det, "evidence_disagreement") or has_signal(cal, "evidence_disagreement"):
        rules.append("evidence_disagreement")
    return rules


def _decide_legacy(
    det_cases: dict,
    cal_cases: dict,
    routing: dict,
    dataset_validation_cases: dict | None = None,
    *,
    task_mode: str | None = None,
    profile: dict | None = None,
) -> dict:
    dataset_validation_cases = dataset_validation_cases or {}
    routing_cases = _routing_cases(routing)
    hard_domains, soft_domains = policy_domains(task_mode, profile)
    decisions = {}
    case_ids = set(det_cases) | set(cal_cases) | set(routing_cases) | set(dataset_validation_cases)
    for case_id in sorted(case_ids):
        det_present = case_id in det_cases and isinstance(det_cases[case_id], dict) and bool(det_cases[case_id])
        cal_present = case_id in cal_cases and isinstance(cal_cases[case_id], dict) and bool(cal_cases[case_id])
        det = det_cases.get(case_id) if det_present else {}
        cal = cal_cases.get(case_id) if cal_present else {}
        validation = dataset_validation_cases.get(case_id) or {}
        det_tri = triage(det)
        cal_tri = triage(cal)
        domain = primary_domain(det) or primary_domain(cal)
        rules = dataset_validation_hard_failure_rules(validation)
        if det_present:
            rules.extend(deterministic_hard_failure_rules(det, hard_domains))
        hard_failure = bool(rules)
        missing = [source for source, present in (("deterministic", det_present), ("calibrated", cal_present)) if not present]
        route_record = routing_cases.get(case_id) or {}

        if hard_failure:
            final = "reject"
            basis = "deterministic_hard_failure"
            routed = False
        elif missing:
            final = "insufficient_evidence"
            basis = "insufficient_evidence"
            routed = False
            rules.extend(f"insufficient_evidence:missing_{source}_case" for source in missing)
        else:
            review_rules = _review_rules(det, cal, soft_domains)
            routed = bool(route_record.get("route_to_review")) or bool(review_rules)
            if route_record.get("route_to_review"):
                rules.extend(route_record.get("policy_rules_triggered") or route_record.get("route_reasons") or ["review_routing"])
            rules.extend(review_rules)
        if not hard_failure and not missing and routed:
            final = "review"
            basis = "review_routing"
        elif not hard_failure and not missing:
            final = _normalize_action(cal_tri.get("recommendation"))
            basis = "calibrated_qc"
            rules.append(f"calibrated_recommendation_mapped:{cal_tri.get('recommendation')}->{final}")
            if final == "insufficient_evidence":
                basis = "insufficient_evidence"
                rules.append("insufficient_evidence:unrecognized_calibrated_recommendation")

        requires_human_review = final == "review"
        decisions[case_id] = {
            "final_decision": final,
            "decision_basis": basis,
            "hard_failure": hard_failure,
            "requires_human_review": requires_human_review,
            "route_to_review": requires_human_review,
            "deterministic_recommendation": det_tri.get("recommendation"),
            "calibrated_recommendation": cal_tri.get("recommendation"),
            "dataset_validation_status": validation.get("status"),
            "dataset_validation_omit_from_training": validation.get("omit_from_training"),
            "risk_level": cal_tri.get("risk_level") or det_tri.get("risk_level"),
            "primary_domain": domain,
            "policy_version": POLICY_VERSION,
            "policy_rules_triggered": sorted(set(rules)),
        }
    counts = Counter(v["final_decision"] for v in decisions.values())
    return {
        "policy_version": POLICY_VERSION,
        "summary": {"n_cases": len(decisions), "decision_counts": dict(sorted(counts.items()))},
        "decisions": decisions,
    }


def _domain_complete(case: dict, domain: str) -> bool:
    info = (case.get("qc_domains") or {}).get(domain)
    return isinstance(info, dict) and isinstance(info.get("severity"), str)


def _non_blocking_warning(case: dict) -> bool:
    if _normalize_action(triage(case).get("recommendation")) == "warning":
        return True
    return any(
        isinstance(info, dict)
        and info.get("enabled", True) is not False
        and info.get("severity") == "low_warning"
        for info in (case.get("qc_domains") or {}).values()
    )


def _visible_pancreas_context(
    *,
    det: dict,
    cal: dict,
    validation: dict,
    comparison: dict,
    routing: dict,
    det_present: bool,
    cal_present: bool,
) -> dict:
    det_triage = triage(det)
    measurements = det.get("measurements") or {}
    target_presence = det.get("target_presence") or {}
    geometry_complete = _domain_complete(det, "geometry_integrity")
    geometry_severity = ((det.get("qc_domains") or {}).get("geometry_integrity") or {}).get("severity")
    geometry_hard_failure = geometry_complete and geometry_severity == "critical"
    geometry_valid = not geometry_hard_failure if geometry_complete else None
    fov_domain_complete = _domain_complete(det, "fov_integrity")
    fov_assessed = any(
        key in measurements
        for key in ("border_touching", "partial_visibility_likely", "anatomical_truncation_suspected")
    )
    fov_complete = fov_domain_complete and fov_assessed

    annotation_presence = target_presence.get("annotation_presence")
    measured_pancreas_present = measurements.get("pancreas_present")
    if annotation_presence in {"present", "absent"}:
        mask_present = annotation_presence == "present"
    elif isinstance(measured_pancreas_present, bool):
        mask_present = measured_pancreas_present
    else:
        mask_present = None
    target_complete = (
        target_presence.get("expected_presence") == "required"
        and target_presence.get("observed_presence") in {"present", "partial", "absent", "uncertain"}
        and target_presence.get("visible_target_annotation_status") in {"complete", "incomplete", "uncertain"}
    )
    blocking = (
        det_triage.get("recommendation") == "omit_from_training"
        or (det.get("case_status") or {}).get("omit_from_training") is True
        or det.get("blocking_processing_error") is True
        or measurements.get("blocking_processing_error") is True
    )
    explicit_readable = det.get("image_readable", measurements.get("image_readable"))
    image_readable = (
        explicit_readable
        if isinstance(explicit_readable, bool)
        else (not blocking if det_present else None)
    )

    recommendation_change = comparison.get("recommendation_changed")
    if not isinstance(recommendation_change, bool):
        recommendation_change = comparison.get("recommendation_change") not in {None, "unchanged"}
    risk_change = comparison.get("risk_level_changed")
    if not isinstance(risk_change, bool):
        risk_change = comparison.get("risk_change") not in {None, "unchanged"}

    return {
        "validation": {
            "omit_from_training": validation.get("omit_from_training") is True,
            "status_invalid": str(validation.get("status", "")).strip().lower() == "invalid",
        },
        "deterministic": {
            "present": det_present,
            "blocking_processing_error": blocking,
            "image_readable": image_readable,
            "geometry": {
                "complete": geometry_complete,
                "valid": geometry_valid,
                "hard_failure": geometry_hard_failure,
            },
            "fov": {"complete": fov_complete},
            "non_blocking_warning": _non_blocking_warning(det),
            "target": {
                "evidence_complete": target_complete,
                "expected_presence": target_presence.get("expected_presence"),
                "observed_presence": target_presence.get("observed_presence"),
                "visible_target_annotation_status": target_presence.get("visible_target_annotation_status"),
                "mask_present": mask_present,
                "mask_empty": not mask_present if isinstance(mask_present, bool) else None,
                "mask_valid": mask_present is True and geometry_valid is True,
            },
        },
        "calibrated": {
            "present": cal_present,
            "non_blocking_warning": _non_blocking_warning(cal),
        },
        "comparison": {
            "recommendation_changed": recommendation_change,
            "risk_level_changed": risk_change,
        },
        "routing": {"route_to_review": routing.get("route_to_review") is True},
    }


def _decide_visible_pancreas(
    det_cases: dict,
    cal_cases: dict,
    routing: dict,
    dataset_validation_cases: dict,
    comparison: dict,
    policy: FirstMatchPolicy,
) -> dict:
    routing_cases = _routing_cases(routing)
    comparison_cases = comparison_case_map(comparison)
    case_ids = set(det_cases) | set(cal_cases) | set(routing_cases) | set(dataset_validation_cases) | set(comparison_cases)
    decisions = {}
    hard_rule_ids = {f"VP-{index:03d}" for index in range(1, 7)}
    for case_id in sorted(case_ids):
        det_present = case_id in det_cases and isinstance(det_cases[case_id], dict) and bool(det_cases[case_id])
        cal_present = case_id in cal_cases and isinstance(cal_cases[case_id], dict) and bool(cal_cases[case_id])
        det = det_cases.get(case_id) if det_present else {}
        cal = cal_cases.get(case_id) if cal_present else {}
        validation = dataset_validation_cases.get(case_id) or {}
        route_record = routing_cases.get(case_id) or {}
        policy_decision = policy.evaluate(_visible_pancreas_context(
            det=det,
            cal=cal,
            validation=validation,
            comparison=comparison_cases.get(case_id) or {},
            routing=route_record,
            det_present=det_present,
            cal_present=cal_present,
        ))
        if policy_decision.rule_id is None:
            raise PolicyError(f"No visible-pancreas policy rule matched case {case_id}")
        final = policy_decision.action
        hard_failure = policy_decision.rule_id in hard_rule_ids
        if hard_failure:
            basis = "deterministic_hard_failure"
        elif final == "insufficient_evidence":
            basis = "insufficient_evidence"
        elif final == "review":
            basis = "review_routing"
        else:
            basis = "calibrated_qc"
        det_tri = triage(det)
        cal_tri = triage(cal)
        decisions[case_id] = {
            "final_decision": final,
            "decision_basis": basis,
            "hard_failure": hard_failure,
            "requires_human_review": final == "review",
            "route_to_review": final == "review",
            "deterministic_recommendation": det_tri.get("recommendation"),
            "calibrated_recommendation": cal_tri.get("recommendation"),
            "dataset_validation_status": validation.get("status"),
            "dataset_validation_omit_from_training": validation.get("omit_from_training"),
            "risk_level": cal_tri.get("risk_level") or det_tri.get("risk_level"),
            "primary_domain": primary_domain(det) or primary_domain(cal),
            "policy_id": policy_decision.policy_id,
            "policy_version": policy_decision.policy_version,
            "policy_rules_triggered": [policy_decision.rule_id],
            "decision_trace": policy_decision.trace(),
        }
    counts = Counter(row["final_decision"] for row in decisions.values())
    return {
        "policy_id": policy.policy_id,
        "policy_version": policy.version,
        "summary": {"n_cases": len(decisions), "decision_counts": dict(sorted(counts.items()))},
        "decisions": decisions,
    }


def decide(
    det_cases: dict,
    cal_cases: dict,
    routing: dict,
    dataset_validation_cases: dict | None = None,
    *,
    comparison: dict | None = None,
    task_mode: str | None = None,
    profile: dict | None = None,
    policy_path: Path | None = None,
) -> dict:
    if task_mode != "pancreas_only":
        return _decide_legacy(
            det_cases,
            cal_cases,
            routing,
            dataset_validation_cases,
            task_mode=task_mode,
            profile=profile,
        )
    selected_policy = load_first_match_policy(
        policy_path or VISIBLE_PANCREAS_POLICY_PATH,
        allowed_roots=VISIBLE_PANCREAS_POLICY_INPUTS,
    )
    return _decide_visible_pancreas(
        det_cases,
        cal_cases,
        routing,
        dataset_validation_cases or {},
        comparison or {},
        selected_policy,
    )


def _write_csv(path: Path, decisions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "final_decision", "decision_basis", "hard_failure", "requires_human_review", "route_to_review", "deterministic_recommendation", "calibrated_recommendation", "dataset_validation_status", "dataset_validation_omit_from_training", "risk_level", "primary_domain", "policy_id", "policy_version", "matched_rule_id", "matched_rule_name", "matched_rule_rationale", "policy_rules_triggered"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for case_id, row in decisions.items():
            out = {"case_id": case_id, **row}
            matched_rule = (out.pop("decision_trace", None) or {}).get("matched_rule") or {}
            out["matched_rule_id"] = matched_rule.get("id")
            out["matched_rule_name"] = matched_rule.get("name")
            out["matched_rule_rationale"] = matched_rule.get("rationale")
            out["policy_rules_triggered"] = ";".join(out.get("policy_rules_triggered") or [])
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compose final deterministic QC decisions.")
    parser.add_argument("--deterministic", required=True, type=Path)
    parser.add_argument("--calibrated", required=True, type=Path)
    parser.add_argument("--routing", required=True, type=Path)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--dataset-validation", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--policy",
        type=Path,
        default=None,
        help="Immutable decision-policy snapshot (required for orchestrated pancreas-only runs).",
    )
    args = parser.parse_args()
    if args.task_mode == "pancreas_only" and args.policy is None:
        parser.error("--policy is required for pancreas_only CLI execution")
    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    profile = load_task_profile(args.task_mode, paths.project_root)
    data = decide(
        load_qc_case_map(args.deterministic),
        load_qc_case_map(args.calibrated),
        load_artifact_data(args.routing),
        load_qc_case_map(args.dataset_validation) if args.dataset_validation else None,
        comparison=load_artifact_data(args.comparison) if args.comparison else None,
        task_mode=args.task_mode,
        profile=profile,
        policy_path=args.policy,
    )
    input_artifacts = [
        artifact_descriptor(args.deterministic),
        artifact_descriptor(args.calibrated),
        artifact_descriptor(args.routing),
    ]
    if args.dataset_validation:
        input_artifacts.append(artifact_descriptor(args.dataset_validation, "dataset_validation"))
    if args.comparison:
        input_artifacts.append(artifact_descriptor(args.comparison, "qc_comparison"))
    input_resources = []
    configuration = {}
    if args.task_mode == "pancreas_only":
        selected_policy_path = (args.policy or VISIBLE_PANCREAS_POLICY_PATH).resolve()
        input_resources.append(resource_descriptor(selected_policy_path, "decision_policy"))
        configuration["decision_policy"] = {
            "path": str(selected_policy_path),
            "id": data["policy_id"],
            "version": data["policy_version"],
        }
    write_artifact(
        args.output,
        artifact_type="final_qc_decisions",
        generator="FinalDecisionAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        input_resources=input_resources,
        input_artifacts=input_artifacts,
        configuration=configuration,
        run_id=args.run_id,
        project_root=paths.project_root,
    )
    _write_csv(args.csv or args.output.with_suffix(".csv"), data["decisions"])
    print(f"Wrote final decision artifact: {args.output}")


if __name__ == "__main__":
    main()
