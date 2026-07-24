#!/usr/bin/env python3
"""ReviewRoutingAgent: deterministic routing of uncertain soft QC cases."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

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
from artifacts.io import artifact_descriptor, write_artifact

POLICY_VERSION = "v1.0"


def _partial_visible_target_annotation_uncertain(case: dict) -> bool:
    target = case.get("target_presence") or {}
    return (
        target.get("observed_presence") == "partial"
        and target.get("visible_target_annotation_status") == "uncertain"
    )


def _required_pancreas_mask_missing_or_empty(case: dict) -> bool:
    target = case.get("target_presence") or {}
    measurements = case.get("measurements") or {}
    return target.get("expected_presence") == "required" and (
        target.get("annotation_presence") == "absent"
        or measurements.get("pancreas_present") is False
    )


def route(
    det_cases: dict,
    cal_cases: dict,
    comparison: dict,
    dataset_validation_cases: dict | None = None,
    *,
    task_mode: str | None = None,
    profile: dict | None = None,
) -> dict:
    dataset_validation_cases = dataset_validation_cases or {}
    comparison_rows = comparison_case_map(comparison)
    changed_case_ids = {
        str(row.get("case_id"))
        for row in comparison.get("changed_cases", []) or []
        if isinstance(row, dict)
    }
    hard_domains, soft_domains = policy_domains(task_mode, profile)
    records: dict[str, dict] = {}
    case_ids = set(det_cases) | set(cal_cases) | set(comparison_rows) | set(dataset_validation_cases)
    for case_id in sorted(case_ids):
        det_present = case_id in det_cases and isinstance(det_cases[case_id], dict) and bool(det_cases[case_id])
        cal_present = case_id in cal_cases and isinstance(cal_cases[case_id], dict) and bool(cal_cases[case_id])
        det = det_cases.get(case_id) if det_present else {}
        cal = cal_cases.get(case_id) if cal_present else {}
        validation = dataset_validation_cases.get(case_id) or {}
        domain = primary_domain(det) or primary_domain(cal)
        rules = dataset_validation_hard_failure_rules(validation)
        if det_present:
            rules.extend(deterministic_hard_failure_rules(det, hard_domains))
            if task_mode == "pancreas_only" and _required_pancreas_mask_missing_or_empty(det):
                rules.append("deterministic_required_pancreas_mask_missing_or_empty")
        hard_failure = bool(rules)
        missing = [source for source, present in (("deterministic", det_present), ("calibrated", cal_present)) if not present]

        if hard_failure:
            status = "hard_failure"
            route_reasons: list[str] = []
        elif missing:
            rules.extend(f"insufficient_evidence:missing_{source}_case" for source in missing)
            status = "insufficient_evidence"
            route_reasons = []
        else:
            route_reasons = []
            comparison_row = comparison_rows.get(case_id) or {}
            if (
                evidence_disagrees(det, cal)
                or comparison_row.get("changed") is True
                or comparison_row.get("evidence_disagreement") is True
                or case_id in changed_case_ids
            ):
                route_reasons.append("deterministic_calibrated_disagreement")
            if has_signal(det, "uncertainty", "uncertain", "evidence_uncertainty"):
                route_reasons.append("deterministic_uncertainty")
            if has_signal(cal, "uncertainty", "uncertain", "evidence_uncertainty"):
                route_reasons.append("calibrated_uncertainty")
            if has_signal(comparison_row, "uncertainty", "uncertain", "evidence_uncertainty"):
                route_reasons.append("comparison_uncertainty")
            if has_signal(det, "evidence_disagreement") or has_signal(cal, "evidence_disagreement"):
                route_reasons.append("evidence_disagreement")
            if task_mode == "pancreas_only" and _partial_visible_target_annotation_uncertain(det):
                route_reasons.append("partial_visible_target_annotation_uncertain")
            if triage(det).get("recommendation") == "review":
                route_reasons.append("deterministic_recommendation_review")
            if triage(cal).get("recommendation") == "review":
                route_reasons.append("calibrated_recommendation_review")
            for source, case in (("deterministic", det), ("calibrated", cal)):
                risk = triage(case).get("risk_level")
                if risk in REVIEW_RISKS:
                    route_reasons.append(f"{source}_{risk}_risk")
                for soft_domain in sorted(soft_domains):
                    severity = domain_severity(case, soft_domain)
                    if severity in REVIEW_SEVERITIES:
                        route_reasons.append(f"{source}_soft_domain:{soft_domain}:{severity}")
            route_reasons = sorted(set(route_reasons))
            rules.extend(route_reasons)
            status = "routed" if route_reasons else "not_routed"

        record = {
            "case_id": case_id,
            "status": status,
            "route_to_review": status == "routed",
            "route_reasons": route_reasons,
            "deterministic_recommendation": triage(det).get("recommendation"),
            "calibrated_recommendation": triage(cal).get("recommendation"),
            "dataset_validation_status": validation.get("status"),
            "dataset_validation_omit_from_training": validation.get("omit_from_training"),
            "primary_domain": domain,
            "hard_failure": hard_failure,
            "evidence_status": "insufficient_evidence" if missing else "complete",
            "policy_version": POLICY_VERSION,
            "policy_rules_triggered": sorted(set(rules)),
        }
        records[case_id] = record

    routed = [record for record in records.values() if record["route_to_review"]]
    not_routed = [record for record in records.values() if not record["route_to_review"]]
    return {
        "policy_version": POLICY_VERSION,
        "summary": {
            "n_cases": len(records),
            "n_routed": len(routed),
            "n_not_routed": len(not_routed),
            "n_hard_failures": sum(record["hard_failure"] for record in records.values()),
            "n_insufficient_evidence": sum(record["evidence_status"] == "insufficient_evidence" for record in records.values()),
        },
        "cases": records,
        "routed_cases": routed,
        "not_routed_cases": not_routed,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["case_id", "status", "route_to_review", "route_reasons", "deterministic_recommendation", "calibrated_recommendation", "dataset_validation_status", "dataset_validation_omit_from_training", "primary_domain", "hard_failure", "evidence_status", "policy_version", "policy_rules_triggered"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["route_reasons"] = ";".join(out.get("route_reasons") or [])
            out["policy_rules_triggered"] = ";".join(out.get("policy_rules_triggered") or [])
            writer.writerow(out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Route uncertain QC cases for optional review.")
    parser.add_argument("--deterministic", required=True, type=Path)
    parser.add_argument("--calibrated", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--dataset-validation", type=Path, default=None)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    profile = load_task_profile(args.task_mode, paths.project_root)
    data = route(
        load_qc_case_map(args.deterministic),
        load_qc_case_map(args.calibrated),
        load_artifact_data(args.comparison),
        load_qc_case_map(args.dataset_validation) if args.dataset_validation else None,
        task_mode=args.task_mode,
        profile=profile,
    )
    input_artifacts = [
        artifact_descriptor(args.deterministic),
        artifact_descriptor(args.calibrated),
        artifact_descriptor(args.comparison),
    ]
    if args.dataset_validation:
        input_artifacts.append(artifact_descriptor(args.dataset_validation, "dataset_validation"))
    write_artifact(
        args.output,
        artifact_type="review_routing",
        generator="ReviewRoutingAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        input_artifacts=input_artifacts,
        run_id=args.run_id,
        project_root=paths.project_root,
    )
    _write_csv(args.csv or args.output.with_name("manual_review_queue.csv"), data["routed_cases"])
    print(f"Wrote review routing artifact: {args.output}")


if __name__ == "__main__":
    main()
