#!/usr/bin/env python3
"""ReasoningAgent: produce evidence-grounded explanations without authority."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.structured_reasoning import (
    CaseBackend,
    assert_no_forbidden_keys,
    comparison_index,
    comparison_snapshot,
    context_status,
    expected_context_case_ids,
    identity_value,
    load_case_artifact,
    load_mapping_artifact,
    metadata_alignment_issues,
    qc_snapshot,
    require_fields,
    source_reference_map,
)
from artifacts.io import artifact_descriptor, write_artifact


_REASONING_FORBIDDEN = {"final_decision", "route_to_review", "policy_action"}
_REASONING_FORBIDDEN_TOKENS = {"decision", "route", "routing", "action"}
_REASONING_REQUIRED = {
    "case_id",
    "status",
    "task_profile_reference",
    "evidence_references",
    "evidence_summary",
    "conflicts",
    "uncertainty_annotations",
    "limitations",
}


def _different(left: Any, right: Any) -> bool:
    return left != right and not (left is None and right is None)


class DeterministicReasoningBackend:
    """Deterministic template backend; a frozen provider can implement CaseBackend."""

    name = "deterministic_template"
    frozen = True

    def generate_case(self, case_input: Mapping[str, Any]) -> dict[str, Any]:
        case_id = str(case_input["case_id"])
        deterministic = qc_snapshot(case_input.get("deterministic"))
        calibrated = qc_snapshot(case_input.get("calibrated"))
        comparison = comparison_snapshot(case_input.get("comparison"))
        references = dict(case_input["evidence_references"])
        missing = [
            source
            for source, snapshot in (
                ("deterministic", deterministic),
                ("calibrated", calibrated),
                ("comparison", comparison),
            )
            if snapshot["availability"] == "missing"
        ]

        conflicts: list[dict[str, Any]] = []
        if deterministic["availability"] == calibrated["availability"] == "available":
            for field in ("reported_recommendation", "risk_level", "score", "primary_domain"):
                if _different(deterministic.get(field), calibrated.get(field)):
                    conflicts.append({
                        "field": field,
                        "deterministic_value": deterministic.get(field),
                        "calibrated_value": calibrated.get(field),
                        "description": "Deterministic and calibrated evidence report different values.",
                        "evidence_references": [references["deterministic"], references["calibrated"]],
                    })

        uncertainty: list[dict[str, Any]] = []
        for source in missing:
            uncertainty.append({
                "code": f"missing_{source}_case",
                "severity": "high" if source != "comparison" else "medium",
                "description": f"Case {case_id} is absent from the {source} artifact.",
                "evidence_references": [references[source]],
            })
        if conflicts:
            uncertainty.append({
                "code": "deterministic_calibrated_conflict",
                "severity": "medium",
                "description": "The evidence variants disagree on one or more reported fields.",
                "evidence_references": [references["deterministic"], references["calibrated"]],
            })
        if comparison["availability"] == "available":
            comparison_changed = bool(comparison.get("changed"))
            if comparison_changed != bool(conflicts):
                uncertainty.append({
                    "code": "comparison_alignment_inconsistency",
                    "severity": "medium",
                    "description": "The comparison changed marker is inconsistent with the aligned evidence summaries.",
                    "evidence_references": [references["comparison"]],
                })
        if not case_input.get("declared_in_context", True):
            uncertainty.append({
                "code": "case_not_declared_in_context",
                "severity": "medium",
                "description": f"Case {case_id} is not listed among the context case identifiers.",
                "evidence_references": [case_input["task_profile_reference"]],
            })
        for issue in case_input.get("global_alignment_issues", []):
            uncertainty.append({
                "code": "artifact_identity_mismatch",
                "severity": "high",
                "description": issue,
                "evidence_references": list(references.values()),
            })

        available_evidence = 2 - sum(source in missing for source in ("deterministic", "calibrated"))
        if available_evidence == 0:
            status = "insufficient_evidence"
        elif missing or uncertainty:
            status = "partial"
        else:
            status = "complete"
        limitations = [
            "Template reasoning summarizes structured artifacts and does not inspect source images.",
            "The explanatory output does not replace deterministic QC or policy evaluation.",
        ]
        if missing:
            limitations.append("Missing case-level inputs limit cross-artifact interpretation.")
        return {
            "case_id": case_id,
            "status": status,
            "task_profile_reference": case_input["task_profile_reference"],
            "evidence_references": references,
            "evidence_summary": {
                "deterministic": deterministic,
                "calibrated": calibrated,
                "comparison": comparison,
            },
            "conflicts": conflicts,
            "uncertainty_annotations": uncertainty,
            "limitations": limitations,
        }


def _validate_case_output(case_id: str, output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError(f"Reasoning backend output for {case_id} must be an object")
    require_fields(output, _REASONING_REQUIRED, label="Reasoning")
    if str(output.get("case_id")) != case_id:
        raise ValueError(f"Reasoning backend changed case_id {case_id!r}")
    assert_no_forbidden_keys(
        output,
        exact=_REASONING_FORBIDDEN,
        forbidden_tokens=_REASONING_FORBIDDEN_TOKENS,
    )
    return output


def build_reasoning_data(
    *,
    context_path: Path,
    deterministic_path: Path,
    calibrated_path: Path,
    comparison_path: Path,
    backend: CaseBackend | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = backend or DeterministicReasoningBackend()
    context_metadata, context_data = load_mapping_artifact(context_path)
    deterministic_cases, deterministic_metadata = load_case_artifact(deterministic_path)
    calibrated_cases, calibrated_metadata = load_case_artifact(calibrated_path)
    comparison_metadata, comparison_data = load_mapping_artifact(comparison_path)
    comparison_cases, comparison_positions, comparison_issues = comparison_index(comparison_data)

    identity_issues = metadata_alignment_issues(
        context_metadata,
        context_data,
        {
            "deterministic": (deterministic_metadata, {}),
            "calibrated": (calibrated_metadata, {}),
            "comparison": (comparison_metadata, comparison_data),
        },
    )
    validation_status = context_status(context_data)
    if validation_status not in {
        "passed",
        "valid",
        "valid_with_warnings",
        "validated",
        "success",
    }:
        identity_issues.append(f"context_validation_status:{validation_status}")
    identity_issues.extend(comparison_issues)
    identity_issues = sorted(set(identity_issues))

    expected_ids = expected_context_case_ids(context_data)
    case_ids = set(deterministic_cases) | set(calibrated_cases) | set(comparison_cases) | expected_ids
    context_reference = {
        **artifact_descriptor(context_path, "validated_run_context"),
        "task_mode": identity_value(context_metadata, context_data, "task_mode"),
    }
    cases: dict[str, dict[str, Any]] = {}
    for case_id in sorted(case_ids):
        references = source_reference_map(
            case_id,
            deterministic_path,
            calibrated_path,
            comparison_path,
            comparison_positions.get(case_id),
        )
        output = backend.generate_case({
            "case_id": case_id,
            "deterministic": deterministic_cases.get(case_id),
            "calibrated": calibrated_cases.get(case_id),
            "comparison": comparison_cases.get(case_id),
            "evidence_references": references,
            "task_profile_reference": context_reference,
            "declared_in_context": not expected_ids or case_id in expected_ids,
            "global_alignment_issues": identity_issues,
        })
        cases[case_id] = _validate_case_output(case_id, output)

    statuses = {case["status"] for case in cases.values()}
    if not cases or statuses == {"insufficient_evidence"}:
        status = "insufficient_evidence"
    elif statuses == {"complete"}:
        status = "complete"
    else:
        status = "partial"
    data = {
        "status": status,
        "backend": {"name": backend.name, "frozen": bool(backend.frozen)},
        "alignment": {
            "context_validation_status": validation_status,
            "issues": identity_issues,
            "case_counts": {
                "context": len(expected_ids),
                "deterministic": len(deterministic_cases),
                "calibrated": len(calibrated_cases),
                "comparison": len(comparison_cases),
                "output": len(cases),
            },
        },
        "cases": cases,
        "limitations": [
            "This artifact is explanatory and has no decision, routing, or policy authority."
        ],
    }
    assert_no_forbidden_keys(
        data,
        exact=_REASONING_FORBIDDEN,
        forbidden_tokens=_REASONING_FORBIDDEN_TOKENS,
    )
    effective_context_metadata = dict(context_metadata)
    for field in ("run_id", "dataset_name", "task_mode"):
        effective_context_metadata.setdefault(field, context_data.get(field))
    return data, effective_context_metadata


def generate_reasoning_artifact(
    *,
    context_path: Path,
    deterministic_path: Path,
    calibrated_path: Path,
    comparison_path: Path,
    output_path: Path,
    backend: CaseBackend | None = None,
    dataset_name: str | None = None,
    task_mode: str | None = None,
    run_id: str | None = None,
) -> dict:
    data, context_metadata = build_reasoning_data(
        context_path=context_path,
        deterministic_path=deterministic_path,
        calibrated_path=calibrated_path,
        comparison_path=comparison_path,
        backend=backend,
    )
    backend = backend or DeterministicReasoningBackend()
    return write_artifact(
        output_path,
        artifact_type="reasoning_artifact",
        generator="ReasoningAgent",
        data=data,
        dataset_name=dataset_name or context_metadata.get("dataset_name") or "unknown",
        task_mode=task_mode or context_metadata.get("task_mode") or "unknown",
        input_artifacts=[
            artifact_descriptor(context_path, "validated_run_context"),
            artifact_descriptor(deterministic_path, "qc_report_deterministic"),
            artifact_descriptor(calibrated_path, "qc_report_calibrated"),
            artifact_descriptor(comparison_path, "qc_comparison"),
        ],
        configuration={
            "backend": backend.name,
            "frozen": bool(backend.frozen),
            "explanatory_only": True,
        },
        run_id=run_id if run_id is not None else context_metadata.get("run_id"),
        project_root=_PROJECT_ROOT,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Generate an explanatory reasoning artifact from validated QC evidence."
    )
    parser.add_argument("--context", required=True, type=Path)
    parser.add_argument("--deterministic", required=True, type=Path)
    parser.add_argument("--calibrated", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="template", choices=("template",))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--task-mode", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    generate_reasoning_artifact(
        context_path=args.context,
        deterministic_path=args.deterministic,
        calibrated_path=args.calibrated,
        comparison_path=args.comparison,
        output_path=args.output,
        backend=DeterministicReasoningBackend(),
        dataset_name=args.dataset_name,
        task_mode=args.task_mode,
        run_id=args.run_id,
    )
    print(f"Wrote reasoning artifact: {args.output}")


if __name__ == "__main__":
    main()
