#!/usr/bin/env python3
"""MedicalCriticAgent: audit explanation grounding without decision authority."""

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
    load_case_artifact,
    load_mapping_artifact,
    metadata_alignment_issues,
    qc_snapshot,
    require_fields,
    source_reference_map,
)
from artifacts.io import artifact_descriptor, write_artifact


_CRITIQUE_EXACT_FORBIDDEN = {"final_decision", "route_to_review", "policy_action"}
_CRITIQUE_FORBIDDEN_TOKENS = {"decision", "route", "routing", "action", "block", "blocking"}
_CRITIQUE_REQUIRED = {
    "case_id",
    "status",
    "audit_scope",
    "supported_claim_checks",
    "unsupported_claim_checks",
    "concerns",
    "evidence_gaps",
    "severity",
    "limitations",
    "binding",
}
_SEVERITIES = {"none", "low", "medium", "high", "critical"}
_SEVERITY_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}


class DeterministicMedicalCriticBackend:
    """Deterministically check reasoning claims against their source artifacts."""

    name = "deterministic_template"
    frozen = True

    def generate_case(self, case_input: Mapping[str, Any]) -> dict[str, Any]:
        case_id = str(case_input["case_id"])
        reasoning = case_input.get("reasoning")
        references = dict(case_input["evidence_references"])
        expected = {
            "deterministic": qc_snapshot(case_input.get("deterministic")),
            "calibrated": qc_snapshot(case_input.get("calibrated")),
            "comparison": comparison_snapshot(case_input.get("comparison")),
        }
        supported: list[dict[str, Any]] = []
        unsupported: list[dict[str, Any]] = []
        concerns: list[dict[str, Any]] = []
        gaps: list[dict[str, str]] = []
        severity = "none"

        def raise_severity(value: str) -> None:
            nonlocal severity
            if _SEVERITY_RANK[value] > _SEVERITY_RANK[severity]:
                severity = value

        if not isinstance(reasoning, dict):
            gaps.append({
                "code": "missing_reasoning_case",
                "description": f"No reasoning entry is available for case {case_id}.",
            })
            concerns.append({
                "code": "reasoning_unavailable",
                "severity": "high",
                "description": "Grounding and consistency cannot be fully audited without reasoning.",
                "evidence_references": list(references.values()),
            })
            raise_severity("high")
        else:
            summaries = reasoning.get("evidence_summary")
            if not isinstance(summaries, dict):
                summaries = {}
                gaps.append({
                    "code": "missing_reasoning_evidence_summary",
                    "description": "Reasoning does not contain a structured evidence summary.",
                })
                raise_severity("high")
            fields = {
                "deterministic": ("availability", "reported_recommendation", "risk_level", "score", "primary_domain"),
                "calibrated": ("availability", "reported_recommendation", "risk_level", "score", "primary_domain"),
                "comparison": ("availability", "changed", "recommendation_change", "risk_change", "score_delta"),
            }
            for source, source_fields in fields.items():
                claimed = summaries.get(source)
                if not isinstance(claimed, dict):
                    gaps.append({
                        "code": f"missing_{source}_reasoning_summary",
                        "description": f"Reasoning omits its {source} evidence summary.",
                    })
                    raise_severity("high")
                    continue
                for field in source_fields:
                    if field not in claimed and field not in expected[source]:
                        continue
                    check = {
                        "claim": f"{source}.{field}",
                        "reasoning_value": claimed.get(field),
                        "evidence_value": expected[source].get(field),
                        "evidence_reference": references[source],
                    }
                    if claimed.get(field) == expected[source].get(field):
                        check["status"] = "supported"
                        supported.append(check)
                    else:
                        check["status"] = "unsupported"
                        unsupported.append(check)
                        raise_severity("high")

            actual_conflict_fields = {
                field
                for field in ("reported_recommendation", "risk_level", "score", "primary_domain")
                if expected["deterministic"].get("availability") == "available"
                and expected["calibrated"].get("availability") == "available"
                and expected["deterministic"].get(field) != expected["calibrated"].get(field)
            }
            reported_conflicts = reasoning.get("conflicts") or []
            reported_fields = {
                item.get("field")
                for item in reported_conflicts
                if isinstance(item, dict)
            }
            omitted_conflicts = sorted(actual_conflict_fields - reported_fields)
            if omitted_conflicts:
                concerns.append({
                    "code": "unreported_evidence_conflict",
                    "severity": "medium",
                    "description": "Reasoning does not report conflicts for: " + ", ".join(omitted_conflicts) + ".",
                    "evidence_references": [references["deterministic"], references["calibrated"]],
                })
                raise_severity("medium")

        for issue in case_input.get("global_alignment_issues", []):
            concerns.append({
                "code": "artifact_identity_mismatch",
                "severity": "high",
                "description": issue,
                "evidence_references": list(references.values()),
            })
            raise_severity("high")

        for source, snapshot in expected.items():
            if snapshot["availability"] == "missing":
                gaps.append({
                    "code": f"missing_{source}_evidence_case",
                    "description": f"Case {case_id} is absent from the {source} evidence artifact.",
                })
                raise_severity("medium" if source == "comparison" else "high")
        if unsupported:
            concerns.append({
                "code": "unsupported_reasoning_claims",
                "severity": "high",
                "description": f"{len(unsupported)} structured reasoning claim(s) do not match cited evidence.",
                "evidence_references": list(references.values()),
            })

        status = "complete" if isinstance(reasoning, dict) and not gaps else "audit_limited"
        return {
            "case_id": case_id,
            "status": status,
            "audit_scope": "grounding_and_internal_consistency_only",
            "supported_claim_checks": supported,
            "unsupported_claim_checks": unsupported,
            "concerns": concerns,
            "evidence_gaps": gaps,
            "severity": severity,
            "limitations": [
                "The template critic checks structured grounding and consistency only.",
                "It does not inspect images or infer independent medical facts.",
            ],
            "binding": False,
        }


def _validate_case_output(case_id: str, output: Any) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError(f"Medical critic backend output for {case_id} must be an object")
    require_fields(output, _CRITIQUE_REQUIRED, label="Medical critic")
    if str(output.get("case_id")) != case_id:
        raise ValueError(f"Medical critic backend changed case_id {case_id!r}")
    if output.get("severity") not in _SEVERITIES:
        raise ValueError(f"Invalid critique severity for {case_id}: {output.get('severity')!r}")
    if output.get("binding") is not False:
        raise ValueError("Medical critic output must set binding=false")
    assert_no_forbidden_keys(
        output,
        exact=_CRITIQUE_EXACT_FORBIDDEN,
        forbidden_tokens=_CRITIQUE_FORBIDDEN_TOKENS,
    )
    return output


def build_medical_critique_data(
    *,
    reasoning_path: Path,
    deterministic_path: Path,
    calibrated_path: Path,
    comparison_path: Path,
    backend: CaseBackend | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    backend = backend or DeterministicMedicalCriticBackend()
    reasoning_cases, reasoning_metadata = load_case_artifact(reasoning_path)
    deterministic_cases, deterministic_metadata = load_case_artifact(deterministic_path)
    calibrated_cases, calibrated_metadata = load_case_artifact(calibrated_path)
    comparison_metadata, comparison_data = load_mapping_artifact(comparison_path)
    comparison_cases, comparison_positions, comparison_issues = comparison_index(comparison_data)
    identity_issues = metadata_alignment_issues(
        reasoning_metadata,
        {},
        {
            "deterministic": (deterministic_metadata, {}),
            "calibrated": (calibrated_metadata, {}),
            "comparison": (comparison_metadata, comparison_data),
        },
    )
    alignment_issues = sorted(set(comparison_issues + identity_issues))
    case_ids = set(reasoning_cases) | set(deterministic_cases) | set(calibrated_cases) | set(comparison_cases)

    cases: dict[str, dict[str, Any]] = {}
    for case_id in sorted(case_ids):
        references = source_reference_map(
            case_id,
            deterministic_path,
            calibrated_path,
            comparison_path,
            comparison_positions.get(case_id),
        )
        references["reasoning"] = {
            **artifact_descriptor(reasoning_path, "reasoning_artifact"),
            "case_pointer": f"/data/cases/{case_id.replace('~', '~0').replace('/', '~1')}",
        }
        output = backend.generate_case({
            "case_id": case_id,
            "reasoning": reasoning_cases.get(case_id),
            "deterministic": deterministic_cases.get(case_id),
            "calibrated": calibrated_cases.get(case_id),
            "comparison": comparison_cases.get(case_id),
            "evidence_references": references,
            "global_alignment_issues": alignment_issues,
        })
        cases[case_id] = _validate_case_output(case_id, output)

    if not cases:
        status = "insufficient_evidence"
    elif all(case["status"] == "complete" for case in cases.values()) and not alignment_issues:
        status = "complete"
    else:
        status = "partial"
    data = {
        "status": status,
        "backend": {"name": backend.name, "frozen": bool(backend.frozen)},
        "audit_scope": "grounding_and_internal_consistency_only",
        "alignment": {
            "issues": alignment_issues,
            "case_counts": {
                "reasoning": len(reasoning_cases),
                "deterministic": len(deterministic_cases),
                "calibrated": len(calibrated_cases),
                "comparison": len(comparison_cases),
                "output": len(cases),
            },
        },
        "cases": cases,
        "limitations": [
            "This non-binding artifact audits explanation grounding and internal consistency only."
        ],
        "binding": False,
    }
    assert_no_forbidden_keys(
        data,
        exact=_CRITIQUE_EXACT_FORBIDDEN,
        forbidden_tokens=_CRITIQUE_FORBIDDEN_TOKENS,
    )
    return data, reasoning_metadata


def generate_medical_critique_artifact(
    *,
    reasoning_path: Path,
    deterministic_path: Path,
    calibrated_path: Path,
    comparison_path: Path,
    output_path: Path,
    backend: CaseBackend | None = None,
    dataset_name: str | None = None,
    task_mode: str | None = None,
    run_id: str | None = None,
) -> dict:
    data, reasoning_metadata = build_medical_critique_data(
        reasoning_path=reasoning_path,
        deterministic_path=deterministic_path,
        calibrated_path=calibrated_path,
        comparison_path=comparison_path,
        backend=backend,
    )
    backend = backend or DeterministicMedicalCriticBackend()
    return write_artifact(
        output_path,
        artifact_type="medical_critique",
        generator="MedicalCriticAgent",
        data=data,
        dataset_name=dataset_name or reasoning_metadata.get("dataset_name") or "unknown",
        task_mode=task_mode or reasoning_metadata.get("task_mode") or "unknown",
        input_artifacts=[
            artifact_descriptor(reasoning_path, "reasoning_artifact"),
            artifact_descriptor(deterministic_path, "qc_report_deterministic"),
            artifact_descriptor(calibrated_path, "qc_report_calibrated"),
            artifact_descriptor(comparison_path, "qc_comparison"),
        ],
        configuration={
            "backend": backend.name,
            "frozen": bool(backend.frozen),
            "explanatory_only": True,
        },
        run_id=run_id if run_id is not None else reasoning_metadata.get("run_id"),
        project_root=_PROJECT_ROOT,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit reasoning grounding and consistency without making decisions."
    )
    parser.add_argument("--reasoning", required=True, type=Path)
    parser.add_argument("--deterministic", required=True, type=Path)
    parser.add_argument("--calibrated", required=True, type=Path)
    parser.add_argument("--comparison", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--backend", default="template", choices=("template",))
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--task-mode", default=None)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args(argv)
    generate_medical_critique_artifact(
        reasoning_path=args.reasoning,
        deterministic_path=args.deterministic,
        calibrated_path=args.calibrated,
        comparison_path=args.comparison,
        output_path=args.output,
        backend=DeterministicMedicalCriticBackend(),
        dataset_name=args.dataset_name,
        task_mode=args.task_mode,
        run_id=args.run_id,
    )
    print(f"Wrote medical critique artifact: {args.output}")


if __name__ == "__main__":
    main()
