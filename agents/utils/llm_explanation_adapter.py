"""Deterministic Evidence Abstraction Adapter A_tau and output validator C_tau."""

from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path
from typing import Any

from agents.utils.explanation_contracts import (
    CANONICAL_ACTIONS,
    EXPLANATION_OUTPUT_KEYS,
    NOT_APPLICABLE,
    SCHEMA_VERSION,
    SUPPORTED_TASKS,
    TERMINAL_CORRUPTION_RE,
    UNKNOWN,
    UNSUPPORTED_CAUSAL_RE,
    UNSUPPORTED_MEDICAL_RE,
    assert_no_authority_fields,
    contains_path,
    iter_numbers,
    iter_strings,
    strip_ansi,
)
from agents.utils.llm_backend import ChatBackend, ProviderError
from agents.utils.run_artifact_index import ADMIN_ARTIFACTS, RunArtifactIndex


PROMPT_TEMPLATE = """You are the advisory explanation component of AgentQC.

Use only the supplied compact structured evidence.

Return ONLY valid JSON.
Do not output Markdown.
Do not output headings or introductory text.
Do not repeat the complete input.
Do not alter identifiers, values, task names, routing reasons, or final actions.
Do not calculate measurements.
Do not infer missing medical facts.
Do not infer the cause of an error unless explicitly supported.
Do not modify routing or deterministic policy.
Do not claim medical correctness.
Do not expose filesystem paths.
If the evidence is incomplete, malformed, or contradictory, state this explicitly.

The deterministic policy is authoritative.

Return exactly these keys:
{
  "case_id": "...",
  "evidence_summary": "...",
  "reason_for_review_or_action": "...",
  "unresolved_items": [],
  "possible_consistency_questions": [],
  "limitations": [],
  "explicit_statement_that_deterministic_policy_is_authoritative": "..."
}

Maximum explanation length: 250 words.
"""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_explanation_index(run_dir: Path) -> RunArtifactIndex:
    return RunArtifactIndex.load_admin(
        run_dir,
        allowed_task_modes=set(SUPPORTED_TASKS),
        verify_pancreas_policy=False,
    )


def _value_or_na(group: str, key: str, measurements: dict[str, Any], applicable: set[str]) -> Any:
    if group not in applicable:
        return NOT_APPLICABLE
    return measurements[key] if key in measurements else None


def _domain(case: dict[str, Any], domain: str) -> dict[str, Any] | str:
    value = (case.get("qc_domains") or {}).get(domain)
    return value if isinstance(value, dict) else NOT_APPLICABLE


def _abnormal_domains(case: dict[str, Any]) -> list[dict[str, Any]]:
    abnormal = []
    for domain, value in sorted((case.get("qc_domains") or {}).items()):
        if isinstance(value, dict) and value.get("severity") not in {None, "normal", "not_applicable"}:
            abnormal.append({
                "domain": domain,
                "severity": value.get("severity"),
                "component_score": value.get("component_score"),
            })
    return abnormal


def _critique_summary(critique: dict[str, Any]) -> dict[str, Any]:
    supported = critique.get("supported_claim_checks") or []
    unsupported = critique.get("unsupported_claim_checks") or []
    concerns = critique.get("concerns") or []
    contradictory = [item for item in concerns if isinstance(item, dict) and "contradict" in str(item).lower()]
    return {
        "status": critique.get("status"),
        "severity": critique.get("severity"),
        "binding": critique.get("binding"),
        "supported_claim_count": len(supported) if isinstance(supported, list) else 0,
        "unsupported_claim_count": len(unsupported) if isinstance(unsupported, list) else 0,
        "contradictory_claim_count": len(contradictory),
        "evidence_gap_count": len(critique.get("evidence_gaps") or []),
    }


def _provenance_refs(index: RunArtifactIndex, case_id: str) -> list[dict[str, Any]]:
    refs = []
    for name, (_filename, _artifact_type) in ADMIN_ARTIFACTS.items():
        if name == "evaluation":
            pointer = "/data"
        else:
            pointer = index.pointer(name, case_id)
        refs.append({
            "reference_id": name,
            "artifact_id": index.metadata.get(name, {}).get("artifact_id"),
            "json_pointer": pointer,
        })
    return refs


def build_compact_evidence(index: RunArtifactIndex, case_id: str) -> dict[str, Any]:
    task_id = index.metadata["validated_context"].get("task_mode")
    if task_id not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task mode for explanation: {task_id}")
    task = SUPPORTED_TASKS[str(task_id)]
    applicable = set(task["applicable_groups"])
    det = index.case("deterministic", case_id) or {}
    cal = index.case("calibrated", case_id) or {}
    routing = index.case("routing", case_id) or {}
    decision = index.case("final", case_id) or {}
    reasoning = index.case("reasoning", case_id) or {}
    critique = index.case("critique", case_id) or {}
    measurements = det.get("measurements") if isinstance(det.get("measurements"), dict) else {}
    route_reasons = routing.get("route_reasons") or decision.get("policy_rules_triggered") or []
    visualization_limitation = "No canonical validated visualization artifact was available to the adapter."
    compact = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "scientific_task_name": task["scientific_task_name"],
        "task_notation": task["task_notation"],
        "implementation_task_id": task_id,
        "task_specific_evidence_applicability": {
            "pancreas": "applicable",
            "geometry": "applicable",
            "fov": "applicable",
            "metadata": "applicable",
            "lesion": "applicable" if "lesion" in applicable else NOT_APPLICABLE,
            "subregions": "applicable" if "subregions" in applicable else NOT_APPLICABLE,
        },
        "observed_final_policy_action": decision.get("final_decision"),
        "observed_deterministic_recommendation": decision.get("deterministic_recommendation"),
        "observed_calibrated_recommendation": decision.get("calibrated_recommendation"),
        "final_action": decision.get("final_decision"),
        "decision_basis": decision.get("decision_basis"),
        "requires_human_review": decision.get("requires_human_review"),
        "deterministic_recommendation": decision.get("deterministic_recommendation"),
        "route_to_review": routing.get("route_to_review", decision.get("route_to_review")),
        "risk_level": decision.get("risk_level"),
        "score": (det.get("triage") or {}).get("score"),
        "primary_domain": decision.get("primary_domain"),
        "primary_failure_mode": det.get("primary_failure_mode") or (det.get("primary_issue") or {}).get("failure_mode"),
        "route_reasons": route_reasons if isinstance(route_reasons, list) else [],
        "policy_rules_triggered": decision.get("policy_rules_triggered") or routing.get("policy_rules_triggered") or [],
        "abnormal_domains": _abnormal_domains(det),
        "critical_findings": [
            item for item in _abnormal_domains(det) if item.get("severity") in {"high_warning", "critical"}
        ],
        "relevant_measurements": {
            "pancreas_presence": _value_or_na("pancreas", "pancreas_present", measurements, applicable),
            "pancreas_geometry": _domain(det, "geometry_integrity"),
            "pancreas_volume_mm3": _value_or_na("pancreas", "pancreas_volume_mm3", measurements, applicable),
            "field_of_view_integrity": _domain(det, "fov_integrity"),
            "metadata_completeness": _domain(det, "metadata_completeness"),
            "pancreas_annotation_status": (det.get("target_presence") or {}).get("visible_target_annotation_status"),
            "lesion_presence": _value_or_na("lesion", "lesion_present", measurements, applicable),
            "lesion_volume_mm3": _value_or_na("lesion", "lesion_volume_mm3", measurements, applicable),
            "lesion_pancreas_overlap": _value_or_na("lesion", "lesion_pancreas_overlap", measurements, applicable),
            "lesion_localization": _domain(det, "lesion_localization") if "lesion" in applicable else NOT_APPLICABLE,
            "lesion_annotation_consistency": _domain(det, "lesion_burden") if "lesion" in applicable else NOT_APPLICABLE,
            "lesion_attenuation_evidence": _domain(det, "attenuation_integrity") if "lesion" in applicable else NOT_APPLICABLE,
            "head_body_tail_annotation_availability": _value_or_na("subregions", "subregion_masks_present", measurements, applicable),
            "subregion_completeness": _domain(det, "region_consistency") if "subregions" in applicable else NOT_APPLICABLE,
            "subregion_overlap": _value_or_na("subregions", "subregion_overlap", measurements, applicable),
            "region_volume_consistency": _value_or_na("subregions", "region_volume_consistency", measurements, applicable),
            "subregion_annotation_uncertainty": _value_or_na("subregions", "subregion_annotation_uncertainty", measurements, applicable),
        },
        "evidence_conflicts": reasoning.get("conflicts") or [],
        "unresolved_evidence": (reasoning.get("uncertainty_annotations") or []) + (critique.get("evidence_gaps") or []),
        "visualization_status": UNKNOWN,
        "visualization_error_type": None,
        "visualization_error": None,
        "advisory_reasoning_summary": {
            "status": reasoning.get("status"),
            "limitations": reasoning.get("limitations") or [],
        },
        "advisory_critique_summary": _critique_summary(critique),
        "evidence_status": routing.get("evidence_status") or reasoning.get("status") or UNKNOWN,
        "limitations": [
            "Compact evidence is selected from validated AgentQC artifacts only.",
            "The adapter does not infer evidence, calculate measurements, or change final policy.",
            visualization_limitation,
        ],
        "provenance_reference_ids": _provenance_refs(index, case_id),
    }
    validate_compact_evidence(compact)
    return compact


def validate_compact_evidence(compact: dict[str, Any]) -> None:
    required = {
        "schema_version", "case_id", "scientific_task_name", "task_notation", "implementation_task_id",
        "observed_final_policy_action", "observed_deterministic_recommendation",
        "observed_calibrated_recommendation", "route_to_review", "route_reasons", "decision_basis",
        "risk_level", "primary_domain", "primary_failure_mode", "unresolved_evidence",
        "relevant_measurements", "provenance_reference_ids",
    }
    missing = required - set(compact)
    if missing:
        raise ValueError(f"Compact evidence missing fields: {sorted(missing)}")
    if compact["implementation_task_id"] not in SUPPORTED_TASKS:
        raise ValueError("Compact evidence has unsupported implementation_task_id")
    for text in iter_strings(compact):
        if contains_path(text):
            raise ValueError("Compact evidence exposes a filesystem path")
        if TERMINAL_CORRUPTION_RE.search(text):
            raise ValueError("Compact evidence contains terminal control text")


def render_prompt(compact: dict[str, Any]) -> list[dict[str, str]]:
    payload = json.dumps(compact, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": PROMPT_TEMPLATE},
        {"role": "user", "content": payload},
    ]


def parse_and_validate_llm_output(raw: str, compact: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("LLM response is empty")
    sanitized = strip_ansi(raw)
    if sanitized != raw and TERMINAL_CORRUPTION_RE.search(raw):
        raise ValueError("LLM response contains ANSI or terminal control sequences")
    try:
        parsed = json.loads(sanitized)
    except json.JSONDecodeError as exc:
        raise ValueError("LLM response is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError("LLM response must be a JSON object")
    if set(parsed) != EXPLANATION_OUTPUT_KEYS:
        raise ValueError("LLM response does not contain exactly the required keys")
    assert_no_authority_fields(parsed)
    if parsed.get("case_id") != compact.get("case_id"):
        raise ValueError("LLM response changed the case_id")
    for key in ("unresolved_items", "possible_consistency_questions", "limitations"):
        if not isinstance(parsed.get(key), list) or not all(isinstance(item, str) for item in parsed[key]):
            raise ValueError(f"LLM response field {key} must be a list of strings")
    for key in ("evidence_summary", "reason_for_review_or_action", "explicit_statement_that_deterministic_policy_is_authoritative"):
        if not isinstance(parsed.get(key), str) or not parsed[key].strip():
            raise ValueError(f"LLM response field {key} must be a nonempty string")
    text = "\n".join(iter_strings(parsed))
    if contains_path(text):
        raise ValueError("LLM response exposes filesystem paths")
    if UNSUPPORTED_MEDICAL_RE.search(text):
        raise ValueError("LLM response asserts unsupported medical correctness or facts")
    if UNSUPPORTED_CAUSAL_RE.search(text) and "visualization" in text.lower():
        raise ValueError("LLM response introduces unsupported visualization-error causality")
    final_action = str(compact.get("observed_final_policy_action") or "")
    mentioned_actions = {action for action in CANONICAL_ACTIONS if re.search(rf"\b{re.escape(action)}\b", text)}
    if mentioned_actions - {final_action}:
        raise ValueError("LLM response altered or contradicted the final action")
    route_reasons = set(str(item) for item in compact.get("route_reasons") or [])
    invented_reasons = [match for match in re.findall(r"[a-z]+(?:_[a-z0-9]+)+", text) if match not in route_reasons]
    allowed_identifiers = {str(compact.get("case_id")), str(compact.get("task_notation")), str(compact.get("implementation_task_id"))}
    invented_reasons = [item for item in invented_reasons if item not in allowed_identifiers and item not in str(compact)]
    if invented_reasons:
        raise ValueError("LLM response includes unsupported identifier-like claims")
    supplied_numbers = {str(number) for number in iter_numbers(compact)}
    for number in re.findall(r"(?<![A-Za-z_])-?\d+(?:\.\d+)?(?![A-Za-z_])", text):
        if number not in supplied_numbers and number not in {"250"}:
            raise ValueError("LLM response includes a number not supplied in compact evidence")
    if "deterministic" not in parsed["explicit_statement_that_deterministic_policy_is_authoritative"].lower():
        raise ValueError("LLM response omits deterministic policy authority")
    if "authoritative" not in parsed["explicit_statement_that_deterministic_policy_is_authoritative"].lower():
        raise ValueError("LLM response omits deterministic policy authority")
    return parsed


def fallback_explanation(compact: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": compact["case_id"],
        "evidence_summary": (
            f"For {compact['scientific_task_name']} ({compact['task_notation']}), compact evidence reports "
            f"primary domain {compact.get('primary_domain')} and final deterministic-policy action "
            f"{compact.get('observed_final_policy_action')}."
        ),
        "reason_for_review_or_action": (
            f"Decision basis: {compact.get('decision_basis')}; route_to_review: "
            f"{compact.get('route_to_review')}; route reasons: {compact.get('route_reasons')}."
        ),
        "unresolved_items": [json.dumps(item, sort_keys=True, ensure_ascii=True) for item in compact.get("unresolved_evidence") or []],
        "possible_consistency_questions": [],
        "limitations": list(compact.get("limitations") or []),
        "explicit_statement_that_deterministic_policy_is_authoritative": (
            "The deterministic AgentQC policy is authoritative; this explanation is nonbinding."
        ),
    }


def explain_case(index: RunArtifactIndex, backend: ChatBackend, case_id: str) -> dict[str, Any]:
    compact = build_compact_evidence(index, case_id)
    messages = render_prompt(compact)
    raw_response = None
    parsed_response = None
    fallback = None
    validation = {"usable_for_reporting": False, "status": "not_run", "errors": []}
    try:
        raw_response = backend.complete(messages)
        parsed_response = parse_and_validate_llm_output(raw_response, compact)
        validation = {"usable_for_reporting": True, "status": "passed", "errors": []}
    except (ProviderError, ValueError) as exc:
        validation = {"usable_for_reporting": False, "status": "failed", "errors": [f"{type(exc).__name__}: {exc}"]}
        fallback = fallback_explanation(compact)
    return {
        "case_id": case_id,
        "compact_input": compact,
        "rendered_prompt": messages,
        "raw_response": raw_response,
        "parsed_response": parsed_response,
        "validation_result": validation,
        "fallback_response": fallback,
        "reported_response": parsed_response if parsed_response is not None else fallback,
        "execution_timestamp": utc_now(),
        "provider": backend.config.provider,
        "model": backend.config.model,
        "decoding_parameters": {
            "temperature": backend.config.temperature,
            "top_p": backend.config.top_p,
            "seed": backend.config.seed,
            "context_length": backend.config.context_length,
            "max_tokens": backend.config.max_tokens,
        },
        "nonbinding": True,
    }
