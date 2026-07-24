#!/usr/bin/env python3
"""Terminal review interface over immutable, validated QC run artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.llm_backend import (
    ChatBackend,
    ProviderError,
    build_backend,
    load_interaction_logging,
    load_provider_config,
)
from agents.utils.run_artifact_index import (
    RunArtifactError,
    RunArtifactIndex,
    UnknownCaseError,
    json_pointer,
)


CANONICAL_ACTIONS = {"keep", "warning", "review", "reject", "insufficient_evidence"}
_ABNORMAL_SEVERITIES = {"low_warning", "moderate_warning", "high_warning", "critical"}


@dataclass
class ToolResult:
    tool: str
    args: dict[str, Any]
    text: str
    payload: dict[str, Any]
    citations: list[str]


@dataclass
class InteractionState:
    classification_used: bool = False
    fallback: bool = False
    error: str | None = None
    prompt: list[dict[str, str]] | None = None
    record: dict[str, Any] = field(default_factory=dict)


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _dedupe(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _compact(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))


class InteractiveReviewAgent:
    """Dispatch authorized read-only tools with optional intent classification."""

    def __init__(
        self,
        index: RunArtifactIndex,
        backend: ChatBackend,
        *,
        log_path: Path | None = None,
        log_interactions: bool = True,
        log_query_text: bool = False,
        max_displayed_cases: int = 50,
    ):
        self.index = index
        self.backend = backend
        self.log_path = Path(log_path) if log_path is not None else None
        self.log_interactions = log_interactions and self.log_path is not None
        self.log_query_text = log_query_text
        self.max_displayed_cases = max_displayed_cases
        self.last_state = InteractionState()

    def answer(self, query: str) -> str:
        state = InteractionState()
        result = self._dispatch(query)
        if result.tool == "unknown" and self.backend.enabled:
            messages = self._classification_prompt(query)
            state.prompt = messages
            state.classification_used = True
            try:
                result = self._execute_classification(self.backend.complete(messages))
            except Exception as exc:
                state.error = f"{type(exc).__name__}: {exc}"
                state.fallback = True

        sections = ["AUTHORITATIVE DETERMINISTIC RESULT", result.text]
        sections.extend(["SOURCES", *[f"- {citation}" for citation in _dedupe(result.citations)]])
        output = "\n".join(sections)
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        result_hash = hashlib.sha256(output.encode("utf-8")).hexdigest()
        state.record = {
            "timestamp": _utc_now(),
            "mode": self.index.mode,
            "provider": self.backend.config.provider,
            "model": self.backend.config.model,
            "query_sha256": query_hash,
            "selected_tool": result.tool,
            "tool_args": result.args,
            "citations": _dedupe(result.citations),
            "deterministic_result_sha256": result_hash,
            "llm_classification_used": state.classification_used,
            "fallback": state.fallback,
            "error": state.error,
        }
        if self.log_query_text:
            state.record["query"] = query
        if self.index.mode == "admin":
            state.record.update({
                "run_id": self.index.metadata["validated_context"].get("run_id"),
                "dataset_name": self.index.metadata["validated_context"].get("dataset_name"),
                "task_mode": self.index.metadata["validated_context"].get("task_mode"),
                "artifact_ids": {
                    name: metadata.get("artifact_id") for name, metadata in self.index.metadata.items()
                },
            })
        else:
            state.record["package_id"] = self.index.package_id
        self.last_state = state
        self._log(state.record)
        return output

    def _log(self, record: dict[str, Any]) -> None:
        if not self.log_interactions or self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        os.chmod(self.log_path, 0o600)
        with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=True) + "\n")

    def _classification_prompt(self, query: str) -> list[dict[str, str]]:
        if self.index.mode == "reviewer":
            schema = {
                "help": {}, "reviewer_summary": {}, "protocol": {},
                "case_detail": {"case": "string"}, "paths": {"case": "string"},
                "compare_resources": {"cases": ["string", "string"]},
            }
        else:
            schema = {
                "help": {}, "admin_summary": {}, "protocol": {},
                "case_detail": {"case": "string"}, "evidence": {"case": "string"},
                "paths": {"case": "string"}, "compare": {"cases": ["string", "string"]},
                "list_action": {"action": "string"}, "list_rule": {"rule": "string"},
                "golden_selection": {"case": "string"},
            }
        system = (
            "Classify the user query into exactly one allowed read-only command. Return one JSON object "
            "with exactly keys tool and args, no markdown or prose. Do not answer the query. Use only the "
            "provided command schema; if no command fits, return {\"tool\":\"unknown\",\"args\":{}}."
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps({
                "query": query,
                "command_schema": schema,
            }, sort_keys=True, ensure_ascii=True)},
        ]

    def _execute_classification(self, response: str) -> ToolResult:
        try:
            classified = json.loads(response)
        except json.JSONDecodeError as exc:
            raise ProviderError("Intent classifier did not return strict JSON") from exc
        if not isinstance(classified, dict) or set(classified) != {"tool", "args"}:
            raise ProviderError("Intent classification must contain exactly tool and args")
        tool = classified.get("tool")
        args = classified.get("args")
        if not isinstance(tool, str) or not isinstance(args, dict):
            raise ProviderError("Intent classification has invalid types")
        allowed = {
            "reviewer": {"help", "reviewer_summary", "protocol", "case_detail", "paths", "compare_resources"},
            "admin": {
                "help", "admin_summary", "protocol", "case_detail", "evidence", "paths", "compare",
                "list_action", "list_rule", "golden_selection",
            },
        }[self.index.mode]
        if tool not in allowed:
            raise ProviderError("Intent classifier selected a forbidden or unknown tool")
        no_args = {"help", "reviewer_summary", "admin_summary", "protocol"}
        if tool in no_args:
            if args:
                raise ProviderError("Classified tool does not accept arguments")
            return {"help": self._help, "reviewer_summary": self._summary,
                    "admin_summary": self._summary, "protocol": self._protocol}[tool]()
        if tool in {"case_detail", "evidence", "paths", "golden_selection"}:
            if set(args) != {"case"} or not isinstance(args["case"], str):
                raise ProviderError("Classified case tool requires one string case argument")
            self.index.resolve_case(args["case"])
            return {
                "case_detail": self._case_detail,
                "evidence": self._evidence,
                "paths": self._paths,
                "golden_selection": self._selection,
            }[tool](args["case"])
        if tool in {"compare", "compare_resources"}:
            cases = args.get("cases")
            if set(args) != {"cases"} or not isinstance(cases, list) or len(cases) != 2 or not all(
                isinstance(case, str) for case in cases
            ):
                raise ProviderError("Classified comparison requires exactly two string cases")
            for case in cases:
                self.index.resolve_case(case)
            return self._compare(cases[0], cases[1])
        if tool == "list_action":
            if set(args) != {"action"} or args.get("action") not in CANONICAL_ACTIONS:
                raise ProviderError("Classified action is not canonical")
            return self._action(args["action"])
        if tool == "list_rule":
            if set(args) != {"rule"} or not isinstance(args.get("rule"), str):
                raise ProviderError("Classified rule argument is invalid")
            result = self._rule(args["rule"])
            if result.payload.get("status") == "unknown_rule":
                raise ProviderError("Classified rule is not authorized")
            return result
        raise ProviderError("Intent classification could not be executed")

    def _dispatch(self, query: str) -> ToolResult:
        text = query.strip()
        if not text:
            return self._help()
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()
        lowered = [token.lower() for token in tokens]
        joined = " ".join(lowered)

        if joined in {"help", "?", "commands"} or "what can" in joined:
            return self._help()
        if joined in {"summary", "run summary", "package summary"}:
            return self._summary()
        if joined in {"protocol", "review protocol", "show protocol"}:
            return self._protocol()

        if self.index.mode == "reviewer" and self._is_blinded_request(joined):
            return self._blinded()

        if lowered and lowered[0] == "compare" and len(tokens) >= 3:
            return self._compare(tokens[1], tokens[2])
        if "compare" in lowered:
            case_tokens = self._case_like_tokens(tokens)
            if len(case_tokens) >= 2:
                return self._compare(case_tokens[0], case_tokens[1])

        if lowered and lowered[0] in {"paths", "path"} and len(tokens) >= 2:
            return self._paths(tokens[1])
        if "path" in joined or "resource" in joined:
            case_token = self._one_case_token(tokens)
            if case_token:
                return self._paths(case_token)

        if self.index.mode == "admin":
            case_token = self._one_case_token(tokens)
            if self._looks_like_selection(joined):
                if case_token:
                    return self._selection(case_token)
            if case_token and "evidence" in joined:
                return self._evidence(case_token)
            if case_token and any(word in joined for word in (
                "case", "detail", "why", "what happened", "happened", "decision", "action",
                "rule", "match", "review", "reject", "warning", "keep", "vp-", "vp0",
            )):
                return self._case_detail(case_token)
            rule_id_match = re.search(r"(?i)\bvp[-_ ]?(\d+)\b", text)
            if rule_id_match and any(word in joined for word in ("show", "list", "match", "rule", "cases")):
                return self._rule(rule_id_match.group(0))
            if lowered and lowered[0] in {"show", "list"} and len(tokens) >= 3:
                if lowered[1] == "action":
                    return self._action(tokens[2])
                if lowered[1] == "rule":
                    return self._rule(tokens[2])
            action_match = re.search(
                r"(?i)(?:(?:show|list)\s+(?:cases?\s+)?(?:with\s+)?action|"
                r"cases?\s+with\s+action)\s+([a-z_]+)", text
            )
            if action_match:
                return self._action(action_match.group(1))
            rule_match = re.search(
                r"(?i)(?:(?:show|list)\s+(?:cases?\s+)?rule|cases?\s+matching\s+rule)\s+"
                r"(vp[-_ ]?\d+)", text
            )
            if rule_match:
                return self._rule(rule_match.group(1))
            if re.search(r"\b(?:need|requires?|requiring)\s+(?:human\s+)?review\b", joined):
                return self._action("review")
            natural_actions = {
                "rejected": "reject", "reject": "reject", "kept": "keep", "warnings": "warning",
                "warning": "warning", "reviewed": "review",
            }
            if any(word in joined for word in ("which", "how many", "show", "list", "cases")):
                for word, action in natural_actions.items():
                    if re.search(rf"\b{word}\b", joined):
                        return self._action(action)
            if any(word in joined for word in ("show", "list")):
                for action in sorted(CANONICAL_ACTIONS, key=len, reverse=True):
                    if re.search(rf"(?<![a-z_]){re.escape(action)}(?![a-z_])", joined):
                        return self._action(action)
        if lowered and lowered[0] in {"case", "detail", "details", "show"} and len(tokens) >= 2:
            explicit_case = self._one_case_token(tokens)
            if explicit_case:
                return self._case_detail(explicit_case)
        case_token = self._one_case_token(tokens)
        if case_token and any(word in joined for word in (
            "case", "detail", "show", "what happened", "happened",
        )):
            return self._case_detail(case_token)
        return self._unknown_query()

    @staticmethod
    def _is_blinded_request(joined: str) -> bool:
        return bool(re.search(
            r"\b(?:why|action|rule|evidence|decision|route|routing|triage|score|risk|"
            r"selected|selection|golden package|system)\b", joined
        ))

    @staticmethod
    def _looks_like_selection(joined: str) -> bool:
        return bool(
            "golden" in joined
            or "package" in joined
            or "selection stratum" in joined
            or "sampling stratum" in joined
            or re.search(r"\bselected\s+(?:into|for)\s+(?:the\s+)?(?:sample|package)\b", joined)
        )

    def _case_like_tokens(self, tokens: list[str]) -> list[str]:
        matches: list[str] = []
        for token in tokens:
            try:
                self.index.resolve_case(token.strip("?,.:;()[]"))
            except UnknownCaseError:
                continue
            matches.append(token.strip("?,.:;()[]"))
        return matches

    def _one_case_token(self, tokens: list[str]) -> str | None:
        matches = self._case_like_tokens(tokens)
        if matches:
            return matches[-1]
        candidate_pattern = re.compile(r"(?i)(?:pants[_-]?\d+|\d{3,}|case[-_]\w+)")
        for token in reversed(tokens):
            candidate = token.strip("?,.:;()[]")
            if candidate_pattern.fullmatch(candidate):
                return candidate
        return None

    def _resolve(self, raw: str, tool: str) -> tuple[str | None, ToolResult | None]:
        try:
            return self.index.resolve_case(raw), None
        except UnknownCaseError as exc:
            if self.index.mode == "reviewer":
                citations = ["review_manifest.csv#case_id"]
            else:
                citations = ["validated_run_context.json#/data/case_ids"]
            result = ToolResult(
                tool=tool,
                args={"case": raw},
                text=str(exc) + ". No case data were returned.",
                payload={"status": "unknown_case"},
                citations=citations,
            )
            return None, result

    def _help(self) -> ToolResult:
        if self.index.mode == "reviewer":
            commands = (
                "Reviewer commands: help, protocol, summary, case/detail <case>, paths <case>, "
                "compare <case1> <case2>. Only package membership, paths, and neutral resource "
                "availability are accessible."
            )
            citations = [
                "review_manifest.csv#header",
                "REVIEW_PROTOCOL.md#review-protocol",
            ]
        else:
            commands = (
                "Admin commands: help, summary, case/why/detail <case>, evidence <case>, paths <case>, "
                "compare <case1> <case2>, show/list action <action>, show/list rule <VP-id>, and "
                "why selected/in golden package <case>."
            )
            citations = [
                "validated_run_context.json#/metadata",
                "final_qc_decisions.json#/data/summary",
            ]
        return ToolResult("help", {}, commands, {"commands": commands}, citations)

    def _summary(self) -> ToolResult:
        if self.index.mode == "reviewer":
            resources = list(self.index.reviewer_resources.values())
            image_available = sum(Path(row.image_path).is_file() for row in resources)
            mask_total = sum(len(row.mask_paths) for row in resources)
            masks_available = sum(
                Path(path).is_file() for row in resources for path in row.mask_paths.values()
            )
            payload = {
                "package_cases": len(resources),
                "images_available": image_available,
                "images_total": len(resources),
                "masks_available": masks_available,
                "masks_total": mask_total,
            }
            text = (
                f"Blinded package: {len(resources)} cases; images available {image_available}/{len(resources)}; "
                f"masks available {masks_available}/{mask_total}."
            )
            return ToolResult(
                "reviewer_summary", {}, text, payload,
                ["review_manifest.csv#rows"],
            )
        stats = self.index.artifacts["evaluation"].get("dataset_statistics") or {}
        identity = {field: self.index.metadata["validated_context"].get(field) for field in (
            "run_id", "dataset_name", "task_mode"
        )}
        payload = {
            **identity,
            "dataset_statistics": stats,
            "golden_package_cases": len(self.index.manifest),
            "authority": "final policy actions are the source of truth",
        }
        text = (
            f"Run {identity['run_id']} for {identity['dataset_name']} / {identity['task_mode']}: "
            f"{stats.get('n_cases')} cases; actions {_compact(stats.get('decision_counts') or {})}; "
            f"human review required {stats.get('human_review_required')}; hard failures "
            f"{stats.get('hard_failures')}; golden package cases {len(self.index.manifest)}. "
            "Final policy actions are the source of truth."
        )
        return ToolResult(
            "admin_summary", {}, text, payload,
            [
                "validated_run_context.json#/metadata",
                "eval_report.json#/data/dataset_statistics",
                "final_qc_decisions.json#/data/summary",
            ],
        )

    def _protocol(self) -> ToolResult:
        if self.index.mode == "admin":
            assert self.index.run_dir is not None
            protocol_path = self.index.run_dir / "golden_review_package" / "reviewer_package" / "REVIEW_PROTOCOL.md"
            text = protocol_path.read_text(encoding="utf-8") if protocol_path.is_file() else "Review protocol is unavailable."
            citations = [
                "golden_review_package/reviewer_package/REVIEW_PROTOCOL.md#review-protocol",
                "validated_run_context.json#/metadata",
            ]
        else:
            text = self.index.protocol or "Review protocol is unavailable."
            citations = ["REVIEW_PROTOCOL.md#review-protocol"]
        return ToolResult(
            "protocol", {}, text, {"protocol": text},
            citations,
        )

    def _blinded(self) -> ToolResult:
        text = (
            "Blinded-access restriction: reviewer mode only provides package case IDs, image/mask "
            "paths, neutral resource availability, and protocol/help. This request is not available."
        )
        return ToolResult(
            "blinded_access", {}, text, {"status": "blinded_access"},
            ["review_manifest.csv#header"],
        )

    def _case_detail(self, raw_case: str) -> ToolResult:
        case_id, error = self._resolve(raw_case, "case_detail")
        if error:
            return error
        assert case_id is not None
        if self.index.mode == "reviewer":
            return self._reviewer_resources(case_id, "case_detail")

        validation = self.index.case("dataset_validation", case_id) or {}
        det = self.index.case("deterministic", case_id) or {}
        cal = self.index.case("calibrated", case_id) or {}
        comparison = self.index.case("comparison", case_id) or {}
        reasoning = self.index.case("reasoning", case_id) or {}
        critique = self.index.case("critique", case_id) or {}
        routing = self.index.case("routing", case_id) or {}
        decision = self.index.case("final", case_id) or {}
        matched = ((decision.get("decision_trace") or {}).get("matched_rule") or {})
        rule_id = str(matched.get("id") or "").upper()
        yaml_rule = self.index.policy_rules.get(rule_id, (None, {}))[1]
        det_abnormal = self._abnormal_domains(det)
        cal_abnormal = self._abnormal_domains(cal)
        det_triage = self._triage_brief(det)
        cal_triage = self._triage_brief(cal)
        comparison_brief = {
            key: comparison.get(key)
            for key in (
                "changed",
                "recommendation_change",
                "risk_change",
                "score_delta",
                "domain_deterministic",
                "domain_calibrated",
            )
        }
        golden = self.index.system_reference.get(case_id)
        golden_stratum = golden[1].get("selection_stage") if golden else None
        payload = {
            "case_id": case_id,
            "final_policy_action": decision.get("final_decision"),
            "matched_yaml_rule": {
                "id": yaml_rule.get("id"),
                "name": yaml_rule.get("name"),
                "rationale": yaml_rule.get("rationale"),
                "action": yaml_rule.get("action"),
            },
            "validation": {
                "status": validation.get("status"),
                "omit_from_training": validation.get("omit_from_training"),
                "errors": validation.get("errors") or [],
            },
            "target_presence": det.get("target_presence"),
            "abnormal_domains": {"deterministic": det_abnormal, "calibrated": cal_abnormal},
            "triage": {"deterministic": det_triage, "calibrated": cal_triage},
            "comparison": comparison_brief,
            "routing": {
                "route_to_review": routing.get("route_to_review"),
                "reasons": routing.get("route_reasons") or [],
            },
            "explanations_nonbinding": {
                "reasoning_status": reasoning.get("status"),
                "critique_status": critique.get("status"),
                "critique_binding": critique.get("binding"),
            },
            "golden_selection_stratum": golden_stratum,
            "provenance": {
                "run_id": self.index.metadata["final"].get("run_id"),
                "dataset_name": self.index.metadata["final"].get("dataset_name"),
                "task_mode": self.index.metadata["final"].get("task_mode"),
                "policy_id": decision.get("policy_id"),
                "policy_version": decision.get("policy_version"),
            },
        }
        text = "\n".join([
            f"Case: {case_id}",
            f"FINAL POLICY ACTION (source of truth): {decision.get('final_decision')}",
            (
                f"Matched YAML: {yaml_rule.get('id')} {yaml_rule.get('name')} -> "
                f"{yaml_rule.get('action')}"
            ),
            f"Reason: {yaml_rule.get('rationale')}",
            (
                f"Validation: status={validation.get('status')}, "
                f"omit={validation.get('omit_from_training')}, "
                f"errors={_compact(validation.get('errors') or [])}"
            ),
            f"Target presence: {_compact(det.get('target_presence'))}",
            (
                f"Abnormal domains: deterministic={_compact(det_abnormal)}, "
                f"calibrated={_compact(cal_abnormal)}"
            ),
            f"Triage: deterministic={_compact(det_triage)}, calibrated={_compact(cal_triage)}",
            f"Calibration comparison: {_compact(comparison_brief)}",
            (
                f"Routing: route={routing.get('route_to_review')}, "
                f"reasons={_compact(routing.get('route_reasons') or [])}"
            ),
            (
                f"Explanatory artifacts: reasoning={reasoning.get('status')}, "
                f"critique={critique.get('status')}, binding={critique.get('binding')}; "
                "both are explanatory and nonbinding"
            ),
            (
                f"Provenance: run={payload['provenance']['run_id']}, "
                f"dataset={payload['provenance']['dataset_name']}, "
                f"task={payload['provenance']['task_mode']}, "
                f"policy={payload['provenance']['policy_id']}@{payload['provenance']['policy_version']}"
            ),
            f"Golden selection stratum: {golden_stratum if golden_stratum else 'not selected/unavailable'}",
        ])
        rule_index = self.index.policy_rules.get(rule_id, (0, {}))[0]
        citations = [
            self.index.citation("final", self.index.pointer("final", case_id)),
            f"{self.index.policy_path.name}#{json_pointer('policy', 'rules', rule_index)}",
            self.index.citation("dataset_validation", self.index.pointer("dataset_validation", case_id)),
            self.index.citation("deterministic", self.index.pointer("deterministic", case_id)),
            self.index.citation("calibrated", self.index.pointer("calibrated", case_id)),
            self.index.citation("comparison", self.index.pointer("comparison", case_id)),
            self.index.citation("routing", self.index.pointer("routing", case_id)),
            self.index.citation("reasoning", self.index.pointer("reasoning", case_id)),
            self.index.citation("critique", self.index.pointer("critique", case_id)),
        ]
        if golden:
            citations.append(f"golden_review_package/system_reference.csv#row={golden[0]}")
        return ToolResult("case_detail", {"case": case_id}, text, payload, citations)

    @staticmethod
    def _abnormal_domains(case: dict[str, Any]) -> list[dict[str, Any]]:
        abnormal = []
        for domain, value in sorted((case.get("qc_domains") or {}).items()):
            if isinstance(value, dict) and value.get("severity") in _ABNORMAL_SEVERITIES:
                abnormal.append({
                    "domain": domain,
                    "severity": value.get("severity"),
                    "component_score": value.get("component_score"),
                })
        return abnormal

    @staticmethod
    def _triage_brief(case: dict[str, Any]) -> dict[str, Any]:
        triage = case.get("triage") or {}
        return {
            "recommendation": triage.get("recommendation"),
            "risk_level": triage.get("risk_level"),
            "score": triage.get("score"),
        }

    def _evidence(self, raw_case: str) -> ToolResult:
        case_id, error = self._resolve(raw_case, "evidence")
        if error:
            return error
        assert case_id is not None
        det = self.index.case("deterministic", case_id) or {}
        cal = self.index.case("calibrated", case_id) or {}
        comparison = self.index.case("comparison", case_id) or {}
        payload = {
            "case_id": case_id,
            "target_presence": det.get("target_presence"),
            "deterministic": {"triage": det.get("triage"), "abnormal_domains": self._abnormal_domains(det)},
            "calibrated": {"triage": cal.get("triage"), "abnormal_domains": self._abnormal_domains(cal)},
            "comparison": comparison,
        }
        text = f"Evidence for {case_id}: {_compact(payload)}. Final policy action remains the source of truth."
        return ToolResult(
            "evidence", {"case": case_id}, text, payload,
            [
                self.index.citation("deterministic", self.index.pointer("deterministic", case_id)),
                self.index.citation("calibrated", self.index.pointer("calibrated", case_id)),
                self.index.citation("comparison", self.index.pointer("comparison", case_id)),
                self.index.citation("final", self.index.pointer("final", case_id)),
            ],
        )

    def _paths(self, raw_case: str) -> ToolResult:
        case_id, error = self._resolve(raw_case, "paths")
        if error:
            return error
        assert case_id is not None
        if self.index.mode == "reviewer":
            return self._reviewer_resources(case_id, "paths")
        validation = self.index.case("dataset_validation", case_id) or {}
        resources = validation.get("resources") or {}
        image = ((resources.get("image") or {}).get("path"))
        segmentation_dir = ((resources.get("segmentation_dir") or {}).get("path"))
        required = self.index.artifacts["dataset_validation"].get("required_segmentations") or {}
        masks = {
            name: str(Path(segmentation_dir) / filename)
            for name, filename in required.items()
            if segmentation_dir and isinstance(filename, str)
        }
        payload = {
            "case_id": case_id,
            "image": {"path": image, "available": bool(image and Path(image).is_file())},
            "masks": {
                name: {"path": path, "available": Path(path).is_file()} for name, path in masks.items()
            },
        }
        return ToolResult(
            "paths", {"case": case_id}, f"Resources for {case_id}: {_compact(payload)}", payload,
            [self.index.citation("dataset_validation", self.index.pointer("dataset_validation", case_id))],
        )

    def _reviewer_resources(self, case_id: str, tool: str) -> ToolResult:
        row = self.index.reviewer_resources[case_id]
        payload = {
            "review_id": row.review_id,
            "case_id": case_id,
            "image": {"path": row.image_path, "available": Path(row.image_path).is_file()},
            "masks": {
                name: {"path": path, "available": Path(path).is_file()}
                for name, path in sorted(row.mask_paths.items())
            },
        }
        text = f"Blinded reviewer resources for {case_id}: {_compact(payload)}"
        return ToolResult(
            tool, {"case": case_id}, text, payload,
            [f"review_manifest.csv#row={row.manifest_row}"],
        )

    def _compare(self, raw_left: str, raw_right: str) -> ToolResult:
        left, error = self._resolve(raw_left, "compare")
        if error:
            return error
        right, error = self._resolve(raw_right, "compare")
        if error:
            return error
        assert left is not None and right is not None
        if self.index.mode == "reviewer":
            rows = [self.index.reviewer_resources[left], self.index.reviewer_resources[right]]
            payload = {
                row.case_id: {
                    "image_available": Path(row.image_path).is_file(),
                    "mask_availability": {
                        name: Path(path).is_file() for name, path in sorted(row.mask_paths.items())
                    },
                    "image_path": row.image_path,
                    "mask_paths": row.mask_paths,
                }
                for row in rows
            }
            return ToolResult(
                "compare_resources", {"cases": [left, right]},
                f"Blinded resource comparison: {_compact(payload)}", payload,
                [f"review_manifest.csv#row={row.manifest_row}" for row in rows],
            )
        payload: dict[str, Any] = {}
        citations: list[str] = []
        for case_id in (left, right):
            decision = self.index.case("final", case_id) or {}
            det = self.index.case("deterministic", case_id) or {}
            cal = self.index.case("calibrated", case_id) or {}
            comparison = self.index.case("comparison", case_id) or {}
            payload[case_id] = {
                "final_policy_action": decision.get("final_decision"),
                "matched_rule": ((decision.get("decision_trace") or {}).get("matched_rule")),
                "deterministic_triage": det.get("triage"),
                "calibrated_triage": cal.get("triage"),
                "calibration_change": comparison,
            }
            citations.extend([
                self.index.citation("final", self.index.pointer("final", case_id)),
                self.index.citation("deterministic", self.index.pointer("deterministic", case_id)),
                self.index.citation("calibrated", self.index.pointer("calibrated", case_id)),
                self.index.citation("comparison", self.index.pointer("comparison", case_id)),
            ])
        text = (
            f"Case comparison: {_compact(payload)}. Final policy actions are the source of truth; "
            "the comparison does not create a new decision."
        )
        return ToolResult("compare", {"cases": [left, right]}, text, payload, citations)

    def _action(self, raw_action: str) -> ToolResult:
        action = raw_action.strip().lower()
        if action not in CANONICAL_ACTIONS:
            return ToolResult(
                "list_action", {"action": raw_action},
                f"Unknown canonical action: {raw_action}. No cases were returned.",
                {"status": "unknown_action"},
                ["final_qc_decisions.json#/data/summary/decision_counts"],
            )
        cases = sorted(
            case_id for case_id, decision in self.index.case_maps["final"].items()
            if isinstance(decision, dict) and decision.get("final_decision") == action
        )
        displayed = cases[:self.max_displayed_cases]
        truncated = len(displayed) < len(cases)
        payload = {
            "action": action,
            "count": len(cases),
            "displayed_count": len(displayed),
            "truncated": truncated,
            "cases": displayed,
        }
        text = (
            f"Final policy action {action}: {len(cases)} cases; displaying {len(displayed)}"
            f"{' (truncated)' if truncated else ''}. {', '.join(displayed) if displayed else 'None'}. "
            "These final policy actions are the source of truth."
        )
        return ToolResult(
            "list_action", {"action": action}, text, payload,
            ["final_qc_decisions.json#/data/decisions", "final_qc_decisions.json#/data/summary/decision_counts"],
        )

    def _rule(self, raw_rule: str) -> ToolResult:
        digits = re.search(r"(?i)vp[-_ ]?(\d+)", raw_rule)
        rule_id = f"VP-{int(digits.group(1)):03d}" if digits else raw_rule.upper()
        indexed = self.index.policy_rules.get(rule_id)
        if indexed is None:
            return ToolResult(
                "list_rule", {"rule": raw_rule}, f"Unknown policy rule: {raw_rule}. No cases were returned.",
                {"status": "unknown_rule"},
                [f"{self.index.policy_path.name}#/policy/rules"],
            )
        rule_index, rule = indexed
        cases = sorted(
            case_id for case_id, decision in self.index.case_maps["final"].items()
            if isinstance(decision, dict) and rule_id in (decision.get("policy_rules_triggered") or [])
        )
        displayed = cases[:self.max_displayed_cases]
        truncated = len(displayed) < len(cases)
        payload = {
            "rule": {
                "id": rule.get("id"), "name": rule.get("name"), "rationale": rule.get("rationale"),
                "action": rule.get("action"), "when": rule.get("when"),
            },
            "count": len(cases),
            "displayed_count": len(displayed),
            "truncated": truncated,
            "cases": displayed,
        }
        text = (
            f"YAML rule {rule_id}: name={rule.get('name')}; action={rule.get('action')}; "
            f"rationale={rule.get('rationale')}; when={_compact(rule.get('when'))}. Matched cases "
            f"({len(cases)} total; displaying {len(displayed)}{' (truncated)' if truncated else ''}): "
            f"{', '.join(displayed) if displayed else 'None'}. Final policy actions are the source of truth."
        )
        return ToolResult(
            "list_rule", {"rule": rule_id}, text, payload,
            [
                f"{self.index.policy_path.name}#{json_pointer('policy', 'rules', rule_index)}",
                "final_qc_decisions.json#/data/decisions",
            ],
        )

    def _selection(self, raw_case: str) -> ToolResult:
        case_id, error = self._resolve(raw_case, "golden_selection")
        if error:
            return error
        assert case_id is not None
        manifest = self.index.manifest.get(case_id)
        reference = self.index.system_reference.get(case_id)
        if not manifest:
            payload = {"case_id": case_id, "selected": False}
            text = f"{case_id} is not in the golden review manifest."
            citations = [
                "golden_review_package/reviewer_package/review_manifest.csv#case_id",
                self.index.citation("final", self.index.pointer("final", case_id)),
            ]
        elif not reference:
            payload = {"case_id": case_id, "selected": True, "selection_stratum": None}
            text = f"{case_id} is in the golden review manifest; selection stratum is unavailable."
            citations = [
                f"golden_review_package/reviewer_package/review_manifest.csv#row={manifest[0]}",
                self.index.citation("final", self.index.pointer("final", case_id)),
            ]
        else:
            stage = reference[1].get("selection_stage")
            payload = {
                "case_id": case_id,
                "selected": True,
                "selection_stratum": stage,
                "selection_seed": reference[1].get("selection_seed"),
            }
            text = (
                f"{case_id} was selected into the golden package in deterministic stratum {stage} "
                f"with seed {reference[1].get('selection_seed')}. This is a package sampling stratum, "
                "not a new QC decision. Final policy action remains the source of truth."
            )
            citations = [
                f"golden_review_package/reviewer_package/review_manifest.csv#row={manifest[0]}",
                f"golden_review_package/system_reference.csv#row={reference[0]}",
                self.index.citation("final", self.index.pointer("final", case_id)),
            ]
        return ToolResult("golden_selection", {"case": case_id}, text, payload, citations)

    def _unknown_query(self) -> ToolResult:
        citations = (
            ["review_manifest.csv#header"]
            if self.index.mode == "reviewer"
            else ["validated_run_context.json#/metadata"]
        )
        return ToolResult(
            "unknown", {}, "Unknown request. Use help for authorized commands. No data were returned.",
            {"status": "unknown_request"}, citations,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="Completed run directory (admin mode only)")
    parser.add_argument(
        "--review-package",
        type=Path,
        help="Sanitized directory containing only review_manifest.csv and REVIEW_PROTOCOL.md (reviewer mode only)",
    )
    parser.add_argument("--mode", required=True, choices=("admin", "reviewer"))
    parser.add_argument("--query", help="Run one query and exit; otherwise start an interactive REPL")
    parser.add_argument("--llm-config", type=Path, default=_PROJECT_ROOT / "configs" / "llm_config.yaml")
    parser.add_argument("--provider", choices=("none", "ollama", "openai"), help="Override configured provider")
    parser.add_argument(
        "--allow-external-provider",
        action="store_true",
        help="Explicitly permit an external provider such as OpenAI for query intent classification only",
    )
    logging_group = parser.add_mutually_exclusive_group()
    logging_group.add_argument("--log", action="store_true", help="Enable interaction JSONL logging")
    logging_group.add_argument("--no-log", action="store_true", help="Disable interaction JSONL logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "admin":
        if args.run_dir is None or args.review_package is not None:
            print("error: admin mode requires --run-dir and does not accept --review-package", file=sys.stderr)
            return 2
        index_root = args.run_dir
    else:
        if args.review_package is None or args.run_dir is not None:
            print("error: reviewer mode requires --review-package and does not accept --run-dir", file=sys.stderr)
            return 2
        index_root = args.review_package
    try:
        index = RunArtifactIndex.load(index_root, args.mode)
        provider_config = load_provider_config(args.llm_config, args.provider)
        configured_logging, log_filename, log_query_text, max_cases = load_interaction_logging(args.llm_config)
        backend = build_backend(
            provider_config,
            allow_external_provider=args.allow_external_provider,
        )
    except (RunArtifactError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.mode == "admin":
        assert index.run_dir is not None
        log_path = index.run_dir / "logs" / log_filename
        default_logging = configured_logging
    else:
        log_path = _PROJECT_ROOT / "logs" / "reviewer" / f"{index.package_id}-{log_filename}"
        default_logging = False
    selected_logging = True if args.log else False if args.no_log else default_logging
    agent = InteractiveReviewAgent(
        index,
        backend,
        log_path=log_path,
        log_interactions=selected_logging,
        log_query_text=log_query_text,
        max_displayed_cases=max_cases,
    )
    if args.query is not None:
        print(agent.answer(args.query))
        return 0

    print(
        f"Interactive review ({args.mode}); provider={provider_config.provider}; "
        f"model={provider_config.model or 'none'}; type help or quit."
    )
    while True:
        try:
            query = input("review> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if query.lower() in {"quit", "exit"}:
            return 0
        print(agent.answer(query))


if __name__ == "__main__":
    raise SystemExit(main())
