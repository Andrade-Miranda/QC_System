"""Fail-closed evaluator for deterministic first-match YAML policies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


_MISSING = object()


class PolicyError(ValueError):
    """Raised when a policy or its evaluation context is invalid."""


@dataclass(frozen=True)
class PolicyDecision:
    policy_id: str
    policy_version: str
    action: str
    rule_id: str | None
    rule_name: str | None
    rationale: str | None

    def trace(self) -> dict[str, Any]:
        return {
            "policy_id": self.policy_id,
            "policy_version": self.policy_version,
            "matched_rule": {
                "id": self.rule_id,
                "name": self.rule_name,
                "rationale": self.rationale,
            } if self.rule_id is not None else None,
        }


class FirstMatchPolicy:
    """Validated, deterministic first-match policy."""

    def __init__(self, policy: dict[str, Any], *, allowed_roots: set[str] | None = None):
        if not isinstance(policy, dict):
            raise PolicyError("Policy must be a mapping")
        self.policy_id = _required_string(policy, "id")
        self.version = _required_string(policy, "version")
        if policy.get("evaluation") != "first_match":
            raise PolicyError("Only evaluation: first_match is supported")

        actions = policy.get("canonical_actions")
        if not isinstance(actions, list) or not actions or not all(isinstance(v, str) for v in actions):
            raise PolicyError("canonical_actions must be a non-empty string list")
        self.canonical_actions = frozenset(actions)
        self.no_match_action = _required_string(policy, "no_match_action")
        if self.no_match_action not in self.canonical_actions:
            raise PolicyError("no_match_action is not canonical")

        rules = policy.get("rules")
        if not isinstance(rules, list) or not rules:
            raise PolicyError("rules must be a non-empty list")
        rule_ids = [_required_string(rule, "id") for rule in rules if isinstance(rule, dict)]
        if len(rule_ids) != len(rules) or len(set(rule_ids)) != len(rule_ids):
            raise PolicyError("Every rule must have a unique string id")
        if policy.get("precedence") != rule_ids:
            raise PolicyError("precedence must exactly match rule order")

        self.allowed_roots = frozenset(allowed_roots) if allowed_roots is not None else None
        for rule in rules:
            _required_string(rule, "name")
            _required_string(rule, "rationale")
            if rule.get("action") not in self.canonical_actions:
                raise PolicyError(f"Rule {rule['id']} has a non-canonical action")
            _validate_condition(rule.get("when"), self.allowed_roots, rule["id"])
        self.rules = tuple(rules)

    def evaluate(self, context: dict[str, Any]) -> PolicyDecision:
        if not isinstance(context, dict):
            raise PolicyError("Policy context must be a mapping")
        if self.allowed_roots is not None:
            forbidden = set(context) - self.allowed_roots
            if forbidden:
                raise PolicyError(f"Policy context contains forbidden roots: {sorted(forbidden)}")
        for rule in self.rules:
            if _matches(rule["when"], context):
                return PolicyDecision(
                    policy_id=self.policy_id,
                    policy_version=self.version,
                    action=rule["action"],
                    rule_id=rule["id"],
                    rule_name=rule["name"],
                    rationale=rule["rationale"],
                )
        return PolicyDecision(
            policy_id=self.policy_id,
            policy_version=self.version,
            action=self.no_match_action,
            rule_id=None,
            rule_name=None,
            rationale=None,
        )


def load_first_match_policy(
    path: Path,
    *,
    allowed_roots: set[str] | None = None,
) -> FirstMatchPolicy:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    policy = raw.get("policy") if isinstance(raw, dict) else None
    return FirstMatchPolicy(policy, allowed_roots=allowed_roots)


def _required_string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise PolicyError(f"{key} must be a non-empty string")
    return value


def _validate_condition(condition: Any, allowed_roots: frozenset[str] | None, rule_id: str) -> None:
    if not isinstance(condition, dict):
        raise PolicyError(f"Rule {rule_id} condition must be a mapping")
    composite = [key for key in ("all", "any", "not") if key in condition]
    if composite:
        if len(composite) != 1 or len(condition) != 1:
            raise PolicyError(f"Rule {rule_id} condition has ambiguous composition")
        key = composite[0]
        children = condition[key]
        if key == "not":
            _validate_condition(children, allowed_roots, rule_id)
            return
        if not isinstance(children, list):
            raise PolicyError(f"Rule {rule_id} {key} condition must be a list")
        for child in children:
            _validate_condition(child, allowed_roots, rule_id)
        return

    path = condition.get("path")
    operator = condition.get("operator")
    if not isinstance(path, str) or not path or not isinstance(operator, str):
        raise PolicyError(f"Rule {rule_id} leaf requires path and operator")
    if allowed_roots is not None and path.split(".", 1)[0] not in allowed_roots:
        raise PolicyError(f"Rule {rule_id} uses forbidden policy input: {path}")
    if operator not in {"equals", "not_equals", "in", "not_in", "exists", "truthy", "falsy"}:
        raise PolicyError(f"Rule {rule_id} uses unsupported operator: {operator}")
    if operator in {"in", "not_in"} and not isinstance(condition.get("value"), list):
        raise PolicyError(f"Rule {rule_id} operator {operator} requires a list value")


def _resolve(context: dict[str, Any], path: str) -> Any:
    value: Any = context
    for part in path.split("."):
        if not isinstance(value, dict) or part not in value:
            return _MISSING
        value = value[part]
    return value


def _matches(condition: dict[str, Any], context: dict[str, Any]) -> bool:
    if "all" in condition:
        return all(_matches(child, context) for child in condition["all"])
    if "any" in condition:
        return any(_matches(child, context) for child in condition["any"])
    if "not" in condition:
        return not _matches(condition["not"], context)

    actual = _resolve(context, condition["path"])
    operator = condition["operator"]
    expected = condition.get("value")
    if operator == "exists":
        return (actual is not _MISSING) is bool(expected)
    if actual is _MISSING:
        return False
    if operator == "equals":
        return actual == expected
    if operator == "not_equals":
        return actual != expected
    if operator == "in":
        return actual in expected
    if operator == "not_in":
        return actual not in expected
    if operator == "truthy":
        return bool(actual)
    if operator == "falsy":
        return not bool(actual)
    return False
