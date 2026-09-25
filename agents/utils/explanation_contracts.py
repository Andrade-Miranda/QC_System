"""Contracts for nonbinding task-aware LLM explanations."""

from __future__ import annotations

import re
from typing import Any


SCHEMA_VERSION = "1.0"
NOT_APPLICABLE = "not_applicable"
UNKNOWN = "unknown"

SUPPORTED_TASKS = {
    "pancreas_only": {
        "scientific_task_name": "pancreas-segmentation task",
        "task_notation": "tau_P",
        "applicable_groups": {"pancreas", "geometry", "fov", "metadata"},
    },
    "pancreas_lesion": {
        "scientific_task_name": "pancreatic-lesion segmentation task",
        "task_notation": "tau_L",
        "applicable_groups": {"pancreas", "geometry", "fov", "metadata", "lesion"},
    },
    "pancreas_lesion_subregions": {
        "scientific_task_name": "pancreatic-lesion subregion task",
        "task_notation": "tau_S",
        "applicable_groups": {"pancreas", "geometry", "fov", "metadata", "lesion", "subregions"},
    },
}

EXPLANATION_OUTPUT_KEYS = {
    "case_id",
    "evidence_summary",
    "reason_for_review_or_action",
    "unresolved_items",
    "possible_consistency_questions",
    "limitations",
    "explicit_statement_that_deterministic_policy_is_authoritative",
}

AUTHORITY_FIELD_PATTERNS = (
    "decision_override",
    "new_decision",
    "recommended_action",
    "routing_override",
    "route_override",
    "score_override",
    "policy_override",
    "threshold_override",
)

CANONICAL_ACTIONS = {"keep", "warning", "review", "reject", "insufficient_evidence"}
ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PATH_RE = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\\\|\.nii(?:\.gz)?|labelsTr/|imagesTr/)")
TERMINAL_CORRUPTION_RE = re.compile(r"(?:\r\x1b|\x1b|\[K|\[A|\[B)")
UNSUPPORTED_MEDICAL_RE = re.compile(
    r"\b(?:diagnos(?:is|tic)|malignant|benign|cancer|adenocarcinoma|metasta(?:sis|tic)|"
    r"clinical(?:ly)? correct|medically correct|medical correctness|expert validated)\b",
    re.IGNORECASE,
)
UNSUPPORTED_CAUSAL_RE = re.compile(r"\b(?:caused by|due to|because of|resulted from)\b", re.IGNORECASE)


def strip_ansi(value: str) -> str:
    return ANSI_RE.sub("", value)


def contains_path(value: str) -> bool:
    return bool(PATH_RE.search(value))


def iter_strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_strings(item)


def iter_numbers(value: Any):
    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from iter_numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from iter_numbers(item)


def assert_no_authority_fields(value: Any, *, path: str = "") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if any(pattern in lowered for pattern in AUTHORITY_FIELD_PATTERNS):
                raise ValueError(f"LLM output contains forbidden authority field: {path}/{key}")
            assert_no_authority_fields(item, path=f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_no_authority_fields(item, path=f"{path}/{index}")
