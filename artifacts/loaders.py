"""Shared loaders for legacy and artifact-wrapped QC JSON payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import load_artifact


def unwrap_payload(path: Path) -> tuple[Any, dict, dict]:
    """Return (payload, payload_metadata, artifact_metadata).

    Supports:
      - flat case dictionaries
      - legacy {metadata, cases}
      - artifact {metadata, data}
      - artifact {metadata, data: {metadata, cases}}
    """
    artifact_meta, payload = load_artifact(path)
    payload_meta = {}
    if isinstance(payload, dict) and "metadata" in payload and "cases" in payload:
        payload_meta = payload.get("metadata") or {}
    return payload, payload_meta, artifact_meta


def load_cases_payload(path: Path) -> tuple[dict, dict, dict]:
    payload, payload_meta, artifact_meta = unwrap_payload(path)
    if isinstance(payload, dict) and "cases" in payload:
        return payload.get("cases") or {}, payload_meta, artifact_meta
    if isinstance(payload, dict):
        return payload, payload_meta, artifact_meta
    return {}, payload_meta, artifact_meta


def load_summary_cases(path: Path) -> tuple[dict, dict, dict]:
    return load_cases_payload(path)


def load_qc_cases(path: Path) -> tuple[dict, dict, dict]:
    return load_cases_payload(path)
