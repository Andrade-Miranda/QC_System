"""Confirmed-negative lesion manifest utilities.

The manifest is provenance for lesion-capable segmentation profiles only.  It
allows an absent or empty lesion annotation to be consumed as an explicitly
confirmed negative sample instead of a silent valid negative.  It is not an
external medical-correctness claim unless the source itself is a locked human or
golden label.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import hash_file


LESION_TASK_MODES = {"pancreas_lesion", "pancreas_lesion_subregions"}
REQUIRED_CONFIRMATION_FIELDS = {
    "case_id",
    "task_mode",
    "lesion_status",
    "confirmed_by",
    "confirmation_date",
    "confirmation_source",
    "confirmation_scope",
}


def is_lesion_task_mode(task_mode: str) -> bool:
    return str(task_mode) in LESION_TASK_MODES


def load_confirmed_negative_lesions(
    path: Path | str | None,
    *,
    task_mode: str,
    known_case_ids: set[str] | list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Load, validate, and normalize a confirmed-negative lesion manifest.

    Accepted payloads are either a raw JSON object with
    ``confirmed_negative_lesions`` or an artifact envelope whose ``data`` block
    contains that object.  The returned mapping is deterministic and contains a
    source hash for run provenance.
    """
    if path is None:
        return {
            "status": "not_provided",
            "source": None,
            "count": 0,
            "case_ids": [],
            "cases": {},
        }
    if not is_lesion_task_mode(task_mode):
        raise ValueError(
            "Confirmed-negative lesion manifests apply only to lesion-capable "
            f"task modes; got {task_mode!r}"
        )
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    data = payload.get("data") if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(data, dict):
        raise ValueError("Confirmed-negative lesion manifest must be a JSON object")
    records = data.get("confirmed_negative_lesions")
    if not isinstance(records, list):
        raise ValueError("confirmed_negative_lesions must be a list")
    known = set(known_case_ids or [])
    cases: dict[str, dict[str, Any]] = {}
    for index, raw in enumerate(records):
        if not isinstance(raw, dict):
            raise ValueError(f"confirmed_negative_lesions[{index}] must be an object")
        missing = sorted(
            field for field in REQUIRED_CONFIRMATION_FIELDS
            if not isinstance(raw.get(field), str) or not str(raw.get(field)).strip()
        )
        if missing:
            raise ValueError(
                f"confirmed_negative_lesions[{index}] missing required fields: {', '.join(missing)}"
            )
        case_id = str(raw["case_id"]).strip()
        if case_id in cases:
            raise ValueError(f"Duplicate confirmed-negative lesion case_id: {case_id}")
        if known and case_id not in known:
            raise ValueError(f"Confirmed-negative lesion case not present in run cases: {case_id}")
        if str(raw["task_mode"]).strip() != task_mode:
            raise ValueError(
                "Confirmed-negative lesion task_mode mismatch for "
                f"{case_id}: {raw['task_mode']!r} != {task_mode!r}"
            )
        if not is_lesion_task_mode(str(raw["task_mode"]).strip()):
            raise ValueError(
                "Confirmed-negative lesion records apply only to lesion-capable "
                f"task modes: {raw['task_mode']!r}"
            )
        if str(raw["lesion_status"]).strip() != "confirmed_absent":
            raise ValueError(
                f"Confirmed-negative lesion {case_id} must use lesion_status='confirmed_absent'"
            )
        record = {key: raw[key] for key in sorted(REQUIRED_CONFIRMATION_FIELDS)}
        for key, value in raw.items():
            if key not in record and isinstance(value, (str, int, float, bool)):
                record[key] = value
        cases[case_id] = record
    return {
        "status": "loaded",
        "source": {
            "path": str(path.resolve()),
            "sha256": hash_file(path),
        },
        "count": len(cases),
        "case_ids": sorted(cases),
        "cases": {case_id: cases[case_id] for case_id in sorted(cases)},
    }


def confirmation_for_case(confirmed: dict[str, Any] | None, case_id: str) -> dict[str, Any] | None:
    cases = (confirmed or {}).get("cases") or {}
    value = cases.get(case_id)
    return value if isinstance(value, dict) else None
