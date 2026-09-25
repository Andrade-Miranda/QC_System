"""Shared helpers for public AgentQC reproduction scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


REPRO_ROOT = Path(__file__).resolve().parent
REPO_ROOT = REPRO_ROOT.parent
ARTIFACTS = REPRO_ROOT / "artifacts"
OUTPUTS = REPRO_ROOT / "outputs"
EXPECTED = REPRO_ROOT / "expected_results.json"


def load_expected() -> dict[str, Any]:
    with EXPECTED.open(encoding="utf-8") as fh:
        return json.load(fh)


def ensure_output_dir(path: Path | None = None) -> Path:
    out = path or OUTPUTS
    out.mkdir(parents=True, exist_ok=True)
    return out


def assert_close(actual: float, expected: float, *, tolerance: float = 0.05, label: str) -> None:
    if abs(actual - expected) > tolerance:
        raise AssertionError(f"{label}: expected {expected}, observed {actual}")
