"""Task profile loading utilities.

Task profiles make task modes declarative instead of hardcoding workflow
behavior in orchestrators and agents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover
    raise ImportError("PyYAML is required: pip install pyyaml") from exc

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_DIR = _PROJECT_ROOT / "configs" / "task_profiles"


def task_profiles_dir(project_root: Path | None = None) -> Path:
    return (Path(project_root) / "configs" / "task_profiles") if project_root else _PROFILE_DIR


def load_task_profile(task_mode: str, project_root: Path | None = None) -> dict[str, Any]:
    path = task_profiles_dir(project_root) / f"{task_mode}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task profile not found: {path}")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Task profile must be a mapping: {path}")
    return data


def required_segmentations(profile: dict[str, Any]) -> dict[str, str]:
    return dict(profile.get("REQUIRED_SEGMENTATIONS") or {})


def active_qc_domains(profile: dict[str, Any]) -> dict[str, bool]:
    return dict(profile.get("ACTIVE_QC_DOMAINS") or {})


def hard_failure_domains(profile: dict[str, Any]) -> set[str]:
    return set(profile.get("HARD_FAILURE_DOMAINS") or [])
