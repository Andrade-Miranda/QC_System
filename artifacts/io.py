"""Read/write helpers for immutable QC artifacts."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

from .hashing import hash_file, hash_json_payload


ARTIFACT_SCHEMA_VERSION = "1.0"


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _git_commit(project_root: Path | None = None) -> str | None:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def artifact_descriptor(path: Path, artifact_type: str | None = None) -> dict:
    path = Path(path)
    if artifact_type is None and path.exists() and path.is_file():
        try:
            raw = read_json(path)
            if isinstance(raw, dict):
                metadata = raw.get("metadata") or {}
                artifact_type = metadata.get("artifact_type")
        except Exception:
            artifact_type = None
    return {
        "artifact_type": artifact_type,
        "path": str(path),
        "exists": path.exists(),
        "sha256": hash_file(path) if path.exists() and path.is_file() else None,
    }


def build_metadata(
    *,
    artifact_type: str,
    generator: str,
    dataset_name: str,
    task_mode: str,
    version: str = "1.0",
    input_resources: list[dict] | None = None,
    input_artifacts: list[dict] | None = None,
    configuration: dict | None = None,
    run_id: str | None = None,
    project_root: Path | None = None,
) -> dict:
    return {
        "artifact_type": artifact_type,
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "generator": generator,
        "generator_version": version,
        "created_at": _utc_now(),
        "run_id": run_id,
        "artifact_id": None,
        "dataset_name": dataset_name,
        "task_mode": task_mode,
        "input_resources": input_resources or [],
        "input_artifacts": input_artifacts or [],
        "configuration": configuration or {},
        "software_version": {
            "git_commit": _git_commit(project_root),
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "determinism": {
            "random_seed": None,
            "nondeterministic_components": [],
        },
    }


def _stable_for_artifact_id(value: Any) -> Any:
    """Remove volatile timestamp fields before artifact identity hashing."""
    if isinstance(value, dict):
        return {
            k: _stable_for_artifact_id(v)
            for k, v in value.items()
            if k not in {"created_at", "timestamp"}
        }
    if isinstance(value, list):
        return [_stable_for_artifact_id(v) for v in value]
    return value


def write_artifact(
    path: Path,
    *,
    artifact_type: str,
    generator: str,
    data: Any,
    dataset_name: str,
    task_mode: str,
    version: str = "1.0",
    input_resources: list[dict] | None = None,
    input_artifacts: list[dict] | None = None,
    configuration: dict | None = None,
    run_id: str | None = None,
    project_root: Path | None = None,
) -> dict:
    """Write an artifact envelope and return the written payload."""
    metadata = build_metadata(
        artifact_type=artifact_type,
        generator=generator,
        dataset_name=dataset_name,
        task_mode=task_mode,
        version=version,
        input_resources=input_resources,
        input_artifacts=input_artifacts,
        configuration=configuration,
        run_id=run_id,
        project_root=project_root,
    )
    payload = {"metadata": metadata, "data": data}
    metadata["artifact_id"] = hash_json_payload({
        "metadata": {**metadata, "created_at": None, "artifact_id": None},
        "data": _stable_for_artifact_id(data),
    })
    write_json(path, payload)
    return payload


def load_artifact(path: Path) -> tuple[dict, Any]:
    """Load an artifact envelope; legacy JSON returns empty metadata and raw data."""
    raw = read_json(path)
    if isinstance(raw, dict) and "metadata" in raw and "data" in raw:
        return raw.get("metadata") or {}, raw.get("data")
    return {}, raw
