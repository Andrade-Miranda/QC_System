"""Run-manifest and root-alias helpers for paper-trackable outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .hashing import hash_file
from .io import write_json


def _entry(path: Path, role: str) -> dict[str, Any]:
    path = Path(path)
    return {
        "role": role,
        "path": str(path),
        "exists": path.exists(),
        "sha256": hash_file(path) if path.is_file() else None,
    }


def build_run_manifest(
    *,
    run_id: str,
    dataset_name: str,
    task_mode: str,
    run_dir: Path,
    artifacts: dict[str, Path],
    resources: dict[str, Path] | None = None,
    aliases: dict[str, Path] | None = None,
) -> dict[str, Any]:
    return {
        "manifest_version": "1.0",
        "run_id": run_id,
        "dataset_name": dataset_name,
        "task_mode": task_mode,
        "run_dir": str(Path(run_dir)),
        "artifacts": {name: _entry(path, name) for name, path in sorted(artifacts.items())},
        "resources": {name: _entry(path, name) for name, path in sorted((resources or {}).items())},
        "aliases": {name: str(path) for name, path in sorted((aliases or {}).items())},
        "authority": {
            "final_actions": "final_qc_decisions.json",
            "reasoning_critique_llm_outputs": "explanatory_nonbinding",
        },
    }


def write_run_manifest(path: Path, **kwargs: Any) -> dict[str, Any]:
    manifest_path = Path(path).resolve()
    artifacts = dict(kwargs.get("artifacts") or {})
    kwargs["artifacts"] = {
        name: artifact_path
        for name, artifact_path in artifacts.items()
        if Path(artifact_path).resolve() != manifest_path
    }
    manifest = build_run_manifest(**kwargs)
    write_json(path, manifest)
    return manifest


def write_pointer_alias(path: Path, *, run_id: str, run_dir: Path, target: Path, role: str) -> dict[str, Any]:
    payload = {
        "alias_version": "1.0",
        "role": role,
        "run_id": run_id,
        "run_dir": str(Path(run_dir)),
        "target": str(Path(target)),
        "target_sha256": hash_file(target) if Path(target).is_file() else None,
    }
    write_json(path, payload)
    return payload
