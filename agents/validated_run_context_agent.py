#!/usr/bin/env python3
"""Produce the validated, resolved context for an AgentQC run."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
import re
from typing import Any

import yaml

from artifacts import resource_descriptor, write_artifact
from artifacts.confirmed_negative_lesions import load_confirmed_negative_lesions
from artifacts.hashing import hash_json_payload
from artifacts.loaders import load_summary_cases


REQUIRED_RUN_ARTIFACTS = [
    "validated_context",
    "dataset_validation",
    "deterministic_evidence",
    "calibrated_evidence",
    "comparison",
    "reasoning",
    "critique",
    "routing",
    "final_decisions",
]
_SEMANTIC_VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")


def _load_yaml(path: Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def task_profile_reference(task_mode: str, profile_path: Path) -> dict[str, str]:
    """Return the additive lower-case profile identity or a safe legacy identity."""
    profile_path = Path(profile_path).resolve()
    profile = _load_yaml(profile_path)
    declared = profile.get("task_profile")
    declared = declared if isinstance(declared, dict) else {}
    profile_id = str(declared.get("id") or task_mode).strip() or task_mode
    version = str(declared.get("version") or "1.0.0").strip() or "1.0.0"
    if not _SEMANTIC_VERSION.fullmatch(version):
        raise ValueError(f"Task profile version must use MAJOR.MINOR.PATCH: {version!r}")
    reference_path = str(declared.get("path") or profile_path)
    return {"id": profile_id, "version": version, "path": reference_path}


def _named_resource(name: str, path: Path, *, required: bool) -> dict[str, Any]:
    descriptor = resource_descriptor(path, name, required=required)
    descriptor["name"] = name
    return descriptor


def _resolved_path_configuration(
    paths: Any,
    *,
    run_dir: Path | None,
    output_dirs: Mapping[str, Path] | None,
) -> dict[str, Any]:
    output_dirs = output_dirs or {}
    result = {
        "paths_yaml": str(Path(paths.paths_yaml).resolve()),
        "project_root": str(Path(paths.project_root).resolve()),
        "dataset_name": paths.dataset_name,
        "raw_dataset_root": str(Path(paths.raw_root).resolve()),
        "raw_images_dir": str(Path(paths.raw_images_dir).resolve()),
        "raw_labels_dir": str(Path(paths.raw_labels_dir).resolve()),
        "raw_metadata": str(Path(paths.raw_metadata).resolve()),
        "image_suffix": paths.image_suffix,
        "raw_dataset_root_configured": bool(getattr(paths, "raw_dataset_root_configured", True)),
        "summary_dir": str(Path(output_dirs.get("summary_dir", paths.summary_dir)).resolve()),
        "qc_dir": str(Path(output_dirs.get("qc_dir", paths.qc_dir)).resolve()),
        "logs_dir": str(Path(paths.logs_dir).resolve()),
        "thresholds_config": str(Path(paths.thresholds_config).resolve()),
        "default_threshold_method": paths.default_threshold_method,
    }
    if "task_dir" in output_dirs:
        result["task_dir"] = str(Path(output_dirs["task_dir"]).resolve())
    if run_dir is not None:
        result["run_dir"] = str(Path(run_dir).resolve())
    return result


def _resolved_threshold_configuration(
    paths: Any,
    *,
    reuse_calibrated: bool,
    calibrated_thresholds_path: Path,
) -> dict[str, Any]:
    deterministic_path = Path(paths.thresholds_config).resolve()
    calibrated_path = Path(calibrated_thresholds_path).resolve()
    calibrated_exists = calibrated_path.is_file()
    result: dict[str, Any] = {
        "default_method": paths.default_threshold_method,
        "deterministic": {
            "path": str(deterministic_path),
            "exists": deterministic_path.is_file(),
            "values": _load_yaml(deterministic_path),
        },
        "calibrated": {
            "path": str(calibrated_path),
            "exists_at_validation": calibrated_exists,
            "source": "existing_file" if reuse_calibrated else "calibration_stage",
        },
    }
    if reuse_calibrated and calibrated_exists:
        result["calibrated"]["values"] = _load_yaml(calibrated_path)
    return result


def _known_case_ids(paths: Any) -> list[str]:
    images_dir = Path(paths.raw_images_dir)
    if not images_dir.is_dir():
        return []
    suffix = str(paths.image_suffix)
    case_ids = []
    for image_path in images_dir.glob(f"*{suffix}"):
        name = image_path.name
        case_ids.append(name[: -len(suffix)] if suffix and name.endswith(suffix) else image_path.stem)
    return sorted(set(case_ids))


def validate_reusable_calibrated_thresholds(
    thresholds_path: Path,
    *,
    summary_path: Path,
    task_mode: str,
) -> None:
    """Reject calibrated thresholds created for another summary or task."""
    thresholds_path = Path(thresholds_path).resolve()
    metadata = _load_yaml(thresholds_path).get("THRESHOLD_METADATA") or {}
    calibrated_from = metadata.get("calibrated_from_summary")
    if not calibrated_from:
        raise ValueError(
            f"Reusable calibrated thresholds lack calibrated_from_summary: {thresholds_path}"
        )
    payload_hash = metadata.get("calibrated_from_summary_payload_sha256")
    if payload_hash:
        cases, _, _ = load_summary_cases(summary_path)
        if hash_json_payload(cases) != payload_hash:
            raise ValueError(
                "Reusable calibrated thresholds were generated from different summary content"
            )
    elif Path(str(calibrated_from)).resolve() != Path(summary_path).resolve():
        raise ValueError(
            "Reusable calibrated thresholds were generated from a different summary: "
            f"{calibrated_from}"
        )
    calibrated_task_mode = metadata.get("task_mode")
    if calibrated_task_mode and str(calibrated_task_mode) != task_mode:
        raise ValueError(
            "Reusable calibrated thresholds were generated for task mode "
            f"{calibrated_task_mode!r}, not {task_mode!r}"
        )


def generate_validated_run_context(
    *,
    paths: Any,
    task_mode: str,
    profile_path: Path,
    run_id: str,
    output_path: Path,
    run_dir: Path | None = None,
    output_dirs: Mapping[str, Path] | None = None,
    reuse_calibrated: bool = False,
    calibrated_thresholds_path: Path,
    confirmed_negative_lesions_path: Path | None = None,
    validation_status: str = "valid",
    warnings: Sequence[str] = (),
    required_artifacts: Sequence[str] = REQUIRED_RUN_ARTIFACTS,
) -> dict[str, Any]:
    """Write a schema-conforming context after configuration validation succeeds."""
    if validation_status not in {"valid", "valid_with_warnings", "invalid"}:
        raise ValueError(f"Unsupported validation status: {validation_status}")
    profile_path = Path(profile_path).resolve()
    case_ids = _known_case_ids(paths)
    confirmed_negative_lesions = load_confirmed_negative_lesions(
        confirmed_negative_lesions_path,
        task_mode=task_mode,
        known_case_ids=set(case_ids),
    )
    dataset_resources = [
        _named_resource("raw_dataset_root", paths.raw_root, required=True),
        _named_resource("raw_images_dir", paths.raw_images_dir, required=True),
        _named_resource("raw_labels_dir", paths.raw_labels_dir, required=True),
        _named_resource("metadata", paths.raw_metadata, required=False),
    ]
    data = {
        "run_id": run_id,
        "dataset_name": paths.dataset_name,
        "task_mode": task_mode,
        "task_profile": task_profile_reference(task_mode, profile_path),
        "dataset_resources": dataset_resources,
        "path_configuration": _resolved_path_configuration(
            paths, run_dir=run_dir, output_dirs=output_dirs
        ),
        "threshold_configuration": _resolved_threshold_configuration(
            paths,
            reuse_calibrated=reuse_calibrated,
            calibrated_thresholds_path=calibrated_thresholds_path,
        ),
        "validation_status": validation_status,
        "warnings": [str(warning) for warning in warnings],
        "required_artifacts": list(required_artifacts),
        "case_ids": case_ids,
        "confirmed_negative_lesions": {
            key: confirmed_negative_lesions[key]
            for key in ("status", "source", "count", "case_ids")
            if key in confirmed_negative_lesions
        },
    }
    input_resources = [
        *dataset_resources,
        _named_resource("paths_config", paths.paths_yaml, required=True),
        _named_resource("task_profile", profile_path, required=True),
        _named_resource("deterministic_thresholds", paths.thresholds_config, required=True),
        _named_resource(
            "calibrated_thresholds",
            calibrated_thresholds_path,
            required=reuse_calibrated,
        ),
    ]
    if confirmed_negative_lesions_path is not None:
        input_resources.append(
            _named_resource(
                "confirmed_negative_lesions",
                confirmed_negative_lesions_path,
                required=True,
            )
        )
    return write_artifact(
        output_path,
        artifact_type="validated_run_context",
        generator="ValidatedRunContextAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=task_mode,
        input_resources=input_resources,
        configuration={
            "config_validation": validation_status,
            "explanatory_agents_have_decision_authority": False,
        },
        run_id=run_id,
        project_root=paths.project_root,
    )
