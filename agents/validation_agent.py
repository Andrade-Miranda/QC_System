#!/usr/bin/env python3
"""ValidationAgent: produce dataset_validation.json as a provenance artifact."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import build_output_dirs, resolve_project_paths, VALID_TASK_MODES
from agents.utils.task_profiles import load_task_profile, required_segmentations
from artifacts import resource_descriptor, write_artifact
from artifacts.confirmed_negative_lesions import (
    is_lesion_task_mode,
    load_confirmed_negative_lesions,
)

def _case_id_from_image(path: Path, suffix: str) -> str:
    name = path.name
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def _lesion_annotation_evidence(paths, img_path: Path, seg_dir: Path, task_mode: str) -> dict:
    separate_path = seg_dir / "pancreatic_lesion.nii.gz"
    try:
        from scripts.summarize_dataset import (  # heavy NIfTI stack; import only when needed
            build_lesion_annotation_evidence,
            load_sitk_image,
            sitk_to_uint8,
        )
    except ModuleNotFoundError as exc:
        return {
            "status": "insufficient_evidence",
            "lesion_present": None,
            "absence_confirmed": False,
            "selected_source": None,
            "sources": [{
                "name": "separate_lesion_mask",
                "path": str(separate_path),
                "exists": separate_path.exists(),
                "readable": False,
                "geometry_valid": False,
                "status": "dependency_unavailable",
                "positive_voxels": None,
                "lesion_present": None,
            }],
            "validation_error": f"nifti_dependency_unavailable:{exc.name}",
        }

    ref_sitk = load_sitk_image(img_path)
    separate_sitk = load_sitk_image(separate_path)
    separate_arr = sitk_to_uint8(separate_sitk) if separate_sitk is not None else None
    evidence = build_lesion_annotation_evidence(
        separate_path=separate_path,
        separate_sitk=separate_sitk,
        separate_arr=separate_arr,
        ref_sitk=ref_sitk,
        task_mode=task_mode,
    )
    if ref_sitk is None:
        evidence["status"] = "insufficient_evidence"
        evidence["validation_error"] = "ct_image_unreadable"
    return evidence


def validate_dataset(
    paths,
    task_mode: str,
    profile: dict,
    *,
    confirmed_negative_lesions_path: Path | None = None,
) -> dict:
    required = required_segmentations(profile)
    errors: list[str] = []
    raw_root_configured = getattr(paths, "raw_dataset_root_configured", True)
    if not raw_root_configured:
        errors.append("RAW_DATASET_ROOT is not configured")
    if not paths.raw_root.exists():
        errors.append(f"RAW_DATASET_ROOT not found: {paths.raw_root}")
    if not paths.raw_images_dir.exists():
        errors.append(f"Images directory not found: {paths.raw_images_dir}")
    if not paths.raw_labels_dir.exists():
        errors.append(f"Labels directory not found: {paths.raw_labels_dir}")

    image_paths = sorted(paths.raw_images_dir.glob(f"*{paths.image_suffix}")) if paths.raw_images_dir.exists() else []
    image_case_ids = {_case_id_from_image(img_path, paths.image_suffix) for img_path in image_paths}
    try:
        confirmed_negatives = load_confirmed_negative_lesions(
            confirmed_negative_lesions_path,
            task_mode=task_mode,
            known_case_ids=image_case_ids,
        )
    except (OSError, ValueError) as exc:
        confirmed_negatives = {
            "status": "invalid",
            "source": {"path": str(confirmed_negative_lesions_path)} if confirmed_negative_lesions_path else None,
            "count": 0,
            "case_ids": [],
            "cases": {},
            "error": str(exc),
        }
        errors.append(f"invalid_confirmed_negative_lesions:{exc}")
    cases: dict[str, dict] = {}
    for img_path in image_paths:
        case_id = _case_id_from_image(img_path, paths.image_suffix)
        seg_dir = paths.raw_labels_dir / case_id / "segmentations"
        missing = []
        missing_confirmed_negative = []
        lesion_evidence = None
        for seg_key, seg_name in required.items():
            if seg_key == "lesion_mask" and is_lesion_task_mode(task_mode):
                lesion_evidence = _lesion_annotation_evidence(paths, img_path, seg_dir, task_mode)
            if (seg_dir / seg_name).exists():
                if seg_key == "lesion_mask" and is_lesion_task_mode(task_mode):
                    if lesion_evidence.get("status") not in {"lesion_present", "confirmed_absent"}:
                        missing.append("insufficient_lesion_annotation_evidence")
                continue
            missing.append(seg_name)
        case_errors = []
        if missing:
            case_errors.append("missing_segmentations:" + ",".join(missing))
        confirmation = (confirmed_negatives.get("cases") or {}).get(case_id)
        cases[case_id] = {
            "status": "valid" if not case_errors else "invalid",
            "omit_from_training": bool(case_errors),
            "errors": case_errors,
            "confirmed_negative_lesion": confirmation,
            "lesion_annotation_evidence": lesion_evidence,
            "missing_segmentations_covered_by_confirmation": missing_confirmed_negative,
            "resources": {
                "image": resource_descriptor(img_path, "image", required=True),
                "segmentation_dir": resource_descriptor(seg_dir, "segmentation_dir", required=True),
            },
        }

    label_case_dirs = sorted(p for p in paths.raw_labels_dir.iterdir() if p.is_dir()) if paths.raw_labels_dir.exists() else []
    labels_without_images = [p.name for p in label_case_dirs if p.name not in image_case_ids]
    status = "passed" if not errors and all(c["status"] == "valid" for c in cases.values()) else "failed"
    return {
        "status": status,
        "dataset_name": paths.dataset_name,
        "task_mode": task_mode,
        "n_images": len(image_paths),
        "n_cases": len(cases),
        "n_invalid_cases": sum(1 for c in cases.values() if c["status"] != "valid"),
        "errors": errors,
        "labels_without_images": labels_without_images,
        "cases": cases,
        "required_segmentations": required,
        "confirmed_negative_lesions": {
            key: confirmed_negatives[key]
            for key in ("status", "source", "count", "case_ids")
            if key in confirmed_negatives
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw dataset resources and write dataset_validation.json.")
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion_subregions", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--output", default=None, help="Output artifact path.")
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--confirmed-negative-lesions",
        type=Path,
        default=None,
        help="Optional JSON manifest of explicitly confirmed absent lesion cases.",
    )
    args = parser.parse_args()

    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    profile = load_task_profile(args.task_mode, paths.project_root)
    task_dirs = build_output_dirs(paths.summary_dir.parent, args.task_mode)
    out = Path(args.output) if args.output else task_dirs["task_dir"] / "dataset_validation.json"
    data = validate_dataset(
        paths,
        args.task_mode,
        profile,
        confirmed_negative_lesions_path=args.confirmed_negative_lesions,
    )
    input_resources = [
        resource_descriptor(paths.raw_root, "raw_dataset_root"),
        resource_descriptor(paths.raw_images_dir, "raw_images_dir"),
        resource_descriptor(paths.raw_labels_dir, "raw_labels_dir"),
        resource_descriptor(paths.raw_metadata, "metadata", required=False),
        resource_descriptor(paths.paths_yaml, "paths_config"),
    ]
    if args.confirmed_negative_lesions is not None:
        input_resources.append(
            resource_descriptor(args.confirmed_negative_lesions, "confirmed_negative_lesions", required=True)
        )
    write_artifact(
        out,
        artifact_type="dataset_validation",
        generator="ValidationAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        input_resources=input_resources,
        configuration={"task_profile": profile},
        run_id=args.run_id,
        project_root=paths.project_root,
    )
    print(f"Wrote dataset validation artifact: {out}")


if __name__ == "__main__":
    main()
