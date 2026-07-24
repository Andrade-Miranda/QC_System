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


def _case_id_from_image(path: Path, suffix: str) -> str:
    name = path.name
    if suffix and name.endswith(suffix):
        return name[: -len(suffix)]
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


def validate_dataset(paths, task_mode: str, profile: dict) -> dict:
    required = required_segmentations(profile)
    required_seg_files = tuple(required.values())
    errors: list[str] = []
    if not paths.raw_root.exists():
        errors.append(f"RAW_DATASET_ROOT not found: {paths.raw_root}")
    if not paths.raw_images_dir.exists():
        errors.append(f"Images directory not found: {paths.raw_images_dir}")
    if not paths.raw_labels_dir.exists():
        errors.append(f"Labels directory not found: {paths.raw_labels_dir}")

    image_paths = sorted(paths.raw_images_dir.glob(f"*{paths.image_suffix}")) if paths.raw_images_dir.exists() else []
    cases: dict[str, dict] = {}
    for img_path in image_paths:
        case_id = _case_id_from_image(img_path, paths.image_suffix)
        seg_dir = paths.raw_labels_dir / case_id / "segmentations"
        missing = [name for name in required_seg_files if not (seg_dir / name).exists()]
        case_errors = []
        if missing:
            case_errors.append("missing_segmentations:" + ",".join(missing))
        cases[case_id] = {
            "status": "valid" if not case_errors else "invalid",
            "omit_from_training": bool(case_errors),
            "errors": case_errors,
            "resources": {
                "image": resource_descriptor(img_path, "image", required=True),
                "segmentation_dir": resource_descriptor(seg_dir, "segmentation_dir", required=True),
            },
        }

    label_case_dirs = sorted(p for p in paths.raw_labels_dir.iterdir() if p.is_dir()) if paths.raw_labels_dir.exists() else []
    image_case_ids = set(cases)
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate raw dataset resources and write dataset_validation.json.")
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion_subregions", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--output", default=None, help="Output artifact path.")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    profile = load_task_profile(args.task_mode, paths.project_root)
    task_dirs = build_output_dirs(paths.summary_dir.parent, args.task_mode)
    out = Path(args.output) if args.output else task_dirs["task_dir"] / "dataset_validation.json"
    data = validate_dataset(paths, args.task_mode, profile)
    write_artifact(
        out,
        artifact_type="dataset_validation",
        generator="ValidationAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        input_resources=[
            resource_descriptor(paths.raw_root, "raw_dataset_root"),
            resource_descriptor(paths.raw_images_dir, "raw_images_dir"),
            resource_descriptor(paths.raw_labels_dir, "raw_labels_dir"),
            resource_descriptor(paths.raw_metadata, "metadata", required=False),
            resource_descriptor(paths.paths_yaml, "paths_config"),
        ],
        configuration={"task_profile": profile},
        run_id=args.run_id,
        project_root=paths.project_root,
    )
    print(f"Wrote dataset validation artifact: {out}")


if __name__ == "__main__":
    main()
