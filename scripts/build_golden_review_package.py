#!/usr/bin/env python3
"""Build a reproducible, system-action-blinded human review package."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from artifacts.io import artifact_descriptor, load_artifact
from artifacts.hashing import hash_json_payload


REVIEW_FIELDS = [
    "review_id",
    "case_id",
    "image_path",
    "mask_paths",
    "annotation_completeness",
    "expected_action",
    "reviewer_id",
    "confidence",
    "rationale",
]
SYSTEM_FIELDS = [
    "package_id",
    "review_id",
    "case_id",
    "selection_stage",
    "system_action",
    "primary_domain",
    "policy_rules_triggered",
    "target_observed_presence",
    "target_annotation_presence",
    "visible_target_annotation_status",
    "deterministic_recommendation",
    "deterministic_risk_level",
    "deterministic_score",
    "dataset_validation_status",
    "source_run_id",
    "dataset_name",
    "task_mode",
    "selection_seed",
    "final_decisions_sha256",
    "dataset_validation_sha256",
    "deterministic_evidence_sha256",
]
QUOTAS = {"keep": 15, "warning": 15, "review": 15}
TARGET_SIZE = 60


def _cases(data: dict, key: str) -> dict[str, dict]:
    value = data.get(key)
    if isinstance(value, dict):
        return {
            str(case_id): row
            for case_id, row in value.items()
            if isinstance(row, dict)
        }
    if isinstance(value, list):
        return {
            str(row["case_id"]): row
            for row in value
            if isinstance(row, dict) and row.get("case_id") is not None
        }
    return {}


def _stable_key(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def _feature_tokens(candidate: dict) -> set[str]:
    final = candidate["final"]
    evidence = candidate["evidence"]
    target = evidence.get("target_presence") or {}
    tokens = {
        f"domain:{final.get('primary_domain') or 'none'}",
        f"target_presence:{target.get('observed_presence') or 'unknown'}",
        f"annotation_presence:{target.get('annotation_presence') or 'unknown'}",
        f"annotation_status:{target.get('visible_target_annotation_status') or 'unknown'}",
    }
    tokens.update(f"policy:{rule}" for rule in final.get("policy_rules_triggered") or [])
    return tokens


def _edge_score(candidate: dict) -> int:
    final = candidate["final"]
    evidence = candidate["evidence"]
    validation = candidate["validation"]
    target = evidence.get("target_presence") or {}
    score = 0
    if target.get("observed_presence") == "uncertain":
        score += 8
    elif target.get("observed_presence") == "partial":
        score += 5
    if target.get("visible_target_annotation_status") == "uncertain":
        score += 3
    domain = final.get("primary_domain")
    if domain and domain != "metadata_completeness":
        score += 4
    elif domain:
        score += 1
    risk = final.get("risk_level") or (evidence.get("triage") or {}).get("risk_level")
    score += {"medium": 3, "high": 6, "critical": 8}.get(str(risk).lower(), 0)
    if validation.get("status") != "valid" or validation.get("omit_from_training") is True:
        score += 8
    score += min(len(final.get("policy_rules_triggered") or []), 4)
    return score


def _choose_diverse(
    candidates: list[dict],
    count: int,
    seed: int,
    *,
    seen: set[str] | None = None,
) -> list[dict]:
    remaining = list(candidates)
    selected: list[dict] = []
    covered = set(seen or ())
    while remaining and len(selected) < count:
        best = min(
            remaining,
            key=lambda row: (
                -len(_feature_tokens(row) - covered),
                -_edge_score(row),
                _stable_key(row["case_id"], seed),
            ),
        )
        selected.append(best)
        covered.update(_feature_tokens(best))
        remaining.remove(best)
    return selected


def _resource_paths(validation: dict, required_segmentations: dict) -> tuple[str, str]:
    resources = validation.get("resources") or {}
    image_path = str((resources.get("image") or {}).get("path") or "")
    segmentation_dir = (resources.get("segmentation_dir") or {}).get("path")
    masks = {}
    if segmentation_dir:
        masks = {
            str(name): str(Path(segmentation_dir) / filename)
            for name, filename in sorted(required_segmentations.items())
        }
    for name, descriptor in sorted(resources.items()):
        if name in {"image", "segmentation_dir"} or not isinstance(descriptor, dict):
            continue
        if "mask" in name or "segmentation" in name:
            path = descriptor.get("path")
            if path:
                masks.setdefault(str(name), str(path))
    return image_path, json.dumps(masks, sort_keys=True, separators=(",", ":"))


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _protocol(package_id: str) -> str:
    return f"""# Human Golden Review Protocol

Package ID: `{package_id}`

## Scope

Review dataset suitability for the configured segmentation task. Do not assign a diagnosis, disease subtype, prognosis, or any other medical label. The reviewer records only annotation completeness and the expected dataset-curation action. Do not consult system outputs.

## Procedure

1. Open the image at `image_path` and each mask listed in the JSON object in `mask_paths`.
2. Inspect technical usability, image-mask geometry, target-mask presence, visible-target coverage, and field-of-view limitations relevant to the configured segmentation task.
3. Complete every reviewer field without changing `review_id`, `case_id`, or resource paths.
4. Review cases in `review_id` order and do not consult system outputs.

## Reviewer Fields

- `annotation_completeness`: `complete`, `incomplete`, or `uncertain`.
- `expected_action`: `keep`, `warning`, `review`, `reject`, or `insufficient_evidence`.
- `reviewer_id`: a non-empty stable reviewer identifier; do not enter a patient identifier.
- `confidence`: decimal confidence from `0` to `1`, inclusive.
- `rationale`: a concise, non-empty explanation grounded in the image, mask, and task. Do not include diagnostic labels.

Use `insufficient_evidence` when the available resources do not support a defensible curation action. Use `review` when additional human adjudication is the expected action rather than guessing.

After all rows are complete, return the reviewer package to the study administrator. Distribution, filesystem permissions, and reviewer access controls are operational responsibilities outside this package.
"""


def build_review_package(
    final_path: Path,
    validation_path: Path,
    deterministic_path: Path,
    output_dir: Path,
    *,
    seed: int = 20260722,
    overwrite: bool = False,
) -> dict:
    """Build separate administrative and reviewer-only package outputs."""
    final_metadata, final_data = load_artifact(final_path)
    validation_metadata, validation_data = load_artifact(validation_path)
    deterministic_metadata, deterministic_data = load_artifact(deterministic_path)
    if not all(isinstance(value, dict) for value in (final_data, validation_data, deterministic_data)):
        raise ValueError("All input artifacts must contain JSON object data.")

    metadata_values = [final_metadata, validation_metadata, deterministic_metadata]
    aligned = {}
    for field in ("run_id", "dataset_name", "task_mode"):
        values = {metadata.get(field) for metadata in metadata_values if metadata.get(field) is not None}
        if len(values) > 1:
            raise ValueError(f"Input artifact metadata mismatch for {field}: {sorted(values)}")
        aligned[field] = next(iter(values), None)

    final_cases = _cases(final_data, "decisions")
    validation_cases = _cases(validation_data, "cases")
    evidence_cases = _cases(deterministic_data, "cases")
    case_ids = set(final_cases) & set(validation_cases) & set(evidence_cases)
    candidates = [
        {
            "case_id": case_id,
            "final": final_cases[case_id],
            "validation": validation_cases[case_id],
            "evidence": evidence_cases[case_id],
        }
        for case_id in sorted(case_ids)
        if (validation_cases[case_id].get("resources") or {}).get("image", {}).get("path")
    ]
    if len(candidates) < TARGET_SIZE:
        raise ValueError(
            f"At least {TARGET_SIZE} aligned cases with image paths are required; found {len(candidates)}."
        )

    selected: list[dict] = []
    stages: dict[str, str] = {}
    for action, quota in QUOTAS.items():
        available = [row for row in candidates if row["final"].get("final_decision") == action]
        if len(available) < quota:
            raise ValueError(f"Review package requires {quota} {action} cases; found {len(available)}.")
        chosen = _choose_diverse(available, quota, seed)
        selected.extend(chosen)
        stages.update({row["case_id"]: f"quota:{action}" for row in chosen})

    rejects = [row for row in candidates if row["final"].get("final_decision") == "reject"]
    chosen_rejects = _choose_diverse(rejects, min(15, len(rejects)), seed)
    selected.extend(chosen_rejects)
    stages.update({row["case_id"]: "quota:reject" for row in chosen_rejects})

    selected_ids = {row["case_id"] for row in selected}
    fill_n = TARGET_SIZE - len(selected)
    if fill_n:
        remaining = [row for row in candidates if row["case_id"] not in selected_ids]
        covered = set().union(*(_feature_tokens(row) for row in selected)) if selected else set()
        fill = _choose_diverse(remaining, fill_n, seed, seen=covered)
        if len(fill) != fill_n:
            raise ValueError(f"Could not fill review package to exactly {TARGET_SIZE} cases.")
        selected.extend(fill)
        stages.update({row["case_id"]: "diverse_edge_fill" for row in fill})

    if len(selected) != TARGET_SIZE or len({row["case_id"] for row in selected}) != TARGET_SIZE:
        raise RuntimeError("Review selection did not produce exactly 60 unique cases.")

    descriptors = {
        "final": artifact_descriptor(final_path),
        "validation": artifact_descriptor(validation_path),
        "deterministic": artifact_descriptor(deterministic_path),
    }
    package_id = hash_json_payload({
        "seed": seed,
        "inputs": {name: value["sha256"] for name, value in descriptors.items()},
        "case_ids": sorted(row["case_id"] for row in selected),
    })
    ordered = sorted(selected, key=lambda row: _stable_key(row["case_id"], seed + 1))
    review_ids = {row["case_id"]: f"R{index:03d}" for index, row in enumerate(ordered, 1)}
    required_segmentations = validation_data.get("required_segmentations") or {}

    review_rows = []
    system_rows = []
    for row in sorted(selected, key=lambda item: review_ids[item["case_id"]]):
        case_id = row["case_id"]
        final = row["final"]
        evidence = row["evidence"]
        target = evidence.get("target_presence") or {}
        triage = evidence.get("triage") or {}
        image_path, mask_paths = _resource_paths(row["validation"], required_segmentations)
        review_rows.append({
            "review_id": review_ids[case_id],
            "case_id": case_id,
            "image_path": image_path,
            "mask_paths": mask_paths,
            "annotation_completeness": "",
            "expected_action": "",
            "reviewer_id": "",
            "confidence": "",
            "rationale": "",
        })
        system_rows.append({
            "package_id": package_id,
            "review_id": review_ids[case_id],
            "case_id": case_id,
            "selection_stage": stages[case_id],
            "system_action": final.get("final_decision"),
            "primary_domain": final.get("primary_domain") or "",
            "policy_rules_triggered": json.dumps(final.get("policy_rules_triggered") or [], separators=(",", ":")),
            "target_observed_presence": target.get("observed_presence") or "",
            "target_annotation_presence": target.get("annotation_presence") or "",
            "visible_target_annotation_status": target.get("visible_target_annotation_status") or "",
            "deterministic_recommendation": triage.get("recommendation") or "",
            "deterministic_risk_level": triage.get("risk_level") or "",
            "deterministic_score": triage.get("score") if triage.get("score") is not None else "",
            "dataset_validation_status": row["validation"].get("status") or "",
            "source_run_id": aligned["run_id"] or "",
            "dataset_name": aligned["dataset_name"] or validation_data.get("dataset_name") or "unknown",
            "task_mode": aligned["task_mode"] or validation_data.get("task_mode") or "unknown",
            "selection_seed": seed,
            "final_decisions_sha256": descriptors["final"]["sha256"] or "",
            "dataset_validation_sha256": descriptors["validation"]["sha256"] or "",
            "deterministic_evidence_sha256": descriptors["deterministic"]["sha256"] or "",
        })

    output_dir = Path(output_dir)
    reviewer_dir = output_dir / "reviewer_package"
    outputs = [
        reviewer_dir / "review_manifest.csv",
        output_dir / "system_reference.csv",
        reviewer_dir / "REVIEW_PROTOCOL.md",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Review package output already exists: {existing[0]}")
    if reviewer_dir.exists():
        unexpected = sorted(
            path.name for path in reviewer_dir.iterdir()
            if path.name not in {"review_manifest.csv", "REVIEW_PROTOCOL.md"}
        )
        if unexpected:
            raise ValueError(f"Reviewer package contains unexpected entries: {unexpected}")
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewer_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(outputs[0], REVIEW_FIELDS, review_rows)
    _write_csv(outputs[1], SYSTEM_FIELDS, system_rows)
    outputs[2].write_text(_protocol(package_id), encoding="utf-8")
    return {
        "package_id": package_id,
        "n_cases": len(selected),
        "action_counts": {
            action: sum(row["final"].get("final_decision") == action for row in selected)
            for action in ("keep", "warning", "review", "reject", "insufficient_evidence")
        },
        "outputs": [str(path) for path in outputs],
        "reviewer_package": str(reviewer_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--final-decisions", type=Path, default=None)
    parser.add_argument("--dataset-validation", type=Path, default=None)
    parser.add_argument("--deterministic-evidence", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.run_dir:
        final_path = args.final_decisions or args.run_dir / "final_qc_decisions.json"
        validation_path = args.dataset_validation or args.run_dir / "dataset_validation.json"
        deterministic_path = args.deterministic_evidence or args.run_dir / "qc_report_deterministic.json"
    else:
        final_path = args.final_decisions
        validation_path = args.dataset_validation
        deterministic_path = args.deterministic_evidence
    if not all((final_path, validation_path, deterministic_path)):
        parser.error("Provide --run-dir or all three input artifact paths.")
    summary = build_review_package(
        final_path,
        validation_path,
        deterministic_path,
        args.output_dir,
        seed=args.seed,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
