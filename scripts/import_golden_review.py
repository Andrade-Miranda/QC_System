#!/usr/bin/env python3
"""Import a completed blinded review CSV as a golden-label artifact."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from artifacts import hash_json_payload, resource_descriptor, write_artifact


ANNOTATION_COMPLETENESS = {"complete", "incomplete", "uncertain"}
EXPECTED_ACTIONS = {"keep", "warning", "review", "reject", "insufficient_evidence"}
REQUIRED_REVIEW_FIELDS = {
    "review_id",
    "case_id",
    "annotation_completeness",
    "expected_action",
    "reviewer_id",
    "confidence",
    "rationale",
}
REQUIRED_REFERENCE_FIELDS = {
    "package_id",
    "review_id",
    "case_id",
    "source_run_id",
    "dataset_name",
    "task_mode",
    "selection_seed",
    "final_decisions_sha256",
    "dataset_validation_sha256",
    "deterministic_evidence_sha256",
}


def _read_csv(path: Path, required_fields: set[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = sorted(required_fields - fields)
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        return [dict(row) for row in reader]


def _unique_rows(rows: list[dict[str, str]], path: Path) -> dict[str, dict[str, str]]:
    indexed = {}
    for line_number, row in enumerate(rows, 2):
        review_id = (row.get("review_id") or "").strip()
        if not review_id:
            raise ValueError(f"{path}:{line_number}: review_id is required.")
        if review_id in indexed:
            raise ValueError(f"{path}:{line_number}: duplicate review_id {review_id!r}.")
        indexed[review_id] = row
    return indexed


def _one_reference_value(rows: list[dict[str, str]], field: str) -> str:
    values = {(row.get(field) or "").strip() for row in rows}
    if len(values) != 1 or not next(iter(values), ""):
        raise ValueError(f"system_reference.csv must contain one non-empty {field} value.")
    return next(iter(values))


def _case_ids(indexed: dict[str, dict[str, str]], path: Path) -> set[str]:
    case_ids = []
    for review_id, row in indexed.items():
        case_id = (row.get("case_id") or "").strip()
        if not case_id:
            raise ValueError(f"{path}: review_id {review_id!r} has no case_id.")
        case_ids.append(case_id)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"{path} contains duplicate case_id values.")
    return set(case_ids)


def import_golden_review(
    review_csv: Path,
    system_reference: Path,
    output: Path,
    *,
    overwrite: bool = False,
) -> dict:
    """Validate a completed package and write a provenance-aware artifact."""
    review_rows = _read_csv(review_csv, REQUIRED_REVIEW_FIELDS)
    reference_rows = _read_csv(system_reference, REQUIRED_REFERENCE_FIELDS)
    if not review_rows:
        raise ValueError("Completed review CSV contains no cases.")
    reviews = _unique_rows(review_rows, review_csv)
    references = _unique_rows(reference_rows, system_reference)
    review_case_ids = _case_ids(reviews, review_csv)
    reference_case_ids = _case_ids(references, system_reference)
    if set(reviews) != set(references):
        missing = sorted(set(references) - set(reviews))
        unexpected = sorted(set(reviews) - set(references))
        raise ValueError(
            "Review/reference package mismatch: "
            f"missing review IDs={missing}, unexpected review IDs={unexpected}."
        )
    if review_case_ids != reference_case_ids:
        raise ValueError("Review/reference package case IDs do not match.")

    labels = {}
    for review_id in sorted(reviews):
        row = reviews[review_id]
        reference = references[review_id]
        case_id = (row.get("case_id") or "").strip()
        reference_case_id = (reference.get("case_id") or "").strip()
        if not case_id or case_id != reference_case_id:
            raise ValueError(
                f"{review_csv}: review_id {review_id!r} has case_id {case_id!r}; "
                f"expected {reference_case_id!r}."
            )
        completeness = (row.get("annotation_completeness") or "").strip().lower()
        if completeness not in ANNOTATION_COMPLETENESS:
            raise ValueError(
                f"{review_csv}: {review_id} annotation_completeness must be one of "
                f"{sorted(ANNOTATION_COMPLETENESS)}."
            )
        expected_action = (row.get("expected_action") or "").strip().lower()
        if expected_action not in EXPECTED_ACTIONS:
            raise ValueError(
                f"{review_csv}: {review_id} expected_action must be one of "
                f"{sorted(EXPECTED_ACTIONS)}."
            )
        reviewer_id = (row.get("reviewer_id") or "").strip()
        if not reviewer_id:
            raise ValueError(f"{review_csv}: {review_id} reviewer_id is required.")
        raw_confidence = (row.get("confidence") or "").strip()
        try:
            confidence = float(raw_confidence)
        except ValueError as exc:
            raise ValueError(f"{review_csv}: {review_id} confidence must be numeric.") from exc
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(f"{review_csv}: {review_id} confidence must be between 0 and 1.")
        rationale = (row.get("rationale") or "").strip()
        if not rationale:
            raise ValueError(f"{review_csv}: {review_id} rationale is required.")
        labels[case_id] = {
            "case_id": case_id,
            "review_id": review_id,
            "annotation_completeness": completeness,
            "expected_action": expected_action,
            "reviewer_id": reviewer_id,
            "confidence": confidence,
            "rationale": rationale,
        }

    package_id = _one_reference_value(reference_rows, "package_id")
    source_run_id = _one_reference_value(reference_rows, "source_run_id")
    dataset_name = _one_reference_value(reference_rows, "dataset_name")
    task_mode = _one_reference_value(reference_rows, "task_mode")
    selection_seed_text = _one_reference_value(reference_rows, "selection_seed")
    try:
        selection_seed = int(selection_seed_text)
    except ValueError as exc:
        raise ValueError("system_reference.csv selection_seed must be an integer.") from exc
    source_hashes = {
        "final": _one_reference_value(reference_rows, "final_decisions_sha256"),
        "validation": _one_reference_value(reference_rows, "dataset_validation_sha256"),
        "deterministic": _one_reference_value(reference_rows, "deterministic_evidence_sha256"),
    }
    expected_package_id = hash_json_payload({
        "seed": selection_seed,
        "inputs": source_hashes,
        "case_ids": sorted(reference_case_ids),
    })
    if package_id != expected_package_id:
        raise ValueError("system_reference.csv package_id does not match its cases and source provenance.")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Golden-label artifact already exists: {output}")

    reviewer_counts = Counter(row["reviewer_id"] for row in labels.values())
    action_counts = Counter(row["expected_action"] for row in labels.values())
    data = {
        "review_provenance": {
            "package_id": package_id,
            "source_run_id": source_run_id,
            "selection_seed": selection_seed,
            "source_artifact_sha256": source_hashes,
            "review_csv": resource_descriptor(review_csv, "completed_review_csv"),
            "system_reference": resource_descriptor(system_reference, "system_reference_csv"),
        },
        "annotation_schema": {
            "scope": "dataset_curation_only_no_medical_labels",
            "annotation_completeness": sorted(ANNOTATION_COMPLETENESS),
            "expected_actions": sorted(EXPECTED_ACTIONS),
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "summary": {
            "n_labels": len(labels),
            "reviewer_counts": dict(sorted(reviewer_counts.items())),
            "expected_action_counts": dict(sorted(action_counts.items())),
        },
        "labels": labels,
    }
    payload = write_artifact(
        output,
        artifact_type="golden_labels",
        generator="HumanGoldenReviewImporter",
        data=data,
        dataset_name=dataset_name,
        task_mode=task_mode,
        input_resources=[
            resource_descriptor(review_csv, "completed_review_csv"),
            resource_descriptor(system_reference, "system_reference_csv"),
        ],
        configuration={
            "package_id": package_id,
            "annotation_scope": "dataset_curation_only_no_medical_labels",
        },
        run_id=source_run_id,
        project_root=_ROOT,
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    review_source = parser.add_mutually_exclusive_group(required=True)
    review_source.add_argument("--review-csv", type=Path)
    review_source.add_argument(
        "--review-package",
        type=Path,
        help="Sanitized reviewer package containing review_manifest.csv",
    )
    parser.add_argument("--system-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    review_csv = args.review_csv or args.review_package / "review_manifest.csv"
    payload = import_golden_review(
        review_csv,
        args.system_reference,
        args.output,
        overwrite=args.overwrite,
    )
    print(json.dumps({
        "output": str(args.output),
        "artifact_id": payload["metadata"]["artifact_id"],
        "n_labels": payload["data"]["summary"]["n_labels"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
