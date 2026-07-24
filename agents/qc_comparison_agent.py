#!/usr/bin/env python3
"""QCComparisonAgent: compare deterministic and calibrated QC artifacts."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import resolve_project_paths, VALID_TASK_MODES
from artifacts.loaders import load_qc_cases
from artifacts.io import artifact_descriptor, write_artifact


_REC_ORDER = {"keep": 0, "keep_with_metadata_warning": 0, "review": 1, "exclude": 2, "omit_from_training": 3}
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def _report_cases(path: Path) -> tuple[dict, dict]:
    cases, payload_meta, artifact_meta = load_qc_cases(path)
    return cases, {**payload_meta, "artifact_metadata": artifact_meta} if artifact_meta else payload_meta


def _triage(case: dict) -> dict:
    tri = case.get("triage") or {}
    return {
        "recommendation": tri.get("recommendation", case.get("recommendation", "unknown")),
        "risk_level": tri.get("risk_level", case.get("risk_level", "unknown")),
        "score": tri.get("score", case.get("qc_score", case.get("score"))),
        "driving_domain": case.get("driving_domain") or (case.get("primary_issue") or {}).get("domain"),
    }


def _change(a: str, b: str, order: dict) -> str:
    av, bv = order.get(a, -1), order.get(b, -1)
    if bv > av:
        return "escalated"
    if bv < av:
        return "deescalated"
    return "unchanged"


def compare(det_cases: dict, cal_cases: dict) -> dict:
    rows = []
    rec_transitions = Counter()
    domain_changes = Counter()
    for case_id in sorted(set(det_cases) | set(cal_cases)):
        d = _triage(det_cases.get(case_id) or {})
        c = _triage(cal_cases.get(case_id) or {})
        score_delta = None
        if d["score"] is not None and c["score"] is not None:
            score_delta = float(c["score"]) - float(d["score"])
        changed = (
            d["recommendation"] != c["recommendation"]
            or d["risk_level"] != c["risk_level"]
            or d["driving_domain"] != c["driving_domain"]
            or (score_delta is not None and abs(score_delta) >= 1.0)
        )
        rec_transitions[(d["recommendation"], c["recommendation"])] += 1
        if d["driving_domain"] != c["driving_domain"]:
            domain_changes[f"{d['driving_domain']}->{c['driving_domain']}"] += 1
        rows.append({
            "case_id": case_id,
            "recommendation_deterministic": d["recommendation"],
            "recommendation_calibrated": c["recommendation"],
            "recommendation_change": _change(d["recommendation"], c["recommendation"], _REC_ORDER),
            "risk_deterministic": d["risk_level"],
            "risk_calibrated": c["risk_level"],
            "risk_change": _change(d["risk_level"], c["risk_level"], _RISK_ORDER),
            "score_deterministic": d["score"],
            "score_calibrated": c["score"],
            "score_delta": score_delta,
            "domain_deterministic": d["driving_domain"],
            "domain_calibrated": c["driving_domain"],
            "changed": changed,
        })
    changed_rows = [r for r in rows if r["changed"]]
    return {
        "summary": {
            "n_cases_deterministic": len(det_cases),
            "n_cases_calibrated": len(cal_cases),
            "n_cases_compared": len(rows),
            "n_changed_cases": len(changed_rows),
            "n_recommendation_changes": sum(1 for r in rows if r["recommendation_deterministic"] != r["recommendation_calibrated"]),
            "n_risk_changes": sum(1 for r in rows if r["risk_deterministic"] != r["risk_calibrated"]),
            "recommendation_transitions": {f"{a}->{b}": n for (a, b), n in sorted(rec_transitions.items())},
            "domain_transition_counts": dict(sorted(domain_changes.items())),
        },
        "changed_cases": changed_rows,
        "all_cases": rows,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("case_id\n", encoding="utf-8")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_txt(path: Path, data: dict) -> None:
    s = data["summary"]
    lines = [
        "QC Calibration Impact Report",
        "============================",
        f"Cases compared: {s['n_cases_compared']}",
        f"Changed cases: {s['n_changed_cases']}",
        f"Recommendation changes: {s['n_recommendation_changes']}",
        f"Risk changes: {s['n_risk_changes']}",
        "",
        "Recommendation transitions:",
    ]
    for key, n in s["recommendation_transitions"].items():
        lines.append(f"  {key}: {n}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare deterministic and calibrated QC reports.")
    parser.add_argument("--deterministic", required=True, type=Path)
    parser.add_argument("--calibrated", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--txt", type=Path, default=None)
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    det_cases, det_meta = _report_cases(args.deterministic)
    cal_cases, cal_meta = _report_cases(args.calibrated)
    data = compare(det_cases, cal_cases)
    data["input_metadata"] = {"deterministic": det_meta, "calibrated": cal_meta}
    write_artifact(
        args.output,
        artifact_type="qc_comparison",
        generator="QCComparisonAgent",
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        input_artifacts=[artifact_descriptor(args.deterministic, "qc_report_deterministic"), artifact_descriptor(args.calibrated, "qc_report_calibrated")],
        run_id=args.run_id,
        project_root=paths.project_root,
    )
    _write_csv(args.csv or args.output.with_suffix(".csv"), data["all_cases"])
    _write_txt(args.txt or args.output.with_name("calibration_impact_report.txt"), data)
    print(f"Wrote QC comparison artifact: {args.output}")


if __name__ == "__main__":
    main()
