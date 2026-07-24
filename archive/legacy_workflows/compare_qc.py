#!/usr/bin/env python3
"""
QC Run Comparison
=================
Compares two qc_report.json files produced by qc_agent.py and prints a diff
table of every case whose recommendation, risk level, score, or driving
domain changed between the two runs.

Useful after re-tuning thresholds to see which cases flipped.

Usage:
    python scripts/compare_qc.py REPORT_A REPORT_B
    python scripts/compare_qc.py REPORT_A REPORT_B --output diff.csv
    python scripts/compare_qc.py REPORT_A REPORT_B --changed-only --output diff.csv
    python scripts/compare_qc.py REPORT_A REPORT_B --label-a "v1" --label-b "v2"

Exit code: 0 always (comparison is informational).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from artifacts.loaders import load_qc_cases

# ===========================================================================
# HELPERS
# ===========================================================================

_VERDICT_ORDER = {"keep": 0, "review": 1, "exclude": 2, "omit_from_training": 3}
_RISK_ORDER    = {"low": 0, "medium": 1, "high": 2}


def _load_report(path: Path) -> tuple[dict, dict]:
    """Load legacy qc_report.json or artifact-wrapped QC report.

    Returns (cases_dict, metadata_dict).
    """
    cases, payload_meta, artifact_meta = load_qc_cases(path)
    meta = dict(payload_meta)
    if artifact_meta:
        meta["artifact_metadata"] = artifact_meta
    return cases, meta


def _fmt(val) -> str:
    if val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.1f}"
    return str(val)


def _escalation(old: str, new: str, order: dict) -> str:
    """Return '↑' (escalation), '↓' (de-escalation), or '=' (unchanged)."""
    o, n = order.get(old, -1), order.get(new, -1)
    if o < n:
        return "↑"
    if o > n:
        return "↓"
    return "="


def _score_delta_str(a, b) -> str:
    if a is None or b is None:
        return "—"
    try:
        d = float(b) - float(a)
        sign = "+" if d >= 0 else ""
        return f"{sign}{d:.1f}"
    except (TypeError, ValueError):
        return "—"


# ===========================================================================
# COMPARISON LOGIC
# ===========================================================================

def compare_reports(
    cases_a: dict,
    cases_b: dict,
    label_a: str = "A",
    label_b: str = "B",
    changed_only: bool = False,
) -> list[dict]:
    """Return a list of per-case comparison rows (dicts).

    Each row contains:
        case_id, rec_a, rec_b, rec_change (↑/↓/=),
        risk_a, risk_b, score_a, score_b, score_delta,
        domain_a, domain_b, changed (bool).
    """
    all_ids = sorted(set(cases_a) | set(cases_b))
    rows: list[dict] = []

    for case_id in all_ids:
        ca = cases_a.get(case_id)
        cb = cases_b.get(case_id)

        rec_a   = (ca or {}).get("recommendation", "—")
        rec_b   = (cb or {}).get("recommendation", "—")
        risk_a  = (ca or {}).get("risk_level",     "—")
        risk_b  = (cb or {}).get("risk_level",     "—")
        score_a = (ca or {}).get("qc_score")
        score_b = (cb or {}).get("qc_score")
        dom_a   = (ca or {}).get("driving_domain",  "—")
        dom_b   = (cb or {}).get("driving_domain",  "—")

        # A case is "changed" if anything clinically relevant shifted
        changed = (rec_a != rec_b
                   or risk_a != risk_b
                   or dom_a != dom_b
                   or (score_a is not None and score_b is not None
                       and abs(float(score_b) - float(score_a)) >= 1.0))

        if changed_only and not changed:
            continue

        rows.append({
            "case_id":    case_id,
            f"rec_{label_a}":   rec_a,
            f"rec_{label_b}":   rec_b,
            "rec_change": _escalation(rec_a, rec_b, _VERDICT_ORDER),
            f"risk_{label_a}":  risk_a,
            f"risk_{label_b}":  risk_b,
            "risk_change": _escalation(risk_a, risk_b, _RISK_ORDER),
            f"score_{label_a}": score_a,
            f"score_{label_b}": score_b,
            "score_delta": _score_delta_str(score_a, score_b),
            f"domain_{label_a}": dom_a,
            f"domain_{label_b}": dom_b,
            "changed":    changed,
            "only_in":    (label_a if cb is None
                           else (label_b if ca is None else "both")),
        })

    return rows


def _print_summary(rows: list[dict], cases_a: dict, cases_b: dict,
                   label_a: str, label_b: str) -> None:
    """Print a concise human-readable summary to stdout."""
    n_total = max(len(cases_a), len(cases_b))
    changed = [r for r in rows if r["changed"]]
    only_a  = [r for r in rows if r["only_in"] == label_a]
    only_b  = [r for r in rows if r["only_in"] == label_b]

    # Transition matrix: rec_A → rec_B counts
    transitions: dict[tuple[str, str], int] = {}
    for r in rows:
        if r["only_in"] == "both":
            key = (r[f"rec_{label_a}"], r[f"rec_{label_b}"])
            transitions[key] = transitions.get(key, 0) + 1

    print(f"\n{'='*64}")
    print(f"  QC Report Comparison: {label_a}  vs  {label_b}")
    print(f"{'='*64}")
    print(f"  Cases in {label_a}        : {len(cases_a)}")
    print(f"  Cases in {label_b}        : {len(cases_b)}")
    if only_a:
        print(f"  Only in {label_a}         : {len(only_a)}"
              f"  ({', '.join(r['case_id'] for r in only_a[:5])}"
              f"{'...' if len(only_a) > 5 else ''})")
    if only_b:
        print(f"  Only in {label_b}         : {len(only_b)}"
              f"  ({', '.join(r['case_id'] for r in only_b[:5])}"
              f"{'...' if len(only_b) > 5 else ''})")
    print(f"  Changed cases       : {len(changed)} / {n_total}")

    # Recommendation transition matrix
    verdicts = ("keep", "review", "exclude", "omit_from_training")
    any_transition = any(a != b for a, b in transitions)
    if any_transition:
        print(f"\n  Recommendation transitions ({label_a} \u2192 {label_b}):")
        header = f"  {'':18s}"
        for v in verdicts:
            if any(b == v for _, b in transitions):
                header += f"{v:>12}"
        print(header)
        for va in verdicts:
            if not any(a == va for a, _ in transitions):
                continue
            row_str = f"  {va:<18s}"
            for vb in verdicts:
                if any(b == vb for _, b in transitions):
                    n = transitions.get((va, vb), 0)
                    marker = " (*)" if va != vb and n > 0 else ""
                    row_str += f"{str(n) + marker:>12}"
            print(row_str)

    # Escalations / de-escalations
    escalated   = [r for r in changed if r["rec_change"] == "↑"]
    deescalated = [r for r in changed if r["rec_change"] == "↓"]
    score_only  = [r for r in changed if r["rec_change"] == "="
                   and r["only_in"] == "both"]

    if escalated:
        print(f"\n  Escalated ({len(escalated)}):")
        for r in escalated[:20]:
            print(f"    {r['case_id']:<20}  "
                  f"{r[f'rec_{label_a}']:>10} \u2192 {r[f'rec_{label_b}']:<10}  "
                  f"score {_fmt(r[f'score_{label_a}'])} \u2192 {_fmt(r[f'score_{label_b}'])} "
                  f"(\u0394{r['score_delta']})")
        if len(escalated) > 20:
            print(f"    ... and {len(escalated) - 20} more")

    if deescalated:
        print(f"\n  De-escalated ({len(deescalated)}):")
        for r in deescalated[:20]:
            print(f"    {r['case_id']:<20}  "
                  f"{r[f'rec_{label_a}']:>10} \u2192 {r[f'rec_{label_b}']:<10}  "
                  f"score {_fmt(r[f'score_{label_a}'])} \u2192 {_fmt(r[f'score_{label_b}'])} "
                  f"(\u0394{r['score_delta']})")
        if len(deescalated) > 20:
            print(f"    ... and {len(deescalated) - 20} more")

    if score_only:
        print(f"\n  Score/domain changed (same recommendation, {len(score_only)} cases):")
        for r in score_only[:10]:
            print(f"    {r['case_id']:<20}  "
                  f"rec={r[f'rec_{label_a}']:<8}  "
                  f"score {_fmt(r[f'score_{label_a}'])} \u2192 {_fmt(r[f'score_{label_b}'])} "
                  f"(\u0394{r['score_delta']})  "
                  f"domain {r[f'domain_{label_a}']} \u2192 {r[f'domain_{label_b}']}")
        if len(score_only) > 10:
            print(f"    ... and {len(score_only) - 10} more")

    print()


def _write_csv(rows: list[dict], output_path: Path, label_a: str, label_b: str) -> None:
    if not rows:
        print(f"[INFO] No rows to write to {output_path}")
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Diff CSV written: {output_path}  ({len(rows)} rows)")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two qc_report.json files and print a diff table.")
    parser.add_argument("report_a", metavar="REPORT_A",
                        help="Path to the baseline qc_report.json (run A).")
    parser.add_argument("report_b", metavar="REPORT_B",
                        help="Path to the comparison qc_report.json (run B).")
    parser.add_argument("--output", "-o", default=None, metavar="FILE",
                        help="Optional path to write the diff table as a CSV file.")
    parser.add_argument("--changed-only", action="store_true", dest="changed_only",
                        help="Only include cases where something changed in the output "
                             "(stdout print always shows all changed cases regardless).")
    parser.add_argument("--label-a", default="A", dest="label_a",
                        help="Display label for REPORT_A (default: 'A').")
    parser.add_argument("--label-b", default="B", dest="label_b",
                        help="Display label for REPORT_B (default: 'B').")
    args = parser.parse_args()

    path_a = Path(args.report_a)
    path_b = Path(args.report_b)

    if not path_a.exists():
        print(f"[ERROR] REPORT_A not found: {path_a}")
        raise SystemExit(1)
    if not path_b.exists():
        print(f"[ERROR] REPORT_B not found: {path_b}")
        raise SystemExit(1)

    print(f"Loading {path_a} ...")
    cases_a, meta_a = _load_report(path_a)
    print(f"  {len(cases_a)} cases  "
          f"(task_mode={meta_a.get('task_mode', '?')}, "
          f"created={meta_a.get('created_at', '?')})")

    print(f"Loading {path_b} ...")
    cases_b, meta_b = _load_report(path_b)
    print(f"  {len(cases_b)} cases  "
          f"(task_mode={meta_b.get('task_mode', '?')}, "
          f"created={meta_b.get('created_at', '?')})")

    # Warn if task modes differ
    if (meta_a.get("task_mode") and meta_b.get("task_mode")
            and meta_a["task_mode"] != meta_b["task_mode"]):
        print(f"\n[WARNING] task_mode mismatch: "
              f"{args.label_a}={meta_a['task_mode']}, "
              f"{args.label_b}={meta_b['task_mode']}")
        print("          Comparison may not be meaningful across different task modes.\n")

    rows = compare_reports(
        cases_a, cases_b,
        label_a=args.label_a,
        label_b=args.label_b,
        changed_only=False,  # always compute full rows; CSV uses changed_only filter
    )

    _print_summary(rows, cases_a, cases_b, args.label_a, args.label_b)

    if args.output:
        out_rows = [r for r in rows if r["changed"]] if args.changed_only else rows
        _write_csv(out_rows, Path(args.output), args.label_a, args.label_b)


if __name__ == "__main__":
    main()
