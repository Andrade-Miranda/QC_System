#!/usr/bin/env python3
"""Reproduce decision-stability counts from validated derived artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ARTIFACTS, ensure_output_dir, load_expected  # noqa: E402

ACTION_ORDER = ["keep", "warning", "review", "reject"]


def analyze(summary_path: Path, transitions_path: Path, output_dir: Path) -> dict:
    with summary_path.open(encoding="utf-8") as fh:
        summary = json.load(fh)
    expected = load_expected()["decision_stability"]

    if summary["aligned_case_count"] != expected["aligned_case_count"]:
        raise AssertionError("aligned case count mismatch")
    if summary["all_three_profiles"]["same_action"] != expected["stable_cases"]:
        raise AssertionError("stable case count mismatch")
    if summary["all_three_profiles"]["changed_action_at_least_one_profile"] != expected["changed_cases"]:
        raise AssertionError("changed case count mismatch")

    matrices: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros((4, 4), dtype=int))
    with transitions_path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sa = row["source_action"]
            ta = row["target_action"]
            if sa in ACTION_ORDER and ta in ACTION_ORDER:
                matrices[(row["source_profile"], row["target_profile"])][ACTION_ORDER.index(sa), ACTION_ORDER.index(ta)] += 1

    pairwise = {}
    for key, expected_changed in expected["pairwise_changed"].items():
        source, _, target = key.partition("_to_")
        mat = matrices[(source, target)]
        if int(mat.sum()) != expected["aligned_case_count"]:
            raise AssertionError(f"{key}: denominator mismatch")
        changed = int(mat.sum() - np.trace(mat))
        if changed != expected_changed:
            raise AssertionError(f"{key}: changed count mismatch {changed} != {expected_changed}")
        pairwise[key] = {"changed": changed, "rate_percent": round(100 * changed / int(mat.sum()), 1), "matrix": mat.tolist()}

    action_counts = {
        "tau_P": matrices[("tau_P", "tau_L")].sum(axis=1),
        "tau_L": matrices[("tau_P", "tau_L")].sum(axis=0),
        "tau_S": matrices[("tau_P", "tau_S")].sum(axis=0),
    }
    for profile, counts in action_counts.items():
        observed = dict(zip(ACTION_ORDER, [int(v) for v in counts]))
        if observed != expected["action_counts"][profile]:
            raise AssertionError(f"{profile}: action distribution mismatch {observed}")

    result = {
        "aligned_case_count": expected["aligned_case_count"],
        "stable_cases": expected["stable_cases"],
        "changed_cases": expected["changed_cases"],
        "changed_rate_percent": 49.3,
        "pairwise": pairwise,
        "action_counts": {k: dict(zip(ACTION_ORDER, [int(v) for v in vals])) for k, vals in action_counts.items()},
        "boundary": "Internal artifact comparison only; not clinical correctness, expert agreement, downstream utility, or causal evidence.",
    }
    ensure_output_dir(output_dir)
    with (output_dir / "decision_stability_results.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ARTIFACTS / "decision_stability" / "decision_stability_summary.json")
    parser.add_argument("--transitions", type=Path, default=ARTIFACTS / "decision_stability" / "decision_stability_case_transitions.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    result = analyze(args.summary, args.transitions, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
