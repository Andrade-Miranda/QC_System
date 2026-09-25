#!/usr/bin/env python3
"""Verify and export stable-vs-task-changing evidence-domain results."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ARTIFACTS, assert_close, ensure_output_dir, load_expected  # noqa: E402


def analyze(input_csv: Path, output_dir: Path) -> list[dict]:
    expected = load_expected()["evidence_domains"]
    rows: list[dict] = []
    with input_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            domain = row["evidence_domain"]
            stable_n = int(row["stable_N"])
            changing_n = int(row["profile_changing_N"])
            if stable_n != expected["stable_n"] or changing_n != expected["task_changing_n"]:
                raise AssertionError(f"{domain}: denominator mismatch")
            stable = float(row["stable_percent"])
            changing = float(row["profile_changing_percent"])
            assert_close(stable, expected["percentages"][domain]["stable"], label=f"{domain} stable")
            assert_close(changing, expected["percentages"][domain]["task_changing"], label=f"{domain} task-changing")
            rows.append({
                "evidence_domain": domain,
                "stable_n": int(row["stable_n"]),
                "stable_N": stable_n,
                "stable_percent": stable,
                "task_changing_n": int(row["profile_changing_n"]),
                "task_changing_N": changing_n,
                "task_changing_percent": changing,
            })
    ensure_output_dir(output_dir)
    with (output_dir / "evidence_domain_results.json").open("w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "boundary": "Case-level multi-label descriptive frequencies; rows are not mutually exclusive and are not causal effects."}, fh, indent=2)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ARTIFACTS / "evidence_domains" / "changed_vs_stable_evidence_domains.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    rows = analyze(args.input, args.output_dir)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
