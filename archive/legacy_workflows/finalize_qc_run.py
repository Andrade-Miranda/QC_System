#!/usr/bin/env python3
"""Regenerate comparison, routing, final decisions, and eval for a QC run."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TASK_MODES = ["pancreas_only", "pancreas_lesion", "pancreas_lesion_subregions"]


def run(cmd: list[str]) -> None:
    print(f"\n{'='*80}\nRunning: {' '.join(cmd)}\n{'='*80}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _common(args) -> list[str]:
    out = ["--task-mode", args.task_mode]
    if args.paths_yaml:
        out += ["--paths-yaml", args.paths_yaml]
    if args.run_id:
        out += ["--run-id", args.run_id]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize an artifact QC run from existing deterministic/calibrated reports.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default=None, choices=_TASK_MODES)
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    run_dir = args.run_dir
    graph_path = run_dir / "execution_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {}
    if args.task_mode is None:
        args.task_mode = graph.get("task_mode") or "pancreas_lesion"
    if args.run_id is None:
        args.run_id = graph.get("run_id")
    common = _common(args)
    py = sys.executable
    run([
        py, str(_ROOT / "agents" / "qc_comparison_agent.py"), *common,
        "--deterministic", str(run_dir / "qc_report_deterministic.json"),
        "--calibrated", str(run_dir / "qc_report_calibrated.json"),
        "--output", str(run_dir / "qc_comparison.json"),
        "--csv", str(run_dir / "qc_comparison.csv"),
        "--txt", str(run_dir / "calibration_impact_report.txt"),
    ])
    run([
        py, str(_ROOT / "agents" / "review_routing_agent.py"), *common,
        "--deterministic", str(run_dir / "qc_report_deterministic.json"),
        "--calibrated", str(run_dir / "qc_report_calibrated.json"),
        "--comparison", str(run_dir / "qc_comparison.json"),
        "--output", str(run_dir / "review_routing.json"),
        "--csv", str(run_dir / "manual_review_queue.csv"),
    ])
    run([
        py, str(_ROOT / "agents" / "final_decision_agent.py"), *common,
        "--deterministic", str(run_dir / "qc_report_deterministic.json"),
        "--calibrated", str(run_dir / "qc_report_calibrated.json"),
        "--routing", str(run_dir / "review_routing.json"),
        "--output", str(run_dir / "final_qc_decisions.json"),
        "--csv", str(run_dir / "final_qc_decisions.csv"),
    ])
    run([
        py, str(_ROOT / "agents" / "evaluation_agent.py"), *common,
        "--final-decisions", str(run_dir / "final_qc_decisions.json"),
        "--comparison", str(run_dir / "qc_comparison.json"),
        "--routing", str(run_dir / "review_routing.json"),
        "--output", str(run_dir / "eval_report.json"),
    ])


if __name__ == "__main__":
    main()
