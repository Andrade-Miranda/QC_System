#!/usr/bin/env python3
"""Regenerate eval_report.json for an artifact QC run directory."""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def build_evaluation_command(
    run_dir: Path,
    *,
    task_mode: str,
    run_id: str | None,
    paths_yaml: str | None,
    golden_labels: Path | None,
    python: str = sys.executable,
) -> list[str]:
    """Build a complete post-hoc audit command for an existing run."""
    cmd = [
        python,
        str(_ROOT / "agents" / "evaluation_agent.py"),
        "--task-mode", task_mode,
        "--output", str(run_dir / "eval_report.json"),
    ]
    artifact_flags = {
        "--validated-context": "validated_run_context.json",
        "--dataset-validation": "dataset_validation.json",
        "--deterministic-evidence": "qc_report_deterministic.json",
        "--calibrated-evidence": "qc_report_calibrated.json",
        "--comparison": "qc_comparison.json",
        "--reasoning": "reasoning_artifact.json",
        "--critique": "medical_critique.json",
        "--routing": "review_routing.json",
        "--final-decisions": "final_qc_decisions.json",
    }
    for flag, filename in artifact_flags.items():
        path = run_dir / filename
        if path.exists():
            cmd.extend([flag, str(path)])
    if paths_yaml:
        cmd.extend(["--paths-yaml", paths_yaml])
    if run_id:
        cmd.extend(["--run-id", run_id])
    if golden_labels is not None:
        cmd.extend(["--golden-labels", str(golden_labels)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate eval_report.json for a completed artifact QC run.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default=None, choices=["pancreas_only", "pancreas_lesion", "pancreas_lesion_subregions"])
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--golden-labels", type=Path, default=None)
    args = parser.parse_args()
    run_dir = args.run_dir
    graph_path = run_dir / "execution_graph.json"
    graph = json.loads(graph_path.read_text(encoding="utf-8")) if graph_path.exists() else {}
    task_mode = args.task_mode or graph.get("task_mode") or "pancreas_lesion"
    run_id = args.run_id or graph.get("run_id")
    default_golden = run_dir / "golden_review_package" / "golden_labels.json"
    golden_labels = args.golden_labels or (default_golden if default_golden.exists() else None)
    cmd = build_evaluation_command(
        run_dir,
        task_mode=task_mode,
        run_id=run_id,
        paths_yaml=args.paths_yaml,
        golden_labels=golden_labels,
    )
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
