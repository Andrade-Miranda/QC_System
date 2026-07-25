#!/usr/bin/env python3
"""OrchestratorAgent for the artifact-based deterministic QC workflow."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import build_output_dirs, resolve_project_paths, VALID_TASK_MODES
from agents.final_decision_agent import VISIBLE_PANCREAS_POLICY_PATH
from agents.validated_run_context_agent import (
    generate_validated_run_context,
    validate_reusable_calibrated_thresholds,
)
from artifacts import read_json, resource_descriptor, write_artifact, write_json
from artifacts.io import artifact_descriptor
from artifacts.run_manifest import write_pointer_alias, write_run_manifest


def _task_profile_path(project_root: Path, task_mode: str) -> Path:
    return project_root / "configs" / "task_profiles" / f"{task_mode}.yaml"


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("\n" + "=" * 80)
    print("Running: " + " ".join(cmd))
    print("=" * 80)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def _step(step_id: str, agent: str, output: Path | None, *dependencies: str) -> dict:
    return {
        "id": step_id,
        "agent": agent,
        "dependencies": list(dependencies),
        "output": str(output) if output is not None else None,
    }


def build_execution_graph(
    *,
    run_id: str,
    dataset_name: str,
    task_mode: str,
    run_dir: Path,
    steps: list[dict],
    graph_path: Path,
) -> dict:
    """Build the serial execution record while retaining explicit data dependencies."""
    seen: set[str] = set()
    for step in steps:
        step_id = step.get("id")
        if not step_id or step_id in seen:
            raise ValueError(f"Execution step IDs must be unique and non-empty: {step_id!r}")
        missing = set(step.get("dependencies") or []) - seen
        if missing:
            raise ValueError(
                f"Execution step {step_id!r} precedes dependencies: {sorted(missing)}"
            )
        seen.add(step_id)
    return {
        "run_id": run_id,
        "dataset_name": dataset_name,
        "task_mode": task_mode,
        "run_dir": str(run_dir),
        "execution_graph_output": str(graph_path),
        "steps": steps,
    }


def _wrap_json_artifact(
    *,
    src: Path,
    dst: Path,
    artifact_type: str,
    generator: str,
    paths,
    task_mode: str,
    run_id: str,
    input_resources: list[dict] | None = None,
    input_artifacts: list[dict] | None = None,
    configuration: dict | None = None,
) -> None:
    data = read_json(src)
    write_artifact(
        dst,
        artifact_type=artifact_type,
        generator=generator,
        data=data,
        dataset_name=paths.dataset_name,
        task_mode=task_mode,
        input_resources=input_resources or [],
        input_artifacts=input_artifacts or [],
        configuration=configuration or {},
        run_id=run_id,
        project_root=paths.project_root,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the artifact-based deterministic QC multi-agent workflow.")
    parser.add_argument("--paths-yaml", default=None)
    parser.add_argument("--task-mode", default="pancreas_lesion", choices=sorted(VALID_TASK_MODES))
    parser.add_argument("--score-version", default="hybrid", choices=["v2", "hybrid"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-summary", action="store_true", help="Reuse existing task summary artifact/source JSON.")
    parser.add_argument("--skip-calibration", action="store_true", help="Reuse explicitly supplied calibrated thresholds.")
    parser.add_argument(
        "--calibrated-thresholds",
        type=Path,
        default=None,
        help="Existing calibrated thresholds to reuse with --skip-calibration.",
    )
    parser.add_argument("--golden-labels", type=Path, default=None, help="Optional golden labels artifact for evaluation.")
    parser.add_argument(
        "--confirmed-negative-lesions",
        type=Path,
        default=None,
        help="Optional JSON manifest of explicitly confirmed absent lesion cases.",
    )
    args = parser.parse_args()

    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    task_dirs = build_output_dirs(paths.summary_dir.parent, args.task_mode)
    task_profile = _task_profile_path(paths.project_root, args.task_mode)
    run_id = args.run_id or uuid.uuid4().hex[:12]
    run_dir = args.run_dir or task_dirs["task_dir"] / "runs" / run_id
    if run_dir.exists() and not args.overwrite:
        raise SystemExit(f"[ERROR] Run directory exists: {run_dir}\n        Use --overwrite or choose --run-id/--run-dir.")
    run_dir.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    root = paths.project_root
    paths_arg = ["--paths-yaml", str(paths.paths_yaml)] if paths.paths_yaml else []

    execution_steps: list[dict] = []

    _run([
        py, str(root / "scripts" / "validate_config.py"),
        *paths_arg,
        "--task-mode", args.task_mode,
    ], cwd=root)
    execution_steps.append(_step("config_validation", "ConfigValidation", None))

    summary_src = task_dirs["summary_dir"] / f"{paths.dataset_name}_summary.json"
    if args.skip_calibration and args.calibrated_thresholds is None:
        raise SystemExit("[ERROR] --skip-calibration requires --calibrated-thresholds.")
    if not args.skip_calibration and args.calibrated_thresholds is not None:
        raise SystemExit("[ERROR] --calibrated-thresholds requires --skip-calibration.")
    calibrated_thresholds = (
        args.calibrated_thresholds.resolve()
        if args.calibrated_thresholds is not None
        else run_dir / "thresholds.calibrated.yaml"
    )
    if args.skip_calibration:
        try:
            validate_reusable_calibrated_thresholds(
                calibrated_thresholds,
                summary_path=summary_src,
                task_mode=args.task_mode,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise SystemExit(f"[ERROR] Cannot reuse calibrated thresholds: {exc}") from exc

    context_path = run_dir / "validated_run_context.json"
    generate_validated_run_context(
        paths=paths,
        task_mode=args.task_mode,
        profile_path=task_profile,
        run_id=run_id,
        output_path=context_path,
        run_dir=run_dir,
        output_dirs=task_dirs,
        reuse_calibrated=args.skip_calibration,
        calibrated_thresholds_path=calibrated_thresholds,
        confirmed_negative_lesions_path=args.confirmed_negative_lesions,
    )
    execution_steps.append(
        _step("validated_context", "ValidatedRunContextAgent", context_path, "config_validation")
    )

    validation_path = run_dir / "dataset_validation.json"
    validation_cmd = [
        py, str(root / "agents" / "validation_agent.py"),
        *paths_arg,
        "--task-mode", args.task_mode,
        "--output", str(validation_path),
        "--run-id", run_id,
    ]
    if args.confirmed_negative_lesions is not None:
        validation_cmd.extend(["--confirmed-negative-lesions", str(args.confirmed_negative_lesions)])
    _run(validation_cmd, cwd=root)
    execution_steps.append(
        _step("dataset_validation", "ValidationAgent", validation_path, "validated_context")
    )

    if not args.skip_summary:
        cmd = [
            py, str(root / "scripts" / "summarize_dataset.py"),
            *paths_arg,
            "--task-mode", args.task_mode,
            "--overwrite",
        ]
        if args.workers > 1:
            cmd += ["--workers", str(args.workers)]
        _run(cmd, cwd=root)
    summary_artifact = run_dir / "summary.json"
    _wrap_json_artifact(
        src=summary_src,
        dst=summary_artifact,
        artifact_type="summary",
        generator="SummaryAgent",
        paths=paths,
        task_mode=args.task_mode,
        run_id=run_id,
        input_resources=[resource_descriptor(paths.raw_root, "raw_dataset_root"), resource_descriptor(paths.paths_yaml, "paths_config")],
        input_artifacts=[artifact_descriptor(validation_path, "dataset_validation")],
        configuration={"task_profile": str(task_profile)},
    )
    execution_steps.append(
        _step("summary", "SummaryAgent", summary_artifact, "dataset_validation")
    )

    if not args.skip_calibration:
        _run([
            py, str(root / "scripts" / "calibrate_thresholds.py"),
            "--summary", str(summary_artifact),
            "--base-config", str(paths.thresholds_config),
            "--output", str(calibrated_thresholds),
            "--task-mode", args.task_mode,
            "--task-profile", str(task_profile),
        ], cwd=root)
    calibration_src = calibrated_thresholds.with_suffix(".calibration_report.json")
    calibration_artifact = run_dir / "threshold_calibration.json"
    if calibration_src.exists():
        _wrap_json_artifact(
            src=calibration_src,
            dst=calibration_artifact,
            artifact_type="threshold_calibration",
            generator="CalibrationAgent",
            paths=paths,
            task_mode=args.task_mode,
            run_id=run_id,
            input_resources=[resource_descriptor(paths.thresholds_config, "deterministic_thresholds")],
            input_artifacts=[artifact_descriptor(summary_artifact, "summary")],
            configuration={"task_profile": str(task_profile)},
        )
        execution_steps.append(
            _step("calibration", "CalibrationAgent", calibration_artifact, "summary")
        )
    elif not args.skip_calibration:
        execution_steps.append(
            _step(
                "calibration",
                "CalibrationAgent",
                calibrated_thresholds,
                "summary",
            )
        )

    det_raw_dir = run_dir / "raw_qc_deterministic"
    cal_raw_dir = run_dir / "raw_qc_calibrated"
    confirmed_args = (
        ["--confirmed-negative-lesions", str(args.confirmed_negative_lesions)]
        if args.confirmed_negative_lesions is not None
        else []
    )
    _run([
        py, str(root / "agents" / "qc_agent.py"),
        *paths_arg,
        "--json", str(summary_artifact),
        "--report-dir", str(det_raw_dir),
        "--threshold-method", "deterministic",
        "--task-mode", args.task_mode,
        "--score-version", args.score_version,
        "--overwrite",
        "--no-interactive",
        *confirmed_args,
    ], cwd=root)
    _run([
        py, str(root / "agents" / "qc_agent.py"),
        *paths_arg,
        "--json", str(summary_artifact),
        "--report-dir", str(cal_raw_dir),
        "--thresholds-yaml", str(calibrated_thresholds),
        "--task-mode", args.task_mode,
        "--score-version", args.score_version,
        "--overwrite",
        "--no-interactive",
        *confirmed_args,
    ], cwd=root)

    det_artifact = run_dir / "qc_report_deterministic.json"
    cal_artifact = run_dir / "qc_report_calibrated.json"
    _wrap_json_artifact(
        src=det_raw_dir / "qc_report.json",
        dst=det_artifact,
        artifact_type="qc_report_deterministic",
        generator="DeterministicQCAgent",
        paths=paths,
        task_mode=args.task_mode,
        run_id=run_id,
        input_resources=[resource_descriptor(paths.thresholds_config, "deterministic_thresholds"), resource_descriptor(task_profile, "task_profile")],
        input_artifacts=[artifact_descriptor(summary_artifact, "summary")],
        configuration={"score_version": args.score_version},
    )
    _wrap_json_artifact(
        src=cal_raw_dir / "qc_report.json",
        dst=cal_artifact,
        artifact_type="qc_report_calibrated",
        generator="CalibratedQCAgent",
        paths=paths,
        task_mode=args.task_mode,
        run_id=run_id,
        input_resources=[resource_descriptor(calibrated_thresholds, "calibrated_thresholds"), resource_descriptor(task_profile, "task_profile")],
        input_artifacts=[
            artifact_descriptor(summary_artifact, "summary"),
            *(
                [artifact_descriptor(calibration_artifact, "threshold_calibration")]
                if calibration_artifact.exists()
                else []
            ),
        ],
        configuration={"score_version": args.score_version},
    )
    _copy_if_exists(det_raw_dir / "qc_report.csv", run_dir / "qc_report_deterministic.csv")
    _copy_if_exists(det_raw_dir / "qc_report.txt", run_dir / "qc_report_deterministic.txt")
    _copy_if_exists(cal_raw_dir / "qc_report.csv", run_dir / "qc_report_calibrated.csv")
    _copy_if_exists(cal_raw_dir / "qc_report.txt", run_dir / "qc_report_calibrated.txt")
    execution_steps.extend([
        _step("deterministic_qc", "DeterministicQCAgent", det_artifact, "summary", "validated_context"),
        _step(
            "calibrated_qc",
            "CalibratedQCAgent",
            cal_artifact,
            "summary",
            "calibration" if any(step["id"] == "calibration" for step in execution_steps) else "validated_context",
        ),
    ])

    comparison = run_dir / "qc_comparison.json"
    _run([
        py, str(root / "agents" / "qc_comparison_agent.py"),
        *paths_arg,
        "--task-mode", args.task_mode,
        "--deterministic", str(det_artifact),
        "--calibrated", str(cal_artifact),
        "--output", str(comparison),
        "--csv", str(run_dir / "qc_comparison.csv"),
        "--txt", str(run_dir / "calibration_impact_report.txt"),
        "--run-id", run_id,
    ], cwd=root)
    execution_steps.append(
        _step("comparison", "QCComparisonAgent", comparison, "deterministic_qc", "calibrated_qc")
    )

    reasoning = run_dir / "reasoning_artifact.json"
    _run([
        py, str(root / "agents" / "reasoning_agent.py"),
        "--context", str(context_path),
        "--deterministic", str(det_artifact),
        "--calibrated", str(cal_artifact),
        "--comparison", str(comparison),
        "--output", str(reasoning),
        "--dataset-name", paths.dataset_name,
        "--task-mode", args.task_mode,
        "--run-id", run_id,
    ], cwd=root)
    execution_steps.append(
        _step(
            "reasoning",
            "ReasoningAgent",
            reasoning,
            "validated_context",
            "deterministic_qc",
            "calibrated_qc",
            "comparison",
        )
    )

    critique = run_dir / "medical_critique.json"
    _run([
        py, str(root / "agents" / "medical_critic_agent.py"),
        "--reasoning", str(reasoning),
        "--deterministic", str(det_artifact),
        "--calibrated", str(cal_artifact),
        "--comparison", str(comparison),
        "--output", str(critique),
        "--dataset-name", paths.dataset_name,
        "--task-mode", args.task_mode,
        "--run-id", run_id,
    ], cwd=root)
    execution_steps.append(
        _step(
            "medical_critique",
            "MedicalCriticAgent",
            critique,
            "reasoning",
            "deterministic_qc",
            "calibrated_qc",
            "comparison",
        )
    )

    routing = run_dir / "review_routing.json"
    _run([
        py, str(root / "agents" / "review_routing_agent.py"),
        *paths_arg,
        "--task-mode", args.task_mode,
        "--deterministic", str(det_artifact),
        "--calibrated", str(cal_artifact),
        "--comparison", str(comparison),
        "--dataset-validation", str(validation_path),
        "--output", str(routing),
        "--csv", str(run_dir / "manual_review_queue.csv"),
        "--run-id", run_id,
    ], cwd=root)
    execution_steps.append(
        _step(
            "routing",
            "ReviewRoutingAgent",
            routing,
            "dataset_validation",
            "deterministic_qc",
            "calibrated_qc",
            "comparison",
        )
    )
    final = run_dir / "final_qc_decisions.json"
    final_cmd = [
        py, str(root / "agents" / "final_decision_agent.py"),
        *paths_arg,
        "--task-mode", args.task_mode,
        "--deterministic", str(det_artifact),
        "--calibrated", str(cal_artifact),
        "--routing", str(routing),
        "--comparison", str(comparison),
        "--dataset-validation", str(validation_path),
        "--output", str(final),
        "--csv", str(run_dir / "final_qc_decisions.csv"),
        "--run-id", run_id,
    ]
    if args.task_mode == "pancreas_only":
        policy_snapshot = run_dir / "decision_policy.yaml"
        shutil.copy2(VISIBLE_PANCREAS_POLICY_PATH, policy_snapshot)
        final_cmd.extend(["--policy", str(policy_snapshot)])
    _run(final_cmd, cwd=root)
    execution_steps.append(
        _step(
            "final_decisions",
            "FinalDecisionAgent",
            final,
            "dataset_validation",
            "deterministic_qc",
            "calibrated_qc",
            "comparison",
            "routing",
        )
    )
    eval_report = run_dir / "eval_report.json"
    evaluation_cmd = [
        py, str(root / "agents" / "evaluation_agent.py"),
        *paths_arg,
        "--task-mode", args.task_mode,
        "--validated-context", str(context_path),
        "--dataset-validation", str(validation_path),
        "--deterministic-evidence", str(det_artifact),
        "--calibrated-evidence", str(cal_artifact),
        "--comparison", str(comparison),
        "--reasoning", str(reasoning),
        "--critique", str(critique),
        "--routing", str(routing),
        "--final-decisions", str(final),
        "--output", str(eval_report),
        "--run-id", run_id,
    ]
    if args.golden_labels is not None:
        evaluation_cmd.extend(["--golden-labels", str(args.golden_labels.resolve())])
    _run(evaluation_cmd, cwd=root)
    execution_steps.append(
        _step(
            "evaluation",
            "EvaluationAgent",
            eval_report,
            "validated_context",
            "dataset_validation",
            "deterministic_qc",
            "calibrated_qc",
            "comparison",
            "reasoning",
            "medical_critique",
            "routing",
            "final_decisions",
        )
    )

    graph_path = run_dir / "execution_graph.json"
    graph = build_execution_graph(
        run_id=run_id,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        run_dir=run_dir,
        steps=execution_steps,
        graph_path=graph_path,
    )
    write_json(graph_path, graph)
    manifest_path = run_dir / "run_manifest.json"
    aliases = {
        "latest_run": task_dirs["task_dir"] / "latest_run.json",
        "latest_eval_report": task_dirs["task_dir"] / "latest_eval_report.json",
        "latest_final_qc_decisions": task_dirs["task_dir"] / "latest_final_qc_decisions.json",
    }
    artifact_paths = {
        "validated_run_context": context_path,
        "dataset_validation": validation_path,
        "summary": summary_artifact,
        "deterministic_qc": det_artifact,
        "calibrated_qc": cal_artifact,
        "comparison": comparison,
        "reasoning": reasoning,
        "medical_critique": critique,
        "routing": routing,
        "final_decisions": final,
        "evaluation": eval_report,
        "execution_graph": graph_path,
    }
    if (run_dir / "decision_policy.yaml").exists():
        artifact_paths["decision_policy"] = run_dir / "decision_policy.yaml"
    write_run_manifest(
        manifest_path,
        run_id=run_id,
        dataset_name=paths.dataset_name,
        task_mode=args.task_mode,
        run_dir=run_dir,
        artifacts=artifact_paths,
        resources={
            "paths_config": paths.paths_yaml,
            "task_profile": task_profile,
            "thresholds_config": paths.thresholds_config,
            **(
                {"confirmed_negative_lesions": args.confirmed_negative_lesions}
                if args.confirmed_negative_lesions is not None
                else {}
            ),
        },
        aliases=aliases,
    )
    write_pointer_alias(
        aliases["latest_run"],
        run_id=run_id,
        run_dir=run_dir,
        target=manifest_path,
        role="latest_run_manifest",
    )
    write_pointer_alias(
        aliases["latest_eval_report"],
        run_id=run_id,
        run_dir=run_dir,
        target=eval_report,
        role="latest_eval_report",
    )
    write_pointer_alias(
        aliases["latest_final_qc_decisions"],
        run_id=run_id,
        run_dir=run_dir,
        target=final,
        role="latest_final_qc_decisions",
    )
    print(f"\nArtifact QC workflow complete: {run_dir}")


if __name__ == "__main__":
    main()
