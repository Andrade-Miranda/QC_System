from __future__ import annotations

import json
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from agents import orchestrator_agent
from agents.orchestrator_agent import build_execution_graph
from agents.validated_run_context_agent import (
    REQUIRED_RUN_ARTIFACTS,
    generate_validated_run_context,
    task_profile_reference,
    validate_reusable_calibrated_thresholds,
)
from artifacts.validation import validate_artifact
from artifacts.hashing import hash_json_payload


JSONSCHEMA_AVAILABLE = importlib.util.find_spec("jsonschema") is not None


class ValidatedRunContextTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "dataset" / "imagesTr"
        self.labels = self.root / "dataset" / "labelsTr"
        self.images.mkdir(parents=True)
        self.labels.mkdir(parents=True)
        (self.images / "case_b_0000.nii.gz").touch()
        (self.images / "case_a_0000.nii.gz").touch()
        self.metadata = self.root / "dataset" / "metadata.xlsx"
        self.paths_yaml = self.root / "paths.yaml"
        self.paths_yaml.write_text("PROJECT_ROOT: .\n", encoding="utf-8")
        self.thresholds = self.root / "thresholds.yaml"
        self.thresholds.write_text(
            "THRESHOLD_METADATA:\n  method: deterministic\nMAX_QC_SCORE_FOR_KEEP: 35\n",
            encoding="utf-8",
        )
        self.calibrated = self.root / "thresholds.calibrated.yaml"
        self.paths = SimpleNamespace(
            paths_yaml=self.paths_yaml,
            project_root=self.root,
            dataset_name="fixture_dataset",
            raw_root=self.root / "dataset",
            raw_images_dir=self.images,
            raw_labels_dir=self.labels,
            raw_metadata=self.metadata,
            image_suffix="_0000.nii.gz",
            summary_dir=self.root / "outputs" / "summary",
            qc_dir=self.root / "outputs" / "qc",
            logs_dir=self.root / "logs",
            thresholds_config=self.thresholds,
            default_threshold_method="deterministic",
        )

    def _profile(self, text: str) -> Path:
        path = self.root / "profile.yaml"
        path.write_text(text, encoding="utf-8")
        return path

    def test_profile_reference_prefers_lowercase_contract(self) -> None:
        profile = self._profile(
            "TASK_MODE: pancreas_only\ntask_profile:\n  id: visible_pancreas\n  version: 2.1.0\n"
        )

        reference = task_profile_reference("pancreas_only", profile)

        self.assertEqual(reference["id"], "visible_pancreas")
        self.assertEqual(reference["version"], "2.1.0")
        self.assertEqual(reference["path"], str(profile.resolve()))

    def test_profile_reference_has_safe_legacy_fallback(self) -> None:
        profile = self._profile("TASK_MODE: pancreas_lesion\n")

        reference = task_profile_reference("pancreas_lesion", profile)

        self.assertEqual(reference["id"], "pancreas_lesion")
        self.assertEqual(reference["version"], "1.0.0")

    def test_generated_context_contains_resolved_production_inputs(self) -> None:
        profile = self._profile("TASK_MODE: pancreas_lesion\n")
        output = self.root / "run" / "validated_run_context.json"

        payload = generate_validated_run_context(
            paths=self.paths,
            task_mode="pancreas_lesion",
            profile_path=profile,
            run_id="run-001",
            output_path=output,
            calibrated_thresholds_path=self.calibrated,
        )

        self.assertTrue(output.is_file())
        self.assertEqual(payload["metadata"]["artifact_type"], "validated_run_context")
        data = payload["data"]
        self.assertEqual(data["run_id"], "run-001")
        self.assertEqual(data["validation_status"], "valid")
        self.assertEqual(data["warnings"], [])
        self.assertEqual(data["case_ids"], ["case_a", "case_b"])
        self.assertEqual(data["required_artifacts"], REQUIRED_RUN_ARTIFACTS)
        self.assertEqual(
            data["path_configuration"]["raw_images_dir"], str(self.images.resolve())
        )
        self.assertEqual(
            data["threshold_configuration"]["deterministic"]["values"]["MAX_QC_SCORE_FOR_KEEP"],
            35,
        )
        self.assertEqual(
            {resource["name"] for resource in data["dataset_resources"]},
            {"raw_dataset_root", "raw_images_dir", "raw_labels_dir", "metadata"},
        )
        json.dumps(payload)

    def test_reused_calibration_must_match_summary_and_task(self) -> None:
        summary = self.root / "summary.json"
        summary.write_text("{}\n", encoding="utf-8")
        self.calibrated.write_text(
            "THRESHOLD_METADATA:\n"
            f"  calibrated_from_summary: {summary}\n"
            "  task_mode: pancreas_only\n",
            encoding="utf-8",
        )
        validate_reusable_calibrated_thresholds(
            self.calibrated,
            summary_path=summary,
            task_mode="pancreas_only",
        )
        with self.assertRaisesRegex(ValueError, "different summary"):
            validate_reusable_calibrated_thresholds(
                self.calibrated,
                summary_path=self.root / "other.json",
                task_mode="pancreas_only",
            )
        with self.assertRaisesRegex(ValueError, "task mode"):
            validate_reusable_calibrated_thresholds(
                self.calibrated,
                summary_path=summary,
                task_mode="pancreas_lesion",
            )

    def test_reused_calibration_accepts_identical_summary_content_at_new_path(self) -> None:
        cases = {"case_a": {"value": 1}}
        original = self.root / "original-summary.json"
        original.write_text(json.dumps(cases), encoding="utf-8")
        wrapped = self.root / "run" / "summary.json"
        wrapped.parent.mkdir()
        wrapped.write_text(
            json.dumps({"metadata": {"artifact_type": "summary"}, "data": {"cases": cases}}),
            encoding="utf-8",
        )
        self.calibrated.write_text(
            "THRESHOLD_METADATA:\n"
            f"  calibrated_from_summary: {original}\n"
            f"  calibrated_from_summary_payload_sha256: {hash_json_payload(cases)}\n"
            "  task_mode: pancreas_only\n",
            encoding="utf-8",
        )

        validate_reusable_calibrated_thresholds(
            self.calibrated,
            summary_path=wrapped,
            task_mode="pancreas_only",
        )

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "optional jsonschema package is unavailable")
    def test_generated_context_satisfies_schema(self) -> None:
        profile = self._profile(
            "TASK_MODE: pancreas_only\ntask_profile:\n  id: visible_pancreas\n  version: 1.0.0\n"
        )
        payload = generate_validated_run_context(
            paths=self.paths,
            task_mode="pancreas_only",
            profile_path=profile,
            run_id="run-schema",
            output_path=self.root / "validated_run_context.json",
            calibrated_thresholds_path=self.calibrated,
        )

        validate_artifact(payload, "validated_run_context")

    def test_execution_graph_records_output_and_dependency_order(self) -> None:
        steps = [
            {"id": "deterministic_qc", "dependencies": []},
            {"id": "calibrated_qc", "dependencies": []},
            {"id": "comparison", "dependencies": ["deterministic_qc", "calibrated_qc"]},
            {"id": "reasoning", "dependencies": ["comparison"]},
            {"id": "medical_critique", "dependencies": ["reasoning"]},
            {"id": "routing", "dependencies": ["comparison"]},
            {"id": "final_decisions", "dependencies": ["routing"]},
            {"id": "evaluation", "dependencies": ["reasoning", "medical_critique", "final_decisions"]},
        ]
        graph_path = self.root / "execution_graph.json"

        graph = build_execution_graph(
            run_id="run-001",
            dataset_name="fixture_dataset",
            task_mode="pancreas_lesion",
            run_dir=self.root,
            steps=steps,
            graph_path=graph_path,
        )

        self.assertEqual(graph["execution_graph_output"], str(graph_path))
        order = {step["id"]: index for index, step in enumerate(graph["steps"])}
        for step in graph["steps"]:
            for dependency in step["dependencies"]:
                self.assertLess(order[dependency], order[step["id"]])

    def test_orchestrator_wires_explanations_and_audit_without_policy_authority(self) -> None:
        profile_dir = self.root / "configs" / "task_profiles"
        profile_dir.mkdir(parents=True)
        (profile_dir / "pancreas_lesion.yaml").write_text(
            "TASK_MODE: pancreas_lesion\n", encoding="utf-8"
        )
        expected_summary = (
            self.paths.summary_dir.parent
            / "pancreas_lesion"
            / "summary"
            / "fixture_dataset_summary.json"
        )
        self.calibrated.write_text(
            "THRESHOLD_METADATA:\n"
            "  method: calibrated\n"
            f"  calibrated_from_summary: {expected_summary}\n"
            "  task_mode: pancreas_lesion\n",
            encoding="utf-8",
        )
        run_dir = self.root / "run"
        golden = self.root / "golden.json"
        golden.write_text("{}\n", encoding="utf-8")
        commands: list[list[str]] = []

        def fake_run(command: list[str], *, cwd: Path) -> None:
            commands.append(command)
            script = Path(command[1]).name
            if script == "summarize_dataset.py":
                summary = (
                    self.paths.summary_dir.parent
                    / "pancreas_lesion"
                    / "summary"
                    / "fixture_dataset_summary.json"
                )
                summary.parent.mkdir(parents=True, exist_ok=True)
                summary.write_text("{}\n", encoding="utf-8")
            elif script == "qc_agent.py":
                report_dir = Path(command[command.index("--report-dir") + 1])
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "qc_report.json").write_text("{}\n", encoding="utf-8")
            elif "--output" in command:
                output = Path(command[command.index("--output") + 1])
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("{}\n", encoding="utf-8")

        argv = [
            "orchestrator_agent.py",
            "--task-mode",
            "pancreas_lesion",
            "--run-dir",
            str(run_dir),
            "--skip-calibration",
            "--calibrated-thresholds",
            str(self.calibrated),
            "--golden-labels",
            str(golden),
        ]
        with (
            patch.object(orchestrator_agent, "resolve_project_paths", return_value=self.paths),
            patch.object(orchestrator_agent, "_run", side_effect=fake_run),
            patch("sys.argv", argv),
        ):
            orchestrator_agent.main()

        by_script = {Path(command[1]).name: command for command in commands}
        sequence = [Path(command[1]).name for command in commands]
        self.assertLess(sequence.index("qc_comparison_agent.py"), sequence.index("reasoning_agent.py"))
        self.assertLess(sequence.index("reasoning_agent.py"), sequence.index("medical_critic_agent.py"))
        self.assertLess(sequence.index("medical_critic_agent.py"), sequence.index("review_routing_agent.py"))

        routing = by_script["review_routing_agent.py"]
        final = by_script["final_decision_agent.py"]
        for command in (routing, final):
            self.assertIn("--dataset-validation", command)
            self.assertNotIn("--reasoning", command)
            self.assertNotIn("--critique", command)
        self.assertIn("--comparison", final)

        evaluation = by_script["evaluation_agent.py"]
        for flag in (
            "--validated-context",
            "--dataset-validation",
            "--deterministic-evidence",
            "--calibrated-evidence",
            "--comparison",
            "--reasoning",
            "--critique",
            "--routing",
            "--final-decisions",
            "--golden-labels",
        ):
            self.assertIn(flag, evaluation)

        graph = json.loads((run_dir / "execution_graph.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [step["id"] for step in graph["steps"]][-6:],
            ["comparison", "reasoning", "medical_critique", "routing", "final_decisions", "evaluation"],
        )


if __name__ == "__main__":
    unittest.main()
