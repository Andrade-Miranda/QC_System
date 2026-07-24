from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_qc_run import build_evaluation_command


class EvaluateRunCommandTests(unittest.TestCase):
    def test_complete_artifact_set_and_golden_labels_are_forwarded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            filenames = (
                "validated_run_context.json",
                "dataset_validation.json",
                "qc_report_deterministic.json",
                "qc_report_calibrated.json",
                "qc_comparison.json",
                "reasoning_artifact.json",
                "medical_critique.json",
                "review_routing.json",
                "final_qc_decisions.json",
            )
            for filename in filenames:
                (run_dir / filename).touch()
            golden = run_dir / "golden_labels.json"
            golden.touch()

            command = build_evaluation_command(
                run_dir,
                task_mode="pancreas_only",
                run_id="run-1",
                paths_yaml="paths.yaml",
                golden_labels=golden,
                python="python",
            )

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
                self.assertIn(flag, command)
            self.assertEqual(command[command.index("--golden-labels") + 1], str(golden))


if __name__ == "__main__":
    unittest.main()
