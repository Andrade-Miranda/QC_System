from __future__ import annotations

import importlib.util
import tempfile
import sys
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.medical_critic_agent import (
    DeterministicMedicalCriticBackend,
    generate_medical_critique_artifact,
    main as critic_main,
)
from agents.reasoning_agent import (
    DeterministicReasoningBackend,
    generate_reasoning_artifact,
    main as reasoning_main,
)
from artifacts.io import load_artifact, write_artifact
from artifacts.validation import validate_artifact


JSONSCHEMA_AVAILABLE = importlib.util.find_spec("jsonschema") is not None


def _qc_case(recommendation: str, risk: str, score: int, domain: str | None) -> dict:
    return {
        "triage": {"recommendation": recommendation, "risk_level": risk, "score": score},
        "primary_issue": {"domain": domain} if domain else None,
        "qc_domains": {
            "geometry_integrity": {
                "severity": "normal" if domain != "geometry_integrity" else "high_warning",
                "component_score": score,
            }
        },
    }


def _contains_key(value, forbidden: set[str]) -> bool:
    if isinstance(value, dict):
        return any(key in forbidden or _contains_key(child, forbidden) for key, child in value.items())
    if isinstance(value, list):
        return any(_contains_key(child, forbidden) for child in value)
    return False


class ExplanatoryAgentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.context = self.root / "validated_run_context.json"
        self.deterministic = self.root / "qc_report_deterministic.json"
        self.calibrated = self.root / "qc_report_calibrated.json"
        self.comparison = self.root / "qc_comparison.json"
        common = {
            "dataset_name": "fixture_dataset",
            "task_mode": "pancreas_lesion",
            "run_id": "fixture_run",
        }
        write_artifact(
            self.context,
            artifact_type="validated_run_context",
            generator="Test",
            data={
                "status": "passed",
                "task_mode": "pancreas_lesion",
                "case_ids": ["case_a", "case_b"],
            },
            **common,
        )
        deterministic_cases = {
            "case_a": _qc_case("keep", "low", 0, None),
            "case_b": _qc_case("keep", "low", 10, "geometry_integrity"),
        }
        calibrated_cases = {
            "case_a": _qc_case("keep", "low", 0, None),
        }
        write_artifact(
            self.deterministic,
            artifact_type="qc_report_deterministic",
            generator="Test",
            data={"cases": deterministic_cases},
            **common,
        )
        write_artifact(
            self.calibrated,
            artifact_type="qc_report_calibrated",
            generator="Test",
            data={"cases": calibrated_cases},
            **common,
        )
        write_artifact(
            self.comparison,
            artifact_type="qc_comparison",
            generator="Test",
            data={
                "all_cases": [
                    {
                        "case_id": "case_a",
                        "changed": False,
                        "recommendation_change": "unchanged",
                        "risk_change": "unchanged",
                        "score_delta": 0,
                        "domain_deterministic": None,
                        "domain_calibrated": None,
                    },
                    {
                        "case_id": "case_b",
                        "changed": True,
                        "recommendation_change": "unknown",
                        "risk_change": "unknown",
                        "score_delta": None,
                        "domain_deterministic": "geometry_integrity",
                        "domain_calibrated": None,
                    },
                ]
            },
            **common,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_reasoning_is_wrapped_explanatory_and_explicit_about_missing_case(self) -> None:
        output = self.root / "reasoning_artifact.json"
        payload = generate_reasoning_artifact(
            context_path=self.context,
            deterministic_path=self.deterministic,
            calibrated_path=self.calibrated,
            comparison_path=self.comparison,
            output_path=output,
            backend=DeterministicReasoningBackend(),
        )

        self.assertEqual(payload["metadata"]["artifact_type"], "reasoning_artifact")
        self.assertEqual(payload["data"]["cases"]["case_a"]["status"], "complete")
        missing = payload["data"]["cases"]["case_b"]
        self.assertEqual(missing["status"], "partial")
        self.assertIn(
            "missing_calibrated_case",
            {item["code"] for item in missing["uncertainty_annotations"]},
        )
        self.assertIsNotNone(
            missing["evidence_references"]["deterministic"]["case_pointer"]
        )
        self.assertFalse(
            _contains_key(payload, {"final_decision", "route_to_review", "policy_action"})
        )

    def test_critic_checks_grounding_and_is_non_binding(self) -> None:
        reasoning = self.root / "reasoning_artifact.json"
        generate_reasoning_artifact(
            context_path=self.context,
            deterministic_path=self.deterministic,
            calibrated_path=self.calibrated,
            comparison_path=self.comparison,
            output_path=reasoning,
        )
        metadata, data = load_artifact(reasoning)
        data["cases"]["case_a"]["evidence_summary"]["deterministic"]["score"] = 999
        write_artifact(
            reasoning,
            artifact_type="reasoning_artifact",
            generator="TestTamper",
            data=data,
            dataset_name=metadata["dataset_name"],
            task_mode=metadata["task_mode"],
            run_id=metadata["run_id"],
        )
        output = self.root / "medical_critique.json"
        payload = generate_medical_critique_artifact(
            reasoning_path=reasoning,
            deterministic_path=self.deterministic,
            calibrated_path=self.calibrated,
            comparison_path=self.comparison,
            output_path=output,
            backend=DeterministicMedicalCriticBackend(),
        )

        case = payload["data"]["cases"]["case_a"]
        self.assertFalse(payload["data"]["binding"])
        self.assertFalse(case["binding"])
        self.assertEqual(case["severity"], "high")
        self.assertIn(
            "deterministic.score",
            {check["claim"] for check in case["unsupported_claim_checks"]},
        )
        forbidden = {
            "final_decision",
            "route_to_review",
            "policy_action",
            "recommended_routing",
            "blocks_automatic_acceptance",
        }
        self.assertFalse(_contains_key(payload, forbidden))

    def test_valid_with_warnings_context_is_accepted(self) -> None:
        write_artifact(
            self.context,
            artifact_type="validated_run_context",
            generator="Test",
            data={
                "validation_status": "valid_with_warnings",
                "task_mode": "pancreas_lesion",
                "case_ids": ["case_a", "case_b"],
                "warnings": ["Non-blocking fixture warning."],
            },
            dataset_name="fixture_dataset",
            task_mode="pancreas_lesion",
            run_id="fixture_run",
        )

        payload = generate_reasoning_artifact(
            context_path=self.context,
            deterministic_path=self.deterministic,
            calibrated_path=self.calibrated,
            comparison_path=self.comparison,
            output_path=self.root / "reasoning_with_warnings.json",
        )

        self.assertEqual(
            payload["data"]["alignment"]["context_validation_status"],
            "valid_with_warnings",
        )
        self.assertNotIn(
            "context_validation_status:valid_with_warnings",
            payload["data"]["alignment"]["issues"],
        )

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "optional jsonschema package is unavailable")
    def test_generated_explanatory_artifacts_validate_against_schemas(self) -> None:
        reasoning_path = self.root / "reasoning_for_schema.json"
        critique_path = self.root / "critique_for_schema.json"
        reasoning = generate_reasoning_artifact(
            context_path=self.context,
            deterministic_path=self.deterministic,
            calibrated_path=self.calibrated,
            comparison_path=self.comparison,
            output_path=reasoning_path,
        )
        critique = generate_medical_critique_artifact(
            reasoning_path=reasoning_path,
            deterministic_path=self.deterministic,
            calibrated_path=self.calibrated,
            comparison_path=self.comparison,
            output_path=critique_path,
        )

        validate_artifact(reasoning, "reasoning_artifact")
        validate_artifact(critique, "medical_critique")

    def test_cli_interfaces_write_artifacts(self) -> None:
        reasoning = self.root / "cli_reasoning.json"
        critique = self.root / "cli_critique.json"
        reasoning_main([
            "--context", str(self.context),
            "--deterministic", str(self.deterministic),
            "--calibrated", str(self.calibrated),
            "--comparison", str(self.comparison),
            "--output", str(reasoning),
        ])
        critic_main([
            "--reasoning", str(reasoning),
            "--deterministic", str(self.deterministic),
            "--calibrated", str(self.calibrated),
            "--comparison", str(self.comparison),
            "--output", str(critique),
        ])
        reasoning_metadata, _ = load_artifact(reasoning)
        critique_metadata, _ = load_artifact(critique)
        self.assertEqual(reasoning_metadata["artifact_type"], "reasoning_artifact")
        self.assertEqual(critique_metadata["artifact_type"], "medical_critique")

    def test_backend_authority_fields_are_rejected(self) -> None:
        class InvalidReasoningBackend(DeterministicReasoningBackend):
            def generate_case(self, case_input):
                output = super().generate_case(case_input)
                output["policy_actions"] = []
                return output

        with self.assertRaisesRegex(ValueError, "Forbidden explanatory output field"):
            generate_reasoning_artifact(
                context_path=self.context,
                deterministic_path=self.deterministic,
                calibrated_path=self.calibrated,
                comparison_path=self.comparison,
                output_path=self.root / "invalid_reasoning.json",
                backend=InvalidReasoningBackend(),
            )


if __name__ == "__main__":
    unittest.main()
