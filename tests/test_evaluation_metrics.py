import json
import tempfile
import unittest
from pathlib import Path

from agents.evaluation_agent import evaluate


class EvaluationMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def artifact(self, name, data, **metadata):
        path = self.root / name
        payload = {
            "metadata": {
                "artifact_type": metadata.pop("artifact_type", name.removesuffix(".json")),
                "run_id": "run-1",
                "dataset_name": "dataset-1",
                "task_mode": "pancreas_lesion",
                **metadata,
            },
            "data": data,
        }
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    def final_artifact(self, decisions):
        return self.artifact("final.json", {"decisions": decisions})

    def test_zero_denominators_are_not_applicable(self):
        final = self.final_artifact({})
        comparison = self.artifact("comparison.json", {"summary": {"n_cases_compared": 0}, "all_cases": []})

        report = evaluate(final, comparison, None)

        stability = report["evidence_stability"]
        for name in ("calibration_sensitivity_rate", "agreement_rate", "mean_absolute_score_shift"):
            self.assertEqual(stability[name]["status"], "not_applicable")
            self.assertIsNone(stability[name]["value"])
            self.assertEqual(stability[name]["denominator"], 0)
            self.assertEqual(
                set(stability[name]) & {
                    "value", "numerator", "denominator", "status", "eligible_n", "missing_n", "excluded_n"
                },
                {"value", "numerator", "denominator", "status", "eligible_n", "missing_n", "excluded_n"},
            )

    def test_case_id_mismatch_is_reported(self):
        validation = self.artifact("validation.json", {"cases": {"case-a": {}, "case-b": {}}})
        final = self.final_artifact({
            "case-a": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "keep",
                "calibrated_recommendation": "keep",
            },
            "case-c": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "keep",
                "calibrated_recommendation": "keep",
            },
        })

        report = evaluate(final, dataset_validation_path=validation)

        check = report["artifact_integrity"]["case_id_set_match"]
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["mismatches"]["final_decisions"]["missing_case_ids"], ["case-b"])
        self.assertEqual(check["mismatches"]["final_decisions"]["unexpected_case_ids"], ["case-c"])

    def test_review_action_must_be_in_review_queue(self):
        final = self.final_artifact({
            "case-a": {
                "final_decision": "review",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "review",
                "calibrated_recommendation": "review",
            }
        })
        routing = self.artifact("routing.json", {"routed_cases": [], "not_routed_cases": [{"case_id": "case-a"}]})

        report = evaluate(final, routing_path=routing)

        check = report["routing_behavior"]["review_action_queue_consistency"]
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["numerator"], 0)
        self.assertEqual(check["denominator"], 1)
        self.assertEqual(check["review_actions_missing_from_queue"], ["case-a"])

    def test_hard_failure_cannot_be_kept(self):
        final = self.final_artifact({
            "case-a": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": True,
                "requires_human_review": False,
                "deterministic_recommendation": "exclude",
                "calibrated_recommendation": "keep",
            }
        })

        report = evaluate(final)

        check = report["policy_safety"]["hard_failure_preservation"]
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["value"], 0)
        self.assertEqual(check["hard_failures_inappropriately_kept"], ["case-a"])

    def test_missing_optional_artifacts_remain_unavailable(self):
        final = self.final_artifact({
            "case-a": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "keep",
                "calibrated_recommendation": "keep",
            }
        })

        report = evaluate(final)

        self.assertEqual(report["reasoning_validity"]["status"], "unavailable")
        self.assertEqual(report["critique_validity"]["status"], "unavailable")
        self.assertEqual(report["routing_behavior"]["status"], "unavailable")
        self.assertEqual(
            report["routing_behavior"]["review_action_queue_consistency"]["status"],
            "not_applicable",
        )
        self.assertEqual(report["task_consistency"]["status"], "unavailable")
        self.assertNotIn("golden_case_validation", report)
        self.assertEqual(report["dataset_statistics"]["n_cases"], 1)

    def test_structured_explanation_references_are_validated(self):
        final = self.final_artifact({
            "case-a": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "keep",
                "calibrated_recommendation": "keep",
            }
        })
        case_reference = {
            "artifact_type": "qc_report_deterministic",
            "path": "qc_report_deterministic.json",
            "case_pointer": "/data/cases/case-a",
        }
        comparison_reference = {
            "artifact_type": "qc_comparison",
            "path": "qc_comparison.json",
            "case_pointer": "/data/all_cases/0",
        }
        reasoning = self.artifact("reasoning.json", {
            "cases": {
                "case-a": {
                    "case_id": "case-a",
                    "evidence_references": {
                        "deterministic": case_reference,
                        "comparison": comparison_reference,
                    },
                }
            }
        })
        critique = self.artifact("critique.json", {
            "cases": {
                "case-a": {
                    "case_id": "case-a",
                    "supported_claim_checks": [{"evidence_reference": case_reference}],
                    "concerns": [{"evidence_references": [comparison_reference]}],
                }
            }
        })

        report = evaluate(final, reasoning_path=reasoning, critique_path=critique)

        reasoning_check = report["reasoning_validity"]["evidence_reference_validity"]
        critique_check = report["critique_validity"]["evidence_reference_validity"]
        self.assertEqual(reasoning_check["status"], "passed")
        self.assertEqual(reasoning_check["denominator"], 2)
        self.assertEqual(critique_check["status"], "passed")
        self.assertEqual(critique_check["denominator"], 2)

    def test_malformed_structured_reference_is_reported(self):
        final = self.final_artifact({
            "case-a": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "keep",
                "calibrated_recommendation": "keep",
            }
        })
        reasoning = self.artifact("reasoning.json", {
            "cases": {
                "case-a": {
                    "case_id": "case-a",
                    "evidence_references": {
                        "deterministic": {
                            "artifact_type": "qc_report_deterministic",
                            "case_pointer": "/data/cases/different-case",
                        }
                    },
                }
            }
        })

        report = evaluate(final, reasoning_path=reasoning)

        check = report["reasoning_validity"]["evidence_reference_validity"]
        self.assertEqual(check["status"], "failed")
        self.assertEqual(check["numerator"], 0)
        self.assertEqual(check["denominator"], 1)

    def test_golden_validation_reports_action_and_safety_metrics(self):
        decisions = {
            "case-a": "keep",
            "case-b": "keep",
            "case-c": "review",
            "case-d": "keep",
            "case-e": "warning",
            "case-f": "reject",
            "case-unlabeled": "warning",
        }
        final = self.final_artifact({
            case_id: {
                "final_decision": action,
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": action == "review",
                "deterministic_recommendation": action,
                "calibrated_recommendation": action,
            }
            for case_id, action in decisions.items()
        })
        expected = {
            "case-a": "keep",
            "case-b": "warning",
            "case-c": "review",
            "case-d": "reject",
            "case-e": "reject",
            "case-f": "insufficient_evidence",
        }
        golden = self.artifact("golden.json", {
            "labels": {
                case_id: {"case_id": case_id, "expected_action": action}
                for case_id, action in expected.items()
            }
        }, artifact_type="golden_labels")

        report = evaluate(final, golden_labels_path=golden)
        validation = report["golden_case_validation"]

        self.assertEqual(validation["label_coverage"]["numerator"], 6)
        self.assertEqual(validation["label_coverage"]["denominator"], 7)
        self.assertEqual(validation["overall_agreement"]["numerator"], 2)
        self.assertEqual(validation["overall_agreement"]["denominator"], 6)
        self.assertEqual(validation["per_action_agreement"]["review"]["value"], 1)
        self.assertEqual(validation["confusion_counts"]["reject"]["keep"], 1)
        self.assertEqual(validation["confusion_counts"]["reject"]["warning"], 1)
        self.assertEqual(validation["unsafe_keep_count"], 2)
        self.assertEqual(validation["unsafe_keep_rate"]["value"], 1)
        self.assertEqual(validation["review_coverage"]["value"], 1)

    def test_golden_action_metrics_are_safe_with_zero_denominators(self):
        final = self.final_artifact({
            "case-a": {
                "final_decision": "keep",
                "decision_basis": "calibrated_qc",
                "hard_failure": False,
                "requires_human_review": False,
                "deterministic_recommendation": "keep",
                "calibrated_recommendation": "keep",
            }
        })
        golden = self.artifact("golden.json", {
            "labels": {"case-a": {"case_id": "case-a", "expected_action": "keep"}}
        }, artifact_type="golden_labels")

        validation = evaluate(final, golden_labels_path=golden)["golden_case_validation"]

        self.assertEqual(validation["unsafe_keep_rate"]["status"], "not_applicable")
        self.assertIsNone(validation["unsafe_keep_rate"]["value"])
        self.assertEqual(validation["review_coverage"]["status"], "not_applicable")
        self.assertEqual(
            validation["per_action_agreement"]["reject"]["status"],
            "not_applicable",
        )


if __name__ == "__main__":
    unittest.main()
