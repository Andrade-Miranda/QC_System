from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from agents.llm_explanation_agent import build_explanation_artifact
from agents.llm_explanation_agent import main as explanation_main
from agents.utils.explanation_contracts import NOT_APPLICABLE
from agents.utils.llm_backend import ChatBackend, ProviderConfig, ProviderError
from agents.utils.llm_explanation_adapter import (
    build_compact_evidence,
    fallback_explanation,
    load_explanation_index,
    parse_and_validate_llm_output,
)
from artifacts.hashing import hash_file
from artifacts.io import write_artifact
from artifacts.validation import ArtifactValidationError, validate_artifact


ROOT = Path(__file__).resolve().parents[1]


class FakeBackend(ChatBackend):
    def __init__(self, response: str, *, fail: bool = False):
        super().__init__(ProviderConfig(
            provider="ollama",
            model="devstral:test",
            timeout_seconds=1,
            max_tokens=256,
            base_url="http://unused",
            temperature=0,
            top_p=1,
            seed=0,
        ))
        self.response = response
        self.fail = fail
        self.messages = None

    def complete(self, messages):
        self.messages = messages
        if self.fail:
            raise ProviderError("simulated failure")
        return self.response


def valid_response(case_id: str, action: str = "review") -> str:
    return json.dumps({
        "case_id": case_id,
        "evidence_summary": f"Compact evidence reports final policy action {action}.",
        "reason_for_review_or_action": "The supplied route reasons and decision basis explain the action.",
        "unresolved_items": [],
        "possible_consistency_questions": [],
        "limitations": ["This is nonbinding and uses compact evidence only."],
        "explicit_statement_that_deterministic_policy_is_authoritative": "The deterministic policy is authoritative.",
    })


class LLMExplanationAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

    def _artifact(self, run_dir: Path, filename: str, artifact_type: str, task_mode: str, data: dict):
        return write_artifact(
            run_dir / filename,
            artifact_type=artifact_type,
            generator="TestFixture",
            data=data,
            dataset_name="PantsMini",
            task_mode=task_mode,
            run_id=f"run-{task_mode}",
            project_root=ROOT,
        )

    def _make_run(self, task_mode: str) -> Path:
        run_dir = self.root / task_mode
        run_dir.mkdir()
        cases = ["case_keep", "case_reject", "case_review"]
        task_profile = {
            "pancreas_only": ("pancreas_segmentation_visible", "1.0.0"),
            "pancreas_lesion": ("pancreatic_lesion_segmentation", "1.0.0"),
            "pancreas_lesion_subregions": ("pancreatic_lesion_subregion_segmentation", "1.0.0"),
        }[task_mode]
        self._artifact(run_dir, "validated_run_context.json", "validated_run_context", task_mode, {
            "run_id": f"run-{task_mode}",
            "dataset_name": "PantsMini",
            "task_mode": task_mode,
            "case_ids": cases,
            "task_profile": {"id": task_profile[0], "version": task_profile[1]},
            "dataset_resources": [],
            "path_configuration": {},
            "threshold_configuration": {},
            "validation_status": "valid",
            "warnings": [],
        })
        validation_cases = {
            case: {"status": "valid", "omit_from_training": False, "errors": [], "resources": {}}
            for case in cases
        }
        self._artifact(run_dir, "dataset_validation.json", "dataset_validation", task_mode, {
            "status": "valid", "dataset_name": "PantsMini", "task_mode": task_mode,
            "n_images": 3, "n_cases": 3, "n_invalid_cases": 0, "errors": [],
            "labels_without_images": [], "cases": validation_cases, "required_segmentations": {},
        })
        det_cases = {}
        cal_cases = {}
        final_cases = {}
        routing_cases = {}
        for case, action in zip(cases, ["keep", "reject", "review"]):
            recommendation = "keep_with_metadata_warning" if case == "case_review" else action
            measurements = {
                "pancreas_present": True,
                "pancreas_volume_mm3": 100.0,
            }
            if task_mode != "pancreas_only":
                measurements.update({
                    "lesion_present": True,
                    "lesion_volume_mm3": 12.5,
                    "lesion_pancreas_overlap": 0.9,
                })
            if task_mode == "pancreas_lesion_subregions":
                measurements.update({
                    "subregion_masks_present": True,
                    "subregion_overlap": 0.1,
                    "region_volume_consistency": "consistent",
                })
            det_cases[case] = {
                "target_presence": {"visible_target_annotation_status": "uncertain" if case == "case_review" else "complete"},
                "measurements": measurements,
                "triage": {"recommendation": recommendation, "risk_level": "low", "score": 10 if case == "case_review" else 0},
                "qc_domains": {
                    "geometry_integrity": {"severity": "normal", "component_score": 0},
                    "fov_integrity": {"severity": "normal", "component_score": 0},
                    "metadata_completeness": {"severity": "low_warning" if case == "case_review" else "normal", "component_score": 8 if case == "case_review" else 0},
                    "lesion_localization": {"severity": "normal", "component_score": 0},
                    "lesion_burden": {"severity": "normal", "component_score": 0},
                    "attenuation_integrity": {"severity": "normal", "component_score": 0},
                    "region_consistency": {"severity": "normal", "component_score": 0},
                },
                "primary_failure_mode": "missing_metadata" if case == "case_review" else None,
            }
            cal_cases[case] = dict(det_cases[case])
            final_cases[case] = {
                "final_decision": action,
                "decision_basis": "review_routing" if case == "case_review" else "deterministic_qc",
                "hard_failure": action == "reject",
                "requires_human_review": action == "review",
                "deterministic_recommendation": recommendation,
                "calibrated_recommendation": recommendation,
                "risk_level": "low",
                "primary_domain": "metadata_completeness" if case == "case_review" else "geometry_integrity",
                "policy_version": "v1.0",
                "policy_rules_triggered": ["partial_visible_target_annotation_uncertain"] if case == "case_review" else [action],
                "route_to_review": action == "review",
                "decision_trace": {"matched_rule": {"id": "fixture", "name": action, "rationale": "fixture"}},
            }
            routing_cases[case] = {
                "case_id": case,
                "route_to_review": action == "review",
                "route_reasons": ["partial_visible_target_annotation_uncertain"] if case == "case_review" else [],
                "hard_failure": action == "reject",
                "deterministic_recommendation": recommendation,
                "calibrated_recommendation": recommendation,
                "primary_domain": final_cases[case]["primary_domain"],
                "evidence_status": "complete",
            }
        self._artifact(run_dir, "qc_report_deterministic.json", "qc_report_deterministic", task_mode, {"cases": det_cases})
        self._artifact(run_dir, "qc_report_calibrated.json", "qc_report_calibrated", task_mode, {"cases": cal_cases})
        self._artifact(run_dir, "qc_comparison.json", "qc_comparison", task_mode, {
            "all_cases": [{"case_id": case, "changed": False, "score_delta": 0} for case in cases]
        })
        reasoning_cases = {
            case: {
                "case_id": case,
                "status": "complete",
                "task_profile_reference": {"artifact_type": "validated_run_context", "path": "validated_run_context.json", "exists": True, "sha256": None},
                "evidence_references": {
                    name: {"artifact_type": name, "path": f"{name}.json", "exists": True, "sha256": None, "case_pointer": f"/data/cases/{case}"}
                    for name in ["deterministic", "calibrated", "comparison"]
                },
                "evidence_summary": {
                    "deterministic": {"availability": "available", "reported_recommendation": "keep", "risk_level": "low", "score": 0, "primary_domain": None, "abnormal_domains": []},
                    "calibrated": {"availability": "available", "reported_recommendation": "keep", "risk_level": "low", "score": 0, "primary_domain": None, "abnormal_domains": []},
                    "comparison": {"availability": "available", "changed": False, "recommendation_change": "unchanged", "risk_change": "unchanged", "score_delta": 0, "domain_deterministic": None, "domain_calibrated": None},
                },
                "conflicts": [], "uncertainty_annotations": [], "limitations": ["Structured evidence only."],
            } for case in cases
        }
        self._artifact(run_dir, "reasoning_artifact.json", "reasoning_artifact", task_mode, {
            "status": "complete", "backend": {"name": "template", "frozen": True},
            "alignment": {"context_validation_status": "valid", "issues": [], "case_counts": {"context": 3, "deterministic": 3, "calibrated": 3, "comparison": 3, "output": 3}},
            "cases": reasoning_cases, "limitations": [],
        })
        critique_cases = {
            case: {
                "case_id": case, "status": "complete", "audit_scope": "grounding_and_internal_consistency_only",
                "supported_claim_checks": [], "unsupported_claim_checks": [], "concerns": [],
                "evidence_gaps": [], "severity": "none", "limitations": ["Structured audit only."], "binding": False,
            } for case in cases
        }
        self._artifact(run_dir, "medical_critique.json", "medical_critique", task_mode, {
            "status": "complete", "backend": {"name": "template", "frozen": True},
            "audit_scope": "grounding_and_internal_consistency_only",
            "alignment": {"issues": [], "case_counts": {"reasoning": 3, "deterministic": 3, "calibrated": 3, "comparison": 3, "output": 3}},
            "cases": critique_cases, "limitations": [], "binding": False,
        })
        self._artifact(run_dir, "review_routing.json", "review_routing", task_mode, {
            "summary": {"n_routed": 1, "n_not_routed": 2}, "cases": routing_cases,
            "routed_cases": [routing_cases["case_review"]], "not_routed_cases": [routing_cases["case_keep"], routing_cases["case_reject"]],
        })
        self._artifact(run_dir, "final_qc_decisions.json", "final_qc_decisions", task_mode, {
            "summary": {"n_cases": 3, "decision_counts": {"keep": 1, "reject": 1, "review": 1}},
            "decisions": final_cases,
        })
        self._artifact(run_dir, "eval_report.json", "eval_report", task_mode, {
            "dataset_statistics": {"n_cases": 3, "decision_counts": {"keep": 1, "reject": 1, "review": 1}, "decision_basis_counts": {"review_routing": 1, "deterministic_qc": 2}, "human_review_required": 1, "hard_failures": 1},
            "calibration_impact": {}, "review_routing": {}, "artifact_consistency": {},
        })
        return run_dir

    def test_compact_evidence_supports_all_task_profiles(self):
        expectations = {
            "pancreas_only": ("pancreas-segmentation task", "tau_P", NOT_APPLICABLE, NOT_APPLICABLE),
            "pancreas_lesion": ("pancreatic-lesion segmentation task", "tau_L", True, NOT_APPLICABLE),
            "pancreas_lesion_subregions": ("pancreatic-lesion subregion task", "tau_S", True, True),
        }
        for task_mode, (name, notation, lesion_expected, subregion_expected) in expectations.items():
            with self.subTest(task_mode=task_mode):
                index = load_explanation_index(self._make_run(task_mode))
                compact = build_compact_evidence(index, "case_review")
                self.assertEqual(compact["scientific_task_name"], name)
                self.assertEqual(compact["task_notation"], notation)
                self.assertEqual(compact["implementation_task_id"], task_mode)
                self.assertEqual(compact["observed_final_policy_action"], "review")
                self.assertEqual(compact["observed_deterministic_recommendation"], "keep_with_metadata_warning")
                self.assertEqual(compact["route_reasons"], ["partial_visible_target_annotation_uncertain"])
                self.assertEqual(compact["visualization_status"], "unknown")
                self.assertIsNone(compact["visualization_error"])
                self.assertEqual(compact["relevant_measurements"]["lesion_presence"], lesion_expected)
                self.assertEqual(compact["relevant_measurements"]["head_body_tail_annotation_availability"], subregion_expected)

    def test_valid_generation_writes_schema_valid_artifact_and_preserves_final_hash(self):
        run_dir = self._make_run("pancreas_lesion_subregions")
        before = hash_file(run_dir / "final_qc_decisions.json")
        artifact = build_explanation_artifact(run_dir, FakeBackend(valid_response("case_review")), case_ids=["case_review"])
        after = hash_file(run_dir / "final_qc_decisions.json")
        self.assertEqual(before, after)
        validate_artifact(artifact, "llm_explanation_artifact")
        case = artifact["data"]["cases"]["case_review"]
        self.assertTrue(case["validation_result"]["usable_for_reporting"])
        prompt_text = json.dumps(case["rendered_prompt"])
        self.assertNotIn("qc_report_deterministic", prompt_text)
        self.assertNotIn("/home/", prompt_text)

    def test_invalid_outputs_trigger_fallback(self):
        run_dir = self._make_run("pancreas_only")
        artifact = build_explanation_artifact(run_dir, FakeBackend("not json"), case_ids=["case_review"])
        case = artifact["data"]["cases"]["case_review"]
        self.assertFalse(case["validation_result"]["usable_for_reporting"])
        self.assertIsNotNone(case["fallback_response"])
        self.assertEqual(case["reported_response"], case["fallback_response"])

    def test_case_identifier_and_action_cannot_change(self):
        index = load_explanation_index(self._make_run("pancreas_only"))
        compact = build_compact_evidence(index, "case_review")
        with self.assertRaisesRegex(ValueError, "case_id"):
            parse_and_validate_llm_output(valid_response("case_keep"), compact)
        altered = json.loads(valid_response("case_review"))
        altered["evidence_summary"] = "Compact evidence reports final policy action reject."
        with self.assertRaisesRegex(ValueError, "final action"):
            parse_and_validate_llm_output(json.dumps(altered), compact)

    def test_paths_ansi_unsupported_claims_and_visualization_causality_rejected(self):
        index = load_explanation_index(self._make_run("pancreas_lesion"))
        compact = build_compact_evidence(index, "case_review")
        for text in [
            "\u001b[31mred\u001b[0m",
            "/home/user/PanTS_00007001_0000.nii.gz",
            "This proves medical correctness.",
            "The visualization failed because of a corrupted dataset.",
        ]:
            payload = json.loads(valid_response("case_review"))
            payload["limitations"] = [text]
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    parse_and_validate_llm_output(json.dumps(payload), compact)

    def test_provider_failure_does_not_modify_final_action(self):
        run_dir = self._make_run("pancreas_lesion")
        before = (run_dir / "final_qc_decisions.json").read_text(encoding="utf-8")
        artifact = build_explanation_artifact(run_dir, FakeBackend("", fail=True), case_ids=["case_review"])
        after = (run_dir / "final_qc_decisions.json").read_text(encoding="utf-8")
        self.assertEqual(before, after)
        self.assertFalse(artifact["data"]["cases"]["case_review"]["validation_result"]["usable_for_reporting"])

    def test_cli_returns_nonzero_on_schema_validation_failure(self):
        run_dir = self._make_run("pancreas_only")
        with patch(
            "agents.llm_explanation_agent.validate_artifact",
            side_effect=ArtifactValidationError("boom"),
        ):
            code = explanation_main(["--run-dir", str(run_dir), "--case", "case_review", "--provider", "none"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
