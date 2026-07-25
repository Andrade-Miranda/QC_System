from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from agents.final_decision_agent import decide
from agents.interactive_review_agent import InteractiveReviewAgent
from agents.utils.llm_backend import ChatBackend, ProviderConfig, ProviderError, build_backend
from agents.utils.run_artifact_index import RunArtifactError, RunArtifactIndex
from artifacts.hashing import hash_file, hash_json_payload


ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "policies" / "visible_pancreas_v1.yaml"


class FakeBackend(ChatBackend):
    def __init__(self, response: str, *, fail: bool = False, provider: str = "ollama"):
        super().__init__(ProviderConfig(
            provider=provider,
            model="test-model",
            timeout_seconds=1,
            max_tokens=64,
            base_url="http://unused",
            api_key_env="UNUSED_KEY" if provider == "openai" else None,
        ))
        self.response = response
        self.fail = fail
        self.messages = None
        self.calls = 0

    def complete(self, messages):
        self.calls += 1
        self.messages = messages
        if self.fail:
            raise ProviderError("provider unavailable")
        return self.response


class NoneTestBackend(FakeBackend):
    def __init__(self):
        super().__init__("", provider="none")

    def complete(self, messages):
        raise AssertionError("disabled backend must not be called")


class InteractiveReviewAgentTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.run_dir = self.root / "run-1"
        self.run_dir.mkdir()
        self._build_run()

    def test_admin_index_rejects_non_pancreas_only_runs(self):
        for path in self.run_dir.glob("*.json"):
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metadata"]["task_mode"] = "pancreas_lesion"
            if path.name == "validated_run_context.json":
                payload["data"]["task_mode"] = "pancreas_lesion"
            path.write_text(json.dumps(payload), encoding="utf-8")

        with self.assertRaisesRegex(RunArtifactError, "supports pancreas_only"):
            RunArtifactIndex.load_admin(self.run_dir)

    def _artifact(self, filename, artifact_type, data, **metadata_override):
        metadata = {
            "artifact_type": artifact_type,
            "artifact_schema_version": "1.0",
            "generator": "TestFixture",
            "generator_version": "1.0",
            "created_at": "2026-07-22T12:00:00Z",
            "run_id": "run-1",
            "artifact_id": f"{artifact_type}-id",
            "dataset_name": "PantsMini",
            "task_mode": "pancreas_only",
            "input_resources": [],
            "input_artifacts": [],
            "configuration": {},
        }
        metadata.update(metadata_override)
        path = self.run_dir / filename
        path.write_text(json.dumps({"metadata": metadata, "data": data}), encoding="utf-8")
        return path

    @staticmethod
    def _qc_case(*, partial=False, abnormal=False):
        return {
            "target_presence": {
                "target": "pancreas",
                "role": "primary_target",
                "expected_presence": "required",
                "observed_presence": "partial" if partial else "present",
                "annotation_presence": "present",
                "visible_target_annotation_status": "uncertain" if partial else "complete",
            },
            "measurements": {
                "pancreas_present": True,
                "pancreas_volume_mm3": 100,
                "border_touching": {},
                "partial_visibility_likely": partial,
                "anatomical_truncation_suspected": False,
            },
            "triage": {"recommendation": "keep", "risk_level": "low", "score": 4 if abnormal else 0},
            "qc_domains": {
                "geometry_integrity": {"severity": "normal", "component_score": 0},
                "fov_integrity": {
                    "severity": "low_warning" if abnormal else "normal",
                    "component_score": 4 if abnormal else 0,
                },
            },
        }

    @staticmethod
    def _reasoning_case(case_id):
        ref = lambda kind, pointer: {
            "artifact_type": kind, "path": f"{kind}.json", "case_pointer": pointer
        }
        snapshot = {
            "availability": "available", "reported_recommendation": "keep", "risk_level": "low",
            "score": 0, "primary_domain": None, "abnormal_domains": [],
        }
        return {
            "case_id": case_id,
            "status": "complete",
            "task_profile_reference": {"artifact_type": "validated_run_context"},
            "evidence_references": {
                "deterministic": ref("qc_report_deterministic", f"/data/cases/{case_id}"),
                "calibrated": ref("qc_report_calibrated", f"/data/cases/{case_id}"),
                "comparison": ref("qc_comparison", "/data/all_cases/0"),
            },
            "evidence_summary": {
                "deterministic": snapshot,
                "calibrated": snapshot,
                "comparison": {
                    "availability": "available", "changed": False,
                    "recommendation_change": "unchanged", "risk_change": "unchanged",
                    "score_delta": 0, "domain_deterministic": None, "domain_calibrated": None,
                },
            },
            "conflicts": [], "uncertainty_annotations": [], "limitations": ["Structured evidence only."],
        }

    @staticmethod
    def _critique_case(case_id):
        return {
            "case_id": case_id, "status": "complete",
            "audit_scope": "grounding_and_internal_consistency_only",
            "supported_claim_checks": [], "unsupported_claim_checks": [], "concerns": [],
            "evidence_gaps": [], "severity": "none", "limitations": ["Structured audit only."],
            "binding": False,
        }

    def _build_run(self):
        cases = ["PanTS_00000001", "PanTS_00000002"]
        shutil.copy2(POLICY_PATH, self.run_dir / "decision_policy.yaml")
        image_dir = self.root / "images"
        mask_dir = self.root / "labels"
        image_dir.mkdir()
        mask_dir.mkdir()
        validation_cases = {}
        for case_id in cases:
            image = image_dir / f"{case_id}_0000.nii.gz"
            case_masks = mask_dir / case_id
            case_masks.mkdir()
            mask = case_masks / "pancreas.nii.gz"
            image.write_text("image", encoding="utf-8")
            mask.write_text("mask", encoding="utf-8")
            validation_cases[case_id] = {
                "status": "valid", "omit_from_training": False, "errors": [],
                "resources": {
                    "image": {"path": str(image), "exists": True},
                    "segmentation_dir": {"path": str(case_masks), "exists": True},
                },
            }
        det_cases = {cases[0]: self._qc_case(partial=True, abnormal=True), cases[1]: self._qc_case()}
        cal_cases = json.loads(json.dumps(det_cases))
        comparison_rows = [{
            "case_id": case_id, "changed": False, "recommendation_change": "unchanged",
            "risk_change": "unchanged", "score_delta": 0,
            "recommendation_deterministic": "keep", "recommendation_calibrated": "keep",
        } for case_id in cases]
        routing_cases = {
            cases[0]: {"case_id": cases[0], "route_to_review": True,
                       "route_reasons": ["partial_visible_target_annotation_uncertain"],
                       "hard_failure": False},
            cases[1]: {"case_id": cases[1], "route_to_review": False, "route_reasons": [],
                       "hard_failure": False},
        }
        routing_data = {
            "cases": routing_cases,
            "summary": {"n_routed": 1, "n_not_routed": 1},
            "routed_cases": [routing_cases[cases[0]]],
            "not_routed_cases": [routing_cases[cases[1]]],
        }
        comparison_data = {"all_cases": comparison_rows}
        final_data = decide(
            det_cases, cal_cases, routing_data, validation_cases,
            comparison=comparison_data, task_mode="pancreas_only",
            policy_path=self.run_dir / "decision_policy.yaml",
        )

        self._artifact("validated_run_context.json", "validated_run_context", {
            "case_ids": cases, "run_id": "run-1", "dataset_name": "PantsMini",
            "task_mode": "pancreas_only",
            "task_profile": {"id": "pancreas_segmentation_visible", "version": "1.0.0"},
            "dataset_resources": [], "path_configuration": {}, "threshold_configuration": {},
            "validation_status": "valid", "warnings": [],
        })
        validation_path = self._artifact("dataset_validation.json", "dataset_validation", {
            "cases": validation_cases, "required_segmentations": {"pancreas_mask": "pancreas.nii.gz"},
            "dataset_name": "PantsMini", "task_mode": "pancreas_only",
        })
        det_path = self._artifact("qc_report_deterministic.json", "qc_report_deterministic", {"cases": det_cases})
        self._artifact("qc_report_calibrated.json", "qc_report_calibrated", {"cases": cal_cases})
        self._artifact("qc_comparison.json", "qc_comparison", comparison_data)
        reasoning_cases = {case_id: self._reasoning_case(case_id) for case_id in cases}
        self._artifact("reasoning_artifact.json", "reasoning_artifact", {
            "status": "complete", "backend": {"name": "deterministic_template", "frozen": True},
            "alignment": {"context_validation_status": "valid", "issues": [], "case_counts": {
                "context": 2, "deterministic": 2, "calibrated": 2, "comparison": 2, "output": 2,
            }},
            "cases": reasoning_cases, "limitations": ["Nonbinding."],
        })
        critique_cases = {case_id: self._critique_case(case_id) for case_id in cases}
        self._artifact("medical_critique.json", "medical_critique", {
            "status": "complete", "backend": {"name": "deterministic_template", "frozen": True},
            "audit_scope": "grounding_and_internal_consistency_only",
            "alignment": {"issues": [], "case_counts": {
                "reasoning": 2, "deterministic": 2, "calibrated": 2, "comparison": 2, "output": 2,
            }},
            "cases": critique_cases, "limitations": ["Nonbinding."], "binding": False,
        })
        self._artifact("review_routing.json", "review_routing", routing_data)
        final_path = self._artifact(
            "final_qc_decisions.json", "final_qc_decisions", final_data,
            configuration={"decision_policy": {
                "id": "visible_pancreas_v1", "version": "1.0.0",
                "path": str((self.run_dir / "decision_policy.yaml").resolve()),
            }},
            input_resources=[{
                "resource_type": "decision_policy", "path": str(self.run_dir / "decision_policy.yaml"),
                "sha256": hash_file(self.run_dir / "decision_policy.yaml"),
            }],
        )
        self._artifact("eval_report.json", "eval_report", {
            "dataset_statistics": {
                "n_cases": 2, "decision_counts": final_data["summary"]["decision_counts"],
                "decision_basis_counts": {"review_routing": 1, "calibrated_qc": 1},
                "human_review_required": 1, "hard_failures": 0,
            },
            "calibration_impact": {}, "review_routing": {"n_routed": 1, "n_not_routed": 1},
            "artifact_consistency": {"final_cases": 2, "routed_cases": 1},
            "external_validation_limitations": {
                "status": "not_externally_validated", "limitations": ["Internal audit only."],
            },
        })

        package = self.run_dir / "golden_review_package"
        reviewer_package = package / "reviewer_package"
        reviewer_package.mkdir(parents=True)
        manifest_fields = [
            "review_id", "case_id", "image_path", "mask_paths", "annotation_completeness",
            "expected_action", "reviewer_id", "confidence", "rationale",
        ]
        manifest_row = {
            "review_id": "R001", "case_id": cases[0],
            "image_path": validation_cases[cases[0]]["resources"]["image"]["path"],
            "mask_paths": json.dumps({"pancreas_mask": str(mask_dir / cases[0] / "pancreas.nii.gz")}),
            "annotation_completeness": "", "expected_action": "", "reviewer_id": "",
            "confidence": "", "rationale": "",
        }
        with (reviewer_package / "review_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=manifest_fields)
            writer.writeheader()
            writer.writerow(manifest_row)
        source_hashes = {
            "final": hash_file(final_path), "validation": hash_file(validation_path),
            "deterministic": hash_file(det_path),
        }
        package_id = hash_json_payload({"seed": 17, "inputs": source_hashes, "case_ids": [cases[0]]})
        reference_row = {
            "package_id": package_id, "review_id": "R001", "case_id": cases[0],
            "selection_stage": "quota:review", "system_action": "review",
            "policy_rules_triggered": '["VP-009"]', "source_run_id": "run-1",
            "dataset_name": "PantsMini", "task_mode": "pancreas_only", "selection_seed": "17",
            "final_decisions_sha256": source_hashes["final"],
            "dataset_validation_sha256": source_hashes["validation"],
            "deterministic_evidence_sha256": source_hashes["deterministic"],
        }
        with (package / "system_reference.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(reference_row))
            writer.writeheader()
            writer.writerow(reference_row)
        (reviewer_package / "REVIEW_PROTOCOL.md").write_text(
            f"# Review Protocol\nPackage ID: `{package_id}`\nOpen the image and masks.\n", encoding="utf-8"
        )
        self.reviewer_package = reviewer_package

    def _admin(self, backend=None, **kwargs):
        return InteractiveReviewAgent(
            RunArtifactIndex.load_admin(self.run_dir), backend or NoneTestBackend(),
            log_interactions=False, **kwargs,
        )

    def test_admin_explanation_uses_recomputed_snapshot_policy(self):
        output = self._admin().answer("What happened to pants-1?")
        self.assertIn("FINAL POLICY ACTION (source of truth): review", output)
        self.assertIn("VP-009 partial_visible_target_annotation_uncertain", output)
        self.assertIn("decision_policy.yaml#/policy/rules/8", output)

    def test_natural_queries_and_case_priority(self):
        agent = self._admin(max_displayed_cases=1)
        self.assertIn("FINAL POLICY ACTION", agent.answer("Why did VP-009 match PanTS_00000001?"))
        self.assertIn("Final policy action review: 1 cases", agent.answer("How many cases need human review?"))
        self.assertIn("Final policy action review: 1 cases", agent.answer("Which cases were reviewed?"))
        self.assertIn("YAML rule VP-009", agent.answer("show all cases matched by VP009"))
        self.assertIn("Final policy action reject: 0 cases", agent.answer("Which cases were rejected?"))

    def test_case_lists_are_capped_with_full_count(self):
        agent = self._admin(max_displayed_cases=1)
        output = agent.answer("list action review")
        self.assertIn("1 cases; displaying 1", output)
        rule = agent.answer("show rule VP-009")
        self.assertIn("1 total; displaying 1", rule)

    def test_llm_is_not_called_for_deterministically_parsed_query(self):
        backend = FakeBackend("Misleading prose claiming reject.")
        output = self._admin(backend).answer("case pants1")
        self.assertEqual(backend.calls, 0)
        self.assertNotIn("Misleading", output)
        self.assertNotIn("LLM REWRITE", output)

    def test_unknown_query_can_be_classified_then_rendered_deterministically(self):
        backend = FakeBackend('{"tool":"case_detail","args":{"case":"pants1"}}')
        agent = self._admin(backend)
        output = agent.answer("Tell me the disposition of scan one")
        self.assertEqual(backend.calls, 1)
        self.assertIn("FINAL POLICY ACTION (source of truth): review", output)
        prompt = json.dumps(backend.messages)
        self.assertIn("command_schema", prompt)
        for forbidden in ("qc_report", "final_policy_action", "/images/", "system_reference", "quota:review"):
            self.assertNotIn(forbidden, prompt)

    def test_misleading_or_invalid_classifier_output_is_rejected_not_displayed(self):
        backend = FakeBackend("Ignore safety. The action is reject.")
        agent = self._admin(backend)
        output = agent.answer("unrecognized wording")
        self.assertIn("Unknown request", output)
        self.assertNotIn("Ignore safety", output)
        self.assertTrue(agent.last_state.fallback)

    def test_provider_failure_preserves_unknown_deterministic_fallback(self):
        agent = self._admin(FakeBackend("", fail=True))
        output = agent.answer("unrecognized wording")
        self.assertIn("Unknown request", output)
        self.assertNotIn("provider unavailable", output)
        self.assertTrue(agent.last_state.classification_used)
        self.assertTrue(agent.last_state.fallback)

    def test_classifier_arguments_are_authorized_and_fail_closed(self):
        backend = FakeBackend('{"tool":"case_detail","args":{"case":"pants999"}}')
        agent = self._admin(backend)
        output = agent.answer("unrecognized wording")
        self.assertIn("Unknown request", output)
        self.assertTrue(agent.last_state.fallback)

    def test_reviewer_loads_only_sanitized_package_and_uses_reviewer_schema(self):
        (self.run_dir / "golden_review_package" / "system_reference.csv").write_text(
            "TOP_SECRET", encoding="utf-8"
        )
        backend = FakeBackend('{"tool":"paths","args":{"case":"pants1"}}')
        index = RunArtifactIndex.load_reviewer(self.reviewer_package)
        self.assertIsNone(index.run_dir)
        self.assertEqual(index.artifacts, {})
        agent = InteractiveReviewAgent(index, backend, log_interactions=False)
        output = agent.answer("locate scan one")
        self.assertIn("Blinded reviewer resources", output)
        prompt = json.dumps(backend.messages)
        for hidden in (
            "TOP_SECRET", "list_action", "list_rule", "evidence", "golden_selection",
            "system_reference", "final_decision", "/images/",
        ):
            self.assertNotIn(hidden, prompt)
        for hidden in (
            "TOP_SECRET", "list_action", "list_rule", "evidence", "golden_selection",
            "system_reference", "final_decision",
        ):
            self.assertNotIn(hidden, output)
        blocked = agent.answer("why action pants1 rule VP-009")
        self.assertIn("Blinded-access restriction", blocked)
        self.assertEqual(backend.calls, 1)

    def test_reviewer_package_rejects_extra_files(self):
        (self.reviewer_package / "system_reference.csv").write_text("forbidden", encoding="utf-8")
        with self.assertRaisesRegex(RunArtifactError, "unauthorized entries"):
            RunArtifactIndex.load_reviewer(self.reviewer_package)

    def test_logging_uses_hashes_identity_and_mode_0600(self):
        log_path = self.root / "admin.jsonl"
        agent = InteractiveReviewAgent(
            RunArtifactIndex.load_admin(self.run_dir), NoneTestBackend(),
            log_path=log_path, log_interactions=True,
        )
        agent.answer("summary")
        record = json.loads(log_path.read_text(encoding="utf-8"))
        self.assertNotIn("query", record)
        self.assertEqual(len(record["query_sha256"]), 64)
        self.assertEqual(len(record["deterministic_result_sha256"]), 64)
        self.assertEqual(record["run_id"], "run-1")
        self.assertIn("artifact_ids", record)
        self.assertEqual(os.stat(log_path).st_mode & 0o777, 0o600)

        reviewer_log = self.root / "reviewer.jsonl"
        reviewer = InteractiveReviewAgent(
            RunArtifactIndex.load_reviewer(self.reviewer_package), NoneTestBackend(),
            log_path=reviewer_log, log_interactions=True,
        )
        reviewer.answer("paths pants1")
        reviewer_record = json.loads(reviewer_log.read_text(encoding="utf-8"))
        self.assertEqual(reviewer_record["package_id"], reviewer.index.package_id)
        for hidden in ("run_id", "dataset_name", "task_mode", "artifact_ids", "query"):
            self.assertNotIn(hidden, reviewer_record)

    def test_metadata_schema_hash_policy_and_golden_integrity_fail_closed(self):
        path = self.run_dir / "qc_report_calibrated.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["dataset_name"] = "other"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RunArtifactError, "Metadata mismatch"):
            RunArtifactIndex.load_admin(self.run_dir)

    def test_available_schema_validation_fails_closed(self):
        path = self.run_dir / "validated_run_context.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload["data"]["warnings"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RunArtifactError, "does not satisfy validated_run_context"):
            RunArtifactIndex.load_admin(self.run_dir)

    def test_loaded_input_artifact_hash_mismatch_fails_closed(self):
        path = self.run_dir / "eval_report.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["metadata"]["input_artifacts"] = [{
            "artifact_type": "final_qc_decisions",
            "path": str(self.run_dir / "final_qc_decisions.json"),
            "sha256": "0" * 64,
        }]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RunArtifactError, "Input-artifact hash mismatch"):
            RunArtifactIndex.load_admin(self.run_dir)

    def test_golden_source_hash_and_package_identity_fail_closed(self):
        path = self.run_dir / "golden_review_package" / "system_reference.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        rows[0]["final_decisions_sha256"] = "0" * 64
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(RunArtifactError, "source hashes"):
            RunArtifactIndex.load_admin(self.run_dir)

    def test_policy_recomposition_detects_tampered_final_data(self):
        shutil.rmtree(self.run_dir / "golden_review_package")
        path = self.run_dir / "final_qc_decisions.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["data"]["decisions"]["PanTS_00000001"]["decision_basis"] = "tampered"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with self.assertRaisesRegex(RunArtifactError, "recomputed deterministic policy composition"):
            RunArtifactIndex.load_admin(self.run_dir)

    def test_external_backend_requires_explicit_permission(self):
        config = ProviderConfig(
            provider="openai", model="test", timeout_seconds=1, max_tokens=1,
            base_url="https://example.invalid", api_key_env="KEY",
        )
        with self.assertRaisesRegex(ProviderError, "allow-external-provider"):
            build_backend(config)
        self.assertIsNotNone(build_backend(config, allow_external_provider=True))

    def test_cli_mode_paths_and_one_shot(self):
        script = str(ROOT / "agents" / "interactive_review_agent.py")
        admin = subprocess.run([
            sys.executable, script, "--run-dir", str(self.run_dir), "--mode", "admin",
            "--query", "summary", "--provider", "none", "--no-log",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(admin.returncode, 0, admin.stderr)
        self.assertIn("Run run-1", admin.stdout)

        external_rejected = subprocess.run([
            sys.executable, script, "--run-dir", str(self.run_dir), "--mode", "admin",
            "--query", "summary", "--provider", "openai", "--no-log",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(external_rejected.returncode, 2)
        self.assertIn("allow-external-provider", external_rejected.stderr)

        rejected = subprocess.run([
            sys.executable, script, "--run-dir", str(self.run_dir), "--mode", "reviewer",
            "--query", "summary", "--provider", "none",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(rejected.returncode, 2)
        self.assertIn("requires --review-package", rejected.stderr)

        reviewer = subprocess.run([
            sys.executable, script, "--review-package", str(self.reviewer_package),
            "--mode", "reviewer", "--query", "summary", "--provider", "none",
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(reviewer.returncode, 0, reviewer.stderr)
        self.assertIn("Blinded package", reviewer.stdout)
        self.assertFalse((ROOT / "logs" / "reviewer").exists())


if __name__ == "__main__":
    unittest.main()
