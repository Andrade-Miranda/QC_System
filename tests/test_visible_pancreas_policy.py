from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path

import yaml

from agents.final_decision_agent import (
    VISIBLE_PANCREAS_POLICY_INPUTS,
    VISIBLE_PANCREAS_POLICY_PATH,
    decide,
)
from agents.utils.deterministic_policy import PolicyError, load_first_match_policy


def _qc_case(
    *,
    observed: str = "present",
    annotation_status: str = "complete",
    annotation_presence: str = "present",
    recommendation: str = "keep",
    geometry_severity: str = "normal",
    warning: bool = False,
    image_readable: bool | None = None,
) -> dict:
    pancreas_present = annotation_presence == "present"
    domains = {
        "geometry_integrity": {"severity": geometry_severity},
        "fov_integrity": {"severity": "normal"},
        "metadata_completeness": {"severity": "low_warning" if warning else "normal"},
    }
    case = {
        "triage": {"recommendation": recommendation, "risk_level": "low", "score": 0},
        "qc_domains": domains,
        "measurements": {
            "pancreas_present": pancreas_present,
            "pancreas_volume_mm3": 100.0 if pancreas_present else 0.0,
            "border_touching": {},
            "partial_visibility_likely": observed == "partial",
            "anatomical_truncation_suspected": False,
        },
        "target_presence": {
            "expected_presence": "required",
            "observed_presence": observed,
            "annotation_presence": annotation_presence,
            "visible_target_annotation_status": annotation_status,
        },
    }
    if image_readable is not None:
        case["image_readable"] = image_readable
    return case


class VisiblePancreasPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validation = {"case": {"status": "valid", "omit_from_training": False}}
        raw = yaml.safe_load(VISIBLE_PANCREAS_POLICY_PATH.read_text(encoding="utf-8"))["policy"]
        self.yaml_rules = {rule["id"]: rule for rule in raw["rules"]}

    def _decision(
        self,
        det: dict | None = None,
        cal: dict | None = None,
        *,
        validation: dict | None = None,
        routing: dict | None = None,
        comparison: dict | None = None,
    ) -> tuple[dict, dict]:
        det_cases = {} if det is None else {"case": det}
        cal_cases = {} if cal is None else {"case": cal}
        result = decide(
            det_cases,
            cal_cases,
            routing or {},
            self.validation if validation is None else validation,
            comparison=comparison or {},
            task_mode="pancreas_only",
        )
        return result, result["decisions"]["case"]

    def test_yaml_policy_outcomes_precedence_and_exact_traces(self) -> None:
        valid_det = _qc_case()
        valid_cal = _qc_case()
        scenarios = [
            (
                "validation invalid precedes all evidence",
                valid_det,
                valid_cal,
                {"case": {"status": "invalid", "omit_from_training": False}},
                {"cases": {"case": {"route_to_review": True}}},
                {},
                "VP-001",
                "reject",
            ),
            ("unreadable image", _qc_case(image_readable=False), valid_cal, None, {}, {}, "VP-002", "reject"),
            ("invalid geometry", _qc_case(geometry_severity="critical"), valid_cal, None, {}, {}, "VP-003", "reject"),
            ("required target absent", _qc_case(observed="absent"), valid_cal, None, {}, {}, "VP-004", "reject"),
            (
                "empty required mask precedes uncertain presence",
                _qc_case(observed="uncertain", annotation_status="uncertain", annotation_presence="absent"),
                valid_cal,
                None,
                {},
                {},
                "VP-005",
                "reject",
            ),
            (
                "explicit incomplete annotation precedes route",
                _qc_case(annotation_status="incomplete"),
                valid_cal,
                None,
                {"cases": {"case": {"route_to_review": True}}},
                {},
                "VP-006",
                "reject",
            ),
            ("missing calibrated evidence", valid_det, None, None, {}, {}, "VP-007", "insufficient_evidence"),
            (
                "uncertain target presence",
                _qc_case(observed="uncertain", annotation_status="uncertain"),
                valid_cal,
                None,
                {},
                {},
                "VP-008",
                "insufficient_evidence",
            ),
            (
                "partial annotation uncertainty precedes explicit route",
                _qc_case(observed="partial", annotation_status="uncertain"),
                valid_cal,
                None,
                {"cases": {"case": {"route_to_review": True}}},
                {},
                "VP-009",
                "review",
            ),
            (
                "explicit deterministic route",
                valid_det,
                valid_cal,
                None,
                {"cases": {"case": {"route_to_review": True}}},
                {},
                "VP-010",
                "review",
            ),
            (
                "comparison instability",
                valid_det,
                valid_cal,
                None,
                {},
                {"all_cases": [{"case_id": "case", "recommendation_change": "escalated", "risk_change": "unchanged"}]},
                "VP-011",
                "review",
            ),
            (
                "complete partial annotation",
                _qc_case(observed="partial", annotation_status="complete"),
                valid_cal,
                None,
                {},
                {},
                "VP-012",
                "warning",
            ),
            (
                "nonblocking warning",
                valid_det,
                _qc_case(recommendation="keep_with_metadata_warning", warning=True),
                None,
                {},
                {},
                "VP-013",
                "warning",
            ),
            ("valid full visible pancreas", valid_det, valid_cal, None, {}, {}, "VP-014", "keep"),
            (
                "unrecognized evidence fails closed as missing evidence",
                _qc_case(annotation_status="unexpected"),
                valid_cal,
                None,
                {},
                {},
                "VP-007",
                "insufficient_evidence",
            ),
        ]

        for name, det, cal, validation, routing, comparison, rule_id, action in scenarios:
            with self.subTest(name=name):
                result, decision = self._decision(
                    det,
                    cal,
                    validation=validation,
                    routing=routing,
                    comparison=comparison,
                )
                expected_rule = self.yaml_rules[rule_id]
                self.assertEqual(decision["final_decision"], action)
                self.assertEqual(decision["policy_rules_triggered"], [rule_id])
                self.assertEqual(result["policy_id"], "visible_pancreas_v1")
                self.assertEqual(result["policy_version"], "1.0.0")
                self.assertEqual(decision["decision_trace"]["matched_rule"], {
                    "id": expected_rule["id"],
                    "name": expected_rule["name"],
                    "rationale": expected_rule["rationale"],
                })

    def test_missing_target_evidence_is_insufficient_not_geometry_reject(self) -> None:
        det = _qc_case()
        del det["target_presence"]
        _, decision = self._decision(det, _qc_case())
        self.assertEqual(decision["final_decision"], "insufficient_evidence")
        self.assertEqual(decision["policy_rules_triggered"], ["VP-007"])

    def test_critique_cannot_enter_policy_context_or_decide_interface(self) -> None:
        policy = load_first_match_policy(
            VISIBLE_PANCREAS_POLICY_PATH,
            allowed_roots=VISIBLE_PANCREAS_POLICY_INPUTS,
        )
        with self.assertRaisesRegex(PolicyError, "forbidden roots"):
            policy.evaluate({"critique": {"recommendation": "reject"}})
        self.assertNotIn("critique", inspect.signature(decide).parameters)
        self.assertNotIn("reasoning", inspect.signature(decide).parameters)

        clean_result, _ = self._decision(_qc_case(), _qc_case())
        det_with_critique = _qc_case()
        det_with_critique["critique"] = {
            "route_to_review": True,
            "recommendation": "reject",
        }
        result_with_critique, _ = self._decision(det_with_critique, _qc_case())
        self.assertEqual(result_with_critique, clean_result)

    def test_non_pancreas_policy_path_ignores_comparison_and_is_unchanged(self) -> None:
        det = {"case": {"triage": {"recommendation": "keep", "risk_level": "low"}, "qc_domains": {}}}
        cal = {"case": {"triage": {"recommendation": "keep", "risk_level": "low"}, "qc_domains": {}}}
        baseline = decide(det, cal, {}, task_mode="pancreas_lesion")
        with_comparison = decide(
            det,
            cal,
            {},
            comparison={"all_cases": [{"case_id": "case", "changed": True}]},
            task_mode="pancreas_lesion",
        )
        self.assertEqual(with_comparison, baseline)
        self.assertEqual(baseline["policy_version"], "v1.0")

    def test_explicit_policy_snapshot_is_used_while_direct_default_is_preserved(self) -> None:
        default_result, default_decision = self._decision(_qc_case(), _qc_case())
        self.assertEqual(default_decision["final_decision"], "keep")

        raw = yaml.safe_load(VISIBLE_PANCREAS_POLICY_PATH.read_text(encoding="utf-8"))
        rule = next(item for item in raw["policy"]["rules"] if item["id"] == "VP-014")
        rule["action"] = "warning"
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = Path(temp_dir) / "decision_policy.yaml"
            snapshot.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
            snapshot_result = decide(
                {"case": _qc_case()},
                {"case": _qc_case()},
                {},
                self.validation,
                comparison={},
                task_mode="pancreas_only",
                policy_path=snapshot,
            )

        self.assertEqual(snapshot_result["decisions"]["case"]["final_decision"], "warning")
        self.assertEqual(default_result["decisions"]["case"]["final_decision"], "keep")


if __name__ == "__main__":
    unittest.main()
