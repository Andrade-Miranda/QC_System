import unittest

from agents.final_decision_agent import decide
from agents.review_routing_agent import route


def _case(recommendation="keep", risk="low", domains=None, primary_domain=None, **extra):
    case = {
        "triage": {"recommendation": recommendation, "risk_level": risk},
        "qc_domains": domains or {},
    }
    if primary_domain:
        case["primary_issue"] = {"domain": primary_domain}
    case.update(extra)
    return case


class RoutingPolicyTests(unittest.TestCase):
    def test_deterministic_hard_failure_is_rejected_and_not_routed(self):
        det = {
            "hard": _case(
                "exclude",
                "high",
                {"geometry_integrity": {"severity": "critical"}},
                "geometry_integrity",
            )
        }
        cal = {"hard": _case("keep")}
        routing = route(det, cal, {"changed_cases": [{"case_id": "hard", "changed": True}]})

        record = routing["cases"]["hard"]
        self.assertTrue(record["hard_failure"])
        self.assertFalse(record["route_to_review"])
        self.assertEqual(record["status"], "hard_failure")

        decision = decide(det, cal, routing)["decisions"]["hard"]
        self.assertEqual(decision["final_decision"], "reject")
        self.assertFalse(decision["requires_human_review"])
        self.assertFalse(decision["route_to_review"])
        self.assertEqual(decision["calibrated_recommendation"], "keep")
        self.assertTrue(decision["policy_rules_triggered"])

    def test_review_recommendations_are_routed_and_review_is_consistent(self):
        det = {"review": _case("review", "medium")}
        cal = {"review": _case("keep", "low")}
        routing = route(det, cal, {})

        self.assertTrue(routing["cases"]["review"]["route_to_review"])
        self.assertEqual([row["case_id"] for row in routing["routed_cases"]], ["review"])

        decision = decide(det, cal, routing)["decisions"]["review"]
        self.assertEqual(decision["final_decision"], "review")
        self.assertTrue(decision["requires_human_review"])
        self.assertTrue(decision["route_to_review"])

        legacy_routing = {"routed_cases": [{"case_id": "legacy", "route_to_review": True}]}
        legacy = decide({"legacy": _case()}, {"legacy": _case()}, legacy_routing)["decisions"]["legacy"]
        self.assertEqual(legacy["final_decision"], "review")
        self.assertTrue(legacy["requires_human_review"])

    def test_shared_uncertainty_and_score_disagreement_keep_stages_aligned(self):
        scenarios = {
            "triage-uncertainty": (
                _case(triage={"recommendation": "keep", "risk_level": "low", "uncertainty": "yes"}),
                _case(),
                {},
                "deterministic_uncertainty",
            ),
            "score-disagreement": (
                _case(triage={"recommendation": "keep", "risk_level": "low", "score": 1}),
                _case(triage={"recommendation": "keep", "risk_level": "low", "score": 2}),
                {},
                "deterministic_calibrated_disagreement",
            ),
        }

        for case_id, (det_case, cal_case, comparison, expected_rule) in scenarios.items():
            with self.subTest(case_id=case_id):
                routing = route({case_id: det_case}, {case_id: cal_case}, comparison)
                record = routing["cases"][case_id]
                decision = decide({case_id: det_case}, {case_id: cal_case}, routing)["decisions"][case_id]

                self.assertTrue(record["route_to_review"])
                self.assertIn(expected_rule, record["policy_rules_triggered"])
                self.assertEqual(decision["final_decision"], "review")
                self.assertIn(expected_rule, decision["policy_rules_triggered"])

    def test_partial_uncertain_visible_target_routes_from_deterministic_evidence(self):
        target_presence = {
            "observed_presence": "partial",
            "visible_target_annotation_status": "uncertain",
        }
        det = {"partial": _case(target_presence=target_presence)}
        cal = {"partial": _case()}

        routing = route(det, cal, {}, task_mode="pancreas_only")
        record = routing["cases"]["partial"]

        self.assertTrue(record["route_to_review"])
        self.assertEqual(record["status"], "routed")
        self.assertIn("partial_visible_target_annotation_uncertain", record["route_reasons"])
        self.assertIn("partial_visible_target_annotation_uncertain", record["policy_rules_triggered"])
        self.assertEqual([row["case_id"] for row in routing["routed_cases"]], ["partial"])

    def test_partial_uncertain_route_ignores_calibrated_and_critique_lookalikes(self):
        target_presence = {
            "observed_presence": "partial",
            "visible_target_annotation_status": "uncertain",
        }
        det = {
            "case": _case(
                medical_critique={"target_presence": target_presence},
                reasoning={"target_presence": target_presence},
            )
        }
        cal = {"case": _case(target_presence=target_presence)}

        routing = route(det, cal, {}, task_mode="pancreas_only")

        self.assertFalse(routing["cases"]["case"]["route_to_review"])
        self.assertNotIn("partial_visible_target_annotation_uncertain", routing["cases"]["case"]["route_reasons"])

    def test_empty_required_pancreas_is_hard_failure_and_not_queued(self):
        det = {
            "empty": _case(
                target_presence={
                    "expected_presence": "required",
                    "observed_presence": "uncertain",
                    "annotation_presence": "absent",
                    "visible_target_annotation_status": "uncertain",
                },
                measurements={"pancreas_present": False},
            )
        }
        cal = {"empty": _case("keep")}

        routing = route(det, cal, {}, task_mode="pancreas_only")
        record = routing["cases"]["empty"]

        self.assertTrue(record["hard_failure"])
        self.assertFalse(record["route_to_review"])
        self.assertEqual(record["status"], "hard_failure")
        self.assertIn(
            "deterministic_required_pancreas_mask_missing_or_empty",
            record["policy_rules_triggered"],
        )

    def test_legacy_metadata_warning_maps_to_warning_action(self):
        det = {"warning": _case("keep_with_metadata_warning")}
        cal = {"warning": _case("keep_with_metadata_warning")}
        routing = route(det, cal, {})

        self.assertFalse(routing["cases"]["warning"]["route_to_review"])
        decision = decide(det, cal, routing)["decisions"]["warning"]
        self.assertEqual(decision["final_decision"], "warning")
        self.assertEqual(decision["deterministic_recommendation"], "keep_with_metadata_warning")
        self.assertEqual(decision["calibrated_recommendation"], "keep_with_metadata_warning")

    def test_missing_case_input_is_insufficient_evidence(self):
        det = {"missing": _case("keep")}
        routing = route(det, {}, {})

        record = routing["cases"]["missing"]
        self.assertEqual(record["status"], "insufficient_evidence")
        self.assertEqual(record["evidence_status"], "insufficient_evidence")
        self.assertFalse(record["route_to_review"])

        decision = decide(det, {}, routing)["decisions"]["missing"]
        self.assertEqual(decision["final_decision"], "insufficient_evidence")
        self.assertEqual(decision["decision_basis"], "insufficient_evidence")

    def test_invalid_dataset_validation_is_hard_failure_and_rejected(self):
        det = {
            "invalid-status": _case("review", "medium"),
            "omitted": _case("keep"),
        }
        cal = {case_id: _case("keep") for case_id in det}
        validation = {
            "invalid-status": {"status": "invalid", "omit_from_training": False},
            "omitted": {"status": "valid", "omit_from_training": True},
        }

        routing = route(det, cal, {}, validation)
        for case_id in validation:
            with self.subTest(stage="routing", case_id=case_id):
                record = routing["cases"][case_id]
                self.assertTrue(record["hard_failure"])
                self.assertFalse(record["route_to_review"])
                self.assertEqual(record["status"], "hard_failure")
                self.assertTrue(any(rule.startswith("dataset_validation_") for rule in record["policy_rules_triggered"]))

        decisions = decide(det, cal, routing, validation)["decisions"]
        for case_id in validation:
            with self.subTest(stage="decision", case_id=case_id):
                decision = decisions[case_id]
                self.assertEqual(decision["final_decision"], "reject")
                self.assertEqual(decision["decision_basis"], "deterministic_hard_failure")
                self.assertTrue(decision["hard_failure"])
                self.assertFalse(decision["requires_human_review"])

    def test_validation_only_cases_are_included(self):
        validation = {
            "invalid-only": {"status": "invalid", "omit_from_training": True},
            "valid-only": {"status": "valid", "omit_from_training": False},
        }

        routing = route({}, {}, {}, validation)
        self.assertEqual(set(routing["cases"]), set(validation))
        self.assertEqual(routing["cases"]["invalid-only"]["status"], "hard_failure")
        self.assertEqual(routing["cases"]["valid-only"]["status"], "insufficient_evidence")

        decisions = decide({}, {}, routing, validation)["decisions"]
        self.assertEqual(set(decisions), set(validation))
        self.assertEqual(decisions["invalid-only"]["final_decision"], "reject")
        self.assertEqual(decisions["invalid-only"]["decision_basis"], "deterministic_hard_failure")
        self.assertEqual(decisions["valid-only"]["final_decision"], "insufficient_evidence")

    def test_valid_dataset_validation_preserves_qc_policy(self):
        det = {"valid": _case("keep")}
        cal = {"valid": _case("keep")}
        validation = {"valid": {"status": "valid", "omit_from_training": False}}

        routing = route(det, cal, {}, validation)
        record = routing["cases"]["valid"]
        self.assertFalse(record["hard_failure"])
        self.assertFalse(record["route_to_review"])
        self.assertEqual(record["dataset_validation_status"], "valid")

        decision = decide(det, cal, routing, validation)["decisions"]["valid"]
        self.assertEqual(decision["final_decision"], "keep")
        self.assertEqual(decision["decision_basis"], "calibrated_qc")
        self.assertFalse(decision["hard_failure"])
        self.assertEqual(decision["dataset_validation_omit_from_training"], False)

    def test_profile_domains_control_hard_failures_and_soft_warning_routes(self):
        profile = {
            "ACTIVE_QC_DOMAINS": {
                "profile_hard": True,
                "profile_soft": True,
                "fov_integrity": False,
            },
            "HARD_FAILURE_DOMAINS": ["profile_hard"],
        }
        det = {
            "profile-hard": _case(
                "keep",
                "low",
                {"profile_hard": {"severity": "critical"}},
                "profile_hard",
            ),
            "profile-soft": _case(
                "keep",
                "low",
                {
                    "profile_soft": {"severity": "moderate_warning"},
                    "fov_integrity": {"severity": "high_warning"},
                },
                "profile_soft",
            ),
        }
        cal = {
            "profile-hard": _case("keep"),
            "profile-soft": _case("keep"),
        }
        routing = route(det, cal, {}, profile=profile)

        self.assertTrue(routing["cases"]["profile-hard"]["hard_failure"])
        self.assertFalse(routing["cases"]["profile-hard"]["route_to_review"])
        soft_reasons = routing["cases"]["profile-soft"]["route_reasons"]
        self.assertTrue(routing["cases"]["profile-soft"]["route_to_review"])
        self.assertTrue(any("profile_soft" in reason for reason in soft_reasons))
        self.assertFalse(any("fov_integrity" in reason for reason in soft_reasons))

        decisions = decide(det, cal, routing, profile=profile)["decisions"]
        self.assertEqual(decisions["profile-hard"]["final_decision"], "reject")
        self.assertEqual(decisions["profile-soft"]["final_decision"], "review")

        disabled_det = {
            "disabled": _case(
                "keep",
                "low",
                {"lesion_localization": {"severity": "critical"}},
                "lesion_localization",
            )
        }
        disabled_cal = {"disabled": _case("keep")}
        loaded_profile_routing = route(disabled_det, disabled_cal, {}, task_mode="pancreas_only")
        self.assertFalse(loaded_profile_routing["cases"]["disabled"]["hard_failure"])
        self.assertFalse(loaded_profile_routing["cases"]["disabled"]["route_to_review"])


if __name__ == "__main__":
    unittest.main()
