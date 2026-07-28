import json
import tempfile
import unittest
from pathlib import Path

from agents.qc_agent import generate_json_report, run_qc


def _case(*, pancreas_volume=0.0, pancreas_empty=True, lesion_volume=0.0):
    lesion_present = lesion_volume > 0
    return {
        "case_status": {"omit_from_training": False},
        "quality_control": {
            "pancreas_mask_empty": pancreas_empty,
            "lesion_mask_empty": not lesion_present,
            "total_overlap_rate_vs_lesion": 0.9 if lesion_present else None,
        },
        "pancreas": {
            "volume_mm3": pancreas_volume,
            "bounding_box_zyx": [[10, 10, 10], [40, 40, 40]] if pancreas_volume else None,
            "regions": {},
            "region_consistency": {},
        },
        "lesions": {
            "total_volume_mm3": lesion_volume,
            "n_lesions": 1 if lesion_present else 0,
            "per_lesion": [],
        },
        "image": {"shape_zyx": [64, 64, 64], "spacing_xyz_mm": [1.0, 1.0, 1.0]},
        "metadata": {
            "sex": "F",
            "age": 60,
            "ct_phase": "portal_venous",
            "manufacturer": "test",
        },
        "hu_statistics": {
            "hu_statistics_status": "pancreas_only_negative_case",
            "pancreas_median_hu": 80.0,
        },
    }


def _json_case(tmp_path, raw_case, task_mode, confirmed_negative_lesions=None):
    thresholds = {
        "TASK_MODE": task_mode,
        "FOV_POLICY": {
            "border_touching_severity": "low_warning",
            "missing_region_severity": "moderate_warning",
            "incomplete_coverage_severity": "moderate_warning",
            "truncation_severity": "high_warning",
        },
        "SCORE_WEIGHTS": {
            task_mode: {
                "lesion_localization": 0.0 if task_mode == "pancreas_only" else 1.0,
                "lesion_burden": 0.0 if task_mode == "pancreas_only" else 1.0,
                "region_consistency": 0.0,
            },
        },
    }
    results = run_qc(
        {"case": raw_case},
        thresholds,
        confirmed_negative_lesions=confirmed_negative_lesions,
    )
    report_path = tmp_path / "qc.json"
    generate_json_report(
        results,
        report_path,
        data={"case": raw_case},
        thr=thresholds,
    )
    return results["case"], json.loads(report_path.read_text())["case"]


class PancreasOnlySafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_empty_pancreas_is_not_a_negative_or_keep_in_pancreas_only(self):
        result, emitted = _json_case(self.tmp_path, _case(), "pancreas_only")

        self.assertEqual(result["sample_type"], "missing_required_target")
        self.assertEqual(result["recommendation"], "exclude")
        self.assertEqual(result["decision_hint"], "exclude_missing_required_target")
        self.assertNotIn("negative_case", emitted["retrieval_tags"])
        self.assertIn("required_target_missing", emitted["retrieval_tags"])
        self.assertEqual(emitted["target_presence"], {
            "target": "pancreas",
            "role": "primary_target",
            "expected_presence": "required",
            "annotation_presence": "absent",
            "observed_presence": "uncertain",
            "visible_target_annotation_status": "uncertain",
        })

    def test_nonempty_pancreas_is_valid_primary_target_and_emits_evidence(self):
        raw_case = _case(pancreas_volume=50_000.0, pancreas_empty=False)
        result, emitted = _json_case(self.tmp_path, raw_case, "pancreas_only")

        self.assertEqual(result["sample_type"], "positive")
        self.assertEqual(result["recommendation"], "keep")
        self.assertEqual(emitted["training_objective"], "pancreas_segmentation")
        self.assertIn("primary_target_pancreas", emitted["retrieval_tags"])
        self.assertEqual(emitted["target_presence"]["annotation_presence"], "present")
        self.assertEqual(emitted["target_presence"]["observed_presence"], "present")
        self.assertEqual(emitted["target_presence"]["visible_target_annotation_status"], "complete")
        self.assertEqual(emitted["measurements"]["pancreas_volume_mm3"], 50_000.0)
        self.assertEqual(emitted["hu_statistics"]["pancreas_median_hu"], 80.0)

    def test_lesion_negative_requires_separate_annotation_confirmation(self):
        raw_case = _case(pancreas_volume=50_000.0, pancreas_empty=False)
        result, emitted = _json_case(self.tmp_path, raw_case, "pancreas_lesion")

        self.assertEqual(result["sample_type"], "negative")
        self.assertEqual(result["recommendation"], "exclude")
        self.assertIn("unconfirmed_negative_lesion", emitted["retrieval_tags"])

        manifest_confirmed, _ = _json_case(
            self.tmp_path,
            raw_case,
            "pancreas_lesion",
            confirmed_negative_lesions={
                "case": {
                    "lesion_status": "confirmed_absent",
                    "confirmation_source": "fixture",
                    "confirmed_by": "test",
                }
            },
        )
        self.assertEqual(manifest_confirmed["recommendation"], "exclude")

        annotation_confirmed_case = _case(pancreas_volume=50_000.0, pancreas_empty=False)
        annotation_confirmed_case["quality_control"]["lesion_annotation_evidence"] = {
            "status": "confirmed_absent",
            "selected_source": "separate_lesion_mask",
            "lesion_present": False,
            "absence_confirmed": True,
            "sources": [],
        }
        confirmed, confirmed_emitted = _json_case(
            self.tmp_path,
            annotation_confirmed_case,
            "pancreas_lesion",
        )
        self.assertEqual(confirmed["recommendation"], "keep")
        self.assertEqual(confirmed["decision_hint"], "keep_negative")
        self.assertEqual(emitted["training_objective"], "pancreas_lesion_segmentation")
        self.assertIn("negative_case", confirmed_emitted["retrieval_tags"])
        self.assertEqual(confirmed_emitted["target_presence"]["target"], "pancreas")
        self.assertEqual(confirmed_emitted["target_presence"]["role"], "context")
        self.assertEqual(confirmed_emitted["target_presence"]["expected_presence"], "required")


if __name__ == "__main__":
    unittest.main()
