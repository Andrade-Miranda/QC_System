from __future__ import annotations

import unittest
from pathlib import Path

from scripts.calibrate_thresholds import calibrate


class CalibrationTaskScopeTests(unittest.TestCase):
    def test_only_task_calibratable_domains_are_considered(self) -> None:
        base = {
            "CALIBRATION_POLICY": {
                "version": "v1",
                "reference_clean": {"min_lesion_overlap_for_reference": 0.8},
                "minimums": {"min_reference_cases": 1, "min_reference_fraction": 0.0},
                "fov_integrity": {"calibrate": False},
                "pancreas_context": {"calibrate": False},
                "lesion_burden": {"calibrate": False},
            },
            "QC_PROFILE_THRESHOLDS": {},
        }
        cases = {
            "case-1": {
                "case_status": {"omit_from_training": False},
                "quality_control": {"pancreas_mask_empty": False, "lesion_mask_empty": True},
                "pancreas": {"volume_mm3": 50_000},
                "image": {"spacing_xyz_mm": [1, 1, 1]},
            }
        }

        _, report = calibrate(
            base,
            cases,
            Path("summary.json"),
            task_mode="pancreas_only",
            calibratable_domains={"fov_integrity", "pancreas_context"},
        )

        self.assertEqual(report["calibratable_domains"], ["fov_integrity", "pancreas_context"])
        self.assertEqual(report["domains"]["lesion_burden"]["status"], "not_calibratable_for_task")

    def test_unknown_calibratable_domain_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported calibratable domains"):
            calibrate(
                {},
                {},
                Path("summary.json"),
                task_mode="pancreas_only",
                calibratable_domains={"unknown_domain"},
            )


if __name__ == "__main__":
    unittest.main()
