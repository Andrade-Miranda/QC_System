from __future__ import annotations

from pathlib import Path
import sys
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.utils.paths import APPROVED_V1_TASK_MODES, VALID_TASK_MODES
from agents.utils.task_profiles import load_task_profile


class TaskProfileTests(unittest.TestCase):
    def test_exact_approved_v1_profiles_are_executable(self) -> None:
        profile_dir = ROOT / "configs" / "task_profiles"

        self.assertEqual(VALID_TASK_MODES, APPROVED_V1_TASK_MODES)
        self.assertEqual({path.stem for path in profile_dir.glob("*.yaml")}, APPROVED_V1_TASK_MODES)
        for task_mode in sorted(APPROVED_V1_TASK_MODES):
            with self.subTest(task_mode=task_mode):
                profile = load_task_profile(task_mode, ROOT)
                self.assertEqual(profile["TASK_MODE"], task_mode)
                self.assertIn("REQUIRED_SEGMENTATIONS", profile)
                self.assertIn("ACTIVE_QC_DOMAINS", profile)

    def test_lesion_profiles_keep_lesion_domains_and_subregion_boundary(self) -> None:
        lesion = yaml.safe_load((ROOT / "configs" / "task_profiles" / "pancreas_lesion.yaml").read_text())
        subregions = yaml.safe_load((ROOT / "configs" / "task_profiles" / "pancreas_lesion_subregions.yaml").read_text())

        self.assertTrue(lesion["ACTIVE_QC_DOMAINS"]["lesion_localization"])
        self.assertFalse(lesion["ACTIVE_QC_DOMAINS"]["region_consistency"])
        self.assertTrue(subregions["ACTIVE_QC_DOMAINS"]["lesion_localization"])
        self.assertTrue(subregions["ACTIVE_QC_DOMAINS"]["region_consistency"])
        self.assertIn("lesion_mask", lesion["REQUIRED_SEGMENTATIONS"])
        self.assertIn("lesion_mask", subregions["REQUIRED_SEGMENTATIONS"])


if __name__ == "__main__":
    unittest.main()
