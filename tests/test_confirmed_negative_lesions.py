from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.qc_agent import run_qc
from agents.validation_agent import validate_dataset
from artifacts.confirmed_negative_lesions import load_confirmed_negative_lesions


class ConfirmedNegativeLesionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "imagesTr"
        self.labels = self.root / "labelsTr"
        self.seg_dir = self.labels / "case_001" / "segmentations"
        self.images.mkdir(parents=True)
        self.seg_dir.mkdir(parents=True)
        (self.images / "case_001_0000.nii.gz").write_text("image", encoding="utf-8")
        (self.seg_dir / "pancreas.nii.gz").write_text("mask", encoding="utf-8")
        self.manifest = self.root / "confirmed_negative_lesions.json"
        self.manifest.write_text(json.dumps({
            "confirmed_negative_lesions": [{
                "case_id": "case_001",
                "task_mode": "pancreas_lesion",
                "lesion_status": "confirmed_absent",
                "confirmed_by": "fixture_reviewer",
                "confirmation_date": "2026-07-24",
                "confirmation_source": "fixture_manifest",
                "confirmation_scope": "Lesion absent for lesion segmentation task.",
            }]
        }), encoding="utf-8")
        self.paths = SimpleNamespace(
            raw_root=self.root,
            raw_images_dir=self.images,
            raw_labels_dir=self.labels,
            raw_metadata=self.root / "metadata.xlsx",
            image_suffix="_0000.nii.gz",
            dataset_name="fixture_dataset",
        )
        self.profile = {
            "REQUIRED_SEGMENTATIONS": {
                "pancreas_mask": "pancreas.nii.gz",
                "lesion_mask": "pancreatic_lesion.nii.gz",
            }
        }

    def test_missing_lesion_mask_is_not_converted_to_negative_by_manifest(self) -> None:
        without_confirmation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(without_confirmation["status"], "failed")
        self.assertIn("missing_segmentations:pancreatic_lesion.nii.gz", without_confirmation["cases"]["case_001"]["errors"])

        with_confirmation = validate_dataset(
            self.paths,
            "pancreas_lesion",
            self.profile,
            confirmed_negative_lesions_path=self.manifest,
        )
        self.assertEqual(with_confirmation["status"], "failed")
        self.assertTrue(with_confirmation["cases"]["case_001"]["omit_from_training"])
        self.assertIn("missing_segmentations:pancreatic_lesion.nii.gz", with_confirmation["cases"]["case_001"]["errors"])
        self.assertEqual(
            with_confirmation["cases"]["case_001"]["confirmed_negative_lesion"]["lesion_status"],
            "confirmed_absent",
        )

    def test_manifest_fails_closed_for_task_or_case_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "task_mode mismatch"):
            load_confirmed_negative_lesions(self.manifest, task_mode="pancreas_lesion_subregions")
        with self.assertRaisesRegex(ValueError, "not present"):
            load_confirmed_negative_lesions(
                self.manifest,
                task_mode="pancreas_lesion",
                known_case_ids={"case_other"},
            )

    def test_manifest_fails_closed_for_non_lesion_task_mode(self) -> None:
        pancreas_only_profile = {"REQUIRED_SEGMENTATIONS": {"pancreas_mask": "pancreas.nii.gz"}}

        with self.assertRaisesRegex(ValueError, "lesion-capable task modes"):
            load_confirmed_negative_lesions(self.manifest, task_mode="pancreas_only")

        validation = validate_dataset(
            self.paths,
            "pancreas_only",
            pancreas_only_profile,
            confirmed_negative_lesions_path=self.manifest,
        )
        self.assertEqual(validation["status"], "failed")
        self.assertIn("invalid_confirmed_negative_lesions", validation["errors"][0])
        self.assertEqual(validation["confirmed_negative_lesions"]["status"], "invalid")

    @staticmethod
    def _summary_case() -> dict:
        return {
            "case_status": {"omit_from_training": False},
            "quality_control": {"pancreas_mask_empty": False, "lesion_mask_empty": True},
            "pancreas": {"volume_mm3": 50000, "regions": {}, "region_consistency": None},
            "lesions": {"n_lesions": 0, "total_volume_mm3": 0.0, "per_lesion": []},
            "image": {"spacing_xyz_mm": [1, 1, 1], "shape_zyx": [64, 64, 64]},
            "metadata": {"sex": "F", "age": 50, "ct_phase": "venous", "manufacturer": "fixture"},
            "hu_statistics": {"hu_statistics_status": "pancreas_only_negative_case"},
        }

    def test_qc_uses_annotation_evidence_not_manifest_for_empty_lesion_status(self) -> None:
        data = {"case_001": self._summary_case()}
        unconfirmed = run_qc(data, {"TASK_MODE": "pancreas_lesion"})["case_001"]
        self.assertEqual(unconfirmed["recommendation"], "exclude")
        self.assertIn("unconfirmed_negative_lesion", {tag for _, tag, _ in unconfirmed["flags"]})

        confirmed = run_qc(
            data,
            {"TASK_MODE": "pancreas_lesion"},
            confirmed_negative_lesions={
                "case_001": {
                    "lesion_status": "confirmed_absent",
                    "confirmation_source": "fixture_manifest",
                    "confirmed_by": "fixture_reviewer",
                }
            },
        )["case_001"]
        self.assertEqual(confirmed["recommendation"], "exclude")
        self.assertFalse(confirmed["evidence"]["lesion_absence_confirmation"]["confirmed"])


if __name__ == "__main__":
    unittest.main()
