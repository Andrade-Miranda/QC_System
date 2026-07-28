from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest

try:
    import numpy as np
    import SimpleITK as sitk
except ModuleNotFoundError:  # pragma: no cover - dependency presence varies by test host
    np = None
    sitk = None


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.qc_agent import run_qc
from agents.validation_agent import validate_dataset
if np is not None and sitk is not None:
    from scripts import summarize_dataset as sd
else:  # pragma: no cover
    sd = None


class AnnotationDerivedLesionNegativeTests(unittest.TestCase):
    def setUp(self) -> None:
        if np is None or sitk is None or sd is None:
            self.skipTest("numpy and SimpleITK are required for synthetic NIfTI annotation tests")
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.images = self.root / "imagesTr"
        self.labels = self.root / "labelsTr"
        self.case_id = "case_001"
        self.seg_dir = self.labels / self.case_id / "segmentations"
        self.images.mkdir(parents=True)
        self.seg_dir.mkdir(parents=True)
        self.paths = SimpleNamespace(
            raw_root=self.root,
            raw_images_dir=self.images,
            raw_labels_dir=self.labels,
            raw_metadata=self.root / "metadata.xlsx",
            image_suffix="_0000.nii.gz",
            dataset_name="fixture_dataset",
            raw_dataset_root_configured=True,
        )
        sd._PATHS = self.paths
        self.profile = {
            "REQUIRED_SEGMENTATIONS": {
                "pancreas_mask": "pancreas.nii.gz",
                "lesion_mask": "pancreatic_lesion.nii.gz",
            }
        }
        self._write_image(self.images / f"{self.case_id}_0000.nii.gz", np.zeros((5, 5, 5), dtype=np.int16))
        pancreas = np.zeros((5, 5, 5), dtype=np.uint8)
        pancreas[1:4, 1:4, 1:4] = 1
        self._write_image(self.seg_dir / "pancreas.nii.gz", pancreas)

    @staticmethod
    def _write_image(path: Path, arr: np.ndarray) -> None:
        img = sitk.GetImageFromArray(arr)
        img.SetSpacing((1.0, 1.0, 1.0))
        sitk.WriteImage(img, str(path))

    def _summary(self) -> dict:
        return {self.case_id: sd.process_case(self.case_id, {}, "pancreas_lesion")}

    def _qc(self) -> dict:
        return run_qc(self._summary(), {"TASK_MODE": "pancreas_lesion"})[self.case_id]

    def test_non_empty_separate_lesion_annotation_is_positive(self) -> None:
        lesion = np.zeros((5, 5, 5), dtype=np.uint8)
        lesion[2, 2, 2] = 1
        self._write_image(self.seg_dir / "pancreatic_lesion.nii.gz", lesion)

        result = self._qc()
        self.assertTrue(result["evidence"]["lesion_present"])
        self.assertEqual(result["evidence"]["lesion_annotation_evidence"]["status"], "lesion_present")
        self.assertNotIn("unconfirmed_negative_lesion", {tag for _, tag, _ in result["flags"]})

    def test_empty_valid_separate_lesion_annotation_confirms_absence(self) -> None:
        self._write_image(self.seg_dir / "pancreatic_lesion.nii.gz", np.zeros((5, 5, 5), dtype=np.uint8))

        result = self._qc()
        self.assertFalse(result["evidence"]["lesion_present"])
        self.assertTrue(result["evidence"]["lesion_absence_confirmation"]["confirmed"])
        self.assertEqual(result["evidence"]["lesion_absence_confirmation"]["source"], "valid_annotation_evidence")
        self.assertIn("confirmed_negative_lesion", {tag for _, tag, _ in result["flags"]})

    def test_combined_labels_with_lesion_label_does_not_replace_missing_separate_mask(self) -> None:
        shared = np.zeros((5, 5, 5), dtype=np.uint8)
        shared[1:4, 1:4, 1:4] = 1
        shared[2, 2, 2] = 2
        self._write_image(self.seg_dir / "combined_labels.nii.gz", shared)

        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "failed")
        self.assertIn("missing_segmentations:pancreatic_lesion.nii.gz", validation["cases"][self.case_id]["errors"])
        result = self._qc()
        self.assertEqual(result["recommendation"], "exclude")
        self.assertIn("lesion_annotation_insufficient_evidence", {tag for _, tag, _ in result["flags"]})

    def test_combined_labels_without_lesion_label_does_not_confirm_missing_separate_mask_absence(self) -> None:
        shared = np.zeros((5, 5, 5), dtype=np.uint8)
        shared[1:4, 1:4, 1:4] = 1
        self._write_image(self.seg_dir / "combined_labels.nii.gz", shared)

        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "failed")
        result = self._qc()
        self.assertEqual(result["recommendation"], "exclude")
        self.assertFalse(result["evidence"].get("lesion_absence_confirmation", {}).get("confirmed", False))

    def test_missing_lesion_annotation_fails_closed(self) -> None:
        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "failed")
        self.assertIn("missing_segmentations:pancreatic_lesion.nii.gz", validation["cases"][self.case_id]["errors"])

        result = self._qc()
        self.assertEqual(result["recommendation"], "exclude")
        self.assertIn("lesion_annotation_insufficient_evidence", {tag for _, tag, _ in result["flags"]})

    def test_unreadable_lesion_annotation_fails_closed(self) -> None:
        (self.seg_dir / "pancreatic_lesion.nii.gz").write_bytes(b"")

        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "failed")
        result = self._qc()
        tags = {tag for _, tag, _ in result["flags"]}
        self.assertEqual(result["recommendation"], "exclude")
        self.assertIn("lesion_annotation_insufficient_evidence", tags)

    def test_nonempty_geometry_mismatched_separate_lesion_annotation_fails_closed(self) -> None:
        lesion = np.zeros((5, 5, 5), dtype=np.uint8)
        lesion[2, 2, 2] = 1
        img = sitk.GetImageFromArray(lesion)
        img.SetSpacing((2.0, 1.0, 1.0))
        sitk.WriteImage(img, str(self.seg_dir / "pancreatic_lesion.nii.gz"))

        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "failed")
        self.assertIn("missing_segmentations:insufficient_lesion_annotation_evidence", validation["cases"][self.case_id]["errors"])
        result = self._qc()
        tags = {tag for _, tag, _ in result["flags"]}
        self.assertEqual(result["recommendation"], "exclude")
        self.assertFalse(result["evidence"]["lesion_present"])
        self.assertIn("lesion_annotation_insufficient_evidence", tags)

    def test_combined_labels_disagreement_has_no_effect_on_empty_separate_mask_status(self) -> None:
        self._write_image(self.seg_dir / "pancreatic_lesion.nii.gz", np.zeros((5, 5, 5), dtype=np.uint8))
        shared = np.zeros((5, 5, 5), dtype=np.uint8)
        shared[2, 2, 2] = 2
        self._write_image(self.seg_dir / "combined_labels.nii.gz", shared)

        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "passed")
        result = self._qc()
        self.assertFalse(result["evidence"]["lesion_present"])
        self.assertTrue(result["evidence"]["lesion_absence_confirmation"]["confirmed"])
        self.assertEqual(result["evidence"]["lesion_annotation_evidence"]["selected_source"], "separate_lesion_mask")
        self.assertNotIn("contradictions", result["evidence"]["lesion_annotation_evidence"])
        self.assertNotIn("lesion_annotation_contradiction", {tag for _, tag, _ in result["flags"]})

    def test_combined_labels_disagreement_has_no_effect_on_nonempty_separate_mask_status(self) -> None:
        lesion = np.zeros((5, 5, 5), dtype=np.uint8)
        lesion[2, 2, 2] = 1
        self._write_image(self.seg_dir / "pancreatic_lesion.nii.gz", lesion)
        self._write_image(self.seg_dir / "combined_labels.nii.gz", np.zeros((5, 5, 5), dtype=np.uint8))

        validation = validate_dataset(self.paths, "pancreas_lesion", self.profile)
        self.assertEqual(validation["status"], "passed")
        result = self._qc()
        self.assertTrue(result["evidence"]["lesion_present"])
        self.assertEqual(result["evidence"]["lesion_annotation_evidence"]["selected_source"], "separate_lesion_mask")
        self.assertNotIn("contradictions", result["evidence"]["lesion_annotation_evidence"])
        self.assertNotIn("lesion_annotation_contradiction", {tag for _, tag, _ in result["flags"]})


if __name__ == "__main__":
    unittest.main()
