from __future__ import annotations

import csv
import json
import subprocess
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.build_golden_review_package import build_review_package
from scripts.import_golden_review import import_golden_review


ROOT = Path(__file__).resolve().parents[1]


class GoldenReviewWorkflowTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.final_path, self.validation_path, self.evidence_path = self._input_artifacts()

    def _artifact(self, name, artifact_type, data):
        path = self.root / name
        path.write_text(json.dumps({
            "metadata": {
                "artifact_type": artifact_type,
                "run_id": "source-run",
                "dataset_name": "dataset-1",
                "task_mode": "pancreas_only",
            },
            "data": data,
        }), encoding="utf-8")
        return path

    def _input_artifacts(self):
        decisions = {}
        validation = {}
        evidence = {}
        action_counts = {"keep": 20, "warning": 20, "review": 20, "reject": 4}
        index = 0
        for action, count in action_counts.items():
            for _ in range(count):
                index += 1
                case_id = f"case-{index:03d}"
                domain = ("geometry_integrity", "fov_integrity", "pancreas_context", None)[index % 4]
                decisions[case_id] = {
                    "final_decision": action,
                    "primary_domain": domain,
                    "risk_level": "high" if action == "reject" else "medium" if action == "review" else "low",
                    "policy_rules_triggered": [f"rule-{index % 7}"],
                }
                validation[case_id] = {
                    "status": "valid",
                    "omit_from_training": False,
                    "resources": {
                        "image": {"path": f"/dataset/images/{case_id}.nii.gz", "exists": True},
                        "segmentation_dir": {"path": f"/dataset/labels/{case_id}", "exists": True},
                    },
                }
                presence = ("present", "partial", "uncertain")[index % 3]
                evidence[case_id] = {
                    "target_presence": {
                        "observed_presence": presence,
                        "annotation_presence": "absent" if presence == "uncertain" else "present",
                        "visible_target_annotation_status": "complete" if presence == "present" else "uncertain",
                    },
                    "triage": {
                        "recommendation": "exclude" if action == "reject" else action,
                        "risk_level": decisions[case_id]["risk_level"],
                        "score": index,
                    },
                }
        final_path = self._artifact("final.json", "final_qc_decisions", {"decisions": decisions})
        validation_path = self._artifact("validation.json", "dataset_validation", {
            "dataset_name": "dataset-1",
            "task_mode": "pancreas_only",
            "required_segmentations": {"pancreas_mask": "pancreas.nii.gz"},
            "cases": validation,
        })
        evidence_path = self._artifact("evidence.json", "qc_report_deterministic", {"cases": evidence})
        return final_path, validation_path, evidence_path

    @staticmethod
    def _rows(path):
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))

    def test_package_is_exact_blinded_stratified_and_reproducible(self):
        first = self.root / "package-1"
        second = self.root / "package-2"
        summary = build_review_package(
            self.final_path, self.validation_path, self.evidence_path, first, seed=17
        )
        build_review_package(
            self.final_path, self.validation_path, self.evidence_path, second, seed=17
        )

        manifest = self._rows(first / "reviewer_package" / "review_manifest.csv")
        reference = self._rows(first / "system_reference.csv")
        self.assertEqual(summary["n_cases"], 60)
        self.assertEqual(len(manifest), 60)
        self.assertEqual(len({row["case_id"] for row in manifest}), 60)
        self.assertNotIn("system_action", manifest[0])
        for field in ("annotation_completeness", "expected_action", "reviewer_id", "confidence", "rationale"):
            self.assertTrue(all(row[field] == "" for row in manifest))
        self.assertTrue(all("pancreas_mask" in row["mask_paths"] for row in manifest))

        counts = Counter(row["system_action"] for row in reference)
        self.assertGreaterEqual(counts["keep"], 15)
        self.assertGreaterEqual(counts["warning"], 15)
        self.assertGreaterEqual(counts["review"], 15)
        self.assertEqual(counts["reject"], 4)
        self.assertEqual(
            (first / "reviewer_package" / "review_manifest.csv").read_bytes(),
            (second / "reviewer_package" / "review_manifest.csv").read_bytes(),
        )
        self.assertEqual(
            (first / "system_reference.csv").read_bytes(),
            (second / "system_reference.csv").read_bytes(),
        )
        reviewer_files = sorted(path.name for path in (first / "reviewer_package").iterdir())
        self.assertEqual(reviewer_files, ["REVIEW_PROTOCOL.md", "review_manifest.csv"])
        protocol = (first / "reviewer_package" / "REVIEW_PROTOCOL.md").read_text(encoding="utf-8")
        self.assertIn("Do not assign a diagnosis", protocol)
        self.assertNotIn("system_reference", protocol)

    def test_completed_review_import_writes_provenance_artifact(self):
        package = self.root / "package"
        build_review_package(
            self.final_path, self.validation_path, self.evidence_path, package, seed=17
        )
        manifest_path = package / "reviewer_package" / "review_manifest.csv"
        rows = self._rows(manifest_path)
        for row in rows:
            row.update({
                "annotation_completeness": "complete",
                "expected_action": "warning",
                "reviewer_id": "reviewer-1",
                "confidence": "0.85",
                "rationale": "Task-specific annotation review completed.",
            })
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        output = self.root / "golden_labels.json"
        payload = import_golden_review(manifest_path, package / "system_reference.csv", output)

        self.assertEqual(payload["metadata"]["artifact_type"], "golden_labels")
        self.assertEqual(payload["metadata"]["run_id"], "source-run")
        self.assertEqual(payload["data"]["summary"]["n_labels"], 60)
        self.assertEqual(len(payload["data"]["labels"]), 60)
        first_label = next(iter(payload["data"]["labels"].values()))
        self.assertEqual(first_label["annotation_completeness"], "complete")
        self.assertEqual(first_label["confidence"], 0.85)
        self.assertTrue(payload["data"]["review_provenance"]["review_csv"]["sha256"])

        cli_output = self.root / "golden_labels_cli.json"
        result = subprocess.run([
            "/usr/bin/python3", str(ROOT / "scripts" / "import_golden_review.py"),
            "--review-package", str(package / "reviewer_package"),
            "--system-reference", str(package / "system_reference.csv"),
            "--output", str(cli_output),
        ], cwd=ROOT, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(cli_output.is_file())

    def test_import_rejects_incomplete_or_invalid_annotations(self):
        package = self.root / "package-invalid"
        build_review_package(
            self.final_path, self.validation_path, self.evidence_path, package, seed=17
        )
        manifest_path = package / "reviewer_package" / "review_manifest.csv"
        rows = self._rows(manifest_path)
        for row in rows:
            row.update({
                "annotation_completeness": "complete",
                "expected_action": "keep",
                "reviewer_id": "reviewer-1",
                "confidence": "0.8",
                "rationale": "Reviewed for task-specific curation.",
            })
        rows[0]["confidence"] = "1.2"
        with manifest_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        with self.assertRaisesRegex(ValueError, "confidence must be between 0 and 1"):
            import_golden_review(
                manifest_path,
                package / "system_reference.csv",
                self.root / "invalid.json",
            )


if __name__ == "__main__":
    unittest.main()
