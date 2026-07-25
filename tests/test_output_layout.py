from __future__ import annotations

from pathlib import Path
import json
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from artifacts.run_manifest import write_pointer_alias, write_run_manifest


class OutputLayoutTests(unittest.TestCase):
    def test_run_manifest_and_aliases_are_lightweight_pointers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "Fixture" / "pancreas_lesion" / "runs" / "run-001"
            run_dir.mkdir(parents=True)
            final = run_dir / "final_qc_decisions.json"
            eval_report = run_dir / "eval_report.json"
            final.write_text('{"final": true}\n', encoding="utf-8")
            eval_report.write_text('{"eval": true}\n', encoding="utf-8")
            task_dir = run_dir.parents[1]
            manifest_path = run_dir / "run_manifest.json"
            aliases = {
                "latest_run": task_dir / "latest_run.json",
                "latest_eval_report": task_dir / "latest_eval_report.json",
            }

            write_run_manifest(
                manifest_path,
                run_id="run-001",
                dataset_name="Fixture",
                task_mode="pancreas_lesion",
                run_dir=run_dir,
                artifacts={
                    "final_decisions": final,
                    "evaluation": eval_report,
                    "run_manifest": manifest_path,
                },
                aliases=aliases,
            )
            write_pointer_alias(
                aliases["latest_run"],
                run_id="run-001",
                run_dir=run_dir,
                target=manifest_path,
                role="latest_run_manifest",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            alias = json.loads(aliases["latest_run"].read_text(encoding="utf-8"))
            self.assertEqual(manifest["authority"]["reasoning_critique_llm_outputs"], "explanatory_nonbinding")
            self.assertNotIn("run_manifest", manifest["artifacts"])
            self.assertTrue(manifest["artifacts"]["final_decisions"]["exists"])
            self.assertIsNotNone(manifest["artifacts"]["final_decisions"]["sha256"])
            self.assertEqual(alias["target"], str(manifest_path))
            self.assertEqual(alias["role"], "latest_run_manifest")


if __name__ == "__main__":
    unittest.main()
