from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.utils.paths import resolve_project_paths
from agents.validation_agent import validate_dataset


class PathConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.project = Path(self.temp_dir.name)
        (self.project / "configs").mkdir()
        (self.project / "datasets" / "dataset_configs").mkdir(parents=True)
        (self.project / "datasets" / "dataset_configs" / "fixture.yaml").write_text(
            "DATASET_NAME: Fixture\nRAW_DATASET_ROOT: /dataset/root\n",
            encoding="utf-8",
        )
        self.paths_yaml = self.project / "configs" / "paths.yaml"
        self.paths_yaml.write_text(
            "\n".join([
                f"PROJECT_ROOT: {self.project}",
                "DATASET_CONFIG: datasets/dataset_configs/fixture.yaml",
                "OUTPUT_ROOT: outputs",
                "LOGS_DIR: logs",
            ]),
            encoding="utf-8",
        )

    def test_local_override_and_environment_precedence(self) -> None:
        (self.project / "configs" / "paths.local.yaml").write_text(
            "RAW_DATASET_ROOT: /local/root\n",
            encoding="utf-8",
        )
        old = os.environ.get("AGENTQC_RAW_DATASET_ROOT")
        try:
            if old is not None:
                del os.environ["AGENTQC_RAW_DATASET_ROOT"]
            local_paths = resolve_project_paths(self.paths_yaml)
            self.assertEqual(str(local_paths.raw_root), "/local/root")

            os.environ["AGENTQC_RAW_DATASET_ROOT"] = "/env/root"
            env_paths = resolve_project_paths(self.paths_yaml)
            self.assertEqual(str(env_paths.raw_root), "/env/root")
        finally:
            if old is None:
                os.environ.pop("AGENTQC_RAW_DATASET_ROOT", None)
            else:
                os.environ["AGENTQC_RAW_DATASET_ROOT"] = old

    def test_unconfigured_raw_dataset_root_fails_closed_validation(self) -> None:
        (self.project / "datasets" / "dataset_configs" / "fixture.yaml").write_text(
            "DATASET_NAME: Fixture\nRAW_DATASET_ROOT: ''\n",
            encoding="utf-8",
        )
        paths = resolve_project_paths(self.paths_yaml)
        validation = validate_dataset(paths, "pancreas_only", {"REQUIRED_SEGMENTATIONS": {}})
        self.assertEqual(validation["status"], "failed")
        self.assertIn("RAW_DATASET_ROOT is not configured", validation["errors"])


if __name__ == "__main__":
    unittest.main()
