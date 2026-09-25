from __future__ import annotations

import builtins
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from artifacts.validation import (
    ArtifactValidationError,
    JsonSchemaUnavailableError,
    load_schema,
    validate_artifact,
)


SCHEMA_DIR = ROOT / "artifacts" / "schemas"
PROFILE_DIR = ROOT / "configs" / "task_profiles"
POLICY_DIR = ROOT / "configs" / "policies"
FIXTURE_DIR = Path(__file__).with_name("fixtures") / "contracts"
JSONSCHEMA_AVAILABLE = importlib.util.find_spec("jsonschema") is not None
CANONICAL_ACTIONS = ["keep", "warning", "review", "reject", "insufficient_evidence"]


class ContractTests(unittest.TestCase):
    def test_every_schema_and_fixture_loads(self) -> None:
        schemas = sorted(SCHEMA_DIR.glob("*.schema.json"))
        fixtures = sorted(FIXTURE_DIR.glob("*.json"))

        self.assertTrue(schemas)
        self.assertEqual(
            {path.name.replace(".schema.json", ".json") for path in schemas},
            {path.name for path in fixtures},
        )
        for path in schemas:
            schema = load_schema(path)
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
            self.assertEqual(schema["$id"], path.name)
        for path in fixtures:
            self.assertIsInstance(json.loads(path.read_text(encoding="utf-8")), dict)

    def test_every_task_profile_loads_and_preserves_legacy_fields(self) -> None:
        required_legacy_fields = {
            "TASK_MODE",
            "REQUIRED_COMPONENTS",
            "REQUIRED_SEGMENTATIONS",
            "ACTIVE_QC_DOMAINS",
            "HARD_FAILURE_DOMAINS",
            "CALIBRATABLE_DOMAINS",
        }
        profiles = sorted(PROFILE_DIR.glob("*.yaml"))

        self.assertTrue(profiles)
        for path in profiles:
            profile = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertLessEqual(required_legacy_fields, profile.keys(), str(path))

    def test_visible_pancreas_profile_contract(self) -> None:
        raw = yaml.safe_load((PROFILE_DIR / "pancreas_only.yaml").read_text(encoding="utf-8"))
        profile = raw["task_profile"]
        target = profile["targets"]["pancreas"]

        self.assertEqual(raw["TASK_MODE"], "pancreas_only")
        self.assertEqual(raw["REQUIRED_SEGMENTATIONS"], {"pancreas_mask": "pancreas.nii.gz"})
        self.assertEqual(profile["id"], "pancreas_segmentation_visible")
        self.assertEqual(profile["version"], "1.0.0")
        self.assertEqual(profile["modality"], "CT")
        self.assertEqual(profile["training_target"], "visible_pancreas")
        self.assertEqual(target["expected_presence"], "required")
        self.assertIs(target["allow_negative_cases"], False)
        self.assertEqual(target["valid_observed_presence"], ["present", "partial"])
        self.assertEqual(target["partial_fov_policy"], "allow_if_visible_target_annotated")
        self.assertEqual(set(profile["required_evidence"]), {"geometry", "fov", "target_presence"})
        self.assertEqual(profile["decision"]["outcomes"], CANONICAL_ACTIONS)

    def test_every_policy_loads_with_explicit_precedence(self) -> None:
        policies = sorted(POLICY_DIR.glob("*.yaml"))

        self.assertTrue(policies)
        for path in policies:
            policy = yaml.safe_load(path.read_text(encoding="utf-8"))["policy"]
            rules = policy["rules"]
            rule_ids = [rule["id"] for rule in rules]
            self.assertTrue(policy["version"])
            self.assertEqual(policy["evaluation"], "first_match")
            self.assertIn(policy["no_match_action"], CANONICAL_ACTIONS)
            self.assertEqual(policy["canonical_actions"], CANONICAL_ACTIONS)
            self.assertEqual(policy["precedence"], rule_ids)
            self.assertTrue(all(rule["action"] in CANONICAL_ACTIONS for rule in rules))
            self.assertTrue(all("when" in rule and "rationale" in rule for rule in rules))
            if policy["id"] == "visible_pancreas_v1":
                self.assertEqual(
                    rule_ids,
                    [f"VP-{index:03d}" for index in range(1, 15)],
                )
                self.assertFalse(
                    any("critique" in str(rule.get("when", {})).lower() for rule in rules)
                )

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "optional jsonschema package is unavailable")
    def test_representative_fixtures_validate_against_every_schema(self) -> None:
        for schema in sorted(SCHEMA_DIR.glob("*.schema.json")):
            fixture = FIXTURE_DIR / schema.name.replace(".schema.json", ".json")
            instance = json.loads(fixture.read_text(encoding="utf-8"))
            with self.subTest(schema=schema.name):
                validate_artifact(instance, schema)

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "optional jsonschema package is unavailable")
    def test_validation_reports_instance_paths(self) -> None:
        fixture = json.loads((FIXTURE_DIR / "eval_report.json").read_text(encoding="utf-8"))
        del fixture["data"]["dataset_statistics"]["n_cases"]

        with self.assertRaisesRegex(ArtifactValidationError, "data.dataset_statistics"):
            validate_artifact(fixture, "eval_report")

    @unittest.skipUnless(JSONSCHEMA_AVAILABLE, "optional jsonschema package is unavailable")
    def test_explanatory_schemas_reject_authority_fields(self) -> None:
        authority_fields = {
            "reasoning_artifact": ("route_to_review", True),
            "medical_critique": ("blocks_automatic_keep", False),
            "llm_explanation_artifact": ("decision_override", "keep"),
        }
        for schema_name, (field, value) in authority_fields.items():
            fixture_path = FIXTURE_DIR / f"{schema_name}.json"
            instance = json.loads(fixture_path.read_text(encoding="utf-8"))
            instance["data"]["cases"]["case_001"][field] = value
            with self.subTest(schema=schema_name, field=field):
                with self.assertRaises(ArtifactValidationError):
                    validate_artifact(instance, schema_name)

        nested = json.loads(
            (FIXTURE_DIR / "reasoning_artifact.json").read_text(encoding="utf-8")
        )
        nested["data"]["cases"]["case_001"]["evidence_references"]["deterministic"][
            "policy_action"
        ] = "keep"
        with self.assertRaises(ArtifactValidationError):
            validate_artifact(nested, "reasoning_artifact")

    def test_validation_fails_clearly_without_jsonschema(self) -> None:
        original_import = builtins.__import__

        def import_without_jsonschema(name, *args, **kwargs):
            if name == "jsonschema":
                raise ImportError("simulated missing optional dependency")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_jsonschema):
            with self.assertRaisesRegex(
                JsonSchemaUnavailableError,
                "python -m pip install jsonschema",
            ):
                validate_artifact({}, "common_artifact_envelope")


if __name__ == "__main__":
    unittest.main()
