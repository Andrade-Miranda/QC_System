"""Validated, mode-aware indexes over completed QC run artifacts."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from artifacts.hashing import hash_file, hash_json_payload
from artifacts.io import load_artifact, read_json
from artifacts.validation import ArtifactValidationError, JsonSchemaUnavailableError, validate_artifact


ADMIN_ARTIFACTS = {
    "validated_context": ("validated_run_context.json", "validated_run_context"),
    "dataset_validation": ("dataset_validation.json", "dataset_validation"),
    "deterministic": ("qc_report_deterministic.json", "qc_report_deterministic"),
    "calibrated": ("qc_report_calibrated.json", "qc_report_calibrated"),
    "comparison": ("qc_comparison.json", "qc_comparison"),
    "reasoning": ("reasoning_artifact.json", "reasoning_artifact"),
    "critique": ("medical_critique.json", "medical_critique"),
    "routing": ("review_routing.json", "review_routing"),
    "final": ("final_qc_decisions.json", "final_qc_decisions"),
    "evaluation": ("eval_report.json", "eval_report"),
}
CASE_MAP_KEYS = {
    "dataset_validation": "cases",
    "deterministic": "cases",
    "calibrated": "cases",
    "reasoning": "cases",
    "critique": "cases",
    "routing": "cases",
    "final": "decisions",
}
_IDENTITY_FIELDS = ("run_id", "dataset_name", "task_mode")
SCHEMA_BY_ARTIFACT = {
    "validated_context": "validated_run_context",
    "reasoning": "reasoning_artifact",
    "critique": "medical_critique",
    "routing": "review_routing",
    "final": "final_reliability_decisions",
    "evaluation": "eval_report",
}


class RunArtifactError(ValueError):
    """Raised when a run cannot be indexed without ambiguity."""


class UnknownCaseError(RunArtifactError):
    """Raised when a case reference is not in the authorized case set."""


def json_pointer(*parts: object) -> str:
    escaped = [str(part).replace("~", "~0").replace("/", "~1") for part in parts]
    return "/" + "/".join(escaped)


def _load_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error) as exc:
        raise RunArtifactError(f"Cannot read {path.name}: {exc}") from exc


def _unique_rows(rows: list[dict[str, str]], path: Path) -> dict[str, tuple[int, dict[str, str]]]:
    indexed: dict[str, tuple[int, dict[str, str]]] = {}
    for row_number, row in enumerate(rows, start=2):
        case_id = str(row.get("case_id") or "").strip()
        if not case_id:
            raise RunArtifactError(f"{path.name} row {row_number} has no case_id")
        if case_id in indexed:
            raise RunArtifactError(f"{path.name} contains duplicate case_id {case_id}")
        indexed[case_id] = (row_number, row)
    return indexed


def _case_aliases(case_id: str) -> set[str]:
    aliases = {case_id.lower(), re.sub(r"[^a-z0-9]", "", case_id.lower())}
    match = re.fullmatch(r"(?i)pants[_-]?(\d+)", case_id)
    if match:
        number = str(int(match.group(1)))
        aliases.update({number, f"pants{number}", f"pants{int(number):08d}"})
    return aliases


@dataclass(frozen=True)
class ReviewerResource:
    review_id: str
    case_id: str
    image_path: str
    mask_paths: dict[str, str]
    manifest_row: int


class RunArtifactIndex:
    """Read-only index whose loaded fields are constrained by its mode."""

    def __init__(
        self,
        root: Path,
        mode: str,
        *,
        allowed_task_modes: set[str] | None = None,
        verify_pancreas_policy: bool = True,
    ):
        resolved = Path(root).expanduser().resolve()
        self.run_dir: Path | None = resolved if mode == "admin" else None
        self.package_dir: Path | None = resolved if mode == "reviewer" else None
        self.mode = mode
        self.allowed_task_modes = allowed_task_modes
        self.verify_pancreas_policy = verify_pancreas_policy
        self.artifacts: dict[str, dict[str, Any]] = {}
        self.metadata: dict[str, dict[str, Any]] = {}
        self.case_maps: dict[str, dict[str, Any]] = {}
        self.case_pointers: dict[str, dict[str, str]] = {}
        self.manifest: dict[str, tuple[int, dict[str, str]]] = {}
        self.system_reference: dict[str, tuple[int, dict[str, str]]] = {}
        self.reviewer_resources: dict[str, ReviewerResource] = {}
        self.protocol: str | None = None
        self.policy: dict[str, Any] | None = None
        self.policy_path: Path | None = None
        self.policy_rules: dict[str, tuple[int, dict[str, Any]]] = {}
        self.artifact_paths: dict[str, Path] = {}
        self.package_id: str | None = None
        self._aliases: dict[str, str] = {}

    @classmethod
    def load(
        cls,
        root: Path,
        mode: str,
        *,
        allowed_task_modes: set[str] | None = None,
        verify_pancreas_policy: bool = True,
    ) -> "RunArtifactIndex":
        if mode not in {"admin", "reviewer"}:
            raise RunArtifactError(f"Unsupported mode: {mode}")
        index = cls(
            root,
            mode,
            allowed_task_modes=allowed_task_modes,
            verify_pancreas_policy=verify_pancreas_policy,
        )
        selected_root = index.package_dir if mode == "reviewer" else index.run_dir
        if selected_root is None or not selected_root.is_dir():
            raise RunArtifactError(f"{mode.title()} root directory does not exist: {selected_root}")
        if mode == "reviewer":
            index._load_reviewer_view()
        else:
            index._load_admin_view()
        return index

    @classmethod
    def load_admin(
        cls,
        run_dir: Path,
        *,
        allowed_task_modes: set[str] | None = None,
        verify_pancreas_policy: bool = True,
    ) -> "RunArtifactIndex":
        return cls.load(
            run_dir,
            "admin",
            allowed_task_modes=allowed_task_modes,
            verify_pancreas_policy=verify_pancreas_policy,
        )

    @classmethod
    def load_reviewer(cls, review_package: Path) -> "RunArtifactIndex":
        return cls.load(review_package, "reviewer")

    def resolve_case(self, value: str) -> str:
        raw = Path(str(value).strip()).name
        raw = re.sub(r"_0000\.nii(?:\.gz)?$", "", raw, flags=re.IGNORECASE)
        raw = re.sub(r"\.nii(?:\.gz)?$", "", raw, flags=re.IGNORECASE)
        candidates = {raw.lower(), re.sub(r"[^a-z0-9]", "", raw.lower())}
        match = re.fullmatch(r"(?i)(?:pants[_-]?)?(\d+)", raw)
        if match:
            number = int(match.group(1))
            candidates.update({str(number), f"pants{number}", f"pants{number:08d}"})
        matches = {self._aliases[candidate] for candidate in candidates if candidate in self._aliases}
        if len(matches) != 1:
            raise UnknownCaseError(f"Unknown or ambiguous case: {value}")
        return matches.pop()

    def case(self, artifact: str, case_id: str) -> dict[str, Any] | None:
        value = self.case_maps.get(artifact, {}).get(case_id)
        return value if isinstance(value, dict) else None

    def pointer(self, artifact: str, case_id: str) -> str:
        return self.case_pointers.get(artifact, {}).get(case_id, "/data")

    def citation(self, artifact: str, pointer: str) -> str:
        return f"{ADMIN_ARTIFACTS[artifact][0]}#{pointer}"

    def _set_aliases(self, case_ids: set[str]) -> None:
        aliases: dict[str, str] = {}
        collisions: set[str] = set()
        for case_id in sorted(case_ids):
            for alias in _case_aliases(case_id):
                if alias in aliases and aliases[alias] != case_id:
                    collisions.add(alias)
                else:
                    aliases[alias] = case_id
        for alias in collisions:
            aliases.pop(alias, None)
        self._aliases = aliases

    def _load_reviewer_view(self) -> None:
        assert self.package_dir is not None
        allowed_names = {"review_manifest.csv", "REVIEW_PROTOCOL.md"}
        unexpected = sorted(path.name for path in self.package_dir.iterdir() if path.name not in allowed_names)
        if unexpected:
            raise RunArtifactError(f"Reviewer package contains unauthorized entries: {unexpected}")
        manifest_path = self.package_dir / "review_manifest.csv"
        if not manifest_path.is_file():
            raise RunArtifactError("Reviewer mode requires review_manifest.csv in --review-package")
        rows = _load_csv(manifest_path)
        indexed = _unique_rows(rows, manifest_path)
        resources: dict[str, ReviewerResource] = {}
        for case_id, (row_number, row) in indexed.items():
            image_path = str(row.get("image_path") or "").strip()
            try:
                masks = json.loads(row.get("mask_paths") or "{}")
            except json.JSONDecodeError as exc:
                raise RunArtifactError(
                    f"review_manifest.csv row {row_number} has invalid mask_paths JSON"
                ) from exc
            if not image_path or not isinstance(masks, dict) or not all(
                isinstance(key, str) and isinstance(value, str) for key, value in masks.items()
            ):
                raise RunArtifactError(f"review_manifest.csv row {row_number} has invalid resources")
            resources[case_id] = ReviewerResource(
                review_id=str(row.get("review_id") or ""),
                case_id=case_id,
                image_path=image_path,
                mask_paths=masks,
                manifest_row=row_number,
            )
        self.reviewer_resources = resources
        self._set_aliases(set(resources))
        protocol_path = self.package_dir / "REVIEW_PROTOCOL.md"
        if protocol_path.is_file():
            self.protocol = protocol_path.read_text(encoding="utf-8")
        package_match = re.search(r"(?im)^Package ID:\s*`([0-9a-f]+)`\s*$", self.protocol or "")
        self.package_id = package_match.group(1) if package_match else hash_json_payload({
            "manifest_sha256": hash_file(manifest_path),
            "protocol_sha256": hash_file(protocol_path),
        })

    def _load_admin_view(self) -> None:
        assert self.run_dir is not None
        for name, (filename, artifact_type) in ADMIN_ARTIFACTS.items():
            path = self.run_dir / filename
            if not path.is_file():
                raise RunArtifactError(f"Completed run artifact is missing: {filename}")
            try:
                raw = read_json(path)
                schema = SCHEMA_BY_ARTIFACT.get(name)
                if schema is not None:
                    validate_artifact(raw, schema)
                metadata, data = load_artifact(path)
            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
                ArtifactValidationError,
                JsonSchemaUnavailableError,
            ) as exc:
                raise RunArtifactError(f"Cannot load {filename}: {exc}") from exc
            if not isinstance(metadata, dict) or not isinstance(data, dict):
                raise RunArtifactError(f"{filename} is not a validated artifact envelope")
            if metadata.get("artifact_type") != artifact_type:
                raise RunArtifactError(
                    f"{filename} artifact_type mismatch: expected {artifact_type!r}, "
                    f"got {metadata.get('artifact_type')!r}"
                )
            self.metadata[name] = metadata
            self.artifacts[name] = data
            self.artifact_paths[name] = path

        baseline = self.metadata["validated_context"]
        identity = {field: baseline.get(field) for field in _IDENTITY_FIELDS}
        if not all(isinstance(value, str) and value for value in identity.values()):
            raise RunArtifactError("validated_run_context.json has incomplete run/dataset/task metadata")
        for name, metadata in self.metadata.items():
            for field, expected in identity.items():
                if metadata.get(field) != expected:
                    raise RunArtifactError(
                        f"Metadata mismatch in {ADMIN_ARTIFACTS[name][0]}: "
                        f"{field}={metadata.get(field)!r}, expected {expected!r}"
                    )
        context_data = self.artifacts["validated_context"]
        for field, expected in identity.items():
            if context_data.get(field) != expected:
                raise RunArtifactError(
                    f"validated_run_context.json data/metadata mismatch for {field}"
                )
        allowed = self.allowed_task_modes or {"pancreas_only"}
        if identity["task_mode"] not in allowed:
            if self.allowed_task_modes is None:
                raise RunArtifactError(
                    "Admin interactive review currently supports pancreas_only runs only"
                )
            raise RunArtifactError(
                "Admin artifact index does not support this task mode for the requested consumer"
            )

        raw_case_ids = context_data.get("case_ids")
        if not isinstance(raw_case_ids, list) or not raw_case_ids or not all(
            isinstance(case_id, str) and case_id for case_id in raw_case_ids
        ):
            raise RunArtifactError("validated_run_context.json has no valid case_ids")
        reference = set(raw_case_ids)
        if len(reference) != len(raw_case_ids):
            raise RunArtifactError("validated_run_context.json contains duplicate case IDs")
        self.case_maps["validated_context"] = {case_id: {} for case_id in raw_case_ids}
        self.case_pointers["validated_context"] = {
            case_id: json_pointer("data", "case_ids", index)
            for index, case_id in enumerate(raw_case_ids)
        }

        for name, key in CASE_MAP_KEYS.items():
            values = self.artifacts[name].get(key)
            if not isinstance(values, dict):
                raise RunArtifactError(f"{ADMIN_ARTIFACTS[name][0]} has no data/{key} mapping")
            case_ids = set(values)
            if case_ids != reference:
                missing = sorted(reference - case_ids)[:5]
                extra = sorted(case_ids - reference)[:5]
                raise RunArtifactError(
                    f"Case alignment mismatch in {ADMIN_ARTIFACTS[name][0]}: "
                    f"missing={missing}, extra={extra}"
                )
            self.case_maps[name] = values
            self.case_pointers[name] = {
                case_id: json_pointer("data", key, case_id) for case_id in values
            }

        comparison_rows = self.artifacts["comparison"].get("all_cases")
        if not isinstance(comparison_rows, list):
            raise RunArtifactError("qc_comparison.json has no data/all_cases list")
        comparison: dict[str, dict[str, Any]] = {}
        comparison_pointers: dict[str, str] = {}
        for index, row in enumerate(comparison_rows):
            if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
                raise RunArtifactError(f"qc_comparison.json has invalid row {index}")
            case_id = row["case_id"]
            if case_id in comparison:
                raise RunArtifactError(f"qc_comparison.json contains duplicate case {case_id}")
            comparison[case_id] = row
            comparison_pointers[case_id] = json_pointer("data", "all_cases", index)
        if set(comparison) != reference:
            raise RunArtifactError("Case alignment mismatch in qc_comparison.json")
        self.case_maps["comparison"] = comparison
        self.case_pointers["comparison"] = comparison_pointers
        self._verify_input_artifact_hashes()
        self._set_aliases(reference)
        self._load_admin_golden(identity, reference)
        if identity["task_mode"] == "pancreas_only" and self.verify_pancreas_policy:
            self._load_policy()

    def _verify_input_artifact_hashes(self) -> None:
        by_type = {
            metadata.get("artifact_type"): name
            for name, metadata in self.metadata.items()
            if isinstance(metadata.get("artifact_type"), str)
        }
        by_filename = {path.name: name for name, path in self.artifact_paths.items()}
        for owner, metadata in self.metadata.items():
            descriptors = metadata.get("input_artifacts") or []
            if not isinstance(descriptors, list):
                raise RunArtifactError(f"{ADMIN_ARTIFACTS[owner][0]} input_artifacts is not a list")
            for descriptor in descriptors:
                if not isinstance(descriptor, dict):
                    raise RunArtifactError(f"{ADMIN_ARTIFACTS[owner][0]} has an invalid input descriptor")
                target = by_type.get(descriptor.get("artifact_type"))
                raw_path = descriptor.get("path")
                if target is None and isinstance(raw_path, str):
                    target = by_filename.get(Path(raw_path).name)
                if target is None:
                    continue
                expected = descriptor.get("sha256")
                if not isinstance(expected, str) or not expected:
                    raise RunArtifactError(
                        f"{ADMIN_ARTIFACTS[owner][0]} lacks a hash for loaded input {ADMIN_ARTIFACTS[target][0]}"
                    )
                if hash_file(self.artifact_paths[target]) != expected:
                    raise RunArtifactError(
                        f"Input-artifact hash mismatch: {ADMIN_ARTIFACTS[owner][0]} -> {ADMIN_ARTIFACTS[target][0]}"
                    )

    def _load_admin_golden(self, identity: dict[str, str], reference: set[str]) -> None:
        assert self.run_dir is not None
        package = self.run_dir / "golden_review_package"
        manifest_path = package / "reviewer_package" / "review_manifest.csv"
        reference_path = package / "system_reference.csv"
        if manifest_path.is_file():
            self.manifest = _unique_rows(_load_csv(manifest_path), manifest_path)
            extra = set(self.manifest) - reference
            if extra:
                raise RunArtifactError(f"review_manifest.csv contains unknown cases: {sorted(extra)[:5]}")
        if reference_path.is_file():
            self.system_reference = _unique_rows(_load_csv(reference_path), reference_path)
            if not self.manifest:
                raise RunArtifactError("system_reference.csv exists without a sanitized reviewer manifest")
            if self.manifest and set(self.system_reference) != set(self.manifest):
                raise RunArtifactError("Golden manifest and system reference case sets differ")
            for case_id, (row_number, row) in self.system_reference.items():
                if case_id not in reference:
                    raise RunArtifactError(f"system_reference.csv row {row_number} has unknown case")
                if row.get("review_id") != self.manifest[case_id][1].get("review_id"):
                    raise RunArtifactError(f"system_reference.csv row {row_number} review_id mismatch")
                for csv_field, identity_field in (
                    ("source_run_id", "run_id"),
                    ("dataset_name", "dataset_name"),
                    ("task_mode", "task_mode"),
                ):
                    if row.get(csv_field) != identity[identity_field]:
                        raise RunArtifactError(
                            f"system_reference.csv row {row_number} metadata mismatch for {csv_field}"
                        )
            self._verify_golden_provenance()

    def _verify_golden_provenance(self) -> None:
        rows = [row for _, row in self.system_reference.values()]
        fields = {
            "package_id",
            "selection_seed",
            "final_decisions_sha256",
            "dataset_validation_sha256",
            "deterministic_evidence_sha256",
        }
        values: dict[str, str] = {}
        for field in fields:
            distinct = {str(row.get(field) or "") for row in rows}
            if len(distinct) != 1 or not next(iter(distinct), ""):
                raise RunArtifactError(f"system_reference.csv has inconsistent {field}")
            values[field] = next(iter(distinct))
        actual_hashes = {
            "final": hash_file(self.artifact_paths["final"]),
            "validation": hash_file(self.artifact_paths["dataset_validation"]),
            "deterministic": hash_file(self.artifact_paths["deterministic"]),
        }
        stored_hashes = {
            "final": values["final_decisions_sha256"],
            "validation": values["dataset_validation_sha256"],
            "deterministic": values["deterministic_evidence_sha256"],
        }
        if actual_hashes != stored_hashes:
            raise RunArtifactError("Golden system reference source hashes do not match loaded artifacts")
        try:
            seed = int(values["selection_seed"])
        except ValueError as exc:
            raise RunArtifactError("Golden selection seed is not an integer") from exc
        expected_package_id = hash_json_payload({
            "seed": seed,
            "inputs": stored_hashes,
            "case_ids": sorted(self.system_reference),
        })
        if values["package_id"] != expected_package_id:
            raise RunArtifactError("Golden package identity does not match source hashes and cases")
        self.package_id = expected_package_id

    def _load_policy(self) -> None:
        final_metadata = self.metadata["final"]
        descriptor = (final_metadata.get("configuration") or {}).get("decision_policy") or {}
        configured_path = descriptor.get("path")
        if not isinstance(configured_path, str):
            raise RunArtifactError("Final decisions do not identify the executable policy")
        policy_path = Path(configured_path).expanduser().resolve()
        assert self.run_dir is not None
        if policy_path != (self.run_dir / "decision_policy.yaml").resolve() or not policy_path.is_file():
            raise RunArtifactError("Final decision policy is not the immutable run snapshot decision_policy.yaml")
        policy_resources = [
            resource for resource in final_metadata.get("input_resources", [])
            if isinstance(resource, dict) and resource.get("resource_type") == "decision_policy"
        ]
        if len(policy_resources) != 1 or not isinstance(policy_resources[0].get("sha256"), str):
            raise RunArtifactError("Final decisions lack immutable decision-policy provenance")
        if hash_file(policy_path) != policy_resources[0]["sha256"]:
            raise RunArtifactError("Executable decision policy hash differs from final-decision provenance")
        try:
            raw = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise RunArtifactError(f"Cannot load decision policy: {exc}") from exc
        policy = raw.get("policy") if isinstance(raw, dict) else None
        if not isinstance(policy, dict) or not isinstance(policy.get("rules"), list):
            raise RunArtifactError("Decision policy has no rules")
        final_data = self.artifacts["final"]
        if policy.get("id") != final_data.get("policy_id") or str(policy.get("version")) != str(
            final_data.get("policy_version")
        ):
            raise RunArtifactError("Final decisions and executable policy identity differ")
        rules: dict[str, tuple[int, dict[str, Any]]] = {}
        for index, rule in enumerate(policy["rules"]):
            if not isinstance(rule, dict) or not isinstance(rule.get("id"), str):
                raise RunArtifactError(f"Decision policy has invalid rule at index {index}")
            rules[rule["id"].upper()] = (index, rule)
        self.policy = policy
        self.policy_path = policy_path
        self.policy_rules = rules
        for case_id, decision in self.case_maps["final"].items():
            if not isinstance(decision, dict):
                raise RunArtifactError(f"Final decision for {case_id} is not a mapping")
            trace = decision.get("decision_trace") or {}
            matched = trace.get("matched_rule")
            if not isinstance(matched, dict) or not isinstance(matched.get("id"), str):
                raise RunArtifactError(f"Final decision for {case_id} has no matched YAML rule")
            rule_id = matched["id"].upper()
            indexed = rules.get(rule_id)
            if indexed is None:
                raise RunArtifactError(f"Final decision for {case_id} references unknown rule {rule_id}")
            rule = indexed[1]
            expected_trace = {
                "id": rule.get("id"),
                "name": rule.get("name"),
                "rationale": rule.get("rationale"),
            }
            if matched != expected_trace:
                raise RunArtifactError(f"Final decision trace for {case_id} differs from YAML rule {rule_id}")
            if decision.get("final_decision") != rule.get("action"):
                raise RunArtifactError(f"Final action for {case_id} differs from YAML rule {rule_id}")
            if decision.get("policy_rules_triggered") != [rule_id]:
                raise RunArtifactError(f"Final decision for {case_id} has an ambiguous policy-rule trace")
            if trace.get("policy_id") != policy.get("id") or str(trace.get("policy_version")) != str(
                policy.get("version")
            ):
                raise RunArtifactError(f"Final decision policy trace for {case_id} is misaligned")
        from agents.final_decision_agent import decide

        recomputed = decide(
            self.case_maps["deterministic"],
            self.case_maps["calibrated"],
            self.artifacts["routing"],
            self.case_maps["dataset_validation"],
            comparison=self.artifacts["comparison"],
            task_mode="pancreas_only",
            policy_path=policy_path,
        )
        if recomputed != self.artifacts["final"]:
            raise RunArtifactError("Stored final decisions differ from recomputed deterministic policy composition")
