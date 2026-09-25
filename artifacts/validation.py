"""Lightweight validation helpers for versioned AgentQC JSON artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


SCHEMA_DIRECTORY = Path(__file__).with_name("schemas")


class JsonSchemaUnavailableError(RuntimeError):
    """Raised when artifact validation is requested without jsonschema."""


class ArtifactValidationError(ValueError):
    """Raised when an artifact does not satisfy its JSON Schema contract."""


def _jsonschema():
    try:
        import jsonschema
    except ImportError as exc:
        raise JsonSchemaUnavailableError(
            "Artifact validation requires the optional 'jsonschema' package; "
            "install it with `python -m pip install jsonschema`."
        ) from exc
    return jsonschema


def schema_path(schema: str | Path) -> Path:
    """Resolve a schema name, filename, or explicit path."""
    candidate = Path(schema)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate
    name = candidate.name
    if not name.endswith(".schema.json"):
        name = f"{name.removesuffix('.json')}.schema.json"
    return SCHEMA_DIRECTORY / name


def load_schema(schema: str | Path) -> dict[str, Any]:
    """Load a local JSON Schema and fail clearly for unknown contracts."""
    path = schema_path(schema)
    if not path.is_file():
        raise FileNotFoundError(f"Artifact schema not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Artifact schema must be a JSON object: {path}")
    return value


def validate_artifact(instance: Any, schema: str | Path) -> None:
    """Validate an artifact and raise one readable error containing all issues."""
    jsonschema = _jsonschema()
    try:
        from referencing import Registry, Resource
    except ImportError as exc:
        raise JsonSchemaUnavailableError(
            "Artifact validation requires the optional 'referencing' package installed with jsonschema."
        ) from exc
    path = schema_path(schema).resolve()
    contract = load_schema(path)
    validator_class = jsonschema.validators.validator_for(contract)
    validator_class.check_schema(contract)
    resources = []
    for schema_file in SCHEMA_DIRECTORY.glob("*.schema.json"):
        raw_schema = load_schema(schema_file)
        resource = Resource.from_contents(raw_schema)
        resources.append((schema_file.resolve().as_uri(), resource))
        resources.append((schema_file.name, resource))
    registry = Registry().with_resources(resources)
    validator = validator_class(
        contract,
        registry=registry,
        format_checker=jsonschema.FormatChecker(),
    )
    errors = sorted(validator.iter_errors(instance), key=lambda error: list(error.absolute_path))
    if not errors:
        return
    details = []
    for error in errors:
        location = ".".join(str(part) for part in error.absolute_path) or "<root>"
        details.append(f"{location}: {error.message}")
    raise ArtifactValidationError(
        f"Artifact does not satisfy {path.name}:\n- " + "\n- ".join(details)
    )
