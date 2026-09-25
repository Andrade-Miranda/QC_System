#!/usr/bin/env python3
"""Generate optional nonbinding LLM explanation artifacts from validated QC runs."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.explanation_contracts import SUPPORTED_TASKS
from agents.utils.llm_backend import build_backend, load_provider_config
from agents.utils.llm_explanation_adapter import explain_case, load_explanation_index
from agents.utils.run_artifact_index import ADMIN_ARTIFACTS
from artifacts.io import artifact_descriptor, write_artifact
from artifacts.validation import ArtifactValidationError, JsonSchemaUnavailableError, validate_artifact


def build_explanation_artifact(
    run_dir: Path,
    backend,
    *,
    case_ids: list[str] | None = None,
    max_cases: int | None = None,
) -> dict[str, Any]:
    index = load_explanation_index(run_dir)
    available = sorted(index.case_maps["final"])
    selected = case_ids or available
    if max_cases is not None:
        selected = selected[:max_cases]
    resolved = [index.resolve_case(case_id) for case_id in selected]
    cases = {case_id: explain_case(index, backend, case_id) for case_id in resolved}
    task_mode = index.metadata["validated_context"]["task_mode"]
    data = {
        "status": "complete" if all(case["validation_result"]["usable_for_reporting"] for case in cases.values()) else "partial",
        "adapter": {
            "name": "Evidence Abstraction Adapter",
            "notation": "A_tau",
            "deterministic": True,
            "learned": False,
            "extracts_new_evidence": False,
        },
        "validator": {"name": "strict output-grounding validator", "notation": "C_tau"},
        "backend": {
            "provider": backend.config.provider,
            "model": backend.config.model,
            "temperature": backend.config.temperature,
            "top_p": backend.config.top_p,
            "seed": backend.config.seed,
            "context_length": backend.config.context_length,
            "max_tokens": backend.config.max_tokens,
        },
        "task": {
            "scientific_task_name": SUPPORTED_TASKS[task_mode]["scientific_task_name"],
            "task_notation": SUPPORTED_TASKS[task_mode]["task_notation"],
            "implementation_task_id": task_mode,
        },
        "authority_boundary": {
            "deterministic_policy_symbol": "pi_tau",
            "final_policy_action_source": "final_qc_decisions.json",
            "adapter_may_change_policy": False,
            "adapter_may_change_routing": False,
            "adapter_may_change_scores": False,
            "adapter_may_extract_new_evidence": False,
            "llm_output_is_nonbinding": True,
        },
        "alignment": {"case_count": len(cases), "requested_case_count": len(selected)},
        "cases": cases,
        "limitations": [
            "Optional explanatory artifact only; not consumed by routing or final policy.",
            "Visualization status is unknown unless a canonical validated visualization artifact is present.",
        ],
    }
    return write_artifact(
        Path(run_dir) / "llm_explanation_artifact.json",
        artifact_type="llm_explanation_artifact",
        generator="LLMExplanationAgent",
        data=data,
        dataset_name=index.metadata["validated_context"]["dataset_name"],
        task_mode=task_mode,
        input_artifacts=[
            artifact_descriptor(path, ADMIN_ARTIFACTS[name][1])
            for name, path in sorted(index.artifact_paths.items())
        ],
        configuration={"case_ids": resolved, "backend_provider": backend.config.provider},
        run_id=index.metadata["validated_context"].get("run_id"),
        project_root=_PROJECT_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate nonbinding AgentQC LLM explanations from compact evidence.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument("--max-cases", type=int)
    parser.add_argument("--llm-config", type=Path, default=_PROJECT_ROOT / "configs" / "llm_config.yaml")
    parser.add_argument("--provider", choices=("none", "ollama", "openai"))
    parser.add_argument("--allow-external-provider", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    config = load_provider_config(args.llm_config, provider_override=args.provider)
    config = dataclasses.replace(config, extended_ollama_options=True)
    backend = build_backend(config, allow_external_provider=args.allow_external_provider)
    artifact = build_explanation_artifact(args.run_dir, backend, case_ids=args.cases, max_cases=args.max_cases)
    output = args.output or (args.run_dir / "llm_explanation_artifact.json")
    if output != args.run_dir / "llm_explanation_artifact.json":
        from artifacts.io import write_json
        write_json(output, artifact)
    try:
        validate_artifact(artifact, "llm_explanation_artifact")
    except JsonSchemaUnavailableError as exc:
        print(f"warning: explanation artifact schema validation not completed: {exc}", file=sys.stderr)
    except ArtifactValidationError as exc:
        print(f"error: explanation artifact schema validation failed: {exc}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
