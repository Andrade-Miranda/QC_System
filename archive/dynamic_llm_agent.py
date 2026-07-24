#!/usr/bin/env python3
"""
Dynamic LLM Agent
==================
Optional LLM-backed reasoning layer on top of the rule-based QCAgent.

When an LLM provider is configured (see configs/llm_config.yaml), this agent
can answer free-text questions about individual cases, explain QC decisions,
and suggest threshold adjustments.

When no provider is configured (provider: none), the agent falls back to the
rule-based GlossaryManager / QCAgent responses.

The active dataset is determined by RAW_DATASET_ROOT in configs/paths.yaml.

Usage:
    python agents/dynamic_llm_agent.py [--paths-yaml configs/paths.yaml]
    python agents/dynamic_llm_agent.py --no-interactive --query "why is <case_id> high-risk?"
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Bootstrap: make agents/ importable when this script is run directly
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from agents.utils.paths import (
    resolve_project_paths,
    ensure_output_directories,
    build_output_dirs,
    setup_file_logging,
    ProjectPaths,
    VALID_TASK_MODES,
)

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

logger = logging.getLogger("dynamic_llm_agent")


# =============================================================================
# LLM CONFIG LOADER
# =============================================================================

def load_llm_config(paths: ProjectPaths) -> dict[str, Any]:
    """Load configs/llm_config.yaml. Returns empty dict on failure."""
    cfg_path = paths.llm_config
    if not cfg_path.exists() or not _HAS_YAML:
        return {}
    with open(cfg_path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# =============================================================================
# LLM BACKEND ABSTRACTION
# =============================================================================

class LLMBackend:
    """
    Minimal abstraction over LLM providers.
    Extend this class with additional providers as needed.
    """

    def __init__(self, cfg: dict[str, Any]):
        self.provider = cfg.get("provider", "none").lower()
        self.cfg = cfg
        self._client = None
        if self.provider not in ("none", ""):
            self._init_client()

    def _init_client(self) -> None:
        if self.provider == "openai":
            try:
                import openai, os
                api_key = os.environ.get(
                    self.cfg.get("openai", {}).get("api_key_env", "OPENAI_API_KEY"), "")
                self._client = openai.OpenAI(api_key=api_key)
            except ImportError:
                logger.warning("openai package not installed — LLM disabled.")
                self.provider = "none"
        elif self.provider == "ollama":
            try:
                import requests
                self._client = requests
            except ImportError:
                logger.warning("requests package not installed — LLM disabled.")
                self.provider = "none"

    def chat(self, system_prompt: str, user_message: str) -> str:
        """Send a message and return the assistant reply as a string."""
        if self.provider == "none" or self._client is None:
            return "[LLM disabled — set provider in configs/llm_config.yaml]"

        if self.provider == "openai":
            return self._chat_openai(system_prompt, user_message)
        if self.provider == "ollama":
            return self._chat_ollama(system_prompt, user_message)
        return "[Unknown provider]"

    def _chat_openai(self, system_prompt: str, user_message: str) -> str:
        oc = self.cfg.get("openai", {})
        resp = self._client.chat.completions.create(
            model       = oc.get("model", "gpt-4o"),
            temperature = oc.get("temperature", 0.2),
            max_tokens  = oc.get("max_tokens", 1024),
            messages    = [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        )
        return resp.choices[0].message.content.strip()

    def _chat_ollama(self, system_prompt: str, user_message: str) -> str:
        import json as _json
        oc = self.cfg.get("ollama", {})
        url = oc.get("base_url", "http://localhost:11434") + "/api/chat"
        payload = {
            "model":    oc.get("model", "llama3"),
            "stream":   False,
            "options":  {"temperature": oc.get("temperature", 0.2)},
            "messages": [
                {"role": "system",  "content": system_prompt},
                {"role": "user",    "content": user_message},
            ],
        }
        resp = self._client.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()

    @property
    def available(self) -> bool:
        return self.provider not in ("none", "")


# =============================================================================
# DYNAMIC LLM AGENT
# =============================================================================

_SYSTEM_PROMPT = """You are an expert medical imaging QC analyst for a
pancreas lesion segmentation dataset. You help researchers understand QC results,
diagnose data quality issues, and decide which cases to keep, review, or exclude.

You have access to per-case QC statistics (scores, risk levels, flags, geometry
checks) and the dataset's curation thresholds. Always be concise and evidence-based.
When a case has a lesion affine mismatch, note that the mask origin or direction
disagrees with the CT image header, which may indicate a preprocessing error.

You can reason about CT attenuation using the hu_statistics block from each case:
  - tumor_median_hu, pancreas_median_hu: robust central tendencies (preferred over mean)
  - delta_hu_tumor_vs_pancreas: tumor_median − pancreas_median
  - z_score_hu: delta normalized by pancreas_std (primary classification signal)
  - tumor_attenuation: primary label (Hypo / Iso / Hyper / Unknown)
  - attenuation_rule_used: "z_score" when z is valid, "delta_hu_fallback" otherwise
  - delta_hu_attenuation: label from delta_hu alone (for cross-checking)
  - attenuation_rule_disagreement: true when z_score and delta labels conflict
  - suspicious_hu_distribution: true when stats are unreliable (few voxels / bimodal)

Clinical interpretation guidelines:
  - Hypo (z < −2 or delta < −10 HU): consistent with PDAC or cystic lesion
    in venous/pancreatic phase. Most common pattern.
  - Hyper (z > +2 or delta > +10 HU): may indicate neuroendocrine tumor (NET,
    especially arterial phase) or incorrect annotation.
  - Iso (−2 ≤ z ≤ +2 or |delta| ≤ 10 HU): plausible; consider indirect signs
    for PDAC. Could indicate poor phase or subtraction artifact.
  - rule_disagreement = true warrants human review of the distribution.
"""


class DynamicLLMAgent:
    """
    Wraps QCAgent with an optional LLM reasoning layer.

    Interaction logging: when log_interactions=true in llm_config.yaml, every
    query/response pair is appended to logs/<interaction_log_filename>.
    """

    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self._dataset_name = paths.dataset_name or "dataset"
        llm_cfg = load_llm_config(paths)
        self.llm = LLMBackend(llm_cfg)
        self._log_interactions = llm_cfg.get("log_interactions", True)
        self._interaction_log  = (
            paths.logs_dir / llm_cfg.get("interaction_log_filename",
                                          "dynamic_agent_interactions.jsonl")
        )
        self.qc_agent  = None
        self.qc_results: dict = {}
        self.summary_data: dict = {}

        # Import QCAgent lazily to avoid circular import
        try:
            from agents.qc_agent import QCAgent
            self._QCAgent = QCAgent
        except ImportError:
            self._QCAgent = None

    def load_data(self, summary_json: Path | None = None) -> None:
        """Load summary JSON and run QC analysis."""
        json_path = summary_json or self.paths.summary_json
        if not json_path.exists():
            raise FileNotFoundError(f"Summary JSON not found: {json_path}")
        with open(json_path, encoding="utf-8") as fh:
            self.summary_data = json.load(fh)
        if self._QCAgent is not None:
            self.qc_agent   = self._QCAgent(self.summary_data,
                                             report_dir=self.paths.qc_dir)
            self.qc_results = self.qc_agent.qc_results
        logger.info("Loaded %d cases from %s", len(self.summary_data), json_path)

    def query(self, user_message: str) -> str:
        """
        Answer a free-text query.
        Falls back to rule-based QCAgent if LLM is unavailable.
        """
        if self.llm.available:
            context = self._build_context(user_message)
            full_prompt = f"{context}\n\nUser question: {user_message}"
            response = self.llm.chat(_SYSTEM_PROMPT, full_prompt)
        else:
            # Fallback: delegate to QCAgent interactive responder
            if self.qc_agent is not None:
                response = self.qc_agent.respond(user_message)
            else:
                response = (
                    "LLM not configured and QCAgent not loaded. "
                    "Run load_data() first or configure an LLM provider in "
                    "configs/llm_config.yaml."
                )

        if self._log_interactions:
            self._log(user_message, response)
        return response

    def _build_context(self, query: str) -> str:
        """Build a short dataset context string to prepend to the LLM prompt."""
        lines = [f"Dataset: {getattr(self, '_dataset_name', 'dataset')} ({len(self.summary_data)} cases)"]
        if self.qc_results:
            from collections import Counter
            risk_counts = Counter(v.get("risk_level", "?")
                                  for v in self.qc_results.values())
            lines.append(
                f"Risk distribution: high={risk_counts.get('high', 0)}, "
                f"medium={risk_counts.get('medium', 0)}, "
                f"low={risk_counts.get('low', 0)}"
            )

            # Attenuation distribution (add when query is attenuation-related
            # or as a brief summary for general queries)
            q_lower = query.lower()
            att_relevant = any(w in q_lower for w in (
                "attenuat", "hu", "hypo", "hyper", "iso", "hounsfield",
                "density", "pdac", "net", "neuroendocrine",
            ))
            att_counts: Counter = Counter()
            for raw_v in self.summary_data.values():
                att = (raw_v.get("hu_statistics") or {}).get("tumor_attenuation")
                if att:
                    att_counts[att] += 1
            if att_counts:
                att_summary = (f"Attenuation distribution — "
                               f"Hypo: {att_counts.get('Hypo', 0)}, "
                               f"Iso: {att_counts.get('Iso', 0)}, "
                               f"Hyper: {att_counts.get('Hyper', 0)}, "
                               f"Unknown: {att_counts.get('Unknown', 0)}")
                if att_relevant:
                    lines.append(att_summary)

        # If query names a specific case, attach its QC entry and HU stats
        import re
        m = re.search(r"PanTS_\d{8}", query)
        if m:
            cid = m.group()
            qc  = self.qc_results.get(cid)
            raw = self.summary_data.get(cid)
            if qc:
                lines.append(f"\nQC result for {cid}:")
                lines.append(json.dumps(qc, indent=2)[:800])
            if raw and "geometry" in raw:
                lines.append(f"\nGeometry for {cid}:")
                lines.append(json.dumps(raw["geometry"]["summary"], indent=2))
            if raw:
                hu = raw.get("hu_statistics")
                if hu and hu.get("tumor_median_hu") is not None:
                    lines.append(f"\nHU statistics for {cid}:")
                    lines.append(json.dumps(hu, indent=2))
        return "\n".join(lines)

    def _log(self, query: str, response: str) -> None:
        self._interaction_log.parent.mkdir(parents=True, exist_ok=True)
        import datetime
        record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "query":     query,
            "response":  response,
            "provider":  self.llm.provider,
        }
        with open(self._interaction_log, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run_interactive(self) -> None:
        """Simple REPL loop."""
        print("\n" + "=" * 60)
        print("  Dynamic LLM Agent")
        print(f"  LLM provider : {self.llm.provider}")
        print("  Type 'exit' or Ctrl-C to quit.")
        print("=" * 60 + "\n")
        try:
            while True:
                try:
                    user_in = input("You> ").strip()
                except EOFError:
                    break
                if user_in.lower() in ("exit", "quit", "q"):
                    break
                if not user_in:
                    continue
                response = self.query(user_in)
                print(f"\nAgent> {response}\n")
        except KeyboardInterrupt:
            pass
        print("\nGoodbye.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dynamic LLM Agent — LLM-backed QC reasoning.")
    parser.add_argument("--paths-yaml",     default=None,
                        help="Path to paths.yaml (default: configs/paths.yaml)")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Do not start interactive loop")
    parser.add_argument("--query",          default=None,
                        help="Single query to answer and exit (implies --no-interactive)")
    parser.add_argument("--task-mode",      default=None, dest="task_mode",
                        choices=sorted(VALID_TASK_MODES),
                        help="Task mode (default: read from thresholds.yaml). "
                             + " | ".join(sorted(VALID_TASK_MODES)))
    args = parser.parse_args()

    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    ensure_output_directories(paths)  # creates shared logs_dir only

    # Resolve task_mode: CLI > thresholds.yaml > default
    _thr_task_mode = "pancreas_lesion_subregions"
    _thr_path = paths.thresholds_config
    if _thr_path.exists() and _HAS_YAML:
        try:
            _thr_raw = yaml.safe_load(_thr_path.read_text(encoding="utf-8")) or {}
            _v = str(_thr_raw.get("TASK_MODE", "")).strip().lower()
            if _v and _v != "auto" and _v in VALID_TASK_MODES:
                _thr_task_mode = _v
        except Exception:
            pass
    _task_mode = args.task_mode if args.task_mode else _thr_task_mode

    # Build task-mode-specific dirs
    _base_output_dir = paths.summary_dir.parent  # outputs/<dataset_name>/
    _task_dirs = build_output_dirs(_base_output_dir, _task_mode)

    # Resolve task-mode-specific paths
    paths.summary_json = _task_dirs["summary_dir"] / paths.summary_json.name
    paths.qc_dir       = _task_dirs["qc_dir"]
    lg = setup_file_logging(paths.logs_dir / "dynamic_llm_agent.log",
                             logger_name="dynamic_llm_agent")

    agent = DynamicLLMAgent(paths)
    lg.info("Loading data from %s", paths.summary_json)
    agent.load_data()

    if args.query:
        print(agent.query(args.query))
    elif not args.no_interactive:
        agent.run_interactive()


if __name__ == "__main__":
    main()
