#!/usr/bin/env python3
"""CLI wrapper for optional nonbinding AgentQC LLM explanations."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm_explanation_agent import main


if __name__ == "__main__":
    raise SystemExit(main())
