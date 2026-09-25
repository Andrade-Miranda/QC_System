#!/usr/bin/env python3
"""Run public AgentQC paper/poster reproduction checks."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print("+", " ".join(args))
    subprocess.run([sys.executable, *args], cwd=ROOT.parent, check=True)


def main() -> None:
    run(["reproducibility/decision_stability/analyze_decision_stability.py"])
    run(["reproducibility/evidence_domains/analyze_evidence_domains.py"])
    run(["reproducibility/representative_cases/export_representative_cases.py"])
    run(["reproducibility/reasoning_critic_audit/verify_audit.py"])
    run(["reproducibility/figures/poster/generate_poster_figures.py"])
    print("All public reproduction checks completed.")


if __name__ == "__main__":
    main()
