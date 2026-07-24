#!/usr/bin/env python3
"""Primary QC runner for the artifact-based deterministic workflow."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    cmd = [sys.executable, str(_ROOT / "agents" / "orchestrator_agent.py"), *sys.argv[1:]]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
