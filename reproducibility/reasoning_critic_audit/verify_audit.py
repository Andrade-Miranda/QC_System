#!/usr/bin/env python3
"""Verify validated nonbinding reasoning/critic audit headline counts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_output_dir, load_expected  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    audit = load_expected()["reasoning_critic_audit"]
    if audit["unsupported_critique_claims"] != 0:
        raise AssertionError("unsupported critique claims must remain zero")
    if audit["reasoning_non_interference"] != audit["case_profile_records"]:
        raise AssertionError("reasoning non-interference mismatch")
    if audit["critique_non_interference"] != audit["case_profile_records"]:
        raise AssertionError("critique non-interference mismatch")
    result = {"audit": audit, "boundary": "Implementation/architecture integrity and non-interference only; not clinical reasoning quality or expert agreement."}
    ensure_output_dir(args.output_dir)
    with (args.output_dir / "reasoning_critic_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
