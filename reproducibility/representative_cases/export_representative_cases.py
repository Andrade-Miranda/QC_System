#!/usr/bin/env python3
"""Export validated representative evidence-to-policy traces."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ensure_output_dir, load_expected  # noqa: E402


CASES = {
    "PanTS_00007012": {
        "transition": "tau_L keep -> tau_S review",
        "task_relevant_evidence": "Region consistency becomes task-relevant; moderate warning, score 7, relative error 0.178.",
        "routing_policy_consequence": "Region-consistency moderate-warning routing changes basis from calibrated QC to review routing.",
    },
    "PanTS_00007024": {
        "transition": "tau_L warning -> tau_S review",
        "task_relevant_evidence": "Subregion/FOV evidence becomes task-relevant; FOV and region consistency moderate warning, coverage ratio 0.371, region score 6.",
        "routing_policy_consequence": "FOV and region-consistency routing determine review; metadata alone maps to warning.",
    },
    "PanTS_00007053": {
        "transition": "tau_P warning -> tau_L review",
        "task_relevant_evidence": "Lesion-capable attenuation evidence becomes task-relevant; invalid tumor HU statistics, high warning, score 10.",
        "routing_policy_consequence": "Attenuation high-warning routing changes basis from calibrated QC to review routing.",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    args = parser.parse_args()
    expected = load_expected()["representative_cases"]
    for case_id, data in expected.items():
        if CASES[case_id]["transition"] != data["transition"]:
            raise AssertionError(f"{case_id}: transition mismatch")
    ensure_output_dir(args.output_dir)
    output = {"cases": CASES, "boundary": "Representative traces are descriptive artifact-to-policy examples, not clinical-correctness labels."}
    with (args.output_dir / "representative_cases.json").open("w", encoding="utf-8") as fh:
        json.dump(output, fh, indent=2)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
