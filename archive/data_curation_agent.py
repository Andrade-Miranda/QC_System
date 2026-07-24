#!/usr/bin/env python3
"""
Data Curation System
=====================
Two-agent pipeline for auditable data curation of medical image datasets.

Agents:
    QCAgent             -- (in qc_agent.py) quality analysis, scoring, risk levels
    DataCurationAgent   -- combines QC evidence + thresholds -> curation decisions

Supporting classes:
    ThresholdConfig     -- load/save/update user thresholds (YAML or JSON)
    RuleEngine          -- apply deterministic curation rules
    SplitManager        -- stratified train/val/test splitting
    ReportGenerator     -- write all output reports
    InteractiveConsole  -- terminal interaction loop (rich-based)

The active dataset is determined by RAW_DATASET_ROOT in configs/paths.yaml.

Usage examples:
    # Full pipeline, list-only output, interactive
    python data_curation_agent.py \\
        --summary-json <dataset>_summary.json \\
        --output-dir   curated_dataset \\
        --interactive

    # Non-interactive, generate all outputs
    python data_curation_agent.py \\
        --summary-json <dataset>_summary.json \\
        --qc-report    qc_report.json \\
        --output-dir   curated_dataset \\
        --config       thresholds.yaml \\
        --mode         list \\
        --seed         42

    # Symlink mode
    python data_curation_agent.py \\
        --summary-json <dataset>_summary.json \\
        --output-dir   curated_dataset \\
        --mode         symlink \\
        --images-dir   imagesTr \\
        --labels-dir   labelsTr
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import random
import shutil
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from textwrap import dedent
from typing import Any

# Optional imports with graceful fallback
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from sklearn.model_selection import StratifiedShuffleSplit
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    HAS_RICH = True
    _console = Console()
except ImportError:
    HAS_RICH = False
    _console = None

# Import QCAgent from sibling module
import sys

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
    VALID_TASK_MODES,
)
from agents.qc_agent import QCAgent, run_qc, SKIPPED_CASES
from artifacts.loaders import load_summary_cases

# ── Default paths (resolved lazily via paths.yaml) ────────────────────────────
def _default_paths():
    try:
        return resolve_project_paths()
    except Exception:
        return None

_DEFAULT_PATHS   = _default_paths()
DEFAULT_JSON     = (_DEFAULT_PATHS.summary_json    if _DEFAULT_PATHS else _HERE / "dataset_summary.json")
DEFAULT_QC_JSON  = (_DEFAULT_PATHS.qc_report_json  if _DEFAULT_PATHS else _HERE / "qc_report.json")
DEFAULT_OUT_DIR  = (_DEFAULT_PATHS.curation_dir    if _DEFAULT_PATHS else _HERE / "curated_dataset")
DEFAULT_CONFIG   = (_DEFAULT_PATHS.thresholds_config if _DEFAULT_PATHS else _HERE / "thresholds.yaml")

RANDOM_SEED = 42


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

class ThresholdConfig:
    """
    Loads, stores, and validates user-defined thresholds.
    Supports YAML and JSON config files.
    """

    DEFAULTS: dict[str, Any] = {
        "MIN_LESION_PANCREAS_OVERLAP" : 0.80,
        "MAX_REGION_RELATIVE_ERROR"   : 0.05,
        "MIN_LESION_VOLUME_MM3"       : 10,
        "MAX_LESION_VOLUME_MM3"       : 200_000,
        "MAX_LESION_PANCREAS_RATIO"   : 0.80,
        "MIN_PANCREAS_VOLUME_MM3"     : 20_000,
        "MAX_PANCREAS_VOLUME_MM3"     : 200_000,
        "MAX_QC_SCORE_FOR_KEEP"       : 30,
        "MAX_QC_SCORE_FOR_REVIEW"     : 60,
        # Split ratios
        "TRAIN_RATIO"                 : 0.70,
        "VAL_RATIO"                   : 0.15,
        "TEST_RATIO"                  : 0.15,
        # Behaviour flags
        "INCLUDE_REVIEW_IN_SPLITS"    : False,
        "EXCLUDE_EMPTY_PANCREAS"      : True,
        "EXCLUDE_SHAPE_MISMATCH"      : True,
        "EXCLUDE_LESION_OUTSIDE"      : True,
        # Hybrid scoring parameters (used by QCAgent when --score-version hybrid)
        "HYBRID_ALPHA"                         : 0.4,
        "HYBRID_BETA"                          : 1.5,
        "HYBRID_WEIGHT_LESION_BURDEN"          : 1.0,
        "HYBRID_WEIGHT_PANCREAS_CONTEXT"       : 0.8,
        "HYBRID_WEIGHT_REGION_CONSISTENCY"     : 0.6,
        "HYBRID_WEIGHT_METADATA"               : 0.3,
        "HYBRID_WEIGHT_ATTENUATION_CONSISTENCY": 0.4,
    }

    def __init__(self, config_path: Path | None = None):
        self.values: dict[str, Any] = dict(self.DEFAULTS)
        if config_path and config_path.exists():
            self._load(config_path)

    def _load(self, path: Path) -> None:
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml") and HAS_YAML:
            loaded = yaml.safe_load(text) or {}
        else:
            loaded = json.loads(text)
        for k, v in loaded.items():
            if k in self.DEFAULTS:
                self.values[k] = type(self.DEFAULTS[k])(v)

    def save(self, path: Path) -> None:
        if path.suffix in (".yaml", ".yml") and HAS_YAML:
            path.write_text(yaml.dump(self.values, default_flow_style=False),
                            encoding="utf-8")
        else:
            path.write_text(json.dumps(self.values, indent=2), encoding="utf-8")

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    def set(self, key: str, value: str) -> bool:
        """Try to set threshold from string value. Returns True if successful."""
        if key not in self.DEFAULTS:
            return False
        expected_type = type(self.DEFAULTS[key])
        try:
            if expected_type == bool:
                self.values[key] = value.lower() in ("true", "1", "yes")
            else:
                self.values[key] = expected_type(value)
            return True
        except (ValueError, TypeError):
            return False

    def summary(self) -> str:
        lines = ["Current thresholds:", f"  {'Parameter':<42} {'Value':>12}",
                 f"  {'─'*42} {'─'*12}"]
        for k, v in sorted(self.values.items()):
            lines.append(f"  {k:<42} {str(v):>12}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RuleEngine:
    """
    Applies deterministic curation rules combining QC evidence and thresholds.
    Returns a decision dict for each case.
    """

    def __init__(self, thr: ThresholdConfig):
        self.thr = thr

    def decide(self, case_id: str, qc: dict, summary: dict) -> dict:
        """
        qc      : result from QCAgent.get_case_qc(case_id)
        summary : entry from <dataset>_summary.json for this case

        Returns:
            {decision, reasons, training_impact, threshold_evidence, source}
        """
        score    = qc.get("qc_score") or qc.get("score") or 0
        risk     = qc.get("risk_level", "low")
        qc_rec   = qc.get("recommendation", "keep")
        conflicts_raw = qc.get("conflicts") or qc.get("flags") or []

        # Normalise conflicts to list of tag strings
        if conflicts_raw and isinstance(conflicts_raw[0], dict):
            conflict_tags = {c["tag"] for c in conflicts_raw}
        elif conflicts_raw and isinstance(conflicts_raw[0], (list, tuple)):
            conflict_tags = {c[1] for c in conflicts_raw}
        else:
            conflict_tags = set()

        evidence = qc.get("evidence") or {}
        reasons: list[str] = []
        impacts: list[str] = []
        threshold_evidence: dict[str, Any] = {}
        decision = "keep"

        pan  = summary.get("pancreas") or {}
        les  = summary.get("lesions") or {}
        qcd  = summary.get("quality_control") or {}

        pvol    = pan.get("volume_mm3") or 0.0
        total_lv = les.get("total_volume_mm3") or 0.0
        ov_rate = qcd.get("total_overlap_rate_vs_lesion")
        rel_err = (pan.get("region_consistency") or {}).get("relative_error_vs_pancreas") or 0.0

        # ── Hard exclusion rules (from QC flags) ──────────────────────────

        if "unreadable_nifti" in conflict_tags:
            decision = "exclude"
            reasons.append("Unreadable NIfTI file (non-orthonormal direction cosines)")
            impacts.append("Cannot be loaded; excluded until header is fixed")

        if self.thr.get("EXCLUDE_EMPTY_PANCREAS") and "pancreas_mask_empty" in conflict_tags:
            decision = "exclude"
            reasons.append("Pancreas mask is completely empty")
            impacts.append("Empty masks bias model toward background prediction")

        if self.thr.get("EXCLUDE_SHAPE_MISMATCH") and "shape_mismatch" in conflict_tags:
            decision = "exclude"
            reasons.append("Implausible image shape detected")
            impacts.append("Shape mismatch indicates file integrity issue")

        if "invalid_spacing" in conflict_tags:
            sev = "CRITICAL" if any(
                (isinstance(c, dict) and c.get("tag") == "invalid_spacing"
                 and c.get("severity") == "CRITICAL")
                or (isinstance(c, (list, tuple)) and len(c) >= 2
                    and c[1] == "invalid_spacing" and c[0] == "CRITICAL")
                for c in conflicts_raw
            ) else "WARNING"
            if sev == "CRITICAL":
                decision = "exclude"
                reasons.append("Invalid (zero or negative) spacing detected")
                impacts.append("Invalid spacing prevents correct mm3 computation")

        # ── Lesion overlap threshold ───────────────────────────────────────
        if total_lv > 0 and ov_rate is not None:
            min_ov = self.thr.get("MIN_LESION_PANCREAS_OVERLAP", 0.80)
            threshold_evidence["overlap_rate_vs_lesion"] = ov_rate
            threshold_evidence["MIN_LESION_PANCREAS_OVERLAP"] = min_ov
            if self.thr.get("EXCLUDE_LESION_OUTSIDE") and ov_rate < min_ov:
                if decision != "exclude":
                    decision = "exclude" if ov_rate < 0.10 else "review"
                reasons.append(
                    f"Lesion-pancreas overlap {ov_rate:.1%} < threshold {min_ov:.1%}")
                impacts.append("Incorrect lesion location priors degrade lesion segmentation")

        # ── Region relative error threshold ───────────────────────────────
        max_err = self.thr.get("MAX_REGION_RELATIVE_ERROR", 0.05)
        threshold_evidence["region_relative_error"] = rel_err
        threshold_evidence["MAX_REGION_RELATIVE_ERROR"] = max_err
        if rel_err > max_err and decision not in ("exclude",):
            decision = "review" if rel_err < 0.30 else "exclude"
            reasons.append(
                f"Region sum error {rel_err:.1%} > threshold {max_err:.1%}")
            impacts.append("Inconsistent region labels harm sub-task training")

        # ── Pancreas volume thresholds ─────────────────────────────────────
        min_pvol = self.thr.get("MIN_PANCREAS_VOLUME_MM3", 20_000)
        max_pvol = self.thr.get("MAX_PANCREAS_VOLUME_MM3", 200_000)
        threshold_evidence["pancreas_volume_mm3"] = pvol
        if 0 < pvol < min_pvol and decision not in ("exclude",):
            decision = "review"
            reasons.append(
                f"Pancreas volume {pvol:,.0f} mm3 < threshold {min_pvol:,} mm3")
            impacts.append("Small pancreas may destabilise Dice loss")
        if pvol > max_pvol and decision not in ("exclude",):
            decision = "review"
            reasons.append(
                f"Pancreas volume {pvol:,.0f} mm3 > threshold {max_pvol:,} mm3")
            impacts.append("Large pancreas may indicate annotation spill-over")

        # ── Lesion volume thresholds ───────────────────────────────────────
        if total_lv > 0:
            min_lv = self.thr.get("MIN_LESION_VOLUME_MM3", 10)
            max_lv = self.thr.get("MAX_LESION_VOLUME_MM3", 200_000)
            threshold_evidence["total_lesion_volume_mm3"] = total_lv
            if total_lv < min_lv and decision not in ("exclude",):
                decision = "review"
                reasons.append(
                    f"Lesion volume {total_lv:.1f} mm3 < threshold {min_lv} mm3")
            if total_lv > max_lv and decision not in ("exclude",):
                decision = "review"
                reasons.append(
                    f"Lesion volume {total_lv:.0f} mm3 > threshold {max_lv:,} mm3")

        # ── QC score thresholds ────────────────────────────────────────────
        max_keep   = self.thr.get("MAX_QC_SCORE_FOR_KEEP", 30)
        max_review = self.thr.get("MAX_QC_SCORE_FOR_REVIEW", 60)
        threshold_evidence["qc_score"] = score
        threshold_evidence["MAX_QC_SCORE_FOR_KEEP"] = max_keep
        threshold_evidence["MAX_QC_SCORE_FOR_REVIEW"] = max_review
        if score > max_review and decision not in ("exclude",):
            decision = "exclude"
            reasons.append(
                f"QC score {score} > MAX_QC_SCORE_FOR_REVIEW ({max_review})")
            impacts.append("High QC score indicates multiple overlapping quality issues")
        elif score > max_keep and decision not in ("exclude", "review"):
            decision = "review"
            reasons.append(
                f"QC score {score} > MAX_QC_SCORE_FOR_KEEP ({max_keep})")

        # ── High-risk QC from agent ────────────────────────────────────────
        if risk == "high" and decision not in ("exclude",):
            decision = "exclude"
            reasons.append(f"QC risk level: {risk} (score={score})")
            impacts.append("High-risk cases contain CRITICAL quality flags")
        elif risk == "medium" and decision not in ("exclude", "review"):
            decision = "review"
            reasons.append(f"QC risk level: {risk} (score={score})")

        # ── Missing metadata (keep unless other problems) ──────────────────
        if "missing_metadata" in conflict_tags and decision == "keep":
            decision = "review"
            reasons.append("Missing metadata fields (sex/age/ct_phase/manufacturer)")
            impacts.append("Missing metadata limits stratified analysis")

        # Default: no issues found
        if not reasons:
            reasons.append("All QC checks passed within thresholds")

        # Training impact summary
        if not impacts:
            impacts.append("No known training impact")

        return {
            "decision"           : decision,
            "source"             : "rule_engine",
            "qc_risk_level"      : risk,
            "qc_score"           : score,
            "conflicts"          : list(conflict_tags),
            "evidence"           : evidence,
            "threshold_evidence" : threshold_evidence,
            "reasons"            : reasons,
            "training_impact"    : "; ".join(impacts),
            "recommendation"     : qc_rec,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SPLIT MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class SplitManager:
    """
    Creates stratified (or random fallback) train/val/test splits.
    Only uses 'keep' cases unless include_review=True.
    """

    def __init__(self, thr: ThresholdConfig, seed: int = RANDOM_SEED):
        self.thr  = thr
        self.seed = seed

    def _stratification_key(self, case_id: str, summary: dict,
                             qc: dict | None = None) -> str:
        """Build a stratification label from available metadata."""
        meta = summary.get("metadata") or {}
        les  = summary.get("lesions") or {}
        pan  = summary.get("pancreas") or {}

        ct_phase = str(meta.get("ct_phase") or "unknown")[:20]
        manufact = str(meta.get("manufacturer") or "unknown")[:10]
        sex      = str(meta.get("sex") or "unknown")[:1]
        n_les    = les.get("n_lesions") or 0
        les_cat  = "0" if n_les == 0 else ("1" if n_les <= 2 else "3+")
        pvol     = pan.get("volume_mm3") or 0
        pcat     = "small" if pvol < 30_000 else ("large" if pvol > 100_000 else "normal")
        risk     = (qc.get("risk_level") or "low") if qc else "low"

        return f"{ct_phase}|{sex}|{les_cat}|{pcat}|{risk}"

    def split(self, keep_cases: list[str], summary_data: dict,
              qc_data: dict | None = None) -> dict[str, list[str]]:
        """
        Returns {train: [...], val: [...], test: [...]}.
        Falls back to random split if stratification fails.
        """
        if not keep_cases:
            return {"train": [], "val": [], "test": []}

        rng = random.Random(self.seed)
        train_r = self.thr.get("TRAIN_RATIO", 0.70)
        val_r   = self.thr.get("VAL_RATIO",   0.15)

        n       = len(keep_cases)
        n_test  = max(1, round(n * self.thr.get("TEST_RATIO", 0.15)))
        n_val   = max(1, round(n * val_r))
        n_train = n - n_val - n_test

        # Build stratification labels
        labels = [self._stratification_key(
                      c, summary_data.get(c, {}),
                      (qc_data or {}).get(c))
                  for c in keep_cases]

        label_counts = Counter(labels)
        stratify_ok = (
            HAS_SKLEARN
            and HAS_NUMPY
            and all(v >= 2 for v in label_counts.values())
            and len(label_counts) < n // 2
        )

        fallback_note = ""
        if stratify_ok:
            try:
                import numpy as np
                cases_arr  = np.array(keep_cases)
                labels_arr = np.array(labels)

                # First split off test
                sss_test = StratifiedShuffleSplit(
                    n_splits=1, test_size=n_test, random_state=self.seed)
                train_val_idx, test_idx = next(sss_test.split(cases_arr, labels_arr))

                # Then split train/val from the remainder
                sss_val = StratifiedShuffleSplit(
                    n_splits=1, test_size=n_val, random_state=self.seed + 1)
                rem_labels = labels_arr[train_val_idx]
                train_idx_local, val_idx_local = next(
                    sss_val.split(train_val_idx, rem_labels))

                train_cases = cases_arr[train_val_idx[train_idx_local]].tolist()
                val_cases   = cases_arr[train_val_idx[val_idx_local]].tolist()
                test_cases  = cases_arr[test_idx].tolist()
                method = "stratified"
            except Exception as exc:
                stratify_ok = False
                fallback_note = f"Stratification failed ({exc}), using random split."

        if not stratify_ok:
            shuffled = list(keep_cases)
            rng.shuffle(shuffled)
            test_cases  = shuffled[:n_test]
            val_cases   = shuffled[n_test:n_test + n_val]
            train_cases = shuffled[n_test + n_val:]
            method = "random"
            if not fallback_note:
                fallback_note = "Stratification skipped (sklearn/numpy unavailable or sample too small)."

        return {
            "train"        : sorted(train_cases),
            "val"          : sorted(val_cases),
            "test"         : sorted(test_cases),
            "method"       : method,
            "fallback_note": fallback_note,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════

class ReportGenerator:
    """Writes all output reports to the output directory."""

    def __init__(self, output_dir: Path):
        self.out = output_dir
        self.out.mkdir(parents=True, exist_ok=True)

    def write_case_lists(self, decisions: dict[str, dict],
                         splits: dict[str, list[str]]) -> None:
        keep    = sorted(k for k, v in decisions.items() if v["final_decision"] == "keep")
        review  = sorted(k for k, v in decisions.items() if v["final_decision"] == "review")
        exclude = sorted(k for k, v in decisions.items() if v["final_decision"] == "exclude")

        for name, cases in (("keep_cases", keep), ("review_cases", review),
                             ("excluded_cases", exclude),
                             ("train_cases", splits.get("train", [])),
                             ("val_cases",   splits.get("val", [])),
                             ("test_cases",  splits.get("test", []))):
            (self.out / f"{name}.txt").write_text(
                "\n".join(cases), encoding="utf-8")

    def write_decisions_json(self, decisions: dict[str, dict]) -> None:
        (self.out / "curation_decisions.json").write_text(
            json.dumps(decisions, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_decisions_csv(self, decisions: dict[str, dict]) -> None:
        fieldnames = [
            "case_id", "automatic_decision", "user_override", "final_decision",
            "qc_risk_level", "qc_score", "reasons", "training_impact",
            "recommendation",
        ]
        rows = []
        for cid, d in decisions.items():
            rows.append({
                "case_id"           : cid,
                "automatic_decision": d.get("automatic_decision", ""),
                "user_override"     : d.get("user_override") or "",
                "final_decision"    : d.get("final_decision", ""),
                "qc_risk_level"     : d.get("qc_risk_level", ""),
                "qc_score"          : d.get("qc_score", ""),
                "reasons"           : "; ".join(d.get("reasons") or []),
                "training_impact"   : d.get("training_impact", ""),
                "recommendation"    : d.get("recommendation", ""),
            })
        with (self.out / "curation_decisions.csv").open(
                "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def write_filtered_summary(self, summary_data: dict,
                                decisions: dict[str, dict]) -> None:
        keep_ids = {k for k, v in decisions.items() if v["final_decision"] == "keep"}
        filtered = {k: v for k, v in summary_data.items() if k in keep_ids}
        (self.out / "qc_filtered_summary.json").write_text(
            json.dumps(filtered, indent=2, ensure_ascii=False), encoding="utf-8")

    def write_split_report(self, summary_data: dict, decisions: dict[str, dict],
                            splits: dict[str, list[str]], thr: ThresholdConfig,
                            overrides: dict[str, str]) -> None:
        keep    = [k for k, v in decisions.items() if v["final_decision"] == "keep"]
        review  = [k for k, v in decisions.items() if v["final_decision"] == "review"]
        exclude = [k for k, v in decisions.items() if v["final_decision"] == "exclude"]

        L: list[str] = []
        L += ["=" * 72, "  PANTSMINI — DATA CURATION SPLIT REPORT", "=" * 72, ""]

        # 1. Dataset summary
        L += ["1. DATASET SUMMARY", "─" * 50,
              f"   Cases in JSON    : {len(summary_data)}",
              f"   Skipped (NIfTI)  : {len(SKIPPED_CASES)}", ""]

        # 2. QC summary
        high_r  = sum(1 for v in decisions.values() if v.get("qc_risk_level") == "high")
        med_r   = sum(1 for v in decisions.values() if v.get("qc_risk_level") == "medium")
        low_r   = sum(1 for v in decisions.values() if v.get("qc_risk_level") == "low")
        L += ["2. QC SUMMARY", "─" * 50,
              f"   High risk        : {high_r}",
              f"   Medium risk      : {med_r}",
              f"   Low risk         : {low_r}", ""]

        # 3. Curation summary
        L += ["3. CURATION SUMMARY", "─" * 50,
              f"   Keep             : {len(keep)}",
              f"   Review           : {len(review)}",
              f"   Exclude          : {len(exclude)}", ""]

        # 4. Thresholds
        L += ["4. THRESHOLDS USED", "─" * 50]
        for k, v in sorted(thr.values.items()):
            L.append(f"   {k:<42} {v}")
        L.append("")

        # 5. Train/val/test counts
        L += ["5. TRAIN / VAL / TEST SPLITS", "─" * 50,
              f"   Train            : {len(splits.get('train', []))}",
              f"   Validation       : {len(splits.get('val', []))}",
              f"   Test             : {len(splits.get('test', []))}",
              f"   Split method     : {splits.get('method', 'unknown')}"]
        if splits.get("fallback_note"):
            L.append(f"   Note             : {splits['fallback_note']}")
        L.append("")

        # 6. Stratification summary
        L += ["6. STRATIFICATION (train set metadata distribution)", "─" * 50]
        train_metas = [summary_data.get(c, {}).get("metadata") or {}
                       for c in splits.get("train", [])]
        if train_metas:
            for field in ("ct_phase", "manufacturer", "sex"):
                cntr = Counter(m.get(field) or "unknown" for m in train_metas)
                L.append(f"   {field}:")
                for k, v in cntr.most_common():
                    L.append(f"     {str(k):<30}: {v}")
        L.append("")

        # 7. User overrides
        L += ["7. USER OVERRIDES", "─" * 50]
        if overrides:
            for cid, val in sorted(overrides.items()):
                L.append(f"   {cid} -> {val}")
        else:
            L.append("   None")
        L.append("")

        # 8. Excluded cases with reasons
        L += ["8. EXCLUDED CASES", "─" * 50]
        for cid in sorted(exclude):
            d = decisions[cid]
            L.append(f"   {cid}  [risk={d.get('qc_risk_level','?')} "
                     f"score={d.get('qc_score','?')}]")
            for r in d.get("reasons", []):
                L.append(f"     - {r}")
        L.append("")

        # 9. Review cases
        L += ["9. REVIEW CASES", "─" * 50]
        for cid in sorted(review):
            d = decisions[cid]
            L.append(f"   {cid}  [risk={d.get('qc_risk_level','?')} "
                     f"score={d.get('qc_score','?')}]")
            for r in d.get("reasons", []):
                L.append(f"     - {r}")
        L.append("")

        # 10. Recommendations
        L += ["10. TRAINING RECOMMENDATIONS", "─" * 50,
              "   * Use only 'keep' cases for supervised training (default).",
              "   * All CRITICAL QC cases have been excluded automatically.",
              "   * Review cases require manual inspection before use.",
              "   * Re-run with --interactive to override individual decisions.",
              "   * Use 'regenerate splits' after any override.",
              "=" * 72]

        (self.out / "split_report.txt").write_text("\n".join(L), encoding="utf-8")

    def save_all(self, summary_data: dict, decisions: dict[str, dict],
                 splits: dict[str, list[str]], thr: ThresholdConfig,
                 overrides: dict[str, str]) -> None:
        self.write_case_lists(decisions, splits)
        self.write_decisions_json(decisions)
        self.write_decisions_csv(decisions)
        self.write_filtered_summary(summary_data, decisions)
        self.write_split_report(summary_data, decisions, splits, thr, overrides)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA CURATION AGENT
# ═══════════════════════════════════════════════════════════════════════════════

class DataCurationAgent:
    """
    Communicates with QCAgent to build auditable curation decisions.

    Communication API:
        qc_agent.get_case_qc(case_id)          -> QC dict
        qc_agent.get_high_risk_cases()         -> list
        qc_agent.get_medium_risk_cases()       -> list
        qc_agent.get_cases_with_conflict(tag)  -> list
        qc_agent.explain_case(case_id)         -> str
        qc_agent.recommend_action(case_id)     -> str
    """

    def __init__(self, summary_data: dict, qc_agent: QCAgent,
                 thr: ThresholdConfig, seed: int = RANDOM_SEED):
        self.summary_data  = summary_data
        self.qc_agent      = qc_agent
        self.thr           = thr
        self.rule_engine   = RuleEngine(thr)
        self.split_manager = SplitManager(thr, seed)

        # {case_id -> full decision dict}
        self.decisions: dict[str, dict] = {}
        # {case_id -> "keep" | "review" | "exclude"}  user overrides
        self.overrides: dict[str, str] = {}
        # Current splits
        self.splits: dict[str, list[str]] = {}

        self._run_curation()

    # ── Core curation ─────────────────────────────────────────────────────────

    def _run_curation(self) -> None:
        """Process all cases through QCAgent + RuleEngine."""
        self.decisions = {}
        for case_id, case_summary in self.summary_data.items():
            # Query QCAgent
            qc_result = self.qc_agent.get_case_qc(case_id)

            # Apply rules
            rule_result = self.rule_engine.decide(case_id, qc_result, case_summary)

            auto_decision = rule_result["decision"]
            user_override = self.overrides.get(case_id)
            final_decision = user_override or auto_decision

            # Build full traceable record
            self.decisions[case_id] = {
                "case_id"            : case_id,
                "qc_risk_level"      : rule_result["qc_risk_level"],
                "qc_score"           : rule_result["qc_score"],
                "conflicts"          : rule_result["conflicts"],
                "evidence"           : rule_result["evidence"],
                "threshold_evidence" : rule_result["threshold_evidence"],
                "automatic_decision" : auto_decision,
                "user_override"      : user_override,
                "final_decision"     : final_decision,
                "reasons"            : rule_result["reasons"],
                "training_impact"    : rule_result["training_impact"],
                "recommendation"     : rule_result["recommendation"],
            }

        # Also record SKIPPED cases
        for case_id in SKIPPED_CASES:
            if case_id not in self.decisions:
                self.decisions[case_id] = {
                    "case_id"            : case_id,
                    "qc_risk_level"      : "high",
                    "qc_score"           : 100,
                    "conflicts"          : ["unreadable_nifti"],
                    "evidence"           : {"reason": "non_orthonormal_direction_cosines"},
                    "threshold_evidence" : {},
                    "automatic_decision" : "exclude",
                    "user_override"      : None,
                    "final_decision"     : "exclude",
                    "reasons"            : ["Unreadable NIfTI file"],
                    "training_impact"    : "Cannot be loaded",
                    "recommendation"     : "exclude",
                }

        self._regenerate_splits()

    def _regenerate_splits(self) -> None:
        keep = [k for k, v in self.decisions.items()
                if v["final_decision"] == "keep"]
        include_review = self.thr.get("INCLUDE_REVIEW_IN_SPLITS", False)
        if include_review:
            keep += [k for k, v in self.decisions.items()
                     if v["final_decision"] == "review"]
        # Build a QC data lookup for stratification
        qc_lookup = {cid: {"risk_level": v.get("qc_risk_level", "low")}
                     for cid, v in self.decisions.items()}
        self.splits = self.split_manager.split(keep, self.summary_data, qc_lookup)

    # ── Communication: query QCAgent ──────────────────────────────────────────

    def query_qc(self, case_id: str) -> dict:
        """Direct passthrough to QCAgent."""
        return self.qc_agent.get_case_qc(case_id)

    def ask_qc_for_conflicts(self, tag: str) -> list[str]:
        return self.qc_agent.get_cases_with_conflict(tag)

    def ask_qc_to_explain(self, case_id: str) -> str:
        return self.qc_agent.explain_case(case_id)

    def ask_qc_to_recommend(self, case_id: str) -> str:
        return self.qc_agent.recommend_action(case_id)

    # ── Override mechanism ────────────────────────────────────────────────────

    def override(self, case_id: str, decision: str) -> str:
        """Apply user override for a case and re-run splits."""
        cid = self.qc_agent._canon(case_id)
        if decision not in ("keep", "review", "exclude"):
            return f"Invalid decision '{decision}'. Use: keep, review, exclude"
        if cid not in self.decisions and cid not in self.summary_data:
            return f"Case '{cid}' not found."
        old = self.decisions.get(cid, {}).get("final_decision", "unknown")
        self.overrides[cid] = decision
        if cid in self.decisions:
            self.decisions[cid]["user_override"] = decision
            self.decisions[cid]["final_decision"] = decision
        self._regenerate_splits()
        return (f"Override applied: {cid}  {old} -> {decision}\n"
                f"Splits regenerated. New counts: "
                f"train={len(self.splits.get('train',[]))}  "
                f"val={len(self.splits.get('val',[]))}  "
                f"test={len(self.splits.get('test',[]))}")

    # ── Save all outputs ───────────────────────────────────────────────────────

    def save_all(self, output_dir: Path, mode: str = "list",
                 images_dir: Path | None = None,
                 labels_dir: Path | None = None) -> None:
        rg = ReportGenerator(output_dir)
        rg.save_all(self.summary_data, self.decisions, self.splits,
                    self.thr, self.overrides)

        if mode in ("symlink", "copy") and images_dir and labels_dir:
            self._link_or_copy(output_dir, mode, images_dir, labels_dir)

    def _link_or_copy(self, output_dir: Path, mode: str,
                       images_dir: Path, labels_dir: Path) -> None:
        for split_name in ("train", "val", "test"):
            for subdir_src, subdir_dst in ((images_dir, "imagesTr"), (labels_dir, "labelsTr")):
                dst = output_dir / split_name / subdir_dst
                dst.mkdir(parents=True, exist_ok=True)
                for case_id in self.splits.get(split_name, []):
                    for f in subdir_src.glob(f"{case_id}*"):
                        target = dst / f.name
                        if mode == "symlink":
                            if not target.exists():
                                target.symlink_to(f.resolve())
                        else:
                            shutil.copy2(f, target)

    # ── Stats ─────────────────────────────────────────────────────────────────

    def summary_text(self) -> str:
        total   = len(self.summary_data)
        keep    = sum(1 for v in self.decisions.values() if v["final_decision"] == "keep")
        review  = sum(1 for v in self.decisions.values() if v["final_decision"] == "review")
        exclude = sum(1 for v in self.decisions.values() if v["final_decision"] == "exclude")
        train   = len(self.splits.get("train", []))
        val_n   = len(self.splits.get("val", []))
        test_n  = len(self.splits.get("test", []))

        return (
            f"DATA CURATION SUMMARY\n{'─'*50}\n"
            f"  Total cases               : {total}\n"
            f"  Keep                      : {keep}  ({keep/total:.1%})\n"
            f"  Review (needs inspection) : {review}  ({review/total:.1%})\n"
            f"  Exclude                   : {exclude}  ({exclude/total:.1%})\n"
            f"{'─'*50}\n"
            f"  Train split               : {train}\n"
            f"  Validation split          : {val_n}\n"
            f"  Test split                : {test_n}\n"
            f"  Split method              : {self.splits.get('method','?')}\n"
            f"{'─'*50}\n"
            f"  User overrides active     : {len(self.overrides)}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# INTERACTIVE CONSOLE
# ═══════════════════════════════════════════════════════════════════════════════

class InteractiveConsole:
    """
    Terminal interface for the curation system.

    Supported commands:
        show high-risk cases / excluded / review / keep
        explain PanTS_XXXXX
        why was PanTS_XXXXX excluded?
        should I keep PanTS_XXXXX?
        override PanTS_XXXXX keep/review/exclude
        change threshold <KEY> <VALUE>
        show thresholds
        regenerate splits
        summarize conflicts
        summarize CT phase distribution
        summarize manufacturer distribution
        show cases with <issue_type>
        qc detail PanTS_XXXXX
        qc show high risk
        save reports
        help
        exit / quit
    """

    BANNER = dedent("""\
        +================================================================+
        |      Data Curation Agent  --  type 'help'                   |
        +================================================================+""")

    HELP = dedent("""\
        COMMANDS
        ────────────────────────────────────────────────────────────────
        OVERVIEW
          summary                          Curation summary
          show thresholds                  Current threshold values

        LISTS
          show high-risk cases             High-risk QC cases
          show excluded                    Excluded cases
          show review                      Review cases
          show keep                        Keep cases
          show train / val / test          Split case lists
          show cases with <tag>            Cases with a specific QC conflict
              e.g. 'show cases with lesion_outside_pancreas'

        CASE INVESTIGATION
          explain PanTS_XXXXX              Full QC detail (from QCAgent)
          why excluded PanTS_XXXXX         Curation decision reasons
          should I keep PanTS_XXXXX        Recommendation
          qc detail PanTS_XXXXX            QCAgent detail view
          qc show high risk                QCAgent filtered list

        OVERRIDES
          override PanTS_XXXXX keep        Force decision to keep
          override PanTS_XXXXX review      Force decision to review
          override PanTS_XXXXX exclude     Force decision to exclude

        CONFIGURATION
          change threshold <KEY> <VALUE>   Update a threshold
              e.g. 'change threshold MIN_LESION_PANCREAS_OVERLAP 0.70'
          regenerate splits                Re-run splits (after overrides/thresholds)

        DISTRIBUTIONS
          summarize conflicts              Conflict tag frequency
          summarize ct phase               CT phase distribution
          summarize manufacturer           Scanner manufacturer distribution
          summarize sex                    Sex distribution

        OUTPUT
          save reports                     Write all outputs to output dir

        EXIT
          quit / exit""")

    _CASE_RE = re.compile(r"PanTS_\d{3,}", re.IGNORECASE)

    def __init__(self, curator: DataCurationAgent, output_dir: Path,
                 mode: str = "list",
                 images_dir: Path | None = None,
                 labels_dir: Path | None = None):
        self.curator    = curator
        self.output_dir = output_dir
        self.mode       = mode
        self.images_dir = images_dir
        self.labels_dir = labels_dir

    def _extract_case(self, text: str) -> str | None:
        m = self._CASE_RE.search(text)
        if not m:
            return None
        digits = re.search(r"\d+", m.group()).group()
        return f"PanTS_{digits.zfill(8)}"

    def _print(self, text: str) -> None:
        if HAS_RICH and _console:
            _console.print(text)
        else:
            print(text)

    def _handle(self, raw: str) -> bool:
        """Process one user command. Returns False to quit."""
        t = raw.strip().lower()
        case_id_raw = self._extract_case(raw)

        # ── Exit ──────────────────────────────────────────────────────────
        if t in ("quit", "exit", "bye", "q"):
            self._print("Goodbye.")
            return False

        # ── Help ──────────────────────────────────────────────────────────
        if "help" in t:
            self._print(self.HELP)
            return True

        # ── Summary ───────────────────────────────────────────────────────
        if t in ("summary", "overview", "status"):
            self._print(self.curator.summary_text())
            return True

        # ── Show thresholds ───────────────────────────────────────────────
        if "threshold" in t and ("show" in t or "list" in t or t == "show thresholds"):
            self._print(self.curator.thr.summary())
            return True

        # ── Change threshold ──────────────────────────────────────────────
        if t.startswith("change threshold") or t.startswith("set threshold"):
            parts = raw.strip().split()
            if len(parts) >= 4:
                key, val = parts[2], parts[3]
                if self.curator.thr.set(key, val):
                    self.curator.rule_engine = RuleEngine(self.curator.thr)
                    self.curator._run_curation()
                    self._print(f"Threshold {key} = {val}\n"
                                f"Curation re-run. {self.curator.summary_text()}")
                else:
                    self._print(f"Unknown threshold key: '{key}'. "
                                f"Type 'show thresholds' for valid keys.")
            else:
                self._print("Usage: change threshold <KEY> <VALUE>")
            return True

        # ── Override ──────────────────────────────────────────────────────
        if t.startswith("override") and case_id_raw:
            parts = t.split()
            new_dec = parts[-1] if parts[-1] in ("keep","review","exclude") else None
            if new_dec:
                self._print(self.curator.override(case_id_raw, new_dec))
            else:
                self._print("Usage: override PanTS_XXXXX <keep|review|exclude>")
            return True

        # ── Regenerate splits ─────────────────────────────────────────────
        if "regenerate" in t or ("regen" in t and "split" in t):
            self.curator._regenerate_splits()
            sp = self.curator.splits
            self._print(f"Splits regenerated: "
                        f"train={len(sp.get('train',[]))}  "
                        f"val={len(sp.get('val',[]))}  "
                        f"test={len(sp.get('test',[]))}")
            return True

        # ── Save reports ──────────────────────────────────────────────────
        if "save" in t and ("report" in t or "output" in t or t == "save"):
            self.curator.save_all(self.output_dir, self.mode,
                                   self.images_dir, self.labels_dir)
            self._print(f"All reports saved to {self.output_dir}/")
            return True

        # ── QC passthrough ────────────────────────────────────────────────
        if t.startswith("qc "):
            qc_query = raw[3:].strip()
            resp = self.curator.qc_agent.respond(qc_query)
            self._print(resp or "")
            return True

        # ── Explain / detail ──────────────────────────────────────────────
        if case_id_raw and any(w in t for w in ("explain","detail","show case","info")):
            self._print(self.curator.ask_qc_to_explain(case_id_raw))
            return True

        # ── Why excluded ──────────────────────────────────────────────────
        if case_id_raw and any(w in t for w in ("why","reason","decision","excluded")):
            cid = self.curator.qc_agent._canon(case_id_raw)
            d   = self.curator.decisions.get(cid)
            if not d:
                self._print(f"Case '{cid}' not in decisions.")
                return True
            lines = [
                f"DECISION for {cid}: {d['final_decision'].upper()}",
                f"  Automatic decision : {d['automatic_decision']}",
                f"  User override      : {d.get('user_override') or 'None'}",
                f"  QC risk            : {d['qc_risk_level']}  (score={d['qc_score']})",
                "  Reasons:",
            ]
            for r in d.get("reasons", []):
                lines.append(f"    - {r}")
            lines.append(f"  Training impact    : {d.get('training_impact','')}")
            self._print("\n".join(lines))
            return True

        # ── Should I keep ─────────────────────────────────────────────────
        if case_id_raw and any(w in t for w in ("should","keep","recommend")):
            self._print(self.curator.ask_qc_to_recommend(case_id_raw))
            return True

        # ── Show cases with conflict ───────────────────────────────────────
        if "show cases with" in t:
            tag = t.replace("show cases with", "").strip()
            cases = self.curator.ask_qc_for_conflicts(tag)
            if cases:
                self._print(f"Cases with '{tag}'  ({len(cases)}):\n" +
                             "\n".join(f"  {c}" for c in cases))
            else:
                self._print(f"No cases with conflict tag '{tag}'.")
            return True

        # ── Show case lists ────────────────────────────────────────────────
        show_map = {
            "high-risk": lambda: [k for k, v in self.curator.decisions.items()
                                   if v["qc_risk_level"] == "high"],
            "high risk": lambda: [k for k, v in self.curator.decisions.items()
                                   if v["qc_risk_level"] == "high"],
            "excluded" : lambda: sorted(k for k, v in self.curator.decisions.items()
                                         if v["final_decision"] == "exclude"),
            "review"   : lambda: sorted(k for k, v in self.curator.decisions.items()
                                         if v["final_decision"] == "review"),
            "keep"     : lambda: sorted(k for k, v in self.curator.decisions.items()
                                         if v["final_decision"] == "keep"),
            "train"    : lambda: self.curator.splits.get("train", []),
            "val"      : lambda: self.curator.splits.get("val", []),
            "test"     : lambda: self.curator.splits.get("test", []),
        }
        for key, getter in show_map.items():
            if f"show {key}" in t or t == f"show {key} cases":
                cases = getter()
                self._print(f"{key.upper()}  ({len(cases)} cases):\n" +
                             "\n".join(f"  {c}" for c in sorted(cases)[:200]) +
                             (f"\n  ... and {len(cases)-200} more" if len(cases) > 200 else ""))
                return True

        # ── Summaries / distributions ─────────────────────────────────────
        if "summarize conflict" in t or "conflict summary" in t:
            cntr = Counter(tag for v in self.curator.decisions.values()
                           for tag in v.get("conflicts", []))
            lines = ["Conflict tag frequency:", f"  {'Tag':<40} {'Count':>5}",
                     f"  {'─'*40} {'─'*5}"]
            for tag, cnt in cntr.most_common():
                lines.append(f"  {tag:<40} {cnt:>5}")
            self._print("\n".join(lines))
            return True

        dist_map = {
            "ct phase"     : "ct_phase",
            "ct_phase"     : "ct_phase",
            "manufacturer" : "manufacturer",
            "sex"          : "sex",
        }
        for kw, field in dist_map.items():
            if kw in t and ("summarize" in t or "distribution" in t or "distrib" in t):
                cntr = Counter(
                    (self.curator.summary_data.get(c) or {}).get("metadata", {}).get(field)
                    or "unknown"
                    for c in self.curator.decisions
                    if self.curator.decisions[c]["final_decision"] == "keep"
                )
                lines = [f"{field} distribution (keep cases):",
                         f"  {'Value':<30} {'Count':>5}",
                         f"  {'─'*30} {'─'*5}"]
                for k, v in cntr.most_common():
                    lines.append(f"  {str(k):<30} {v:>5}")
                self._print("\n".join(lines))
                return True

        # ── Fallback: forward to QCAgent ──────────────────────────────────
        resp = self.curator.qc_agent.respond(raw)
        if resp:
            self._print(resp)
        else:
            self._print("I didn't understand. Type 'help' for available commands.")
        return True

    def run(self) -> None:
        self._print(self.BANNER)
        self._print(self.curator.summary_text())
        self._print("\nType 'help' for commands.\n")

        while True:
            try:
                raw = input("Curate> ").strip()
            except (EOFError, KeyboardInterrupt):
                self._print("\nGoodbye.")
                break
            if not raw:
                continue
            try:
                should_continue = self._handle(raw)
            except Exception as exc:
                self._print(f"[Error: {exc}]")
                should_continue = True
            if not should_continue:
                break


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data Curation System -- two-agent QC + curation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=dedent("""\
            Examples:
              # Full pipeline, interactive
              python data_curation_agent.py --summary-json <dataset>_summary.json --interactive

              # Non-interactive, custom config, symlink mode
              python data_curation_agent.py \\
                --summary-json <dataset>_summary.json \\
                --output-dir   curated_dataset \\
                --config       thresholds.yaml \\
                --mode         symlink \\
                --images-dir   imagesTr \\
                --labels-dir   labelsTr \\
                --seed         42"""))

    parser.add_argument("--summary-json",  default=None,
                        help="Path to <dataset>_summary.json (default: from paths.yaml)")
    parser.add_argument("--qc-report",     default=None,
                        help="Path to existing qc_report.json (optional; recomputed if absent)")
    parser.add_argument("--output-dir",    default=None,
                        help="Output directory for curation files (default: from paths.yaml)")
    parser.add_argument("--config",        default=None,
                        help="Path to thresholds.yaml or thresholds.json (default: from paths.yaml)")
    parser.add_argument("--paths-yaml",    default=None,
                        help="Path to paths.yaml (default: configs/paths.yaml)")
    parser.add_argument("--mode",          default="list",
                        choices=["list", "symlink", "copy"],
                        help="Output mode: list (default), symlink, copy")
    parser.add_argument("--images-dir",    default=None,
                        help="Source imagesTr directory (for symlink/copy mode)")
    parser.add_argument("--labels-dir",    default=None,
                        help="Source labelsTr directory (for symlink/copy mode)")
    parser.add_argument("--seed",          type=int, default=RANDOM_SEED,
                        help="Random seed for splits (default: 42)")
    parser.add_argument("--interactive",   action="store_true",
                        help="Launch interactive terminal after generating outputs")
    parser.add_argument("--no-reports",    action="store_true",
                        help="Skip writing output files (useful when just exploring interactively)")
    parser.add_argument("--task-mode",     default=None, dest="task_mode",
                        choices=sorted(VALID_TASK_MODES),
                        help="Task mode override (default: read from thresholds.yaml). "
                             + " | ".join(sorted(VALID_TASK_MODES)))

    args = parser.parse_args()

    # Load centralized path config
    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    ensure_output_directories(paths)  # creates shared logs_dir only

    config_path = Path(args.config) if args.config else paths.thresholds_config
    images_dir  = Path(args.images_dir) if args.images_dir else paths.raw_images_dir
    labels_dir  = Path(args.labels_dir) if args.labels_dir else paths.raw_labels_dir

    # Resolve task_mode: CLI > thresholds.yaml > default
    _thr_task_mode = "pancreas_lesion_subregions"
    if config_path.exists():
        try:
            import yaml as _yaml
            _thr_raw = _yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            _v = str(_thr_raw.get("TASK_MODE", "")).strip().lower()
            if _v and _v != "auto" and _v in VALID_TASK_MODES:
                _thr_task_mode = _v
        except Exception:
            pass
    _task_mode = args.task_mode if args.task_mode else _thr_task_mode

    # Build task-mode-specific output dirs: outputs/<dataset>/<task_mode>/
    _base_output_dir = Path(args.output_dir).parent if args.output_dir else paths.summary_dir.parent
    _task_dirs = build_output_dirs(_base_output_dir, _task_mode)

    summary_path  = Path(args.summary_json) if args.summary_json else (
        _task_dirs["summary_dir"] / f"{paths.dataset_name}_summary.json"
    )
    output_dir    = Path(args.output_dir) if args.output_dir else _task_dirs["curation_dir"]
    qc_report_dir = _task_dirs["qc_dir"]

    # Set up file logging
    lg = setup_file_logging(
        paths.logs_dir / "data_curation_agent.log",
        logger_name="data_curation_agent",
    )
    lg.info("Starting DataCurationAgent")
    lg.info("TASK_MODE     : %s", _task_mode)
    lg.info("Summary JSON  : %s", summary_path)
    lg.info("Curation dir  : %s", output_dir)
    lg.info("Config        : %s", config_path)

    # ── Load summary JSON ──────────────────────────────────────────────────
    if not summary_path.exists():
        print(f"[ERROR] Summary JSON not found: {summary_path}")
        print("        Run summarize_pantsmini.py first.")
        raise SystemExit(1)

    print(f"Loading {summary_path} ...")
    summary_data, _summary_meta, _summary_artifact_meta = load_summary_cases(summary_path)
    print(f"  {len(summary_data)} cases loaded.")

    # ── Threshold config ─────────────────────────────────────────────────────
    thr = ThresholdConfig(config_path if config_path.exists() else None)
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        thr.save(config_path)
        print(f"  Default thresholds saved to {config_path}")

    # ── QC Agent ─────────────────────────────────────────────────────
    qc_json_path = Path(args.qc_report) if args.qc_report else _task_dirs["qc_dir"] / "qc_report.json"
    if qc_json_path.exists():
        print(f"Loading existing QC report: {qc_json_path} ...")
        # QCAgent still re-runs from JSON; the existing report is informational only
    else:
        print("No existing QC report found -- will recompute from summary JSON.")

    print("Initialising QCAgent ...")
    qc_agent = QCAgent(summary_data, report_dir=qc_report_dir, thr=thr.values)
    print(f"  QC complete: {len(qc_agent.get_high_risk_cases())} high-risk, "
          f"{len(qc_agent.get_medium_risk_cases())} medium-risk.")

    if not args.no_reports:
        print(f"Saving QC reports to {qc_report_dir} ...")
        qc_agent.save_reports(qc_report_dir)
        print("  qc_report.txt  qc_report.json  qc_report.csv  written.")

    # ── Data Curation Agent ────────────────────────────────────────────────
    print("Running DataCurationAgent ...")
    curator = DataCurationAgent(summary_data, qc_agent, thr, seed=args.seed)
    print(f"  {curator.summary_text()}")

    # ── Save outputs ───────────────────────────────────────────────────────
    if not args.no_reports:
        print(f"Saving curation outputs to {output_dir} ...")
        curator.save_all(output_dir, args.mode, images_dir, labels_dir)
        print("  All outputs written.")

    # ── Interactive mode ───────────────────────────────────────────────────
    if args.interactive:
        console = InteractiveConsole(
            curator, output_dir, mode=args.mode,
            images_dir=images_dir, labels_dir=labels_dir)
        console.run()


if __name__ == "__main__":
    main()
