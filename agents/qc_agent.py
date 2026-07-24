#!/usr/bin/env python3
"""
QC Agent  (v3)
==============
Analyses <dataset_name>_summary.json, assigns QC scores and risk levels,
detects conflicts, and generates three report formats:
  - qc_report.txt
  - qc_report.json
  - qc_report.csv

Can also be launched interactively for expert QC discussion.

The active dataset is determined by RAW_DATASET_ROOT in configs/paths.yaml.

Usage:
    python qc_agent.py [--json PATH] [--report-dir DIR] [--no-interactive]

Task-aware QC (v3):
    - pancreas_only requires non-empty pancreas supervision.
    - pancreas_lesion preserves valid lesion-negative samples.
    - Lesion-positive cases use pancreas context for localization QC.
"""
from __future__ import annotations

import argparse
import csv
import warnings
import datetime
import difflib
import json
import re
import statistics
import sys

try:
    import yaml as _yaml
    _HAS_YAML = True
except ModuleNotFoundError:  # pragma: no cover
    _HAS_YAML = False
from collections import Counter
from pathlib import Path
from textwrap import dedent
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
    setup_file_logging,
    build_output_dirs,
    VALID_TASK_MODES,
)
from artifacts.loaders import load_summary_cases

def _get_omitted_cases(data: dict) -> list[str]:
    """Return sorted list of cases where case_status.omit_from_training is True."""
    return sorted(
        k for k, v in data.items()
        if (v.get("case_status") or {}).get("omit_from_training", False)
    )

# ── Training objective ────────────────────────────────────────────────────────
TRAINING_OBJECTIVE = "pancreas_lesion_segmentation"


def _training_objective(task_mode: str) -> str:
    return "pancreas_segmentation" if task_mode == "pancreas_only" else TRAINING_OBJECTIVE


def _primary_target(task_mode: str) -> str:
    return "pancreas" if task_mode == "pancreas_only" else "lesion"

# ── Default QC thresholds ──────────────────────────────────────────────────────
THR_REGION_ERR_CRIT = 0.30
THR_REGION_ERR_WARN = 0.10
# Overlap thresholds for task-aware lesion-pancreas QC
THR_LESION_STRICT   = 0.70   # keep_positive if overlap ≥ this
THR_LESION_MIN      = 0.40   # review_positive if overlap ≥ this (but < STRICT)
                              # review_or_exclude_positive if overlap < this
THR_LESION_CRIT     = 0.10   # lesion_outside flag (legacy compat)
THR_LESION_WARN     = 0.50   # lesion_partial_outside flag (legacy compat)
THR_PAN_CRIT        = 5_000
THR_PAN_WARN        = 10_000
THR_PAN_LARGE       = 200_000
THR_LES_MIN         = 10
THR_LES_MAX         = 200_000
# Partial-visibility thresholds
THR_PARTIAL_PAN_VOLUME  = 30_000   # mm³ — pancreas below this may be truncated
BORDER_MARGIN_VOXELS    = 3        # voxels — bbox within this → border touch
THR_LES_PAN_RATIO   = 0.80

# ── QC scoring weights (penalty points per flag, 0–100 scale) ────────────────
# Task-aware adjustments:
#   - pancreas_mask_empty is no longer automatically critical; weight depends
#     on whether a lesion is present (handled in run_qc, not just by weight).
#   - lesion_no_anatomical_validation: new flag for pancreas-absent + lesion-present.
#
# v4 SCORING NOTE:
#   Overlap-related flags are kept for explainability (reports/compatibility)
#   but do NOT contribute to the numeric score.  The score uses a single
#   monotonic overlap_penalty() instead.  Component-based scoring prevents
#   double-counting and ensures lower overlap → equal or higher penalty.

# ---------------------------------------------------------------------------
# Component penalty functions
# ---------------------------------------------------------------------------

def _overlap_penalty(ov_rate, lesion_present: bool, pancreas_present: bool) -> int:
    """Continuous monotonic penalty for lesion-pancreas overlap.

    Penalty varies smoothly so cases with different overlap values receive
    meaningfully different scores.  Lower overlap always yields equal or
    higher penalty (strict monotonicity).

    Curve version: v2-continuous
    Special cases:
        lesion absent              →  0   (valid negative sample)
        lesion present, no pancreas→  40  (cannot validate)
        overlap None               →  45  (data quality issue)

    Continuous bands:
        overlap ≥ 0.80  →   0                            (good)
        0.30 ≤ ov < 0.80 →  round(60 * (0.80 - ov) / 0.50)  (moderate→high)
        0.05 ≤ ov < 0.30 →  round(60 + 25 * (0.30 - ov) / 0.30)  (very high)
        ov < 0.05        →  85                           (catastrophic)

    Example outputs:
        ov=0.80 → 0   ov=0.62 → 22   ov=0.50 → 36
        ov=0.45 → 42  ov=0.30 → 60   ov=0.27 → 68
        ov=0.15 → 73  ov=0.05 → 83   ov=0.01 → 85
    """
    if not lesion_present:
        return 0
    if not pancreas_present:
        return 40
    if ov_rate is None:
        return 45

    ov = max(0.0, min(1.0, ov_rate))

    if ov >= 0.80:
        return 0
    if ov < 0.05:
        return 85
    if ov < 0.30:
        return round(60 + 25 * (0.30 - ov) / 0.30)
    # 0.30 ≤ ov < 0.80
    return round(60 * (0.80 - ov) / 0.50)


# Component caps — enforce task priority: lesion errors dominate.
# For pancreas_lesion_segmentation, region consistency is informative but
# must not drive the overall score.  Each cap is applied before summing.
_COMPONENT_CAPS: dict[str, int] = {
    "geometry_integrity"   : 80,   # shape + spacing issues
    "lesion_localization"  : 70,   # monotonic overlap penalty
    "lesion_burden"        : 20,   # small/large lesion anomalies
    "pancreas_context"     : 20,   # pancreas size / absence context
    "fov_integrity"        : 15,   # border touching / truncation / missing regions
    # region_consistency: 0 for positive cases (overlap already judges quality),
    #                     10 for negative cases (only structural signal available).
    "region_consistency"   : 10,   # head/body/tail deviation — max for negative cases
    "metadata_completeness": 10,   # missing fields
    "attenuation_integrity": 10,   # suspicious HU values (secondary QC signal only)
}

# ── Hybrid scoring parameters ─────────────────────────────────────────────────
# Additive geometry gate + multiplicative secondary amplifier.
# alpha controls how strongly secondary issues amplify a borderline overlap.
# Secondary issues cannot independently condemn a good-overlap case
# (P_overlap = 0  ⟹  amplifier has no effect).
_HYBRID_ALPHA: float = 0.4
_HYBRID_SECONDARY_WEIGHTS: dict[str, float] = {
    "lesion_burden"        : 1.0,
    "pancreas_context"     : 0.8,
    "fov_integrity"        : 0.6,
    "region_consistency"   : 0.6,
    "metadata_completeness": 0.3,
    "attenuation_integrity": 0.4,
}
# beta: direct fallback coefficient applied to the raw secondary weighted sum
# when P_overlap == 0 (negative cases / perfect localization).  Unlike alpha
# (which scales a fractional amplifier), beta acts on the raw weighted sum
# directly on the 0-100 penalty scale, so a value of ~1.5 is appropriate.
_HYBRID_BETA: float = 1.5


def _geometry_penalty(flags: list) -> int:
    """Combined penalty for shape_mismatch and invalid_spacing.

    Max contribution: 80 (shape_mismatch 40 + CRITICAL spacing 40).
    """
    tags = {t for _, t, _ in flags}
    p = 0
    if "shape_mismatch" in tags:
        p += 40
    if "invalid_spacing" in tags:
        sev = next((s for s, t, _ in flags if t == "invalid_spacing"), "WARNING")
        p += 40 if sev == "CRITICAL" else 20
    return min(p, _COMPONENT_CAPS["geometry_integrity"])


def _region_penalty_raw(flags: list, rel_err: float = 0.0,
                        thr: "dict | None" = None) -> int:
    """Uncapped anatomical region consistency score.

    Returned for reporting. Cap is applied separately in
    _compute_score_components() so the uncapped value is still visible.

    rel_err: actual head+body+tail vs total pancreas relative error (0–1+).
             Scales the region_sum_error penalty continuously within its
             min–max bounds instead of using a flat integer step.
             CRITICAL band: 10 (at threshold 0.25) → 30 (at rel_err ≥ 1.0).
             WARNING  band:  1 (at low end)        → 10 (at threshold 0.25).
    """
    tags = {t for _, t, _ in flags}
    pts = (thr or {}).get("REGION_SCORE_POINTS") or {}
    incomplete_pts = int(pts["incomplete_anatomical_coverage"])
    missing_all_pts = int(pts["subregions_expected_but_missing"])
    missing_region_pts = int(pts["missing_subregion"])
    p = 0
    if "region_sum_error_critical" in tags:
        # Continuous: 10 at crit threshold (0.25) → 30 at rel_err ≥ 1.0
        p += min(round(10 + 20 * max(rel_err - 0.25, 0.0) / 0.75), 30)
    if "region_sum_error_warning" in tags:
        # Continuous: 1 at low warning level → 10 approaching crit threshold
        p += max(1, min(round(10 * rel_err / 0.25), 10))
    for t in ("head_body_overlap", "head_tail_overlap", "body_tail_overlap"):
        if t in tags:
            p += 5
    if "incomplete_anatomical_coverage" in tags:
        p += incomplete_pts
    if "subregions_expected_but_missing" in tags:
        p += missing_all_pts
    for rn in ("head", "body", "tail"):
        tag = f"{rn}_region_empty"
        if tag in tags:
            sev = next((s for s, t, _ in flags if t == tag), "INFO")
            if sev == "WARNING":
                p += missing_region_pts
    return p


def _fov_penalty(flags: list, evidence: dict, thr: "dict | None" = None) -> int:
    """Penalty for field-of-view truncation using pancreas/image FOV evidence."""
    pts = (thr or {}).get("FOV_SCORE_POINTS") or {}
    border_pts = int(pts["border_touching"])
    partial_pts = int(pts["partial_visibility_likely"])
    trunc_pts = int(pts["anatomical_truncation_suspected"])
    p = 0

    bt = evidence.get("border_touching") or {}
    if any(bt.values()):
        p += border_pts
    if evidence.get("partial_visibility_likely"):
        p += partial_pts
    if evidence.get("anatomical_truncation_suspected"):
        p += trunc_pts

    return min(p, _COMPONENT_CAPS["fov_integrity"])


def _lesion_size_penalty(flags: list) -> int:
    """Penalty for lesion size anomalies.  Capped at 20.

    For pancreas_lesion_segmentation an extremely small lesion is notable
    but must not overshadow a bad overlap.  high_lesion_pancreas_ratio is
    included here because it is proportional to lesion size.
    """
    tags = {t for _, t, _ in flags}
    p = 0
    if "very_small_lesion" in tags:
        p += 15
    if "very_large_lesion" in tags:
        p += 15
    if "high_lesion_pancreas_ratio" in tags:
        p += 10
    return min(p, _COMPONENT_CAPS["lesion_burden"])


def _pancreas_context_penalty(flags: list, pancreas_present: bool, lesion_present: bool) -> int:
    """Penalty for pancreas context issues (size, absence).  Capped at 20.

    Pancreas absence + lesion present is already penalised heavily via
    _overlap_penalty (→ 35); here we only add a small structural signal.
    """
    tags = {t for _, t, _ in flags}
    p = 0
    if "very_small_pancreas" in tags:
        p += 20
    elif "small_pancreas" in tags:
        p += 10
    elif "large_pancreas" in tags:
        p += 5
    if "required_target_missing" in tags:
        p += 20
    if "pancreas_mask_empty" in tags and lesion_present:
        # absence + lesion: already covered by overlap_penalty; add small bump
        p += 5
    return min(p, _COMPONENT_CAPS["pancreas_context"])


def _metadata_penalty(flags: list) -> int:
    tags = {t for _, t, _ in flags}
    if "missing_metadata" not in tags:
        return 0
    # Proportional to fraction of expected fields missing (4 total: sex, age, ct_phase, manufacturer).
    # n=1→2, n=2→5, n=3→8, n=4→10 (banker's rounding keeps values distinct).
    n_missing = len(next((m for s, t, m in flags if t == "missing_metadata"), "").split(","))
    return min(round(10 * n_missing / 4), _COMPONENT_CAPS["metadata_completeness"])


def _attenuation_consistency_penalty(flags: list) -> int:
    """Small penalty for suspicious HU distributions.  Cap at 10.

    This is a secondary QC signal.  Normal physiological variation
    (Hypo / Iso / Hyper) does NOT trigger this penalty — only values
    outside the plausible soft-tissue CT range [-200, 400] HU or
    NaN / invalid statistics are penalised.
    """
    tags = {t for _, t, _ in flags}
    if "suspicious_hu_distribution" in tags:
        return _COMPONENT_CAPS["attenuation_integrity"]
    return 0


def _compute_score_components(
    flags: list,
    ov_rate,
    lesion_present: bool,
    pancreas_present: bool,
    evidence: dict,
    rel_err: float = 0.0,
    thr: "dict | None" = None,
) -> dict:
    """Return named penalty components that sum to the final QC score.

    Component max caps (enforced before summing):
        geometry_integrity     80  — shape / spacing
        lesion_localization    70  — continuous monotonic overlap penalty (v2)
        lesion_burden          20  — small / large / ratio anomalies
        pancreas_context       20  — pancreas size / absence context
        fov_integrity          15  — border touching / truncation / missing regions
        region_consistency      0  — zero for positive cases (overlap already captures quality)
                               10  — for negative cases (only structural signal available)
        metadata_completeness  10  — missing fields

    Region consistency is 0 for positive cases: when a lesion is present the
    overlap penalty already fully judges annotation quality; sub-region
    decomposition errors are irrelevant noise for training sample ranking.
    For negative cases (no lesion) the region consistency is the primary
    structural sanity check, so a cap of 10 applies.

    Overlap penalty uses a continuous curve (v2-continuous): lower overlap
    always produces equal or higher penalty with no hard bin edges.
    """
    raw_region = _region_penalty_raw(flags, rel_err=rel_err, thr=thr)
    # Task-aware conditional cap: positive cases → 0, negative cases → 10
    _region_cap = 0 if lesion_present else _COMPONENT_CAPS["region_consistency"]
    capped_region = min(raw_region, _region_cap)
    return {
        # Scored components — canonical domain names (1:1 with qc_domains output)
        "geometry_integrity"   : _geometry_penalty(flags),
        "lesion_localization"  : _overlap_penalty(ov_rate, lesion_present, pancreas_present),
        "lesion_burden"        : _lesion_size_penalty(flags),
        "pancreas_context"     : _pancreas_context_penalty(flags, pancreas_present, lesion_present),
        "fov_integrity"        : _fov_penalty(flags, evidence, thr),
        "region_consistency"   : capped_region,
        "metadata_completeness": _metadata_penalty(flags),
        "attenuation_integrity": _attenuation_consistency_penalty(flags),
        # Informational fields (prefix _ → excluded from score sum)
        "_region_raw"          : raw_region,
        "_overlap_curve"       : "v2-continuous",
    }


def _compute_score_hybrid(
    score_components: dict,
    alpha: float = _HYBRID_ALPHA,
    beta: float = _HYBRID_BETA,
    secondary_weights: dict | None = None,
) -> tuple[int, dict]:
    """Hybrid additive-gate + multiplicative-amplifier QC scoring with fallback.

    Full formula:
        score = P_geom
              + P_overlap * (1 + alpha * S_capped / 100)   [amplifier, active when P_overlap > 0]
              + I(P_overlap=0) * beta * S_raw               [fallback, active when P_overlap == 0]
        qc_score = min(round(score), 100)

    S_capped  — secondary weighted sum using capped component scores
                (prevents secondary issues from dominating when overlap exists).
    S_raw     — same weights but region_consistency uses its uncapped raw value
                (_region_raw); other components have no separate raw value.

    Geometry is an additive hard gate, excluded from both amplifier and fallback.
    When P_overlap > 0 the fallback term is zero (formula collapses to v1 hybrid).
    When P_overlap == 0 the amplifier term is zero and secondary issues contribute
    directly via beta * S_raw — a structurally bad negative case is still penalised.

    Returns (qc_score: int, scoring_details: dict).
    """
    if secondary_weights is None:
        secondary_weights = _HYBRID_SECONDARY_WEIGHTS

    p_geom    = float(score_components.get("geometry_integrity", 0))
    p_overlap = float(score_components.get("lesion_localization", 0))

    # S_capped: standard capped values — used in the multiplicative amplifier
    secondary_weighted_sum = sum(
        weight * float(score_components.get(key, 0))
        for key, weight in secondary_weights.items()
    )

    # S_raw: uncapped region_consistency; all other components have no raw variant
    def _raw(key: str) -> float:
        if key == "region_consistency":
            return float(score_components.get("_region_raw",
                         score_components.get("region_consistency", 0)))
        return float(score_components.get(key, 0))

    fallback_secondary_sum = sum(
        weight * _raw(key) for key, weight in secondary_weights.items()
    )

    fallback_active   = (p_overlap == 0.0)
    fallback_term     = beta * fallback_secondary_sum if fallback_active else 0.0

    secondary_amplifier = 1.0 + alpha * secondary_weighted_sum / 100.0
    raw_score           = p_geom + p_overlap * secondary_amplifier + fallback_term
    qc_score            = min(round(raw_score), 100)

    scoring_details: dict = {
        "score_formula"                   : "geometry_gate_plus_overlap_amplifier_plus_fallback",
        "alpha"                           : alpha,
        "beta"                            : beta,
        "secondary_weighted_sum"          : round(secondary_weighted_sum, 3),
        "secondary_amplifier"             : round(secondary_amplifier, 4),
        "fallback_active"                 : fallback_active,
        "fallback_secondary_sum_raw"      : round(fallback_secondary_sum, 3),
        "fallback_term"                   : round(fallback_term, 3),
        "raw_score_before_clamp"          : round(raw_score, 2),
        "qc_score_after_clamp"            : qc_score,
        "geometry_is_additive_gate"       : True,
        "geometry_excluded_from_amplifier": True,
    }
    return qc_score, scoring_details


# ---------------------------------------------------------------------------
# Risk interpretation helpers
# ---------------------------------------------------------------------------

_FAILURE_MODE_LABELS: dict[str, str] = {
    "geometry_integrity"   : "geometry_integrity_failure",
    "lesion_localization"  : "lesion_localization_failure",
    "lesion_burden"        : "lesion_burden_abnormality",
    "pancreas_context"     : "pancreas_context_issue",
    "fov_integrity"        : "fov_integrity_issue",
    "region_consistency"   : "anatomical_region_inconsistency",
    "metadata_completeness": "missing_metadata",
    "attenuation_integrity": "suspicious_attenuation",
}


def _dominant_failure_mode(sc: dict) -> str:
    """Return human-readable label for the highest-scoring component."""
    scored = {k: v for k, v in sc.items()
              if not k.startswith("_") and v > 0}
    if not scored:
        return "none"
    dominant_key = max(scored, key=lambda k: scored[k])
    return _FAILURE_MODE_LABELS.get(dominant_key, dominant_key)


def _overlap_severity_label(penalty: int,
                             lesion_present: bool,
                             pancreas_present: bool) -> str:
    if not lesion_present:    return "not_applicable"
    if not pancreas_present:  return "cannot_evaluate"
    if penalty == 0:          return "normal"
    if penalty <= 25:         return "minor"
    if penalty <= 45:         return "moderate"
    if penalty <= 65:         return "high"
    if penalty <= 80:         return "very_high"
    return "catastrophic"


def _build_risk_interpretation(sc: dict,
                                 lesion_present: bool,
                                 pancreas_present: bool,
                                 hu_evidence: dict | None = None,
                                 task_mode: str = "pancreas_lesion") -> dict:
    """Build a 4-dimension human-readable risk interpretation dict.

    Each dimension covers an independent failure mode so users can see
    WHY the final score is high without inspecting every flag.
    """
    ov_pen     = sc.get("lesion_localization", 0)
    burden_pen = sc.get("lesion_burden", 0)
    geom_pen   = sc.get("geometry_integrity", 0)
    region_pen = sc.get("region_consistency", 0)
    raw_region = sc.get("_region_raw", region_pen)

    # ── Spatial plausibility ──────────────────────────────────────────────
    sp_sev = _overlap_severity_label(ov_pen, lesion_present, pancreas_present)
    if task_mode == "pancreas_only":
        sp_sev = "not_applicable"
        sp_desc = "Lesion localization is not applicable to pancreas-only segmentation."
    elif not lesion_present:
        sp_desc = "Not applicable: negative sample — no lesion to localize."
    elif not pancreas_present:
        sp_desc = "Cannot evaluate: pancreas mask absent with lesion present."
    elif ov_pen == 0:
        sp_desc = "Good: lesion is well-localized within pancreas (overlap ≥ 80%)."
    elif ov_pen <= 25:
        sp_desc = f"Minor concern: lesion mostly within pancreas (penalty={ov_pen})."
    elif ov_pen <= 45:
        sp_desc = f"Moderate concern: lesion partially outside pancreas (penalty={ov_pen})."
    elif ov_pen <= 65:
        sp_desc = f"High concern: lesion substantially outside pancreas (penalty={ov_pen})."
    elif ov_pen <= 80:
        sp_desc = f"Very high concern: lesion mostly outside pancreas (penalty={ov_pen})."
    else:
        sp_desc = (f"Catastrophic: lesion barely overlaps pancreas near-zero "
                   f"(penalty={ov_pen}).")

    # ── Burden plausibility ───────────────────────────────────────────────
    if task_mode == "pancreas_only":
        bu_sev = "not_applicable"
        bu_desc = "Lesion burden is not applicable to pancreas-only segmentation."
    elif not lesion_present:
        bu_sev  = "not_applicable"
        bu_desc = "Not applicable: no lesion present."
    elif burden_pen == 0:
        bu_sev  = "normal"
        bu_desc = "Normal: lesion size and ratio within expected range."
    elif burden_pen <= 10:
        bu_sev  = "minor"
        bu_desc = "Minor concern: lesion size mildly outside normal range."
    elif burden_pen <= 15:
        bu_sev  = "moderate"
        bu_desc = "Moderate concern: lesion size or lesion/pancreas ratio notable."
    else:
        bu_sev  = "high"
        bu_desc = ("High concern: lesion size or lesion/pancreas ratio "
                   "substantially abnormal.")

    # ── Geometry integrity ────────────────────────────────────────────────
    if geom_pen == 0:
        ge_sev  = "normal"
        ge_desc = "Normal: no shape or spacing issues detected."
    elif geom_pen <= 25:
        ge_sev  = "minor"
        ge_desc = "Minor: highly anisotropic spacing detected."
    elif geom_pen <= 50:
        ge_sev  = "moderate"
        ge_desc = "Moderate: shape or spacing issue detected."
    else:
        ge_sev  = "severe"
        ge_desc = "Severe: critical shape or spacing mismatch."

    # ── Anatomical region integrity ───────────────────────────────────────
    if region_pen == 0:
        re_sev  = "normal"
        re_desc = "Normal: head/body/tail consistent with whole pancreas."
    elif region_pen <= 5:
        re_sev  = "minor"
        re_desc = "Minor: small sub-region deviation detected."
    elif region_pen <= 10:
        re_sev  = "moderate"
        re_desc = "Moderate: notable sub-region deviation or missing region."
    else:
        re_sev  = "high"
        re_desc = (f"High: significant sub-region inconsistency "
                   f"(raw={raw_region}, capped={region_pen} — "
                   f"task=pancreas_lesion_segmentation).")

    # ── Attenuation plausibility (secondary — informational / explainability) ──────
    hu_ev = hu_evidence or {}
    # Support both new schema (tumor_median_hu) and legacy (mean_hu_tumor)
    median_hu    = hu_ev.get("tumor_median_hu") or hu_ev.get("mean_hu_tumor")
    attenuate    = hu_ev.get("tumor_attenuation")
    delta_hu     = hu_ev.get("delta_hu")
    z_score      = hu_ev.get("z_score_hu")
    suspicious_hu= hu_ev.get("suspicious_hu_distribution", False)

    if task_mode == "pancreas_only":
        att_sev = "not_applicable"
        att_desc = "Tumor attenuation is not applicable to pancreas-only segmentation."
    elif not lesion_present:
        att_sev  = "not_applicable"
        att_desc = "Not applicable: no lesion present."
    elif median_hu is None:
        att_sev  = "not_computed"
        att_desc = "HU statistics not available (CT not loaded or lesion absent)."
    elif suspicious_hu:
        att_sev  = "warning"
        hu_str   = f"{median_hu:.1f} HU" if median_hu is not None else "N/A"
        att_desc = (f"Suspicious HU distribution ({hu_str} median) — verify CT image, "
                    f"mask quality, or acquisition phase.")
    elif attenuate == "Hypo":
        delta_str = f"{delta_hu:+.0f}" if delta_hu is not None else "N/A"
        z_str     = f"{z_score:.2f}" if z_score is not None else "N/A"
        att_sev  = "normal"
        att_desc = (f"Hypoattenuating lesion (median={median_hu:.1f} HU, delta={delta_str} HU, "
                    f"z={z_str}) — consistent with PDAC or cystic lesion.")
    elif attenuate == "Hyper":
        delta_str = f"{delta_hu:+.0f}" if delta_hu is not None else "N/A"
        z_str     = f"{z_score:.2f}" if z_score is not None else "N/A"
        att_sev  = "informational"
        att_desc = (f"Hyperattenuating lesion (median={median_hu:.1f} HU, delta={delta_str} HU, "
                    f"z={z_str}) — may indicate NET or annotation inconsistency.")
    elif attenuate == "Iso":
        delta_str = f"{delta_hu:+.0f}" if delta_hu is not None else "N/A"
        z_str     = f"{z_score:.2f}" if z_score is not None else "N/A"
        att_sev  = "normal"
        att_desc = (f"Isoattenuating lesion (median={median_hu:.1f} HU, delta={delta_str} HU, "
                    f"z={z_str}) — plausible; consider indirect signs if PDAC suspected.")
    else:  # Unknown (pancreas absent)
        att_sev  = "unknown"
        hu_str   = f"{median_hu:.1f}" if median_hu is not None else "N/A"
        att_desc = (f"Pancreas mask absent; relative attenuation cannot be computed "
                    f"(tumor median HU={hu_str}).")

    return {
        "spatial_plausibility": {
            "severity": sp_sev, "description": sp_desc,
        },
        "anatomical_burden_plausibility": {
            "severity": bu_sev, "description": bu_desc,
        },
        "geometry_integrity": {
            "severity": ge_sev, "description": ge_desc,
        },
        "anatomical_region_integrity": {
            "severity": re_sev, "description": re_desc,
        },
        "attenuation_plausibility": {
            "severity": att_sev, "description": att_desc,
        },
    }


# ---------------------------------------------------------------------------
# Per-domain verdict helpers  (v4 — solves scalar conflation)
# ---------------------------------------------------------------------------

# Default per-domain thresholds (can be overridden via DOMAIN_THRESHOLDS in
# thresholds.yaml).  Values are penalty-point thresholds on each domain's
# ── v5 Multi-domain QC profile ───────────────────────────────────────────────
# The QC system produces a *profile* (one entry per domain) rather than a
# single scalar.  The scalar (triage_score) is preserved for ranking only.
#
# Eight canonical domains (1:1 with score components):
#   geometry_integrity     — shape, spacing, mask/image alignment
#   lesion_localization    — lesion-pancreas overlap
#   lesion_burden          — lesion size and ratio anomalies
#   pancreas_context       — pancreas size / absence context
#   fov_integrity          — field-of-view coverage / truncation plausibility
#   region_consistency     — head/body/tail sub-region decomposition
#   attenuation_integrity  — HU distribution plausibility
#   metadata_completeness  — missing clinical metadata fields
#
# Five severity levels (ordered):
#   normal            — no issues detected
#   low_warning       — minor informational concern
#   moderate_warning  — notable; worth reviewing
#   high_warning      — significant; prioritise for review
#   critical          — must not be used for training without human inspection

_CANONICAL_DOMAINS: tuple[str, ...] = (
    "geometry_integrity",
    "lesion_localization",
    "lesion_burden",
    "pancreas_context",
    "fov_integrity",
    "region_consistency",
    "attenuation_integrity",
    "metadata_completeness",
)
# Alias kept for any code that still uses _PROFILE_DOMAINS
_PROFILE_DOMAINS = _CANONICAL_DOMAINS

_SEVERITY_ORDER: tuple[str, ...] = (
    "normal", "low_warning", "moderate_warning", "high_warning", "critical",
)

# Maps each flag tag to its owning domain (canonical 7-domain schema).
_FLAG_DOMAIN_MAP: dict[str, str] = {
    # geometry_integrity — structural mask / image integrity
    "shape_mismatch":                     "geometry_integrity",
    "invalid_spacing":                    "geometry_integrity",
    # lesion_localization — lesion localisation
    "lesion_overlap_critical_fail":       "lesion_localization",
    "lesion_overlap_strict_fail":         "lesion_localization",
    "lesion_outside_pancreas":            "lesion_localization",
    "lesion_partial_outside":             "lesion_localization",
    "lesion_no_anatomical_validation":    "lesion_localization",
    "pancreas_mask_empty":                "lesion_localization",
    "pancreas_hole_annotation_suspected": "lesion_localization",
    "lesion_in_subregion_spillover":      "lesion_localization",
    "lesion_mask_empty":                  "lesion_localization",
    "required_target_missing":            "pancreas_context",
    # lesion_burden — lesion size plausibility
    "very_small_lesion":                  "lesion_burden",
    "very_large_lesion":                  "lesion_burden",
    "high_lesion_pancreas_ratio":         "lesion_burden",
    # pancreas_context — pancreas morphology
    "very_small_pancreas":                "pancreas_context",
    "small_pancreas":                     "pancreas_context",
    "large_pancreas":                     "pancreas_context",
    # fov_integrity — field of view / missing regions
    "incomplete_anatomical_coverage":     "fov_integrity",
    "head_region_empty":                  "fov_integrity",
    "body_region_empty":                  "fov_integrity",
    "tail_region_empty":                  "fov_integrity",
    "subregions_expected_but_missing":    "fov_integrity",
    # region_consistency — sub-region geometry
    "region_sum_error_critical":          "region_consistency",
    "region_sum_error_warning":           "region_consistency",
    "head_body_overlap":                  "region_consistency",
    "head_tail_overlap":                  "region_consistency",
    "body_tail_overlap":                  "region_consistency",
    "subregions_not_applicable":          "region_consistency",
    # attenuation_integrity — HU distribution
    "suspicious_hu_distribution":         "attenuation_integrity",
    "attenuation_rule_disagreement":      "attenuation_integrity",
    # metadata_completeness — clinical metadata
    "missing_metadata":                   "metadata_completeness",
}

# Maps each domain to the score_components key that contributes to its score (1:1).
_PROFILE_SCORE_COMPONENTS: dict[str, list[str]] = {
    "geometry_integrity"   : ["geometry_integrity"],
    "lesion_localization"  : ["lesion_localization"],
    "lesion_burden"        : ["lesion_burden"],
    "pancreas_context"     : ["pancreas_context"],
    "fov_integrity"        : ["fov_integrity"],
    "region_consistency"   : ["region_consistency"],
    "attenuation_integrity": ["attenuation_integrity"],
    "metadata_completeness": ["metadata_completeness"],
}

# Default per-domain severity thresholds (overridable via QC_PROFILE_THRESHOLDS
# in thresholds.yaml).  Values are penalty-point breakpoints.
_DEFAULT_SEVERITY_THRESHOLDS: dict[str, dict[str, int]] = {
    "geometry_integrity"   : {"low_warning":  1, "moderate_warning": 20, "high_warning": 35, "critical": 40},
    "lesion_localization"  : {"low_warning":  1, "moderate_warning": 15, "high_warning": 35, "critical": 40},
    "lesion_burden"        : {"low_warning":  1, "moderate_warning":  8, "high_warning": 15, "critical": 20},
    "pancreas_context"     : {"low_warning":  1, "moderate_warning":  8, "high_warning": 15, "critical": 20},
    "fov_integrity"        : {"low_warning":  1, "moderate_warning":  5, "high_warning": 10, "critical": 15},
    "region_consistency"   : {"low_warning":  1, "moderate_warning":  5, "high_warning": 10, "critical": 15},
    "attenuation_integrity": {"low_warning":  1, "moderate_warning":  5, "high_warning": 10, "critical": 15},
    "metadata_completeness": {"low_warning":  1, "moderate_warning":  4, "high_warning":  8, "critical": 10},
}

def _build_qc_profile(
    flags: list,
    score_components: dict,
    thr: "dict | None" = None,
) -> dict:
    """Build the multi-dimensional QC profile (v5).

    Returns a dict keyed by domain name, each containing:
        score     — sum of contributing component penalties (float)
        severity  — one of: normal | low_warning | moderate_warning |
                    high_warning | critical
        evidence  — list of {tag, severity, message} for non-INFO flags
                    belonging to this domain
    """
    dt = (thr or {}).get("QC_PROFILE_THRESHOLDS") or {}
    metadata_policy = (thr or {}).get("METADATA_POLICY") or {}
    metadata_max = str(metadata_policy.get("max_severity", "low_warning"))

    def _thresholds(domain: str) -> dict[str, int]:
        defaults = _DEFAULT_SEVERITY_THRESHOLDS.get(
            domain, {"low_warning": 1, "moderate_warning": 10, "high_warning": 30, "critical": 40}
        )
        override = dt.get(domain) or {}
        return {k: int(override.get(k, defaults[k]))
                for k in ("low_warning", "moderate_warning", "high_warning", "critical")}

    def _score_to_severity(score: float, domain: str, has_critical_flag: bool) -> str:
        if domain == "metadata_completeness":
            if score <= 0 and not has_critical_flag:
                return "normal"
            return metadata_max if metadata_max in _SEVERITY_ORDER else "low_warning"
        if has_critical_flag:
            return "critical"
        t = _thresholds(domain)
        if score >= t["critical"]:          return "critical"
        if score >= t["high_warning"]:      return "high_warning"
        if score >= t["moderate_warning"]:  return "moderate_warning"
        if score >= t["low_warning"]:       return "low_warning"
        return "normal"

    result: dict = {}
    for domain in _PROFILE_DOMAINS:
        comp_keys   = _PROFILE_SCORE_COMPONENTS.get(domain, [])
        domain_score = round(sum(float(score_components.get(k, 0)) for k in comp_keys), 2)

        domain_flags = [
            {"tag": tag, "severity": sev.lower(), "message": msg}
            for sev, tag, msg in flags
            if _FLAG_DOMAIN_MAP.get(tag) == domain and sev != "INFO"
        ]
        has_critical = any(
            sev == "CRITICAL"
            for sev, tag, _ in flags
            if _FLAG_DOMAIN_MAP.get(tag) == domain
        )
        result[domain] = {
            "score":    domain_score,
            "severity": _score_to_severity(domain_score, domain, has_critical),
            "evidence": domain_flags,
        }
    return result


def _recommend_from_profile(qc_profile: dict, thr: "dict | None" = None) -> "tuple[str, str | None]":
    """Derive (recommendation, driving_domain) via strict domain gating.

    Priority hierarchy (first matching rule wins):
      1. geometry_integrity critical         → exclude
      2. lesion_localization critical        → exclude
      3. lesion_burden critical              → review
      4. pancreas_context critical           → review
    5. fov_integrity critical              → review
    6. region_consistency critical         → review
    7. attenuation_integrity critical      → review
    8. geometry_integrity high_warning     → review
    9. lesion_localization high_warning    → review
     10. lesion_burden high_warning          → review
     11. pancreas_context high_warning       → review
     12. fov_integrity high_warning          → review
     13. region_consistency high_warning     → review
     14. attenuation_integrity high_warning  → review
      15. metadata_completeness low_warning   → keep_with_metadata_warning
      16. any domain moderate_warning         → review
      17. any non-metadata low_warning only   → keep  (noted in driving_domain)
      18. all normal                          → keep

    metadata_completeness alone can NEVER produce risk > low.
    """
    def _sev(domain: str) -> str:
        return qc_profile.get(domain, {}).get("severity", "normal")

    # Rules 1–2: critical in structural domains → exclude
    for domain in ("geometry_integrity", "lesion_localization"):
        if _sev(domain) == "critical":
            return "exclude", domain

    metadata_policy = (thr or {}).get("METADATA_POLICY") or {}
    metadata_rec = str(metadata_policy.get("recommendation", "keep_with_metadata_warning"))

    # Rules 3–6: critical in secondary structural domains → review
    for domain in ("lesion_burden", "pancreas_context", "fov_integrity",
                   "region_consistency", "attenuation_integrity"):
        if _sev(domain) == "critical":
            return "review", domain

    # Rules 7–12: high_warning in structural domains → review
    for domain in ("geometry_integrity", "lesion_localization", "lesion_burden",
                   "pancreas_context", "fov_integrity", "region_consistency", "attenuation_integrity"):
        if _sev(domain) == "high_warning":
            return "review", domain

    # Rule 14: any moderate_warning → review
    for domain in _CANONICAL_DOMAINS:
        if domain == "metadata_completeness":
            continue
        if _sev(domain) == "moderate_warning":
            return "review", domain

    # Rule 13: metadata warning → keep_with_metadata_warning. Metadata is capped
    # to low_warning because the primary task is image segmentation.
    if _sev("metadata_completeness") == "low_warning":
        return metadata_rec, "metadata_completeness"

    # Rule 15: any low_warning → keep (minor issues noted)
    for domain in _CANONICAL_DOMAINS:
        if _sev(domain) == "low_warning":
            return "keep", domain

    return "keep", None


def _scalar_recommendation(score: float, flags: list, thr: "dict | None" = None) -> str:
    keep_max = float((thr or {}).get("MAX_QC_SCORE_FOR_KEEP", 30))
    review_max = float((thr or {}).get("MAX_QC_SCORE_FOR_REVIEW", 60))
    has_critical = any(sev == "CRITICAL" for sev, _, _ in flags)
    if score > review_max:
        return "exclude"
    if score > keep_max or has_critical:
        return "review"
    return "keep"


def _merge_recommendations(profile_recommendation: str, scalar_recommendation: str) -> str:
    _REC_SEVERITY = {"keep": 0, "keep_with_metadata_warning": 0, "review": 1, "exclude": 2}
    return (
        scalar_recommendation
        if _REC_SEVERITY.get(scalar_recommendation, 0) > _REC_SEVERITY.get(profile_recommendation, 0)
        else profile_recommendation
    )


def _risk_from_profile(qc_profile: dict) -> str:
    """Derive risk_level from domain severities.

    Structural domains (geometry_integrity, lesion_localization, lesion_burden,
    pancreas_context, region_consistency, attenuation_integrity) drive risk:
      critical or high_warning  → "high"
      moderate_warning          → "medium"
      low_warning or normal     → "low"
    metadata_completeness alone never raises risk above "low"
    (image-only segmentation: missing metadata is not a training-data defect).
    """
    _structural = [d for d in _CANONICAL_DOMAINS if d != "metadata_completeness"]
    worst_structural = max(
        (_severity_rank(qc_profile.get(d, {}).get("severity", "normal"))
         for d in _structural),
        default=0,
    )
    if worst_structural >= _severity_rank("high_warning"):
        return "high"
    if worst_structural >= _severity_rank("moderate_warning"):
        return "medium"
    return "low"


def _profile_from_domains(qc_domains: dict) -> dict:
    return {
        domain: {
            "score": info.get("component_score", 0),
            "severity": info.get("severity", "normal"),
            "evidence": [],
        }
        for domain, info in qc_domains.items()
        if domain in _CANONICAL_DOMAINS
    }


def _threshold_metadata(thr: "dict | None") -> dict:
    meta = dict((thr or {}).get("THRESHOLD_METADATA") or {})
    meta.setdefault("method", "deterministic")
    meta.setdefault("source", "manual")
    return meta


def _effective_thresholds(thr: "dict | None") -> dict:
    if not thr:
        return {}
    keys = (
        "MAX_QC_SCORE_FOR_KEEP",
        "MAX_QC_SCORE_FOR_REVIEW",
        "SCORE_WEIGHTS",
        "FOV_SCORE_POINTS",
        "REGION_SCORE_POINTS",
        "FOV_POLICY",
        "METADATA_POLICY",
        "QC_PROFILE_THRESHOLDS",
    )
    return {k: thr[k] for k in keys if k in thr}


def _domain_verdict(domain: str, severity: str, triage_recommendation: str) -> str:
    """Return a domain-local verdict aligned with domain semantics.

    metadata_completeness is informational for all current task modes, so it
    should mirror the task-level keep/keep_with_metadata_warning behavior rather
    than the generic review/exclude ladder used by structural domains.
    """
    if domain == "metadata_completeness":
        if triage_recommendation == "keep_with_metadata_warning":
            return "keep_with_metadata_warning"
        return "keep"
    if severity == "critical":
        return "exclude"
    if severity in ("high_warning", "moderate_warning"):
        return "review"
    return "keep"


def _classify_sample(
    pancreas_present: bool,
    lesion_present: bool,
    task_mode: str = "pancreas_lesion",
) -> str:
    """Return a sample type relative to the active training target.

    Lesion-task behavior remains positive/negative. Pancreas-only cases cannot
    become valid negatives because pancreas foreground is required supervision.
    """
    if task_mode == "pancreas_only":
        return "positive" if pancreas_present else "missing_required_target"
    return "positive" if lesion_present else "negative"


def _task_aware_decision_hint(
    pancreas_present: bool,
    lesion_present: bool,
    ov_rate: float | None,
    thr_min: float,
    thr_strict: float,
    task_mode: str = "pancreas_lesion",
) -> tuple[str, list[str]]:
    """Compute task-aware decision_hint and explanatory reasons.

    Four canonical cases for pancreas_lesion_segmentation:
      1. pancreas+  lesion+  → overlap-based keep/review/exclude
      2. pancreas+  lesion-  → keep_negative (valid negative sample)
      3. pancreas-  lesion-  → keep_negative (background negative)
      4. pancreas-  lesion+  → anatomical validation impossible → review_positive

    Returns:
        (decision_hint, [reason strings])
    """
    reasons: list[str] = []

    if task_mode == "pancreas_only":
        if pancreas_present:
            return "keep_positive", [
                "Required primary target pancreas is annotated"
            ]
        return "exclude_missing_required_target", [
            "Required primary target pancreas is empty or missing"
        ]

    if not lesion_present:
        # Cases 2 and 3: negative sample regardless of pancreas status
        if not pancreas_present:
            reasons.append(
                "No lesion and no pancreas mask: valid background negative sample")
        else:
            reasons.append(
                "Pancreas present, no lesion: valid negative sample")
        return "keep_negative", reasons

    # lesion_present = True from here on (positive sample)
    if not pancreas_present:
        # Case 4: cannot validate lesion location anatomically
        reasons.append(
            "Pancreas mask absent: cannot anatomically validate lesion location")
        return "review_positive", reasons

    # Case 1: both present — use overlap
    if ov_rate is None:
        reasons.append(
            "Lesion present but overlap with pancreas could not be computed")
        return "review_positive", reasons

    if ov_rate >= thr_strict:
        reasons.append(
            f"Lesion-pancreas overlap {ov_rate:.1%} ≥ strict threshold "
            f"{thr_strict:.0%}: good anatomical agreement")
        return "keep_positive", reasons
    elif ov_rate >= thr_min:
        reasons.append(
            f"Lesion-pancreas overlap {ov_rate:.1%} is between minimum "
            f"{thr_min:.0%} and strict {thr_strict:.0%}: moderate agreement")
        return "review_positive", reasons
    else:
        reasons.append(
            f"Lesion-pancreas overlap {ov_rate:.1%} < minimum threshold "
            f"{thr_min:.0%}: lesion may be outside pancreas")
        return "review_or_exclude_positive", reasons


def _detect_dataset_task_mode(data: dict) -> str:
    """Auto-detect task mode by scanning summary data for sub-region annotations.

    Returns "pancreas_lesion_subregions" if \u22651% of cases that have a
    pancreas also have non-zero head/body/tail sub-region volume; otherwise
    returns "pancreas_lesion".
    """
    cases_with_pancreas   = 0
    cases_with_subregions = 0
    for d in data.values():
        pan  = d.get("pancreas") or {}
        pvol = pan.get("volume_mm3") or 0.0
        if pvol > 0:
            cases_with_pancreas += 1
            reg = pan.get("regions") or {}
            if any(
                ((reg.get(rn) or {}).get("volume_mm3") or 0.0) > 0
                for rn in ("head", "body", "tail")
            ):
                cases_with_subregions += 1
    if cases_with_pancreas == 0:
        return "pancreas_lesion"
    ratio = cases_with_subregions / cases_with_pancreas
    return "pancreas_lesion_subregions" if ratio >= 0.01 else "pancreas_lesion"


def _get_task_mode(thr: "dict | None", data: "dict | None" = None) -> str:
    """Return the effective task mode, with auto-detection fallback.

    Priority:
      1. TASK_MODE in thresholds.yaml (unless value is "auto").
      2. Legacy REQUIRE_SUBREGIONS key.
      3. Auto-detect from summary data (when data is provided or mode=="auto").
      4. Hardcoded default: "pancreas_lesion_subregions" (backward-compat).

    Supported TASK_MODE values:
        "auto"                        — detect from summary data (recommended default).
        "pancreas_lesion"             — skip all sub-region checks.
        "pancreas_lesion_subregions"  — full sub-region QC.
    """
    if thr:
        if "TASK_MODE" in thr:
            mode = str(thr["TASK_MODE"]).strip().lower()
            if mode != "auto":
                return mode
            # "auto" falls through to data-driven detection below
        elif "REQUIRE_SUBREGIONS" in thr:
            return (
                "pancreas_lesion_subregions"
                if thr["REQUIRE_SUBREGIONS"]
                else "pancreas_lesion"
            )
    if data is not None:
        return _detect_dataset_task_mode(data)
    return "pancreas_lesion_subregions"   # hardcoded fallback (backward-compat)


def _subregion_data_present(reg: dict, rc: dict) -> bool:
    """Return True if at least one sub-region mask has non-zero volume."""
    return any(
        ((reg.get(rn) or {}).get("volume_mm3") or 0.0) > 0
        for rn in ("head", "body", "tail")
    )


def _partial_visibility_assessment(
    pan: dict,
    reg: dict,
    img: dict,
    use_subregion_coverage: bool = False,
) -> dict:
    """
    Assess whether partial pancreas visibility is likely due to limited
    field of view, acquisition truncation, or incomplete annotation.

    Uses the pancreas bounding box (from <dataset>_summary.json) to detect
    whether the mask touches image borders, and falls back to a volume + missing-
    region heuristic when bbox is unavailable.

    Returns
    -------
    dict with keys:
        border_touching              — per-axis boolean flags (6 keys)
        any_border_touch             — bool
        region_coverage_ratio        — float  (fraction of pancreas vol covered
                                               by union of sub-region overlaps)
        regions_present              — list[str]  (non-empty region names)
        partial_visibility_likely    — bool
        anatomical_truncation_suspected — bool
    """
    shape = img.get("shape_zyx")          # [nz, ny, nx]
    bbox  = pan.get("bounding_box_zyx")   # [[z_min,y_min,x_min],[z_max,y_max,x_max]]
    pvol  = pan.get("volume_mm3") or 0.0

    border_touching: dict[str, bool] = {
        "touches_z_min": False, "touches_z_max": False,
        "touches_y_min": False, "touches_y_max": False,
        "touches_x_min": False, "touches_x_max": False,
    }

    if bbox is not None and shape is not None:
        bmin, bmax = bbox[0], bbox[1]     # each is [z, y, x]
        nz, ny, nx = shape
        m = BORDER_MARGIN_VOXELS
        border_touching["touches_z_min"] = bmin[0] <= m
        border_touching["touches_z_max"] = bmax[0] >= nz - 1 - m
        border_touching["touches_y_min"] = bmin[1] <= m
        border_touching["touches_y_max"] = bmax[1] >= ny - 1 - m
        border_touching["touches_x_min"] = bmin[2] <= m
        border_touching["touches_x_max"] = bmax[2] >= nx - 1 - m

    any_border_touch = any(border_touching.values())

    # Coverage ratio: fraction of pancreas explained by union of sub-regions
    # (approximation: sum of per-region overlaps, valid when regions don't
    # overlap each other significantly)
    o_h = (reg.get("head") or {}).get("overlap_with_pancreas_mm3") or 0.0
    o_b = (reg.get("body") or {}).get("overlap_with_pancreas_mm3") or 0.0
    o_t = (reg.get("tail") or {}).get("overlap_with_pancreas_mm3") or 0.0
    coverage_mm3   = o_h + o_b + o_t
    coverage_ratio = round(coverage_mm3 / pvol, 4) if use_subregion_coverage and pvol > 0 else None

    regions_present = [
        rn for rn in ("head", "body", "tail")
        if ((reg.get(rn) or {}).get("volume_mm3") or 0.0) > 0
    ]
    n_present = len(regions_present)

    # Heuristic: partial visibility is likely when:
    #   A) bbox directly touches image border (hard evidence), OR
    #   B) only a subset of expected regions is present AND the pancreas
    #      is small (possibly because the scan captured only part of it)
    small_pancreas = pvol < THR_PARTIAL_PAN_VOLUME
    partial_visibility_likely = any_border_touch or (
        use_subregion_coverage and n_present < 3 and small_pancreas
    )
    anatomical_truncation_suspected = (
        use_subregion_coverage and any_border_touch and n_present < 3
    )

    return {
        "border_touching":                  border_touching,
        "any_border_touch":                 any_border_touch,
        "region_coverage_ratio":            coverage_ratio,
        "regions_present":                  regions_present,
        "partial_visibility_likely":        partial_visibility_likely,
        "anatomical_truncation_suspected":  anatomical_truncation_suspected,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CORE QC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════

def run_qc(data: dict, thr: dict | None = None, score_version: str = "v2") -> dict:
    """
    Analyse every case and return:
        {case_id: {flags, score, risk_level, evidence, recommendation,
                   sample_type, decision_hint, anatomical_validation_possible,
                   task_aware_reasons, score_version, scoring_details}}
    thr: optional threshold override dict (keys as in ThresholdConfig).
    score_version: "v2" (additive, default) or "hybrid" (additive geometry
        gate + multiplicative secondary amplifier).

    Task-aware logic (v3) for TRAINING_OBJECTIVE='pancreas_lesion_segmentation':
      All four combinations of pancreas/lesion presence are handled
      explicitly instead of treating empty pancreas as always critical.
    """
    thr = dict(thr or {})
    thr.setdefault("FOV_SCORE_POINTS", {
        "border_touching": 2,
        "partial_visibility_likely": 4,
        "anatomical_truncation_suspected": 8,
    })
    thr.setdefault("REGION_SCORE_POINTS", {
        "incomplete_anatomical_coverage": 6,
        "subregions_expected_but_missing": 10,
        "missing_subregion": 4,
    })
    thr.setdefault("FOV_POLICY", {
        "border_touching_severity": "low_warning",
        "missing_region_severity": "moderate_warning",
        "incomplete_coverage_severity": "moderate_warning",
        "truncation_severity": "high_warning",
    })

    def _t(key: str, default: float) -> float:
        return float(thr[key]) if thr and key in thr else default

    t = {
        "region_err_crit" : _t("MAX_REGION_RELATIVE_ERROR", THR_REGION_ERR_CRIT),
        "region_err_warn" : _t("MAX_REGION_RELATIVE_ERROR", THR_REGION_ERR_WARN) / 3,
        # Task-aware lesion-overlap thresholds
        "lesion_strict"   : _t("STRICT_LESION_PANCREAS_OVERLAP", THR_LESION_STRICT),
        "lesion_min"      : _t("MIN_LESION_PANCREAS_OVERLAP",    THR_LESION_MIN),
        # Legacy compat (kept for flag labelling)
        "lesion_crit"     : _t("MIN_LESION_PANCREAS_OVERLAP", THR_LESION_CRIT) * 0.2,
        "lesion_warn"     : _t("MIN_LESION_PANCREAS_OVERLAP", THR_LESION_WARN),
        "pan_crit"        : _t("MIN_PANCREAS_VOLUME_MM3", THR_PAN_CRIT) * 0.5,
        "pan_warn"        : _t("MIN_PANCREAS_VOLUME_MM3", THR_PAN_WARN),
        "pan_large"       : _t("MAX_PANCREAS_VOLUME_MM3", THR_PAN_LARGE),
        "les_min"         : _t("MIN_LESION_VOLUME_MM3", THR_LES_MIN),
        "les_max"         : _t("MAX_LESION_VOLUME_MM3", THR_LES_MAX),
        "les_pan_ratio"   : _t("MAX_LESION_PANCREAS_RATIO", THR_LES_PAN_RATIO),
    }

    # Determine effective task mode once for the whole dataset.
    # Passing `data` enables auto-detection when TASK_MODE="auto" (or absent).
    _dataset_task_mode = _get_task_mode(thr, data)

    results: dict = {}

    for case_id, d in data.items():
        # ── Omitted cases: CT unreadable / file missing ─────────────────────────
        cs = d.get("case_status") or {}
        if cs.get("omit_from_training", False):
            _etype   = cs.get("error_type", "unknown_error")
            _estage  = cs.get("error_stage", "unknown")
            _emsg    = cs.get("error_message", "")
            results[case_id] = {
                "flags": [
                    ("CRITICAL", "omit_from_training",
                     f"Case omitted from training: {_etype} at {_estage}"),
                ],
                "score"                         : 100,
                "score_components"              : {
                    "lesion_localization"  : 0,
                    "geometry_integrity"   : 100,
                    "lesion_burden"        : 0,
                    "pancreas_context"     : 0,
                    "fov_integrity"        : 0,
                    "region_consistency"   : 0,
                    "metadata_completeness": 0,
                    "attenuation_integrity": 0,
                },
                "dominant_failure_mode"         : "geometry_integrity_failure",
                "risk_interpretation"           : {
                    "geometry_integrity": {
                        "severity"   : "severe",
                        "description": f"CT image cannot be loaded: {_emsg[:120]}",
                    },
                },
                "risk_level"                    : "critical",
                "evidence"                      : {
                    "pancreas_present"         : False,
                    "lesion_present"           : False,
                    "pancreas_volume_mm3"      : 0.0,
                    "lesion_volume_mm3"        : 0.0,
                    "overlap_rate_vs_lesion"   : None,
                    "suspicious_hu_distribution": False,
                    "omit_from_training"       : True,
                    "error_type"               : _etype,
                    "error_stage"              : _estage,
                },
                "recommendation"                : "omit_from_training",
                "sample_type"                   : "omitted",
                "decision_hint"                 : "omit_from_training",
                "anatomical_validation_possible": False,
                "task_aware_reasons"            : [
                    f"Case omitted: {cs.get('omit_reason', 'unknown')}",
                ],
                # QC profile: omitted cases always fail geometry_integrity (critical)
                "qc_profile"                    : {
                    "geometry_integrity"   : {"score": 100.0, "severity": "critical",
                                               "evidence": [{"tag": "omitted", "severity": "critical",
                                                             "message": "Case omitted from training"}]},
                    "lesion_localization"  : {"score":   0.0, "severity": "normal",  "evidence": []},
                    "lesion_burden"        : {"score":   0.0, "severity": "normal",  "evidence": []},
                    "pancreas_context"     : {"score":   0.0, "severity": "normal",  "evidence": []},
                    "fov_integrity"        : {"score":   0.0, "severity": "normal",  "evidence": []},
                    "region_consistency"   : {"score":   0.0, "severity": "normal",  "evidence": []},
                    "attenuation_integrity": {"score":   0.0, "severity": "normal",  "evidence": []},
                    "metadata_completeness": {"score":   0.0, "severity": "normal",  "evidence": []},
                },
                # Enriched qc_domains: single source of truth
                "qc_domains"                    : {
                    "geometry_integrity"   : {"component_score": 100, "severity": "critical",  "verdict": "exclude",
                                               "subchecks": {}},
                    "lesion_localization"  : {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                    "lesion_burden"        : {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                    "pancreas_context"     : {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                    "fov_integrity"        : {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                    "region_consistency"   : {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                    "attenuation_integrity": {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                    "metadata_completeness": {"component_score":   0, "severity": "normal",   "verdict": "keep",
                                               "subchecks": {}},
                },
                "triage"                        : {
                    "score":          100,
                    "rank_hint":      "omit",
                    "risk_level":     "critical",
                    "recommendation": "omit_from_training",
                    "score_version":  "v2",
                },
                "primary_issue"                 : {
                    "domain":       "geometry_integrity",
                    "failure_mode": "geometry_integrity_failure",
                },
                "driving_domain"                : "geometry_integrity",
                "task_mode"                     : _dataset_task_mode,
                "training_objective"             : _training_objective(_dataset_task_mode),
            }
            continue

        flags: list[tuple[str, str, str]] = []
        evidence: dict[str, Any] = {}

        qc   = d.get("quality_control") or {}
        pan  = d.get("pancreas") or {}
        les  = d.get("lesions") or {}
        img  = d.get("image") or {}
        meta = d.get("metadata") or {}
        reg  = pan.get("regions") or {}
        rc   = pan.get("region_consistency") or {}

        pvol     = pan.get("volume_mm3") or 0.0
        total_lv = les.get("total_volume_mm3") or 0.0
        n_les    = les.get("n_lesions") or 0
        # Use the reference declared by summarize_dataset (task-mode-aware);
        # fall back to total_overlap_rate_vs_lesion for backward compatibility.
        _loa      = qc.get("lesion_overlap_analysis") or {}
        _ref_key  = _loa.get("overlap_reference_used_for_qc") or "combined_filled_pancreas"
        _ref_rate = (_loa.get(_ref_key) or {}).get("overlap_rate_vs_lesion")
        ov_rate   = _ref_rate if _ref_rate is not None else qc.get("total_overlap_rate_vs_lesion")
        spacing  = img.get("spacing_xyz_mm")
        shape    = img.get("shape_zyx")

        # ── Derive boolean presence flags ─────────────────────────────────
        pancreas_present = not bool(qc.get("pancreas_mask_empty")) and pvol > 0
        lesion_present   = not (qc.get("lesion_mask_empty") or
                                (n_les == 0 and total_lv == 0))

        # Record primary numeric evidence used by all downstream consumers
        evidence["pancreas_present"]      = pancreas_present
        evidence["lesion_present"]        = lesion_present
        evidence["pancreas_volume_mm3"]   = pvol
        evidence["lesion_volume_mm3"]     = total_lv
        evidence["overlap_rate_vs_lesion"] = ov_rate
        if _loa:
            evidence["hole_annotation_detected"] = _loa.get("hole_filling_changed_overlap", False)
            evidence["regions_added_overlap"]    = _loa.get("regions_added_overlap", False)
            evidence["overlap_ref_used_for_qc"]  = _loa.get("overlap_reference_used_for_qc")

        # ── 1. Task-aware classification (v3 core logic) ──────────────────
        # This block determines sample_type and decision_hint using the
        # four canonical cases for pancreas_lesion_segmentation.
        # It does NOT replace structural QC checks below; those remain
        # independent and cumulative.

        sample_type = _classify_sample(
            pancreas_present, lesion_present, _dataset_task_mode
        )
        decision_hint, task_reasons = _task_aware_decision_hint(
            pancreas_present, lesion_present, ov_rate,
            t["lesion_min"], t["lesion_strict"],
            _dataset_task_mode,
        )
        anatomical_validation_possible = pancreas_present

        # Add task-aware flags (only for cases that need attention)
        if _dataset_task_mode == "pancreas_only":
            if not pancreas_present:
                flags.append(("CRITICAL", "required_target_missing",
                              "Required primary target pancreas is empty or missing"))
        elif not lesion_present:
            # Negative sample: informational only, no penalty
            flags.append(("INFO", "lesion_mask_empty",
                           "No lesion mask — valid negative sample"))
        else:
            # Positive sample: evaluate overlap quality
            if not pancreas_present:
                # Case 4: lesion present, pancreas absent → cannot validate
                flags.append(("WARNING", "lesion_no_anatomical_validation",
                               "Lesion present but pancreas mask absent: "
                               "anatomical validation impossible"))
            elif ov_rate is not None:
                evidence["overlap_rate_vs_lesion"] = ov_rate
                evidence["total_lesion_volume_mm3"] = total_lv
                # Informational annotation-convention diagnostics
                if _loa.get("hole_filling_changed_overlap"):
                    flags.append(("INFO", "pancreas_hole_annotation_suspected",
                                   "Overlap improved >10 pp after hole-filling — "
                                   "pancreas mask likely uses lesion-as-hole "
                                   "annotation convention"))
                if _loa.get("regions_added_overlap"):
                    flags.append(("INFO", "lesion_in_subregion_spillover",
                                   "Overlap improved >10 pp after adding sub-region "
                                   "masks — lesion may be in a zone not covered by "
                                   "the total pancreas mask"))
                if ov_rate < t["lesion_min"]:
                    # Below minimum threshold: one canonical flag, details in evidence.
                    sev = "CRITICAL" if ov_rate < t["lesion_crit"] else "WARNING"
                    flags.append((sev, "lesion_overlap_critical_fail",
                                   f"Lesion-pancreas overlap {ov_rate:.1%} < "
                                   f"minimum {t['lesion_min']:.0%}; "
                                   f"lesion={total_lv:.0f} mm3"))
                    # Alias tags kept for DataCurationAgent / interactive assistant
                    # compatibility — INFO so they do not appear in non-INFO reports.
                    if ov_rate < t["lesion_crit"]:
                        flags.append(("INFO", "lesion_outside_pancreas",
                                       f"alias:lesion_overlap_critical_fail "
                                       f"(overlap={ov_rate:.1%})"))
                    else:
                        flags.append(("INFO", "lesion_partial_outside",
                                       f"alias:lesion_overlap_critical_fail "
                                       f"(overlap={ov_rate:.1%})"))
                elif ov_rate < t["lesion_strict"]:
                    # Between min and strict: one canonical flag.
                    flags.append(("WARNING", "lesion_overlap_strict_fail",
                                   f"Lesion-pancreas overlap {ov_rate:.1%} < "
                                   f"strict threshold {t['lesion_strict']:.0%}; "
                                   f"lesion={total_lv:.0f} mm3"))
                    # INFO alias for interactive queries
                    flags.append(("INFO", "lesion_partial_outside",
                                   f"alias:lesion_overlap_strict_fail "
                                   f"(overlap={ov_rate:.1%})"))

        # ── 2. Pancreas mask status (informational for negatives) ─────────
        # CHANGED in v3: empty pancreas on a negative case is INFO, not CRITICAL.
        if not pancreas_present and _dataset_task_mode != "pancreas_only":
            if lesion_present:
                # Already flagged above as WARNING; add CRITICAL for scoring
                flags.append(("CRITICAL", "pancreas_mask_empty",
                               "Pancreas mask empty with lesion present: "
                               "cannot validate lesion location"))
            else:
                flags.append(("INFO", "pancreas_mask_empty",
                               "Pancreas mask empty — case treated as background negative"))

        # ── 3-5. Sub-region checks (task-mode-dependent) ──────────────────────
        # task_mode controls whether sub-region QC runs at all.
        # Section 6 (pancreas volume extremes) always runs when pancreas is present.
        subregion_qc: dict = {
            "enabled": False,
            "reason": "pancreas_not_present",
            "status": "not_applicable",
        }
        if pancreas_present:
            task_mode          = _dataset_task_mode
            _subregions_found  = _subregion_data_present(reg, rc)
            # Always compute partial visibility: border_touching is used by
            # the volume/FOV integrity block regardless of task_mode.
            pv = _partial_visibility_assessment(
                pan, reg, img,
                use_subregion_coverage=(
                    task_mode == "pancreas_lesion_subregions" and _subregions_found
                ),
            )
            evidence["border_touching"] = pv["border_touching"]
            evidence["partial_visibility_likely"] = pv["partial_visibility_likely"]
            evidence["anatomical_truncation_suspected"] = pv["anatomical_truncation_suspected"]

            if task_mode in ("pancreas_only", "pancreas_lesion"):
                # Sub-region QC explicitly disabled — one INFO note, no penalties.
                flags.append(("INFO", "subregions_not_applicable",
                               f"Subregion QC disabled: TASK_MODE={task_mode}"))
                subregion_qc = {
                    "enabled": False,
                    "reason": f"TASK_MODE={task_mode}",
                    "status": "not_applicable",
                }

            elif not _subregions_found:
                # Safety check: mode expects sub-regions but none detected.
                flags.append(("WARNING", "subregions_expected_but_missing",
                               "TASK_MODE=pancreas_lesion_subregions but no "
                               "head/body/tail annotations found — "
                               "region checks skipped to avoid false penalties"))
                subregion_qc = {
                    "enabled": True,
                    "status": "skipped_no_data",
                    "reason": "TASK_MODE requires subregions but none detected",
                }

            else:
                # ── 3. Sub-region empty masks ──────────────────────────────
                evidence["region_coverage_ratio"] = pv["region_coverage_ratio"]

                for rname in ("head", "body", "tail"):
                    r  = reg.get(rname) or {}
                    rv = r.get("volume_mm3") or 0.0
                    evidence[f"region_{rname}_volume_mm3"] = rv
                    if rv == 0:
                        if pv["partial_visibility_likely"]:
                            flags.append(("INFO", f"{rname}_region_empty",
                                           f"Pancreas {rname} sub-region absent — "
                                           f"likely partial anatomical coverage"))
                        else:
                            flags.append(("WARNING", f"{rname}_region_empty",
                                           f"Pancreas {rname} sub-region mask is empty"))

                # ── 4. Sub-region sum consistency ──────────────────────────
                rel_err = rc.get("relative_error_vs_pancreas") or 0.0
                evidence["region_relative_error"] = rel_err
                if rel_err > t["region_err_crit"]:
                    if pv["partial_visibility_likely"]:
                        _trunc_note = (
                            "border-truncated FOV"
                            if pv["anatomical_truncation_suspected"]
                            else "partial visibility suspected"
                        )
                        flags.append(("WARNING", "incomplete_anatomical_coverage",
                                       f"Partial anatomical coverage: "
                                       f"{pv['region_coverage_ratio']:.1%} of pancreas "
                                       f"explained by regions ({_trunc_note}). "
                                       f"Region sum deviates {rel_err:.1%}"))
                    else:
                        flags.append(("CRITICAL", "region_sum_error_critical",
                                       f"Head+body+tail deviates {rel_err:.1%} from total pancreas"))
                elif rel_err > t["region_err_warn"]:
                    flags.append(("WARNING", "region_sum_error_warning",
                                   f"Head+body+tail deviates {rel_err:.1%} from total pancreas"))

                # ── 5. Sub-region mutual overlaps ──────────────────────────
                for r1, r2 in (("head","body"), ("head","tail"), ("body","tail")):
                    k   = f"{r1}_{r2}_overlap_mm3"
                    val = rc.get(k) or 0.0
                    evidence[k] = val
                    if val > 0:
                        flags.append(("WARNING", f"{r1}_{r2}_overlap",
                                       f"{r1.capitalize()}-{r2} overlap: {val:.0f} mm3"))

                # Build subregion_qc summary for this case
                _region_tags = {
                    "region_sum_error_critical", "region_sum_error_warning",
                    "incomplete_anatomical_coverage",
                    "head_region_empty", "body_region_empty", "tail_region_empty",
                    "head_body_overlap", "head_tail_overlap", "body_tail_overlap",
                }
                _active = [(sev, tag) for sev, tag, _ in flags if tag in _region_tags]
                _region_status = "passed"
                if any(sev == "CRITICAL" for sev, _ in _active):
                    _region_status = "failed"
                elif any(sev == "WARNING" for sev, _ in _active):
                    _region_status = "warning"
                subregion_qc = {
                    "enabled": True,
                    "status": _region_status,
                    "region_coverage_ratio": pv["region_coverage_ratio"],
                    "region_relative_error": evidence.get("region_relative_error", 0.0),
                }

            # ── 6. Pancreas volume extremes (always, independent of task_mode) ──
            if 0 < pvol < t["pan_crit"]:
                flags.append(("CRITICAL", "very_small_pancreas",
                               f"Pancreas volume {pvol:.0f} mm3 (< {t['pan_crit']:.0f} mm3)"))
            elif 0 < pvol < t["pan_warn"]:
                flags.append(("WARNING", "small_pancreas",
                               f"Pancreas volume {pvol:.0f} mm3 (< {t['pan_warn']:.0f} mm3)"))
            if pvol > t["pan_large"]:
                flags.append(("WARNING", "large_pancreas",
                               f"Pancreas volume {pvol:.0f} mm3 (> {t['pan_large']:.0f} mm3)"))

        # ── 7. Lesion volume extremes (independent of pancreas) ───────────
        if lesion_present and total_lv > 0:
            evidence["total_lesion_volume_mm3"] = total_lv
            if total_lv < t["les_min"]:
                flags.append(("WARNING", "very_small_lesion",
                               f"Total lesion volume {total_lv:.1f} mm3 "
                               f"(< {t['les_min']} mm3)"))
            if total_lv > t["les_max"]:
                flags.append(("WARNING", "very_large_lesion",
                               f"Total lesion volume {total_lv:.0f} mm3 "
                               f"(> {t['les_max']:.0f} mm3)"))

        # ── 8. Lesion-to-pancreas ratio (only when both present) ──────────
        if pancreas_present and lesion_present and pvol > 0 and total_lv > 0:
            ratio = total_lv / pvol
            evidence["lesion_pancreas_ratio"] = ratio
            if ratio > t["les_pan_ratio"]:
                flags.append(("WARNING", "high_lesion_pancreas_ratio",
                               f"Lesion/pancreas ratio {ratio:.2f} "
                               f"(> {t['les_pan_ratio']:.2f})"))

        # ── 9. Lesion sub-region localisation (informational) ────────────
        # Sum overlap_with_regions across all per-lesion components so that
        # multi-focal cases report the combined overlap per sub-region.
        if lesion_present:
            per_lesion = les.get("per_lesion") or []
            _ov_head = sum(
                (c.get("overlap_with_regions") or {}).get("head", {}).get("volume_mm3", 0.0) or 0.0
                for c in per_lesion
            )
            _ov_body = sum(
                (c.get("overlap_with_regions") or {}).get("body", {}).get("volume_mm3", 0.0) or 0.0
                for c in per_lesion
            )
            _ov_tail = sum(
                (c.get("overlap_with_regions") or {}).get("tail", {}).get("volume_mm3", 0.0) or 0.0
                for c in per_lesion
            )
            evidence["h_e_overlap_mm3"]   = _ov_head
            evidence["b_o_overlap_mm3"]   = _ov_body
            evidence["t_a_overlap_mm3"]   = _ov_tail

        # ── 10. Invalid / anisotropic spacing (always checked) ─────────────
        if spacing:
            evidence["spacing_xyz_mm"] = spacing
            try:
                sp = [float(s) for s in spacing]
                bad = any(s <= 0 or s > 10 for s in sp)
                anisotropic = len(sp) >= 2 and (max(sp) / (min(sp) + 1e-9)) > 5
                if bad:
                    flags.append(("CRITICAL", "invalid_spacing",
                                   f"Spacing contains invalid value: {spacing}"))
                elif anisotropic:
                    flags.append(("WARNING", "invalid_spacing",
                                   f"Highly anisotropic spacing {spacing}"))
            except (TypeError, ValueError):
                flags.append(("WARNING", "invalid_spacing",
                               f"Could not parse spacing: {spacing}"))

        # ── 11. Shape plausibility (always checked) ────────────────────────
        if shape:
            evidence["shape_zyx"] = shape
            if any(s < 10 for s in shape):
                flags.append(("CRITICAL", "shape_mismatch",
                               f"Implausible image shape {shape}"))

        # ── 12. Missing metadata ───────────────────────────────────────────
        missing_fields = [k for k in ("sex", "age", "ct_phase", "manufacturer")
                          if not meta.get(k)]
        if missing_fields:
            evidence["missing_metadata_fields"] = missing_fields
            flags.append(("INFO", "missing_metadata",
                           f"Missing metadata: {', '.join(missing_fields)}"))

        # ── 13. HU statistics validation (read-only — values from <dataset>_summary.json) ──
        # qc_agent.py never recomputes HU; it only consumes values produced by
        # summarize_dataset.py.  Flags only implausible HU distributions or
        # disagreements that warrant clinical review.
        hu_stats = d.get("hu_statistics") or {}

        # Primary fields (new schema); fall back to old names for backward compat
        tumor_median_hu   = (hu_stats.get("tumor_median_hu")
                             or hu_stats.get("mean_hu_tumor"))
        tumor_mean_hu     = (hu_stats.get("tumor_mean_hu")
                             or hu_stats.get("mean_hu_tumor"))
        tumor_attenuation = hu_stats.get("tumor_attenuation")
        delta_hu_val      = hu_stats.get("delta_hu_tumor_vs_pancreas")
        z_score_hu        = hu_stats.get("z_score_hu")
        suspicious_hu     = bool(hu_stats.get("suspicious_hu_distribution", False))
        rule_disagreement = bool(hu_stats.get("attenuation_rule_disagreement", False))

        if lesion_present:
            if tumor_median_hu is not None:
                evidence["tumor_median_hu"]    = tumor_median_hu
                evidence["tumor_attenuation"]  = tumor_attenuation
                evidence["delta_hu"]           = delta_hu_val
                evidence["z_score_hu"]         = z_score_hu
                # Flag HU values outside the plausible soft-tissue CT range.
                # Normal pancreas/lesion HU: roughly -100 to +200 HU.
                # Allow [-200, 400] to accommodate fat, fibrosis, and calcification.
                if not (-200 <= tumor_median_hu <= 400):
                    suspicious_hu = True
                    flags.append(("WARNING", "suspicious_hu_distribution",
                                   f"Tumor median HU {tumor_median_hu:.1f} outside expected "
                                   f"soft-tissue range [-200, 400] — verify CT image or annotation"))
                elif tumor_median_hu != tumor_median_hu:   # NaN guard
                    suspicious_hu = True
                    flags.append(("WARNING", "suspicious_hu_distribution",
                                   "NaN / invalid tumor HU statistics detected"))
            if suspicious_hu and not any(t == "suspicious_hu_distribution" for _, t, _ in flags):
                flags.append(("INFO", "suspicious_hu_distribution",
                               "Suspicious HU distribution flagged by summarizer "
                               "(small sample, high std, or mean–median discrepancy)"))
            if rule_disagreement:
                flags.append(("INFO", "attenuation_rule_disagreement",
                               f"Delta-based ({hu_stats.get('delta_hu_attenuation')}) and "
                               f"z-score ({hu_stats.get('tumor_attenuation')}) attenuation "
                               f"labels disagree — review HU distribution"))
        # Negative cases: record pancreas attenuation context but never flag
        # missing tumor HU as suspicious — it is expected.
        evidence["suspicious_hu_distribution"] = suspicious_hu if lesion_present else False

        # ── Compute QC score via named components ─────────────────────────
        # _region_raw is informational only; excluded from the sum.
        # region_consistency is capped at 15 (task = pancreas_lesion_seg).
        score_components = _compute_score_components(
            flags, ov_rate, lesion_present, pancreas_present, evidence,
            rel_err=evidence.get("region_relative_error") or 0.0,
            thr=thr,
        )

        # ── Task-mode region-consistency uncapping ────────────────────────
        # For pancreas_lesion_subregions, region consistency must be evaluated
        # for ALL cases (including positive ones), because validating the
        # head/body/tail decomposition is the explicit goal of this mode.
        # The default cap of 0 for positive cases is overridden here.
        if _dataset_task_mode == "pancreas_lesion_subregions":
            score_components["region_consistency"] = min(
                int(score_components.get("_region_raw",
                    score_components["region_consistency"])),
                _COMPONENT_CAPS["region_consistency"],
            )

        # ── Per-mode zero mask from SCORE_WEIGHTS ─────────────────────────
        # Components whose weight == 0.0 for this task mode are zeroed out
        # so that irrelevant anatomy (e.g. lesion in pancreas_only mode) does
        # not contribute to either the score or the domain verdicts.
        _sw = (thr or {}).get("SCORE_WEIGHTS", {}).get(_dataset_task_mode, {})
        for _sw_key, _sw_weight in _sw.items():
            if float(_sw_weight) == 0.0 and _sw_key in score_components:
                score_components[_sw_key] = 0

        _scoring_details: dict | None = None
        if score_version == "hybrid":
            # Allow alpha and per-component weights to be overridden via thresholds.yaml.
            # Priority: SCORE_WEIGHTS[task_mode] > flat HYBRID_WEIGHT_* keys > defaults.
            _h_alpha = float(thr["HYBRID_ALPHA"]) if thr and "HYBRID_ALPHA" in thr else _HYBRID_ALPHA
            _h_beta  = float(thr["HYBRID_BETA"])  if thr and "HYBRID_BETA"  in thr else _HYBRID_BETA
            _h_weights = dict(_HYBRID_SECONDARY_WEIGHTS)
            # 1. Apply flat HYBRID_WEIGHT_* overrides (backward compat)
            if thr:
                _w_map = {
                    "HYBRID_WEIGHT_LESION_BURDEN"          : "lesion_burden",
                    "HYBRID_WEIGHT_PANCREAS_CONTEXT"       : "pancreas_context",
                    "HYBRID_WEIGHT_REGION_CONSISTENCY"     : "region_consistency",
                    "HYBRID_WEIGHT_METADATA"               : "metadata_completeness",
                    "HYBRID_WEIGHT_ATTENUATION_CONSISTENCY": "attenuation_integrity",
                }
                for _yaml_key, _score_key in _w_map.items():
                    if _yaml_key in thr:
                        _h_weights[_score_key] = float(thr[_yaml_key])
            # 2. If SCORE_WEIGHTS[task_mode] is defined, it takes final precedence
            #    (only the secondary-weight keys overlap; geometry/overlap handled
            #    structurally by the formula, not through _h_weights).
            _sw_mode = (thr or {}).get("SCORE_WEIGHTS", {}).get(_dataset_task_mode, {})
            _sw_secondary_keys = set(_HYBRID_SECONDARY_WEIGHTS.keys())
            for _sw_k, _sw_v in _sw_mode.items():
                if _sw_k in _sw_secondary_keys:
                    _h_weights[_sw_k] = float(_sw_v)
            score, _scoring_details = _compute_score_hybrid(
                score_components, alpha=_h_alpha, beta=_h_beta, secondary_weights=_h_weights
            )
        else:   # "v2" — original additive sum
            score = min(
                sum(v for k, v in score_components.items() if not k.startswith("_")),
                100,
            )

        dominant_failure_mode = _dominant_failure_mode(score_components)
        risk_interpretation   = _build_risk_interpretation(
            score_components, lesion_present, pancreas_present,
            hu_evidence=evidence,
            task_mode=_dataset_task_mode,
        )

        # ── v5 Multi-domain QC profile ──────────────────────────────────────
        # Build the new multi-dimensional profile before computing the
        # final recommendation.  The profile drives the decision; the scalar
        # triage score is used only for ranking / sorting.
        qc_profile = _build_qc_profile(flags, score_components, thr)

        # Gating recommendation from profile (domain-severity hierarchy)
        profile_recommendation, profile_driver = _recommend_from_profile(qc_profile, thr)

        # Risk level derived from worst domain severity
        risk = _risk_from_profile(qc_profile)

        # ── Scalar-based recommendation (backward compat fallback) ─────────
        # Primary gate: score-based (0-100 scale).
        # Safety net: any CRITICAL-severity flag always escalates to "review".
        scalar_recommendation = _scalar_recommendation(score, flags, thr)

        # Merge: profile recommendation is authoritative; scalar may escalate.
        # keep_with_metadata_warning has the same escalation priority as "keep"
        # so that a high score (scalar "review") can still override it.
        recommendation = _merge_recommendations(profile_recommendation, scalar_recommendation)

        if _dataset_task_mode == "pancreas_only" and not pancreas_present:
            recommendation = "exclude"

        # Override: negative samples with only INFO flags are always keep
        if (_dataset_task_mode != "pancreas_only" and
                sample_type == "negative" and recommendation == "exclude"):
            non_info_tags = {tag for sev, tag, _ in flags if sev != "INFO"}
            if non_info_tags <= {"pancreas_mask_empty"}:
                recommendation = "keep"

        # Driving domain: use profile driver if set, else note scalar escalation
        driving_domain_out: "str | None" = profile_driver if profile_driver is not None else (
            "scalar_score" if recommendation not in ("keep", "keep_with_metadata_warning") else None
        )

        # Primary issue (v6 — single source of truth, replaces dominant_failure_mode + driving_domain)
        primary_issue = {
            "domain":       driving_domain_out,
            "failure_mode": dominant_failure_mode if dominant_failure_mode != "none" else None,
        }

        # ── Build enriched qc_domains (single source of truth) ────────────
        # Merges qc_profile (score + severity + evidence) with component list + verdict.
        # subchecks (geometry_integrity, fov_integrity, etc.) are added in
        # generate_json_report() where raw data is available.
        qc_domains = {
            domain: {
                "component_score": round(info["score"], 2),
                "severity":        info["severity"],
                "verdict":        _domain_verdict(domain, info["severity"], recommendation),
                "subchecks": {},
            }
            for domain, info in qc_profile.items()
        }
        if not subregion_qc.get("enabled", False):
            qc_domains["region_consistency"] = {
                "component_score": 0,
                "severity":        "not_applicable",
                "verdict":         "not_applicable",
                "enabled":         False,
                "subchecks":       {},
            }

        # ── Triage block (v6) — primary decision block ─────────────────────
        # rank_hint uses structural severity only; metadata alone must not
        # produce "priority_review".
        _worst_structural_sev = max(
            (info["severity"] for domain, info in qc_profile.items()
             if domain != "metadata_completeness"),
            key=lambda s: _SEVERITY_ORDER.index(s),
            default="normal",
        )
        _rank_hint = (
            "omit"                  if recommendation == "exclude" else
            "metadata_warning"      if recommendation == "keep_with_metadata_warning" else
            "priority_review"       if _worst_structural_sev in ("high_warning", "critical") else
            "review"                if recommendation == "review" else
            "low_priority"          if _worst_structural_sev == "low_warning" else
            "routine"
        )
        _score_ver_str = (
            "hybrid-additive-geometry-multiplicative-secondary"
            if score_version == "hybrid" else "v2"
        )
        triage: dict = {
            "score":          score,
            "rank_hint":      _rank_hint,
            "risk_level":     risk,
            "recommendation": recommendation,
            "score_version":  _score_ver_str,
        }
        if _scoring_details:
            triage["scoring_details"] = _scoring_details

        # decision_hint: legacy alias consistent with triage.recommendation (rule 7)
        _hint_prefix_map = {
            "exclude":                    "exclude",
            "omit_from_training":         "omit",
            "review":                     "review",
            "keep_with_metadata_warning": "keep_with_metadata_warning",
            "keep":                       "keep",
        }
        decision_hint_out = (
            f"{_hint_prefix_map.get(recommendation, recommendation)}_{sample_type}"
        )

        results[case_id] = {
            "flags"                         : flags,
            "score"                         : score,
            "score_components"              : score_components,
            "score_version"                 : _score_ver_str,
            "scoring_details"               : _scoring_details,
            "dominant_failure_mode"         : dominant_failure_mode,
            "risk_interpretation"           : risk_interpretation,
            "risk_level"                    : risk,
            "evidence"                      : evidence,
            "recommendation"                : recommendation,
            # Task-aware fields (v3)
            "sample_type"                   : sample_type,
            "decision_hint"                 : decision_hint_out,
            "anatomical_validation_possible": anatomical_validation_possible,
            "task_aware_reasons"            : task_reasons,
            "subregion_qc"                  : subregion_qc,
            # Kept internal for TXT reporter backward compat
            "qc_profile"                    : qc_profile,
            # v6 triage block (primary decision: score + risk + recommendation)
            "triage"                        : triage,
            # v6 enriched qc_domains: single source of truth (replaces qc_profile in output)
            "qc_domains"                    : qc_domains,
            "driving_domain"                : driving_domain_out,
            # v6 primary issue (replaces dominant_failure_mode + driving_domain in output)
            "primary_issue"                 : primary_issue,
            # dataset-level context (replicated per-case for retrieval_summary)
            "task_mode"                     : _dataset_task_mode,
            "training_objective"             : _training_objective(_dataset_task_mode),
        }

    return results


def _tags(flags: list) -> set:
    return {t for _, t, _ in flags}


# ─────────────────────────────────────────────────────────────────────────────
# Structured integrity block helpers  (v4)
# Each returns {"severity": one_of("normal","low_warning","moderate_warning",
#                                  "high_warning","critical","not_applicable"),
#               "description": str}
# These are output-only — they never affect the numeric QC score.
# ─────────────────────────────────────────────────────────────────────────────

_SEV_ORDER: tuple[str, ...] = (
    "not_applicable", "normal", "low_warning", "moderate_warning", "high_warning", "critical"
)

def _escalate_sev(current: str, candidate: str) -> str:
    """Return the higher of two severity strings."""
    return candidate if _SEV_ORDER.index(candidate) > _SEV_ORDER.index(current) else current


def _fov_integrity_block(ev: dict, flags: list) -> dict:
    """FOV integrity — partial visibility / border truncation."""
    sev = "normal"
    parts: list[str] = []

    bt = ev.get("border_touching") or {}
    border_sides = [k.replace("touches_", "").replace("_", " ")
                    for k, touched in bt.items() if touched]
    if border_sides:
        parts.append(f"pancreas touches {', '.join(border_sides)} border")
        sev = _escalate_sev(sev, "low_warning")

    flag_tags_sev = {t: s for s, t, _ in flags}
    missing_regions = [rn for rn in ("head", "body", "tail")
                       if flag_tags_sev.get(f"{rn}_region_empty", "INFO") not in ("INFO",)]
    if missing_regions:
        parts.append(f"missing {'/'.join(missing_regions)} region")
        sev = _escalate_sev(sev, "moderate_warning")

    if ev.get("anatomical_truncation_suspected"):
        parts.append("anatomical truncation suspected")
        sev = _escalate_sev(sev, "high_warning")
    elif ev.get("partial_visibility_likely") and border_sides:
        sev = _escalate_sev(sev, "moderate_warning")

    if "incomplete_anatomical_coverage" in flag_tags_sev:
        cov = ev.get("region_coverage_ratio")
        cov_str = f"{cov:.0%}" if cov is not None else "unknown"
        parts.append(f"incomplete anatomical coverage ({cov_str} of pancreas covered by regions)")
        sev = _escalate_sev(sev, "moderate_warning")

    desc = "No FOV integrity issues detected." if not parts else "; ".join(parts).capitalize() + "."
    values = {
        "border_sides"                  : border_sides,
        "partial_visibility_likely"     : bool(ev.get("partial_visibility_likely", False)),
        "anatomical_truncation_suspected": bool(ev.get("anatomical_truncation_suspected", False)),
        "region_coverage_ratio"         : ev.get("region_coverage_ratio"),
        "missing_regions"               : missing_regions,
    }
    return {"severity": sev, "description": desc, "values": values}


def _attenuation_integrity_block(hu: dict, suspicious: bool) -> dict:
    """Attenuation integrity — HU plausibility check."""
    status = hu.get("hu_statistics_status", "")
    if status in ("pancreas_only_negative_case",
                  "no_pancreas_no_lesion_negative_case"):
        return {"severity": "not_applicable",
                "description": "Negative case — tumor HU not applicable.",
                "values": {}}

    if not suspicious:
        return {"severity": "normal", "description": "HU distribution appears plausible.",
                "values": {
                    "tumor_median_hu": hu.get("tumor_median_hu"),
                    "tumor_mean_hu":   hu.get("tumor_mean_hu"),
                    "z_score_hu":      hu.get("z_score_hu"),
                }}

    # Use median as primary (new schema); fall back to mean (legacy)
    mean_hu = hu.get("tumor_median_hu") or hu.get("tumor_mean_hu") or hu.get("mean_hu_tumor")
    if mean_hu is None or mean_hu != mean_hu:   # NaN guard
        return {"severity": "moderate_warning",
                "description": "NaN or invalid tumor HU statistics; verify CT image quality.",
                "values": {"tumor_median_hu": None, "z_score_hu": hu.get("z_score_hu")}}

    if mean_hu < -500 or mean_hu > 1000:
        sev = "critical"
    elif mean_hu < -300 or mean_hu > 600:
        sev = "high_warning"
    else:
        sev = "moderate_warning"
    desc = (f"Tumor mean HU {mean_hu:.1f} outside expected soft-tissue range "
            f"[-200, 400]; verify CT image or annotation.")
    values = {
        "tumor_median_hu"            : hu.get("tumor_median_hu"),
        "tumor_mean_hu"              : hu.get("tumor_mean_hu"),
        "z_score_hu"                 : hu.get("z_score_hu"),
        "tumor_attenuation"          : hu.get("tumor_attenuation"),
        "suspicious_hu_distribution" : True,
    }
    return {"severity": sev, "description": desc, "values": values}


def _lesion_localization_integrity_block(
    ov_rate: float | None,
    lesion_present: bool,
    pancreas_present: bool,
    flags: list,
) -> dict:
    """Lesion localization integrity — overlap-based severity."""
    if not lesion_present:
        return {"severity": "not_applicable",
                "description": "Negative sample — no lesion present.",
                "values": {"lesion_present": False, "overlap_rate_vs_lesion": None}}
    if not pancreas_present:
        return {"severity": "high_warning",
                "description": "Pancreas mask absent; anatomical validation of lesion location impossible.",
                "values": {"lesion_present": True, "pancreas_present": False,
                           "overlap_rate_vs_lesion": None}}
    if ov_rate is None:
        return {"severity": "moderate_warning",
                "description": "Lesion-pancreas overlap could not be computed.",
                "values": {"lesion_present": True, "pancreas_present": True,
                           "overlap_rate_vs_lesion": None}}

    if ov_rate >= 0.80:
        return {"severity": "normal",
                "description": f"Good lesion localization — overlap {ov_rate:.1%} within pancreas.",
                "values": {"overlap_rate_vs_lesion": round(ov_rate, 4),
                           "lesion_present": True, "pancreas_present": True}}
    elif ov_rate >= 0.55:
        sev  = "low_warning"
        desc = f"Minor localization concern — overlap {ov_rate:.1%}; anatomically plausible."
    elif ov_rate >= 0.25:
        sev  = "moderate_warning"
        desc = f"Moderate localization concern — overlap {ov_rate:.1%}; manual review recommended."
    elif ov_rate >= 0.05:
        sev  = "high_warning"
        desc = f"Poor localization — overlap {ov_rate:.1%}; lesion substantially outside pancreas."
    else:
        sev  = "critical"
        desc = f"Critical localization failure — overlap {ov_rate:.1%}; lesion near-entirely outside pancreas."
    return {"severity": sev, "description": desc,
            "values": {"overlap_rate_vs_lesion": round(ov_rate, 4),
                       "lesion_present": True, "pancreas_present": True}}


def _lesion_burden_subcheck(ev: dict) -> dict:
    """Lesion burden — volume ratio anomaly."""
    lv = ev.get("total_lesion_volume_mm3") or ev.get("lesion_volume_mm3") or 0
    pv = ev.get("pancreas_volume_mm3") or 0
    ratio = (lv / pv) if pv > 0 else None

    if not ev.get("lesion_present", False):
        return {"severity": "not_applicable",
                "description": "Negative sample — lesion burden not applicable.",
                "values": {"lesion_volume_mm3": lv, "pancreas_volume_mm3": pv, "ratio": None}}

    if ratio is None:
        sev  = "not_applicable"
        desc = "Lesion/pancreas volume ratio could not be computed."
    elif ratio > 1.5:
        sev  = "high_warning"
        desc = f"Lesion volume ({lv:,.0f} mm³) greatly exceeds pancreas ({pv:,.0f} mm³); ratio={ratio:.2f} — likely over-segmentation."
    elif ratio > 0.8:
        sev  = "moderate_warning"
        desc = f"Lesion volume ratio {ratio:.2f} — large lesion relative to pancreas; verify annotation."
    elif ratio > 0.4:
        sev  = "low_warning"
        desc = f"Lesion volume ratio {ratio:.2f} — moderately large lesion."
    else:
        sev  = "normal"
        desc = f"Lesion volume ratio {ratio:.2f} — within expected range."

    return {"severity": sev, "description": desc,
            "values": {"lesion_volume_mm3": lv, "pancreas_volume_mm3": pv,
                       "ratio": round(ratio, 3) if ratio is not None else None}}


def _pancreas_context_subcheck(ev: dict) -> dict:
    """Pancreas context — organ size plausibility."""
    pv = ev.get("pancreas_volume_mm3")
    pancreas_p = bool(ev.get("pancreas_present", False))

    if not pancreas_p:
        return {"severity": "not_applicable",
                "description": "Pancreas mask absent — no anatomical context available.",
                "values": {"pancreas_present": False, "pancreas_volume_mm3": None}}

    if pv is None:
        return {"severity": "not_applicable",
                "description": "Pancreas volume not available.",
                "values": {"pancreas_present": True, "pancreas_volume_mm3": None}}

    # Normal adult pancreas ~50 000–120 000 mm³ (~50–120 cm³)
    if pv < 10_000:
        sev  = "high_warning"
        desc = f"Very small pancreas ({pv:,.0f} mm³); possible atrophy, post-op resection, or segmentation error."
    elif pv < 25_000:
        sev  = "moderate_warning"
        desc = f"Small pancreas ({pv:,.0f} mm³); may indicate atrophic disease."
    elif pv > 250_000:
        sev  = "moderate_warning"
        desc = f"Unusually large pancreas ({pv:,.0f} mm³); verify segmentation."
    else:
        sev  = "normal"
        desc = f"Pancreas volume {pv:,.0f} mm³ — within plausible range."

    return {"severity": sev, "description": desc,
            "values": {"pancreas_present": True, "pancreas_volume_mm3": pv}}


def _region_consistency_subcheck(ev: dict, sc: dict) -> dict:
    """Region consistency — head/body/tail deviation from total volume."""
    raw    = sc.get("_region_raw", 0)
    capped = sc.get("region_consistency", 0)
    dev    = ev.get("region_relative_error") or ev.get("region_sum_deviation")

    if raw == 0 and capped == 0:
        return {"severity": "normal",
                "description": "Head/body/tail sub-region volumes are consistent with total pancreas volume.",
                "values": {"region_relative_error": dev, "raw_penalty": 0, "capped_penalty": 0}}

    if dev is not None:
        dev_pct = f"{dev:.1%}"
    else:
        dev_pct = "unknown"

    if raw >= 30:
        sev  = "high_warning"
        desc = f"Large sub-region volume inconsistency (deviation {dev_pct}); region masks may overlap or be missing."
    elif raw >= 15:
        sev  = "moderate_warning"
        desc = f"Moderate sub-region inconsistency (deviation {dev_pct}); verify head/body/tail annotations."
    else:
        sev  = "low_warning"
        desc = f"Minor sub-region inconsistency (deviation {dev_pct}); informational."

    return {"severity": sev, "description": desc,
            "values": {"region_relative_error": dev, "raw_penalty": raw, "capped_penalty": capped}}


def _region_consistency_not_applicable_subcheck(task_mode: str, reason: str = "") -> dict:
    """Region consistency — task-mode explanation for skipped sub-region QC."""
    if reason == "pancreas_not_present":
        description = "Subregion QC not applicable because pancreas mask is absent."
    else:
        description = f"Subregion QC disabled because TASK_MODE={task_mode}."
    return {
        "severity": "not_applicable",
        "description": description,
        "values": {
            "task_mode": task_mode,
        },
    }


def _metadata_completeness_subcheck(ev: dict) -> dict:
    """Metadata completeness — missing or implausible fields."""
    missing = ev.get("missing_metadata_fields") or []
    if isinstance(missing, str):
        missing = [f.strip() for f in missing.split(",") if f.strip()]

    if not missing:
        return {"severity": "normal",
                "description": "All expected metadata fields are present.",
                "values": {"missing_fields": [], "n_missing": 0}}

    n = len(missing)
    if n >= 4:
        sev  = "moderate_warning"
        desc = f"{n} metadata fields missing: {', '.join(missing)}. Limits stratified splits."
    elif n >= 2:
        sev  = "low_warning"
        desc = f"{n} metadata fields missing: {', '.join(missing)}."
    else:
        sev  = "low_warning"
        desc = f"Metadata field missing: {', '.join(missing)}."

    return {"severity": sev, "description": desc,
            "values": {"missing_fields": missing, "n_missing": n}}


def _severity_rank(severity: str) -> int:
    return _SEV_ORDER.index(severity) if severity in _SEV_ORDER else 0


def _max_subcheck_severity(subchecks: dict, *, metadata: bool = False) -> str:
    severities = [info.get("severity", "normal") for info in subchecks.values()]
    if not severities:
        return "normal"
    sev = max(severities, key=_severity_rank)
    if metadata and _severity_rank(sev) > _severity_rank("low_warning"):
        return "low_warning"
    return sev


def _combine_domain_severity(score_severity: str, subchecks: dict, *, metadata: bool = False) -> str:
    subcheck_severity = _max_subcheck_severity(subchecks, metadata=metadata)
    sev = score_severity if _severity_rank(score_severity) >= _severity_rank(subcheck_severity) else subcheck_severity
    if metadata and _severity_rank(sev) > _severity_rank("low_warning"):
        return "low_warning"
    return sev


def _primary_subcheck(domain: str | None, qc_domains: dict) -> str | None:
    if not domain:
        return None
    subchecks = (qc_domains.get(domain) or {}).get("subchecks") or {}
    if not subchecks:
        return None
    return max(subchecks, key=lambda key: _severity_rank(subchecks[key].get("severity", "normal")))


def _geometry_integrity_subchecks(flags: list, raw_data: dict) -> dict:
    flag_tags_sev = {t: (s, m) for s, t, m in flags}
    geo_s = ((raw_data.get("geometry") or {}).get("summary") or {})
    cs = raw_data.get("case_status") or {}

    affine_mismatch = bool(geo_s.get("pancreas_affine_mismatch") or geo_s.get("lesion_affine_mismatch"))
    spacing_flag = flag_tags_sev.get("invalid_spacing")
    shape_flag = "shape_mismatch" in flag_tags_sev
    non_orth = cs.get("error_type") == "non_orthonormal_direction_cosines"

    return {
        "affine_alignment": {
            "severity": "high_warning" if affine_mismatch else "normal",
            "description": "Mask affine mismatch with CT image." if affine_mismatch else "Image and mask affines are aligned.",
            "values": {
                "pancreas_affine_mismatch": bool(geo_s.get("pancreas_affine_mismatch", False)),
                "lesion_affine_mismatch": bool(geo_s.get("lesion_affine_mismatch", False)),
            },
        },
        "spacing": {
            "severity": ("high_warning" if spacing_flag and spacing_flag[0] == "CRITICAL"
                         else "moderate_warning" if spacing_flag else "normal"),
            "description": spacing_flag[1] if spacing_flag else "Voxel spacing is within expected limits.",
            "values": {"spacing_xyz_mm": (raw_data.get("image") or {}).get("spacing_xyz_mm")},
        },
        "shape": {
            "severity": "high_warning" if shape_flag else "normal",
            "description": flag_tags_sev["shape_mismatch"][1] if shape_flag else "Image shape is plausible.",
            "values": {"shape_zyx": (raw_data.get("image") or {}).get("shape_zyx")},
        },
        "direction_cosines": {
            "severity": "critical" if non_orth else "normal",
            "description": "Non-orthonormal direction cosines; CT cannot be loaded." if non_orth else "Direction cosines are loadable.",
            "values": {"non_orthonormal_cosines": non_orth},
        },
    }


def _lesion_localization_subchecks(
    ov_rate: float | None,
    lesion_present: bool,
    pancreas_present: bool,
    flags: list,
) -> dict:
    return {
        "overlap": _lesion_localization_integrity_block(ov_rate, lesion_present, pancreas_present, flags),
        "pancreas_presence_context": {
            "severity": "high_warning" if lesion_present and not pancreas_present else "normal",
            "description": (
                "Pancreas mask absent; lesion localization cannot be anatomically validated."
                if lesion_present and not pancreas_present else
                "Pancreas context is available for lesion localization."
            ),
            "values": {"lesion_present": lesion_present, "pancreas_present": pancreas_present},
        },
    }


def _lesion_burden_subchecks(ev: dict) -> dict:
    lv = ev.get("total_lesion_volume_mm3") or ev.get("lesion_volume_mm3") or 0
    pv = ev.get("pancreas_volume_mm3") or 0
    ratio = (lv / pv) if pv > 0 else None
    lesion_present = bool(ev.get("lesion_present", False))

    if not lesion_present:
        volume = {"severity": "not_applicable", "description": "Negative sample — lesion volume not applicable.",
                  "values": {"lesion_volume_mm3": lv}}
    elif lv < THR_LES_MIN:
        volume = {"severity": "moderate_warning", "description": f"Very small lesion volume ({lv:,.0f} mm³).",
                  "values": {"lesion_volume_mm3": lv}}
    elif lv > THR_LES_MAX:
        volume = {"severity": "high_warning", "description": f"Very large lesion volume ({lv:,.0f} mm³).",
                  "values": {"lesion_volume_mm3": lv}}
    else:
        volume = {"severity": "normal", "description": f"Lesion volume {lv:,.0f} mm³ is within configured limits.",
                  "values": {"lesion_volume_mm3": lv}}

    ratio_block = _lesion_burden_subcheck(ev)
    ratio_block["values"]["ratio"] = round(ratio, 3) if ratio is not None else None
    return {"lesion_volume": volume, "lesion_pancreas_ratio": ratio_block}


def _pancreas_context_subchecks(ev: dict) -> dict:
    pancreas_present = bool(ev.get("pancreas_present", False))
    return {
        "organ_presence": {
            "severity": "not_applicable" if not pancreas_present else "normal",
            "description": "Pancreas mask is absent." if not pancreas_present else "Pancreas mask is present.",
            "values": {"pancreas_present": pancreas_present},
        },
        "organ_volume": _pancreas_context_subcheck(ev),
    }


def _fov_integrity_subchecks(ev: dict, flags: list, thr: "dict | None" = None) -> dict:
    base = _fov_integrity_block(ev, flags)
    policy = (thr or {}).get("FOV_POLICY") or {}
    border_sev = str(policy["border_touching_severity"])
    coverage_sev = str(policy["incomplete_coverage_severity"])
    missing_sev = str(policy["missing_region_severity"])
    truncation_sev = str(policy["truncation_severity"])
    values = base.get("values", {})
    border_sides = values.get("border_sides", [])
    flag_tags = {t for _, t, _ in flags}
    missing_regions = values.get("missing_regions", [])
    coverage_issue = "incomplete_anatomical_coverage" in flag_tags or bool(missing_regions)
    truncation = bool(values.get("anatomical_truncation_suspected", False))
    return {
        "border_touching": {
            "severity": border_sev if border_sides else "normal",
            "description": f"Pancreas touches {', '.join(border_sides)} border." if border_sides else "No border-touching pancreas signal.",
            "values": {"border_sides": border_sides},
        },
        "anatomical_coverage": {
            "severity": (missing_sev if missing_regions else coverage_sev) if coverage_issue else "normal",
            "description": base["description"] if coverage_issue else "Anatomical coverage appears complete.",
            "values": {"region_coverage_ratio": values.get("region_coverage_ratio"), "missing_regions": missing_regions},
        },
        "truncation": {
            "severity": truncation_sev if truncation else "normal",
            "description": "Anatomical truncation suspected." if truncation else "No anatomical truncation suspected.",
            "values": {"partial_visibility_likely": values.get("partial_visibility_likely"),
                       "anatomical_truncation_suspected": truncation},
        },
    }


def _region_consistency_subchecks(ev: dict, sc: dict, enabled: bool, task_mode: str, reason: str = "") -> dict:
    applicability = {
        "severity": "normal" if enabled else "not_applicable",
        "description": "Subregion QC is enabled for this task mode." if enabled else _region_consistency_not_applicable_subcheck(task_mode, reason)["description"],
        "values": {"task_mode": task_mode, "enabled": enabled},
    }
    if not enabled:
        na = {"severity": "not_applicable", "description": "Subregion QC disabled for this case.", "values": {}}
        return {
            "task_mode_applicability": applicability,
            "subregion_volume_sum": na,
            "subregion_overlap": na,
            "missing_subregions": na,
        }
    missing = [r for r in ("head", "body", "tail") if ev.get(f"{r}_volume_mm3") == 0]
    overlap_values = {
        "head_body_overlap_mm3": ev.get("head_body_overlap_mm3"),
        "head_tail_overlap_mm3": ev.get("head_tail_overlap_mm3"),
        "body_tail_overlap_mm3": ev.get("body_tail_overlap_mm3"),
    }
    has_overlap = any((v or 0) > 0 for v in overlap_values.values())
    return {
        "task_mode_applicability": applicability,
        "subregion_volume_sum": _region_consistency_subcheck(ev, sc),
        "subregion_overlap": {
            "severity": "moderate_warning" if has_overlap else "normal",
            "description": "Subregion masks overlap." if has_overlap else "No subregion overlap detected.",
            "values": overlap_values,
        },
        "missing_subregions": {
            "severity": "moderate_warning" if missing else "normal",
            "description": f"Missing subregions: {', '.join(missing)}." if missing else "All expected subregions are present.",
            "values": {"missing_subregions": missing},
        },
    }


def _attenuation_integrity_subchecks(hu: dict, suspicious: bool) -> dict:
    dist = _attenuation_integrity_block(hu, suspicious)
    att = hu.get("tumor_attenuation", "Unknown")
    rule_disagree = bool(hu.get("attenuation_rule_disagreement", False))
    return {
        "hu_distribution": dist,
        "attenuation_class": {
            "severity": "not_applicable" if att in ("Not applicable", None) else "normal",
            "description": f"Tumor attenuation class: {att}." if att not in (None, "Unknown") else "Tumor attenuation class unknown.",
            "values": {"tumor_attenuation": att, "delta_hu_attenuation": hu.get("delta_hu_attenuation")},
        },
        "rule_disagreement": {
            "severity": "low_warning" if rule_disagree else "normal",
            "description": "Attenuation classification rules disagree." if rule_disagree else "Attenuation classification rules agree.",
            "values": {"attenuation_rule_disagreement": rule_disagree},
        },
    }


def _metadata_completeness_subchecks(ev: dict) -> dict:
    fields = _metadata_completeness_subcheck(ev)
    if _severity_rank(fields.get("severity", "normal")) > _severity_rank("low_warning"):
        fields = dict(fields)
        fields["severity"] = "low_warning"
    return {"fields": fields}


# ═══════════════════════════════════════════════════════════════════════════════
# REPORT GENERATORS
# ═══════════════════════════════════════════════════════════════════════════════

def generate_txt_report(data: dict, qc_results: dict, report_path: Path) -> None:
    def _cases_with_tag(tag: str) -> list[str]:
        return sorted(k for k, v in qc_results.items()
                      if any(t == tag for _, t, _ in v["flags"]))

    def _cases_any(tags: list[str]) -> list[str]:
        return sorted(set(c for tg in tags for c in _cases_with_tag(tg)))

    empty_pan  = _cases_with_tag("pancreas_mask_empty")
    lesion_out = _cases_any(["lesion_outside_pancreas", "lesion_partial_outside",
                              "lesion_overlap_critical_fail"])
    region_err = _cases_any(["region_sum_error_critical", "region_sum_error_warning"])
    partial_cov = _cases_with_tag("incomplete_anatomical_coverage")
    small_pan  = _cases_any(["very_small_pancreas", "small_pancreas"])
    miss_reg   = _cases_any(["head_region_empty", "body_region_empty", "tail_region_empty"])
    high_risk  = sorted(k for k, v in qc_results.items() if v["risk_level"] == "high")
    med_risk   = sorted(k for k, v in qc_results.items() if v["risk_level"] == "medium")
    crit_risk  = sorted(k for k, v in qc_results.items() if v.get("risk_level") == "critical")
    omitted_cases = _get_omitted_cases(data)
    low_risk   = sorted(k for k, v in qc_results.items() if v["risk_level"] == "low")

    # Task-aware sample counts
    positives    = sorted(k for k, v in qc_results.items()
                          if v.get("sample_type") == "positive")
    negatives    = sorted(k for k, v in qc_results.items()
                          if v.get("sample_type") == "negative")
    no_anat_val  = sorted(k for k, v in qc_results.items()
                          if not v.get("anatomical_validation_possible")
                          and v.get("sample_type") == "positive")
    keep_pos     = sorted(k for k, v in qc_results.items()
                          if v.get("decision_hint") == "keep_positive")
    review_pos   = sorted(k for k, v in qc_results.items()
                          if v.get("decision_hint") in ("review_positive",
                                                         "review_or_exclude_positive"))

    # HU / attenuation case lists (built from summary data, not qc_results)
    def _hu_attenuate(attenuate: str) -> list[str]:
        return sorted(k for k, v in data.items()
                      if (v.get("hu_statistics") or {}).get("tumor_attenuation") == attenuate)

    hypo_cases          = _hu_attenuate("Hypo")
    hyper_cases         = _hu_attenuate("Hyper")
    iso_cases           = _hu_attenuate("Iso")
    unknown_att_cases   = _hu_attenuate("Unknown")
    not_applicable_cases = _hu_attenuate("Not applicable")
    suspicious_hu_cases = sorted(k for k, v in qc_results.items()
                                  if v.get("evidence", {}).get("suspicious_hu_distribution"))

    L: list[str] = []
    L += ["=" * 72,
          "  PANTSMINI DATASET — QUALITY CONTROL REPORT  (v5)",
          f"  Training objective: {_training_objective(next(iter(qc_results.values()), {}).get('task_mode', 'pancreas_lesion'))}",
          "=" * 72, "",
          f"  Total cases in JSON      : {len(data)}",
          f"  Omitted (unloadable CT)  : {len(omitted_cases)}",
          f"  Cases with QC flags      : {len(qc_results)}", "",
          f"  SAMPLE TYPES (task-aware)",
          f"    Positive (lesion)      : {len(positives)}",
          f"    Negative (no lesion)   : {len(negatives)}", "",
          f"  TASK-AWARE DECISIONS",
          f"    keep_positive          : {len(keep_pos)}",
          f"    review_positive        : {len(review_pos)}",
          f"    No anatomical val.     : {len(no_anat_val)}  "
          f"(pancreas absent + lesion present)", "",
          f"  RISK LEVELS",
          f"    Critical (omitted)     : {len(crit_risk)}",
          f"    High risk              : {len(high_risk)}",
          f"    Medium risk            : {len(med_risk)}",
          f"    Low risk               : {len(low_risk)}", ""]

    _SEV_LABEL: dict[str, str] = {
        "normal":           "NORMAL",
        "low_warning":      "LOW WARNING",
        "moderate_warning": "WARNING",
        "high_warning":     "HIGH WARNING",
        "critical":         "CRITICAL",
    }

    def _section(title: str, cases: list[str], note: str) -> None:
        L.extend(["─" * 72, f"  {title}", "─" * 72, f"  {note}", ""])
        for c in cases:
            v     = qc_results.get(c, {})
            _raw  = data.get(c) or {}
            ev    = v.get("evidence", {})
            flags = v.get("flags", [])
            score = v.get("score", "?")
            risk  = v.get("risk_level", "?")
            hint  = v.get("decision_hint", "?")
            pi    = v.get("primary_issue", {}) or {}
            msgs  = "; ".join(m for sev, _, m in flags if sev != "INFO")[:100]
            sc    = v.get("score_components", {})
            sd    = v.get("scoring_details")
            sv    = v.get("score_version", "v2")
            qc_doms = v.get("qc_domains", {})
            triage_block = v.get("triage", {})
            triage_score = triage_block.get("score", score)
            rec  = triage_block.get("recommendation", v.get("recommendation", "?"))

            _hu      = (_raw.get("hu_statistics") or {})
            _susp_hu = bool(ev.get("suspicious_hu_distribution", False))
            ov_rate  = ev.get("overlap_rate_vs_lesion")
            lesion_p = bool(ev.get("lesion_present", False))
            pan_p    = bool(ev.get("pancreas_present", False))

            _pi_domain  = pi.get("domain") or "—"
            _pi_failure = pi.get("failure_mode") or "—"

            # ── v6 compact case header ────────────────────────────────────
            L.append(f"  {c}  [score={triage_score} | {risk} risk | rec={rec} | {_pi_domain}/{_pi_failure}]")
            L.append(f"    QC DOMAINS")
            _dom_labels = {
                "geometry_integrity"   : "Geometry integrity",
                "lesion_localization"  : "Lesion localization",
                "lesion_burden"        : "Lesion burden",
                "pancreas_context"     : "Pancreas context",
                "fov_integrity"        : "FOV integrity",
                "region_consistency"   : "Region consistency",
                "attenuation_integrity": "Attenuation integrity",
                "metadata_completeness": "Metadata completeness",
            }
            for domain, label in _dom_labels.items():
                d_info  = qc_doms.get(domain, {})
                sev     = d_info.get("severity", "normal")
                sev_l   = _SEV_LABEL.get(sev, sev.upper())
                d_score = d_info.get("component_score", d_info.get("score", 0))
                ev_tags = [e["tag"] for e in d_info.get("evidence", [])]
                ev_str  = f"  ← {', '.join(ev_tags)}" if ev_tags else ""
                _excl   = "  [excl. risk]" if domain == "metadata_completeness" and sev != "normal" else ""
                L.append(f"      {label:<28} {sev_l:<16} (score={d_score:.0f}){_excl}{ev_str}")
            if msgs:
                L.append(f"    Details               : {msgs}")
            if sd:
                L.append(f"    SCORING DETAILS")
                L.append(f"      Version                 : {sv}")
                L.append(f"      Geometry gate           : {sc.get('geometry_integrity', 0)}")
                L.append(f"      Overlap penalty         : {sc.get('lesion_localization', 0)}")
                L.append(f"      Secondary weighted sum  : {sd.get('secondary_weighted_sum', 0)}")
                L.append(f"      Alpha                   : {sd.get('alpha', 0)}")
                L.append(f"      Secondary amplifier     : {sd.get('secondary_amplifier', 0)}")
                L.append(f"      Fallback active         : {sd.get('fallback_active', False)}")
                if sd.get("fallback_active"):
                    L.append(f"      Beta                    : {sd.get('beta', 0)}")
                    L.append(f"      Fallback term           : {sd.get('fallback_term', 0)}")
                L.append(f"      Final score             : {triage_score}")
            L.append("")
        L.append("")

    L += ["─" * 72,
          f"  SECTION 1  — OMITTED CASES ({len(omitted_cases)} cases)  [CRITICAL]",
          "─" * 72,
          "  CT image could not be loaded (file missing or non-orthonormal geometry).",
          "  These cases are excluded from training splits.", ""]
    for c in omitted_cases:
        cs  = (data.get(c) or {}).get("case_status") or {}
        err = cs.get("error_type", "unknown")
        msg = cs.get("error_message", "")[:80]
        L.append(f"  {c}  [{err}]  {msg}")
    L.append("")

    _section(f"SECTION 2  — EMPTY PANCREAS MASK ({len(empty_pan)} cases)",
             empty_pan,
             "pancreas.nii.gz is entirely zero. "
             "If lesion absent: treated as valid negative. "
             "If lesion present: anatomical validation impossible [WARNING].")
    _section(f"SECTION 3  — LESION OVERLAP ISSUES ({len(lesion_out)} cases)  [CRITICAL/WARNING]",
             lesion_out,
             "Lesion-pancreas overlap below threshold. "
             f"Minimum: {THR_LESION_MIN:.0%}  Strict: {THR_LESION_STRICT:.0%}.")
    _section(f"SECTION 4  — REGION SUM INCONSISTENCY ({len(region_err)} cases)  [CRITICAL/WARNING]",
             region_err, "Head+body+tail deviates > 10 % from total pancreas.")
    _section(f"SECTION 4b — PARTIAL ANATOMICAL COVERAGE ({len(partial_cov)} cases)  [WARNING]",
             partial_cov,
             "Partial pancreas visibility suspected due to truncated field of view. "
             "Missing anatomical regions may be expected. "
             "Safe for pancreas/lesion segmentation; exclude from sub-region tasks.")
    _section(f"SECTION 5  — MISSING SUB-REGION MASK ({len(miss_reg)} cases)  [WARNING]",
             miss_reg, "At least one of pancreas head/body/tail mask is empty "
             "(not explained by partial visibility).")
    _section(f"SECTION 6  — ABNORMAL PANCREAS VOLUME ({len(small_pan)} cases)  [CRITICAL/WARNING]",
             small_pan, f"Pancreas volume < {THR_PAN_WARN:,} mm3.")

    # ── SECTION 7: HU attenuation statistics ─────────────────────────────────

    def _hu_display(case_id: str) -> list[str]:
        """Format per-case HU statistics block in the txt report."""
        hu      = (data.get(case_id) or {}).get("hu_statistics") or {}
        status  = hu.get("hu_statistics_status", "")
        phase   = hu.get("ct_phase") or "N/A"
        t_med   = hu.get("tumor_median_hu")
        p_med   = hu.get("pancreas_median_hu")
        delta   = hu.get("delta_hu_tumor_vs_pancreas")
        z       = hu.get("z_score_hu")
        att     = hu.get("tumor_attenuation", "Unknown")
        rule    = hu.get("attenuation_rule_used", "unknown")
        delta_att = hu.get("delta_hu_attenuation", "Unknown")
        disagree  = hu.get("attenuation_rule_disagreement", False)
        susp      = hu.get("suspicious_hu_distribution", False)

        def _fmt(v, fmt=".1f") -> str:
            return format(v, fmt) if v is not None else "N/A"

        is_negative = status in ("pancreas_only_negative_case",
                                 "no_pancreas_no_lesion_negative_case")

        lines = [
            f"    HU STATISTICS  [{status}]",
            f"      CT phase                 : {phase}",
        ]
        if not is_negative:
            lines += [
                f"      Tumor median HU          : {_fmt(t_med)}",
            ]
        lines += [
            f"      Pancreas median HU       : {_fmt(p_med)}",
        ]
        if not is_negative:
            lines += [
                f"      Delta HU                 : {_fmt(delta, '+.1f') if delta is not None else 'N/A'}",
                f"      Z-score HU               : {_fmt(z, '.2f') if z is not None else 'N/A'}",
                f"      Attenuation              : {att}",
                f"      Rule used                : {rule}",
                f"      Delta-based label        : {delta_att}",
                f"      Rule disagreement        : {'true' if disagree else 'false'}",
                f"      Suspicious distribution  : {'true' if susp else 'false'}",
            ]
        else:
            lines += [
                f"      Pancreas std HU          : {_fmt(hu.get('pancreas_std_hu'))}",
                f"      Pancreas p05/p95 HU      : {_fmt(hu.get('pancreas_p05_hu'))} / {_fmt(hu.get('pancreas_p95_hu'))}",
                f"      Pancreas voxel count     : {hu.get('pancreas_voxel_count') or 'N/A'}",
            ]

        # Clinical interpretation
        if is_negative:
            if status == "no_pancreas_no_lesion_negative_case":
                interp = ("HU attenuation not applicable: negative case without lesion or "
                          "pancreas mask.")
            else:
                interp = ("HU attenuation not applicable: negative case without lesion. "
                          f"Normal pancreas reference computed (median {_fmt(p_med)} HU, "
                          f"phase: {phase}).")
        elif att == "Hypo":
            interp = (f"Hypoattenuating lesion relative to normal pancreas in {phase} phase "
                      f"based on {rule} classification — consistent with PDAC or cystic lesion.")
        elif att == "Hyper":
            interp = (f"Hyperattenuating lesion in {phase} phase based on {rule} classification "
                      f"— may indicate neuroendocrine tumor or annotation inconsistency.")
        elif att == "Iso":
            interp = (f"Isoattenuating lesion relative to normal pancreas in {phase} phase "
                      f"based on {rule} classification — consider indirect signs if PDAC suspected.")
        elif att == "Unknown":
            interp = "Pancreas mask absent; attenuation relative to pancreas cannot be computed."
        else:
            interp = "HU classification not available."
        if disagree and not is_negative:
            interp += " Note: z-score and delta-HU labels disagree — review distribution."
        lines.append(f"      Interpretation: {interp}")
        return lines

    L += ["", "─" * 72,
          f"  SECTION 7  — LESION ATTENUATION STATISTICS (HU)",
          "─" * 72,
          "  Computed from CT voxel intensities (tumor vs. normal-pancreas reference).",
          "  Primary rule: z_score_hu (normalized contrast). Fallback: delta_hu.",
          "  Normal pancreas reference = pancreas AND NOT tumor, eroded + percentile-clipped.",
          "  Negative cases: pancreas HU computed when pancreas present; tumor fields null.",
          "  Attenuation is a secondary QC signal — used for reporting,",
          "  explainability, and dataset characterization only.", ""]

    for att_label, att_cases in [
        ("Hypoattenuating  (Hypo)",          hypo_cases),
        ("Hyperattenuating (Hyper)",          hyper_cases),
        ("Isoattenuating   (Iso)",            iso_cases),
        ("Unknown (pancreas absent)",         unknown_att_cases),
        ("Not applicable (negative cases)",   not_applicable_cases),
    ]:
        L.append(f"  {att_label}: {len(att_cases)} cases")
        for c in att_cases:
            L.append(f"    {c}")
            L.extend(_hu_display(c))
        L.append("")

    if suspicious_hu_cases:
        L += [f"  SUSPICIOUS HU VALUES: {len(suspicious_hu_cases)} cases",
              "  (HU outside expected soft-tissue range [-200, 400] — verify CT image)", ""]
        for c in suspicious_hu_cases:
            hu  = (data.get(c) or {}).get("hu_statistics") or {}
            L.extend(_hu_display(c))
        L.append("")
    else:
        L.append("  No suspicious HU values detected.")
        L.append("")

    L += ["=" * 72, "  SUMMARY TABLE", "=" * 72,
          f"  {'Issue type':<50} {'Count':>5}",
          f"  {'─'*50} {'─'*5}"]
    for label, count in [
        ("Omitted cases (CT unloadable)",                    len(omitted_cases)),
        ("Positive samples (lesion present)",                len(positives)),
        ("Negative samples (no lesion)",                     len(negatives)),
        ("keep_positive",                                    len(keep_pos)),
        ("review_positive / review_or_exclude_positive",     len(review_pos)),
        ("No anatomical validation (pancreas absent+lesion)", len(no_anat_val)),
        ("Empty pancreas mask",                              len(empty_pan)),
        ("Lesion overlap issues",                            len(lesion_out)),
        ("Region sum inconsistency (> 10 %)",                len(region_err)),
        ("Partial anatomical coverage (truncated FOV)",      len(partial_cov)),
        ("Missing sub-region mask (head/body/tail)",         len(miss_reg)),
        ("Abnormal pancreas volume (< 10 000 mm3)",          len(small_pan)),
        ("High-risk cases",                                  len(high_risk)),
        ("Medium-risk cases",                                len(med_risk)),
        ("Critical-risk cases (CT unloadable)",              len(crit_risk)),
        ("Hypoattenuating lesions",                          len(hypo_cases)),
        ("Hyperattenuating lesions",                         len(hyper_cases)),
        ("Isoattenuating lesions",                           len(iso_cases)),
        ("Attenuation unknown (no pancreas)",                len(unknown_att_cases)),
        ("Attenuation not applicable (negative cases)",      len(not_applicable_cases)),
        ("Suspicious HU values",                             len(suspicious_hu_cases)),
    ]:
        L.append(f"  {label:<50} {count:>5}")
    L.append("=" * 72)

    report_path.write_text("\n".join(L), encoding="utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Schema consistency assertions  (rule 10)
# Called on every case before writing JSON output.  Violations are logged as
# warnings rather than hard errors to avoid blocking report generation.
# ─────────────────────────────────────────────────────────────────────────────

def _assert_case_consistency(case_id: str, v: dict) -> list[str]:
    """Return a list of consistency violation messages (empty = all OK)."""
    issues: list[str] = []
    score        = v.get("score")
    triage       = v.get("triage", {})
    pi           = v.get("primary_issue", {}) or {}
    dom_fm       = v.get("dominant_failure_mode")
    decision_h   = v.get("decision_hint", "")
    rec          = v.get("recommendation", "")
    risk         = v.get("risk_level", "")
    qc_doms      = v.get("qc_domains", {})
    sc           = v.get("score_components", {})

    # 1. qc_score == triage.score
    if triage.get("score") is not None and score is not None:
        if triage["score"] != score:
            issues.append(
                f"qc_score ({score}) != triage.score ({triage['score']})"
            )

    # 2. dominant_failure_mode == primary_issue.failure_mode
    if dom_fm and pi.get("failure_mode") and dom_fm != pi["failure_mode"]:
        issues.append(
            f"dominant_failure_mode ({dom_fm!r}) != primary_issue.failure_mode "
            f"({pi['failure_mode']!r})"
        )

    # 3. decision_hint is consistent with triage.recommendation
    triage_rec = triage.get("recommendation", rec)
    if decision_h and triage_rec:
        _hint_prefix_map = {
            "exclude": "exclude", "omit_from_training": "omit",
            "review": "review", "keep_with_metadata_warning": "keep_with_metadata_warning",
            "keep": "keep",
        }
        expected_prefix = _hint_prefix_map.get(triage_rec, triage_rec)
        if not decision_h.startswith(expected_prefix):
            issues.append(
                f"decision_hint ({decision_h!r}) inconsistent with "
                f"triage.recommendation ({triage_rec!r})"
            )

    # 4. metadata-only issues must not produce high risk
    if risk == "high" and qc_doms:
        structural_issues = any(
            qc_doms.get(d, {}).get("severity", "normal") not in ("normal", "low_warning")
            for d in _PROFILE_DOMAINS if d != "metadata_completeness"
        )
        if not structural_issues:
            issues.append(
                f"risk_level='high' but no structural domain has severity > low_warning "
                f"(metadata-only issue should not produce high risk)"
            )

    # 5. qc_domains keys must all be canonical
    bad_dom_keys = [k for k in qc_doms if k not in _CANONICAL_DOMAINS]
    if bad_dom_keys:
        issues.append(f"qc_domains contains non-canonical keys: {bad_dom_keys}")

    # 6. primary_issue.domain must be canonical (if set)
    pi_domain = pi.get("domain")
    if pi_domain and pi_domain not in _CANONICAL_DOMAINS:
        issues.append(
            f"primary_issue.domain ({pi_domain!r}) is not a canonical domain name"
        )

    # 7. score_components keys must be canonical (excluding _ private keys)
    bad_sc_keys = [k for k in sc if not k.startswith("_") and k not in _CANONICAL_DOMAINS]
    if bad_sc_keys:
        issues.append(f"score_components contains non-canonical keys: {bad_sc_keys}")

    return issues


# ─────────────────────────────────────────────────────────────────────────────
# RAG / retrieval tag builder
# Produces a compact list of string tags for fast semantic retrieval.
# Tags are intentionally human-readable and collision-free across cases.
# ─────────────────────────────────────────────────────────────────────────────

def _build_retrieval_tags(
    v: dict,
    ev: dict,
    triage: dict,
    primary_issue: dict,
    qc_domains: dict,
    hu: dict,
    subregion_qc: dict,
) -> list[str]:
    tags: list[str] = []

    task_mode = v.get("task_mode", "unknown")
    primary_target = _primary_target(task_mode)
    tags.extend((task_mode, f"primary_target_{primary_target}"))

    # Sample type
    sample_type = v.get("sample_type", "unknown")
    tags.append(f"{sample_type}_case")
    if sample_type == "missing_required_target":
        tags.append("required_target_missing")

    # Top-level recommendation and risk
    rec  = triage.get("recommendation", "keep")
    risk = triage.get("risk_level", "low")
    tags.append(rec)
    tags.append(f"{risk}_risk")

    # Primary issue domain and failure mode
    pi_domain  = primary_issue.get("domain")
    pi_failure = primary_issue.get("failure_mode")
    if pi_domain:
        tags.append(f"primary_{pi_domain}")
    if pi_failure:
        tags.append(pi_failure)

    # Overlap quality bucket (lesion cases only)
    ov = ev.get("overlap_rate_vs_lesion")
    if ov is not None:
        if   ov >= 0.90: tags.append("excellent_overlap")
        elif ov >= 0.70: tags.append("good_overlap")
        elif ov >  0.0:  tags.append("poor_overlap")
        elif ov == 0.0 and ev.get("lesion_present"): tags.append("zero_overlap")

    # Domain severities — only actionable non-normal states
    for domain, info in qc_domains.items():
        sev = info.get("severity", "normal")
        if sev not in ("normal", "none", "not_applicable"):
            tags.append(f"{domain}_{sev}")

    # HU attenuation (tumor, positive cases)
    att = hu.get("tumor_attenuation")
    if att and att not in ("Not applicable", "Unknown", None):
        tags.append(f"tumor_{att.lower()}")  # tumor_hypo / tumor_hyper / tumor_iso

    # Subregion integrity
    srq_status = subregion_qc.get("status", "not_applicable")
    if srq_status == "warning":
        tags.append("subregion_inconsistency")
    elif srq_status == "failed":
        tags.append("subregion_failed")

    # Omitted marker
    if rec == "omit_from_training":
        tags.append("omitted")

    # HU distribution anomaly
    if ev.get("suspicious_hu_distribution"):
        tags.append("suspicious_hu")

    # FOV truncation
    if ev.get("anatomical_truncation_suspected"):
        tags.append("fov_truncated")

    return list(dict.fromkeys(tags))


def _build_target_presence(task_mode: str, ev: dict) -> dict:
    """Describe deterministic annotation presence and conservative FOV presence."""
    annotation_present = bool(ev.get("pancreas_present", False))
    fov_known = any(
        key in ev
        for key in ("border_touching", "partial_visibility_likely",
                    "anatomical_truncation_suspected")
    )
    partial = bool(
        ev.get("partial_visibility_likely")
        or ev.get("anatomical_truncation_suspected")
        or any((ev.get("border_touching") or {}).values())
    )

    if annotation_present:
        observed_presence = "partial" if partial else "present"
    else:
        observed_presence = "uncertain"

    annotation_complete = fov_known and not partial and annotation_present
    return {
        "target": "pancreas",
        "role": "primary_target" if task_mode == "pancreas_only" else "context",
        "expected_presence": "required",
        "annotation_presence": "present" if annotation_present else "absent",
        "observed_presence": observed_presence,
        "visible_target_annotation_status": "complete" if annotation_complete else "uncertain",
    }


def generate_json_report(qc_results: dict, report_path: Path,
                          data: dict | None = None,
                          report_metadata: dict | None = None,
                          thr: dict | None = None) -> None:
    # HU fields that live exclusively in hu_statistics; strip from measurements
    _HU_EV_FIELDS = frozenset({
        "tumor_median_hu", "tumor_mean_hu", "tumor_attenuation", "z_score_hu",
        "mean_hu_tumor", "mean_hu_pancreas",
        "delta_hu", "suspicious_hu_distribution",
    })
    _EV_PRIMARY = frozenset({
        "pancreas_present", "lesion_present",
        "pancreas_volume_mm3", "lesion_volume_mm3", "overlap_rate_vs_lesion",
    })

    out: dict = {}
    for case_id, v in qc_results.items():
        ev    = v["evidence"]
        sc    = v.get("score_components", {})
        _raw  = (data or {}).get(case_id) or {}
        _hu   = _raw.get("hu_statistics") or {}
        flags = v.get("flags", [])

        suspicious_hu = bool(ev.get("suspicious_hu_distribution", False))
        ov_rate       = ev.get("overlap_rate_vs_lesion")
        lesion_p      = bool(ev.get("lesion_present", False))
        pancreas_p    = bool(ev.get("pancreas_present", False))

        # Consistency assertions (rule 10): violations are warnings, not errors
        for _issue in _assert_case_consistency(case_id, v):
            warnings.warn(f"[QC consistency] {case_id}: {_issue}", stacklevel=2)

        # ── Layer 4: Triage — scoring_details updated ──────────────────────
        # • qc_score_after_clamp removed (= triage.score, redundant)
        # • score_components moved here from top-level
        _triage_in = v.get("triage", {"score": v["score"]})
        _sd_orig   = _triage_in.get("scoring_details") or {}
        _sc_clean  = {k: v2 for k, v2 in sc.items() if not k.startswith("_")}
        _sd_out    = {k: v2 for k, v2 in _sd_orig.items()
                      if k != "qc_score_after_clamp"}
        if _sc_clean:
            _sd_out["score_components"] = _sc_clean
        _triage_out = {k: v2 for k, v2 in _triage_in.items()
                       if k != "scoring_details"}
        if _sd_out:
            _triage_out["scoring_details"] = _sd_out

        # ── Layer 3: Primary issue ──────────────────────────────────────────
        _pi = v.get("primary_issue")

        # ── Subregion QC ────────────────────────────────────────────────────
        _subregion_qc = v.get("subregion_qc", {
            "enabled": False, "reason": "not_computed", "status": "not_applicable",
        })

        # ── Layer 2: QC domains — canonical 8-domain schema ────────────────
        # domain fields: component_score / severity / verdict / subchecks
        # metadata_completeness annotated with excluded_from_risk_escalation
        _qc_domains_in  = v.get("qc_domains", {})
        _qc_domains_out = {}
        _task_mode_for_case = v.get("task_mode", "unknown")
        _score_weights = (thr or {}).get("SCORE_WEIGHTS", {}).get(_task_mode_for_case, {})
        for domain, dinfo in _qc_domains_in.items():
            _d: dict = {
                "component_score": dinfo.get("component_score", dinfo.get("score", 0)),
                "severity": dinfo["severity"],
                "verdict":  dinfo.get("verdict", "keep"),
            }
            if domain == "metadata_completeness":
                _d["excluded_from_risk_escalation"] = True
            # Build granular subchecks from the helper functions (all return
            # {severity, description, values}).  Domain severity follows the
            # highest subcheck severity, except metadata is capped at low_warning
            # because this QC system's primary task is image segmentation.
            if domain == "geometry_integrity":
                _d["subchecks"] = _geometry_integrity_subchecks(flags, _raw)
            elif domain == "lesion_localization":
                _d["subchecks"] = _lesion_localization_subchecks(ov_rate, lesion_p, pancreas_p, flags)
            elif domain == "fov_integrity":
                _d["subchecks"] = _fov_integrity_subchecks(ev, flags, thr)
            elif domain == "attenuation_integrity":
                _d["subchecks"] = _attenuation_integrity_subchecks(_hu, suspicious_hu)
            elif domain == "lesion_burden":
                _d["subchecks"] = _lesion_burden_subchecks(ev)
            elif domain == "pancreas_context":
                _d["subchecks"] = _pancreas_context_subchecks(ev)
            elif domain == "region_consistency":
                _enabled = bool(_subregion_qc.get("enabled", False))
                _task_mode = v.get("task_mode", "unknown")
                _reason = _subregion_qc.get("reason", "")
                _d["enabled"] = _enabled
                _d["subchecks"] = _region_consistency_subchecks(ev, sc, _enabled, _task_mode, _reason)
                if not _enabled:
                    _d["component_score"] = 0
                    _d["severity"] = "not_applicable"
                    _d["verdict"] = "not_applicable"
            elif domain == "metadata_completeness":
                _d["subchecks"] = _metadata_completeness_subchecks(ev)
            else:
                _d["subchecks"] = {}
            _disabled_by_weight = float(_score_weights.get(domain, 1.0)) == 0.0
            if _disabled_by_weight:
                _d["component_score"] = 0
                _d["severity"] = "not_applicable"
                _d["verdict"] = "not_applicable"
                _d["enabled"] = False
            elif domain != "region_consistency" or _d.get("enabled", True):
                _d["severity"] = _combine_domain_severity(
                    _d.get("severity", "normal"),
                    _d.get("subchecks", {}),
                    metadata=(domain == "metadata_completeness")
                )
                _d["verdict"] = _domain_verdict(domain, _d["severity"], _triage_out.get("recommendation", "keep"))
            _qc_domains_out[domain] = _d

        _final_profile = _profile_from_domains(_qc_domains_out)
        _profile_rec, _profile_driver = _recommend_from_profile(_final_profile, thr)
        _scalar_rec = _scalar_recommendation(float(_triage_out.get("score", 0)), flags, thr)
        _final_rec = _merge_recommendations(_profile_rec, _scalar_rec)
        if _task_mode_for_case == "pancreas_only" and not pancreas_p:
            _final_rec = "exclude"
        _final_risk = _risk_from_profile(_final_profile)
        _worst_structural_sev = max(
            (info["severity"] for domain, info in _final_profile.items()
             if domain != "metadata_completeness"),
            key=_severity_rank,
            default="normal",
        )
        _triage_out["recommendation"] = _final_rec
        _triage_out["risk_level"] = _final_risk
        _triage_out["rank_hint"] = (
            "omit"                  if _final_rec == "exclude" else
            "metadata_warning"      if _final_rec == "keep_with_metadata_warning" else
            "priority_review"       if _worst_structural_sev in ("high_warning", "critical") else
            "review"                if _final_rec == "review" else
            "low_priority"          if _worst_structural_sev == "low_warning" else
            "routine"
        )
        for domain, _d in _qc_domains_out.items():
            if not _d.get("enabled", True):
                continue
            _d["verdict"] = _domain_verdict(domain, _d["severity"], _final_rec)

        _pi_out = dict(_pi or {})
        _driver = _profile_driver
        if not _driver and _final_rec not in ("keep", "keep_with_metadata_warning"):
            scored_domains = {
                domain: float(info.get("component_score", 0))
                for domain, info in _qc_domains_out.items()
                if domain in _CANONICAL_DOMAINS and not str(info.get("severity", "normal")).startswith("not_applicable")
            }
            _driver = max(scored_domains, key=scored_domains.get) if scored_domains else None
        if _driver:
            _pi_out = {
                "domain": _driver,
                "failure_mode": _FAILURE_MODE_LABELS.get(_driver, _driver),
            }
        elif not _pi_out.get("domain") and not _pi_out.get("failure_mode"):
            _pi_out = {}
        _pi_subcheck = _primary_subcheck(_pi_out.get("domain"), _qc_domains_out)
        if _pi_subcheck:
            _pi_out["subcheck"] = _pi_subcheck

        # ── Retrieval tags (RAG) ────────────────────────────────────────────
        _retrieval_tags = _build_retrieval_tags(
            v, ev, _triage_out, _pi_out, _qc_domains_out, _hu, _subregion_qc,
        )

        # ── Layer 1: Measurements — cleaned (renamed from "evidence") ───────
        # • total_lesion_volume_mm3: omit when equal to lesion_volume_mm3 (single lesion)
        # • h_e / b_o / t_a overlap fields: omit when all zero (no subregion info)
        # • hole_annotation_detected / regions_added_overlap: omit when False (noise-free)
        _lv    = ev.get("lesion_volume_mm3")
        _tlv   = ev.get("total_lesion_volume_mm3")
        _h_e   = ev.get("h_e_overlap_mm3", 0) or 0
        _b_o   = ev.get("b_o_overlap_mm3", 0) or 0
        _t_a   = ev.get("t_a_overlap_mm3", 0) or 0
        _meas_exclude = set(_HU_EV_FIELDS)
        if _tlv == _lv:
            _meas_exclude.add("total_lesion_volume_mm3")
        if _h_e == 0 and _b_o == 0 and _t_a == 0:
            _meas_exclude.update({"h_e_overlap_mm3", "b_o_overlap_mm3", "t_a_overlap_mm3"})
        if not ev.get("hole_annotation_detected", False):
            _meas_exclude.add("hole_annotation_detected")
        if not ev.get("regions_added_overlap", False):
            _meas_exclude.add("regions_added_overlap")

        measurements = {
            "pancreas_present"      : ev.get("pancreas_present"),
            "lesion_present"        : ev.get("lesion_present"),
            "pancreas_volume_mm3"   : ev.get("pancreas_volume_mm3"),
            "lesion_volume_mm3"     : ev.get("lesion_volume_mm3"),
            "overlap_rate_vs_lesion": ev.get("overlap_rate_vs_lesion"),
            **{k: v2 for k, v2 in ev.items()
               if k not in _EV_PRIMARY | _meas_exclude},
        }

        # ── HU block (unchanged) ────────────────────────────────────────────
        hu_block = {
            "ct_phase"                      : _hu.get("ct_phase"),
            "hu_statistics_status"          : _hu.get("hu_statistics_status"),
            "tumor_mean_hu"                 : _hu.get("tumor_mean_hu"),
            "tumor_median_hu"               : _hu.get("tumor_median_hu"),
            "tumor_std_hu"                  : _hu.get("tumor_std_hu"),
            "tumor_p05_hu"                  : _hu.get("tumor_p05_hu"),
            "tumor_p25_hu"                  : _hu.get("tumor_p25_hu"),
            "tumor_p75_hu"                  : _hu.get("tumor_p75_hu"),
            "tumor_p95_hu"                  : _hu.get("tumor_p95_hu"),
            "tumor_voxel_count"             : _hu.get("tumor_voxel_count"),
            "pancreas_mean_hu"              : _hu.get("pancreas_mean_hu"),
            "pancreas_median_hu"            : _hu.get("pancreas_median_hu"),
            "pancreas_std_hu"               : _hu.get("pancreas_std_hu"),
            "pancreas_p05_hu"               : _hu.get("pancreas_p05_hu"),
            "pancreas_p25_hu"               : _hu.get("pancreas_p25_hu"),
            "pancreas_p75_hu"               : _hu.get("pancreas_p75_hu"),
            "pancreas_p95_hu"               : _hu.get("pancreas_p95_hu"),
            "pancreas_voxel_count"          : _hu.get("pancreas_voxel_count"),
            "delta_hu_tumor_vs_pancreas"    : _hu.get("delta_hu_tumor_vs_pancreas"),
            "z_score_hu"                    : _hu.get("z_score_hu"),
            "tumor_attenuation"             : _hu.get("tumor_attenuation", "Unknown"),
            "attenuation_rule_used"         : _hu.get("attenuation_rule_used", "unknown"),
            "delta_hu_attenuation"          : _hu.get("delta_hu_attenuation", "Unknown"),
            "attenuation_rule_disagreement" : bool(_hu.get("attenuation_rule_disagreement", False)),
            "suspicious_hu_distribution"    : suspicious_hu,
        }

        _retrieval_summary = {
            "task_mode"             : v.get("task_mode", "unknown"),
            "training_objective"    : _training_objective(v.get("task_mode", "unknown")),
            "primary_target"        : _primary_target(v.get("task_mode", "unknown")),
            "sample_type"          : v.get("sample_type", "unknown"),
            "primary_issue_domain" : _pi_out.get("domain", "none"),
            "primary_issue_subcheck": _pi_out.get("subcheck", "none"),
            "primary_issue_failure_mode": _pi_out.get("failure_mode", "none"),
            "risk_level"           : _triage_out.get("risk_level", "low"),
            "recommendation"       : _triage_out.get("recommendation", "keep"),
        }

        out[case_id] = {
            "training_objective": _training_objective(v.get("task_mode", "unknown")),
            "target_presence": _build_target_presence(v.get("task_mode", "unknown"), ev),
            # ── Layer 4: Triage (authoritative decision) ───────────────────
            "triage"          : _triage_out,
            # ── Layer 3: Primary issue ─────────────────────────────────────
            "primary_issue"   : _pi_out if _pi_out else None,
            # ── Layer 2: Domain assessment ─────────────────────────────────
            "qc_domains"      : _qc_domains_out,
            # ── RAG retrieval index ────────────────────────────────────────
            "retrieval_tags"     : _retrieval_tags,
            "retrieval_summary"  : _retrieval_summary,
            "measurements"       : measurements,
            "hu_statistics"      : hu_block,
        }

    payload = {"metadata": report_metadata, "cases": out} if report_metadata else out
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                           encoding="utf-8")


def generate_csv_report(qc_results: dict, report_path: Path, data: dict | None = None) -> None:
    all_tags: list[str] = sorted({t for v in qc_results.values()
                                   for _, t, _ in v["flags"]})
    base_fields = [
                   # ── primary columns ──────────────────────────────────────────
                   "case_id",
                   "triage_score", "risk_level", "recommendation",
                   "primary_issue_domain", "primary_issue_subcheck", "primary_issue_failure_mode",
                   "geometry_integrity_score",    "geometry_integrity_severity",    "geometry_integrity_verdict",
                   "lesion_localization_score",   "lesion_localization_severity",   "lesion_localization_verdict",
                   "lesion_burden_score",         "lesion_burden_severity",         "lesion_burden_verdict",
                   "pancreas_context_score",      "pancreas_context_severity",      "pancreas_context_verdict",
                   "fov_integrity_score",         "fov_integrity_severity",         "fov_integrity_verdict",
                   "region_consistency_score",    "region_consistency_severity",    "region_consistency_verdict",
                   "attenuation_integrity_score", "attenuation_integrity_severity", "attenuation_integrity_verdict",
                   "metadata_completeness_score", "metadata_completeness_severity", "metadata_completeness_verdict",
                   # ── numeric evidence ────────────────────────────────────────
                   "pancreas_present", "lesion_present",
                   "pancreas_volume_mm3", "lesion_volume_mm3",
                   "overlap_rate_vs_lesion",
                   "n_conflicts", "region_relative_error",
                   "region_coverage_ratio",
                   "partial_visibility_likely",
                   "anatomical_truncation_suspected",
                   "spacing_x", "spacing_y", "spacing_z",
                   # ── HU statistics ───────────────────────────────────────────
                   "hu_statistics_status",
                   "tumor_median_hu", "tumor_mean_hu", "tumor_std_hu",
                   "pancreas_median_hu", "pancreas_std_hu",
                   "delta_hu_tumor_vs_pancreas", "z_score_hu",
                   "tumor_attenuation", "attenuation_rule_used",
                   "suspicious_hu_distribution",
                   # ── retrieval tags ──────────────────────────────────────────
                   "retrieval_tags",
                   ]
    fieldnames = base_fields + all_tags

    rows: list[dict] = []
    for case_id, v in qc_results.items():
        case_tags = _tags(v["flags"])
        ev   = v["evidence"]
        sp   = ev.get("spacing_xyz_mm") or []
        sc   = v.get("score_components", {})
        qc_doms = v.get("qc_domains", {})
        _triage = v.get("triage", {})
        _pi     = v.get("primary_issue", {}) or {}
        _raw_case = (data or {}) if isinstance(data, dict) else {}
        hu   = (_raw_case.get(case_id) or {}).get("hu_statistics") or {}

        def _dom_val(domain: str, field: str, default=""):
            return qc_doms.get(domain, {}).get(field, default)

        row: dict = {
            "case_id"                           : case_id,
            "triage_score"                      : _triage.get("score", v["score"]),
            "risk_level"                        : _triage.get("risk_level", v["risk_level"]),
            "recommendation"                    : _triage.get("recommendation", v["recommendation"]),
            "primary_issue_domain"              : _pi.get("domain", ""),
            "primary_issue_subcheck"            : _pi.get("subcheck", ""),
            "primary_issue_failure_mode"        : _pi.get("failure_mode", ""),
            "geometry_integrity_score"          : _dom_val("geometry_integrity", "component_score"),
            "geometry_integrity_severity"       : _dom_val("geometry_integrity", "severity"),
            "geometry_integrity_verdict"        : _dom_val("geometry_integrity", "verdict"),
            "lesion_localization_score"         : _dom_val("lesion_localization", "component_score"),
            "lesion_localization_severity"      : _dom_val("lesion_localization", "severity"),
            "lesion_localization_verdict"       : _dom_val("lesion_localization", "verdict"),
            "lesion_burden_score"               : _dom_val("lesion_burden", "component_score"),
            "lesion_burden_severity"            : _dom_val("lesion_burden", "severity"),
            "lesion_burden_verdict"             : _dom_val("lesion_burden", "verdict"),
            "pancreas_context_score"            : _dom_val("pancreas_context", "component_score"),
            "pancreas_context_severity"         : _dom_val("pancreas_context", "severity"),
            "pancreas_context_verdict"          : _dom_val("pancreas_context", "verdict"),
            "fov_integrity_score"               : _dom_val("fov_integrity", "component_score"),
            "fov_integrity_severity"            : _dom_val("fov_integrity", "severity"),
            "fov_integrity_verdict"             : _dom_val("fov_integrity", "verdict"),
            "region_consistency_score"          : _dom_val("region_consistency", "component_score"),
            "region_consistency_severity"       : _dom_val("region_consistency", "severity"),
            "region_consistency_verdict"        : _dom_val("region_consistency", "verdict"),
            "attenuation_integrity_score"       : _dom_val("attenuation_integrity", "component_score"),
            "attenuation_integrity_severity"    : _dom_val("attenuation_integrity", "severity"),
            "attenuation_integrity_verdict"     : _dom_val("attenuation_integrity", "verdict"),
            "metadata_completeness_score"       : _dom_val("metadata_completeness", "component_score"),
            "metadata_completeness_severity"    : _dom_val("metadata_completeness", "severity"),
            "metadata_completeness_verdict"     : _dom_val("metadata_completeness", "verdict"),
            "pancreas_present"                  : int(ev.get("pancreas_present") or False),
            "lesion_present"                    : int(ev.get("lesion_present") or False),
            "pancreas_volume_mm3"               : ev.get("pancreas_volume_mm3", ""),
            "lesion_volume_mm3"                 : ev.get("lesion_volume_mm3", ""),
            "overlap_rate_vs_lesion"            : ev.get("overlap_rate_vs_lesion", ""),
            "n_conflicts"                       : sum(1 for s, _, _ in v["flags"] if s != "INFO"),
            "region_relative_error"             : ev.get("region_relative_error", ""),
            "region_coverage_ratio"             : ev.get("region_coverage_ratio", ""),
            "partial_visibility_likely"         : int(ev["partial_visibility_likely"])
                                                  if ev.get("partial_visibility_likely") is not None
                                                  else "",
            "anatomical_truncation_suspected"   : int(ev["anatomical_truncation_suspected"])
                                                  if ev.get("anatomical_truncation_suspected") is not None
                                                  else "",
            "spacing_x"                         : sp[0] if len(sp) > 0 else "",
            "spacing_y"                         : sp[1] if len(sp) > 1 else "",
            "spacing_z"                         : sp[2] if len(sp) > 2 else "",
            "hu_statistics_status"              : hu.get("hu_statistics_status", ""),
            "tumor_median_hu"                   : hu.get("tumor_median_hu", ""),
            "tumor_mean_hu"                     : hu.get("tumor_mean_hu", ""),
            "tumor_std_hu"                      : hu.get("tumor_std_hu", ""),
            "pancreas_median_hu"                : hu.get("pancreas_median_hu", ""),
            "pancreas_std_hu"                   : hu.get("pancreas_std_hu", ""),
            "delta_hu_tumor_vs_pancreas"        : hu.get("delta_hu_tumor_vs_pancreas", ""),
            "z_score_hu"                        : hu.get("z_score_hu", ""),
            "tumor_attenuation"                 : hu.get("tumor_attenuation", ""),
            "attenuation_rule_used"             : hu.get("attenuation_rule_used", ""),
            "suspicious_hu_distribution"        : int(ev.get("suspicious_hu_distribution") or False),
            "retrieval_tags"                    : "|".join(v.get("retrieval_tags", [])),
        }
        for tag in all_tags:
            row[tag] = 1 if tag in case_tags else 0
        rows.append(row)

    with report_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def generate_debug_json(qc_results: dict, report_path: Path,
                        report_metadata: dict | None = None) -> None:
    """Write qc_report_debug.json — internal scoring details for each case.

    Adds a ``_debug`` block per case containing:
    - ``score_components``  : raw and capped per-domain penalty values
    - ``flags``             : all QC flags as (severity, tag, message) triples
    - ``evidence``          : full evidence dict from run_qc
    - ``scoring_details``   : hybrid formula breakdown from triage
    - ``consistency_issues``: any violations detected by _assert_case_consistency
    """
    out: dict = {}
    if report_metadata:
        out["_report_metadata"] = report_metadata

    out["cases"] = {}
    for case_id, v in qc_results.items():
        sc     = v.get("score_components", {})
        flags  = v.get("flags", [])
        ev     = v.get("evidence", {})
        triage = v.get("triage", {})
        sd     = (triage.get("scoring_details") or {})

        out["cases"][case_id] = {
            "score"              : v.get("score"),
            "risk_level"         : v.get("risk_level"),
            "score_components"   : sc,
            "flags"              : [
                {"severity": s, "tag": t, "message": m}
                for s, t, m in flags
            ],
            "evidence"           : ev,
            "scoring_details"    : sd,
            "consistency_issues" : _assert_case_consistency(case_id, v),
        }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

CAUSES: dict[str, str] = {
    "pancreas_mask_empty": dedent("""\
        * CT acquired without the pancreas in field-of-view
        * Post-operative total pancreatectomy
        * Segmentation algorithm failure (out-of-distribution case)
        * File export error
        * Incorrect case pairing"""),
    "lesion_outside_pancreas": dedent("""\
        * Coordinate system mismatch between CT image and label file
        * Wrong label file paired with this CT image
        * Extra-pancreatic lesion (duodenal tumor, lymph node)
        * Annotation spill-over beyond the pancreas boundary"""),
    "region_sum_error": dedent("""\
        * Sub-region masks derived from an older version of the whole-pancreas mask
        * Automated sub-region splitting failed
        * Manual annotation inconsistency between annotators"""),
    "incomplete_anatomical_coverage": dedent("""\
        * Scan has limited field of view — only part of the pancreas is visible
        * Post-processing truncation removed part of the anatomy
        * Incomplete annotation where only visible regions were segmented
        * Only body+tail visible (common in portal-venous phase or cropped scans)"""),
    "small_pancreas": dedent("""\
        * Post-surgical pancreatic remnant (partial pancreatectomy)
        * Pancreatic atrophy (chronic pancreatitis)
        * Segmentation failure — only a fragment was captured"""),
    "invalid_spacing": dedent("""\
        * Highly anisotropic acquisition (thick axial slices)
        * Header corruption during DICOM-to-NIfTI conversion
        * Wrong units (stored in cm instead of mm)"""),
    "shape_mismatch": dedent("""\
        * Partial volume export (only a slab was saved)
        * Crop/pad error during preprocessing
        * Wrong case pairing"""),
}

CORRECTIONS: dict[str, str] = {
    "pancreas_mask_empty": dedent("""\
        Task-aware (v3): action depends on lesion status.
        If lesion absent:
          → Case is a valid negative. No correction needed unless pancreas
            is clinically expected to be present.
        If lesion present:
          1. [VERIFY]      Open CT — is pancreas visible?
          2. [RE-SEGMENT]  Re-segment manually or with an automated tool
          3. [REVIEW]      If pancreas truly absent, case cannot be
                           anatomically validated — tag as review_positive"""),
    "lesion_no_anatomical_validation": dedent("""\
        1. [INSPECT]     Visually confirm lesion location on CT
        2. [RE-SEGMENT]  Try to generate a pancreas mask for this case
        3. [REVIEW]      If pancreas absent by design (post-op), accept as
                         unvalidated positive with reduced weight"""),
    "lesion_outside_pancreas": dedent("""\
        1. [INSPECT]     Visually overlay lesion and pancreas masks on CT
        2. [RE-REGISTER] Apply the missing spatial transform
        3. [CLIP]        lesion_fixed = lesion AND pancreas
        4. [EXCLUDE]     For lesion segmentation training"""),
    "region_sum_error": dedent("""\
        1. [INSPECT]     Overlay head/body/tail on whole-pancreas mask
        2. [RECOMPUTE]   Re-run automated sub-region splitting
        3. [CLIP]        head_fixed = head AND pancreas"""),
    "incomplete_anatomical_coverage": dedent("""\
        Partial anatomical coverage — likely expected for truncated scans.
        1. [VERIFY]    Check CT at scan borders for visible pancreas extent
        2. [ACCEPT]    If truncation is confirmed, no correction needed;
                       keep for pancreas/lesion segmentation tasks
        3. [EXCLUDE]   Remove from head/body/tail sub-region tasks
        4. [ANNOTATE]  Add missing sub-region masks only if region is visible"""),
    "small_pancreas": dedent("""\
        1. [INSPECT]     Verify mask covers visible pancreas
        2. [CHECK-META]  Confirm clinical context: post-surgery? atrophy?
        3. [RE-SEGMENT]  Re-run with adjusted parameters"""),
    "invalid_spacing": dedent("""\
        1. [CHECK-HEADER] Verify spacing in NIfTI header with nibabel
        2. [FIX-HEADER]   Scale spacing if units are wrong
        3. [RE-EXPORT]    Re-export from original DICOM"""),
}

TRAINING_IMPACTS: dict[str, str] = {
    "pancreas_mask_empty"             : (
        "If lesion absent: valid negative sample — no training impact. "
        "If lesion present: anatomical validation impossible; review before use."),
    "lesion_outside_pancreas"         : "Wrong-location labels -> inflated FPR. EXCLUDE from lesion training.",
    "lesion_no_anatomical_validation" : (
        "Cannot confirm lesion is inside pancreas. Use with caution; "
        "may act as noisy positive — apply loss weighting or review manually."),
    "region_sum_error"                : "Inconsistent region supervision. EXCLUDE from head/body/tail tasks (>30%).",
    "incomplete_anatomical_coverage"  : (
        "Only a subset of anatomical regions is present due to truncated FOV. "
        "Safe for pancreas/lesion segmentation. EXCLUDE from head/body/tail sub-region tasks."),
    "small_pancreas"                  : "Severe class imbalance in sample -> destabilises Dice loss. Reduce weight.",
    "invalid_spacing"                 : "Anisotropy causes feature anisotropy. Resample before training.",
}


# ═══════════════════════════════════════════════════════════════════════════════
# GLOSSARY MANAGER
# ═══════════════════════════════════════════════════════════════════════════════

class GlossaryManager:
    """
    Interactive glossary for dataset QC terms.

    Loaded once by QCAgent and queried via  help <term>  in the console.
    Dynamically augments static definitions with live examples from the
    in-memory QC results so counts and case lists stay current.
    """

    # ── Static definitions ────────────────────────────────────────────────────
    _DEFINITIONS: dict[str, dict] = {
        # ── Score / risk ──────────────────────────────────────────────────────
        "qc_score": {
            "aliases": ["qc score", "risk score", "score", "final score"],
            "definition": (
                "Integer 0–100 summarising annotation quality risk. "
                "Computed from all active score_components using the configured "
                "score version: additive sums components, hybrid uses geometry "
                "and lesion localization directly plus weighted secondary components "
                "(burden, context, FOV, region, attenuation, metadata). "
                "Low <31 — safe to train. Medium 31–60 — review before use. "
                "High ≥61 — likely problematic; exclude or correct first."
            ),
            "relevance": (
                "Drives the keep / review / exclude decision. A score of 0 "
                "does NOT mean perfect — it means no detected anomaly."
            ),
            "visual_check": "Check the Images for the largest contributors in score_components.",
            "score_impact": "IS the score; each component is capped at its individual maximum.",
            "decision_impact": "High → review_or_exclude_positive or exclude. Low → keep.",
        },
        "score_components": {
            "aliases": ["components", "score breakdown", "subscores"],
            "definition": (
                "Dict of per-dimension penalty scores that contribute to the final qc_score. "
                "Keys: lesion_localization (cap 70), geometry_integrity (cap 80), "
                "lesion_burden (cap 20), pancreas_context (cap 20), "
                "fov_integrity (cap 15), region_consistency (cap 10), "
                "attenuation_integrity (cap 10), metadata_completeness (cap 10). "
                "Private keys starting with _ are informational only and not summed."
            ),
            "relevance": "Shows which QC dimension drives the total risk score.",
            "visual_check": "Compare the dominant component against the case's annotations.",
            "score_impact": "Each key contributes its value, capped at its per-key maximum.",
            "decision_impact": "Component with the highest value becomes dominant_failure_mode.",
        },
        "dominant_failure_mode": {
            "aliases": ["dominant issue", "dominant failure", "main issue", "top issue"],
            "definition": (
                "Human-readable label for the score_component with the highest value. "
                "Possible values: lesion_localization_failure, geometry_integrity_failure, "
                "lesion_burden_abnormality, pancreas_context_issue, "
                "fov_integrity_issue, anatomical_region_inconsistency, "
                "suspicious_attenuation, missing_metadata, none."
            ),
            "relevance": (
                "Two cases can share the same total score but differ entirely in what is "
                "wrong. The dominant_failure_mode tells you WHERE to look first."
            ),
            "visual_check": (
                "lesion_localization_failure → check lesion mask placement in 3D slicer. "
                "geometry_integrity_failure → check voxel spacing and affine. "
                "lesion_burden_abnormality → compare lesion/pancreas volume ratio."
            ),
            "score_impact": "Determined by max(score_components); does not directly add to score.",
            "decision_impact": "Lesion-localization failures with high penalty → review_or_exclude.",
        },
        # ── Score components ──────────────────────────────────────────────────
        "lesion_localization": {
            "aliases": [
                "overlap severity", "overlap score", "lesion overlap penalty",
                "lesion pancreas overlap", "overlap penalty",
                "overlap_severity_score", "lesion_localization",
            ],
            "definition": (
                "Penalty (0–85) for poor lesion-to-pancreas spatial overlap. "
                "Uses a continuous v2 curve: "
                "overlap ≥80% → 0; 30–80% → 1–60 (linear); "
                "5–30% → 60–85 (steep); <5% → 85 (catastrophic). "
                "Capped at 70 in the component sum."
            ),
            "relevance": (
                "The primary QC dimension for pancreas lesion segmentation. "
                "A lesion annotated outside the pancreas is a wrong-location label "
                "that will train the model to predict lesions in incorrect anatomy."
            ),
            "visual_check": (
                "Open the case in 3D Slicer. Overlay lesion.nii.gz on pancreas.nii.gz. "
                "Visually confirm whether the lesion is anatomically plausible even if "
                "the numeric overlap is low (e.g. peri-pancreatic lesions on the boundary)."
            ),
            "score_impact": "Dominates qc_score when overlap is poor. Cap=70.",
            "decision_impact": (
                "overlap < 40% → review_or_exclude_positive. "
                "overlap 40–70% → review_positive. "
                "overlap ≥ 70% → keep_positive (if no other issues)."
            ),
        },
        "geometry_integrity": {
            "aliases": [
                "geometry", "affine", "affine mismatch", "spacing mismatch",
                "geometry penalty", "geometry integrity",
                "geometry_score", "geometry_integrity",
            ],
            "definition": (
                "Penalty (0–80) for geometric header problems: "
                "affine mismatch between image and mask, abnormal voxel spacing, "
                "non-orthonormal direction cosines, or extreme anisotropy. "
                "These indicate the mask was not generated in the same coordinate "
                "space as the image."
            ),
            "relevance": (
                "A geometry failure means the mask and image are misaligned. "
                "Training on misaligned data produces a model that hallucinates "
                "lesions or systematically mislocates them."
            ),
            "visual_check": (
                "Load image.nii.gz and lesion.nii.gz in the same 3D Slicer session. "
                "If the lesion appears in the wrong anatomical position, the affine "
                "is wrong. Also check spacing values in the JSON evidence."
            ),
            "score_impact": "Highest individual cap (80) — can dominate the total score.",
            "decision_impact": "High geometry score → likely exclude unless correctable.",
        },
        "lesion_burden": {
            "aliases": [
                "lesion burden", "burden score", "lesion size", "burden abnormality",
                "lesion volume", "lesion pancreas ratio",
                "lesion_burden_score", "lesion_burden",
            ],
            "definition": (
                "Penalty (0–20) for abnormal lesion-to-pancreas volume ratio. "
                "Triggered when lesion volume exceeds a large fraction of the "
                "pancreas volume, suggesting either an over-segmented lesion or "
                "a whole-organ involvement (e.g. diffuse PDAC)."
            ),
            "relevance": (
                "Very large lesions relative to the pancreas create extreme class "
                "imbalance per sample. They can destabilise Dice-based losses "
                "and should be weighted or reviewed."
            ),
            "visual_check": (
                "Compare lesion_volume_mm3 vs pancreas_volume_mm3 in the JSON evidence. "
                "Ratio > 1.0 means the lesion is larger than the whole pancreas — "
                "verify it is not a whole-pancreas mis-segmentation."
            ),
            "score_impact": "Minor contributor; cap=20.",
            "decision_impact": "Rarely decisive alone; elevates score when combined with overlap issues.",
        },
        "pancreas_context": {
            "aliases": [
                "pancreas context", "context score", "pancreas size",
                "small pancreas", "pancreas context penalty",
                "pancreas_context_score", "pancreas_context",
            ],
            "definition": (
                "Penalty (0–20) for abnormal pancreas context: "
                "very small or absent pancreas, or pancreas volume far outside "
                "the expected range for the training cohort. "
                "Small pancreas → atrophic disease; absent → no anatomical validation."
            ),
            "relevance": (
                "Pancreas atrophy is associated with chronic pancreatitis and "
                "post-operative states. Very small pancreas volumes inflate class "
                "imbalance and may represent annotation of scar tissue rather than "
                "true organ."
            ),
            "visual_check": "Check pancreas_volume_mm3 in evidence. Normal adult range: ~50–120 cm³.",
            "score_impact": "Minor contributor; cap=20.",
            "decision_impact": "Informs keep/review but rarely triggers exclusion alone.",
        },
        "region_consistency": {
            "aliases": [
                "region consistency", "region score", "region sum", "region error",
                "head body tail", "sub-region", "region inconsistency",
                "region_consistency_score", "region_consistency",
            ],
            "definition": (
                "Penalty (0–15, capped) for mismatch between head+body+tail sub-region "
                "volumes and total pancreas volume. Triggered when the deviation exceeds "
                "10%. Raw penalty can be higher but is capped at 15 for the "
                "pancreas_lesion_segmentation task because sub-region consistency "
                "is secondary to lesion localisation."
            ),
            "relevance": (
                "Sub-region masks are used for auxiliary training tasks (head/body/tail "
                "segmentation). Inconsistency suggests overlapping or missing region "
                "annotations. For lesion training it is informational."
            ),
            "visual_check": (
                "Compare region volumes in the evidence dict. "
                "Deviation > 30% is critical; 10–30% is a warning."
            ),
            "score_impact": "Capped at 15 deliberately — does not dominate lesion QC scores.",
            "decision_impact": "Low weight for pancreas_lesion_segmentation task.",
        },
        "metadata_completeness": {
            "aliases": [
                "metadata", "metadata penalty", "missing metadata",
                "sex", "age", "manufacturer", "missing fields",
                "metadata_score", "metadata_completeness",
            ],
            "definition": (
                "Penalty (0–10) for missing or implausible metadata fields: "
                "sex, age, CT manufacturer, scanner model, study type, CT phase. "
                "Each missing field adds a small fixed penalty."
            ),
            "relevance": (
                "Missing metadata complicates stratified splits, site-aware "
                "normalisation, and harmonisation experiments. It does not affect "
                "image quality or annotation correctness directly."
            ),
            "visual_check": "Check the metadata block in 'detail PanTS_XXXXX'.",
            "score_impact": "Minor contributor; cap=10.",
            "decision_impact": "Metadata alone never triggers exclusion.",
        },
        # ── Evidence / overlap fields ─────────────────────────────────────────
        "b_o_overlap_mm3": {
            "aliases": ["bo overlap", "body overlap", "b_o overlap"],
            "definition": (
                "Volume (mm³) of the lesion mask that overlaps with the pancreas "
                "body sub-region mask. Stored in the evidence dict. "
                "Helps localise where in the pancreas the lesion sits."
            ),
            "relevance": "Useful for sub-region lesion localisation analysis.",
            "visual_check": "Compare b_o_overlap_mm3 vs h_e_overlap_mm3 and t_a_overlap_mm3.",
            "score_impact": "Not directly scored; contributes to the overall overlap_rate_vs_lesion.",
            "decision_impact": "Informational only.",
        },
        "h_e_overlap_mm3": {
            "aliases": ["he overlap", "head overlap", "h_e overlap"],
            "definition": (
                "Volume (mm³) of the lesion mask that overlaps with the pancreas "
                "head sub-region mask. Stored in the evidence dict."
            ),
            "relevance": "Most pancreatic ductal adenocarcinomas arise in the head — high h_e_overlap is expected.",
            "visual_check": "If h_e_overlap_mm3 is 0 and lesion is present, verify head mask integrity.",
            "score_impact": "Informational only.",
            "decision_impact": "Informational only.",
        },
        "t_a_overlap_mm3": {
            "aliases": ["ta overlap", "tail overlap", "t_a overlap"],
            "definition": (
                "Volume (mm³) of the lesion mask that overlaps with the pancreas "
                "tail sub-region mask. Stored in the evidence dict."
            ),
            "relevance": "Useful for sub-region lesion localisation analysis.",
            "visual_check": "Compare t_a_overlap_mm3 vs h_e_overlap_mm3 and b_o_overlap_mm3.",
            "score_impact": "Informational only.",
            "decision_impact": "Informational only.",
        },
        "region_consistency_note": {
            "aliases": ["consistency note", "region note", "capped region"],
            "definition": (
                "Free-text note in the JSON report explaining whether the "
                "region_consistency component score was capped, and the raw (uncapped) penalty. "
                "Example: 'capped at 15 (raw=25) for pancreas_lesion_segmentation task'."
            ),
            "relevance": "Allows downstream consumers to retrieve the uncapped penalty if needed.",
            "visual_check": "Inspect uncapped_region_penalty in the JSON report.",
            "score_impact": "The capped value is used in scoring; raw is stored for audit.",
            "decision_impact": "None — informational annotation.",
        },
        "lesion_pancreas_ratio": {
            "aliases": ["volume ratio", "lesion ratio", "burden ratio", "lesion to pancreas"],
            "definition": (
                "lesion_volume_mm3 / pancreas_volume_mm3. "
                "Ratio > 1.0 means the lesion is larger than the whole pancreas. "
                "Stored in evidence; used by lesion_burden (cap 20)."
            ),
            "relevance": (
                "High ratios indicate either diffuse organ involvement or over-segmentation. "
                "Both scenarios require review before training."
            ),
            "visual_check": "Check total_lesion_volume_mm3 and pancreas_volume_mm3 in 'detail'.",
            "score_impact": "Feeds into lesion_burden (cap 20).",
            "decision_impact": "Ratio > 0.5 → likely minor flag; ratio > 1.0 → review.",
        },
        # ── Anatomical validation ─────────────────────────────────────────────
        "anatomical_validation_possible": {
            "aliases": [
                "anatomical validation", "anat validation", "validation possible",
                "can validate", "no anatomical validation",
            ],
            "definition": (
                "Boolean. True when both pancreas mask and lesion mask are present "
                "and non-empty, allowing the lesion-pancreas overlap to be computed. "
                "False when pancreas mask is absent while a lesion is present — "
                "the lesion location cannot be verified."
            ),
            "relevance": (
                "If False on a positive case, the case is structurally unvalidatable. "
                "It may still be usable if you trust the source annotation, but "
                "it should be flagged for manual review."
            ),
            "visual_check": "Check whether pancreas.nii.gz is all-zeros for this case.",
            "score_impact": "When False + lesion present: lesion_no_anatomical_validation flag adds to score.",
            "decision_impact": "Positive + no anat validation → review_positive or exclude.",
        },
        "partial_anatomical_coverage": {
            "aliases": [
                "partial coverage", "anatomical coverage", "partial visibility",
                "incomplete coverage",
            ],
            "definition": (
                "Qualitative flag indicating that the field of view (FOV) likely does not "
                "cover the full pancreas. Detected by checking whether pancreas voxels "
                "touch the image border in Z, and comparing sub-region coverage ratios. "
                "See also: partial_visibility_likely."
            ),
            "relevance": (
                "Truncated FOV means the training sample teaches the model on a "
                "partial organ. For whole-pancreas segmentation this is fine if "
                "handled correctly; for sub-region tasks it introduces bias."
            ),
            "visual_check": "Check whether the pancreas is cut off at the top/bottom in the axial view.",
            "score_impact": "Triggers incomplete_anatomical_coverage flag; small geometry penalty.",
            "decision_impact": "Safe for pancreas/lesion tasks. Exclude from head/body/tail tasks.",
        },
        "incomplete_anatomical_coverage": {
            "aliases": [
                "incomplete coverage", "fov truncation", "truncated fov",
                "incomplete anatomical",
            ],
            "definition": (
                "QC flag (WARNING) indicating that the CT volume covers only a subset "
                "of the expected anatomy due to FOV truncation. Sub-region masks for "
                "the missing part will be empty or undersized."
            ),
            "relevance": "Same as partial_anatomical_coverage — see above.",
            "visual_check": "Scroll through the axial slices looking for abrupt organ truncation.",
            "score_impact": "Small contribution to geometry_integrity.",
            "decision_impact": "Keep for pancreas/lesion tasks. Review for sub-region tasks.",
        },
        "partial_visibility_likely": {
            "aliases": ["partial visibility", "visibility flag", "partial vis"],
            "definition": (
                "Boolean stored in the evidence dict. True when heuristics detect "
                "that the pancreas is probably cut at the image boundary "
                "(e.g. border-touching voxels, low region_coverage_ratio)."
            ),
            "relevance": "Soft signal for FOV truncation; used to generate incomplete_anatomical_coverage.",
            "visual_check": "Also check border_touching and region_coverage_ratio in the evidence.",
            "score_impact": "Indirect — feeds into geometry penalty when combined with other signals.",
            "decision_impact": "Informational; triggers deeper review of anatomical coverage.",
        },
        "anatomical_truncation_suspected": {
            "aliases": ["truncation", "truncation suspected", "fov cut"],
            "definition": (
                "Boolean in evidence. Stronger signal than partial_visibility_likely. "
                "True when region sum error + border touching + low coverage ratio "
                "all point to FOV truncation as the cause of region inconsistency."
            ),
            "relevance": (
                "Distinguishes annotation error from legitimate scan truncation. "
                "If True, the region_sum_error may not reflect an annotation mistake "
                "but simply a cropped FOV."
            ),
            "visual_check": "Verify in the original DICOM whether the scan was acquired with restricted FOV.",
            "score_impact": "Reduces region penalty if truncation is the most likely cause.",
            "decision_impact": "Affects whether region_sum_error is treated as annotation error or scan artifact.",
        },
        # ── Decision hints ────────────────────────────────────────────────────
        "keep_positive": {
            "aliases": ["keep positive", "keep lesion", "keep decision"],
            "definition": (
                "decision_hint assigned to positive cases where overlap ≥ 70% "
                "and no critical geometry flags. The case is considered safe "
                "to include as a positive training sample."
            ),
            "relevance": "These cases form the clean positive training set.",
            "visual_check": "Spot-check a sample of keep_positive cases to validate the threshold.",
            "score_impact": "Typically score < 31 (low risk).",
            "decision_impact": "Include in training without further review.",
        },
        "keep_negative": {
            "aliases": ["keep negative", "negative sample", "background sample"],
            "definition": (
                "decision_hint for negative samples (no lesion mask) with no "
                "critical flags. Valid background training examples for lesion "
                "segmentation. The model needs these to learn true-negative anatomy."
            ),
            "relevance": (
                "Negatives teach the model not to predict lesions in healthy anatomy. "
                "Removing them creates a dataset with only positives, which inflates "
                "false-positive rate at inference."
            ),
            "visual_check": "Verify that lesion.nii.gz is absent or all-zeros.",
            "score_impact": "Score is typically 0 unless metadata or region issues exist.",
            "decision_impact": "Include in training. Do NOT exclude negatives without cause.",
        },
        "review_positive": {
            "aliases": ["review positive", "review lesion", "needs review"],
            "definition": (
                "decision_hint for positive cases where overlap is between 40–70% "
                "OR other non-critical flags are present. The case may still be "
                "usable but warrants visual inspection before training."
            ),
            "relevance": "Boundary cases where the automatic QC is uncertain.",
            "visual_check": "Open in 3D Slicer and confirm that the lesion annotation is plausible.",
            "score_impact": "Typically score 31–60 (medium risk).",
            "decision_impact": "Manually inspect; keep or escalate to review_or_exclude_positive.",
        },
        "review_or_exclude_positive": {
            "aliases": [
                "review or exclude", "exclude positive", "high risk positive",
                "exclude lesion",
            ],
            "definition": (
                "decision_hint for positive cases where overlap < 40% "
                "OR geometry failures are present. The annotation is likely wrong. "
                "Default action is to exclude unless a domain expert confirms the "
                "annotation is anatomically valid (e.g. peri-pancreatic lesion)."
            ),
            "relevance": (
                "These cases will actively harm training if included: "
                "they teach the model that lesions outside the pancreas are correct."
            ),
            "visual_check": (
                "Check if the lesion is peri-pancreatic (could still be valid) "
                "or clearly in the wrong organ (liver, spleen, kidney)."
            ),
            "score_impact": "Score ≥ 61 (high risk).",
            "decision_impact": "Exclude by default; keep only after expert confirmation.",
        },
    }

    def __init__(self) -> None:
        self._qc_results: dict = {}
        # Build a flat alias → canonical_key index
        self._alias_index: dict[str, str] = {}
        for key, entry in self._DEFINITIONS.items():
            self._alias_index[key.lower()] = key
            for alias in entry.get("aliases", []):
                self._alias_index[alias.lower()] = key

    def load_qc_results(self, qc_results: dict) -> None:
        """Attach live QC results so explain() can show dynamic examples."""
        self._qc_results = qc_results

    # ── Public API ────────────────────────────────────────────────────────────

    def explain(self, term: str) -> str:
        """Return a full glossary entry for *term*, with live examples."""
        canonical = self._resolve(term)
        if canonical is None:
            return self._not_found(term)

        entry = self._DEFINITIONS[canonical]
        lines = [
            f"GLOSSARY: {canonical}",
            "─" * 60,
            f"Definition:\n  {entry['definition']}",
            "",
            f"Why it matters:\n  {entry['relevance']}",
            "",
            f"Effect on QC score:\n  {entry['score_impact']}",
            "",
            f"Effect on keep/review/exclude:\n  {entry['decision_impact']}",
            "",
            f"What to check visually:\n  {entry['visual_check']}",
        ]

        examples = self._find_examples(canonical)
        if examples:
            lines += ["", f"Live examples ({len(examples)} cases):"]
            for ex in examples[:5]:
                lines.append(f"  {ex}")
            if len(examples) > 5:
                lines.append(f"  … and {len(examples) - 5} more cases.")

        lines += ["", f"Aliases: {', '.join(entry.get('aliases', []))}"]
        return "\n".join(lines)

    def suggest_terms(self, term: str) -> str:
        """Return closest matching glossary terms when *term* is not found."""
        all_aliases = list(self._alias_index.keys())
        close = difflib.get_close_matches(term.lower(), all_aliases, n=5, cutoff=0.4)
        if not close:
            # Fall back to substring matching
            close = [a for a in all_aliases if term.lower() in a or a in term.lower()][:5]
        if not close:
            canonical_list = sorted(self._DEFINITIONS.keys())
            return (f"Unknown term: '{term}'\n"
                    f"Available glossary terms:\n  " +
                    "\n  ".join(canonical_list))
        suggestions = sorted({self._alias_index[c] for c in close})
        return (f"Unknown term: '{term}'\n"
                f"Did you mean one of:\n  " +
                "\n  ".join(suggestions) +
                "\n\nType  help <term>  with one of the above.")

    def list_terms(self) -> str:
        """Return a formatted list of all glossary terms."""
        lines = ["Available glossary terms (type  help <term>  for details):", ""]
        for key, entry in sorted(self._DEFINITIONS.items()):
            aliases = entry.get("aliases", [])
            alias_str = f"  ({', '.join(aliases[:3])})" if aliases else ""
            lines.append(f"  {key}{alias_str}")
        return "\n".join(lines)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _resolve(self, term: str) -> str | None:
        """Return the canonical key for *term*, or None if not found."""
        t = term.lower().strip()
        if t in self._alias_index:
            return self._alias_index[t]
        # Try close match
        close = difflib.get_close_matches(t, list(self._alias_index.keys()), n=1, cutoff=0.6)
        if close:
            return self._alias_index[close[0]]
        return None

    def _find_examples(self, canonical: str) -> list[str]:
        """Return short descriptive strings for cases related to *canonical*."""
        if not self._qc_results:
            return []
        examples: list[str] = []

        if canonical == "lesion_localization":
            for cid, v in self._qc_results.items():
                sc = v.get("score_components", {})
                ov = sc.get("lesion_localization", 0)
                if ov > 0:
                    rate = v.get("evidence", {}).get("overlap_rate_vs_lesion")
                    rate_str = f"{rate:.1%}" if rate is not None else "N/A"
                    examples.append(f"{cid}: penalty={ov}, overlap={rate_str}")
            examples.sort(key=lambda s: -int(s.split("penalty=")[1].split(",")[0]))

        elif canonical == "lesion_burden":
            for cid, v in self._qc_results.items():
                sc = v.get("score_components", {})
                if sc.get("lesion_burden", 0) > 0:
                    ev = v.get("evidence", {})
                    lv = ev.get("total_lesion_volume_mm3", 0)
                    pv = ev.get("pancreas_volume_mm3", 0)
                    ratio = f"{lv/pv:.2f}" if pv else "N/A"
                    examples.append(
                        f"{cid}: lesion={lv:,.0f} mm³, "
                        f"pancreas={pv:,.0f} mm³, ratio={ratio}")

        elif canonical == "geometry_integrity":
            for cid, v in self._qc_results.items():
                if v.get("score_components", {}).get("geometry_integrity", 0) > 0:
                    tags = [t for _, t, _ in v.get("flags", [])
                            if "affine" in t or "spacing" in t or "geometry" in t]
                    examples.append(f"{cid}: flags={tags or 'geometry issue'}")

        elif canonical == "region_consistency":
            for cid, v in self._qc_results.items():
                raw = v.get("score_components", {}).get("_region_raw", 0)
                if raw > 0:
                    dev = v.get("evidence", {}).get("region_sum_deviation")
                    dev_str = f"{dev:.1%}" if dev is not None else "N/A"
                    examples.append(f"{cid}: raw_penalty={raw}, deviation={dev_str}")
            examples.sort(key=lambda s: -int(s.split("raw_penalty=")[1].split(",")[0]))

        elif canonical in ("dominant_failure_mode", "score_components", "qc_score"):
            for cid, v in self._qc_results.items():
                if v.get("score", 0) > 60:
                    dom = v.get("dominant_failure_mode", "none")
                    examples.append(
                        f"{cid}: score={v['score']}, dominant={dom}, "
                        f"risk={v.get('risk_level','?')}")
            examples.sort(key=lambda s: -int(s.split("score=")[1].split(",")[0]))

        elif canonical in (
            "anatomical_validation_possible",
            "partial_anatomical_coverage",
            "incomplete_anatomical_coverage",
            "partial_visibility_likely",
            "anatomical_truncation_suspected",
        ):
            ev_key_map = {
                "anatomical_validation_possible"  : "anatomical_validation_possible",
                "partial_visibility_likely"       : "partial_visibility_likely",
                "anatomical_truncation_suspected" : "anatomical_truncation_suspected",
            }
            tag_map = {
                "incomplete_anatomical_coverage": "incomplete_anatomical_coverage",
                "partial_anatomical_coverage"   : "incomplete_anatomical_coverage",
            }
            if canonical in ev_key_map:
                ev_key = ev_key_map[canonical]
                for cid, v in self._qc_results.items():
                    val = v.get("evidence", {}).get(ev_key)
                    if val is False or val is True:
                        examples.append(f"{cid}: {ev_key}={val}")
                    # Keep only the surprising cases (False for validation_possible)
                if canonical == "anatomical_validation_possible":
                    examples = [e for e in examples if "=False" in e]
            elif canonical in tag_map:
                tag = tag_map[canonical]
                for cid, v in self._qc_results.items():
                    if any(t == tag for _, t, _ in v.get("flags", [])):
                        examples.append(f"{cid}: {tag}")

        elif canonical in (
            "keep_positive", "keep_negative",
            "review_positive", "review_or_exclude_positive",
        ):
            for cid, v in self._qc_results.items():
                if v.get("decision_hint") == canonical:
                    examples.append(
                        f"{cid}: score={v.get('score','?')}, "
                        f"risk={v.get('risk_level','?')}")

        elif canonical in ("b_o_overlap_mm3", "h_e_overlap_mm3", "t_a_overlap_mm3", "lesion_pancreas_ratio"):
            ev_key = {"b_o_overlap_mm3": "b_o_overlap_mm3",
                      "h_e_overlap_mm3": "h_e_overlap_mm3",
                      "t_a_overlap_mm3": "t_a_overlap_mm3",
                      "lesion_pancreas_ratio": None}[canonical]
            for cid, v in self._qc_results.items():
                ev = v.get("evidence", {})
                if canonical == "lesion_pancreas_ratio":
                    lv = ev.get("total_lesion_volume_mm3", 0) or 0
                    pv = ev.get("pancreas_volume_mm3", 0) or 0
                    if pv > 0 and lv / pv > 0.4:
                        examples.append(f"{cid}: ratio={lv/pv:.2f} (lesion={lv:,.0f}, pan={pv:,.0f})")
                else:
                    val = ev.get(ev_key)
                    if val and val > 0:
                        examples.append(f"{cid}: {ev_key}={val:,.0f} mm³")

        return examples[:20]  # limit to 20 so output stays readable

    def _not_found(self, term: str) -> str:
        return self.suggest_terms(term)


# ═══════════════════════════════════════════════════════════════════════════════
# QC AGENT CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class QCAgent:
    """
    Expert QC agent for medical image datasets.

    Public API:
        get_high_risk_cases()         -> list[str]
        get_medium_risk_cases()       -> list[str]
        save_reports(report_dir)      -> None
    """

    BANNER = dedent("""\
        +==============================================================+
        |   QC Assistant v2  --  type 'help' for commands             |
        +==============================================================+""")

    HELP = dedent("""\
        COMMANDS
        --------
        summary / overview         Dataset QC overview
        stats                      Volume & issue statistics
        show high risk             High-risk cases (score ≥ 61)
        show medium risk           Medium-risk cases
        show empty pancreas        Empty pancreas mask cases
        show lesion outside        Lesion overlap / mismatch cases
        show no anatomical val     Positive cases with no pancreas (unvalidatable)
        show positives             All positive samples (lesion present)
        show negatives             All negative samples (no lesion)
        show region error          Region annotation issues (sum error + partial coverage)
        show partial coverage      Partial anatomical coverage (truncated FOV)
        show small pancreas        Abnormally small pancreas cases
        show missing region        Missing sub-region mask cases
        show unreadable            Unreadable NIfTI cases
        show hypoattenuating       Hypoattenuating lesion cases
        show hyperattenuating      Hyperattenuating lesion cases
        show isoattenuating        Isoattenuating lesion cases
        show suspicious hu         Lesions with suspicious HU values
        attenuation summary        Dataset-wide lesion attenuation distribution

        detail PanTS_XXXXX         Full QC detail for a case
        exclude PanTS_XXXXX        Exclusion recommendation
        fix PanTS_XXXXX            Correction strategies
        compare PanTS_X PanTS_Y    Side-by-side comparison

        causes                     Root causes for all issue types
        training implications      Training impact analysis
        harmonization              Multi-site harmonization strategies
        annotation issues          Annotation inconsistency analysis
        domain shift               Domain shift risk assessment

        report                     Regenerate all QC reports

        help <term>                Explain a QC term (glossary)
        glossary                   List all available glossary terms
        quit / exit                Exit""")

    _CASE_RE = re.compile(r"PanTS_\d{3,}", re.IGNORECASE)

    @staticmethod
    def _canon(case_id: str) -> str:
        m = re.search(r"\d+", case_id)
        if not m:
            return case_id
        return f"PanTS_{m.group().zfill(8)}"

    def __init__(self, data: dict, report_dir: Path | None = None,
                 thr: dict | None = None, score_version: str = "v2"):
        self.data           = data
        self.report_dir     = report_dir or _HERE
        self.thr            = thr
        self.score_version  = score_version
        self.qc_results     = run_qc(data, thr, score_version=score_version)
        self._omitted_cases = _get_omitted_cases(data)

        self._by_tag: dict[str, list[str]] = {}
        for cid, v in self.qc_results.items():
            for _, t, _ in v["flags"]:
                self._by_tag.setdefault(t, [])
                if cid not in self._by_tag[t]:
                    self._by_tag[t].append(cid)
        for lst in self._by_tag.values():
            lst.sort()

        self.glossary = GlossaryManager()
        self.glossary.load_qc_results(self.qc_results)

    # ── Public API ────────────────────────────────────────────────────────────

    def get_high_risk_cases(self) -> list[str]:
        return sorted(k for k, v in self.qc_results.items()
                      if v["risk_level"] == "high")

    def get_medium_risk_cases(self) -> list[str]:
        return sorted(k for k, v in self.qc_results.items()
                      if v["risk_level"] == "medium")

    def save_reports(self, report_dir: Path | None = None,
                     report_metadata: dict | None = None,
                     debug: bool = False) -> None:
        d = report_dir or self.report_dir
        d.mkdir(parents=True, exist_ok=True)
        generate_txt_report(self.data, self.qc_results, d / "qc_report.txt")
        generate_json_report(self.qc_results, d / "qc_report.json",
                             data=self.data, report_metadata=report_metadata,
                             thr=self.thr)
        generate_csv_report(self.qc_results, d / "qc_report.csv", data=self.data)
        if debug:
            generate_debug_json(self.qc_results, d / "qc_report_debug.json",
                                report_metadata=report_metadata)

    # ── Intent classification ─────────────────────────────────────────────────

    def _extract_cases(self, text: str) -> list[str]:
        results = []
        for m in self._CASE_RE.findall(text):
            digits = re.search(r"\d+", m).group()
            results.append(f"PanTS_{digits.zfill(8)}")
        return results

    def _classify(self, text: str) -> tuple[str, list]:
        t     = text.lower().strip()
        cases = self._extract_cases(text)

        if cases:
            if any(w in t for w in ("exclude", "should i", "keep or", "remove", "drop")):
                return "exclude", cases
            if any(w in t for w in ("fix", "correct", "repair", "how to", "solution")):
                return "fix", cases
            if any(w in t for w in ("compare", "vs", "versus", "difference")):
                return "compare", cases
            return "detail", cases

        if any(w in t for w in ("help", "command", "what can", "?")):
            # "help <term>" → glossary lookup; bare "help" → command list
            prefix = next((w for w in ("help",) if t.startswith(w)), None)
            if prefix and len(t) > len(prefix) and t[len(prefix)] == " ":
                term = t[len(prefix):].strip()
                return "glossary_explain", [term]
            return "help", []
        if t in ("glossary", "glossary list", "list glossary", "list terms"):
            return "glossary_list", []
        if any(w in t for w in ("summary", "overview", "status")):
            return "summary", []
        if any(w in t for w in ("stat", "distribution", "count", "breakdown")):
            return "stats", []
        if "report" in t or "regenerate" in t or "regen" in t or "save" in t:
            return "regen_report", []

        if "high risk" in t or "high-risk" in t:
            return "show", ["high_risk"]
        if "medium risk" in t or "medium-risk" in t:
            return "show", ["medium_risk"]
        if "critical" in t:
            return "show", ["high_risk"]
        if "warning" in t and "show" in t:
            return "show", ["warning"]
        if "empty" in t and "pancreas" in t:
            return "show", ["pancreas_mask_empty"]
        if "lesion" in t and any(w in t for w in ("outside", "mismatch", "overlap", "partial")):
            return "show", ["lesion_outside"]
        if any(w in t for w in ("no anatomical", "unvalidat", "no anat")):
            return "show", ["no_anatomical_val"]
        if "positive" in t and "show" in t:
            return "show", ["positives"]
        if "negative" in t and "show" in t:
            return "show", ["negatives"]
        if "region" in t and any(w in t for w in ("error", "inconsisten", "sum")):
            return "show", ["region_error"]
        if any(w in t for w in ("partial coverage", "partial visibility", "truncat")):
            return "show", ["partial_coverage"]
        if "small" in t and "pancreas" in t:
            return "show", ["small_pancreas"]
        if "missing" in t and "region" in t:
            return "show", ["missing_region"]
        if any(w in t for w in ("unreadable", "skipped", "nifti header")):
            return "show", ["unreadable"]

        if "training" in t or ("model" in t and "impact" in t):
            return "training", []
        if "harmoniz" in t:
            return "harmonization", []
        if "annotation" in t:
            return "annotation", []
        if "domain shift" in t or ("domain" in t and "shift" in t):
            return "domain_shift", []
        if "cause" in t or ("why" in t and "issue" in t):
            return "causes_general", []
        if any(w in t for w in ("quit", "exit", "bye")):
            return "quit", []

        # HU / attenuation queries
        if "hypoattenuating" in t or ("hypo" in t and "attenuating" in t):
            return "show", ["hypoattenuating"]
        if "hyperattenuating" in t or ("hyper" in t and "attenuating" in t):
            return "show", ["hyperattenuating"]
        if "isoattenuating" in t or ("iso" in t and "attenuating" in t):
            return "show", ["isoattenuating"]
        if "suspicious hu" in t or ("suspicious" in t and "hu" in t):
            return "show", ["suspicious_hu"]
        if "attenuation" in t and any(w in t for w in ("summary", "distribution",
                                                        "overview", "stat", "breakdown")):
            return "attenuation_summary", []

        return "unknown", []

    # ── Dispatcher ────────────────────────────────────────────────────────────

    def respond(self, user_input: str) -> str | None:
        intent, params = self._classify(user_input)
        if intent == "quit":
            return None
        try:
            if intent == "help":          return self.HELP
            if intent == "glossary_explain":
                return self.glossary.explain(params[0] if params else "")
            if intent == "glossary_list":
                return self.glossary.list_terms()
            if intent == "summary":       return self._r_summary()
            if intent == "stats":         return self._r_stats()
            if intent == "regen_report":  return self._regen_report()
            if intent == "training":      return self._r_training()
            if intent == "harmonization": return self._r_harmonization()
            if intent == "annotation":    return self._r_annotation()
            if intent == "domain_shift":  return self._r_domain_shift()
            if intent == "causes_general":return self._r_causes_general()
            if intent == "show":
                return self._r_show(params[0] if params else "high_risk")
            if intent == "attenuation_summary":
                return self._r_attenuation_summary()
            if intent == "detail":
                return self._r_detail(params[0] if params else "")
            if intent == "exclude":
                return self._r_exclude(params[0] if params else "")
            if intent == "fix":
                return self._r_fix(params[0] if params else "")
            if intent == "compare":
                return self._r_compare(
                    params[0] if len(params) > 0 else "",
                    params[1] if len(params) > 1 else "")
        except Exception as exc:
            return f"[Internal error: {exc}]"

        return ("I didn't understand. Type 'help' for commands.\n"
                "  Examples: 'detail PanTS_00007002'  'show high risk'  "
                "'training implications'")

    # ── Responders ────────────────────────────────────────────────────────────

    def _regen_report(self) -> str:
        self.save_reports()
        return (f"QC reports written to {self.report_dir}:\n"
                "  qc_report.txt  qc_report.json  qc_report.csv")

    def _r_summary(self) -> str:
        total = len(self.data)
        high  = len(self.get_high_risk_cases())
        med   = len(self.get_medium_risk_cases())
        low   = sum(1 for v in self.qc_results.values() if v["risk_level"] == "low")
        ok    = total - len(self.qc_results)

        ep  = len(self._by_tag.get("pancreas_mask_empty", []))
        lo  = len(set(self._by_tag.get("lesion_outside_pancreas", []) +
                      self._by_tag.get("lesion_partial_outside", []) +
                      self._by_tag.get("lesion_overlap_critical_fail", [])))
        re_ = len(set(self._by_tag.get("region_sum_error_critical", []) +
                      self._by_tag.get("region_sum_error_warning", [])))
        sp  = len(set(self._by_tag.get("very_small_pancreas", []) +
                      self._by_tag.get("small_pancreas", [])))
        mr  = len(set(self._by_tag.get("head_region_empty", []) +
                      self._by_tag.get("body_region_empty", []) +
                      self._by_tag.get("tail_region_empty", [])))

        # Task-aware counts
        positives   = sum(1 for v in self.qc_results.values()
                          if v.get("sample_type") == "positive")
        negatives   = sum(1 for v in self.qc_results.values()
                          if v.get("sample_type") == "negative")
        no_anat_val = sum(1 for v in self.qc_results.values()
                          if not v.get("anatomical_validation_possible")
                          and v.get("sample_type") == "positive")
        keep_pos    = sum(1 for v in self.qc_results.values()
                          if v.get("decision_hint") == "keep_positive")
        review_pos  = sum(1 for v in self.qc_results.values()
                          if v.get("decision_hint") in ("review_positive",
                                                         "review_or_exclude_positive"))

        return (
            f"PANTSMINI QC OVERVIEW (v3 — task-aware)\n"
            f"  Training objective: {_training_objective(next(iter(self.qc_results.values()), {}).get('task_mode', 'pancreas_lesion'))}\n"
            f"{'─'*52}\n"
            f"  Total cases analysed      : {total}\n"
            f"  Omitted (CT unloadable)   : {len(self._omitted_cases)}\n"
            f"  Positive samples (lesion) : {positives}\n"
            f"  Negative samples (no les) : {negatives}\n"
            f"{'─'*52}\n"
            f"  TASK-AWARE DECISIONS\n"
            f"  keep_positive             : {keep_pos}\n"
            f"  review_positive           : {review_pos}\n"
            f"  No anatomical validation  : {no_anat_val}  (pancreas absent + lesion)\n"
            f"{'─'*52}\n"
            f"  RISK LEVELS\n"
            f"  High risk                 : {high}  ({high/total:.1%})\n"
            f"  Medium risk               : {med}  ({med/total:.1%})\n"
            f"  Low risk (flagged)        : {low}  ({low/total:.1%})\n"
            f"  No issues                 : {ok}  ({ok/total:.1%})\n"
            f"{'─'*52}\n"
            f"  BY ISSUE TYPE\n"
            f"  Empty pancreas mask       : {ep} cases\n"
            f"  Lesion overlap issues     : {lo} cases\n"
            f"  Region sum inconsistency  : {re_} cases\n"
            f"  Missing sub-region mask   : {mr} cases\n"
            f"  Abnormal pancreas volume  : {sp} cases\n"
            f"{'─'*52}\n"
            f"  Type 'show high risk'  'training implications'  'detail PanTS_XXXXX'"
        )

    def _r_stats(self) -> str:
        tag_counts = Counter(t for v in self.qc_results.values()
                              for _, t, _ in v["flags"])
        lines = ["Issue tag counts:", f"  {'Tag':<42} {'Cases':>5}", f"  {'─'*42} {'─'*5}"]
        for tag, cnt in tag_counts.most_common():
            lines.append(f"  {tag:<42} {cnt:>5}")

        scores = [v["score"] for v in self.qc_results.values() if v["score"] > 0]
        if scores:
            lines += [
                f"\nQC score distribution ({len(scores)} flagged):",
                f"  Min    : {min(scores)}",
                f"  Max    : {max(scores)}",
                f"  Mean   : {statistics.mean(scores):.1f}",
                f"  Median : {statistics.median(scores):.1f}",
            ]
        return "\n".join(lines)

    def _r_show(self, filter_type: str) -> str:
        label_map: dict[str, tuple[str, list[str]]] = {
            "warning"           : ("WARNING cases",
                                   [k for k, v in self.qc_results.items()
                                    if not any(s == "CRITICAL" for s, _, _ in v["flags"])
                                    and any(s == "WARNING" for s, _, _ in v["flags"])]),
            "high_risk"         : ("High-risk cases  (score ≥ 61)", self.get_high_risk_cases()),
            "medium_risk"       : ("Medium-risk cases", self.get_medium_risk_cases()),
            "pancreas_mask_empty": ("Empty pancreas mask",
                                    self._by_tag.get("pancreas_mask_empty", [])),
            "region_error"      : ("Region annotation issues",
                                   sorted(set(
                                       self._by_tag.get("region_sum_error_critical", []) +
                                       self._by_tag.get("region_sum_error_warning", []) +
                                       self._by_tag.get("incomplete_anatomical_coverage", [])))),
            "partial_coverage"  : ("Partial anatomical coverage (truncated FOV)",
                                   self._by_tag.get("incomplete_anatomical_coverage", [])),
            "small_pancreas"    : ("Abnormal pancreas volume",
                                   sorted(set(
                                       self._by_tag.get("very_small_pancreas", []) +
                                       self._by_tag.get("small_pancreas", [])))),
            "missing_region"    : ("Missing sub-region mask",
                                   sorted(set(
                                       self._by_tag.get("head_region_empty", []) +
                                       self._by_tag.get("body_region_empty", []) +
                                       self._by_tag.get("tail_region_empty", [])))),
            "lesion_outside"    : ("Lesion overlap issues",
                                   sorted(set(
                                       self._by_tag.get("lesion_outside_pancreas", []) +
                                       self._by_tag.get("lesion_partial_outside", []) +
                                       self._by_tag.get("lesion_overlap_critical_fail", [])))),
            "no_anatomical_val" : ("No anatomical validation (pancreas absent + lesion)",
                                   sorted(k for k, v in self.qc_results.items()
                                          if not v.get("anatomical_validation_possible")
                                          and v.get("sample_type") == "positive")),
            "positives"         : ("Positive samples (lesion present)",
                                   sorted(k for k, v in self.qc_results.items()
                                          if v.get("sample_type") == "positive")),
            "negatives"         : ("Negative samples (no lesion)",
                                   sorted(k for k, v in self.qc_results.items()
                                          if v.get("sample_type") == "negative")),
            "hypoattenuating"   : ("Hypoattenuating lesion cases",
                                   sorted(k for k, v in self.data.items()
                                          if (v.get("hu_statistics") or {}).get(
                                              "tumor_attenuation") == "Hypo")),
            "hyperattenuating"  : ("Hyperattenuating lesion cases",
                                   sorted(k for k, v in self.data.items()
                                          if (v.get("hu_statistics") or {}).get(
                                              "tumor_attenuation") == "Hyper")),
            "isoattenuating"    : ("Isoattenuating lesion cases",
                                   sorted(k for k, v in self.data.items()
                                          if (v.get("hu_statistics") or {}).get(
                                              "tumor_attenuation") == "Iso")),
            "suspicious_hu"     : ("Cases with suspicious HU values",
                                   sorted(k for k, v in self.qc_results.items()
                                          if v.get("evidence", {}).get(
                                              "suspicious_hu_distribution"))),
        }

        if filter_type == "unreadable":
            omitted = self._omitted_cases
            lines = [f"Omitted cases ({len(omitted)} cases)  [CT unloadable]:",
                     "─" * 60]
            for c in omitted:
                cs  = (self.data.get(c) or {}).get("case_status") or {}
                err = cs.get("error_type", "unknown")
                lines.append(f"  {c}  [{err}]")
            lines.append("\nFix: use nibabel to reset qform/sform, or re-export from DICOM.")
            return "\n".join(lines)

        if filter_type not in label_map:
            return f"Unknown filter '{filter_type}'. Type 'help' for options."

        title, cases = label_map[filter_type]
        cases = sorted(set(cases))
        if not cases:
            return f"No cases found for filter '{filter_type}'."

        lines = [f"{title}  ({len(cases)} cases)", "─" * 60]
        for c in cases:
            v    = self.qc_results.get(c, {})
            risk = v.get("risk_level", "?")
            sc   = v.get("score", "?")
            hint = v.get("decision_hint", "?")
            msgs = "; ".join(m for sev, _, m in v.get("flags", [])
                             if sev != "INFO")[:70]
            lines.append(f"  [score={sc} {risk} | {hint}] {c}  -- {msgs}")
        lines.append("\n  Type 'detail PanTS_XXXXX' for a full breakdown.")
        return "\n".join(lines)

    def _r_attenuation_summary(self) -> str:
        """Return dataset-wide lesion attenuation distribution summary."""
        def _att_cases(att: str) -> list[str]:
            return sorted(k for k, v in self.data.items()
                          if (v.get("hu_statistics") or {}).get("tumor_attenuation") == att)

        hypo    = _att_cases("Hypo")
        hyper   = _att_cases("Hyper")
        iso     = _att_cases("Iso")
        unknown = _att_cases("Unknown")
        no_stats = sorted(k for k, v in self.data.items()
                          if v.get("hu_statistics", {}).get("mean_hu_tumor") is None
                          and (v.get("hu_statistics") or {}).get("tumor_attenuation") is None)
        suspicious = sorted(k for k, v in self.qc_results.items()
                             if v.get("evidence", {}).get("suspicious_hu_distribution"))

        total_pos = sum(1 for v in self.qc_results.values()
                        if v.get("sample_type") == "positive")

        lines = [
            "LESION ATTENUATION DISTRIBUTION",
            "─" * 60,
            f"  (Based on {total_pos} positive cases with lesion present)",
            "",
            f"  Hypoattenuating  (Hypo)   : {len(hypo):>4}  "
            f"— consistent with PDAC, cystic lesions",
            f"  Isoattenuating   (Iso)    : {len(iso):>4}  "
            f"— plausible; may be challenging to detect",
            f"  Hyperattenuating (Hyper)  : {len(hyper):>4}  "
            f"— may indicate neuroendocrine tumor",
            f"  Unknown (no pancreas mask): {len(unknown):>4}  "
            f"— attenuation relative to pancreas unavailable",
            f"  No HU stats (no lesion)   : {len(no_stats):>4}",
            f"  Suspicious HU values      : {len(suspicious):>4}  "
            f"— outside [-200, 400] HU range",
            "",
            "Type 'show hypoattenuating' / 'show hyperattenuating' for case lists.",
        ]
        if hyper:
            lines += ["", f"Hyperattenuating cases ({len(hyper)}):"]
            for c in hyper[:15]:
                hu = (self.data.get(c) or {}).get("hu_statistics") or {}
                lines.append(f"  {c}  mean_tumor={hu.get('mean_hu_tumor')} HU  "
                              f"delta={hu.get('delta_hu_tumor_vs_pancreas')} HU")
            if len(hyper) > 15:
                lines.append(f"  … and {len(hyper)-15} more.")
        return "\n".join(lines)

    def _r_detail(self, case_id: str) -> str:
        case_id = self._canon(case_id)
        # Omitted cases are in self.data with null fields
        if case_id not in self.data:
            return f"Case '{case_id}' not found in the JSON. Check the ID."

        d     = self.data[case_id]
        v     = self.qc_results.get(case_id, {})
        img   = d.get("image") or {}
        meta  = d.get("metadata") or {}
        pan   = d.get("pancreas") or {}
        les   = d.get("lesions") or {}
        qc    = d.get("quality_control") or {}
        rc    = pan.get("region_consistency") or {}
        reg   = pan.get("regions") or {}
        flags = v.get("flags", [])

        # Task-aware fields (v3)
        sample_type  = v.get("sample_type", "unknown")
        hint         = v.get("decision_hint", "unknown")
        anat_val     = v.get("anatomical_validation_possible", False)
        task_reasons = v.get("task_aware_reasons", [])

        L = [f"{'─'*62}",
             f"  {case_id}",
             f"    score={v.get('score',0)} | risk={v.get('risk_level','?')} | "
             f"rec={v.get('recommendation','?')}",
             f"    sample_type={sample_type} | decision_hint={hint}",
             f"    anatomical_validation_possible={anat_val}",
             f"{'─'*62}",
             "  TASK-AWARE REASONING"]
        for r in task_reasons:
            L.append(f"    • {r}")
        if not task_reasons:
            L.append("    (none)")

        L += ["  IMAGE",
              f"    Shape (z,y,x)      : {img.get('shape_zyx')}",
              f"    Spacing (x,y,z) mm : {img.get('spacing_xyz_mm')}",
              f"    Voxel volume mm3   : {img.get('voxel_volume_mm3')}",
              "  METADATA"]
        for k, label in (("sex","Sex"), ("age","Age"), ("ct_phase","CT phase"),
                         ("manufacturer","Manufacturer"),
                         ("manufacturer_model","Scanner model"),
                         ("study_type","Study type"),
                         ("site_nationality","Nationality")):
            L.append(f"    {label:<20}: {meta.get(k, 'N/A')}")

        pvol = pan.get("volume_mm3") or 0
        rel_err = rc.get("relative_error_vs_pancreas")
        L += ["  PANCREAS",
              f"    Present            : {v.get('evidence',{}).get('pancreas_present', 'N/A')}",
              f"    Volume mm3         : {pvol:,.1f}",
              f"    Region sum error   : {f'{rel_err:.1%}' if rel_err is not None else 'N/A'}",
              "  REGIONS (vol mm3)"]
        for rn in ("head", "body", "tail"):
            r  = reg.get(rn) or {}
            rv = r.get("volume_mm3") or 0
            rr = r.get("rate_vs_pancreas")
            L.append(f"    {rn:<6}: {rv:>10,.0f} mm3   "
                     f"{f'{rr:.1%}' if rr is not None else 'N/A'}")

        n_les = les.get("n_lesions", 0)
        tv    = les.get("total_volume_mm3") or 0
        ov_r  = qc.get("total_overlap_rate_vs_lesion")
        L += ["  LESIONS",
              f"    Present            : {v.get('evidence',{}).get('lesion_present', 'N/A')}",
              f"    N lesions          : {n_les}",
              f"    Total volume mm3   : {tv:,.1f}",
              f"    Overlap w/ pancreas: {f'{ov_r:.1%}' if ov_r is not None else 'N/A'}"]

        L.append("  QC FLAGS")
        non_info = [(s, t, m) for s, t, m in flags if s != "INFO"]
        info_fl  = [(s, t, m) for s, t, m in flags if s == "INFO"]
        if non_info:
            for sev_f, tag, msg in non_info:
                L.append(f"    [{sev_f}] {msg}")
        if info_fl:
            for _, _, msg in info_fl:
                L.append(f"    [INFO] {msg}")
        if not flags:
            L.append("    None -- case appears clean")

        # HU statistics section
        hu = d.get("hu_statistics") or {}
        mean_hu_tumor    = hu.get("tumor_mean_hu")    or hu.get("mean_hu_tumor")
        mean_hu_pancreas = hu.get("pancreas_mean_hu") or hu.get("mean_hu_pancreas")
        delta_hu         = hu.get("delta_hu_tumor_vs_pancreas")
        tumor_atten      = hu.get("tumor_attenuation")
        if mean_hu_tumor is not None:
            if tumor_atten == "Hypo":
                interp = (f"Tumor is hypoattenuating relative to pancreas "
                          f"({delta_hu:+.0f} HU) — consistent with PDAC.")
            elif tumor_atten == "Hyper":
                interp = (f"Hyperattenuating lesion ({delta_hu:+.0f} HU) — "
                          f"may indicate neuroendocrine tumor or annotation error.")
            elif tumor_atten == "Iso":
                interp = f"Isoattenuating lesion ({delta_hu:+.0f} HU) — plausible."
            elif tumor_atten == "Unknown":
                interp = (f"Pancreas mask absent; relative attenuation cannot be "
                          f"computed (tumor HU={mean_hu_tumor:.1f}).")
            else:
                interp = "Attenuation classification unavailable."
            L += ["  HU STATISTICS",
                  f"    Mean HU tumor      : {mean_hu_tumor}",
                  f"    Mean HU pancreas   : {mean_hu_pancreas if mean_hu_pancreas is not None else 'N/A'}",
                  f"    Delta HU           : {delta_hu if delta_hu is not None else 'N/A'}",
                  f"    Tumor attenuation  : {tumor_atten}",
                  f"    Interpretation     : {interp}"]
        else:
            L.append("  HU STATISTICS  : Not available (lesion absent or CT not loaded)")

        L += [f"{'─'*62}",
              f"  Type 'exclude {case_id}' or 'fix {case_id}' for recommendations."]
        return "\n".join(L)

    def _r_exclude(self, case_id: str) -> str:
        case_id = self._canon(case_id)
        if case_id not in self.data:
            return f"Case '{case_id}' not found."

        v = self.qc_results.get(case_id)
        if not v or not v["flags"]:
            return (f"RECOMMENDATION for {case_id}: KEEP\n"
                    "  No QC issues detected. Suitable for all training splits.")

        rec   = v["recommendation"]
        score = v["score"]
        risk  = v["risk_level"]
        tags  = _tags(v["flags"])
        d     = self.data[case_id]
        qc    = d.get("quality_control") or {}
        pan   = d.get("pancreas") or {}
        rc    = pan.get("region_consistency") or {}

        verdict = {"exclude": "EXCLUDE", "review": "REVIEW BEFORE USE",
                   "keep": "KEEP (with caveats)"}.get(rec, "REVIEW")
        L = [f"RECOMMENDATION for {case_id}: {verdict}",
             f"  QC score: {score}  |  risk: {risk}", ""]

        if "pancreas_mask_empty" in tags:
            L += ["-> Pancreas mask is empty.",
                  "   Cannot be used for any pancreas segmentation task."]
        if "lesion_outside_pancreas" in tags or "lesion_partial_outside" in tags:
            ov = qc.get("total_overlap_rate_vs_lesion", 0) or 0
            sev_str = "CRITICAL" if ov < 0.10 else "WARNING"
            L += [f"-> Lesion-pancreas overlap: {ov:.1%}  [{sev_str}]",
                  "   Exclude from lesion segmentation training/validation."]
        if "region_sum_error_critical" in tags or "region_sum_error_warning" in tags:
            err = rc.get("relative_error_vs_pancreas", 0) or 0
            L += [f"-> Region sum error: {err:.1%}.",
                  "   Exclude from region-specific training if error > 30 %."]
        if "very_small_pancreas" in tags:
            pvol = pan.get("volume_mm3", 0)
            L += [f"-> Pancreas volume: {pvol:,.0f} mm3 -- critically small.",
                  "   Exclude from volume estimation tasks."]
        elif "small_pancreas" in tags:
            pvol = pan.get("volume_mm3", 0)
            L += [f"-> Pancreas volume: {pvol:,.0f} mm3 -- below expected range.",
                  "   Flag for clinical review."]

        L += ["", f"  Type 'fix {case_id}' for correction strategies."]
        return "\n".join(L)

    def _r_fix(self, case_id: str) -> str:
        case_id = self._canon(case_id)
        if case_id not in self.data:
            return f"Case '{case_id}' not found."
        v = self.qc_results.get(case_id)
        if not v or not v["flags"]:
            return f"No issues found for {case_id}. No corrections needed."

        tags = _tags(v["flags"])
        L = [f"CORRECTION STRATEGIES for {case_id}:"]
        key_map = [
            ("pancreas_mask_empty",           "pancreas_mask_empty"),
            ("lesion_no_anatomical_validation","lesion_no_anatomical_validation"),
            ("lesion_outside_pancreas",        "lesion_outside_pancreas"),
            ("lesion_overlap_critical_fail",   "lesion_outside_pancreas"),
            ("lesion_partial_outside",         "lesion_outside_pancreas"),
            ("region_sum_error_critical",      "region_sum_error"),
            ("region_sum_error_warning",       "region_sum_error"),
            ("very_small_pancreas",            "small_pancreas"),
            ("small_pancreas",                 "small_pancreas"),
            ("invalid_spacing",                "invalid_spacing"),
        ]
        seen: set[str] = set()
        for tag, key in key_map:
            if tag in tags and key not in seen and key in CORRECTIONS:
                L += ["", f"-- [{tag}] --", CORRECTIONS[key]]
                seen.add(key)
        return "\n".join(L)

    def _r_compare(self, case1: str, case2: str) -> str:
        case1, case2 = self._canon(case1), self._canon(case2)
        missing = [c for c in (case1, case2) if c not in self.data]
        if missing:
            return f"Case(s) not found: {', '.join(missing)}"

        def _brief(cid: str) -> dict:
            d  = self.data[cid]
            v  = self.qc_results.get(cid, {})
            fl = v.get("flags", [])
            return dict(
                risk    = v.get("risk_level", "low"),
                score   = v.get("score", 0),
                pvol    = f"{(d.get('pancreas') or {}).get('volume_mm3',0):,.0f} mm3",
                n_les   = (d.get("lesions") or {}).get("n_lesions", 0),
                flags   = [f"[{s}] {m}" for s, _, m in fl if s != "INFO"] or ["None"],
                spacing = str((d.get("image") or {}).get("spacing_xyz_mm", "N/A")),
                shape   = str((d.get("image") or {}).get("shape_zyx", "N/A")),
            )

        a, b = _brief(case1), _brief(case2)
        W = 22

        def row(lbl: str, va: Any, vb: Any) -> str:
            return f"  {lbl:<28} {str(va):<{W}} {str(vb):<{W}}"

        L = [f"{'─'*72}",
             f"  {'ATTRIBUTE':<28} {case1:<{W}} {case2:<{W}}",
             f"{'─'*72}",
             row("Risk level",      a["risk"],    b["risk"]),
             row("QC score",        a["score"],   b["score"]),
             row("Pancreas volume", a["pvol"],    b["pvol"]),
             row("N lesions",       a["n_les"],   b["n_les"]),
             row("Spacing",         a["spacing"], b["spacing"]),
             row("Shape",           a["shape"],   b["shape"]),
             f"{'─'*72}",
             f"  QC flags  {case1}:"]
        for f_ in a["flags"]:
            L.append(f"    * {f_}")
        L.append(f"  QC flags  {case2}:")
        for f_ in b["flags"]:
            L.append(f"    * {f_}")
        L.append(f"{'─'*72}")
        return "\n".join(L)

    def _r_training(self) -> str:
        total = len(self.data)
        high  = len(self.get_high_risk_cases())
        med   = len(self.get_medium_risk_cases())
        safe  = total - high - med
        parts = [
            f"TRAINING IMPLICATIONS\n{'─'*65}",
            f"  {total} total  |  {high} high-risk  |  {med} medium-risk  |  {safe} clean\n",
            "Recommended strategy:",
            "  1. Exclude all high-risk cases from clean training set",
            "  2. Use clean set for validation and test splits",
            "  3. Medium-risk: keep in training with reduced weight / loss masking",
            "  4. Curriculum: start with clean, add difficult cases later\n",
            "IMPACT BY ISSUE TYPE",
        ]
        for key, text in TRAINING_IMPACTS.items():
            parts += [f"\n  [{key}]", f"  {text}"]
        return "\n".join(parts)

    def _r_harmonization(self) -> str:
        return dedent("""\
            HARMONIZATION STRATEGIES
            ────────────────────────────────────────────────────────────
            1. SPACING NORMALIZATION
               Resample to common isotropic spacing (e.g. 1x1x1 mm).
               nnU-Net handles this automatically.

            2. INTENSITY NORMALIZATION
               Clip CT to [-200, 300] HU, then z-score per case.
               Optional: histogram matching across sites.

            3. SUB-REGION MASK CONSISTENCY
               head_fixed = head AND pancreas  (repeat for body, tail)

            4. LESION / PANCREAS ALIGNMENT
               Apply rigid registration for cases with overlap < 50%.
               Tools: SimpleElastix, ANTs, nibabel affine correction.

            5. SCANNER STRATIFICATION
               Site-stratified cross-validation to avoid data leakage.
               Consider domain adversarial training.

            6. CT PHASE SEGREGATION
               Never mix arterial/venous/non-contrast in the same fold.
               Train phase-specific models or apply phase conditioning.""")

    def _r_annotation(self) -> str:
        ep  = len(self._by_tag.get("pancreas_mask_empty", []))
        lo  = len(set(self._by_tag.get("lesion_outside_pancreas", []) +
                      self._by_tag.get("lesion_partial_outside", [])))
        re_ = len(set(self._by_tag.get("region_sum_error_critical", []) +
                      self._by_tag.get("region_sum_error_warning", [])))
        return dedent(f"""\
            ANNOTATION INCONSISTENCY ANALYSIS
            ────────────────────────────────────────────────────────────
            1. EMPTY PANCREAS MASKS  ({ep} cases)
               Probability of annotation error: HIGH
               Action: cross-check against original DICOM source.

            2. LESION/PANCREAS SPATIAL MISMATCH  ({lo} cases)
               Probability of error: HIGH for overlap < 10%
               Action: visually inspect all cases with overlap < 10%.

            3. SUB-REGION VOLUME INCONSISTENCY  ({re_} cases)
               Probability of error: MEDIUM
               Action: re-run sub-region splitting on the final mask.

            4. INTER-RATER VARIABILITY
               Multi-site dataset -- head/body/tail boundary definitions
               are not standardized across sites.

            5. TEMPORAL INCONSISTENCY
               Data spans multiple years; protocol changes may introduce
               annotation style drift within a single site.""")

    def _r_domain_shift(self) -> str:
        phases  = Counter((d.get("metadata") or {}).get("ct_phase") or "unknown"
                          for d in self.data.values())
        nats    = Counter((d.get("metadata") or {}).get("site_nationality") or "unknown"
                          for d in self.data.values())
        makers  = Counter((d.get("metadata") or {}).get("manufacturer") or "unknown"
                          for d in self.data.values())

        def _dist(counter: Counter, label: str) -> str:
            lines = [f"\n  {label}:"]
            for k, v in counter.most_common():
                lines.append(f"    {str(k):<32}: {v:>4}  ({v/len(self.data):.1%})")
            return "\n".join(lines)

        return ("DOMAIN SHIFT RISK ASSESSMENT\n" + "─" * 65 +
                _dist(phases, "CT phase distribution") +
                _dist(nats,   "Site nationality") +
                _dist(makers, "Scanner manufacturer") +
                dedent("""

            Key risks:
            CT PHASE   -- train phase-specific models or use phase conditioning.
            MULTI-SITE -- site-stratified CV; intensity normalization per case.
            SCANNER    -- include manufacturer as covariate in performance analysis.
            TEMPORAL   -- verify year distribution is balanced across folds."""))

    def _r_causes_general(self) -> str:
        L = ["Causes of common QC issues:", "─" * 60]
        for tag, text in CAUSES.items():
            L += ["", f"[{tag}]", text]
        L.append("\nType 'fix PanTS_XXXXX' for per-case correction strategies.")
        return "\n".join(L)

    # ── Interactive loop ──────────────────────────────────────────────────────

    def run(self) -> None:
        print(self.BANNER)
        high = len(self.get_high_risk_cases())
        med  = len(self.get_medium_risk_cases())
        print(f"\n  Loaded {len(self.data)} cases | "
              f"{high} high-risk | {med} medium-risk | "
              f"reports saved to {self.report_dir.name}/")
        print("  Type 'help' to see all commands.\n")

        while True:
            try:
                raw = input("QC> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAgent: Goodbye.")
                break
            if not raw:
                continue
            response = self.respond(raw)
            if response is None:
                print("Agent: Goodbye.")
                break
            print(f"\n{response}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="QC Agent v3 -- QC analysis + interactive assistant.")
    parser.add_argument("--json", default=None,
                        help="Path to <dataset>_summary.json (default: derived from "
                             "--task-mode / paths.yaml). Overrides auto-discovery.")
    parser.add_argument("--report-dir", default=None,
                        help="Output directory for qc_report.{txt,json,csv} "
                             "(default: <output_dir>/<task_mode>/qc/).")
    parser.add_argument("--paths-yaml", default=None,
                        help="Path to paths.yaml (default: configs/paths.yaml)")
    parser.add_argument("--thresholds-yaml", default=None,
                        help=("Path to deterministic or calibrated thresholds YAML "
                              "(explicit override for --threshold-method)."))
    parser.add_argument("--threshold-method", default=None,
                        choices=["deterministic", "calibrated"],
                        help=("Threshold set to use when --thresholds-yaml is not provided "
                              "(default: DEFAULT_THRESHOLD_METHOD from paths.yaml)."))
    parser.add_argument("--task-mode", default=None, dest="task_mode",
                        choices=sorted(VALID_TASK_MODES),
                        help="Override TASK_MODE (highest priority; overrides "
                             "thresholds.yaml). Choices: "
                             + " | ".join(sorted(VALID_TASK_MODES)) + ".")
    parser.add_argument("--output-dir", default=None, dest="output_dir",
                        help="Dataset-level output root (default: outputs/<dataset_name>/ "
                             "from paths.yaml). Task-specific sub-folders are used "
                             "automatically: <output_dir>/<task_mode>/summary/ and "
                             "<output_dir>/<task_mode>/qc/.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing QC report files. "
                             "Aborts with an error if reports already exist and "
                             "this flag is absent.")
    parser.add_argument("--no-interactive", action="store_true",
                        help="Generate reports and exit without interactive loop")
    parser.add_argument("--score-version", default="v2",
                        choices=["v2", "hybrid"],
                        dest="score_version",
                        help=(
                            "Scoring formula to use.  "
                            "v2 (default): additive sum of penalty components.  "
                            "hybrid: additive geometry gate + overlap amplified by "
                            "secondary issues (geometry + overlap*(1 + alpha*secondary/100)).  "
                            "v2 is preserved and unchanged."
                        ))
    parser.add_argument("--debug", action="store_true",
                        help=("Write qc_report_debug.json alongside the standard reports. "
                              "Includes raw score_components, flags, evidence, and "
                              "scoring_details for each case."))
    args = parser.parse_args()

    # Load centralized path config
    paths = resolve_project_paths(Path(args.paths_yaml) if args.paths_yaml else None)
    ensure_output_directories(paths)

    # Load thresholds (spatial QC gates + hybrid scoring params)
    thr: dict | None = None
    _threshold_method = (
        args.threshold_method
        or getattr(paths, "default_threshold_method", "deterministic")
        or "deterministic"
    ).strip().lower()
    if _threshold_method not in {"deterministic", "calibrated"}:
        raise SystemExit(
            f"[ERROR] Unsupported threshold method: {_threshold_method!r}\n"
            "        Expected 'deterministic' or 'calibrated'."
        )
    if args.thresholds_yaml:
        _thr_path = Path(args.thresholds_yaml)
    elif _threshold_method == "calibrated":
        raise SystemExit(
            "[ERROR] --threshold-method calibrated requires --thresholds-yaml "
            "with an explicit run-scoped calibrated threshold file."
        )
    else:
        _thr_path = paths.thresholds_config

    if _threshold_method == "calibrated" and not _thr_path.exists():
        raise SystemExit(
            f"[ERROR] Calibrated thresholds not found: {_thr_path}\n"
            "        Run scripts/calibrate_thresholds.py with an explicit --output first, "
            "or use --threshold-method deterministic."
        )
    if _thr_path.exists():
        if _HAS_YAML:
            with open(_thr_path, encoding="utf-8") as _f:
                thr = _yaml.safe_load(_f) or {}
        else:  # fall back to JSON if PyYAML is not installed
            try:
                thr = json.loads(_thr_path.read_text(encoding="utf-8"))
            except Exception:
                thr = None

    # Resolve TASK_MODE: CLI arg > thresholds.yaml > default
    _thr_mode = str((thr or {}).get("TASK_MODE", "pancreas_lesion_subregions")).strip().lower()
    if _thr_mode == "auto":
        _thr_mode = "pancreas_lesion_subregions"
    _task_mode = args.task_mode if args.task_mode else _thr_mode
    if _task_mode not in VALID_TASK_MODES:
        raise ValueError(
            f"Unsupported TASK_MODE: '{_task_mode}'. "
            f"Valid values: {sorted(VALID_TASK_MODES)}"
        )

    # Build task-specific output dirs
    _base_output_dir = (
        Path(args.output_dir) if args.output_dir
        else paths.summary_dir.parent   # outputs/<dataset_name>/
    )
    _task_dirs = build_output_dirs(_base_output_dir, _task_mode)

    # Resolve summary JSON path
    if args.json:
        json_path = Path(args.json)
    else:
        json_path = _task_dirs["summary_dir"] / f"{paths.dataset_name}_summary.json"
        if not json_path.exists():
            # Backward-compat fallback to legacy location
            _legacy = paths.summary_json
            if _legacy.exists():
                print(f"[WARNING] Task-specific summary not found: {json_path}")
                print(f"          Falling back to legacy path: {_legacy}")
                json_path = _legacy
            # If neither exists, the error below will catch it

    report_dir = Path(args.report_dir) if args.report_dir else _task_dirs["qc_dir"]

    # Set up file logging
    lg = setup_file_logging(
        paths.logs_dir / "qc_agent.log",
        logger_name="qc_agent",
    )
    lg.info("Starting QC Agent")
    lg.info("TASK_MODE    : %s", _task_mode)
    lg.info("Summary JSON : %s", json_path)
    lg.info("Report dir   : %s", report_dir)
    lg.info("Thresholds   : %s", _thr_path)

    if not json_path.exists():
        print(f"[ERROR] JSON not found: {json_path}")
        print("        Run scripts/summarize_dataset.py first.")
        lg.error("JSON not found: %s", json_path)
        raise SystemExit(1)

    # Overwrite guard
    _report_files = [
        report_dir / "qc_report.json",
        report_dir / "qc_report.txt",
        report_dir / "qc_report.csv",
    ]
    if not args.overwrite and any(p.exists() for p in _report_files):
        _existing = [str(p) for p in _report_files if p.exists()]
        raise SystemExit(
            "[ERROR] QC report files already exist:\n  " + "\n  ".join(_existing) +
            "\n        Use --overwrite to replace them."
        )

    print(f"Loading {json_path} ...")
    data, _summary_meta, _summary_artifact_meta = load_summary_cases(json_path)
    print(f"  {len(data)} cases loaded.")
    lg.info("%d cases loaded", len(data))

    _report_metadata = {
        "dataset_name":   paths.dataset_name,
        "task_mode":      _task_mode,
        "summary_source": str(json_path),
        "output_dir":     str(report_dir),
        "created_at":     datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "schema_version": "4",
        "training_objective": _training_objective(_task_mode),
        "primary_target": _primary_target(_task_mode),
    }

    # Inject the CLI-resolved task mode so run_qc() always sees the correct value,
    # even when thresholds.yaml has a different TASK_MODE or "auto".
    thr = dict(thr) if thr else {}
    thr["TASK_MODE"] = _task_mode
    _thr_meta = _threshold_metadata(thr)
    _report_metadata.update({
        "threshold_method": _thr_meta.get("method", "deterministic"),
        "threshold_source": str(_thr_path),
        "threshold_metadata": _thr_meta,
        "effective_thresholds": _effective_thresholds(thr),
    })

    agent = QCAgent(data, report_dir=report_dir, thr=thr, score_version=args.score_version)
    print(f"Saving QC reports to {report_dir} ...")
    agent.save_reports(report_metadata=_report_metadata, debug=args.debug)
    suffix = "  qc_report.txt  qc_report.json  qc_report.csv"
    if args.debug:
        suffix += "  qc_report_debug.json"
    print(f"{suffix}  written.\n")
    lg.info("QC reports written to %s", report_dir)

    if not args.no_interactive:
        agent.run()


if __name__ == "__main__":
    main()
