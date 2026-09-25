#!/usr/bin/env python3
"""Generate current poster result figures from validated public artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import ARTIFACTS, ensure_output_dir, load_expected  # noqa: E402

ACTION_ORDER = ["keep", "warning", "review", "reject"]
ACTION_LABELS = ["Keep", "Warning", "Review", "Reject"]
PROFILE_LABELS = {"tau_P": r"$\tau_P$", "tau_L": r"$\tau_L$", "tau_S": r"$\tau_S$"}
COLORS = {"navy": "#17324D", "blue": "#176B87", "light_blue": "#D9ECF2", "teal": "#2A9D8F", "gold": "#E9C46A", "red": "#D45D5D", "gray": "#64748B", "ink": "#1F2937"}


def save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{name}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def load_matrices(transitions: Path) -> dict[tuple[str, str], np.ndarray]:
    mats = defaultdict(lambda: np.zeros((4, 4), dtype=int))
    with transitions.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["source_action"] in ACTION_ORDER and row["target_action"] in ACTION_ORDER:
                mats[(row["source_profile"], row["target_profile"])][ACTION_ORDER.index(row["source_action"]), ACTION_ORDER.index(row["target_action"])] += 1
    expected = load_expected()["decision_stability"]["pairwise_changed"]
    for key, changed in expected.items():
        source, _, target = key.partition("_to_")
        mat = mats[(source, target)]
        assert int(mat.sum()) == 1000
        assert int(mat.sum() - np.trace(mat)) == changed
    return dict(mats)


def action_profiles(summary_path: Path, transitions: Path, output_dir: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    mats = load_matrices(transitions)
    counts = {"tau_P": mats[("tau_P", "tau_L")].sum(axis=1), "tau_L": mats[("tau_P", "tau_L")].sum(axis=0), "tau_S": mats[("tau_P", "tau_S")].sum(axis=0)}
    expected_counts = load_expected()["decision_stability"]["action_counts"]
    for profile, vals in counts.items():
        assert dict(zip(ACTION_ORDER, [int(v) for v in vals])) == expected_counts[profile]

    fig = plt.figure(figsize=(18, 9))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.15, 1, 1, 1], height_ratios=[0.95, 1.05], wspace=0.56, hspace=0.43)
    ax0 = fig.add_subplot(gs[:, 0]); ax0.axis("off")
    changed = summary["all_three_profiles"]["changed_action_at_least_one_profile"]; stable = summary["all_three_profiles"]["same_action"]
    ax0.text(0.5, 0.95, "Task-aware action stability", ha="center", va="top", fontsize=18, weight="bold", color=COLORS["navy"])
    ax0.text(0.5, 0.80, f"{changed}/1000", ha="center", va="center", fontsize=42, weight="bold", color=COLORS["blue"])
    ax0.text(0.5, 0.70, "cases changed action\nunder ≥1 profile", ha="center", va="center", fontsize=16, weight="bold", color=COLORS["ink"])
    ax0.text(0.5, 0.60, "49.3% is the global any-profile statistic", ha="center", va="center", fontsize=11, color=COLORS["gray"])
    ax0.barh([0], [stable], color=COLORS["light_blue"], height=0.25, left=0); ax0.barh([0], [changed], color=COLORS["blue"], height=0.25, left=stable)
    ax0.set_xlim(0, 1000); ax0.set_ylim(-0.5, 1.2)
    ax0.text(stable / 2, 0, "507 stable", ha="center", va="center", fontsize=12, color=COLORS["navy"], weight="bold")
    ax0.text(stable + changed / 2, 0, "493 changing", ha="center", va="center", fontsize=12, color="white", weight="bold")
    ax0.text(0.02, 0.20, "Pairwise percentages in the matrices answer a\ndifferent question: how many cases change\nin that specific profile comparison.", transform=ax0.transAxes, fontsize=10.5, color=COLORS["ink"], va="top")

    ax_bar = fig.add_subplot(gs[0, 1:]); x = np.arange(3); bottoms = np.zeros(3); vals = np.array([counts[p] for p in ["tau_P", "tau_L", "tau_S"]])
    for i, (label, col) in enumerate(zip(ACTION_LABELS, [COLORS["teal"], COLORS["gold"], COLORS["blue"], COLORS["red"]])):
        ax_bar.bar(x, vals[:, i], bottom=bottoms, color=col, width=0.62, label=label)
        for j, v in enumerate(vals[:, i]):
            ax_bar.text(j, bottoms[j] + v / 2, str(int(v)), ha="center", va="center", fontsize=11, color="white" if label in {"Keep", "Review", "Reject"} else COLORS["navy"], weight="bold")
        bottoms += vals[:, i]
    ax_bar.set_xticks(x, [r"$\tau_P$ visible pancreas", r"$\tau_L$ pancreas lesion", r"$\tau_S$ lesion + subregions"]); ax_bar.set_ylabel("Cases (n=1000/profile)")
    ax_bar.set_title("Final-action distribution changes with task profile", pad=22); ax_bar.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15)); ax_bar.spines[["top", "right"]].set_visible(False); ax_bar.set_ylim(0, 1120)

    for k, pair in enumerate([("tau_P", "tau_L"), ("tau_P", "tau_S"), ("tau_L", "tau_S")]):
        ax = fig.add_subplot(gs[1, k + 1]); mat = mats[pair]; changed_n = int(mat.sum() - np.trace(mat)); rate = 100 * changed_n / int(mat.sum())
        ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(m.max() for m in mats.values()))
        ax.set_title(f"{PROFILE_LABELS[pair[0]]} → {PROFILE_LABELS[pair[1]]}\n{changed_n}/1000 changed ({rate:.1f}%)", fontsize=14)
        ax.set_xticks(np.arange(4), ACTION_LABELS, rotation=35, ha="right"); ax.set_yticks(np.arange(4), ACTION_LABELS); ax.set_xlabel("Target action")
        for i in range(4):
            for j in range(4):
                val = int(mat[i, j]); ax.text(j, i, str(val), ha="center", va="center", fontsize=10, color="white" if val > 230 else COLORS["ink"], weight="bold" if i != j and val else "normal")
        ax.tick_params(length=0)
    fig.suptitle("Same 1,000 cases, different task requirements, different deterministic curation actions", fontsize=20, weight="bold", color=COLORS["navy"], y=0.99)
    fig.text(0.53, 0.01, "Counts are configured action transitions, not clinical-correctness labels. Pairwise matrices exclude no cases; each denominator is 1,000.", ha="center", fontsize=11, color=COLORS["gray"])
    save(fig, output_dir, "action_profiles")


def evidence_domains(domains_csv: Path, output_dir: Path) -> None:
    rows = list(csv.DictReader(domains_csv.open(newline="", encoding="utf-8")))
    priority = ["region_consistency", "fov_integrity", "attenuation_integrity", "lesion_localization", "geometry_integrity", "pancreas_context", "metadata_completeness", "lesion_burden"]
    by_name = {r["evidence_domain"]: r for r in rows}; names = [p for p in priority if p in by_name]
    stable = np.array([float(by_name[n]["stable_percent"]) for n in names]); changing = np.array([float(by_name[n]["profile_changing_percent"]) for n in names])
    labels = ["Region\nconsistency", "FOV\nintegrity", "Attenuation\nintegrity", "Lesion\nlocalization", "Geometry\nintegrity", "Pancreas\ncontext", "Metadata\ncompleteness", "Lesion\nburden"]
    y = np.arange(len(names))[::-1]; fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(y - 0.18, stable[::-1], height=0.34, color=COLORS["light_blue"], label="Profile-stable cases (n=507)"); ax.barh(y + 0.18, changing[::-1], height=0.34, color=COLORS["blue"], label="Profile-changing cases (n=493)")
    ax.set_yticks(y, labels[::-1]); ax.set_xlim(0, 78); ax.set_xlabel("Cases containing evidence domain (%)"); ax.set_title("Evidence-domain prevalence in stable vs profile-changing cases"); ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", color="#E5E7EB", linewidth=0.8); ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    for yi, sv, cv in zip(y, stable[::-1], changing[::-1]):
        ax.text(sv + 1.0, yi - 0.18, f"{sv:.1f}%", va="center", fontsize=10, color=COLORS["navy"]); ax.text(cv + 1.0, yi + 0.18, f"{cv:.1f}%", va="center", fontsize=10, color=COLORS["blue"], weight="bold")
    fig.text(0.5, 0.01, "Multi-label descriptive frequencies aggregated over profile records; prevalence differences are not causal effects.", ha="center", fontsize=11, color=COLORS["gray"])
    save(fig, output_dir, "evidence_domains")


def case_pathways(output_dir: Path) -> None:
    from representative_cases.export_representative_cases import CASES
    fig, ax = plt.subplots(figsize=(17, 7.2)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.97, "Representative evidence-to-policy traces", ha="center", va="top", fontsize=22, weight="bold", color=COLORS["navy"])
    ax.text(0.5, 0.90, "Validated artifact values show how task relevance changes routing; actions are not clinical-correctness labels.", ha="center", va="top", fontsize=12, color=COLORS["gray"])
    colors = [COLORS["blue"], COLORS["teal"], COLORS["red"]]
    for i, (case_id, data) in enumerate(CASES.items()):
        x = 0.035 + i * 0.322; edge = colors[i]; rect = FancyBboxPatch((x, 0.18), 0.285, 0.64, boxstyle="round,pad=0.02,rounding_size=0.025", fc="white", ec=edge, lw=2.2); ax.add_patch(rect)
        ax.text(x + 0.012, 0.72, case_id, ha="left", va="top", fontsize=17, weight="bold", color=COLORS["navy"]); ax.text(x + 0.012, 0.63, data["transition"], ha="left", va="top", fontsize=14, weight="bold", color=edge)
        ax.text(x + 0.012, 0.49, "Task-relevant evidence", ha="left", va="top", fontsize=11, weight="bold", color=COLORS["gray"]); ax.text(x + 0.012, 0.43, data["task_relevant_evidence"], ha="left", va="top", fontsize=11, color=COLORS["ink"], wrap=True)
        ax.text(x + 0.012, 0.30, "Routing consequence", ha="left", va="top", fontsize=11, weight="bold", color=COLORS["gray"]); ax.text(x + 0.012, 0.24, data["routing_policy_consequence"], ha="left", va="top", fontsize=11, color=COLORS["ink"], wrap=True)
    fig.text(0.5, 0.055, "Source: revised supplementary evidence-to-policy traces. PanTS_00007013 is excluded because task-contract interpretation remains insufficiently documented.", ha="center", fontsize=10.5, color=COLORS["gray"])
    save(fig, output_dir, "case_pathways")


def workflow(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 5.4)); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.5, 0.95, "AgentQC authority path: evidence is interpreted through the downstream task", ha="center", va="top", fontsize=20, weight="bold", color=COLORS["navy"])
    steps = [("QC evidence", "geometry • FOV\ntarget state localization\nattenuation • metadata"), ("Task requirements", "$\\tau_P$ pancreas\n$\\tau_L$ lesion • $\\tau_S$ subregions"), ("Routing / policy", "deterministic escalation\nexplicit final-action rules"), ("Final curation action", "Keep • Warning\nReview • Reject")]
    for x, (title, sub) in zip([0.04, 0.29, 0.54, 0.79], steps):
        box = FancyBboxPatch((x, 0.40), 0.17, 0.34, boxstyle="round,pad=0.02,rounding_size=0.025", fc=COLORS["light_blue"], ec=COLORS["blue"], lw=2.5); ax.add_patch(box)
        ax.text(x + 0.085, 0.63, title, ha="center", va="center", fontsize=15.5, weight="bold", color=COLORS["navy"]); ax.text(x + 0.085, 0.49, sub, ha="center", va="center", fontsize=9.8, color=COLORS["ink"])
    for x in [0.22, 0.47, 0.72]: ax.annotate("", xy=(x + 0.05, 0.57), xytext=(x, 0.57), arrowprops=dict(arrowstyle="->", color=COLORS["blue"], lw=3))
    audit = FancyBboxPatch((0.27, 0.09), 0.46, 0.17, boxstyle="round,pad=0.02,rounding_size=0.025", fc="#FFF7E0", ec=COLORS["gold"], lw=2.0, linestyle="--"); ax.add_patch(audit)
    ax.text(0.50, 0.19, "Nonbinding explanation / audit", ha="center", va="center", fontsize=14, weight="bold", color=COLORS["navy"]); ax.text(0.50, 0.12, "ReasoningAgent and MedicalCriticAgent inspect artifacts;\nthey do not control routing or final actions.", ha="center", va="center", fontsize=10.5, color=COLORS["ink"])
    ax.annotate("", xy=(0.50, 0.39), xytext=(0.50, 0.27), arrowprops=dict(arrowstyle="<->", color=COLORS["gold"], lw=2.2, linestyle="--"))
    save(fig, output_dir, "workflow")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[2] / "outputs" / "poster_figures")
    args = parser.parse_args(); out = ensure_output_dir(args.output_dir)
    action_profiles(ARTIFACTS / "decision_stability" / "decision_stability_summary.json", ARTIFACTS / "decision_stability" / "decision_stability_case_transitions.csv", out)
    evidence_domains(ARTIFACTS / "evidence_domains" / "changed_vs_stable_evidence_domains.csv", out)
    case_pathways(out); workflow(out)
    print(f"Generated poster figures in {out}")


if __name__ == "__main__":
    main()
