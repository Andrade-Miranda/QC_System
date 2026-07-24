#!/usr/bin/env bash
# =============================================================================
# qc_commands.sh — quick-reference launcher for the QC pipeline
#
# Usage:
#   bash scripts/qc_commands.sh <command> [extra-args ...]
#
# Commands:
#   validate          Validate configs/thresholds.yaml + configs/paths.yaml
#   summarize         Summarize dataset (serial, uses thresholds.yaml TASK_MODE)
#   summarize-fast    Summarize with 8 parallel workers
#   qc                Run artifact QC workflow
#   qc-fast           Artifact QC workflow with 8 workers and overwrite
#   qc-all            Re-run QC for all three task modes (with overwrite)
#   evaluate          Rebuild eval_report.json for an existing run
#   golden-build      Build a blinded golden review package
#   golden-import     Import a completed review package as golden labels
#   admin             Open interactive review in admin mode
#   reviewer          Open interactive review in reviewer mode
#   help              Print this message
#
# Examples:
#   bash scripts/qc_commands.sh validate
#   bash scripts/qc_commands.sh qc --task-mode pancreas_lesion
#   bash scripts/qc_commands.sh qc-fast --task-mode pancreas_lesion_subregions
#   bash scripts/qc_commands.sh evaluate --run-dir "$RUN_DIR"
#   bash scripts/qc_commands.sh golden-build --run-dir "$RUN_DIR" \
#       --output-dir "$RUN_DIR/golden_review_package"
#   bash scripts/qc_commands.sh admin --run-dir "$RUN_DIR"
#   bash scripts/qc_commands.sh reviewer --review-package "$REVIEWER_PACKAGE"
# =============================================================================

set -euo pipefail

# ── Resolve project root (one level above this script) ───────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python}"

# ── Helpers ──────────────────────────────────────────────────────────────────
sep() { printf '\n%s\n' "$(printf '=%.0s' {1..60})"; }
run() { sep; echo "  $*"; sep; "$@"; }

# ── Dispatch ─────────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true                 # remaining args forwarded to the underlying script

case "$CMD" in

  # -- Config validation -------------------------------------------------------
  validate)
    run "$PYTHON" "$ROOT/scripts/validate_config.py" "$@"
    ;;

  # -- Dataset summarization (serial) -----------------------------------------
  summarize)
    run "$PYTHON" "$ROOT/scripts/summarize_dataset.py" "$@"
    ;;

  # -- Dataset summarization (8 parallel workers) -----------------------------
  summarize-fast)
    run "$PYTHON" "$ROOT/scripts/summarize_dataset.py" --workers 8 "$@"
    ;;

  # -- Full pipeline: validate → summarize → QC --------------------------------
  qc)
    run "$PYTHON" "$ROOT/scripts/run_qc.py" "$@"
    ;;

  # -- Full pipeline, 8 workers + overwrite -----------------------------------
  qc-fast)
    run "$PYTHON" "$ROOT/scripts/run_qc.py" --workers 8 --overwrite "$@"
    ;;

  # -- Re-run QC for all three task modes ------------------------------------
  qc-all)
    for mode in pancreas_only pancreas_lesion pancreas_lesion_subregions; do
      sep
      echo "  ► task-mode: $mode"
      run "$PYTHON" "$ROOT/scripts/run_qc.py" --task-mode "$mode" --workers 8 --overwrite "$@"
    done
    sep
    echo "  ✓ All three task modes complete."
    sep
    ;;

  # -- Rebuild eval artifact for an existing run ------------------------------
  evaluate)
    run "$PYTHON" "$ROOT/scripts/evaluate_qc_run.py" "$@"
    ;;

  # -- Golden review package --------------------------------------------------
  golden-build)
    run "$PYTHON" "$ROOT/scripts/build_golden_review_package.py" "$@"
    ;;

  golden-import)
    run "$PYTHON" "$ROOT/scripts/import_golden_review.py" "$@"
    ;;

  # -- Read-only interactive review -------------------------------------------
  admin)
    run "$PYTHON" "$ROOT/agents/interactive_review_agent.py" --mode admin "$@"
    ;;

  reviewer)
    run "$PYTHON" "$ROOT/agents/interactive_review_agent.py" --mode reviewer "$@"
    ;;

  # -- Help -------------------------------------------------------------------
  help|--help|-h)
    # Print only the header block (lines before the blank line after the last '=')
    awk '/^set -euo pipefail/{exit} /^#!/{next} /^#/{print substr($0,3)}' "$BASH_SOURCE"
    ;;

  *)
    echo "Unknown command: '$CMD'.  Run '$0 help' to see available commands." >&2
    exit 1
    ;;

esac
