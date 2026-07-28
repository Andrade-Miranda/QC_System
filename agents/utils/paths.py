"""
QC System — centralized path management.

Usage
-----
    from agents.utils.paths import load_paths_config, resolve_project_paths, ensure_output_directories

    paths = resolve_project_paths()   # loads configs/paths.yaml relative to PROJECT_ROOT

    # Dataset name is inferred automatically from RAW_DATASET_ROOT:
    paths.dataset_name           # e.g. "PantsMini"

    # All output dirs are dataset-specific:
    paths.summary_dir / f"{paths.dataset_name}_summary.json"
    paths.logs_dir / "summarize.log"
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False

# ---------------------------------------------------------------------------
# Locate the project root from this file's position:
#   agents/utils/paths.py  →  ../..  →  QC_System/
# ---------------------------------------------------------------------------
_THIS_FILE     = Path(__file__).resolve()
_AGENTS_UTILS  = _THIS_FILE.parent          # agents/utils/
_AGENTS_DIR    = _AGENTS_UTILS.parent       # agents/
_PROJECT_ROOT  = _AGENTS_DIR.parent         # QC_System/

_CONFIGS_DIR   = _PROJECT_ROOT / "configs"
_DEFAULT_PATHS_YAML = _CONFIGS_DIR / "paths.yaml"

_ENV_CONFIG_OVERRIDES = {
    "AGENTQC_PROJECT_ROOT": "PROJECT_ROOT",
    "AGENTQC_DATASET_CONFIG": "DATASET_CONFIG",
    "AGENTQC_RAW_DATASET_ROOT": "RAW_DATASET_ROOT",
    "AGENTQC_OUTPUT_ROOT": "OUTPUT_ROOT",
    "AGENTQC_LOGS_DIR": "LOGS_DIR",
}

logger = logging.getLogger(__name__)

APPROVED_V1_TASK_MODES: frozenset[str] = frozenset({
    "pancreas_only",
    "pancreas_lesion",
    "pancreas_lesion_subregions",
})


def _discover_task_modes() -> frozenset:
    profiles = _CONFIGS_DIR / "task_profiles"
    modes = {p.stem for p in profiles.glob("*.yaml")} if profiles.exists() else set()
    # V1 is deliberately limited to the three approved segmentation profiles.
    # Extra YAML files do not become executable task modes without an approved
    # implementation plan for their evidence and decision semantics.
    return frozenset((modes & APPROVED_V1_TASK_MODES) or APPROVED_V1_TASK_MODES)


# Supported task modes — sourced from configs/task_profiles/*.yaml when present.
VALID_TASK_MODES: frozenset = _discover_task_modes()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class ProjectPaths:
    """Resolved absolute Path objects for every directory / file in the system."""

    # Active dataset identity
    dataset_name:   str  = ""              # inferred from RAW_DATASET_ROOT folder name

    # Raw dataset (immutable)
    raw_root:       Path = field(default_factory=Path)
    raw_images_dir: Path = field(default_factory=Path)
    raw_labels_dir: Path = field(default_factory=Path)
    raw_metadata:   Path = field(default_factory=Path)
    image_suffix:   str  = "_0000.nii.gz"

    # Project root (QC framework root — shared across datasets)
    project_root:   Path = field(default_factory=Path)

    # Dataset-specific output dirs  (outputs/<dataset_name>/<subdir>)
    qc_dir:         Path = field(default_factory=Path)
    summary_dir:    Path = field(default_factory=Path)

    # Shared (not dataset-specific)
    logs_dir:       Path = field(default_factory=Path)

    # Specific output file
    summary_json:       Path = field(default_factory=Path)

    # Config files
    thresholds_config:            Path = field(default_factory=Path)
    default_threshold_method:     str = "deterministic"
    paths_yaml:                   Path = field(default_factory=Path)
    raw_dataset_root_configured:  bool = False


def _expand_path_value(value: Any) -> str:
    return os.path.expandvars(os.path.expanduser(str(value).strip()))


def _resolve_project_root(value: Any) -> Path:
    raw = _expand_path_value(value) if value else ""
    if not raw:
        return _PROJECT_ROOT
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _resolve_path(value: Any, base: Path) -> Path:
    raw = _expand_path_value(value) if value else ""
    if not raw:
        return Path()
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _apply_env_overrides(cfg: dict[str, Any], *, keys: set[str] | None = None) -> None:
    for env_name, cfg_key in _ENV_CONFIG_OVERRIDES.items():
        if keys is not None and cfg_key not in keys:
            continue
        env_value = os.environ.get(env_name)
        if env_value is not None and env_value.strip():
            cfg[cfg_key] = env_value


def load_paths_config(yaml_path: Path | None = None) -> dict[str, Any]:
    """
    Load paths.yaml (and the optional dataset config it points to) and return
    the merged result as a plain dict.

    Merge order (later values win):
        1. configs/paths.yaml       — framework defaults + active DATASET_CONFIG pointer
        2. datasets/dataset_configs/<name>.yaml  — dataset-specific overrides

    yaml_path defaults to configs/paths.yaml relative to the project root
    inferred from this file's location.
    """
    yaml_path = yaml_path or _DEFAULT_PATHS_YAML

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"paths.yaml not found at {yaml_path}. "
            "Run from the QC_System directory or pass yaml_path explicitly."
        )

    if not _HAS_YAML:
        raise ImportError("PyYAML is required: pip install pyyaml")

    with open(yaml_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    # Environment PROJECT_ROOT / DATASET_CONFIG may affect dataset-config lookup.
    _apply_env_overrides(cfg, keys={"PROJECT_ROOT", "DATASET_CONFIG"})

    # Resolve PROJECT_ROOT early so we can find the dataset config.
    project_root = _resolve_project_root(cfg.get("PROJECT_ROOT", ""))

    # Optionally merge a per-dataset config on top
    dataset_config_rel = cfg.get("DATASET_CONFIG", "")
    if dataset_config_rel:
        ds_cfg_path = Path(dataset_config_rel)
        if not ds_cfg_path.is_absolute():
            ds_cfg_path = project_root / ds_cfg_path
        if ds_cfg_path.exists():
            with open(ds_cfg_path, encoding="utf-8") as fh:
                ds_cfg = yaml.safe_load(fh) or {}
            cfg.update(ds_cfg)   # dataset config wins over paths.yaml defaults
            logger.debug("Merged dataset config from %s", ds_cfg_path)
        else:
            logger.warning("DATASET_CONFIG points to missing file: %s", ds_cfg_path)

    # Optional gitignored local override.  This is intentionally loaded after
    # the tracked dataset config so workstation-specific paths never need to be
    # committed.  Default: configs/paths.local.yaml next to paths.yaml.
    local_override = cfg.get("LOCAL_PATHS_CONFIG", "paths.local.yaml")
    if local_override:
        local_path = Path(_expand_path_value(local_override))
        if not local_path.is_absolute():
            local_path = yaml_path.parent / local_path
        if local_path.exists():
            with open(local_path, encoding="utf-8") as fh:
                local_cfg = yaml.safe_load(fh) or {}
            cfg.update(local_cfg)
            logger.debug("Merged local paths override from %s", local_path)

    # Environment variables have final precedence over tracked and local YAML.
    _apply_env_overrides(cfg)

    return cfg


def resolve_project_paths(yaml_path: Path | None = None) -> ProjectPaths:
    """
    Load paths.yaml and return a fully-resolved ProjectPaths dataclass.

    All relative paths in the YAML are resolved relative to PROJECT_ROOT.
    PROJECT_ROOT is taken from the YAML value (absolute) or inferred from
    this file's location if the YAML value is relative.
    """
    cfg = load_paths_config(yaml_path)

    project_root = _resolve_project_root(cfg.get("PROJECT_ROOT", ""))

    raw_root_str = _expand_path_value(cfg.get("RAW_DATASET_ROOT", ""))
    raw_root_configured = bool(raw_root_str)
    raw_root     = _resolve_path(raw_root_str, project_root) if raw_root_configured else Path()

    # Dataset name: explicit key wins, else infer from the last folder component
    # e.g. /path/to/PantsMini  →  "PantsMini"
    dataset_name = (
        cfg.get("DATASET_NAME")
        or (raw_root.name if raw_root.name else "unknown_dataset")
    )

    def _abs(relative_key: str, fallback: str = "") -> Path:
        """Resolve a relative-path config value against project_root."""
        val = cfg.get(relative_key, fallback)
        p   = Path(_expand_path_value(val))
        return p.resolve() if p.is_absolute() else (project_root / p).resolve()

    # Dataset-specific output root:  outputs/<dataset_name>/
    output_root   = _abs("OUTPUT_ROOT", "outputs")
    dataset_out   = output_root / dataset_name

    qc_subdir     = cfg.get("OUTPUT_QC_SUBDIR",       "qc")
    sum_subdir    = cfg.get("OUTPUT_SUMMARY_SUBDIR",   "summaries")

    qc_dir       = dataset_out / qc_subdir
    summary_dir  = dataset_out / sum_subdir
    logs_dir     = _abs("LOGS_DIR", "logs")   # shared across datasets

    # Summary JSON filename: explicit override or auto-generated from dataset name
    summary_json_filename = cfg.get("SUMMARY_JSON_FILENAME") or f"{dataset_name}_summary.json"

    return ProjectPaths(
        # dataset identity
        dataset_name   = dataset_name,

        # raw dataset
        raw_root       = raw_root,
        raw_images_dir = raw_root / cfg.get("RAW_IMAGES_SUBDIR", "imagesTr"),
        raw_labels_dir = raw_root / cfg.get("RAW_LABELS_SUBDIR", "labelsTr"),
        raw_metadata   = raw_root / cfg.get("RAW_METADATA_FILE", "metadata.xlsx"),
        image_suffix   = cfg.get("IMAGE_SUFFIX", "_0000.nii.gz"),

        # project root
        project_root   = project_root,

        # generated dirs (dataset-specific)
        qc_dir         = qc_dir,
        summary_dir    = summary_dir,
        logs_dir       = logs_dir,

        # generated files
        summary_json    = summary_dir / summary_json_filename,

        # configs
        thresholds_config            = _abs("THRESHOLDS_CONFIG", "configs/thresholds.yaml"),
        default_threshold_method     = str(cfg.get("DEFAULT_THRESHOLD_METHOD", "deterministic")).strip().lower(),
        paths_yaml                   = yaml_path or _DEFAULT_PATHS_YAML,
        raw_dataset_root_configured  = raw_root_configured,
    )


def ensure_output_directories(paths: ProjectPaths) -> None:
    """
    Create the shared (non-dataset-specific) output directories.
    Dataset- and task-mode-specific dirs are created by build_output_dirs().
    Never touches any directory under raw_root.
    """
    # Only the shared logs dir is created here; everything else lives under
    # outputs/<dataset_name>/<task_mode>/ and is created by build_output_dirs().
    paths.logs_dir.mkdir(parents=True, exist_ok=True)


def build_output_dirs(
    base_output_dir: "Path | str",
    task_mode: str,
    *,
    create: bool = True,
) -> dict:
    """Build (and optionally create) task-mode-specific output subdirectories.

    Creates the following structure under *base_output_dir*::

        <base_output_dir>/<task_mode>/
            qc/
            summary/

    Args:
        base_output_dir: Dataset-level output root
                         (e.g. ``outputs/PantsMini/``).
        task_mode:       One of VALID_TASK_MODES.
        create:          If True, create directories (parents=True,
                         exist_ok=True).  Set to False for dry-run checks.

    Returns:
        dict with keys ``task_dir``, ``qc_dir``, and ``summary_dir`` (all
        ``Path`` objects).

    Raises:
        ValueError: if *task_mode* is not in VALID_TASK_MODES.
    """
    if task_mode not in VALID_TASK_MODES:
        raise ValueError(
            f"Unsupported TASK_MODE: '{task_mode}'. "
            f"Valid values: {sorted(VALID_TASK_MODES)}"
        )
    task_dir     = Path(base_output_dir) / task_mode
    qc_dir       = task_dir / "qc"
    summary_dir  = task_dir / "summary"
    if create:
        for d in (qc_dir, summary_dir):
            d.mkdir(parents=True, exist_ok=True)
    return {
        "task_dir":     task_dir,
        "qc_dir":       qc_dir,
        "summary_dir":  summary_dir,
    }


def setup_file_logging(
    log_path: Path,
    logger_name: str = "pantsmini_qc",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Configure a named logger that writes to *log_path* as well as stdout.
    Returns the logger.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lg = logging.getLogger(logger_name)
    lg.setLevel(level)
    if not lg.handlers:
        fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        lg.addHandler(fh)
        lg.addHandler(ch)
    return lg
