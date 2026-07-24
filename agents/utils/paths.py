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

logger = logging.getLogger(__name__)

def _discover_task_modes() -> frozenset:
    profiles = _CONFIGS_DIR / "task_profiles"
    modes = {p.stem for p in profiles.glob("*.yaml")} if profiles.exists() else set()
    return frozenset(modes or {
        "pancreas_only",
        "pancreas_lesion",
        "pancreas_lesion_subregions",
    })


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

    # Resolve PROJECT_ROOT early so we can find the dataset config
    project_root_str = cfg.get("PROJECT_ROOT", "")
    project_root = Path(project_root_str).resolve() if project_root_str else _PROJECT_ROOT

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

    return cfg


def resolve_project_paths(yaml_path: Path | None = None) -> ProjectPaths:
    """
    Load paths.yaml and return a fully-resolved ProjectPaths dataclass.

    All relative paths in the YAML are resolved relative to PROJECT_ROOT.
    PROJECT_ROOT is taken from the YAML value (absolute) or inferred from
    this file's location if the YAML value is relative.
    """
    cfg = load_paths_config(yaml_path)

    raw_root_str = cfg.get("RAW_DATASET_ROOT", "")
    raw_root     = Path(raw_root_str).resolve() if raw_root_str else Path()

    # Dataset name: explicit key wins, else infer from the last folder component
    # e.g. /path/to/PantsMini  →  "PantsMini"
    dataset_name = (
        cfg.get("DATASET_NAME")
        or (raw_root.name if raw_root.name else "unknown_dataset")
    )

    project_root_str = cfg.get("PROJECT_ROOT", "")
    project_root = Path(project_root_str).resolve() if project_root_str else _PROJECT_ROOT

    def _abs(relative_key: str, fallback: str = "") -> Path:
        """Resolve a relative-path config value against project_root."""
        val = cfg.get(relative_key, fallback)
        p   = Path(val)
        return p if p.is_absolute() else (project_root / p)

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
