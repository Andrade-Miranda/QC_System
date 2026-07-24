"""Deterministic hashing helpers for resources and artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def hash_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str | None:
    """Return a SHA256 hash for *path*, or None when the file is absent."""
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_json_payload(payload: Any) -> str:
    """Return a SHA256 hash for a JSON-serializable payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def resource_descriptor(path: Path, resource_type: str, *, required: bool = True) -> dict:
    """Build a provenance descriptor for an immutable external resource."""
    path = Path(path)
    exists = path.exists()
    info = {
        "resource_type": resource_type,
        "path": str(path),
        "exists": exists,
        "required": required,
    }
    if exists and path.is_file():
        stat = path.stat()
        info.update({
            "sha256": hash_file(path),
            "size_bytes": stat.st_size,
        })
    elif exists and path.is_dir():
        info.update({"kind": "directory"})
    return info
