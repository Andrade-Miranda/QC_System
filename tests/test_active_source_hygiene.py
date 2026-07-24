from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_SOURCE_DIRS = ("agents", "artifacts", "scripts")


class ActiveSourceHygieneTests(unittest.TestCase):
    def test_active_python_does_not_import_archive(self) -> None:
        violations: list[str] = []
        for directory in ACTIVE_SOURCE_DIRS:
            for path in (PROJECT_ROOT / directory).rglob("*.py"):
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        modules = [node.module or ""]
                    else:
                        continue
                    if any(module == "archive" or module.startswith("archive.") for module in modules):
                        violations.append(str(path.relative_to(PROJECT_ROOT)))
        self.assertEqual(violations, [])


if __name__ == "__main__":
    unittest.main()
