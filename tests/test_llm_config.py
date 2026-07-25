from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.utils.llm_backend import load_provider_config


class LLMConfigTests(unittest.TestCase):
    def test_ollama_default_is_devstral_and_remains_intent_only_provider(self) -> None:
        config = load_provider_config(ROOT / "configs" / "llm_config.yaml")

        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "devstral:latest")
        self.assertEqual(config.base_url, "http://localhost:11434")


if __name__ == "__main__":
    unittest.main()
