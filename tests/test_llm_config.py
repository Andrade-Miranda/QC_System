from __future__ import annotations

from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from unittest.mock import patch

from agents.utils.llm_backend import OllamaBackend, ProviderConfig, load_provider_config


class LLMConfigTests(unittest.TestCase):
    def test_ollama_default_is_devstral_and_remains_intent_only_provider(self) -> None:
        config = load_provider_config(ROOT / "configs" / "llm_config.yaml")

        self.assertEqual(config.provider, "ollama")
        self.assertEqual(config.model, "devstral:latest")
        self.assertEqual(config.base_url, "http://localhost:11434")
        self.assertFalse(config.extended_ollama_options)

    def test_extended_ollama_options_are_opt_in(self) -> None:
        captured = []

        def fake_post(_url, payload, _timeout, **_kwargs):
            captured.append(payload)
            return {"message": {"content": "{}"}}

        base = ProviderConfig(
            provider="ollama",
            model="devstral:test",
            timeout_seconds=1,
            max_tokens=32,
            base_url="http://unused",
            temperature=0,
            top_p=0.5,
            seed=123,
            context_length=2048,
        )
        with patch("agents.utils.llm_backend._post_json", side_effect=fake_post):
            OllamaBackend(base).complete([])
            OllamaBackend(ProviderConfig(**{**base.__dict__, "extended_ollama_options": True})).complete([])

        self.assertNotIn("top_p", captured[0]["options"])
        self.assertNotIn("seed", captured[0]["options"])
        self.assertNotIn("num_ctx", captured[0]["options"])
        self.assertEqual(captured[1]["options"]["top_p"], 0.5)
        self.assertEqual(captured[1]["options"]["seed"], 123)
        self.assertEqual(captured[1]["options"]["num_ctx"], 2048)


if __name__ == "__main__":
    unittest.main()
