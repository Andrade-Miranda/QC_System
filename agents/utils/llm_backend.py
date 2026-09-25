"""Provider-neutral, standard-library chat backends for intent classification."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ProviderError(RuntimeError):
    """Raised when an optional provider cannot return a usable classification."""


@dataclass(frozen=True)
class ProviderConfig:
    provider: str
    model: str | None
    timeout_seconds: float
    max_tokens: int
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = 0
    context_length: int | None = None
    extended_ollama_options: bool = False


class ChatBackend:
    """Minimal interface used by the interactive review agent."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.provider != "none"

    def complete(self, messages: list[dict[str, str]]) -> str:
        raise NotImplementedError


class NoneBackend(ChatBackend):
    def complete(self, messages: list[dict[str, str]]) -> str:
        raise ProviderError("LLM provider is disabled")


class OllamaBackend(ChatBackend):
    def complete(self, messages: list[dict[str, str]]) -> str:
        endpoint = f"{str(self.config.base_url).rstrip('/')}/api/chat"
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
            },
        }
        if self.config.extended_ollama_options:
            payload["options"]["top_p"] = self.config.top_p
        if self.config.extended_ollama_options and self.config.seed is not None:
            payload["options"]["seed"] = self.config.seed
        if self.config.extended_ollama_options and self.config.context_length is not None:
            payload["options"]["num_ctx"] = self.config.context_length
        response = _post_json(endpoint, payload, self.config.timeout_seconds)
        content = ((response.get("message") or {}).get("content")) if isinstance(response, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama returned no message content")
        return content.strip()


class OpenAIBackend(ChatBackend):
    def complete(self, messages: list[dict[str, str]]) -> str:
        api_key = os.environ.get(str(self.config.api_key_env or ""))
        if not api_key:
            raise ProviderError(f"OpenAI API key environment variable is not set: {self.config.api_key_env}")
        endpoint = str(self.config.base_url).rstrip("/")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
        }
        response = _post_json(
            endpoint,
            payload,
            self.config.timeout_seconds,
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            content = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("OpenAI returned no message content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("OpenAI returned empty message content")
        return content.strip()


def _post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Content-Type": "application/json", **(headers or {})}
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=request_headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except (OSError, TimeoutError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        raise ProviderError(f"Provider request failed: {exc}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderError("Provider returned invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise ProviderError("Provider returned a non-object response")
    return parsed


def load_provider_config(path: Path, provider_override: str | None = None) -> ProviderConfig:
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderError(f"Cannot load LLM configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProviderError("LLM configuration must be a mapping")
    agent = raw.get("interactive_review_agent") or {}
    if not isinstance(agent, dict):
        raise ProviderError("interactive_review_agent configuration must be a mapping")
    provider = str(provider_override or agent.get("provider") or raw.get("provider") or "none").lower()
    if provider not in {"none", "ollama", "openai"}:
        raise ProviderError(f"Unsupported LLM provider: {provider}")
    timeout = agent.get("timeout_seconds", raw.get("timeout_seconds", 20))
    max_tokens = agent.get("max_tokens", 512)
    temperature = agent.get("temperature", raw.get("temperature", 0))
    top_p = agent.get("top_p", raw.get("top_p", 1))
    seed = agent.get("seed", raw.get("seed", 0))
    context_length = agent.get("context_length", raw.get("context_length"))
    try:
        timeout_value = float(timeout)
        max_tokens_value = int(max_tokens)
        temperature_value = float(temperature)
        top_p_value = float(top_p)
        seed_value = None if seed is None else int(seed)
        context_length_value = None if context_length is None else int(context_length)
    except (TypeError, ValueError) as exc:
        raise ProviderError("LLM decoding configuration values must be numeric") from exc
    if timeout_value <= 0 or max_tokens_value <= 0:
        raise ProviderError("LLM timeout and max_tokens must be positive")
    if not 0 <= temperature_value <= 2 or not 0 < top_p_value <= 1:
        raise ProviderError("LLM temperature and top_p are outside supported ranges")
    if context_length_value is not None and context_length_value <= 0:
        raise ProviderError("LLM context_length must be positive when configured")
    section = raw.get(provider) or {}
    if not isinstance(section, dict):
        raise ProviderError(f"{provider} configuration must be a mapping")
    model = None if provider == "none" else section.get("model")
    if provider != "none" and (not isinstance(model, str) or not model):
        raise ProviderError(f"{provider} model must be configured")
    if provider == "ollama":
        base_url = str(section.get("base_url") or "http://localhost:11434")
        api_key_env = None
    elif provider == "openai":
        base_url = str(section.get("base_url") or "https://api.openai.com/v1/chat/completions")
        api_key_env = str(section.get("api_key_env") or "OPENAI_API_KEY")
    else:
        base_url = None
        api_key_env = None
    return ProviderConfig(
        provider=provider,
        model=model,
        timeout_seconds=timeout_value,
        max_tokens=max_tokens_value,
        base_url=base_url,
        api_key_env=api_key_env,
        temperature=temperature_value,
        top_p=top_p_value,
        seed=seed_value,
        context_length=context_length_value,
    )


def load_interaction_logging(path: Path) -> tuple[bool, str, bool, int]:
    """Load interaction/log display settings without exposing provider internals."""
    try:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ProviderError(f"Cannot load LLM configuration: {exc}") from exc
    if not isinstance(raw, dict):
        raise ProviderError("LLM configuration must be a mapping")
    enabled = raw.get("log_interactions", True)
    if not isinstance(enabled, bool):
        raise ProviderError("log_interactions must be boolean")
    filename = raw.get("interaction_log_filename", "interactive_review_interactions.jsonl")
    if not isinstance(filename, str) or not filename or Path(filename).name != filename:
        raise ProviderError("interaction_log_filename must be a plain filename")
    log_query_text = raw.get("log_query_text", False)
    if not isinstance(log_query_text, bool):
        raise ProviderError("log_query_text must be boolean")
    agent = raw.get("interactive_review_agent") or {}
    max_cases = agent.get("max_displayed_cases", 50) if isinstance(agent, dict) else 50
    if not isinstance(max_cases, int) or max_cases <= 0:
        raise ProviderError("interactive_review_agent.max_displayed_cases must be a positive integer")
    return enabled, filename, log_query_text, max_cases


def build_backend(config: ProviderConfig, *, allow_external_provider: bool = False) -> ChatBackend:
    if config.provider == "ollama":
        return OllamaBackend(config)
    if config.provider == "openai":
        if not allow_external_provider:
            raise ProviderError("OpenAI requires explicit --allow-external-provider")
        return OpenAIBackend(config)
    return NoneBackend(config)
