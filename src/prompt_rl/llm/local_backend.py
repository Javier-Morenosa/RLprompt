"""Backend for local / OpenAI-compatible LLM servers (LM Studio, Ollama, vLLM, etc.)."""

from __future__ import annotations

from typing import Any, Optional

from prompt_rl.llm.base import LLMBackend, LLMResponse
from prompt_rl.llm.openai_backend import OpenAIBackend


# Common default base URLs for local servers (OpenAI-compatible API)
LM_STUDIO_DEFAULT_BASE = "http://localhost:1234/v1"
OLLAMA_OPENAI_BASE = "http://localhost:11434/v1"  # Ollama with OpenAI compatibility


class LocalLLMBackend(OpenAIBackend):
    """
    Backend for local or custom LLM servers that expose an OpenAI-compatible API.

    Use this when running a model locally (e.g. LM Studio, Ollama, vLLM, LiteLLM)
    instead of calling the OpenAI cloud API. No OpenAI API key is required unless
    your local server enforces one.

    Examples:
        LM Studio (default port 1234):
            LocalLLMBackend(model="local-model", base_url="http://localhost:1234/v1")
        Ollama (OpenAI-compatible endpoint):
            LocalLLMBackend(model="llama3.2", base_url="http://localhost:11434/v1")
        Custom server with API key:
            LocalLLMBackend(model="my-model", base_url="https://...", api_key="...")
    """

    def __init__(
        self,
        model: str = "local-model",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        url = base_url if base_url is not None else LM_STUDIO_DEFAULT_BASE
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=url,
            **client_kwargs,
        )
