"""
Backend para modelos locales (Ollama con Gemma, LM Studio, etc.).

Usa la API compatible con OpenAI para conectar a Ollama en localhost.
"""

from __future__ import annotations

from typing import Any, Optional

from prompt_rl.llm.base import LLMBackend, LLMResponse

OLLAMA_BASE = "http://localhost:11434/v1"
LM_STUDIO_BASE = "http://localhost:1234/v1"


def _check_openai() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Se requiere openai para conectar a Ollama: pip install openai"
        ) from exc


class LocalLLMBackend(LLMBackend):
    """
    Backend para servidores locales OpenAI-compatible (Ollama, LM Studio, vLLM).

    Ejemplos:
        Ollama Gemma: LocalLLMBackend(model="gemma3:4b")
        LM Studio:    LocalLLMBackend(model="local-model", base_url=LM_STUDIO_BASE)
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        _check_openai()
        from openai import OpenAI

        self.model = model
        kwargs: dict[str, Any] = {
            "api_key": api_key or "not-needed",
            "base_url": base_url or OLLAMA_BASE,
        }
        kwargs.update(client_kwargs)
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            usage=usage,
            raw=resp,
        )

    def generate_with_system(
        self,
        system_prompt: str,
        user_message: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **kwargs: Any,
    ) -> LLMResponse:
        """Completion con system prompt + mensaje de usuario (para Actor/validación)."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = resp.choices[0]
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            raw=resp,
        )
