"""Base interfaces for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Response from an LLM completion call."""

    text:  str
    model: str = ""
    usage: Optional[dict[str, int]] = None
    raw:   Optional[Any] = None


class LLMBackend(ABC):
    """Abstract base for LLM backends (OpenAI-compatible cloud, local Ollama/Groq)."""

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generate a completion for the given prompt text.

        Args:
            prompt      : Plain-text prompt sent as the user message.
            max_tokens  : Maximum tokens to generate.
            temperature : Sampling temperature.

        Returns:
            LLMResponse with .text and optional .usage dict.
        """
        ...
