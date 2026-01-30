"""Base interfaces for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from prompt_rl.core.prompt import Prompt


@dataclass
class LLMResponse:
    """Response from an LLM."""

    text: str
    model: str = ""
    usage: Optional[dict[str, int]] = None
    raw: Optional[Any] = None


class LLMBackend(ABC):
    """Base interface for LLM backends (OpenAI, local, etc.)."""

    @abstractmethod
    def complete(
        self,
        prompt: Prompt | str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Generates a completion given a prompt.

        Args:
            prompt: Prompt or prompt text.
            max_tokens: Maximum tokens to generate.
            temperature: Temperature for sampling.
            **kwargs: Additional backend arguments.

        Returns:
            LLMResponse with text and metadata.
        """
        ...

    def refine_prompt(
        self,
        prompt: Prompt | str,
        instruction: str = "Improve this prompt while keeping its intent.",
        **kwargs: Any,
    ) -> LLMResponse:
        """
        Asks the LLM to refine a prompt according to an instruction.
        Useful as default action in the RL environment.
        """
        text = prompt.text if isinstance(prompt, Prompt) else prompt
        user = f"{instruction}\n\nCurrent prompt:\n{text}"
        return self.complete(user, **kwargs)
