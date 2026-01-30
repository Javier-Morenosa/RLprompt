"""Prompt representation and refinement history."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Prompt:
    """
    Represents a prompt with metadata for the RL refinement loop.

    Attributes:
        text: Text content of the prompt.
        version: Version number in the refinement history.
        metadata: Optional metadata (e.g. metrics, scores).
    """

    text: str
    version: int = 0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.version < 0:
            raise ValueError("version must be >= 0")

    def with_refinement(self, new_text: str, **meta) -> "Prompt":
        """Creates a new version of the prompt after a refinement step."""
        return Prompt(
            text=new_text,
            version=self.version + 1,
            metadata={**self.metadata, **meta},
        )

    def __str__(self) -> str:
        return self.text

    def __len__(self) -> int:
        return len(self.text)


@dataclass
class PromptHistory:
    """History of prompt versions during refinement."""

    prompts: list[Prompt] = field(default_factory=list)

    def append(self, prompt: Prompt) -> None:
        self.prompts.append(prompt)

    @property
    def current(self) -> Optional[Prompt]:
        return self.prompts[-1] if self.prompts else None

    def __len__(self) -> int:
        return len(self.prompts)
