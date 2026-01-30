"""Modular prompt genome: editable sections (tone, constraints, examples)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


DEFAULT_SECTIONS = ("system_role", "tone", "constraints", "examples", "format")


@dataclass
class PromptGenome:
    """
    Prompt genome: modular structure by sections.
    Each individual in the population has a genome that can be mutated and crossed.
    """

    sections: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.sections:
            self.sections = {k: "" for k in DEFAULT_SECTIONS}

    def to_prompt_text(self, section_order: Optional[tuple[str, ...]] = None) -> str:
        """Renders the genome as system prompt text."""
        order = section_order or tuple(self.sections.keys())
        parts = []
        for key in order:
            if key in self.sections and self.sections[key].strip():
                parts.append(self.sections[key].strip())
        return "\n\n".join(parts)

    def copy(self) -> PromptGenome:
        return PromptGenome(
            sections=dict(self.sections),
            metadata=dict(self.metadata),
        )

    def get_section(self, key: str) -> str:
        return self.sections.get(key, "")

    def set_section(self, key: str, value: str) -> None:
        self.sections[key] = value

    def __len__(self) -> int:
        return len(self.to_prompt_text())
