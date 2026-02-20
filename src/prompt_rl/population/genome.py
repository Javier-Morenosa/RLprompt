"""PromptGenome — modular prompt structured as named sections."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PromptGenome:
    """
    A system prompt represented as a dictionary of named sections.

    Typical sections: system_role, tone, constraints, examples, format.
    Sections with empty content are skipped when rendering.

    The genome is the unit stored in the Leaderboard. Its to_text() output
    is what gets written to system_prompt.md as the active policy.
    """

    sections: dict[str, str]      = field(default_factory=dict)
    metadata: dict[str, Any]      = field(default_factory=dict)

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def from_text(cls, text: str) -> "PromptGenome":
        """
        Build a genome from raw prompt text by splitting on the first
        blank line (paragraph boundary).

        - Two or more paragraphs → system_role + instructions
        - Single paragraph       → system_role only
        """
        text = text.strip()
        if "\n\n" in text:
            parts = text.split("\n\n", 1)
            sections = {"system_role": parts[0], "instructions": parts[1]}
        elif "\n" in text:
            parts = text.split("\n", 1)
            sections = {"system_role": parts[0], "instructions": parts[1]}
        else:
            sections = {"system_role": text}
        return cls(sections=sections)

    # ── Rendering ─────────────────────────────────────────────────────────────

    def to_text(self, order: tuple[str, ...] | None = None) -> str:
        """Render all non-empty sections joined by double newline."""
        keys  = order or tuple(self.sections.keys())
        parts = [
            self.sections[k].strip()
            for k in keys
            if k in self.sections and self.sections[k].strip()
        ]
        return "\n\n".join(parts)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def copy(self) -> "PromptGenome":
        return PromptGenome(
            sections=dict(self.sections),
            metadata=dict(self.metadata),
        )

    def get(self, key: str) -> str:
        return self.sections.get(key, "")

    def set(self, key: str, value: str) -> None:
        self.sections[key] = value

    def __len__(self) -> int:
        return len(self.to_text())
