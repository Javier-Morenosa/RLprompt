"""PerceptionCycle — the fundamental data unit of the online RL loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PerceptionCycle:
    """
    One complete observation-feedback loop.

    Phases (mirrors the subconscious comparison model):
      ① Predictive Model  — active system_prompt (Actor's current policy)
      ② System Action     — user_query + bot_response
      ③ Observation Phase — where the human looked: dwell, select, review
      ④ ACC Signal        — verdict (CORRECTO / INCORRECTO) + optional comment

    Design rule: the Critic receives ONLY (system_prompt, verdict, comment).
    It is intentionally blind to user_query and bot_response — forcing it
    to learn to improve prompts from reward signals alone, not from content.
    """

    system_prompt: str
    user_query:    str           # captured for logging; NOT passed to Critic
    bot_response:  str           # captured for logging; NOT passed to Critic
    verdict:       str           # "CORRECTO" | "INCORRECTO"
    comment:       str           # human correction text (may be empty)
    dwell_seconds: float = 0.0   # implicit reading-time signal
    observations:  list[str] = field(default_factory=list)
    timestamp:     str = field(
        default_factory=lambda: datetime.now().isoformat()
    )

    @property
    def is_correct(self) -> bool:
        return self.verdict == "CORRECTO"

    @property
    def has_correction(self) -> bool:
        return bool(self.comment.strip())
