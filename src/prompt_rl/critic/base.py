"""PerceptionCritic — abstract interface for the online Critic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from prompt_rl.core.cycle import PerceptionCycle


@dataclass
class CriticOutput:
    """Result of one Critic evaluation."""

    critic_score:    float  # quality estimate for the current policy [0.0 – 1.0]
    proposed_prompt: str    # refined system prompt (JSON policy or legacy text)
    reasoning:       str    # one-line explanation
    nota:            str = ""  # hypothesis / what to watch in the next cycle
    # When comment is present: how the Critic treated it
    comment_treatment:        str = ""  # "direct_rule" | "refinement"
    comment_treatment_reasoning: str = ""  # why this choice


@runtime_checkable
class PerceptionCritic(Protocol):
    """
    Critic protocol for the online RL loop.

    Contract:
        - Receives: cycle.system_prompt, cycle.verdict, cycle.comment
        - BLIND to: cycle.user_query, cycle.bot_response

    Rationale: the Critic must learn to improve system prompts from reward
    signals (human verdict + correction) alone — not from conversation content.
    This prevents it from overfitting to specific queries and forces it to
    develop general prompt-quality intuition.
    """

    def evaluate(self, cycle: PerceptionCycle) -> CriticOutput:
        ...
