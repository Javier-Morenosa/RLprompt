"""Critic: evaluates quality (prompt_used, query, response) → score."""

from __future__ import annotations

from typing import Any, Optional, Protocol

from prompt_rl.actor_critic.actor import CandidateResponse


class Critic(Protocol):
    """
    Critic_θ(prompt, query, response) → expected_score.

    Learns to predict:
    - Probability of positive feedback
    - Estimated response quality
    - Alignment with objectives
    """

    def score(
        self,
        system_prompt: str,
        query: str,
        response: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        """Estimated quality score (e.g. probability of positive feedback)."""
        ...

    def update(
        self,
        system_prompt: str,
        query: str,
        response: str,
        reward: float,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Estimator update (e.g. gradient). Optional."""
        ...


class MockCritic:
    """Test Critic: fixed score or length/heuristic-based."""

    def __init__(self, default_score: float = 0.5) -> None:
        self.default_score = default_score

    def score(
        self,
        system_prompt: str,
        query: str,
        response: str,
        context: Optional[dict[str, Any]] = None,
    ) -> float:
        if not response.strip():
            return 0.0
        length_bonus = min(len(response) / 500.0, 0.3)
        return self.default_score + length_bonus

    def update(
        self,
        system_prompt: str,
        query: str,
        response: str,
        reward: float,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        pass
