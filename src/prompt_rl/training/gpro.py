"""
GPRO: Generalized Policy Optimization.

GPRO is the optimization algorithm that trains the Actor-Critic.
It updates the Actor (policy) and the Critic (value estimator) using
batches of (query, candidates, chosen_index, reward) collected from
the training loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol

from prompt_rl.actor_critic import Actor, Critic


@dataclass
class GPROTransition:
    """One transition for GPRO: query, candidates, chosen index, reward."""

    query: str
    system_prompt_used: str
    response_shown: str
    chosen_index: int
    reward: float
    candidates: list[Any] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)


class GPROOptimizer(Protocol):
    """
    GPRO optimizer: trains Actor and Critic from a batch of transitions.

    Implement update(batch, actor, critic) to perform one or more gradient
    steps (e.g. policy gradient / clipped objective for Actor, MSE for Critic).
    """

    def update(
        self,
        batch: list[GPROTransition],
        actor: Actor,
        critic: Critic,
        **kwargs: Any,
    ) -> dict[str, float]:
        """
        Performs one GPRO update step on the given batch.
        Returns optional metrics (e.g. actor_loss, critic_loss).
        """
        ...


class NoOpGPROOptimizer:
    """
    No-op GPRO optimizer: does not update Actor or Critic.
    Use as a placeholder until a real GPRO implementation is plugged in.
    """

    def update(
        self,
        batch: list[GPROTransition],
        actor: Actor,
        critic: Critic,
        **kwargs: Any,
    ) -> dict[str, float]:
        return {}
