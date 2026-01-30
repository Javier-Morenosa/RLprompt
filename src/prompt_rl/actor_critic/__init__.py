"""
Actor-Critic for generation and evaluation.

- Actor: generates/selects candidate responses (which prompt to use, which R to show).
- Critic: evaluates (prompt_used, query, response) → estimated score.
"""

from prompt_rl.actor_critic.actor import (
    Actor,
    RandomActor,
    get_actor_temperature,
    generate_candidates_parallel,
)
from prompt_rl.actor_critic.critic import Critic, MockCritic

__all__ = [
    "Actor",
    "RandomActor",
    "get_actor_temperature",
    "generate_candidates_parallel",
    "Critic",
    "MockCritic",
]
