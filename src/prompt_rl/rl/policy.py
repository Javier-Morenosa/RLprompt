"""Policies (agents) for the refinement environment."""

from typing import Any, Protocol

from prompt_rl.core.prompt import Prompt


class Policy(Protocol):
    """Protocol for policies that choose actions given state."""

    def select_action(self, state: Prompt, **kwargs: Any) -> Any:
        """Selects an action given the state (current prompt)."""
        ...


class RandomPolicy:
    """
    Policy that returns random/sampled actions.
    Useful as a baseline or placeholder until connecting a real RL agent.
    """

    def __init__(self, action_space: Any = None) -> None:
        self.action_space = action_space

    def select_action(self, state: Prompt, **kwargs: Any) -> Any:
        if self.action_space is not None and hasattr(self.action_space, "sample"):
            return self.action_space.sample()
        return state.text
