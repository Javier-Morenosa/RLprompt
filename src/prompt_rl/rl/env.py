"""RL environment for prompt refinement."""

from dataclasses import dataclass
from typing import Any, Callable, Optional

from prompt_rl.core.prompt import Prompt


@dataclass
class EnvStep:
    """Result of one environment step."""

    observation: Prompt
    reward: float
    done: bool
    info: dict


class PromptRefinementEnv:
    """
    Gym-like environment for prompt refinement with RL.

    - State: current prompt (Prompt).
    - Action: refinement (new text or parameters).
    - Reward: configurable function (quality, length, etc.).
    """

    def __init__(
        self,
        initial_prompt: Prompt,
        refine_fn: Callable[[Prompt, Any], Prompt],
        reward_fn: Callable[[Prompt, Prompt, Any], float],
        max_steps: int = 10,
    ) -> None:
        """
        Args:
            initial_prompt: Initial prompt.
            refine_fn: (current_prompt, action) -> refined_prompt.
            reward_fn: (prev_prompt, new_prompt, info) -> reward.
            max_steps: Maximum steps per episode.
        """
        self.initial_prompt = initial_prompt
        self.refine_fn = refine_fn
        self.reward_fn = reward_fn
        self.max_steps = max_steps
        self._step_count: int = 0
        self._current_prompt: Optional[Prompt] = None
        self._prev_prompt: Optional[Prompt] = None

    def reset(self) -> Prompt:
        """Resets the environment and returns the initial state."""
        self._step_count = 0
        self._current_prompt = self.initial_prompt
        self._prev_prompt = None
        assert self._current_prompt is not None
        return self._current_prompt

    def step(self, action: Any, info: Optional[dict] = None) -> EnvStep:
        """
        Performs one step: applies the action (refinement) and computes reward.

        Args:
            action: Agent action (e.g. refined text or parameters).
            info: Additional info for reward_fn (e.g. LLM response).

        Returns:
            EnvStep with observation, reward, done, info.
        """
        if self._current_prompt is None:
            raise RuntimeError("Call reset() before step().")

        self._prev_prompt = self._current_prompt
        self._current_prompt = self.refine_fn(self._current_prompt, action)
        self._step_count += 1

        reward = self.reward_fn(
            self._prev_prompt,
            self._current_prompt,
            info or {},
        )
        done = self._step_count >= self.max_steps

        return EnvStep(
            observation=self._current_prompt,
            reward=reward,
            done=done,
            info={"step": self._step_count, **(info or {})},
        )

    @property
    def current_prompt(self) -> Optional[Prompt]:
        return self._current_prompt

    @property
    def step_count(self) -> int:
        return self._step_count
