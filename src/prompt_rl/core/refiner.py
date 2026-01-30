"""Prompt refinement loop (orchestration without RL)."""

from typing import Callable, Optional

from prompt_rl.core.prompt import Prompt, PromptHistory


class RefinementLoop:
    """
    Refinement loop that applies a refinement function iteratively.
    Designed to be combined with an RL agent that chooses refinement actions.
    """

    def __init__(
        self,
        refine_fn: Callable[[Prompt], Prompt],
        max_steps: int = 10,
        stop_fn: Optional[Callable[[Prompt, Prompt], bool]] = None,
    ) -> None:
        """
        Args:
            refine_fn: Function that takes a Prompt and returns a refined one.
            max_steps: Maximum number of iterations.
            stop_fn: If provided, (prev, next) -> bool; True to stop.
        """
        self.refine_fn = refine_fn
        self.max_steps = max_steps
        self.stop_fn = stop_fn

    def run(self, initial: Prompt) -> PromptHistory:
        """Runs the loop until max_steps or until stop_fn returns True."""
        history = PromptHistory()
        current = initial
        history.append(current)

        for _ in range(self.max_steps - 1):
            next_prompt = self.refine_fn(current)
            history.append(next_prompt)
            if self.stop_fn and self.stop_fn(current, next_prompt):
                break
            current = next_prompt

        return history
