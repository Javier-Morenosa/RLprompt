from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_rl.actor_critic_loop.config import ActorCriticConfig
from prompt_rl.actor_critic_loop.critic import LLMCritic, CriticInput
from prompt_rl.actor_critic_loop.actor import LLMActor
from prompt_rl.actor_critic_loop.feedback import (
    HumanMultiSelectFeedback,
    MultiSelectResult,
    compute_rewards_from_selection,
)


@dataclass
class LoopState:
    iteration: int = 0
    base_instruction: str = ""
    refinement_hint: Optional[str] = None
    history: list[dict[str, Any]] = field(default_factory=list)


class ActorCriticLoop:
    def __init__(
        self,
        actor: LLMActor,
        critic: LLMCritic,
        feedback: HumanMultiSelectFeedback,
        config: Optional[ActorCriticConfig] = None,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.feedback = feedback
        self.config = config or ActorCriticConfig()
        self._state = LoopState()

    def run_iteration(self, query: str) -> dict[str, Any]:
        cfg = self.config
        state = self._state
        state.iteration += 1

        prompts = self.actor.generate_prompt_variations(
            state.base_instruction or "You are a helpful assistant.",
            cfg.num_variations,
            refinement_hint=state.refinement_hint,
            query=query,
        )

        outputs = self.actor.generate_responses(prompts, query)
        responses = [o.response for o in outputs]

        result = self.feedback.get_selection(query, responses)
        rewards = compute_rewards_from_selection(result.selected_indices, len(outputs))
        total_correct = sum(1 for r in rewards if r >= 0.5)
        prev_total = state.history[-1]["total_correct"] if state.history else None

        critic_inputs = [
            CriticInput(system_prompt=o.system_prompt, reward=rewards[o.index], has_feedback_comment=False, human_comment=None)
            for o in outputs
        ]
        refinement_hint = self.critic.get_refinement_direction(
            critic_inputs,
            context={"previous_total_correct": prev_total},
        )
        state.refinement_hint = refinement_hint

        record = {
            "iteration": state.iteration,
            "query": query,
            "prompts": prompts,
            "responses": responses,
            "selected_indices": result.selected_indices,
            "rewards": rewards,
            "total_correct": total_correct,
            "refinement_hint": refinement_hint,
        }
        state.history.append(record)
        return record

    def run(
        self,
        queries: list[str],
        base_instruction: str = "You are a helpful assistant.",
    ) -> list[dict[str, Any]]:
        self._state.base_instruction = base_instruction
        self._state.iteration = 0
        results = []
        for q in queries:
            record = self.run_iteration(q)
            results.append(record)
        return results
