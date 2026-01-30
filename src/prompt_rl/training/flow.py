"""
Integrated optimization flow:

  Population → Query → Actor (generates K responses, selects R*) → Show R*
  → Human Feedback → Critic (updates) → Guided evolution (fitness, mutate/crossover)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from prompt_rl.actor_critic import Actor, Critic
from prompt_rl.actor_critic.actor import (
    CandidateResponse,
    generate_candidates,
    generate_candidates_parallel,
    get_actor_temperature,
)
from prompt_rl.evolution import Population, PromptGenome
from prompt_rl.evolution.population import Individual
from prompt_rl.feedback import FeedbackAggregator
from prompt_rl.llm.base import LLMBackend
from prompt_rl.rl.rewards import HybridReward


@dataclass
class FlowStepResult:
    """Result of one flow step: query → feedback → reward."""

    query: str
    system_prompt_used: str
    prompt_index_used: int
    response_shown: str
    candidates: list[CandidateResponse]
    feedback_score: float
    critic_score: float
    reward_total: float
    info: dict[str, Any] = field(default_factory=dict)
    temperature_used: Optional[float] = None  # If Actor provided context-dependent temperature


class HybridOptimizationFlow:
    """
    Orchestrates one full step:
    1. Get top-K prompts from population
    2. Generate K responses (one per prompt)
    3. Actor selects which to show
    4. (External) Receive human feedback
    5. Critic evaluates and updates
    6. Compute composite reward
    """

    def __init__(
        self,
        llm: LLMBackend,
        population: Population,
        actor: Actor,
        critic: Critic,
        reward_fn: Optional[HybridReward] = None,
        feedback_aggregator: Optional[FeedbackAggregator] = None,
        top_k_prompts: int = 5,
        max_tokens: int = 512,
        temperature: float = 0.7,
        parallel_eval: bool = False,
        max_workers: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.population = population
        self.actor = actor
        self.critic = critic
        self.reward_fn = reward_fn or HybridReward()
        self.feedback_aggregator = feedback_aggregator or FeedbackAggregator()
        self.top_k_prompts = top_k_prompts
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.parallel_eval = parallel_eval
        self.max_workers = max_workers

    def get_prompt_texts(self) -> list[str]:
        """Top-K prompt texts from population (by fitness)."""
        return self.population.get_prompt_texts(top_k=self.top_k_prompts)

    def step(
        self,
        query: str,
        human_feedback: float,
        coherence: float = 0.0,
        tokens_used: float = 0.0,
        safety_penalty: float = 0.0,
        **kwargs: Any,
    ) -> FlowStepResult:
        """
        Performs one step: generate candidates, Actor chooses, Critic scores, reward.
        human_feedback: aggregated user score [0,1] (thumbs, rating, etc.).
        """
        prompts = self.get_prompt_texts()
        if not prompts:
            raise RuntimeError("Empty population or no prompts with fitness.")

        # Actor can learn context-dependent temperature; fallback to flow default
        temperature = get_actor_temperature(
            self.actor, query, context=kwargs, default=self.temperature
        )
        if temperature is None:
            temperature = self.temperature
        if self.parallel_eval:
            candidates = generate_candidates_parallel(
                self.llm,
                prompts,
                query,
                max_tokens=self.max_tokens,
                temperature=temperature,
                max_workers=self.max_workers,
                **kwargs,
            )
        else:
            candidates = generate_candidates(
                self.llm,
                prompts,
                query,
                max_tokens=self.max_tokens,
                temperature=temperature,
                **kwargs,
            )
        if not candidates:
            raise RuntimeError("No candidates generated.")

        idx = self.actor.select_response(query, candidates, context=kwargs)
        chosen = candidates[idx]
        response_shown = chosen.text

        critic_score = self.critic.score(
            chosen.system_prompt,
            query,
            response_shown,
            context={"candidates": candidates, **kwargs},
        )
        self.critic.update(
            chosen.system_prompt,
            query,
            response_shown,
            reward=human_feedback,
            context=kwargs,
        )

        info = {
            "human_feedback": human_feedback,
            "critic_score": critic_score,
            "response_coherence": coherence,
            "token_efficiency": tokens_used,
            "safety_penalty": safety_penalty,
        }
        reward_total = self.reward_fn(info)

        self.population.sort_by_fitness()
        ind_index = min(chosen.prompt_index, len(self.population.individuals) - 1)
        current_f = self.population.individuals[ind_index].fitness
        self.population.update_fitness(ind_index, 0.9 * current_f + 0.1 * reward_total)

        return FlowStepResult(
            query=query,
            system_prompt_used=chosen.system_prompt,
            prompt_index_used=chosen.prompt_index,
            response_shown=response_shown,
            candidates=candidates,
            feedback_score=human_feedback,
            critic_score=critic_score,
            reward_total=reward_total,
            info=info,
            temperature_used=temperature,
        )
