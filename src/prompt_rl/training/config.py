"""Hybrid system parametrization: Actor (θ_actor), Critic (θ_critic), evolutionary."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class EvolutionaryParams:
    """
    Evolutionary population parameters.

    Population = { individuals (prompt_1..prompt_N), fitness (from Critic),
    mutation_rate α, crossover_rate β, elite top_K }.
    """

    population_size: int = 10  # N: number of individuals
    elite_size: int = 2       # top_K: best prompts preserved each generation
    mutation_rate: float = 0.3   # α: per-section mutation probability
    crossover_rate: float = 0.5  # β: probability to take from parent A in crossover
    evolution_interval: int = 5   # E: run evolution every E episodes
    # Early stopping: stop evolution when best fitness improves by less than
    # min_improvement over the last patience generations
    min_improvement: float = 0.0   # threshold; 0 = disabled
    patience: int = 5             # N generations to look back


@dataclass
class ActorCriticParams:
    """
    Actor-Critic parameters (θ_actor, θ_critic).

    Actor_θ(prompt, context, candidate_responses) → probabilities:
      - Which prompt/response to use per query; exploration vs exploitation.
    Critic_θ(prompt, query, response) → expected_score:
      - Probability of positive feedback; quality; alignment.
    """

    num_candidates: int = 5   # K: candidate responses per query
    update_interval: int = 10  # M: update Actor/Critic every M queries
    exploration: float = 0.1   # Exploration factor (e.g. ε or temperature)
    # Parallel evaluation: run LLM generation and/or Critic scoring in parallel
    parallel_eval: bool = False
    max_workers: Optional[int] = None  # None = min(K, 4) for generation


@dataclass
class RewardWeights:
    """λi for composite reward R_total."""

    lambda_feedback: float = 1.0
    lambda_critic: float = 0.8
    lambda_coherence: float = 0.3
    lambda_tokens: float = -0.001
    lambda_safety: float = -1.0


@dataclass
class TrainingConfig:
    """Full training configuration."""

    evolutionary: EvolutionaryParams = field(default_factory=EvolutionaryParams)
    actor_critic: ActorCriticParams = field(default_factory=ActorCriticParams)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_queries_per_phase: Optional[int] = None
    bootstrap_queries: int = 20
