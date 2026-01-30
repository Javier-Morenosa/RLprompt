"""
Training process in phases:

  Phase 1: Initialization (initial population, Actor/Critic, bootstrap with feedback)
  Phase 2: Main loop (generation, selection, feedback, update, evolution)
  Phase 3: Refinement (meta-learning, pruning, specialization)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, Optional

from prompt_rl.actor_critic.actor import generate_candidates
from prompt_rl.evolution import (
    Population,
    PromptGenome,
    mutate_genome,
    crossover_genomes,
)
from prompt_rl.evolution.population import Individual
from prompt_rl.training.config import TrainingConfig
from prompt_rl.training.flow import HybridOptimizationFlow, FlowStepResult
from prompt_rl.training.gpro import GPROOptimizer, GPROTransition
from prompt_rl.training.metrics import MetricsCollector


class TrainingPhase(Enum):
    INIT = "init"
    MAIN = "main"
    REFINEMENT = "refinement"


class TrainingLoop:
    """
    Training loop for the hybrid system.

    Phase 1: Create initial population, initialize Actor/Critic, collect bootstrap.
    Phase 2: Per iteration: queries → Actor selects → feedback → Critic → every E episodes evolve.
    Phase 3: Refinement (pruning, hyperparameters, subpopulations).
    """

    def __init__(
        self,
        config: TrainingConfig,
        flow: HybridOptimizationFlow,
        mutation_rate: Optional[float] = None,
        crossover_rate: Optional[float] = None,
        evolution_interval: Optional[int] = None,
        gpro_optimizer: Optional[GPROOptimizer] = None,
        metrics_collector: Optional[MetricsCollector] = None,
    ) -> None:
        self.config = config
        self.flow = flow
        self.gpro_optimizer = gpro_optimizer
        self.metrics_collector = metrics_collector
        self.ev_params = config.evolutionary
        if mutation_rate is not None:
            self.ev_params.mutation_rate = mutation_rate
        if crossover_rate is not None:
            self.ev_params.crossover_rate = crossover_rate
        if evolution_interval is not None:
            self.ev_params.evolution_interval = evolution_interval

        self._phase = TrainingPhase.INIT
        self._query_count = 0
        self._episode_count = 0
        self._history: list[FlowStepResult] = []
        self._best_fitness_per_generation: list[float] = []
        self._evolution_stopped = False

    @property
    def phase(self) -> TrainingPhase:
        return self._phase

    @property
    def query_count(self) -> int:
        return self._query_count

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def evolution_stopped(self) -> bool:
        """True if evolution was stopped due to minimal improvement over the last N generations."""
        return self._evolution_stopped

    def run_initialization(
        self,
        initial_genomes: list[PromptGenome],
        bootstrap_callback: Optional[Callable[[str, list[str], int], float]] = None,
    ) -> None:
        """
        Phase 1: Initial population from genomes; optionally collect bootstrap
        feedback with bootstrap_callback(query, responses, chosen_index) -> feedback_score.
        """
        self._phase = TrainingPhase.INIT
        self._evolution_stopped = False
        self._best_fitness_per_generation = []
        pop = self.flow.population
        pop.individuals.clear()
        pop.generation = 0
        for g in initial_genomes:
            pop.add(Individual(genome=g, fitness=0.0, generation=0))

        if bootstrap_callback and initial_genomes:
            # Simular algunas consultas para dar fitness inicial al Critic
            n_bootstrap = min(self.config.bootstrap_queries, 5)
            for _ in range(n_bootstrap):
                # Un paso simulado: el callback proporciona feedback
                query = "bootstrap_query_placeholder"
                prompts = self.flow.get_prompt_texts()
                if not prompts:
                    break
                candidates = generate_candidates(
                    self.flow.llm,
                    prompts[: self.flow.top_k_prompts],
                    query,
                    max_tokens=self.flow.max_tokens,
                    temperature=self.flow.temperature,
                )
                if not candidates:
                    break
                idx = self.flow.actor.select_response(query, candidates)
                fb = bootstrap_callback(query, [c.text for c in candidates], idx)
                self.flow.step(query, human_feedback=fb)

        self._phase = TrainingPhase.MAIN

    def step(
        self,
        query: str,
        human_feedback: float,
        coherence: float = 0.0,
        tokens_used: float = 0.0,
        safety_penalty: float = 0.0,
    ) -> FlowStepResult:
        """One step of the main loop: query + feedback → flow → optional evolution."""
        result = self.flow.step(
            query,
            human_feedback=human_feedback,
            coherence=coherence,
            tokens_used=tokens_used,
            safety_penalty=safety_penalty,
        )
        self._query_count += 1
        self._history.append(result)
        if self.metrics_collector is not None:
            self.metrics_collector.update_step(result)

        # Every M queries: GPRO updates Actor-Critic
        M = self.config.actor_critic.update_interval
        if M > 0 and self._query_count % M == 0:
            self._episode_count += 1
            if self.gpro_optimizer is not None and len(self._history) >= M:
                batch = [
                    GPROTransition(
                        query=r.query,
                        system_prompt_used=r.system_prompt_used,
                        response_shown=r.response_shown,
                        chosen_index=r.prompt_index_used,
                        reward=r.reward_total,
                        candidates=r.candidates,
                        context=r.info,
                    )
                    for r in self._history[-M:]
                ]
                self.gpro_optimizer.update(batch, self.flow.actor, self.flow.critic)

            # Every E episodes: guided evolution (skip if stopped by min-improvement criterion)
            E = self.ev_params.evolution_interval
            if E > 0 and self._episode_count % E == 0 and not self._evolution_stopped:
                self._maybe_evolve()

        return result

    def _maybe_evolve(self) -> None:
        """Runs evolution if min-improvement criterion is not met; otherwise sets evolution_stopped."""
        pop = self.flow.population
        if not pop.individuals:
            return
        pop.sort_by_fitness()
        current_best = pop.individuals[0].fitness
        patience = self.ev_params.patience
        min_improvement = self.ev_params.min_improvement
        if min_improvement > 0 and patience > 0 and len(self._best_fitness_per_generation) >= patience:
            best_n_ago = self._best_fitness_per_generation[-patience]
            if (current_best - best_n_ago) < min_improvement:
                self._evolution_stopped = True
                return
        self._evolve_population()
        self._best_fitness_per_generation.append(current_best)

    def _evolve_population(self) -> None:
        """Guided evolution: Critic fitness, elite, mutate/crossover, replace worst."""
        pop = self.flow.population
        elite = pop.elite(k=self.ev_params.elite_size)
        worst = pop.worst(k=len(pop) - len(elite))
        if not elite or not worst:
            return

        import random
        rng = random.Random()
        new_individuals: list[Individual] = list(elite)

        for _ in range(len(worst)):
            p1, p2 = rng.sample(elite, min(2, len(elite)))
            child_genome = crossover_genomes(
                p1.genome,
                p2.genome,
                crossover_rate=self.ev_params.crossover_rate,
                rng=rng,
            )
            child_genome = mutate_genome(
                child_genome,
                mutation_rate=self.ev_params.mutation_rate,
                rng=rng,
            )
            new_individuals.append(Individual(genome=child_genome, fitness=0.0, generation=pop.generation + 1))

        pop.individuals = new_individuals
        pop.generation += 1
        if self.metrics_collector is not None:
            self.metrics_collector.update_after_evolution(pop, pop.generation)

    def run_refinement(
        self,
        prune_duplicates: bool = True,
        min_fitness_threshold: Optional[float] = None,
    ) -> None:
        """
        Phase 3: Refinement.
        - prune_duplicates: remove very similar individuals
        - min_fitness_threshold: remove below fitness threshold
        """
        self._phase = TrainingPhase.REFINEMENT
        pop = self.flow.population
        pop.sort_by_fitness()

        if min_fitness_threshold is not None:
            pop.individuals = [i for i in pop.individuals if i.fitness >= min_fitness_threshold]

        if prune_duplicates and len(pop.individuals) > 2:
            seen_texts: set[str] = set()
            kept = []
            for ind in pop.individuals:
                t = ind.to_prompt_text()
                if t not in seen_texts:
                    seen_texts.add(t)
                    kept.append(ind)
            pop.individuals = kept

    def get_history(self) -> list[FlowStepResult]:
        return list(self._history)
