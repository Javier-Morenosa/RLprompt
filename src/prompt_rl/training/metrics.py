"""
Metrics to measure that the framework is working.

- Fitness over time (best, mean per generation)
- Variance between prompts (fitness variance across population)
- User satisfaction (mean human feedback over a window)
- Convergence speed (generations to threshold, or slope of best fitness)
- Diversity of the population (diversity_score, fitness variance, optional entropy)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from prompt_rl.evolution.population import Population
from prompt_rl.training.flow import FlowStepResult


@dataclass
class FitnessOverTime:
    """Fitness over time: one entry per generation or step."""

    generation: int
    best_fitness: float
    mean_fitness: float
    min_fitness: float = 0.0


@dataclass
class DiversityMetrics:
    """Diversity of the population."""

    diversity_score: float  # Jaccard-based (Population.diversity_score)
    fitness_variance: float  # Variance of fitness across individuals
    fitness_std: float = 0.0  # Std dev of fitness


@dataclass
class FrameworkMetrics:
    """Aggregated metrics to assess that the framework is working."""

    fitness_over_time: list[FitnessOverTime] = field(default_factory=list)
    variance_between_prompts: list[float] = field(default_factory=list)  # per generation
    user_satisfaction: float = 0.0  # mean human feedback over window
    user_satisfaction_window: list[float] = field(default_factory=list)
    convergence_generations: Optional[int] = None  # generations to reach threshold (if set)
    convergence_slope: Optional[float] = None  # slope of best_fitness over last N generations
    diversity_per_generation: list[DiversityMetrics] = field(default_factory=list)
    last_generation: int = 0


def variance_between_prompts(population: Population) -> float:
    """Variance of fitness across individuals. Lower can mean prompts are more aligned."""
    if not population.individuals:
        return 0.0
    fitnesses = [ind.fitness for ind in population.individuals]
    n = len(fitnesses)
    mean = sum(fitnesses) / n
    return sum((f - mean) ** 2 for f in fitnesses) / n


def diversity_metrics(population: Population) -> DiversityMetrics:
    """Diversity metrics for the current population."""
    if not population.individuals:
        return DiversityMetrics(diversity_score=0.0, fitness_variance=0.0, fitness_std=0.0)
    div = population.diversity_score()
    fitnesses = [ind.fitness for ind in population.individuals]
    n = len(fitnesses)
    mean = sum(fitnesses) / n
    var = sum((f - mean) ** 2 for f in fitnesses) / n if n > 0 else 0.0
    std = var ** 0.5
    return DiversityMetrics(diversity_score=div, fitness_variance=var, fitness_std=std)


def convergence_slope(best_fitness_history: list[float], last_n: int = 5) -> Optional[float]:
    """Linear slope of best fitness over the last N generations (positive = improving)."""
    if len(best_fitness_history) < 2 or last_n < 2:
        return None
    segment = best_fitness_history[-last_n:]
    n = len(segment)
    x_mean = (n - 1) / 2.0
    y_mean = sum(segment) / n
    num = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(segment))
    den = sum((i - x_mean) ** 2 for i in range(n))
    if den == 0:
        return None
    return num / den


def generations_to_threshold(
    best_fitness_history: list[float],
    threshold: float,
) -> Optional[int]:
    """First generation index at which best fitness >= threshold; None if never reached."""
    for gen, b in enumerate(best_fitness_history):
        if b >= threshold:
            return gen
    return None


class MetricsCollector:
    """
    Collects metrics from the training loop and population to measure
    that the framework is working.
    """

    def __init__(
        self,
        satisfaction_window: int = 50,
        convergence_threshold: Optional[float] = None,
        slope_window: int = 5,
    ) -> None:
        self.satisfaction_window = satisfaction_window
        self.convergence_threshold = convergence_threshold
        self.slope_window = slope_window
        self._history: list[FlowStepResult] = []
        self._best_fitness_per_generation: list[float] = []
        self._mean_fitness_per_generation: list[float] = []
        self._min_fitness_per_generation: list[float] = []
        self._variance_per_generation: list[float] = []
        self._diversity_per_generation: list[DiversityMetrics] = []
        self._last_generation_snapshot: int = -1

    def update_step(self, result: FlowStepResult) -> None:
        """Call after each flow step (query → feedback)."""
        self._history.append(result)

    def update_after_evolution(self, population: Population, generation: int) -> None:
        """Call after each evolution step to record population metrics."""
        if generation <= self._last_generation_snapshot:
            return
        self._last_generation_snapshot = generation
        population.sort_by_fitness()
        if not population.individuals:
            return
        fitnesses = [ind.fitness for ind in population.individuals]
        best = fitnesses[0]
        mean = sum(fitnesses) / len(fitnesses)
        min_f = min(fitnesses)
        self._best_fitness_per_generation.append(best)
        self._mean_fitness_per_generation.append(mean)
        self._min_fitness_per_generation.append(min_f)
        self._variance_per_generation.append(variance_between_prompts(population))
        self._diversity_per_generation.append(diversity_metrics(population))

    def get_metrics(self) -> FrameworkMetrics:
        """Returns current aggregated metrics."""
        fitness_over_time = []
        for gen in range(len(self._best_fitness_per_generation)):
            best = self._best_fitness_per_generation[gen]
            mean = self._mean_fitness_per_generation[gen] if gen < len(self._mean_fitness_per_generation) else best
            min_f = self._min_fitness_per_generation[gen] if gen < len(self._min_fitness_per_generation) else 0.0
            fitness_over_time.append(
                FitnessOverTime(generation=gen, best_fitness=best, mean_fitness=mean, min_fitness=min_f)
            )

        satisfaction_scores = [r.feedback_score for r in self._history[-self.satisfaction_window :]]
        user_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores) if satisfaction_scores else 0.0

        conv_gen = None
        if self.convergence_threshold is not None and self._best_fitness_per_generation:
            conv_gen = generations_to_threshold(self._best_fitness_per_generation, self.convergence_threshold)

        slope = convergence_slope(self._best_fitness_per_generation, self.slope_window)

        return FrameworkMetrics(
            fitness_over_time=fitness_over_time,
            variance_between_prompts=list(self._variance_per_generation),
            user_satisfaction=user_satisfaction,
            user_satisfaction_window=satisfaction_scores,
            convergence_generations=conv_gen,
            convergence_slope=slope,
            diversity_per_generation=list(self._diversity_per_generation),
            last_generation=len(self._best_fitness_per_generation) - 1 if self._best_fitness_per_generation else 0,
        )

    def get_fitness_over_time(self) -> list[tuple[int, float, float]]:
        """Returns [(generation, best_fitness, variance)] for plotting."""
        out = []
        for gen, best in enumerate(self._best_fitness_per_generation):
            var = self._variance_per_generation[gen] if gen < len(self._variance_per_generation) else 0.0
            out.append((gen, best, var))
        return out
