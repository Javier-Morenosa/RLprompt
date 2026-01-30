"""Population of N prompt variants (individuals with genome)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from prompt_rl.evolution.genome import PromptGenome


@dataclass
class Individual:
    """Individual: genome + fitness (Critic score)."""

    genome: PromptGenome
    fitness: float = 0.0
    generation: int = 0

    def to_prompt_text(self, section_order: Optional[tuple[str, ...]] = None) -> str:
        return self.genome.to_prompt_text(section_order)


@dataclass
class Population:
    """
    Population of prompts for the evolutionary algorithm.
    Keeps N individuals, fitness from Critic, elite and operators.
    """

    individuals: list[Individual] = field(default_factory=list)
    generation: int = 0
    elite_size: int = 2

    def __len__(self) -> int:
        return len(self.individuals)

    def add(self, individual: Individual) -> None:
        individual.generation = self.generation
        self.individuals.append(individual)

    def sort_by_fitness(self, descending: bool = True) -> None:
        """Sorts individuals by fitness (best first if descending=True)."""
        self.individuals.sort(key=lambda x: x.fitness, reverse=descending)

    def elite(self, k: Optional[int] = None) -> list[Individual]:
        """Returns the top-k individuals (elite)."""
        k = k or self.elite_size
        self.sort_by_fitness()
        return self.individuals[: min(k, len(self.individuals))]

    def worst(self, k: int = 1) -> list[Individual]:
        """Returns the k worst individuals (candidates for replacement)."""
        self.sort_by_fitness()
        return self.individuals[-k:] if k <= len(self.individuals) else []

    def update_fitness(self, index: int, fitness: float) -> None:
        if 0 <= index < len(self.individuals):
            self.individuals[index].fitness = fitness

    def get_prompt_texts(
        self,
        section_order: Optional[tuple[str, ...]] = None,
        top_k: Optional[int] = None,
    ) -> list[str]:
        """List of prompt texts (optionally only top_k by fitness)."""
        if top_k is not None:
            self.sort_by_fitness()
            individuals = self.individuals[: top_k]
        else:
            individuals = self.individuals
        return [ind.to_prompt_text(section_order) for ind in individuals]

    def diversity_score(self, section_order: Optional[tuple[str, ...]] = None) -> float:
        """
        Simple diversity estimate: average pairwise Jaccard dissimilarity of
        prompt texts (1 - intersection/union of token sets). Higher = more variety.
        Returns 0 if fewer than 2 individuals.
        """
        texts = self.get_prompt_texts(section_order=section_order)
        if len(texts) < 2:
            return 0.0
        tokens_list = [set(t.split()) for t in texts]
        n = len(tokens_list)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                a, b = tokens_list[i], tokens_list[j]
                inter = len(a & b)
                union = len(a | b)
                if union > 0:
                    total += 1.0 - (inter / union)
                    count += 1
        return total / count if count else 0.0
