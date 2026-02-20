"""Leaderboard — ranked collection of prompt candidates by fitness."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from prompt_rl.population.genome import PromptGenome


@dataclass
class Individual:
    """One entry in the leaderboard: a genome with its cumulative fitness score."""

    genome:     PromptGenome
    fitness:    float = 0.0
    generation: int   = 0

    def to_text(self) -> str:
        return self.genome.to_text()


class Leaderboard:
    """
    Fixed-capacity ranked list of prompt candidates.

    Tracks the top-N system prompts by R_total fitness, giving a historical
    view of how the policy has evolved. No mutation or crossover is performed
    here — this is a pure fitness leaderboard, not an evolutionary engine.

    Serialises to / deserialises from population.json (same schema as before
    to remain compatible with existing runtime state files).
    """

    def __init__(self, capacity: int = 20) -> None:
        self.capacity:    int             = capacity
        self.individuals: list[Individual] = []
        self.generation:  int             = 0

    # ── Mutation ──────────────────────────────────────────────────────────────

    def add(self, individual: Individual) -> None:
        individual.generation = self.generation
        self.individuals.append(individual)
        if len(self.individuals) > self.capacity:
            self._sort()
            self.individuals = self.individuals[: self.capacity]
        self.generation += 1

    def _sort(self) -> None:
        self.individuals.sort(key=lambda x: x.fitness, reverse=True)

    # ── Queries ───────────────────────────────────────────────────────────────

    def elite(self, k: int = 1) -> list[Individual]:
        self._sort()
        return self.individuals[: min(k, len(self.individuals))]

    def __len__(self) -> int:
        return len(self.individuals)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        data = {
            "generation": self.generation,
            "elite_size": 2,  # kept for schema compat; unused here
            "individuals": [
                {
                    "sections":   ind.genome.sections,
                    "fitness":    ind.fitness,
                    "generation": ind.generation,
                }
                for ind in self.individuals
            ],
        }
        Path(path).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self.generation  = data.get("generation", 0)
            self.individuals = [
                Individual(
                    genome=PromptGenome(sections=d.get("sections", {})),
                    fitness=d.get("fitness", 0.0),
                    generation=d.get("generation", 0),
                )
                for d in data.get("individuals", [])
            ]
        except Exception:
            pass

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: object) -> "Leaderboard":
        lb = cls(**kwargs)  # type: ignore[arg-type]
        lb.load(path)
        return lb
