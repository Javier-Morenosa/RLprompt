"""
Guided evolutionary component: population of prompts with modular genome.

- PromptGenome: modular structure (tone, constraints, examples)
- Population: N variants of the system prompt
- Operators: mutation, crossover, guided injection by Actor-Critic
"""

from prompt_rl.evolution.genome import PromptGenome
from prompt_rl.evolution.population import Population
from prompt_rl.evolution.operators import (
    mutate_genome,
    crossover_genomes,
    guided_mutation,
)

__all__ = [
    "PromptGenome",
    "Population",
    "mutate_genome",
    "crossover_genomes",
    "guided_mutation",
]
