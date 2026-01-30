"""Evolutionary operators: mutation, crossover, guided injection."""

from __future__ import annotations

import random
from typing import Callable, Optional

from prompt_rl.evolution.genome import PromptGenome


def mutate_genome(
    genome: PromptGenome,
    mutation_rate: float = 0.3,
    section_mutation_fn: Optional[Callable[[str], str]] = None,
    rng: Optional[random.Random] = None,
) -> PromptGenome:
    """
    Mutation: modify genome sections (tone, constraints, examples).
    Each section is mutated with probability mutation_rate.
    """
    rng = rng or random.Random()
    result = genome.copy()
    default_mutate = section_mutation_fn or _default_section_mutate

    for key in list(result.sections.keys()):
        if rng.random() < mutation_rate and result.sections[key]:
            result.sections[key] = default_mutate(result.sections[key])
    return result


def _default_section_mutate(text: str) -> str:
    """Default mutation: small perturbation (e.g. add/replace phrase)."""
    if not text.strip():
        return text
    lines = text.strip().split("\n")
    if lines and random.random() < 0.5:
        lines[-1] = lines[-1] + "."
    return "\n".join(lines)


def crossover_genomes(
    parent_a: PromptGenome,
    parent_b: PromptGenome,
    crossover_rate: float = 0.5,
    rng: Optional[random.Random] = None,
) -> PromptGenome:
    """
    Crossover: combine successful sections from two prompts.
    For each section, choose from parent_a or parent_b according to crossover_rate.
    """
    rng = rng or random.Random()
    sections = {}
    all_keys = set(parent_a.sections) | set(parent_b.sections)
    for key in all_keys:
        if rng.random() < crossover_rate:
            sections[key] = parent_a.get_section(key)
        else:
            sections[key] = parent_b.get_section(key)
    return PromptGenome(sections=sections)


def guided_mutation(
    genome: PromptGenome,
    direction_hint: str,
    strength: float = 0.5,
    mutation_fn: Optional[Callable[[PromptGenome, str, float], PromptGenome]] = None,
) -> PromptGenome:
    """
    Guided injection (intelligent exploration): Actor-Critic suggests mutation
    directions instead of purely random mutation. Use direction_hint from the
    Actor (e.g. policy gradient or preference) or from reward signals.
    direction_hint: text or descriptor of where to mutate (e.g. "more concise").
    strength: intensity of the guidance.
    mutation_fn: optional custom function (genome, direction_hint, strength) -> genome,
        e.g. one that uses Actor gradients or an LLM to apply the hint.
    """
    if mutation_fn is not None:
        return mutation_fn(genome, direction_hint, strength)
    result = genome.copy()
    key = "constraints"
    current = result.get_section(key)
    prefix = f"[Hint: {direction_hint}] "
    if strength > 0.5:
        result.set_section(key, (prefix + current).strip() if current else prefix.strip())
    return result
