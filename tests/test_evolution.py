"""Tests for the evolution module."""

import pytest
from prompt_rl.evolution import PromptGenome, Population, mutate_genome, crossover_genomes
from prompt_rl.evolution.population import Individual


def test_genome_to_prompt_text() -> None:
    g = PromptGenome(sections={"system_role": "You are an assistant.", "tone": "Formal."})
    text = g.to_prompt_text()
    assert "You are an assistant" in text
    assert "Formal" in text


def test_population_elite() -> None:
    pop = Population(elite_size=2)
    pop.add(Individual(PromptGenome(sections={"a": "1"}), fitness=0.5))
    pop.add(Individual(PromptGenome(sections={"a": "2"}), fitness=0.9))
    pop.add(Individual(PromptGenome(sections={"a": "3"}), fitness=0.1))
    elite = pop.elite(k=2)
    assert len(elite) == 2
    assert elite[0].fitness == 0.9
    assert elite[1].fitness == 0.5


def test_mutate_genome() -> None:
    g = PromptGenome(sections={"tone": "Formal.\nBrief."})
    g2 = mutate_genome(g, mutation_rate=0.0)
    assert g2.sections["tone"] == g.sections["tone"]
    g3 = mutate_genome(g, mutation_rate=1.0)
    assert g3.sections is not g.sections


def test_crossover_genomes() -> None:
    a = PromptGenome(sections={"x": "A", "y": "A"})
    b = PromptGenome(sections={"x": "B", "y": "B"})
    c = crossover_genomes(a, b, crossover_rate=1.0)
    assert c.get_section("x") == "A"
    assert c.get_section("y") == "A"
    c2 = crossover_genomes(a, b, crossover_rate=0.0)
    assert c2.get_section("x") == "B"
    assert c2.get_section("y") == "B"


def test_population_diversity_score() -> None:
    pop = Population()
    pop.add(Individual(PromptGenome(sections={"a": "hello world"}), fitness=0.5))
    pop.add(Individual(PromptGenome(sections={"a": "foo bar"}), fitness=0.5))
    d = pop.diversity_score()
    assert 0 <= d <= 1
    assert d > 0
    pop2 = Population()
    pop2.add(Individual(PromptGenome(sections={"a": "same text"}), fitness=0.0))
    assert pop2.diversity_score() == 0.0
