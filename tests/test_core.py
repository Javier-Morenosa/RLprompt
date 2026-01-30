"""Tests for the core module."""

import pytest
from prompt_rl.core.prompt import Prompt, PromptHistory
from prompt_rl.core.refiner import RefinementLoop


def test_prompt_creation() -> None:
    p = Prompt("Hola", version=0)
    assert p.text == "Hola"
    assert p.version == 0
    assert len(p) == 4


def test_prompt_with_refinement() -> None:
    p = Prompt("Hola")
    p2 = p.with_refinement("Hola mundo")
    assert p2.text == "Hola mundo"
    assert p2.version == 1
    assert p.version == 0


def test_prompt_invalid_version() -> None:
    with pytest.raises(ValueError):
        Prompt("x", version=-1)


def test_prompt_history() -> None:
    h = PromptHistory()
    assert h.current is None
    h.append(Prompt("A"))
    h.append(Prompt("B", version=1))
    assert len(h) == 2
    assert h.current is not None
    assert h.current.text == "B"


def test_refinement_loop() -> None:
    def refine(p: Prompt) -> Prompt:
        return p.with_refinement(p.text + ".")

    loop = RefinementLoop(refine_fn=refine, max_steps=3)
    initial = Prompt("Test")
    history = loop.run(initial)
    assert len(history.prompts) == 3
    assert history.current is not None
    assert history.current.text == "Test.."
