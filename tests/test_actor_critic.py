"""Tests for the actor_critic module."""

from prompt_rl.actor_critic import RandomActor, MockCritic
from prompt_rl.actor_critic.actor import CandidateResponse, generate_candidates
from prompt_rl.llm import MockLLM


def test_random_actor_select() -> None:
    actor = RandomActor()
    candidates = [
        CandidateResponse("A", "sys1", 0),
        CandidateResponse("B", "sys2", 1),
    ]
    idx = actor.select_response("query", candidates)
    assert 0 <= idx <= 1


def test_mock_critic_score() -> None:
    critic = MockCritic(default_score=0.5)
    s = critic.score("sys", "query", "response text here")
    assert 0.0 <= s <= 1.0


def test_generate_candidates() -> None:
    llm = MockLLM(default_response="OK")
    prompts = ["Sys A", "Sys B"]
    candidates = generate_candidates(llm, prompts, "Hello")
    assert len(candidates) == 2
    assert candidates[0].system_prompt == "Sys A"
    assert candidates[1].system_prompt == "Sys B"
