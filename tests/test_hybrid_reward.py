"""Tests for the hybrid reward."""

from prompt_rl.rl.rewards import hybrid_reward, HybridReward


def test_hybrid_reward_basic() -> None:
    info = {
        "human_feedback": 0.8,
        "critic_score": 0.7,
        "response_coherence": 0.5,
        "token_efficiency": 100,
        "safety_penalty": 0.0,
    }
    r = hybrid_reward(info)
    assert r > 0


def test_hybrid_reward_class() -> None:
    fn = HybridReward(lambda_feedback=1.0, lambda_critic=0.5)
    r = fn({"human_feedback": 1.0, "critic_score": 0.0})
    assert r == 1.0
