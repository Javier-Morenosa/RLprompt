"""Reward functions for prompt refinement."""

from __future__ import annotations

from typing import Any, Callable, Optional, Protocol

from prompt_rl.core.prompt import Prompt


class RewardFn(Protocol):
    """Protocol for reward functions."""

    def __call__(
        self,
        prev_prompt: Prompt,
        new_prompt: Prompt,
        info: dict[str, Any],
    ) -> float:
        ...


def scalar_reward(
    prev_prompt: Prompt,
    new_prompt: Prompt,
    info: dict[str, Any],
    key: str = "score",
    default: float = 0.0,
) -> float:
    """
    Scalar reward extracted from info[key].
    Useful when the LLM or an evaluator returns a score in info.
    """
    return float(info.get(key, default))


class ScalarReward:
    """Wrapper to use scalar_reward as a callable with fixed key/config."""

    def __init__(self, key: str = "score", default: float = 0.0) -> None:
        self.key = key
        self.default = default

    def __call__(
        self,
        prev_prompt: Prompt,
        new_prompt: Prompt,
        info: dict[str, Any],
    ) -> float:
        return scalar_reward(prev_prompt, new_prompt, info, self.key, self.default)


def length_penalty(
    prev_prompt: Prompt,
    new_prompt: Prompt,
    info: dict[str, Any],
    max_length: int = 2000,
    scale: float = -0.001,
) -> float:
    """
    Penalty for prompt length (avoid excessively long prompts).
    """
    length = len(new_prompt.text)
    if length <= max_length:
        return 0.0
    return scale * (length - max_length)


def composite_reward(
    prev_prompt: Prompt,
    new_prompt: Prompt,
    info: dict[str, Any],
    *rewards: Callable[[Prompt, Prompt, dict], float],
    weights: Optional[list[float]] = None,
) -> float:
    """Combines multiple reward functions with optional weights."""
    if weights is None:
        weights = [1.0] * len(rewards)
    if len(weights) != len(rewards):
        raise ValueError("weights must have the same length as rewards")
    total = 0.0
    for w, r in zip(weights, rewards):
        total += w * r(prev_prompt, new_prompt, info)
    return total


# --- Hybrid system composite reward (Actor-Critic + Evolutionary + Human Feedback) ---

def hybrid_reward(
    info: dict[str, Any],
    lambda_feedback: float = 1.0,
    lambda_critic: float = 0.8,
    lambda_coherence: float = 0.3,
    lambda_tokens: float = -0.001,
    lambda_safety: float = -1.0,
) -> float:
    """
    R_total = λ1 * human_feedback + λ2 * critic_score + λ3 * response_coherence
              + λ4 * token_efficiency + λ5 * safety_penalty

    info should contain (as available):
      - human_feedback: float [0,1] (direct user signal)
      - critic_score: float (Critic estimate)
      - response_coherence: float (automatic metric, e.g. length/consistency)
      - token_efficiency: int or float (tokens used; normalized as bonus if low)
      - safety_penalty: float (positive = violation, penalize)
    """
    feedback = float(info.get("human_feedback", 0.0))
    critic = float(info.get("critic_score", 0.0))
    coherence = float(info.get("response_coherence", 0.0))
    tokens_raw = info.get("token_efficiency", 0)
    tokens = float(tokens_raw) if tokens_raw is not None else 0.0
    token_bonus = 1.0 / (1.0 + tokens / 500.0) if tokens >= 0 else 0.0
    safety = float(info.get("safety_penalty", 0.0))

    return (
        lambda_feedback * feedback
        + lambda_critic * critic
        + lambda_coherence * coherence
        + lambda_tokens * token_bonus
        + lambda_safety * safety
    )


class HybridReward:
    """Wrapper with configurable λi for use in the training loop."""

    def __init__(
        self,
        lambda_feedback: float = 1.0,
        lambda_critic: float = 0.8,
        lambda_coherence: float = 0.3,
        lambda_tokens: float = -0.001,
        lambda_safety: float = -1.0,
    ) -> None:
        self.lambda_feedback = lambda_feedback
        self.lambda_critic = lambda_critic
        self.lambda_coherence = lambda_coherence
        self.lambda_tokens = lambda_tokens
        self.lambda_safety = lambda_safety

    def __call__(self, info: dict[str, Any]) -> float:
        return hybrid_reward(
            info,
            lambda_feedback=self.lambda_feedback,
            lambda_critic=self.lambda_critic,
            lambda_coherence=self.lambda_coherence,
            lambda_tokens=self.lambda_tokens,
            lambda_safety=self.lambda_safety,
        )
