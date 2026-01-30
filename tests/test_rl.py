"""Tests for the RL module."""

from prompt_rl.core.prompt import Prompt
from prompt_rl.rl import PromptRefinementEnv, ScalarReward


def _refine(prompt: Prompt, action: str) -> Prompt:
    return prompt.with_refinement(action)


def test_env_reset_and_step() -> None:
    env = PromptRefinementEnv(
        initial_prompt=Prompt("Hi"),
        refine_fn=_refine,
        reward_fn=ScalarReward(key="score"),
        max_steps=2,
    )
    state = env.reset()
    assert state.text == "Hi"
    step = env.step("Hi there", info={"score": 1.0})
    assert step.reward == 1.0
    assert step.observation.text == "Hi there"
    assert step.done is False
    step2 = env.step("Hi there!", info={"score": 0.5})
    assert step2.done is True


def test_scalar_reward_default() -> None:
    r = ScalarReward(key="missing", default=-1.0)
    reward = r(Prompt("a"), Prompt("b"), {})
    assert reward == -1.0
