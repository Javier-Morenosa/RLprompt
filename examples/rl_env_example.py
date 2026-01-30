"""
Example: using the RL environment for prompt refinement.
Reward is taken from info["score"] (simulated).
"""

from prompt_rl.core.prompt import Prompt
from prompt_rl.rl import PromptRefinementEnv, ScalarReward


def refine_fn(prompt: Prompt, action: str) -> Prompt:
    """Action = new prompt text."""
    return prompt.with_refinement(action)


def main() -> None:
    initial = Prompt("Write a title for an article about AI.")

    env = PromptRefinementEnv(
        initial_prompt=initial,
        refine_fn=refine_fn,
        reward_fn=ScalarReward(key="score", default=0.0),
        max_steps=3,
    )

    state = env.reset()
    total_reward = 0.0

    actions = [
        "Generate a short, catchy title for an article about artificial intelligence.",
        "Generate a title of fewer than 10 words for an article about applied AI.",
    ]

    for i, action in enumerate(actions):
        info = {"score": 0.3 + i * 0.35}
        step = env.step(action, info=info)
        state = step.observation
        total_reward += step.reward
        print(f"Step {i + 1}: reward={step.reward:.2f}, done={step.done}")
        if step.done:
            break

    print(f"\nFinal prompt: {state.text}")
    print(f"Total reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
