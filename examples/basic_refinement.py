"""
Basic example: prompt refinement with loop and mock LLM.
"""

from prompt_rl import Prompt, RefinementLoop
from prompt_rl.llm import MockLLM


def main() -> None:
    initial = Prompt("Explain what photosynthesis is.")

    llm = MockLLM(default_response="Explain clearly and concisely what photosynthesis is.")

    def refine(prompt: Prompt) -> Prompt:
        response = llm.refine_prompt(
            prompt,
            instruction="Improve the clarity of the prompt without changing its goal.",
        )
        return prompt.with_refinement(response.text.strip())

    loop = RefinementLoop(refine_fn=refine, max_steps=3)
    history = loop.run(initial)

    print("Refinement history:")
    for i, p in enumerate(history.prompts):
        print(f"  v{p.version}: {p.text[:80]}..." if len(p.text) > 80 else f"  v{p.version}: {p.text}")
    print(f"\nFinal version: {history.current}")


if __name__ == "__main__":
    main()
