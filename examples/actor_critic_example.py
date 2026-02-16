import random
import sys

from prompt_rl.llm import MockLLM
from prompt_rl.actor_critic_loop import (
    ActorCriticConfig,
    LLMActor,
    LLMCritic,
    HumanMultiSelectFeedback,
    ActorCriticLoop,
)

try:
    from prompt_rl.llm import LocalLLMBackend
except ImportError:
    LocalLLMBackend = None


def main() -> None:
    use_mock = "--mock" in sys.argv
    if use_mock:
        llm = MockLLM(default_response="Sample response.")
        prompt_llm = response_llm = llm
        print("Using MockLLM")
    elif LocalLLMBackend:
        llm = LocalLLMBackend(model="gemma3:1b", base_url="http://localhost:11434/v1")
        prompt_llm = response_llm = llm
        print("Using Ollama gemma3:1b")
    else:
        print("Install prompt-rl[openai]. Using MockLLM.")
        llm = MockLLM(default_response="Sample response.")
        prompt_llm = response_llm = llm

    config = ActorCriticConfig(num_variations=5)
    actor = LLMActor(
        prompt_llm=prompt_llm,
        response_llm=response_llm,
        max_tokens=256,
        temperature=0.8,
    )
    critic = LLMCritic(
        llm=llm,
        max_tokens=256,
        temperature=0.3,
    )
    def mock_select_all_correct(q: str, resps: list[str]) -> list[int]:
        n = min(len(resps), random.randint(2, len(resps)))
        return random.sample(range(len(resps)), n)
    feedback = HumanMultiSelectFeedback(callback=mock_select_all_correct)
    loop = ActorCriticLoop(actor=actor, critic=critic, feedback=feedback, config=config)
    results = loop.run(
        queries=["What is machine learning?", "Explain neural networks in one sentence."],
        base_instruction="You are a helpful assistant.",
    )
    for r in results:
        print(f"Iteration {r['iteration']}, total_correct: {r['total_correct']}/{len(r['responses'])}, selected: {r['selected_indices']}")
        print(f"Refinement hint: {r['refinement_hint'][:200]}...")


if __name__ == "__main__":
    main()
