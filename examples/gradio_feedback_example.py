import argparse
import sys

try:
    from prompt_rl.actor_critic_loop import (
        launch_integrated,
        ActorCriticConfig,
        LLMActor,
        LLMCritic,
    )
    from prompt_rl.llm import MockLLM
except ImportError:
    print("Install: pip install prompt-rl[gradio]")
    sys.exit(1)

try:
    from prompt_rl.llm import LocalLLMBackend
except ImportError:
    LocalLLMBackend = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Use MockLLM instead of real Ollama")
    parser.add_argument("--model", default="gemma3:1b", help="Ollama model name")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama API URL")
    parser.add_argument("--port", type=int, default=7863, help="Gradio server port")
    args = parser.parse_args()

    if args.mock:
        llm = MockLLM(default_response="Sample response.")
        prompt_llm = response_llm = llm
        print("Using MockLLM")
    elif LocalLLMBackend:
        llm = LocalLLMBackend(model=args.model, base_url=args.base_url)
        prompt_llm = response_llm = llm
        print(f"Using Ollama: {args.model}")
    else:
        llm = MockLLM(default_response="Sample response.")
        prompt_llm = response_llm = llm
        print("Using MockLLM (install prompt-rl[openai] for real LLM)")

    config = ActorCriticConfig(num_variations=10)
    actor = LLMActor(
        prompt_llm=prompt_llm,
        response_llm=response_llm,
        max_tokens=256,
        temperature=0.8,
    )
    critic = LLMCritic(llm=llm, max_tokens=256, temperature=0.3)
    launch_integrated(
        actor=actor,
        critic=critic,
        base_instruction="You are a helpful assistant.",
        num_variations=config.num_variations,
        server_name="127.0.0.1",
        server_port=args.port,
    )


if __name__ == "__main__":
    main()
