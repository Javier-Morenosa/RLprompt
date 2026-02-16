"""
Hybrid system with real local LLM (Ollama, LM Studio, etc.).

Requires:
  pip install prompt-rl[openai]
  Ollama running: ollama run llama3.2
  Or LM Studio with a model loaded on port 1234

Usage:
  python examples/hybrid_local_llm_example.py
  python examples/hybrid_local_llm_example.py --model mistral --base-url http://localhost:11434/v1
  python examples/hybrid_local_llm_example.py --model local-model --base-url http://localhost:1234/v1  # LM Studio
"""

import argparse
import sys

from prompt_rl.actor_critic import Actor, Critic, RandomActor, MockCritic
from prompt_rl.evolution import PromptGenome, Population
from prompt_rl.evolution.population import Individual
from prompt_rl.rl.rewards import HybridReward

try:
    from prompt_rl.llm import LocalLLMBackend
except ImportError:
    LocalLLMBackend = None
from prompt_rl.training import TrainingLoop, TrainingConfig, HybridOptimizationFlow
from prompt_rl.training.config import EvolutionaryParams, ActorCriticParams


def make_initial_genomes(n: int = 5) -> list[PromptGenome]:
    """Initial varied population (tone, constraints)."""
    templates = [
        {"system_role": "You are a helpful assistant.", "tone": "Formal.", "constraints": "Brief answers."},
        {"system_role": "You are a friendly assistant.", "tone": "Casual.", "constraints": "Max 2 paragraphs."},
        {"system_role": "You are a technical expert.", "tone": "Precise.", "constraints": "Include examples if applicable."},
        {"system_role": "Concise assistant.", "tone": "Neutral.", "constraints": "One sentence when possible."},
        {"system_role": "Educational assistant.", "tone": "Clear.", "constraints": "Explain step by step."},
    ]
    genomes = []
    for i in range(n):
        t = templates[i % len(templates)]
        g = PromptGenome(sections=dict(t))
        genomes.append(g)
    return genomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Actor-Critic with local LLM")
    parser.add_argument(
        "--model",
        default="llama3.2",
        help="Model name (Ollama: llama3.2, mistral, phi, etc.; LM Studio: local-model)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434/v1",
        help="API base URL (Ollama: 11434, LM Studio: 1234)",
    )
    parser.add_argument(
        "--queries",
        nargs="+",
        default=["What is AI?", "Explain machine learning in one sentence.", "Give a coding example."],
        help="Queries to run",
    )
    args = parser.parse_args()

    if LocalLLMBackend is None:
        print("Install OpenAI client for local API: pip install prompt-rl[openai]")
        sys.exit(1)
    llm = LocalLLMBackend(model=args.model, base_url=args.base_url)

    config = TrainingConfig(
        evolutionary=EvolutionaryParams(
            population_size=8,
            elite_size=2,
            mutation_rate=0.3,
            crossover_rate=0.5,
            evolution_interval=2,
        ),
        actor_critic=ActorCriticParams(num_candidates=3, update_interval=2),
        bootstrap_queries=2,
    )

    population = Population(elite_size=config.evolutionary.elite_size)
    actor: Actor = RandomActor()
    critic: Critic = MockCritic(default_score=0.5)
    reward_fn = HybridReward(
        lambda_feedback=1.0,
        lambda_critic=0.8,
        lambda_coherence=0.3,
    )
    flow = HybridOptimizationFlow(
        llm=llm,
        population=population,
        actor=actor,
        critic=critic,
        reward_fn=reward_fn,
        top_k_prompts=3,
        max_tokens=128,
        temperature=0.7,
    )

    loop = TrainingLoop(config=config, flow=flow)

    print(f"Using local LLM: {args.model} at {args.base_url}")
    print("Phase 1: Initialization...")

    def bootstrap_cb(query: str, responses: list[str], chosen_index: int) -> float:
        return 0.5 + 0.1 * chosen_index

    loop.run_initialization(make_initial_genomes(5), bootstrap_callback=bootstrap_cb)

    print("Initial population:", len(flow.population.individuals))
    print("Top-3 prompt texts:", flow.get_prompt_texts())
    print("\nPhase 2: Main loop (queries with real LLM responses)...")

    for q in args.queries:
        result = loop.step(
            query=q,
            human_feedback=0.7,
            coherence=0.5,
            tokens_used=100.0,
            safety_penalty=0.0,
        )
        resp_preview = result.response_shown[:60] + "..." if len(result.response_shown) > 60 else result.response_shown
        print(f"  Query: {q[:50]}...")
        print(f"  Response: {resp_preview}")
        print(f"  reward_total={result.reward_total:.3f}\n")

    print("Total queries:", loop.query_count)
    print("Episodes:", loop.episode_count)
    print("Population generation:", flow.population.generation)

    print("\nPhase 3: Refinement...")
    loop.run_refinement(prune_duplicates=True, min_fitness_threshold=None)
    print("After refinement, individuals:", len(flow.population.individuals))
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        err_str = str(e).lower()
        if "connection" in err_str or "connect" in err_str or "denied" in err_str:
            print("\n[!] No se pudo conectar al servidor LLM local.")
            print("    Inicia Ollama: ollama run llama3.2")
            print("    O LM Studio con un modelo cargado en puerto 1234")
        print(f"\nError: {e}")
        sys.exit(1)
