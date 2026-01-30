"""
Hybrid system example: Evolutionary + Actor-Critic + Human Feedback.

Flow:
  1. Population of N prompts (modular genome)
  2. Per query: generate K responses (one per prompt), Actor selects one
  3. Simulate human feedback (in production: thumbs, rating, A/B)
  4. Critic scores and updates; composite reward R_total
  5. Every E episodes: evolution (elite, mutation, crossover)
"""

from prompt_rl.actor_critic import Actor, Critic, RandomActor, MockCritic
from prompt_rl.evolution import PromptGenome, Population
from prompt_rl.evolution.population import Individual
from prompt_rl.feedback import FeedbackAggregator
from prompt_rl.llm import MockLLM
from prompt_rl.rl.rewards import HybridReward
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

    llm = MockLLM(default_response="This is a sample response from the model.")
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
        max_tokens=64,
        temperature=0.7,
    )

    loop = TrainingLoop(config=config, flow=flow)

    # Phase 1: Initialization
    genomes = make_initial_genomes(5)
    def bootstrap_cb(query: str, responses: list[str], chosen_index: int) -> float:
        return 0.5 + 0.1 * chosen_index
    loop.run_initialization(genomes, bootstrap_callback=bootstrap_cb)

    print("Initial population:", len(flow.population.individuals))
    print("Top-3 prompt texts:", flow.get_prompt_texts())

    # Phase 2: Main loop (simulate queries and feedback)
    queries = ["What is AI?", "Explain in one sentence.", "Give an example."]
    for q in queries:
        result = loop.step(
            query=q,
            human_feedback=0.7,
            coherence=0.5,
            tokens_used=100.0,
            safety_penalty=0.0,
        )
        print(f"Query: {q[:40]}... → reward_total={result.reward_total:.3f}")

    print("Total queries:", loop.query_count)
    print("Episodes:", loop.episode_count)
    print("Population generation:", flow.population.generation)

    # Phase 3: Refinement (optional)
    loop.run_refinement(prune_duplicates=True, min_fitness_threshold=None)
    print("After refinement, individuals:", len(flow.population.individuals))


if __name__ == "__main__":
    main()
