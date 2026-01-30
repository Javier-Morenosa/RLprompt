# Documentation

You can add additional framework documentation here (API, tutorials, etc.).

## Architecture and parametrization

- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Hybrid system flow (Evolutionary + Actor-Critic + Human Feedback), components.
- **[PARAMETRIZATION.md](PARAMETRIZATION.md)**: Detailed parameters—Actor (θ_actor), Critic (θ_critic), evolutionary (α, β, elite top_K)—and their mapping to config/code.
- **[GPRO.md](GPRO.md)**: GPRO (Generalized Policy Optimization)—the algorithm that trains the Actor-Critic (batch updates every M queries).
- **[PARALLEL_EVAL.md](PARALLEL_EVAL.md)**: Parallelization of evaluations—parallel candidate generation and parallel Critic batch scoring.
- **[METRICS.md](METRICS.md)**: How to measure that the framework is working—fitness over time, variance between prompts, user satisfaction, convergence speed, diversity metrics.
- **[KEY_INNOVATIONS.md](KEY_INNOVATIONS.md)**: Four key innovations—guaranteed diversity, fast learning, scalability, intelligent exploration—and how the framework supports them.

## Modules

- **core**: `Prompt`, `PromptHistory`, `RefinementLoop`
- **rl**: `PromptRefinementEnv`, rewards (`ScalarReward`, `composite_reward`), policies
- **llm**: `LLMBackend`, `MockLLM`, `OpenAIBackend` (optional)
- **evolution**: `PromptGenome`, `Population` (including `diversity_score()`), `mutate_genome`, `crossover_genomes`, `guided_mutation`
