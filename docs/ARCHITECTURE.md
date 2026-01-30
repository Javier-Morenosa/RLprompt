# Hybrid System Architecture

Combined system: **Evolutionary Algorithm** + **Actor-Critic** + **Human Feedback Loop** for optimizing system prompts with LLMs.

## Flow diagram

```
┌─────────────────────────────────────────────────┐
│  PROMPT POPULATION (Evolutionary)                │
│  [P1, P2, ..., PN]  (modular genome)             │
└──────────────┬──────────────────────────────────┘
               │
               ▼
      ┌────────────────────┐
      │  User Query        │
      └────────┬───────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  ACTOR: Generation        │
    │  - Apply each Pi (top-K)  │
    │  - Generate responses Ri │
    │  - Select best R*         │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  Show R* to human         │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  HUMAN FEEDBACK           │
    │  - Rating / Preference    │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  CRITIC: Update            │
    │  - Evaluate (Pi, Ri, fb)  │
    │  - Update estimator       │
    └──────────┬───────────────┘
               │
               ▼
    ┌──────────────────────────┐
    │  GUIDED EVOLUTION         │
    │  - Fitness = scores       │
    │  - Mutate / Crossover      │
    │  - New generation         │
    └──────────────────────────┘
```

## Components

### 1. Evolutionary (`evolution/`)

- **PromptGenome**: modular genome (sections: `system_role`, `tone`, `constraints`, `examples`, `format`).
- **Population**: N individuals (genome + fitness). Elite, worst, replacement.
- **Operators**: `mutate_genome`, `crossover_genomes`, `guided_mutation` (direction suggested by Actor-Critic).

### 2. Actor-Critic (`actor_critic/`)

- **Actor**: given (query, candidates), selects which to show. `Actor_θ(prompt, context, candidate_responses) → index`.
- **Critic**: `Critic_θ(prompt, query, response) → expected_score`. Updates with real feedback.
- **generate_candidates**: for each system prompt (top-K), generates one response with the LLM.

### 3. Human Feedback (`feedback/`)

- **Explicit**: thumbs up/down, ratings, corrections → score [0,1].
- **Implicit**: reading time, re-prompts, acceptance.
- **Preference**: A vs B comparisons.
- **FeedbackAggregator**: combines signals into a single score.

### 4. Composite reward (`rl/rewards.py`)

```
R_total = λ1 * human_feedback + λ2 * critic_score + λ3 * response_coherence
          + λ4 * token_efficiency + λ5 * safety_penalty
```

The λi are configurable (`RewardWeights` in `training/config.py`).

### 5. Training (`training/`)

- **TrainingConfig**: evolutionary params (α, β, elite, E), Actor-Critic (K, M), λi weights, bootstrap.
- **HybridOptimizationFlow**: one step: query → generate → Actor chooses → feedback → Critic → reward → update fitness.
- **TrainingLoop**:
  - **Phase 1 (Init)**: initial population, bootstrap with feedback.
  - **Phase 2 (Main)**: query loop, every M update Actor-Critic, every E evolve population.
  - **Phase 3 (Refinement)**: pruning, fitness threshold, duplicate removal.
- **When to stop evolution:** optional early stopping when best fitness improves by less than `min_improvement` over the last `patience` generations (`EvolutionaryParams.min_improvement`, `EvolutionaryParams.patience`). Check `TrainingLoop.evolution_stopped`.

## Quick use

See `examples/hybrid_system_example.py`: initialization, main loop and refinement with MockLLM and test Actor/Critic.

## Parametrization

- **Actor (θ_actor):** `Actor_θ(prompt, context, candidate_responses) → probabilities`. Learns which prompt/response to use; exploration vs exploitation. Config: `num_candidates` (K), `update_interval` (M), `exploration`.
- **Critic (θ_critic):** `Critic_θ(prompt, query, response) → expected_score`. Learns probability of positive feedback, quality, alignment. Trained with human feedback.
- **Evolutionary:** Population = { individuals, fitness (from Critic), mutation_rate α, crossover_rate β, elite top_K }. Config: `population_size` (N), `elite_size` (top_K), `mutation_rate` (α), `crossover_rate` (β), `evolution_interval` (E).
- **Reward:** `lambda_feedback`, `lambda_critic`, `lambda_coherence`, `lambda_tokens`, `lambda_safety`.

See [PARAMETRIZATION.md](PARAMETRIZATION.md) for full detail.
