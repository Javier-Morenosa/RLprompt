# Detailed Parametrization

This document describes the parameters of the hybrid system: Actor (θ_actor), Critic (θ_critic), the evolutionary population, and **GPRO** as the optimization algorithm that trains the Actor-Critic.

---

## Actor parameters (θ_actor)

The Actor is a policy (e.g. neural network or learned model) that maps context and candidate responses to a **selection** (or probabilities over candidates).

### Signature

```python
# Neural network or model that learns:
Actor_θ(prompt, context, candidate_responses) → probabilities
```

- **Inputs:** current prompt(s) in use, query/context, list of K candidate responses (one per prompt).
- **Output:** probabilities over candidates (or a single selected index). Used to choose which response to show to the user.

### What the Actor learns to optimize

- **Which prompt to use** depending on query type (e.g. formal vs casual, technical vs general).
- **Which response to select** to maximize expected human feedback (reward).
- **Exploration vs exploitation:** e.g. via temperature, ε-greedy, or entropy bonus in the policy loss.
- **Which temperature to use for LLM sampling:** context-dependent (e.g. higher for creative/open-ended queries, lower for factual).

### Context-dependent temperature (Actor-controlled)

The Actor can learn **which temperature to use** for generation depending on context. This is exposed as:

- **`get_temperature(query, context) → float | None`** on the Actor. If the Actor returns a value, the flow uses it when calling the LLM to generate candidates; otherwise the flow default (e.g. 0.7) is used.
- **Helper:** `get_actor_temperature(actor, query, context, default)` in `prompt_rl.actor_critic.actor` returns the Actor’s temperature or the default.
- **Flow:** `HybridOptimizationFlow.step()` calls `get_actor_temperature()` before `generate_candidates()`; the value used is stored in `FlowStepResult.temperature_used`.

A learned Actor can output temperature from the same policy (e.g. a second head or a learned scalar from context) and train it with the same reward signal so that high-reward contexts get appropriate temperature (e.g. low for factual, high for creative).

**Example (heuristic):** return low temperature for short/factual-looking queries, higher for long or open-ended ones:

```python
class MyActor:
    def get_temperature(self, query: str, context: Optional[dict] = None) -> Optional[float]:
        # e.g. short query → low temp; long/open-ended → higher
        if len(query.split()) < 5:
            return 0.3
        return 0.8
    # ... implement select_response, get_selection_probs
```

### Framework mapping

- **Protocol:** `Actor` in `prompt_rl.actor_critic.actor`:
  - `select_response(query, candidates, context) → int` (index of response to show).
  - `get_selection_probs(query, candidates, context) → list[float]` (for PPO/A2C).
  - `get_temperature(query, context) → float | None` (context-dependent temperature for generation).
- **Config:** `ActorCriticParams` in `prompt_rl.training.config`:
  - `num_candidates` (K): number of candidate responses per query.
  - `update_interval` (M): how often to update the policy (e.g. every M queries).
  - `exploration`: exploration factor (e.g. ε or temperature for sampling).

When implementing a learned Actor (e.g. neural), you would add your own θ_actor (weights, learning rate, etc.) and train it with **GPRO (Generalized Policy Optimization)**, which updates both the Actor and the Critic using batches of (query, candidates, chosen_index, reward). See [GPRO.md](GPRO.md).

---

## Critic parameters (θ_critic)

The Critic is a value estimator that predicts expected quality (or return) for a (prompt, query, response) triple.

### Signature

```python
# Value estimator
Critic_θ(prompt, query, response) → expected_score
```

- **Inputs:** system prompt used, user query, model response.
- **Output:** scalar expected score (e.g. probability of positive feedback, or estimated return).

### What the Critic learns to predict

- **Probability of positive feedback** (e.g. thumbs up, high rating).
- **Estimated response quality** (relevance, correctness, style).
- **Alignment with objectives** (safety, brevity, task success).

### Framework mapping

- **Protocol:** `Critic` in `prompt_rl.actor_critic.critic`:
  - `score(system_prompt, query, response, context) → float`.
  - `update(system_prompt, query, response, reward, context)`: update the estimator (e.g. gradient step).
- **Training:** Human feedback is the reward used in `update(...)`. The Critic is trained on (state, action, reward) where state = (prompt, query), action = response, reward = human_feedback (and optionally R_total).

When implementing a learned Critic (e.g. neural), you would add θ_critic (weights, learning rate) and train it with **GPRO**, which performs value updates (e.g. MSE on reward) on batches collected every M queries. See [GPRO.md](GPRO.md).

---

## Evolutionary parameters

The evolutionary component maintains a **population** of N prompt genomes and updates it using fitness (from the Critic) and genetic operators.

### Structure

```python
population = {
    'individuals': [prompt_1, ..., prompt_N],   # N prompt genomes
    'fitness': [score_1, ..., score_N],          # From Critic (and reward)
    'mutation_rate': α,                          # Per-section mutation probability
    'crossover_rate': β,                         # Probability to take from parent A
    'elite': top_K,                              # Number of best prompts preserved
}
```

- **individuals:** List of N `Individual` (each has a `PromptGenome` and a `fitness`).
- **fitness:** Scores come from the Critic (and composite reward) when a prompt is used; can be smoothed over time.
- **α (mutation_rate):** Probability of mutating each genome section (tone, constraints, etc.).
- **β (crossover_rate):** For each section, probability of taking from parent A vs B in crossover.
- **elite (top_K):** Number of best individuals kept unchanged each generation; the rest are replaced by offspring (crossover + mutation).

### Framework mapping

- **Population:** `Population` in `prompt_rl.evolution.population`:
  - `individuals`, `elite_size` (top_K), `generation`.
  - `elite(k)`, `worst(k)`, `update_fitness(index, fitness)`, `get_prompt_texts(top_k=...)`.
- **Config:** `EvolutionaryParams` in `prompt_rl.training.config`:
  - `population_size` (N).
  - `elite_size` (top_K).
  - `mutation_rate` (α).
  - `crossover_rate` (β).
  - `evolution_interval` (E): run evolution every E episodes.
  - `min_improvement`: early-stop threshold; stop evolution when best fitness improves by less than this over the last `patience` generations (0 = disabled).
  - `patience`: number of generations to look back; if (current_best - best_N_ago) < min_improvement, evolution is stopped (`TrainingLoop.evolution_stopped` = True).

---

## Summary table

| Symbol / concept | Config / code | Description |
|------------------|----------------|-------------|
| θ_actor | Actor protocol + your model | Policy: (prompt, context, candidates) → selection/probs |
| θ_critic | Critic protocol + your model | Value: (prompt, query, response) → expected score |
| K | `num_candidates` | Number of candidate responses per query |
| M | `update_interval` | Update Actor/Critic every M queries |
| exploration | `exploration` | Exploration factor (ε, temperature, etc.) |
| N | `population_size` | Number of individuals in the population |
| α | `mutation_rate` | Section mutation probability |
| β | `crossover_rate` | Crossover probability (parent A vs B) |
| top_K (elite) | `elite_size` | Number of best individuals preserved |
| E | `evolution_interval` | Evolve population every E episodes |
| min_improvement | `min_improvement` | Early stop: stop evolution if improvement over last N gen < this (0 = off) |
| patience | `patience` | N generations for early stop |

All of these are set via `TrainingConfig` (evolutionary, actor_critic, reward_weights) and used by `TrainingLoop` and `HybridOptimizationFlow`.

---

## GPRO: optimization algorithm

**GPRO (Generalized Policy Optimization)** is the algorithm that trains the Actor-Critic. It updates the Actor (policy) and the Critic (value estimator) using batches of transitions collected every **M** queries.

- **Interface:** `GPROOptimizer` in `prompt_rl.training.gpro` with `update(batch, actor, critic) → metrics`. The batch is a list of `GPROTransition` (query, system_prompt_used, response_shown, chosen_index, reward, candidates, context).
- **Training loop:** When `TrainingLoop` is constructed with `gpro_optimizer=my_optimizer`, every M queries it builds a batch from the last M `FlowStepResult`s and calls `gpro_optimizer.update(batch, flow.actor, flow.critic)`.
- **Placeholder:** `NoOpGPROOptimizer` does nothing; use it until you plug in a real GPRO implementation (e.g. policy gradient + MSE, or group-normalized advantages + clipped objective).

See [GPRO.md](GPRO.md) for full detail.
