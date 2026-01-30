# Key Innovations of This Approach

This document describes the four main innovations of the hybrid evolutionary + Actor-Critic + human feedback system.

---

## 1. Guaranteed diversity

**Idea:** The evolutionary population keeps **variety** instead of collapsing to a single best prompt. Multiple prompt variants (different tone, constraints, examples) are maintained so the system can adapt to different query types and user preferences.

**How the framework supports it:**

- **Population of N individuals:** `Population` holds N prompt genomes; fitness is per individual, but selection preserves an **elite** and replaces only the worst. Crossover and mutation continuously introduce new variants.
- **Modular genome:** Each individual is a `PromptGenome` with sections (`system_role`, `tone`, `constraints`, `examples`, `format`). Diversity is natural across sections and combinations.
- **Diversity helpers:** `Population.diversity_score()` (optional) estimates variety (e.g. pairwise text dissimilarity). You can use it to reject low-diversity generations or to encourage exploration.
- **Refinement phase:** `run_refinement(prune_duplicates=True)` removes near-duplicates while keeping distinct strategies.

**Usage:** Keep `population_size` (e.g. 10–20) and use crossover + mutation every E episodes so the population does not converge to a single point.

---

## 2. Fast learning

**Idea:** **Actor-Critic** adapts **selection in real time**. The Actor learns which prompt to use for which query and which response to show; the Critic is updated from human feedback so that the system improves quickly with each batch of labels.

**How the framework supports it:**

- **Actor:** Chooses which of K candidate responses to show (`select_response(query, candidates)`). With a learned policy (e.g. neural), it can specialize by query type and user context.
- **Critic:** Estimates quality `score(prompt, query, response)` and is updated with `update(..., reward=human_feedback)`. It generalizes from limited human labels to score many (prompt, query, response) triples.
- **Update interval:** `ActorCriticParams.update_interval` (M) controls how often the policy and value function are updated (e.g. every M queries), so learning is tuned to your data rate.
- **Composite reward:** `human_feedback` is fed into `R_total` and into the Critic, so every feedback signal drives both the Critic and the evolutionary fitness.

**Usage:** Use a small M (e.g. 5–10) for fast adaptation; combine with human feedback (e.g. Gradio UI) so the Critic gets a steady stream of rewards.

---

## 3. Scalability

**Idea:** **Human feedback trains the Critic**; once trained, the **Critic can evaluate thousands of variants** without more human labels. You label a subset of (prompt, query, response), the Critic learns to predict quality, and then you use the Critic to score large populations and rank prompts.

**How the framework supports it:**

- **Critic as proxy:** Human feedback is the reward for the Critic. After enough feedback, `critic.score(prompt, query, response)` approximates human preference. You can call it for any prompt/query/response without asking a human.
- **Fitness at scale:** In `HybridOptimizationFlow` and `TrainingLoop`, fitness is updated from the Critic (and reward). Evolution uses these scores to select elite and replace worst individuals. The same Critic can score the full population (or a large sample) each generation.
- **Batch evaluation:** For large N, you can evaluate individuals in batches (e.g. score many prompts on a fixed set of queries) and assign fitness from mean Critic score. The framework does not assume that every individual is shown to a human.
- **Optional human loop:** You only need human feedback for the responses you actually show (Actor’s choice). The rest can be Critic-only for scalability.

**Usage:** Start with human feedback on a subset of traffic; once the Critic is reliable, use it to score all individuals every E episodes and drive evolution without increasing human load.

---

## 4. Intelligent exploration

**Idea:** **Mutations are guided by Actor (and Critic)**, not purely random. The Actor’s gradients or preferences (e.g. “more concise”, “more formal”) suggest **directions** for mutation, so evolution explores in promising regions instead of at random.

**How the framework supports it:**

- **Guided mutation:** `guided_mutation(genome, direction_hint, strength)` applies a **direction hint** (e.g. from the Actor or from a reward signal) to the genome, e.g. by adding a constraint or biasing a section. This is “intelligent” exploration: the hint encodes where to explore.
- **Actor-driven hint:** When using a learned Actor, you can derive a text or vector hint from the policy (e.g. “prefer shorter responses” or “more technical”) and pass it as `direction_hint`. Optionally, `guided_mutation(..., mutation_fn=custom_fn)` lets you use Actor gradients (e.g. embed gradient in a prompt for an LLM) to generate the mutated genome.
- **Evolution loop:** In `TrainingLoop._evolve_population()`, you can replace plain `mutate_genome(child_genome, ...)` with `guided_mutation(child_genome, direction_hint=actor.get_direction(query_context), ...)` so each mutation is biased by the current policy.
- **Strength:** The `strength` parameter controls how strongly the hint affects the genome; you can anneal it over generations (strong early, weak later) for refinement.

**Usage:** Implement an Actor that exposes a `get_direction()` (or similar) from its policy or from reward gradients, and call `guided_mutation(genome, direction_hint=..., strength=...)` in the evolution step. For full gradient-based mutation, provide a custom `mutation_fn` that uses an LLM or another model conditioned on the Actor’s gradient.

---

## Summary

| Innovation            | Mechanism in the framework                                      |
|-----------------------|-----------------------------------------------------------------|
| **Guaranteed diversity** | Population of N, modular genome, crossover/mutation, diversity_score, prune_duplicates |
| **Fast learning**       | Actor-Critic, update_interval M, human_feedback → Critic and R_total   |
| **Scalability**         | Critic trained on human feedback; score thousands of variants without new labels |
| **Intelligent exploration** | guided_mutation(direction_hint, strength), optional Actor-driven hint or custom mutation_fn |

These four properties are designed to work together: diversity keeps the population useful, fast learning adapts the Actor-Critic to users, scalability lets the Critic score the whole population, and intelligent exploration makes evolution more sample-efficient.
