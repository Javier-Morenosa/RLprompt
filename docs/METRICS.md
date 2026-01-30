# How to Measure That the Framework Is Working

This document describes metrics you can use to assess that the hybrid system is improving over time: fitness over time, variance between prompts, user satisfaction, convergence speed, and population diversity.

---

## 1. Improvement in fitness over time

**What to measure:** Best fitness (and optionally mean fitness) per generation should trend upward as evolution and the Critic improve.

**How:**

- Use **`MetricsCollector`** and pass it to **`TrainingLoop(..., metrics_collector=collector)`**. After each evolution step, the collector records best, mean, and min fitness per generation.
- **`get_metrics().fitness_over_time`** is a list of **`FitnessOverTime`** (generation, best_fitness, mean_fitness, min_fitness).
- Plot **best_fitness** and **mean_fitness** vs generation; both should generally increase (with possible plateaus).

**Helper:** `loop.get_history()` gives per-step results; `metrics_collector.get_fitness_over_time()` returns `[(generation, best_fitness, variance)]` for quick plotting.

---

## 2. Reduction of variance between prompts

**What to measure:** Variance of fitness across the population. A decrease can indicate that prompts are becoming more consistently good; you may also want to avoid variance going to zero (loss of diversity).

**How:**

- **`variance_between_prompts(population)`** in `prompt_rl.training.metrics` returns the variance of fitness across individuals.
- **`MetricsCollector`** records this per generation; **`get_metrics().variance_between_prompts`** is a list of variances (one per generation).
- **`DiversityMetrics.fitness_variance`** and **`fitness_std`** from **`diversity_metrics(population)`** give the same notion for the current population.

**Interpretation:** Decreasing variance + increasing mean fitness suggests more uniform quality; if variance goes to zero, check diversity metrics (see below).

---

## 3. User satisfaction

**What to measure:** Mean human feedback (e.g. thumbs, ratings) over a sliding window. Higher values indicate that the system is showing responses that users like more.

**How:**

- **`FlowStepResult.feedback_score`** is the human feedback for each step.
- **`MetricsCollector`** keeps the last `satisfaction_window` steps (default 50); **`get_metrics().user_satisfaction`** is the mean of **`feedback_score`** over that window.
- **`get_metrics().user_satisfaction_window`** is the list of recent feedback scores you can use for your own plots or stats.

**Usage:** Create the collector with **`MetricsCollector(satisfaction_window=100)`** and monitor **`user_satisfaction`** over training; it should trend up if the framework is improving.

---

## 4. Speed of convergence

**What to measure:** How quickly fitness reaches a target (e.g. 90% of a known max) or how fast the best fitness is improving (slope).

**How:**

- **`generations_to_threshold(best_fitness_history, threshold)`** in `prompt_rl.training.metrics`: returns the first generation index at which best fitness ≥ threshold, or `None` if never reached.
- **`convergence_slope(best_fitness_history, last_n=5)`**: linear slope of best fitness over the last N generations; positive = improving.
- **`MetricsCollector(convergence_threshold=0.8, slope_window=5)`**: then **`get_metrics().convergence_generations`** is the generation at which the threshold was reached (if any), and **`get_metrics().convergence_slope`** is the recent slope.

**Usage:** Compare runs (e.g. different α, β, or population size) by **convergence_generations** (lower = faster) and **convergence_slope** (positive and larger = faster recent improvement).

---

## 5. Diversity of the population

**What to measure:** Whether the population keeps a variety of prompts (avoid collapse to a single strategy). Useful for “guaranteed diversity” and for detecting over-convergence.

**How:**

- **`Population.diversity_score()`** (in `prompt_rl.evolution.population`): average pairwise Jaccard dissimilarity of prompt texts (tokens). Higher = more variety; 0 = all identical.
- **`diversity_metrics(population)`** in `prompt_rl.training.metrics` returns **`DiversityMetrics`**:
  - **`diversity_score`**: same Jaccard-based score.
  - **`fitness_variance`**: variance of fitness across individuals.
  - **`fitness_std`**: standard deviation of fitness.
- **`MetricsCollector`** records **`DiversityMetrics`** after each evolution; **`get_metrics().diversity_per_generation`** is a list of these per generation.

**Interpretation:**

- **diversity_score** should stay above a minimum you choose (e.g. 0.2) so the population does not collapse.
- **fitness_variance** can decrease as quality improves, but **diversity_score** staying stable indicates you are keeping variety while improving.

---

## Summary table

| Metric | What it measures | Where to get it |
|--------|------------------|------------------|
| **Fitness over time** | Best / mean fitness per generation | `MetricsCollector` → `get_metrics().fitness_over_time`, `get_fitness_over_time()` |
| **Variance between prompts** | Variance of fitness across population | `variance_between_prompts(pop)`, `get_metrics().variance_between_prompts`, `DiversityMetrics.fitness_variance` |
| **User satisfaction** | Mean human feedback over window | `get_metrics().user_satisfaction`, `user_satisfaction_window` |
| **Convergence speed** | Generations to threshold, slope of best fitness | `generations_to_threshold()`, `convergence_slope()`, `get_metrics().convergence_generations`, `convergence_slope` |
| **Diversity** | Jaccard diversity, fitness variance/std | `Population.diversity_score()`, `diversity_metrics(pop)`, `get_metrics().diversity_per_generation` |

---

## Example: wiring and logging

```python
from prompt_rl.training import TrainingLoop, MetricsCollector, FrameworkMetrics

collector = MetricsCollector(
    satisfaction_window=50,
    convergence_threshold=0.85,
    slope_window=5,
)
loop = TrainingLoop(config=config, flow=flow, metrics_collector=collector)

# ... run loop ...

m = collector.get_metrics()
print("User satisfaction (recent):", m.user_satisfaction)
print("Convergence slope:", m.convergence_slope)
print("Generations to 0.85:", m.convergence_generations)
for gen, div in enumerate(m.diversity_per_generation):
    print(f"Gen {gen} diversity_score={div.diversity_score:.3f} fitness_var={div.fitness_variance:.3f}")
```
