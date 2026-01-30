# prompt-rl Usage Guide

This guide explains how to install, configure, and use **prompt-rl** in a professional workflow. For architecture and theory, see [ARCHITECTURE.md](ARCHITECTURE.md), [PARAMETRIZATION.md](PARAMETRIZATION.md), and [KEY_INNOVATIONS.md](KEY_INNOVATIONS.md).

---

## Table of contents

1. [Overview](#1-overview)
2. [Installation](#2-installation)
3. [Quick start](#3-quick-start)
4. [Core concepts](#4-core-concepts)
5. [Configuration](#5-configuration)
6. [Workflow: phases and loop](#6-workflow-phases-and-loop)
7. [Examples walkthrough](#7-examples-walkthrough)
8. [API overview](#8-api-overview)
9. [Best practices](#9-best-practices)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Overview

**prompt-rl** is a reinforcement learning framework for **prompt refinement** with LLMs. It supports:

- **Standalone RL:** A single prompt refined step-by-step with a configurable reward (e.g. quality, length).
- **Hybrid system:** Evolutionary population of prompts + Actor-Critic + human feedback: the population evolves, the Actor selects which response to show, the Critic is trained on feedback, and GPRO updates both.

Use it when you want to **optimize system prompts** (or prompt strategies) using rewards from human feedback, automatic metrics, or a learned Critic.

---

## 2. Installation

### Requirements

- **Python** ≥ 3.9
- **pip** (or another PEP 517-compatible installer)

### From source (recommended for development)

```bash
git clone https://github.com/your-username/prompt-rl.git
cd prompt-rl
pip install -e .
```

### Optional dependencies

| Extra        | Purpose                          | Install                    |
|-------------|-----------------------------------|----------------------------|
| `openai`    | OpenAI API or local model (OpenAI-compatible backend) | `pip install -e ".[openai]"` |
| `gradio`    | Gradio UI for human feedback      | `pip install -e ".[gradio]"` |
| `dev`       | Tests, coverage, lint             | `pip install -e ".[dev]"` |

Install multiple: `pip install -e ".[openai,gradio,dev]"`.

### Verify installation

```bash
python -c "import prompt_rl; print(prompt_rl.__version__)"
```

---

## 3. Quick start

### Minimal: prompt and refinement loop

```python
from prompt_rl import Prompt, RefinementLoop
from prompt_rl.llm import MockLLM

llm = MockLLM(default_response="Improved prompt.")
initial = Prompt("Explain quantum computing in one paragraph.")

def refine(p: Prompt) -> Prompt:
    r = llm.refine_prompt(p, instruction="Make it clearer for beginners.")
    return p.with_refinement(r.text.strip())

loop = RefinementLoop(refine_fn=refine, max_steps=3)
history = loop.run(initial)
print(history.current.text)
```

### RL environment (single prompt, reward-driven)

```python
from prompt_rl.core.prompt import Prompt
from prompt_rl.rl import PromptRefinementEnv, ScalarReward

def refine(prompt: Prompt, action: str) -> Prompt:
    return prompt.with_refinement(action)

env = PromptRefinementEnv(
    initial_prompt=Prompt("Write a short product description."),
    refine_fn=refine,
    reward_fn=ScalarReward(key="score"),
    max_steps=5,
)
state = env.reset()
step = env.step("Write a 2-sentence product description.", info={"score": 0.8})
print(step.reward, step.observation.text)
```

### Full hybrid system (evolution + Actor-Critic + feedback)

See [Section 6](#6-workflow-phases-and-loop) and `examples/hybrid_system_example.py`.

### OpenAI API key or local model

Install `.[openai]` then choose one:

| Use case | Backend | Notes |
|----------|---------|--------|
| **OpenAI cloud** | `OpenAIBackend(model="gpt-4o-mini", api_key="sk-...")` | Set `OPENAI_API_KEY` or pass `api_key`. |
| **Local model** | `LocalLLMBackend(model="local-model")` | No API key. Default base URL: `http://localhost:1234/v1` (LM Studio). For Ollama: `base_url="http://localhost:11434/v1"`, `model="llama3.2"`. |

Both use the same `complete()` / `refine_prompt()` interface, so you can swap backends without changing the rest of your code.

---

## 4. Core concepts

| Concept | Description |
|--------|-------------|
| **Prompt** | Text + version + metadata; supports `with_refinement(new_text)`. |
| **PromptGenome** | Modular prompt (sections: system_role, tone, constraints, examples, format). |
| **Population** | N individuals (genome + fitness); elite, worst, crossover, mutation. |
| **Actor** | Selects which of K candidate responses to show; can output context-dependent temperature. |
| **Critic** | Scores (prompt, query, response); trained with human feedback. |
| **GPRO** | Optimization algorithm that updates Actor and Critic from batches of transitions. |
| **Human feedback** | Explicit (thumbs, ratings), implicit (time, re-prompts), preferences (A vs B). |
| **Composite reward** | R_total = λ₁·feedback + λ₂·critic + λ₃·coherence + λ₄·tokens + λ₅·safety. |

---

## 5. Configuration

All training parameters live in **`TrainingConfig`** and its nested dataclasses.

### Evolutionary parameters

```python
from prompt_rl.training.config import EvolutionaryParams

EvolutionaryParams(
    population_size=10,    # N individuals
    elite_size=2,         # top-K preserved each generation
    mutation_rate=0.3,    # α
    crossover_rate=0.5,   # β
    evolution_interval=5,  # evolve every E episodes
    min_improvement=0.01,  # early stop if improvement < this over last N gen (0 = off)
    patience=5,           # N generations for early stop
)
```

### Actor-Critic parameters

```python
from prompt_rl.training.config import ActorCriticParams

ActorCriticParams(
    num_candidates=5,     # K responses per query
    update_interval=10,   # GPRO update every M queries
    exploration=0.1,
    parallel_eval=False,  # set True to generate candidates in parallel
    max_workers=None,     # threads for parallel eval (None = min(K, 4))
)
```

### Reward weights (λᵢ)

```python
from prompt_rl.training.config import RewardWeights

RewardWeights(
    lambda_feedback=1.0,
    lambda_critic=0.8,
    lambda_coherence=0.3,
    lambda_tokens=-0.001,
    lambda_safety=-1.0,
)
```

### Full config

```python
from prompt_rl.training import TrainingConfig

config = TrainingConfig(
    evolutionary=EvolutionaryParams(...),
    actor_critic=ActorCriticParams(...),
    reward_weights=RewardWeights(...),
    bootstrap_queries=20,
    max_queries_per_phase=None,
)
```

See [PARAMETRIZATION.md](PARAMETRIZATION.md) for symbols (α, β, K, M, E, λᵢ) and [GPRO.md](GPRO.md) for the optimizer.

---

## 6. Workflow: phases and loop

The hybrid system runs in three phases.

### Phase 1: Initialization

- Build initial **population** from a list of **PromptGenome**s.
- Optionally run **bootstrap** queries and collect human feedback to seed the Critic.

```python
from prompt_rl.training import TrainingLoop, TrainingConfig, HybridOptimizationFlow
from prompt_rl.training.config import EvolutionaryParams, ActorCriticParams
from prompt_rl.evolution import PromptGenome, Population
from prompt_rl.evolution.population import Individual
from prompt_rl.actor_critic import RandomActor, MockCritic
from prompt_rl.llm import MockLLM
from prompt_rl.rl.rewards import HybridReward

config = TrainingConfig(
    evolutionary=EvolutionaryParams(population_size=8, elite_size=2),
    actor_critic=ActorCriticParams(num_candidates=3, update_interval=2),
)
llm = MockLLM()
population = Population(elite_size=2)
flow = HybridOptimizationFlow(
    llm=llm, population=population,
    actor=RandomActor(), critic=MockCritic(),
    reward_fn=HybridReward(), top_k_prompts=3,
)
loop = TrainingLoop(config=config, flow=flow)

genomes = [
    PromptGenome(sections={"system_role": "You are helpful.", "tone": "Formal."}),
    PromptGenome(sections={"system_role": "You are friendly.", "tone": "Casual."}),
    # ... more
]
loop.run_initialization(genomes, bootstrap_callback=None)
```

### Phase 2: Main loop

For each user query:

1. **Flow step:** Get top-K prompts → generate K responses → Actor selects one → you show it and collect **human feedback**.
2. **Call** `loop.step(query, human_feedback=score, ...)`.
3. Every **M** queries, GPRO updates Actor/Critic (if a `GPROOptimizer` is provided).
4. Every **E** episodes, evolution runs (mutate/crossover, replace worst); optional early stop if improvement &lt; `min_improvement` over `patience` generations.

```python
for query in user_queries:
    # In production: get human_feedback from your UI (e.g. Gradio) or reward model
    human_feedback = 0.7  # e.g. from HumanFeedbackCollector.get_feedback(...)
    result = loop.step(query, human_feedback=human_feedback)
    # result.response_shown, result.reward_total, result.critic_score, etc.
    if loop.evolution_stopped:
        break
```

### Phase 3: Refinement

Optional: prune duplicates, drop individuals below a fitness threshold.

```python
loop.run_refinement(prune_duplicates=True, min_fitness_threshold=0.1)
```

### Metrics

Attach a **MetricsCollector** to monitor fitness over time, variance between prompts, user satisfaction, convergence, and diversity:

```python
from prompt_rl.training import MetricsCollector

collector = MetricsCollector(satisfaction_window=50, convergence_threshold=0.85)
loop = TrainingLoop(config=config, flow=flow, metrics_collector=collector)
# ... run loop ...
m = collector.get_metrics()
print(m.user_satisfaction, m.convergence_slope, m.diversity_per_generation)
```

See [METRICS.md](METRICS.md).

---

## 7. Examples walkthrough

| Example | What it does |
|--------|----------------|
| **`examples/basic_refinement.py`** | Single prompt + RefinementLoop + MockLLM. |
| **`examples/rl_env_example.py`** | PromptRefinementEnv + ScalarReward; simulates steps with a score in `info`. |
| **`examples/hybrid_system_example.py`** | Full hybrid: init → loop.step with simulated feedback → refinement; uses MockLLM, RandomActor, MockCritic. |
| **`examples/gradio_feedback_example.py`** | Gradio UI: standalone feedback form or collector that blocks until user submits. |

Run from the repo root:

```bash
pip install -e .
python examples/basic_refinement.py
python examples/hybrid_system_example.py
python examples/gradio_feedback_example.py          # standalone UI
python examples/gradio_feedback_example.py collector  # collector demo
```

---

## 8. API overview

### Package layout

```
prompt_rl
├── core          Prompt, PromptHistory, RefinementLoop
├── rl            PromptRefinementEnv, ScalarReward, HybridReward, Policy
├── llm           LLMBackend, LLMResponse, MockLLM, OpenAIBackend, LocalLLMBackend (optional)
├── evolution     PromptGenome, Population, mutate_genome, crossover_genomes, guided_mutation
├── actor_critic  Actor, Critic, RandomActor, MockCritic, generate_candidates, generate_candidates_parallel
├── feedback      HumanFeedback, FeedbackAggregator, Gradio: create_feedback_interface, HumanFeedbackCollector
├── training      TrainingConfig, TrainingLoop, HybridOptimizationFlow, GPROOptimizer, MetricsCollector
└── utils         map_parallel, run_parallel, score_batch_parallel
```

### Key imports

```python
from prompt_rl import Prompt, RefinementLoop
from prompt_rl.core.prompt import PromptHistory
from prompt_rl.rl import PromptRefinementEnv, ScalarReward, HybridReward
from prompt_rl.llm import MockLLM, OpenAIBackend, LocalLLMBackend
from prompt_rl.evolution import PromptGenome, Population, mutate_genome, crossover_genomes
from prompt_rl.actor_critic import Actor, Critic, RandomActor, MockCritic, generate_candidates_parallel
from prompt_rl.training import (
    TrainingConfig, TrainingLoop, HybridOptimizationFlow,
    GPROOptimizer, GPROTransition, MetricsCollector, FrameworkMetrics,
)
from prompt_rl.training.config import EvolutionaryParams, ActorCriticParams, RewardWeights
from prompt_rl.utils import map_parallel, score_batch_parallel
```

---

## 9. Best practices

1. **Start with mocks:** Use `MockLLM`, `RandomActor`, `MockCritic` to validate the pipeline before wiring real APIs and models.
2. **Tune evolution:** Use `min_improvement` and `patience` to stop evolution when progress plateaus; monitor `loop.evolution_stopped`.
3. **Monitor diversity:** Track `Population.diversity_score()` and `get_metrics().diversity_per_generation` so the population does not collapse.
4. **Human feedback:** Use the Gradio UI or your own collector so every shown response gets a feedback signal for the Critic and R_total.
5. **Parallel eval:** Set `parallel_eval=True` and `max_workers` when K is large or LLM latency is high (see [PARALLEL_EVAL.md](PARALLEL_EVAL.md)).
6. **Metrics:** Always attach a `MetricsCollector` when running the full loop; use `get_metrics()` and `get_fitness_over_time()` for logging and plots.
7. **Config in one place:** Build a single `TrainingConfig` (and optional `GPROOptimizer`, `MetricsCollector`) and pass them into `TrainingLoop` and `HybridOptimizationFlow`.

---

## 10. Troubleshooting

| Issue | What to check |
|-------|----------------|
| **Empty population / no candidates** | Ensure population has individuals with fitness (run bootstrap or assign initial fitness). Ensure `top_k_prompts` ≤ population size. |
| **Evolution never runs** | `evolution_interval` E must be reached (every E episodes, where an episode is every M queries). Check `loop.episode_count` and `config.evolutionary.evolution_interval`. |
| **Evolution stops too early** | Reduce `min_improvement` or increase `patience`; or set `min_improvement=0` to disable early stopping. |
| **Low diversity** | Increase `mutation_rate` or `crossover_rate`; use `guided_mutation` with hints; check `Population.diversity_score()` and avoid over-pruning in refinement. |
| **GPRO not updating** | Pass a non–no-op `GPROOptimizer` to `TrainingLoop` and implement `update(batch, actor, critic)`. |
| **Gradio import error** | Install with `pip install -e ".[gradio]"`. |
| **OpenAI / LLM errors** | Install `.[openai]`. For cloud: set `OPENAI_API_KEY` or pass `api_key` to `OpenAIBackend`. For local: use `LocalLLMBackend` with `base_url` (e.g. LM Studio `http://localhost:1234/v1`, Ollama `http://localhost:11434/v1`) and no API key. |

For more detail on architecture, parameters, and metrics, use the [docs](README.md) index.
