# RLPrompt



<img width="1572" height="512" alt="Group 3" src="https://github.com/user-attachments/assets/6a328fd7-abb0-4788-9153-e19851a412db" />





Reinforcement learning framework for **prompt refinement** with LLMs (Large Language Models).
## Description

`RLPrompt` provides the structure for:

- **Representing prompts** with history and metadata
- **Integrating LLM backends** (OpenAI API with API key, local model via OpenAI-compatible server, or mock for development)
- **Modeling refinement as an RL environment**: state = current prompt, action = refinement, reward = quality/configurable
- **Refinement loops** orchestrated by RL policies (agents)

Ideal for researching and automating prompt optimization via RL (reward by output quality, length, etc.).

## Requirements

- Python >= 3.9

## Installation

```bash
# Base installation (no API dependencies)
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With Gradio UI for human feedback
pip install -e ".[gradio]"

# Development (tests, lint)
pip install -e ".[dev]"
```

## Project structure

```
prompt_rl/
├── src/prompt_rl/
│   ├── core/           # Prompt, PromptHistory, RefinementLoop
│   ├── rl/             # PromptRefinementEnv, rewards (ScalarReward, HybridReward), policies
│   ├── llm/            # LLMBackend, MockLLM, OpenAIBackend, LocalLLMBackend
│   ├── evolution/       # Population, modular genome, mutation, crossover, guided injection
│   ├── actor_critic/   # Actor (selection), Critic (score), generate_candidates
│   ├── feedback/       # Human feedback: explicit, implicit, preferences; optional Gradio UI
│   └── training/       # TrainingLoop, HybridOptimizationFlow, config (λi, α, β, elite)
├── tests/
├── examples/
└── docs/
```

### Hybrid system architecture

Combined **Evolutionary + Actor-Critic + Human Feedback** system:

- **Population** of N prompts (modular genome: tone, constraints, examples).
- **Actor**: generates K responses (one per prompt), selects which to show.
- **Critic**: evaluates (prompt, query, response) → score; updates with feedback.
- **Human Feedback**: explicit (thumbs, ratings), implicit (time, re-prompts), preferences (A vs B).
- **Composite reward**: `R_total = λ1·feedback + λ2·critic + λ3·coherence + λ4·tokens + λ5·safety`.
- **Guided evolution**: Critic fitness, elite, mutation/crossover every E episodes.

**GPRO (Generalized Policy Optimization)** is the algorithm that trains the Actor-Critic: every M queries the training loop passes a batch of transitions to a `GPROOptimizer` (see [docs/GPRO.md](docs/GPRO.md) and `prompt_rl.training.gpro`).

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and `examples/hybrid_system_example.py`.

### Key innovations

- **Guaranteed diversity:** Evolutionary population keeps N variants (modular genome, crossover/mutation); optional `Population.diversity_score()` and `prune_duplicates` in refinement.
- **Fast learning:** Actor-Critic adapts selection in real time; human feedback updates the Critic every M queries.
- **Scalability:** Human feedback trains the Critic; the Critic then scores thousands of prompt/response variants without more labels.
- **Intelligent exploration:** `guided_mutation(genome, direction_hint, strength)` uses Actor-driven or reward-based hints instead of purely random mutation.

See [docs/KEY_INNOVATIONS.md](docs/KEY_INNOVATIONS.md) for details.

## Quick start

### Basic prompt and refinement

```python
from prompt_rl import Prompt, RefinementLoop
from prompt_rl.core.prompt import PromptHistory

p = Prompt("Summarize the following text in one sentence.", version=0)
history = PromptHistory()
history.append(p)

# Manual refinement
p2 = p.with_refinement("Summarize the following text in exactly one sentence, without adding opinions.")
history.append(p2)
```

### RL environment + reward

```python
from prompt_rl.core.prompt import Prompt
from prompt_rl.rl import PromptRefinementEnv, ScalarReward

def refine(prompt: Prompt, action: str) -> Prompt:
    return prompt.with_refinement(action)

env = PromptRefinementEnv(
    initial_prompt=Prompt("Write a haiku."),
    refine_fn=refine,
    reward_fn=ScalarReward(key="score"),
    max_steps=5,
)

state = env.reset()
# action = policy.select_action(state)  # your RL policy
# step_result = env.step(action, info={"score": 0.8})
```

### Human feedback with Gradio (optional)

```bash
pip install prompt-rl[gradio]
```

```python
# Standalone: launch app; user enters query/response and submits ratings
from prompt_rl.feedback import launch_standalone
launch_standalone(server_port=7860)

# Integrated: training loop blocks until user submits feedback
from prompt_rl.feedback import HumanFeedbackCollector
collector = HumanFeedbackCollector(server_port=7860)
collector.start()
result = collector.get_feedback(query="...", response="...")
score = result.score  # use in training (e.g. human_feedback=score)
```

See `examples/gradio_feedback_example.py` (run as-is for standalone; `python gradio_feedback_example.py collector` for collector demo).

### Mock LLM (no API)

```python
from prompt_rl.llm import MockLLM, LLMResponse

llm = MockLLM(default_response="Refined prompt from model.")
resp = llm.complete("Improve this prompt.")
print(resp.text)
```

### Hybrid system (Evolutionary + Actor-Critic + Human Feedback)

```python
from prompt_rl.evolution import PromptGenome, Population
from prompt_rl.actor_critic import RandomActor, MockCritic
from prompt_rl.training import TrainingLoop, TrainingConfig, HybridOptimizationFlow
from prompt_rl.training.config import EvolutionaryParams, ActorCriticParams
from prompt_rl.llm import MockLLM
from prompt_rl.rl.rewards import HybridReward

# Config: population, elite, α, β, E, K, M, λi
config = TrainingConfig(
    evolutionary=EvolutionaryParams(population_size=10, elite_size=2, mutation_rate=0.3),
    actor_critic=ActorCriticParams(num_candidates=5, update_interval=10),
)
# Set up flow (llm, population, actor, critic, reward_fn) and TrainingLoop
# Phase 1: run_initialization(genomes, bootstrap_callback)
# Phase 2: loop.step(query, human_feedback, ...)
# Phase 3: loop.run_refinement(prune_duplicates=True)
```

See `examples/hybrid_system_example.py`.

### OpenAI API or local model (optional)

Install the `openai` extra to use the OpenAI cloud API or any OpenAI-compatible local server (LM Studio, Ollama, vLLM, etc.):

```bash
pip install prompt-rl[openai]
```

**OpenAI cloud (API key):**

```python
from prompt_rl.llm import OpenAIBackend

# Uses OPENAI_API_KEY env var if api_key not set
llm = OpenAIBackend(model="gpt-4o-mini", api_key="sk-...")
resp = llm.refine_prompt("Translate to English: Hello world.")
```

**Local model (no API key):**

Use a local server that exposes an OpenAI-compatible API (e.g. LM Studio on port 1234, Ollama with `ollama serve` and OpenAI compatibility):

```python
from prompt_rl.llm import LocalLLMBackend

# LM Studio default (http://localhost:1234/v1)
llm = LocalLLMBackend(model="local-model")
# Or custom base_url, e.g. Ollama: base_url="http://localhost:11434/v1", model="llama3.2"
resp = llm.refine_prompt("Translate to English: Hello world.")
```

## Tests

```bash
pytest tests/ -v
```

## License

MIT. See [LICENSE](LICENSE).

## Contributing

Contributions are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, code style, and how to submit pull requests.
