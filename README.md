# RLPrompt

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/Javier-Morenosa/RLprompt)

<img width="1572" height="512" alt="Group 3" src="https://github.com/user-attachments/assets/6a328fd7-abb0-4788-9153-e19851a412db" />

Online Reinforcement Learning library for **system-prompt refinement** with human feedback, backed by a local LLM Critic.

---

## How it works

Every human interaction produces a **Perception Cycle** (stimulus → observation → verdict).
The **TwoStageCritic** (1.1 Backward + 1.2 Optimizer) uses full conversation context
to produce actionable feedback and a refined prompt.
**CriticValidationLoop** validates proposals by re-asking the same question before accepting.
A statistical **Update Gate** decides whether the proposed prompt replaces the live policy.

```
Human interaction
      │
      ▼
PerceptionCycle (system_prompt, user_query, bot_response, verdict, comment, observations)
      │
      ▼
TwoStageCritic  →  Critic 1.1 (Backward): feedback accionable con contexto
      │           →  Critic 1.2 (Optimizer): nuevo system prompt (BLIND a query/response)
      ▼
CriticValidationLoop  →  Re-pregunta al Actor con prompt propuesto
      │                →  Judge: ¿se solucionó? Si no: ciclo virtual, otra propuesta
      ▼
HybridReward   R = λ_fb·H + λ_c·C − λ_ch·word_change_ratio
      │
      ▼
RewardHistory  (rolling window, convergence tracking)
      │
      ▼
UpdateGate     degradation OR forced correction?
      │
      ├── YES → ActivePolicy.write()  →  system_prompt.md updated
      └── NO  → "Politica estable — sin actualizacion"
```

---

## Project structure

```
RLprompt/
│
├── src/prompt_rl/               ← Library (importable package)
│   ├── core/
│   │   ├── cycle.py             # PerceptionCycle — fundamental data unit
│   │   └── policy.py            # ActivePolicy — manages system_prompt.md
│   ├── llm/
│   │   ├── base.py              # LLMBackend ABC + LLMResponse
│   │   └── local_backend.py     # LocalLLMBackend (Ollama/Gemma)
│   ├── critic/
│   │   ├── base.py              # PerceptionCritic protocol + CriticOutput
│   │   ├── two_stage_critic.py  # TwoStageCritic (Backward + Optimizer)
│   │   └── llm_critic.py        # LLMPerceptionCritic (blind, legacy)
│   ├── feedback/
│   │   └── signals.py           # thumbs_to_score, reading_time_to_score, FeedbackAggregator
│   ├── rl/
│   │   ├── reward.py            # HybridReward + word_change_ratio
│   │   ├── history.py           # RewardHistory (rolling window + convergence)
│   │   └── gate.py              # UpdateGate (degradation / forced)
│   ├── population/
│   │   ├── genome.py            # PromptGenome (modular prompt sections)
│   │   └── leaderboard.py       # Leaderboard (fitness-ranked candidates)
│   └── loop/
│       └── online.py            # OnlineCriticLoop — ties everything together
│
├── demos/human_watch/           ← Human-Watch demo (encapsulated)
│   ├── server.py                # FastAPI chat server + RAG + feedback UI
│   ├── monitor.py               # Playwright perception monitor
│   ├── evaluator.py             # Critic subprocess (thin wrapper)
│   ├── run_backend.py           # Launcher: server + monitor
│   ├── run_server.py            # Server-only launcher
│   ├── reset_to_state_zero.py   # Reset state files
│   └── tests/
│       ├── test_flow.py         # Simulated flow test
│       └── test_monitor.py      # E2E integration test (14 checks)
│
├── examples/                    ← Library examples
│   ├── two_stage_example.py
│   └── validation_loop_example.py
│
├── data/                        ← Archivos de estado (.md, .json)
│   ├── system_prompt.md         ← Active Actor policy (hot-reloaded)
│   ├── interactions.md         ← Append-only perception cycle log
│   ├── reward_history.json     ← Rolling reward window + convergence state
│   ├── population.json        ← Fitness leaderboard (up to 20 entries)
│   ├── critic_memory.json
│   ├── critic_memory.md
│   ├── prompts/                ← Backups versionados del policy
│   └── logs/                   ← Archivos de interactions archivados
├── evaluator.log                ← Evaluator subprocess stdout/stderr
└── prompts/prompt_vN.md         ← Backups before each policy update
```

---

## Library

### Installation

```bash
pip install -e .                      # base (LocalLLMBackend para Ollama/Gemma)
pip install -e ".[dev]"               # + pytest, ruff
```

### Key types

| Type | Module | Description |
|---|---|---|
| `PerceptionCycle` | `core.cycle` | One feedback loop (system_prompt + verdict + comment + dwell) |
| `ActivePolicy` | `core.policy` | Reads / writes `system_prompt.md` with versioned backup |
| `PerceptionCritic` | `critic.base` | Protocol: `evaluate(cycle) -> CriticOutput` |
| `TwoStageCritic` | `critic.two_stage_critic` | Critic en dos etapas (Backward + Optimizer) |
| `CriticValidationLoop` | `validation.loop` | Valida propuestas antes de aceptar |
| `CriticOutput` | `critic.base` | `(critic_score, proposed_prompt, reasoning)` |
| `HybridReward` | `rl.reward` | `R = λ_fb·H + λ_c·C − λ_ch·change_ratio` |
| `RewardHistory` | `rl.history` | Rolling window + convergence state + persistence |
| `UpdateGate` | `rl.gate` | Fires on degradation or forced human correction |
| `AlwaysUpdateGate` | `rl.gate` | Updates every cycle |
| `OnlineCriticLoop` | `loop.online` | Orchestrates one full RL step per `PerceptionCycle` |
| `PromptGenome` | `population.genome` | Modular prompt with named sections |
| `Leaderboard` | `population.leaderboard` | Top-N candidates ranked by fitness |

### Quickstart

```python
from prompt_rl import (
    PerceptionCycle, ActivePolicy,
    TwoStageCritic, CriticValidationLoop, Actor, LLMValidationJudge,
    OnlineCriticLoop, RewardHistory, Leaderboard,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

backend = LocalLLMBackend(model="gemma3:4b")
critic = TwoStageCritic(backend=backend)
validated_critic = CriticValidationLoop(
    critic=critic,
    actor=Actor(backend=backend),
    judge=LLMValidationJudge(backend=backend),
    max_iterations=3,
)
policy  = ActivePolicy(path="system_prompt.md")
history = RewardHistory.from_file("reward_history.json")
lb      = Leaderboard.from_file("population.json")

loop = OnlineCriticLoop(
    critic=validated_critic,
    policy=policy,
    history=history,
    leaderboard=lb,
)

cycle = PerceptionCycle(
    system_prompt="Eres un asistente de negocio...",
    user_query="¿El plan Pro incluye IVA?",
    bot_response="No lo sé.",
    verdict="INCORRECTO",
    comment="Siempre incluye IVA del 21 %.",
    dwell_seconds=4.2,
)

result = loop.process_cycle(cycle)
loop.save_state("reward_history.json", "population.json")

print(result.gate.reason)     # "forced"
print(result.converged)       # False
```

### TwoStageCritic design

1. **Critic 1.1 (Backward)**: Full context (conversation + human feedback + cursor trace) → actionable feedback.
2. **Critic 1.2 (Optimizer)**: BLIND to user_query/bot_response — only system prompt + feedback → new prompt. Avoids overfitting.

**Optimizer options** (Critic 1.2):
- `gradient_memory`: N últimos feedbacks a incluir → evita repeticiones si el feedback se repite.
- `constraints`: restricciones en lenguaje natural (ej. `["Responde en español.", "Mantén el prompt conciso."]`).
- `in_context_examples`: ejemplos antes→después para guiar el optimizer.

```python
critic = TwoStageCritic(
    backend=backend,
    gradient_memory=3,
    constraints=["Responde siempre en español."],
    in_context_examples=["Antes: 'X' → Después: 'X mejorado porque...'"],
)
```

`CursorTrace.from_observations(cycle.observations)` parses [DWELL], [SELECT], [CLICK], [REVIEW_RAG].

### Convergence criterion

After **N consecutive stable cycles** (default N=5) where:
- `verdict == CORRECTO`, and
- `word_change_ratio(current, proposed) < ε` (default ε=0.05)

`RewardHistory.converged` becomes `True`. The monitor then skips
the evaluator subprocess, stopping further refinement automatically.
Convergence resets whenever the gate fires and the policy is updated.

---

## Human-Watch implementation

Human-Watch is the production runtime built on the library.
It closes the real feedback loop with a human operator in the loop.

### How to run

**Prerequisites:** Ollama with `gemma3:1b` + `gemma3:4b` pulled;
`pip install -e ".[human-watch]"`;
`playwright install chromium` once.

**Option A — Single command** (recommended):

```bash
# From project root: starts server + monitor
python -m demos.human_watch.run_backend
# or after pip install:
rlprompt-backend
```

**Option B — Separate terminals:**

```bash
# Terminal 1 — chat server
python -m demos.human_watch.run_server
# or: rlprompt-serve

# Terminal 2 — perception monitor (opens Chromium)
python -m demos.human_watch.monitor
```

### Perception Cycle (4 phases)

```
① Predictive Model   — active system_prompt.md
② System Action      — user query + Gemma 3:1b response
③ Observation Phase  — [DWELL] [SELECT] [REVIEW_RAG]
④ ACC Signal         — CORRECTO / INCORRECTO + optional comment
[RAW]                — click/cursor telemetry
```

### Dashboard

`http://localhost:8000/dashboard` — prompt version, reward history,
accuracy, top-3 fitness, last 5 human corrections.

### Tests de flujo

```bash
# From project root
# Test sin Playwright: simula eventos y verifica ciclo + evaluator
python -m demos.human_watch.tests.test_flow   # requiere servidor en :8000

# Test e2e con navegador (requiere: playwright install)
python -m demos.human_watch.tests.test_monitor   # requiere servidor en :8000
```

- **test_flow.py**: Verifica que texto → Incorrecto → evaluator se ejecute. No requiere Playwright.
- **test_monitor.py**: Test e2e completo con Playwright; valida 14 aserciones del ciclo en `interactions.md`.

---

## Reward formula

```
R_total = λ_feedback · H + λ_critic · C − λ_change · word_change_ratio

H  = human_feedback  [0, 1]   thumbs + dwell (FeedbackAggregator)
C  = critic_score    [0, 1]   TwoStageCritic output
λ_change             penalises large rewrites (keeps changes minimal)
```

Default weights: `λ_feedback=0.5, λ_critic=0.5, λ_change=0.3`.

### Update triggers

| Trigger | Condition |
|---|---|
| Degradation | `R_curr < R_avg × 0.8` (rolling window of 10) |
| Forced | `verdict == INCORRECTO` AND non-empty comment |
| Stable | Neither — policy unchanged |

### CriticValidationLoop (default)

When verdict is INCORRECTO, validate before accepting:

1. Critic proposes new system prompt
2. Re-ask the **same** question to the actor with the new prompt
3. ValidationJudge evaluates (with original feedback) whether the problem was fixed
4. If not: virtual cycle → Critic proposes again → repeat (up to `max_iterations`)

### AlwaysUpdateGate

To update on **every** cycle (no degradation/forced checks):

```python
from prompt_rl import OnlineCriticLoop, AlwaysUpdateGate, ...

loop = OnlineCriticLoop(
    critic=critic,
    policy=policy,
    gate=AlwaysUpdateGate(),
    ...
)
```

---

## Tests

```bash
# Library unit tests
pytest tests/ -v

# Human-Watch integration tests (requires server on :8000)
python -m demos.human_watch.tests.test_flow
python -m demos.human_watch.tests.test_monitor
```

---

## References

- [Expanding the Capabilities of Reinforcement Learning](https://arxiv.org/pdf/2602.02482)
- [Teaching Models to Teach Themselves](https://arxiv.org/pdf/2601.18778)

## License

MIT. See [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
