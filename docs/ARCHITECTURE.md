# Architecture

## Library vs Implementation

| Layer | Files | Responsibility |
|---|---|---|
| **Library** | `src/prompt_rl/` | Reusable RL logic — no I/O, no HTTP, no Playwright |
| **Implementation** | `demos/human_watch/` (server.py, monitor.py, evaluator.py) | Human-Watch runtime — specific to this deployment |

The implementation imports the library. The library knows nothing about HTTP,
browser automation, or Ollama configuration.

---

## Online RL loop (event-driven)

Human-Watch uses an **event-driven online RL** paradigm — not episodic gym-style RL.
There is no `reset() → step() → done` cycle. Instead, a `PerceptionCycle` arrives
whenever a human completes a feedback interaction, and the loop processes it immediately.

```
Browser event (feedback submitted)
        │
        ▼
demos/human_watch/monitor.py  ──── writes ──→  data/interactions.md
        │
        └── _trigger_evaluator()
                  │
                  │  convergence gate: skip if history.converged == True
                  │
                  ▼
           demos/human_watch/evaluator.py  (subprocess)
                  │
                  ▼
          ┌───────────────────────────────────────────────────┐
          │           OnlineCriticLoop.process_cycle()        │
          │                                                   │
          │  PerceptionCycle                                  │
          │       │                                           │
          │       ▼                                           │
          │  TwoStageCritic (Backward + Optimizer)            │
          │  ← full context: system_prompt, user_query,      │
          │    bot_response, verdict, comment, observations    │
          │       │                                           │
          │       ▼  CriticValidationLoop: re-ask Actor,      │
          │          Judge validates before accepting         │
          │       │                                           │
          │       ▼  CriticOutput (score, proposed_prompt)    │
          │                                                   │
          │  FeedbackAggregator (thumbs + dwell)  → H         │
          │                                                   │
          │  HybridReward.compute()                           │
          │  R = λ_fb·H + λ_c·C − λ_ch·word_change_ratio     │
          │       │                                           │
          │       ▼                                           │
          │  RewardHistory.append()  (convergence tracking)   │
          │  Leaderboard.add()                                │
          │       │                                           │
          │       ▼                                           │
          │  UpdateGate.evaluate()                            │
          │       │                                           │
          │       ├── should_update=True                      │
          │       │       └── ActivePolicy.write()            │
          │       │           data/system_prompt.md updated   │
          │       │           history.bump_version()          │
          │       │                                           │
          │       └── should_update=False                     │
          │               "Politica estable"                  │
          │                                                   │
          └───────────────────────────────────────────────────┘
                  │
                  └── loop.save_state(data/reward_history.json, data/population.json)
```

---

## Library modules

### `core/`
- **`PerceptionCycle`** — the fundamental data unit; carries system_prompt, verdict,
  comment, dwell_seconds, and (for logging only) user_query + bot_response.
- **`ActivePolicy`** — reads/writes `data/system_prompt.md` with versioned backup in `data/prompts/`.

### `llm/`
- **`LLMBackend`** (ABC) — `complete(prompt: str) -> LLMResponse`.
- **`LocalLLMBackend`** — Ollama / Gemma / Groq (OpenAI-compatible API).

### `critic/`
- **`PerceptionCritic`** (Protocol) — `evaluate(cycle) -> CriticOutput`.
- **`TwoStageCritic`** — Critic 1.1 (Backward): full context → feedback; Critic 1.2 (Optimizer): BLIND, feedback → new prompt.
- **`CriticOutput`** — `(critic_score: float, proposed_prompt: str, reasoning: str)`.

### `validation/`
- **`CriticValidationLoop`** — wraps any Critic; re-asks Actor, Judge validates before accepting.
- **`Actor`** — generates responses with system_prompt + user_query.
- **`LLMValidationJudge`** — judges if new response fixes the problem.

### `feedback/`
- **`thumbs_to_score`**, **`reading_time_to_score`** — signal converters.
- **`FeedbackAggregator`** — weighted combination of explicit + implicit signals → `H ∈ [0,1]`.

### `rl/`
- **`HybridReward`** — `R = λ_fb·H + λ_c·C − λ_ch·word_change_ratio`.
  `word_change_ratio` is the fraction of words that changed between the current
  and proposed prompt. It discourages large rewrites.
- **`RewardHistory`** — fixed-size rolling window; computes `R_avg`; tracks convergence.
  Persists to / loads from `reward_history.json`.
- **`UpdateGate`** — fires on degradation (`R_curr < R_avg·0.8`) or forced correction
  (`INCORRECTO` + non-empty comment).

### `population/`
- **`PromptGenome`** — prompt as a dict of named sections (`system_role`, `instructions`, …).
  Factory: `PromptGenome.from_text(text)`. Renders with `to_text()`.
- **`Leaderboard`** — top-N `Individual` entries ranked by fitness.
  No mutation or crossover — this is a pure fitness leaderboard.
  Persists to / loads from `data/population.json`.

### `loop/`
- **`OnlineCriticLoop`** — orchestrates one full RL step per `PerceptionCycle`.
  Accepts all components via constructor injection; ships with sensible defaults.
  `process_cycle(cycle) -> LoopResult`.
  `save_state(history_path, leaderboard_path)` / `load_state(…)`.

---

## Convergence

`RewardHistory` tracks `consecutive_stable`: the number of consecutive cycles where:
- `verdict == CORRECTO`, and
- `word_change_ratio < ε` (the Critic is proposing near-zero changes)

When `consecutive_stable >= convergence_window` (default 5), `history.converged = True`.

`monitor.py` reads this flag before spawning the evaluator subprocess. When `converged`,
the Critic is not called and no further policy updates occur. Convergence is reset
automatically via `history.bump_version()` whenever the gate fires and the policy changes.

---

## State files

| File | Owner | Contents |
|---|---|---|
| `data/system_prompt.md` | `ActivePolicy` | Live Actor policy; hot-reloaded by server on every request |
| `data/interactions.md` | `demos/human_watch/monitor.py` | Append-only Perception Cycle log |
| `data/reward_history.json` | `RewardHistory` | Rolling window, version, convergence state |
| `data/population.json` | `Leaderboard` | Up to 20 prompt candidates ranked by fitness |
| `data/critic_memory.md`, `.json` | `CriticMemory` | Memoria del Critic |
| `data/evaluator.log` | `demos/human_watch/monitor.py` | Evaluator subprocess stdout/stderr |
| `data/prompts/prompt_vN.md` | `ActivePolicy` | Backup before each policy overwrite |
