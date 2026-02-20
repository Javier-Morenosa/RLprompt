# Documentation

## Core docs

- **[ARCHITECTURE.md](ARCHITECTURE.md)** — library vs implementation, full RL loop diagram,
  module descriptions, convergence criterion, state files in `data/`.
- **[USAGE_GUIDE.md](USAGE_GUIDE.md)** — guía paso a paso para ejecutar Human-Watch (servidor, monitor, feedback, convergencia).

## Estructura actual

- **Library:** `src/prompt_rl/` (core, critic, loop, validation, rl, feedback, population, llm)
- **Demo:** `demos/human_watch/` (server, monitor, evaluator, run_backend, reset)
- **Estado:** todos los `.md` y `.json` de runtime en `data/` (system_prompt, interactions, reward_history, population, critic_memory, etc.)

## Módulos de la librería

| Package | Key types |
|---|---|
| `core` | `PerceptionCycle`, `ActivePolicy`, `policy_schema` |
| `llm` | `LLMBackend`, `LocalLLMBackend` (Ollama/Gemma/Groq) |
| `validation` | `CriticValidationLoop`, `Actor`, `LLMValidationJudge` |
| `critic` | `PerceptionCritic` (Protocol), `TwoStageCritic`, `CriticOutput`, `CriticMemory` |
| `feedback` | `FeedbackAggregator`, `thumbs_to_score`, `reading_time_to_score` |
| `rl` | `HybridReward`, `RewardHistory`, `UpdateGate`, `GateResult` |
| `population` | `PromptGenome`, `Leaderboard`, `Individual` |
| `loop` | `OnlineCriticLoop`, `LoopResult` |

## Docs conceptuales

Los siguientes documentos describen extensiones conceptuales o planes futuros (p. ej. evolución, GPRO, métricas avanzadas). La implementación actual es Human-Watch con **TwoStageCritic** y **OnlineCriticLoop**:

- [PARAMETRIZATION.md](PARAMETRIZATION.md) — parámetros Actor/Critic, población, GPRO
- [GPRO.md](GPRO.md) — Generalized Policy Optimization
- [KEY_INNOVATIONS.md](KEY_INNOVATIONS.md) — diversidad, escalabilidad, exploración guiada
- [METRICS.md](METRICS.md) — fitness, convergencia, diversidad
- [PARALLEL_EVAL.md](PARALLEL_EVAL.md) — paralelización de evaluaciones
