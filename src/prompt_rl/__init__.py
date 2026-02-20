"""
prompt_rl — Online RL library for system-prompt refinement with human feedback.

Public surface:

    Core
        PerceptionCycle   — the fundamental data unit
        ActivePolicy      — manages system_prompt.md

    Critic
        PerceptionCritic      — protocol
        LLMPerceptionCritic   — blind (solo verdict+comment)
        TwoStageCritic        — Critic en dos etapas con contexto completo
        CriticBackward        — Critic 1.1: feedback accionable
        CriticOptimizer       — Critic 1.2: nuevo system prompt
        CriticOutput          — score + proposed_prompt + reasoning + nota
        CriticMemory          — persistent journal

    RL components
        HybridReward      — R = λ_fb*H + λ_c*C - λ_ch*change_ratio
        RewardHistory     — rolling window + convergence tracking
        UpdateGate        — degradation / forced update conditions
        AlwaysUpdateGate  — actualiza en cada ciclo

    Feedback
        FeedbackAggregator
        thumbs_to_score, reading_time_to_score
        CursorTrace, HumanFeedbackResult  — trazado cursor + evaluación humana

    Population
        PromptGenome      — modular prompt structure
        Leaderboard       — fitness-ranked candidate store
        Individual

    Loop
        OnlineCriticLoop  — ties everything together
        LoopResult

    Validation (flujo de validación con refinamiento)
        Actor               — genera respuestas con system_prompt + user_query
        ValidationJudge     — juzga si la nueva respuesta solucionó el problema
        CriticValidationLoop — wrapper del critic con validación iterativa

    LLM backends
        LocalLLMBackend   — Ollama (Gemma) local
"""

__version__ = "1.0.0"

from prompt_rl.core.cycle            import PerceptionCycle
from prompt_rl.core.policy           import ActivePolicy
from prompt_rl.critic.base             import CriticOutput, PerceptionCritic
from prompt_rl.critic.llm_critic       import LLMPerceptionCritic
from prompt_rl.critic.two_stage_critic import TwoStageCritic
from prompt_rl.critic.memory           import CriticMemory
from prompt_rl.rl.reward             import HybridReward, word_change_ratio
from prompt_rl.rl.history            import RewardHistory
from prompt_rl.rl.gate               import UpdateGate, GateResult, AlwaysUpdateGate
from prompt_rl.feedback.signals      import (
    FeedbackAggregator,
    thumbs_to_score,
    reading_time_to_score,
)
from prompt_rl.feedback.cursor_trace import CursorTrace, HumanFeedbackResult
from prompt_rl.population.genome      import PromptGenome
from prompt_rl.population.leaderboard import Individual, Leaderboard
from prompt_rl.loop.online            import OnlineCriticLoop, LoopResult
from prompt_rl.validation             import (
    Actor,
    ValidationJudge,
    LLMValidationJudge,
    CriticValidationLoop,
    ValidationLoopResult,
)
from prompt_rl.llm.base      import LLMBackend, LLMResponse
from prompt_rl.llm.local_backend import LocalLLMBackend

__all__ = [
    # core
    "PerceptionCycle", "ActivePolicy",
    # critic
    "PerceptionCritic", "LLMPerceptionCritic", "TwoStageCritic",
    "CriticOutput", "CriticMemory",
    # rl
    "HybridReward", "word_change_ratio", "RewardHistory",
    "UpdateGate", "GateResult", "AlwaysUpdateGate",
    # feedback
    "FeedbackAggregator", "thumbs_to_score", "reading_time_to_score",
    "CursorTrace", "HumanFeedbackResult",
    # population
    "PromptGenome", "Individual", "Leaderboard",
    # loop
    "OnlineCriticLoop", "LoopResult",
    # validation
    "Actor", "ValidationJudge", "LLMValidationJudge",
    "CriticValidationLoop", "ValidationLoopResult",
    # llm
    "LLMBackend", "LLMResponse", "LocalLLMBackend",
]
