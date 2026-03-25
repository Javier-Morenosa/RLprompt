"""
prompt_rl — RL library for system-prompt refinement with LLMs.

Supports two operating modes:

  Online (chatbot)  — human feedback drives the loop in real time via a
                      FastAPI server + Playwright monitor (demos/human_watch).

  Offline (dataset) — a Q&A dataset with train/test split drives the loop
                      automatically; correctness is evaluated deterministically
                      (no extra LLM calls). See prompt_rl.dataset.

Public surface:

    Core
        PerceptionCycle   — the fundamental data unit
        ActivePolicy      — manages system_prompt.md

    Critic
        PerceptionCritic      — protocol
        LLMPerceptionCritic   — blind (solo verdict+comment)
        TwoStageCritic        — context-aware two-stage critic
        CriticBackward        — stage 1: actionable feedback
        CriticOptimizer       — stage 2: new system prompt
        CriticOutput          — score + proposed_prompt + reasoning
        CriticMemory          — persistent critic journal

    RL components
        HybridReward      — R = λ_fb*H + λ_c*C - λ_ch*change_ratio
        RewardHistory     — rolling window + convergence tracking
        UpdateGate        — degradation / forced update conditions
        AlwaysUpdateGate  — updates on every cycle

    Feedback (online mode)
        FeedbackAggregator
        thumbs_to_score, reading_time_to_score
        CursorTrace, HumanFeedbackResult

    Population
        PromptGenome      — modular prompt structure
        Leaderboard       — fitness-ranked candidate store
        Individual

    Loop
        OnlineCriticLoop  — single-event processor (chatbot mode)
        LoopResult
        DatasetLoop       — dataset iterator (offline mode)
        DatasetResult, EpochMetrics

    Dataset (offline mode)
        DatasetSample     — question + answer + extracted + metadata
        DatasetSplit      — train/test split with reproducible shuffling
        DatasetJudge      — deterministic correctness protocol
        ExactMatchJudge   — normalized exact-string match
        ContainsMatchJudge — substring containment match

    Validation
        Actor               — generates responses from system_prompt + query
        ValidationJudge     — checks if new response fixes the problem
        LLMValidationJudge  — LLM-based judge
        CriticValidationLoop — critic wrapper with iterative validation

    LLM backends
        LLMBackend, LLMResponse
        LocalLLMBackend   — OpenAI-compatible endpoint (Groq, Ollama, etc.)
"""

__version__ = "1.0.0"

from prompt_rl.core.cycle            import PerceptionCycle
from prompt_rl.core.policy           import ActivePolicy
from prompt_rl.critic.base             import CriticOutput, PerceptionCritic
from prompt_rl.critic.llm_critic       import LLMPerceptionCritic
from prompt_rl.critic.two_stage_critic import TwoStageCritic
from prompt_rl.critic.backward        import CriticBackward, BackwardOutput
from prompt_rl.critic.optimizer       import CriticOptimizer
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
from prompt_rl.llm.base          import LLMBackend, LLMResponse
from prompt_rl.llm.local_backend import LocalLLMBackend
from prompt_rl.dataset           import (
    DatasetSample,
    DatasetSplit,
    DatasetJudge,
    ExactMatchJudge,
    ContainsMatchJudge,
    DatasetLoop,
    DatasetResult,
    EpochMetrics,
)

__all__ = [
    # core
    "PerceptionCycle", "ActivePolicy",
    # critic
    "PerceptionCritic", "LLMPerceptionCritic", "TwoStageCritic",
    "CriticBackward", "BackwardOutput", "CriticOptimizer",
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
    # dataset (offline mode)
    "DatasetSample", "DatasetSplit",
    "DatasetJudge", "ExactMatchJudge", "ContainsMatchJudge",
    "DatasetLoop", "DatasetResult", "EpochMetrics",
]
