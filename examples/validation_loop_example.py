"""
Ejemplo: Flujo de validación con refinamiento iterativo.

Cuando el humano marca INCORRECTO:
1. Critic propone nuevo system prompt
2. Se re-pregunta la MISMA pregunta al actor con el nuevo prompt
3. ValidationJudge evalúa (con feedback original) si se solucionó
4. Si no: ciclo virtual → Critic propone otro prompt → repetir (hasta max_iterations)

Requisitos: pip install -e .  (Ollama con gemma3:4b)
"""

from prompt_rl import (
    PerceptionCycle,
    ActivePolicy,
    TwoStageCritic,
    CriticValidationLoop,
    Actor,
    LLMValidationJudge,
    OnlineCriticLoop,
    RewardHistory,
    Leaderboard,
    AlwaysUpdateGate,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

# Mismo backend para critic, actor y judge (o distintos si se desea)
backend = LocalLLMBackend(model="gemma3:4b")

critic = TwoStageCritic(backend=backend)
actor = Actor(backend=backend)
judge = LLMValidationJudge(backend=backend)

# Critic envuelto en validación: re-pregunta y refina hasta que el judge diga SI
validated_critic = CriticValidationLoop(
    critic=critic,
    actor=actor,
    judge=judge,
    max_iterations=3,
    skip_validation_if_correct=True,
)

policy = ActivePolicy(path="data/system_prompt.md")
history = RewardHistory.from_file("data/reward_history.json")
lb = Leaderboard.from_file("data/population.json")
gate = AlwaysUpdateGate()

loop = OnlineCriticLoop(
    critic=validated_critic,
    policy=policy,
    gate=gate,
    history=history,
    leaderboard=lb,
)

cycle = PerceptionCycle(
    system_prompt="Eres un asistente. Responde en español.",
    user_query="¿El plan Pro incluye IVA?",
    bot_response="No lo sé.",
    verdict="INCORRECTO",
    comment="Siempre incluye IVA del 21 %.",
    dwell_seconds=4.2,
    observations=[],
)

result = loop.process_cycle(cycle)
loop.save_state("data/reward_history.json", "data/population.json")

print("Gate:", result.gate.reason)
print("Proposed (validado):", result.critic_output.proposed_prompt[:200], "...")
