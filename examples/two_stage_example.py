"""
Ejemplo: TwoStageCritic — Critic en dos etapas con contexto completo.

Recibe user_query y bot_response. Flujo: HumanFeedback → Critic 1.1 (Backward) → Critic 1.2 (Optimizer).
AlwaysUpdateGate: actualiza el system prompt en cada ciclo.

Requisitos: pip install -e .  (Ollama con gemma3:4b)
"""

from prompt_rl import (
    PerceptionCycle,
    ActivePolicy,
    TwoStageCritic,
    OnlineCriticLoop,
    RewardHistory,
    Leaderboard,
    AlwaysUpdateGate,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

backend = LocalLLMBackend(model="gemma3:4b")
critic = TwoStageCritic(
    backend=backend,
    include_cursor_trace=True,
    gradient_memory=3,  # incluir últimos 3 feedbacks para evitar repeticiones
    constraints=["Responde siempre en español.", "Mantén el prompt conciso."],
    in_context_examples=[
        "Antes: 'Eres un asistente.' → Después: 'Eres un asistente profesional. Responde en español.'",
        "Antes: 'Responde preguntas.' → Después: 'Responde preguntas de forma clara. Indica siempre fuentes.'",
    ],
)

policy = ActivePolicy(path="data/system_prompt.md")
history = RewardHistory.from_file("data/reward_history.json")
lb = Leaderboard.from_file("data/population.json")

# AlwaysUpdateGate = actualiza en cada iteración
gate = AlwaysUpdateGate()

loop = OnlineCriticLoop(
    critic=critic,
    policy=policy,
    history=history,
    leaderboard=lb,
    gate=gate,
)

# Ciclo con observaciones (trazado cursor)
cycle = PerceptionCycle(
    system_prompt="Eres un asistente de negocio. Responde en español.",
    user_query="¿El plan Pro incluye IVA?",
    bot_response="No lo sé.",
    verdict="INCORRECTO",
    comment="Siempre incluye IVA del 21 %.",
    dwell_seconds=4.2,
    observations=[
        "[DWELL] 4.2s",
        "[SELECT] No lo sé",
    ],
)

result = loop.process_cycle(cycle)
loop.save_state("data/reward_history.json", "data/population.json")

print("Gate:", result.gate.reason)
print("Proposed prompt:", result.critic_output.proposed_prompt[:200], "...")
