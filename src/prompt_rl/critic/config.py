"""Critic configuration — structured JSON for the Critic's system instructions."""

from __future__ import annotations

import json

# ── What the Critic does NOT know (blind context) ──────────────────────────────
CRITIC_BLIND_CONTEXT = {
    "no_conoces": [
        "El contenido del RAG/catálogo de la empresa. El servidor lo inyecta al Actor; tú nunca lo ves.",
        "La consulta del usuario (qué preguntó).",
        "La respuesta del bot (qué dijo).",
    ],
    "solo_conoces": [
        "El system_prompt actual del Actor (la política).",
        "El veredicto humano (CORRECTO/INCORRECTO).",
        "La corrección o comentario adicional del humano.",
        "Las señales de comportamiento (DWELL, SELECT, CLICK, REVIEW_RAG).",
        "El historial de rewards y tu propia memoria de ciclos anteriores.",
    ],
}

# ── Critic hard rules (never violate) ─────────────────────────────────────────
CRITIC_HARD_RULES = [
    "IDIOMA: TODO tu output (reasoning, proposed_policy, nota) DEBE estar en ESPAÑOL. Prohibido inglés.",
    "PROHIBIDO reescribir la política. NUNCA cambies role completo. NUNCA reemplaces la lista entera de hard_rules. Una interacción = máximo UNA regla nueva o UN ajuste puntual.",
    "COPIAR primero: toma la política actual y COPIALA. Solo añade UNA hard_rule nueva (si hay comentario) o modifica UNA regla existente. No inventes, no reestructures.",
    "Ejemplo correcto: política actual tiene 5 hard_rules -> proposed_policy tiene las mismas 5 + 1 nueva. Ejemplo PROHIBIDO: proposed_policy con role y reglas totalmente diferentes.",
]

# ── Output format ─────────────────────────────────────────────────────────────
CRITIC_OUTPUT_SCHEMA = {
    "critic_score": "float 0.0-1.0",
    "reasoning": "string EN ESPAÑOL",
    "comment_treatment": "direct_rule | refinement",
    "comment_treatment_reasoning": "string EN ESPAÑOL",
    "proposed_policy": {
        "role": "string EN ESPAÑOL — COPIAR del actual",
        "hard_rules": "lista — COPIAR + añadir 0-1 regla OBLIGATORIA nueva",
        "context_amplification": "lista — COPIAR del actual; info/recordatorios que amplían contexto (no reglas rígidas)",
        "soft_guidelines": "lista — COPIAR del actual",
    },
    "nota": "string EN ESPAÑOL",
}


def build_critic_context_json() -> str:
    """Build the JSON context for the Critic prompt."""
    return json.dumps(
        {
            "contexto_ceguera": CRITIC_BLIND_CONTEXT,
            "reglas_criticas_del_critico": CRITIC_HARD_RULES,
            "formato_salida": CRITIC_OUTPUT_SCHEMA,
        },
        ensure_ascii=False,
        indent=2,
    )
