"""LLMPerceptionCritic — Critic backed by a local LLM (Gemma via Ollama).

Blind-critic design: the Critic receives ONLY
    system_prompt + verdict + comment + dwell_seconds + observations.
It is intentionally blind to user_query and bot_response — forcing it to
learn reward patterns from behavioral signals rather than memorising
conversation content.

The observations list (cycle.observations) carries three categories of
structured strings that are split and placed in dedicated prompt sections:

    Behavioral signals   — lines containing [DWELL], [SELECT], [CLICK],
                           [REVIEW_RAG] — raw human interaction traces.
    Trend context        — lines injected by OnlineCriticLoop from
                           RewardHistory.trend_summary() — statistics about
                           the policy's recent performance.
    Memory               — lines prefixed with [MEMORIA] — the Critic's own
                           past reasoning and hypotheses from CriticMemory.
"""

from __future__ import annotations

import json
import re
import sys

from prompt_rl.core.cycle       import PerceptionCycle
from prompt_rl.core.policy_schema import PolicySchema, parse_policy
from prompt_rl.rl.reward        import word_change_ratio, MAX_POLICY_CHANGE_RATIO
from prompt_rl.critic.base     import CriticOutput
from prompt_rl.critic.config   import build_critic_context_json
from prompt_rl.llm.base        import LLMBackend

# ── Observation classifier ────────────────────────────────────────────────────

_BEHAVIORAL_TAGS = ("[DWELL]", "[SELECT]", "[CLICK]", "[REVIEW_RAG]")
_MEMORY_MARKER   = "[MEMORIA]"


def _split_observations(
    obs: list[str],
) -> tuple[list[str], list[str], list[str]]:
    """
    Split cycle.observations into three groups:
        behavioral  — human interaction traces (DWELL / SELECT / CLICK / REVIEW_RAG)
        trend       — reward-history statistics (injected by OnlineCriticLoop)
        memory      — Critic's past conclusions (injected by CriticMemory)
    """
    behavioral: list[str] = []
    trend:      list[str] = []
    memory:     list[str] = []
    for o in obs:
        if o.startswith(_MEMORY_MARKER):
            memory.append(o[len(_MEMORY_MARKER):].lstrip())
        elif any(t in o for t in _BEHAVIORAL_TAGS):
            behavioral.append(o)
        else:
            trend.append(o)
    return behavioral, trend, memory


# ── Critic prompt ─────────────────────────────────────────────────────────────

def _build_critic_prompt(
    system_prompt: str,
    verdict: str,
    comment: str,
    dwell_seconds: str,
    signals_block: str,
    trend_block: str,
    memory_block: str,
) -> str:
    context_json = build_critic_context_json()
    return f"""\
=== TU CONFIGURACION (JSON) ===
{context_json}

Tu rol: eres el disenador de politicas del Actor. COPIAS la politica actual y añades SOLO lo minimo (maximo 1 regla nueva o 1 ajuste).

INSTRUCCION CLAVE: Abre la politica actual, COPIALA integra en proposed_policy, y solo añade UNA hard_rule nueva si el comentario lo exige. NO reescribas role. NO reemplaces las reglas existentes. TODO en ESPAÑOL.

Violar las reglas_criticas = propuesta RECHAZADA automaticamente.

=== POLITICA ACTUAL DEL ACTOR ===
{system_prompt}

=== CICLO DE FEEDBACK ===
Veredicto humano  : {verdict}
Correccion humana : "{comment}"
Tiempo interaccion: {dwell_seconds}s

=== SEÑALES DE COMPORTAMIENTO ===
{signals_block}

=== HISTORIAL DE REWARDS ===
{trend_block}

=== TU MEMORIA (reflexiones de ciclos anteriores) ===
{memory_block}

=== DECISION SOBRE EL COMENTARIO ===
- direct_rule: comentario explicito y accionable -> añadir como nueva hard_rule.
- refinement: comentario vago o que se solapa con historia -> sintetizar/refinar.

DISTINCIÓN hard_rules vs context_amplification:
- hard_rules: comportamientos OBLIGATORIOS ("siempre indica el IVA", "responde en español").
- context_amplification: info que amplía contexto sin ser imperativa ("el catálogo es fuente de verdad", "hay planes Premium").

RESPONDE SOLO CON ESTE JSON. TODO en ESPAÑOL. SIN codigo markdown:
{{"critic_score": 0.5, "reasoning": "Razon en espanol", "comment_treatment": "direct_rule", "comment_treatment_reasoning": "Explicacion en espanol", "proposed_policy": {{"role": "COPIAR role actual", "hard_rules": ["reglas actuales", "solo 1 nueva si hay comentario"], "context_amplification": ["copiar del actual"], "soft_guidelines": ["copiar actual"]}}, "nota": "Hipotesis en espanol"}}

Escala critic_score: 0.0-0.2=fallo grave, 0.3-0.5=parcial, 0.6-0.8=aceptable, 0.9-1.0=excelente."""


class LLMPerceptionCritic:
    """
    Critic backed by a local LLM (e.g. Gemma 3:4b via Ollama).

    Blind to user_query and bot_response by design.
    Receives behavioral signals, reward history, and its own past memory
    via the cycle.observations list.
    """

    def __init__(
        self,
        backend:     LLMBackend,
        max_tokens:  int   = 2048,
        temperature: float = 0.1,
        verbose:     bool  = False,
    ) -> None:
        self._backend     = backend
        self._max_tokens  = max_tokens
        self._temperature = temperature
        self._verbose     = verbose

    @staticmethod
    def _esc(s: str) -> str:
        """Escape { and } in user-provided strings before .format() call."""
        return s.replace("{", "{{").replace("}", "}}")

    @staticmethod
    def _fmt_block(lines: list[str], empty_msg: str) -> str:
        return (
            "\n".join(f"  {ln}" for ln in lines)
            if lines
            else f"  {empty_msg}"
        )

    @staticmethod
    def _format_policy_for_critic(raw: str) -> str:
        """Format current policy for Critic: structured JSON or legacy plain text."""
        policy, is_structured = parse_policy(raw)
        if is_structured:
            return policy.to_json()
        return f"(texto plano legacy)\n{raw[:1500]}" + ("..." if len(raw) > 1500 else "")

    def evaluate(self, cycle: PerceptionCycle) -> CriticOutput:
        behavioral, trend, memory = _split_observations(cycle.observations)

        system_prompt_display = self._format_policy_for_critic(cycle.system_prompt)

        signals_block = self._fmt_block(
            behavioral,
            "(sin trazas de interaccion — el humano no navego el RAG)",
        )
        trend_block = self._fmt_block(
            trend,
            "(sin historial de rewards previo — primera iteracion)",
        )
        memory_block = self._fmt_block(
            memory,
            "(sin entradas de memoria previas — primera iteracion)",
        )

        prompt_text = _build_critic_prompt(
            system_prompt  = system_prompt_display,
            verdict        = cycle.verdict,
            comment        = cycle.comment or "(sin comentario)",
            dwell_seconds  = f"{cycle.dwell_seconds:.1f}",
            signals_block  = signals_block,
            trend_block    = trend_block,
            memory_block   = memory_block,
        )

        try:
            if self._verbose:
                sys.stdout.write("\n[Critic] Prompt enviado al LLM (~{} chars):\n".format(len(prompt_text)))
                preview = prompt_text[:800] + "..." if len(prompt_text) > 800 else prompt_text
                sys.stdout.write(preview + "\n\n")
                sys.stdout.flush()

            resp = self._backend.complete(
                prompt_text,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            raw = resp.text.strip()
            if self._verbose:
                sys.stdout.write("[Critic] Respuesta raw del LLM:\n" + raw + "\n\n")
                sys.stdout.flush()
            m   = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                data = json.loads(m.group())
                # Prefer proposed_policy (structured) over proposed_prompt (legacy)
                proposed_policy = data.get("proposed_policy")
                if isinstance(proposed_policy, dict):
                    policy = PolicySchema.from_dict(proposed_policy)
                    proposed_prompt = policy.to_json()
                else:
                    proposed_prompt = (
                        data.get("proposed_prompt", cycle.system_prompt)
                        or cycle.system_prompt
                    )
                treatment = str(data.get("comment_treatment", "")).lower()
                if treatment not in ("direct_rule", "refinement"):
                    treatment = ""
                # Sanitización: si el LLM alucinó y propone reescritura completa, forzar incremental
                if word_change_ratio(cycle.system_prompt, proposed_prompt) > MAX_POLICY_CHANGE_RATIO:
                    policy, _ = parse_policy(cycle.system_prompt)
                    if cycle.comment and cycle.comment.strip():
                        rule = cycle.comment.strip()
                        if len(rule) > 80:
                            rule = rule[:77] + "..."
                        policy.hard_rules.append(rule)
                    proposed_prompt = policy.to_json()
                    if self._verbose:
                        sys.stdout.write(
                            "[Critic] Sanitizado: LLM propuso reescritura -> forzado incremental\n"
                        )
                        sys.stdout.flush()
                return CriticOutput(
                    critic_score=float(data.get("critic_score", 0.5)),
                    proposed_prompt=proposed_prompt,
                    reasoning=data.get("reasoning", ""),
                    nota=data.get("nota", ""),
                    comment_treatment=treatment,
                    comment_treatment_reasoning=data.get("comment_treatment_reasoning", ""),
                )
        except Exception:
            pass

        # Safe fallback: keep current prompt, neutral score
        return CriticOutput(
            critic_score=0.5,
            proposed_prompt=cycle.system_prompt,
            reasoning="fallback — LLM error or invalid JSON",
            nota="",
        )
