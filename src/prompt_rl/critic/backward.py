"""
Critic 1.1 (Backward): produce feedback accionable a partir de la evaluación humana.

Contexto completo: conversación (system, input, output),
feedback humano + correcto/incorrecto + trazado cursor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import re

from prompt_rl.core.policy_schema import parse_policy, PolicySchema
from prompt_rl.feedback.cursor_trace import HumanFeedbackResult
from prompt_rl.critic.two_stage_prompts import (
    CONVERSATION_TEMPLATE,
    CONVERSATION_START,
    OBJECTIVE_INSTRUCTION,
    EVALUATE_VARIABLE,
    EVALUATE_BY_COMPONENT,
    COMPONENT_BLOCK_TEMPLATE,
    COMPONENT_NAMES,
    COMPONENT_DESCRIPTIONS,
    CRITIC_BACKWARD_SYSTEM,
    GRADIENT_TEMPLATE,
)
from prompt_rl.llm.base import LLMBackend


def _truncate(text: str, n_words: int = 80) -> str:
    words = text.split()
    if len(words) <= 2 * n_words:
        return text
    return " ".join(words[:n_words]) + " (...) " + " ".join(words[-n_words:])


def _parse_feedback_by_component(raw: str) -> dict[str, str]:
    """Extrae feedback por componente del texto de respuesta del Backward."""
    result: dict[str, str] = {c: "" for c in COMPONENT_NAMES}
    pattern = r'<FEEDBACK\s+component="([^"]+)">\s*(.*?)\s*</FEEDBACK>'
    for m in re.finditer(pattern, raw, re.DOTALL):
        comp, text = m.group(1).strip(), m.group(2).strip()
        if comp in COMPONENT_NAMES:
            result[comp] = text
    return result


@dataclass
class BackwardOutput:
    """
    Salida del Critic 1.1: feedback accionable + contexto (2.3).

    Cuando component_propagation está activo: feedback_by_component contiene
    feedback específico por rol, hard_rules, context_amplification, soft_guidelines.
    feedback_text sigue siendo la concatenación para compatibilidad.
    """

    feedback_text: str
    context: dict[str, str] = field(default_factory=dict)
    feedback_by_component: dict[str, str] | None = None

    def to_gradient_blind(self) -> str:
        """
        Formato BLIND para el Critic 1.2 (Optimizer).

        Solo incluye el feedback accionable. NO incluye user_query ni bot_response.
        Con feedback_by_component: concatena para uso legacy.
        """
        if self.feedback_by_component:
            parts = [
                f"[{k}]\n{v}" for k, v in self.feedback_by_component.items()
                if v and v.lower() not in ("ok", "mantener", "-")
            ]
            if parts:
                return "\n\n".join(parts)
        return self.feedback_text

    def to_gradient_with_context(self) -> str:
        """Formato con contexto completo (conversación). Para uso cuando se necesita."""
        if not self.context:
            return self.feedback_text
        return GRADIENT_TEMPLATE.format(
            feedback=self.feedback_text,
            context=self.context.get("context", ""),
            response_desc=self.context.get("response_desc", ""),
            variable_desc=self.context.get("variable_desc", ""),
        )


def _build_components_block(policy: PolicySchema) -> str:
    """Construye el bloque de componentes para el prompt de propagación."""
    parts = []
    if policy.role.strip():
        content = policy.role.strip()
        parts.append(
            COMPONENT_BLOCK_TEMPLATE.format(
                name="role",
                desc=COMPONENT_DESCRIPTIONS["role"],
                content=content,
            )
        )
    if policy.hard_rules:
        content = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(policy.hard_rules))
        parts.append(
            COMPONENT_BLOCK_TEMPLATE.format(
                name="hard_rules",
                desc=COMPONENT_DESCRIPTIONS["hard_rules"],
                content=content,
            )
        )
    if policy.context_amplification:
        content = "\n".join(f"  - {c}" for c in policy.context_amplification)
        parts.append(
            COMPONENT_BLOCK_TEMPLATE.format(
                name="context_amplification",
                desc=COMPONENT_DESCRIPTIONS["context_amplification"],
                content=content,
            )
        )
    if policy.soft_guidelines:
        content = "\n".join(f"  - {g}" for g in policy.soft_guidelines)
        parts.append(
            COMPONENT_BLOCK_TEMPLATE.format(
                name="soft_guidelines",
                desc=COMPONENT_DESCRIPTIONS["soft_guidelines"],
                content=content,
            )
        )
    return "\n\n".join(parts) if parts else ""


class CriticBackward:
    """
    Critic 1.1: Backward agent.
    Recibe contexto completo (conversación + evaluación humana).
    Produce feedback accionable. Con component_propagation: feedback por
    role, hard_rules, context_amplification, soft_guidelines.
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        component_propagation: bool = True,
    ):
        self._backend = backend
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._component_propagation = component_propagation

    def run(
        self,
        system_prompt: str,
        prompt: str,
        response_value: str,
        human_evaluation: HumanFeedbackResult,
        variable_desc: str = "system prompt que guía al modelo",
        variable_short: str | None = None,
        response_desc: str = "respuesta del modelo",
    ) -> BackwardOutput:
        evaluation_string = human_evaluation.to_evaluation_string()
        variable_short = variable_short or _truncate(system_prompt)

        conversation = CONVERSATION_TEMPLATE.format(
            system_prompt=system_prompt,
            prompt=prompt,
            response_value=response_value,
        )
        human_block = f"<EVALUACION_HUMANA>\n{evaluation_string}\n</EVALUACION_HUMANA>"

        # Propagación por componentes: policy estructurada → feedback por componente
        policy, is_structured = parse_policy(system_prompt)
        feedback_by_component: dict[str, str] | None = None

        if self._component_propagation and is_structured:
            components_block = _build_components_block(policy)
            backward_prompt = (
                CONVERSATION_START.format(
                    conversation=conversation,
                    variable_desc=variable_desc,
                )
                + OBJECTIVE_INSTRUCTION.format(human_evaluation_block=human_block)
                + EVALUATE_BY_COMPONENT.format(components_block=components_block)
            )
            full_prompt = f"{CRITIC_BACKWARD_SYSTEM}\n\n---\n\n{backward_prompt}"
            resp = self._backend.complete(
                full_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            raw = resp.text.strip()
            feedback_by_component = _parse_feedback_by_component(raw)
            feedback_text = "\n\n".join(
                f"[{k}]\n{v}" for k, v in feedback_by_component.items() if v
            ) or raw
        else:
            backward_prompt = (
                CONVERSATION_START.format(
                    conversation=conversation,
                    variable_desc=variable_desc,
                )
                + OBJECTIVE_INSTRUCTION.format(human_evaluation_block=human_block)
                + EVALUATE_VARIABLE.format(
                    variable_desc=variable_desc,
                    variable_short=variable_short,
                )
            )
            full_prompt = f"{CRITIC_BACKWARD_SYSTEM}\n\n---\n\n{backward_prompt}"
            resp = self._backend.complete(
                full_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            feedback_text = resp.text.strip()

        return BackwardOutput(
            feedback_text=feedback_text,
            context={
                "context": conversation,
                "response_desc": response_desc,
                "variable_desc": variable_desc,
            },
            feedback_by_component=feedback_by_component,
        )
