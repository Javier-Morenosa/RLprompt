"""
Critic 1.2 (Optimizer): genera un nuevo system prompt a partir del feedback del Critic 1.1.

Soporta gradient_memory (feedbacks anteriores), constraints, in_context_examples,
y propagación por componentes (feedback específico por role, hard_rules, etc.).
"""

from __future__ import annotations

import json
import re

from prompt_rl.core.policy_schema import parse_policy, PolicySchema
from prompt_rl.critic.backward import BackwardOutput
from prompt_rl.critic.two_stage_prompts import (
    CRITIC_OPTIMIZER_SYSTEM,
    OPTIMIZER_PREFIX,
    OPTIMIZER_SUFFIX,
    OPTIMIZER_GRADIENT_MEMORY,
    OPTIMIZER_CONSTRAINTS,
    OPTIMIZER_IN_CONTEXT_EXAMPLES,
    OPTIMIZER_BY_COMPONENT_SYSTEM,
    OPTIMIZER_BY_COMPONENT_PREFIX,
    COMPONENT_NAMES,
    COMPONENT_DESCRIPTIONS,
)
from prompt_rl.llm.base import LLMBackend

DEFAULT_TAGS = ["<IMPROVED_SYSTEM_PROMPT>", "</IMPROVED_SYSTEM_PROMPT>"]


def _truncate(text: str, n_words: int = 80) -> str:
    words = text.split()
    if len(words) <= 2 * n_words:
        return text
    return " ".join(words[:n_words]) + " (...) " + " ".join(words[-n_words:])


def _format_constraint_text(constraints: list[str]) -> str:
    return "\n".join(f"  {i+1}. {c}" for i, c in enumerate(constraints))


def _format_gradient_memory(feedbacks: list[str]) -> str:
    return "\n".join(
        f"<FEEDBACK-{i+1}>{fb}</FEEDBACK-{i+1}>"
        for i, fb in enumerate(feedbacks)
    )


def _format_feedback_by_component(feedback_by_component: dict[str, str]) -> str:
    """Formatea feedback por componente para el prompt del Optimizer."""
    parts = []
    for name in COMPONENT_NAMES:
        fb = feedback_by_component.get(name, "").strip()
        desc = COMPONENT_DESCRIPTIONS.get(name, name)
        if fb and fb.lower() not in ("ok", "mantener", "-"):
            parts.append(f"<{name.upper()} desc=\"{desc}\">\n{fb}\n</{name.upper()}>")
    return "\n\n".join(parts) if parts else ""


class CriticOptimizer:
    """
    Critic 1.2: Optimizer agent.
    Recibe output del Critic 1.1 y genera nuevo system prompt.

    Opciones:
      gradient_memory: número de feedbacks anteriores a incluir (0=desactivado)
      constraints: restricciones en lenguaje natural
      in_context_examples: ejemplos de mejoras para guiar el optimizer
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        new_prompt_tags: list[str] | None = None,
        gradient_memory: int = 0,
        constraints: list[str] | None = None,
        in_context_examples: list[str] | None = None,
    ):
        self._backend = backend
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._tags = new_prompt_tags or DEFAULT_TAGS
        self._gradient_memory_size = max(0, gradient_memory)
        self._constraints = constraints or []
        self._in_context_examples = in_context_examples or []
        # Buffer de feedbacks recientes para gradient memory
        self._gradient_memory_buffer: list[str] = []

    def run(
        self,
        current_system_prompt: str,
        backward_output: BackwardOutput,
        variable_desc: str = "system prompt que guía al modelo",
        variable_short: str | None = None,
    ) -> str:
        policy, is_structured = parse_policy(current_system_prompt)
        # Ruta por componentes: feedback por predecessor + policy estructurada
        if (
            backward_output.feedback_by_component
            and is_structured
            and _format_feedback_by_component(backward_output.feedback_by_component)
        ):
            return self._run_by_component(
                policy=policy,
                backward_output=backward_output,
            )

        # Ruta legacy: feedback único
        variable_short = variable_short or _truncate(current_system_prompt)
        variable_grad = backward_output.to_gradient_blind()

        optimizer_system = CRITIC_OPTIMIZER_SYSTEM.format(
            new_variable_start_tag=self._tags[0],
            new_variable_end_tag=self._tags[1],
        )

        user_prompt = OPTIMIZER_PREFIX.format(
            variable_desc=variable_desc,
            variable_short=variable_short,
            variable_grad=variable_grad,
        )

        # Gradient memory: feedbacks de ciclos anteriores
        if self._gradient_memory_size > 0 and self._gradient_memory_buffer:
            user_prompt += OPTIMIZER_GRADIENT_MEMORY.format(
                gradient_memory=_format_gradient_memory(self._gradient_memory_buffer),
            )
        if self._gradient_memory_size > 0:
            self._gradient_memory_buffer.append(variable_grad)
            self._gradient_memory_buffer = self._gradient_memory_buffer[
                -self._gradient_memory_size :
            ]

        if self._constraints:
            user_prompt += OPTIMIZER_CONSTRAINTS.format(
                constraint_text=_format_constraint_text(self._constraints),
            )

        if self._in_context_examples:
            user_prompt += OPTIMIZER_IN_CONTEXT_EXAMPLES.format(
                in_context_examples="\n".join(self._in_context_examples),
            )

        user_prompt += OPTIMIZER_SUFFIX.format(
            new_variable_start_tag=self._tags[0],
            new_variable_end_tag=self._tags[1],
        )
        full_prompt = f"{optimizer_system}\n\n---\n\n{user_prompt}"

        resp = self._backend.complete(
            full_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        return self._extract_from_tags(resp.text.strip())

    def _run_by_component(
        self,
        policy: PolicySchema,
        backward_output: BackwardOutput,
    ) -> str:
        """Optimiza cada componente según su feedback (propagación por predecessors)."""
        fb_formatted = _format_feedback_by_component(
            backward_output.feedback_by_component  # type: ignore[arg-type]
        )
        if not fb_formatted:
            return policy.to_json()

        optimizer_system = OPTIMIZER_BY_COMPONENT_SYSTEM.format(
            new_variable_start_tag=self._tags[0],
            new_variable_end_tag=self._tags[1],
        )
        user_prompt = OPTIMIZER_BY_COMPONENT_PREFIX.format(
            policy_json=policy.to_json(),
            feedback_by_component=fb_formatted,
        )
        if self._gradient_memory_size > 0 and self._gradient_memory_buffer:
            user_prompt += OPTIMIZER_GRADIENT_MEMORY.format(
                gradient_memory=_format_gradient_memory(self._gradient_memory_buffer),
            )
        user_prompt += OPTIMIZER_SUFFIX.format(
            new_variable_start_tag=self._tags[0],
            new_variable_end_tag=self._tags[1],
        )

        if self._constraints:
            user_prompt += OPTIMIZER_CONSTRAINTS.format(
                constraint_text=_format_constraint_text(self._constraints),
            )

        full_prompt = f"{optimizer_system}\n\n---\n\n{user_prompt}"
        resp = self._backend.complete(
            full_prompt,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )

        # Gradient memory para próximo ciclo
        if self._gradient_memory_size > 0:
            variable_grad = backward_output.to_gradient_blind()
            self._gradient_memory_buffer.append(variable_grad)
            self._gradient_memory_buffer = self._gradient_memory_buffer[
                -self._gradient_memory_size :
            ]

        raw = self._extract_from_tags(resp.text.strip())
        # Parsear JSON; si falla, devolver policy actual
        parsed = PolicySchema.from_json(raw)
        if parsed is not None:
            return parsed.to_json()
        # Fallback: intentar extraer JSON de markdown
        if "```" in raw:
            for block in re.findall(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL):
                p = PolicySchema.from_json(block.strip())
                if p is not None:
                    return p.to_json()
        return policy.to_json()

    def _extract_from_tags(self, raw: str) -> str:
        """Extrae texto entre las etiquetas de respuesta."""
        pattern = re.escape(self._tags[0]) + r"(.*?)" + re.escape(self._tags[1])
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            return match.group(1).strip()
        try:
            parts = raw.split(self._tags[0])
            if len(parts) >= 2:
                return parts[1].split(self._tags[1])[0].strip()
        except IndexError:
            pass
        raise ValueError(
            f"El optimizador no devolvió el formato esperado. "
            f"Se esperaba texto entre {self._tags[0]}...{self._tags[1]}. "
            f"Respuesta: {raw[:400]}..."
        )
