"""
ValidationJudge: juzga si la nueva respuesta del actor (con el prompt propuesto)
ha solucionado el problema indicado en el feedback humano anterior.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from prompt_rl.llm.base import LLMBackend

def _parse_judge_verdict(text: str) -> bool:
    """Extrae SI/NO de la respuesta del Judge. La primera palabra determina el veredicto."""
    first_line = text.split("\n")[0].strip()
    if not first_line:
        return False
    normalized = first_line.upper().replace("Í", "I").replace("É", "E")
    first_word = normalized.split()[0] if normalized.split() else ""
    if first_word in ("SI", "YES"):
        return True
    if first_word == "NO":
        return False
    # "Sí" -> "SI" tras normalizar; prefijo "SI." o "SI,"
    prefix = normalized[:12]
    if prefix.startswith("SI") or prefix.startswith("YES"):
        return True
    if prefix.startswith("NO"):
        return False
    return False


JUDGE_SYSTEM = """Eres un juez de validación estricto. Tu única tarea es determinar si una nueva respuesta del modelo ha solucionado el problema que el evaluador humano señaló.

Criterios para SI (aprobar la nueva respuesta):
- La nueva respuesta debe responder CORRECTAMENTE la pregunta del usuario en el contexto del feedback.
- Debe abordar EXPLÍCITAMENTE el problema señalado (ej: si faltaba IVA, debe mencionarlo).
- No basta con ser vagamente mejor: debe resolver el defecto concreto.

Criterios para NO (rechazar):
- La nueva respuesta repite el mismo error o lo ignora.
- Responde otra cosa o se desvía de la pregunta.
- Es ambigua o incompleta respecto al problema señalado.

Responde en la primera línea exactamente: SI o NO (en mayúsculas, nada más).
Opcionalmente añade una breve razón en la siguiente línea."""


@dataclass
class ValidationResult:
    fixed: bool
    reasoning: str = ""


class ValidationJudge(ABC):
    """Protocolo: juzga si la nueva respuesta solucionó el problema."""

    @abstractmethod
    def judge(
        self,
        user_query: str,
        original_feedback: str,
        original_response: str,
        new_response: str,
    ) -> ValidationResult:
        """Retorna si la nueva respuesta solucionó el problema (fixed=True/False)."""
        ...


class LLMValidationJudge(ValidationJudge):
    """Juez basado en LLM que evalúa si la nueva respuesta soluciona el feedback."""

    def __init__(
        self,
        backend: LLMBackend,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self._backend = backend
        self._max_tokens = max_tokens
        self._temperature = temperature

    def judge(
        self,
        user_query: str,
        original_feedback: str,
        original_response: str,
        new_response: str,
    ) -> ValidationResult:
        user_message = f"""PREGUNTA DEL USUARIO: "{user_query}"

FEEDBACK DEL EVALUADOR (qué estaba mal): "{original_feedback}"

RESPUESTA ORIGINAL (incorrecta): "{original_response}"

NUEVA RESPUESTA (con prompt mejorado): "{new_response}"

¿La nueva respuesta soluciona el problema señalado en el feedback?
Responde en la primera línea: SI o NO."""

        if hasattr(self._backend, "generate_with_system"):
            resp = self._backend.generate_with_system(
                system_prompt=JUDGE_SYSTEM,
                user_message=user_message,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
        else:
            full_prompt = f"{JUDGE_SYSTEM}\n\n---\n\n{user_message}"
            resp = self._backend.complete(
                full_prompt,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )

        text = resp.text.strip()
        fixed = _parse_judge_verdict(text)
        return ValidationResult(fixed=fixed, reasoning=text)
