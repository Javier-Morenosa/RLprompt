"""
Actor / ResponseGenerator: genera respuestas dado un system prompt y una pregunta.

Se usa en el flujo de validación para re-preguntar al actor con el nuevo
system prompt propuesto por el crítico.
"""

from __future__ import annotations

from prompt_rl.llm.base import LLMBackend, LLMResponse


class Actor:
    """
    Genera respuestas usando un LLM con system prompt + user query.

    Requiere LocalLLMBackend (con generate_with_system).
    """

    def __init__(
        self,
        backend: LLMBackend,
        max_tokens: int = 512,
        temperature: float = 0.2,
    ):
        if not hasattr(backend, "generate_with_system"):
            raise TypeError(
                "Actor requiere LocalLLMBackend (con generate_with_system)"
            )
        self._backend = backend
        self._max_tokens = max_tokens
        self._temperature = temperature

    def generate(self, system_prompt: str, user_query: str) -> str:
        """Genera una respuesta con el system prompt y la pregunta del usuario."""
        resp = self._backend.generate_with_system(
            system_prompt=system_prompt,
            user_message=user_query,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
        )
        return resp.text.strip()
