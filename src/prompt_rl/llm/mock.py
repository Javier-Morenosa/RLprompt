"""Mock backend for testing and development without an API."""

from prompt_rl.core.prompt import Prompt
from prompt_rl.llm.base import LLMBackend, LLMResponse


class MockLLM(LLMBackend):
    """
    Test LLM backend that returns predefined responses.
    Useful for tests and development without consuming APIs.
    """

    def __init__(
        self,
        default_response: str = "Mock response",
        model: str = "mock",
    ) -> None:
        self.default_response = default_response
        self.model = model
        self._call_count = 0

    def complete(
        self,
        prompt: Prompt | str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: object,
    ) -> LLMResponse:
        self._call_count += 1
        text = prompt.text if isinstance(prompt, Prompt) else prompt
        response_text = self.default_response if not text.strip() else f"[Refined] {text[:50]}..."
        return LLMResponse(
            text=response_text,
            model=self.model,
            usage={"prompt_tokens": len(text.split()), "completion_tokens": 10},
        )

    @property
    def call_count(self) -> int:
        return self._call_count
