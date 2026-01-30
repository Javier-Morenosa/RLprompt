"""LLM module: backends and integration with language models."""

from prompt_rl.llm.base import LLMBackend, LLMResponse
from prompt_rl.llm.mock import MockLLM

# Optional backends (require pip install prompt-rl[openai])
try:
    from prompt_rl.llm.openai_backend import OpenAIBackend
    from prompt_rl.llm.local_backend import LocalLLMBackend
except ImportError:
    OpenAIBackend = None  # type: ignore[misc, assignment]
    LocalLLMBackend = None  # type: ignore[misc, assignment]

__all__ = [
    "LLMBackend",
    "LLMResponse",
    "MockLLM",
    "OpenAIBackend",
    "LocalLLMBackend",
]
