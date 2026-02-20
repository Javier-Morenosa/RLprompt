"""LLM backends: solo local (Ollama con Gemma)."""

from prompt_rl.llm.base import LLMBackend, LLMResponse
from prompt_rl.llm.local_backend import LocalLLMBackend

__all__ = ["LLMBackend", "LLMResponse", "LocalLLMBackend"]
