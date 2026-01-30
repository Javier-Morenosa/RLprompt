"""Backend for OpenAI API (optional, requires openai installed)."""

from typing import Any, Optional

from prompt_rl.core.prompt import Prompt
from prompt_rl.llm.base import LLMBackend, LLMResponse


def _check_openai() -> None:
    try:
        import openai  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "To use OpenAIBackend install: pip install prompt-rl[openai]"
        ) from e


class OpenAIBackend(LLMBackend):
    """
    Backend that uses the OpenAI API (GPT-4, GPT-3.5, etc.) or any
    OpenAI-compatible endpoint (e.g. local server).

    - **OpenAI cloud:** Set `api_key` or use env `OPENAI_API_KEY`; leave
      `base_url` unset.
    - **Local / custom endpoint:** Set `base_url` (e.g. LM Studio, Ollama,
      vLLM). For local servers that do not require auth, you can pass
      a placeholder api_key (e.g. "not-needed") if the client requires it.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        **client_kwargs: Any,
    ) -> None:
        _check_openai()
        from openai import OpenAI

        self.model = model
        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        kwargs.update(client_kwargs)
        self._client = OpenAI(**kwargs)

    def complete(
        self,
        prompt: Prompt | str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        text = prompt.text if isinstance(prompt, Prompt) else prompt
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": text}],
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = resp.choices[0]
        usage = None
        if resp.usage:
            usage = {
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            }
        return LLMResponse(
            text=choice.message.content or "",
            model=self.model,
            usage=usage,
            raw=resp,
        )
