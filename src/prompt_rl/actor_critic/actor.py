"""Actor: generates responses with multiple prompts and selects which to show."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from prompt_rl.llm.base import LLMBackend, LLMResponse


@dataclass
class CandidateResponse:
    """Candidate response: text + prompt_used + index."""

    text: str
    system_prompt: str
    prompt_index: int
    raw: Optional[LLMResponse] = None


class Actor(Protocol):
    """
    Actor_θ(prompt, context, candidate_responses) → probabilities.

    Learns to optimize:
    - Which prompt to use depending on query type
    - Which response to select to maximize expected feedback
    - Exploration vs exploitation balance
    - Which temperature to use for generation (optional, context-dependent)
    """

    def select_response(
        self,
        query: str,
        candidates: list[CandidateResponse],
        context: Optional[dict[str, Any]] = None,
    ) -> int:
        """Returns the index of the response to show to the human."""
        ...

    def get_selection_probs(
        self,
        query: str,
        candidates: list[CandidateResponse],
        context: Optional[dict[str, Any]] = None,
    ) -> list[float]:
        """Optional: probabilities over candidates (for PPO/A2C)."""
        ...

    def get_temperature(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[float]:
        """
        Optional: context-dependent temperature for LLM sampling.
        The Actor can learn which temperature to use (e.g. higher for creative
        queries, lower for factual). Return None to use the flow default.
        """
        ...


class RandomActor:
    """Actor that selects randomly (baseline)."""

    def __init__(self, rng: Optional[Any] = None) -> None:
        import random
        self._rng = rng or random.Random()

    def select_response(
        self,
        query: str,
        candidates: list[CandidateResponse],
        context: Optional[dict[str, Any]] = None,
    ) -> int:
        if not candidates:
            raise ValueError("candidates cannot be empty")
        return self._rng.randint(0, len(candidates) - 1)

    def get_selection_probs(
        self,
        query: str,
        candidates: list[CandidateResponse],
        context: Optional[dict[str, Any]] = None,
    ) -> list[float]:
        n = len(candidates)
        return [1.0 / n] * n

    def get_temperature(
        self,
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> Optional[float]:
        """Baseline: no context-dependent temperature; use flow default."""
        return None


def get_actor_temperature(
    actor: Actor,
    query: str,
    context: Optional[dict[str, Any]] = None,
    default: Optional[float] = None,
) -> Optional[float]:
    """
    Returns the Actor's context-dependent temperature if implemented and not None;
    otherwise returns default. Use for LLM sampling (e.g. in generate_candidates).
    """
    t = actor.get_temperature(query, context)
    return t if t is not None else default


def _generate_one(
    llm: LLMBackend,
    query: str,
    max_tokens: int,
    temperature: float,
    index: int,
    system_prompt: str,
    **kwargs: Any,
) -> CandidateResponse:
    full_prompt = f"{system_prompt}\n\nUser: {query}"
    resp = llm.complete(
        full_prompt, max_tokens=max_tokens, temperature=temperature, **kwargs
    )
    return CandidateResponse(
        text=resp.text,
        system_prompt=system_prompt,
        prompt_index=index,
        raw=resp,
    )


def generate_candidates(
    llm: LLMBackend,
    system_prompts: list[str],
    query: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    **kwargs: Any,
) -> list[CandidateResponse]:
    """
    For each system prompt in system_prompts, generates one response.
    Returns K candidate responses (one per prompt). Sequential.
    """
    return [
        _generate_one(llm, query, max_tokens, temperature, i, sp, **kwargs)
        for i, sp in enumerate(system_prompts)
    ]


def generate_candidates_parallel(
    llm: LLMBackend,
    system_prompts: list[str],
    query: str,
    max_tokens: int = 512,
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    **kwargs: Any,
) -> list[CandidateResponse]:
    """
    Same as generate_candidates but runs LLM calls in parallel (ThreadPoolExecutor).
    Use when K is large or LLM latency is high. max_workers: None = min(K, 4).
    """
    from prompt_rl.utils.parallel import map_parallel

    if not system_prompts:
        return []
    max_workers = max_workers or min(len(system_prompts), 4)

    def fn(item: tuple[int, str]) -> CandidateResponse:
        i, sp = item
        return _generate_one(llm, query, max_tokens, temperature, i, sp, **kwargs)

    items = list(enumerate(system_prompts))
    return map_parallel(fn, items, max_workers=max_workers)
