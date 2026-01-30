"""
Parallelization of evaluations.

- run_parallel: run a callable on each item in parallel (ThreadPoolExecutor).
- map_parallel: map(fn, items) in parallel; returns list of results in order.
Use for parallel LLM generation (K candidates) or parallel Critic scoring.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_parallel(
    fn: Callable[..., R],
    items: list[tuple[Any, ...]],
    max_workers: Optional[int] = None,
) -> list[R]:
    """
    Runs fn(*item) for each item in items, in parallel.
    items[i] is the tuple of positional arguments for the i-th call.
    Returns results in the same order as items.
    """
    if not items:
        return []
    max_workers = max_workers or min(len(items), 4)
    results: list[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(fn, *item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                raise
    return [r for r in results if r is not None]


def map_parallel(
    fn: Callable[[T], R],
    items: list[T],
    max_workers: Optional[int] = None,
) -> list[R]:
    """
    Maps fn over items in parallel. Returns results in the same order as items.
    """
    if not items:
        return []
    max_workers = max_workers or min(len(items), 4)
    results: list[Optional[R]] = [None] * len(items)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(fn, item): i for i, item in enumerate(items)}
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception:
                raise
    return list(results)  # type: ignore[return-value]


def score_batch_parallel(
    score_fn: Callable[[str, str, str], float],
    items: list[tuple[str, str, str]],
    max_workers: Optional[int] = None,
) -> list[float]:
    """
    Scores each (system_prompt, query, response) with score_fn in parallel.
    items[i] = (prompt, query, response). Returns list of scores in order.
    Use for parallel Critic evaluation over many triples.
    """
    if not items:
        return []

    def fn(item: tuple[str, str, str]) -> float:
        return score_fn(item[0], item[1], item[2])

    return map_parallel(fn, items, max_workers=max_workers)
