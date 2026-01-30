"""Human feedback signals: explicit, implicit, preferences."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional


class FeedbackType(Enum):
    EXPLICIT = "explicit"
    IMPLICIT = "implicit"
    PREFERENCE = "preference"


@dataclass
class HumanFeedback:
    """Base human feedback signal."""

    feedback_type: FeedbackType
    value: float
    raw: Optional[dict[str, Any]] = None


@dataclass
class ExplicitFeedback(HumanFeedback):
    """Explicit: thumbs up/down, ratings (1-5), corrections."""

    def __post_init__(self) -> None:
        self.feedback_type = FeedbackType.EXPLICIT


def thumbs_to_score(thumbs_up: bool) -> float:
    """Converts thumbs up/down to score in [0, 1]."""
    return 1.0 if thumbs_up else 0.0


def rating_to_score(rating: float, min_r: float = 1.0, max_r: float = 5.0) -> float:
    """Normalizes rating (e.g. 1-5) to [0, 1]."""
    return (rating - min_r) / (max_r - min_r) if max_r > min_r else 0.0


@dataclass
class ImplicitFeedback(HumanFeedback):
    """Implicit: reading time, re-prompts, acceptance (no correction)."""

    def __post_init__(self) -> None:
        self.feedback_type = FeedbackType.IMPLICIT


def reading_time_to_score(seconds: float, good_threshold: float = 30.0) -> float:
    """Reading time: very short or very long may indicate disinterest."""
    if seconds <= 0:
        return 0.0
    if seconds <= good_threshold:
        return min(1.0, seconds / good_threshold)
    return max(0.0, 1.0 - (seconds - good_threshold) / 60.0)


@dataclass
class PreferenceFeedback(HumanFeedback):
    """Preference: A vs B (which the user chose)."""

    chosen_index: int = 0
    candidates: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.feedback_type = FeedbackType.PREFERENCE
        self.value = 1.0


def preference_to_scores(chosen_index: int, num_candidates: int = 2) -> list[float]:
    """Converts A vs B preference into per-candidate scores."""
    scores = [0.0] * num_candidates
    if 0 <= chosen_index < num_candidates:
        scores[chosen_index] = 1.0
    return scores


@dataclass
class FeedbackAggregator:
    """
    Aggregates multiple signals into a single score for reward.
    Weights per type: explicit > preference > implicit (configurable).
    """

    weights: dict[FeedbackType, float] = field(default_factory=lambda: {
        FeedbackType.EXPLICIT: 1.0,
        FeedbackType.PREFERENCE: 0.9,
        FeedbackType.IMPLICIT: 0.5,
    })

    def add(self, fb: HumanFeedback) -> None:
        """Registers a feedback (optional: internal buffer)."""
        pass

    def aggregate(
        self,
        explicit: Optional[float] = None,
        implicit: Optional[float] = None,
        preference_chosen_index: Optional[int] = None,
        num_preference_candidates: int = 2,
    ) -> float:
        """
        Combines signals into a score [0, 1].
        If a signal is not provided, it does not contribute.
        """
        total = 0.0
        weight_sum = 0.0
        if explicit is not None:
            total += self.weights[FeedbackType.EXPLICIT] * explicit
            weight_sum += self.weights[FeedbackType.EXPLICIT]
        if implicit is not None:
            total += self.weights[FeedbackType.IMPLICIT] * implicit
            weight_sum += self.weights[FeedbackType.IMPLICIT]
        if preference_chosen_index is not None:
            s = 1.0
            total += self.weights[FeedbackType.PREFERENCE] * s
            weight_sum += self.weights[FeedbackType.PREFERENCE]
        if weight_sum <= 0:
            return 0.0
        return total / weight_sum
