"""Human feedback signals: conversion and aggregation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class FeedbackType(Enum):
    EXPLICIT   = "explicit"
    IMPLICIT   = "implicit"
    PREFERENCE = "preference"


# ── Score converters ──────────────────────────────────────────────────────────

def thumbs_to_score(thumbs_up: bool) -> float:
    """
    Thumbs up -> +1.0, thumbs down -> -1.0.

    The [-1, +1] range is intentional: an incorrect response is a real
    penalty, not neutral.  HybridReward and FeedbackAggregator both
    accept negative values from this signal.
    """
    return 1.0 if thumbs_up else -1.0


def rating_to_score(rating: float, min_r: float = 1.0, max_r: float = 5.0) -> float:
    """Normalise a 1–5 rating to [0, 1]."""
    return (rating - min_r) / (max_r - min_r) if max_r > min_r else 0.0


def reading_time_to_score(seconds: float, good_threshold: float = 30.0) -> float:
    """
    Implicit score from reading/dwell time.
    Too short (skimmed) or too long (confused) both lower the score.
    """
    if seconds <= 0:
        return 0.0
    if seconds <= good_threshold:
        return min(1.0, seconds / good_threshold)
    return max(0.0, 1.0 - (seconds - good_threshold) / 60.0)


# ── Aggregator ────────────────────────────────────────────────────────────────

@dataclass
class FeedbackAggregator:
    """
    Weighted combination of explicit, implicit, and preference signals
    into a single human_feedback score in [0, 1].

    Signals that are not provided (None) do not contribute to the result.
    """

    weights: dict[FeedbackType, float] = field(default_factory=lambda: {
        FeedbackType.EXPLICIT:   1.0,
        FeedbackType.PREFERENCE: 0.9,
        FeedbackType.IMPLICIT:   0.5,
    })

    def aggregate(
        self,
        explicit:                   Optional[float] = None,
        implicit:                   Optional[float] = None,
        preference_chosen_index:    Optional[int]   = None,
        num_preference_candidates:  int             = 2,
    ) -> float:
        total      = 0.0
        weight_sum = 0.0
        if explicit is not None:
            total      += self.weights[FeedbackType.EXPLICIT] * explicit
            weight_sum += self.weights[FeedbackType.EXPLICIT]
        if implicit is not None:
            total      += self.weights[FeedbackType.IMPLICIT] * implicit
            weight_sum += self.weights[FeedbackType.IMPLICIT]
        if preference_chosen_index is not None:
            total      += self.weights[FeedbackType.PREFERENCE] * 1.0
            weight_sum += self.weights[FeedbackType.PREFERENCE]
        return total / weight_sum if weight_sum > 0 else 0.0
