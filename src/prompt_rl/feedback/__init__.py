"""Human feedback signals: conversion utilities and aggregation."""

from prompt_rl.feedback.signals import (
    FeedbackAggregator,
    FeedbackType,
    thumbs_to_score,
    rating_to_score,
    reading_time_to_score,
)
from prompt_rl.feedback.cursor_trace import (
    CursorTrace,
    HumanFeedbackResult,
)

__all__ = [
    "FeedbackAggregator",
    "FeedbackType",
    "thumbs_to_score",
    "rating_to_score",
    "reading_time_to_score",
    "CursorTrace",
    "HumanFeedbackResult",
]
