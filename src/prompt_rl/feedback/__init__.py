"""
Human Feedback Loop: explicit, implicit and preference signals.

- Explicit: thumbs up/down, ratings, corrections
- Implicit: reading time, re-prompts, acceptance
- Preferences: A vs B comparisons

Optional Gradio UI: install with pip install prompt-rl[gradio], then use
create_feedback_interface, HumanFeedbackCollector, launch_standalone.
"""

from prompt_rl.feedback.signals import (
    HumanFeedback,
    ExplicitFeedback,
    ImplicitFeedback,
    PreferenceFeedback,
    FeedbackAggregator,
)

__all__ = [
    "HumanFeedback",
    "ExplicitFeedback",
    "ImplicitFeedback",
    "PreferenceFeedback",
    "FeedbackAggregator",
]

try:
    from prompt_rl.feedback.gradio_ui import (
        create_feedback_interface,
        HumanFeedbackCollector,
        FeedbackResult,
        launch_standalone,
    )
    __all__ += ["create_feedback_interface", "HumanFeedbackCollector", "FeedbackResult", "launch_standalone"]
except ImportError:
    pass
