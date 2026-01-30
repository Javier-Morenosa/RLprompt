"""
Gradio interface for collecting human feedback.

Supports:
- Explicit: thumbs up/down, rating (1-5)
- Preference: A vs B comparison (two responses, choose one)

Use standalone (launch demo) or integrated via HumanFeedbackCollector
and get_feedback() for the training loop.
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from prompt_rl.feedback.signals import (
    FeedbackAggregator,
    thumbs_to_score,
    rating_to_score,
)


@dataclass
class FeedbackResult:
    """Result of one feedback submission from the UI."""

    score: float
    thumbs_up: Optional[bool] = None
    rating: Optional[float] = None
    preference_index: Optional[int] = None
    raw: dict[str, Any] = field(default_factory=dict)


def _check_gradio() -> None:
    try:
        import gradio as gr  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Gradio is required for the feedback UI. Install with: pip install prompt-rl[gradio]"
        ) from e


def create_feedback_interface(
    aggregator: Optional[FeedbackAggregator] = None,
    title: str = "Human Feedback",
    description: str = "Rate the model response to improve the system.",
) -> Any:
    """
    Creates a Gradio Blocks interface for collecting feedback.

    Returns a gr.Blocks app. Launch with app.launch() or use with
    HumanFeedbackCollector for queue-based integration with the training loop.

    Args:
        aggregator: Optional FeedbackAggregator for combining signals.
        title: App title.
        description: Short description shown in the UI.
    """
    _check_gradio()
    import gradio as gr

    agg = aggregator or FeedbackAggregator()

    def submit_feedback(
        query: str,
        response: str,
        thumbs: Optional[str],
        rating: Optional[float],
        response_b: str,
        preference: Optional[str],
    ) -> str:
        explicit = None
        if thumbs is not None and thumbs.strip():
            explicit = thumbs_to_score(thumbs.strip().lower() == "thumbs up")
        if rating is not None and rating >= 1:
            rn = rating_to_score(float(rating), 1.0, 5.0)
            explicit = rn if explicit is None else (explicit + rn) / 2

        pref_index = None
        if preference is not None and preference.strip() and (response.strip() or response_b.strip()):
            pref_index = 0 if preference.strip().upper() == "A" else 1

        score = agg.aggregate(
            explicit=explicit,
            implicit=None,
            preference_chosen_index=pref_index,
            num_preference_candidates=2 if (response.strip() and response_b.strip()) else 1,
        )
        if explicit is None and pref_index is None:
            score = 0.0
        return f"**Thank you.** Aggregated score: **{score:.2f}**"

    with gr.Blocks(title=title, css=".feedback-box { max-width: 800px; }") as app:
        gr.Markdown(f"## {title}\n{description}")
        with gr.Row():
            query_box = gr.Textbox(
                label="Query",
                placeholder="User query shown to the model",
                lines=2,
            )
        with gr.Row():
            response_a = gr.Textbox(
                label="Response A",
                placeholder="Model response to rate",
                lines=6,
            )
            response_b = gr.Textbox(
                label="Response B (optional, for A vs B)",
                placeholder="Second response if comparing",
                lines=6,
            )
        with gr.Row():
            thumbs = gr.Radio(
                choices=["Thumbs up", "Thumbs down"],
                label="Quick rating",
                value=None,
            )
            rating = gr.Slider(
                minimum=1,
                maximum=5,
                step=1,
                value=0,
                label="Rating (1-5)",
            )
            preference = gr.Radio(
                choices=["A", "B"],
                label="Prefer (A vs B)",
                value=None,
            )
        submit_btn = gr.Button("Submit feedback")
        out_msg = gr.Markdown("")
        submit_btn.click(
            fn=submit_feedback,
            inputs=[query_box, response_a, thumbs, rating, response_b, preference],
            outputs=[out_msg],
        )
    return app


def _run_app_with_queues(
    task_queue: queue.Queue,
    result_queue: queue.Queue,
    aggregator: Optional[FeedbackAggregator] = None,
    server_name: str = "0.0.0.0",
    server_port: Optional[int] = None,
) -> None:
    _check_gradio()
    import gradio as gr

    agg = aggregator or FeedbackAggregator()
    current = {"query": "", "response": "", "response_b": ""}

    def load_next() -> tuple[str, str, str]:
        try:
            item = task_queue.get(timeout=0.5)
            if isinstance(item, tuple):
                if len(item) == 2:
                    q, r = item
                    current["query"], current["response"], current["response_b"] = q, r, ""
                    return q, r, ""
                if len(item) >= 3:
                    q, r, rb = item[0], item[1], item[2]
                    current["query"], current["response"], current["response_b"] = q, r, rb
                    return q, r, rb
            return current["query"], current["response"], current["response_b"]
        except queue.Empty:
            return current["query"], current["response"], current["response_b"]

    def submit_feedback(thumbs: Optional[str], rating: Optional[float], preference: Optional[str]) -> str:
        explicit = None
        if thumbs and thumbs.strip():
            explicit = thumbs_to_score(thumbs.strip().lower() == "thumbs up")
        if rating is not None and rating >= 1:
            rn = rating_to_score(float(rating), 1.0, 5.0)
            explicit = rn if explicit is None else (explicit + rn) / 2
        pref_index = None
        if preference and current["response_b"]:
            pref_index = 0 if preference.strip().upper() == "A" else 1
        score = agg.aggregate(
            explicit=explicit,
            implicit=None,
            preference_chosen_index=pref_index,
            num_preference_candidates=2 if current["response_b"] else 1,
        )
        if explicit is None and pref_index is None:
            score = 0.0
        thumbs_up = None
        if thumbs and thumbs.strip():
            thumbs_up = thumbs.strip().lower() == "thumbs up"
        result_queue.put(FeedbackResult(score=score, thumbs_up=thumbs_up, rating=rating, preference_index=pref_index))
        return f"Thanks. Score: {score:.2f}"

    with gr.Blocks(title="Human Feedback") as app:
        gr.Markdown("## Human Feedback\nRate the response(s) to improve the system.")
        query_box = gr.Textbox(label="Query", lines=2, interactive=False)
        response_a = gr.Textbox(label="Response A", lines=6, interactive=False)
        response_b = gr.Textbox(label="Response B (optional)", lines=6, interactive=False)
        load_btn = gr.Button("Load next task")
        load_btn.click(fn=load_next, inputs=[], outputs=[query_box, response_a, response_b])
        thumbs = gr.Radio(choices=["Thumbs up", "Thumbs down"], label="Quick rating")
        rating = gr.Slider(1, 5, step=1, value=0, label="Rating (1-5)")
        preference = gr.Radio(choices=["A", "B"], label="Prefer (A vs B)")
        submit_btn = gr.Button("Submit feedback")
        out_msg = gr.Markdown("")
        submit_btn.click(
            fn=submit_feedback,
            inputs=[thumbs, rating, preference],
            outputs=[out_msg],
        )
    app.queue()
    app.launch(server_name=server_name, server_port=server_port or 7860)


class HumanFeedbackCollector:
    """
    Collects human feedback via a Gradio UI running in a background thread.
    Use get_feedback(query, response) or get_feedback_preference(query, response_a, response_b)
    to push a task to the UI and block until the user submits feedback.
    """

    def __init__(
        self,
        aggregator: Optional[FeedbackAggregator] = None,
        server_name: str = "0.0.0.0",
        server_port: Optional[int] = None,
    ) -> None:
        _check_gradio()
        self.aggregator = aggregator or FeedbackAggregator()
        self._task_queue: queue.Queue = queue.Queue()
        self._result_queue: queue.Queue = queue.Queue()
        self._server_name = server_name
        self._server_port = server_port
        self._thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        """Starts the Gradio UI in a background thread."""
        if self._started:
            return
        self._thread = threading.Thread(
            target=_run_app_with_queues,
            args=(self._task_queue, self._result_queue),
            kwargs={"aggregator": self.aggregator, "server_name": self._server_name, "server_port": self._server_port},
            daemon=True,
        )
        self._thread.start()
        self._started = True

    def get_feedback(self, query: str, response: str) -> FeedbackResult:
        """
        Submits (query, response) to the UI and blocks until the user submits feedback.
        Call start() before the first get_feedback.
        """
        if not self._started:
            self.start()
        self._task_queue.put((query, response))
        return self._result_queue.get()

    def get_feedback_preference(self, query: str, response_a: str, response_b: str) -> FeedbackResult:
        """Submits an A vs B task and blocks until the user chooses and submits."""
        if not self._started:
            self.start()
        self._task_queue.put((query, response_a, response_b))
        return self._result_queue.get()


def launch_standalone(
    aggregator: Optional[FeedbackAggregator] = None,
    server_name: str = "127.0.0.1",
    server_port: Optional[int] = None,
) -> None:
    """
    Launches a standalone Gradio app for feedback (no queues).
    User enters query/response manually and submits ratings.
    """
    _check_gradio()
    app = create_feedback_interface(aggregator=aggregator)
    app.queue()
    app.launch(server_name=server_name, server_port=server_port or 7860)
