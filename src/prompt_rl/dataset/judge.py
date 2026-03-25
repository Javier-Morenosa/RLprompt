"""DatasetJudge — deterministic judges for offline prompt optimization.

Unlike :class:`~prompt_rl.validation.judge.LLMValidationJudge`, these judges
use no LLM calls: correctness is determined by exact or pattern matching
against the ground-truth answer stored in the :class:`DatasetSample`.

Each judge implements two interfaces:

* **DatasetJudge** (outer loop): ``is_correct(response, sample)`` and
  ``feedback(response, sample)`` — used by :class:`DatasetLoop` to decide
  the verdict for each training sample.

* **ValidationJudge** (inner loop): ``judge(user_query, original_feedback,
  original_response, new_response)`` — used by
  :class:`~prompt_rl.validation.loop.CriticValidationLoop` to verify that a
  proposed new prompt actually fixes the wrong answer before committing it.

Call :meth:`set_sample` before each sample when using as a ``ValidationJudge``
inside a ``CriticValidationLoop``.
"""

from __future__ import annotations

import re
from typing import Protocol, runtime_checkable

from prompt_rl.dataset.sample import DatasetSample
from prompt_rl.validation.judge import ValidationResult


@runtime_checkable
class DatasetJudge(Protocol):
    """Protocol for deterministic dataset judges."""

    def is_correct(self, response: str, sample: DatasetSample) -> bool:
        """Return True if *response* matches the ground truth in *sample*."""
        ...

    def feedback(self, response: str, sample: DatasetSample) -> str:
        """Return a generic correction hint for the Critic (no ground truth leaked)."""
        ...


class ExactMatchJudge:
    """Deterministic judge based on normalized exact-string matching.

    Suitable for datasets with unambiguous numeric or short-text answers
    (e.g. GSM8K, arithmetic, multiple-choice with single letter answers).

    The comparison pipeline is:
        1. Extract the relevant fragment from both *response* and *sample.answer*
           using *extract_pattern* (if provided).
        2. Normalize both fragments with *normalize_fn* (strip, lower,
           remove punctuation and whitespace by default).
        3. Compare the normalized strings.

    Also implements ``ValidationJudge`` for use inside
    :class:`~prompt_rl.validation.loop.CriticValidationLoop`. Call
    :meth:`set_sample` with the current sample before calling
    :meth:`~prompt_rl.validation.loop.CriticValidationLoop.evaluate`.

    Args:
        extract_pattern : Regex with one capture group used to extract the
                          answer from a longer response. Defaults to the GSM8K
                          format ``#### <answer>``. Set to ``None`` to use the
                          full response text.
        normalize_fn    : Callable ``str -> str`` applied to both response and
                          ground truth before comparison. Defaults to
                          lower-case + remove non-numeric characters.
        feedback_msg    : Fixed string returned by :meth:`feedback`. Should
                          NOT contain the ground truth answer.
    """

    _DEFAULT_FEEDBACK = (
        "La respuesta fue incorrecta. "
        "Mejora el prompt para que el modelo razone mejor paso a paso."
    )

    def __init__(
        self,
        extract_pattern: str | None = r"####\s*(.+)$",
        normalize_fn: "callable[[str], str] | None" = None,
        feedback_msg: str = _DEFAULT_FEEDBACK,
        include_ground_truth: bool = True,
    ) -> None:
        self._pattern = (
            re.compile(extract_pattern, re.MULTILINE) if extract_pattern else None
        )
        self._normalize = normalize_fn or self._default_normalize
        self._feedback_msg = feedback_msg
        self._include_ground_truth = include_ground_truth
        self._current_sample: DatasetSample | None = None

    @staticmethod
    def _default_normalize(text: str) -> str:
        text = text.strip().lower()
        # Keep only digits, decimal points and minus sign
        text = re.sub(r"[,\s]+", "", text)
        text = re.sub(r"[^\d.\-]", "", text)
        return text

    def _extract(self, text: str) -> str:
        if self._pattern:
            m = self._pattern.search(text)
            if m:
                return m.group(1).strip()
        return text.strip()

    # ── DatasetJudge interface ────────────────────────────────────────────────

    def is_correct(self, response: str, sample: DatasetSample) -> bool:
        gt = sample.extracted or self._extract(sample.answer)
        pred = self._extract(response)
        return self._normalize(pred) == self._normalize(gt)

    def feedback(self, response: str, sample: DatasetSample) -> str:
        if not self._include_ground_truth:
            return self._feedback_msg
        model_answer = self._extract(response)
        correct_answer = sample.extracted or self._extract(sample.answer)
        return (
            f"{self._feedback_msg} "
            f"El modelo respondio '{model_answer}' pero la respuesta correcta es '{correct_answer}'."
        )

    # ── ValidationJudge interface (for CriticValidationLoop) ─────────────────

    def set_sample(self, sample: DatasetSample) -> None:
        """Set the current sample used by :meth:`judge` for validation calls."""
        self._current_sample = sample

    def judge(
        self,
        user_query: str,
        original_feedback: str,
        original_response: str,
        new_response: str,
    ) -> ValidationResult:
        """Check if *new_response* is correct for the current sample.

        Raises:
            RuntimeError: if :meth:`set_sample` has not been called.
        """
        if self._current_sample is None:
            raise RuntimeError(
                "Call ExactMatchJudge.set_sample(sample) before using as ValidationJudge."
            )
        fixed = self.is_correct(new_response, self._current_sample)
        return ValidationResult(fixed=fixed, reasoning="exact-match")


class ContainsMatchJudge:
    """Judge that checks whether the response *contains* the expected answer.

    Useful for open-ended questions where the answer may appear anywhere in a
    longer response (e.g. multiple-choice with explanation, fill-in-the-blank).

    The comparison is case-insensitive by default. Normalization is applied to
    both the extracted ground truth and the full response text.

    Args:
        extract_pattern : Regex to extract the expected answer from
                          *sample.answer*. ``None`` uses the full answer.
        case_sensitive  : Whether the contains check is case-sensitive
                          (default ``False``).
        feedback_msg    : Fixed hint returned by :meth:`feedback`.
    """

    _DEFAULT_FEEDBACK = (
        "La respuesta fue incorrecta. "
        "Asegurate de que el modelo incluya la respuesta correcta de forma explicita."
    )

    def __init__(
        self,
        extract_pattern: str | None = None,
        case_sensitive: bool = False,
        feedback_msg: str = _DEFAULT_FEEDBACK,
        include_ground_truth: bool = True,
    ) -> None:
        self._pattern = (
            re.compile(extract_pattern, re.MULTILINE) if extract_pattern else None
        )
        self._case_sensitive = case_sensitive
        self._feedback_msg = feedback_msg
        self._include_ground_truth = include_ground_truth
        self._current_sample: DatasetSample | None = None

    def _extract(self, text: str) -> str:
        if self._pattern:
            m = self._pattern.search(text)
            if m:
                return m.group(1).strip()
        return text.strip()

    def _normalize(self, text: str) -> str:
        return text if self._case_sensitive else text.lower()

    # ── DatasetJudge interface ────────────────────────────────────────────────

    def is_correct(self, response: str, sample: DatasetSample) -> bool:
        gt = sample.extracted or self._extract(sample.answer)
        return self._normalize(gt) in self._normalize(response)

    def feedback(self, response: str, sample: DatasetSample) -> str:
        if not self._include_ground_truth:
            return self._feedback_msg
        correct_answer = sample.extracted or self._extract(sample.answer)
        return (
            f"{self._feedback_msg} "
            f"La respuesta correcta es '{correct_answer}'."
        )

    # ── ValidationJudge interface ─────────────────────────────────────────────

    def set_sample(self, sample: DatasetSample) -> None:
        self._current_sample = sample

    def judge(
        self,
        user_query: str,
        original_feedback: str,
        original_response: str,
        new_response: str,
    ) -> ValidationResult:
        if self._current_sample is None:
            raise RuntimeError(
                "Call ContainsMatchJudge.set_sample(sample) before using as ValidationJudge."
            )
        fixed = self.is_correct(new_response, self._current_sample)
        return ValidationResult(fixed=fixed, reasoning="contains-match")
