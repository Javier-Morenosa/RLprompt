"""Tests for core primitives: PerceptionCycle, HybridReward, word_change_ratio."""

import pytest
from prompt_rl.core.cycle import PerceptionCycle
from prompt_rl.rl.reward import HybridReward, word_change_ratio


# ── PerceptionCycle ───────────────────────────────────────────────────────────

class TestPerceptionCycle:
    def _make(self, verdict="CORRECTO", comment=""):
        return PerceptionCycle(
            system_prompt="Be helpful.",
            user_query="What is 2+2?",
            bot_response="4",
            verdict=verdict,
            comment=comment,
        )

    def test_is_correct_true(self):
        assert self._make(verdict="CORRECTO").is_correct is True

    def test_is_correct_false(self):
        assert self._make(verdict="INCORRECTO").is_correct is False

    def test_has_correction_with_comment(self):
        assert self._make(comment="The answer is 5").has_correction is True

    def test_has_correction_empty(self):
        assert self._make(comment="").has_correction is False

    def test_has_correction_whitespace_only(self):
        assert self._make(comment="   ").has_correction is False

    def test_observations_default_empty(self):
        cycle = self._make()
        assert cycle.observations == []

    def test_dwell_seconds_default(self):
        cycle = self._make()
        assert cycle.dwell_seconds == 0.0

    def test_timestamp_set(self):
        cycle = self._make()
        assert cycle.timestamp != ""


# ── word_change_ratio ─────────────────────────────────────────────────────────

class TestWordChangeRatio:
    def test_identical_strings(self):
        assert word_change_ratio("hello world", "hello world") == 0.0

    def test_completely_different(self):
        ratio = word_change_ratio("one two three", "four five six")
        assert ratio == 1.0

    def test_empty_old(self):
        # old is empty → ratio should be 1.0 (everything changed)
        ratio = word_change_ratio("", "new text here")
        assert 0.0 <= ratio <= 1.0

    def test_partial_change(self):
        ratio = word_change_ratio("hello world foo", "hello world bar")
        assert 0.0 < ratio < 1.0

    def test_result_in_range(self):
        ratio = word_change_ratio("a b c d e", "a b c x y")
        assert 0.0 <= ratio <= 1.0

    def test_single_word_changed(self):
        ratio = word_change_ratio("hello world", "hello there")
        assert ratio > 0.0

    def test_single_word_same(self):
        ratio = word_change_ratio("hello", "hello")
        assert ratio == 0.0


# ── HybridReward ──────────────────────────────────────────────────────────────

class TestHybridReward:
    def setup_method(self):
        self.reward = HybridReward(
            lambda_feedback=0.9,
            lambda_critic=0.1,
            lambda_change=0.3,
        )

    def test_correcto_zero_change_positive_reward(self):
        R = self.reward.compute(
            human_feedback=1.0,
            critic_score=1.0,
            current_prompt="same prompt",
            proposed_prompt="same prompt",
        )
        assert R > 0.0

    def test_incorrecto_with_change_negative_reward(self):
        # feedback=0, critic=0, and prompt has changed → penalty makes R < 0
        R = self.reward.compute(
            human_feedback=0.0,
            critic_score=0.0,
            current_prompt="a b c d e",
            proposed_prompt="x y z w v",
        )
        assert R < 0.0

    def test_large_change_lowers_reward(self):
        R_small = self.reward.compute(
            human_feedback=1.0,
            critic_score=1.0,
            current_prompt="a b c d e",
            proposed_prompt="a b c d f",   # 1/5 words changed
        )
        R_large = self.reward.compute(
            human_feedback=1.0,
            critic_score=1.0,
            current_prompt="a b c d e",
            proposed_prompt="x y z w v",   # 5/5 words changed
        )
        assert R_small > R_large

    def test_rewrite_blocked_above_threshold(self):
        old = " ".join(["word"] * 20)
        new = " ".join(["new"] * 20)
        blocked = self.reward.is_rewrite_blocked(old, new)
        assert blocked is True

    def test_rewrite_not_blocked_small_change(self):
        old = "You are a helpful assistant. Answer clearly."
        new = "You are a helpful assistant. Answer clearly and concisely."
        blocked = self.reward.is_rewrite_blocked(old, new)
        assert blocked is False

    def test_lambda_feedback_dominates(self):
        # With lambda_feedback=0.9 human signal should dominate
        R_correct = self.reward.compute(
            human_feedback=1.0,
            critic_score=0.0,
            current_prompt="p",
            proposed_prompt="p",
        )
        R_incorrect = self.reward.compute(
            human_feedback=0.0,
            critic_score=1.0,
            current_prompt="p",
            proposed_prompt="p",
        )
        assert R_correct > R_incorrect
