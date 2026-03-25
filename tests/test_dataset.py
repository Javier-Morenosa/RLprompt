"""Tests for dataset primitives: DatasetSample, DatasetSplit, ExactMatchJudge, ContainsMatchJudge."""

import pytest
from prompt_rl.dataset.sample import DatasetSample, DatasetSplit
from prompt_rl.dataset.judge import ExactMatchJudge, ContainsMatchJudge


# ── DatasetSample ─────────────────────────────────────────────────────────────

class TestDatasetSample:
    def test_basic_fields(self):
        s = DatasetSample(question="2+2?", answer="#### 4", extracted="4")
        assert s.question == "2+2?"
        assert s.answer == "#### 4"
        assert s.extracted == "4"

    def test_extracted_defaults_empty(self):
        s = DatasetSample(question="q", answer="a")
        assert s.extracted == ""

    def test_metadata_defaults_empty(self):
        s = DatasetSample(question="q", answer="a")
        assert s.metadata == {}

    def test_metadata_stored(self):
        s = DatasetSample(question="q", answer="a", metadata={"source": "gsm8k"})
        assert s.metadata["source"] == "gsm8k"


# ── DatasetSplit ──────────────────────────────────────────────────────────────

class TestDatasetSplit:
    def _samples(self, n=10):
        return [DatasetSample(question=f"q{i}", answer=f"a{i}") for i in range(n)]

    def test_from_list_sizes(self):
        split = DatasetSplit.from_list(self._samples(10), train_ratio=0.8, seed=42)
        assert len(split.train) == 8
        assert len(split.test) == 2

    def test_from_list_total_preserved(self):
        samples = self._samples(10)
        split = DatasetSplit.from_list(samples, train_ratio=0.7, seed=42)
        assert len(split.train) + len(split.test) == 10

    def test_from_list_reproducible(self):
        samples = self._samples(20)
        s1 = DatasetSplit.from_list(samples, seed=42)
        s2 = DatasetSplit.from_list(samples, seed=42)
        assert [s.question for s in s1.train] == [s.question for s in s2.train]

    def test_from_list_different_seeds(self):
        samples = self._samples(20)
        s1 = DatasetSplit.from_list(samples, seed=1)
        s2 = DatasetSplit.from_list(samples, seed=2)
        # Very unlikely to be identical with different seeds
        assert [s.question for s in s1.train] != [s.question for s in s2.train]

    def test_from_list_invalid_ratio(self):
        with pytest.raises(ValueError):
            DatasetSplit.from_list(self._samples(), train_ratio=1.5)

    def test_from_list_zero_ratio(self):
        with pytest.raises(ValueError):
            DatasetSplit.from_list(self._samples(), train_ratio=0.0)

    def test_from_dicts(self):
        train = [{"question": "q1", "answer": "a1", "extracted": "1"}]
        test  = [{"question": "q2", "answer": "a2", "extracted": "2"}]
        split = DatasetSplit.from_dicts(train, test)
        assert split.train[0].question == "q1"
        assert split.test[0].extracted == "2"

    def test_from_dicts_custom_keys(self):
        train = [{"q": "question", "a": "answer"}]
        test  = [{"q": "q2",       "a": "a2"}]
        split = DatasetSplit.from_dicts(train, test, question_key="q", answer_key="a")
        assert split.train[0].question == "question"

    def test_from_dicts_metadata_captured(self):
        train = [{"question": "q", "answer": "a", "difficulty": "hard"}]
        split = DatasetSplit.from_dicts(train, [])
        assert split.train[0].metadata["difficulty"] == "hard"


# ── ExactMatchJudge ───────────────────────────────────────────────────────────

class TestExactMatchJudge:
    def setup_method(self):
        self.judge = ExactMatchJudge(extract_pattern=r"####\s*(.+)$")

    def _sample(self, extracted="42"):
        return DatasetSample(question="q", answer=f"#### {extracted}", extracted=extracted)

    # is_correct
    def test_correct_exact(self):
        assert self.judge.is_correct("The answer is #### 42", self._sample("42"))

    def test_incorrect(self):
        assert not self.judge.is_correct("The answer is #### 99", self._sample("42"))

    def test_correct_strips_whitespace(self):
        # Leading/trailing whitespace around the extracted number should not matter
        assert self.judge.is_correct("####  42 ", self._sample("42"))

    def test_correct_with_comma_in_number(self):
        # "1,000" and "1000" should match (commas stripped by normalization)
        assert self.judge.is_correct("#### 1,000", self._sample("1000"))

    def test_correct_case_insensitive_number(self):
        assert self.judge.is_correct("#### 42", self._sample("42"))

    def test_uses_extracted_field_over_answer(self):
        sample = DatasetSample(question="q", answer="#### 99", extracted="42")
        assert self.judge.is_correct("#### 42", sample)

    def test_no_pattern_uses_full_response(self):
        judge = ExactMatchJudge(extract_pattern=None)
        sample = DatasetSample(question="q", answer="Paris", extracted="Paris")
        assert judge.is_correct("Paris", sample)

    # feedback with ground truth
    def test_feedback_includes_model_answer(self):
        feedback = self.judge.feedback("#### 5", self._sample("7"))
        assert "5" in feedback

    def test_feedback_includes_correct_answer(self):
        feedback = self.judge.feedback("#### 5", self._sample("7"))
        assert "7" in feedback

    def test_feedback_without_ground_truth(self):
        judge = ExactMatchJudge(include_ground_truth=False)
        feedback = judge.feedback("#### 5", self._sample("7"))
        assert "7" not in feedback

    def test_feedback_custom_message(self):
        judge = ExactMatchJudge(feedback_msg="Custom message.")
        feedback = judge.feedback("#### 5", self._sample("7"))
        assert "Custom message." in feedback

    # ValidationJudge interface
    def test_validation_judge_correct(self):
        sample = self._sample("42")
        self.judge.set_sample(sample)
        result = self.judge.judge("q", "fb", "old", "#### 42")
        assert result.fixed is True

    def test_validation_judge_incorrect(self):
        sample = self._sample("42")
        self.judge.set_sample(sample)
        result = self.judge.judge("q", "fb", "old", "#### 99")
        assert result.fixed is False

    def test_validation_judge_raises_without_set_sample(self):
        judge = ExactMatchJudge()
        with pytest.raises(RuntimeError, match="set_sample"):
            judge.judge("q", "fb", "old", "new")


# ── ContainsMatchJudge ────────────────────────────────────────────────────────

class TestContainsMatchJudge:
    def setup_method(self):
        self.judge = ContainsMatchJudge()

    def _sample(self, extracted="Paris"):
        return DatasetSample(question="q", answer=extracted, extracted=extracted)

    def test_correct_contains(self):
        assert self.judge.is_correct("The capital is Paris, located in France.", self._sample("Paris"))

    def test_incorrect_not_contains(self):
        assert not self.judge.is_correct("The capital is London.", self._sample("Paris"))

    def test_case_insensitive_by_default(self):
        assert self.judge.is_correct("the capital is paris.", self._sample("Paris"))

    def test_case_sensitive_mode(self):
        judge = ContainsMatchJudge(case_sensitive=True)
        assert not judge.is_correct("the capital is paris.", self._sample("Paris"))
        assert judge.is_correct("the capital is Paris.", self._sample("Paris"))

    def test_feedback_includes_correct_answer(self):
        feedback = self.judge.feedback("London", self._sample("Paris"))
        assert "Paris" in feedback

    def test_feedback_without_ground_truth(self):
        judge = ContainsMatchJudge(include_ground_truth=False)
        feedback = judge.feedback("London", self._sample("Paris"))
        assert "Paris" not in feedback

    def test_validation_judge_correct(self):
        sample = self._sample("Paris")
        self.judge.set_sample(sample)
        result = self.judge.judge("q", "fb", "old", "The answer is Paris.")
        assert result.fixed is True

    def test_validation_judge_incorrect(self):
        sample = self._sample("Paris")
        self.judge.set_sample(sample)
        result = self.judge.judge("q", "fb", "old", "The answer is London.")
        assert result.fixed is False

    def test_validation_judge_raises_without_set_sample(self):
        judge = ContainsMatchJudge()
        with pytest.raises(RuntimeError, match="set_sample"):
            judge.judge("q", "fb", "old", "new")
