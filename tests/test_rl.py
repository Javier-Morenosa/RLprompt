"""Tests for RL components: RewardHistory, UpdateGate, AlwaysUpdateGate."""

import json
import tempfile
from pathlib import Path

import pytest
from prompt_rl.rl.history import RewardHistory
from prompt_rl.rl.gate import UpdateGate, AlwaysUpdateGate, GateResult


# ── RewardHistory ─────────────────────────────────────────────────────────────

class TestRewardHistory:
    def setup_method(self):
        self.h = RewardHistory(
            window_size=5,
            convergence_window=3,
            convergence_change_eps=0.05,
        )

    def _append_correcto(self, R=0.8, change_ratio=0.01):
        self.h.append(R=R, verdict="CORRECTO", critic_score=0.9, change_ratio=change_ratio)

    def _append_incorrecto(self, R=-0.1):
        self.h.append(R=R, verdict="INCORRECTO", critic_score=0.3, change_ratio=0.2)

    # convergence
    def test_no_convergence_initially(self):
        assert self.h.converged is False
        assert self.h.consecutive_stable == 0

    def test_convergence_after_n_stable(self):
        for _ in range(3):
            self._append_correcto()
        assert self.h.converged is True
        assert self.h.consecutive_stable == 3

    def test_convergence_reset_by_incorrecto(self):
        for _ in range(2):
            self._append_correcto()
        self._append_incorrecto()
        assert self.h.converged is False
        assert self.h.consecutive_stable == 0

    def test_convergence_reset_by_large_change(self):
        # change_ratio >= eps → not stable
        for _ in range(3):
            self.h.append(R=0.8, verdict="CORRECTO", critic_score=0.9, change_ratio=0.10)
        assert self.h.converged is False

    def test_bump_version_resets_convergence(self):
        for _ in range(3):
            self._append_correcto()
        assert self.h.converged is True
        self.h.bump_version()
        assert self.h.converged is False
        assert self.h.consecutive_stable == 0

    def test_bump_version_increments(self):
        assert self.h.version == 0
        self.h.bump_version()
        assert self.h.version == 1
        self.h.bump_version()
        assert self.h.version == 2

    # rolling window
    def test_window_capped(self):
        for i in range(10):
            self._append_correcto(R=float(i))
        assert len(self.h.entries) == 5

    def test_window_keeps_latest(self):
        for i in range(7):
            self._append_correcto(R=float(i))
        rewards = [e["R"] for e in self.h.entries]
        assert rewards[-1] == 6.0

    # R_avg
    def test_r_avg_empty(self):
        assert self.h.R_avg() == 0.0

    def test_r_avg_single(self):
        self._append_correcto(R=0.5)
        assert abs(self.h.R_avg() - 0.5) < 1e-6

    def test_r_avg_multiple(self):
        self._append_correcto(R=0.4)
        self._append_correcto(R=0.6)
        assert abs(self.h.R_avg() - 0.5) < 1e-6

    # is_degrading
    def test_not_degrading_with_one_entry(self):
        self._append_correcto(R=0.8)
        assert self.h.is_degrading(R_curr=0.1) is False

    def test_degrading_when_far_below_avg(self):
        for _ in range(4):
            self._append_correcto(R=0.9)
        assert self.h.is_degrading(R_curr=0.1) is True

    def test_not_degrading_near_avg(self):
        for _ in range(4):
            self._append_correcto(R=0.9)
        assert self.h.is_degrading(R_curr=0.85) is False

    # trend_label
    def test_trend_sin_datos(self):
        assert self.h.trend_label() == "SIN_DATOS"

    def test_trend_degradante(self):
        for _ in range(3):
            self._append_incorrecto()
        assert self.h.trend_label() == "DEGRADANTE"

    def test_trend_positiva(self):
        for _ in range(5):
            self._append_correcto(R=0.9)
        assert self.h.trend_label() == "POSITIVA"

    # persistence
    def test_save_and_load(self):
        for _ in range(3):
            self._append_correcto()
        self.h.bump_version()

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            self.h.save(path)
            h2 = RewardHistory(window_size=5, convergence_window=3)
            h2.load(path)
            assert h2.version == 1
            assert h2.converged is False   # bumped → reset
            assert len(h2.entries) == 3
        finally:
            path.unlink(missing_ok=True)

    def test_load_nonexistent_file(self):
        # Should not raise, just keep defaults
        self.h.load("/nonexistent/path/file.json")
        assert self.h.version == 0
        assert self.h.entries == []

    def test_load_corrupt_json(self):
        with tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w", encoding="utf-8"
        ) as f:
            f.write("{ invalid json }")
            path = Path(f.name)
        try:
            self.h.load(path)  # should not raise
            assert self.h.version == 0
        finally:
            path.unlink(missing_ok=True)

    def test_from_file_missing(self):
        h = RewardHistory.from_file("/nonexistent/file.json")
        assert h.version == 0


# ── UpdateGate ────────────────────────────────────────────────────────────────

class TestUpdateGate:
    def setup_method(self):
        self.gate = UpdateGate(threshold_degradation=0.2)

    def _eval(self, R_curr=0.8, R_avg=0.9, verdict="CORRECTO",
              comment="", is_degrading=False):
        return self.gate.evaluate(
            R_curr=R_curr,
            R_avg=R_avg,
            verdict=verdict,
            comment=comment,
            is_degrading=is_degrading,
        )

    def test_stable_no_update(self):
        result = self._eval()
        assert result.should_update is False
        assert result.reason == "stable"

    def test_degradation_triggers_update(self):
        result = self._eval(is_degrading=True)
        assert result.should_update is True
        assert result.reason == "degradation"

    def test_forced_incorrecto_with_comment(self):
        result = self._eval(verdict="INCORRECTO", comment="The answer is wrong.")
        assert result.should_update is True
        assert result.reason == "forced"

    def test_incorrecto_without_comment_no_update(self):
        result = self._eval(verdict="INCORRECTO", comment="")
        assert result.should_update is False

    def test_incorrecto_whitespace_comment_no_update(self):
        result = self._eval(verdict="INCORRECTO", comment="   ")
        assert result.should_update is False

    def test_degradation_takes_priority_over_stable(self):
        result = self._eval(verdict="CORRECTO", is_degrading=True)
        assert result.should_update is True
        assert result.reason == "degradation"

    def test_result_is_gatetype(self):
        result = self._eval()
        assert isinstance(result, GateResult)


class TestAlwaysUpdateGate:
    def setup_method(self):
        self.gate = AlwaysUpdateGate()

    def test_always_updates_correcto(self):
        r = self.gate.evaluate(0.8, 0.5, "CORRECTO", "", False)
        assert r.should_update is True
        assert r.reason == "always"

    def test_always_updates_incorrecto(self):
        r = self.gate.evaluate(-0.1, 0.5, "INCORRECTO", "", False)
        assert r.should_update is True

    def test_always_updates_no_degradation(self):
        r = self.gate.evaluate(0.9, 0.9, "CORRECTO", "", False)
        assert r.should_update is True
