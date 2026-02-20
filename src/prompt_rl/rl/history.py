"""RewardHistory — rolling reward window with convergence tracking."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


class RewardHistory:
    """
    Maintains a fixed-size rolling window of reward entries and tracks
    policy convergence.

    Convergence is declared after N consecutive 'stable' cycles, where
    stable means: verdict == CORRECTO  AND  word_change_ratio < eps.

    When the history file is loaded at startup, convergence state is
    restored so monitor.py can gate the evaluator correctly.
    """

    def __init__(
        self,
        window_size:            int   = 10,
        convergence_window:     int   = 5,
        convergence_change_eps: float = 0.05,
    ) -> None:
        self.window_size            = window_size
        self.convergence_window     = convergence_window
        self.convergence_change_eps = convergence_change_eps

        self._history:           list[dict] = []
        self._version:           int  = 0
        self._consecutive_stable: int  = 0
        self._converged:          bool = False

    # ── Append ────────────────────────────────────────────────────────────────

    def append(
        self,
        R:             float,
        verdict:       str,
        critic_score:  float,
        change_ratio:  float,
        dwell_seconds: float = 0.0,
    ) -> None:
        self._history.append({
            "R":             round(R, 6),
            "ts":            datetime.now().isoformat(),
            "verdict":       verdict,
            "critic_score":  round(critic_score, 4),
            "change_ratio":  round(change_ratio, 4),
            "dwell_seconds": round(dwell_seconds, 2),
            "version":       self._version,
        })
        if len(self._history) > self.window_size:
            self._history = self._history[-self.window_size:]

        # Convergence tracking
        is_stable = (
            verdict == "CORRECTO"
            and change_ratio < self.convergence_change_eps
        )
        self._consecutive_stable = self._consecutive_stable + 1 if is_stable else 0
        self._converged = self._consecutive_stable >= self.convergence_window

    # ── Derived statistics ────────────────────────────────────────────────────

    def R_avg(self) -> float:
        if not self._history:
            return 0.0
        return sum(e["R"] for e in self._history) / len(self._history)

    def is_degrading(self, R_curr: float, threshold: float = 0.2) -> bool:
        """
        True when R_curr is significantly below the rolling average.

        Uses an absolute margin (avg - |avg|*threshold) so the check
        works correctly when R_avg is negative.  A minimum margin of
        0.05 prevents false positives near zero.

        Examples (threshold=0.2):
            avg= 0.90 → fires if R_curr < 0.90 - 0.18 = 0.72
            avg=-0.45 → fires if R_curr < -0.45 - 0.09 = -0.54
            avg= 0.00 → fires if R_curr < 0.00 - 0.05 = -0.05
        """
        if len(self._history) <= 1:
            return False
        avg    = self.R_avg()
        margin = max(abs(avg) * threshold, 0.05)
        return R_curr < avg - margin

    # ── Version management ────────────────────────────────────────────────────

    def bump_version(self) -> int:
        """
        Increment version counter and reset convergence streak.
        Called whenever a policy update is applied.
        """
        self._version += 1
        self._consecutive_stable = 0
        self._converged = False
        return self._version

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def version(self) -> int:
        return self._version

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def consecutive_stable(self) -> int:
        return self._consecutive_stable

    @property
    def entries(self) -> list[dict]:
        return list(self._history)

    def trend_label(self) -> str:
        """
        Short trend classification based on the last 5 verdicts.
        Returns one of: DEGRADANTE | INESTABLE | POSITIVA | NEUTRAL | SIN_DATOS.
        """
        if not self._history:
            return "SIN_DATOS"
        recent   = self._history[-5:]
        n_incorr = sum(1 for e in recent if e["verdict"] == "INCORRECTO")
        if   n_incorr >= 3:         return "DEGRADANTE"
        elif n_incorr == 2:         return "INESTABLE"
        elif self.R_avg() > 0.05:   return "POSITIVA"
        else:                       return "NEUTRAL"

    def trend_summary(self) -> list[str]:
        """
        Structured reward-context strings intended to be injected into
        PerceptionCycle.observations so the Critic can reason about the
        policy's performance history without any additional API changes.

        Returns a list of plain strings like:
            "policy_version: v3"
            "reward_avg (n=7): -0.0864"
            "reward_trend: INESTABLE"
            ...
        """
        if not self._history:
            return ["historial_reward: sin datos previos"]

        n        = len(self._history)
        avg      = self.R_avg()
        recent   = self._history[-5:]
        verdicts = [e["verdict"] for e in recent]
        rewards  = [e["R"]       for e in recent]

        return [
            f"policy_version: v{self._version}",
            f"reward_avg (n={n}): {avg:+.4f}",
            f"reward_trend: {self.trend_label()}",
            "ultimos_rewards: " + "  ".join(f"{r:+.2f}" for r in rewards),
            "ultimos_veredictos: " + " | ".join(verdicts),
            f"racha_estable: {self._consecutive_stable}/{self.convergence_window}",
        ]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {
                    "version":            self._version,
                    "consecutive_stable": self._consecutive_stable,
                    "converged":          self._converged,
                    "history":            self._history,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def load(self, path: str | Path) -> None:
        p = Path(path)
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            self._history             = data.get("history", [])
            self._version             = data.get("version", 0)
            self._consecutive_stable  = data.get("consecutive_stable", 0)
            self._converged           = data.get("converged", False)
        except Exception:
            pass

    @classmethod
    def from_file(cls, path: str | Path, **kwargs: object) -> "RewardHistory":
        h = cls(**kwargs)  # type: ignore[arg-type]
        h.load(path)
        return h
