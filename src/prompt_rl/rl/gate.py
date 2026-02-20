"""UpdateGate — decides when to apply a policy update."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GateResult:
    should_update: bool
    reason:        str   # "degradation" | "forced" | "stable" | "always"


class UpdateGate:
    """
    Fires (should_update=True) when EITHER condition holds:

    A. Degradation: R_curr < R_avg * (1 - threshold_degradation)
       The policy has gotten measurably worse than its rolling average.

    B. Forced: verdict == INCORRECTO  AND  comment is non-empty.
       The human explicitly corrected the bot and left a note.

    In all other cases the gate returns 'stable' and the policy is left as-is.
    """

    def __init__(self, threshold_degradation: float = 0.2) -> None:
        self.threshold_degradation = threshold_degradation

    def evaluate(
        self,
        R_curr:       float,
        R_avg:        float,
        verdict:      str,
        comment:      str,
        is_degrading: bool,
    ) -> GateResult:
        if is_degrading:
            return GateResult(should_update=True, reason="degradation")
        if verdict == "INCORRECTO" and bool(comment.strip()):
            return GateResult(should_update=True, reason="forced")
        return GateResult(should_update=False, reason="stable")


class AlwaysUpdateGate:
    """
    Gate que siempre permite actualizar en cada iteración/ciclo.
    No aplica condiciones de degradación ni forzado.
    """

    def evaluate(
        self,
        R_curr:       float,
        R_avg:        float,
        verdict:      str,
        comment:      str,
        is_degrading: bool,
    ) -> GateResult:
        return GateResult(should_update=True, reason="always")
