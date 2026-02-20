"""OnlineCriticLoop — the main event-driven RL loop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from prompt_rl.core.cycle          import PerceptionCycle
from prompt_rl.core.policy         import ActivePolicy
from prompt_rl.core.policy_schema  import PolicySchema, parse_policy
from prompt_rl.critic.base         import CriticOutput, PerceptionCritic
from prompt_rl.critic.memory       import CriticMemory
from prompt_rl.feedback.signals    import (
    FeedbackAggregator,
    reading_time_to_score,
    thumbs_to_score,
)
from prompt_rl.population.genome      import PromptGenome
from prompt_rl.population.leaderboard import Individual, Leaderboard
from prompt_rl.rl.gate     import GateResult, UpdateGate
from prompt_rl.rl.history  import RewardHistory
from prompt_rl.rl.reward   import HybridReward, word_change_ratio

# Tags that identify behavioral-signal lines in cycle.observations
_BEHAVIORAL_TAGS = ("[DWELL]", "[SELECT]", "[CLICK]", "[REVIEW_RAG]")


@dataclass
class LoopResult:
    """Everything that happened during one process_cycle() call."""

    R_total:        float
    critic_output:  CriticOutput
    gate:           GateResult
    human_feedback: float
    change_ratio:   float
    converged:      bool


class OnlineCriticLoop:
    """
    Ties together: Critic → Reward → Gate → Policy update.

    One call to process_cycle() per completed PerceptionCycle.

    Convergence
    -----------
    When history.converged is True the evaluator should NOT be triggered
    (the monitor.py gate reads history.converged before spawning the subprocess).
    Convergence is reset automatically when bump_version() is called (i.e.
    every time the gate fires and the policy is updated).

    CriticMemory
    ------------
    Optional.  When provided, the loop:
      • Injects the Critic's past reasoning into cycle.observations
        (as [MEMORIA]-prefixed lines) before calling critic.evaluate().
      • Records a new memory entry after each cycle (verdict, reward,
        reasoning, nota/hypothesis, key behavioral signals).

    Usage
    -----
    >>> loop = OnlineCriticLoop(critic=..., policy=...)
    >>> result = loop.process_cycle(cycle)
    >>> loop.save_state(history_path, leaderboard_path)
    """

    def __init__(
        self,
        critic:      PerceptionCritic,
        policy:      ActivePolicy,
        reward_fn:   Optional[HybridReward]      = None,
        gate:        Optional[UpdateGate]         = None,
        history:     Optional[RewardHistory]      = None,
        leaderboard: Optional[Leaderboard]        = None,
        feedback:    Optional[FeedbackAggregator] = None,
        memory:      Optional[CriticMemory]       = None,
    ) -> None:
        self.critic      = critic
        self.policy      = policy
        self.reward_fn   = reward_fn   or HybridReward()
        self.gate        = gate        or UpdateGate()
        self.history     = history     or RewardHistory()
        self.leaderboard = leaderboard or Leaderboard()
        self._feedback   = feedback    or FeedbackAggregator()
        self.memory      = memory      # None = memory disabled

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def converged(self) -> bool:
        return self.history.converged

    def process_cycle(self, cycle: PerceptionCycle) -> LoopResult:
        """
        Full RL step for one completed PerceptionCycle.

        Fast path para CORRECTO: no invoca al Critic (ahorra 2+ LLM calls).
        Solo registra R sintético en history para convergencia y R_avg.

        1. Read current policy text.
        2. Para CORRECTO: fast path (sin Critic). Para INCORRECTO: Critic + Reward + Gate.
        3–10. History, leaderboard, gate, policy update, memory.
        """
        current_prompt = self.policy.read()

        # ── Fast path: CORRECTO ───────────────────────────────────────────────
        # No invocar Critic: ahorra Backward + Optimizer (+ validación si aplica).
        # Solo R sintético para convergencia y R_avg.
        if cycle.is_correct:
            implicit = (
                reading_time_to_score(cycle.dwell_seconds)
                if cycle.dwell_seconds > 0
                else None
            )
            human_feedback = self._feedback.aggregate(
                explicit=thumbs_to_score(True),
                implicit=implicit,
            )
            critic_out = CriticOutput(
                critic_score=0.9,
                proposed_prompt=current_prompt,
                reasoning="Ciclo CORRECTO — sin refinamiento (fast path)",
                nota="",
            )
            change_ratio = 0.0
            R_total = self.reward_fn.compute(
                human_feedback=human_feedback,
                critic_score=0.9,
                current_prompt=current_prompt,
                proposed_prompt=current_prompt,
            )
            self.history.append(
                R=R_total,
                verdict=cycle.verdict,
                critic_score=0.9,
                change_ratio=change_ratio,
                dwell_seconds=cycle.dwell_seconds,
            )
            self.leaderboard.add(
                Individual(
                    genome=PromptGenome.from_text(current_prompt),
                    fitness=R_total,
                )
            )
            gate_result = self.gate.evaluate(
                R_curr=R_total,
                R_avg=self.history.R_avg(),
                verdict=cycle.verdict,
                comment=cycle.comment,
                is_degrading=self.history.is_degrading(R_total),
            )
            if self.memory is not None:
                key_signals = [
                    o for o in cycle.observations
                    if any(t in o for t in _BEHAVIORAL_TAGS)
                ][:5]
                self.memory.record(
                    policy_version=self.history.version,
                    verdict=cycle.verdict,
                    critic_score=0.9,
                    R_total=R_total,
                    reward_trend=self.history.trend_label(),
                    change_ratio=change_ratio,
                    gate_reason=gate_result.reason,
                    policy_updated=False,
                    reasoning="Ciclo CORRECTO — fast path (sin Critic)",
                    nota="",
                    key_signals=key_signals,
                )
            return LoopResult(
                R_total=R_total,
                critic_output=critic_out,
                gate=gate_result,
                human_feedback=human_feedback,
                change_ratio=change_ratio,
                converged=self.converged,
            )

        # ── Full path: INCORRECTO (o CORRECTO si se desactiva fast path) ───────
        # 2. Enrich observations
        enriched = list(cycle.observations) + self.history.trend_summary()
        if self.memory is not None:
            enriched += self.memory.observation_lines()
        cycle.observations = enriched

        # 3. Critic (blind to query/response)
        critic_out = self.critic.evaluate(cycle)

        # 4. Human feedback score
        implicit = (
            reading_time_to_score(cycle.dwell_seconds)
            if cycle.dwell_seconds > 0
            else None
        )
        human_feedback = self._feedback.aggregate(
            explicit=thumbs_to_score(cycle.is_correct),
            implicit=implicit,
        )

        # 5. Reward
        change_ratio = word_change_ratio(current_prompt, critic_out.proposed_prompt)
        R_total      = self.reward_fn.compute(
            human_feedback=human_feedback,
            critic_score=critic_out.critic_score,
            current_prompt=current_prompt,
            proposed_prompt=critic_out.proposed_prompt,
        )

        # ── 6. History ────────────────────────────────────────────────────────
        self.history.append(
            R=R_total,
            verdict=cycle.verdict,
            critic_score=critic_out.critic_score,
            change_ratio=change_ratio,
            dwell_seconds=cycle.dwell_seconds,
        )

        # ── 7. Leaderboard ────────────────────────────────────────────────────
        self.leaderboard.add(
            Individual(
                genome=PromptGenome.from_text(critic_out.proposed_prompt),
                fitness=R_total,
            )
        )

        # ── 8. Gate ───────────────────────────────────────────────────────────
        gate_result = self.gate.evaluate(
            R_curr=R_total,
            R_avg=self.history.R_avg(),
            verdict=cycle.verdict,
            comment=cycle.comment,
            is_degrading=self.history.is_degrading(R_total),
        )

        # ── 8b. Block full rewrites ────────────────────────────────────────────
        # Una iteracion de feedback NUNCA puede provocar reescribir toda la politica
        rewrite_blocked = self.reward_fn.is_rewrite_blocked(
            current_prompt, critic_out.proposed_prompt
        )
        if rewrite_blocked:
            gate_result = GateResult(
                should_update=False,
                reason="rewrite_blocked",
            )

        # ── 9. Policy update ──────────────────────────────────────────────────
        if gate_result.should_update:
            # Fallback: Critic returned the identical prompt on a forced correction.
            # Use string equality (not change_ratio) because word_change_ratio
            # returns 0 when text is *appended* — appending is valid and should
            # not trigger this fallback; only a truly unchanged prompt should.
            if (
                gate_result.reason == "forced"
                and critic_out.proposed_prompt.strip() == current_prompt.strip()
                and cycle.comment.strip()
            ):
                # Fallback: inject comment as direct hard rule (structured format)
                policy, _ = parse_policy(current_prompt)
                rule_text = f"Indica siempre: {cycle.comment.strip()}" if len(cycle.comment) < 50 else cycle.comment.strip()
                policy.hard_rules.append(rule_text)
                injected = policy.to_json()
                critic_out = CriticOutput(
                    critic_score=critic_out.critic_score,
                    proposed_prompt=injected,
                    reasoning=f"inyeccion-directa (fallback): {cycle.comment.strip()[:80]}",
                    nota=critic_out.nota,
                )
                change_ratio = word_change_ratio(current_prompt, injected)

            version = self.history.bump_version()
            self.policy.write(critic_out.proposed_prompt, version)

        # ── 10. Critic memory ─────────────────────────────────────────────────
        if self.memory is not None:
            key_signals = [
                o for o in cycle.observations
                if any(t in o for t in _BEHAVIORAL_TAGS)
            ][:5]
            self.memory.record(
                policy_version=self.history.version,
                verdict=cycle.verdict,
                critic_score=critic_out.critic_score,
                R_total=R_total,
                reward_trend=self.history.trend_label(),
                change_ratio=change_ratio,
                gate_reason=gate_result.reason,
                policy_updated=gate_result.should_update,
                reasoning=critic_out.reasoning,
                nota=critic_out.nota,
                key_signals=key_signals,
            )

        return LoopResult(
            R_total=R_total,
            critic_output=critic_out,
            gate=gate_result,
            human_feedback=human_feedback,
            change_ratio=change_ratio,
            converged=self.converged,
        )

    # ── Persistence helpers ───────────────────────────────────────────────────

    def save_state(
        self, history_path: str, leaderboard_path: str
    ) -> None:
        self.history.save(history_path)
        self.leaderboard.save(leaderboard_path)

    def load_state(
        self, history_path: str, leaderboard_path: str
    ) -> None:
        self.history.load(history_path)
        self.leaderboard.load(leaderboard_path)
