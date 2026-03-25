"""DatasetLoop — offline prompt optimization over Q&A datasets.

Mirrors the online :class:`~prompt_rl.loop.online.OnlineCriticLoop` but
iterates over a dataset instead of waiting for live human feedback.

Typical usage::

    from prompt_rl.dataset import DatasetLoop, DatasetSplit, ExactMatchJudge
    from prompt_rl import TwoStageCritic, Actor, CriticValidationLoop
    from prompt_rl import LLMValidationJudge, ActivePolicy
    from prompt_rl.llm.local_backend import LocalLLMBackend

    backend = LocalLLMBackend(model="llama-3.1-8b-instant", base_url=..., api_key=...)
    judge   = ExactMatchJudge()
    critic  = CriticValidationLoop(
        critic=TwoStageCritic(backend=backend),
        actor=Actor(backend=backend),
        judge=judge,
    )
    policy  = ActivePolicy(path="data/system_prompt.md")

    loop = DatasetLoop(critic=critic, policy=policy, actor=Actor(backend=backend), judge=judge)
    result = loop.train(split.train, test_samples=split.test, verbose=True)
    print(f"Accuracy: {result.acc_before:.1%} -> {result.acc_after:.1%}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from prompt_rl.dataset.sample import DatasetSample
    from prompt_rl.dataset.judge import DatasetJudge


@dataclass
class EpochMetrics:
    """Accuracy snapshot at a point during training.

    Attributes:
        epoch     : Epoch index (0-based). ``-1`` means pre-training baseline.
        step      : Training step within the epoch when snapshot was taken.
                    ``-1`` means end-of-epoch.
        accuracy  : Fraction of test samples answered correctly.
        n_correct : Absolute number of correct answers.
        n_total   : Total samples evaluated.
        n_updates : Cumulative policy updates at the time of this snapshot.
    """

    epoch:     int
    step:      int
    accuracy:  float
    n_correct: int
    n_total:   int
    n_updates: int


@dataclass
class DatasetResult:
    """Result returned by :meth:`DatasetLoop.train`.

    Attributes:
        acc_before   : Test accuracy with an empty system prompt (baseline).
        acc_after    : Test accuracy after training.
        delta        : ``acc_after - acc_before``.
        n_train      : Number of training samples.
        n_test       : Number of test samples (0 if no test set provided).
        n_updates    : Total number of policy updates applied during training.
        epochs       : Number of training epochs completed.
        per_epoch    : List of :class:`EpochMetrics` snapshots (baseline +
                       any mid-training + final).
        final_prompt : System prompt text at the end of training.
    """

    acc_before:   float
    acc_after:    float
    delta:        float
    n_train:      int
    n_test:       int
    n_updates:    int
    epochs:       int
    per_epoch:    list[EpochMetrics] = field(default_factory=list)
    final_prompt: str = ""


class DatasetLoop:
    """Offline prompt optimization loop for Q&A datasets.

    Wraps :class:`~prompt_rl.loop.online.OnlineCriticLoop` to drive it with
    dataset samples instead of live human feedback.  The judge is called
    deterministically (no extra LLM call) to decide verdict and comment for
    each sample.  The same judge also serves as a
    :class:`~prompt_rl.validation.judge.ValidationJudge` inside
    :class:`~prompt_rl.validation.loop.CriticValidationLoop`, so the Critic
    can verify its proposed prompts before committing them.

    Args:
        critic      : A :class:`~prompt_rl.critic.base.PerceptionCritic` —
                      typically :class:`~prompt_rl.validation.loop.CriticValidationLoop`
                      wrapping :class:`~prompt_rl.critic.two_stage_critic.TwoStageCritic`.
        policy      : :class:`~prompt_rl.core.policy.ActivePolicy` managing the
                      prompt file on disk.
        actor       : :class:`~prompt_rl.validation.actor.Actor` that generates
                      responses from a system prompt + question.
        judge       : A :class:`DatasetJudge` (e.g. :class:`ExactMatchJudge`)
                      that determines correctness and feedback without LLM calls.
        reward_fn   : Optional :class:`~prompt_rl.rl.reward.HybridReward`.
                      Created with defaults if omitted.
        gate        : Optional :class:`~prompt_rl.rl.gate.UpdateGate`.
                      Created with defaults if omitted.
        history     : Optional :class:`~prompt_rl.rl.history.RewardHistory`.
        leaderboard : Optional :class:`~prompt_rl.population.leaderboard.Leaderboard`.
        memory      : Optional :class:`~prompt_rl.critic.memory.CriticMemory`.
        verbose     : Print progress to stdout (default ``False``).
    """

    def __init__(
        self,
        critic,
        policy,
        actor,
        judge,
        reward_fn=None,
        gate=None,
        history=None,
        leaderboard=None,
        memory=None,
        verbose: bool = False,
    ) -> None:
        from prompt_rl.loop.online import OnlineCriticLoop

        self.actor   = actor
        self.judge   = judge
        self.policy  = policy
        self.verbose = verbose
        self._loop   = OnlineCriticLoop(
            critic=critic,
            policy=policy,
            reward_fn=reward_fn,
            gate=gate,
            history=history,
            leaderboard=leaderboard,
            memory=memory,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        samples: "list[DatasetSample]",
        system_prompt: str | None = None,
    ) -> EpochMetrics:
        """Evaluate the current (or given) system prompt on *samples*.

        Args:
            samples       : List of :class:`DatasetSample` to evaluate on.
            system_prompt : Override the prompt to evaluate. Uses the current
                            policy file if ``None``.

        Returns:
            :class:`EpochMetrics` with accuracy, counts, and ``epoch=-1``.
        """
        if system_prompt is None:
            system_prompt = self.policy.read()
        correct = 0
        for sample in samples:
            response = self.actor.generate(system_prompt, sample.question)
            if self.judge.is_correct(response, sample):
                correct += 1
        n = len(samples)
        return EpochMetrics(
            epoch=-1,
            step=-1,
            accuracy=correct / n if n > 0 else 0.0,
            n_correct=correct,
            n_total=n,
            n_updates=self._loop.history.version,
        )

    def train(
        self,
        train_samples: "list[DatasetSample]",
        test_samples: "list[DatasetSample] | None" = None,
        eval_every: int = 0,
        max_epochs: int = 1,
        verbose: bool | None = None,
    ) -> DatasetResult:
        """Run the offline RL training loop.

        Args:
            train_samples : Samples used to drive Critic updates.
            test_samples  : Hold-out samples for accuracy evaluation.
                            If ``None``, ``acc_before``/``acc_after`` will be 0.
            eval_every    : Evaluate on *test_samples* every N training steps.
                            ``0`` means only at start and end (default).
            max_epochs    : Number of full passes over *train_samples* (default 1).
            verbose       : Override instance-level ``verbose`` flag.

        Returns:
            :class:`DatasetResult` with accuracy metrics and final prompt.
        """
        from prompt_rl.core.cycle import PerceptionCycle

        verbose = self.verbose if verbose is None else verbose
        per_epoch: list[EpochMetrics] = []
        total_updates = 0

        # Baseline: evaluate with empty prompt before any training
        acc_before = 0.0
        if test_samples:
            if verbose:
                print(
                    f"[DatasetLoop] Baseline evaluation on {len(test_samples)} test samples..."
                )
            baseline = self.evaluate(test_samples, system_prompt="")
            baseline.epoch = -1
            per_epoch.append(baseline)
            acc_before = baseline.accuracy
            if verbose:
                print(f"[DatasetLoop] Baseline accuracy: {acc_before:.1%}")

        for epoch in range(max_epochs):
            if verbose:
                print(
                    f"\n[DatasetLoop] Epoch {epoch + 1}/{max_epochs} "
                    f"— {len(train_samples)} samples"
                )

            for step, sample in enumerate(train_samples):
                system_prompt = self.policy.read()
                response = self.actor.generate(system_prompt, sample.question)
                is_correct = self.judge.is_correct(response, sample)

                # Inject sample into the judge so CriticValidationLoop can
                # verify proposed prompts deterministically (no extra LLM call).
                if not is_correct and hasattr(self.judge, "set_sample"):
                    self.judge.set_sample(sample)

                cycle = PerceptionCycle(
                    system_prompt=system_prompt,
                    user_query=sample.question,
                    bot_response=response,
                    verdict="CORRECTO" if is_correct else "INCORRECTO",
                    comment=(
                        self.judge.feedback(response, sample)
                        if not is_correct
                        else ""
                    ),
                )

                loop_result = self._loop.process_cycle(cycle)
                if loop_result.gate.should_update:
                    total_updates += 1

                if verbose and (step + 1) % 10 == 0:
                    print(
                        f"  [{step + 1}/{len(train_samples)}] "
                        f"verdict={'OK' if is_correct else 'FAIL'} "
                        f"updates={total_updates}"
                    )

                # Mid-training evaluation
                if eval_every > 0 and (step + 1) % eval_every == 0 and test_samples:
                    snap = self.evaluate(test_samples)
                    snap.epoch = epoch
                    snap.step = step + 1
                    snap.n_updates = total_updates
                    per_epoch.append(snap)
                    if verbose:
                        print(
                            f"  [eval @ step {step + 1}] "
                            f"acc={snap.accuracy:.1%} updates={total_updates}"
                        )

        # Final evaluation
        acc_after = 0.0
        if test_samples:
            if verbose:
                print(
                    f"\n[DatasetLoop] Final evaluation on {len(test_samples)} test samples..."
                )
            final = self.evaluate(test_samples)
            final.epoch = max_epochs - 1
            final.step = -1
            final.n_updates = total_updates
            per_epoch.append(final)
            acc_after = final.accuracy
            if verbose:
                delta = acc_after - acc_before
                print(
                    f"[DatasetLoop] Final accuracy: {acc_after:.1%} "
                    f"(delta={delta:+.1%}, updates={total_updates})"
                )

        return DatasetResult(
            acc_before=acc_before,
            acc_after=acc_after,
            delta=acc_after - acc_before,
            n_train=len(train_samples),
            n_test=len(test_samples) if test_samples else 0,
            n_updates=total_updates,
            epochs=max_epochs,
            per_epoch=per_epoch,
            final_prompt=self.policy.read(),
        )

    def save_state(self, history_path: str, leaderboard_path: str) -> None:
        """Persist reward history and leaderboard to disk."""
        self._loop.save_state(history_path, leaderboard_path)
