"""HybridReward — composite reward for the perception-cycle RL loop."""

from __future__ import annotations

import difflib


def word_change_ratio(old: str, new: str) -> float:
    """
    Fraction of words in `old` that differ from `new` (word-level Levenshtein).
    Returns a value in [0.0, 1.0].  0.0 = identical, 1.0 = completely rewritten.
    """
    old_words = old.split()
    new_words = new.split()
    total     = max(len(old_words), 1)
    matcher   = difflib.SequenceMatcher(None, old_words, new_words, autojunk=False)
    matches   = sum(block.size for block in matcher.get_matching_blocks())
    return min((total - matches) / total, 1.0)


# Threshold: if policy change exceeds this, the update is BLOCKED and heavily penalised
MAX_POLICY_CHANGE_RATIO = 0.35  # 35% words changed = considered full rewrite (prompts largos)

# Umbrales para prompts pequeños: más permisivo al inicio, menos al crecer
_WORDS_PERMISSIVE = 50   # hasta 50 palabras: muy permisivo
_WORDS_MODERATE = 100    # 50-100: moderado
_WORDS_STRICT = 150     # 100-150: transición a estricto


class HybridReward:
    """
    R_total = λ_feedback * H + λ_critic * C - λ_change * change_ratio
             - rewrite_penalty (if change_ratio > MAX_POLICY_CHANGE_RATIO)

    H             : human feedback score  [0, 1]
    C             : critic score          [0, 1]
    λ_change      : penalises large rewrites
    rewrite_penalty: extra penalty when Critic proposes full rewrite (blocked)
    """

    def __init__(
        self,
        lambda_feedback:   float = 0.9,
        lambda_critic:     float = 0.1,
        lambda_change:     float = 0.3,
        max_change_ratio:  float = MAX_POLICY_CHANGE_RATIO,
        rewrite_penalty:   float = 2.0,
    ) -> None:
        self.lambda_feedback  = lambda_feedback
        self.lambda_critic    = lambda_critic
        self.lambda_change   = lambda_change
        self.max_change_ratio = max_change_ratio
        self.rewrite_penalty = rewrite_penalty

    def get_effective_max_change_ratio(self, current_prompt: str) -> float:
        """
        Umbral dinámico: prompt vacío = nunca bloquear; prompt grande = estricto (0.35).
        A medida que el prompt crece, se vuelve menos permisivo.
        """
        n = len(current_prompt.strip().split())
        if n == 0:
            return 1.0  # bootstrap: nunca bloquear
        if n <= _WORDS_PERMISSIVE:
            return 0.9  # muy permisivo
        if n <= _WORDS_MODERATE:
            return 0.65  # moderado
        if n <= _WORDS_STRICT:
            return 0.5  # transición
        return self.max_change_ratio  # estricto (0.35)

    def compute(
        self,
        human_feedback:  float,
        critic_score:    float,
        current_prompt:  str,
        proposed_prompt: str,
    ) -> float:
        change_ratio = word_change_ratio(current_prompt, proposed_prompt)
        effective_threshold = self.get_effective_max_change_ratio(current_prompt)
        R = (
            self.lambda_feedback * human_feedback
            + self.lambda_critic   * critic_score
            - self.lambda_change   * change_ratio
        )
        # Penalizacion maxima si el Critic propone reescritura completa (según umbral efectivo)
        if change_ratio > effective_threshold:
            R -= self.rewrite_penalty
        return R

    def is_rewrite_blocked(self, current_prompt: str, proposed_prompt: str) -> bool:
        """True if the proposed change exceeds the allowed threshold (full rewrite).
        Umbral dinámico: vacío=permisivo, grande=estricto."""
        effective_threshold = self.get_effective_max_change_ratio(current_prompt)
        return word_change_ratio(current_prompt, proposed_prompt) > effective_threshold
