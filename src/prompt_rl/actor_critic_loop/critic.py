from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from prompt_rl.llm.base import LLMBackend


@dataclass
class CriticInput:
    """Input for the Critic. The Critic never sees query nor responses."""
    system_prompt: str
    reward: float
    has_feedback_comment: bool = False
    human_comment: Optional[str] = None


CRITIC_SYSTEM_PROMPT = """You are the Critic in an Actor-Critic system for refining system prompts.

Your role:
- You guide an Actor (student) who generates system prompt variations. The model uses each prompt to answer a user question. Humans select ALL responses that are correct (relevant, on-topic, answering the question). Each selected = reward 1, not selected = reward 0. More correct = higher total reward.
- You ONLY receive per variation: (system_prompt, reward). You never see the query nor the responses. You infer from rewards which prompts work.
- Reward 0 often means: the response was irrelevant, off-topic, or did not answer the user's question. Tell the Actor to strengthen prompts so the model answers directly and stays on topic.
- The human may optionally provide a feedback comment. If present, use it to better infer preferences.
- If total correct INCREASED: your refinement worked. Continue that direction.
- If total correct STABLE: may be near optimum. Consolidate or fine-tune.
- If total correct DECREASED or many rewards are 0: emphasize relevance—prompts must make the model answer the user's question directly.
- Your goal: help the Actor maximize correct responses by inferring from reward signals and optional human comments."""


class LLMCritic:
    def __init__(
        self,
        llm: LLMBackend,
        system_prompt: Optional[str] = None,
        max_tokens: int = 256,
        temperature: float = 0.3,
    ) -> None:
        self.llm = llm
        self._system_prompt = system_prompt or CRITIC_SYSTEM_PROMPT
        self.max_tokens = max_tokens
        self.temperature = temperature

    def get_refinement_direction(
        self,
        inputs: list[CriticInput],
        context: Optional[dict[str, Any]] = None,
    ) -> str:
        total_correct = sum(1 for inp in inputs if inp.reward >= 0.5)
        prev_total = (context or {}).get("previous_total_correct")
        iter_info = ""
        if prev_total is not None:
            if total_correct > prev_total:
                iter_info = f"\n\nPrevious iteration: {prev_total} correct. This iteration: {total_correct} correct. Reward INCREASED — the refinement worked. Continue this direction."
            elif total_correct == prev_total:
                iter_info = f"\n\nPrevious iteration: {prev_total} correct. This iteration: {total_correct} correct. Reward STABLE — may be near optimum. Consolidate or fine-tune."
            else:
                iter_info = f"\n\nPrevious iteration: {prev_total} correct. This iteration: {total_correct} correct. Reward DECREASED — adjust direction."
        parts = []
        batch_comment = None
        for i, inp in enumerate(inputs, 1):
            r = 1.0 if inp.reward >= 0.5 else 0.0
            parts.append(f"[{i}] System prompt: {inp.system_prompt}\n    Reward: {r:.0f} (1=correct, 0=incorrect)")
            if inp.has_feedback_comment and inp.human_comment:
                batch_comment = inp.human_comment
        user_content = "Given these (system_prompt, reward) pairs per variation. Reward=1 means human selected as correct, 0 means not selected. You do NOT see query nor responses.\n\n" + "\n\n".join(parts)
        if batch_comment:
            user_content += f"\n\nHuman feedback comment (optional, for the whole batch): {batch_comment}"
        user_content += f"\n\nTotal correct this iteration: {total_correct} of {len(inputs)}." + iter_info
        user_content += "\n\nProvide a concise refinement instruction for the Actor (2-4 sentences)."
        full_prompt = f"{self._system_prompt}\n\n{user_content}"
        resp = self.llm.complete(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        return resp.text.strip()
