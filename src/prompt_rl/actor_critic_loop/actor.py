from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from prompt_rl.llm.base import LLMBackend


@dataclass
class ActorOutput:
    system_prompt: str
    response: str
    index: int


ACTOR_SYSTEM_PROMPT = """You are the Actor (student) in an Actor-Critic system for refining system prompts.

Your role:
- You generate system prompt variations. Each variation will be given to a model along with a user question. The model must ANSWER THAT QUESTION directly and relevantly. Responses that ignore, deflect, or are off-topic receive reward 0.
- A human selects ALL responses they consider correct (relevant and answering the question). Each selected = reward 1, not selected = 0. Your goal: maximize the NUMBER of correct responses.
- The Critic only sees (system_prompt, reward) and optionally a human feedback comment. It never sees the query nor the responses. It infers refinement direction from reward patterns and tells you how to improve.
- Every system prompt you generate MUST instruct the model to: answer the user's question directly, stay on topic, and be relevant. Never generate prompts that would cause tangential or unrelated answers.
- Output ONLY the system prompt text. No preamble, no explanation."""


class LLMActor:
    def __init__(
        self,
        prompt_llm: LLMBackend,
        response_llm: LLMBackend,
        system_prompt: Optional[str] = None,
        max_tokens: int = 512,
        temperature: float = 0.8,
    ) -> None:
        self.prompt_llm = prompt_llm
        self.response_llm = response_llm
        self._system_prompt = system_prompt or ACTOR_SYSTEM_PROMPT
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate_prompt_variations(
        self,
        base_or_instruction: str,
        num_variations: int,
        refinement_hint: Optional[str] = None,
        query: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        user = f"Generate {num_variations} distinct system prompt variations. Each will be used with a user question; the model MUST answer that question directly and relevantly."
        if query:
            user += f"\n\nUser question (responses must address this): {query}"
        if refinement_hint:
            user += f"\n\nCritic's refinement hint: {refinement_hint}"
        user += f"\n\nBase instruction: {base_or_instruction}\n\nEach prompt must ensure the model answers the user's question on-topic. Output each prompt on a separate line, prefixed with 'PROMPT N:' for N=1 to {num_variations}."
        full = f"{self._system_prompt}\n\n{user}"
        resp = self.prompt_llm.complete(
            full,
            max_tokens=self.max_tokens * 2,
            temperature=self.temperature,
            **(context or {}),
        )
        prompts = []
        for line in resp.text.strip().split("\n"):
            line = line.strip()
            if line.upper().startswith("PROMPT ") and ":" in line:
                _, _, rest = line.partition(":")
                prompts.append(rest.strip())
            elif line and ":" in line:
                idx = line.index(":")
                rest = line[idx + 1 :].strip()
                if rest:
                    prompts.append(rest)
            elif line:
                prompts.append(line)
        while len(prompts) < num_variations:
            prompts.append(prompts[-1] if prompts else base_or_instruction)
        return prompts[:num_variations]

    def generate_responses(
        self,
        system_prompts: list[str],
        query: str,
        context: Optional[dict[str, Any]] = None,
    ) -> list[ActorOutput]:
        outputs = []
        for i, sp in enumerate(system_prompts):
            full = f"""{sp}

---
USER QUESTION (answer this directly and relevantly):
{query}

ASSISTANT RESPONSE:"""
            resp = self.response_llm.complete(
                full,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                **(context or {}),
            )
            outputs.append(ActorOutput(system_prompt=sp, response=resp.text.strip(), index=i))
        return outputs
