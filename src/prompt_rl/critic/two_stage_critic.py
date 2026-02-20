"""
TwoStageCritic — Critic en dos etapas con contexto completo.

Recibe user_query y bot_response para producir feedback accionable.
Flujo: HumanFeedback → Critic 1.1 (Backward) → Critic 1.2 (Optimizer).

Implementa PerceptionCritic para integrarse en OnlineCriticLoop.
"""

from __future__ import annotations

from prompt_rl.core.cycle import PerceptionCycle
from prompt_rl.critic.base import CriticOutput, PerceptionCritic
from prompt_rl.critic.backward import CriticBackward, BackwardOutput
from prompt_rl.critic.optimizer import CriticOptimizer
from prompt_rl.feedback.cursor_trace import CursorTrace, HumanFeedbackResult
from prompt_rl.llm.base import LLMBackend


class TwoStageCritic:
    """
    Critic en dos etapas con contexto completo.

    Recibe el ciclo completo (incluye user_query y bot_response).
    Usa CursorTrace desde cycle.observations para enriquecer el feedback.

    Flujo:
      1. HumanFeedbackResult desde verdict + comment + CursorTrace(observations)
      2. Critic 1.1 (Backward): produce feedback accionable con contexto
      3. Critic 1.2 (Optimizer): genera nuevo system prompt
    """

    def __init__(
        self,
        backend: LLMBackend,
        backward_max_tokens: int = 1024,
        optimizer_max_tokens: int = 2048,
        temperature: float = 0.2,
        include_cursor_trace: bool = True,
        gradient_memory: int = 0,
        constraints: list[str] | None = None,
        in_context_examples: list[str] | None = None,
        component_propagation: bool = True,
        verbose: bool = False,
    ):
        """
        gradient_memory: número de feedbacks anteriores a incluir en el optimizer (0=off).
        constraints: restricciones en lenguaje natural (ej. "responde en español").
        in_context_examples: ejemplos de mejoras para guiar el optimizer.
        component_propagation: feedback por componente (role, hard_rules, etc.) cuando la policy es JSON.
        verbose: si True, imprime Backward (1.1) y Optimizer (1.2) en detalle.
        """
        self._backend = backend
        self._verbose = verbose
        self._include_cursor = include_cursor_trace
        self._backward = CriticBackward(
            backend=backend,
            max_tokens=backward_max_tokens,
            temperature=temperature,
            component_propagation=component_propagation,
        )
        self._optimizer = CriticOptimizer(
            backend=backend,
            max_tokens=optimizer_max_tokens,
            temperature=temperature,
            gradient_memory=gradient_memory,
            constraints=constraints or [],
            in_context_examples=in_context_examples or [],
        )

    def evaluate(self, cycle: PerceptionCycle) -> CriticOutput:
        """Implementa PerceptionCritic. Usa user_query y bot_response (contexto completo)."""

        cursor_trace = None
        if self._include_cursor and cycle.observations:
            cursor_trace = CursorTrace.from_observations(cycle.observations)

        human_eval = HumanFeedbackResult(
            feedback_text=cycle.comment or "(sin comentario explícito)",
            is_correct=cycle.is_correct,
            cursor_trace=cursor_trace,
        )

        backward_out = self._backward.run(
            system_prompt=cycle.system_prompt,
            prompt=cycle.user_query,
            response_value=cycle.bot_response,
            human_evaluation=human_eval,
            variable_desc="system prompt que guía al modelo",
            response_desc="respuesta del modelo",
        )

        if self._verbose:
            print("\n" + "=" * 60)
            print("[TwoStageCritic] 1.1 BACKWARD — feedback accionable")
            print("=" * 60)
            print(f"  feedback_text:\n{backward_out.feedback_text[:500]}{'...' if len(backward_out.feedback_text) > 500 else ''}")
            if backward_out.feedback_by_component:
                print("  feedback_by_component:")
                for k, v in backward_out.feedback_by_component.items():
                    preview = (v[:120] + "...") if len(v) > 120 else v
                    status = "OK" if v.strip().lower() in ("ok", "mantener", "") else "cambio"
                    print(f"    [{k}] ({status}): {preview}")
            print()

        try:
            proposed_prompt = self._optimizer.run(
                current_system_prompt=cycle.system_prompt,
                backward_output=backward_out,
                variable_desc="system prompt que guía al modelo",
            )
        except ValueError as e:
            # Fallback: mantener prompt actual
            proposed_prompt = cycle.system_prompt

        if self._verbose:
            print("[TwoStageCritic] 1.2 OPTIMIZER — propuesta generada")
            print("-" * 60)
            print(f"  gradient_blind (input): {backward_out.to_gradient_blind()[:300]}...")
            print(f"  proposed_prompt ({len(proposed_prompt)} chars):")
            for line in proposed_prompt[:400].splitlines():
                print(f"    | {line}")
            if len(proposed_prompt) > 400:
                print("    | ...")
            print()

        # critic_score: heurística simple; el gate y reward usan human_feedback principalmente
        critic_score = 0.6 if cycle.is_correct else 0.4
        if backward_out.feedback_text and len(backward_out.feedback_text) > 50:
            critic_score = 0.55  # feedback sustancial

        return CriticOutput(
            critic_score=critic_score,
            proposed_prompt=proposed_prompt,
            reasoning=backward_out.feedback_text[:200] + ("..." if len(backward_out.feedback_text) > 200 else ""),
            nota="TwoStageCritic — contexto completo (user_query + bot_response)",
        )
