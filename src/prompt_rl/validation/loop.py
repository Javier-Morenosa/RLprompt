"""
CriticValidationLoop: flujo de validación con refinamiento iterativo.

1. Critic propone nuevo system prompt
2. Se re-pregunta la misma cuestión al actor con el nuevo prompt
3. El juez evalúa (con el feedback original) si se solucionó el problema
4. Si no: se crea un ciclo virtual y el critic propone otro prompt
5. Se repite hasta que el juez diga SI o se alcance max_iterations
"""

from __future__ import annotations

from dataclasses import dataclass, field

from prompt_rl.core.cycle import PerceptionCycle
from prompt_rl.critic.base import CriticOutput, PerceptionCritic
from prompt_rl.validation.actor import Actor
from prompt_rl.validation.judge import ValidationJudge, ValidationResult


@dataclass
class ValidationLoopResult:
    """Resultado del bucle de validación."""

    critic_output: CriticOutput
    fixed: bool
    iterations: int
    validation_results: list[ValidationResult] = field(default_factory=list)


class CriticValidationLoop:
    """
    Wrapper del Critic que añade validación + refinamiento.

    Cuando el verdict original es INCORRECTO:
    1. Critic propone prompt
    2. Actor responde con ese prompt a la misma pregunta
    3. Judge evalúa si se solucionó
    4. Si no: ciclo virtual → Critic propone de nuevo → repetir
    """

    def __init__(
        self,
        critic: PerceptionCritic,
        actor: Actor,
        judge: ValidationJudge,
        max_iterations: int = 3,
        skip_validation_if_correct: bool = True,
        verbose: bool = False,
    ):
        self._critic = critic
        self._verbose = verbose
        self._actor = actor
        self._judge = judge
        self._max_iterations = max_iterations
        self._skip_if_correct = skip_validation_if_correct

    def evaluate(self, cycle: PerceptionCycle) -> CriticOutput:
        """
        Evalúa el ciclo; si verdict=INCORRECTO, ejecuta bucle de validación.
        """
        critic_out = self._critic.evaluate(cycle)

        if self._skip_if_correct and cycle.is_correct:
            return critic_out

        if cycle.verdict != "INCORRECTO":
            return critic_out

        return self._run_validation_loop(cycle, critic_out).critic_output

    def _run_validation_loop(
        self,
        cycle: PerceptionCycle,
        initial_output: CriticOutput,
    ) -> ValidationLoopResult:
        proposed_prompt = initial_output.proposed_prompt
        current_cycle = cycle
        validation_results: list[ValidationResult] = []

        if self._verbose:
            print("\n" + "=" * 60)
            print("[CriticValidationLoop] Bucle de validación (Actor + Judge)")
            print("=" * 60)

        for i in range(self._max_iterations):
            if self._verbose:
                print(f"\n[CriticValidationLoop] Iteración {i + 1}/{self._max_iterations}")
            new_response = self._actor.generate(proposed_prompt, cycle.user_query)
            if self._verbose:
                print(f"  Actor (nueva respuesta): {new_response[:200]}{'...' if len(new_response) > 200 else ''}")
            result = self._judge.judge(
                user_query=cycle.user_query,
                original_feedback=cycle.comment or "(sin comentario)",
                original_response=cycle.bot_response,
                new_response=new_response,
            )
            validation_results.append(result)
            if self._verbose:
                print(f"  Judge: {'SI (solucionado)' if result.fixed else 'NO'}")
                if result.reasoning:
                    print(f"  reasoning: {result.reasoning[:150]}{'...' if len(result.reasoning) > 150 else ''}")

            if result.fixed:
                return ValidationLoopResult(
                    critic_output=CriticOutput(
                        critic_score=initial_output.critic_score,
                        proposed_prompt=proposed_prompt,
                        reasoning=f"{initial_output.reasoning} [Validado en iteración {i+1}]",
                        nota=initial_output.nota,
                    ),
                    fixed=True,
                    iterations=i + 1,
                    validation_results=validation_results,
                )

            if i + 1 >= self._max_iterations:
                break

            if self._verbose:
                print(f"  Ciclo virtual -> Backward + Optimizer proponen de nuevo...")
            # Backward recibe contexto (user_query, new_response) para entender el fallo,
            # pero su salida debe ser ciega (sin alusiones a user_query) para el Optimizer.
            virtual_cycle = PerceptionCycle(
                system_prompt=proposed_prompt,
                user_query=cycle.user_query,
                bot_response=new_response,
                verdict="INCORRECTO",
                comment=f"{cycle.comment} La respuesta con el prompt mejorado sigue sin ser correcta.",
                dwell_seconds=0.0,
                observations=[],
            )
            critic_out = self._critic.evaluate(virtual_cycle)
            proposed_prompt = critic_out.proposed_prompt

        return ValidationLoopResult(
            critic_output=CriticOutput(
                critic_score=initial_output.critic_score,
                proposed_prompt=proposed_prompt,
                reasoning=f"{initial_output.reasoning} [No validado tras {self._max_iterations} iteraciones]",
                nota=initial_output.nota,
            ),
            fixed=False,
            iterations=self._max_iterations,
            validation_results=validation_results,
        )
