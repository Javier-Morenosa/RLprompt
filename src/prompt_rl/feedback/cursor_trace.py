"""
Cursor trace y HumanFeedback para enriquecer la evaluación (punto 2.1).

El feedback viene del humano (verdict + comment). Se incorpora el trazado del cursor
para detectar pistas de razonamiento: dónde pausó, qué seleccionó, etc.

Compatible con cycle.observations existentes: [DWELL], [SELECT], [CLICK], [REVIEW_RAG].
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

# Tags existentes en RLprompt para señales comportamentales
_BEHAVIORAL_TAGS = ("[DWELL]", "[SELECT]", "[CLICK]", "[REVIEW_RAG]")


@dataclass
class CursorTrace:
    """
    Trazado del cursor/mouse del evaluador.
    Proporciona pistas sobre el razonamiento: qué regiones revisó, dónde pausó, etc.
    """

    events: list[dict[str, Any]] = field(default_factory=list)
    """
    Lista de eventos. Cada evento puede tener:
    - type: "move" | "click" | "scroll" | "selection" | "pause"
    - x, y: coordenadas (para move, click)
    - target_text: fragmento de texto bajo el cursor (si aplica)
    - duration_ms: tiempo en milisegundos (para pause)
    - scroll_delta: cantidad de scroll (para scroll)
    """

    @classmethod
    def from_observations(cls, observations: list[str]) -> "CursorTrace":
        """
        Construye CursorTrace desde cycle.observations existentes.
        Parsea [DWELL], [SELECT], [CLICK], [REVIEW_RAG].
        """
        events: list[dict[str, Any]] = []
        for obs in observations:
            obs = obs.strip()
            if not obs:
                continue
            if "[DWELL]" in obs:
                # ej: "[DWELL] 4.2s" o "[DWELL] 4200ms"
                m = re.search(r"(\d+(?:\.\d+)?)\s*(?:s|sec|seconds?|ms)?", obs, re.I)
                if m:
                    val = float(m.group(1))
                    events.append({"type": "pause", "duration_ms": val * 1000 if val < 100 else val})
            elif "[SELECT]" in obs:
                # ej: "[SELECT] fragmento de texto"
                text = obs.replace("[SELECT]", "").strip()
                events.append({"type": "selection", "target_text": text[:200]})
            elif "[CLICK]" in obs:
                events.append({"type": "click", "raw": obs})
            elif "[REVIEW_RAG]" in obs:
                events.append({"type": "scroll", "raw": obs})
        return cls(events=events)

    def to_analysis_string(self) -> str:
        """
        Convierte el trazado en un string analítico para enriquecer el feedback.
        """
        if not self.events:
            return ""

        parts: list[str] = []
        pauses = [e for e in self.events if e.get("type") == "pause" and e.get("duration_ms", 0) > 1000]
        selections = [e for e in self.events if e.get("type") == "selection"]

        if pauses:
            total_ms = sum(e.get("duration_ms", 0) for e in pauses)
            parts.append(
                f"El evaluador pausó {len(pauses)} vez(es) (~{total_ms/1000:.1f}s), "
                "posible indicador de reflexión o duda."
            )
        if selections:
            parts.append(
                f"El evaluador seleccionó {len(selections)} fragmento(s): "
                + ", ".join(repr(e.get("target_text", "")[:40]) for e in selections[:3])
            )

        return " ".join(parts) if parts else ""

    def to_raw_string(self) -> str:
        """Representación raw del trazado."""
        if not self.events:
            return ""
        return json.dumps(self.events, ensure_ascii=False, indent=2)


@dataclass
class HumanFeedbackResult:
    """
    Resultado de la evaluación humana (equivalente al output del TextLoss).
    Incluye feedback, veredicto y análisis del cursor si existe.
    """

    feedback_text: str
    is_correct: bool
    cursor_trace: Optional[CursorTrace] = None

    def to_evaluation_string(self) -> str:
        """Construye el string de evaluación completo para Critic 1.1."""
        verdict = "CORRECTA" if self.is_correct else "INCORRECTA"
        parts = [
            f"<VEREDICTO>{verdict}</VEREDICTO>",
            "",
            "<FEEDBACK_DEL_EVALUADOR>",
            self.feedback_text,
            "</FEEDBACK_DEL_EVALUADOR>",
        ]
        if self.cursor_trace and self.cursor_trace.events:
            analysis = self.cursor_trace.to_analysis_string()
            if analysis:
                parts.extend(["", "<ANÁLISIS_TRAZADO_CURSOR>", analysis, "</ANÁLISIS_TRAZADO_CURSOR>"])
        return "\n".join(parts)
