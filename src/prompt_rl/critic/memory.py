"""CriticMemory — persistent journal of Critic conclusions and hypotheses.

The Critic is the policy designer: it decides how the Actor's system prompt
should evolve based on human feedback signals.  CriticMemory gives it a
temporal sense — it can read its own past reasoning, validate or discard
prior hypotheses, and detect recurring failure patterns.

Lifecycle per PerceptionCycle
------------------------------
1. OnlineCriticLoop reads memory.observation_lines() and appends them to
   cycle.observations with the [MEMORIA] prefix.
2. LLMPerceptionCritic.evaluate() finds those lines, strips the prefix, and
   places them in the dedicated "TU MEMORIA" section of its prompt.
3. After the gate fires (or not), the loop calls memory.record() with the
   full LoopResult data including the Critic's own nota hypothesis.
4. The memory file is saved: critic_memory.json (machine) + critic_memory.md
   (human-readable dashboard).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

_MAX_ENTRIES     = 20   # entries kept on disk
_CONTEXT_ENTRIES = 6    # entries injected into the Critic prompt

MEMORY_MARKER = "[MEMORIA]"


class CriticMemory:
    """
    Rolling journal of Critic evaluations.

    Each entry records: timestamp, policy version, verdict, reward,
    trend, reasoning (from CriticOutput), nota/hypothesis (from
    CriticOutput), and the key behavioral signals observed.

    The markdown file (critic_memory.md) is a human-readable dashboard.
    The JSON sidecar (critic_memory.json) is used for reliable re-loading.
    """

    def __init__(self, path: str | Path = "critic_memory.md") -> None:
        self.path    = Path(path)
        self._json   = self.path.with_suffix(".json")
        self._entries: list[dict] = []
        self._load()

    # ── Recording ────────────────────────────────────────────────────────────

    def record(
        self,
        *,
        policy_version: int,
        verdict:        str,
        critic_score:   float,
        R_total:        float,
        reward_trend:   str,
        change_ratio:   float,
        gate_reason:    str,
        policy_updated: bool,
        reasoning:      str,
        nota:           str,
        key_signals:    list[str],
    ) -> None:
        """Append one entry to the journal and persist to disk."""
        entry = {
            "ts":             datetime.now().strftime("%Y-%m-%d %H:%M"),
            "policy_version": policy_version,
            "verdict":        verdict,
            "critic_score":   round(critic_score, 3),
            "R_total":        round(R_total, 4),
            "reward_trend":   reward_trend,
            "change_pct":     f"{change_ratio * 100:.1f}%",
            "gate_reason":    gate_reason,
            "policy_updated": policy_updated,
            "reasoning":      reasoning[:200],
            "nota":           nota[:200],
            "key_signals":    key_signals[:5],
        }
        self._entries.append(entry)
        if len(self._entries) > _MAX_ENTRIES:
            self._entries = self._entries[-_MAX_ENTRIES:]
        self._save()

    # ── Context for Critic prompt ─────────────────────────────────────────────

    def observation_lines(self, n: int = _CONTEXT_ENTRIES) -> list[str]:
        """
        Return last N entries as [MEMORIA]-prefixed observation strings.

        Designed to be appended to cycle.observations so the Critic LLM
        can read its own past conclusions in the prompt.

        Example output:
            "[MEMORIA] v3 | 2026-02-19 09:42 | INCORRECTO | R=-0.12 ..."
            "[MEMORIA]   razonamiento: no menciono el IVA; regla añadida"
            "[MEMORIA]   hipotesis: vigilar preguntas de precio con impuestos"
        """
        if not self._entries:
            return [f"{MEMORY_MARKER} (sin entradas previas — primera iteracion)"]

        lines  = []
        recent = self._entries[-n:]
        for e in reversed(recent):   # most recent first
            upd  = "politica-actualizada" if e["policy_updated"] else "sin-cambio"
            lines.append(
                f"{MEMORY_MARKER} v{e['policy_version']} | {e['ts']} | "
                f"{e['verdict']} | R={e['R_total']:+} score={e['critic_score']} "
                f"cambio={e['change_pct']} | {upd}"
            )
            if e["reasoning"]:
                lines.append(f"{MEMORY_MARKER}   razonamiento: {e['reasoning']}")
            if e["nota"]:
                lines.append(f"{MEMORY_MARKER}   hipotesis: {e['nota']}")
            if e["key_signals"]:
                lines.append(
                    f"{MEMORY_MARKER}   señales: {' | '.join(e['key_signals'])}"
                )
        return lines

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        # JSON sidecar — used for reliable reloading
        self._json.write_text(
            json.dumps(self._entries, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        # Human-readable markdown dashboard
        lines: list[str] = [
            "# Memoria del Critico\n\n",
            f"> Actualizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n",
            f"> Entradas guardadas: {len(self._entries)} / {_MAX_ENTRIES}  \n",
            f"> Entradas inyectadas al Critico: ultimas {_CONTEXT_ENTRIES}\n\n",
        ]
        for e in reversed(self._entries):
            upd = "**SI**" if e["policy_updated"] else "no"
            lines.append(f"## [{e['ts']}] Policy v{e['policy_version']}\n\n")
            lines.append(
                f"| Campo | Valor |\n|---|---|\n"
                f"| Veredicto | {e['verdict']} |\n"
                f"| Critic score | {e['critic_score']} |\n"
                f"| Reward total | {e['R_total']:+} |\n"
                f"| Tendencia | {e['reward_trend']} |\n"
                f"| Cambio politica | {e['change_pct']} |\n"
                f"| Politica actualizada | {upd} |\n"
                f"| Gate | {e['gate_reason']} |\n\n"
            )
            if e["reasoning"]:
                lines.append(f"**Razonamiento:** {e['reasoning']}\n\n")
            if e["nota"]:
                lines.append(
                    f"**Hipotesis para siguiente ciclo:** {e['nota']}\n\n"
                )
            if e["key_signals"]:
                lines.append(
                    f"**Señales clave:** {' | '.join(e['key_signals'])}\n\n"
                )
            lines.append("---\n\n")
        self.path.write_text("".join(lines), encoding="utf-8")

    def _load(self) -> None:
        if self._json.exists():
            try:
                data = json.loads(self._json.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = data
            except Exception:
                pass
