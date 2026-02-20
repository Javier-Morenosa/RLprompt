#!/usr/bin/env python3
"""
Resetea todos los archivos de memoria y estado a valor inicial (estado 0).

Ejecutar desde la raíz del proyecto:
    python -m demos.human_watch.reset_to_state_zero

Permite empezar un testeo desde cero sin arrastrar estado previo.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from demos.human_watch.common import DATA_DIR, PROJECT_ROOT

ROOT = PROJECT_ROOT

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    print(f"  [OK] {path.name}")


def reset_critic_memory() -> None:
    """Vacía la memoria del Critic (critic_memory.md + critic_memory.json)."""
    json_path = DATA_DIR / "critic_memory.json"
    md_path = DATA_DIR / "critic_memory.md"
    json_path.write_text("[]", encoding="utf-8")
    md_path.write_text(
        "# Memoria del Critico\n\n"
        "> Estado reseteado. Sin entradas previas.\n\n",
        encoding="utf-8",
    )
    print("  [OK] critic_memory.json")
    print("  [OK] critic_memory.md")


def reset_interactions() -> None:
    """Vacía el log de interacciones (interactions.md)."""
    path = DATA_DIR / "interactions.md"
    path.write_text("", encoding="utf-8")
    print("  [OK] interactions.md")


def reset_system_prompt() -> None:
    """Restaura system_prompt.md a la política mínima (asistente + responde en español)."""
    from prompt_rl.core.policy_schema import MINIMAL_POLICY_JSON

    path = DATA_DIR / "system_prompt.md"
    path.write_text(MINIMAL_POLICY_JSON.strip(), encoding="utf-8")
    print("  [OK] system_prompt.md (política mínima)")


def reset_reward_history() -> None:
    """Resetea el historial de recompensas y convergencia."""
    path = DATA_DIR / "reward_history.json"
    data = {
        "version": 0,
        "consecutive_stable": 0,
        "converged": False,
        "history": [],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  [OK] reward_history.json")


def reset_population() -> None:
    """Vacía el leaderboard de población."""
    path = DATA_DIR / "population.json"
    data = {
        "generation": 0,
        "elite_size": 2,
        "individuals": [],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("  [OK] population.json")


def main() -> int:
    print("Reset a estado 0 — RLprompt")
    print("=" * 40)

    DATA_DIR.mkdir(exist_ok=True)
    reset_critic_memory()
    reset_interactions()
    reset_system_prompt()
    reset_reward_history()
    reset_population()

    print()
    print("Listo. Puedes empezar un testeo desde cero.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
