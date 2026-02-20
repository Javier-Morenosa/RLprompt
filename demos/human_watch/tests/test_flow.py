"""
Test del flujo: texto -> Incorrecto -> evaluator ejecutado.

Simula los eventos del monitor llamando a _dispatch directamente (sin Playwright).
Verifica que se escribe un ciclo en interactions.md y que el evaluator se ejecuta.

Requiere: servidor en localhost:8000 (para system prompt) y Ollama para el evaluator.
Si el servidor no está disponible, usa un system prompt de prueba.
"""

import sys
import time
from pathlib import Path

from demos.human_watch.common import DATA_DIR, PROJECT_ROOT
from demos.human_watch import monitor as _mon
from demos.human_watch.monitor import (
    LOG_FILE,
    _dispatch,
    _fetch_system_prompt,
    _write_session_footer,
    _write_session_header,
)

TEST_SESSION = "TEST-FLOW"


def run() -> bool:
    _mon._cycle = None
    _mon._cycle_count = 0

    print("[test_flow] Obteniendo system prompt...")
    try:
        system_prompt = _fetch_system_prompt()
        if "unreachable" in system_prompt:
            system_prompt = '{"role": "Asistente de prueba.", "hard_rules": [], "context_amplification": [], "soft_guidelines": []}'
            print("[test_flow] Servidor no disponible, usando prompt de prueba")
        else:
            print(f"[test_flow] OK — {len(system_prompt)} chars")
    except Exception:
        system_prompt = '{"role": "Asistente de prueba.", "hard_rules": [], "context_amplification": [], "soft_guidelines": []}'
        print("[test_flow] Usando prompt de prueba")

    _write_session_header(TEST_SESSION, system_prompt)

    # Simular flujo: user_query -> bot_response -> feedback Incorrecto + comment
    print("[test_flow] (1) Simulando user_query...")
    _dispatch(
        {"type": "user_query", "text": "el precio del plan pro incluye IVA?"},
        TEST_SESSION,
        system_prompt,
    )

    print("[test_flow] (2) Simulando bot_response...")
    _dispatch(
        {
            "type": "bot_response",
            "text": "El Plan Pro cuesta 79 €/mes. El precio incluye IVA del 21 %.",
        },
        TEST_SESSION,
        system_prompt,
    )

    print("[test_flow] (3) Simulando feedback Incorrecto...")
    _dispatch(
        {"type": "feedback", "value": "no", "label": "Incorrecto (No)", "lastBotMessage": ""},
        TEST_SESSION,
        system_prompt,
    )

    print("[test_flow] (4) Simulando comment (corrección)...")
    _dispatch(
        {"type": "comment", "text": "La respuesta no especifico el limite de usuarios."},
        TEST_SESSION,
        system_prompt,
    )

    _write_session_footer(TEST_SESSION)

    # Esperar evaluator
    evaluator_log = DATA_DIR / "evaluator.log"
    size_before = evaluator_log.stat().st_size if evaluator_log.exists() else 0
    evaluator_ran = False
    print("[test_flow] Esperando evaluator (hasta 45s)...")
    for _ in range(45):
        time.sleep(1)
        if evaluator_log.exists() and evaluator_log.stat().st_size > size_before:
            print(f"[test_flow] Evaluator ejecutado (log +{evaluator_log.stat().st_size - size_before} bytes)")
            evaluator_ran = True
            break
    else:
        print(
            "[test_flow] AVISO: evaluator.log no creció "
            "(¿Ollama corriendo? ¿Modelo con memoria suficiente?)"
        )

    # Verificar interactions.md
    log = LOG_FILE.read_text(encoding="utf-8")
    block = log[log.find(f"Sesion {TEST_SESSION}"):]

    checks = {
        "Sesion escrita": f"Sesion {TEST_SESSION}" in block,
        "Ciclo con veredicto INCORRECTO": "INCORRECTO" in block,
        "Comentario de corrección": "CORRECCION" in block,
        "Evaluator ejecutado": evaluator_ran,
    }

    print()
    all_ok = True
    for name, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\n[test_flow] TODOS LOS CHECKS PASARON")
    else:
        print("\n[test_flow] ALGUNOS CHECKS FALLARON")

    return all_ok


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
