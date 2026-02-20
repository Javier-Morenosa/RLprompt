"""
Automated end-to-end test for monitor.py (Perception Cycle format).

Simulates one complete subconscious comparison loop:
  ① System prompt captured from server
  ② User query sent → bot responds
  ③ Human "looks" at RAG (dwell) and selects text (observation phase)
  ④ Human gives feedback Incorrecto + comentario (se envía al hacer clic)

Verifica que interactions.md contiene un ciclo bien formado y que el evaluator
se ejecutó (evaluator.log actualizado o reward_history con nueva entrada).
"""

import asyncio
import json
import sys
import time
from pathlib import Path

from playwright.async_api import async_playwright

from demos.human_watch.common import DATA_DIR, PROJECT_ROOT
from demos.human_watch import monitor as _mon
from demos.human_watch.monitor import (
    EVENT_LISTENER_JS,
    LOG_FILE,
    PerceptionCycle,
    _dispatch,
    _fetch_system_prompt,
    _write_session_footer,
    _write_session_header,
)

TEST_SESSION = "TEST-AUTO"


async def run() -> bool:
    # Reset monitor state for a clean test run
    _mon._cycle = None
    _mon._cycle_count = 0

    print("[test] Fetching system prompt...")
    system_prompt = _fetch_system_prompt()
    if "unreachable" in system_prompt:
        print(f"[test] ERROR: {system_prompt}")
        return False
    print(f"[test] OK — {len(system_prompt)} chars\n")

    _write_session_header(TEST_SESSION, system_prompt)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        page = await browser.new_page()

        async def handle(event_json: str) -> None:
            event = json.loads(event_json)
            _dispatch(event, TEST_SESSION, system_prompt)

        await page.expose_function("sendEventToPython", handle)
        await page.add_init_script(EVENT_LISTENER_JS)

        print("[test] Opening http://localhost:8000 ...")
        await page.goto("http://localhost:8000")
        await page.wait_for_load_state("networkidle")
        print("[test] Page loaded\n")

        # ── ② Stimulus: send a chat message ──────────────────────────────────
        print("[test] (2) Sending user query...")
        await page.fill("#user-input", "el precio del plan pro incluye IVA?")
        await page.click("#send-btn")
        await page.wait_for_timeout(6000)   # wait for Gemma

        # ── (3) Observation: dwell on RAG, select text, review ───────────────
        print("[test] (3) Dwelling on RAG section...")
        await page.wait_for_selector("[data-section]", timeout=8000)
        await page.locator("[data-section]").first.hover()
        await page.wait_for_timeout(1100)   # > DWELL_MS

        print("[test] (3) Selecting text in RAG...")
        await page.evaluate("""() => {
            const el = document.querySelector('[data-section]');
            const range = document.createRange();
            range.selectNodeContents(el);
            window.getSelection().removeAllRanges();
            window.getSelection().addRange(range);
            document.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
        }""")
        await page.wait_for_timeout(400)

        print("[test] (3) Clicking Review RAG button...")
        await page.click("#review-rag-btn")
        await page.wait_for_timeout(400)

        # ── (4) ACC signal: Incorrecto envía feedback + comentario de inmediato ──
        print("[test] (4) Escribiendo corrección en textarea...")
        await page.fill("#comment-input", "La respuesta no especifico el limite de usuarios.")
        await page.wait_for_timeout(200)

        print("[test] (4) Clicking Incorrecto (No) — envía feedback y comentario...")
        await page.click("#btn-no")
        await page.wait_for_timeout(1500)   # tiempo para monitor escribir + evaluator iniciar

        await browser.close()

    _write_session_footer(TEST_SESSION)

    # ── Esperar a que el evaluator procese (LLM puede tardar) ───────────────────
    evaluator_log = DATA_DIR / "evaluator.log"
    size_before = evaluator_log.stat().st_size if evaluator_log.exists() else 0
    evaluator_ran = False
    print("[test] Esperando ejecución del evaluator (hasta 45s)...")
    for _ in range(45):
        time.sleep(1)
        if evaluator_log.exists() and evaluator_log.stat().st_size > size_before:
            print(f"[test] Evaluator ejecutado (log creció {evaluator_log.stat().st_size - size_before} bytes)")
            evaluator_ran = True
            break
    else:
        print("[test] AVISO: No se detectó nueva escritura en evaluator.log (puede estar aún procesando)")

    # ── Assertions ────────────────────────────────────────────────────────────
    log = LOG_FILE.read_text(encoding="utf-8")
    session_start = log.find(f"Sesion {TEST_SESSION}")
    assert session_start >= 0, "Session header not found"
    block = log[session_start:]

    checks = {
        "Session header":           f"Sesion {TEST_SESSION}"        in block,
        "Ciclo block written":      "#### Ciclo 1"                  in block,
        "(1) Predictive model":     "Modelo Predictivo"             in block,
        "(2) User query":           "Consulta del Usuario"          in block,
        "(2) Bot response":         "Accion del Sistema"            in block,
        "(3) Observation (DWELL)":  "[DWELL]"                       in block,
        "(3) Observation (SELECT)": "[SELECT]"                      in block,
        "(3) Observation (REVIEW)": "[REVIEW_RAG]"                  in block,
        "(4) ACC verdict":          "INCORRECTO"                    in block,
        "(4) Correction comment":   "CORRECCION"                    in block,
        "[RAW] telemetry section":  "Telemetria Cruda"              in block,
        "[RAW] click logged":       "[CLICK]"                       in block,
        "[RAW] cursor logged":      "[CURSOR]"                      in block,
        "Session footer":           f"Fin de Sesion {TEST_SESSION}" in block,
        "Evaluator ejecutado":      evaluator_ran,
    }

    print()
    all_ok = True
    for check, passed in checks.items():
        print(f"  [{'PASS' if passed else 'FAIL'}] {check}")
        if not passed:
            all_ok = False

    if all_ok:
        print("\n[test] ALL CHECKS PASSED")
        print(f"\n--- interactions.md (Ciclo 1 de {TEST_SESSION}) ---")
        start = block.find("#### Ciclo 1")
        end   = block.find("---", start) + 3
        sys_out = block[start:end].encode("utf-8")
        sys.stdout.buffer.write(sys_out)
        sys.stdout.buffer.write(b"\n")
        sys.stdout.buffer.flush()
    else:
        print("\n[test] SOME CHECKS FAILED")

    return all_ok


if __name__ == "__main__":
    ok = asyncio.run(run())
    raise SystemExit(0 if ok else 1)
