"""
Human-Watch: Playwright monitoring agent.

Structured as a Perception Cycle engine. Each bot response opens a new cycle.
Feedback closes it and writes to interactions.md.

Usage:
  1. Start the chat server:  python -m uvicorn demos.human_watch.server:app --port 8000
  2. Run this script:        python -m demos.human_watch.monitor
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from demos.human_watch.common import DATA_DIR, PROJECT_ROOT

# Asegurar que prompt_rl es importable
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from playwright.async_api import async_playwright

TARGET_URL    = os.environ.get("TARGET_URL", "http://localhost:8000")
LOG_FILE      = DATA_DIR / "interactions.md"
HISTORY_FILE  = DATA_DIR / "reward_history.json"

EVENT_LISTENER_JS = """
(function () {
  document.addEventListener('click', function (e) {
    var t = e.target;
    var label = (t.innerText || t.value || t.id || t.className || t.tagName || '')
                  .trim().substring(0, 100);
    sendEventToPython(JSON.stringify({
      type:    'click',
      element: t.tagName,
      id:      t.id || '',
      label:   label
    }));
  }, true);

  document.addEventListener('mouseup', function () {
    var sel = window.getSelection ? window.getSelection().toString().trim() : '';
    if (sel && sel.length > 2) {
      sendEventToPython(JSON.stringify({
        type: 'selection',
        text: sel.substring(0, 500)
      }));
    }
  });

  console.log('[Human-Watch] Event listener active.');
})();
"""


def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


@dataclass
class PerceptionCycle:
    cycle_num:     int
    session_id:    str
    system_prompt: str
    started_at:    str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S.%f")[:-3])
    user_query:    str = ""
    bot_response:  str = ""
    observations:  list[str] = field(default_factory=list)
    raw_events:    list[str] = field(default_factory=list)
    verdict:       str = ""
    comment:       str = ""

    def has_stimulus(self) -> bool:
        return bool(self.bot_response)

    def is_judged(self) -> bool:
        return bool(self.verdict)


_cycle:       PerceptionCycle | None = None
_cycle_count: int = 0


def _new_cycle(session_id: str, system_prompt: str) -> PerceptionCycle:
    global _cycle_count
    _cycle_count += 1
    return PerceptionCycle(
        cycle_num=_cycle_count,
        session_id=session_id,
        system_prompt=system_prompt,
    )


_ARCHIVE_THRESHOLD_BYTES = 200 * 1024


def _maybe_archive() -> None:
    if not LOG_FILE.exists() or LOG_FILE.stat().st_size < _ARCHIVE_THRESHOLD_BYTES:
        return
    logs_dir = DATA_DIR / "logs"
    logs_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    archive_path = logs_dir / f"interactions_{date_str}.md"
    counter = 1
    while archive_path.exists():
        archive_path = logs_dir / f"interactions_{date_str}_{counter}.md"
        counter += 1
    archive_path.write_bytes(LOG_FILE.read_bytes())
    LOG_FILE.write_text("", encoding="utf-8")
    print(f"[Monitor] Archivado -> {archive_path}")


def _write_session_header(session_id: str, system_prompt: str) -> None:
    _maybe_archive()
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n# Sesion {session_id} — {ts}\n")
        f.write(f"**System Prompt activo:** \"{system_prompt}\"\n")


def _write_session_footer(session_id: str) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n# Fin de Sesion {session_id}\n")


def _write_cycle(c: PerceptionCycle) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(f"\n#### Ciclo {c.cycle_num} — Sesion {c.session_id} — {ts}\n\n")

        f.write("**① Modelo Predictivo (System Prompt):**\n")
        f.write(f'"{c.system_prompt}"\n\n')

        if c.user_query:
            f.write("**② Consulta del Usuario:**\n")
            f.write(f'"{c.user_query}"\n\n')
        f.write("**② Accion del Sistema (Respuesta del Bot):**\n")
        f.write(f'"{c.bot_response or "(sin respuesta registrada)"}"\n\n')

        f.write("**③ Fase de Observacion — Busqueda de Informacion:**\n")
        if c.observations:
            for line in c.observations:
                f.write(f"{line}\n")
        else:
            f.write("- (sin observaciones: el humano no consulto el RAG)\n")
        f.write("\n")

        f.write("**④ Senal ACC — Veredicto de Comparacion:**\n")
        if c.verdict == "CORRECTO":
            f.write("- Resultado: CORRECTO — El modelo predictivo coincidio con la realidad\n")
        elif c.verdict == "INCORRECTO":
            f.write("- Resultado: INCORRECTO — Discrepancia detectada (ACC activada)\n")
            if c.comment:
                f.write(f'- [CORRECCION] "{c.comment}"\n')
        else:
            f.write("- (sesion terminada sin veredicto explicito)\n")
        f.write("\n")

        elapsed = (
            datetime.strptime(ts.split(" ")[1], "%H:%M:%S")
            - datetime.strptime(c.started_at.split(".")[0], "%H:%M:%S")
        )
        f.write(f"**[RAW] Telemetria Cruda — duracion del ciclo: {elapsed.seconds}s**\n")
        f.write(f"- Ciclo iniciado: {c.started_at} | Ciclo cerrado: {ts.split(' ')[1]}\n")
        if c.raw_events:
            for line in c.raw_events:
                f.write(f"{line}\n")
        else:
            f.write("- (sin eventos crudos registrados)\n")

        f.write("\n---\n")
        f.flush()
        if hasattr(os, "fsync"):
            try:
                os.fsync(f.fileno())
            except (OSError, AttributeError):
                pass


def _trigger_evaluator() -> None:
    time.sleep(0.5)
    if HISTORY_FILE.exists():
        try:
            from prompt_rl.rl.history import RewardHistory
            history = RewardHistory.from_file(HISTORY_FILE)
            if history.converged:
                print(
                    f"  [Monitor] Policy convergida "
                    f"({history.consecutive_stable} ciclos estables) — evaluator omitido."
                )
                return
        except Exception:
            pass

    log_path = DATA_DIR / "evaluator.log"
    stream_to_console = os.environ.get("RUN_BACKEND_VERBOSE", "").lower() in ("1", "true", "yes")
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "demos.human_watch.evaluator"]
    if os.environ.get("EVALUATOR_VERBOSE"):
        cmd.append("--verbose")
    try:
        if stream_to_console:
            subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=None,
                stderr=None,
                env=env,
            )
            print("  [Monitor] Evaluator lanzado (salida en consola)")
        else:
            log_fh = log_path.open("a", encoding="utf-8")
            log_fh.write(f"\n--- {datetime.now().isoformat()} ---\n")
            log_fh.flush()
            subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=log_fh,
                stderr=log_fh,
                env=env,
            )
            verbose_hint = " (verbose)" if os.environ.get("EVALUATOR_VERBOSE") else ""
            print(f"  [Monitor] Evaluator lanzado -> {log_path}{verbose_hint}")
    except Exception as exc:
        print(f"  [Monitor] ERROR al lanzar evaluator: {exc}")


def _dispatch(event: dict, session_id: str, system_prompt: str) -> None:
    global _cycle
    etype = event.get("type", "unknown")
    ts = _ts()

    if etype == "user_query":
        if _cycle and _cycle.is_judged():
            _write_cycle(_cycle)
            _trigger_evaluator()
        elif _cycle and not _cycle.is_judged():
            _write_cycle(_cycle)
        _cycle = _new_cycle(session_id, system_prompt)
        _cycle.user_query = event.get("text", "")
        print(f"  [Monitor] Ciclo {_cycle.cycle_num} abierto | Consulta: \"{_cycle.user_query[:60]}\"")

    elif etype == "bot_response":
        if _cycle is None:
            _cycle = _new_cycle(session_id, system_prompt)
        _cycle.bot_response = event.get("text", "")
        print(f"  [Monitor] Ciclo {_cycle.cycle_num} | Bot respondió ({len(_cycle.bot_response)} chars)")

    elif etype == "mouse_dwell" and _cycle and _cycle.has_stimulus():
        zone = event.get("zone", "?")
        section = event.get("section", "?")
        preview = event.get("preview", "")
        x, y = event.get("x", 0), event.get("y", 0)
        _cycle.observations.append(
            f"  - [{ts}] [DWELL] {zone} → seccion \"{section}\" @ ({x},{y}) | \"{preview}\""
        )
        print(f"  [Monitor] DWELL {zone}/{section} @ ({x},{y})")

    elif etype == "selection" and _cycle and _cycle.has_stimulus():
        text = event.get("text", "")
        _cycle.observations.append(
            f"  - [{ts}] [SELECT] Texto seleccionado del RAG: \"{text}\""
        )
        print(f"  [Monitor] SELECT \"{text[:60]}\"")

    elif etype == "review_rag" and _cycle and _cycle.has_stimulus():
        _cycle.observations.append(
            f"  - [{ts}] [REVIEW_RAG] El humano senalo revision activa de la documentacion"
        )
        print(f"  [Monitor] REVIEW_RAG")

    elif etype == "incorrect_with_comment" and _cycle:
        if not _cycle.has_stimulus() and event.get("lastBotMessage"):
            _cycle.bot_response = event.get("lastBotMessage", "")
        if _cycle.has_stimulus():
            _cycle.verdict = "INCORRECTO"
            _cycle.comment = event.get("text", "")
            print(f"  [Monitor] Ciclo {_cycle.cycle_num} | Veredicto: INCORRECTO | Correccion: \"{_cycle.comment[:60]}\"")
            _write_cycle(_cycle)
            print(f"  [Monitor] Ciclo {_cycle.cycle_num} cerrado | Escrito en log")
            _trigger_evaluator()
            _cycle = None
        else:
            print("  [Monitor] incorrect_with_comment ignorado: ciclo sin bot_response")

    elif etype == "feedback" and _cycle and _cycle.has_stimulus():
        value = event.get("value", "")
        _cycle.verdict = "CORRECTO" if value == "yes" else "INCORRECTO"
        print(f"  [Monitor] Ciclo {_cycle.cycle_num} | Veredicto: {_cycle.verdict}")

    elif etype == "comment" and _cycle:
        _cycle.comment = event.get("text", "")
        print(f"  [Monitor] Ciclo {_cycle.cycle_num} | Corrección: \"{_cycle.comment[:60]}\"")
        if _cycle.is_judged():
            _write_cycle(_cycle)
            print(f"  [Monitor] Ciclo {_cycle.cycle_num} cerrado | Escrito en log")
            _trigger_evaluator()
            _cycle = None

    elif etype == "click":
        label = event.get("label", "")
        elem = event.get("element", "?")
        eid = event.get("id", "")
        id_str = f"#{eid}" if eid else ""
        line = f"  - [{_ts()}] [CLICK] {elem}{id_str} | \"{label}\""
        if _cycle:
            _cycle.raw_events.append(line)
        print(f"  [Monitor] CLICK \"{label[:60]}\"")

    elif etype == "cursor_sample":
        zone = event.get("zone", "?")
        section = event.get("section", "?")
        x = event.get("x", 0)
        y = event.get("y", 0)
        label = event.get("label", "")
        line = f"  - [{_ts()}] [CURSOR] {zone}/{section} @ ({x},{y}) | \"{label}\""
        if _cycle and _cycle.has_stimulus():
            _cycle.raw_events.append(line)


def _fetch_system_prompt() -> str:
    try:
        import urllib.request
        with urllib.request.urlopen(f"{TARGET_URL}/system-prompt", timeout=3) as resp:
            return json.loads(resp.read()).get("system_prompt", "N/A")
    except Exception:
        return "N/A (server unreachable)"


async def main() -> None:
    global _cycle
    session_id = str(uuid.uuid4())[:8].upper()
    system_prompt = _fetch_system_prompt()

    print(f"[Monitor] Sesión {session_id} | URL: {TARGET_URL}")
    print(f"[Monitor] Log: {LOG_FILE.resolve()}\n")

    _write_session_header(session_id, system_prompt)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=False)
        page = await browser.new_page()

        async def _handle(event_json: str) -> None:
            try:
                event = json.loads(event_json)
                _dispatch(event, session_id, system_prompt)
            except json.JSONDecodeError as exc:
                print(f"[Monitor] JSON inválido: {exc}")

        await page.expose_function("sendEventToPython", _handle)
        await page.add_init_script(EVENT_LISTENER_JS)
        await page.goto(TARGET_URL)

        print("[Monitor] Navegador abierto. Cierra la ventana para terminar.\n")
        await page.wait_for_event("close", timeout=0)

    if _cycle and _cycle.has_stimulus():
        print(f"  [Monitor] Ciclo {_cycle.cycle_num} cerrado al finalizar sesión")
        _write_cycle(_cycle)

    _write_session_footer(session_id)
    print(f"\n[Monitor] Sesión {session_id} finalizada.")


if __name__ == "__main__":
    asyncio.run(main())
