"""
run_backend.py - Lanza y monitorea todo el backend de Human-Watch desde CMD.

Inicia el servidor (uvicorn) en segundo plano, espera a que esté listo,
y luego ejecuta el monitor (Playwright + navegador). Al cerrar el navegador,
se detiene el servidor. La salida del monitor (ciclos, veredictos) aparece
en la consola. Con --verbose, también se muestra la evolución completa de
lo que recibe y procesa el evaluator (ciclo, Critic, reward, gate).

Uso:
    python -m demos.human_watch.run_backend              # puerto 8000
    python -m demos.human_watch.run_backend -v            # modo verbose
    python -m demos.human_watch.run_backend --verbose 9000
"""

import argparse
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from demos.human_watch.common import PROJECT_ROOT

_PORT_DEFAULT = 8000


def _free_port(port: int) -> None:
    """Libera el puerto matando procesos que lo usen (Windows netstat / POSIX lsof)."""
    freed = False
    try:
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        pids = set()
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                if parts:
                    pids.add(parts[-1])
        for pid in pids:
            subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
            print(f"[Backend] Puerto {port} liberado (PID {pid})")
            freed = True
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True
            )
            for pid in result.stdout.split():
                subprocess.run(["kill", "-9", pid])
                print(f"[Backend] Puerto {port} liberado (PID {pid})")
                freed = True
        except Exception:
            pass

    if freed:
        time.sleep(0.8)


def _wait_for_server(url: str, timeout_sec: float = 30) -> bool:
    """Espera a que el servidor responda. Devuelve True si está listo."""
    start = time.time()
    while (time.time() - start) < timeout_sec:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lanza y monitorea el backend Human-Watch (servidor + monitor)."
    )
    parser.add_argument(
        "port",
        nargs="?",
        type=int,
        default=_PORT_DEFAULT,
        help=f"Puerto del servidor (default: {_PORT_DEFAULT})",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Mostrar evolución completa: datos que llegan al monitor y procesamiento del evaluator",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    port = args.port
    base_url = f"http://localhost:{port}"

    os.chdir(PROJECT_ROOT)

    _free_port(port)

    # Iniciar servidor en segundo plano (módulo demos.human_watch.server)
    print(f"[Backend] Iniciando servidor en {base_url} ...")
    server_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "demos.human_watch.server:app",
            "--port", str(port),
            "--reload",
        ],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    if not _wait_for_server(base_url):
        server_proc.kill()
        print("[Backend] ERROR: El servidor no respondió a tiempo.")
        sys.exit(1)

    if args.verbose:
        print("[Backend] Modo verbose: verás la evolución del monitor y del evaluator en la consola.\n")

    print(f"[Backend] Servidor listo en {base_url}\n")

    try:
        env = dict(os.environ)
        env["TARGET_URL"] = base_url
        if args.verbose:
            env["RUN_BACKEND_VERBOSE"] = "1"
            env["EVALUATOR_VERBOSE"] = "1"
        subprocess.run(
            [sys.executable, "-m", "demos.human_watch.monitor"],
            cwd=PROJECT_ROOT,
            env=env,
        )
    finally:
        server_proc.terminate()
        try:
            server_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server_proc.kill()
        print("\n[Backend] Servidor detenido.")


if __name__ == "__main__":
    main()
