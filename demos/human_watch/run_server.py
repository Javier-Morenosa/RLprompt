"""
run_server.py — Safe launcher for the Human-Watch chat server.

Automatically frees port 8000 (kills any process using it) before
starting uvicorn, so you never hit "address already in use" errors.

Usage:
    python -m demos.human_watch.run_server          # default port 8000
    python -m demos.human_watch.run_server 9000    # custom port
"""

import os
import subprocess
import sys
import time

from demos.human_watch.common import PROJECT_ROOT


def free_port(port: int) -> None:
    """Kill every process listening on *port* (Windows netstat / POSIX lsof)."""
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
            subprocess.run(["taskkill", "/PID", pid, "/F"],
                           capture_output=True)
            print(f"[run_server] Freed port {port} (killed PID {pid})")
            freed = True
    except FileNotFoundError:
        try:
            result = subprocess.run(
                ["lsof", "-ti", f"tcp:{port}"], capture_output=True, text=True
            )
            for pid in result.stdout.split():
                subprocess.run(["kill", "-9", pid])
                print(f"[run_server] Freed port {port} (killed PID {pid})")
                freed = True
        except Exception:
            pass

    if freed:
        time.sleep(0.8)


def main() -> None:
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    os.chdir(PROJECT_ROOT)
    free_port(port)
    print(f"[run_server] Starting uvicorn on http://localhost:{port}")
    subprocess.run(
        [
            sys.executable, "-m", "uvicorn",
            "demos.human_watch.server:app",
            "--port", str(port),
            "--reload",
        ],
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
