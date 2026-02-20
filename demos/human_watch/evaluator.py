"""
Human-Watch runtime — Evaluator subprocess.

Reads the last completed PerceptionCycle from interactions.md,
runs it through the library's OnlineCriticLoop, and saves state.

Called by monitor.py as a non-blocking subprocess.
Backend: Groq (GROQ_API_KEY en env o .env)
"""

import argparse
import os
import sys
import time
from pathlib import Path

from demos.human_watch.common import DATA_DIR, PROJECT_ROOT

# Cargar .env desde la raiz del proyecto
def _load_dotenv():
    p = PROJECT_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                k, v = s.split("=", 1)
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v

_load_dotenv()

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import re

from prompt_rl import (
    ActivePolicy,
    Actor,
    CriticMemory,
    CriticValidationLoop,
    Leaderboard,
    LLMValidationJudge,
    OnlineCriticLoop,
    PerceptionCycle,
    RewardHistory,
    TwoStageCritic,
)
from prompt_rl.core.policy_schema import ensure_policy_file
from prompt_rl.llm.local_backend import LocalLLMBackend

GROQ_BASE     = "https://api.groq.com/openai/v1"
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")
CRITIC_MODEL  = "llama-3.1-8b-instant"
ACTOR_MODEL   = "llama-3.1-8b-instant"

LOG_FILE           = DATA_DIR / "interactions.md"
SYSTEM_PROMPT_FILE = DATA_DIR / "system_prompt.md"
SYSTEM_PROMPT_LEGACY = DATA_DIR / "system_prompt.txt"
HISTORY_FILE       = DATA_DIR / "reward_history.json"
POPULATION_FILE    = DATA_DIR / "population.json"
MEMORY_FILE        = DATA_DIR / "critic_memory.md"


def _parse_observations(block: str) -> list[str]:
    start = block.find("③ Fase de Observacion")
    end = block.find("④ Senal ACC", start) if start >= 0 else -1
    if start < 0:
        return []
    section = block[start: end if end > start else len(block)]
    events = []
    for line in section.splitlines():
        line = line.strip().lstrip("- ")
        if any(tag in line for tag in ("[DWELL]", "[SELECT]", "[REVIEW_RAG]")):
            events.append(line)
    return events


def _parse_clicks(block: str) -> list[str]:
    start = block.find("[RAW] Telemetria Cruda")
    if start < 0:
        return []
    section = block[start:]
    clicks = []
    for line in section.splitlines():
        line = line.strip().lstrip("- ")
        if "[CLICK]" in line:
            clicks.append(line)
        if len(clicks) >= 10:
            break
    return clicks


def _parse_last_cycle(log_text: str) -> dict:
    last_idx = log_text.rfind("#### Ciclo ")
    if last_idx < 0:
        return {}

    end_idx = log_text.find("\n---", last_idx)
    block = log_text[last_idx: end_idx if end_idx >= 0 else len(log_text)]

    verdict = ""
    if "Resultado: CORRECTO" in block:
        verdict = "CORRECTO"
    elif "Resultado: INCORRECTO" in block:
        verdict = "INCORRECTO"

    comment = ""
    m = re.search(r'\[CORRECCION\]\s*"([^"]*)"', block)
    if m:
        comment = m.group(1)

    user_query = ""
    m = re.search(r'Consulta del Usuario[^\n]*\n"([^"]*)"', block)
    if m:
        user_query = m.group(1)

    bot_response = ""
    m = re.search(r'Accion del Sistema[^\n]*\n"([^"]*)"', block)
    if m:
        bot_response = m.group(1)

    observations = _parse_observations(block) + _parse_clicks(block)

    return {
        "verdict": verdict,
        "comment": comment,
        "user_query": user_query,
        "bot_response": bot_response,
        "dwell_seconds": block.count("[DWELL]") * 0.8,
        "observations": observations,
    }


def _log_verbose_cycle(cycle) -> None:
    print("\n" + "=" * 60)
    print("[Evaluator] 1. CICLO ENVIADO AL CRITIC")
    print("=" * 60)
    print(f"  system_prompt ({len(cycle.system_prompt)} chars):")
    preview = cycle.system_prompt[:400] + "..." if len(cycle.system_prompt) > 400 else cycle.system_prompt
    for line in preview.splitlines():
        print(f"    | {line}")
    print(f"  verdict:      {cycle.verdict}")
    print(f"  comment:      {repr(cycle.comment[:200])}..." if len(cycle.comment) > 200 else f"  comment:      {repr(cycle.comment)}")
    print()


def _log_verbose_reward(result, loop) -> None:
    rf = loop.reward_fn
    H = result.human_feedback
    C = result.critic_output.critic_score
    ch = result.change_ratio
    print("\n" + "=" * 60)
    print("[Evaluator] 3. REWARD CALCULADO")
    print("=" * 60)
    print(f"  R_total = {result.R_total:+.4f}")
    print()


def _log_verbose_gate(result, loop) -> None:
    print("=" * 60)
    print("[Evaluator] 4. GATE")
    print("=" * 60)
    print(f"  should_update: {result.gate.should_update}")
    print(f"  reason:        {result.gate.reason}")
    print()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluator — process last cycle through Critic")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args()


def run() -> None:
    args = _parse_args()
    verbose = args.verbose or os.environ.get("EVALUATOR_VERBOSE", "").lower() in ("1", "true", "yes")

    if not LOG_FILE.exists():
        print("[Evaluator] interactions.md not found — skipping.")
        return

    log_text = LOG_FILE.read_text(encoding="utf-8")
    fields = _parse_last_cycle(log_text)

    if not fields:
        print("[Evaluator] No cycle found in log — skipping.")
        return

    if not fields.get("verdict"):
        for attempt in range(3):
            time.sleep(1.0)
            log_text = LOG_FILE.read_text(encoding="utf-8")
            fields = _parse_last_cycle(log_text)
            if fields.get("verdict"):
                break
        if not fields.get("verdict"):
            print("[Evaluator] Last cycle has no verdict yet - skipping.")
            return

    history = RewardHistory.from_file(HISTORY_FILE)
    if history.converged:
        print(f"[Evaluator] Policy converged — evaluation suspended.")
        return

    system_prompt = ensure_policy_file(SYSTEM_PROMPT_FILE, SYSTEM_PROMPT_LEGACY)
    obs = fields.get("observations", [])
    print(
        f"[Evaluator] Behavioral signals: "
        f"{sum(1 for o in obs if any(t in o for t in ('[DWELL]','[SELECT]','[REVIEW_RAG]')))} "
        f"obs + {sum(1 for o in obs if '[CLICK]' in o)} clicks"
    )
    cycle = PerceptionCycle(
        system_prompt=system_prompt,
        user_query=fields.get("user_query", ""),
        bot_response=fields.get("bot_response", ""),
        verdict=fields["verdict"],
        comment=fields.get("comment", ""),
        dwell_seconds=fields.get("dwell_seconds", 0.0),
        observations=obs,
    )

    if verbose:
        _log_verbose_cycle(cycle)

    if not GROQ_API_KEY:
        print("[Evaluator] GROQ_API_KEY no definida. Define la variable o crea .env")
    print(f"[Evaluator] Critic: {CRITIC_MODEL} | Actor: {ACTOR_MODEL}")

    critic_backend = LocalLLMBackend(
        model=CRITIC_MODEL, base_url=GROQ_BASE, api_key=GROQ_API_KEY or "not-set"
    )
    actor_backend = LocalLLMBackend(
        model=ACTOR_MODEL, base_url=GROQ_BASE, api_key=GROQ_API_KEY or "not-set"
    )
    base_critic = TwoStageCritic(backend=critic_backend, verbose=verbose)
    actor = Actor(backend=actor_backend)
    judge = LLMValidationJudge(backend=critic_backend)
    critic = CriticValidationLoop(
        critic=base_critic,
        actor=actor,
        judge=judge,
        max_iterations=3,
        skip_validation_if_correct=True,
        verbose=verbose,
    )
    policy = ActivePolicy(path=SYSTEM_PROMPT_FILE, backup_dir=str(DATA_DIR / "prompts"))
    lb = Leaderboard.from_file(POPULATION_FILE)
    mem = CriticMemory(path=MEMORY_FILE)
    loop = OnlineCriticLoop(
        critic=critic,
        policy=policy,
        history=history,
        leaderboard=lb,
        memory=mem,
    )

    print("[Evaluator] Running Critic...")
    result = loop.process_cycle(cycle)

    if verbose:
        print("\n" + "=" * 60)
        print("[Evaluator] 2. RESPUESTA DEL CRITIC")
        print("=" * 60)
        co = result.critic_output
        print(f"  critic_score: {co.critic_score:.4f}")
        print(f"  reasoning:    {co.reasoning[:150]}...")
        _log_verbose_reward(result, loop)
        _log_verbose_gate(result, loop)

    print(
        f"[Evaluator] critic_score={result.critic_output.critic_score:.2f}  "
        f"R={result.R_total:+.4f}  "
        f"gate={result.gate.reason}  "
        f"change={result.change_ratio*100:.1f}%"
    )
    if result.gate.should_update:
        print(f"[Evaluator] Policy updated -> v{loop.history.version} ({result.gate.reason})")
    else:
        print("[Evaluator] Policy stable — no update.")

    if result.converged:
        print("[Evaluator] *** CONVERGENCIA ALCANZADA ***")

    loop.save_state(str(HISTORY_FILE), str(POPULATION_FILE))


if __name__ == "__main__":
    run()
