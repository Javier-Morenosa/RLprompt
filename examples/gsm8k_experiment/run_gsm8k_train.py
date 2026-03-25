"""
Experimento GSM8K: entrenar un system prompt en 80 muestras, evaluar en 20.

Objetivo: el loop de refinamiento genera un system prompt que mejora la
accuracy en el conjunto de test.  Usa semilla fija para reproducibilidad.
El Critic recibe feedback genérico (sin la respuesta correcta).

Ahora usa las primitivas de la librería:
  - DatasetSplit.from_dicts()  — gestiona el split train/test
  - ExactMatchJudge            — juez determinístico (sin LLM extra)
  - DatasetLoop                — itera sobre el dataset y llama a OnlineCriticLoop
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT   = Path(__file__).resolve().parent.parent.parent
GSM8K_DIR      = PROJECT_ROOT / "data" / "gsm8k"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "gsm8k_experiment"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _load_dotenv() -> None:
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

from prompt_rl import (
    ActivePolicy,
    Actor,
    CriticValidationLoop,
    DatasetLoop,
    DatasetSplit,
    ExactMatchJudge,
    Leaderboard,
    RewardHistory,
    TwoStageCritic,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

# Config
GROQ_BASE    = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL        = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
SEED         = 42


def load_split() -> DatasetSplit:
    """Carga train y test desde data/gsm8k/ como DatasetSplit."""
    train_raw = json.loads((GSM8K_DIR / "train.json").read_text(encoding="utf-8"))
    test_raw  = json.loads((GSM8K_DIR / "test.json").read_text(encoding="utf-8"))
    return DatasetSplit.from_dicts(
        train=train_raw,
        test=test_raw,
        question_key="question",
        answer_key="answer",
        extracted_key="extracted",
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Experimento GSM8K: refinar prompt desde vacio"
    )
    parser.add_argument(
        "--curate",
        action="store_true",
        help="Usar curate_hard_gsm8k.py (muestras donde prompt vacio falla)",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        metavar="N",
        help="Evaluar en test cada N pasos de entrenamiento (0 = solo al final)",
    )
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY no definida. Define la variable o crea .env")
        sys.exit(1)

    # Descargar o curar datos si no existen
    if args.curate or not (GSM8K_DIR / "train.json").exists():
        script = "curate_hard_gsm8k.py" if args.curate else "download_gsm8k.py"
        cmd = [sys.executable, str(PROJECT_ROOT / "examples" / "gsm8k_experiment" / script)]
        if args.curate and (GSM8K_DIR / "hard_pool.json").exists():
            cmd.append("--use-cached")
        print(f"Preparando datos con {script}...")
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    split = load_split()
    print(f"Train: {len(split.train)} muestras, Test: {len(split.test)} muestras")

    # Usar acc_empty de meta.json si disponible (evita re-evaluar con prompt vacio)
    meta_path = GSM8K_DIR / "meta.json"
    acc_empty_cached = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("curated") and "acc_empty" in meta:
            acc_empty_cached = meta["acc_empty"]
            print(f"  (datos curados: acc_empty={acc_empty_cached:.1%} en cache)")

    # Backends
    backend = LocalLLMBackend(model=MODEL, base_url=GROQ_BASE, api_key=GROQ_API_KEY)
    actor   = Actor(backend=backend, max_tokens=512, temperature=0.1)

    # Judge determinístico (ExactMatchJudge cubre tanto DatasetJudge
    # como ValidationJudge para CriticValidationLoop)
    judge = ExactMatchJudge(
        extract_pattern=r"####\s*(.+)$",
        feedback_msg=(
            "El calculo o razonamiento fue incorrecto. "
            "Mejora el system prompt para que el modelo siga "
            "los pasos correctos en problemas matematicos."
        ),
    )

    # Critic con validacion iterativa usando el mismo judge determinístico
    critic = CriticValidationLoop(
        critic=TwoStageCritic(backend=backend, verbose=False),
        actor=actor,
        judge=judge,
        max_iterations=3,
        skip_validation_if_correct=True,
        verbose=False,
    )

    # Policy, history, leaderboard — partir de prompt vacio
    policy_path     = EXPERIMENT_DIR / "system_prompt.md"
    history_path    = EXPERIMENT_DIR / "reward_history.json"
    leaderboard_path = EXPERIMENT_DIR / "population.json"

    policy = ActivePolicy(path=str(policy_path), backup_dir=str(EXPERIMENT_DIR / "prompts"))
    policy.write("", 0)

    # DatasetLoop — une todo
    loop = DatasetLoop(
        critic=critic,
        policy=policy,
        actor=actor,
        judge=judge,
        history=RewardHistory(),
        leaderboard=Leaderboard(),
        verbose=True,
    )

    # Si los datos son curados, saltamos la evaluación baseline (ya está en meta.json)
    test_for_baseline = None if acc_empty_cached is not None else split.test

    result = loop.train(
        train_samples=split.train,
        test_samples=split.test,
        eval_every=args.eval_every,
    )

    # Si usamos acc_empty desde cache, sobreescribir el resultado
    if acc_empty_cached is not None:
        result.acc_before = acc_empty_cached
        result.delta = result.acc_after - acc_empty_cached
        print(f"\nAccuracy (cache) con prompt vacio: {acc_empty_cached:.1%}")

    loop.save_state(str(history_path), str(leaderboard_path))

    print(f"\nAccuracy: {result.acc_before:.1%} -> {result.acc_after:.1%} "
          f"(delta={result.delta:+.1%}, updates={result.n_updates})")

    # Guardar resultados
    results_data = {
        "acc_before": result.acc_before,
        "acc_after":  result.acc_after,
        "delta":      result.delta,
        "n_updates":  result.n_updates,
        "n_train":    result.n_train,
        "n_test":     result.n_test,
        "seed":       SEED,
    }
    out_path = EXPERIMENT_DIR / "results.json"
    out_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")
    print(f"Resultados guardados en {out_path}")


if __name__ == "__main__":
    main()
