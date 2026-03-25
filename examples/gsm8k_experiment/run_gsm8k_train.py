"""
Experimento GSM8K: entrenar un system prompt en 80 muestras, evaluar en 20.

Objetivo: el loop de refinamiento genera un system prompt que mejora las métricas en test.
Usa semilla para reproducibilidad. Feedback genérico (sin respuesta correcta) al Critic.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GSM8K_DIR = PROJECT_ROOT / "data" / "gsm8k"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "gsm8k_experiment"
EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)

# Añadir src al path
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Cargar .env
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

# Dependencias
from prompt_rl import (
    ActivePolicy,
    Actor,
    CriticValidationLoop,
    Leaderboard,
    LLMValidationJudge,
    OnlineCriticLoop,
    PerceptionCycle,
    RewardHistory,
    TwoStageCritic,
)
from prompt_rl.llm.local_backend import LocalLLMBackend
from prompt_rl.validation.judge import ValidationJudge, ValidationResult

# Config
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"
SEED = 42

# Feedback genérico para el Critic (sin dar la respuesta correcta)
GENERIC_FEEDBACK_INCORRECT = (
    "El cálculo o razonamiento fue incorrecto. "
    "Mejora el system prompt para que el modelo siga los pasos correctos en problemas matemáticos."
)


def extract_answer_from_response(text: str) -> str:
    """Extrae la respuesta numérica de la respuesta del modelo."""
    text = text.strip()
    # Formato GSM8K: #### 42
    if "####" in text:
        return text.split("####")[-1].strip()
    # Último número en la respuesta (común en modelos)
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    if numbers:
        return numbers[-1]
    return ""


def normalize_answer(s: str) -> str:
    """Normaliza para comparación (ej. 12.0 vs 12)."""
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


# Judge determinístico para GSM8K (usa ground truth, no LLM)
class GSM8KJudge(ValidationJudge):
    """Juez que compara la respuesta extraída con ground truth."""

    def __init__(self) -> None:
        self._current_gt: str = ""

    def set_ground_truth(self, gt: str) -> None:
        self._current_gt = normalize_answer(gt)

    def judge(
        self,
        user_query: str,
        original_feedback: str,
        original_response: str,
        new_response: str,
    ) -> ValidationResult:
        extracted = normalize_answer(extract_answer_from_response(new_response))
        fixed = extracted == self._current_gt
        return ValidationResult(fixed=fixed, reasoning=f"Extracted: {extracted}, GT: {self._current_gt}")


def load_data() -> tuple[list[dict], list[dict]]:
    """Carga train y test desde data/gsm8k/."""
    train = json.loads((GSM8K_DIR / "train.json").read_text(encoding="utf-8"))
    test = json.loads((GSM8K_DIR / "test.json").read_text(encoding="utf-8"))
    return train, test


def evaluate_accuracy(
    actor: Actor,
    system_prompt: str,
    samples: list[dict],
) -> float:
    """Evalúa accuracy en un conjunto de muestras."""
    correct = 0
    for s in samples:
        response = actor.generate(system_prompt, s["question"])
        extracted = normalize_answer(extract_answer_from_response(response))
        gt = normalize_answer(s["extracted"])
        if extracted == gt:
            correct += 1
    return correct / len(samples) if samples else 0.0


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Experimento GSM8K: refinar prompt desde vacío")
    parser.add_argument(
        "--curate",
        action="store_true",
        help="Usar curate_hard_gsm8k.py para generar dataset de muestras difíciles (prompt vacío falla)",
    )
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY no definida. Define la variable o crea .env")
        sys.exit(1)

    # Descargar o curar datos
    if args.curate or not (GSM8K_DIR / "train.json").exists():
        if args.curate:
            hard_pool = GSM8K_DIR / "hard_pool.json"
            use_cached = "--use-cached" if hard_pool.exists() else ""
            print("Curando dataset (muestras difíciles para prompt vacío)...")
            script = "curate_hard_gsm8k.py"
        else:
            use_cached = ""
            print("Descargando GSM8K...")
            script = "download_gsm8k.py"
        import subprocess
        cmd = [sys.executable, str(PROJECT_ROOT / "examples" / "gsm8k_experiment" / script)]
        if use_cached:
            cmd.append(use_cached)
        subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

    train, test = load_data()
    print(f"Train: {len(train)} muestras, Test: {len(test)} muestras")

    # Si hay datos curados, usar acc_empty de meta.json (evita re-evaluar con prompt vacío)
    meta_path = GSM8K_DIR / "meta.json"
    acc_empty_cached = None
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("curated") and "acc_empty" in meta:
            acc_empty_cached = meta["acc_empty"]
            print(f"  (datos curados: acc_empty en caché)")

    # Backends
    backend = LocalLLMBackend(
        model=MODEL,
        base_url=GROQ_BASE,
        api_key=GROQ_API_KEY,
    )

    # Critic (TwoStageCritic) + Judge para validación
    # Usamos GSM8KJudge (determinístico) en lugar de LLMValidationJudge
    gsm8k_judge = GSM8KJudge()
    critic = TwoStageCritic(backend=backend, verbose=False)
    actor = Actor(backend=backend, max_tokens=512, temperature=0.1)
    validated_critic = CriticValidationLoop(
        critic=critic,
        actor=actor,
        judge=gsm8k_judge,
        max_iterations=3,
        skip_validation_if_correct=True,
        verbose=False,
    )

    # Policy, history, leaderboard
    policy_path = EXPERIMENT_DIR / "system_prompt.md"
    history_path = EXPERIMENT_DIR / "reward_history.json"
    leaderboard_path = EXPERIMENT_DIR / "population.json"

    # Partir de system prompt vacío
    policy = ActivePolicy(path=str(policy_path), backup_dir=str(EXPERIMENT_DIR / "prompts"))
    policy.write("", 0)

    history = RewardHistory()
    leaderboard = Leaderboard()

    loop = OnlineCriticLoop(
        critic=validated_critic,
        policy=policy,
        history=history,
        leaderboard=leaderboard,
    )

    # Accuracy con prompt vacío (antes del entrenamiento)
    if acc_empty_cached is not None:
        acc_empty = acc_empty_cached
        print(f"\nAccuracy con prompt vacío (caché): {acc_empty:.1%}")
    else:
        acc_empty = evaluate_accuracy(actor, "", test)
        print(f"\nAccuracy con prompt vacío: {acc_empty:.1%}")

    # Entrenamiento: una pasada sobre train
    n_updates = 0
    for i, sample in enumerate(train):
        system_prompt = policy.read()
        response = actor.generate(system_prompt, sample["question"])
        gt = sample["extracted"]
        extracted = normalize_answer(extract_answer_from_response(response))
        is_correct = extracted == normalize_answer(gt)

        verdict = "CORRECTO" if is_correct else "INCORRECTO"
        comment = "" if is_correct else GENERIC_FEEDBACK_INCORRECT

        cycle = PerceptionCycle(
            system_prompt=system_prompt,
            user_query=sample["question"],
            bot_response=response,
            verdict=verdict,
            comment=comment,
            dwell_seconds=0.0,
            observations=[],
        )

        if verdict == "INCORRECTO":
            gsm8k_judge.set_ground_truth(gt)

        result = loop.process_cycle(cycle)
        if result.gate.should_update:
            n_updates += 1

        if (i + 1) % 10 == 0:
            print(f"  Procesadas {i + 1}/{len(train)} muestras, {n_updates} actualizaciones")

    loop.save_state(str(history_path), str(leaderboard_path))

    # Accuracy final en test (prompt elaborado por el Critic)
    final_prompt = policy.read()
    acc_after = evaluate_accuracy(actor, final_prompt, test)
    print(f"\nAccuracy final (prompt refinado): {acc_after:.1%}")
    print(f"Actualizaciones de policy: {n_updates}")
    print(f"Delta (vacío → refinado): {acc_after - acc_empty:+.1%}")

    # Guardar resultados
    results = {
        "acc_empty": acc_empty,
        "acc_after": acc_after,
        "delta": acc_after - acc_empty,
        "n_updates": n_updates,
        "n_train": len(train),
        "n_test": len(test),
        "seed": SEED,
    }
    (EXPERIMENT_DIR / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    print(f"\nResultados guardados en {EXPERIMENT_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
