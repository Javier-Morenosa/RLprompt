"""
Filtrado en dos pasadas: selecciona muestras GSM8K que el modelo con prompt vacío falla.

1. Carga un pool grande (300 muestras)
2. Evalúa cada una con prompt vacío
3. Conserva solo las que el modelo falla → guarda en hard_pool.json
4. Split 80 train / 20 test desde ese subconjunto

Si hard_pool.json existe (mismo seed/pool_size), se puede usar --use-cached para
saltar la evaluación y reutilizar las muestras difíciles.

Objetivo: crear un dataset donde el LLM crudo tenga baja accuracy, dejando margen
para que el Critic demuestre mejora al refinar el prompt.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GSM8K_DIR = PROJECT_ROOT / "data" / "gsm8k"
GSM8K_DIR.mkdir(parents=True, exist_ok=True)

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

from prompt_rl import Actor
from prompt_rl.llm.local_backend import LocalLLMBackend

# Mismo modelo y temperatura que run_gsm8k_train
GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"
TEMPERATURE = 0.1
SEED = 42
POOL_SIZE = 300
MIN_HARD_SAMPLES = 100
TRAIN_RATIO = 0.8


def extract_answer(text: str) -> str:
    if "####" in text:
        return text.split("####")[-1].strip()
    numbers = re.findall(r"[-+]?\d*\.?\d+", text)
    return numbers[-1] if numbers else ""


def normalize_answer(s: str) -> str:
    s = str(s).strip()
    try:
        return str(int(float(s)))
    except ValueError:
        return s


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Curar GSM8K: seleccionar muestras difíciles para prompt vacío")
    parser.add_argument(
        "--use-cached",
        action="store_true",
        help="Reutilizar hard_pool.json sin re-evaluar (salta las llamadas al LLM)",
    )
    args = parser.parse_args()

    hard_pool_path = GSM8K_DIR / "hard_pool.json"
    meta_path = GSM8K_DIR / "meta.json"

    hard_samples = None
    acc_empty_on_pool = None

    # Si --use-cached y existe hard_pool con mismo config, cargar y split
    if args.use_cached and hard_pool_path.exists():
        cached = json.loads(hard_pool_path.read_text(encoding="utf-8"))
        if cached.get("seed") == SEED and cached.get("pool_size") == POOL_SIZE:
            hard_samples = cached["hard_samples"]
            acc_empty_on_pool = cached.get("acc_empty_on_pool")
            if acc_empty_on_pool is None:
                acc_empty_on_pool = 1.0 - len(hard_samples) / cached.get("pool_size", POOL_SIZE)
            print(f"Usando hard_pool.json en caché: {len(hard_samples)} muestras difíciles", flush=True)
        else:
            args.use_cached = False

    if not args.use_cached:
        if not os.environ.get("GROQ_API_KEY"):
            _load_dotenv()
        if not os.environ.get("GROQ_API_KEY"):
            print("ERROR: GROQ_API_KEY no definida. Define la variable o crea .env", flush=True)
            sys.exit(1)

        from datasets import load_dataset
        import random

        print("Descargando GSM8K...", flush=True)
        ds = load_dataset("openai/gsm8k", "main")
        train_data = list(ds["train"])

        random.seed(SEED)
        pool = random.sample(train_data, min(POOL_SIZE, len(train_data)))

        samples = [
            {"question": x["question"], "answer": x["answer"], "extracted": extract_answer(x["answer"])}
            for x in pool
        ]

        print(f"\nPasada 1: evaluando {len(samples)} muestras con prompt vacío...", flush=True)
        backend = LocalLLMBackend(
            model=MODEL,
            base_url=GROQ_BASE,
            api_key=os.environ.get("GROQ_API_KEY", ""),
        )
        actor = Actor(backend=backend, max_tokens=512, temperature=TEMPERATURE)

        hard_samples = []
        for i, s in enumerate(samples):
            response = actor.generate("", s["question"])
            extracted = normalize_answer(extract_answer(response))
            gt = normalize_answer(s["extracted"])
            if extracted != gt:
                hard_samples.append(s)
            if (i + 1) % 10 == 0:
                print(f"  Evaluadas {i + 1}/{len(samples)}, hard: {len(hard_samples)}", flush=True)

        acc_empty_on_pool = 1.0 - len(hard_samples) / len(samples)
        print(f"\nResultado pasada 1: {len(hard_samples)} muestras difíciles de {len(samples)}", flush=True)
        print(f"  Accuracy con prompt vacío en pool: {acc_empty_on_pool:.1%}", flush=True)

        # Guardar hard_pool para reutilizar sin re-evaluar
        hard_pool_path.write_text(
            json.dumps(
                {
                    "seed": SEED,
                    "pool_size": POOL_SIZE,
                    "hard_samples": hard_samples,
                    "acc_empty_on_pool": acc_empty_on_pool,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"  Guardado hard_pool.json ({len(hard_samples)} muestras)", flush=True)

    if len(hard_samples) < MIN_HARD_SAMPLES:
        print(
            f"\nADVERTENCIA: Solo {len(hard_samples)} muestras difíciles "
            f"(mínimo {MIN_HARD_SAMPLES}). Aumenta POOL_SIZE o prueba otro SEED.",
            flush=True,
        )
        if len(hard_samples) < 20:
            print("No hay suficientes muestras para train/test. Abortando.", flush=True)
            sys.exit(1)

    # Split 80/20 desde el subconjunto difícil
    import random
    random.seed(SEED)
    random.shuffle(hard_samples)
    n_train = int(len(hard_samples) * TRAIN_RATIO)
    train_out = hard_samples[:n_train]
    test_out = hard_samples[n_train:]

    train_path = GSM8K_DIR / "train.json"
    test_path = GSM8K_DIR / "test.json"
    meta_path = GSM8K_DIR / "meta.json"

    train_path.write_text(
        json.dumps(train_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    test_path.write_text(
        json.dumps(test_out, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    # acc_empty en test = 0 (test es subconjunto de hard = todas fallaron con prompt vacío)
    acc_empty_test = 0.0

    meta_path.write_text(
        json.dumps(
            {
                "seed": SEED,
                "pool_size": POOL_SIZE,
                "n_hard": len(hard_samples),
                "acc_empty_on_pool": acc_empty_on_pool,
                "acc_empty": acc_empty_test,
                "n_train": len(train_out),
                "n_test": len(test_out),
                "train_ratio": TRAIN_RATIO,
                "curated": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nGuardado en {GSM8K_DIR}", flush=True)
    print(f"  train.json: {len(train_out)} muestras (difíciles)", flush=True)
    print(f"  test.json:  {len(test_out)} muestras (difíciles)", flush=True)
    print(f"  meta.json:  curated=True, acc_empty={acc_empty_test:.1%}", flush=True)


if __name__ == "__main__":
    main()
