"""
Descarga GSM8K, muestra 100 aleatorias con semilla, split 80 train / 20 test.
Guarda en data/gsm8k/.
"""

import json
import random
from pathlib import Path

# Semilla para reproducibilidad
SEED = 42
N_SAMPLES = 100
TRAIN_RATIO = 0.8  # 80 train, 20 test

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GSM8K_DIR = PROJECT_ROOT / "data" / "gsm8k"
GSM8K_DIR.mkdir(parents=True, exist_ok=True)


def extract_answer(text: str) -> str:
    """Extrae la respuesta numérica de GSM8K (formato: ... #### 42)."""
    if "####" in text:
        return text.split("####")[-1].strip()
    return text.strip()


def main() -> None:
    from datasets import load_dataset

    print("Descargando GSM8K...")
    ds = load_dataset("openai/gsm8k", "main")

    # Combinar train para tener pool más grande, luego samplear
    train_data = list(ds["train"])
    random.seed(SEED)
    sampled = random.sample(train_data, min(N_SAMPLES, len(train_data)))

    n_train = int(len(sampled) * TRAIN_RATIO)
    train_split = sampled[:n_train]
    test_split = sampled[n_train:]

    # Formato: {"question": "...", "answer": "..."}
    # answer en GSM8K tiene formato "step1... step2... #### 42"
    train_out = [
        {"question": x["question"], "answer": x["answer"], "extracted": extract_answer(x["answer"])}
        for x in train_split
    ]
    test_out = [
        {"question": x["question"], "answer": x["answer"], "extracted": extract_answer(x["answer"])}
        for x in test_split
    ]

    train_path = GSM8K_DIR / "train.json"
    test_path = GSM8K_DIR / "test.json"
    meta_path = GSM8K_DIR / "meta.json"

    train_path.write_text(json.dumps(train_out, indent=2, ensure_ascii=False), encoding="utf-8")
    test_path.write_text(json.dumps(test_out, indent=2, ensure_ascii=False), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {"seed": SEED, "n_train": len(train_out), "n_test": len(test_out), "train_ratio": TRAIN_RATIO},
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Guardado en {GSM8K_DIR}")
    print(f"  train.json: {len(train_out)} muestras")
    print(f"  test.json:  {len(test_out)} muestras")
    print(f"  meta.json:  seed={SEED}")


if __name__ == "__main__":
    main()
