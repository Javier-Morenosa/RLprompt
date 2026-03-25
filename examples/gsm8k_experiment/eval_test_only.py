"""
Evalúa solo el test: prompt vacío vs prompt refinado (system_prompt.md).
Sin entrenamiento. Útil para verificar accuracy rápidamente.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GSM8K_DIR = PROJECT_ROOT / "data" / "gsm8k"
EXPERIMENT_DIR = PROJECT_ROOT / "data" / "gsm8k_experiment"

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


GROQ_BASE = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"


def extract_answer_from_response(text: str) -> str:
    text = text.strip()
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


def evaluate_accuracy(actor, system_prompt: str, samples: list) -> float:
    correct = 0
    for s in samples:
        response = actor.generate(system_prompt, s["question"])
        extracted = normalize_answer(extract_answer_from_response(response))
        gt = normalize_answer(s["extracted"])
        if extracted == gt:
            correct += 1
    return correct / len(samples) if samples else 0.0


def main() -> None:
    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY no definida.")
        sys.exit(1)

    test = json.loads((GSM8K_DIR / "test.json").read_text(encoding="utf-8"))
    backend = LocalLLMBackend(model=MODEL, base_url=GROQ_BASE, api_key=GROQ_API_KEY)
    actor = Actor(backend=backend, max_tokens=512, temperature=0.1)

    # Prompt vacío
    acc_empty = evaluate_accuracy(actor, "", test)
    print(f"Accuracy con prompt vacío: {acc_empty:.1%} ({int(acc_empty * len(test))}/{len(test)})")

    # Prompt refinado (system_prompt.md)
    refined_path = EXPERIMENT_DIR / "system_prompt.md"
    if not refined_path.exists():
        print(f"ERROR: No existe {refined_path}")
        sys.exit(1)
    refined_prompt = refined_path.read_text(encoding="utf-8").strip()
    acc_refined = evaluate_accuracy(actor, refined_prompt, test)
    print(f"Accuracy con prompt refinado: {acc_refined:.1%} ({int(acc_refined * len(test))}/{len(test)})")

    print(f"\nDelta (vacío → refinado): {acc_refined - acc_empty:+.1%}")


if __name__ == "__main__":
    main()
