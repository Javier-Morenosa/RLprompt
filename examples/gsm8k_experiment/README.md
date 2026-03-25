# Experimento GSM8K

Entrena un system prompt en 80 muestras de GSM8K y evalúa en 20 (split con semilla 42).

## Requisitos

```bash
pip install -e ".[gsm8k]"
```

Variable de entorno `GROQ_API_KEY` (o `.env` en la raíz del proyecto).

## Uso

```bash
# Descargar datos (100 muestras, 80 train / 20 test)
python examples/gsm8k_experiment/download_gsm8k.py

# Ejecutar entrenamiento + evaluación
python examples/gsm8k_experiment/run_gsm8k_train.py
```

El script `run_gsm8k_train.py` descarga automáticamente si `data/gsm8k/train.json` no existe.

## Salidas

- `data/gsm8k/` — train.json, test.json, meta.json
- `data/gsm8k_experiment/` — system_prompt.md, reward_history.json, results.json

## Diseño

- **Feedback genérico**: El Critic no recibe la respuesta correcta; solo "El cálculo fue incorrecto..."
- **Judge determinístico**: GSM8KJudge compara la respuesta extraída con ground truth (solo para validación en el virtual loop)
- **Sin data leakage**: Train y test son splits distintos; el feedback al Critic no incluye la respuesta correcta
