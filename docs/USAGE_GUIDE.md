# Guía de uso — Human-Watch en directo

Esta guía te lleva paso a paso desde arrancar el sistema hasta observar en tiempo real cómo el Crítico refina el prompt y la política converge.

---

## Índice

1. [Requisitos previos](#1-requisitos-previos)
2. [Reset del estado (limpieza opcional)](#2-reset-del-estado)
3. [Arrancar el sistema (3 terminales)](#3-arrancar-el-sistema)
4. [Interactuar con el chat](#4-interactuar-con-el-chat)
5. [Dar feedback y cerrar ciclos](#5-dar-feedback-y-cerrar-ciclos)
6. [Leer la salida de los terminales](#6-leer-la-salida-de-los-terminales)
7. [Dashboard — ver el estado global](#7-dashboard)
8. [Observar la convergencia](#8-observar-la-convergencia)
9. [Ejecutar el test de integración](#9-test-de-integracion)
10. [Escenarios de prueba sugeridos](#10-escenarios-de-prueba)
11. [Solución de problemas](#11-solucion-de-problemas)

---

## 1. Requisitos previos

Si es la primera vez, instala el paquete y dependencias Human-Watch:

```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt
pip install -e ".[human-watch]"
playwright install chromium
```

**Backend LLM:** Human-Watch usa **Groq** (Llama 3.1 8B) por defecto. Necesitas `GROQ_API_KEY` en el entorno o en `.env` en la raíz del proyecto. Alternativamente, puedes configurar Ollama para los ejemplos de la librería.

Verifica que todo esté listo:

```bash
# Dependencias Python
python -c "import fastapi, uvicorn, playwright; print('OK')"

# API key (Groq)
# Define GROQ_API_KEY en .env o como variable de entorno
```

Si usas Ollama para los ejemplos (`two_stage_example`, `validation_loop_example`):
```bash
ollama list   # deben aparecer gemma3:1b y gemma3:4b
ollama serve  # si no está corriendo
```

---

## 2. Reset del estado

Si quieres empezar desde cero (sin historial de sesiones anteriores), ejecuta esto **antes** de arrancar:

```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt
python -m demos.human_watch.reset_to_state_zero
# o: rlprompt-reset
```

Restaura automáticamente critic_memory, interactions.md, system_prompt.md (política mínima), reward_history.json y population.json.

Si prefieres **continuar** desde donde lo dejaste, no hagas nada — el estado persiste entre sesiones.

---

## 3. Arrancar el sistema

**Opción A — Un solo comando** (recomendado):

```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt
python -m demos.human_watch.run_backend
# o tras pip install:
rlprompt-backend
```

Inicia servidor + monitor en uno. Cierra el navegador para detener todo.

**Opción B — Terminales separadas**

### Terminal 1 — Servidor de chat (Actor)

```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt
python -m demos.human_watch.run_server
# o: rlprompt-serve
```

Salida esperada:
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
...
```

El servidor estará disponible en `http://localhost:8000`.

### Terminal 2 — Monitor de percepción (Human-Watch)

```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt
python -m demos.human_watch.monitor
```

Salida esperada:
```
[Monitor] Sesión A3F2B1C4 | URL: http://localhost:8000
[Monitor] Log: C:\...\data\interactions.md

[Monitor] Navegador abierto. Cierra la ventana para terminar.
```

Se abrirá una ventana de **Chromium** con el chat. **No la cierres** — es donde interactúas.

### Terminal 3 — Log del evaluador (opcional pero recomendado)

```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt

# Sigue el log del evaluador en tiempo real (PowerShell)
Get-Content data/evaluator.log -Wait -Tail 20
```

O si prefieres ver el archivo directamente después de cada ciclo:
```bash
# En cmd clásico
type data\evaluator.log
```

---

## 4. Interactuar con el chat

En la ventana de Chromium verás:

```
┌──────────────────────────────┐  ┌─────────────────────────┐
│  Chatbot de Negocio          │  │ Información de la        │
│                              │  │ Empresa (RAG)            │
│  [ventana de chat]           │  │                          │
│                              │  │ Planes y Precios         │
│  [input]  [Enviar]           │  │  - Plan Básico: 29€/mes  │
│                              │  │  - Plan Pro: 79€/mes     │
│  ¿Fue útil la respuesta?     │  │                          │
│  [✓ Correcto] [✗ Incorrecto] │  │ Política de Devoluciones │
│  [comentario...] [Enviar]    │  │ ...                      │
└──────────────────────────────┘  └─────────────────────────┘
```

**Preguntas de prueba sugeridas** (escríbelas en el input y pulsa Enviar):

| Pregunta | Respuesta esperada del bot |
|---|---|
| `¿el plan básico incluye IVA?` | Sí, incluye IVA del 21% |
| `¿cuántos usuarios tiene el plan pro?` | 25 usuarios |
| `¿cuál es la política de devoluciones?` | 30 días, sin condiciones |
| `¿el soporte técnico 24/7 está en todos los planes?` | Solo en Plan Pro |
| `¿cuánto cuesta el plan enterprise?` | No existe, no está en el catálogo |

Mientras el bot responde, la **Terminal 2** mostrará:
```
  [Ciclo 1 abierto] Consulta: "¿el plan básico incluye IVA?"
  [Ciclo 1] Respuesta del bot recibida (87 chars)
```

---

## 5. Dar feedback y cerrar ciclos

Después de leer la respuesta del bot, tienes que completar el ciclo de percepción:

### Fase de observación (opcional pero recomendada)

Antes de dar feedback, **mueve el ratón sobre el panel RAG** de la derecha. Mantén el cursor quieto más de 1 segundo sobre alguna sección. En la Terminal 2 aparecerá:
```
  -> [DWELL] RAG_PANEL/Planes y Precios @ (712, 234)
```

Esto simula que el humano consultó la documentación para verificar la respuesta.

También puedes **seleccionar texto** en el RAG (arrastra con el ratón) para marcar la información relevante:
```
  -> [SELECT] "Plan Básico: 29 €/mes — 5 usuarios, 10 GB..."
```

Y pulsar el botón **"👁 Revisando documentación"** para registrar una revisión activa:
```
  -> [REVIEW_RAG]
```

### Dar el veredicto — ACC Signal

**Si la respuesta fue correcta:**
1. Pulsa `✓ Correcto (Sí)`
2. (Opcional) Escribe un comentario positivo → pulsa `Enviar comentario`

**Si la respuesta fue incorrecta:**
1. Pulsa `✗ Incorrecto (No)`
2. Escribe una corrección concreta, por ejemplo:
   - `"No mencionó que incluye IVA del 21%"`
   - `"Debería especificar el límite de 25 usuarios"`
   - `"No aclaró que el soporte 24/7 es solo para Plan Pro"`
3. Pulsa `Enviar comentario`

Al enviar el comentario, el ciclo se **cierra** y la Terminal 2 mostrará:
```
  [Ciclo 1] Veredicto: INCORRECTO
  [Ciclo 1] Correccion: "No mencionó que incluye IVA del 21%"
  [Ciclo 1 cerrado y escrito]
  [Evaluator] Lanzado en segundo plano -> data/evaluator.log
```

---

## 6. Leer la salida de los terminales

### Terminal 2 (monitor) — lo que ves durante un ciclo

```
  [Ciclo 1 abierto] Consulta: "¿el plan básico incluye IVA?"
  -> [DWELL] RAG_PANEL/Planes y Precios @ (712, 234)
  -> [SELECT] "Plan Básico: 29 €/mes..."
  -> [REVIEW_RAG]
  [Ciclo 1] Veredicto: INCORRECTO
  [Ciclo 1] Correccion: "No mencionó que incluye IVA del 21%"
  [Ciclo 1 cerrado y escrito]
  [Evaluator] Lanzado en segundo plano -> data/evaluator.log
```

### Terminal 3 (data/evaluator.log) — lo que hace el Crítico

Unos segundos después del ciclo, el evaluador termina y `data/evaluator.log` muestra algo así:

```
--- 2026-02-19T15:32:10 ---
[Evaluator] Critic model: gemma3:4b
[Evaluator] Running Critic...
[Evaluator] critic_score=0.42  R=-0.0730  gate=forced  change=18.3%  stable_streak=0/5  converged=False
[Evaluator] Policy updated -> v6 (forced)
```

**Qué significa cada campo:**

| Campo | Significado |
|---|---|
| `critic_score` | Puntuación que el Crítico da al prompt actual (0.0–1.0) |
| `R` | Recompensa total: `λ_fb·H + λ_c·C − λ_ch·change_ratio` |
| `gate` | Por qué se actualizó: `forced` (INCORRECTO+comentario) o `degradation` |
| `change` | % de palabras que el Crítico cambió en el prompt |
| `stable_streak` | Ciclos consecutivos estables / umbral de convergencia |
| `converged` | Si es `True`, el evaluador ya no se lanza en ciclos futuros |

Si la política **no se actualizó** (respuesta fue correcta, R no degrada):
```
[Evaluator] critic_score=0.91  R=+0.8320  gate=stable  change=1.2%  stable_streak=2/5  converged=False
[Evaluator] Policy stable — no update.
```

#### Modo verbose — ver todo el flujo

Para depurar o inspeccionar el pipeline completo, activa el modo verbose:

**Opción A — variable de entorno** (el monitor pasa `--verbose` al evaluador):
```powershell
# Windows
$env:EVALUATOR_VERBOSE = "1"
python -m demos.human_watch.monitor
```
```bash
# Linux / macOS
export EVALUATOR_VERBOSE=1
python -m demos.human_watch.monitor
```

**Opción B — ejecución manual:**
```bash
python -m demos.human_watch.evaluator --verbose
```

En `data/evaluator.log` verás:

1. **Ciclo enviado al Critic** — `system_prompt`, `verdict`, `comment`, `observations`
2. **Prompt al LLM** — texto que recibe el Critic (preview) y respuesta raw
3. **Reward calculado** — H, C, change_ratio, fórmula y R_total
4. **Gate** — `should_update`, `reason`, R_avg, `stable_streak`
5. **Resumen** — igual que el output normal

### Estructura JSON de la política

El `data/system_prompt.md` usa formato JSON con tres campos:

```json
{
  "role": "Descripción del rol del bot",
  "hard_rules": [
    "Regla obligatoria 1",
    "Regla obligatoria 2"
  ],
  "soft_guidelines": [
    "Directriz recomendada"
  ]
}
```

- **role**: identidad y contexto base del Actor
- **hard_rules**: reglas que el Actor debe cumplir siempre
- **soft_guidelines**: recomendaciones (opcional)

El Critic decide cómo tratar el comentario del humano:
- **direct_rule**: lo incorpora como nueva `hard_rule` (instrucción explícita)
- **refinement**: sintetiza/refina en base al historial y la memoria

### Ver el prompt actualizado

```bash
type data/system_prompt.md
```

Después de una actualización forzada verás el prompt refinado por el Crítico (en JSON).

> **Hot-reload:** El servidor lee `data/system_prompt.md` en cada petición de chat. Cuando el Crítico actualiza el archivo, la siguiente respuesta del bot ya usa el prompt nuevo — no hace falta reiniciar.

Para verificar que el Actor relee el prompt en cada consulta, arranca el servidor con verbose:

**CMD:**
```cmd
set SERVER_VERBOSE=1
python -m demos.human_watch.run_server
```

**PowerShell:**
```powershell
$env:SERVER_VERBOSE = "1"
python -m demos.human_watch.run_server
```
En la terminal verás en cada chat algo como:
`[Actor] Hot-reload: system_prompt.md leido | 345 chars | mtime=... | hash=...` Por ejemplo:
```
Eres un asistente de negocio amable y profesional. Responde ÚNICAMENTE con información
del catálogo de la empresa. Si el usuario pregunta por precios, indica siempre si el
precio incluye IVA (21%) y el número de usuarios incluidos. Si el usuario pregunta algo
que no está en el catálogo, indícalo claramente. Responde en español y de forma concisa.
```

---

## 7. Dashboard

Abre en el navegador: `http://localhost:8000/dashboard`

Muestra:

```
┌─────────────┐ ┌─────────────┐ ┌──────────────────────┐
│ Versión: 6  │ │ Accuracy    │ │ Individuos (gen 6)   │
│ del Prompt  │ │ 60% (3/5)   │ │ 6                    │
└─────────────┘ └─────────────┘ └──────────────────────┘

┌─ System Prompt Activo ──────────────────────────────────┐
│ Eres un asistente de negocio amable...                  │
└─────────────────────────────────────────────────────────┘

┌─ Historial de Recompensas ──────────────────────────────┐
│ 2026-02-19 15:28  ████████████████░░░░  +0.4821  CORRECTO   │
│ 2026-02-19 15:30  ████░░░░░░░░░░░░░░░░  -0.0730  INCORRECTO │
│ ...                                                     │
└─────────────────────────────────────────────────────────┘

┌─ Top-3 Fitness ──────┐  ┌─ Últimas 5 Correcciones ────┐
│ #1: +0.4821          │  │ • No mencionó IVA del 21%   │
│ #2: +0.2340          │  │ • Faltó el límite usuarios  │
│ #3: -0.0730          │  │ ...                          │
└──────────────────────┘  └─────────────────────────────┘
```

Recarga la página tras cada ciclo para ver la actualización.

---

## 8. Observar la convergencia

La convergencia se activa cuando durante **5 ciclos consecutivos** ocurre todo lo siguiente:
- El humano dice `CORRECTO`
- El Crítico propone cambios mínimos (`< 5%` de las palabras)

### Cómo provocar la convergencia

Repite este patrón 5 veces seguidas:

1. Pregunta sobre el catálogo (algo que el bot responde bien)
2. Marca `✓ Correcto`
3. Envía el comentario en blanco (o no lo envíes — el ciclo se cierra igualmente al dar veredicto positivo sin comentario)

> **Nota:** un ciclo CORRECTO sin comentario se cierra en la siguiente consulta, cuando el monitor abre un nuevo ciclo.

### Lo que verás cuando converja

En `data/evaluator.log`:
```
[Evaluator] critic_score=0.94  R=+0.8910  gate=stable  change=0.8%  stable_streak=5/5  converged=True
[Evaluator] Policy stable — no update.
[Evaluator] *** CONVERGENCIA ALCANZADA — evaluaciones suspendidas ***
```

En el siguiente ciclo, desde la Terminal 2:
```
  [Ciclo 6 cerrado y escrito]
  [Evaluator] Policy converged (5 stable cycles) — skipped.
```

El evaluador ya no se lanza. El sistema se ha estabilizado.

### Resetear la convergencia

Para que el sistema vuelva a aprender después de converger, basta con dar un feedback `INCORRECTO` con comentario. El gate lo detectará como `forced`, actualizará la política, y `bump_version()` reseteará el contador de estabilidad a 0.

---

## 9. Test de integración

Verifica que el pipeline de percepción (monitor → data/interactions.md) funciona correctamente, **sin necesidad de interacción manual**.

### Requisitos

El servidor debe estar corriendo (Terminal 1):
```bash
python -m demos.human_watch.run_server
```

### Ejecutar el test

En una terminal nueva:
```bash
cd C:\Users\Usuario\Downloads\Openclaw\RLprompt
python -m demos.human_watch.tests.test_monitor
```

El test lanza un Chromium en modo headless, simula toda la secuencia (envío de mensaje → dwell → selección → feedback INCORRECTO + comentario) y verifica 14 puntos del log:

```
[test] Fetching system prompt...
[test] OK — 187 chars

[test] Opening http://localhost:8000 ...
[test] Page loaded

[test] (2) Sending user query...
[test] (3) Dwelling on RAG section...
[test] (3) Selecting text in RAG...
[test] (3) Clicking Review RAG button...
[test] (4) Clicking Incorrecto (No)...
[test] (4) Submitting correction comment...

  [PASS] Session header
  [PASS] Ciclo block written
  [PASS] (1) Predictive model
  [PASS] (2) User query
  [PASS] (2) Bot response
  [PASS] (3) Observation (DWELL)
  [PASS] (3) Observation (SELECT)
  [PASS] (3) Observation (REVIEW)
  [PASS] (4) ACC verdict
  [PASS] (4) Correction comment
  [PASS] [RAW] telemetry section
  [PASS] [RAW] click logged
  [PASS] [RAW] cursor logged
  [PASS] Session footer

[test] ALL CHECKS PASSED
```

> **Importante:** el test prueba el pipeline monitor → `data/interactions.md`. No invoca al Crítico ni a `evaluator.py` — eso es intencional. El test aísla la capa de percepción del RL loop.

---

## 10. Escenarios de prueba

### Escenario A — Refinamiento forzado (INCORRECTO + comentario)

**Objetivo:** ver cómo el Crítico edita el prompt en una sola iteración.

1. Pregunta: `¿el plan básico cuántos usuarios incluye?`
2. Si el bot no menciona "5 usuarios" → marca `INCORRECTO` + escribe `"Debe indicar que el Plan Básico incluye 5 usuarios"`
3. Observa `data/evaluator.log` → verás `gate=forced` y el prompt actualizado
4. Repite la misma pregunta → el bot debería ahora incluir ese dato

---

### Escenario B — Ciclo correcto (sin actualización)

**Objetivo:** confirmar que la política no cambia cuando el bot responde bien.

1. Pregunta: `¿cuánto cuesta el plan pro?`
2. Si el bot responde correctamente → marca `✓ Correcto`
3. En `data/evaluator.log` verás `gate=stable` → "Policy stable — no update."
4. `data/system_prompt.md` no cambia

---

### Escenario C — Convergencia acelerada

**Objetivo:** llegar a `converged=True` en 5 ciclos.

Repite 5 veces:
1. Pregunta algo que el bot responde bien
2. Marca `✓ Correcto`
3. No añadas comentario (o añade uno vacío)
4. Envía la siguiente pregunta

En el quinto ciclo, `data/evaluator.log` mostrará `converged=True` y el monitor empezará a saltarse el evaluador.

---

### Escenario D — Recuperación tras convergencia

**Objetivo:** demostrar que el sistema retoma el aprendizaje si el humano detecta un error nuevo.

1. Lleva el sistema a convergencia (Escenario C)
2. Pregunta algo que el bot responde mal
3. Marca `INCORRECTO` + escribe la corrección
4. El evaluador se lanza de nuevo (`gate=forced`)
5. La convergencia se resetea a `stable_streak=0`

---

## 11. Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| El navegador no abre | Playwright no tiene Chromium | `playwright install chromium` |
| `[Evaluator] GROQ_API_KEY no definida` | Falta la API key de Groq | Crea `.env` con `GROQ_API_KEY=...` o exporta la variable |
| El evaluador no produce output en el log | Ollama/Groq tardó mucho o falló | Comprueba la API key; revisa `data/evaluator.log` completo |
| El prompt no cambia tras INCORRECTO | El comentario estaba vacío | El gate `forced` requiere comentario no vacío; escribe una corrección concreta |
| `data/interactions.md` crece demasiado | Muchas sesiones acumuladas | Se archiva automáticamente en `data/logs/` al superar 200 KB; o bórralo manualmente |
| El test falla en `(2) Bot response` | Gemma tardó más de 6 s | Aumenta el timeout en `test_monitor.py` línea 67: `await page.wait_for_timeout(10000)` |
| `data/reward_history.json` — KeyError | Fichero de una versión antigua | Ejecuta `rlprompt-reset` o bórralo y deja que se regenere |
| El servidor no carga en puerto 8000 | Puerto ocupado | `python -m demos.human_watch.run_server 9000` y `TARGET_URL=http://localhost:9000` para el monitor |

---

## Flujo completo resumido

```
[Terminal 1]                    [Chromium]               [Terminal 2]          [evaluator.log]
run_server / run_backend  →   chat UI abierto
                         escribe pregunta    →   [Ciclo N abierto]
                         bot responde        →   Respuesta recibida
                         mueve ratón RAG     →   [DWELL] / [SELECT]
                         pulsa Incorrecto    →   Veredicto: INCORRECTO
                         escribe corrección  →   [Ciclo N cerrado]
                         pulsa Enviar        →   [Evaluator] Lanzado
                                                                    →  Critic llama a LLM (Groq/Ollama)
                                                                    →  R=-0.07  gate=forced
                                                                    →  Policy updated -> vN+1
                         siguiente pregunta  →   [Ciclo N+1 abierto]
```
