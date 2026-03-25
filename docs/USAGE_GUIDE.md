# RLPrompt — Usage Guide

RLPrompt optimizes a system prompt through reinforcement learning.
It supports two operating modes that share the same internal engine:

| Mode | Signal source | Judge | Use case |
|---|---|---|---|
| **Online** | Live human feedback | LLM-based (subjective) | Chatbot, customer service, conversational AI |
| **Offline** | Q&A dataset with ground truth | Deterministic (no extra LLM) | Benchmarks, math, MCQ, any task with known answers |

---

## Setup

### Requirements

- Python 3.9+
- An OpenAI-compatible LLM endpoint (Groq, Ollama, LM Studio, etc.)
- A `GROQ_API_KEY` (or equivalent) in your environment or a `.env` file

### Install

```bash
git clone https://github.com/Javier-Morenosa/RLprompt.git
cd RLprompt

# Library only
pip install -e .

# Library + Human-Watch demo (FastAPI + Playwright)
pip install -e ".[human-watch]"
playwright install chromium

# Library + dataset experiments (HuggingFace datasets)
pip install -e ".[gsm8k]"

# Everything
pip install -e ".[all]"
```

### Environment variables

Create a `.env` file in the project root (already gitignored):

```env
GROQ_API_KEY=gsk_...

# Optional overrides (defaults shown)
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant
CRITIC_MODEL=llama-3.1-8b-instant
ACTOR_MODEL=llama-3.1-8b-instant
```

Any OpenAI-compatible endpoint works. For local models via Ollama:

```env
GROQ_BASE_URL=http://localhost:11434/v1
GROQ_API_KEY=ollama
GROQ_MODEL=gemma3:4b
```

---

## Mode 1 — Online (Chatbot with Human Feedback)

### How it works

```
Human types a message
        │
        ▼
FastAPI server generates a response  (Actor)
        │
        ▼
Playwright monitor records the full interaction
        │
Human labels: CORRECTO / INCORRECTO + optional correction text
        │
        ▼
PerceptionCycle
(system_prompt, user_query, bot_response, verdict, comment, cursor signals)
        │
        ▼
TwoStageCritic
  ├─ Backward:   "what went wrong and why?"  →  actionable feedback
  └─ Optimizer:  "write a better prompt"     →  proposed system_prompt
        │
        ▼
CriticValidationLoop
  Re-asks the Actor with the proposed prompt
  └─ LLMValidationJudge: "did this fix the problem?"
     ├─ YES → accept proposal
     └─ NO  → send feedback to Critic again (up to 3 iterations)
        │
        ▼
HybridReward:  R = λ_fb·H + λ_c·C − λ_ch·word_change_ratio
        │
        ▼
UpdateGate — should the policy be updated?
  ├─ YES → write system_prompt.md  (hot-reloaded by the server on next request)
  └─ NO  → keep current policy
        │
        ▼
5 consecutive stable CORRECTO cycles → converged, evaluator stops
```

### Quick start

```bash
# Single command (recommended)
python -m demos.human_watch.run_backend
# or after pip install:
rlprompt-backend
```

Or in two separate terminals:

```bash
# Terminal 1 — chat server
python -m demos.human_watch.run_server

# Terminal 2 — Playwright monitor (opens browser automatically)
python -m demos.human_watch.monitor
```

Open `http://localhost:8000` in the Chromium window and start chatting.
Every time you label a response, the monitor writes a `PerceptionCycle` to
`data/interactions.md` and spawns the evaluator subprocess.

Open `http://localhost:8000/dashboard` to see the live reward history,
active system prompt, and leaderboard.

### State files

| File | Description |
|---|---|
| `data/system_prompt.md` | Active policy — hot-reloaded on every chat request |
| `data/interactions.md` | Append-only log of all perception cycles |
| `data/reward_history.json` | Rolling reward window + convergence state |
| `data/population.json` | Top-N prompt candidates ranked by fitness |
| `data/critic_memory.md` | Human-readable Critic journal (shown in dashboard) |
| `evaluator.log` | Evaluator subprocess output |

### Reading the evaluator output

After each labeled cycle, `data/evaluator.log` shows:

```
[Evaluator] critic_score=0.42  R=-0.0730  gate=forced  change=18.3%
[Evaluator] Policy updated -> v6 (forced)
```

| Field | Meaning |
|---|---|
| `critic_score` | Critic's self-assessed score for the current prompt (0–1) |
| `R` | Total reward: `λ_fb·H + λ_c·C − λ_ch·change_ratio` |
| `gate` | Why the policy was updated: `forced` (INCORRECTO+comment) or `degradation` |
| `change` | % of words the Critic changed in the proposed prompt |

For a stable cycle (no update):
```
[Evaluator] critic_score=0.91  R=+0.8320  gate=stable  change=1.2%
[Evaluator] Policy stable — no update.
```

Enable verbose mode to see the full Critic pipeline:

```bash
# Windows PowerShell
$env:EVALUATOR_VERBOSE = "1"

# Linux / macOS
export EVALUATOR_VERBOSE=1
```

### Convergence

After **5 consecutive** stable CORRECTO cycles with less than 5% word change,
the system converges and the evaluator stops running automatically:

```
[Evaluator] *** CONVERGENCIA ALCANZADA — evaluaciones suspendidas ***
```

To resume learning after convergence, simply send one INCORRECTO verdict
with a correction comment — the gate fires as `forced` and resets the
stability counter to 0.

### Reset

```bash
python -m demos.human_watch.reset_to_state_zero
# or
rlprompt-reset
```

### Use the library directly (online mode)

```python
from prompt_rl import (
    PerceptionCycle, ActivePolicy,
    TwoStageCritic, CriticValidationLoop,
    Actor, LLMValidationJudge,
    OnlineCriticLoop, RewardHistory, Leaderboard,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

backend = LocalLLMBackend(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
)

critic = CriticValidationLoop(
    critic=TwoStageCritic(backend=backend),
    actor=Actor(backend=backend),
    judge=LLMValidationJudge(backend=backend),
    max_iterations=3,
    skip_validation_if_correct=True,
)

loop = OnlineCriticLoop(
    critic=critic,
    policy=ActivePolicy(path="system_prompt.md"),
    history=RewardHistory.from_file("reward_history.json"),
    leaderboard=Leaderboard.from_file("population.json"),
)

# Process one human interaction
cycle = PerceptionCycle(
    system_prompt="You are a helpful assistant.",
    user_query="What is the refund policy?",
    bot_response="We offer a 30-day money-back guarantee.",
    verdict="INCORRECTO",
    comment="The policy is 14 days, not 30.",
)

result = loop.process_cycle(cycle)
print(f"R={result.R_total:+.4f}  updated={result.gate.should_update}")
```

---

## Mode 2 — Offline (Dataset with Ground Truth)

### How it works

```
Load dataset  →  DatasetSplit (train / test)
        │
        ▼
For each sample in train:

  Actor generates response with current system_prompt
        │
        ▼
  ExactMatchJudge.is_correct(response, sample)
  │
  ├─ CORRECTO  →  fast path (no Critic call, reward still recorded)
  │
  └─ INCORRECTO
          │
          ▼
          ExactMatchJudge.feedback(response, sample)
          → "El modelo respondio '5' pero la respuesta correcta es '7'."
          │
          │  This comment is the ground truth signal — equivalent to
          │  the human correction comment in online mode.
          │
          ▼
          TwoStageCritic receives verdict + correction comment
            ├─ Backward:   "what rule is the model missing?"
            └─ Optimizer:  "write a better system_prompt"
          │
          ▼
          CriticValidationLoop re-asks Actor with proposed prompt
            ExactMatchJudge.judge() — deterministic check (no extra LLM call)
            ├─ FIXED   → accept proposal
            └─ STILL WRONG → send feedback again (up to 3 iterations)
          │
          ▼
          UpdateGate — write new system_prompt if gate fires
        │
        ▼
Evaluate on test set  →  DatasetResult (acc_before, acc_after, delta)
```

The ground truth answer plays the same role as the human correction comment
in online mode — it tells the Critic exactly what was wrong and what the
correct answer is, so the Backward stage can generate specific, actionable
feedback for the Optimizer.

### Quick start — GSM8K experiment

```bash
# Step 1 — download 100 GSM8K samples (80 train / 20 test)
python examples/gsm8k_experiment/download_gsm8k.py

# Step 2 — train and evaluate
python examples/gsm8k_experiment/run_gsm8k_train.py

# Optional: evaluate on test every 20 training steps
python examples/gsm8k_experiment/run_gsm8k_train.py --eval-every 20

# Optional: curate a harder subset (only samples where the empty prompt fails)
python examples/gsm8k_experiment/run_gsm8k_train.py --curate
```

Results are saved to `data/gsm8k_experiment/results.json`:

```json
{
  "acc_before": 0.30,
  "acc_after":  0.55,
  "delta":      0.25,
  "n_updates":  7,
  "n_train":    80,
  "n_test":     20
}
```

### Use the library directly (offline mode)

```python
import json
from prompt_rl import (
    ActivePolicy, Actor,
    TwoStageCritic, CriticValidationLoop,
    DatasetLoop, DatasetSplit, ExactMatchJudge,
    RewardHistory, Leaderboard,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

# 1. Backend
backend = LocalLLMBackend(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
)
actor = Actor(backend=backend, max_tokens=512, temperature=0.1)

# 2. Load your dataset
train_raw = json.loads(open("data/train.json", encoding="utf-8").read())
test_raw  = json.loads(open("data/test.json",  encoding="utf-8").read())

split = DatasetSplit.from_dicts(
    train=train_raw,
    test=test_raw,
    question_key="question",    # adjust to your field names
    answer_key="answer",
    extracted_key="extracted",  # normalized answer for comparison
)

# Or split from a single list:
# split = DatasetSplit.from_list(all_samples, train_ratio=0.8, seed=42)

# 3. Judge — deterministic, sends ground truth to the Critic as feedback
judge = ExactMatchJudge(
    extract_pattern=r"####\s*(.+)$",  # extract answer after ####
    include_ground_truth=True,        # "correct answer is X" in Critic feedback
)

# 4. Critic with iterative validation using the same deterministic judge
critic = CriticValidationLoop(
    critic=TwoStageCritic(backend=backend),
    actor=actor,
    judge=judge,          # ExactMatchJudge also implements ValidationJudge
    max_iterations=3,
    skip_validation_if_correct=True,
)

# 5. Policy — start from an empty prompt
policy = ActivePolicy(path="data/system_prompt.md", backup_dir="data/prompts")
policy.write("", 0)

# 6. DatasetLoop
loop = DatasetLoop(
    critic=critic,
    policy=policy,
    actor=actor,
    judge=judge,
    history=RewardHistory(),
    leaderboard=Leaderboard(),
    verbose=True,
)

# 7. Train and evaluate
result = loop.train(
    train_samples=split.train,
    test_samples=split.test,
    eval_every=20,    # 0 = evaluate only at start and end
    max_epochs=1,
)

loop.save_state("data/reward_history.json", "data/population.json")

print(f"Accuracy: {result.acc_before:.1%} -> {result.acc_after:.1%}  "
      f"(delta={result.delta:+.1%}, updates={result.n_updates})")
print(f"\nFinal prompt:\n{result.final_prompt}")
```

### Bring your own dataset

Any dataset with a question and a known correct answer works.
Adapt the judge to your answer format:

| Task type | Judge | Notes |
|---|---|---|
| Math / numeric (GSM8K) | `ExactMatchJudge(extract_pattern=r"####\s*(.+)$")` | Extracts number after `####` |
| Short answer / fill-in-blank | `ExactMatchJudge(extract_pattern=None)` | Compares full response |
| Multiple choice (A/B/C/D) | `ExactMatchJudge(extract_pattern=r"\b([A-D])\b")` | Extracts single letter |
| Open-ended with known keyword | `ContainsMatchJudge()` | Response must contain the expected text |

```python
from prompt_rl import DatasetSample, DatasetSplit, ExactMatchJudge

# Multiple choice example
samples = [
    DatasetSample(
        question="What is the capital of France?\nA) London  B) Paris  C) Berlin  D) Rome",
        answer="B) Paris",
        extracted="B",
    ),
    # ...
]

split = DatasetSplit.from_list(samples, train_ratio=0.8, seed=42)
judge = ExactMatchJudge(extract_pattern=r"\b([A-D])\b")
```

---

## Architecture overview

```
prompt_rl/
├── core/          PerceptionCycle, ActivePolicy
├── llm/           LLMBackend (ABC), LocalLLMBackend
├── critic/        TwoStageCritic (Backward + Optimizer), CriticMemory
├── validation/    Actor, LLMValidationJudge, CriticValidationLoop
├── rl/            HybridReward, RewardHistory, UpdateGate
├── population/    PromptGenome, Leaderboard
├── feedback/      FeedbackAggregator, CursorTrace       [online only]
├── loop/          OnlineCriticLoop  ← shared engine
└── dataset/       DatasetSample, DatasetSplit            [offline only]
                   ExactMatchJudge, ContainsMatchJudge
                   DatasetLoop, DatasetResult, EpochMetrics

demos/
└── human_watch/   FastAPI server + Playwright monitor    [online demo]

examples/
└── gsm8k_experiment/  download, curate, train, eval     [offline demo]
```

Both modes share the same `OnlineCriticLoop` engine internally.
`DatasetLoop` wraps it and drives it with dataset samples instead of
live human interactions.

---

## HybridReward tuning

```
R = λ_feedback · H + λ_critic · C − λ_change · word_change_ratio
```

| Parameter | Default | Description |
|---|---|---|
| `lambda_feedback` | `0.9` | Weight of human/judge verdict (dominant signal) |
| `lambda_critic` | `0.1` | Weight of Critic's self-assessed score |
| `lambda_change` | `0.3` | Penalty for large prompt rewrites |
| `max_change_ratio` | `0.35` | Rewrites beyond 35% of words are blocked entirely |

```python
from prompt_rl import HybridReward

reward_fn = HybridReward(lambda_feedback=0.9, lambda_critic=0.1, lambda_change=0.3)
loop = DatasetLoop(..., reward_fn=reward_fn)
```

---

## FAQ

**Q: Can I use any LLM, not just Groq?**
Yes. `LocalLLMBackend` works with any OpenAI-compatible endpoint.
Set `GROQ_BASE_URL` and `GROQ_MODEL` (or pass them directly to `LocalLLMBackend`).

**Q: What happens when the policy converges (online mode)?**
After 5 consecutive CORRECTO cycles with less than 5% word change,
`RewardHistory.converged` becomes `True` and the monitor stops spawning
the evaluator. Send one INCORRECTO verdict to resume learning.

**Q: Can I run multiple epochs over the dataset?**
```python
result = loop.train(split.train, test_samples=split.test, max_epochs=3)
```

**Q: The Critic keeps rewriting the whole prompt. How do I prevent that?**
`UpdateGate` blocks rewrites where `word_change_ratio > max_change_ratio`
(default 35%). Reduce `max_change_ratio` or increase `lambda_change` to
penalize large changes more.

**Q: How do I inspect what the Critic is doing?**
```python
critic = TwoStageCritic(backend=backend, verbose=True)
loop   = CriticValidationLoop(..., verbose=True)
```
In online mode, check `data/critic_memory.md` or `http://localhost:8000/dashboard`.

**Q: The browser does not open (online mode).**
Run `playwright install chromium` then retry.

**Q: `GROQ_API_KEY` not set error.**
Create `.env` in the project root with `GROQ_API_KEY=gsk_...`
