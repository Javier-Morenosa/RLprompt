# RLPrompt

<div align="center">

[![PyPI](https://img.shields.io/pypi/v/prompt-rl?color=blue&label=pip%20install%20prompt-rl)](https://pypi.org/project/prompt-rl/)
[![Python](https://img.shields.io/pypi/pyversions/prompt-rl)](https://pypi.org/project/prompt-rl/)
[![CI](https://github.com/Javier-Morenosa/RLprompt/actions/workflows/ci.yml/badge.svg)](https://github.com/Javier-Morenosa/RLprompt/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Reinforcement learning library for automatic system-prompt refinement.**
Works with any OpenAI-compatible LLM endpoint — Groq, Ollama, LM Studio, and more.

[Installation](#installation) · [Online mode](#mode-1--online-chatbot) · [Offline mode](#mode-2--offline-dataset) · [Docs](docs/USAGE_GUIDE.md) · [PyPI](https://pypi.org/project/prompt-rl/)

</div>

---

<img width="1572" height="512" alt="RLPrompt banner" src="https://github.com/user-attachments/assets/6a328fd7-abb0-4788-9153-e19851a412db" />

---

## What is RLPrompt?

RLPrompt automatically improves a system prompt through reinforcement learning.
Every time a response is labeled correct or incorrect, a two-stage LLM Critic
analyzes what went wrong, proposes a refined prompt, validates it, and updates
the live policy — without any manual prompt engineering.

It supports two operating modes on the same engine:

| Mode | Signal | Judge | Use case |
|---|---|---|---|
| **Online** | Live human feedback | LLM-based | Chatbot, customer service, conversational AI |
| **Offline** | Q&A dataset + ground truth | Deterministic | Math, MCQ, benchmarks, any task with known answers |

---

## Installation

```bash
pip install prompt-rl
```

```bash
# With Human-Watch demo (FastAPI + Playwright chatbot)
pip install "prompt-rl[human-watch]"
playwright install chromium

# With dataset tools (HuggingFace datasets)
pip install "prompt-rl[gsm8k]"

# Everything
pip install "prompt-rl[all]"
```

---

## How it works

```
Verdict (human or ground truth)
        │
        ▼
PerceptionCycle  ─  system_prompt + query + response + verdict + comment
        │
        ▼
TwoStageCritic
  ├─ Backward   →  "what went wrong and why?"   (actionable feedback)
  └─ Optimizer  →  "write a better prompt"       (proposed system_prompt)
        │
        ▼
CriticValidationLoop  →  re-asks Actor with proposed prompt
  └─ Judge: fixed?  ├─ YES → accept
                    └─ NO  → iterate (up to 3×)
        │
        ▼
HybridReward   R = λ_fb · H + λ_c · C − λ_ch · word_change_ratio
        │
        ▼
UpdateGate  ──  degradation or forced correction?
  ├─ YES  →  ActivePolicy.write()  →  system_prompt updated
  └─ NO   →  policy unchanged

5 consecutive stable cycles  →  converged, refinement stops
```

---

## Mode 1 — Online (Chatbot)

A FastAPI server + Playwright monitor close the feedback loop in real time.
The human labels each response; the Critic refines the prompt automatically.

```bash
# Start server + monitor in one command
rlprompt-backend
```

Or use the library directly:

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
)

loop = OnlineCriticLoop(
    critic=critic,
    policy=ActivePolicy(path="system_prompt.md"),
    history=RewardHistory.from_file("reward_history.json"),
    leaderboard=Leaderboard.from_file("population.json"),
)

cycle = PerceptionCycle(
    system_prompt="You are a helpful assistant.",
    user_query="What is the refund policy?",
    bot_response="We offer a 30-day guarantee.",
    verdict="INCORRECTO",
    comment="The policy is 14 days, not 30.",
)

result = loop.process_cycle(cycle)
print(result.gate.reason)   # "forced"
```

---

## Mode 2 — Offline (Dataset)

Train a system prompt from scratch using a Q&A dataset with known answers.
No human in the loop — the ground truth answer is the feedback signal.

```python
import json
from prompt_rl import (
    ActivePolicy, Actor,
    TwoStageCritic, CriticValidationLoop,
    DatasetLoop, DatasetSplit, ExactMatchJudge,
    RewardHistory, Leaderboard,
)
from prompt_rl.llm.local_backend import LocalLLMBackend

backend = LocalLLMBackend(
    model="llama-3.1-8b-instant",
    base_url="https://api.groq.com/openai/v1",
    api_key="gsk_...",
)
actor = Actor(backend=backend, max_tokens=512, temperature=0.1)

# Load your dataset
train = json.loads(open("data/train.json", encoding="utf-8").read())
test  = json.loads(open("data/test.json",  encoding="utf-8").read())
split = DatasetSplit.from_dicts(train, test)

# Deterministic judge — sends ground truth to the Critic as feedback
judge = ExactMatchJudge(extract_pattern=r"####\s*(.+)$", include_ground_truth=True)

critic = CriticValidationLoop(
    critic=TwoStageCritic(backend=backend),
    actor=actor,
    judge=judge,   # also acts as ValidationJudge inside the loop
    max_iterations=3,
)

policy = ActivePolicy(path="system_prompt.md")
policy.write("", 0)   # start from empty prompt

loop = DatasetLoop(
    critic=critic, policy=policy, actor=actor, judge=judge,
    history=RewardHistory(), leaderboard=Leaderboard(), verbose=True,
)

result = loop.train(split.train, test_samples=split.test)
print(f"{result.acc_before:.1%} → {result.acc_after:.1%}  ({result.delta:+.1%})")
```

### GSM8K quick start

```bash
python examples/gsm8k_experiment/download_gsm8k.py
python examples/gsm8k_experiment/run_gsm8k_train.py
```

---

## Bring your own dataset

| Task | Judge | Pattern |
|---|---|---|
| Math / numeric | `ExactMatchJudge` | `r"####\s*(.+)$"` |
| Multiple choice | `ExactMatchJudge` | `r"\b([A-D])\b"` |
| Short answer | `ExactMatchJudge` | `extract_pattern=None` |
| Open-ended keyword | `ContainsMatchJudge` | — |

---

## Reward formula

```
R = λ_feedback · H + λ_critic · C − λ_change · word_change_ratio

H  — human/judge verdict score   [0, 1]
C  — critic self-assessed score  [0, 1]
```

Defaults: `λ_feedback=0.9  λ_critic=0.1  λ_change=0.3`

Update triggers: **degradation** (`R_curr < R_avg × 0.8`) or **forced** (`INCORRECTO` + non-empty comment).

---

## Tests

```bash
pytest tests/ -v      # 88 tests, no LLM required
```

---

## References

- [Expanding the Capabilities of Reinforcement Learning](https://arxiv.org/pdf/2602.02482)
- [Teaching Models to Teach Themselves](https://arxiv.org/pdf/2601.18778)

---

## License

MIT — see [LICENSE](LICENSE).
