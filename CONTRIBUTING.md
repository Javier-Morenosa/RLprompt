# Contributing to prompt-rl

Thank you for your interest in contributing to **prompt-rl**. This document explains how to set up your environment, run tests, and submit changes.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Getting started](#getting-started)
3. [Development setup](#development-setup)
4. [Running tests](#running-tests)
5. [Code style and linting](#code-style-and-linting)
6. [Submitting changes](#submitting-changes)
7. [Documentation](#documentation)
8. [Project structure](#project-structure)

---

## Code of conduct

- Be respectful and constructive in issues and pull requests.
- Focus on technical merit; avoid off-topic or personal remarks.
- By contributing, you agree that your contributions will be licensed under the same license as the project (MIT).

---

## Getting started

- **Bug reports and feature requests:** Open an [issue](https://github.com/your-username/prompt-rl/issues). For bugs, include Python version, steps to reproduce, and expected vs actual behavior.
- **Small fixes:** Feel free to open a pull request directly (e.g. typos, docs).
- **Larger changes:** Open an issue first to discuss design or scope, then submit a PR when ready.

---

## Development setup

1. **Fork and clone the repository:**

   ```bash
   git clone https://github.com/YOUR_USERNAME/prompt-rl.git
   cd prompt-rl
   ```

2. **Create a virtual environment (recommended):**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   # or:  .venv\Scripts\activate   # Windows
   ```

3. **Install in editable mode with dev and optional dependencies:**

   ```bash
   pip install -e ".[dev,openai,gradio]"
   ```

   This gives you tests, coverage, linting, and optional backends for local experimentation.

---

## Running tests

- Run the full test suite:

  ```bash
  pytest
  ```

- Run with coverage:

  ```bash
  pytest --cov=prompt_rl --cov-report=term-missing
  ```

- Run a specific test file or test:

  ```bash
  pytest tests/test_core.py
  pytest tests/test_core.py -k "test_prompt"
  ```

All new code is expected to be covered by tests where practical. Bug fixes should include a test that reproduces the bug.

---

## Code style and linting

- **Style:** The project uses [Ruff](https://docs.astral.sh/ruff/) for linting and formatting. Configuration is in `pyproject.toml` (e.g. `line-length = 100`, `target-version = "py39"`).

- **Check and fix:**

  ```bash
  ruff check .
  ruff format .
  ```

- **Conventions:**
  - Use type hints for public APIs and new code where it improves clarity.
  - Docstrings: prefer clear, concise descriptions; follow existing style in the codebase.
  - Keep the codebase and all docs/comments in **English**.

---

## Submitting changes

1. **Branch:** Create a branch from `main` (e.g. `fix/mock-llm-edge-case` or `feat/add-xyz`).

2. **Commit:** Use clear, descriptive commit messages. Prefer present tense (“Add X” rather than “Added X”).

3. **Tests and lint:** Ensure `pytest` passes and `ruff check .` / `ruff format .` are clean before submitting.

4. **Pull request:**
   - Target the `main` branch.
   - Describe what the PR does and how it was tested.
   - Reference any related issue (e.g. “Fixes #123”).
   - Keep PRs focused; split large changes into smaller ones when possible.

5. **Review:** Maintainers will review and may request changes. Once approved, your PR will be merged.

---

## Documentation

- **User-facing:** [README.md](README.md) and [docs/USAGE_GUIDE.md](docs/USAGE_GUIDE.md) cover installation, quick start, and usage.
- **Design and theory:** See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/PARAMETRIZATION.md](docs/PARAMETRIZATION.md), [docs/KEY_INNOVATIONS.md](docs/KEY_INNOVATIONS.md), [docs/GPRO.md](docs/GPRO.md), [docs/PARALLEL_EVAL.md](docs/PARALLEL_EVAL.md), and [docs/METRICS.md](docs/METRICS.md).

When adding features or changing behavior, please update the README or the relevant doc so that the docs stay in sync with the code.

---

## Project structure

```
prompt_rl/
├── src/prompt_rl/     # Main package (core, rl, llm, evolution, actor_critic, feedback, training, utils)
├── tests/             # Pytest tests
├── examples/          # Example scripts (basic_refinement, hybrid_system, gradio_feedback, rl_env)
├── docs/              # Markdown documentation
├── pyproject.toml     # Build and tool config (Ruff, pytest)
├── README.md
├── CONTRIBUTING.md    # This file
└── LICENSE
```

- New features should fit into the existing modules or extend them via clear interfaces (e.g. new `LLMBackend`, `Actor`, `Critic`, or reward components).
- Add or extend examples in `examples/` when introducing new workflows.

---

Thank you for contributing to prompt-rl.
