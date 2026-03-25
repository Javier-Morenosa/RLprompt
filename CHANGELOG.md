# Changelog

## [1.0.0] - 2025-03-07

### Changed

- **Versión unificada** a 1.0.0 en pyproject.toml, __init__.py y README
- **Metadatos:** authors y project.urls actualizados (Javier-Morenosa/RLprompt)
- **Documentación:** corrección de referencias obsoletas (critic.two_stage → critic.two_stage_critic, system_prompt.txt → system_prompt.md)
- **CONTRIBUTING:** pip install sin openai redundante; referencias a tests actualizadas (integración en demos/human_watch/tests/)
- **Docs conceptuales:** aviso explícito de que el código de ejemplo es de diseño y puede no existir en la implementación actual

### Added

- **Exports públicos:** CriticBackward, CriticOptimizer, BackwardOutput añadidos a la API pública del paquete

### Fixed

- **.gitignore:** exclusión de datos de runtime y experimentos (data/*.json, data/*.md, data/gsm8k/, data/gsm8k_experiment/, data/logs/, data/prompts/, evaluator.log)

---

## [0.3.0] - 2025-02-20

### Changed

- **Estructura como librería** (estilo TextGrad): demo Human-Watch encapsulada en `demos/human_watch/`
- Archivos de la raíz (`server.py`, `monitor.py`, `evaluator.py`, etc.) movidos a `demos/human_watch/`
- Paths centralizados en `PROJECT_ROOT` (raíz del repo); archivos de estado siguen en la raíz
- Entry points: `rlprompt-backend`, `rlprompt-serve`, `rlprompt-reset`
- Ejecución: `python -m demos.human_watch.run_backend` o `rlprompt-backend`

### Removed

- Ejemplos legacy que importaban módulos inexistentes: `actor_critic_example.py`, `gradio_feedback_example.py`, `hybrid_system_example.py`, `hybrid_local_llm_example.py`, `rl_env_example.py`

### Kept

- Ejemplos viables: `examples/two_stage_example.py`, `examples/validation_loop_example.py`

### References

- [Expanding the Capabilities of Reinforcement Learning](https://arxiv.org/pdf/2602.02482)
- [Teaching Models to Teach Themselves](https://arxiv.org/pdf/2601.18778)
