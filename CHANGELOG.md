# Changelog

## [0.3.0] - 2025-02-16

### Added

- **Actor-Critic loop** (`prompt_rl.actor_critic_loop`): flujo autónomo para refinamiento de prompts con feedback humano
  - `LLMActor`: genera variaciones de system prompt y respuestas; recibe la query para prompts orientados a la pregunta
  - `LLMCritic`: recibe solo (system_prompt, reward) y opcionalmente comentario humano; nunca ve query ni respuestas
  - `HumanMultiSelectFeedback`: selección múltiple de respuestas correctas
  - `launch_integrated()`: UI Gradio integrada con bucle automático tras cada Submit selection
- `ActorCriticConfig`, `ActorCriticLoop` para ejecución programática
- Ejemplos: `gradio_feedback_example.py` (Ollama/Mock, `--port`), `actor_critic_example.py`

### Changed

- Formato de prompt en `generate_responses`: separación clara USER QUESTION / ASSISTANT RESPONSE para respuestas más relevantes
- Actor y Critic: instrucciones reforzadas para que las respuestas sean relevantes a la pregunta del usuario

### Fixed

- Respuestas irrelevantes: Actor ahora recibe la query al generar variaciones; prompts refuerzan responder directamente

### References

Esta actualización se inspira en:

- [Expanding the Capabilities of Reinforcement Learning](https://arxiv.org/pdf/2602.02482)
- [Teaching Models to Teach Themselves: Reasoning at the Edge of Learnability](https://arxiv.org/pdf/2601.18778)
