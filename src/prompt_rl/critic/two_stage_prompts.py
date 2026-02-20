"""
Prompts para el pipeline de critic en dos etapas.

Critic 1.1 (Backward): contexto completo (conversación, evaluación humana + cursor).
Critic 1.2 (Optimizer): genera nuevo system prompt a partir del feedback.
"""

# ── Critic 1.1 (Backward) ───────────────────────────────────────────────────────

GLOSSARY_BACKWARD = """
### Glosario:
# - <LM_SYSTEM_PROMPT>: El system prompt que guiaba al modelo.
# - <LM_INPUT>: La entrada al modelo (user query).
# - <LM_OUTPUT>: La respuesta del modelo.
# - <EVALUACION_HUMANA>: Feedback y veredicto del evaluador humano.
# - <VARIABLE>: El fragmento a mejorar (system prompt).
# - <ROLE>: Descripción del rol de la variable."""

CRITIC_BACKWARD_SYSTEM = (
    "Eres parte de un sistema de optimización que mejora textos (variables). "
    "Eres el motor de feedback (gradiente). "
    "Tu única responsabilidad es dar feedback inteligente y crítico constructivo a las variables, "
    "dado el objetivo especificado en <EVALUACION_HUMANA> </EVALUACION_HUMANA>. "
    "Las variables pueden ser system prompts, soluciones, etc. "
    "Solo proporciona estrategias, explicaciones y métodos de cambio. "
    "NO propongas una nueva versión de la variable; eso lo hará el optimizador. "
    "Tu trabajo es enviar feedback y crítica (calcular 'gradientes'). "
    "Ejemplos: 'Dado que los modelos fallan en X...', 'Añadir Y puede corregir porque...'. "
    "Si la respuesta fue correcta y el feedback es positivo, indica qué aspectos mantener.\n\n"
    "CRÍTICO - CEGUERA PARA EL OPTIMIZADOR: "
    "Tu feedback será consumido por el Optimizer (etapa 1.2), que trabaja a CIEGAS de la conversación: "
    "NO ves user_query ni bot_response. Por tanto, tu feedback NO DEBE contener alusiones a la pregunta concreta "
    "(ej: prohibido 'cuando pregunten por el Plan Pro...', 'el usuario preguntó sobre X...'). "
    "Escribe feedback GENÉRICO y aplicable al system prompt: ej. 'Incluir en la información de planes que los precios llevan IVA' "
    "en vez de 'Mencionar IVA cuando pregunten por el Plan Pro'.\n"
    f"{GLOSSARY_BACKWARD}"
)

CONVERSATION_TEMPLATE = (
    "<LM_SYSTEM_PROMPT> {system_prompt} </LM_SYSTEM_PROMPT>\n\n"
    "<LM_INPUT> {prompt} </LM_INPUT>\n\n"
    "<LM_OUTPUT> {response_value} </LM_OUTPUT>\n\n"
)

CONVERSATION_START = (
    "Darás feedback a una variable con el siguiente rol: <ROLE> {variable_desc} </ROLE>. "
    "Aquí hay una conversación con un modelo de lenguaje:\n\n"
    "{conversation}"
)

OBJECTIVE_INSTRUCTION = (
    "La respuesta del LM fue evaluada por un humano.\n\n"
    "{human_evaluation_block}\n\n"
    "Tu objetivo es dar feedback y crítica a la variable para abordar la evaluación humana. "
    "Extrae las pistas relevantes del feedback y del trazado de cursor si existe. "
    "Recuerda: escribe feedback GENÉRICO (sin referencias a la pregunta concreta), "
    "porque el Optimizer lo lee a ciegas.\n\n"
)

EVALUATE_VARIABLE = (
    "Queremos feedback para el {variable_desc}. "
    "Da feedback al siguiente fragmento:\n\n"
    "<VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Describe cómo podría mejorarse para satisfacer la evaluación humana. "
    "Feedback GENÉRICO (sin alusiones a la pregunta del usuario): el Optimizer trabaja a ciegas.\n\n"
)

# ── Propagación por componentes (predecessors) ─────────────────────────────────
COMPONENT_NAMES = ["role", "hard_rules", "context_amplification", "soft_guidelines"]

COMPONENT_DESCRIPTIONS = {
    "role": "rol e identidad del asistente (quién es, qué hace)",
    "hard_rules": "reglas obligatorias de comportamiento (imperativos, siempre cumplir)",
    "context_amplification": "ampliación de contexto (información, fuentes de verdad)",
    "soft_guidelines": "directrices de estilo y formato (recomendaciones)",
}

EVALUATE_BY_COMPONENT = (
    "El system prompt está estructurado en 4 componentes. "
    "Da feedback ESPECÍFICO para cada componente que necesite mejora.\n\n"
    "{components_block}\n\n"
    "Para cada componente, indica cómo mejorarlo según la evaluación humana. "
    "Si un componente está bien, escribe 'OK' o 'mantener'. "
    "Feedback GENÉRICO (sin mencionar la pregunta del usuario): el Optimizer trabaja a ciegas. "
    "Responde con el formato exacto:\n\n"
    "<FEEDBACK component=\"role\">\n...\n</FEEDBACK>\n"
    "<FEEDBACK component=\"hard_rules\">\n...\n</FEEDBACK>\n"
    "<FEEDBACK component=\"context_amplification\">\n...\n</FEEDBACK>\n"
    "<FEEDBACK component=\"soft_guidelines\">\n...\n</FEEDBACK>"
)

COMPONENT_BLOCK_TEMPLATE = (
    "<COMPONENT name=\"{name}\" desc=\"{desc}\">\n{content}\n</COMPONENT>"
)

# Optimizer con feedback por componente
OPTIMIZER_BY_COMPONENT_SYSTEM = (
    "Eres un optimizador de system prompts estructurados en JSON. "
    "Recibirás el prompt actual (JSON) y feedback ESPECÍFICO por componente. "
    "Solo actualiza los componentes que tengan feedback no trivial (evita 'OK', 'mantener', vacío). "
    "Mantén la estructura JSON. Responde SOLO con el JSON mejorado entre {new_variable_start_tag} y {new_variable_end_tag}."
)

OPTIMIZER_BY_COMPONENT_PREFIX = (
    "System prompt actual (JSON):\n<VARIABLE>\n{policy_json}\n</VARIABLE>\n\n"
    "Feedback por componente:\n\n{feedback_by_component}\n\n"
    "Genera el JSON mejorado incorporando únicamente los cambios necesarios. "
)

GRADIENT_TEMPLATE = (
    "Aquí hay una conversación:\n\n<CONVERSATION>{context}</CONVERSATION>\n\n"
    "El output se usa como {response_desc}\n\n"
    "Feedback obtenido para {variable_desc}:\n\n<FEEDBACK>{feedback}</FEEDBACK>\n\n"
)

# ── Critic 1.2 (Optimizer) ───────────────────────────────────────────────────────
# BLIND: No recibe user_query ni bot_response. Solo el system prompt actual y
# el feedback accionable del Critic 1.1. Evita overfitting a conversaciones concretas.

CRITIC_OPTIMIZER_SYSTEM = (
    "Eres parte de un sistema de optimización que mejora system prompts. "
    "Recibirás el system prompt actual y feedback de un crítico sobre cómo mejorarlo. "
    "NO ves la conversación ni la respuesta del modelo; solo el feedback. "
    "Tu trabajo es generar una NUEVA versión del system prompt que incorpore el feedback. "
    "El feedback puede ser ruidoso; identifica qué es importante. "
    "IMPORTANTE: Responde SOLO con el prompt mejorado entre {new_variable_start_tag} y {new_variable_end_tag}."
)

OPTIMIZER_PREFIX = (
    "Rol de la variable: <ROLE>{variable_desc}</ROLE>.\n\n"
    "Variable actual (system prompt a mejorar):\n<VARIABLE> {variable_short} </VARIABLE>\n\n"
    "Feedback del crítico (sin contexto de conversación):\n\n<FEEDBACK>{variable_grad}</FEEDBACK>\n\n"
    "Mejora la variable usando únicamente el feedback anterior. "
)

OPTIMIZER_SUFFIX = (
    "Envía SOLO el prompt mejorado entre {new_variable_start_tag} y {new_variable_end_tag}."
)

# ── Gradient memory: feedbacks pasados ─────────────────────────────────────────
# Feedback repetido en varios ciclos sugiere que los cambios son insuficientes.
OPTIMIZER_GRADIENT_MEMORY = (
    "\n\n### Feedback de ciclos anteriores (cambios insuficientes)\n"
    "Si el feedback se repite, haz cambios más significativos:\n\n"
    "{gradient_memory}"
)

# ── Constraints: restricciones en lenguaje natural ─────────────────────────────
OPTIMIZER_CONSTRAINTS = (
    "\n\n### Restricciones obligatorias\n"
    "Debes cumplir:\n\n"
    "{constraint_text}\n\n"
)

# ── In-context examples: ejemplos antes/después ───────────────────────────────
OPTIMIZER_IN_CONTEXT_EXAMPLES = (
    "\n\n### Ejemplos de mejoras (sigue una estructura similar)\n"
    "<EXAMPLES>\n{in_context_examples}\n</EXAMPLES>\n\n"
)
