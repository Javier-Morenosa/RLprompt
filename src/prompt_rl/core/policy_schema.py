"""Structured policy schema — JSON format for Actor system prompt.

The Actor receives a policy with:
  - role: identity and task (edited by Critic)
  - hard_rules: mandatory behavioral rules (always enforced)
  - context_amplification: info/reminders that extend context (not rigid rules)
  - soft_guidelines: style/format recommendations

ACTOR_STRUCTURE_PREAMBLE: meta-instructions inherent to the system, always
prepended before the policy content. The Critic never edits this.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Inherent to the system — always prepended before policy content, never edited by Critic
ACTOR_STRUCTURE_PREAMBLE = """Tu system prompt está estructurado en 3 bloques. Interpreta cada uno así:

• REGLAS OBLIGATORIAS: Comportamientos que DEBES cumplir siempre. Son imperativos.
• AMPLIACIÓN DE CONTEXTO: Información que amplía tu conocimiento. Úsala para enriquecer respuestas; no son reglas rígidas.
• DIRECTRICES: Recomendaciones de estilo y formato."""

# Política mínima al reset: solo identidad de asistente (evita alucinaciones con prompt vacío)
MINIMAL_POLICY_JSON = json.dumps(
    {
        "role": "Eres un asistente. Responde usando la información del catálogo cuando la tengas.",
        "hard_rules": ["Responde siempre en español"],
        "context_amplification": [],
        "soft_guidelines": [],
    },
    ensure_ascii=False,
    indent=2,
)

# Default policy when system_prompt.md is empty or missing (role = identity only, no structure)
DEFAULT_POLICY_JSON = json.dumps(
    {
        "role": "Eres un asistente de negocio amable y profesional. Responde ÚNICAMENTE con información del catálogo.",
        "hard_rules": [
            "Si el usuario pregunta algo que no está en el catálogo, indícalo claramente",
            "Responde siempre en español",
        ],
        "context_amplification": [
            "El catálogo (siguiente sección) es tu única fuente de verdad para precios, planes y políticas",
        ],
        "soft_guidelines": ["Responde de forma concisa"],
    },
    ensure_ascii=False,
    indent=2,
)


def ensure_policy_file(path: Path, legacy_path: Path | None = None) -> str:
    """
    Ensure the policy file exists and has content. If empty or missing:
    - Migrate from legacy_path if it exists
    - Otherwise write MINIMAL_POLICY_JSON and return it
    Returns the file content (or minimal policy after writing).
    """
    # Migrate legacy first
    if legacy_path and not path.exists() and legacy_path.exists():
        legacy_path.rename(path)
    # Read; if empty or missing, bootstrap
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        text = ""
    if not text:
        path.write_text(MINIMAL_POLICY_JSON.strip(), encoding="utf-8")
        return MINIMAL_POLICY_JSON.strip()
    return text


@dataclass
class PolicySchema:
    role: str = ""
    hard_rules: list[str] = field(default_factory=list)
    context_amplification: list[str] = field(default_factory=list)
    soft_guidelines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "hard_rules": self.hard_rules,
            "context_amplification": self.context_amplification,
            "soft_guidelines": self.soft_guidelines,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def to_llm_text(self) -> str:
        """Build the system prompt string for the LLM."""
        parts = []
        if self.role.strip():
            parts.append(self.role.strip())
        if self.hard_rules:
            rules_text = "\n".join(f"  {i+1}. {r}" for i, r in enumerate(self.hard_rules))
            parts.append("\n--- REGLAS OBLIGATORIAS (cumplir siempre) ---\n" + rules_text)
        if self.context_amplification:
            ctx_text = "\n".join(f"  - {c}" for c in self.context_amplification)
            parts.append("\n--- AMPLIACIÓN DE CONTEXTO ---\n" + ctx_text)
        if self.soft_guidelines:
            guides_text = "\n".join(f"  - {g}" for g in self.soft_guidelines)
            parts.append("\n--- DIRECTRICES (recomendadas) ---\n" + guides_text)
        return "\n".join(parts).strip() if parts else ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PolicySchema:
        return cls(
            role=str(d.get("role", "")),
            hard_rules=list(d.get("hard_rules", []) or []),
            context_amplification=list(d.get("context_amplification", []) or []),
            soft_guidelines=list(d.get("soft_guidelines", []) or []),
        )

    @classmethod
    def from_json(cls, s: str) -> PolicySchema | None:
        """Parse JSON string. Returns None if invalid or not policy JSON."""
        s = s.strip()
        if not s:
            return None
        if not (s.startswith("{") and "}" in s):
            return None
        try:
            data = json.loads(s)
            if isinstance(data, dict) and ("role" in data or "hard_rules" in data):
                return cls.from_dict(data)
        except json.JSONDecodeError:
            pass
        return None

    @classmethod
    def from_legacy_text(cls, text: str) -> PolicySchema:
        """Wrap plain text as policy with empty hard_rules."""
        text = text.strip()
        if not text:
            return cls(role="", hard_rules=[], context_amplification=[], soft_guidelines=[])
        # Put everything in role for backward compatibility
        return cls(role=text, hard_rules=[], context_amplification=[], soft_guidelines=[])


def parse_policy(raw: str) -> tuple[PolicySchema, bool]:
    """
    Parse raw file content. Returns (PolicySchema, is_structured).
    is_structured=True if valid JSON policy, False if legacy plain text.
    """
    parsed = PolicySchema.from_json(raw)
    if parsed is not None:
        return parsed, True
    return PolicySchema.from_legacy_text(raw), False


def build_actor_system_text(policy: PolicySchema) -> str:
    """
    Build full system prompt for Actor: inherent preamble + policy content.
    The preamble explains the structure; it is never edited by the Critic.
    """
    policy_text = policy.to_llm_text()
    if not policy_text:
        return ACTOR_STRUCTURE_PREAMBLE
    return f"{ACTOR_STRUCTURE_PREAMBLE}\n\n{policy_text}"
