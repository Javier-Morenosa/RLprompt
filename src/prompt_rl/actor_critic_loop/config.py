from dataclasses import dataclass


@dataclass
class ActorCriticConfig:
    num_variations: int = 10
    max_tokens: int = 512
    temperature: float = 0.7
