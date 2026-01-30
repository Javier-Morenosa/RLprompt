"""
prompt_rl: Reinforcement learning framework for prompt refinement with LLMs.

Main components:
- core: Prompt representation and refinement loop
- rl: Environment, rewards and policy for RL
- llm: Backends and integration with language models
"""

__version__ = "0.1.0"

from prompt_rl.core.prompt import Prompt
from prompt_rl.core.refiner import RefinementLoop

__all__ = [
    "__version__",
    "Prompt",
    "RefinementLoop",
]
