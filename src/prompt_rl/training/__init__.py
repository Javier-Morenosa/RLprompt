"""
Integrated optimization flow and training process.

- TrainingLoop: initialization, main loop and refinement phases
- GPRO: optimization algorithm that trains the Actor-Critic
- Parameters: Actor (θ_actor), Critic (θ_critic), evolutionary (α, β, elite)
"""

from prompt_rl.training.config import TrainingConfig
from prompt_rl.training.flow import HybridOptimizationFlow
from prompt_rl.training.gpro import GPROOptimizer, GPROTransition, NoOpGPROOptimizer
from prompt_rl.training.loop import TrainingLoop, TrainingPhase
from prompt_rl.training.metrics import (
    MetricsCollector,
    FrameworkMetrics,
    FitnessOverTime,
    DiversityMetrics,
    diversity_metrics,
    variance_between_prompts,
    convergence_slope,
    generations_to_threshold,
)

__all__ = [
    "TrainingLoop",
    "TrainingConfig",
    "TrainingPhase",
    "HybridOptimizationFlow",
    "GPROOptimizer",
    "GPROTransition",
    "NoOpGPROOptimizer",
    "MetricsCollector",
    "FrameworkMetrics",
    "FitnessOverTime",
    "DiversityMetrics",
    "diversity_metrics",
    "variance_between_prompts",
    "convergence_slope",
    "generations_to_threshold",
]
