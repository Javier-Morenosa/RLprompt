"""Dataset-mode primitives for offline prompt optimization.

Enables RL-based prompt refinement using Q&A datasets with train/test splits,
as opposed to the online (chatbot) mode driven by live human feedback.
"""

from prompt_rl.dataset.sample import DatasetSample, DatasetSplit
from prompt_rl.dataset.judge import DatasetJudge, ExactMatchJudge, ContainsMatchJudge
from prompt_rl.dataset.loop import DatasetLoop, DatasetResult, EpochMetrics

__all__ = [
    "DatasetSample",
    "DatasetSplit",
    "DatasetJudge",
    "ExactMatchJudge",
    "ContainsMatchJudge",
    "DatasetLoop",
    "DatasetResult",
    "EpochMetrics",
]
