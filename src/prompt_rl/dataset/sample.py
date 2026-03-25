"""DatasetSample and DatasetSplit — primitives for offline prompt optimization."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetSample:
    """A single question-answer pair for offline prompt optimization.

    Attributes:
        question  : Input presented to the model (user query).
        answer    : Full ground-truth answer (may include reasoning steps).
        extracted : Normalized form of the answer used for comparison
                    (e.g. just the final number). Derived from ``answer``
                    by the judge if left empty.
        metadata  : Any additional fields (source, difficulty, id, ...).
    """

    question:  str
    answer:    str
    extracted: str = ""
    metadata:  dict[str, Any] = field(default_factory=dict)


@dataclass
class DatasetSplit:
    """Train/test split of :class:`DatasetSample` instances.

    Example::

        split = DatasetSplit.from_list(samples, train_ratio=0.8, seed=42)
        print(len(split.train), len(split.test))
    """

    train: list[DatasetSample]
    test:  list[DatasetSample]

    @classmethod
    def from_list(
        cls,
        samples: list[DatasetSample],
        train_ratio: float = 0.8,
        seed: int = 42,
    ) -> "DatasetSplit":
        """Shuffle and split *samples* into train and test sets.

        Args:
            samples     : Full list of samples to split.
            train_ratio : Fraction of samples used for training (default 0.8).
            seed        : Random seed for reproducibility (default 42).
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
        rng = random.Random(seed)
        shuffled = list(samples)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_ratio))
        return cls(train=shuffled[:n_train], test=shuffled[n_train:])

    @classmethod
    def from_dicts(
        cls,
        train: list[dict],
        test: list[dict],
        question_key: str = "question",
        answer_key: str = "answer",
        extracted_key: str = "extracted",
    ) -> "DatasetSplit":
        """Build a split from plain dicts (e.g. loaded from JSON).

        Args:
            train         : List of dicts for the training set.
            test          : List of dicts for the test set.
            question_key  : Dict key for the question field.
            answer_key    : Dict key for the full answer field.
            extracted_key : Dict key for the normalized answer field (optional).
        """
        def _to_sample(d: dict) -> DatasetSample:
            return DatasetSample(
                question=d[question_key],
                answer=d[answer_key],
                extracted=d.get(extracted_key, ""),
                metadata={k: v for k, v in d.items()
                          if k not in (question_key, answer_key, extracted_key)},
            )
        return cls(
            train=[_to_sample(d) for d in train],
            test=[_to_sample(d) for d in test],
        )
