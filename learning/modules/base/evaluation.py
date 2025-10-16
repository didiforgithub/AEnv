from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List

from pydantic import BaseModel, Field, ConfigDict


class EvaluationMetric(Enum):
    ACCURACY = "accuracy"
    COST = "cost"


class EvaluationResult(BaseModel):
    candidate_round: int = Field(...)
    metrics: Dict[EvaluationMetric, float] = Field(default_factory=dict)
    trajectories: List[Dict[str, Any]] = Field(default_factory=list)


class Evaluation(BaseModel):
    """Abstract evaluator interface.

    Concrete implementations should run a candidate and return metrics/trajectories.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    async def evaluate_candidate(self, candidate) -> EvaluationResult:  # noqa: ANN001
        raise NotImplementedError

