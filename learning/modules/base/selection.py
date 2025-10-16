"""
Selection strategies for multi-turn optimization.

Notes:
- CurrentBest: accepts a metrics list but uses the first as primary.
- ParetoFront: expects two metrics (e.g., [ACCURACY, COST]) and returns a non-dominated candidate
  with maximize for ACCURACY-like and minimize for COST-like.
"""

from enum import Enum
from typing import Dict, List, TYPE_CHECKING
from pydantic import BaseModel

from .evaluation import EvaluationMetric
if TYPE_CHECKING:
    from .candidate import Candidate


class SelectionType(Enum):
    PARETO_FRONT = "pareto_front"
    CURRENT_BEST = "current_best"


class Selection(BaseModel):
    def select(self, candidates: Dict[int, "Candidate"], metrics_type: List[EvaluationMetric]) -> "Candidate":
        raise NotImplementedError


class ParetoFrontSelection(Selection):
    def _metric_value(self, cand: 'Candidate', metric: EvaluationMetric) -> float:
        # Defaults: accuracy->0 (maximize), cost->inf (minimize)
        if metric == EvaluationMetric.COST:
            return float(cand.metrics.get(metric, float('inf')))
        return float(cand.metrics.get(metric, 0.0))

    def _dominates(self, a: 'Candidate', b: 'Candidate', m1: EvaluationMetric, m2: EvaluationMetric) -> bool:
        # Assume m1 is to maximize if not COST, and m2 minimize if COST (and symmetric)
        a1, b1 = self._metric_value(a, m1), self._metric_value(b, m1)
        a2, b2 = self._metric_value(a, m2), self._metric_value(b, m2)

        def better(x, y, metric):
            if metric == EvaluationMetric.COST:
                return x < y
            return x > y

        def no_worse(x, y, metric):
            if metric == EvaluationMetric.COST:
                return x <= y
            return x >= y

        cond_all = no_worse(a1, b1, m1) and no_worse(a2, b2, m2)
        cond_any = better(a1, b1, m1) or better(a2, b2, m2)
        return cond_all and cond_any

    def select(self, candidates: Dict[int, "Candidate"], metrics_type: List[EvaluationMetric]) -> "Candidate":
        assert candidates, "No candidates to select from"
        assert len(metrics_type) == 2, "ParetoFrontSelection requires exactly two metrics"
        m1, m2 = metrics_type[0], metrics_type[1]

        cand_list = list(candidates.values())
        pareto: List['Candidate'] = []
        for c in cand_list:
            dominated = False
            for other in cand_list:
                if other is c:
                    continue
                if self._dominates(other, c, m1, m2):
                    dominated = True
                    break
            if not dominated:
                pareto.append(c)

        # If multiple on front, tie-break: higher m1 (maximize unless COST), then lower COST if any, else latest round
        def sort_key(c: 'Candidate'):
            v1 = self._metric_value(c, m1)
            v2 = self._metric_value(c, m2)
            # Convert to sorting where higher is better for primary, lower is better for cost
            primary = v1 if m1 != EvaluationMetric.COST else -v1
            secondary = -v2 if m2 == EvaluationMetric.COST else v2
            return (primary, secondary, c.round)

        pareto.sort(key=sort_key, reverse=True)
        return pareto[0]


class CurrentBestSelection(Selection):
    def select(self, candidates: Dict[int, "Candidate"], metrics_type: List[EvaluationMetric]) -> "Candidate":
        assert candidates, "No candidates to select from"
        if not metrics_type:
            latest_round = max(candidates.keys())
            return candidates[latest_round]
        # Accept a list but only use the first metric as primary
        primary = metrics_type[0]
        best = None
        best_score = float("-inf")
        for c in candidates.values():
            score = c.metrics.get(primary, float("-inf"))
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            latest_round = max(candidates.keys())
            return candidates[latest_round]
        return best
