# core/engines/weighted_a_star.py
"""Weighted A* : priorité = g + w * h. w > 1 → plus rapide, sous-optimal."""
from typing import Optional
from core.engines.heuristic_search import HeuristicSearch
from core.physics import PhysicsConstraints


class WeightedAStar(HeuristicSearch):
    def __init__(self, map_obj, start_point, end_point, observer=None,
                 weight=2.0, physics: Optional[PhysicsConstraints] = None):
        self.weight = weight
        super().__init__(map_obj, start_point, end_point, observer, physics)

    def name(self) -> str:
        w = getattr(self, "weight", 2.0)
        return f"WeightedA* (w={w})"

    def priority(self, g, h):
        return g + self.weight * h
