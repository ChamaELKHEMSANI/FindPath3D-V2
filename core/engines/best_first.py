# core/engines/best_first.py
"""Best-First : priorité = h seule (greedy). Rapide, non optimal."""
from typing import Optional
from core.engines.heuristic_search import HeuristicSearch
from core.physics import PhysicsConstraints


class BestFirst(HeuristicSearch):
    def __init__(self, map_obj, start_point, end_point, observer=None,
                 physics: Optional[PhysicsConstraints] = None):
        super().__init__(map_obj, start_point, end_point, observer, physics)

    def name(self) -> str:
        return "Best-First"

    def priority(self, g, h):
        return h
