# core/engines/dijkstra.py
"""Dijkstra : A* sans heuristique (priorité = g)."""
from typing import Optional
from core.engines.heuristic_search import HeuristicSearch
from core.physics import PhysicsConstraints


class Dijkstra(HeuristicSearch):
    def __init__(self, map_obj, start_point, end_point, observer=None,
                 physics: Optional[PhysicsConstraints] = None):
        super().__init__(map_obj, start_point, end_point, observer, physics)

    def name(self) -> str:
        return "Dijkstra"

    def priority(self, g, h):
        return g
