# core/engines/heuristic_search.py
"""Base commune pour les recherches heuristiques (Dijkstra, Best-First, Weighted A*)."""
import heapq
import numpy as np
from typing import Optional

from core.point import Point
from core.engines.base import Engine
from core.physics import PhysicsConstraints


class HeuristicSearch(Engine):
    """
    Moteur de recherche basé sur une file de priorité.
    Les sous-classes définissent uniquement priority(g, h).
    """

    def __init__(
        self,
        map_obj,
        start_point,
        end_point,
        observer=None,
        physics: Optional[PhysicsConstraints] = None,
    ):
        super().__init__(map_obj, start_point, end_point, observer, physics)

    def _h(self, p):
        """Heuristique octile vers la vraie cible."""
        dx = abs(self.end_point.x - p.x)
        dy = abs(self.end_point.y - p.y)
        dz = abs(self.end_point.z - p.z)
        return dx + dy + dz + (np.sqrt(3) - 3) * min(dx, dy, dz)

    def _g(self, p):
        """Coût g approximatif depuis le départ (forme octile)."""
        dx = abs(p.x - self.start_point.x)
        dy = abs(p.y - self.start_point.y)
        dz = abs(p.z - self.start_point.z)
        return dx + dy + dz + (np.sqrt(3) - 3) * min(dx, dy, dz)

    def priority(self, g, h):
        """À surcharger : retourne la clé de priorité (plus petit = plus prioritaire)."""
        raise NotImplementedError

    def run(self):
        with self._time_context():
            open_heap = []
            open_set = set()
            close_set = set()
            counter = 0

            self.start_point.parent = None
            self.start_point.cost = 0.0
            h0 = self._h(self.start_point)
            heapq.heappush(open_heap, (self.priority(0.0, h0), counter, self.start_point))
            open_set.add(self.start_point)
            counter += 1

            while open_heap:
                if not self.observer.is_running():
                    return False

                _, _, p = heapq.heappop(open_heap)
                open_set.discard(p)
                if p in close_set:
                    continue

                self.stats.nodes_explored += 1
                self.observer.on_explore(p, alpha=0.1)

                if p.isEgale(self.end_point):
                    path = self._build_path(p)
                    self.stats.success = True
                    self._finalize_path(path)
                    return True

                close_set.add(p)

                for n in self._neighbors(p, previous_point=p.parent):
                    if n in close_set or n in open_set:
                        continue
                    n.parent = p
                    g = self._g(n)
                    h = self._h(n)
                    n.cost = g + h
                    heapq.heappush(open_heap, (self.priority(g, h), counter, n))
                    open_set.add(n)
                    counter += 1

            self.stats.success = False
            return False

    def _build_path(self, p):
        path = []
        while p is not None:
            path.insert(0, p)
            if p.isEgale(self.start_point):
                break
            p = p.parent
        return path
