# core/engines/a_star.py
"""A* — recherche heuristique classique (Phase 1 : physique + heuristique vers end)."""
import heapq
import numpy as np
from typing import Optional

from core.point import Point
from core.engines.base import Engine
from core.physics import PhysicsConstraints


class AStar(Engine):

    def __init__(
        self,
        map_obj,
        start_point,
        end_point,
        observer=None,
        physics: Optional[PhysicsConstraints] = None,
        heuristic=None,
    ):
        super().__init__(map_obj, start_point, end_point, observer, physics)
        self.heuristic = heuristic  # None = use built-in _h

    def name(self) -> str:
        return "A*"

    def _h(self, p):
        """Heuristique : objet injectable ou octile vers end_point."""
        if self.heuristic is not None:
            return self.heuristic.estimate(p, self.end_point)
        dx = abs(self.end_point.x - p.x)
        dy = abs(self.end_point.y - p.y)
        dz = abs(self.end_point.z - p.z)
        return dx + dy + dz + (np.sqrt(3) - 3) * min(dx, dy, dz)

    def _g_from_parent(self, parent, child):
        """Coût réel du segment parent → child (euclidien)."""
        return np.sqrt(
            (parent.x - child.x) ** 2
            + (parent.y - child.y) ** 2
            + (parent.z - child.z) ** 2
        )

    def run(self):
        with self._time_context():
            open_heap = []
            open_set = set()
            close_set = set()
            g_score = {self.start_point: 0.0}
            counter = 0

            self.start_point.parent = None
            h0 = self._h(self.start_point)
            heapq.heappush(open_heap, (h0, counter, self.start_point))
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
                    self._last_path = path
                    self.stats.success = True
                    self._finalize_path(path)
                    return True

                close_set.add(p)

                for n in self._neighbors(p, previous_point=p.parent):
                    if n in close_set:
                        continue

                    tentative_g = g_score[p] + self._g_from_parent(p, n)

                    if n not in g_score or tentative_g < g_score[n]:
                        n.parent = p
                        g_score[n] = tentative_g
                        f = tentative_g + self._h(n)
                        n.cost = f
                        if n not in open_set:
                            heapq.heappush(open_heap, (f, counter, n))
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
