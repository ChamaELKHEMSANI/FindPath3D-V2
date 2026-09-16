# core/engines/theta_star.py
"""
Theta* - any-angle pathfinding (Phase 2 raffinement).
Produit des chemins plus fluides en autorisant des lignes de vue
directes entre parents non adjacents.
"""
from __future__ import annotations

import heapq
import math
from typing import Optional

from core.point import Point
from core.engines.base import Engine
from core.physics import PhysicsConstraints


class ThetaStar(Engine):
    """
    Theta* (Nash et al.) sur grille 3D.
    Si line-of-sight(parent(s), n) : on relie n directement à parent(s).
    Sinon comportement type A*.
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

    def name(self) -> str:
        return "Theta*"

    def _h(self, p: Point) -> float:
        dx = abs(self.end_point.x - p.x)
        dy = abs(self.end_point.y - p.y)
        dz = abs(self.end_point.z - p.z)
        return math.sqrt(dx * dx + dy * dy + dz * dz)

    def _dist(self, a: Point, b: Point) -> float:
        return math.sqrt(
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
        )

    def _line_of_sight(self, a: Point, b: Point) -> bool:
        """
        Supercover 3D simplifié : échantillonne le segment a→b.
        Retourne False si un voxel obstacle est croisé.
        """
        dx = b.x - a.x
        dy = b.y - a.y
        dz = b.z - a.z
        steps = max(abs(dx), abs(dy), abs(dz), 1)
        for i in range(steps + 1):
            t = i / steps
            x = int(round(a.x + dx * t))
            y = int(round(a.y + dy * t))
            z = int(round(a.z + dz * t))
            if not self._is_valid(x, y, z):
                return False
            # Physique : pente le long du segment
            if self.physics is not None and i > 0:
                prev_t = (i - 1) / steps
                px = int(round(a.x + dx * prev_t))
                py = int(round(a.y + dy * prev_t))
                pz = int(round(a.z + dz * prev_t))
                if not self.physics.is_valid_move(
                    Point(px, py, pz), Point(x, y, z)
                ):
                    return False
        return True

    def run(self):
        with self._time_context():
            open_heap = []
            g_score = {self.start_point: 0.0}
            parent = {self.start_point: self.start_point}
            closed = set()
            counter = 0

            self.start_point.parent = None
            f0 = self._h(self.start_point)
            heapq.heappush(open_heap, (f0, counter, self.start_point))
            counter += 1
            in_open = {self.start_point}

            while open_heap:
                if not self.observer.is_running():
                    return False
                _, _, s = heapq.heappop(open_heap)
                in_open.discard(s)
                if s in closed:
                    continue
                closed.add(s)
                self.stats.nodes_explored += 1
                self.observer.on_explore(s, alpha=0.1)

                if s.isEgale(self.end_point):
                    path = self._reconstruct(s, parent)
                    self._last_path = path
                    self.stats.success = True
                    self._finalize_path(path)
                    return True

                for n in self._neighbors(s, previous_point=parent.get(s)):
                    if n in closed:
                        continue
                    # Theta* update
                    ps = parent[s]
                    if self._line_of_sight(ps, n):
                        # path  s → n  via parent(s)
                        tentative = g_score[ps] + self._dist(ps, n)
                        better_parent = ps
                    else:
                        tentative = g_score[s] + self._dist(s, n)
                        better_parent = s

                    if n not in g_score or tentative < g_score[n]:
                        g_score[n] = tentative
                        parent[n] = better_parent
                        n.parent = better_parent
                        f = tentative + self._h(n)
                        n.cost = f
                        if n not in in_open:
                            heapq.heappush(open_heap, (f, counter, n))
                            in_open.add(n)
                            counter += 1

            self.stats.success = False
            return False

    def _reconstruct(self, end: Point, parent: dict) -> list:
        path = []
        cur = end
        visited = set()
        while cur is not None:
            path.insert(0, cur)
            if cur.isEgale(self.start_point):
                break
            key = (cur.x, cur.y, cur.z)
            if key in visited:
                break
            visited.add(key)
            cur = parent.get(cur, cur.parent)
        return path
