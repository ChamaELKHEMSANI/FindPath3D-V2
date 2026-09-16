# core/engines/jps.py
"""
Jump Point Search (JPS) 3D simplifié - Phase 3.
Accélère A* en sautant les symétries sur les lignes droites libres.
"""
from __future__ import annotations

import heapq
import math
from typing import List, Optional, Tuple

from core.point import Point
from core.engines.base import Engine
from core.physics import PhysicsConstraints


class JPS(Engine):
    """
    JPS allégé 3D :
    - directions cardinales + diagonales 2D/3D
    - jump le long d'une direction tant que le passage est libre
      et qu'aucun voisin forcé n'apparaît
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
        return "JPS"

    def _h(self, p: Point) -> float:
        dx = abs(self.end_point.x - p.x)
        dy = abs(self.end_point.y - p.y)
        dz = abs(self.end_point.z - p.z)
        return dx + dy + dz + (math.sqrt(3) - 3) * min(dx, dy, dz)

    def _dist(self, a: Point, b: Point) -> float:
        return math.sqrt(
            (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
        )

    def _walkable(self, x, y, z) -> bool:
        return self._is_valid(x, y, z)

    def _jump(
        self, x: int, y: int, z: int, dx: int, dy: int, dz: int
    ) -> Optional[Tuple[int, int, int]]:
        """Avance dans la direction (dx,dy,dz) jusqu'à un jump point ou obstacle."""
        nx, ny, nz = x + dx, y + dy, z + dz
        if not self._walkable(nx, ny, nz):
            return None
        if self.physics is not None:
            if not self.physics.is_valid_move(Point(x, y, z), Point(nx, ny, nz)):
                return None
        if (nx, ny, nz) == (
            self.end_point.x, self.end_point.y, self.end_point.z
        ):
            return (nx, ny, nz)

        # Voisins forcés simplifiés (détection d'obstacle adjacent hors axe)
        if dx != 0 and dy != 0 and dz == 0:
            # diagonale XY
            if (not self._walkable(nx - dx, ny, nz) and self._walkable(nx - dx, ny + dy, nz)) or \
               (not self._walkable(nx, ny - dy, nz) and self._walkable(nx + dx, ny - dy, nz)):
                return (nx, ny, nz)
        elif dx != 0 and dz != 0 and dy == 0:
            if (not self._walkable(nx - dx, ny, nz) and self._walkable(nx - dx, ny, nz + dz)) or \
               (not self._walkable(nx, ny, nz - dz) and self._walkable(nx + dx, ny, nz - dz)):
                return (nx, ny, nz)
        elif dy != 0 and dz != 0 and dx == 0:
            if (not self._walkable(nx, ny - dy, nz) and self._walkable(nx, ny - dy, nz + dz)) or \
               (not self._walkable(nx, ny, nz - dz) and self._walkable(nx, ny + dy, nz - dz)):
                return (nx, ny, nz)
        elif dx != 0 and dy == 0 and dz == 0:
            if (not self._walkable(nx, ny + 1, nz) and self._walkable(nx + dx, ny + 1, nz)) or \
               (not self._walkable(nx, ny - 1, nz) and self._walkable(nx + dx, ny - 1, nz)) or \
               (not self._walkable(nx, ny, nz + 1) and self._walkable(nx + dx, ny, nz + 1)) or \
               (not self._walkable(nx, ny, nz - 1) and self._walkable(nx + dx, ny, nz - 1)):
                return (nx, ny, nz)
        elif dy != 0 and dx == 0 and dz == 0:
            if (not self._walkable(nx + 1, ny, nz) and self._walkable(nx + 1, ny + dy, nz)) or \
               (not self._walkable(nx - 1, ny, nz) and self._walkable(nx - 1, ny + dy, nz)):
                return (nx, ny, nz)
        elif dz != 0 and dx == 0 and dy == 0:
            if (not self._walkable(nx + 1, ny, nz) and self._walkable(nx + 1, ny, nz + dz)) or \
               (not self._walkable(nx - 1, ny, nz) and self._walkable(nx - 1, ny, nz + dz)):
                return (nx, ny, nz)

        # Continuer le saut (récursif sur composantes pour diagonales)
        if dx != 0 and dy != 0 and dz == 0:
            if self._jump(nx, ny, nz, dx, 0, 0) or self._jump(nx, ny, nz, 0, dy, 0):
                return (nx, ny, nz)
        if dx != 0 and dy == 0 and dz != 0:
            if self._jump(nx, ny, nz, dx, 0, 0) or self._jump(nx, ny, nz, 0, 0, dz):
                return (nx, ny, nz)
        if dx == 0 and dy != 0 and dz != 0:
            if self._jump(nx, ny, nz, 0, dy, 0) or self._jump(nx, ny, nz, 0, 0, dz):
                return (nx, ny, nz)
        if dx != 0 and dy != 0 and dz != 0:
            if (self._jump(nx, ny, nz, dx, 0, 0) or
                self._jump(nx, ny, nz, 0, dy, 0) or
                self._jump(nx, ny, nz, 0, 0, dz) or
                self._jump(nx, ny, nz, dx, dy, 0) or
                self._jump(nx, ny, nz, dx, 0, dz) or
                self._jump(nx, ny, nz, 0, dy, dz)):
                return (nx, ny, nz)

        return self._jump(nx, ny, nz, dx, dy, dz)

    def _successors(self, p: Point) -> List[Point]:
        result = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    jp = self._jump(p.x, p.y, p.z, dx, dy, dz)
                    if jp is not None:
                        result.append(Point(*jp))
        return result

    def run(self):
        with self._time_context():
            open_heap = []
            g_score = {self.start_point: 0.0}
            parent = {self.start_point: None}
            closed = set()
            counter = 0
            f0 = self._h(self.start_point)
            heapq.heappush(open_heap, (f0, counter, self.start_point))
            counter += 1
            in_open = {self.start_point}

            while open_heap:
                if not self.observer.is_running():
                    return False
                _, _, p = heapq.heappop(open_heap)
                in_open.discard(p)
                if p in closed:
                    continue
                closed.add(p)
                self.stats.nodes_explored += 1
                self.observer.on_explore(p, alpha=0.1)

                if p.isEgale(self.end_point):
                    path = self._reconstruct(p, parent)
                    self._last_path = path
                    self.stats.success = True
                    self._finalize_path(path)
                    return True

                for n in self._successors(p):
                    if n in closed:
                        continue
                    tentative = g_score[p] + self._dist(p, n)
                    if n not in g_score or tentative < g_score[n]:
                        g_score[n] = tentative
                        parent[n] = p
                        n.parent = p
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
        seen = set()
        while cur is not None:
            path.insert(0, cur)
            if cur.isEgale(self.start_point):
                break
            k = (cur.x, cur.y, cur.z)
            if k in seen:
                break
            seen.add(k)
            cur = parent.get(cur)
        return path
