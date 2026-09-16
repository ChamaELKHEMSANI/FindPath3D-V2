# core/engines/time_astar.py
"""
A* spatio-temporel (Phase 2).
État = (x, y, z, t). Compatible avec DynamicMap.
"""
from __future__ import annotations

import heapq
import math
from typing import Optional, Tuple

from core.point import Point
from core.engines.base import Engine
from core.physics import PhysicsConstraints


class TimePoint:
    """État spatio-temporel."""
    __slots__ = ("x", "y", "z", "t", "cost", "parent")

    def __init__(self, x, y, z, t=0, cost=float("inf")):
        self.x = x
        self.y = y
        self.z = z
        self.t = t
        self.cost = cost
        self.parent = None

    def key(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.z, self.t)

    def pos_key(self) -> Tuple[int, int, int]:
        return (self.x, self.y, self.z)

    def isEgale(self, other) -> bool:
        # Pour l'objectif on ignore souvent le temps exact
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash(self.key())

    def __eq__(self, other):
        return isinstance(other, TimePoint) and self.key() == other.key()

    def __lt__(self, other):
        return self.key() < other.key()

    def __repr__(self):
        return f"TP({self.x},{self.y},{self.z},t={self.t})"


class TimeAStar(Engine):
    """
    A* dans l'espace (x,y,z,t).
    - wait autorisé (rester sur place = +1 temps)
    - horizon temporel max pour éviter explosion d'état
    """

    def __init__(
        self,
        map_obj,
        start_point,
        end_point,
        observer=None,
        physics: Optional[PhysicsConstraints] = None,
        t_start: int = 0,
        t_horizon: int = 200,
        allow_wait: bool = True,
    ):
        super().__init__(map_obj, start_point, end_point, observer, physics)
        self.t_start = t_start
        self.t_horizon = t_horizon
        self.allow_wait = allow_wait

    def name(self) -> str:
        return "TimeA*"

    def _is_blocked(self, x, y, z, t) -> bool:
        # DynamicMap a IsObstacle(..., t=)
        if hasattr(self.map, "IsObstacle"):
            try:
                return self.map.IsObstacle(x, y, z, t)
            except TypeError:
                return self.map.IsObstacle(x, y, z)
        return False

    def _h(self, p: TimePoint) -> float:
        dx = abs(self.end_point.x - p.x)
        dy = abs(self.end_point.y - p.y)
        dz = abs(self.end_point.z - p.z)
        return dx + dy + dz + (math.sqrt(3) - 3) * min(dx, dy, dz)

    def _neighbors(self, p: TimePoint):
        result = []
        # Mouvements 26-connexité + option wait
        deltas = [
            (i, j, k)
            for i in (-1, 0, 1)
            for j in (-1, 0, 1)
            for k in (-1, 0, 1)
            if not (i == 0 and j == 0 and k == 0)
        ]
        if self.allow_wait:
            deltas.append((0, 0, 0))  # wait

        for di, dj, dk in deltas:
            nx, ny, nz = p.x + di, p.y + dj, p.z + dk
            nt = p.t + 1
            if nt > self.t_start + self.t_horizon:
                continue
            if not self.map.IsValidPoint(nx, ny, nz):
                continue
            if self._is_blocked(nx, ny, nz, nt):
                continue
            # Physique (ignore le temps)
            if self.physics is not None and (di, dj, dk) != (0, 0, 0):
                from_p = Point(p.x, p.y, p.z)
                to_p = Point(nx, ny, nz)
                if not self.physics.is_valid_move(from_p, to_p, None):
                    continue
            step = math.sqrt(di * di + dj * dj + dk * dk) if (di or dj or dk) else 1.0
            result.append((nx, ny, nz, nt, step))
        return result

    def run(self):
        with self._time_context():
            start = TimePoint(
                self.start_point.x, self.start_point.y, self.start_point.z,
                self.t_start, cost=0.0,
            )
            open_heap = []
            g_score = {start.key(): 0.0}
            counter = 0
            heapq.heappush(open_heap, (self._h(start), counter, start))
            counter += 1
            closed = set()

            while open_heap:
                if not self.observer.is_running():
                    return False
                _, _, p = heapq.heappop(open_heap)
                if p.key() in closed:
                    continue
                closed.add(p.key())
                self.stats.nodes_explored += 1
                self.observer.on_explore(Point(p.x, p.y, p.z), alpha=0.08)

                if p.isEgale(self.end_point):
                    path = self._build_path(p)
                    self._last_path = path
                    self.stats.success = True
                    self._finalize_path(path)
                    self.stats.extra["t_arrival"] = p.t
                    return True

                for nx, ny, nz, nt, step in self._neighbors(p):
                    nkey = (nx, ny, nz, nt)
                    if nkey in closed:
                        continue
                    tentative = g_score[p.key()] + step
                    if tentative < g_score.get(nkey, float("inf")):
                        g_score[nkey] = tentative
                        n = TimePoint(nx, ny, nz, nt, cost=tentative + self._h(
                            TimePoint(nx, ny, nz, nt)
                        ))
                        n.parent = p
                        heapq.heappush(open_heap, (n.cost, counter, n))
                        counter += 1

            self.stats.success = False
            return False

    def _build_path(self, p: TimePoint):
        path = []
        while p is not None:
            path.insert(0, Point(p.x, p.y, p.z))
            if (p.x, p.y, p.z) == (
                self.start_point.x, self.start_point.y, self.start_point.z
            ) and p.t == self.t_start:
                break
            p = p.parent
        return path
