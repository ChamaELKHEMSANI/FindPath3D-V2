# core/engines/dynamic_pathfinder.py
"""
Replanning incrémental - Phase 2 (raffiné).
Capture robuste du chemin via PathCollector.
"""
from __future__ import annotations

from typing import List, Optional

from core.point import Point
from core.engines.a_star import AStar
from core.engines.time_astar import TimeAStar
from core.path_collector import PathCollector
from core.physics import PhysicsConstraints


class DynamicPathfinder:
    """
    1. Plan initial (A* ou TimeA*).
    2. step() avance le long du chemin.
    3. Si obstacle dynamique → replan depuis la position courante.
    """

    def __init__(
        self,
        grid,
        start: Point,
        end: Point,
        physics: Optional[PhysicsConstraints] = None,
        use_time: bool = False,
        lookahead: int = 0,
        t_horizon: int = 150,
    ):
        self.grid = grid
        self.start = start
        self.end = end
        self.physics = physics
        self.use_time = use_time
        self.lookahead = lookahead
        self.t_horizon = t_horizon

        self.path: List[Point] = []
        self.t = 0
        self.index = 0
        self.replan_count = 0
        self.last_stats = None
        self.fuel_left: float = getattr(grid, "initial_fuel", 1e9)

    def _run_engine(self, start: Point, goal: Point, t_start: int = 0):
        collector = PathCollector()
        if self.use_time:
            eng = TimeAStar(
                self.grid, start, goal,
                observer=collector,
                physics=self.physics,
                t_start=t_start,
                t_horizon=self.t_horizon,
            )
        else:
            eng = AStar(
                self.grid, start, goal,
                observer=collector,
                physics=self.physics,
            )
        ok = eng.run()
        self.last_stats = eng.stats
        path = list(getattr(eng, "_last_path", None) or collector.path or [])
        return ok, path, eng

    def plan_initial(self) -> bool:
        ok, path, _ = self._run_engine(self.start, self.end, t_start=0)
        if ok and path:
            self.path = path
            self.index = 0
            self.t = 0
            self.fuel_left = getattr(self.grid, "initial_fuel", 1e9)
            return True
        self.path = []
        return False

    def current_position(self) -> Optional[Point]:
        if not self.path or self.index >= len(self.path):
            return None
        return self.path[self.index]

    def is_finished(self) -> bool:
        return bool(self.path) and self.index >= len(self.path) - 1

    def _blocked_at(self, p: Point, t: int) -> bool:
        try:
            return self.grid.IsObstacle(p.x, p.y, p.z, t)
        except TypeError:
            return self.grid.IsObstacle(p.x, p.y, p.z)

    def path_valid_from_here(self) -> bool:
        if not self.path or self.index >= len(self.path):
            return False
        t = self.t
        for p in self.path[self.index:]:
            if self._blocked_at(p, t):
                return False
            t += 1
        return True

    def on_grid_change(self, changed_positions=None) -> bool:
        if self.path_valid_from_here():
            return False
        return self.replan()

    def replan(self) -> bool:
        cur = self.current_position() or self.start
        goal = self.end
        if (
            self.lookahead > 0
            and self.path
            and self.index + self.lookahead < len(self.path)
        ):
            goal = self.path[min(len(self.path) - 1, self.index + self.lookahead)]

        ok, new_path, _ = self._run_engine(cur, goal, t_start=self.t)
        self.replan_count += 1
        if not ok or not new_path:
            return False
        # Fusion : conserver le préfixe déjà parcouru
        self.path = self.path[: self.index] + new_path
        return True

    def step(self) -> Optional[Point]:
        if not self.path or self.index >= len(self.path) - 1:
            return self.current_position()

        nxt = self.path[self.index + 1]
        t_next = self.t + 1

        if self._blocked_at(nxt, t_next):
            if not self.replan():
                return self.current_position()
            if self.index >= len(self.path) - 1:
                return self.current_position()
            nxt = self.path[self.index + 1]

        # Consommation carburant (1 unité / pas)
        self.fuel_left -= 1.0
        # Recharge éventuelle
        if hasattr(self.grid, "fuel_at"):
            gain = self.grid.fuel_at(nxt.x, nxt.y, nxt.z)
            if gain > 0:
                self.fuel_left += gain

        if self.fuel_left < 0:
            # En panne : tente un replan vers une station si connue
            self.replan_count += 1
            return self.current_position()

        self.index += 1
        self.t += 1
        return self.path[self.index]

    def run_until_done(self, max_steps: int = 500) -> dict:
        """Exécute step() jusqu'à la fin ou max_steps (utile en tests)."""
        if not self.path:
            self.plan_initial()
        steps = 0
        while not self.is_finished() and steps < max_steps:
            self.step()
            steps += 1
        return self.summary()

    def summary(self) -> dict:
        return {
            "path_len": len(self.path),
            "index": self.index,
            "t": self.t,
            "replan_count": self.replan_count,
            "finished": self.is_finished(),
            "fuel_left": round(self.fuel_left, 2) if self.fuel_left < 1e8 else "∞",
        }
