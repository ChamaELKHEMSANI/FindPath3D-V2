# core/path_collector.py
"""Observer qui capture le chemin trouvé (Phase 2 raffinement)."""
from __future__ import annotations

from typing import List, Optional

from core.observer import GridObserver
from core.point import Point


class PathCollector(GridObserver):
    """Stocke le dernier chemin notifié par on_path."""

    def __init__(self):
        self.path: List[Point] = []
        self.explored: List[Point] = []
        self._running = True

    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False

    def on_explore(self, point, alpha=0.1):
        self.explored.append(point)

    def on_path(self, path):
        self.path = list(path) if path else []

    def reset(self):
        self.path = []
        self.explored = []
        self._running = True
