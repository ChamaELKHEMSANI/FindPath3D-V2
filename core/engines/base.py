# core/engines/base.py
"""Interface abstraite Engine + helpers communs (Phase 1 : support physique)."""
import math
import time
from abc import ABC, abstractmethod
from typing import Optional

from core.observer import GridObserver
from core.stats import EngineStats
from core.physics import PhysicsConstraints


class Engine(ABC):
    def __init__(
        self,
        map_obj,
        start_point,
        end_point,
        observer=None,
        physics: Optional[PhysicsConstraints] = None,
    ):
        self.map = map_obj
        self.start_point = start_point
        self.end_point = end_point
        self.observer = observer or GridObserver()
        # Physique optionnelle (None = comportement historique 26-connexité pure)
        self.physics = physics
        self.stats = EngineStats(
            engine=self.name(),
            size=map_obj.size,
            obstacle=map_obj.obstacle,
            seed=getattr(map_obj, "seed", None),
        )
        if physics is not None:
            self.stats.extra["physics"] = repr(physics)

    @abstractmethod
    def name(self) -> str:
        """Nom court affiché dans l'UI."""

    @abstractmethod
    def run(self) -> bool:
        """Exécute l'algorithme. Retourne True si un chemin a été trouvé."""

    # ------------------------------------------------------------------
    # Helpers communs aux sous-classes
    # ------------------------------------------------------------------
    def _is_valid(self, x, y, z):
        return self.map.IsValidPoint(x, y, z) and not self.map.IsObstacle(x, y, z)

    def _neighbors(self, p, previous_point=None):
        """
        Retourne les voisins 26-connexité valides.
        Si self.physics est défini, applique les contraintes physiques.
        """
        from core.point import Point

        result = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    nx, ny, nz = p.x + i, p.y + j, p.z + k
                    if not self._is_valid(nx, ny, nz):
                        continue
                    candidate = Point(nx, ny, nz)
                    if self.physics is not None:
                        if not self.physics.is_valid_move(p, candidate, previous_point):
                            continue
                    result.append(candidate)
        return result

    def _finalize_path(self, path):
        """Remplit stats.path_length et stats.path_cost puis notifie l'observer."""
        self.stats.path_length = len(path)
        self.stats.path_cost = self._path_cost(path)
        self.observer.on_path(path)
        self.observer.on_save_image()

    @staticmethod
    def _path_cost(path):
        """Coût = distance euclidienne cumulée entre voxels consécutifs."""
        total = 0.0
        for a, b in zip(path, path[1:]):
            total += math.sqrt(
                (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
            )
        return total

    def _time_context(self):
        class _Timer:
            def __enter__(_self):
                _self.t0 = time.time()

            def __exit__(_self, *a):
                self.stats.elapsed = time.time() - _self.t0

        return _Timer()
