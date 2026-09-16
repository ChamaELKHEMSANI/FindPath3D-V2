# core/heuristics.py
"""
Heuristiques injectables (Phase 1).
Permet de remplacer l'heuristique codée en dur par un objet pluggable.
"""
from __future__ import annotations

import heapq
import math
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from core.point import Point


class Heuristic(ABC):
    """Interface commune pour toutes les heuristiques."""

    @abstractmethod
    def estimate(self, from_point: Point, to_point: Point) -> float:
        """Estimation du coût restant (doit être admissible pour optimalité)."""

    def __repr__(self) -> str:
        return self.__class__.__name__


class OctileHeuristic(Heuristic):
    """Heuristique octile 3D classique (admissible pour distance euclidienne)."""

    def estimate(self, from_point: Point, to_point: Point) -> float:
        dx = abs(to_point.x - from_point.x)
        dy = abs(to_point.y - from_point.y)
        dz = abs(to_point.z - from_point.z)
        return dx + dy + dz + (math.sqrt(3) - 3) * min(dx, dy, dz)


class ManhattanHeuristic(Heuristic):
    """Distance de Manhattan pure."""

    def estimate(self, from_point: Point, to_point: Point) -> float:
        return (
            abs(to_point.x - from_point.x)
            + abs(to_point.y - from_point.y)
            + abs(to_point.z - from_point.z)
        )


class EuclideanHeuristic(Heuristic):
    """Distance euclidienne pure."""

    def estimate(self, from_point: Point, to_point: Point) -> float:
        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y
        dz = to_point.z - from_point.z
        return math.sqrt(dx * dx + dy * dy + dz * dz)


class LandmarkHeuristic(Heuristic):
    """
    Heuristique de landmarks (style ALT).
    Pré-calcule les distances exactes (Dijkstra 26-connexité) depuis
    quelques points stratégiques et utilise l'inégalité triangulaire.
    """

    def __init__(
        self,
        grid,
        landmarks: Optional[List[Tuple[int, int, int]]] = None,
        num_extra: int = 0,
    ):
        self.grid = grid
        self.size = grid.size
        if landmarks is None:
            s = self.size - 1
            landmarks = [
                (0, 0, 0),
                (s, 0, 0),
                (0, s, 0),
                (0, 0, s),
                (s, s, s),
            ]
            # Optionnel : quelques points au centre des faces
            if num_extra > 0 and s > 4:
                mid = s // 2
                extras = [
                    (mid, mid, 0),
                    (mid, mid, s),
                    (mid, 0, mid),
                    (0, mid, mid),
                ]
                landmarks.extend(extras[:num_extra])
        self.landmarks = []
        for lm in landmarks:
            p = Point(*lm)
            if grid.IsValidPoint(p.x, p.y, p.z) and not grid.IsObstacle(p.x, p.y, p.z):
                self.landmarks.append(p)
        if not self.landmarks:
            # Fallback : origin si libre
            self.landmarks = [Point(0, 0, 0)]
        self.dist: List[Dict[Tuple[int, int, int], float]] = []
        self._precompute()

    def _neighbors(self, x: int, y: int, z: int):
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                for dk in (-1, 0, 1):
                    if di == 0 and dj == 0 and dk == 0:
                        continue
                    nx, ny, nz = x + di, y + dj, z + dk
                    if not self.grid.IsValidPoint(nx, ny, nz):
                        continue
                    if self.grid.IsObstacle(nx, ny, nz):
                        continue
                    step = math.sqrt(di * di + dj * dj + dk * dk)
                    yield nx, ny, nz, step

    def _dijkstra_from(self, origin: Point) -> Dict[Tuple[int, int, int], float]:
        """Dijkstra exact (heap) depuis origin."""
        dmap: Dict[Tuple[int, int, int], float] = {}
        key0 = (origin.x, origin.y, origin.z)
        if self.grid.IsObstacle(*key0):
            return dmap
        dmap[key0] = 0.0
        heap = [(0.0, origin.x, origin.y, origin.z)]
        while heap:
            dist, x, y, z = heapq.heappop(heap)
            if dist > dmap.get((x, y, z), float("inf")):
                continue
            for nx, ny, nz, step in self._neighbors(x, y, z):
                nd = dist + step
                nkey = (nx, ny, nz)
                if nd < dmap.get(nkey, float("inf")):
                    dmap[nkey] = nd
                    heapq.heappush(heap, (nd, nx, ny, nz))
        return dmap

    def _precompute(self):
        for lm in self.landmarks:
            self.dist.append(self._dijkstra_from(lm))

    def estimate(self, from_point: Point, to_point: Point) -> float:
        """max |d(L, from) - d(L, to)|  (borne inférieure admissible)."""
        best = 0.0
        fkey = (from_point.x, from_point.y, from_point.z)
        tkey = (to_point.x, to_point.y, to_point.z)
        for dmap in self.dist:
            if fkey not in dmap or tkey not in dmap:
                continue
            est = abs(dmap[fkey] - dmap[tkey])
            if est > best:
                best = est
        if best == 0.0:
            return OctileHeuristic().estimate(from_point, to_point)
        return best

    def __repr__(self) -> str:
        return f"LandmarkHeuristic(L={len(self.landmarks)})"
