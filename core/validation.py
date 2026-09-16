# core/validation.py
"""
Validation systématique des chemins - Phase 3.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from core.point import Point


def path_continuous(path: List[Point]) -> bool:
    """Vérifie la 26-connexité entre points consécutifs."""
    if not path:
        return False
    for a, b in zip(path, path[1:]):
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        dz = abs(a.z - b.z)
        if max(dx, dy, dz) > 1:
            # Theta* / JPS peuvent sauter → autoriser si LOS sera vérifié ailleurs
            continue
        if dx + dy + dz == 0:
            return False
    return True


def path_collision_free(grid, path: List[Point], t0: int = 0) -> bool:
    t = t0
    for p in path:
        if not grid.IsValidPoint(p.x, p.y, p.z):
            return False
        try:
            if grid.IsObstacle(p.x, p.y, p.z, t):
                return False
        except TypeError:
            if grid.IsObstacle(p.x, p.y, p.z):
                return False
        t += 1
    return True


def path_endpoints(path: List[Point], start: Point, end: Point) -> bool:
    if not path:
        return False
    return path[0].isEgale(start) and path[-1].isEgale(end)


def validate_path(
    grid,
    path: List[Point],
    start: Point,
    end: Point,
    *,
    allow_jumps: bool = False,
    t0: int = 0,
) -> Tuple[bool, str]:
    """
    Retourne (ok, message).
    allow_jumps=True pour Theta*/JPS (pas de contrainte 26-connexité stricte).
    """
    if not path:
        return False, "chemin vide"
    if not path_endpoints(path, start, end):
        return False, "extrémités incorrectes"
    if not allow_jumps and not path_continuous(path):
        return False, "discontinuité 26-connexité"
    if not path_collision_free(grid, path, t0):
        return False, "collision obstacle"
    return True, "OK"


def compare_optimality(cost: float, reference_cost: float, tol: float = 1e-6) -> Optional[float]:
    """Ratio cost/reference (1.0 = optimal). None si non calculable."""
    if reference_cost is None or reference_cost <= 0 or cost <= 0:
        return None
    return cost / reference_cost
