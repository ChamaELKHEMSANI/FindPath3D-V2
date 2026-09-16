# core/topology.py
"""
Détection de topologie et sélection d'heuristique - Phase 2.
"""
from __future__ import annotations

from typing import Literal

from core.heuristics import (
    Heuristic,
    OctileHeuristic,
    ManhattanHeuristic,
    LandmarkHeuristic,
)

Topology = Literal["open_space", "structured", "dense_maze", "unknown"]


def detect_topology(grid, sample: int = 500) -> Topology:
    """
    Estimation rapide de la topologie.
    - open_space : peu d'obstacles
    - dense_maze : beaucoup d'obstacles / corridors
    - structured : intermédiaire (salles / piliers)
    """
    size = grid.size
    total = size ** 3
    n_obs = len(getattr(grid, "obstacle_point", []) or [])
    # Si DynamicMap, compter via base
    if hasattr(grid, "base"):
        n_obs = len(getattr(grid.base, "obstacle_point", []) or [])
    ratio = n_obs / max(1, total)

    if ratio < 0.08:
        return "open_space"
    if ratio > 0.35:
        return "dense_maze"
    # Échantillon de connectivité locale
    try:
        import random
        rng = random.Random(getattr(grid, "seed", 0) or 0)
        blocked_neighbors = 0
        checks = 0
        for _ in range(min(sample, max(50, total // 20))):
            x = rng.randrange(size)
            y = rng.randrange(size)
            z = rng.randrange(size)
            if grid.IsObstacle(x, y, z):
                continue
            free = 0
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    for dk in (-1, 0, 1):
                        if di == dj == dk == 0:
                            continue
                        nx, ny, nz = x + di, y + dj, z + dk
                        if grid.IsValidPoint(nx, ny, nz) and not grid.IsObstacle(nx, ny, nz):
                            free += 1
            checks += 1
            if free < 6:
                blocked_neighbors += 1
        if checks > 0 and blocked_neighbors / checks > 0.45:
            return "dense_maze"
        if checks > 0 and blocked_neighbors / checks < 0.15:
            return "open_space"
    except Exception:
        pass
    return "structured"


def select_heuristic(grid, topology: Topology | None = None) -> Heuristic:
    """
    Choisit une heuristique adaptée.
    - open_space → Octile (bon marché)
    - dense_maze → Landmark (meilleure borne)
    - structured → Octile par défaut
    """
    topo = topology or detect_topology(grid)
    if topo == "dense_maze" and grid.size <= 40:
        return LandmarkHeuristic(grid)
    if topo == "open_space":
        return OctileHeuristic()
    return OctileHeuristic()


def describe_topology(grid) -> dict:
    topo = detect_topology(grid)
    n_obs = len(getattr(grid, "obstacle_point", []) or [])
    if hasattr(grid, "base"):
        n_obs = len(getattr(grid.base, "obstacle_point", []) or [])
    total = grid.size ** 3
    return {
        "topology": topo,
        "obstacle_ratio": round(n_obs / max(1, total), 4),
        "recommended_heuristic": type(select_heuristic(grid, topo)).__name__,
    }
