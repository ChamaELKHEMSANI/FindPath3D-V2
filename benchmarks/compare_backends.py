#!/usr/bin/env python3
"""
Comparaison Grid vs Octree + impact physique / heuristique.
Usage : python benchmarks/compare_backends.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.point import Point
from core.generators import create_grid
from core.engines.a_star import AStar
from core.physics import DEFAULT_PHYSICS
from core.heuristics import OctileHeuristic, LandmarkHeuristic


CONFIGS = [
    # size, obst, generator, seed
    (15, 8,  "random", 42),
    (20, 12, "caves",  42),
    (25, 15, "rooms",  7),
    (30, 20, "random", 99),
]


def run(engine_cls, grid, start, end, **kwargs):
    eng = engine_cls(grid, start, end, **kwargs)
    t0 = time.perf_counter()
    ok = eng.run()
    dt = time.perf_counter() - t0
    return ok, dt, eng.stats.nodes_explored, eng.stats.path_length, eng.stats.path_cost


def main():
    print(f"{'Config':<28} {'Backend':<8} {'Phys':<6} {'Heur':<10} "
          f"{'OK':>3} {'t(s)':>8} {'nodes':>7} {'len':>5}")
    print("-" * 95)

    for size, obst, gen, seed in CONFIGS:
        start = Point(0, 0, 0)
        end = Point(size - 1, size - 1, size - 1)
        label = f"{gen}/{size}³/o={obst}"

        for backend in ("grid", "octree"):
            grid = create_grid(gen, size, obst, seed, backend=backend)
            if grid.IsObstacle(0, 0, 0) or grid.IsObstacle(end.x, end.y, end.z):
                print(f"{label:<28} {backend:<8} {'—':<6} {'—':<10} skip (start/end blocked)")
                continue

            # 1) Baseline
            ok, dt, nodes, plen, _ = run(AStar, grid, start, end)
            print(f"{label:<28} {backend:<8} {'no':<6} {'builtin':<10} "
                  f"{'Y' if ok else 'N':>3} {dt:8.3f} {nodes:7d} {plen:5d}")

            # 2) Physique DEFAULT
            ok, dt, nodes, plen, _ = run(AStar, grid, start, end, physics=DEFAULT_PHYSICS)
            print(f"{label:<28} {backend:<8} {'DEF':<6} {'builtin':<10} "
                  f"{'Y' if ok else 'N':>3} {dt:8.3f} {nodes:7d} {plen:5d}")

            # 3) Landmark (peut être lent à pré-calculer sur grandes grilles)
            if size <= 25:
                t_pre = time.perf_counter()
                lm = LandmarkHeuristic(grid)
                pre = time.perf_counter() - t_pre
                ok, dt, nodes, plen, _ = run(AStar, grid, start, end, heuristic=lm)
                print(f"{label:<28} {backend:<8} {'no':<6} {'landmark':<10} "
                      f"{'Y' if ok else 'N':>3} {dt:8.3f} {nodes:7d} {plen:5d}  "
                      f"(pre={pre:.2f}s)")

            if backend == "octree":
                print(f"  → Octree stats: {grid.stats()}")

        print()


if __name__ == "__main__":
    main()
