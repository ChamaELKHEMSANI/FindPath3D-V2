# bench.py
"""Banc d'essai CLI. Usage : python bench.py"""
import time
from core.point import Point
from core.generators import create_grid, GENERATORS
from core.engines.a_star import AStar
from core.engines.dijkstra import Dijkstra
from core.engines.weighted_a_star import WeightedAStar
from core.engines.best_first import BestFirst
from core.engines.bfs import BFS
from core.engines.d_star import DStar
from core.engines.ara_star import ARAStar
from core.engines.genetic_program import GeneticProgram


def run_one(engine_cls, grid, start, end, **kwargs):
    engine = engine_cls(grid, start, end, **kwargs)
    t0 = time.time()
    ok = engine.run()
    dt = time.time() - t0
    print(f"  {engine.name():22s} "
          f"{'OK' if ok else 'ÉCHEC':6s} "
          f"t={dt:.3f}s  nœuds={engine.stats.nodes_explored:5d}  "
          f"chemin={engine.stats.path_length:4d}  "
          f"coût={engine.stats.path_cost:7.3f}")
    return ok


if __name__ == "__main__":
    size = 10
    obstacle = 5
    seed = 42

    print(f"\n### Taille={size} obstacle={obstacle} seed={seed} ###")

    for gen_name in ('random', 'maze', 'caves', 'rooms', 'pillars'):
        print(f"\n--- Carte : {gen_name} ---")
        grid = create_grid(gen_name, size, obstacle, seed)
        print(f"  obstacles générés : {len(grid.obstacle_point)}")

        start = Point(0, 0, 0, 0)
        end = Point(size - 1, size - 1, size - 1, 0)

        for cls in (AStar, Dijkstra, WeightedAStar, BestFirst, BFS,
                    DStar, ARAStar, GeneticProgram):
            try:
                run_one(cls, grid, start, end)
            except Exception as exc:
                print(f"  {cls.__name__:22s} ERREUR : {exc}")