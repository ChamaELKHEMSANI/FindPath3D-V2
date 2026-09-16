#!/usr/bin/env python3
"""
Suite de benchmarks de référence - Phase 0.
Usage :
    python -m benchmarks.reference_suite
    ou depuis la racine : python benchmarks/reference_suite.py

Produit un CSV (stdout ou fichier) avec mean/std sur plusieurs runs.
Les configs couvrent Easy / Medium / Hard. Les tailles extrêmes (200+) 
sont commentées pour rester raisonnables sur machine de développement.
"""
from __future__ import annotations

import csv
import statistics
import sys
import time
from pathlib import Path

# Permettre l'exécution depuis la racine ou depuis benchmarks/
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.point import Point
from core.generators import create_grid
from core.engines.a_star import AStar
from core.engines.dijkstra import Dijkstra
from core.engines.weighted_a_star import WeightedAStar
from core.engines.best_first import BestFirst
from core.engines.bfs import BFS
from core.engines.d_star import DStar
from core.engines.ara_star import ARAStar
from core.engines.genetic_program import GeneticProgram


# ---------------------------------------------------------------------------
# Configurations de référence
# Format : (size, obstacle_count, generator, seed, difficulty_label)
# ---------------------------------------------------------------------------
BENCHMARK_CONFIGS = [
    # EASY
    (10, 5,  "random",  42, "easy"),
    (10, 3,  "maze",    42, "easy"),
    (10, 8,  "rooms",   42, "easy"),

    # MEDIUM
    (20, 15, "random",  42, "medium"),
    (20, 10, "caves",   42, "medium"),
    (20, 12, "pillars", 42, "medium"),

    # HARD (taille encore raisonnable pour Phase 0)
    (30, 25, "maze",    42, "hard"),
    (30, 40, "random",  42, "hard"),
    (40, 30, "rooms",   42, "hard"),

    # EXTREME - à activer une fois les optimisations en place
    # (50, 50, "random", 42, "extreme"),
    # (100, 80, "maze",  42, "extreme"),
]

ENGINES = [
    ("A*", AStar),
    ("Dijkstra", Dijkstra),
    ("WeightedA*", WeightedAStar),
    ("Best-First", BestFirst),
    ("BFS", BFS),
    ("D*", DStar),
    ("ARA*", ARAStar),
    ("Genetic", GeneticProgram),
]

REPEATS = 3   # nombre de runs par (config × engine) pour moyenne/écart-type


def run_single(engine_cls, grid, start, end, **kwargs):
    engine = engine_cls(grid, start, end, **kwargs)
    t0 = time.perf_counter()
    ok = engine.run()
    dt = time.perf_counter() - t0
    return {
        "success": ok,
        "time": dt,
        "nodes": engine.stats.nodes_explored,
        "path_length": engine.stats.path_length,
        "path_cost": engine.stats.path_cost,
    }


def main(output_csv: str | None = None):
    rows = []
    print(f"{'Config':<35} {'Engine':<14} {'OK':>3} {'t_mean':>8} {'t_std':>7} "
          f"{'nodes':>7} {'len':>5} {'cost':>8}")
    print("-" * 100)

    for size, obst, gen, seed, label in BENCHMARK_CONFIGS:
        grid = create_grid(gen, size, obst, seed)
        start = Point(0, 0, 0)
        end = Point(size - 1, size - 1, size - 1)
        n_obst = len(grid.obstacle_point)
        cfg_str = f"{label}/{gen}/{size}³/o={obst}(n={n_obst})"

        for eng_name, eng_cls in ENGINES:
            times, nodes, lengths, costs, successes = [], [], [], [], []
            for _ in range(REPEATS):
                try:
                    res = run_single(eng_cls, grid, start, end)
                    times.append(res["time"])
                    nodes.append(res["nodes"])
                    lengths.append(res["path_length"])
                    costs.append(res["path_cost"])
                    successes.append(res["success"])
                except Exception as exc:
                    print(f"  ERREUR {eng_name} sur {cfg_str}: {exc}")
                    times.append(float("nan"))
                    nodes.append(0)
                    lengths.append(0)
                    costs.append(0.0)
                    successes.append(False)

            t_mean = statistics.mean(times) if times else float("nan")
            t_std = statistics.stdev(times) if len(times) > 1 else 0.0
            n_mean = statistics.mean(nodes) if nodes else 0
            len_mean = statistics.mean(lengths) if lengths else 0
            cost_mean = statistics.mean(costs) if costs else 0.0
            ok_ratio = sum(1 for s in successes if s) / len(successes)

            print(f"{cfg_str:<35} {eng_name:<14} {ok_ratio:3.0%} "
                  f"{t_mean:8.3f} {t_std:7.3f} {n_mean:7.0f} "
                  f"{len_mean:5.0f} {cost_mean:8.2f}")

            rows.append({
                "difficulty": label,
                "generator": gen,
                "size": size,
                "obstacle_param": obst,
                "n_obstacles": n_obst,
                "seed": seed,
                "engine": eng_name,
                "ok_ratio": ok_ratio,
                "time_mean": t_mean,
                "time_std": t_std,
                "nodes_mean": n_mean,
                "path_length_mean": len_mean,
                "path_cost_mean": cost_mean,
                "repeats": REPEATS,
            })

    if output_csv:
        out = Path(output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nCSV écrit : {out}")

    return rows


if __name__ == "__main__":
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = str(ROOT / "benchmarks" / "baseline_results.csv")
    main(csv_path)
