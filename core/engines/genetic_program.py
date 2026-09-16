# core/engines/genetic_program.py
"""Algorithme génétique pour la recherche de chemin 3D."""
import random

from core.point import Point
from core.engines.base import Engine


class GeneticProgram(Engine):

    def __init__(self, map_obj, start_point, end_point, observer=None,
                 population_size=10, generations=50, mutation_rate=0.3,
                 seed=None):
        super().__init__(map_obj, start_point, end_point, observer)
        self.population_size = max(2, population_size)
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.rng = random.Random(seed)
        self.population = []

    def name(self) -> str:
        return "GeneticProgram"

    def run(self):
        with self._time_context():
            self.population = []
            for _ in range(self.population_size):
                if not self.observer.is_running():
                    return False
                self.population.append(self._random_path())

            for _ in range(self.generations):
                if not self.observer.is_running():
                    return False
                self.population = self._select_best()
                new_pop = list(self.population)
                parents = self.population if len(self.population) >= 2 else self.population * 2
                while len(new_pop) < self.population_size:
                    if not self.observer.is_running():
                        return False
                    p1, p2 = self.rng.sample(parents, 2)
                    child = self._mutate(self._crossover(p1, p2))
                    new_pop.append(child)
                self.population = new_pop
                self.stats.nodes_explored += self.population_size

                best = min(self.population, key=self._fitness)
                if best:
                    self.observer.on_explore(best[-1], alpha=0.5)

                if self._is_end(best[-1]):
                    break

            best_path = min(self.population, key=self._fitness)
            if not self._is_end(best_path[-1]):
                self.stats.success = False
                return False

            self.stats.path_length = len(best_path)
            self.stats.success = True
            self.observer.on_path(best_path)
            self.observer.on_save_image()
            return True

    # ---- Génération ----
    def _random_path(self, max_len=None):
        max_len = max_len or self.map.size ** 2
        path = [self.start_point]
        current = self.start_point
        while not self._is_end(current) and len(path) < max_len:
            moves = self._valid_moves(path, current)
            if not moves:
                break
            if len(moves) > 3 and self.rng.random() < 0.7:
                current = self.rng.choice(moves[:3])
            else:
                current = self.rng.choice(moves)
            path.append(current)
        return path

    def _valid_moves(self, path, point):
        moves = []
        for i in (-1, 0, 1):
            for j in (-1, 0, 1):
                for k in (-1, 0, 1):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    nx, ny, nz = point.x + i, point.y + j, point.z + k
                    if not self._is_valid(nx, ny, nz):
                        continue
                    if any(q.isEgaleCoord(nx, ny, nz) for q in path):
                        continue
                    moves.append(Point(nx, ny, nz))
        moves.sort(key=lambda p: p.distance(self.end_point))
        return moves

    # ---- Fitness / sélection ----
    def _fitness(self, path):
        if not path:
            return float('inf')
        last = path[-1]
        if self._is_end(last):
            return len(path)
        return last.distance(self.end_point) * 10 + len(path)

    def _select_best(self):
        self.population.sort(key=self._fitness)
        return self.population[:max(2, self.population_size // 2)]

    # ---- Opérateurs ----
    def _crossover(self, p1, p2):
        m = min(len(p1), len(p2))
        if m < 2:
            return p1[:]
        cut = self.rng.randint(1, m - 1)
        return p1[:cut] + p2[cut:]

    def _mutate(self, path):
        if len(path) < 3 or self.rng.random() >= self.mutation_rate:
            return path
        idx = self.rng.randint(1, len(path) - 2)
        moves = self._valid_moves(path, path[idx])
        if moves:
            path[idx] = self.rng.choice(moves)
        return path

    def _is_end(self, p):
        return p.isEgale(self.end_point)