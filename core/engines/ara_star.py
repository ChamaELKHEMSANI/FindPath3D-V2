# core/engines/ara_star.py
"""ARA* — Anytime Repairing A*."""
import heapq
import numpy as np

from core.point import Point
from core.engines.base import Engine


class ARAStar(Engine):

    def __init__(self, map_obj, start_point, end_point, observer=None,
                 epsilon=2.5, epsilon_decay=0.9):
        super().__init__(map_obj, start_point, end_point, observer)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

    def name(self) -> str:
        return "ARA*"

    def _h(self, p):
        return float(np.sqrt((p.x - self.end_point.x) ** 2
                             + (p.y - self.end_point.y) ** 2
                             + (p.z - self.end_point.z) ** 2))

    def _cost(self, a, b):
        return float(np.sqrt((a.x - b.x) ** 2
                             + (a.y - b.y) ** 2
                             + (a.z - b.z) ** 2))

    def _key(self, p):
        g = self.g.get(p, float('inf'))
        rhs = self.rhs.get(p, float('inf'))
        return (min(g, rhs) + self.epsilon * self._h(p), min(g, rhs))

    def run(self):
        with self._time_context():
            self.epsilon = self._initial_epsilon
            self._initialize()

            if not self._improve():
                return False

            while self.epsilon > 1:
                self.epsilon *= self.epsilon_decay
                if self.epsilon < 1:
                    self.epsilon = 1.0
                if not self._improve():
                    return False
                if not self.observer.is_running():
                    return False

            path = self._reconstruct()
            if path is None:
                self.stats.success = False
                return False

            self.stats.path_length = len(path)
            self.stats.success = True
            self.observer.on_path(path)
            self.observer.on_save_image()
            return True

    def _initialize(self):
        self._initial_epsilon = self.epsilon
        self.g = {self.start_point: float('inf'), self.end_point: float('inf')}
        self.rhs = {self.start_point: 0.0, self.end_point: float('inf')}
        self.heap = []
        self.in_open = set()
        self.close_set = set()
        self.counter = 0
        self._push(self.start_point)

    def _push(self, p):
        self.counter += 1
        heapq.heappush(self.heap, (self._key(p), self.counter, p))
        self.in_open.add(p)

    def _update_vertex(self, p):
        if p != self.start_point:
            neighbors = self._neighbors(p)
            if neighbors:
                self.rhs[p] = min(self.g.get(n, float('inf')) + self._cost(p, n)
                                  for n in neighbors)
            else:
                self.rhs[p] = float('inf')

        self.in_open.discard(p)
        if self.g.get(p, float('inf')) != self.rhs.get(p, float('inf')):
            self._push(p)

    def _improve(self):
        max_iter = self.map.size ** 3 * 4
        it = 0
        while self.heap:
            if not self.observer.is_running():
                return False
            if it > max_iter:
                self.observer.on_log("ARA* : dépassement du nombre max d'itérations")
                return False
            it += 1

            top_key = self.heap[0][0]
            end_key = self._key(self.end_point)
            if not (top_key < end_key
                    or self.rhs.get(self.end_point, float('inf'))
                       > self.g.get(self.end_point, float('inf'))):
                break

            _, _, p = heapq.heappop(self.heap)
            self.in_open.discard(p)

            self.stats.nodes_explored += 1
            self.observer.on_explore(p, alpha=0.05)

            if self.g.get(p, float('inf')) > self.rhs.get(p, float('inf')):
                self.g[p] = self.rhs[p]
                self.close_set.add(p)
                for n in self._neighbors(p):
                    self._update_vertex(n)
            else:
                self.g[p] = float('inf')
                self._update_vertex(p)
                for n in self._neighbors(p):
                    self._update_vertex(n)

        return True

    def _reconstruct(self):
        if self.rhs.get(self.end_point, float('inf')) == float('inf'):
            return None

        max_steps = self.map.size ** 3
        path = []
        p = self.end_point
        visited = set()
        steps = 0

        while p != self.start_point and steps < max_steps:
            path.append(p)
            visited.add(p)
            neighbors = [n for n in self._neighbors(p) if n not in visited]
            if not neighbors:
                return None
            p = min(neighbors,
                    key=lambda n: self.g.get(n, float('inf')) + self._cost(p, n))
            steps += 1

        if p != self.start_point:
            return None
        path.append(self.start_point)
        path.reverse()
        return path