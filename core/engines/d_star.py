# core/engines/d_star.py
"""D* Lite — recherche adaptative."""
import heapq
import numpy as np

from core.point import Point
from core.engines.base import Engine


class DStar(Engine):

    def name(self) -> str:
        return "D*"

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
        return (min(g, rhs) + self._h(p), min(g, rhs))

    def run(self):
        with self._time_context():
            self.g = {self.end_point: float('inf')}
            self.rhs = {self.end_point: 0.0}
            heap = []
            in_open = set()
            counter = [0]

            def push(p):
                counter[0] += 1
                heapq.heappush(heap, (self._key(p), counter[0], p))
                in_open.add(p)

            push(self.end_point)

            max_iter = self.map.size ** 3 * 4
            it = 0

            while heap and (self.g.get(self.start_point, float('inf'))
                            != self.rhs.get(self.start_point, float('inf'))):
                if not self.observer.is_running():
                    return False
                it += 1
                if it > max_iter:
                    self.observer.on_log("D* : dépassement du nombre max d'itérations")
                    return False

                _, _, p = heapq.heappop(heap)
                in_open.discard(p)

                self.stats.nodes_explored += 1
                self.observer.on_explore(p, alpha=0.05)

                if self.g.get(p, float('inf')) > self.rhs.get(p, float('inf')):
                    self.g[p] = self.rhs[p]
                else:
                    self.g[p] = float('inf')
                    self._update_vertex(p, push)

                for n in self._neighbors(p):
                    self._update_vertex(n, push)

            path = self._reconstruct()
            if path is None:
                self.stats.success = False
                return False

            self.stats.path_length = len(path)
            self.stats.success = True
            self.observer.on_path(path)
            self.observer.on_save_image()
            return True

    def _update_vertex(self, p, push):
        if p != self.end_point:
            neighbors = self._neighbors(p)
            if neighbors:
                self.rhs[p] = min(self.g.get(n, float('inf')) + self._cost(p, n)
                                  for n in neighbors)
            else:
                self.rhs[p] = float('inf')
        if self.g.get(p, float('inf')) != self.rhs.get(p, float('inf')):
            push(p)

    def _reconstruct(self):
        max_steps = self.map.size ** 3
        path = []
        p = self.start_point
        visited = set()
        steps = 0

        while p != self.end_point and steps < max_steps:
            path.append(p)
            visited.add(p)
            neighbors = [n for n in self._neighbors(p) if n not in visited]
            if not neighbors:
                return None
            p = min(neighbors, key=lambda n: self.g.get(n, float('inf')))
            steps += 1

        if p != self.end_point:
            return None
        path.append(self.end_point)
        return path