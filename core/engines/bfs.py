# core/engines/bfs.py
"""BFS : parcours en largeur. Optimal en nombre de pas, pas en distance."""
from collections import deque
from typing import Optional

from core.engines.base import Engine
from core.physics import PhysicsConstraints


class BFS(Engine):
    def __init__(self, map_obj, start_point, end_point, observer=None,
                 physics: Optional[PhysicsConstraints] = None):
        super().__init__(map_obj, start_point, end_point, observer, physics)

    def name(self) -> str:
        return "BFS"

    def _build_path(self, p):
        path = []
        while p is not None:
            path.insert(0, p)
            if p.isEgale(self.start_point):
                break
            p = p.parent
        return path

    def run(self):
        with self._time_context():
            self.start_point.parent = None
            queue = deque([self.start_point])
            visited = {self.start_point}

            while queue:
                if not self.observer.is_running():
                    return False

                p = queue.popleft()
                self.stats.nodes_explored += 1
                self.observer.on_explore(p, alpha=0.1)

                if p.isEgale(self.end_point):
                    path = self._build_path(p)
                    self.stats.success = True
                    self._finalize_path(path)
                    return True

                for n in self._neighbors(p, previous_point=p.parent):
                    if n in visited:
                        continue
                    n.parent = p
                    visited.add(n)
                    queue.append(n)

            self.stats.success = False
            return False
