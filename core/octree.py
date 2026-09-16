# core/octree.py
"""
Octree spatial - Phase 1 POC.
Structure hiérarchique compatible avec l'API de Grid pour un remplacement
progressif (fallback transparent).

Objectifs du POC :
- Stockage sparse des obstacles
- Requêtes IsObstacle / IsValidPoint O(log n) en moyenne
- Même interface publique que Grid (drop-in)
- Rapport mémoire et statistiques
"""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from core.point import Point


class OctreeNode:
    """Nœud d'un octree 3D."""

    __slots__ = (
        "xmin", "ymin", "zmin", "xmax", "ymax", "zmax",
        "depth", "children", "obstacles", "is_leaf",
    )

    def __init__(
        self,
        xmin: int, ymin: int, zmin: int,
        xmax: int, ymax: int, zmax: int,
        depth: int = 0,
    ):
        self.xmin = xmin
        self.ymin = ymin
        self.zmin = zmin
        self.xmax = xmax
        self.ymax = ymax
        self.zmax = zmax
        self.depth = depth
        self.children: Optional[List[Optional["OctreeNode"]]] = None
        self.obstacles: Set[Tuple[int, int, int]] = set()
        self.is_leaf = True

    @property
    def volume(self) -> int:
        return max(0, (self.xmax - self.xmin) *
                   (self.ymax - self.ymin) *
                   (self.zmax - self.zmin))

    def contains(self, x: int, y: int, z: int) -> bool:
        return (self.xmin <= x < self.xmax and
                self.ymin <= y < self.ymax and
                self.zmin <= z < self.zmax)

    def _child_index(self, x: int, y: int, z: int) -> int:
        mx = (self.xmin + self.xmax) // 2
        my = (self.ymin + self.ymax) // 2
        mz = (self.zmin + self.zmax) // 2
        idx = 0
        if x >= mx:
            idx |= 1
        if y >= my:
            idx |= 2
        if z >= mz:
            idx |= 4
        return idx

    def _ensure_children(self):
        if self.children is not None:
            return
        mx = (self.xmin + self.xmax) // 2
        my = (self.ymin + self.ymax) // 2
        mz = (self.zmin + self.zmax) // 2
        # 8 octants
        bounds = [
            (self.xmin, self.ymin, self.zmin, mx, my, mz),
            (mx, self.ymin, self.zmin, self.xmax, my, mz),
            (self.xmin, my, self.zmin, mx, self.ymax, mz),
            (mx, my, self.zmin, self.xmax, self.ymax, mz),
            (self.xmin, self.ymin, mz, mx, my, self.zmax),
            (mx, self.ymin, mz, self.xmax, my, self.zmax),
            (self.xmin, my, mz, mx, self.ymax, self.zmax),
            (mx, my, mz, self.xmax, self.ymax, self.zmax),
        ]
        self.children = []
        for b in bounds:
            if b[3] > b[0] and b[4] > b[1] and b[5] > b[2]:
                self.children.append(
                    OctreeNode(*b, depth=self.depth + 1)
                )
            else:
                self.children.append(None)
        self.is_leaf = False

    def insert(self, x: int, y: int, z: int, max_depth: int, max_leaf_size: int = 8):
        if not self.contains(x, y, z):
            return
        if self.is_leaf:
            self.obstacles.add((x, y, z))
            # Subdiviser si trop plein et profondeur restante
            if (len(self.obstacles) > max_leaf_size and
                    self.depth < max_depth and
                    self.volume > 1):
                self._subdivide(max_depth, max_leaf_size)
            return
        # Nœud interne
        self._ensure_children()
        idx = self._child_index(x, y, z)
        child = self.children[idx]
        if child is not None:
            child.insert(x, y, z, max_depth, max_leaf_size)

    def _subdivide(self, max_depth: int, max_leaf_size: int):
        pts = list(self.obstacles)
        self.obstacles.clear()
        self._ensure_children()
        for x, y, z in pts:
            idx = self._child_index(x, y, z)
            child = self.children[idx]
            if child is not None:
                child.insert(x, y, z, max_depth, max_leaf_size)

    def query(self, x: int, y: int, z: int) -> bool:
        if not self.contains(x, y, z):
            return False
        if self.is_leaf:
            return (x, y, z) in self.obstacles
        if self.children is None:
            return False
        idx = self._child_index(x, y, z)
        child = self.children[idx]
        if child is None:
            return False
        return child.query(x, y, z)

    def remove(self, x: int, y: int, z: int) -> bool:
        if not self.contains(x, y, z):
            return False
        if self.is_leaf:
            if (x, y, z) in self.obstacles:
                self.obstacles.discard((x, y, z))
                return True
            return False
        if self.children is None:
            return False
        idx = self._child_index(x, y, z)
        child = self.children[idx]
        if child is None:
            return False
        return child.remove(x, y, z)

    def count_obstacles(self) -> int:
        if self.is_leaf:
            return len(self.obstacles)
        total = 0
        if self.children:
            for c in self.children:
                if c is not None:
                    total += c.count_obstacles()
        return total

    def count_nodes(self) -> int:
        if self.is_leaf:
            return 1
        total = 1
        if self.children:
            for c in self.children:
                if c is not None:
                    total += c.count_nodes()
        return total

    def collect_obstacles(self, out: List[Point]):
        if self.is_leaf:
            for x, y, z in self.obstacles:
                out.append(Point(x, y, z))
            return
        if self.children:
            for c in self.children:
                if c is not None:
                    c.collect_obstacles(out)


class OctreeMap:
    """
    Carte 3D basée sur un octree.
    API compatible avec Grid pour un remplacement progressif.
    """

    def __init__(
        self,
        size: int = 50,
        seed=None,
        generator: str = "random",
        obstacle: int = 0,
        max_depth: Optional[int] = None,
        max_leaf_size: int = 8,
    ):
        self.size = size
        self.seed = seed
        self.generator = generator
        self.obstacle = obstacle
        self.max_leaf_size = max_leaf_size
        import numpy as np
        self.rng = np.random.default_rng(seed)
        # Profondeur max : log2(size) arrondi + marge
        if max_depth is None:
            import math
            max_depth = max(1, int(math.ceil(math.log2(max(size, 2)))) + 2)
        self.max_depth = max_depth

        self.root = OctreeNode(0, 0, 0, size, size, size, depth=0)
        # Cache pour compatibilité avec le code qui lit obstacle_point
        self._obstacle_point_cache: Optional[List[Point]] = None
        self._n_obstacles = 0

    # ------------------------------------------------------------------
    # API compatible Grid
    # ------------------------------------------------------------------
    def clear(self):
        self.root = OctreeNode(0, 0, 0, self.size, self.size, self.size, depth=0)
        self._obstacle_point_cache = None
        self._n_obstacles = 0

    def clear_obstacles(self):
        self.clear()

    def IsValidPoint(self, x, y, z) -> bool:
        if x < 0 or y < 0 or z < 0:
            return False
        if x >= self.size or y >= self.size or z >= self.size:
            return False
        return True

    def IsObstacle(self, i, j, k) -> bool:
        return self.root.query(int(i), int(j), int(k))

    def AddObstaclePoint(self, x, y, z):
        if not self.IsValidPoint(x, y, z):
            return
        xi, yi, zi = int(x), int(y), int(z)
        if self.root.query(xi, yi, zi):
            return  # déjà présent
        self.root.insert(xi, yi, zi, self.max_depth, self.max_leaf_size)
        self._n_obstacles += 1
        self._obstacle_point_cache = None  # invalider le cache

    def toggle_obstacle(self, x, y, z) -> bool:
        if not self.IsValidPoint(x, y, z):
            return False
        xi, yi, zi = int(x), int(y), int(z)
        if self.root.query(xi, yi, zi):
            self.root.remove(xi, yi, zi)
            self._n_obstacles = max(0, self._n_obstacles - 1)
            self._obstacle_point_cache = None
            return False
        self.root.insert(xi, yi, zi, self.max_depth, self.max_leaf_size)
        self._n_obstacles += 1
        self._obstacle_point_cache = None
        return True

    @property
    def obstacle_point(self) -> List[Point]:
        if self._obstacle_point_cache is None:
            out: List[Point] = []
            self.root.collect_obstacles(out)
            self._obstacle_point_cache = out
        return self._obstacle_point_cache

    @obstacle_point.setter
    def obstacle_point(self, value):
        # Permet l'affectation directe (rare) - reconstruit l'octree
        self.clear()
        for p in value:
            self.AddObstaclePoint(p.x, p.y, p.z)

    # ------------------------------------------------------------------
    # Stats / debug
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        n_obs = self.root.count_obstacles()
        n_nodes = self.root.count_nodes()
        # Estimation mémoire très approximative
        # (chaque nœud ~ 100-200 bytes + 24 bytes par obstacle dans les feuilles)
        mem_est_bytes = n_nodes * 160 + n_obs * 24
        return {
            "size": self.size,
            "n_obstacles": n_obs,
            "n_nodes": n_nodes,
            "max_depth": self.max_depth,
            "mem_est_kb": round(mem_est_bytes / 1024, 2),
            "sparsity": round(1.0 - n_obs / max(1, self.size ** 3), 4),
        }

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"OctreeMap(size={s['size']}, obstacles={s['n_obstacles']}, "
            f"nodes={s['n_nodes']}, sparsity={s['sparsity']:.1%})"
        )


def grid_to_octree(grid) -> OctreeMap:
    """Construit un OctreeMap à partir d'une Grid existante."""
    ot = OctreeMap(
        size=grid.size,
        seed=getattr(grid, "seed", None),
        generator=getattr(grid, "generator", "random"),
        obstacle=getattr(grid, "obstacle", 0),
    )
    for p in grid.obstacle_point:
        ot.AddObstaclePoint(p.x, p.y, p.z)
    return ot
