# core/map_random.py
"""Carte 3D avec obstacles aléatoires. Lookup obstacle en O(1)."""
import numpy as np
from core.point import Point


class MapRandom:
    def __init__(self, size=50, obstacle=10, seed=None):
        self.size = size
        self.obstacle = obstacle
        self.rng = np.random.default_rng(seed)
        self.obstacle_point = []
        self.obstacle_set = set()
        self.GenerateObstacle()

    def GenerateObstacle(self):
        self.obstacle_point = []
        self.obstacle_set = set()

        for _ in range(self.obstacle):
            x = int(self.rng.integers(0, self.size))
            y = int(self.rng.integers(0, self.size))
            z = int(self.rng.integers(0, self.size))

            rand_obstacle = float(self.rng.random())
            len_obstacle = max(2, int(self.rng.integers(self.size // 2 or 2, self.size // 2 + 2)))
            haut_obstacle = max(2, int(self.rng.integers(self.size // 2 or 2, self.size // 2 + 2)))

            if rand_obstacle > 0.25:
                self._add_vertical_block(x, y, z, len_obstacle, haut_obstacle)
            elif rand_obstacle > 0.125:
                for k in range(haut_obstacle):
                    for l in range(len_obstacle):
                        self.AddObstaclePoint(x, y + l, z + k)
            else:
                for k in range(haut_obstacle):
                    for l in range(len_obstacle):
                        self.AddObstaclePoint(x + l, y, z + k)

    def _add_vertical_block(self, x, y, z, length, height):
        for k in range(height):
            for i in range(length - 4, length + 4):
                self.AddObstaclePoint(x + i, y + length - i, z + k)
                self.AddObstaclePoint(x + i, y + length - i - 1, z + k)
                self.AddObstaclePoint(x + length - i, y + i, z + k)
                self.AddObstaclePoint(x + length - i, y + i - 1, z + k)

    def AddObstaclePoint(self, x, y, z):
        if not self.IsValidPoint(x, y, z):
            return
        key = (x, y, z)
        if key in self.obstacle_set:
            return
        self.obstacle_set.add(key)
        self.obstacle_point.append(Point(x, y, z))

    def IsObstacle(self, i, j, k):
        return (i, j, k) in self.obstacle_set

    def IsValidPoint(self, x, y, z):
        if x < 0 or y < 0 or z < 0:
            return False
        if x >= self.size or y >= self.size or z >= self.size:
            return False
        return True