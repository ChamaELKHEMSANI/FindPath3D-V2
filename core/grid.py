# core/grid.py
"""Grille 3D générique : stockage des obstacles et validation."""
import numpy as np
from core.point import Point


class Grid:
    def __init__(self, size=50, seed=None, generator='random', obstacle=0):
        self.size = size
        self.seed = seed
        self.generator = generator
        self.obstacle = obstacle
        self.rng = np.random.default_rng(seed)
        self.obstacle_point = []
        self.obstacle_set = set()

    def clear(self):
        self.obstacle_point = []
        self.obstacle_set = set()

    def AddObstaclePoint(self, x, y, z):
        if not self.IsValidPoint(x, y, z):
            return
        key = (int(x), int(y), int(z))
        if key in self.obstacle_set:
            return
        self.obstacle_set.add(key)
        self.obstacle_point.append(Point(*key))

    def IsObstacle(self, i, j, k):
        return (i, j, k) in self.obstacle_set

    def IsValidPoint(self, x, y, z):
        if x < 0 or y < 0 or z < 0:
            return False
        if x >= self.size or y >= self.size or z >= self.size:
            return False
        return True
    
    def toggle_obstacle(self, x, y, z):
        """Bascule obstacle/libre. Renvoie True si c'est devenu un obstacle."""
        if not self.IsValidPoint(x, y, z):
            return False
        key = (int(x), int(y), int(z))

        if key in self.obstacle_set:
            self.obstacle_set.discard(key)
            self.obstacle_point = [
                p for p in self.obstacle_point
                if not (p.x == x and p.y == y and p.z == z)
            ]
            return False

        self.obstacle_set.add(key)
        self.obstacle_point.append(Point(*key))
        return True

    def clear_obstacles(self):
        self.obstacle_point = []
        self.obstacle_set = set()