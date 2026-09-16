# core/point.py
"""Point 3D : coordonnées, coût, parent, comparaison et hash."""
import sys
import numpy as np


class Point:
    __slots__ = ('x', 'y', 'z', 'cost', 'parent')

    def __init__(self, x, y, z=0, cost=sys.maxsize):
        self.x = x
        self.y = y
        self.z = z
        self.cost = cost
        self.parent = None

    def __eq__(self, other):
        return (isinstance(other, Point)
                and self.x == other.x
                and self.y == other.y
                and self.z == other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))

    def __lt__(self, other):
        return (self.x, self.y, self.z) < (other.x, other.y, other.z)

    def __repr__(self):
        return f"Point({self.x},{self.y},{self.z})"

    def isEgaleCoord(self, x, y, z):
        return self.x == x and self.y == y and self.z == z

    def isEgale(self, pnt):
        return self.x == pnt.x and self.y == pnt.y and self.z == pnt.z

    def distance(self, pnt):
        return abs(self.x - pnt.x) + abs(self.y - pnt.y) + abs(self.z - pnt.z)

    @staticmethod
    def rand(size, rng=None):
        rng = rng or np.random
        return Point(int(rng.randint(0, size)),
                     int(rng.randint(0, size)),
                     int(rng.randint(0, size)))