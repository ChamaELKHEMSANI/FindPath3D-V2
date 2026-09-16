# core/generators.py
"""Générateurs de cartes : random, maze, caves, rooms, pillars."""
import numpy as np

from core.grid import Grid
from core.octree import OctreeMap, grid_to_octree


# =====================================================================
# 1) Aléatoire (portage de l'ancien MapRandom)
# =====================================================================
def fill_random(grid, obstacle=10):
    for _ in range(obstacle):
        x = int(grid.rng.integers(0, grid.size))
        y = int(grid.rng.integers(0, grid.size))
        z = int(grid.rng.integers(0, grid.size))

        r = float(grid.rng.random())
        length = max(2, int(grid.rng.integers(2, max(3, grid.size // 2))))
        height = max(2, int(grid.rng.integers(2, max(3, grid.size // 2))))

        if r > 0.25:
            _vertical_block(grid, x, y, z, length, height)
        elif r > 0.125:
            for k in range(height):
                for l in range(length):
                    grid.AddObstaclePoint(x, y + l, z + k)
        else:
            for k in range(height):
                for l in range(length):
                    grid.AddObstaclePoint(x + l, y, z + k)


def _vertical_block(grid, x, y, z, length, height):
    for k in range(height):
        for i in range(length - 4, length + 4):
            grid.AddObstaclePoint(x + i, y + length - i, z + k)
            grid.AddObstaclePoint(x + i, y + length - i - 1, z + k)
            grid.AddObstaclePoint(x + length - i, y + i, z + k)
            grid.AddObstaclePoint(x + length - i, y + i - 1, z + k)


# =====================================================================
# 2) Labyrinthe : DFS récursif sur XY, extrudé sur Z
# =====================================================================
def fill_maze(grid, obstacle=10):
    N = grid.size
    if N < 5:
        return fill_random(grid, obstacle)

    rng = np.random.default_rng(grid.seed if grid.seed is not None else 0)
    wall = np.ones((N, N), dtype=bool)   # True = mur

    stack = [(1, 1)]
    wall[1, 1] = False

    while stack:
        x, y = stack[-1]
        candidates = []
        for dx, dy in ((2, 0), (-2, 0), (0, 2), (0, -2)):
            nx, ny = x + dx, y + dy
            if 1 <= nx < N - 1 and 1 <= ny < N - 1 and wall[ny][nx]:
                candidates.append((nx, ny, x + dx // 2, y + dy // 2))

        if not candidates:
            stack.pop()
            continue

        nx, ny, wx, wy = candidates[int(rng.integers(0, len(candidates)))]
        wall[wy][wx] = False
        wall[ny][nx] = False
        stack.append((nx, ny))

    # Extrusion sur Z (toutes les couches identiques)
    for k in range(N):
        for y in range(N):
            for x in range(N):
                if wall[y][x]:
                    grid.AddObstaclePoint(x, y, k)


# =====================================================================
# 3) Cavernes : automate cellulaire 3D
# =====================================================================
def fill_caves(grid, obstacle=10):
    N = grid.size
    rng = np.random.default_rng(grid.seed if grid.seed is not None else 0)

    p = 0.45
    arr = rng.random((N, N, N)) < p

    for _ in range(4):
        new = arr.copy()
        for i in range(N):
            for j in range(N):
                for k in range(N):
                    count = 0
                    for di in (-1, 0, 1):
                        for dj in (-1, 0, 1):
                            for dk in (-1, 0, 1):
                                if di == 0 and dj == 0 and dk == 0:
                                    continue
                                ni, nj, nk = i + di, j + dj, k + dk
                                if 0 <= ni < N and 0 <= nj < N and 0 <= nk < N:
                                    if arr[ni, nj, nk]:
                                        count += 1
                    new[i, j, k] = count >= 13
        arr = new

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if arr[i, j, k]:
                    grid.AddObstaclePoint(i, j, k)


# =====================================================================
# 4) Salles + couloirs : K boîtes reliées par des corridors L
# =====================================================================
def fill_rooms(grid, obstacle=10):
    N = grid.size
    if N < 8:
        return fill_random(grid, obstacle)

    rng = np.random.default_rng(grid.seed if grid.seed is not None else 0)
    wall = np.ones((N, N, N), dtype=bool)

    num_rooms = max(3, N // 3)
    rooms = []
    for _ in range(num_rooms):
        w = int(rng.integers(2, max(3, N // 3)))
        h = int(rng.integers(2, max(3, N // 3)))
        d = int(rng.integers(2, max(3, N // 3)))
        x = int(rng.integers(1, max(2, N - w - 1)))
        y = int(rng.integers(1, max(2, N - h - 1)))
        z = int(rng.integers(1, max(2, N - d - 1)))
        rooms.append((x, y, z, w, h, d))
        wall[x:x + w, y:y + h, z:z + d] = False

    for i in range(1, len(rooms)):
        x1, y1, z1, w1, h1, d1 = rooms[i - 1]
        x2, y2, z2, w2, h2, d2 = rooms[i]
        cx1, cy1, cz1 = x1 + w1 // 2, y1 + h1 // 2, z1 + d1 // 2
        cx2, cy2, cz2 = x2 + w2 // 2, y2 + h2 // 2, z2 + d2 // 2

        x_min, x_max = sorted((cx1, cx2))
        wall[x_min:x_max + 1, cy1, cz1] = False
        y_min, y_max = sorted((cy1, cy2))
        wall[cx2, y_min:y_max + 1, cz1] = False
        z_min, z_max = sorted((cz1, cz2))
        wall[cx2, cy2, z_min:z_max + 1] = False

    for i in range(N):
        for j in range(N):
            for k in range(N):
                if wall[i, j, k]:
                    grid.AddObstaclePoint(i, j, k)


# =====================================================================
# 5) Piliers : colonnes verticales régulières avec jitter
# =====================================================================
def fill_pillars(grid, obstacle=10):
    N = grid.size
    rng = np.random.default_rng(grid.seed if grid.seed is not None else 0)
    spacing = max(3, N // 4)
    height = max(2, N // 2)

    for x in range(1, N, spacing):
        for y in range(1, N, spacing):
            jx = x + int(rng.integers(-1, 2))
            jy = y + int(rng.integers(-1, 2))
            z0 = int(rng.integers(0, max(1, N - height)))
            for z in range(z0, min(z0 + height, N)):
                grid.AddObstaclePoint(jx, jy, z)


# =====================================================================
# Factory
# =====================================================================
GENERATORS = {
    'random': fill_random,
    'maze': fill_maze,
    'caves': fill_caves,
    'rooms': fill_rooms,
    'pillars': fill_pillars,
}

GENERATOR_LABELS = {
    'random':  "Aléatoire",
    'maze':    "Labyrinthe",
    'caves':   "Cavernes",
    'rooms':   "Salles + couloirs",
    'pillars': "Piliers",
}


def create_grid(generator='random', size=10, obstacle=10, seed=None, backend='grid'):
    """
    backend: 'grid' (défaut, historique) ou 'octree' (Phase 1 POC).
    """
    if backend == 'octree':
        grid = OctreeMap(size=size, seed=seed, generator=generator, obstacle=obstacle)
    else:
        grid = Grid(size=size, seed=seed, generator=generator, obstacle=obstacle)
    fn = GENERATORS.get(generator, fill_random)
    fn(grid, obstacle=obstacle)
    return grid


# =====================================================================
# Phase 3 - Générateurs avancés
# =====================================================================

def fill_perlin(grid, obstacle=10):
    """
    Terrain type Perlin simplifié (valeur de bruit 3D → obstacles).
    Sans dépendance externe : hash + interpolation.
    """
    import math
    N = grid.size
    rng = grid.rng
    seed = int(grid.seed) if grid.seed is not None else 0

    def fade(t):
        return t * t * t * (t * (t * 6 - 15) + 10)

    def lerp(a, b, t):
        return a + t * (b - a)

    def grad(hx, hy, hz, x, y, z):
        h = (hx * 374761393 + hy * 668265263 + hz * 1274126177 + seed) & 15
        u = x if h < 8 else y
        v = y if h < 4 else (x if h in (12, 14) else z)
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

    def noise(x, y, z):
        xi, yi, zi = int(math.floor(x)), int(math.floor(y)), int(math.floor(z))
        xf, yf, zf = x - xi, y - yi, z - zi
        u, v, w = fade(xf), fade(yf), fade(zf)
        n000 = grad(xi, yi, zi, xf, yf, zf)
        n001 = grad(xi, yi, zi + 1, xf, yf, zf - 1)
        n010 = grad(xi, yi + 1, zi, xf, yf - 1, zf)
        n011 = grad(xi, yi + 1, zi + 1, xf, yf - 1, zf - 1)
        n100 = grad(xi + 1, yi, zi, xf - 1, yf, zf)
        n101 = grad(xi + 1, yi, zi + 1, xf - 1, yf, zf - 1)
        n110 = grad(xi + 1, yi + 1, zi, xf - 1, yf - 1, zf)
        n111 = grad(xi + 1, yi + 1, zi + 1, xf - 1, yf - 1, zf - 1)
        x1 = lerp(n000, n100, u)
        x2 = lerp(n010, n110, u)
        y1 = lerp(x1, x2, v)
        x3 = lerp(n001, n101, u)
        x4 = lerp(n011, n111, u)
        y2 = lerp(x3, x4, v)
        return lerp(y1, y2, w)

    # Densité liée au paramètre obstacle (approx)
    threshold = 0.15 + min(0.45, obstacle / max(20.0, N * 2))
    scale = max(2.5, N / 8.0)
    for x in range(N):
        for y in range(N):
            for z in range(N):
                n = noise(x / scale, y / scale, z / scale)
                # Basses fréquences = massifs ; threshold → solidité
                if n > threshold:
                    grid.AddObstaclePoint(x, y, z)


def fill_bsp(grid, obstacle=10):
    """
    Donjon BSP 3D simplifié : subdivision récursive → salles + couloirs.
    """
    import math
    N = grid.size
    if N < 8:
        return fill_rooms(grid, obstacle)
    rng = grid.rng

    # Tout est mur au départ
    solid = [[[True for _ in range(N)] for _ in range(N)] for _ in range(N)]

    class Node:
        def __init__(self, x0, y0, z0, x1, y1, z1):
            self.x0, self.y0, self.z0 = x0, y0, z0
            self.x1, self.y1, self.z1 = x1, y1, z1
            self.left = self.right = None
            self.room = None  # (rx, ry, rz, rw, rh, rd)

    def split(node, depth):
        w = node.x1 - node.x0
        h = node.y1 - node.y0
        d = node.z1 - node.z0
        if depth <= 0 or min(w, h, d) < 5:
            # salle intérieure
            rw = max(2, w - 2 - int(rng.integers(0, max(1, w // 3))))
            rh = max(2, h - 2 - int(rng.integers(0, max(1, h // 3))))
            rd = max(2, d - 2 - int(rng.integers(0, max(1, d // 3))))
            rx = node.x0 + 1 + int(rng.integers(0, max(1, w - rw - 1)))
            ry = node.y0 + 1 + int(rng.integers(0, max(1, h - rh - 1)))
            rz = node.z0 + 1 + int(rng.integers(0, max(1, d - rd - 1)))
            node.room = (rx, ry, rz, rw, rh, rd)
            return
        axis = int(rng.integers(0, 3))
        if axis == 0 and w >= 6:
            cut = node.x0 + 3 + int(rng.integers(0, max(1, w - 5)))
            node.left = Node(node.x0, node.y0, node.z0, cut, node.y1, node.z1)
            node.right = Node(cut, node.y0, node.z0, node.x1, node.y1, node.z1)
        elif axis == 1 and h >= 6:
            cut = node.y0 + 3 + int(rng.integers(0, max(1, h - 5)))
            node.left = Node(node.x0, node.y0, node.z0, node.x1, cut, node.z1)
            node.right = Node(node.x0, cut, node.z0, node.x1, node.y1, node.z1)
        elif d >= 6:
            cut = node.z0 + 3 + int(rng.integers(0, max(1, d - 5)))
            node.left = Node(node.x0, node.y0, node.z0, node.x1, node.y1, cut)
            node.right = Node(node.x0, node.y0, cut, node.x1, node.y1, node.z1)
        else:
            rw = max(2, w - 2)
            rh = max(2, h - 2)
            rd = max(2, d - 2)
            node.room = (node.x0 + 1, node.y0 + 1, node.z0 + 1, rw - 1, rh - 1, rd - 1)
            return
        split(node.left, depth - 1)
        split(node.right, depth - 1)

    def carve_room(room):
        rx, ry, rz, rw, rh, rd = room
        for x in range(rx, min(rx + rw, N)):
            for y in range(ry, min(ry + rh, N)):
                for z in range(rz, min(rz + rd, N)):
                    solid[x][y][z] = False

    def center(room):
        rx, ry, rz, rw, rh, rd = room
        return rx + rw // 2, ry + rh // 2, rz + rd // 2

    def carve_corridor(a, b):
        x0, y0, z0 = a
        x1, y1, z1 = b
        x, y, z = x0, y0, z0
        # L-shape 3D
        while x != x1:
            solid[x][y][z] = False
            x += 1 if x1 > x else -1
        while y != y1:
            solid[x][y][z] = False
            y += 1 if y1 > y else -1
        while z != z1:
            solid[x][y][z] = False
            z += 1 if z1 > z else -1
        solid[x1][y1][z1] = False

    def collect_rooms(node, out):
        if node is None:
            return
        if node.room:
            out.append(node.room)
        collect_rooms(node.left, out)
        collect_rooms(node.right, out)

    def connect(node):
        if node is None or (node.left is None and node.right is None):
            return node.room
        rl = connect(node.left)
        rr = connect(node.right)
        if rl and rr:
            carve_corridor(center(rl), center(rr))
        return rl or rr

    depth = max(2, int(math.log2(max(N, 4))) - 1)
    root = Node(0, 0, 0, N, N, N)
    split(root, depth)
    rooms = []
    collect_rooms(root, rooms)
    for r in rooms:
        carve_room(r)
    connect(root)

    for x in range(N):
        for y in range(N):
            for z in range(N):
                if solid[x][y][z]:
                    grid.AddObstaclePoint(x, y, z)


def fill_city(grid, obstacle=10):
    """
    Ville simplifiée : immeubles (blocs) + rues + quelques passages bas.
    """
    N = grid.size
    if N < 10:
        return fill_rooms(grid, obstacle)
    rng = grid.rng
    block = max(3, N // 5)
    street = 1

    for x in range(N):
        for y in range(N):
            # Rues en X et Y
            on_street = (x % (block + street) < street) or (y % (block + street) < street)
            if on_street:
                continue  # libre (rue)
            # Immeuble : occuper une hauteur variable
            h = 2 + int(rng.integers(0, max(2, N // 2)))
            for z in range(min(h, N)):
                grid.AddObstaclePoint(x, y, z)
            # Toit occasionnel avec trou (terrasse)
            if h < N and float(rng.random()) < 0.15:
                for z in range(h, min(h + 1, N)):
                    if float(rng.random()) < 0.7:
                        grid.AddObstaclePoint(x, y, z)


def fill_floating(grid, obstacle=10):
    """
    Îles flottantes : amas de voxels reliés par des ponts étroits.
    """
    N = grid.size
    rng = grid.rng
    n_islands = max(2, N // 6)
    islands = []
    for _ in range(n_islands):
        cx = int(rng.integers(2, max(3, N - 2)))
        cy = int(rng.integers(2, max(3, N - 2)))
        cz = int(rng.integers(1, max(2, N - 1)))
        rad = 1 + int(rng.integers(0, max(1, N // 8)))
        islands.append((cx, cy, cz, rad))
        for x in range(max(0, cx - rad), min(N, cx + rad + 1)):
            for y in range(max(0, cy - rad), min(N, cy + rad + 1)):
                for z in range(max(0, cz - rad // 2), min(N, cz + rad // 2 + 1)):
                    if (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= rad * rad + 1:
                        # Remplir le COMPLEMENTAIRE : on veut du vide autour, solide = île
                        pass
    # Stratégie : tout solide sauf îles + ponts = libre ? 
    # Plus simple : grille vide + îles comme obstacles denses, couloirs libres entre elles
    # → on place les îles comme obstacles, l'espace entre est libre (vol / ponts)
    for cx, cy, cz, rad in islands:
        for x in range(max(0, cx - rad), min(N, cx + rad + 1)):
            for y in range(max(0, cy - rad), min(N, cy + rad + 1)):
                for z in range(max(0, cz - max(1, rad // 2)), min(N, cz + max(1, rad // 2) + 1)):
                    if (x - cx) ** 2 + (y - cy) ** 2 + ((z - cz) * 2) ** 2 <= rad * rad + 1:
                        grid.AddObstaclePoint(x, y, z)
    # Quelques piliers / débris
    extra = max(0, obstacle // 3)
    for _ in range(extra):
        x = int(rng.integers(0, N))
        y = int(rng.integers(0, N))
        z = int(rng.integers(0, N))
        grid.AddObstaclePoint(x, y, z)


# Enregistrement Phase 3
GENERATORS['perlin'] = fill_perlin
GENERATORS['bsp'] = fill_bsp
GENERATORS['city'] = fill_city
GENERATORS['floating'] = fill_floating

GENERATOR_LABELS['perlin'] = "Perlin (terrain)"
GENERATOR_LABELS['bsp'] = "BSP Donjon"
GENERATOR_LABELS['city'] = "Ville urbaine"
GENERATOR_LABELS['floating'] = "Îles flottantes"
