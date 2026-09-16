# ui/pick3d.py
"""Convertit un clic 2D sur le canvas 3D en cellule (i, j, k)."""
from mpl_toolkits.mplot3d import proj3d


def project(ax, x, y, z):
    """Coordonnées display (px, py) d'un point 3D (x, y, z)."""
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax.get_proj())
    return ax.transData.transform((x2, y2))


def find_nearest_cell(ax, click_xy, size, max_distance_px=60):
    """
    Renvoie la cellule (i, j, k) dont le centre projeté est le plus
    proche du clic, ou None si trop loin.
    Deux passes : balayage grossier puis raffinement local.
    """
    cx, cy = click_xy
    stride = max(1, size // 10)

    # --- Passe 1 : grossière ---
    best = None
    best_d = float('inf')
    for i in range(0, size, stride):
        for j in range(0, size, stride):
            for k in range(0, size, stride):
                px, py = project(ax, i + 0.5, j + 0.5, k + 0.5)
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_d:
                    best_d, best = d, (i, j, k)

    if best is None:
        return None

    # --- Passe 2 : raffinement local ---
    bi, bj, bk = best
    for i in range(max(0, bi - stride), min(size, bi + stride + 1)):
        for j in range(max(0, bj - stride), min(size, bj + stride + 1)):
            for k in range(max(0, bk - stride), min(size, bk + stride + 1)):
                px, py = project(ax, i + 0.5, j + 0.5, k + 0.5)
                d = (px - cx) ** 2 + (py - cy) ** 2
                if d < best_d:
                    best_d, best = d, (i, j, k)

    if best_d > max_distance_px ** 2:
        return None
    return best