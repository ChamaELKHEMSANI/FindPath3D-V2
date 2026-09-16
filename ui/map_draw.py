# ui/map_draw.py
"""Canvas matplotlib 3D - Phase 4 : heatmap exploration + chemin + timeline."""
import os
from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from core.point import Point


class MapDraw(FigureCanvas):

    def __init__(self, parent=None, image_folder='./images'):
        self.fig, self.ax = plt.subplots(subplot_kw={'projection': '3d'})
        super().__init__(self.fig)
        if parent is not None:
            self.setParent(parent)
        self.cptImg = 0
        self.image_folder = image_folder

        self._map_obj = None
        self._src = None
        self._dst = None
        self._explored: List[Point] = []
        self._path: List[Point] = []
        self._heatmap_enabled = True
        self._max_explore_draw = 800  # limiter pour perf
        self._view_limits = None
        self.mpl_connect("scroll_event", self._on_scroll)

    # ---------------- Rendu carte ----------------
    def draw_map(self, map_obj, pnt_src, pnt_dest):
        self._map_obj = map_obj
        self._src = pnt_src
        self._dst = pnt_dest
        self._explored = []
        self._path = []
        self._view_limits = None
        self._redraw_base()
        self.draw_idle()

    def _redraw_base(self):
        self.ax.clear()
        if self._map_obj is None:
            return
        size = self._map_obj.size

        self.ax.set_xlabel('x')
        xlim, ylim, zlim = self._current_limits(size)
        self.ax.set_xlim(*xlim)
        self.ax.set_xticklabels([])
        self.ax.set_ylabel('y')
        self.ax.set_ylim(*ylim)
        self.ax.set_yticklabels([])
        self.ax.set_zlabel('z')
        self.ax.set_zlim(*zlim)
        self.ax.set_zticks([])
        self.ax.set_zticklabels([])

        try:
            self.ax.set_box_aspect((1, 1, 1))
        except AttributeError:
            pass

        obs = getattr(self._map_obj, "obstacle_point", None) or []
        if obs:
            voxels = np.zeros((size, size, size), dtype=bool)
            for p in obs:
                if 0 <= p.x < size and 0 <= p.y < size and 0 <= p.z < size:
                    voxels[p.x, p.y, p.z] = True
            self.ax.voxels(
                voxels, facecolors='#8a8a8a', edgecolors='#e6e6e6',
                linewidth=0.18, alpha=0.28,
            )

        if self._src is not None:
            self.drawCube(self._src, couleur='#1f77b4', alpha=0.95)
        if self._dst is not None:
            self.drawCube(self._dst, couleur='#d62728', alpha=0.95)

    def _current_limits(self, size):
        if self._view_limits is None:
            return (0, size), (0, size), (0, size)
        return self._view_limits

    def _limits_from_center(self, cx, cy, cz, radius):
        size = self._map_obj.size
        radius = max(1.0, float(radius))

        def clamp_axis(center):
            span = min(float(size), radius * 2.0)
            low = float(center) - span / 2.0
            high = float(center) + span / 2.0
            if low < 0:
                high -= low
                low = 0.0
            if high > size:
                low -= high - size
                high = float(size)
            return max(0.0, low), min(float(size), high)

        return clamp_axis(cx), clamp_axis(cy), clamp_axis(cz)

    def _set_view_limits(self, limits):
        self._view_limits = limits
        self.redraw_full()

    def _on_scroll(self, event):
        if self._map_obj is None:
            return
        xlim, ylim, zlim = self._current_limits(self._map_obj.size)
        factor = 0.82 if event.button == "up" else 1.22

        def zoom_axis(lim):
            low, high = lim
            center = (low + high) / 2.0
            half = max(0.5, (high - low) * factor / 2.0)
            return center - half, center + half

        xlim = zoom_axis(xlim)
        ylim = zoom_axis(ylim)
        zlim = zoom_axis(zlim)

        size = self._map_obj.size

        def clamp(lim):
            low, high = lim
            span = min(float(size), max(1.0, high - low))
            center = (low + high) / 2.0
            low = center - span / 2.0
            high = center + span / 2.0
            if low < 0:
                high -= low
                low = 0.0
            if high > size:
                low -= high - size
                high = float(size)
            return max(0.0, low), min(float(size), high)

        self._set_view_limits((clamp(xlim), clamp(ylim), clamp(zlim)))

    def zoom_out_full(self):
        self._view_limits = None
        self.redraw_full()

    def set_zoom_region(self, x, y, z, radius):
        if self._map_obj is None:
            return
        size = self._map_obj.size
        x = max(0, min(size - 1, int(x)))
        y = max(0, min(size - 1, int(y)))
        z = max(0, min(size - 1, int(z)))
        radius = max(1, int(radius))
        self._set_view_limits(self._limits_from_center(x, y, z, radius))

    def reset_zoom(self):
        self.zoom_out_full()

    def set_view(self, name):
        views = {
            "iso": (25, -45),
            "top": (90, -90),
            "bottom": (-90, -90),
            "front": (0, -90),
            "back": (0, 90),
            "left": (0, 180),
            "right": (0, 0),
        }
        elev, azim = views.get(name, views["iso"])
        self.ax.view_init(elev=elev, azim=azim)
        self.draw_idle()

    # ---------------- Exploration (heatmap) ----------------
    def record_explore(self, point, alpha=0.1):
        self._explored.append(point)

    def draw_explore_point(self, point, alpha=0.1):
        """Appelé en live pendant l'algo (léger)."""
        self.record_explore(point, alpha)
        # Dessin live discret (cyan)
        self.drawCube(point, couleur='c', alpha=max(0.08, min(0.35, alpha)))
        self.draw_idle()

    def render_heatmap(self):
        """Affiche l'exploration en nuage de points coloré (ordre = chaleur)."""
        if not self._explored or not self._heatmap_enabled:
            return
        pts = self._explored
        if len(pts) > self._max_explore_draw:
            step = max(1, len(pts) // self._max_explore_draw)
            pts = pts[::step]
        xs = [p.x + 0.5 for p in pts]
        ys = [p.y + 0.5 for p in pts]
        zs = [p.z + 0.5 for p in pts]
        colors = np.linspace(0.2, 1.0, len(pts))
        self.ax.scatter(
            xs, ys, zs, c=colors, cmap='cool', s=8, alpha=0.45, depthshade=True
        )

    # ---------------- Chemin ----------------
    def set_path(self, path: List[Point]):
        self._path = list(path) if path else []

    def render_path(self, up_to: Optional[int] = None):
        """Dessine le chemin (optionnellement tronqué pour timeline)."""
        if not self._path:
            return
        path = self._path if up_to is None else self._path[: max(1, up_to)]
        if len(path) < 2:
            if path:
                self.drawCube(path[0], couleur='#2ca02c', alpha=0.9)
            return
        xs = [p.x + 0.5 for p in path]
        ys = [p.y + 0.5 for p in path]
        zs = [p.z + 0.5 for p in path]
        self.ax.plot(xs, ys, zs, color='#2ca02c', linewidth=2.5, alpha=0.95)
        # marqueurs start/end du path
        self.ax.scatter(
            [xs[0]], [ys[0]], [zs[0]], c='#2ca02c', s=40, marker='o'
        )
        self.ax.scatter(
            [xs[-1]], [ys[-1]], [zs[-1]], c='#ff7f0e', s=40, marker='*'
        )

    def redraw_full(self, path_up_to: Optional[int] = None):
        """Reconstruit la scène : base + heatmap + chemin."""
        self._redraw_base()
        self.render_heatmap()
        self.render_path(up_to=path_up_to)
        self.draw_idle()

    # ---------------- Cube ----------------
    @staticmethod
    def _cube_faces(p):
        pts = np.array([
            [p.x, p.y, p.z], [p.x + 1, p.y, p.z],
            [p.x + 1, p.y + 1, p.z], [p.x, p.y + 1, p.z],
            [p.x, p.y, p.z + 1], [p.x + 1, p.y, p.z + 1],
            [p.x + 1, p.y + 1, p.z + 1], [p.x, p.y + 1, p.z + 1],
        ])
        return [
            [pts[0], pts[1], pts[2], pts[3]],
            [pts[4], pts[5], pts[6], pts[7]],
            [pts[0], pts[1], pts[5], pts[4]],
            [pts[2], pts[3], pts[7], pts[6]],
            [pts[1], pts[2], pts[6], pts[5]],
            [pts[4], pts[7], pts[3], pts[0]],
        ]

    def drawCube(self, p, couleur='black', alpha=0.9):
        faces = self._cube_faces(p)
        self.ax.add_collection3d(
            Poly3DCollection(
                faces, facecolors=couleur,
                linewidths=0.4, edgecolors=couleur, alpha=alpha,
            )
        )

    # ---------------- Divers ----------------
    def refresh(self):
        self.draw_idle()

    def SaveImage(self, folder=None):
        folder = folder or self.image_folder
        os.makedirs(folder, exist_ok=True)
        self.cptImg += 1
        filename = os.path.join(folder, f'{self.cptImg}.png')
        try:
            self.fig.savefig(filename, dpi=120, bbox_inches='tight')
        except Exception as exc:
            print(f"[MapDraw] Impossible de sauvegarder {filename} : {exc}")

    def rotate(self):
        self.ax.view_init(elev=20, azim=(self.ax.azim + 30) % 360)
        self.draw_idle()
