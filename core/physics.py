# core/physics.py
"""
Contraintes physiques réalistes pour agents (humanoid / robot).
Utilisées pour filtrer les mouvements valides dans le pathfinding.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from core.point import Point


class PhysicsConstraints:
    """
    Vérifie si un déplacement from → to est physiquement acceptable.

    Paramètres (valeurs par défaut orientées humanoid) :
    - max_slope_deg : pente maximale ascendante (degrés). 45° ≈ 1:1.
    - max_jump_height : hauteur de saut maximale (en voxels).
    - min_turn_radius : rayon de virage minimum (en voxels). 0 = désactivé.
    - allow_diagonal_up : autoriser les montées en diagonale 3D.
    """

    def __init__(
        self,
        max_slope_deg: float = 45.0,
        max_jump_height: float = 1.5,
        min_turn_radius: float = 0.0,
        allow_diagonal_up: bool = True,
    ):
        self.max_slope_deg = max_slope_deg
        self.max_slope = math.tan(math.radians(max_slope_deg)) if max_slope_deg < 89 else 1e9
        self.max_jump_height = max_jump_height
        self.min_turn_radius = min_turn_radius
        self.allow_diagonal_up = allow_diagonal_up

    def is_valid_move(
        self,
        from_point: Point,
        to_point: Point,
        previous_point: Optional[Point] = None,
    ) -> bool:
        """
        Retourne True si le mouvement from → to respecte les contraintes.
        previous_point (optionnel) permet de vérifier le rayon de virage.
        """
        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y
        dz = to_point.z - from_point.z

        # Distance horizontale (XY)
        d_xy = math.sqrt(dx * dx + dy * dy)
        # Distance 3D
        d_3d = math.sqrt(dx * dx + dy * dy + dz * dz)

        if d_3d < 1e-9:
            return False  # même point

        # 1. Hauteur de saut maximale
        if dz > self.max_jump_height:
            return False

        # 2. Pente maximale (uniquement en montée)
        if dz > 0 and d_xy > 1e-9:
            slope = dz / d_xy
            if slope > self.max_slope + 1e-9:
                return False
        elif dz > 0 and d_xy < 1e-9:
            # Montée purement verticale : interdite sauf si jump_height le permet
            # (déjà filtré ci-dessus). On refuse les colonnes verticales pures
            # pour un agent humanoid.
            return False

        # 3. Option : interdire certaines diagonales montantes extrêmes
        if not self.allow_diagonal_up and dz > 0 and (abs(dx) + abs(dy)) > 1:
            return False

        # 4. Rayon de virage (si previous_point fourni et min_turn_radius > 0)
        if previous_point is not None and self.min_turn_radius > 0:
            if not self._turn_ok(previous_point, from_point, to_point):
                return False

        return True

    def _turn_ok(self, prev: Point, curr: Point, nxt: Point) -> bool:
        """Vérifie que le virage n'est pas trop serré."""
        # Vecteurs
        v1 = (curr.x - prev.x, curr.y - prev.y, curr.z - prev.z)
        v2 = (nxt.x - curr.x, nxt.y - curr.y, nxt.z - curr.z)

        len1 = math.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)
        if len1 < 1e-9 or len2 < 1e-9:
            return True

        # Cosinus de l'angle
        dot = (v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]) / (len1 * len2)
        dot = max(-1.0, min(1.0, dot))
        angle_deg = math.degrees(math.acos(dot))

        # Si angle de virage > 90° et distance courte → refuser
        if angle_deg > 90.0 and len2 < self.min_turn_radius:
            return False
        return True

    def filter_neighbors(
        self,
        from_point: Point,
        candidates: list,
        previous_point: Optional[Point] = None,
    ) -> list:
        """Filtre une liste de voisins selon les contraintes."""
        return [
            n for n in candidates
            if self.is_valid_move(from_point, n, previous_point)
        ]

    def __repr__(self) -> str:
        return (
            f"PhysicsConstraints(slope={self.max_slope_deg}°, "
            f"jump={self.max_jump_height}, turn_r={self.min_turn_radius})"
        )


# Instance par défaut (humanoid raisonnable)
DEFAULT_PHYSICS = PhysicsConstraints(
    max_slope_deg=45.0,
    max_jump_height=1.5,
    min_turn_radius=0.0,   # désactivé par défaut pour ne pas casser les chemins existants
    allow_diagonal_up=True,
)

# Variante stricte (plus réaliste, chemins plus longs)
STRICT_PHYSICS = PhysicsConstraints(
    max_slope_deg=40.0,
    max_jump_height=1.2,
    min_turn_radius=1.0,
    allow_diagonal_up=True,
)

# Désactivé (comportement historique 26-connexité pure)
NO_PHYSICS = PhysicsConstraints(
    max_slope_deg=89.0,
    max_jump_height=100.0,
    min_turn_radius=0.0,
    allow_diagonal_up=True,
)
