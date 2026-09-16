# core/dynamics.py
"""
Dynamique environnementale - Phase 2.
Portes temporelles, zones de danger, ressources.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from core.point import Point


@dataclass
class TemporalGate:
    """
    Portail qui n'est praticable que pendant certains intervalles de temps.
    open_intervals : liste de (t_start, t_end) inclusifs.
    """
    pos: Tuple[int, int, int]
    open_intervals: List[Tuple[int, int]] = field(default_factory=list)

    def is_open(self, t: int) -> bool:
        for a, b in self.open_intervals:
            if a <= t <= b:
                return True
        return False


@dataclass
class DangerZone:
    """
    Zone dont le coût (ou le statut obstacle) augmente avec le temps.
    Si danger_level(t) > threshold → considéré comme obstacle.
    """
    pos: Tuple[int, int, int]
    # niveau de base + croissance par unité de temps
    base: float = 0.0
    growth: float = 0.1
    threshold: float = 1.0

    def danger_level(self, t: int) -> float:
        return self.base + self.growth * max(0, t)

    def is_blocked(self, t: int) -> bool:
        return self.danger_level(t) > self.threshold




@dataclass
class FuelStation:
    """Point de recharge : restaure `amount` unités de carburant."""
    pos: Tuple[int, int, int]
    amount: float = 20.0


@dataclass
class MobileObstacle:
    """
    Obstacle qui occupe une position à certains instants (patrouille simplifiée).
    schedule : dict time -> (x,y,z) ou liste de positions cycliques.
    """
    path: List[Tuple[int, int, int]]  # cycle de positions
    period: int = 1  # durée d'occupation de chaque case

    def position_at(self, t: int) -> Tuple[int, int, int]:
        if not self.path:
            return (-1, -1, -1)
        idx = (t // max(1, self.period)) % len(self.path)
        return self.path[idx]

    def occupies(self, x: int, y: int, z: int, t: int) -> bool:
        return self.position_at(t) == (x, y, z)


class DynamicMap:
    """
    Enveloppe autour d'une Grid / OctreeMap ajoutant la dimension temps.
    API :
      IsObstacle(x, y, z, t=0)
      IsValidPoint(x, y, z)  (délègue)
      size, obstacle_point, etc. (délègue)
    """

    def __init__(self, base_map):
        self.base = base_map
        self.size = base_map.size
        self.seed = getattr(base_map, "seed", None)
        self.generator = getattr(base_map, "generator", "dynamic")
        self.obstacle = getattr(base_map, "obstacle", 0)
        self.rng = getattr(base_map, "rng", None)

        self.gates: Dict[Tuple[int, int, int], TemporalGate] = {}
        self.danger_zones: Dict[Tuple[int, int, int], DangerZone] = {}
        self.mobile: List[MobileObstacle] = []
        self.static_blocked: Set[Tuple[int, int, int]] = set()
        self.fuel_stations: Dict[Tuple[int, int, int], FuelStation] = {}
        self.initial_fuel: float = 1e9  # illimité par défaut

        # Pour compatibilité avec le code qui lit obstacle_point
        self._obstacle_cache: Optional[List[Point]] = None

    # ----- délégation -----
    def IsValidPoint(self, x, y, z) -> bool:
        return self.base.IsValidPoint(x, y, z)

    def AddObstaclePoint(self, x, y, z):
        self.base.AddObstaclePoint(x, y, z)
        self._obstacle_cache = None

    def clear(self):
        self.base.clear()
        self.gates.clear()
        self.danger_zones.clear()
        self.mobile.clear()
        self.static_blocked.clear()
        self._obstacle_cache = None

    def clear_obstacles(self):
        self.clear()

    @property
    def obstacle_point(self):
        if self._obstacle_cache is None:
            pts = list(getattr(self.base, "obstacle_point", []))
            self._obstacle_cache = pts
        return self._obstacle_cache

    # ----- dynamique -----
    def add_gate(self, x: int, y: int, z: int, open_intervals: List[Tuple[int, int]]):
        key = (int(x), int(y), int(z))
        self.gates[key] = TemporalGate(pos=key, open_intervals=open_intervals)

    def add_danger_zone(self, x: int, y: int, z: int,
                        base: float = 0.0, growth: float = 0.15, threshold: float = 1.0):
        key = (int(x), int(y), int(z))
        self.danger_zones[key] = DangerZone(
            pos=key, base=base, growth=growth, threshold=threshold
        )

    def add_mobile(self, path: List[Tuple[int, int, int]], period: int = 1):
        self.mobile.append(MobileObstacle(path=path, period=period))

    def add_fuel_station(self, x: int, y: int, z: int, amount: float = 20.0):
        key = (int(x), int(y), int(z))
        self.fuel_stations[key] = FuelStation(pos=key, amount=amount)

    def fuel_at(self, x: int, y: int, z: int) -> float:
        key = (int(x), int(y), int(z))
        if key in self.fuel_stations:
            return self.fuel_stations[key].amount
        return 0.0

    def IsObstacle(self, i, j, k, t: int = 0) -> bool:
        """Obstacle statique OU dynamique au temps t."""
        key = (int(i), int(j), int(k))

        # Statique de base
        if self.base.IsObstacle(i, j, k):
            return True

        # Portes : fermées = obstacle
        if key in self.gates:
            if not self.gates[key].is_open(t):
                return True
            # ouverte → on laisse passer (sauf autre règle)

        # Zones de danger
        if key in self.danger_zones:
            if self.danger_zones[key].is_blocked(t):
                return True

        # Obstacles mobiles
        for m in self.mobile:
            if m.occupies(i, j, k, t):
                return True

        if key in self.static_blocked:
            return True

        return False

    def is_path_still_valid(self, path: List[Point], t_start: int = 0) -> bool:
        """Vérifie si un chemin reste praticable en avançant d'un pas par unité de temps."""
        t = t_start
        for p in path:
            if not self.IsValidPoint(p.x, p.y, p.z):
                return False
            if self.IsObstacle(p.x, p.y, p.z, t):
                return False
            t += 1
        return True

    def stats_dynamic(self) -> dict:
        return {
            "gates": len(self.gates),
            "danger_zones": len(self.danger_zones),
            "mobile": len(self.mobile),
            "base_type": type(self.base).__name__,
        }

    def __repr__(self) -> str:
        s = self.stats_dynamic()
        return (
            f"DynamicMap(base={s['base_type']}, gates={s['gates']}, "
            f"danger={s['danger_zones']}, mobile={s['mobile']})"
        )


def make_demo_dynamic(base_map, seed: int = 42) -> DynamicMap:
    """
    Scénario démo riche :
    - porte centrale (ouverte t=5..25 et 40..60)
    - zone de danger progressive
    - patrouille mobile
    - station de carburant (si size >= 10)
    """
    dm = DynamicMap(base_map)
    s = base_map.size
    mid = s // 2

    dm.add_gate(mid, mid, mid, open_intervals=[(5, 25), (40, 60)])

    if s > 6:
        dm.add_danger_zone(mid + 1, mid, mid, base=0.0, growth=0.2, threshold=1.0)
        dm.add_danger_zone(mid, mid + 1, mid, base=0.3, growth=0.1, threshold=1.2)

    if s > 8:
        path = [(2, 2, 2), (3, 2, 2), (4, 2, 2), (4, 3, 2), (3, 3, 2), (2, 3, 2)]
        dm.add_mobile(path, period=2)

    if s >= 10:
        dm.add_fuel_station(mid // 2, mid // 2, mid // 2, amount=25.0)
        dm.initial_fuel = max(30.0, s * 1.5)

    return dm
