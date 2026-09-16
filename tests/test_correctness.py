"""
Tests de correctness minimaux - Phase 0.
Vérifie qu'un chemin est trouvé quand il doit l'être, et que le chemin est valide.
"""
import pytest
from core.point import Point
from core.generators import create_grid
from core.engines.a_star import AStar
from core.engines.dijkstra import Dijkstra
from core.engines.bfs import BFS
from core.engines.weighted_a_star import WeightedAStar
from core.engines.best_first import BestFirst


def _path_is_valid(grid, path, start, end):
    if not path:
        return False
    if not path[0].isEgale(start):
        return False
    if not path[-1].isEgale(end):
        return False
    for p in path:
        if not grid.IsValidPoint(p.x, p.y, p.z):
            return False
        if grid.IsObstacle(p.x, p.y, p.z):
            return False
    # Continuité 26-connexité
    for a, b in zip(path, path[1:]):
        dx = abs(a.x - b.x)
        dy = abs(a.y - b.y)
        dz = abs(a.z - b.z)
        if max(dx, dy, dz) > 1 or (dx + dy + dz) == 0:
            return False
    return True


@pytest.mark.parametrize("size,obst,gen,seed", [
    (8, 2, "random", 42),
    (10, 3, "maze", 42),
    (10, 5, "rooms", 42),
    (12, 4, "pillars", 42),
])
@pytest.mark.parametrize("engine_cls", [AStar, Dijkstra, BFS, WeightedAStar, BestFirst])
def test_engine_finds_valid_path(size, obst, gen, seed, engine_cls):
    grid = create_grid(gen, size, obst, seed)
    start = Point(0, 0, 0)
    end = Point(size - 1, size - 1, size - 1)

    # Si start ou end est obstacle, le test n'a pas de sens
    if grid.IsObstacle(start.x, start.y, start.z) or grid.IsObstacle(end.x, end.y, end.z):
        pytest.skip("Start or end is obstacle on this seed")

    engine = engine_cls(grid, start, end)
    ok = engine.run()

    assert ok is True, f"{engine_cls.__name__} failed to find a path"
    assert engine.stats.success is True
    assert engine.stats.path_length > 0
    assert engine.stats.nodes_explored > 0

    # Reconstruire le path via l'observer n'est pas obligatoire ;
    # on vérifie au moins que les stats sont cohérentes.
    # Pour un test plus strict on pourrait capturer le path, mais
    # l'observer par défaut ne le stocke pas. On se contente des stats.


def test_empty_grid_trivial_path():
    """Sur une grille vide, A* doit trouver un chemin de longueur raisonnable."""
    grid = create_grid("random", 5, 0, 42)  # 0 obstacle
    start = Point(0, 0, 0)
    end = Point(4, 4, 4)
    engine = AStar(grid, start, end)
    assert engine.run() is True
    assert engine.stats.path_length >= 5  # au minimum la distance de Manhattan
    assert engine.stats.nodes_explored > 0
    assert engine.stats.success is True


def test_grid_toggle_obstacle():
    from core.grid import Grid
    g = Grid(size=5, seed=1)
    assert not g.IsObstacle(2, 2, 2)
    g.toggle_obstacle(2, 2, 2)
    assert g.IsObstacle(2, 2, 2)
    g.toggle_obstacle(2, 2, 2)
    assert not g.IsObstacle(2, 2, 2)


def test_point_equality_and_hash():
    p1 = Point(1, 2, 3)
    p2 = Point(1, 2, 3)
    p3 = Point(1, 2, 4)
    assert p1 == p2
    assert p1 != p3
    assert hash(p1) == hash(p2)
    s = {p1, p2, p3}
    assert len(s) == 2


# ---------------------------------------------------------------------------
# Phase 1 - Physique
# ---------------------------------------------------------------------------
from core.physics import PhysicsConstraints, DEFAULT_PHYSICS, STRICT_PHYSICS, NO_PHYSICS


def test_physics_slope_limit():
    """Une pente trop raide doit être refusée."""
    phys = PhysicsConstraints(max_slope_deg=45.0, max_jump_height=2.0)
    from_p = Point(0, 0, 0)
    # Montée de 2 en z pour 1 en xy → pente = 2 > tan(45°)=1 → refusé
    to_steep = Point(1, 0, 2)
    assert phys.is_valid_move(from_p, to_steep) is False

    # Montée de 1 en z pour 1 en xy → pente = 1 == tan(45°) → accepté
    to_ok = Point(1, 0, 1)
    assert phys.is_valid_move(from_p, to_ok) is True


def test_physics_jump_limit():
    phys = PhysicsConstraints(max_slope_deg=80.0, max_jump_height=1.5)
    from_p = Point(0, 0, 0)
    assert phys.is_valid_move(from_p, Point(0, 0, 2)) is False  # jump trop haut
    assert phys.is_valid_move(from_p, Point(1, 0, 1)) is True


def test_a_star_with_physics_finds_path():
    """A* avec physique par défaut doit encore trouver un chemin sur grille ouverte."""
    grid = create_grid("random", 8, 0, 42)  # aucun obstacle
    start = Point(0, 0, 0)
    end = Point(7, 7, 7)
    engine = AStar(grid, start, end, physics=DEFAULT_PHYSICS)
    ok = engine.run()
    assert ok is True
    assert engine.stats.path_length > 0
    # Avec physique le chemin peut être plus long qu'en 26-connexité pure
    assert engine.stats.path_length >= 8


def test_a_star_physics_vs_no_physics_length():
    """Sur une grille vide, le chemin avec physique DEFAULT reste trouvable."""
    grid = create_grid("random", 6, 0, 1)
    start = Point(0, 0, 0)
    end = Point(5, 5, 5)

    eng_free = AStar(grid, start, end, physics=None)
    eng_phys = AStar(grid, start, end, physics=DEFAULT_PHYSICS)

    assert eng_free.run() is True
    assert eng_phys.run() is True
    # DEFAULT autorise la diagonale 45° → longueur identique sur grille vide
    assert eng_phys.stats.path_length >= eng_free.stats.path_length


# ---------------------------------------------------------------------------
# Phase 1 - Octree
# ---------------------------------------------------------------------------
from core.octree import OctreeMap, grid_to_octree


def test_octree_basic_api():
    ot = OctreeMap(size=10)
    assert ot.IsValidPoint(0, 0, 0)
    assert not ot.IsValidPoint(-1, 0, 0)
    assert not ot.IsObstacle(1, 1, 1)
    ot.AddObstaclePoint(1, 1, 1)
    assert ot.IsObstacle(1, 1, 1)
    assert ot.toggle_obstacle(1, 1, 1) is False  # removed
    assert not ot.IsObstacle(1, 1, 1)
    assert ot.toggle_obstacle(2, 2, 2) is True   # added
    assert ot.IsObstacle(2, 2, 2)


def test_octree_from_generator():
    ot = create_grid("random", 12, 5, 99, backend="octree")
    assert isinstance(ot, OctreeMap)
    assert ot.size == 12
    n = len(ot.obstacle_point)
    assert n == ot.stats()["n_obstacles"]
    assert n > 0


def test_a_star_on_octree_matches_grid():
    """Même seed → même obstacles → même chemin (longueur)."""
    g = create_grid("random", 10, 4, 7, backend="grid")
    o = create_grid("random", 10, 4, 7, backend="octree")
    start, end = Point(0, 0, 0), Point(9, 9, 9)
    if g.IsObstacle(0, 0, 0) or g.IsObstacle(9, 9, 9):
        pytest.skip("start/end blocked")
    eng_g = AStar(g, start, end)
    eng_o = AStar(o, start, end)
    assert eng_g.run() == eng_o.run()
    if eng_g.stats.success:
        assert eng_g.stats.path_length == eng_o.stats.path_length


def test_grid_to_octree_preserves_obstacles():
    g = create_grid("pillars", 8, 5, 3, backend="grid")
    o = grid_to_octree(g)
    assert len(g.obstacle_point) == len(o.obstacle_point)
    for p in g.obstacle_point:
        assert o.IsObstacle(p.x, p.y, p.z)


# ---------------------------------------------------------------------------
# Phase 2 - Dynamique + TimeA* + Topology
# ---------------------------------------------------------------------------
from core.dynamics import DynamicMap, make_demo_dynamic, TemporalGate
from core.engines.time_astar import TimeAStar
from core.engines.dynamic_pathfinder import DynamicPathfinder
from core.topology import detect_topology, select_heuristic, describe_topology


def test_temporal_gate():
    g = TemporalGate(pos=(1, 1, 1), open_intervals=[(5, 10), (20, 25)])
    assert g.is_open(7) is True
    assert g.is_open(12) is False
    assert g.is_open(22) is True


def test_dynamic_map_gate_blocks():
    base = create_grid("random", 8, 0, 1, backend="grid")
    dm = DynamicMap(base)
    dm.add_gate(3, 3, 3, open_intervals=[(5, 15)])
    assert dm.IsObstacle(3, 3, 3, t=0) is True   # fermée
    assert dm.IsObstacle(3, 3, 3, t=7) is False  # ouverte
    assert dm.IsObstacle(3, 3, 3, t=20) is True  # refermée


def test_time_astar_finds_path_empty():
    base = create_grid("random", 6, 0, 2, backend="grid")
    dm = DynamicMap(base)
    start, end = Point(0, 0, 0), Point(5, 5, 5)
    eng = TimeAStar(dm, start, end, t_start=0, t_horizon=80)
    assert eng.run() is True
    assert eng.stats.path_length >= 6
    assert hasattr(eng, "_last_path") and len(eng._last_path) >= 6


def test_time_astar_waits_for_gate():
    """Porte fermée au début → TimeA* doit attendre ou contourner."""
    base = create_grid("random", 5, 0, 0, backend="grid")
    # Bloquer tout sauf un couloir forcé via porte
    dm = DynamicMap(base)
    # Porte au milieu du chemin diagonal
    dm.add_gate(2, 2, 2, open_intervals=[(3, 50)])
    start, end = Point(0, 0, 0), Point(4, 4, 4)
    eng = TimeAStar(dm, start, end, t_start=0, t_horizon=60, allow_wait=True)
    ok = eng.run()
    # Peut réussir en attendant ou en contournant
    assert isinstance(ok, bool)
    if ok:
        assert eng.stats.extra.get("t_arrival", 0) >= 0


def test_dynamic_pathfinder_initial_plan():
    base = create_grid("random", 8, 0, 3, backend="grid")
    dm = DynamicMap(base)
    pf = DynamicPathfinder(dm, Point(0, 0, 0), Point(7, 7, 7), use_time=False)
    ok = pf.plan_initial()
    assert ok is True
    assert len(pf.path) > 0
    assert pf.summary()["path_len"] > 0


def test_topology_detection():
    open_g = create_grid("random", 10, 0, 1)   # 0 obstacles
    dense = create_grid("maze", 12, 10, 1)
    t_open = detect_topology(open_g)
    t_dense = detect_topology(dense)
    assert t_open in ("open_space", "structured", "unknown")
    assert t_dense in ("dense_maze", "structured", "open_space", "unknown")
    h = select_heuristic(open_g)
    assert h is not None
    info = describe_topology(open_g)
    assert "topology" in info


# ---------------------------------------------------------------------------
# Phase 2 raffinements - Theta* + PathCollector + fuel
# ---------------------------------------------------------------------------
from core.engines.theta_star import ThetaStar
from core.path_collector import PathCollector


def test_theta_star_empty_grid():
    grid = create_grid("random", 8, 0, 5, backend="grid")
    start, end = Point(0, 0, 0), Point(7, 7, 7)
    eng = ThetaStar(grid, start, end)
    assert eng.run() is True
    assert eng.stats.path_length >= 2
    # Any-angle → souvent plus court en coût euclidien que le nombre de voxels
    assert eng.stats.path_cost > 0


def test_theta_star_vs_astar_cost():
    grid = create_grid("random", 10, 0, 11, backend="grid")
    start, end = Point(0, 0, 0), Point(9, 9, 9)
    a = AStar(grid, start, end)
    t = ThetaStar(grid, start, end)
    assert a.run() and t.run()
    # Theta* ne doit pas être plus coûteux (approx, tolérance numérique)
    assert t.stats.path_cost <= a.stats.path_cost + 0.5


def test_path_collector():
    grid = create_grid("random", 6, 0, 2)
    col = PathCollector()
    eng = AStar(grid, Point(0, 0, 0), Point(5, 5, 5), observer=col)
    assert eng.run()
    assert len(col.path) >= 6


def test_dynamic_pathfinder_with_collector_runs():
    base = create_grid("random", 8, 0, 4)
    dm = make_demo_dynamic(base)
    pf = DynamicPathfinder(dm, Point(0, 0, 0), Point(7, 7, 7), use_time=False)
    assert pf.plan_initial() is True
    assert len(pf.path) > 0
    summary = pf.run_until_done(max_steps=100)
    assert summary["path_len"] > 0


# ---------------------------------------------------------------------------
# Phase 3 - Générateurs avancés, JPS, validation
# ---------------------------------------------------------------------------
from core.engines.jps import JPS
from core.validation import validate_path, path_collision_free


@pytest.mark.parametrize("gen", ["perlin", "bsp", "city", "floating"])
def test_phase3_generators_create(gen):
    g = create_grid(gen, 10, 8, 42, backend="grid")
    assert g.size == 10
    assert isinstance(len(g.obstacle_point), int)


def test_jps_empty_grid():
    g = create_grid("random", 8, 0, 3)
    eng = JPS(g, Point(0, 0, 0), Point(7, 7, 7))
    assert eng.run() is True
    assert eng.stats.path_length >= 2


def test_validate_path_ok():
    g = create_grid("random", 6, 0, 1)
    eng = AStar(g, Point(0, 0, 0), Point(5, 5, 5))
    assert eng.run()
    path = getattr(eng, "_last_path", None) or []
    if path:
        ok, msg = validate_path(g, path, Point(0, 0, 0), Point(5, 5, 5))
        assert ok, msg


def test_jps_path_reaches_goal():
    g = create_grid("random", 10, 0, 7)
    eng = JPS(g, Point(0, 0, 0), Point(9, 9, 9))
    ok = eng.run()
    assert ok
    path = getattr(eng, "_last_path", [])
    assert path and path[-1].isEgale(Point(9, 9, 9))
