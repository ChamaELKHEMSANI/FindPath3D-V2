# ui/main_window.py
"""Fenêtre principale PyQt5 + worker QThread + dashboard (Phase 1 : physique, backend, heuristique)."""
import traceback

from PyQt5.QtCore import QThread, pyqtSignal, Qt, QTimer
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QLabel, QComboBox, QLineEdit, QPushButton, QMessageBox, QApplication, QSlider
)

from core.point import Point
from core.generators import create_grid
from core.engines.a_star import AStar
from core.engines.dijkstra import Dijkstra
from core.engines.weighted_a_star import WeightedAStar
from core.engines.best_first import BestFirst
from core.engines.bfs import BFS
from core.engines.d_star import DStar
from core.engines.ara_star import ARAStar
from core.engines.genetic_program import GeneticProgram
from core.engines.time_astar import TimeAStar
from core.engines.theta_star import ThetaStar
from core.engines.jps import JPS
from core.dynamics import DynamicMap, make_demo_dynamic
from core.topology import detect_topology, select_heuristic
from core.physics import DEFAULT_PHYSICS, STRICT_PHYSICS, NO_PHYSICS
from core.heuristics import OctileHeuristic, ManhattanHeuristic, LandmarkHeuristic

from ui.map_draw import MapDraw
from ui.qt_observer import QtObserver
from ui.dashboard import DashboardWidget


SIZE_MIN, SIZE_MAX = 10, 80
OBSTACLE_MIN, OBSTACLE_MAX = 0, 200

# ---- Catalogues UI ----
GENERATOR_ITEMS = [
    ('random',  "Aléatoire"),
    ('maze',    "Labyrinthe"),
    ('caves',   "Cavernes"),
    ('rooms',   "Salles + couloirs"),
    ('pillars', "Piliers"),
    ('perlin',  "Perlin (terrain)"),
    ('bsp',     "BSP Donjon"),
    ('city',    "Ville urbaine"),
    ('floating',"Îles flottantes"),
]

ENGINE_ITEMS = [
    ('A-star',          "A*"),
    ('Dijkstra',        "Dijkstra"),
    ('WeightedA*',      "Weighted A*"),
    ('Best-First',      "Best-First"),
    ('BFS',             "BFS"),
    ('D-DStar',         "D*"),
    ('ARA-Star',        "ARA*"),
    ('GeneticProgram',  "Génétique"),
    ('TimeA*',          "TimeA* (dynamique)"),
    ('Theta*',         "Theta* (any-angle)"),
    ('JPS',            "JPS (jump points)"),
]

BACKEND_ITEMS = [
    ('grid',   "Grid (classique)"),
    ('octree', "Octree (Phase 1)"),
    ('dynamic',"Dynamic (Phase 2 démo)"),
]

PHYSICS_ITEMS = [
    ('none',   "Aucune (26-connexité)"),
    ('default',"DEFAULT (45° / jump 1.5)"),
    ('strict', "STRICT (plus réaliste)"),
]

HEURISTIC_ITEMS = [
    ('builtin',   "Intégrée (octile)"),
    ('octile',    "Octile explicite"),
    ('manhattan', "Manhattan"),
    ('landmark',  "Landmark (Dijkstra)"),
    ('adaptive',  "Adaptative (topologie)"),
]


class EngineWorker(QThread):
    engine_finished = pyqtSignal(bool, object)
    error = pyqtSignal(str)

    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine

    def run(self):
        try:
            # Référence optimale via A* silencieux (même physique si possible)
            if not isinstance(self.engine, AStar):
                phys = getattr(self.engine, "physics", None)
                ref = AStar(
                    self.engine.map,
                    self.engine.start_point,
                    self.engine.end_point,
                    physics=phys,
                )
                if ref.run():
                    self.engine.stats.optimal_cost = ref.stats.path_cost

            ok = self.engine.run()
            if isinstance(self.engine, AStar) and ok:
                self.engine.stats.optimal_cost = self.engine.stats.path_cost
        except Exception:
            self.error.emit(traceback.format_exc())
            ok = False

        self.engine_finished.emit(ok, self.engine.stats)


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.worker = None
        self.observer = None

        self.setWindowTitle("FindPath3D — Laboratoire (Phase 1)")
        self.setGeometry(100, 100, 1350, 820)

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ---------- Barre de paramètres ligne 1 ----------
        params = QHBoxLayout()

        params.addWidget(QLabel("Carte :"))
        self.generator_combo = QComboBox()
        for key, label in GENERATOR_ITEMS:
            self.generator_combo.addItem(label, key)
        params.addWidget(self.generator_combo)

        params.addWidget(QLabel("Moteur :"))
        self.combo = QComboBox()
        for key, label in ENGINE_ITEMS:
            self.combo.addItem(label, key)
        params.addWidget(self.combo)

        params.addWidget(QLabel("Taille :"))
        self.size_input = QLineEdit("15")
        self.size_input.setPlaceholderText(f"{SIZE_MIN}-{SIZE_MAX}")
        self.size_input.setFixedWidth(50)
        params.addWidget(self.size_input)

        params.addWidget(QLabel("Obstacle :"))
        self.obstacle_input = QLineEdit("10")
        self.obstacle_input.setPlaceholderText(f"{OBSTACLE_MIN}-{OBSTACLE_MAX}")
        self.obstacle_input.setFixedWidth(50)
        params.addWidget(self.obstacle_input)

        params.addWidget(QLabel("Graine :"))
        self.seed_input = QLineEdit("42")
        self.seed_input.setPlaceholderText("aléatoire")
        self.seed_input.setFixedWidth(60)
        params.addWidget(self.seed_input)

        params.addStretch(1)

        self.run_button = QPushButton("Exécuter")
        self.run_button.clicked.connect(self.RunIt)
        params.addWidget(self.run_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.StopIt)
        self.stop_button.setEnabled(False)
        params.addWidget(self.stop_button)

        self.quit_button = QPushButton("Quitter")
        self.quit_button.clicked.connect(self.endIt)
        params.addWidget(self.quit_button)

        root.addLayout(params)

        # ---------- Barre Phase 1 : backend / physique / heuristique ----------
        params2 = QHBoxLayout()

        params2.addWidget(QLabel("Backend :"))
        self.backend_combo = QComboBox()
        for key, label in BACKEND_ITEMS:
            self.backend_combo.addItem(label, key)
        params2.addWidget(self.backend_combo)

        params2.addWidget(QLabel("Physique :"))
        self.physics_combo = QComboBox()
        for key, label in PHYSICS_ITEMS:
            self.physics_combo.addItem(label, key)
        params2.addWidget(self.physics_combo)

        params2.addWidget(QLabel("Heuristique :"))
        self.heuristic_combo = QComboBox()
        for key, label in HEURISTIC_ITEMS:
            self.heuristic_combo.addItem(label, key)
        params2.addWidget(self.heuristic_combo)

        params2.addStretch(1)
        hint = QLabel("Phase 1 : physique + Octree + heuristiques injectables")
        hint.setStyleSheet("color: #666; font-style: italic;")
        params2.addWidget(hint)

        root.addLayout(params2)

        # ---------- Timeline Phase 4 ----------
        tl = QHBoxLayout()
        tl.addWidget(QLabel("Timeline chemin :"))
        self.timeline_slider = QSlider(Qt.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setEnabled(False)
        self.timeline_slider.valueChanged.connect(self._on_timeline)
        tl.addWidget(self.timeline_slider, stretch=1)
        self.btn_heatmap = QPushButton("Heatmap ON")
        self.btn_heatmap.setCheckable(True)
        self.btn_heatmap.setChecked(True)
        self.btn_heatmap.clicked.connect(self._toggle_heatmap)
        tl.addWidget(self.btn_heatmap)
        self.btn_replay = QPushButton("Rejouer chemin")
        self.btn_replay.clicked.connect(self._replay_path)
        self.btn_replay.setEnabled(False)
        tl.addWidget(self.btn_replay)
        root.addLayout(tl)

        # ---------- Navigation 3D ----------
        zoom = QHBoxLayout()
        zoom.addWidget(QLabel("Navigation 3D : molette = zoom"))

        view_buttons = [
            ("Iso", "iso"),
            ("Top", "top"),
            ("Face", "front"),
            ("Arriere", "back"),
            ("Gauche", "left"),
            ("Droite", "right"),
            ("Dessous", "bottom"),
        ]
        for label, view_name in view_buttons:
            btn = QPushButton(label)
            btn.clicked.connect(
                lambda checked=False, name=view_name: self.canvas.set_view(name)
            )
            zoom.addWidget(btn)

        self.btn_reset_zoom = QPushButton("Vue complete")
        self.btn_reset_zoom.clicked.connect(self._reset_zoom)
        self.btn_reset_zoom.setEnabled(False)
        zoom.addWidget(self.btn_reset_zoom)

        zoom.addStretch(1)
        root.addLayout(zoom)

        # ---------- Split : vue 3D | dashboard ----------
        splitter = QSplitter(Qt.Horizontal)
        self.canvas = MapDraw(self)
        splitter.addWidget(self.canvas)

        self.dashboard = DashboardWidget(self, max_history=100)
        self.dashboard.setMinimumWidth(400)
        splitter.addWidget(self.dashboard)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, stretch=1)

    # ------------------------------------------------------------------
    def _validate(self):
        try:
            size = int(self.size_input.text())
            obs = int(self.obstacle_input.text())
        except ValueError:
            QMessageBox.warning(self, "Erreur",
                                "Taille et obstacle doivent être des entiers.")
            return None
        if not (SIZE_MIN <= size <= SIZE_MAX):
            QMessageBox.warning(self, "Erreur",
                                f"Taille hors bornes ({SIZE_MIN}-{SIZE_MAX}).")
            return None
        if not (OBSTACLE_MIN <= obs <= OBSTACLE_MAX):
            QMessageBox.warning(self, "Erreur",
                                f"Obstacle hors bornes ({OBSTACLE_MIN}-{OBSTACLE_MAX}).")
            return None
        seed_text = self.seed_input.text().strip()
        seed = None
        if seed_text:
            try:
                seed = int(seed_text)
            except ValueError:
                QMessageBox.warning(self, "Erreur",
                                    "La graine doit être un entier.")
                return None
        return size, obs, seed

    def _resolve_physics(self):
        key = self.physics_combo.currentData()
        if key == "default":
            return DEFAULT_PHYSICS
        if key == "strict":
            return STRICT_PHYSICS
        return None  # none → comportement historique

    def _resolve_heuristic(self, map_obj):
        key = self.heuristic_combo.currentData()
        if key == "octile":
            return OctileHeuristic()
        if key == "manhattan":
            return ManhattanHeuristic()
        if key == "landmark":
            return LandmarkHeuristic(map_obj)
        if key == "adaptive":
            return select_heuristic(map_obj)
        return None  # builtin

    def _build_engine(self, key, map_obj, start, end, observer, physics, heuristic):
        # Moteurs qui supportent physics + heuristic (A*)
        if key == "A-star":
            return AStar(map_obj, start, end, observer,
                         physics=physics, heuristic=heuristic)
        # Moteurs qui supportent physics
        if key == "Dijkstra":
            return Dijkstra(map_obj, start, end, observer, physics=physics)
        if key == "WeightedA*":
            return WeightedAStar(map_obj, start, end, observer,
                                 weight=2.0, physics=physics)
        if key == "Best-First":
            return BestFirst(map_obj, start, end, observer, physics=physics)
        if key == "BFS":
            return BFS(map_obj, start, end, observer, physics=physics)
        # Moteurs legacy (pas encore de physics explicite)
        if key == "D-DStar":
            return DStar(map_obj, start, end, observer)
        if key == "ARA-Star":
            return ARAStar(map_obj, start, end, observer)
        if key == "GeneticProgram":
            return GeneticProgram(map_obj, start, end, observer,
                                  population_size=10, generations=50,
                                  mutation_rate=0.3)
        if key == "TimeA*":
            return TimeAStar(map_obj, start, end, observer,
                             physics=physics, t_start=0,
                             t_horizon=max(80, map_obj.size * 4))
        if key == "Theta*":
            return ThetaStar(map_obj, start, end, observer, physics=physics)
        if key == "JPS":
            return JPS(map_obj, start, end, observer, physics=physics)
        raise ValueError(f"Moteur inconnu : {key}")

    # ------------------------------------------------------------------
    def RunIt(self):
        if self.worker is not None:
            return
        params = self._validate()
        if params is None:
            return
        size, obs, seed = params

        self.pnt_src = Point(0, 0, 0, 0)
        self.pnt_dest = Point(size - 1, size - 1, size - 1, 0)

        generator = self.generator_combo.currentData()
        backend = self.backend_combo.currentData()
        if backend == "dynamic":
            base = create_grid(generator, size, obs, seed, backend="grid")
            self.map = make_demo_dynamic(base, seed=seed or 42)
        else:
            self.map = create_grid(generator, size, obs, seed, backend=backend)

        self.canvas.draw_map(self.map, self.pnt_src, self.pnt_dest)
        self.btn_reset_zoom.setEnabled(True)

        self.observer = QtObserver(self)
        self.observer.explore.connect(self._on_explore)
        self.observer.path_found.connect(self._on_path_found)
        self.observer.save_image.connect(self.canvas.SaveImage)
        self.observer.log.connect(print)
        self._last_path = []

        physics = self._resolve_physics()
        heuristic = self._resolve_heuristic(self.map)

        engine = self._build_engine(
            self.combo.currentData(),
            self.map, self.pnt_src, self.pnt_dest,
            self.observer, physics, heuristic,
        )

        # Annoter les stats
        engine.stats.extra["backend"] = backend
        if physics is not None:
            engine.stats.extra["physics"] = repr(physics)
        if heuristic is not None:
            engine.stats.extra["heuristic"] = repr(heuristic)

        self.worker = EngineWorker(engine, self)
        self.worker.engine_finished.connect(self._on_finished)
        self.worker.error.connect(self._on_error)

        self.run_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.worker.start()

    def StopIt(self):
        if self.observer is not None:
            self.observer.stop()

    def endIt(self):
        if self.observer is not None:
            self.observer.stop()
        if self.worker is not None:
            self.worker.wait(2000)
        QApplication.quit()

    # ------------------------------------------------------------------
    def _on_explore(self, point, alpha):
        self.canvas.draw_explore_point(point, alpha=alpha)

    def _reset_zoom(self):
        self.canvas.reset_zoom()

    def _on_path_found(self, path):
        self._last_path = list(path) if path else []

    def _on_timeline(self, value):
        if getattr(self, "_last_path", None):
            self.canvas.redraw_full(path_up_to=value)

    def _toggle_heatmap(self, checked):
        self.canvas._heatmap_enabled = bool(checked)
        self.btn_heatmap.setText("Heatmap ON" if checked else "Heatmap OFF")
        path_up_to = self.timeline_slider.value() if self.timeline_slider.isEnabled() else None
        self.canvas.redraw_full(path_up_to=path_up_to)

    def _replay_path(self):
        path = getattr(self, "_last_path", []) or []
        if not path:
            return
        self.timeline_slider.setValue(1)

        def advance():
            if self.timeline_slider.value() >= len(path):
                return
            self.timeline_slider.setValue(self.timeline_slider.value() + 1)
            QTimer.singleShot(60, advance)

        QTimer.singleShot(60, advance)

    def _on_finished(self, ok, stats):
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None
        self.dashboard.add_run(stats)
        print(stats)
        # Phase 4 : récupérer le chemin si disponible
        path = []
        if self.worker is None:
            pass
        # path via observer signal path_found stocké
        path = getattr(self, "_last_path", []) or []
        if path:
            self.canvas.set_path(path)
            self.timeline_slider.setMaximum(max(1, len(path)))
            self.timeline_slider.setValue(len(path))
            self.timeline_slider.setEnabled(True)
            self.btn_replay.setEnabled(True)
            self.canvas.redraw_full(path_up_to=len(path))

    def _on_error(self, message):
        QMessageBox.critical(self, "Erreur moteur", message)
        self.run_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.worker = None

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        self.endIt()
        super().closeEvent(event)
