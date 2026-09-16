# ui/qt_observer.py
"""
Bridge entre le moteur (thread worker) et l'UI (thread principal).
Émet des signaux Qt ; matplotlib ne s'exécute que dans le thread GUI.

Note : on n'hérite PAS de GridObserver (ABC) pour éviter le conflit
de métaclasses avec QObject. On respecte simplement le même contrat.
"""
from PyQt5.QtCore import QObject, pyqtSignal, QThread


class QtObserver(QObject):
    map_ready = pyqtSignal(object, object, object)
    explore = pyqtSignal(object, float)
    path_found = pyqtSignal(object)
    save_image = pyqtSignal()
    log = pyqtSignal(str)

    def __init__(self, parent=None, yield_ms=0, emit_every=25):
        super().__init__(parent)
        self._running = True
        self._yield_ms = yield_ms
        self._emit_every = max(1, int(emit_every))
        self._explore_count = 0

    # ---- API appelée par le moteur (thread worker) ----
    def is_running(self) -> bool:
        return self._running

    def stop(self):
        self._running = False

    def on_map_ready(self, map_obj, start, end):
        self.map_ready.emit(map_obj, start, end)

    def on_explore(self, point, alpha=0.1):
        self._explore_count += 1
        if self._explore_count % self._emit_every == 0:
            self.explore.emit(point, float(alpha))
            # Yield optionnel : laisse le thread principal traiter les signaux.
            if self._yield_ms > 0:
                QThread.msleep(self._yield_ms)

    def on_path(self, path):
        self.path_found.emit(list(path))

    def on_save_image(self):
        self.save_image.emit()

    def on_log(self, message):
        self.log.emit(message)
