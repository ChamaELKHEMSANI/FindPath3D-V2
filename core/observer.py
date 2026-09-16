# core/observer.py
"""
Interface d'observation. Un moteur notifie son observer au fil de l'exécution.
Classe "normale" (pas d'ABC) pour rester compatible avec QObject (PyQt).
"""


class GridObserver:
    """Observer neutre : ne fait rien. Base pour tous les autres."""

    def is_running(self) -> bool:
        return True

    def on_map_ready(self, map_obj, start, end):
        pass

    def on_explore(self, point, alpha=0.1):
        pass

    def on_path(self, path):
        pass

    def on_save_image(self):
        pass

    def on_log(self, message):
        pass