# core/stats.py
"""Statistiques d'exécution d'un moteur."""
from dataclasses import dataclass, field, asdict
import json


@dataclass
class EngineStats:
    engine: str = ""
    elapsed: float = 0.0
    nodes_explored: int = 0
    path_length: int = 0
    path_cost: float = 0.0
    optimal_cost: float = 0.0         # 0.0 = inconnu
    success: bool = False
    size: int = 0
    obstacle: int = 0
    seed: int = None                  # None = aléatoire
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    @property
    def optimality(self):
        """Ratio coût / optimal. 1.0 = optimal. None si non calculable."""
        if not self.success or self.optimal_cost <= 0 or self.path_cost <= 0:
            return None
        return self.path_cost / self.optimal_cost

    # ------------------------------------------------------------------
    def to_dict(self):
        d = asdict(self)
        d['optimality'] = self.optimality
        return d

    def to_json(self):
        return json.dumps(self.to_dict(), indent=2)

    def __str__(self):
        status = "OK" if self.success else "ÉCHEC"
        opt = self.optimality
        opt_str = f"{opt*100:.1f}%" if opt is not None else "n/a"
        return (f"[{self.engine}] {status} | "
                f"temps={self.elapsed:.3f}s | "
                f"nœuds={self.nodes_explored} | "
                f"longueur={self.path_length} | "
                f"coût={self.path_cost:.3f} | "
                f"optimalité={opt_str}")