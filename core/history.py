# core/history.py
"""Historique en mémoire des exécutions (FIFO borné)."""
from typing import List, Optional
from core.stats import EngineStats


class StatsHistory:
    def __init__(self, max_size: int = 100):
        self._items: List[EngineStats] = []
        self._max = max(1, max_size)

    def add(self, stats: EngineStats) -> None:
        self._items.append(stats)
        if len(self._items) > self._max:
            # On retire les plus anciens
            self._items = self._items[-self._max:]

    def clear(self) -> None:
        self._items.clear()

    def items(self) -> List[EngineStats]:
        return list(self._items)

    def last(self) -> Optional[EngineStats]:
        return self._items[-1] if self._items else None

    def best_optimality(self) -> Optional[EngineStats]:
        """Meilleure optimalité parmi les runs réussis."""
        candidates = [s for s in self._items
                      if s.success and s.optimality is not None]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.optimality)

    def fastest(self) -> Optional[EngineStats]:
        candidates = [s for s in self._items if s.success]
        if not candidates:
            return None
        return min(candidates, key=lambda s: s.elapsed)

    def __len__(self) -> int:
        return len(self._items)