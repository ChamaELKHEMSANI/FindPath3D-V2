# ui/dashboard.py
"""Widget de tableau de bord : résumé + historique des exécutions."""
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor, QBrush, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QGroupBox, QGridLayout
)

from core.history import StatsHistory


COLUMNS = ["#", "Moteur", "OK", "Temps (s)", "Nœuds",
           "Longueur", "Coût", "Optimalité", "Carte"]


class DashboardWidget(QWidget):

    def __init__(self, parent=None, max_history=100):
        super().__init__(parent)
        self.history = StatsHistory(max_size=max_history)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # ---------------- Résumé ----------------
        summary_box = QGroupBox("Résumé")
        summary = QGridLayout(summary_box)
        summary.setHorizontalSpacing(12)
        summary.setVerticalSpacing(4)

        self.lbl_runs = QLabel("0")
        self.lbl_last = QLabel("—")
        self.lbl_fastest = QLabel("—")
        self.lbl_best_opt = QLabel("—")

        bold = QFont()
        bold.setBold(True)
        for lbl in (self.lbl_runs, self.lbl_last, self.lbl_fastest, self.lbl_best_opt):
            lbl.setFont(bold)

        summary.addWidget(QLabel("Exécutions :"), 0, 0)
        summary.addWidget(self.lbl_runs, 0, 1)
        summary.addWidget(QLabel("Dernière :"), 1, 0)
        summary.addWidget(self.lbl_last, 1, 1)
        summary.addWidget(QLabel("Plus rapide :"), 2, 0)
        summary.addWidget(self.lbl_fastest, 2, 1)
        summary.addWidget(QLabel("Meilleure optimalité :"), 3, 0)
        summary.addWidget(self.lbl_best_opt, 3, 1)

        layout.addWidget(summary_box)

        # ---------------- Table ----------------
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        for i in range(len(COLUMNS) - 1):
            self.table.horizontalHeader().setSectionResizeMode(
                i, QHeaderView.ResizeToContents)
        layout.addWidget(self.table, stretch=1)

        # ---------------- Boutons ----------------
        buttons = QHBoxLayout()
        self.btn_clear = QPushButton("Effacer l'historique")
        self.btn_clear.clicked.connect(self.clear)
        buttons.addWidget(self.btn_clear)
        self.btn_export = QPushButton("Exporter CSV")
        self.btn_export.clicked.connect(self.export_csv)
        buttons.addWidget(self.btn_export)
        buttons.addStretch(1)
        layout.addLayout(buttons)

    # ------------------------------------------------------------------
    def add_run(self, stats):
        self.history.add(stats)
        self._refresh()

    def clear(self):
        self.history.clear()
        self._refresh()

    # ------------------------------------------------------------------
    def _refresh(self):
        self._refresh_summary()
        self._refresh_table()

    def _refresh_summary(self):
        runs = len(self.history)
        self.lbl_runs.setText(str(runs))

        last = self.history.last()
        if last is None:
            self.lbl_last.setText("—")
        else:
            self.lbl_last.setText(
                f"{last.engine} · {last.elapsed:.3f}s · "
                f"{last.path_length} voxels"
            )

        fastest = self.history.fastest()
        self.lbl_fastest.setText(
            "—" if fastest is None
            else f"{fastest.engine} · {fastest.elapsed:.3f}s"
        )

        best = self.history.best_optimality()
        if best is None or best.optimality is None:
            self.lbl_best_opt.setText("—")
        else:
            self.lbl_best_opt.setText(
                f"{best.engine} · {best.optimality*100:.1f}%"
            )

    def _refresh_table(self):
        items = self.history.items()
        self.table.setRowCount(len(items))

        green = QBrush(QColor(220, 255, 220))
        red = QBrush(QColor(255, 220, 220))
        amber = QBrush(QColor(255, 245, 210))

        for row, s in enumerate(items):
            cells = [
                str(row + 1),
                s.engine,
                "✓" if s.success else "✗",
                f"{s.elapsed:.3f}",
                str(s.nodes_explored),
                str(s.path_length),
                f"{s.path_cost:.3f}" if s.path_cost > 0 else "—",
                (f"{s.optimality*100:.1f}%" if s.optimality is not None else "—"),
                self._card_label(s),
            ]
            for col, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if col >= 3:
                    item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                if not s.success and col == 2:
                    item.setBackground(red)
                elif s.success and col == 2:
                    item.setBackground(green)
                elif col == 7 and s.optimality is not None:
                    # Colore l'optimalité : vert si optimal, orange sinon
                    if s.optimality <= 1.001:
                        item.setBackground(green)
                    else:
                        item.setBackground(amber)
                self.table.setItem(row, col, item)

        # Auto-scroll sur la dernière ligne
        if items:
            self.table.scrollToBottom()

    @staticmethod
    def _card_label(s):
        seed = "aléa" if s.seed is None else f"seed={s.seed}"
        return f"{s.size}³ / {s.obstacle} / {seed}"

    def export_csv(self):
        """Exporte l'historique vers un fichier CSV."""
        import csv
        from PyQt5.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Exporter historique", "findpath3d_history.csv", "CSV (*.csv)"
        )
        if not path:
            return
        data = self.history.items()
        if not data:
            return
        fieldnames = [
            "engine", "success", "elapsed", "nodes_explored",
            "path_length", "path_cost", "optimal_cost", "size", "obstacle", "seed",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for s in data:
                d = s.to_dict() if hasattr(s, "to_dict") else {
                    "engine": s.engine, "success": s.success, "elapsed": s.elapsed,
                    "nodes_explored": s.nodes_explored, "path_length": s.path_length,
                    "path_cost": s.path_cost, "optimal_cost": s.optimal_cost,
                    "size": s.size, "obstacle": s.obstacle, "seed": s.seed,
                }
                w.writerow({k: d.get(k, "") for k in fieldnames})
