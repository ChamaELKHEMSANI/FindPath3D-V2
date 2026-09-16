# FindPath3D

**Laboratoire interactif de pathfinding 3D** — Python · PyQt5 · matplotlib

FindPath3D est une application pour étudier, comparer et visualiser des algorithmes de recherche de chemin dans des grilles voxel 3D. Elle combine un moteur de simulation extensible, des générateurs de cartes variés, des contraintes physiques, des environnements dynamiques et une interface graphique moderne.

---

## Fonctionnalités

### Algorithmes
| Algorithme | Description |
|------------|-------------|
| **A\*** | Recherche heuristique classique (optimal si heuristique admissible) |
| **Dijkstra** | Plus court chemin sans heuristique |
| **Weighted A\*** | A\* pondéré (plus rapide, sous-optimal) |
| **Best-First** | Recherche gloutonne (priorité = *h*) |
| **BFS** | Parcours en largeur |
| **D\*** | Replanification orientée dynamique |
| **ARA\*** | Anytime Repairing A\* |
| **Genetic** | Approche évolutionnaire expérimentale |
| **Theta\*** | Any-angle (lignes de vue, chemins plus fluides) |
| **JPS** | Jump Point Search 3D (réduction des symétries) |
| **TimeA\*** | A\* spatio-temporel (obstacles dépendant du temps) |

### Représentation spatiale
- **Grid** — grille classique (ensemble d'obstacles)
- **Octree** — structure hiérarchique sparse (API compatible Grid)
- **DynamicMap** — portes temporelles, zones de danger, obstacles mobiles, stations de carburant

### Physique & heuristiques
- Contraintes : pente max, hauteur de saut, rayon de virage (`DEFAULT` / `STRICT`)
- Heuristiques injectables : Octile, Manhattan, Euclidean, Landmark (Dijkstra), **adaptative** (selon topologie)

### Générateurs de cartes
`random` · `maze` · `caves` · `rooms` · `pillars` · `perlin` · `bsp` · `city` · `floating`

### Interface
- Visualisation 3D (matplotlib)
- Heatmap d'exploration
- Timeline du chemin + rejeu animé
- Dashboard (historique, optimalité, **export CSV**)
- Paramètres : backend, physique, heuristique, générateur, graine

---

## Installation

```bash
git clone https://github.com/<votre-compte>/FindPath3D.git
cd FindPath3D
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Dépendances** : `PyQt5`, `matplotlib`, `numpy`, `pytest`

---

## Lancement

```bash
python main.py
```

### Benchmarks & tests

```bash
python bench.py
python benchmarks/compare_backends.py
pytest tests/ -v
```

---

## Structure du dépôt

```
FindPath3D/
├── main.py                 # Point d'entrée PyQt5
├── bench.py                # Banc d'essai CLI
├── requirements.txt
├── core/
│   ├── grid.py / octree.py # Représentations spatiales
│   ├── physics.py          # Contraintes de mouvement
│   ├── heuristics.py       # Heuristiques injectables
│   ├── dynamics.py         # Environnement temporel
│   ├── topology.py         # Détection de topologie
│   ├── validation.py       # Validation de chemins
│   ├── generators.py       # Générateurs de cartes
│   └── engines/            # A*, Theta*, JPS, TimeA*, …
├── ui/                     # Interface PyQt5 + canvas 3D
├── benchmarks/             # Suites de référence
├── tests/                  # Tests de non-régression
└── docs/                   # Architecture, guides, phases
```


---

## Exemple d'utilisation programmatique

```python
from core.generators import create_grid
from core.point import Point
from core.engines.a_star import AStar
from core.physics import DEFAULT_PHYSICS
from core.heuristics import LandmarkHeuristic

grid = create_grid("caves", size=30, obstacle=20, seed=42, backend="octree")
start, end = Point(0, 0, 0), Point(29, 29, 29)

engine = AStar(
    grid, start, end,
    physics=DEFAULT_PHYSICS,
    heuristic=LandmarkHeuristic(grid),
)
if engine.run():
    print(engine.stats)  # temps, nœuds, longueur, coût
```

Environnement dynamique :

```python
from core.dynamics import make_demo_dynamic
from core.engines.time_astar import TimeAStar

base = create_grid("random", 12, 5, 42)
world = make_demo_dynamic(base)
engine = TimeAStar(world, start, end, t_horizon=120)
engine.run()
```

---

## Tests

```bash
pytest tests/ -v
```

Environ **37 tests** de correctness (générateurs, physique, Octree, TimeA\*, Theta\*, JPS, validation, etc.).

---

## Évolution (résumé)

| Phase | Thème |
|-------|--------|
| 0 | Fondations, architecture, benchmarks, tests |
| 1 | Physique, heuristiques, Octree |
| 2 | Dynamique, TimeA\*, Theta\*, replanning, topologie |
| 3 | Générateurs avancés, JPS, validation |
| 4 | Heatmap, timeline, export CSV, documentation |

---

## Licence

À définir par le mainteneur du dépôt (MIT / Apache-2.0 / autre).

## Contribution

Les contributions (issues, correctifs, nouveaux algorithmes ou générateurs) sont les bienvenues via pull requests.

---

*FindPath3D — comprendre le pathfinding 3D en le voyant, en le mesurant, en le faisant évoluer.*
