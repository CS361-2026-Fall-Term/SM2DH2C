"""
Contributors: Sofia Monge and Mylan Bui
File Description: Builds the inputs for the initial probability map (PBI-03) from the
OpenStreetMap scene the 3D environment uses (osm_world.OSMScene).

  terrain_grid()  - Step 1: one terrain type per cell of the probability map's grid.
  terrain_weights() - Step 2: the terrain weight of every cell.
  ring_density()  - Step 3: distance-ring density around the Last Known Position (LKP),
                    scaled by the person category.
  build_initial_map() - Steps 4 and 6: ring density x terrain weight, normalized to sum to 1.
                    Step 5 (excluded zones) is done by ProbabilityMap.exclude_zone().

The grid is always the ProbabilityMap's own grid (its cx, cy, cell, x0, y0), so every value
lines up with the heatmap cell it belongs to. Coordinates are sim metres, LKP at (0, 0).
"""
import math

import numpy as np

# ------------------------------------------------------------------ Step 1-2: terrain
OPEN, FOREST, TRAIL, SIDEWALK, ROAD, PAVED, RAILWAY, ROCK, BARRIER, WATER, BUILDING = range(11)

TERRAIN_NAMES = {
    OPEN: "open", FOREST: "forest", TRAIL: "trail", SIDEWALK: "sidewalk", ROAD: "road",
    PAVED: "paved area", RAILWAY: "railway", ROCK: "steep/rock", BARRIER: "barrier",
    WATER: "water", BUILDING: "building",
}

# Step 2 table. PLACEHOLDER weights, to be replaced with cited lost-person behavior data.
TERRAIN_WEIGHTS = {
    TRAIL: 2.0, SIDEWALK: 1.5, ROAD: 1.2, PAVED: 1.0, OPEN: 1.0, FOREST: 0.6,
    RAILWAY: 0.5, ROCK: 0.3, BARRIER: 0.2, WATER: 0.0, BUILDING: 0.0,
}

# When features overlap in one cell, the first type in this list wins (PBI-03 Step 1).
PRIORITY = [BUILDING, WATER, TRAIL, SIDEWALK, ROAD, RAILWAY, BARRIER, PAVED, ROCK, FOREST, OPEN]

# osm_world ground-surface categories (surface_category) -> terrain type.
SURFACE_TO_TERRAIN = {
    "forest": FOREST, "scrub": FOREST, "rock": ROCK, "water": WATER,
    "parking": PAVED, "plaza": PAVED,
    "grass": OPEN, "farm": OPEN, "sand": OPEN, "dirt": OPEN, "pitch": OPEN, "track": OPEN,
}

# osm_world line categories (LINE_LAYERS) -> terrain type.
LINE_TO_TERRAIN = {
    "path": TRAIL,
    "foot": SIDEWALK, "cycle": SIDEWALK,
    "service": ROAD, "minor": ROAD, "major": ROAD, "crossing": ROAD,
    "rail": RAILWAY,
    "stream": WATER,
}

BARRIER_WIDTH = 0.3     # sim metres; fences, walls and hedges are drawn about this thick
TREE_RADIUS = 1.0       # sim metres; single trees mark the cells under their canopy


def _points_in_polygon(cx, cy, poly):
    """Even-odd test for every cell centre at once."""
    inside = np.zeros(cx.shape, dtype=bool)
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        if y1 == y2:
            continue
        crosses = (y1 > cy) != (y2 > cy)
        inside ^= crosses & (cx < x1 + (cy - y1) * (x2 - x1) / (y2 - y1))
    return inside


def _cell_index(grid, x, y):
    """(iy, ix) of the cell containing (x, y), or None if outside the grid."""
    ix = int(math.floor((x - grid.x0) / grid.cell))
    iy = int(math.floor((y - grid.y0) / grid.cell))
    if 0 <= iy < grid.cx.shape[0] and 0 <= ix < grid.cx.shape[1]:
        return iy, ix
    return None


def _polygon_cells(grid, poly):
    """Cells whose centre is inside the polygon, plus the cells holding its corners and
    centroid, so a feature smaller than a cell (a house on a large --area) still shows."""
    mask = _points_in_polygon(grid.cx, grid.cy, poly)
    centroid = (sum(q[0] for q in poly) / len(poly), sum(q[1] for q in poly) / len(poly))
    for x, y in list(poly) + [centroid]:
        idx = _cell_index(grid, x, y)
        if idx is not None:
            mask[idx] = True
    return mask


def _segment_cells(grid, a, b, width):
    """Cells that a line feature of the given width passes through (any part of the cell)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        t = np.zeros(grid.cx.shape)
    else:
        t = np.clip(((grid.cx - a[0]) * dx + (grid.cy - a[1]) * dy) / length_sq, 0.0, 1.0)
    dist = np.hypot(grid.cx - a[0] - t * dx, grid.cy - a[1] - t * dy)
    return dist <= width / 2 + grid.cell / 2


def terrain_grid(scene, grid):
    """Step 1: terrain type of every cell (TERRAIN_NAMES codes), shape (ny, nx).

    scene: osm_world.OSMScene (surfaces, water, buildings, roads, bridges, barriers, trees)
    grid:  the ProbabilityMap (uses its cx, cy, cell, x0, y0)
    """
    masks = {code: np.zeros(grid.cx.shape, dtype=bool) for code in TERRAIN_NAMES}
    for poly, category in scene.surfaces:
        code = SURFACE_TO_TERRAIN.get(category)
        if code is not None:
            masks[code] |= _points_in_polygon(grid.cx, grid.cy, poly)
    for poly in scene.water:
        masks[WATER] |= _polygon_cells(grid, poly)
    for poly, *_ in scene.buildings:
        masks[BUILDING] |= _polygon_cells(grid, poly)
    for a, b, width, category in scene.roads:
        code = LINE_TO_TERRAIN.get(category)
        if code is not None:
            masks[code] |= _segment_cells(grid, a, b, width)
    for a, b, _kind in getattr(scene, "barriers", []):
        masks[BARRIER] |= _segment_cells(grid, a, b, BARRIER_WIDTH)
    for x, y in scene.trees:
        masks[FOREST] |= np.hypot(grid.cx - x, grid.cy - y) <= max(TREE_RADIUS, grid.cell / 2)

    terrain = np.full(grid.cx.shape, OPEN, dtype=int)
    for code in reversed(PRIORITY):             # lowest priority first; higher ones overwrite
        terrain[masks[code]] = code

    # Bridges (roads, paths, rails on a deck) are walkable even where they cross water, so they
    # are applied last and replace the water below them. Buildings still win.
    for points, width, category in getattr(scene, "bridges", []):
        code = LINE_TO_TERRAIN.get(category)
        if code is None:
            continue
        deck = np.zeros(grid.cx.shape, dtype=bool)
        for a, b in zip(points, points[1:]):
            deck |= _segment_cells(grid, a, b, width)
        terrain[deck & (terrain != BUILDING)] = code
    return terrain


def terrain_weights(terrain):
    """Step 2: terrain weight of every cell."""
    lookup = np.array([TERRAIN_WEIGHTS[code] for code in range(len(TERRAIN_NAMES))])
    return lookup[terrain]


# ------------------------------------------------------------------ Step 3: distance rings
# PLACEHOLDER table, to be replaced with cited lost-person behavior data.
# (outer radius as a fraction of the map half-width h, share of people found inside the ring)
RING_TABLE = [
    (0.23, 0.25),
    (0.46, 0.25),
    (0.77, 0.25),
    (1.23, 0.20),
    (2.46, 0.05),
]

# PLACEHOLDER radius multipliers per person category.
PERSON_CATEGORIES = {
    "hiker": 1.0,
    "runner": 1.3,
    "child": 0.6,
    "elderly/dementia": 0.5,
}
DEFAULT_CATEGORY = "hiker"


def ring_radii(category, map_half):
    """Outer radius of each ring in sim metres for this person category."""
    if category not in PERSON_CATEGORIES:
        raise ValueError(f"Unknown person category '{category}'. "
                         f"Expected one of: {', '.join(PERSON_CATEGORIES)}")
    k = PERSON_CATEGORIES[category]
    return [frac * map_half * k for frac, _share in RING_TABLE]


def ring_density(grid, lkp=(0.0, 0.0), category=DEFAULT_CATEGORY, map_half=None):
    """Step 3: distance-ring density of every cell, shape (ny, nx).

    Each ring's share is spread evenly over the ring's full circular area:
        density = share / (pi * (r_outer^2 - r_inner^2))
    Cells beyond the last ring get 0. Not normalized (Step 6 does that).

    map_half: map half-width h in sim metres (default: sim_config.MAP_HALF at call time).
    """
    if map_half is None:
        import sim_config
        map_half = sim_config.MAP_HALF
    radii = ring_radii(category, map_half)
    d = np.hypot(grid.cx - lkp[0], grid.cy - lkp[1])
    density = np.zeros(grid.cx.shape)
    inner = 0.0
    for outer, (_frac, share) in zip(radii, RING_TABLE):
        in_ring = (d >= inner) & (d < outer)
        density[in_ring] = share / (math.pi * (outer ** 2 - inner ** 2))
        inner = outer
    return density


# ------------------------------------------------------------------ Steps 4-6: initial map
def empty_scene():
    """A scene with no features (practice world without OSM data): every cell is open ground."""
    from types import SimpleNamespace
    return SimpleNamespace(surfaces=[], water=[], buildings=[], roads=[], bridges=[],
                           barriers=[], trees=[])


def _normalized(values):
    total = float(values.sum())
    return values / total if total > 0 else None


def build_initial_map(scene, grid, lkp=(0.0, 0.0), category=DEFAULT_CATEGORY, map_half=None):
    """Steps 4 and 6: the starting probability of every cell, summing to 1.

        raw(cell) = ring_density(cell) x terrain_weight(cell)      (Step 4)
        map(cell) = raw(cell) / sum of all raw values              (Step 6)

    Step 5 (excluded zones) is applied by ProbabilityMap.exclude_zone(), so a zone can be
    re-included later and get these starting values back.

    scene: osm_world.OSMScene, or None for the practice world (all open ground).
    Returns (map, warning): warning is None normally, or a message for the operator when the
    terrain could not be used (every reachable cell is water or building) and a fallback was used.
    Raises ValueError for an unknown person category.
    """
    scene = scene if scene is not None else empty_scene()
    terrain = terrain_grid(scene, grid)
    density = ring_density(grid, lkp, category, map_half)
    result = _normalized(density * terrain_weights(terrain))
    if result is not None:
        return result, None
    result = _normalized(density)
    if result is not None:
        return result, "Terrain could not be applied (no walkable cells); using distance only"
    uniform = np.full(grid.cx.shape, 1.0 / grid.cx.size)
    return uniform, "Terrain and distance could not be applied; using a uniform map"
