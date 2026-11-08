"""
Contributors: Sofia Monge and Mylan Bui
File Description: SPIKE-01 prototype. Generates the initial probability map from terrain data
and a search policy, and checks that the result loads into ProbabilityMap.set_prior().

Investigation question:
    Can we generate a probability map from real terrain data plus a search policy, in a
    format ProbabilityMap.set_prior() accepts, in under 1 second for our search area?

Terrain sources:
    OSM (real world)  - the same OpenStreetMap scene the 3D environment is built from
                        (osm_world.parse_scene), turned into a terrain grid. The grid has exactly
                        the size and cell size main.py uses for its probability map.
    CSV               - a grid of terrain codes (row 0 = north edge), for testing.
    Sample            - a generated forest terrain (default when no source is given).

Approaches compared:
    A. Distance rings   - probability depends only on distance from the Last Known Position
                          (LKP), using lost-person distance percentiles.
    B. Terrain-weighted - terrain weight x smooth falloff from the LKP.
    C. Combined         - A x terrain weights (ring policy shaped by terrain).

Usage (same OSM flags as main.py; the LKP is the map centre, (0, 0)):
    python terrain_prior_spike.py --lat 44.5646 --lon -123.2620 --area 60
    python terrain_prior_spike.py --lat 44.5646 --lon -123.2620 --osm-file map.osm --area 60
    python terrain_prior_spike.py                              # generated sample terrain
    python terrain_prior_spike.py --terrain my_terrain.csv     # terrain CSV
    python terrain_prior_spike.py --benchmark                  # also time larger grids

Outputs (in --out, default spike01_results/): terrain.png, prior_*.png, comparison.png,
prior_*.npy (loadable with ProbabilityMap.set_prior), terrain_grid.csv, results.txt
"""
import argparse
import math
import os
import time

import numpy as np

# ------------------------------------------------------------------ terrain --
# Terrain codes and likelihood weights. Weights are PLACEHOLDERS for the spike;
# final values should come from lost-person behavior data (e.g., Koester, hiker category).
TERRAIN_NAMES = {0: "open", 1: "forest", 2: "trail", 3: "water", 4: "steep/rock",
                 5: "building", 6: "road"}
TERRAIN_WEIGHTS = {0: 1.0, 1: 0.6, 2: 2.0, 3: 0.0, 4: 0.3, 5: 0.0, 6: 1.2}
TERRAIN_COLORS = {0: "#c9e4a6", 1: "#3f7d3a", 2: "#c8a165", 3: "#4a90d9", 4: "#8a8a8a",
                  5: "#8b5a5a", 6: "#3d3d3d"}

# OSM ground surfaces (osm_world.surface_category) -> terrain code.
SURFACE_TO_TERRAIN = {"forest": 1, "scrub": 1, "rock": 4, "water": 3}
# OSM line categories (osm_world.LINE_LAYERS) -> terrain code. Rail is left as open ground.
LINE_TO_TERRAIN = {"path": 2, "foot": 2, "cycle": 2, "stream": 3,
                   "service": 6, "minor": 6, "major": 6, "crossing": 6}
# When features overlap in one cell, the higher-priority terrain wins.
PRIORITY = [5, 3, 2, 6, 4, 1, 0]          # building > water > trail > road > rock > forest > open

# Default grid when no OSM scene is used: main.py's 26 m map with 1 m cells.
DEFAULT_X_RANGE = (-13.0, 13.0)
DEFAULT_Y_RANGE = (-11.0, 13.0)
DEFAULT_CELL = 1.0

# ----------------------------------------------------------- search policy --
# Fraction of lost hikers found within each distance of the LKP, as (fraction, radius).
# PLACEHOLDERS given in FRACTIONS OF THE MAP HALF-WIDTH, so they scale with --area / --map-size.
# Replace with real percentiles (e.g., Koester, "Lost Person Behavior") scaled to the map.
RING_PERCENTILES = [(0.25, 0.23), (0.50, 0.46), (0.75, 0.77), (0.95, 1.23)]


class Grid:
    """Cell centres for a rectangular area, row 0 = south (min y), like ProbabilityMap."""

    def __init__(self, x_range, y_range, cell):
        self.x_range, self.y_range, self.cell = x_range, y_range, cell
        nx = int(round((x_range[1] - x_range[0]) / cell))
        ny = int(round((y_range[1] - y_range[0]) / cell))
        xs = x_range[0] + (np.arange(nx) + 0.5) * cell
        ys = y_range[0] + (np.arange(ny) + 0.5) * cell
        self.cx, self.cy = np.meshgrid(xs, ys)
        self.half = max(abs(v) for v in (*x_range, *y_range))

    @property
    def shape(self):
        return self.cx.shape

    @property
    def extent(self):
        return [self.x_range[0], self.x_range[1], self.y_range[0], self.y_range[1]]


# ------------------------------------------------------- terrain sources --
def sample_terrain(g):
    """Synthetic forest terrain (open, forest, trail, creek, steep), scaled to the grid size."""
    k = g.half / 13.0
    cx, cy = g.cx / k, g.cy / k                     # design coordinates on the 26 m map
    t = np.zeros(g.shape, dtype=int)
    for fx, fy, r in [(-7, 6, 5.0), (6, 7, 4.5), (-6, -6, 4.0), (8, -6, 3.5)]:
        t[(cx - fx) ** 2 + (cy - fy) ** 2 <= r ** 2] = 1
    t[(cx + cy) > 17] = 4
    t[np.abs(cx - (-2 + 1.5 * np.sin(cy / 3.0))) < 0.8] = 3
    t[np.abs(cy - (1 + 0.15 * cx)) < 0.6] = 2
    return t


def load_terrain_csv(path):
    grid = np.loadtxt(path, delimiter=",", dtype=int, ndmin=2)
    unknown = set(np.unique(grid).tolist()) - set(TERRAIN_WEIGHTS)
    if unknown:
        raise SystemExit(f"Unknown terrain codes in {path}: {sorted(unknown)}")
    return np.flipud(grid)


def save_terrain_csv(path, terrain):
    np.savetxt(path, np.flipud(terrain), fmt="%d", delimiter=",")


def _points_in_polygon(cx, cy, poly):
    """Vectorized even-odd test: which cell centres are inside the polygon."""
    inside = np.zeros(cx.shape, dtype=bool)
    n = len(poly)
    for i in range(n):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        if y1 == y2:
            continue
        crosses = (y1 > cy) != (y2 > cy)
        x_at = x1 + (cy - y1) * (x2 - x1) / (y2 - y1)
        inside ^= crosses & (cx < x_at)
    return inside


def _near_segment(cx, cy, a, b, reach):
    """Which cell centres are within `reach` of segment a-b."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = np.zeros(cx.shape) if L2 == 0 else np.clip(((cx - a[0]) * dx + (cy - a[1]) * dy) / L2, 0, 1)
    return np.hypot(cx - a[0] - t * dx, cy - a[1] - t * dy) <= reach


def _polygon_cells(cx, cy, poly, g):
    """Cells covered by a polygon: every cell whose centre is inside it, plus the cells that
    contain its corners and centroid. The extra cells keep small features (a house squeezed
    to 2 m on a large --area) from disappearing between cell centres."""
    mask = _points_in_polygon(cx, cy, poly)
    nx, ny = cx.shape[1], cx.shape[0]
    centroid = (sum(q[0] for q in poly) / len(poly), sum(q[1] for q in poly) / len(poly))
    for x, y in list(poly) + [centroid]:
        ix = int((x - g.x_range[0]) // g.cell)
        iy = int((y - g.y_range[0]) // g.cell)
        if 0 <= ix < nx and 0 <= iy < ny:
            mask[iy, ix] = True
    return mask


def rasterize_scene(scene, g):
    """Turn an osm_world.OSMScene into a terrain grid of TERRAIN_NAMES codes.
    A cell takes the highest-priority feature that covers its centre. Lines (paths, roads,
    streams) count for every cell they pass through, so narrow paths still show on coarse grids."""
    masks = {code: np.zeros(g.shape, dtype=bool) for code in TERRAIN_NAMES}
    for poly, cat in scene.surfaces:
        code = SURFACE_TO_TERRAIN.get(cat)
        if code is not None:
            masks[code] |= _points_in_polygon(g.cx, g.cy, poly)
    for poly in scene.water:
        masks[3] |= _polygon_cells(g.cx, g.cy, poly, g)
    for poly, *_ in scene.buildings:
        masks[5] |= _polygon_cells(g.cx, g.cy, poly, g)
    cell_reach = g.cell / 2                                # any part of the line inside the cell
    for a, b, width, cat in scene.roads:
        code = LINE_TO_TERRAIN.get(cat)
        if code is not None:
            masks[code] |= _near_segment(g.cx, g.cy, a, b, width / 2 + cell_reach)
    for x, y in scene.trees:                               # single trees outside mapped woods
        masks[1] |= np.hypot(g.cx - x, g.cy - y) <= max(1.0, cell_reach)
    terrain = np.zeros(g.shape, dtype=int)
    for code in reversed(PRIORITY):                        # lowest priority first, then overwrite
        terrain[masks[code]] = code
    return terrain


def load_osm_world(args):
    """Build the same OSM scene and probability-map grid that main.py builds for these flags."""
    import sim_config
    import env_sim
    area = sim_config.BASE_MAP_SIZE if args.area is None else args.area
    if not sim_config.MIN_AREA_M <= area <= sim_config.MAX_AREA_M:
        raise SystemExit(f"--area must be between {sim_config.MIN_AREA_M:g} and {sim_config.MAX_AREA_M:g}")
    map_size = args.map_size if args.map_size is not None else sim_config.clamp_map_size(area)
    sim_config.set_map_size(map_size)
    k = sim_config.layout_scale()
    env_sim.scale_layout(k)
    zones = []
    try:                                                   # same keep-clear points as main.py
        from operator_guidance import ZONES, scale_zones
        scale_zones(k)
        zones = list(ZONES.values())
    except Exception:
        pass
    scene = env_sim.use_osm(args.lat, args.lon, area, args.osm_file, keep_clear=zones, seed=args.seed)

    # Same search area and cell size as main.py (search_bounds() + ~30 x 30 heatmap cells).
    pts = [env_sim.DRONE_START_POS] + list(env_sim.OBSTACLE_POSITIONS) + list(env_sim.FLIGHT_PATH)
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    half, margin = sim_config.MAP_HALF, 3.0
    x_range = (max(min(xs) - margin, -half), min(max(xs) + margin, half))
    y_range = (max(min(ys) - margin, -half), min(max(ys) + margin, half))
    cell = max(1.0, max(x_range[1] - x_range[0], y_range[1] - y_range[0]) / 30.0)
    return scene, Grid(x_range, y_range, cell), f"OSM around ({args.lat}, {args.lon}), {area:g} real m"


# ---------------------------------------------------------------- approaches --
def normalize(m):
    m = np.clip(m, 0.0, None)
    total = m.sum()
    return m / total if total > 0 else np.full(m.shape, 1.0 / m.size)


def weights(terrain):
    lut = np.array([TERRAIN_WEIGHTS[c] for c in range(max(TERRAIN_WEIGHTS) + 1)])
    return lut[terrain]


def prior_rings(g, terrain, lkp):
    """Approach A: each ring's share spread over its full circular area (a ring cut off by
    the map edge keeps a realistic density instead of piling into a few cells)."""
    d = np.hypot(g.cx - lkp[0], g.cy - lkp[1])
    m = np.zeros(g.shape)
    rings = [(f, r * g.half) for f, r in RING_PERCENTILES]
    rings.append((1.0, 2.0 * rings[-1][1]))
    prev_f, prev_r = 0.0, 0.0
    for f, r in rings:
        ring = (d >= prev_r) & (d < r)
        m[ring] = (f - prev_f) / (math.pi * (r * r - prev_r * prev_r))
        prev_f, prev_r = f, r
    return normalize(m)


def prior_terrain(g, terrain, lkp):
    """Approach B: terrain weight x Gaussian falloff from the LKP."""
    sigma = 0.6 * g.half
    d2 = (g.cx - lkp[0]) ** 2 + (g.cy - lkp[1]) ** 2
    return normalize(weights(terrain) * np.exp(-d2 / (2 * sigma ** 2)))


def prior_combined(g, terrain, lkp):
    """Approach C: distance-ring policy shaped by terrain weights."""
    return normalize(prior_rings(g, terrain, lkp) * weights(terrain))


APPROACHES = {"A_rings": prior_rings, "B_terrain": prior_terrain, "C_combined": prior_combined}
TITLES = {"A_rings": "A: distance rings", "B_terrain": "B: terrain-weighted",
          "C_combined": "C: rings x terrain"}


# ---------------------------------------------------------------- checks --
def time_it(fn, *args, repeats=20):
    fn(*args)
    t0 = time.perf_counter()
    for _ in range(repeats):
        out = fn(*args)
    return out, (time.perf_counter() - t0) / repeats * 1000.0


def check_set_prior(matrix, g):
    """Load the matrix into the real ProbabilityMap (same grid as main.py) and confirm it is
    accepted unchanged."""
    try:
        from probability_map import ProbabilityMap
    except Exception as exc:
        return f"skipped (could not import ProbabilityMap: {exc.__class__.__name__}: {exc})"
    pm = ProbabilityMap(x_range=g.x_range, y_range=g.y_range, cell_size=g.cell)
    if pm.grid.shape != matrix.shape:
        return f"FAIL: shape {matrix.shape} != map grid {pm.grid.shape}"
    pm.set_prior(matrix)
    ok = np.allclose(pm.grid, matrix) and abs(pm.grid.sum() - 1.0) < 1e-6
    return "PASS (accepted, sums to 1.0, unchanged)" if ok else "FAIL (values changed on load)"


# ---------------------------------------------------------------- plots --
def plot_results(out, g, terrain, priors, lkp):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        return "matplotlib not installed; PNGs skipped"

    codes = sorted(TERRAIN_NAMES)
    cmap = ListedColormap([TERRAIN_COLORS[c] for c in codes])
    panels = [("terrain", None)] + list(priors.items())
    fig_all, axes = plt.subplots(2, 2, figsize=(11, 10))
    for idx, (name, m) in enumerate(panels):
        for target_ax, standalone in ((axes.flat[idx], False), (None, True)):
            if standalone:
                fig, ax = plt.subplots(figsize=(6, 5.5))
            else:
                fig, ax = fig_all, target_ax
            if m is None:
                ax.imshow(terrain, origin="lower", extent=g.extent, cmap=cmap,
                          vmin=-0.5, vmax=len(codes) - 0.5, interpolation="nearest")
                present = [c for c in codes if (terrain == c).any()]
                handles = [plt.Rectangle((0, 0), 1, 1, color=TERRAIN_COLORS[c]) for c in present]
                ax.legend(handles, [TERRAIN_NAMES[c] for c in present], loc="upper left",
                          fontsize=8, framealpha=0.9)
                ax.set_title("Terrain grid")
            else:
                im = ax.imshow(m, origin="lower", extent=g.extent, cmap="YlOrRd", interpolation="nearest")
                fig.colorbar(im, ax=ax, shrink=0.8, label="P(person in cell)")
                ax.set_title(TITLES[name])
            ax.plot(*lkp, "k*", markersize=13)
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
            if standalone:
                fig.tight_layout()
                fig.savefig(os.path.join(out, f"{name if m is None else 'prior_' + name}.png"), dpi=120)
                plt.close(fig)
    fig_all.tight_layout()
    fig_all.savefig(os.path.join(out, "comparison.png"), dpi=110)
    plt.close(fig_all)
    return "PNGs saved (terrain.png, prior_*.png, comparison.png)"


# ---------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, help="LKP latitude (uses the real OSM world, like main.py)")
    ap.add_argument("--lon", type=float, help="LKP longitude")
    ap.add_argument("--area", type=float, default=None, help="real metres of OSM shown (as in main.py)")
    ap.add_argument("--map-size", type=float, default=None, help="sim map width override (as in main.py)")
    ap.add_argument("--osm-file", help="offline OSM export instead of downloading (as in main.py)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--terrain", help="CSV of terrain codes (row 0 = north edge) instead of OSM")
    ap.add_argument("--benchmark", action="store_true", help="also time 100x100 and 500x500 grids")
    ap.add_argument("--out", default="spike01_results")
    args = ap.parse_args()
    if (args.lat is None) != (args.lon is None):
        ap.error("--lat and --lon must be given together")

    t0 = time.perf_counter()
    if args.lat is not None:
        scene, g, source = load_osm_world(args)
        terrain, raster_ms = time_it(rasterize_scene, scene, g, repeats=3)
    else:
        g = Grid(DEFAULT_X_RANGE, DEFAULT_Y_RANGE, DEFAULT_CELL)
        if args.terrain:
            terrain, source = load_terrain_csv(args.terrain), args.terrain
            if terrain.shape != g.shape:
                raise SystemExit(f"Terrain shape {terrain.shape} does not match grid {g.shape}")
        else:
            terrain, source = sample_terrain(g), "generated sample terrain"
        raster_ms = None
    load_s = time.perf_counter() - t0
    lkp = (0.0, 0.0)                                       # the OSM world puts the LKP at the centre

    os.makedirs(args.out, exist_ok=True)
    save_terrain_csv(os.path.join(args.out, "terrain_grid.csv"), terrain)
    lines = ["SPIKE-01 results",
             f"Terrain: {source}",
             f"Grid: {g.shape[1]} x {g.shape[0]} cells of {g.cell:.2f} m, "
             f"x {g.x_range}, y {g.y_range} (same as main.py's probability map)",
             f"LKP: {lkp}", ""]
    lines.append("Terrain cells: " + ", ".join(f"{TERRAIN_NAMES[c]} {int((terrain == c).sum())}"
                                              for c in TERRAIN_NAMES if (terrain == c).any()))
    if raster_ms is not None:
        lines.append(f"OSM load + parse: {load_s:.2f} s | OSM scene -> terrain grid: {raster_ms:.1f} ms")
    lines.append("")

    priors = {}
    for name, fn in APPROACHES.items():
        m, ms = time_it(fn, g, terrain, lkp)
        priors[name] = m
        np.save(os.path.join(args.out, f"prior_{name}.npy"), m)
        iy, ix = np.unravel_index(np.argmax(m), m.shape)
        mass = {c: float(m[terrain == c].sum()) for c in (2, 3, 5, 6)}
        lines.append(f"[{name}] {ms:.3f} ms | sum {m.sum():.6f} | peak ({g.cx[iy, ix]:.1f}, {g.cy[iy, ix]:.1f}) "
                     f"| on trail {mass[2]:.3f} | on road {mass[6]:.3f} | on water {mass[3]:.3f} "
                     f"| in buildings {mass[5]:.3f}")
        lines.append(f"    set_prior check: {check_set_prior(m, g)}")

    if args.benchmark:
        lines += ["", "Benchmark (approach C on larger grids):"]
        for n in (100, 500):
            bg = Grid((-n / 2, n / 2), (-n / 2, n / 2), 1.0)
            bt = sample_terrain(bg)
            _, ms = time_it(prior_combined, bg, bt, (0.0, 0.0), repeats=5)
            lines.append(f"    {n} x {n} cells: {ms:.1f} ms")

    lines += ["", plot_results(args.out, g, terrain, priors, lkp)]
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(args.out, "results.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()