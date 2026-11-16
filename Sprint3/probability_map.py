"""
Contributors: Sofia Monge and Mylan Bui
File Description: Probability-of-area (belief) map for the SAR simulation.
Grid in which each cell holds the probability that the missing person is in that cell.
Updated by:
  - operator sightings        -> probability raised around a point
  - operator exclusions       -> probability set to zero in an area
  - drone coverage, no find   -> Bayesian negative-information update
  - optional diffusion        -> models a person who may be moving
The mission uses the map to choose where the drone searches next.
Detection itself is handled by SearchMonitor; this map only guides where to look.
"""
import numpy as np
import pybullet as p

# Detection chance at the far edge of the camera view, as a share of the close-up chance (pod).
FAR_POD_FACTOR = 0.35

HEATMAP_Z = 0.10   # above the OSM ground layers (roads, grass, parking... top out at ~0.08 m)
HIDDEN_LEVEL = -2  # tile level of a transparent tile (heatmap hidden); -1 = never painted


class ProbabilityMap:
    def __init__(self, x_range=(-12.0, 12.0), y_range=(-12.0, 12.0),
                 cell_size=1.0, sensor_radius=2.0, pod=0.8):
        """
        x_range, y_range: world bounds in meters
        cell_size:        grid resolution in meters
        sensor_radius:    detection radius used by observe()
        pod:              probability of detection when the person is in view
        """
        self.x0, self.x1 = x_range
        self.y0, self.y1 = y_range
        self.cell = cell_size
        self.nx = int(round((self.x1 - self.x0) / cell_size))
        self.ny = int(round((self.y1 - self.y0) / cell_size))
        self.sensor_radius = sensor_radius
        self.pod = pod

        xs = self.x0 + (np.arange(self.nx) + 0.5) * cell_size
        ys = self.y0 + (np.arange(self.ny) + 0.5) * cell_size
        self.cx, self.cy = np.meshgrid(xs, ys)          # shape (ny, nx), indexed [iy, ix]

        self.grid = np.full((self.ny, self.nx), 1.0 / (self.nx * self.ny))
        self.excluded = np.zeros_like(self.grid, dtype=bool)
        self.prior = self.grid.copy()          # starting map; re-included zones get these values back
        self._area_excluded = np.zeros_like(self.excluded)   # from exclude() (circles)
        self._zone_masks = {}                  # zone name -> cells excluded for that zone
        self._last_cell = None

    # ---------- coordinate helpers ----------
    def to_cell(self, xy):
        ix = int((xy[0] - self.x0) // self.cell)
        iy = int((xy[1] - self.y0) // self.cell)
        return min(max(ix, 0), self.nx - 1), min(max(iy, 0), self.ny - 1)

    def to_world(self, ix, iy):
        return (float(self.cx[iy, ix]), float(self.cy[iy, ix]))

    def _disk(self, xy, radius):
        return (self.cx - xy[0]) ** 2 + (self.cy - xy[1]) ** 2 <= radius ** 2

    def _normalize(self):
        self.grid[self.excluded] = 0.0
        total = self.grid.sum()
        if total > 0:
            self.grid /= total
        else:  # all cells excluded/searched: reset to uniform over allowed cells
            self.grid[~self.excluded] = 1.0
            self.grid /= max(self.grid.sum(), 1e-12)

    # ---------- operator updates ----------
    def add_sighting(self, xy, sigma=2.5, weight=0.6):
        """Witness report near xy. weight = trust in the report (0..1)."""
        g = np.exp(-((self.cx - xy[0]) ** 2 + (self.cy - xy[1]) ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        self.grid = (1 - weight) * self.grid + weight * g
        self._normalize()

    def exclude(self, xy, radius):
        """Area marked 'do not search'; person assumed not there."""
        self._area_excluded |= self._disk(xy, radius)
        self.excluded |= self._area_excluded
        self._normalize()

    # ---------- PBI-03: initial map and zone exclusions ----------
    def set_initial(self, matrix):
        """Load the starting map built by terrain_prior.build_initial_map() (same shape as the
        grid). It is kept as self.prior so a re-included zone can get its starting values back."""
        m = np.asarray(matrix, dtype=float)
        if m.shape != self.grid.shape:
            raise ValueError(f"initial map shape {m.shape} does not match the grid {self.grid.shape}")
        self.prior = m / m.sum() if m.sum() > 0 else np.full(m.shape, 1.0 / m.size)
        self.grid = self.prior.copy()
        self._normalize()

    def _box(self, rect):
        x0, y0, x1, y1 = rect
        return (self.cx >= x0) & (self.cx <= x1) & (self.cy >= y0) & (self.cy <= y1)

    def _refresh_excluded(self):
        self.excluded = self._area_excluded.copy()
        for mask in self._zone_masks.values():
            self.excluded |= mask

    def exclude_zone(self, name, rect):
        """'Do not search' for a zone: every cell inside rect = (x0, y0, x1, y1) gets probability 0
        until the zone is re-included. Excluding the same zone twice changes nothing."""
        if name in self._zone_masks:
            return
        self._zone_masks[name] = self._box(rect)
        self._refresh_excluded()
        self._normalize()

    def include_zone(self, name):
        """Re-include an excluded zone: its cells get their starting (prior) values back, then the
        map is re-normalized. Cells still excluded by another zone or area stay at 0."""
        mask = self._zone_masks.pop(name, None)
        if mask is None:
            return
        self._refresh_excluded()
        restore = mask & ~self.excluded
        if restore.any():
            # Put the zone back at the same scale as the rest of the map: its starting values,
            # sized relative to the map's current total over the other allowed cells.
            others = ~self.excluded & ~restore
            prior_others = self.prior[others].sum()
            scale = self.grid[others].sum() / prior_others if prior_others > 0 else 1.0
            self.grid[restore] = self.prior[restore] * scale
        self._normalize()

    def include_all_zones(self):
        """Return to the original plan: every excluded zone is re-included."""
        for name in list(self._zone_masks):
            self.include_zone(name)

    def excluded_zones(self):
        return sorted(self._zone_masks)

    def apply_guidance(self, command, zone=None, rect=None, sighting=None,
                       sighting_sigma=2.5, sighting_weight=0.5):
        """Update the map for one ACCEPTED operator command (operator_guidance.Cmd name):
            INVESTIGATE  -> raise probability around the sighting point
            AVOID        -> exclude the zone's rect
            REINCLUDE    -> re-include the zone
            RETURN_PLAN  -> re-include every zone
            SEARCH_FIRST -> no change (it only reorders the plan)
        """
        if command == "INVESTIGATE" and sighting is not None:
            self.add_sighting(sighting, sigma=sighting_sigma, weight=sighting_weight)
        elif command == "AVOID" and zone is not None and rect is not None:
            self.exclude_zone(zone, rect)
        elif command == "REINCLUDE" and zone is not None:
            self.include_zone(zone)
        elif command == "RETURN_PLAN":
            self.include_all_zones()

    def mark_searched(self, xy, radius):
        """Area already searched by another source (e.g., a ground team)."""
        self.grid[self._disk(xy, radius)] *= (1 - self.pod)
        self._normalize()

    def set_prior(self, matrix):
        """Load an external probability matrix (e.g., from DSSE), resampled to this grid."""
        m = np.asarray(matrix, dtype=float)
        iy = np.arange(self.ny) * m.shape[0] // self.ny
        ix = np.arange(self.nx) * m.shape[1] // self.nx
        self.grid = m[np.ix_(iy, ix)].copy()
        self._normalize()

    # ---------- drone updates ----------
    def observe(self, drone_xy):
        """Circular-footprint update: area around the drone searched, person not found.
        Updates only when the drone enters a new cell."""
        cell = self.to_cell(drone_xy)
        if cell == self._last_cell:
            return
        self._last_cell = cell
        self.grid[self._disk(drone_xy, self.sensor_radius)] *= (1 - self.pod)
        self._normalize()

    def _build_canopy_mask(self, trees, canopy_radius):
        """Boolean grid of cells under a tree canopy. Trees are static, so this is built once."""
        mask = np.zeros(self.grid.shape, dtype=bool)
        for t in trees:
            mask |= (self.cx - t[0]) ** 2 + (self.cy - t[1]) ** 2 <= canopy_radius ** 2
        return mask

    def frustum_mask(self, monitor, cam, ground_z=0.3, return_dist=False):
        """Vectorized version of monitor.in_frustum for every cell at once.
        Same pinhole FOV + max-range test as SearchMonitor.in_frustum, one numpy pass.
        return_dist=True also returns each cell's distance from the camera."""
        f, r, u = monitor._axes()
        vx = self.cx - cam[0]
        vy = self.cy - cam[1]
        vz = ground_z - cam[2]
        dist = np.sqrt(vx * vx + vy * vy + vz * vz)
        xf = vx * f[0] + vy * f[1] + vz * f[2]
        xr = vx * r[0] + vy * r[1] + vz * r[2]
        xu = vx * u[0] + vy * u[1] + vz * u[2]
        ok = (dist <= monitor.range) & (dist >= 1e-6) & (xf > 0.05)
        safe_xf = np.where(ok, xf, 1.0)                  # avoid dividing by ~0 outside the view
        mask = ok & (np.abs(xr / safe_xf) <= monitor.tan_h) & (np.abs(xu / safe_xf) <= monitor.tan_v)
        return (mask, dist) if return_dist else mask

    def observe_camera(self, monitor, drone_pos, dt, trees=(), canopy_radius=1.0,
                       canopy_pod=0.3, ground_z=0.3):
        """Continuous camera-footprint update (call every physics step).

        Uses SearchMonitor's frustum (forward-tilted wedge, FOV + max range). Cells under a
        tree canopy use a lower detection probability.
        pod and canopy_pod are per-SECOND detection chances while a cell is in view, for a cell
        right in front of the camera. Farther cells get less, with the same falloff the detector
        uses for its score (target_detection: quality = 1 - (dist / range)^2, at least 0.2), so a
        glimpse at the edge of the view does not rule a cell out as fast as a close look.
        Each call applies (1 - pod) ** dt, so the result does not depend on the step size.
        """
        if not hasattr(self, "_canopy"):
            self._canopy = self._build_canopy_mask(trees, canopy_radius)
        seen, dist = self.frustum_mask(monitor, drone_pos, ground_z, return_dist=True)
        if not seen.any():
            return
        quality = np.clip(1.0 - (dist / monitor.range) ** 2, FAR_POD_FACTOR, 1.0)
        pod = np.where(self._canopy, canopy_pod, self.pod) * quality
        self.grid[seen] *= ((1.0 - pod) ** dt)[seen]
        self._normalize()

    def diffuse(self, rate=0.05):
        """Spread probability to neighboring cells."""
        pad = np.pad(self.grid, 1, mode="edge")
        neighbors = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
        self.grid = (1 - rate) * self.grid + rate * neighbors
        self._normalize()

    # ---------- queries ----------
    def best_target(self, from_xy=None, distance_weight=0.0, min_dist=0.0, skip=()):
        """World (x, y) of the most promising cell. distance_weight > 0 favors nearby cells.
        min_dist: ignore cells closer than this to from_xy (map-guided search: no 'go nowhere'
        targets). skip: (ix, iy) cells to leave out. Excluded cells are never returned, and the
        result is always a cell centre, so it is inside the search bounds. Returns None only if
        every cell is excluded or skipped."""
        cell = self.best_cell(from_xy, distance_weight, min_dist, skip)
        return None if cell is None else self.to_world(*cell)

    def best_cell(self, from_xy=None, distance_weight=0.0, min_dist=0.0, skip=()):
        """(ix, iy) of the cell best_target() picks, or None (see best_target)."""
        score = self.grid.copy()
        if from_xy is not None:
            d = np.hypot(self.cx - from_xy[0], self.cy - from_xy[1])
            if distance_weight > 0:
                score = score / (1 + distance_weight * d)
            if min_dist > 0:
                score[d < min_dist] = -1.0
        for ix, iy in skip:
            score[iy, ix] = -1.0
        score[self.excluded] = -1.0
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        if score[iy, ix] < 0:
            return None
        return int(ix), int(iy)

    def zone_mass(self, rect):
        """Probability that the person is inside rect (x0, y0, x1, y1)."""
        return float(self.grid[self._box(rect)].sum())

    def mass_near(self, xy, radius):
        """Total probability inside a circle (used for ranking search zones)."""
        return float(self.grid[self._disk(xy, radius)].sum())

    # ---------- visualization ----------
    def draw(self, top_k=60, max_height=1.5):
        """Top_k most likely cells drawn as vertical sticks; taller and redder = more likely.
        Uses debug lines because debug points can hide GUI button labels on some builds."""
        if not hasattr(self, "_line_ids"):
            self._line_ids = []
        flat = np.argsort(self.grid, axis=None)[-top_k:]
        iys, ixs = np.unravel_index(flat, self.grid.shape)
        pmax = max(self.grid.max(), 1e-12)
        for k, (iy, ix) in enumerate(zip(iys, ixs)):
            v = float(self.grid[iy, ix] / pmax)
            x, y = float(self.cx[iy, ix]), float(self.cy[iy, ix])
            start, end = [x, y, 0.02], [x, y, 0.02 + max(0.1, v * max_height)]
            color = [1.0, 1.0 - v, 0.0]
            if k < len(self._line_ids):
                self._line_ids[k] = p.addUserDebugLine(start, end, color, lineWidth=3,
                                                       replaceItemUniqueId=self._line_ids[k])
            else:
                self._line_ids.append(p.addUserDebugLine(start, end, color, lineWidth=3))

    def draw_heatmap(self, alpha=0.45, levels=8, budget=40, pause_rendering=False):
        """Ground heatmap over every cell, scaled to the most likely cell (log scale, 2 decades):
             green  = most likely (the brightest green is the cell best_target() would pick)
             yellow = about a tenth as likely as the best cell
             red    = 1/100 of the best cell or less: ruled out (searched with no find, or excluded)
        Tiles are visual-only (no collision), so they do not block camera rays
        or obstacle sensing. Only tiles whose color changed are updated, and at most
        `budget` tiles are recolored per call (largest color changes first; None = all).
        While the heatmap is hidden the tiles are made transparent the same way.
        The first call builds all tiles; do it before the main loop (see main.py).

        PBI-07: pause_rendering=True recolours with the GUI's rendering paused. In the GUI every
        changeVisualShape waits for the render thread (5-16 ms per tile, so 8 tiles could take
        a whole 60-130 ms frame); paused, all 900 tiles take ~30 ms. main.py does this a few times
        a second for every changed tile instead of a few tiles every frame.
        Returns the number of tiles recoloured."""
        if not hasattr(self, "_tiles"):
            half = 0.45 * self.cell
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, 0.005],
                                      rgbaColor=[1, 1, 0, alpha])
            self._tiles = np.empty(self.grid.shape, dtype=int)
            # tiles are created yellow (the middle level); force a full first paint
            self._tile_level = np.full(self.grid.shape, -1, dtype=int)
            for iy in range(self.ny):
                for ix in range(self.nx):
                    self._tiles[iy, ix] = p.createMultiBody(
                        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                        basePosition=[float(self.cx[iy, ix]), float(self.cy[iy, ix]), HEATMAP_Z])
            self._heat_alpha = alpha
            self.heatmap_visible = True

        if self.heatmap_visible:
            level = self.heat_levels(levels)
        else:                                          # hidden: every tile transparent
            level = np.full(self.grid.shape, HIDDEN_LEVEL, dtype=int)

        diff = np.argwhere(level != self._tile_level)
        if len(diff) == 0:
            return 0
        if budget is not None and len(diff) > budget:
            change = np.abs(level - self._tile_level)[diff[:, 0], diff[:, 1]]
            diff = diff[np.argsort(-change, kind="stable")[:budget]]
        if pause_rendering:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        try:
            for iy, ix in diff:
                lv = level[iy, ix]
                colour = ([0, 0, 0, 0] if lv == HIDDEN_LEVEL
                          else heat_colour(lv / (levels - 1), alpha))   # 0 red, 0.5 yellow, 1 green
                p.changeVisualShape(int(self._tiles[iy, ix]), -1, rgbaColor=colour)
                self._tile_level[iy, ix] = lv          # remember per tile, not all at once
        finally:
            if pause_rendering:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return len(diff)

    def heat_levels(self, levels=8):
        """Heatmap colour level of every cell, 0 (red) .. levels - 1 (green).
        Relative to the BEST cell, on a log scale: 1 -> green, 0.1 -> yellow, <= ~0.01 -> red.
        (Scaling to the mean made every above-average cell the same colour, hiding the peak.)"""
        allowed = ~self.excluded
        top = self.grid[allowed].max() if allowed.any() else 1.0
        ratio = np.maximum(self.grid / max(top, 1e-12), 1e-12)
        frac = np.clip(1.0 + np.log10(ratio) / 2.0, 0.0, 1.0)
        level = np.round(frac * (levels - 1)).astype(int)
        level[self.excluded] = 0
        return level

    def red_mask(self):
        """Cells the heatmap shows red: excluded, or (almost) no chance the person is there
        (water, buildings, or ruled out by searching)."""
        return self.heat_levels() == 0

    def set_heatmap_visible(self, visible, repaint=True):
        """Show or hide the heatmap (operator toggle), e.g. to see the real map underneath.
        The probability map keeps updating while hidden; showing it again repaints every tile.
        repaint=False only flips the setting: the next draw_heatmap() call does the repaint
        (PBI-07: main.py lets its batched heatmap update do it, so the click stays instant)."""
        if not hasattr(self, "_tiles") or visible == self.heatmap_visible:
            return
        self.heatmap_visible = visible
        if repaint:
            self.draw_heatmap(alpha=self._heat_alpha, budget=None, pause_rendering=True)

def heat_colour(v, alpha):
    """Heatmap colour for v in [0, 1]: red (0) -> yellow (0.5) -> green (1)."""
    if v <= 0.5:
        return [1.0, 2.0 * v, 0.0, alpha]
    return [2.0 * (1.0 - v), 1.0 - 0.2 * (v - 0.5) * 2.0, 0.0, alpha]
