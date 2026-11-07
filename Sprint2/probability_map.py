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

HEATMAP_Z = 0.10   # above the OSM ground layers (roads, grass, parking... top out at ~0.08 m)


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
        self.excluded |= self._disk(xy, radius)
        self._normalize()

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

    def frustum_mask(self, monitor, cam, ground_z=0.3):
        """Vectorized version of monitor.in_frustum for every cell at once.
        Same pinhole FOV + max-range test as SearchMonitor.in_frustum, one numpy pass."""
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
        return ok & (np.abs(xr / safe_xf) <= monitor.tan_h) & (np.abs(xu / safe_xf) <= monitor.tan_v)

    def observe_camera(self, monitor, drone_pos, dt, trees=(), canopy_radius=1.0,
                       canopy_pod=0.3, ground_z=0.3):
        """Continuous camera-footprint update (call every physics step).

        Uses SearchMonitor's frustum (forward-tilted wedge, FOV + max range). Cells under a
        tree canopy use a lower detection probability.
        pod and canopy_pod are per-SECOND detection chances while a cell is in view; each
        call applies (1 - pod) ** dt, so the result does not depend on the step size.
        """
        if not hasattr(self, "_canopy"):
            self._canopy = self._build_canopy_mask(trees, canopy_radius)
        seen = self.frustum_mask(monitor, drone_pos, ground_z)
        if not seen.any():
            return
        pod = np.where(self._canopy, canopy_pod, self.pod)
        self.grid[seen] *= ((1.0 - pod) ** dt)[seen]
        self._normalize()

    def diffuse(self, rate=0.05):
        """Spread probability to neighboring cells."""
        pad = np.pad(self.grid, 1, mode="edge")
        neighbors = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
        self.grid = (1 - rate) * self.grid + rate * neighbors
        self._normalize()

    # ---------- queries ----------
    def best_target(self, from_xy=None, distance_weight=0.0):
        """World (x, y) of the most promising cell. distance_weight > 0 favors nearby cells."""
        score = self.grid.copy()
        if from_xy is not None and distance_weight > 0:
            d = np.hypot(self.cx - from_xy[0], self.cy - from_xy[1])
            score = score / (1 + distance_weight * d)
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        return self.to_world(ix, iy)

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

    def draw_heatmap(self, alpha=0.45, levels=8, budget=40):
        """Ground heatmap over every cell:
             red    = ruled out (searched with no find, or excluded)
             orange = less likely
             yellow = possible location
        Tiles are visual-only (no collision), so they do not block camera rays
        or obstacle sensing. Only tiles whose color changed are updated, and at most
        `budget` tiles are recolored per call (largest color changes first) so the cost
        per frame stays flat. Call it often; the map catches up over a few frames.
        The first call builds all tiles; do it before the main loop (see main.py)."""
        if not hasattr(self, "_tiles"):
            half = 0.45 * self.cell
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[half, half, 0.005],
                                      rgbaColor=[1, 1, 0, alpha])
            self._tiles = np.empty(self.grid.shape, dtype=int)
            # tiles are created yellow, which is level (levels - 1)
            self._tile_level = np.full(self.grid.shape, levels - 1, dtype=int)
            for iy in range(self.ny):
                for ix in range(self.nx):
                    self._tiles[iy, ix] = p.createMultiBody(
                        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                        basePosition=[float(self.cx[iy, ix]), float(self.cy[iy, ix]), HEATMAP_Z])
            self._heat_alpha = alpha
            self.heatmap_visible = True

        if not self.heatmap_visible:
            return

        allowed = ~self.excluded
        mean = self.grid[allowed].mean() if allowed.any() else 1.0
        # ratio >= 1: at least as likely as an average allowed cell -> yellow
        ratio = np.clip(self.grid / max(mean, 1e-12), 0.0, 1.0)
        level = np.round(ratio * (levels - 1)).astype(int)
        level[self.excluded] = 0

        diff = np.argwhere(level != self._tile_level)
        if len(diff) == 0:
            return
        if len(diff) > budget:
            change = np.abs(level - self._tile_level)[diff[:, 0], diff[:, 1]]
            diff = diff[np.argsort(-change, kind="stable")[:budget]]
        for iy, ix in diff:
            v = level[iy, ix] / (levels - 1)          # 0 = red, 1 = yellow
            p.changeVisualShape(int(self._tiles[iy, ix]), -1, rgbaColor=[1.0, v, 0.0, alpha])
            self._tile_level[iy, ix] = level[iy, ix]   # remember per tile, not all at once

    def set_heatmap_visible(self, visible):
        """Show or hide the heatmap (operator toggle), e.g. to see the real map underneath.
        The probability map keeps updating while hidden; showing it again repaints every tile."""
        if not hasattr(self, "_tiles") or visible == self.heatmap_visible:
            return
        self.heatmap_visible = visible
        if visible:
            self._tile_level[:] = -1                   # force a full repaint
            self.draw_heatmap(alpha=self._heat_alpha, budget=self._tiles.size)
        else:
            for tile in self._tiles.flat:
                p.changeVisualShape(int(tile), -1, rgbaColor=[0, 0, 0, 0])