"""Probability-of-area (belief) map for the SAR demo.

A grid where each cell holds P(missing person is in this cell).
It is updated by:
  - operator sightings        -> raise probability around a point
  - operator exclusions       -> zero probability in an area
  - drone coverage, no find   -> Bayesian "negative information" update
  - optional diffusion        -> person may be walking around
The mission uses it to decide where the drone should search next.

Detection itself is still decided by SearchMonitor; this map only guides WHERE to look.
"""
import numpy as np
import pybullet as p


class ProbabilityMap:
    def __init__(self, x_range=(-12.0, 12.0), y_range=(-12.0, 12.0),
                 cell_size=1.0, sensor_radius=2.0, pod=0.8):
        """
        x_range, y_range: world bounds in meters (match your env_sim floor)
        cell_size:        grid resolution in meters
        sensor_radius:    how far the drone's (simulated) camera can see
        pod:              probability of detection if the person IS in view
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
        self._points_id = None

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
        else:  # everything excluded/searched: fall back to uniform over allowed cells
            self.grid[~self.excluded] = 1.0
            self.grid /= max(self.grid.sum(), 1e-12)

    # ---------- updates from the operator ----------
    def add_sighting(self, xy, sigma=2.5, weight=0.6):
        """Witness report near xy. weight = how much we trust it (0..1)."""
        g = np.exp(-((self.cx - xy[0]) ** 2 + (self.cy - xy[1]) ** 2) / (2 * sigma ** 2))
        g /= g.sum()
        self.grid = (1 - weight) * self.grid + weight * g
        self._normalize()

    def exclude(self, xy, radius):
        """'Do not search this area' -> person assumed not there."""
        self.excluded |= self._disk(xy, radius)
        self._normalize()

    def mark_searched(self, xy, radius):
        """'This area has already been searched' (e.g., by a ground team)."""
        self.grid[self._disk(xy, radius)] *= (1 - self.pod)
        self._normalize()

    def set_prior(self, matrix):
        """Load an external probability matrix (e.g., from DSSE), resampled to this grid."""
        m = np.asarray(matrix, dtype=float)
        iy = np.arange(self.ny) * m.shape[0] // self.ny
        ix = np.arange(self.nx) * m.shape[1] // self.nx
        self.grid = m[np.ix_(iy, ix)].copy()
        self._normalize()

    # ---------- updates from the drone ----------
    def observe(self, drone_xy):
        """Drone looked here and did NOT find the person (Bayes update).
        Safe to call every step: only updates when the drone enters a new cell."""
        cell = self.to_cell(drone_xy)
        if cell == self._last_cell:
            return
        self._last_cell = cell
        self.grid[self._disk(drone_xy, self.sensor_radius)] *= (1 - self.pod)
        self._normalize()

    def observe_camera(self, monitor, drone_pos, trees=(), canopy_radius=1.0,
                       canopy_pod=0.3, ground_z=0.3):
        """Like observe(), but uses SearchMonitor's real camera footprint (forward-tilted
        wedge, FOV + max range) instead of a circle. Cells under a tree canopy get a
        lower detection probability, since the camera often can't see through it.
        Only updates when the drone enters a new cell."""
        cell = self.to_cell(drone_pos)
        if cell == self._last_cell:
            return
        self._last_cell = cell
        cam = list(drone_pos)
        for iy in range(self.ny):
            for ix in range(self.nx):
                pt = (self.cx[iy, ix], self.cy[iy, ix], ground_z)
                if not monitor.in_frustum(cam, pt):
                    continue
                pod = self.pod
                if any((pt[0] - t[0]) ** 2 + (pt[1] - t[1]) ** 2 <= canopy_radius ** 2 for t in trees):
                    pod = canopy_pod
                self.grid[iy, ix] *= (1 - pod)
        self._normalize()

    def diffuse(self, rate=0.05):
        """Spread probability to neighbors"""
        pad = np.pad(self.grid, 1, mode="edge")
        neighbors = (pad[:-2, 1:-1] + pad[2:, 1:-1] + pad[1:-1, :-2] + pad[1:-1, 2:]) / 4
        self.grid = (1 - rate) * self.grid + rate * neighbors
        self._normalize()

    # ---------- queries ----------
    def best_target(self, from_xy=None, distance_weight=0.0):
        """World (x, y) of the most promising cell. distance_weight > 0 prefers nearby cells."""
        score = self.grid.copy()
        if from_xy is not None and distance_weight > 0:
            d = np.hypot(self.cx - from_xy[0], self.cy - from_xy[1])
            score = score / (1 + distance_weight * d)
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        return self.to_world(ix, iy)

    def mass_near(self, xy, radius):
        """Total probability inside a circle (useful for ranking search zones)."""
        return float(self.grid[self._disk(xy, radius)].sum())

    # ---------- visualization ----------
    def draw(self, top_k=60, max_height=1.5):
        """Draw the top_k most likely cells as vertical sticks on the ground.
        Taller + redder = more likely. Uses debug lines (debug points can blank
        out PyBullet's GUI button labels on some builds)."""
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