"""Map-guided search (PBI: probability map drives navigation after the zone plan).

When the zone plan is finished (or empty), the drone keeps searching on its own: it asks the
probability map for the most likely cell, flies to a VIEWPOINT beside that cell, turns the
camera toward it and looks for a moment, then asks the map again. Every look lowers the
map where the person was not seen (ProbabilityMap.observe_camera), so the next target moves on.

Why a viewpoint and not the cell itself: the camera is tilted 40 deg down from the direction of
travel, so the ground straight under the drone is outside its view. Hovering ON the target would
never observe it. Standing off by about (altitude / tan 40 deg) puts the target in the middle of
the image. The viewpoint is chosen so the line of sight misses tree trunks and canopies, and a
repeated visit to the same cell looks from a different side (a person under a tree may only be
visible from one side).

Which cell: not just the single most likely cell. A lone hot cell 20 m away is a poor trip if a
patch of fairly likely cells is 4 m away, so every cell is scored by
    (probability inside the area the camera covers when looking at it) / (time to fly there + look)
and the best score wins ("most probability found per second"). This sweeps likely regions
instead of zig-zagging across the map between single cells.

Loop prevention (PBI tasks / AC-03):
  * targets within MIN_TARGET_DIST (2 m) of the drone are skipped;
  * the same cell is never selected more than MAX_REPEATS (3) times in a row;
  * the target cell must be inside the camera view during the look (recorded in `log`).

Pure Python + math (no PyBullet), so it is unit-tested without the simulator.
"""
import math

import numpy as np

MIN_TARGET_DIST = 2.0      # m, skip targets this close to the drone
MAX_REPEATS = 3            # max times the same cell may be selected in a row
DISTANCE_WEIGHT = 0.1      # fallback best_target(): mild preference for nearby cells
FOOTPRINT_R = 2.5          # m, radius of ground the camera covers well during a look
LOOK_OVERHEAD_S = 3.5      # s, turn + look at every target (LOOK_S + turning)
DEFAULT_SPEED = 2.0        # m/s, object_avoidance.V_MAX (main.py passes the real value)
LOOK_S = 2.0               # s to look at a target once the camera faces it
MAX_LOOK_S = 8.0           # s, longest look while the detector reports a possible sighting
TURN_RATE = 1.5            # rad/s, same turn rate main.py uses for the drone's nose
CAM_PITCH_DEG = 40.0       # camera tilt below the horizon (target_detection.CAM_PITCH_DEG)
GROUND_Z = 0.3             # height of the ground points the camera checks (frustum_mask)
BEARINGS = 8               # candidate viewpoints around a target
TREE_CLEARANCE = 1.3       # m, a viewpoint must be this far from a trunk (canopy 0.75 + margin)
EDGE_MARGIN = 0.8          # m, viewpoints stay this far inside the map edge
# Tree collision shapes (env_sim.build_world): trunk cylinder r 0.2 centred on the tree position
# (half-height 1.25), canopy sphere r 0.75 centred 1.75 m above it.
CANOPY_DZ, CANOPY_R = 1.75, 0.75
TRUNK_R, TRUNK_HALF = 0.2, 1.25


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def standoff(alt):
    """Horizontal distance at which a ground point is in the centre of the camera image."""
    return max(1.5, min(4.0, (alt - GROUND_Z) / math.tan(math.radians(CAM_PITCH_DEG))))


def line_of_sight_clear(eye, target, trees, margin=0.15, samples=24):
    """True if the straight line from eye (x, y, z) to the ground point target (x, y) misses every
    tree (trunk cylinder + canopy sphere). Trees are env_sim.OBSTACLE_POSITIONS entries (x, y, z)."""
    tz = GROUND_Z
    for k in range(1, samples):
        s = k / samples
        x = target[0] + s * (eye[0] - target[0])
        y = target[1] + s * (eye[1] - target[1])
        z = tz + s * (eye[2] - tz)
        for t in trees:
            dx, dy = x - t[0], y - t[1]
            tz0 = t[2] if len(t) > 2 else 0.5
            if z <= tz0 + TRUNK_HALF and dx * dx + dy * dy <= (TRUNK_R + margin) ** 2:
                return False
            if dx * dx + dy * dy + (z - tz0 - CANOPY_DZ) ** 2 <= (CANOPY_R + margin) ** 2:
                return False
    return True


class MapGuidedSearch:
    """Chooses map-guided targets and runs the fly -> look -> choose cycle.

    main.py calls update() every logic step while the zone plan is empty and the navigator is
    idle (DONE). update() returns a viewpoint (x, y) when the drone must fly somewhere new,
    else None (hover and look; read `heading` for the camera / nose direction).
    """

    def __init__(self, belief, trees=(), map_half=None, blocked=None, log=print, speed=DEFAULT_SPEED):
        self.belief = belief
        self.speed = max(0.1, speed)
        self.trees = [tuple(t) if len(t) > 2 else (t[0], t[1], 0.5) for t in trees]
        self.map_half = map_half
        self.blocked = blocked        # optional f(x, y) -> True if the drone can't hover there
        self.log = log
        self.phase = "IDLE"           # IDLE -> FLY -> LOOK -> FLY ...
        self.target = None            # (x, y) centre of the selected cell
        self.cell = None              # (ix, iy) of the selected cell
        self.viewpoint = None         # (x, y) the drone flies to
        self.heading = 0.0            # camera / nose heading during LOOK (rad)
        self.look_t = 0.0
        self.observed = False         # target was inside the camera view during this look
        self.streak = 0               # times self.cell was selected in a row
        self.selections = []          # every selected cell (ix, iy), in order
        self.log_rows = []            # (t, target, viewpoint, observed) per finished look

    @property
    def active(self):
        return self.phase != "IDLE"

    def cancel(self):
        """The operator gave the drone a zone to search again: stop the map-guided leg."""
        self.phase = "IDLE"
        self.target = self.cell = self.viewpoint = None

    # ------------------------------------------------------------ choosing --
    def choose(self, drone_xy, alt=2.5):
        """Pick the next target cell and a viewpoint for it. Returns (target, viewpoint)."""
        b = self.belief
        skip = [self.cell] if self.cell is not None and self.streak >= MAX_REPEATS else []
        cell = self.best_rate_cell(drone_xy, alt, skip)
        if cell is None:               # every cell is near the drone: drop the distance rule
            cell = b.best_cell(drone_xy, DISTANCE_WEIGHT, 0.0, skip)
        if cell is None:               # nothing allowed at all (should not happen)
            cell = b.best_cell(drone_xy, DISTANCE_WEIGHT)
        if cell is None:
            return None, None
        self.streak = self.streak + 1 if cell == self.cell else 1
        self.cell = cell
        self.selections.append(cell)
        self.target = b.to_world(*cell)
        self.viewpoint = self._viewpoint(self.target, drone_xy, alt, self.streak - 1)
        return self.target, self.viewpoint

    def coverage(self):
        """Probability inside FOOTPRINT_R of every cell (box sum via an integral image)."""
        b = self.belief
        r = max(1, int(round(FOOTPRINT_R / b.cell)))
        g = np.pad(b.grid, r)
        ii = np.zeros((g.shape[0] + 1, g.shape[1] + 1))
        ii[1:, 1:] = g.cumsum(0).cumsum(1)
        k = 2 * r + 1
        return ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]

    def best_rate_cell(self, drone_xy, alt=2.5, skip=()):
        """(ix, iy) with the most probability covered per second of flying + looking, or None.
        Never an excluded cell, a red (near-impossible) cell, or one within MIN_TARGET_DIST."""
        b = self.belief
        d = np.hypot(b.cx - drone_xy[0], b.cy - drone_xy[1])
        fly = np.maximum(0.0, d - standoff(alt)) / self.speed
        score = self.coverage() / (fly + LOOK_OVERHEAD_S)
        bad = b.excluded | (d < MIN_TARGET_DIST)
        if hasattr(b, "red_mask"):
            red = b.red_mask()
            if not (red | bad).all():
                bad = bad | red
        for ix, iy in skip:
            bad[iy, ix] = True
        if bad.all():
            return None
        score[bad] = -1.0
        iy, ix = np.unravel_index(np.argmax(score), score.shape)
        return int(ix), int(iy)

    def _viewpoint(self, target, drone_xy, alt, rotate=0):
        """Spot to look at `target` from: standoff() away, clear of trees, inside the map, with a
        clear line of sight if possible, preferring the shortest flight. A repeat visit (rotate=1,
        2) takes the next-best side so the target is seen from a new angle."""
        r = standoff(alt)
        half = self.map_half
        options = []
        for k in range(BEARINGS):
            a = 2 * math.pi * k / BEARINGS
            v = (target[0] + r * math.cos(a), target[1] + r * math.sin(a))
            if half is not None and (abs(v[0]) > half - EDGE_MARGIN or abs(v[1]) > half - EDGE_MARGIN):
                continue
            if any(math.hypot(v[0] - t[0], v[1] - t[1]) < TREE_CLEARANCE for t in self.trees):
                continue
            if self.blocked is not None and self.blocked(*v):
                continue
            clear = line_of_sight_clear((v[0], v[1], alt), target, self.trees)
            options.append((not clear, math.hypot(v[0] - drone_xy[0], v[1] - drone_xy[1]), v))
        if not options:                       # boxed in: look from wherever the drone is
            return tuple(drone_xy)
        options.sort()
        n_clear = sum(1 for o in options if not o[0]) or len(options)
        return options[rotate % n_clear][2]

    # --------------------------------------------------------------- cycle --
    def update(self, drone_pos, yaw, dt, t, sighting=False, in_view=None):
        """One logic step while the navigator is idle and the zone plan is empty.
        drone_pos: (x, y, z); yaw: current nose heading; sighting: detector has a possible
        sighting; in_view: f((x, y, z)) -> True if the point is in the camera view.
        Returns a viewpoint to fly to, or None to keep hovering (use self.heading)."""
        if self.phase == "IDLE":
            return self._next(drone_pos)
        if self.phase == "FLY":                    # navigator is idle -> we arrived
            self.phase = "LOOK"
            self.look_t = 0.0
            self.observed = False
            self.heading = yaw
        # LOOK: turn to face the target, then hold the view
        want = math.atan2(self.target[1] - drone_pos[1], self.target[0] - drone_pos[0])
        err = _wrap(want - self.heading)
        self.heading = _wrap(self.heading + max(-TURN_RATE * dt, min(TURN_RATE * dt, err)))
        if abs(err) < 0.05:
            self.look_t += dt
        if in_view is not None and not self.observed:
            self.observed = bool(in_view((self.target[0], self.target[1], GROUND_Z)))
        if self.look_t >= LOOK_S and not (sighting and self.look_t < MAX_LOOK_S):
            self.log_rows.append((round(t, 2), self.target, self.viewpoint, self.observed))
            if not self.observed:
                self.log(f"[map] warning: cell {fmt(self.target)} was not in the camera view")
            return self._next(drone_pos)
        return None

    def _next(self, drone_pos):
        target, vp = self.choose(drone_pos[:2], drone_pos[2])
        if target is None:
            self.phase = "IDLE"
            return None
        self.phase = "FLY"
        return vp

    def most_likely(self):
        """Most likely remaining cell overall (for the final report)."""
        return self.belief.best_target()


def fmt(xy):
    return f"({xy[0]:.1f}, {xy[1]:.1f})"


CONFIRM_MAX_S = 8.0        # s, longest the drone holds still to confirm one possible sighting
CONFIRM_COOLDOWN_S = 3.0   # s after a failed confirmation before a new sighting stops the drone


class SightingConfirm:
    """When the detector reports a possible sighting, stop and look until it is confirmed.

    The detector only declares FOUND after its confidence builds up over a second or more of
    steady view. A drone that keeps flying (or turns away) loses the person from the camera
    and the sighting is dropped ("LOST"); that was the main reason searches ran long. While
    `active`, main.py holds the drone in place and points the camera at `heading` (toward the
    detector's estimate of where the person is). Works in the zone phase and the map phase.
    """

    def __init__(self):
        self.active = False
        self.heading = 0.0
        self.started = None
        self.until = -1.0           # sim time before which new sightings are ignored
        self.count = 0              # confirmations attempted

    def update(self, state, estimate, drone_pos, yaw, dt, t):
        """state: detector state ("SEARCHING" / "SIGHTING" / "FOUND"); estimate: its (x, y, z)
        guess of the person's position or None. Returns True while the drone must hold."""
        if not self.active:
            if state != "SIGHTING" or t < self.until or estimate is None:
                return False
            self.active, self.started, self.heading = True, t, yaw
            self.count += 1
        if state != "SIGHTING" or t - self.started >= CONFIRM_MAX_S:
            self.active = False                    # confirmed (FOUND) or lost / timed out
            self.until = t + CONFIRM_COOLDOWN_S
            return False
        if estimate is not None:
            want = math.atan2(estimate[1] - drone_pos[1], estimate[0] - drone_pos[0])
            err = _wrap(want - self.heading)
            self.heading = _wrap(self.heading + max(-TURN_RATE * dt, min(TURN_RATE * dt, err)))
        return True
