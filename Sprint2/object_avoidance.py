"""
Contributor: Chase Vanderlip Mylan Bui (A* Integration)
File Description: 
1.Sense: simulated range sensor that uses pybullet ray casts, 
360 degree x 3 pitch layers and scans around the drone every few physics steps
2.Detect: These hits are stored in persistant obstacle map (2.5D: x, y + Height).
          The route is declared Blocked when:
            - the upcoming part of the planned route crosses an (inflated) obstacle, or
            - a forward "corridor" ray sees something closer than a hard-stop distance.
3.Response: in order of preference
    a) slow down / hover, then re-plan a detour (A* on the map) to the next waypoint
    b) if a waypoint is unreachable, skip to the next reachable waypoint
    c) if no horizontal detour exists, climb in steps and try again
    d) if still blocked, return to launch (RTL)
    e) if RTL is blocked too, hover and report "awaiting operator"
    After clearing an obstacle the drone steps back down to its cruise altitude.
"""
import argparse
import heapq
import math
import os
import sys
import time
import pybullet as p
import pybullet_data
import sim_config
from env_sim import DRONE_START_POS, DRONE_SCALE, OBSTACLE_POSITIONS, TARGET_POSITION, FLIGHT_PATH


# ------------------------------------- Tunables -------------------------------------
DT = 1.0 / 240.0               # Simulation time step in seconds (240 Hz).
SCAN_EVERY = 6                 # Physics steps between sensor scans (40 Hz).
V_MAX = 2.0                    # m/s cruise speed.
V_MIN = 0.3                    # m/s creep speed when something is ahead.
V_Z = 1.0                      # m/s vertical speed.

# Range sensor: a fan of rays around the drone, in 3 pitch layers (up/level/down).
SENSOR_RANGE = 5.0
SENSOR_START = 0.30            # Rays start outside the drone's own body.
YAW_RAYS = 72                  # 5 deg resolution.
PITCHES_DEG = (-25.0, 0.0, 25.0)

# Forward corridor: extra rays straight along the direction of travel. Gives the
# distance to whatever is directlly in the waym and drives slow-down.
CORRIDOR_RANGE = 3.0           # Forward rays along the direction of travel.
CORRIDOR_OFFSETS = (-0.4, 0.0, 0.4)
SLOW_DIST = 3.0                # Start slowing down inside this distance.
STOP_DIST = 0.9                # Hard stop / re-plan inside this distance.
LOOKAHEAD = 4.0                # How far along the route we check for blockage.

# Obstacle map.
CELL = 0.25                    # Map resolution (m).
# Inflation around every sensed hit (m). PBI-04: lowered from 0.70 to 0.50 for generated
# environments (AC-01). The drone's collision body is ~0.12 m in radius (0.08 m x DRONE_SCALE 1.5).
# At 0.45 one seeded run passed 0.29 m from a canopy, just under the 0.3 m target, so 0.50 adds a
# little margin. Gaps narrower than ~1 m may now count as blocked, but trees top out at 3.0 m, so
# the drone climbs over them instead.
SAFETY_RADIUS = 0.50
VERT_CLEAR = 0.75              # Vertical clearance band around the flight altitude (m).
Z_BUCKET = 0.25
GOAL_PROBE_CELLS = 2000        # PBI-04 AC-03: max cells the "is the goal walled in?" check explores


# Reacting to blocked route.
CLIMB_STEP = 1.0               # Metres gained per climb attempt.
MAX_ALT = 5.5                  # Climb threshold, don't go over this.
REPLAN_COOLDOWN = 0.5          # s between sensor-triggered re-plans (hard stops ignore this).

# GUI obstacle-hit marks (drawing only, no effect on navigation).
SHOW_HIT_MARKS = True          # False = don't draw LIDAR hits at all.
MAX_HIT_MARKS = 400            # Oldest marks are recycled past this many.
HIT_MARKS_PER_DRAW = 60        # Cap per redraw (every 0.25 s) to keep frames smooth.

# Only used for safety report.
CRASH_DIST = 0.0               # Body distance <= this counts as contact.
UNSAFE_DIST = 0.35             # Closer than this (propeller reach) counts as unsafe.


# ----------------------------------- Obstacle Map -----------------------------------
class ObstacleMap:
    """Persistent 2.5D map built only from sensor hits.

    Every hit point is stored with its height and inflated by SAFETY_RADIUS in x/y.
    A cell is blocked at a given flight altitude if any inflated hit lies within
    +-VERT_CLEAR of that altitude.
    """

    def __init__(self):
        self.half = sim_config.MAP_HALF       # map covers [-half, half] in x and y
        self.n = int(round(2 * self.half / CELL))
        self.block = {}                       # (ix, iy) -> set(z buckets)
        r = int(math.ceil(SAFETY_RADIUS / CELL))
        self.disc = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
                     if math.hypot(dx, dy) * CELL <= SAFETY_RADIUS]
        self.seen = set()                     # raw (cell, bucket) already inflated
        self.new_hits = []                    # world points, for debug drawing
        self._dead = None                     # (start, alt, map version), region a failed A* explored


    # --- coordinate helpers
    def to_cell(self, x, y):
        return (int(math.floor((x + self.half) / CELL)), int(math.floor((y + self.half) / CELL)))

    def to_xy(self, c):
        return (c[0] * CELL - self.half + CELL / 2, c[1] * CELL - self.half + CELL / 2)

    def in_bounds(self, c):
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    @staticmethod
    def _buckets(alt):
        lo = int(math.floor((alt - VERT_CLEAR) / Z_BUCKET))
        hi = int(math.floor((alt + VERT_CLEAR) / Z_BUCKET))
        return range(lo, hi + 1)

    # --- updates
    def add_hit(self, pt):
        c = self.to_cell(pt[0], pt[1])
        zb = int(math.floor(pt[2] / Z_BUCKET))
        if (c, zb) in self.seen:
            return False
        self.seen.add((c, zb))
        self.new_hits.append(pt)
        for dx, dy in self.disc:
            cc = (c[0] + dx, c[1] + dy)
            self.block.setdefault(cc, set()).add(zb)
        return True

    # --- queries
    def blocked_cell(self, c, alt):
        if not self.in_bounds(c):
            return True
        s = self.block.get(c)
        if not s:
            return False
        return any(b in s for b in self._buckets(alt))

    def blocked(self, x, y, alt):
        return self.blocked_cell(self.to_cell(x, y), alt)

    def line_free(self, a, b, alt):
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / (CELL / 2)))
        for i in range(n + 1):
            t = i / n
            if self.blocked(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), alt):
                return False
        return True

    def nearest_free(self, xy, alt, max_r=3.0):
        c0 = self.to_cell(*xy)
        if not self.blocked_cell(c0, alt):
            return xy
        best, best_d = None, 1e9
        r = int(max_r / CELL)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                c = (c0[0] + dx, c0[1] + dy)
                d = math.hypot(dx, dy)
                if d < best_d and not self.blocked_cell(c, alt):
                    best, best_d = c, d
        return self.to_xy(best) if best else None

    # --- A*
    def _blocked_grid(self, alt):
        """Flat bytearray over every cell (index ix * n + iy): 1 = blocked at this altitude.
        Built once per A* call so the search does plain array lookups (PBI-04 AC-03)."""
        n, buckets = self.n, self._buckets(alt)
        grid = bytearray(n * n)
        for (ix, iy), s in self.block.items():
            if 0 <= ix < n and 0 <= iy < n and any(b in s for b in buckets):
                grid[ix * n + iy] = 1
        return grid

    def _goal_enclosed(self, grid, gi, si, limit=GOAL_PROBE_CELLS):
        """
        Quick unreachable check (PBI-04 AC-03): flood outward from the goal cell, up to `limit`
        cells. Returns True if that region closes off without reaching the start, meaning the goal is
        walled in, so A* doesn't have to search the whole map from the start side to find that out.
        4-connected is enough: A*'s diagonal moves can't cut corners, so they reach the same cells.
        """
        n = self.n
        seen, stack = {gi}, [gi]
        while stack:
            cx, cy = divmod(stack.pop(), n)
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                x2, y2 = cx + dx, cy + dy
                if not (0 <= x2 < n and 0 <= y2 < n):
                    continue
                nb = x2 * n + y2
                if nb == si:
                    return False                 # reached the drone: not enclosed
                if nb in seen or grid[nb]:
                    continue
                if len(seen) >= limit:
                    return False                 # open area: let A* do the real search
                seen.add(nb)
                stack.append(nb)
        return True

    def astar(self, start_xy, goal_xy, alt):
        """
        Shortest obstacle-free path on the grid at altitude `alt`.
        start_xy: drone's current (x, y) in world meters.
        goal_xy:  destination (x, y) in world meters.
        alt:      flight altitude; only obstacles within +-VERT_CLEAR of it count as blocking.
        Returns a list of (x, y) points to fly (not including the start), or None if the
        goal is blocked or unreachable.
        """
        # Convert world positions to grid cells. Give up early if either end is off the map
        # or the goal itself is inside an obstacle.
        s, g = self.to_cell(*start_xy), self.to_cell(*goal_xy)
        if not self.in_bounds(s) or not self.in_bounds(g) or self.blocked_cell(g, alt):
            return None
        n = self.n
        si, gi = s[0] * n + s[1], g[0] * n + g[1]
        # A failed search already explored everything reachable from this start. If nothing new
        # has been sensed since, a goal outside that region is unreachable too: skip the search.
        key = (si, round(alt, 3), len(self.seen))
        if self._dead is not None and self._dead[0] == key and gi not in self._dead[1]:
            return None
        grid = self._blocked_grid(alt)
        if self._goal_enclosed(grid, gi, si):
            return None
        gx, gy = g
        d = 1.4142
        # 8-connected grid: (dx, dy, cost, index step). Straight moves cost 1, diagonals sqrt(2).
        moves = [(1, 0, 1.0, n), (-1, 0, 1.0, -n), (0, 1, 1.0, 1), (0, -1, 1.0, -1),
                 (1, 1, d, n + 1), (1, -1, d, n - 1), (-1, 1, d, -n + 1), (-1, -1, d, -n - 1)]

        # Heuristic: octile distance, the exact cost to the goal if nothing were in the way.
        # It never overestimates, so A* still returns the shortest path.
        def h(i):
            dx, dy = abs(i // n - gx), abs(i % n - gy)
            return (dx + dy) + (d - 2) * min(dx, dy)

        # openq: priority queue of (estimated total cost, cost so far, cell index), cheapest first.
        # came:  cell -> the cell we reached it from, used to rebuild the path.
        # gscore: cell -> cheapest known cost from the start.
        openq = [(h(si), 0.0, si)]
        came, gscore = {}, {si: 0.0}
        while openq:
            _, gc, cur = heapq.heappop(openq)
            # Reached the goal: walk the `came` links backwards to get the path, then smooth it.
            if cur == gi:
                path = [cur]
                while cur in came:
                    cur = came[cur]
                    path.append(cur)
                path.reverse()
                return self._smooth([self.to_xy((i // n, i % n)) for i in path], alt, start_xy, goal_xy)
            # Skip stale queue entries; a cheaper route to this cell was already found.
            if gc > gscore.get(cur, 1e9):
                continue
            cx, cy = divmod(cur, n)
            # Try each neighbor that is on the map and not blocked at this altitude.
            for dx, dy, cost, step in moves:
                x2, y2 = cx + dx, cy + dy
                if not (0 <= x2 < n and 0 <= y2 < n):
                    continue
                nb = cur + step
                if grid[nb]:
                    continue
                # Don't move diagonally past the corner of an obstacle.
                if dx and dy and (grid[cur + dx * n] or grid[cur + dy]):
                    continue  # no corner cutting
                # Record the neighbor if this is the cheapest way found to reach it so far.
                ng = gc + cost
                if ng < gscore.get(nb, 1e9):
                    gscore[nb] = ng
                    came[nb] = cur
                    heapq.heappush(openq, (ng + h(nb), ng, nb))
        # Queue ran out without reaching the goal: no path exists at this altitude.
        self._dead = (key, set(gscore))
        return None


    def _smooth(self, pts, alt, start_xy, goal_xy):
        """
        String-pulling: drop waypoints that have a free straight line between neighbours.
        Grid paths zigzag cell to cell; this turns them into a few straight legs.
        Returns the smoothed points, not including the start.
        """
        # Use the exact start/goal positions instead of their cell centers.
        pts = [start_xy] + pts[1:-1] + [goal_xy]
        out, i = [pts[0]], 0
        # From each kept point, jump to the farthest point that can be reached in a straight,
        # obstacle-free line, and skip everything in between.        
        while i < len(pts) - 1:
            j = len(pts) - 1
            while j > i + 1 and not self.line_free(pts[i], pts[j], alt):
                j -= 1
            out.append(pts[j])
            i = j
        return out[1:]            # first point is where we already are


# -------------------------------------------------------------- navigator --
class Navigator:
    """Sense -> detect blocked route -> respond. Owns the drone's motion."""

    def __init__(self, drone_id, floor_id, target_id, waypoints, gui, log):
        self.drone, self.floor, self.target = drone_id, floor_id, target_id
        self.gui, self.log = gui, log
        self.pos = list(DRONE_START_POS)
        self.wps = [(w[0], w[1]) for w in waypoints]
        self.cruise = waypoints[0][2]
        self.alt = self.cruise
        self.wp_next = 1
        self.route = [(self.wps[i][0], self.wps[i][1], i) for i in range(1, len(self.wps))]
        self.map = ObstacleMap()
        self.state = "NAV"       # NAV, CLIMB, RTL, DONE, RTL_DONE, ABORT
        self.clear_ahead = float("inf")
        self.tick = 0
        self.t = 0.0
        self.events = []
        self.replans = 0
        self.replan_ms = []          # wall time of every replan() call, in ms (PBI-04 AC-03)
        self.skipped = []            # (sim time, waypoint index) of every unreachable waypoint skipped

        self.emergency_stops = 0
        self.last_replan_t = -99.0
        self.target_seen = False
        self._debug_ids = []
        self._text_id = None
        self._last_draw = -1.0
        self._hit_ids = []           # ring buffer of obstacle-hit marks (debug line ids)
        self._hit_slot = 0
        self.horizontal_speed = 0.0  # ramp velocity instead of instant acceleration

    # ---------------------------------------------------------------- log --
    def event(self, kind, msg):
        self.events.append((self.t, kind, msg))
        self.log(f"[t={self.t:6.2f}s] {kind:<8} {msg}")

    # ------------------------------------------------------------- SENSE --
    def sense(self):
        px, py, pz = self.pos
        froms, tos = [], []
        for pd in PITCHES_DEG:
            cp, sp = math.cos(math.radians(pd)), math.sin(math.radians(pd))
            for k in range(YAW_RAYS):
                yaw = 2 * math.pi * k / YAW_RAYS
                d = (cp * math.cos(yaw), cp * math.sin(yaw), sp)
                froms.append([px + d[0] * SENSOR_START, py + d[1] * SENSOR_START, pz + d[2] * SENSOR_START])
                tos.append([px + d[0] * SENSOR_RANGE, py + d[1] * SENSOR_RANGE, pz + d[2] * SENSOR_RANGE])

        # forward corridor along the current direction of travel
        h = self._heading()
        n_scan = len(froms)
        if h is not None:
            perp = (-h[1], h[0])
            for off in CORRIDOR_OFFSETS:
                ox, oy = px + perp[0] * off, py + perp[1] * off
                froms.append([ox + h[0] * SENSOR_START, oy + h[1] * SENSOR_START, pz])
                tos.append([ox + h[0] * CORRIDOR_RANGE, oy + h[1] * CORRIDOR_RANGE, pz])

        results = p.rayTestBatch(froms, tos)
        self.clear_ahead = float("inf")
        for i, (uid, _link, _frac, hit, _n) in enumerate(results):
            if uid < 0 or uid == self.drone or uid == self.floor:
                continue
            if uid == self.target:                    # the casualty is not an obstacle
                if not self.target_seen:
                    self.target_seen = True
                    self.event("TARGET", f"casualty sighted near ({hit[0]:.1f}, {hit[1]:.1f})")
                continue
            self.map.add_hit(hit)
            if i >= n_scan and h is not None:
                self.clear_ahead = min(self.clear_ahead, (hit[0] - px) * h[0] + (hit[1] - py) * h[1])
        # an obstacle beyond the next route point is not "in the way" of this leg
        if self.route and self.clear_ahead != float("inf"):
            d0 = math.hypot(self.route[0][0] - px, self.route[0][1] - py)
            if self.clear_ahead > d0 + 0.3:
                self.clear_ahead = float("inf")

    def _heading(self):
        if not self.route:
            return None
        dx, dy = self.route[0][0] - self.pos[0], self.route[0][1] - self.pos[1]
        d = math.hypot(dx, dy)
        return (dx / d, dy / d) if d > 1e-6 else None

    # ------------------------------------------------------------ DETECT --
    def route_blocked(self, alt=None, lookahead=LOOKAHEAD):
        """True if the next `lookahead` metres of the route cross an inflated obstacle."""
        alt = self.alt if alt is None else alt
        cur = (self.pos[0], self.pos[1])
        travelled = 0.0
        for (x, y, _) in self.route:
            seg = math.hypot(x - cur[0], y - cur[1])
            n = max(1, int(seg / (CELL / 2)))
            for i in range(1, n + 1):
                t = i / n
                sx, sy = cur[0] + t * (x - cur[0]), cur[1] + t * (y - cur[1])
                if travelled + t * seg < 0.3:
                    continue
                if self.map.blocked(sx, sy, alt):
                    return True
                if travelled + t * seg > lookahead:
                    return False
            travelled += seg
            cur = (x, y)
        return False

    def replan(self):
        """Timed wrapper around _replan() (PBI-04 AC-03): records how long each replan takes."""
        t0 = time.perf_counter()
        ok = self._replan()
        self.replan_ms.append((time.perf_counter() - t0) * 1000.0)
        return ok


    # ----------------------------------------------------------- RESPOND --
    def _replan(self):
        """
        Plan a detour to the first reachable remaining waypoint at the current altitude.
        Tries the remaining waypoints in order and uses A* to find an obstacle-free path to each.
        Waypoints that can't be reached are skipped. In RTL mode the only goal is the launch point.
        Returns True if a new route was set, False if no remaining waypoint is reachable
        (on_blocked() then escalates to climbing or return-to-launch).
        """
        start = (self.pos[0], self.pos[1])
        # Route entries are (x, y, waypoint index). Detour points have index None, so this
        # keeps only the real waypoints still ahead of the drone.        
        remaining = [r[2] for r in self.route if r[2] is not None]
        # When returning to launch, the launch point (index -1) is the only goal.        
        if self.state == "RTL":
            remaining = [-1]
        for j in remaining:
            goal = self.wps[j] if j >= 0 else (DRONE_START_POS[0], DRONE_START_POS[1])
            # If the waypoint falls inside an obstacle's safety margin, move it to the
            # nearest free cell; A* can't plan into a blocked goal.
            snapped = self.map.nearest_free(goal, self.alt)
            if snapped is None:
                continue
            # Run A* from the drone's position. No path means try the next waypoint instead.
            path = self.map.astar(start, snapped, self.alt)
            if path is None:
                continue
            if snapped != goal:
                self.event("REPLAN", f"waypoint {j} sits inside an obstacle margin; moved to "
                                     f"({snapped[0]:.1f}, {snapped[1]:.1f})")
                        # Waypoints still in the route before j had no path: name each one (PBI-04 AC-02).
            if j >= 0:
                for s in remaining[:remaining.index(j)]:
                    sx, sy = self.wps[s]
                    self.skipped.append((self.t, s))
                    self.event("UNREACH", f"waypoint {s} at ({sx:.1f}, {sy:.1f}) is unreachable at "
                                          f"alt {self.alt:.1f} m - skipped; heading to waypoint {j}")

            # Build the new route: A* detour points (index None), then the reached waypoint
            # tagged with its index, then every later waypoint unchanged.            
            route = [(x, y, None) for (x, y) in path[:-1]]
            route.append((path[-1][0], path[-1][1], j if j >= 0 else None))
            if j >= 0:
                route += [(self.wps[i][0], self.wps[i][1], i) for i in range(j + 1, len(self.wps))]
            self.route = route
            self.replans += 1
            # Log the detour length for the event history.
            pts = [start] + path
            length = sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
                         for i in range(1, len(pts)))
            self.event("REPLAN", f"detour found: {len(path)} legs, ~{length:.1f} m to waypoint {j} "
                                 f"at alt {self.alt:.1f} m")
            self.draw_route()
            return True
        # Every remaining waypoint was unreachable at this altitude.        
        return False

    def on_blocked(self, reason, emergency=False):
        if not emergency and self.t - self.last_replan_t < REPLAN_COOLDOWN:
            return
        self.last_replan_t = self.t
        self.event("BLOCKED", reason)
        if emergency:
            self.emergency_stops += 1
        # a detour that keeps ending in emergency stops is not working: escalate (climb / RTL)
        if not (emergency and self.emergency_stops > 3) and self.replan():
            return
        self.emergency_stops = 0
        if self.state == "RTL":
            self.state = "ABORT"
            self.event("ABORT", "return-to-launch route blocked too; hovering, awaiting operator")
            return
        if self.alt + CLIMB_STEP <= MAX_ALT + 1e-6:
            self.alt += CLIMB_STEP
            self.state = "CLIMB"
            self.event("CLIMB", f"no horizontal detour; climbing to {self.alt:.1f} m to look over the obstacle")
            return
        # Every remaining waypoint is unreachable even at the maximum altitude: name each one (AC-02).
        for s in [r[2] for r in self.route if r[2] is not None]:
            sx, sy = self.wps[s]
            self.skipped.append((self.t, s))
            self.event("UNREACH", f"waypoint {s} at ({sx:.1f}, {sy:.1f}) is unreachable even at "
                                  f"{self.alt:.1f} m - skipped; returning to launch")
        
        self.event("RTL", "no route at maximum altitude; returning to launch")
        self.alt = max(self.alt, self.cruise)
        self.state = "RTL"
        self.route = []
        if not self.replan():
            self.state = "ABORT"
            self.event("ABORT", "no route home either; hovering, awaiting operator")

    def maybe_descend(self):
        if self.alt <= self.cruise + 1e-6 or self.state not in ("NAV", "RTL"):
            return
        target = max(self.cruise, self.alt - CLIMB_STEP)
        if (not self.map.blocked(self.pos[0], self.pos[1], target)
                and not self.route_blocked(alt=target, lookahead=LOOKAHEAD + 1.0)):
            self.alt = target
            self.event("DESCEND", f"path clear below; returning toward cruise altitude ({self.alt:.1f} m)")

    # ------------------------------------------------------------ UPDATE --
    def update(self, dt):
        self.t += dt
        if self.tick % SCAN_EVERY == 0:
            self.sense()
            if self.state in ("NAV", "RTL") and self.route:
                if self.route_blocked():
                    self.on_blocked("planned route crosses a detected obstacle")
                elif self.clear_ahead < STOP_DIST:
                    self.on_blocked(f"obstacle {self.clear_ahead:.1f} m dead ahead", emergency=True)
                else:
                    self.maybe_descend()
            elif self.state == "CLIMB" and abs(self.pos[2] - self.alt) < 0.05:
                self.state = "NAV"
                if not self.replan():
                    self.on_blocked("still blocked after climbing")
            self.draw_status()
        self.tick += 1
        self.move(dt)

    def move(self, dt):
        # vertical
        dz = self.alt - self.pos[2]
        self.pos[2] += max(-V_Z * dt, min(V_Z * dt, dz))
        if self.state not in ("NAV", "RTL") or not self.route:
            self.horizontal_speed = 0.0
            if self.state == "NAV" and not self.route and abs(self.pos[2] - self.alt) < 0.05:
                self.state = "DONE"
                self.event("DONE", "all waypoints reached")
            elif self.state == "RTL" and not self.route and abs(self.pos[2] - self.alt) < 0.05:
                self.state = "RTL_DONE"
                self.event("DONE", "back at launch point")
            return
        # horizontal: slow down as free space ahead shrinks
        scale = (self.clear_ahead - STOP_DIST) / (SLOW_DIST - STOP_DIST)
        speed = V_MIN + (V_MAX - V_MIN) * max(0.0, min(1.0, scale))
        if self.clear_ahead <= STOP_DIST:
            speed = 0.0
        tx, ty, tag = self.route[0]
        dx, dy = tx - self.pos[0], ty - self.pos[1]
        d = math.hypot(dx, dy)
        # Smooth acceleration/deceleration and slow down near the waypoint.
        # Never teleport across the remaining gap (especially at low speed).
        if d < 0.6:
            speed = min(speed, max(0.15, d * 2.0))
        acceleration = 1.8  # m/s^2
        self.horizontal_speed += max(-acceleration * dt,
                                     min(acceleration * dt, speed - self.horizontal_speed))
        step = min(d, self.horizontal_speed * dt)
        if d <= 0.015:
            self.pos[0], self.pos[1] = tx, ty
            self.route.pop(0)
            if tag is not None:
                self.emergency_stops = 0
                self.wp_next = tag + 1
                self.event("WAYPT", f"reached waypoint {tag}")
            return
        if step > 0:
            self.pos[0] += dx / d * step
            self.pos[1] += dy / d * step

    # -------------------------------------------------------- GUI drawing --
    def draw_route(self):
        if not self.gui:
            return
        for i in self._debug_ids:
            p.removeUserDebugItem(i)
        self._debug_ids = []
        pts = [(self.pos[0], self.pos[1])] + [(r[0], r[1]) for r in self.route]
        for a, b in zip(pts[:-1], pts[1:]):
            self._debug_ids.append(p.addUserDebugLine([a[0], a[1], self.alt], [b[0], b[1], self.alt],
                                                      [0, 1, 0], 2.0))

    def draw_status(self):
        if not self.gui:
            return
        # Obstacle hits are drawn as short red line marks, NOT addUserDebugPoints:
        # on some PyBullet builds debug points break the GUI font, so every button and
        # slider label goes blank as soon as the LIDAR starts reporting hits.
        # A fixed-size ring buffer reuses old ids, so the debug item count stays bounded.
        if SHOW_HIT_MARKS and self.map.new_hits and self.t - self._last_draw > 0.25:
            pts = self.map.new_hits[:HIT_MARKS_PER_DRAW]
            self.map.new_hits = self.map.new_hits[HIT_MARKS_PER_DRAW:]
            for pt in pts:
                a = [pt[0], pt[1], pt[2] - 0.08]
                b = [pt[0], pt[1], pt[2] + 0.08]
                if len(self._hit_ids) < MAX_HIT_MARKS:
                    self._hit_ids.append(p.addUserDebugLine(a, b, [1, 0, 0], 3.0))
                else:
                    i = self._hit_slot
                    self._hit_ids[i] = p.addUserDebugLine(a, b, [1, 0, 0], 3.0,
                                                          replaceItemUniqueId=self._hit_ids[i])
                    self._hit_slot = (i + 1) % MAX_HIT_MARKS
            self._last_draw = self.t
        elif not SHOW_HIT_MARKS:
            self.map.new_hits.clear()
        clear = "--" if self.clear_ahead == float("inf") else f"{self.clear_ahead:.1f} m"
        txt = f"{self.state} | alt {self.pos[2]:.1f} | ahead {clear} | replans {self.replans}"
        kw = dict(textColorRGB=[1, 1, 0], textSize=1.4)
        if self._text_id is None:
            self._text_id = p.addUserDebugText(txt, [self.pos[0], self.pos[1], self.pos[2] + 0.8], **kw)
        else:
            self._text_id = p.addUserDebugText(txt, [self.pos[0], self.pos[1], self.pos[2] + 0.8],
                                               replaceItemUniqueId=self._text_id, **kw)


# ------------------------------------------------------------ world setup --
def find_assets(user_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [user_dir] if user_dir else []
    cands += [here, os.path.join(here, "..", "..", "Environment3D_Spike"),
              os.path.join(here, "Environment3D_Spike"), os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "custom_drone.urdf")):
            return os.path.abspath(c)
    sys.exit("Could not find custom_drone.urdf. Use --assets <Environment3D_Spike folder>.")


def build_world(gui, assets):
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    floor_id = p.loadURDF("plane.urdf")
    tex = os.path.join(assets, "ground_texture.jpg")
    if gui and os.path.exists(tex):
        try:
            p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(tex))
        except Exception as e:  # texture is cosmetic only
            print("Texture not applied:", e)

    drone_id = p.loadURDF(os.path.join(assets, "custom_drone.urdf"), DRONE_START_POS,
                          p.getQuaternionFromEuler([0, 0, 0]), globalScaling=DRONE_SCALE)

    obstacle_ids = []
    for x, y, z in OBSTACLE_POSITIONS:
        tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.5)
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.5, rgbaColor=[0.4, 0.2, 0.05, 1])
        obstacle_ids.append(p.createMultiBody(0, tc, tv, [x, y, z]))
        lc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.75)
        lv = p.createVisualShape(p.GEOM_SPHERE, radius=0.75, rgbaColor=[0.1, 0.45, 0.1, 1])
        obstacle_ids.append(p.createMultiBody(0, lc, lv, [x, y, z + 1.75]))

    target_id = p.loadURDF("humanoid/humanoid.urdf", TARGET_POSITION,
                           p.getQuaternionFromEuler([1.5708, 0, 0]), globalScaling=0.25, useFixedBase=True)
    return floor_id, drone_id, target_id, obstacle_ids


def spawn_blocker(nav, obstacle_ids, height, width):
    """Surprise obstacle: a wall across the drone's direction of travel, ~4 m ahead."""
    h = nav._heading() or (1.0, 0.0)
    cx, cy = nav.pos[0] + h[0] * 4.0, nav.pos[1] + h[1] * 4.0
    yaw = math.atan2(h[1], h[0])
    half = [0.2, width / 2, height / 2]
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.7, 0.1, 0.1, 1])
    bid = p.createMultiBody(0, col, vis, [cx, cy, height / 2], p.getQuaternionFromEuler([0, 0, yaw]))
    obstacle_ids.append(bid)
    nav.event("WORLD", f"(simulation) wall {height:.1f} m high x {width:.1f} m wide appeared at ({cx:.1f}, {cy:.1f})")


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true", help="no GUI, run as fast as possible")
    ap.add_argument("--no-avoid", action="store_true", help="follow the path blindly (baseline)")
    ap.add_argument("--blocker", type=float, default=None, metavar="T", help="spawn a wall at sim time T (s)")
    ap.add_argument("--blocker-height", type=float, default=3.5, help="wall height in m (default 3.5)")
    ap.add_argument("--blocker-width", type=float, default=3.0, help="wall width in m (default 3; 24 spans the map)")
    ap.add_argument("--max-time", type=float, default=240.0, help="sim time limit (s)")
    ap.add_argument("--assets", default=None, help="folder containing custom_drone.urdf")
    args = ap.parse_args()

    gui = not args.headless
    assets = find_assets(args.assets)
    floor_id, drone_id, target_id, obstacle_ids = build_world(gui, assets)
    nav = Navigator(drone_id, floor_id, target_id, FLIGHT_PATH, gui, print)
    quat = p.getQuaternionFromEuler([0, 0, 0])
    print(f"Simulation initialised. Drone {drone_id}, target {target_id}, "
          f"{len(obstacle_ids)} obstacle bodies. Avoidance {'OFF' if args.no_avoid else 'ON'}.")

    min_clear, unsafe_steps, contact_steps = float("inf"), 0, 0
    blocker_done = False
    step = 0
    try:
        while nav.t < args.max_time and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
            if args.blocker is not None and not blocker_done and nav.t >= args.blocker:
                spawn_blocker(nav, obstacle_ids, args.blocker_height, args.blocker_width)
                blocker_done = True

            if args.no_avoid:                       # baseline: blind waypoint following
                nav.t += DT
                nav.tick += 1
                nav.clear_ahead = float("inf")
                nav.move(DT)
            else:
                nav.update(DT)

            p.resetBasePositionAndOrientation(drone_id, nav.pos, quat)
            p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()

            if step % 6 == 0:                       # safety metrics (ground truth, not used by the planner)
                d = 1e9
                for oid in obstacle_ids:
                    pts = p.getClosestPoints(drone_id, oid, 3.0)
                    if pts:
                        d = min(d, min(c[8] for c in pts))
                min_clear = min(min_clear, d)
                if d <= CRASH_DIST:
                    contact_steps += 1
                if d < UNSAFE_DIST:
                    unsafe_steps += 1
            if gui:
                p.resetDebugVisualizerCamera(5, 45, -30, nav.pos)
                time.sleep(DT)
            step += 1
    except KeyboardInterrupt:
        pass

    print("\n==== MISSION REPORT ====")
    print(f"final state        : {nav.state}")
    print(f"sim time           : {nav.t:.1f} s")
    print(f"re-plans           : {nav.replans}")
    print(f"obstacle cells seen: {len(nav.map.seen)}")
    print(f"casualty sighted   : {nav.target_seen}")
    print(f"closest approach   : {min_clear:.2f} m (drone body to obstacle surface)")
    print(f"unsafe samples     : {unsafe_steps}   (closer than {UNSAFE_DIST} m)")
    print(f"contact samples    : {contact_steps}")
    if gui and nav.state in ("DONE", "RTL_DONE", "ABORT"):
        try:
            while True:
                p.stepSimulation()
                time.sleep(DT)
        except KeyboardInterrupt:
            pass
    p.disconnect()


if __name__ == "__main__":
    main()
