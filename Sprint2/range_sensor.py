"""
Contributor: Chase Vanderlip
File Description: Ray layouts for the drone's simulated range sensor (used by object_avoidance.Navigator.sense()).

A layout only decides WHERE the rays go. Navigator.sense() casts them with one p.rayTestBatch call and
decides what each hit means (obstacle, casualty, ground, the drone itself).

BaselineLayout  The original sensor, kept unchanged for A/B testing (`--sensor baseline`):
                72 yaw rays x 3 pitch layers (-25, 0, +25 deg) out to 5 m, plus 3 parallel corridor rays
                along the direction of travel. The fan is fixed in the world, so a thin pole that sits
                between two rays stays invisible until the drone is almost touching it.

EnhancedLayout  The refined sensor (default). Same scan rate (40 Hz) and 252 rays per scan instead of 219,
                but every ray direction is computed once at start-up, so a scan is no slower than
                before (AC-03). Five parts:
                1. Ring (144 rays, 5 m): 48 yaw rays x 3 layers (level and +-tilt) that ROTATE a
                   quarter of the 7.5 deg spacing every scan, while the tilt steps through 24, 12, 18,
                   6 deg. Over 4 scans (0.1 s, 0.2 m of travel at cruise) the ring has looked every
                   1.875 deg in yaw and at 0, +-6 .. +-24 deg in pitch. Hits stay in the persistent
                   obstacle map, so nothing is lost between scans.
                2. Fovea (85 rays, 3.5 m): a cone locked to the direction of travel. A 6 x 6 grid
                   (8 deg spacing, jittered 4 deg on alternate scans, +-24 deg) catches small compact
                   objects (boxes, balls, sign plates). 12 "spokes" with log-spaced radii (0.4, 1.3, 4,
                   13 deg) catch thin poles, wires and struts: as the drone closes in, an object beside
                   the flight line moves outward across the spokes, so even a 1.5 cm rod or a 2 cm wire
                   crosses one well before the stop distance. Plus one ray dead ahead.
                3. Corridor (3 rays, 3 m): the original parallel rays at -0.4, 0, +0.4 m.
                4. Downward push-broom (13 rays, 4 m): a fan across the direction of travel, 30-90 deg
                   below the horizon, that sweeps the ground as the drone moves (low obstacles and the
                   casualty under or beside the flight line).
                5. Up-looking umbrella (7 rays, 2 m): straight up and 6 rays at 60 deg, so lines and
                   roofs overhead are known before the drone decides to climb.
"""
import math

# ------------------------------------------------------------- shared ray settings --
SENSOR_START = 0.30            # Rays start outside the drone's own body (m).

# --- Baseline (original object_avoidance.py values) ---
SENSOR_RANGE = 5.0
YAW_RAYS = 72                  # 5 deg resolution.
PITCHES_DEG = (-25.0, 0.0, 25.0)
CORRIDOR_RANGE = 3.0           # Forward rays along the direction of travel.
CORRIDOR_OFFSETS = (-0.4, 0.0, 0.4)

# --- Enhanced ---
PHASES = 4                     # Interleave phases (1 scan each; 4 scans = 0.1 s at 40 Hz).
PHASE_ORDER = (0.0, 0.5, 0.25, 0.75)   # Fraction of a ray spacing per phase (coarse gaps first).

RING_RANGE = 5.0
RING_YAW_RAYS = 48             # 7.5 deg per scan -> 1.875 deg over the 4 phases
# 3 layers per scan: level, and +-RING_TILT_DEG[phase]. Over 4 scans the ring covers 0, +-6 .. +-24 deg,
# so thin horizontal things beside the drone (a wire's end) are not lost between two layers.
RING_TILT_DEG = (24.0, 12.0, 18.0, 6.0)

FOVEA_RANGE = 3.5              # Forward cone, locked to the heading.
FOVEA_GRID_DEG = (-24.0, -16.0, -8.0, 0.0, 8.0, 16.0)     # + 4 deg on alternate phases: 4 deg grid, -24..+20
FOVEA_SPOKE_DEG = (0.4, 1.3, 4.0, 13.0)                    # log-spaced (ratio ~3)
FOVEA_SPOKE_DIRS = 12          # every 30 deg (includes the 4 axes)

BROOM_RANGE = 4.0              # Downward fan across the direction of travel.
BROOM_DEPRESSION_DEG = tuple(30.0 + 10.0 * k for k in range(7))   # 30 .. 90 deg below the horizon
BROOM_JITTER_DEG = 10.0        # Phase jitter (x PHASE_ORDER) fills the gaps between fan rays.

UP_RANGE = 2.0                 # Overhead umbrella (>= one climb step + clearance).
UP_ELEVATIONS_DEG = ((90.0, 1), (60.0, 6))


class BaselineLayout:
    """The original fixed fan + corridor (unchanged geometry and arithmetic, so A/B runs are exact)."""
    name = "baseline"
    see_through_target = False   # original: a ray that hits the casualty stops there
    tube_filter = False          # original: every corridor hit counts toward "distance ahead"

    def rays(self, pos, heading):
        """Return (froms, tos, first_forward_index). Rays from first_forward_index on are the
        forward rays whose hits give the distance to whatever is ahead."""
        px, py, pz = pos
        froms, tos = [], []
        for pd in PITCHES_DEG:
            cp, sp = math.cos(math.radians(pd)), math.sin(math.radians(pd))
            for k in range(YAW_RAYS):
                yaw = 2 * math.pi * k / YAW_RAYS
                d = (cp * math.cos(yaw), cp * math.sin(yaw), sp)
                froms.append([px + d[0] * SENSOR_START, py + d[1] * SENSOR_START, pz + d[2] * SENSOR_START])
                tos.append([px + d[0] * SENSOR_RANGE, py + d[1] * SENSOR_RANGE, pz + d[2] * SENSOR_RANGE])
        n_scan = len(froms)
        h = heading
        if h is not None:
            perp = (-h[1], h[0])
            for off in CORRIDOR_OFFSETS:
                ox, oy = px + perp[0] * off, py + perp[1] * off
                froms.append([ox + h[0] * SENSOR_START, oy + h[1] * SENSOR_START, pz])
                tos.append([ox + h[0] * CORRIDOR_RANGE, oy + h[1] * CORRIDOR_RANGE, pz])
        return froms, tos, n_scan

    def describe(self):
        n = YAW_RAYS * len(PITCHES_DEG) + len(CORRIDOR_OFFSETS)
        return f"baseline: {n} rays/scan ({YAW_RAYS} yaw x {len(PITCHES_DEG)} pitch + {len(CORRIDOR_OFFSETS)} corridor)"


def _unit(yaw_deg, pitch_deg):
    y, pt = math.radians(yaw_deg), math.radians(pitch_deg)
    return (math.cos(pt) * math.cos(y), math.cos(pt) * math.sin(y), math.sin(pt))


class EnhancedLayout:
    """Interleaved ring + heading-locked fovea + corridor + push-broom + umbrella.

    Ray directions are built once per phase in __init__. A scan only adds the drone position
    (world-fixed rays) or rotates by the heading (heading-locked rays), which is far cheaper than
    the per-scan trigonometry of the baseline and pays for the extra rays.
    """
    name = "enhanced"
    see_through_target = True    # re-cast rays that hit the casualty, so it never hides an obstacle
    tube_filter = True           # only forward hits inside the drone's swept tube set "distance ahead"

    def __init__(self):
        self.scan_no = 0
        self.last_heading = (1.0, 0.0)
        self.world = []          # per phase: list of (fx, fy, fz, tx, ty, tz) offsets from the drone
        self.local = []          # per phase: (forward, left, up) components, scaled to start/end
        for ph in range(PHASES):
            frac = PHASE_ORDER[ph]
            w = []
            # 1. ring, rotated by frac of the yaw spacing
            step = 360.0 / RING_YAW_RAYS
            tilt = RING_TILT_DEG[ph]
            for pd in (-tilt, 0.0, tilt):
                for k in range(RING_YAW_RAYS):
                    w.append(_unit((k + frac) * step, pd))
            # 5. umbrella (world-fixed; rotated with the ring phase)
            for elev, count in UP_ELEVATIONS_DEG:
                for k in range(count):
                    w.append(_unit((k + frac) * 360.0 / count, elev))
            n_ring = RING_YAW_RAYS * 3
            ranges = [RING_RANGE] * n_ring + [UP_RANGE] * (len(w) - n_ring)
            self.world.append([(d[0] * SENSOR_START, d[1] * SENSOR_START, d[2] * SENSOR_START,
                                d[0] * r, d[1] * r, d[2] * r) for d, r in zip(w, ranges)])

            # Heading-locked rays, as (forward, left, up) unit components + range.
            loc = []
            # 4. push-broom, both sides (straight down only once)
            jit = BROOM_JITTER_DEG * frac
            for dep in BROOM_DEPRESSION_DEG:
                dep = min(90.0, dep + jit)
                c, s = math.cos(math.radians(dep)), math.sin(math.radians(dep))
                sides = (1.0,) if dep >= 89.999 else (1.0, -1.0)
                for side in sides:
                    loc.append(((0.0, side * c, -s), BROOM_RANGE))
            self.n_broom = len(loc)
            # 2a. fovea grid (jittered by half a cell on alternate phases in yaw and pitch)
            gy = 4.0 if ph in (1, 3) else 0.0
            gp = 4.0 if ph in (2, 3) else 0.0
            for yd in FOVEA_GRID_DEG:
                for pd in FOVEA_GRID_DEG:
                    loc.append((_unit(yd + gy, pd + gp), FOVEA_RANGE))
            # 2b. fovea spokes (fixed: the drone's own motion sweeps objects across them)
            for k in range(FOVEA_SPOKE_DIRS):
                a = 2 * math.pi * k / FOVEA_SPOKE_DIRS
                for r in FOVEA_SPOKE_DEG:
                    loc.append((_unit(r * math.cos(a), r * math.sin(a)), FOVEA_RANGE))
            loc.append((_unit(0.0, 0.0), FOVEA_RANGE))          # dead ahead
            self.n_fovea = len(loc) - self.n_broom
            # Stored as (from fwd, left, up, to fwd, left, up, lateral offset in m).
            self.local.append([(d[0] * SENSOR_START, d[1] * SENSOR_START, d[2] * SENSOR_START,
                                d[0] * r, d[1] * r, d[2] * r, 0.0) for d, r in loc])
            # 3. corridor (parallel rays, lateral offset in metres)
            for off in CORRIDOR_OFFSETS:
                self.local[-1].append((SENSOR_START, 0.0, 0.0, CORRIDOR_RANGE, 0.0, 0.0, off))

    def rays(self, pos, heading):
        """Return (froms, tos, first_forward_index); see BaselineLayout.rays."""
        px, py, pz = pos
        ph = self.scan_no % PHASES
        self.scan_no += 1
        froms = [(px + a, py + b, pz + c) for a, b, c, _, _, _ in self.world[ph]]
        tos = [(px + d, py + e, pz + f) for _, _, _, d, e, f in self.world[ph]]
        h = heading if heading is not None else self.last_heading
        self.last_heading = h
        hx, hy = h
        loc = self.local[ph]
        # forward f = (hx, hy, 0), left l = (-hy, hx, 0): world = a*f + b*l + c*z (+ lateral offset)
        froms += [(px + a * hx - (b + o) * hy, py + a * hy + (b + o) * hx, pz + c)
                  for a, b, c, _, _, _, o in loc]
        tos += [(px + d * hx - (e + o) * hy, py + d * hy + (e + o) * hx, pz + f)
                for _, _, _, d, e, f, o in loc]
        first_fwd = len(self.world[ph]) + self.n_broom
        return froms, tos, first_fwd

    def describe(self):
        n = len(self.world[0]) + len(self.local[0])
        return (f"enhanced: {n} rays/scan (ring {RING_YAW_RAYS}x3 interleaved x{PHASES}, "
                f"fovea {self.n_fovea}, corridor {len(CORRIDOR_OFFSETS)}, broom {self.n_broom}, "
                f"umbrella {sum(c for _, c in UP_ELEVATIONS_DEG)})")


LAYOUTS = {"baseline": BaselineLayout, "enhanced": EnhancedLayout}


def make_layout(name):
    """Build a sensor layout by name ('baseline' or 'enhanced')."""
    try:
        return LAYOUTS[name]()
    except KeyError:
        raise ValueError(f"unknown sensor layout {name!r}; choose from {sorted(LAYOUTS)}") from None
