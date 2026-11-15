"""
Contributor: Chase Vanderlip
File Description: Range-sensor verification runner. Compares the original ("baseline") and the refined
("enhanced") ray layouts of range_sensor.py on statically placed test obstacles and casualty positions,
and checks the acceptance criteria:

  AC-01  Objects of different sizes / shapes at various positions are detected within the stop
         distance (STOP_DIST, 0.9 m): the obstacle map must mark the flight path as blocked while the
         drone is still >= STOP_DIST from the obstacle surface (detect suite), and the closed-loop
         Navigator must fly courses with those obstacles on its legs without contact (avoid suite).
  AC-02  The casualty is never treated as an obstacle, wherever it is (open ground, on the flight
         line, under a canopy, beside a building or a pole, ...): none of its ray hits may reach the
         obstacle map, and the drone's navigation must be the same with or without the casualty
         there. It must also be identified: sighted, with its location estimated (target suite).
  AC-03  An enhanced scan takes < 20 % longer than a baseline scan (timing suite: the same poses
         replayed for both layouts, interleaved repetitions).

Suites:
  detect  open-loop: the drone flies straight at V_MAX toward one test obstacle and scans at the
          normal rate (every SCAN_EVERY physics steps). Cases vary obstacle type, lateral and
          vertical offset from the flight line, and the direction of travel.
  avoid   closed-loop: full Navigator (sense -> detect -> replan) on two courses in the default
          world, with test obstacles on the legs.
  target  closed-loop: the default survey route with the casualty placed in varied locations.
  sweep   like detect, but with random placements (type, offsets, direction of travel; seeded).
  timing  scan time per sense() call, and Navigator wall time per simulated second.

Usage (from Sprint2/):
    python range_sensor_runs.py                    # every suite, both layouts
    python range_sensor_runs.py --suite detect     # one suite
    python range_sensor_runs.py --quick            # fewer cases / repetitions
Outputs (in --out, default range_sensor_results/): detect.csv, avoid.csv, target.csv, timing.csv,
summary.md and, if matplotlib is installed, detect_ranges.png and scan_time.png.
"""
import argparse
import csv
import math
import os
import statistics
import time

import pybullet as p
import pybullet_data

import env_sim
import object_avoidance as oa
from env_sim import DRONE_START_POS, DRONE_SCALE, FLIGHT_PATH, OBSTACLE_POSITIONS, TARGET_ORIENTATION_EULER, TARGET_SCALE
from object_avoidance import Navigator, DT, SCAN_EVERY, STOP_DIST, V_MAX

LAYOUTS = ("baseline", "enhanced")
ALT = FLIGHT_PATH[0][2]          # cruise altitude (2.5 m)
TIME_LIMIT_PCT = 20.0            # AC-03
HERE = os.path.dirname(os.path.abspath(__file__))


# ======================================================================= helpers ==
class RayProbe:
    """Stands in for the pybullet module inside object_avoidance and records every ray result, so the
    runner can see which bodies were hit without changing the sensor code."""

    def __init__(self, real):
        self._p = real
        self.watch = set()        # obstacle body ids of interest
        self.watch_hit = False    # a watched body was hit since the last reset
        self.target = None
        self.target_hits = []     # hit tuples on the casualty (kept alive so id() stays unique)
        self.target_ids = set()

    def __getattr__(self, name):
        return getattr(self._p, name)

    def rayTestBatch(self, *a, **k):
        res = self._p.rayTestBatch(*a, **k)
        for r in res:
            uid = r[0]
            if uid in self.watch:
                self.watch_hit = True
            elif self.target is not None and uid == self.target:
                self.target_hits.append(r[3])
                self.target_ids.add(id(r[3]))
        return res


def install_probe():
    probe = RayProbe(p)
    oa.p = probe
    return probe


def remove_probe():
    oa.p = p


def connect():
    global _probe_shape
    _probe_shape = None                       # shape ids do not survive a disconnect
    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)


_probe_shape = None


def surface_dist(bodies, pos):
    """Distance (m) from the point pos (the drone centre) to the nearest surface of any body."""
    global _probe_shape
    if _probe_shape is None:
        _probe_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1e-4)
    best = float("inf")
    for b in bodies:
        r = p.getClosestPoints(bodyA=b, bodyB=-1, distance=8.0, collisionShapeB=_probe_shape,
                               collisionShapePositionB=pos)
        if r:
            best = min(best, min(c[8] for c in r))
    return best


def body_clearance(drone_id, bodies):
    """Drone body to obstacle surface distance (<= 0 is contact)."""
    d = float("inf")
    for b in bodies:
        pts = p.getClosestPoints(drone_id, b, 3.0)
        if pts:
            d = min(d, min(c[8] for c in pts))
    return d


def quat_axis_to(direction):
    """Quaternion that turns a shape's +z axis (cylinders) onto `direction`."""
    dx, dy, dz = direction
    n = math.sqrt(dx * dx + dy * dy + dz * dz)
    dx, dy, dz = dx / n, dy / n, dz / n
    ax, ay, az = -dy, dx, 0.0                      # z x d
    s = math.sqrt(ax * ax + ay * ay)
    if s < 1e-9:
        return [0, 0, 0, 1] if dz > 0 else [1, 0, 0, 0]
    ang = math.acos(max(-1.0, min(1.0, dz)))
    k = math.sin(ang / 2) / s
    return [ax * k, ay * k, az * k, math.cos(ang / 2)]


def _body(shape, half=None, radius=None, length=None, pos=(0, 0, 0), orn=(0, 0, 0, 1), rgba=(0.5, 0.5, 0.5, 1)):
    if shape == "box":
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba)
    elif shape == "cyl":
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=rgba)
    else:
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=rgba)
    return p.createMultiBody(0, col, vis, list(pos), list(orn))


# ============================================================ test obstacle catalogue ==
# Every obstacle is placed relative to a point C on the flight line, the direction of travel f and
# its left normal l: `lat` moves it sideways (m), `dz` up/down from the flight altitude (m).
# All placements reach into the drone's swept tube (surface within ~0.35 m of the flight line).
# Sizes: rod 1.5 cm, thin pole 5 cm, pole 15 cm, wire 2 cm, beam 8 cm, boxes 8 / 12 cm, ball 20 cm.
def spawn_obstacle(kind, c, yaw, lat=0.0, dz=0.0, alt=ALT):
    f = (math.cos(yaw), math.sin(yaw), 0.0)
    l = (-math.sin(yaw), math.cos(yaw), 0.0)
    x, y = c[0] + l[0] * lat, c[1] + l[1] * lat
    z = alt + dz
    q_yaw = p.getQuaternionFromEuler([0, 0, yaw])
    ids = []
    if kind == "tree":                       # same shapes as env_sim.build_world
        ids.append(_body("cyl", radius=0.2, length=2.5, pos=(x, y, 0.5), rgba=(0.4, 0.2, 0.05, 1)))
        ids.append(_body("sphere", radius=0.75, pos=(x, y, 2.25), rgba=(0.1, 0.45, 0.1, 1)))
    elif kind == "building":                 # 3 x 3 x 6 m block, front face at C
        ids.append(_body("box", half=[1.5, 1.5, 3.0], pos=(x + f[0] * 1.5, y + f[1] * 1.5, 3.0), orn=q_yaw))
    elif kind == "thin_pole":
        ids.append(_body("cyl", radius=0.025, length=4.0, pos=(x, y, 2.0)))
    elif kind == "pole":
        ids.append(_body("cyl", radius=0.075, length=4.0, pos=(x, y, 2.0)))
    elif kind == "rod":                      # 1.5 cm vertical rod / hanging cable
        ids.append(_body("cyl", radius=0.0075, length=4.0, pos=(x, y, 2.0)))
    elif kind == "wire":                     # power line across the path
        ids.append(_body("cyl", radius=0.01, length=5.0, pos=(x, y, z), orn=quat_axis_to(l)))
    elif kind == "wire_skew":                # crossing the path at 45 deg
        d = (f[0] + l[0], f[1] + l[1], 0.0)
        ids.append(_body("cyl", radius=0.01, length=5.0, pos=(x, y, z), orn=quat_axis_to(d)))
    elif kind == "guy_wire":                 # slanted 45 deg up across the path
        d = (l[0], l[1], 1.0)
        ids.append(_body("cyl", radius=0.015, length=6.0, pos=(x, y, z), orn=quat_axis_to(d)))
    elif kind == "beam":                     # 8 cm horizontal bar across the path
        ids.append(_body("box", half=[0.04, 2.5, 0.04], pos=(x, y, z), orn=q_yaw))
    elif kind == "tiny_box":                 # 8 cm floating box
        ids.append(_body("box", half=[0.04, 0.04, 0.04], pos=(x, y, z), orn=q_yaw))
    elif kind == "small_box":                # 12 cm floating box
        ids.append(_body("box", half=[0.06, 0.06, 0.06], pos=(x, y, z), orn=q_yaw))
    elif kind == "ball":                     # 20 cm floating sphere
        ids.append(_body("sphere", radius=0.10, pos=(x, y, z)))
    elif kind == "sign_face":                # 5 cm post + 0.8 x 0.5 m plate facing the drone
        ids.append(_body("cyl", radius=0.025, length=z - 0.25, pos=(x, y, (z - 0.25) / 2)))
        ids.append(_body("box", half=[0.015, 0.4, 0.25], pos=(x, y, z), orn=q_yaw))
    elif kind == "sign_edge":                # the same sign seen edge-on (3 cm thick)
        ids.append(_body("cyl", radius=0.025, length=z - 0.25, pos=(x, y, (z - 0.25) / 2)))
        ids.append(_body("box", half=[0.4, 0.015, 0.25], pos=(x, y, z), orn=q_yaw))
    elif kind == "wall":                     # 3 m wide x 4 m tall x 0.3 m wall across the path
        ids.append(_body("box", half=[0.15, 1.5, 2.0], pos=(x, y, 2.0), orn=q_yaw))
    elif kind == "alley_wall":               # 13 m long, 7 m tall wall along the path (too tall to climb)
        ids.append(_body("box", half=[6.5, 0.15, 3.5], pos=(x, y, 3.5), orn=q_yaw))
    elif kind == "end_wall":                 # 2.6 m wall closing the alley (climbable)
        ids.append(_body("box", half=[0.15, 1.25, 1.3], pos=(x, y, 1.3), orn=q_yaw))
    elif kind == "wire_overhead":            # 7 m line running along the path, dz above it
        ids.append(_body("cyl", radius=0.01, length=7.0, pos=(x, y, z), orn=quat_axis_to(f)))
    elif kind == "slab_overhead":            # 7 m x 3 m roof / bridge deck, underside dz above the path
        ids.append(_body("box", half=[3.5, 1.5, 0.05], pos=(x, y, z + 0.05), orn=q_yaw))
    elif kind == "low_wall":                 # wall whose top is just under the flight altitude
        ids.append(_body("box", half=[0.1, 2.0, z / 2], pos=(x, y, z / 2), orn=q_yaw))
    elif kind == "overhang":                 # 1.5 m deep x 3 m wide slab, underside just above the path
        ids.append(_body("box", half=[0.75, 1.5, 0.05], pos=(x + f[0] * 0.75, y + f[1] * 0.75, z + 0.05), orn=q_yaw))
    elif kind == "l_corner":                 # concave L of two 2 m walls; drone flies into the corner
        for s in (1.0, -1.0):
            a = yaw + s * math.radians(45.0)
            ux, uy = -math.cos(a), -math.sin(a)          # wall runs back toward the drone side
            ids.append(_body("box", half=[1.0, 0.1, 2.0], pos=(x + ux * 1.0 + f[0] * 0.15,
                                                             y + uy * 1.0 + f[1] * 0.15, 2.0),
                             orn=p.getQuaternionFromEuler([0, 0, a])))
    else:
        raise ValueError(kind)
    return ids


# (kind, lat, dz) cases for the detect suite.
DETECT_CASES = (
    [("tree", lat, 0.0) for lat in (0.0, 0.5, 0.9)]
    + [("building", lat, 0.0) for lat in (0.0, 1.0)]
    + [("thin_pole", lat, 0.0) for lat in (0.0, 0.05, 0.12, 0.25, 0.37)]
    + [("rod", lat, 0.0) for lat in (0.0, 0.04, 0.1, 0.2, 0.3)]
    + [("pole", lat, 0.0) for lat in (0.0, 0.2, 0.4)]
    + [("wire", 0.0, dz) for dz in (0.0, 0.15, -0.15, 0.3, -0.3)]
    + [("wire_skew", 0.0, dz) for dz in (0.0, 0.2, -0.2)]
    + [("guy_wire", lat, 0.0) for lat in (0.0, 0.2, -0.3)]
    + [("beam", 0.0, dz) for dz in (0.0, 0.25, -0.25)]
    + [("small_box", lat, dz) for lat, dz in ((0.0, 0.0), (0.15, 0.1), (-0.2, -0.15), (0.3, 0.2))]
    + [("tiny_box", lat, dz) for lat, dz in ((0.0, 0.0), (0.1, -0.1), (-0.15, 0.2), (0.25, 0.25))]
    + [("ball", lat, dz) for lat, dz in ((0.0, 0.0), (0.2, 0.15), (-0.25, -0.2))]
    + [("sign_face", lat, 0.0) for lat in (0.0, 0.5)]
    + [("sign_edge", lat, 0.0) for lat in (0.0, 0.2)]
    + [("low_wall", 0.0, dz) for dz in (-0.15, -0.3)]
    + [("overhang", 0.0, dz) for dz in (0.15, 0.3)]
    + [("l_corner", lat, 0.0) for lat in (0.0, 0.3)]
)
HEADINGS_DEG = (0.0, 2.5, 47.5, 123.7)      # 2.5 sits halfway between two baseline rays


# ======================================================================= detect ==
def detect_case(layout, kind, lat, dz, heading_deg, probe):
    """Fly straight at V_MAX toward the obstacle, scanning every SCAN_EVERY steps. Returns the
    obstacle-surface distance at the first ray hit and at the first scan whose obstacle map
    marks the path ahead as blocked."""
    connect()
    floor = p.loadURDF("plane.urdf")
    drone = p.loadURDF(os.path.join(HERE, "custom_drone.urdf"), [0, 0, ALT], globalScaling=DRONE_SCALE)
    yaw = math.radians(heading_deg)
    f = (math.cos(yaw), math.sin(yaw))
    ids = spawn_obstacle(kind, (0.0, 0.0), yaw, lat, dz)
    probe.watch, probe.watch_hit = set(ids), False
    start = (-6.0 * f[0], -6.0 * f[1])
    goal = (8.0 * f[0], 8.0 * f[1])
    nav = Navigator(drone, floor, -1, [(start[0], start[1], ALT), (goal[0], goal[1], ALT)], False,
                    lambda *_a: None, sensor=layout)
    nav.pos = [start[0], start[1], ALT]
    nav.route = [(goal[0], goal[1], 1)]
    hit_d = block_d = None
    step, s_done = 0, None
    while s_done is None or s_done < 1.0:
        if step % SCAN_EVERY == 0:
            p.resetBasePositionAndOrientation(drone, nav.pos, [0, 0, 0, 1])   # drone body moves too
            nav.sense()
            d = surface_dist(ids, nav.pos)
            if hit_d is None and probe.watch_hit:
                hit_d = d
            if block_d is None and nav.route_blocked():
                block_d = d
            if d <= 0.13:                           # drone body (0.12 m) touching: stop the run
                break
        nav.pos[0] += f[0] * V_MAX * DT
        nav.pos[1] += f[1] * V_MAX * DT
        along = nav.pos[0] * f[0] + nav.pos[1] * f[1]
        s_done = along if along > 0.6 else None     # flown past the obstacle point
        step += 1
    p.disconnect()
    return hit_d, block_d


def run_detect(quick):
    probe = install_probe()
    rows = []
    cases = DETECT_CASES[::3] if quick else DETECT_CASES
    heads = HEADINGS_DEG[:2] if quick else HEADINGS_DEG
    for kind, lat, dz in cases:
        for hd in heads:
            row = {"kind": kind, "lat_m": lat, "dz_m": dz, "heading_deg": hd}
            for lay in LAYOUTS:
                hit_d, block_d = detect_case(lay, kind, lat, dz, hd, probe)
                row[f"{lay}_hit_m"] = None if hit_d is None else round(hit_d, 2)
                row[f"{lay}_block_m"] = None if block_d is None else round(block_d, 2)
                row[f"{lay}_pass"] = block_d is not None and block_d >= STOP_DIST
            rows.append(row)
            print(f"detect {kind:<10} lat {lat:+.2f} dz {dz:+.2f} hdg {hd:5.1f} | "
                  + " | ".join(f"{lay} hit {fmt_d(row[f'{lay}_hit_m'])} block {fmt_d(row[f'{lay}_block_m'])} "
                               f"{'PASS' if row[f'{lay}_pass'] else 'FAIL'}" for lay in LAYOUTS))
    remove_probe()
    return rows


# Randomised sweep: (kind, lat range, dz range). Every draw still reaches into the drone's swept tube.
SWEEP_KINDS = [("rod", 0.3, 0.0), ("thin_pole", 0.35, 0.0), ("pole", 0.4, 0.0), ("wire", 0.0, 0.3),
               ("wire_skew", 0.0, 0.3), ("guy_wire", 0.35, 0.0), ("beam", 0.0, 0.3), ("tiny_box", 0.25, 0.25),
               ("small_box", 0.3, 0.3), ("ball", 0.3, 0.3), ("sign_edge", 0.35, 0.0), ("sign_face", 0.6, 0.0),
               ("tree", 0.9, 0.0), ("overhang", 0.0, 0.3), ("low_wall", 0.0, 0.3)]


def run_sweep(n, seed):
    """n random placements (kind, sideways / vertical offset, direction of travel), same draws for both
    layouts. Overhangs only get dz > 0 and low walls dz < 0 (their surface faces the drone)."""
    import random
    rng = random.Random(seed)
    probe = install_probe()
    rows = []
    for i in range(n):
        kind, lat_r, dz_r = SWEEP_KINDS[i % len(SWEEP_KINDS)]
        lat = round(rng.uniform(-lat_r, lat_r), 3)
        dz = round(rng.uniform(-dz_r, dz_r), 3)
        if kind == "overhang":
            dz = round(abs(dz) + 0.05, 3)
        elif kind == "low_wall":
            dz = round(-abs(dz) - 0.05, 3)
        hd = round(rng.uniform(0.0, 360.0), 1)
        row = {"kind": kind, "lat_m": lat, "dz_m": dz, "heading_deg": hd}
        for lay in LAYOUTS:
            hit_d, block_d = detect_case(lay, kind, lat, dz, hd, probe)
            row[f"{lay}_hit_m"] = None if hit_d is None else round(hit_d, 2)
            row[f"{lay}_block_m"] = None if block_d is None else round(block_d, 2)
            row[f"{lay}_pass"] = block_d is not None and block_d >= STOP_DIST
        rows.append(row)
    for lay in LAYOUTS:
        print(f"sweep {lay}: {sum(r[f'{lay}_pass'] for r in rows)}/{n} detected within the stop distance")
    remove_probe()
    return rows


def fmt_d(v):
    return "  -  " if v is None else f"{v:5.2f}"


# ======================================================================== avoid ==
SURVEY = [tuple(w) for w in FLIGHT_PATH]
DIAGONAL = [(-10.0, 10.0, ALT), (10.0, 8.6, ALT), (-6.0, -9.5, ALT), (9.5, -3.0, ALT), (-9.0, 2.0, ALT)]
# per course: (leg index, fraction along leg) where test obstacles go; varied lat / dz per kind
SLOTS = {"survey": [(0, 0.5), (2, 0.5), (4, 0.5), (7, 0.55)],
         "diagonal": [(0, 0.35), (1, 0.4), (2, 0.6), (3, 0.5)]}
AVOID_VARIANTS = {
    "thin_pole": [(0.0, 0.0), (0.12, 0.0), (-0.25, 0.0), (0.37, 0.0)],
    "pole": [(0.0, 0.0), (0.2, 0.0), (-0.4, 0.0), (0.1, 0.0)],
    "wire": [(0.0, 0.0), (0.0, 0.15), (0.0, -0.3), (0.0, 0.3)],
    "guy_wire": [(0.0, 0.0), (0.2, 0.0), (-0.3, 0.0), (0.1, 0.0)],
    "beam": [(0.0, 0.0), (0.0, 0.25), (0.0, -0.25), (0.0, 0.1)],
    "small_box": [(0.0, 0.0), (0.15, 0.1), (-0.2, -0.15), (0.3, 0.2)],
    "ball": [(0.0, 0.0), (0.2, 0.15), (-0.25, -0.2), (0.1, -0.1)],
    "sign_edge": [(0.0, 0.0), (0.2, 0.0), (-0.1, 0.0), (0.0, 0.0)],
    "overhang": [(0.0, 0.15), (0.0, 0.3), (0.0, 0.2), (0.0, 0.25)],
    "low_wall": [(0.0, -0.15), (0.0, -0.3), (0.0, -0.2), (0.0, -0.25)],
}


def leg_point(course, leg, frac):
    a, b = course[leg], course[leg + 1]
    return (a[0] + frac * (b[0] - a[0]), a[1] + frac * (b[1] - a[1])), math.atan2(b[1] - a[1], b[0] - a[0])


def fly_course(layout, course, placements, target_xy=None, extra=(), max_time=240.0, probe=None, log=None,
               trees=True, start_z=3.5):
    """Closed-loop mission in the default world. placements: [(kind, c, yaw, lat, dz)].
    extra: [(kind, c, yaw, lat, dz)] fixed scenery (present with or without the casualty).
    Returns a result dict."""
    DRONE_START_POS[:] = [course[0][0], course[0][1], start_z]
    saved = [list(o) for o in OBSTACLE_POSITIONS]
    if not trees:
        OBSTACLE_POSITIONS[:] = []
    floor, drone, trees = env_sim.build_world(gui=False)
    OBSTACLE_POSITIONS[:] = saved
    test_ids = []
    for kind, c, yaw, lat, dz in placements:
        test_ids += spawn_obstacle(kind, c, yaw, lat, dz)
    scenery = []
    for kind, c, yaw, lat, dz in extra:
        scenery += spawn_obstacle(kind, c, yaw, lat, dz)
    target = -1
    if target_xy is not None:
        target = p.loadURDF("humanoid/humanoid.urdf", [target_xy[0], target_xy[1], 0.875],
                            p.getQuaternionFromEuler(TARGET_ORIENTATION_EULER), globalScaling=TARGET_SCALE,
                            useFixedBase=True)
    if probe is not None:
        probe.target = target
        probe.target_hits, probe.target_ids = [], set()
    nav = Navigator(drone, floor, target, list(course), False, log or (lambda *_a: None), sensor=layout)
    mapped_target = [0]
    if probe is not None:
        orig_add = nav.map.add_hit

        def add_hit(pt):
            if id(pt) in probe.target_ids:
                mapped_target[0] += 1
            return orig_add(pt)
        nav.map.add_hit = add_hit
    quat = p.getQuaternionFromEuler([0, 0, 0])
    all_obs = trees + test_ids + scenery
    min_test, min_all, contacts, step = float("inf"), float("inf"), 0, 0
    min_target_h = float("inf")
    sight_h = None
    while nav.t < max_time and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
        nav.update(DT)
        p.resetBasePositionAndOrientation(drone, nav.pos, quat)
        if sight_h is None and nav.target_seen:
            sight_h = math.hypot(nav.pos[0] - target_xy[0], nav.pos[1] - target_xy[1])
        if step % 2 == 0:
            if test_ids:
                min_test = min(min_test, body_clearance(drone, test_ids))
            d = body_clearance(drone, all_obs)
            min_all = min(min_all, d)
            contacts += d <= 0.0
            if target_xy is not None:
                min_target_h = min(min_target_h, math.hypot(nav.pos[0] - target_xy[0], nav.pos[1] - target_xy[1]))
        step += 1
    reached = {int(m.rsplit(" ", 1)[1]) for _, k, m in nav.events if k == "WAYPT"}
    res = {"state": nav.state, "sim_s": round(nav.t, 1), "reached": len(reached),
           "waypoints": len(course) - 1, "contacts": contacts,
           "min_clear_test_m": None if min_test == float("inf") else round(min_test, 3),
           "min_clear_all_m": round(min_all, 3), "replans": nav.replans,
           "climbs": sum(1 for _, k, _m in nav.events if k == "CLIMB"),
           "stops": sum(1 for _, k, m in nav.events if k == "BLOCKED" and "dead ahead" in m),
           "events": [(round(t, 3), k, m) for t, k, m in nav.events if k != "TARGET"],
           "seen": frozenset(nav.map.seen), "target_seen": nav.target_seen, "target_hits": nav.target_hits,
           "target_estimate": nav.target_estimate, "mapped_target": mapped_target[0],
           "min_target_h_m": None if target_xy is None else round(min_target_h, 2),
           "sight_h_m": None if sight_h is None else round(sight_h, 2)}
    p.disconnect()
    return res


# Forced climbs: the drone flies down a 2.2 m wide alley between two 7 m walls that run to the map
# edge, closed at the far end by a 2.6 m wall. There is no way round, so it has to climb. Overhead, a
# line (or a roof) runs along the alley 0.8 m above cruise, so climbing 1 m (CLIMB_STEP) goes straight
# into it. Each scene is flown with the Navigator's overhead check (CLIMB_CHECK) off and on.
STRAIGHT = [(-4.0, 0.0, ALT), (8.0, 0.0, ALT)]
ALLEY = [("alley_wall", (-6.5, 1.1), 0.0, 0.0, 0.0), ("alley_wall", (-6.5, -1.1), 0.0, 0.0, 0.0),
         ("end_wall", (0.0, 0.0), 0.0, 0.0, 0.0)]
CLIMB_SCENES = {
    "alley, line overhead": ALLEY + [("wire_overhead", (-3.5, 0.0), 0.0, 0.0, 0.8)],
    "alley, roof overhead": ALLEY + [("slab_overhead", (-3.5, 0.0), 0.0, 0.0, 0.8)],
    "alley, open sky": ALLEY,
}


def run_avoid(quick):
    rows = []
    for check in (False, True):
        oa.CLIMB_CHECK = check
        for name, places in CLIMB_SCENES.items():
            row = {"course": f"alley, overhead check {'on' if check else 'off'}", "kind": name,
                   "instances": len(places)}
            for lay in LAYOUTS:
                r = fly_course(lay, STRAIGHT, places, trees=False, start_z=ALT)
                # boxed in under a line/roof the drone may legitimately stop and wait for the operator
                ok = r["contacts"] == 0 and (r["reached"] == r["waypoints"] or "open sky" not in name)
                row.update({f"{lay}_contacts": r["contacts"], f"{lay}_min_clear_m": r["min_clear_test_m"],
                            f"{lay}_reached": f"{r['reached']}/{r['waypoints']}", f"{lay}_state": r["state"],
                            f"{lay}_replans": r["replans"], f"{lay}_stops": r["stops"],
                            f"{lay}_climbs": r["climbs"], f"{lay}_sim_s": r["sim_s"], f"{lay}_pass": ok})
            rows.append(row)
            print(f"avoid {row['course']:<26} {name:<21} | " + " | ".join(
                f"{lay} contacts {row[f'{lay}_contacts']:3d} clear {row[f'{lay}_min_clear_m']} climbs "
                f"{row[f'{lay}_climbs']} {row[f'{lay}_state']} {'PASS' if row[f'{lay}_pass'] else 'FAIL'}"
                for lay in LAYOUTS))
    oa.CLIMB_CHECK = True
    kinds = list(AVOID_VARIANTS)[::2] if quick else list(AVOID_VARIANTS)
    for course_name, course in (("survey", SURVEY), ("diagonal", DIAGONAL)):
        for kind in kinds:
            places = []
            for (leg, frac), (lat, dz) in zip(SLOTS[course_name], AVOID_VARIANTS[kind]):
                c, yaw = leg_point(course, leg, frac)
                places.append((kind, c, yaw, lat, dz))
            row = {"course": course_name, "kind": kind, "instances": len(places)}
            for lay in LAYOUTS:
                r = fly_course(lay, course, places)
                ok = r["contacts"] == 0 and r["reached"] == r["waypoints"]
                row.update({f"{lay}_contacts": r["contacts"], f"{lay}_min_clear_m": r["min_clear_test_m"],
                            f"{lay}_reached": f"{r['reached']}/{r['waypoints']}", f"{lay}_state": r["state"],
                            f"{lay}_replans": r["replans"], f"{lay}_stops": r["stops"],
                            f"{lay}_climbs": r["climbs"], f"{lay}_sim_s": r["sim_s"], f"{lay}_pass": ok})
            rows.append(row)
            print(f"avoid {course_name:<8} {kind:<10} | " + " | ".join(
                f"{lay} contacts {row[f'{lay}_contacts']:3d} clear {row[f'{lay}_min_clear_m']} "
                f"reached {row[f'{lay}_reached']} {row[f'{lay}_state']} {'PASS' if row[f'{lay}_pass'] else 'FAIL'}"
                for lay in LAYOUTS))
    return rows


# ======================================================================= target ==
# (name, casualty (x, y), extra scenery). Positions are on / beside the default survey route.
TARGET_CASES = [
    ("original spot", (7.0, -5.5), []),
    ("open, 1.5 m off route", (2.5, -1.5), []),
    ("on the flight line", (7.5, 5.0), []),
    ("on a waypoint", (-5.0, 5.0), []),
    ("under canopy, tree (0, 1)", (0.35, 0.65), []),
    ("under canopy, tree (-4, -1.5)", (-4.0, -1.0), []),
    ("under canopy, tree (8, -2)", (8.45, -2.2), []),
    # 3 x 3 x 6 m building south of leg 4 -> 5 (north face at y = -1.25)
    ("beside building wall", (2.5, -0.95), [("building", (2.5, -1.25), -math.pi / 2, 0.0, 0.0)]),
    ("behind building (far side)", (2.5, -4.6), [("building", (2.5, -1.25), -math.pi / 2, 0.0, 0.0)]),
    ("beside a pole on the route", (-5.3, 7.5), [("pole", (-5.0, 7.5), -math.pi / 2, 0.0, 0.0)]),
    ("between two trees", (-2.0, -0.4), []),
    # 3 m wide x 4 m tall wall across leg 4 -> 5; casualty on the route 0.35 m in front of it
    ("on the route, in front of a wall", (2.0, 0.0), [("wall", (2.5, 0.0), 0.0, 0.0, 0.0)]),
    ("off route, tree (-8, -8)", (-7.6, -8.3), []),
]


def hidden_cells(seen_with, seen_without):
    """Obstacle-map cells seen WITHOUT the casualty that have no counterpart (same or neighbouring
    cell / height bucket) when it is there, i.e. obstacle surface the casualty hid from the sensor.
    The neighbour tolerance absorbs float round-off when a surface lies exactly on a cell edge."""
    n = 0
    for (ix, iy), zb in seen_without - seen_with:
        if not any(((ix + a, iy + b), zb + c) in seen_with
                   for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1)):
            n += 1
    return n


def run_target(quick):
    probe = install_probe()
    rows = []
    cases = TARGET_CASES[::3] if quick else TARGET_CASES
    for name, xy, extra in cases:
        row = {"case": name, "x": xy[0], "y": xy[1]}
        for lay in LAYOUTS:
            with_t = fly_course(lay, SURVEY, [], target_xy=xy, extra=extra, probe=probe)
            without = fly_course(lay, SURVEY, [], target_xy=None, extra=extra, probe=probe)
            est = with_t["target_estimate"]
            err = None if est is None else round(math.hypot(est[0] - xy[0], est[1] - xy[1]), 2)
            same_nav = with_t["events"] == without["events"]
            hidden = hidden_cells(with_t["seen"], without["seen"])
            never_obstacle = with_t["mapped_target"] == 0 and same_nav
            row.update({f"{lay}_seen": with_t["target_seen"], f"{lay}_hits": with_t["target_hits"],
                        f"{lay}_sight_m": with_t["sight_h_m"],
                        f"{lay}_err_m": err, f"{lay}_mapped": with_t["mapped_target"],
                        f"{lay}_same_nav": same_nav, f"{lay}_hidden": hidden,
                        f"{lay}_min_h_m": with_t["min_target_h_m"],
                        f"{lay}_contacts": with_t["contacts"], f"{lay}_state": with_t["state"],
                        f"{lay}_never_obstacle": never_obstacle})
        rows.append(row)
        print(f"target {name:<32} | " + " | ".join(
            f"{lay} seen {row[f'{lay}_seen']!s:<5} at {row[f'{lay}_sight_m']} m hits {row[f'{lay}_hits']:4d} err {row[f'{lay}_err_m']} "
            f"mapped {row[f'{lay}_mapped']} same nav {row[f'{lay}_same_nav']} hidden {row[f'{lay}_hidden']}"
            for lay in LAYOUTS))
    remove_probe()
    return rows


# ======================================================================= timing ==
def replay_poses(course, spacing):
    """Poses (x, y, z, heading) every `spacing` m along a course, as the drone scans them."""
    out = []
    for a, b in zip(course[:-1], course[1:]):
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(L / spacing))
        for i in range(n):
            t = i / n
            out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), ALT, (b[0], b[1])))
    return out


def time_scans(poses, placements):
    """Per-layout sense() times (s) over the same poses. Both layouts scan the SAME world and each pose
    is scanned by both back to back (order alternating), so CPU clock / background-load drift hits
    both equally. Each layout keeps its own fresh obstacle map."""
    DRONE_START_POS[:] = [SURVEY[0][0], SURVEY[0][1], 3.5]
    floor, drone, _trees = env_sim.build_world(gui=False)
    for kind, c, yaw, lat, dz in placements:
        spawn_obstacle(kind, c, yaw, lat, dz)
    navs = {lay: Navigator(drone, floor, -1, SURVEY, False, lambda *_a: None, sensor=lay) for lay in LAYOUTS}
    ts = {lay: [] for lay in LAYOUTS}
    pc = time.perf_counter
    for k, (x, y, z, goal) in enumerate(poses):
        p.resetBasePositionAndOrientation(drone, [x, y, z], [0, 0, 0, 1])
        for lay in (LAYOUTS if k % 2 == 0 else LAYOUTS[::-1]):
            nav = navs[lay]
            nav.pos = [x, y, z]
            nav.route = [(goal[0], goal[1], None)]
            t0 = pc()
            nav.sense()
            ts[lay].append(pc() - t0)
    p.disconnect()
    return ts


def mission_wall(layout, course):
    """Wall time of Navigator.update() per simulated second over a closed-loop mission."""
    DRONE_START_POS[:] = [course[0][0], course[0][1], 3.5]
    floor, drone, _trees = env_sim.build_world(gui=False)
    nav = Navigator(drone, floor, -1, list(course), False, lambda *_a: None, sensor=layout)
    quat = p.getQuaternionFromEuler([0, 0, 0])
    spent = 0.0
    while nav.t < 240.0 and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
        t0 = time.perf_counter()
        nav.update(DT)
        spent += time.perf_counter() - t0
        p.resetBasePositionAndOrientation(drone, nav.pos, quat)
    p.disconnect()
    return spent / max(nav.t, 1e-9), nav.t


class world_variant:
    """Context manager: set up env_sim's world lists for one timing world, restore them afterwards.
    Returns the test obstacles to place on top of it."""

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.saved = ([list(o) for o in OBSTACLE_POSITIONS], list(env_sim.TREE_KINDS), list(env_sim.WALLS),
                      env_sim.OSM_SCENE, list(env_sim.TARGET_POSITION))
        if self.name == "test obstacles":
            return [(k, *leg_point(SURVEY, leg, frac), lat, dz)
                    for k, ((leg, frac), (lat, dz)) in zip(
                        ("thin_pole", "wire", "small_box", "sign_edge"),
                        zip(SLOTS["survey"], ((0.1, 0), (0, 0.15), (0, 0), (0, 0))))]
        if self.name.startswith("generated"):        # route_safety_runs world: 19 obstacles incl. 4 tall walls
            import route_safety_runs
            route_safety_runs.make_world(11)
        if self.name.startswith("OSM"):              # bundled map.osm export, 100 m squeezed into the map
            env_sim.use_osm(44.6488, -123.3123, area_m=100.0, osm_file=os.path.join(HERE, "map.osm"),
                            keep_clear=[tuple(w[:2]) for w in FLIGHT_PATH])
        return []

    def __exit__(self, *exc):
        obs, kinds, walls, scene, target = self.saved
        OBSTACLE_POSITIONS[:], env_sim.TREE_KINDS[:], env_sim.WALLS[:] = obs, kinds, walls
        env_sim.OSM_SCENE = scene
        env_sim.TARGET_POSITION[:] = target
        return False


TIMING_WORLDS = ("default world", "test obstacles", "generated, seed 11", "OSM forest (map.osm)")


def run_timing(quick, reps):
    rows = []
    reps = 2 if quick else reps
    for wname in TIMING_WORLDS:
        for cname, course in (("survey", SURVEY), ("diagonal", DIAGONAL)):
            poses = replay_poses(course, V_MAX * DT * SCAN_EVERY)
            per = {lay: [] for lay in LAYOUTS}
            alls = {lay: [] for lay in LAYOUTS}
            for r in range(reps):
                with world_variant(wname) as placements:
                    ts = time_scans(poses, placements)
                for lay in LAYOUTS:
                    per[lay].append(statistics.mean(ts[lay]))
                    alls[lay] += ts[lay]
            row = {"world": wname, "course": cname, "scans": len(poses), "reps": reps}
            for lay in LAYOUTS:
                srt = sorted(alls[lay])
                row[f"{lay}_mean_us"] = round(statistics.median(per[lay]) * 1e6, 1)
                row[f"{lay}_median_us"] = round(statistics.median(alls[lay]) * 1e6, 1)
                row[f"{lay}_p95_us"] = round(srt[int(0.95 * (len(srt) - 1))] * 1e6, 1)
            row["increase_pct"] = round(100.0 * (row["enhanced_mean_us"] / row["baseline_mean_us"] - 1.0), 1)
            row["pass"] = row["increase_pct"] < TIME_LIMIT_PCT
            rows.append(row)
            print(f"timing {wname:<22} {cname:<8} | baseline {row['baseline_mean_us']:6.1f} us | "
                  f"enhanced {row['enhanced_mean_us']:6.1f} us | {row['increase_pct']:+.1f} % "
                  f"{'PASS' if row['pass'] else 'FAIL'}")
    # where a scan's time goes (default world, survey poses): ray generation, the cast itself, the rest
    # (classifying hits, updating the obstacle map). Explains how 15 % more rays cost no extra time.
    poses = replay_poses(SURVEY, V_MAX * DT * SCAN_EVERY)
    DRONE_START_POS[:] = [SURVEY[0][0], SURVEY[0][1], 3.5]
    floor, drone, _trees = env_sim.build_world(gui=False)
    pc = time.perf_counter
    for lay in LAYOUTS:
        best = None
        for _ in range(reps):
            layout = oa.range_sensor.make_layout(lay)
            nav = Navigator(drone, floor, -1, SURVEY, False, lambda *_a: None, sensor=lay)
            t_build = t_cast = t_sense = 0.0
            n_rays = 0
            for x, y, z, goal in poses:
                h = math.hypot(goal[0] - x, goal[1] - y)
                h = ((goal[0] - x) / h, (goal[1] - y) / h)
                t0 = pc()
                fr, to, _f = layout.rays([x, y, z], h)
                t1 = pc()
                p.rayTestBatch(fr, to)
                t2 = pc()
                nav.pos, nav.route = [x, y, z], [(goal[0], goal[1], None)]
                nav.sense()
                t3 = pc()
                t_build += t1 - t0
                t_cast += t2 - t1
                t_sense += t3 - t2
                n_rays += len(fr)
            n = len(poses)
            cur = (t_sense / n, t_build / n, t_cast / n, n_rays / n)
            best = cur if best is None or cur[0] < best[0] else best
        sense_s, build_s, cast_s, rays = best
        rows.append({"world": "breakdown (default world)", "course": "survey", "scans": len(poses), "reps": reps,
                     f"{lay}_mean_us": round(sense_s * 1e6, 1), "rays_per_scan": round(rays, 1),
                     "build_us": round(build_s * 1e6, 1), "cast_us": round(cast_s * 1e6, 1),
                     "rest_us": round((sense_s - build_s - cast_s) * 1e6, 1), "layout": lay,
                     "increase_pct": None, "pass": None})
        print(f"breakdown {lay:<8}: {rays:.0f} rays/scan | build {build_s * 1e6:5.1f} us | cast "
              f"{cast_s * 1e6:5.1f} us | rest {(sense_s - build_s - cast_s) * 1e6:5.1f} us | total {sense_s * 1e6:5.1f} us")
    p.disconnect()

    # whole-navigator cost per simulated second (sensing + planning + motion)
    for cname, course in (("survey", SURVEY), ("diagonal", DIAGONAL)):
        per = {lay: [] for lay in LAYOUTS}
        for r in range(max(1, reps // 2)):
            for lay in (LAYOUTS if r % 2 == 0 else LAYOUTS[::-1]):
                per[lay].append(mission_wall(lay, course)[0])
        row = {"world": "default world (closed loop)", "course": cname, "scans": "-", "reps": len(per["baseline"])}
        for lay in LAYOUTS:
            row[f"{lay}_mean_us"] = round(statistics.median(per[lay]) * 1e6, 1)   # us of wall time per sim s
            row[f"{lay}_median_us"] = row[f"{lay}_p95_us"] = None
        row["increase_pct"] = round(100.0 * (row["enhanced_mean_us"] / row["baseline_mean_us"] - 1.0), 1)
        row["pass"] = None
        rows.append(row)
        print(f"navigator wall time / sim s, {cname}: baseline {row['baseline_mean_us'] / 1000:.1f} ms | "
              f"enhanced {row['enhanced_mean_us'] / 1000:.1f} ms | {row['increase_pct']:+.1f} %")
    return rows


# ======================================================================= report ==
def write_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols and not k.startswith("_")]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    """Rows of a CSV written by write_csv, with numbers / booleans / blanks converted back."""
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            d = {}
            for k, v in r.items():
                if v in ("True", "False"):
                    d[k] = v == "True"
                elif v == "":
                    d[k] = None
                else:
                    try:
                        d[k] = int(v)
                    except ValueError:
                        try:
                            d[k] = float(v)
                        except ValueError:
                            d[k] = v
            out.append(d)
    return out


def yn(v):
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def summary_md(res, out):
    L = ["# Range sensor verification", "",
         f"Layouts: `{oa.range_sensor.BaselineLayout().describe()}` vs "
         f"`{oa.range_sensor.EnhancedLayout().describe()}`.",
         f"Stop distance {STOP_DIST} m, scan every {SCAN_EVERY} physics steps "
         f"({1 / (DT * SCAN_EVERY):.0f} Hz), cruise {V_MAX} m/s at {ALT} m.", ""]
    verdict = {}
    if "detect" in res:
        rows = res["detect"]
        L += ["## AC-01 detect suite (open loop)", "",
              "Distance from the drone centre to the obstacle surface when the obstacle map first marks "
              "the path ahead as blocked (`block`) and at the first ray hit (`hit`). PASS = blocked at "
              f">= {STOP_DIST} m. `-` = never detected before contact.", "",
              "| Obstacle | lat (m) | dz (m) | heading | baseline hit | baseline block | enhanced hit | enhanced block | baseline | enhanced |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['kind']} | {r['lat_m']:+.2f} | {r['dz_m']:+.2f} | {r['heading_deg']:g} | "
                     f"{fmt_d(r['baseline_hit_m'])} | {fmt_d(r['baseline_block_m'])} | {fmt_d(r['enhanced_hit_m'])} | "
                     f"{fmt_d(r['enhanced_block_m'])} | {yn(r['baseline_pass'])} | {yn(r['enhanced_pass'])} |")
        for lay in LAYOUTS:
            n = sum(r[f"{lay}_pass"] for r in rows)
            L.append(f"\n**{lay}: {n}/{len(rows)} cases detected within the stop distance.**")
        kinds = sorted({r["kind"] for r in rows}, key=lambda k: [r["kind"] for r in rows].index(k))
        L += ["", "Per obstacle type (worst case = shortest blocking distance over its cases):", "",
              "| Obstacle | cases | baseline pass | baseline worst (m) | enhanced pass | enhanced worst (m) |",
              "|---|---|---|---|---|---|"]
        for k in kinds:
            rr = [r for r in rows if r["kind"] == k]
            cells = []
            for lay in LAYOUTS:
                ds = [r[f"{lay}_block_m"] for r in rr]
                worst = "-" if any(d is None for d in ds) else f"{min(ds):.2f}"
                cells += [f"{sum(r[f'{lay}_pass'] for r in rr)}/{len(rr)}", worst]
            L.append(f"| {k} | {len(rr)} | " + " | ".join(cells) + " |")
        verdict["AC-01 detect"] = all(r["enhanced_pass"] for r in rows)
    if "sweep" in res:
        rows = res["sweep"]
        L += ["", f"## AC-01 randomised sweep ({len(rows)} placements, same draws for both layouts)", "",
              "Random obstacle type, sideways / vertical offset into the drone's swept tube and direction of "
              "travel (full table in `sweep.csv`).", "",
              "| Obstacle | cases | baseline pass | baseline worst (m) | enhanced pass | enhanced worst (m) |",
              "|---|---|---|---|---|---|"]
        for k in [k for k, _a, _b in SWEEP_KINDS]:
            rr = [r for r in rows if r["kind"] == k]
            cells = []
            for lay in LAYOUTS:
                ds = [r[f"{lay}_block_m"] for r in rr]
                worst = "-" if any(d is None for d in ds) else f"{min(ds):.2f}"
                cells += [f"{sum(r[f'{lay}_pass'] for r in rr)}/{len(rr)}", worst]
            L.append(f"| {k} | {len(rr)} | " + " | ".join(cells) + " |")
        for lay in LAYOUTS:
            L.append(f"\n**{lay}: {sum(r[f'{lay}_pass'] for r in rows)}/{len(rows)} placements detected within "
                     f"the stop distance.**")
        verdict["AC-01 sweep"] = all(r["enhanced_pass"] for r in rows)
    if "avoid" in res:
        allrows = res["avoid"]
        rows = [r for r in allrows if not r["course"].startswith("alley")]
        L += ["", "## AC-01 avoid suite (closed loop)", "",
              "Full Navigator on two courses in the default world (8 trees) with 4 test obstacles of one type on "
              "the legs (varied lateral / vertical offsets). PASS = no contact and every waypoint reached. "
              "`min clear` = closest drone-body-to-test-obstacle distance (m); under 0.35 m (propeller reach, "
              "UNSAFE_DIST) is a near miss.", "",
              "| Course | Obstacle | baseline contacts | baseline min clear (m) | baseline reached | enhanced contacts | enhanced min clear (m) | enhanced reached | baseline | enhanced |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['course']} | {r['kind']} | {r['baseline_contacts']} | {r['baseline_min_clear_m']} | "
                     f"{r['baseline_reached']} | {r['enhanced_contacts']} | {r['enhanced_min_clear_m']} | "
                     f"{r['enhanced_reached']} | {yn(r['baseline_pass'])} | {yn(r['enhanced_pass'])} |")
        for lay in LAYOUTS:
            clear = [r[f"{lay}_min_clear_m"] for r in rows if r[f"{lay}_min_clear_m"] is not None]
            hit = sum(1 for r in rows if r[f"{lay}_contacts"])
            unsafe = sum(1 for r in rows if r[f"{lay}_min_clear_m"] is not None and r[f"{lay}_min_clear_m"] < 0.35)
            L.append(f"\n**{lay}: {sum(r[f'{lay}_pass'] for r in rows)}/{len(rows)} missions completed without "
                     f"contact; {hit} with contact; {unsafe} closer than 0.35 m; closest approach {min(clear):.2f} m.**")
        climb = [r for r in allrows if r["course"].startswith("alley")]
        if climb:
            L += ["", "Forced climbs: boxed in at the end of an alley (7 m side walls to the map edge, 2.6 m end "
                  "wall) with a line or a roof 0.8 m overhead. Flown with the Navigator's new overhead check "
                  "(`CLIMB_CHECK`) off (old behaviour) and on. PASS = no contact (boxed in under a line, stopping "
                  "and waiting for the operator is the correct outcome).", "",
                  "| Overhead check | Scene | baseline contacts | baseline min clear (m) | baseline climbs | baseline end state | enhanced contacts | enhanced min clear (m) | enhanced climbs | enhanced end state |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
            for r in climb:
                L.append(f"| {r['course'].split('check ')[1]} | {r['kind']} | {r['baseline_contacts']} | "
                         f"{r['baseline_min_clear_m']} | {r['baseline_climbs']} | {r['baseline_state']} | "
                         f"{r['enhanced_contacts']} | {r['enhanced_min_clear_m']} | {r['enhanced_climbs']} | "
                         f"{r['enhanced_state']} |")
        delivered = rows + [r for r in climb if r["course"].endswith("on")]
        verdict["AC-01 avoid"] = all(r["enhanced_pass"] for r in delivered)
    if "target" in res:
        rows = res["target"]
        L += ["", "## AC-02 target suite", "",
              "Survey route flown twice per case: with the casualty and without it. `mapped` = casualty ray "
              "hits that reached the obstacle map (must be 0). `same nav` = identical navigation events "
              "(replans, stops, climbs, waypoints) with and without the casualty. `hidden` = obstacle-map "
              "cells the casualty hid from the sensor (seen without it, missing with it). `err` = horizontal "
              "error of the location estimate (m). `closest` = closest horizontal pass of the drone (m).", "",
              "| Case | closest (m) | baseline seen | baseline err | baseline mapped | baseline same nav | baseline hidden | enhanced seen | enhanced err | enhanced mapped | enhanced same nav | enhanced hidden |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['case']} | {r['enhanced_min_h_m']} | {r['baseline_seen']} | {r['baseline_err_m']} | "
                     f"{r['baseline_mapped']} | {r['baseline_same_nav']} | {r['baseline_hidden']} | "
                     f"{r['enhanced_seen']} | {r['enhanced_err_m']} | {r['enhanced_mapped']} | "
                     f"{r['enhanced_same_nav']} | {r['enhanced_hidden']} |")
        for lay in LAYOUTS:
            ok = sum(r[f"{lay}_never_obstacle"] for r in rows)
            seen = sum(r[f"{lay}_seen"] for r in rows)
            L.append(f"\n**{lay}: never treated as an obstacle in {ok}/{len(rows)} cases; sighted in "
                     f"{seen}/{len(rows)}.**")
        visible = [r for r in rows if "off route" not in r["case"]]
        verdict["AC-02"] = (all(r["enhanced_never_obstacle"] for r in rows)
                            and all(r["enhanced_seen"] and r["enhanced_err_m"] is not None
                                    and r["enhanced_err_m"] <= 0.5 for r in visible))
    if "timing" in res:
        rows = res["timing"]
        L += ["", "## AC-03 timing suite", "",
              "Scan rows: identical poses (one per scan along the course) replayed for both layouts with a fresh "
              "obstacle map; time of one full `sense()` call (ray generation + cast + classification + map "
              "update), median of the per-repetition means, layouts interleaved. Closed-loop rows: wall time "
              "of `Navigator.update()` per simulated second (sensing + planning + motion).", "",
              "| World | Course | scans | baseline (us) | enhanced (us) | increase | baseline p95 (us) | enhanced p95 (us) | < 20 % |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r.get("layout"):
                continue
            L.append(f"| {r['world']} | {r['course']} | {r['scans']} | {r['baseline_mean_us']} | "
                     f"{r['enhanced_mean_us']} | {r['increase_pct']:+.1f} % | {r['baseline_p95_us'] or '-'} | "
                     f"{r['enhanced_p95_us'] or '-'} | {yn(r['pass'])} |")
        bd = [r for r in rows if r.get("layout")]
        if bd:
            L += ["", "Where the time of one scan goes (default world, survey poses, best repetition):", "",
                  "| Layout | rays / scan | ray generation (us) | cast (us) | hit handling + map (us) | total (us) |",
                  "|---|---|---|---|---|---|"]
            for r in bd:
                L.append(f"| {r['layout']} | {r['rays_per_scan']:.0f} | {r['build_us']} | {r['cast_us']} | "
                         f"{r['rest_us']} | {r[r['layout'] + '_mean_us']} |")
        verdict["AC-03"] = all(r["pass"] for r in rows if r["pass"] is not None)
    L += ["", "## Verdict (enhanced layout)", ""] + [f"- {k}: {yn(v)}" for k, v in verdict.items()]
    text = "\n".join(L) + "\n"
    with open(os.path.join(out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text)
    return text, verdict


def charts(res, out):
    """detect_ranges.png (worst-case detection distance per obstacle type) and scan_time.png (AC-03)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    colors = {"enhanced": "#2a78d6", "baseline": "#eb6834"}   # validated categorical pair (CVD dE 24.7)
    ink, muted, grid = "#2b2b29", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink,
                         "xtick.color": muted, "ytick.color": muted, "text.color": ink})

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.yaxis.grid(True, color=grid, lw=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.0, 1.02), ncol=2)

    def kinds_of(rows):
        out_k = []
        for r in rows:
            if r["kind"] not in out_k:
                out_k.append(r["kind"])
        return out_k

    rows = (res.get("detect") or []) + (res.get("sweep") or [])
    if rows:
        kinds = kinds_of(rows)
        fig, ax = plt.subplots(figsize=(11, 4.8))
        w = 0.38
        for j, lay in enumerate(("baseline", "enhanced")):
            worst = []
            for k in kinds:
                ds = [r[f"{lay}_block_m"] for r in rows if r["kind"] == k]
                worst.append(0.0 if any(d is None for d in ds) else min(ds))
            ax.bar([i + (j - 0.5) * (w + 0.02) for i in range(len(kinds))], worst, w, label=lay,
                   color=colors[lay], edgecolor="white", linewidth=1.0)
        ax.axhline(STOP_DIST, color=ink, lw=1.0, ls="--")
        ax.text(-0.6, STOP_DIST + 0.06, f"stop distance {STOP_DIST} m", color=ink, fontsize=9, ha="left")
        ax.set_xticks(range(len(kinds)))
        ax.set_xticklabels([k.replace("_", " ") for k in kinds], rotation=35, ha="right")
        ax.set_xlim(-0.7, len(kinds) - 0.3)
        ax.set_ylabel("worst-case detection distance (m)")
        ax.set_ylim(0, 5.3)                               # head-room for the legend
        n = len(rows)
        ax.set_title(f"Distance from the obstacle when the path is first marked blocked\n"
                     f"worst case per obstacle type over {n} placements (0 = not detected before contact)",
                     fontsize=11, loc="left")
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "detect_ranges.png"), dpi=140)
        plt.close(fig)
    if "timing" in res:
        rows = [r for r in res["timing"] if r["pass"] is not None and not r.get("layout")]
        short = {"default world": "default", "test obstacles": "test obstacles", "generated, seed 11": "generated",
                 "OSM forest (map.osm)": "OSM forest"}
        fig, ax = plt.subplots(figsize=(11, 4.4))
        w = 0.36
        for j, lay in enumerate(("baseline", "enhanced")):
            ax.bar([i + (j - 0.5) * (w + 0.02) for i in range(len(rows))], [r[f"{lay}_mean_us"] for r in rows], w,
                   label=lay, color=colors[lay], edgecolor="white", linewidth=1.0)
        for i, r in enumerate(rows):
            lim = 1.2 * r["baseline_mean_us"]
            ax.plot([i - 0.42, i + 0.42], [lim, lim], color=ink, lw=1.0, ls="--")
            ax.text(i + 0.5 * (w + 0.02), r["enhanced_mean_us"] + 4, f"{r['increase_pct']:+.0f}%", ha="center",
                    va="bottom", fontsize=9, color=ink)
        ax.text(len(rows) - 1 + 0.42, 1.2 * rows[-1]["baseline_mean_us"] + 6, "+20 % limit", ha="right",
                va="bottom", fontsize=9, color=ink)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{short.get(r['world'], r['world'])}\n{r['course']}" for r in rows])
        ax.set_ylabel("time per scan (\u00b5s)")
        ax.set_ylim(0, max(1.2 * r["baseline_mean_us"] for r in rows) * 1.12)
        ax.set_title("Range-sensor scan time on identical poses (AC-03: enhanced must stay under the +20 % line)",
                     fontsize=11, loc="left")
        style(ax)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "scan_time.png"), dpi=140)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", choices=("all", "detect", "sweep", "avoid", "target", "timing"), default="all")
    ap.add_argument("--sweep", type=int, default=300, help="random placements in the sweep suite")
    ap.add_argument("--seed", type=int, default=1, help="sweep seed")
    ap.add_argument("--quick", action="store_true", help="fewer cases and repetitions")
    ap.add_argument("--reps", type=int, default=7, help="timing repetitions per layout")
    ap.add_argument("--out", default="range_sensor_results")
    ap.add_argument("--report-only", action="store_true",
                    help="rebuild summary.md and the charts from the CSV files already in --out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.report_only:
        res = {s: read_csv(os.path.join(args.out, f"{s}.csv")) for s in ("detect", "sweep", "avoid", "target", "timing")
               if os.path.exists(os.path.join(args.out, f"{s}.csv"))}
        text, _ = summary_md(res, args.out)
        charts(res, args.out)
        print(text)
        return
    suites = ("detect", "sweep", "avoid", "target", "timing") if args.suite == "all" else (args.suite,)
    res = {}
    for s in suites:
        t0 = time.perf_counter()
        if s == "detect":
            res[s] = run_detect(args.quick)
        elif s == "sweep":
            res[s] = run_sweep(60 if args.quick else args.sweep, args.seed)
        elif s == "avoid":
            res[s] = run_avoid(args.quick)
        elif s == "target":
            res[s] = run_target(args.quick)
        else:
            res[s] = run_timing(args.quick, args.reps)
        write_csv(os.path.join(args.out, f"{s}.csv"), res[s])
        print(f"--- {s} suite: {time.perf_counter() - t0:.0f} s wall\n")
    text, _ = summary_md(res, args.out)
    charts(res, args.out)
    print(text)
    print(f"Outputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
