"""
PBI-12: range sensor and obstacle-avoidance test scenarios, baseline vs enhanced.

Each scenario builds a small world in PyBullet (DIRECT, no window), flies the Navigator
(object_avoidance.py) along a straight 20 m leg at cruise altitude and measures with ground truth
(PyBullet closest points between the drone body and each object):

  detection   7 obstacle shapes x 3 positions: at what distance from the drone's body the sensor
              first hits the obstacle (AC-01: must be more than STOP_DIST = 0.9 m), for the
              baseline sensor (72 rays x 3 pitches, before PBI-12) and the enhanced one.
  target      the casualty in the open, under a tree, beside a wall: it must never change the
              obstacle map or the route (AC-02).
  timing      time per scan, baseline vs enhanced (AC-03: at most +20 %).
  avoidance   single tree / long row of trees / tall building / low wall / unmapped wall: climb
              over or go round, compared with detour-only flying (AC-04).

Run:  python range_sensor_runs.py          (writes tests/RANGE_SENSOR_TEST_REPORT.md)
The same scenarios are used by tests/test_range_sensor.py.
"""
import contextlib
import math
import os
import statistics
import sys
import time

import pybullet as p

import geo_obstacles
import object_avoidance as oa
import sim_config
from geo_obstacles import Obstacle

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_SIZE = 26.0
CRUISE = 2.5
START, GOAL = (-10.0, 0.0), (10.0, 0.0)
LEG = [(START[0], START[1], CRUISE), (GOAL[0], GOAL[1], CRUISE)]
MAX_T = 200.0

# ------------------------------------------------------------------ test objects
# (name, maker). Unmapped shapes (not in the obstacle table) have an unknown top, so the drone
# must go round them; mapped ones (tree, building) carry their map height.
SHAPES = [
    ("thin pole (10 cm)", "pole", dict(r=0.05, h=5.0)),
    ("lamp post (20 cm)", "pole", dict(r=0.10, h=5.0)),
    ("fence panel (5 cm thick)", "box", dict(half=(0.025, 1.5, 1.5), z=1.5)),
    ("small box (30 cm)", "box", dict(half=(0.15, 0.15, 0.15), z=2.5)),
    ("branch (40 cm ball)", "ball", dict(r=0.20, z=2.6)),
    ("tree (mapped)", "tree", {}),
    ("building 3x3 m, 8 m (mapped)", "building", dict(w=3.0, d=3.0, h=8.0)),
]
# Where along the leg (x) and how far to the side of it (y) each object is put.
POSITIONS = [(0.0, 0.0), (-3.0, 0.3), (4.0, -0.45)]


@contextlib.contextmanager
def settings(**values):
    """Temporarily change object_avoidance settings (SENSOR_MODE, CLIMB_OVER, ...)."""
    old = {k: getattr(oa, k) for k in values}
    try:
        for k, v in values.items():
            setattr(oa, k, v)
        yield
    finally:
        for k, v in old.items():
            setattr(oa, k, v)


def new_world():
    """Empty world: floor + drone at the start of the leg. Returns (floor, drone)."""
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    sim_config.set_map_size(MAP_SIZE)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor = p.loadURDF("plane.urdf")
    drone = p.loadURDF(os.path.join(HERE, "custom_drone.urdf"), [START[0], START[1], CRUISE],
                       p.getQuaternionFromEuler([0, 0, 0]), globalScaling=oa.DRONE_SCALE)
    return floor, drone


def add_shape(kind, x, y, params):
    """Put one test object in the world. Returns (body id, Obstacle or None if unmapped)."""
    if kind == "pole":
        r, h = params["r"], params["h"]
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=r, height=h)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=r, length=h, rgbaColor=[0.4, 0.4, 0.4, 1])
        return p.createMultiBody(0, col, vis, [x, y, h / 2]), None
    if kind == "box":
        half, z = list(params["half"]), params["z"]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.5, 0.4, 0.3, 1])
        return p.createMultiBody(0, col, vis, [x, y, z]), None
    if kind == "ball":
        col = p.createCollisionShape(p.GEOM_SPHERE, radius=params["r"])
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=params["r"], rgbaColor=[0.2, 0.5, 0.2, 1])
        return p.createMultiBody(0, col, vis, [x, y, params["z"]]), None
    if kind == "tree":
        obs, _ = geo_obstacles.obstacles_from_world(tree_positions=[(x, y, geo_obstacles.TREE_BASE_Z)])
    elif kind == "building":
        w, d, h = params["w"], params["d"], params["h"]
        x0, y0, x1, y1 = x - w / 2, y - d / 2, x + w / 2, y + d / 2
        obs = [Obstacle("building", params.get("name", f"building {w:g}x{d:g} m"), 0.0, 0.0, 0.0,
                        height=h, radius=max(w, d) / 2, cx=x, cy=y,
                        footprint=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)], source="test")]
    else:
        raise ValueError(kind)
    ids, table, problems = geo_obstacles.spawn(p, obs)
    if problems:
        raise RuntimeError(problems)
    return ids[0], table[ids[0]]


def add_target(x, y, lying=False):
    """The casualty: a standing (cylinder, 1.7 m) or lying (box) person on the ground."""
    if lying:
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.85, 0.25, 0.15])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.85, 0.25, 0.15], rgbaColor=[0.9, 0.3, 0.1, 1])
        return p.createMultiBody(0, col, vis, [x, y, 0.15])
    col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.25, height=1.7)
    vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.25, length=1.7, rgbaColor=[0.9, 0.3, 0.1, 1])
    return p.createMultiBody(0, col, vis, [x, y, 0.85])


def surface_distance(drone, uid, reach=10.0):
    """Ground truth: distance between the drone's collision body and the object's surface."""
    pts = p.getClosestPoints(drone, uid, reach)
    return min((c[8] for c in pts), default=reach)


def fly(nav, drone, watch=(), max_t=MAX_T):
    """Fly until the Navigator finishes. watch: body ids to measure. Returns a dict with the
    track and, per watched body, the ground-truth distance at first detection and the minimum."""
    quat = p.getQuaternionFromEuler([0, 0, 0])
    first = {u: None for u in watch}
    closest = {u: float("inf") for u in watch}
    track = []
    p.resetBasePositionAndOrientation(drone, nav.pos, quat)
    while nav.t < max_t and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
        nav.update(oa.DT)
        p.resetBasePositionAndOrientation(drone, nav.pos, quat)
        track.append(tuple(nav.pos))
        for u in watch:
            d = surface_distance(drone, u)
            closest[u] = min(closest[u], d)
            if first[u] is None and u in nav.detected:
                first[u] = d
    length = sum(math.dist(track[i][:2], track[i - 1][:2]) for i in range(1, len(track)))
    return {"track": track, "first": first, "closest": closest, "length": length,
            "max_alt": max((q[2] for q in track), default=CRUISE),
            "end_alt": track[-1][2] if track else CRUISE}


def navigator(floor, drone, table, target=-1, waypoints=LEG):
    nav = oa.Navigator(drone, floor, target, waypoints, False, lambda *_a: None, obstacle_info=table)
    nav.pos = [waypoints[0][0], waypoints[0][1], waypoints[0][2]]
    return nav


# ------------------------------------------------------------------ AC-01 detection
def detection_case(shape, pos, mode):
    """Fly the leg past one object. Returns detection distance, closest approach, outcome."""
    name, kind, params = shape
    with settings(SENSOR_MODE=mode):
        floor, drone = new_world()
        uid, ob = add_shape(kind, pos[0], pos[1], params)
        table = {uid: ob} if ob is not None else {}
        nav = navigator(floor, drone, table)
        r = fly(nav, drone, watch=[uid])
    return {"shape": name, "pos": pos, "mode": mode, "detect_m": r["first"][uid],
            "closest_m": r["closest"][uid], "state": nav.state, "replans": nav.replans}


def detection_table(modes=("baseline", "enhanced")):
    return [detection_case(s, pos, m) for s in SHAPES for pos in POSITIONS for m in modes]


# ------------------------------------------------------------------ AC-02 target
TARGET_CASES = [
    ("in the open, on the route", dict(target=(0.0, 0.0)), []),
    ("lying in the open, on the route", dict(target=(0.0, 0.0), lying=True), []),
    ("under a tree canopy", dict(target=(0.35, 1.0)), [("tree", 0.0, 1.0, {})]),
    ("beside a building wall (0.3 m)", dict(target=(0.0, 1.05)),
     [("building", 0.0, 2.8, dict(w=3.0, d=3.0, h=8.0, name="wall building"))]),
    ("between two trees", dict(target=(3.0, 0.0)), [("tree", 3.0, 1.6, {}), ("tree", 3.0, -1.6, {})]),
]


def target_case(case, with_target):
    """Fly the leg with the scene's obstacles, with or without the casualty in it."""
    label, t, scene = case
    floor, drone = new_world()
    table = {}
    for kind, x, y, params in scene:
        uid, ob = add_shape(kind, x, y, params)
        if ob is not None:
            table[uid] = ob
    target = add_target(t["target"][0], t["target"][1], t.get("lying", False)) if with_target else -1
    nav = navigator(floor, drone, table, target=target)
    r = fly(nav, drone)
    events = [(round(tt, 3), k, m) for tt, k, m in nav.events if k != "TARGET"]
    return {"label": label, "target": target, "nav": nav, "events": events,
            "map": {c: frozenset(z) for c, z in nav.map.block.items()},
            "target_seen": nav.target_seen, "track": r["track"]}


# ------------------------------------------------------------------ AC-03 timing
def scan_timing(batches=30, scans=20):
    """Time per scan for both sensors, measured alternately in the same world (a few obstacles in
    range) so both see the same load. Each sensor's time is the FASTEST batch: other work on the
    computer only ever adds time, so the fastest batch is the best measure of the sensor itself
    and the ratio does not jump around between runs. Returns (baseline ms, enhanced ms, rays)."""
    import gc
    floor, drone = new_world()
    table = {}
    for kind, x, y, params in (("tree", -7.0, 1.2, {}), ("pole", -8.0, -1.0, dict(r=0.1, h=5.0)),
                               ("building", -6.0, -3.0, dict(w=2.0, d=2.0, h=8.0))):
        uid, ob = add_shape(kind, x, y, params)
        if ob is not None:
            table[uid] = ob
    nav = navigator(floor, drone, table)
    times = {"baseline": [], "enhanced": []}
    rays = {}
    for mode in ("baseline", "enhanced"):                 # warm-up (caches, first-call costs)
        nav.sensor_mode = mode
        for _ in range(scans):
            nav.sense()
    was_enabled = gc.isenabled()
    gc.disable()                                          # no garbage collection inside a batch
    try:
        for _ in range(batches):
            for mode in ("baseline", "enhanced"):
                nav.sensor_mode = mode
                rays[mode] = nav.ray_count()
                t0 = time.perf_counter()
                for _ in range(scans):
                    nav.sense()
                times[mode].append((time.perf_counter() - t0) / scans * 1000.0)
    finally:
        if was_enabled:
            gc.enable()
    return min(times["baseline"]), min(times["enhanced"]), rays


# ------------------------------------------------------------------ AC-04 avoidance
def hedge(n=19, spacing=1.5, x=0.0):
    """A row of mapped trees across the whole map, perpendicular to the leg."""
    y0 = -spacing * (n - 1) / 2
    return [("tree", x, y0 + i * spacing, {}) for i in range(n)]


AVOID_CASES = [
    ("single tree on the route", [("tree", 0.0, 0.0, {})]),
    ("row of 19 trees across the map", hedge()),
    ("tall building on the route (8 m)", [("building", 0.0, 0.0, dict(w=2.0, d=12.0, h=8.0, name="tall"))]),
    ("low wall across the map (4 m)", [("building", 0.0, 0.0, dict(w=2.0, d=22.0, h=4.0, name="low wall"))]),
    ("unmapped fence across the map", [("box", 0.0, 0.0, dict(half=(0.05, 11.0, 1.5), z=1.5))]),
]


def avoid_case(case, climb_over):
    label, scene = case
    with settings(CLIMB_OVER=climb_over):
        floor, drone = new_world()
        table, uids = {}, []
        for kind, x, y, params in scene:
            uid, ob = add_shape(kind, x, y, params)
            uids.append(uid)
            if ob is not None:
                table[uid] = ob
        nav = navigator(floor, drone, table)
        r = fly(nav, drone, watch=uids)
    return {"label": label, "climb_over": climb_over, "state": nav.state, "t": nav.t,
            "length": r["length"], "max_alt": r["max_alt"], "end_alt": r["end_alt"],
            "climbs": nav.climbs, "skipped": len(nav.skipped),
            "min_clear": min(r["closest"].values()) if uids else float("inf")}


# ------------------------------------------------------------------ report
def write_report(path):
    lines = ["# PBI-12 range sensor and avoidance - test results", "",
             "Generated by `python range_sensor_runs.py` "
             f"(PyBullet {'stand-in' if 'stand-in' in (p.__doc__ or '') else 'engine'}).", ""]
    if "stand-in" in (p.__doc__ or ""):
        lines += ["> These numbers come from a geometric stand-in for PyBullet used during development "
                  "(meshes are approximated by boxes, timing is not the real engine's). Regenerate this "
                  "file with `python range_sensor_runs.py` on a machine with PyBullet before using it.", ""]
    lines += ["Baseline = the range sensor before PBI-12 (72 rays all round x 3 pitches + 3 corridor "
              "rays). Enhanced = PBI-12 (the same all-round fan + a 41-ray forward fan, 1.5 deg apart "
              "over +-30 deg around the direction of travel; the target is see-through; climb over "
              "low obstacles when clearly shorter).", ""]
    b, e, rays = scan_timing()
    lines += ["## AC-03 scan time", "",
              "| Sensor | Rays per scan | Time per scan (fastest of 30 batches) |", "|---|---|---|",
              f"| baseline (before PBI-12) | {rays['baseline']} | {b:.3f} ms |",
              f"| enhanced | {rays['enhanced']} | {e:.3f} ms |", "",
              f"Enhanced / baseline: **{e / b:.2f}x** (limit 1.20x).", ""]
    rows = detection_table()
    lines += ["## AC-01 detection distance", "",
              f"Distance from the drone's body to the object when the sensor first hit it "
              f"(must be more than STOP_DIST = {oa.STOP_DIST} m). Position = (x along the leg, "
              "y to the side).", "",
              "| Object | Position | Baseline | Enhanced | Closest approach (enhanced) | Result |",
              "|---|---|---|---|---|---|"]
    by = {(r["shape"], r["pos"], r["mode"]): r for r in rows}
    ok_all = True
    for s in SHAPES:
        for pos in POSITIONS:
            rb, re_ = by[(s[0], pos, "baseline")], by[(s[0], pos, "enhanced")]
            fmt = lambda v: "missed" if v is None else f"{v:.2f} m"
            ok = re_["detect_m"] is not None and re_["detect_m"] > oa.STOP_DIST and re_["closest_m"] > 0
            ok_all &= ok
            lines.append(f"| {s[0]} | ({pos[0]:g}, {pos[1]:g}) | {fmt(rb['detect_m'])} | "
                         f"{fmt(re_['detect_m'])} | {re_['closest_m']:.2f} m | {'PASS' if ok else 'FAIL'} |")
    lines += ["", f"AC-01: **{'PASS' if ok_all else 'FAIL'}**", ""]
    lines += ["## AC-02 target", "", "Each scene is flown twice, with and without the casualty. "
              "The obstacle map and the route must be identical.", "",
              "| Scene | Target hit by the sensor | Map identical | Route identical | Result |",
              "|---|---|---|---|---|"]
    for case in TARGET_CASES:
        a, b2 = target_case(case, True), target_case(case, False)
        same_map, same_route = a["map"] == b2["map"], a["events"] == b2["events"]
        ok = same_map and same_route and a["target"] not in a["nav"].detected
        lines.append(f"| {case[0]} | {'yes' if a['target_seen'] else 'no'} | {same_map} | "
                     f"{same_route} | {'PASS' if ok else 'FAIL'} |")
    lines += ["", "## AC-04 climb over or go round", "",
              "| Scene | Mode | Result | Distance flown | Max altitude | Ends at | Climbs | "
              "Closest to obstacle |", "|---|---|---|---|---|---|---|---|"]
    for case in AVOID_CASES:
        for climb in (False, True):
            r = avoid_case(case, climb)
            lines.append(f"| {r['label']} | {'climb or detour' if climb else 'detour only'} | "
                         f"{r['state']} ({r['t']:.0f} s) | {r['length']:.1f} m | {r['max_alt']:.2f} m | "
                         f"{r['end_alt']:.2f} m | {r['climbs']} | {r['min_clear']:.2f} m |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


if __name__ == "__main__":
    out = os.path.join(HERE, "tests", "RANGE_SENSOR_TEST_REPORT.md")
    print(write_report(out))
    print(f"\nwritten to {out}", file=sys.stderr)
