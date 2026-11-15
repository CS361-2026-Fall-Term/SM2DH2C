"""
Contributor: Chase Vanderlip
File Description: Verification runner for obstacle detection/avoidance on obstacles generated from
the operator's geographical input (geo_obstacles.py). Flies the Navigator headless through worlds
built from map data and checks the acceptance criteria against GROUND TRUTH (PyBullet closest-point
distances between the drone's collision body and every obstacle), not against what the sensor
reported:

  AC-01  the map's key objects (buildings, single trees, trees scattered in woods) are placed as
         URDF bodies at their real positions (placement error < 5 cm) with no problems reported
  AC-02  every obstacle standing on the route is detected and named, the drone re-plans around it,
         reaches every waypoint, never touches anything and never comes inside UNSAFE_DIST (0.35 m).
         "popup" scenario: an obstacle appears INSIDE the unsafe distance mid-flight; the sensor
         must flag it (UNSAFE), re-plan, back out past 0.35 m within RECOVER_LIMIT_S and finish.
  (AC-03, malformed input, is covered by tests/test_geo_obstacles.py.)

Scenarios
  fixture        tests/fixtures/geo_test_area.json: buildings and trees at PREDEFINED positions on
                 the legs of env_sim.FLIGHT_PATH (26 m map, true scale)
  popup          same map, plus a building that appears 0.25 m beside the drone at t = 3 s
  osm-60/200/500 the real export map.osm (Philomath, OR area) shown 60 / 200 / 500 real metres
                 across a 60 m map

Usage (from Sprint2/):
    python geo_avoidance_runs.py                     # all scenarios
    python geo_avoidance_runs.py --only fixture popup
    python geo_avoidance_runs.py --verbose           # print Navigator events
Outputs (in --out, default geo_avoidance_results/): runs.csv, summary.md
"""
import argparse
import copy
import csv
import math
import os
import tempfile
import time

import pybullet as p

import env_sim
import geo_obstacles
import sim_config
from object_avoidance import Navigator, DT, UNSAFE_DIST

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = os.path.join(HERE, "tests", "fixtures", "geo_test_area.json")
MAP_OSM = os.path.join(HERE, "map.osm")
LKP_FIXTURE = (44.6488, -123.3123)
LKP_MAP_OSM = (44.6488, -123.3123)        # centre of map.osm's bounds

PLACEMENT_TOL_M = 0.05                    # AC-01
RECOVER_LIMIT_S = 2.0                     # AC-02 popup: back outside UNSAFE_DIST within this time
POPUP_T = 3.0                             # when the popup obstacle appears (sim s)
POPUP_GAP = 0.25                          # its distance from the drone's body when it appears (m)
GT_EVERY = 6                              # ground-truth sample every 6 physics steps (40 Hz)

# Obstacles the fixture puts ON the route: each must be detected (AC-02).
FIXTURE_ON_ROUTE = ('building "Test Hall"', 'building "L Annex"', 'building "Garage"')
FIXTURE_TREES_ON_ROUTE = ((-7.0, 10.2), (7.5, 5.0), (7.5, -5.2))

SCENARIOS = {
    "fixture": dict(file=FIXTURE, lkp=LKP_FIXTURE, area=26.0, map_size=26.0, popup=False),
    "popup": dict(file=FIXTURE, lkp=LKP_FIXTURE, area=26.0, map_size=26.0, popup=True),
    "osm-60": dict(file=MAP_OSM, lkp=LKP_MAP_OSM, area=60.0, map_size=60.0, popup=False),
    "osm-200": dict(file=MAP_OSM, lkp=LKP_MAP_OSM, area=200.0, map_size=60.0, popup=False),
    "osm-500": dict(file=MAP_OSM, lkp=LKP_MAP_OSM, area=500.0, map_size=60.0, popup=False),
}

# env_sim's world lists are changed IN PLACE by scale_layout() / use_osm(); keep the originals so
# every scenario starts from the same hand-placed layout.
_ORIGINAL = {name: copy.deepcopy(getattr(env_sim, name)) for name in
             ("DRONE_START_POS", "TARGET_POSITION", "OBSTACLE_POSITIONS", "FLIGHT_PATH", "TREE_KINDS",
              "WALLS")}


def reset_world():
    for name, value in _ORIGINAL.items():
        getattr(env_sim, name)[:] = copy.deepcopy(value)
    env_sim.OSM_SCENE = None
    env_sim.OBSTACLE_INFO.clear()
    env_sim.GEO_PROBLEMS.clear()


# ------------------------------------------------------------------ ground truth --
def clearance(drone_id, ids, reach=1.5):
    """Exact distance from the drone's collision body to the nearest obstacle surface (m), or
    `reach` if nothing is that close. Only bodies whose bounding box is near the drone are asked."""
    lo, hi = p.getAABB(drone_id)
    near = p.getOverlappingObjects([a - reach for a in lo], [b + reach for b in hi]) or []
    best = reach
    for uid, _link in near:
        if uid in ids:
            for c in p.getClosestPoints(drone_id, uid, reach):
                best = min(best, c[8])
    return best


def placement_error(table):
    """Largest difference between where an obstacle should be and where PyBullet has it (m).
    Tree: base position vs the map position. Building/wall: bounding box vs the outline's box."""
    worst = 0.0
    for uid, ob in table.items():
        if ob.kind == "tree":
            pos, _ = p.getBasePositionAndOrientation(uid)
            worst = max(worst, math.dist(pos, (ob.x, ob.y, ob.z)))
        else:
            lo, hi = p.getAABB(uid)
            # union over links (a building is one link; the AABB above is the base's)
            for link in range(p.getNumJoints(uid)):
                l2, h2 = p.getAABB(uid, link)
                lo, hi = [min(a, b) for a, b in zip(lo, l2)], [max(a, b) for a, b in zip(hi, h2)]
            xs, ys = [q[0] for q in ob.footprint], [q[1] for q in ob.footprint]
            want_lo, want_hi = (min(xs), min(ys), 0.0), (max(xs), max(ys), ob.height)
            # PyBullet pads AABBs by its collision margin (a few cm): compare with that removed
            worst = max(worst, max(abs(a - b) for a, b in zip(lo, want_lo)) - 0.02,
                        max(abs(a - b) for a, b in zip(hi, want_hi)) - 0.02)
    return max(0.0, worst)


def spawn_popup(nav, ids):
    """A 2 m x 2 m building that appears beside the drone, its near wall POPUP_GAP from the body."""
    h = nav._heading() or (1.0, 0.0)
    side = (-h[1], h[0])                                   # left of the direction of travel
    body_r = 0.12                                          # drone collision radius (0.08 x 1.5)
    near = POPUP_GAP + body_r
    cx, cy = nav.pos[0] + side[0] * (near + 1.0), nav.pos[1] + side[1] * (near + 1.0)
    ux, uy = h
    corners = [(cx + sx * ux + sy * side[0], cy + sx * uy + sy * side[1])
               for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    ob = geo_obstacles.Obstacle("building", 'building "Pop-up Kiosk"', 0.0, 0.0, 0.0, height=6.0,
                                radius=math.sqrt(2), cx=cx, cy=cy, footprint=corners, source="map",
                                meta={"index": 999})
    new_ids, table, problems = geo_obstacles.spawn(p, [ob], out_dir=tempfile.mkdtemp(prefix="popup_"))
    ids.update(new_ids)
    nav.obstacle_info.update(table)
    nav.event("WORLD", f"(test) {ob.name} appeared {POPUP_GAP:.2f} m from the drone")
    return new_ids


# ------------------------------------------------------------------ one mission --
def fly(name, cfg, max_time, verbose, detailed):
    reset_world()
    sim_config.set_map_size(cfg["map_size"])
    env_sim.scale_layout(sim_config.layout_scale())
    env_sim.DETAILED_VISUALS = detailed
    keep = [] if cfg["file"] == FIXTURE else [tuple(w[:2]) for w in env_sim.FLIGHT_PATH[1:]]
    scene = env_sim.use_osm(*cfg["lkp"], area_m=cfg["area"], osm_file=cfg["file"], keep_clear=keep)
    floor_id, drone_id, obstacle_ids = env_sim.build_world(gui=False)
    table = dict(env_sim.OBSTACLE_INFO)
    expected = len(scene.buildings) + len(scene.trees)
    place_err = placement_error(table)

    log = print if verbose else (lambda *_a, **_k: None)
    nav = Navigator(drone_id, floor_id, -1, env_sim.FLIGHT_PATH, False, log,
                    obstacle_info=env_sim.OBSTACLE_INFO)
    quat = p.getQuaternionFromEuler([0, 0, 0])
    ids = set(obstacle_ids)
    min_clear, unsafe_samples, contacts, step = float("inf"), 0, 0, 0
    popup_ids, popup_t, recovered_t, unsafe_after_recover = None, None, None, 0
    while nav.t < max_time and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
        if cfg["popup"] and popup_ids is None and nav.t >= POPUP_T:
            popup_ids, popup_t = spawn_popup(nav, ids), nav.t
        nav.update(DT)
        p.resetBasePositionAndOrientation(drone_id, nav.pos, quat)
        if step % GT_EVERY == 0:
            d = clearance(drone_id, ids)
            min_clear = min(min_clear, d)
            contacts += d <= 0.0
            if d < UNSAFE_DIST:
                unsafe_samples += 1
                if recovered_t is not None:
                    unsafe_after_recover += 1
            elif popup_t is not None and recovered_t is None:
                recovered_t = nav.t
        step += 1
    p.disconnect()

    events = nav.events
    detected = {nav.obstacle_info[u].name for u in nav.detected if u in nav.obstacle_info}
    reached = {int(m.rsplit(" ", 1)[1]) for _, k, m in events if k == "WAYPT"}
    unsafe_events = [(t, m) for t, k, m in events if k == "UNSAFE"]
    replans_after_unsafe = [t for t, k, _ in events if k == "REPLAN" and unsafe_events
                            and t >= unsafe_events[0][0]]
    kinds = {k: sum(1 for o in table.values() if o.kind == k) for k in ("building", "tree")}
    det_kinds = {k: sum(1 for u in nav.detected if u in table and table[u].kind == k)
                 for k in ("building", "tree")}
    r = {
        "scenario": name, "area_m": cfg["area"], "map_m": cfg["map_size"],
        "buildings": kinds["building"], "trees": kinds["tree"], "expected": expected,
        "placed": len(table), "problems": len(env_sim.GEO_PROBLEMS),
        "placement_err_m": round(place_err, 4),
        "detected_buildings": det_kinds["building"], "detected_trees": det_kinds["tree"],
        "replans": nav.replans, "max_replan_ms": round(max(nav.replan_ms), 2) if nav.replan_ms else 0.0,
        "waypoints": len(env_sim.FLIGHT_PATH) - 1, "reached": len(reached),
        "skipped": sorted({j for _, j in nav.skipped}),
        "min_clear_m": round(min_clear, 3), "unsafe_samples": unsafe_samples, "contacts": contacts,
        "unsafe_alerts": nav.unsafe_alerts,
        "popup_alert_delay_s": (round(unsafe_events[0][0] - popup_t, 3)
                                if popup_t is not None and unsafe_events else None),
        "popup_recover_s": round(recovered_t - popup_t, 2) if recovered_t is not None else None,
        "unsafe_after_recover": unsafe_after_recover,
        "replanned_after_unsafe": bool(replans_after_unsafe),
        "final_state": nav.state, "sim_time_s": round(nav.t, 1),
        "_detected_names": detected, "_events": events, "_table": table,
    }
    return r


# ------------------------------------------------------------------ checks + report --
def check(r):
    ac1 = (r["placed"] == r["expected"] and r["problems"] == 0
           and r["placement_err_m"] < PLACEMENT_TOL_M and r["placed"] > 0)
    flown = r["final_state"] == "DONE" and r["contacts"] == 0
    if r["scenario"] == "popup":
        ac2 = (flown and r["unsafe_alerts"] >= 1 and r["popup_alert_delay_s"] is not None
               and r["popup_alert_delay_s"] <= 0.05 and r["replanned_after_unsafe"]
               and r["popup_recover_s"] is not None and r["popup_recover_s"] <= RECOVER_LIMIT_S
               and r["unsafe_after_recover"] == 0 and r["reached"] == r["waypoints"])
    else:
        ac2 = flown and r["unsafe_samples"] == 0 and r["min_clear_m"] >= UNSAFE_DIST
        if r["scenario"] == "fixture":
            names = r["_detected_names"]
            trees = [o for o in r["_table"].values() if o.kind == "tree"
                     and any(math.dist((o.x, o.y), t) < 0.05 for t in FIXTURE_TREES_ON_ROUTE)]
            trees_seen = all(o.name in names for o in trees) and len(trees) == len(FIXTURE_TREES_ON_ROUTE)
            ac2 = (ac2 and all(n in names for n in FIXTURE_ON_ROUTE) and trees_seen
                   and r["replans"] > 0 and r["reached"] == r["waypoints"])
    return ac1, ac2


def mark(ok):
    return "PASS" if ok else "FAIL"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="+", choices=list(SCENARIOS), help="run only these scenarios")
    ap.add_argument("--max-time", type=float, default=240.0, help="sim time limit per mission (s)")
    ap.add_argument("--detailed", action="store_true", help="build the detailed look (slower; same collision)")
    ap.add_argument("--verbose", action="store_true", help="print Navigator events")
    ap.add_argument("--out", default="geo_avoidance_results")
    args = ap.parse_args()

    rows = []
    for name in args.only or SCENARIOS:
        t0 = time.perf_counter()
        r = fly(name, SCENARIOS[name], args.max_time, args.verbose, args.detailed)
        r["AC-01"], r["AC-02"] = map(mark, check(r))
        rows.append(r)
        print(f"{name:8s} | {r['buildings']:2d} bldg {r['trees']:3d} trees placed (err "
              f"{r['placement_err_m'] * 100:.1f} cm) | detected {r['detected_buildings']} bldg "
              f"{r['detected_trees']} trees | reached {r['reached']}/{r['waypoints']} | clear "
              f"{r['min_clear_m']:.2f} m | unsafe {r['unsafe_samples']} contacts {r['contacts']} | "
              f"{r['final_state']} | {r['AC-01']} {r['AC-02']} ({time.perf_counter() - t0:.0f} s wall)")

    os.makedirs(args.out, exist_ok=True)
    cols = [k for k in rows[0] if not k.startswith("_")]
    with open(os.path.join(args.out, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = ["| Scenario | Area / map (m) | Buildings | Trees | Placed / expected | Placement err (cm) "
             "| Detected (bldg / trees) | Re-plans | Reached | Skipped | Min clearance (m) "
             "| Unsafe samples | Contacts | UNSAFE alerts | AC-01 | AC-02 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['scenario']} | {r['area_m']:g} / {r['map_m']:g} | {r['buildings']} | {r['trees']} "
                     f"| {r['placed']}/{r['expected']} | {r['placement_err_m'] * 100:.1f} "
                     f"| {r['detected_buildings']} / {r['detected_trees']} | {r['replans']} "
                     f"| {r['reached']}/{r['waypoints']} | {', '.join(map(str, r['skipped'])) or '-'} "
                     f"| {r['min_clear_m']:.2f} | {r['unsafe_samples']} | {r['contacts']} "
                     f"| {r['unsafe_alerts']} | {r['AC-01']} | {r['AC-02']} |")
    pop = next((r for r in rows if r["scenario"] == "popup"), None)
    lines += ["", f"Unsafe distance UNSAFE_DIST = {UNSAFE_DIST} m (drone body to obstacle surface, "
                  f"ground truth from PyBullet closest points, sampled at {240 // GT_EVERY} Hz)."]
    if pop:
        lines.append(f"Popup: obstacle appeared {POPUP_GAP} m from the drone at t = {POPUP_T:g} s; UNSAFE "
                     f"logged after {pop['popup_alert_delay_s']} s, re-planned: {pop['replanned_after_unsafe']}, "
                     f"back outside {UNSAFE_DIST} m after {pop['popup_recover_s']} s "
                     f"(limit {RECOVER_LIMIT_S:g} s), unsafe samples after that: {pop['unsafe_after_recover']}.")
    overall = {ac: all(r[ac] == "PASS" for r in rows) for ac in ("AC-01", "AC-02")}
    lines += ["", f"**Summary over {len(rows)} scenarios:** AC-01 {mark(overall['AC-01'])}, "
                  f"AC-02 {mark(overall['AC-02'])}"]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
