"""
Contributors: Mylan Bui and Sofia Monge
File Description: SPIKE-01 AC-03. Compares three search policies in headless runs and records
time-to-find and distance flown.

Policies (investigation question 2):
    sweep  - lawnmower pattern inside each zone, fixed zone order A -> B -> C -> D (ignores the map)
    greedy - always fly to the most likely cell (ProbabilityMap.best_target), repeat after each leg
    mix    - pick the zone with the most probability, sweep it, then pick again

Every policy starts from the same probability map (approach C: rings x terrain) and is run on the
same person positions, one per seed. Person positions are sampled from a separate "ground truth"
(distance rings with water/buildings removed, NO trail boost) so greedy is not graded against
the exact map it searches with.

Usage (same OSM flags as main.py / terrain_prior_spike.py):
    python spike01_policy_runs.py --lat 44.6497 --lon -123.3133 --area 200 --osm-file map.osm
    python spike01_policy_runs.py --lat 44.6497 --lon -123.3133 --area 200 --osm-file map.osm --runs 1 --policies greedy

Outputs (in --out, default spike01_results/): policy_runs.csv, policy_summary.md
"""
import argparse
import csv
import math
import os
import statistics
import time

import numpy as np
import pybullet as p

import env_sim
import osm_world
import sim_config
import terrain_prior_spike as spike
from env_sim import build_world, DRONE_START_POS, OBSTACLE_POSITIONS
from main import set_destination
from object_avoidance import Navigator, DT
from operator_guidance import ZONES, ORIGINAL_PLAN
from probability_map import ProbabilityMap
from target_detection import MissingPerson, SearchMonitor

POLICIES = ("sweep", "greedy", "mix")
CRUISE_ALT = 2.5          # same cruise altitude as main.py
LANE_SPACING = 4.0        # m between sweep lanes; the camera footprint is ~5 m wide at 2.5 m altitude
LKP = (0.0, 0.0)          # the OSM world puts the LKP at the map centre


def quiet(*_args, **_kwargs):
    """Silences Navigator / SearchMonitor event logs so 30 runs don't flood the terminal."""


# ------------------------------------------------------------------ sweep pattern --
def zone_box(centre, g):
    """Square search area around a zone centre (one quadrant of the map), clipped to the grid."""
    h = sim_config.MAP_HALF / 2
    return (max(centre[0] - h, g.x_range[0] + 1.0), min(centre[0] + h, g.x_range[1] - 1.0),
            max(centre[1] - h, g.y_range[0] + 1.0), min(centre[1] + h, g.y_range[1] - 1.0))


def lawnmower(box, start_xy):
    """Back-and-forth lanes (east-west) covering the box, starting from the corner nearest the drone."""
    x0, x1, y0, y1 = box
    n = max(1, math.ceil((y1 - y0) / LANE_SPACING))
    ys = [y0 + (i + 0.5) * (y1 - y0) / n for i in range(n)]
    if abs(start_xy[1] - ys[-1]) < abs(start_xy[1] - ys[0]):
        ys.reverse()
    left_first = abs(start_xy[0] - x0) <= abs(start_xy[0] - x1)
    pts = []
    for i, y in enumerate(ys):
        a, b = (x0, x1) if (i % 2 == 0) == left_first else (x1, x0)
        pts += [(a, y), (b, y)]
    return pts


# ------------------------------------------------------------------ policies --
class Policy:
    """Chooses the drone's next destination each time it finishes a leg. None = plan finished."""

    def __init__(self, name, belief, g):
        self.name, self.belief, self.g = name, belief, g
        self.queue = []                       # remaining sweep waypoints in the current zone
        self.zones_left = list(ORIGINAL_PLAN)

    def next(self, pos):
        if self.name == "greedy":
            return self._greedy(pos)
        if not self.queue:
            if not self.zones_left:
                return None
            if self.name == "mix":
                # Most probable zone first; re-ranked after each zone because the map has changed.
                r = sim_config.MAP_HALF / 2
                self.zones_left.sort(key=lambda z: -self.belief.mass_near(ZONES[z], r))
            zone = self.zones_left.pop(0)
            self.queue = lawnmower(zone_box(ZONES[zone], self.g), pos)
        return self.queue.pop(0)

    def _greedy(self, pos):
        target = self.belief.best_target(pos, distance_weight=0.1)
        # The camera looks forward, so the cell directly under the drone is never in view. If the
        # best cell is where the drone already is, count it as searched so greedy moves on.
        for _ in range(5):
            if math.hypot(target[0] - pos[0], target[1] - pos[1]) >= 1.5:
                break
            self.belief.mark_searched(target, self.belief.cell)
            target = self.belief.best_target(pos, distance_weight=0.1)
        return target


# ------------------------------------------------------------------ person placement --
def person_positions(scene, g, terrain, runs, first_seed):
    """One person position per seed, sampled from the ground-truth map, moved off any tree/building."""
    truth = spike.prior_rings(g, terrain, LKP) * (spike.weights(terrain) > 0)
    flat = (truth / truth.sum()).ravel()
    out = []
    for seed in range(first_seed, first_seed + runs):
        rng = np.random.default_rng(seed)
        iy, ix = np.unravel_index(rng.choice(flat.size, p=flat), truth.shape)
        x = g.cx[iy, ix] + rng.uniform(-g.cell / 2, g.cell / 2)
        y = g.cy[iy, ix] + rng.uniform(-g.cell / 2, g.cell / 2)
        x, y = osm_world.nearest_free_point(scene, (float(x), float(y)), margin=1.2)
        out.append((seed, (round(x, 2), round(y, 2))))
    return out


# ------------------------------------------------------------------ one run --
def run_one(policy_name, seed, person_xy, floor_id, drone_id, g, prior, max_time):
    person = MissingPerson.spawn(xy=person_xy)
    monitor = SearchMonitor(person, gui=False, log=quiet, seed=seed)
    belief = ProbabilityMap(x_range=g.x_range, y_range=g.y_range, cell_size=g.cell, pod=0.65)
    belief.set_prior(prior)
    nav = Navigator(drone_id, floor_id, person.body_id,
                    [DRONE_START_POS, [DRONE_START_POS[0], DRONE_START_POS[1], CRUISE_ALT]], False, quiet)
    nav.route = []
    nav.alt = CRUISE_ALT
    policy = Policy(policy_name, belief, g)

    distance, prev, status = 0.0, list(nav.pos), "timeout"
    set_destination(nav, policy.next(nav.pos[:2]))
    while nav.t < max_time:
        if nav.state in ("NAV", "CLIMB", "RTL"):
            nav.update(DT)
        else:
            nav.t += DT
        if nav.state == "DONE":
            target = policy.next(nav.pos[:2])
            if target is None:
                status = "plan finished"
                break
            set_destination(nav, target)
        elif nav.state in ("ABORT", "RTL_DONE"):
            status = nav.state.lower()
            break

        distance += math.dist(nav.pos, prev)
        prev = list(nav.pos)
        p.resetBasePositionAndOrientation(drone_id, nav.pos, (0, 0, 0, 1))
        monitor.update(nav.pos, nav.t)
        belief.observe_camera(monitor, nav.pos, DT, trees=OBSTACLE_POSITIONS)
        if monitor.found:
            status = "found"
            break

    p.removeBody(person.body_id)
    return {"policy": policy_name, "seed": seed, "person_x": person_xy[0], "person_y": person_xy[1],
            "found": status == "found", "status": status,
            "time_s": round(monitor.found_time if status == "found" else nav.t, 2),
            "distance_m": round(distance, 2)}


# ------------------------------------------------------------------ summary --
def summarize(rows, policies, runs, max_time):
    lines = ["| Policy | Found | Median time-to-find (s) | Median distance flown (m) |",
             "|---|---|---|---|"]
    for name in policies:
        rs = [r for r in rows if r["policy"] == name]
        found = sum(r["found"] for r in rs)
        # A run that never found the person counts as max_time (marked >=), so medians stay over all runs.
        times = [r["time_s"] if r["found"] else max_time for r in rs]
        med_t = statistics.median(times)
        flag = ">=" if med_t >= max_time else ""
        lines.append(f"| {name} | {found}/{runs} | {flag}{med_t:.1f} | {statistics.median(r['distance_m'] for r in rs):.1f} |")
    lines += ["", f"Runs that did not find the person count as {max_time:.0f} s (the time limit) in the median time."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lat", type=float, required=True, help="LKP latitude")
    ap.add_argument("--lon", type=float, required=True, help="LKP longitude")
    ap.add_argument("--area", type=float, default=200.0, help="real metres of OSM shown (as in main.py)")
    ap.add_argument("--map-size", type=float, default=None, help="sim map width override (as in main.py)")
    ap.add_argument("--osm-file", help="offline OSM export instead of downloading")
    ap.add_argument("--seed", type=int, default=0, help="seed for trees scattered in woods (world layout)")
    ap.add_argument("--runs", type=int, default=10, help="runs per policy (one person position per run)")
    ap.add_argument("--first-seed", type=int, default=0, help="seed of the first person position")
    ap.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    ap.add_argument("--max-time", type=float, default=None, help="sim seconds per run (default as main.py)")
    ap.add_argument("--out", default="spike01_results")
    args = ap.parse_args()

    # Same world and probability-map grid as main.py / terrain_prior_spike.py for these flags.
    scene, g, source = spike.load_osm_world(args)
    for name, centre in list(ZONES.items()):           # zone centres moved off buildings, as main.py does
        if tuple(centre) in scene.moved:
            ZONES[name] = scene.moved[tuple(centre)]
    terrain = spike.rasterize_scene(scene, g)
    prior = spike.prior_combined(g, terrain, LKP)
    max_time = args.max_time if args.max_time is not None else 240.0 * max(1.0, sim_config.layout_scale())

    env_sim.DETAILED_VISUALS = False                   # headless: visuals don't matter, collisions are identical
    floor_id, drone_id, _ = build_world(False)
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)

    people = person_positions(scene, g, terrain, args.runs, args.first_seed)
    print(f"\n{source} | zones {dict(ZONES)} | time limit {max_time:.0f} s")
    print("Person positions:", ", ".join(f"seed {s}: {xy}" for s, xy in people))

    rows = []
    for name in args.policies:
        for seed, xy in people:
            t0 = time.perf_counter()
            r = run_one(name, seed, xy, floor_id, drone_id, g, prior, max_time)
            rows.append(r)
            print(f"[{name:<6} seed {seed}] {r['status']:<13} t={r['time_s']:7.1f} s  "
                  f"dist={r['distance_m']:7.1f} m  ({time.perf_counter() - t0:.0f} s wall)")
    p.disconnect()

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "policy_runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows, args.policies, args.runs, max_time)
    with open(os.path.join(args.out, "policy_summary.md"), "w", encoding="utf-8") as f:
        f.write(summary + "\n")
    print("\n" + summary)
    print(f"\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
