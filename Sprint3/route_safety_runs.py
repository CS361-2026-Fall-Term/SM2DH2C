"""
Contributors: Mylan Bui and Sofia Monge
File Description: PBI-04 (#62) verification runner. Flies the Navigator through 20 seeded,
generated environments (env_sim.generate_layout) and checks the acceptance criteria:

  AC-01  8-20 obstacles; every reachable destination (free path >= 1 m wide) is reached with
         zero collisions and minimum clearance >= 0.3 m from any obstacle surface
  AC-02  an unreachable destination is logged by name and the drone is moving again
         within 2 s of simulated time
  AC-03  every A* replan takes < 50 ms (max reported)
  AC-04  the same seed twice gives the same obstacles and the same planned routes

"Reachable" is checked against ground truth: the known obstacle shapes (tree canopies, tall
walls) padded by 0.5 m (half of a 1 m-wide path), searched from the start at every altitude the
drone can climb to. Every 4th seed walls one waypoint in with 7 m walls so AC-02 is exercised.

Usage (from Sprint2/):
    python route_safety_runs.py                 # 20 seeds, each run twice (AC-04)
    python route_safety_runs.py --seeds 3       # quick check
    python route_safety_runs.py --no-repeat     # skip the AC-04 second runs (faster)
Outputs (in --out, default route_safety_results/): runs.csv, summary.md
"""
import argparse
import csv
import math
import os
import random
import time
from collections import deque

import pybullet as p

import env_sim
import sim_config
from env_sim import DRONE_START_POS, FLIGHT_PATH, OBSTACLE_POSITIONS, WALLS
from object_avoidance import Navigator, DT, CLIMB_STEP, MAX_ALT

MIN_CLEARANCE = 0.3          # AC-01
PATH_HALF_WIDTH = 0.5        # AC-01: a path at least 1 m wide
SKIP_MOVE_LIMIT_S = 2.0      # AC-02
REPLAN_LIMIT_MS = 50.0       # AC-03
GT_CELL = 0.1                # ground-truth grid resolution (m)
CANOPY_Z, CANOPY_R = 2.25, 0.75   # tree canopy sphere (env_sim.build_world: z 0.5 + 1.75, r 0.75)


# ------------------------------------------------------------------ world setup --
def layout_for(seed):
    """Obstacle count 8-20 and, every 4th seed, one walled-in waypoint (index 2..n-2)."""
    count = 8 + seed % 13
    enclose_idx = None
    if seed % 4 == 3:
        # Only wall in a waypoint with no other waypoint within 3 m (pen half 1.5 + margin 1.5),
        # so exactly one destination becomes unreachable.
        ok = [i for i in range(2, len(FLIGHT_PATH) - 1)
              if all(math.hypot(FLIGHT_PATH[i][0] - w[0], FLIGHT_PATH[i][1] - w[1]) >= 3.0
                     for k, w in enumerate(FLIGHT_PATH) if k != i)]
        enclose_idx = random.Random(1000 + seed).choice(ok)
    return count, enclose_idx


def make_world(seed):
    count, enclose_idx = layout_for(seed)
    enclose = tuple(FLIGHT_PATH[enclose_idx][:2]) if enclose_idx is not None else None
    others = [w for i, w in enumerate(FLIGHT_PATH) if i != enclose_idx]
    info = env_sim.generate_layout(seed, count, keep_clear=others, enclose=enclose)
    return info, enclose_idx


# ------------------------------------------------------------------ ground truth --
def blocked_grid(alt):
    """Cells within PATH_HALF_WIDTH of an obstacle surface at altitude `alt` (true = blocked)."""
    half = sim_config.MAP_HALF
    n = int(round(2 * half / GT_CELL))
    grid = [[False] * n for _ in range(n)]

    def mark_disc(cx, cy, r):
        r += PATH_HALF_WIDTH
        i0, i1 = int((cx - r + half) / GT_CELL), int((cx + r + half) / GT_CELL)
        j0, j1 = int((cy - r + half) / GT_CELL), int((cy + r + half) / GT_CELL)
        for i in range(max(i0, 0), min(i1 + 1, n)):
            x = -half + (i + 0.5) * GT_CELL
            for j in range(max(j0, 0), min(j1 + 1, n)):
                y = -half + (j + 0.5) * GT_CELL
                if (x - cx) ** 2 + (y - cy) ** 2 <= r * r:
                    grid[i][j] = True

    def mark_box(cx, cy, hx, hy):
        hx, hy = hx + PATH_HALF_WIDTH, hy + PATH_HALF_WIDTH
        for i in range(max(int((cx - hx + half) / GT_CELL), 0), min(int((cx + hx + half) / GT_CELL) + 1, n)):
            for j in range(max(int((cy - hy + half) / GT_CELL), 0), min(int((cy + hy + half) / GT_CELL) + 1, n)):
                grid[i][j] = True

    for x, y, _ in OBSTACLE_POSITIONS:
        dz = alt - CANOPY_Z
        if abs(dz) < CANOPY_R:                       # canopy cross-section at this altitude
            mark_disc(x, y, math.sqrt(CANOPY_R ** 2 - dz ** 2))
    for x, y, hx, hy, h in WALLS:
        if alt <= h:
            mark_box(x, y, hx, hy)
    return grid, n


def reachable_waypoints():
    """Indices of FLIGHT_PATH waypoints (1..) reachable from the start at any climbable altitude."""
    half = sim_config.MAP_HALF
    reach = set()
    alts, a = [], FLIGHT_PATH[0][2]
    while a <= MAX_ALT + 1e-6:
        alts.append(a)
        a += CLIMB_STEP
    for alt in alts:
        grid, n = blocked_grid(alt)
        to_cell = lambda x, y: (min(max(int((x + half) / GT_CELL), 0), n - 1),
                                min(max(int((y + half) / GT_CELL), 0), n - 1))
        start = to_cell(*FLIGHT_PATH[0][:2])
        if grid[start[0]][start[1]]:
            continue
        seen = [[False] * n for _ in range(n)]
        seen[start[0]][start[1]] = True
        q = deque([start])
        while q:
            i, j = q.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
                a2, b2 = i + di, j + dj
                if 0 <= a2 < n and 0 <= b2 < n and not seen[a2][b2] and not grid[a2][b2]:
                    seen[a2][b2] = True
                    q.append((a2, b2))
        for k in range(1, len(FLIGHT_PATH)):
            ci, cj = to_cell(*FLIGHT_PATH[k][:2])
            if seen[ci][cj]:
                reach.add(k)
    return reach


# ------------------------------------------------------------------ one mission --
def fly(seed, max_time, verbose):
    info, enclose_idx = make_world(seed)
    layout = [tuple(o) for o in OBSTACLE_POSITIONS] + [tuple(w) for w in WALLS]
    reachable = reachable_waypoints()

    floor_id, drone_id, obstacle_ids = env_sim.build_world(gui=False)
    log = print if verbose else (lambda *_a, **_k: None)
    nav = Navigator(drone_id, floor_id, -1, FLIGHT_PATH, False, log)   # -1: no target body
    quat = p.getQuaternionFromEuler([0, 0, 0])

    routes = [tuple((round(x, 2), round(y, 2), t) for x, y, t in nav.route)]
    last_replans = 0
    min_clear, contacts = float("inf"), 0
    pending_skips = []                # sim times of skips still waiting for the drone to move
    skip_move_s = []
    while nav.t < max_time and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
        n_skipped = len(nav.skipped)
        nav.update(DT)
        for t_skip, _ in nav.skipped[n_skipped:]:
            pending_skips.append(t_skip)
        if pending_skips and nav.state in ("NAV", "RTL") and nav.horizontal_speed > 0.1:
            skip_move_s += [nav.t - t0 for t0 in pending_skips]
            pending_skips = []
        if nav.replans != last_replans:
            last_replans = nav.replans
            routes.append(tuple((round(x, 2), round(y, 2), t) for x, y, t in nav.route))

        p.resetBasePositionAndOrientation(drone_id, nav.pos, quat)
        d = float("inf")
        for oid in obstacle_ids:
            pts = p.getClosestPoints(drone_id, oid, 3.0)
            if pts:
                d = min(d, min(c[8] for c in pts))
        min_clear = min(min_clear, d)
        if d <= 0.0:
            contacts += 1
    p.disconnect()

    reached = {int(m.rsplit(" ", 1)[1]) for _, k, m in nav.events if k == "WAYPT"}
    skipped = sorted({j for _, j in nav.skipped})
    unreachable = sorted(set(range(1, len(FLIGHT_PATH))) - reachable)
    return {
        "seed": seed, "trees": info["trees"], "walls": info["walls"],
        "obstacles": info["trees"] + info["walls"], "walled_wp": enclose_idx,
        "reachable": len(reachable), "reached_reachable": len(reachable & reached),
        "missed": sorted(reachable - reached), "unreachable": unreachable, "skipped": skipped,
        "min_clear_m": round(min_clear, 3), "contacts": contacts,
        "max_replan_ms": round(max(nav.replan_ms), 2) if nav.replan_ms else 0.0,
        "replans": len(nav.replan_ms),
        "max_skip_move_s": round(max(skip_move_s), 2) if skip_move_s else None,
        "unmoved_skips": len(pending_skips),
        "final_state": nav.state, "sim_time_s": round(nav.t, 1),
        "_layout": layout, "_routes": routes,
    }


# ------------------------------------------------------------------ checks + report --
def check(r, again):
    ac1 = (r["missed"] == [] and r["contacts"] == 0 and r["min_clear_m"] >= MIN_CLEARANCE
           and 8 <= r["obstacles"] <= 20)
    ac2 = (set(r["unreachable"]) <= set(r["skipped"]) and r["unmoved_skips"] == 0
           and (r["max_skip_move_s"] is None or r["max_skip_move_s"] <= SKIP_MOVE_LIMIT_S))
    ac3 = r["max_replan_ms"] < REPLAN_LIMIT_MS
    ac4 = None if again is None else (again["_layout"] == r["_layout"] and again["_routes"] == r["_routes"])
    return ac1, ac2, ac3, ac4


def mark(ok):
    return "n/a" if ok is None else ("PASS" if ok else "FAIL")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20, help="number of seeded environments")
    ap.add_argument("--first-seed", type=int, default=0)
    ap.add_argument("--max-time", type=float, default=240.0, help="sim time limit per mission (s)")
    ap.add_argument("--no-repeat", action="store_true", help="skip the AC-04 second run of each seed")
    ap.add_argument("--verbose", action="store_true", help="print Navigator events")
    ap.add_argument("--out", default="route_safety_results")
    args = ap.parse_args()

    rows = []
    for seed in range(args.first_seed, args.first_seed + args.seeds):
        t0 = time.perf_counter()
        r = fly(seed, args.max_time, args.verbose)
        again = None if args.no_repeat else fly(seed, args.max_time, False)
        r["AC-01"], r["AC-02"], r["AC-03"], r["AC-04"] = map(mark, check(r, again))
        rows.append(r)
        print(f"seed {seed:2d} | {r['obstacles']:2d} obstacles | reached {r['reached_reachable']}/{r['reachable']} "
              f"| skipped {r['skipped'] or '-'} | clear {r['min_clear_m']:.2f} m | contacts {r['contacts']} "
              f"| replan max {r['max_replan_ms']:.1f} ms | {r['AC-01']} {r['AC-02']} {r['AC-03']} {r['AC-04']} "
              f"({time.perf_counter() - t0:.0f} s wall)")

    os.makedirs(args.out, exist_ok=True)
    cols = [k for k in rows[0] if not k.startswith("_")]
    with open(os.path.join(args.out, "runs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    lines = ["| Seed | Obstacles | Walled WP | Reached / reachable | Skipped (logged) | Min clearance (m) "
             "| Contacts | Max replan (ms) | Skip -> moving (s) | AC-01 | AC-02 | AC-03 | AC-04 |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['seed']} | {r['obstacles']} | {r['walled_wp'] if r['walled_wp'] is not None else '-'} "
                     f"| {r['reached_reachable']}/{r['reachable']} | {', '.join(map(str, r['skipped'])) or '-'} "
                     f"| {r['min_clear_m']:.2f} | {r['contacts']} | {r['max_replan_ms']:.1f} "
                     f"| {r['max_skip_move_s'] if r['max_skip_move_s'] is not None else '-'} "
                     f"| {r['AC-01']} | {r['AC-02']} | {r['AC-03']} | {r['AC-04']} |")
    summary = {ac: all(r[ac] in ("PASS", "n/a") for r in rows) for ac in ("AC-01", "AC-02", "AC-03", "AC-04")}
    worst_ms = max(r["max_replan_ms"] for r in rows)
    worst_clear = min(r["min_clear_m"] for r in rows)
    lines += ["", f"**Summary over {len(rows)} seeds** (map {sim_config.map_size():g} m, "
                  f"A* cell {__import__('object_avoidance').CELL} m, safety radius "
                  f"{__import__('object_avoidance').SAFETY_RADIUS} m):",
              f"- AC-01: {mark(summary['AC-01'])} (lowest clearance {worst_clear:.2f} m, "
              f"total contacts {sum(r['contacts'] for r in rows)})",
              f"- AC-02: {mark(summary['AC-02'])}",
              f"- AC-03: {mark(summary['AC-03'])} (max replan {worst_ms:.1f} ms)",
              f"- AC-04: {mark(summary['AC-04']) if not args.no_repeat else 'n/a (--no-repeat)'}"]
    text = "\n".join(lines)
    with open(os.path.join(args.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print("\n" + text + f"\n\nOutputs written to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
