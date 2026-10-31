"""Headless SAR backend that streams to Unity.

    python sar_demo.py                  # PyBullet (p.DIRECT) + Unity bridge
    python sar_demo.py --no-pybullet    # kinematic stand-in, protocol testing only

Plug your own Navigator / SearchMonitor / ProbabilityMap / Mission in at the
marked "HOOK" comments; the bridge calls are the only new code you need.
"""
import argparse
import math
import os
import time

import numpy as np

import terrain as tr
from bridge import Bridge

HERE = os.path.dirname(os.path.abspath(__file__))
TERRAIN_DIR = os.path.join(HERE, "terrain_out")

ALT_AGL = 30.0        # flight height above ground (m)
SPEED = 12.0          # m/s
DETECT_RANGE = 25.0   # m
GRID = 32             # probability grid is GRID x GRID over the terrain
STATE_HZ = 20
PROB_EVERY = 10       # send the probability grid every N state ticks


class PyBulletWorld:
    """Terrain heightfield + drone body + person body inside PyBullet (DIRECT)."""

    def __init__(self, heights, meta, person_xy):
        import pybullet as p
        self.p = p
        self.heights, self.meta = heights, meta
        self.cid = p.connect(p.DIRECT)
        p.setGravity(0, 0, 0, physicsClientId=self.cid)
        n, cell = meta["n"], meta["cell_m"]
        # NOTE: PyBullet centers a heightfield on (0,0) and on the midpoint of its
        # min/max height. We shift the body so world (0,0) is the SW corner.
        shape = p.createCollisionShape(
            p.GEOM_HEIGHTFIELD, meshScale=[cell, cell, 1.0],
            heightfieldData=heights.flatten().tolist(),
            numHeightfieldRows=n, numHeightfieldColumns=n, physicsClientId=self.cid)
        half = meta["size_m"] / 2.0
        mid = (meta["h_min"] + meta["h_max"]) / 2.0
        self.ground = p.createMultiBody(0, shape, basePosition=[half, half, mid],
                                        physicsClientId=self.cid)
        dshape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.5, physicsClientId=self.cid)
        self.drone = p.createMultiBody(1.0, dshape, basePosition=[0, 0, 50],
                                       physicsClientId=self.cid)
        z = tr.height_at(heights, meta, *person_xy) + 0.9
        pshape = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.3, height=1.2,
                                        physicsClientId=self.cid)
        self.person = p.createMultiBody(0, pshape, basePosition=[*person_xy, z],
                                        physicsClientId=self.cid)

    def ground_height(self, x, y):
        r = self.p.rayTest([x, y, 1e4], [x, y, -1e4], physicsClientId=self.cid)[0]
        return r[3][2] if r[0] == self.ground else tr.height_at(self.heights, self.meta, x, y)

    def set_drone(self, pos):
        self.p.resetBasePositionAndOrientation(self.drone, pos, [0, 0, 0, 1],
                                               physicsClientId=self.cid)

    def can_see_person(self, pos):
        """Ray-cast line of sight from drone to person; terrain can occlude."""
        p = self.p
        target = p.getBasePositionAndOrientation(self.person, physicsClientId=self.cid)[0]
        if math.dist(pos, target) > DETECT_RANGE:
            return False, target
        hit = p.rayTest(pos, target, physicsClientId=self.cid)[0]
        return hit[0] == self.person, target


class FakeWorld:
    """No PyBullet: same interface, distance-only detection."""

    def __init__(self, heights, meta, person_xy):
        self.heights, self.meta = heights, meta
        self.person = [*person_xy, tr.height_at(heights, meta, *person_xy) + 0.9]

    def ground_height(self, x, y):
        return tr.height_at(self.heights, self.meta, x, y)

    def set_drone(self, pos):
        pass

    def can_see_person(self, pos):
        return math.dist(pos, self.person) <= DETECT_RANGE, self.person


def lawnmower(size, margin=40.0, lane=40.0):
    pts, y, flip = [], margin, False
    while y <= size - margin:
        xs = (size - margin, margin) if flip else (margin, size - margin)
        pts += [(xs[0], y), (xs[1], y)]
        y += lane
        flip = not flip
    return pts


def new_prob(lkp, size, sigma=90.0):
    c = (np.arange(GRID) + 0.5) * size / GRID
    X, Y = np.meshgrid(c, c)
    g = np.exp(-((X - lkp[0]) ** 2 + (Y - lkp[1]) ** 2) / (2 * sigma ** 2))
    return (g / g.sum()).astype(np.float32)


def run(args):
    if os.path.exists(os.path.join(TERRAIN_DIR, "terrain.raw")):
        heights, meta = tr.load(TERRAIN_DIR)
    else:  # HOOK: replace with the real DTM -> raster export
        heights, size = tr.make_synthetic()
        meta = tr.save(TERRAIN_DIR, heights, size)
    size = meta["size_m"]

    person_xy = (size * 0.7, size * 0.55)
    World = FakeWorld if args.no_pybullet else PyBulletWorld
    world = World(heights, meta, person_xy)

    bridge = Bridge(port=args.port)
    bridge.set_hello({"terrain": meta, "state_hz": STATE_HZ, "grid": GRID})
    bridge.start()
    print(f"bridge listening on 127.0.0.1:{args.port} "
          f"({'fake' if args.no_pybullet else 'pybullet'} world)")

    pos = [40.0, 40.0, 0.0]
    pos[2] = world.ground_height(pos[0], pos[1]) + ALT_AGL
    waypoints = lawnmower(size)  # HOOK: Navigator / Mission plan goes here
    lkp = (size / 2, size / 2)
    prob = new_prob(lkp, size)
    paused, found, tick = False, False, 0
    dt = 1.0 / STATE_HZ

    def send_path():
        flat = [c for wp in waypoints for c in
                (wp[0], wp[1], world.ground_height(wp[0], wp[1]) + ALT_AGL)]
        bridge.send({"type": "path", "pts": flat})

    t0 = time.time()
    next_t = t0
    try:
        while args.duration <= 0 or time.time() - t0 < args.duration:
            for cmd in bridge.poll():  # operator guidance from Unity
                c = cmd.get("cmd")
                if c == "goto":
                    waypoints = [tuple(cmd["pos"][:2])]
                    send_path()
                elif c == "set_lkp":
                    lkp = tuple(cmd["pos"][:2])
                    prob = new_prob(lkp, size)
                    waypoints = lawnmower(size)
                    send_path()
                elif c == "pause":
                    paused = True
                elif c == "resume":
                    paused = False

            if not paused and waypoints:
                tx, ty = waypoints[0]
                dx, dy = tx - pos[0], ty - pos[1]
                d = math.hypot(dx, dy)
                step = SPEED * dt
                if d <= step:
                    pos[0], pos[1] = tx, ty
                    waypoints.pop(0)
                else:
                    pos[0] += dx / d * step
                    pos[1] += dy / d * step
                pos[2] = world.ground_height(pos[0], pos[1]) + ALT_AGL
                world.set_drone(pos)
                # HOOK: SearchMonitor update; here we just discount visited cells
                gi = min(max(int(pos[1] / size * GRID), 0), GRID - 1)
                gj = min(max(int(pos[0] / size * GRID), 0), GRID - 1)
                prob[max(gi - 1, 0):gi + 2, max(gj - 1, 0):gj + 2] *= 0.98

            seen, ppos = world.can_see_person(pos)
            msgs = [{"type": "state", "t": round(time.time() - t0, 3), "paused": paused,
                     "pos": [round(v, 3) for v in pos], "quat": [0, 0, 0, 1],
                     "remaining": len(waypoints)}]
            if tick == 0:
                send_path()
            if tick % PROB_EVERY == 0:
                s = float(prob.sum())
                msgs.append({"type": "prob", "n": GRID,
                             "data": [round(float(v) / (s or 1.0), 6) for v in prob.flatten()]})
            if seen and not found:
                found = True
                msgs.append({"type": "detection", "pos": [round(float(v), 3) for v in ppos],
                             "conf": 0.9})
            for m in msgs:
                bridge.send(m)

            tick += 1
            next_t += dt
            time.sleep(max(0.0, next_t - time.time()))
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pybullet", action="store_true")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--duration", type=float, default=0, help="seconds, 0 = run forever")
    run(ap.parse_args())
