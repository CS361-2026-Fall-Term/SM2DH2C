"""
PA2 Spike - Operator Guidance (Diya, Helena)

Questions this spike answers:
  Frontend: Can PyBullet's built-in GUI show a live dashboard and accept
            preset commands (buttons) while the sim keeps running?
  Backend:  Can a shared mission state object accept preset commands
            mid-mission, change the drone's plan, resume the original plan,
            reject illegal commands, and return a response?

Run:
  python operator_guidance_spike.py --selftest   # backend only, no PyBullet needed
  python operator_guidance_spike.py              # full GUI demo (pip install pybullet)

Notes:
  * The drone is moved kinematically (resetBasePositionAndOrientation) to keep
    the spike small, so "moving/stopped" is computed from position change,
    NOT from p.getBaseVelocity (which reads 0 for kinematic bodies).
  * PyBullet "buttons" are addUserDebugParameter(name, 1, 0, 1): rangeMin >
    rangeMax makes a button; its value goes up by 1 per click, so we detect
    a click by watching the value change.
"""
import math
import sys
import time
from enum import Enum

# ---------------------------------------------------------------- backend --

ZONES = {"A": (6, 6), "B": (-6, 6), "C": (-6, -6), "D": (6, -6)}
ORIGINAL_PLAN = ["A", "B", "C", "D"]
ARRIVE_DIST = 0.5
SIGHTING_DIST = 2.0


class Cmd(Enum):                      # preset instructions only
    INVESTIGATE_D = "Investigate sighting (Zone D)"
    SEARCH_B_FIRST = "Search Zone B first"
    AVOID_C = "Do not search Zone C"
    RETURN_PLAN = "Return to original plan"


class Mission:
    """Shared state: mission state, plan, destination, last response."""

    def __init__(self):
        self.state = "IDLE"           # IDLE, SEARCHING, FOUND
        self.plan = list(ORIGINAL_PLAN)
        self.visited = []
        self.destination = None
        self.response = "Press START"
        self.history = []             # (time, command, response) for evidence

    def _set_dest(self):
        self.destination = ZONES[self.plan[0]] if self.plan else None

    def start(self):
        if self.state != "IDLE":
            return self._reply("START", "Rejected: mission already started")
        self.state = "SEARCHING"
        self._set_dest()
        return self._reply("START", "Mission started")

    def found(self):
        self.state = "FOUND"
        self.destination = None
        return self._reply("DETECT", "Possible person detected - mission complete")

    def arrived(self):
        if self.plan:
            self.visited.append(self.plan.pop(0))
        self._set_dest()

    def handle(self, cmd):
        """Process one preset command; always returns a response string."""
        if self.state != "SEARCHING":
            return self._reply(cmd.name, f"Rejected: mission is {self.state}")
        if cmd is Cmd.INVESTIGATE_D:
            self.plan = ["D"] + [z for z in self.plan if z != "D"]
            msg = "Received: investigating Zone D now"
        elif cmd is Cmd.SEARCH_B_FIRST:
            self.plan = ["B"] + [z for z in self.plan if z != "B"]
            msg = "Received: Zone B moved to front of plan"
        elif cmd is Cmd.AVOID_C:
            self.plan = [z for z in self.plan if z != "C"]
            msg = "Received: Zone C removed from plan"
        elif cmd is Cmd.RETURN_PLAN:
            self.plan = [z for z in ORIGINAL_PLAN if z not in self.visited]
            msg = "Received: resuming original plan"
        else:
            return self._reply(str(cmd), "Rejected: unknown command")
        self._set_dest()
        return self._reply(cmd.name, msg)

    def _reply(self, cmd, msg):
        self.response = msg
        self.history.append((round(time.time(), 2), cmd, msg))
        return msg


def selftest():
    m = Mission()
    assert "Rejected" in m.handle(Cmd.SEARCH_B_FIRST)       # before start: blocked
    m.start()
    assert m.destination == ZONES["A"]
    m.handle(Cmd.SEARCH_B_FIRST)
    assert m.plan[0] == "B" and m.destination == ZONES["B"]
    m.handle(Cmd.AVOID_C)
    assert "C" not in m.plan
    m.handle(Cmd.INVESTIGATE_D)
    assert m.plan[0] == "D"
    m.arrived()                                              # reached D
    m.handle(Cmd.RETURN_PLAN)
    assert m.plan == ["A", "B", "C"]                         # original minus visited
    assert "Rejected" in m.start()                           # illegal transition
    m.found()
    assert "Rejected" in m.handle(Cmd.AVOID_C)               # guidance after FOUND
    print("selftest passed:", len(m.history), "events logged")
    for h in m.history:
        print(" ", h)


# --------------------------------------------------------------- frontend --

def run_gui():
    import pybullet as p
    import pybullet_data

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")
    p.resetDebugVisualizerCamera(22, 45, -50, [0, 0, 0])

    def ball(color, pos, r=0.4):
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=r, rgbaColor=color)
        return p.createMultiBody(0, -1, vis, pos)

    drone = ball([0, 0.4, 1, 1], [0, 0, 2])
    target = ball([1, 0, 0, 1], [ZONES["D"][0], ZONES["D"][1], 0.4])
    for name, (x, y) in ZONES.items():
        p.addUserDebugText(f"Zone {name}", [x, y, 0.1], [0, 0, 0], 1.5)

    # Buttons: rangeMin > rangeMax => button. Value +1 per click.
    start_btn = p.addUserDebugParameter("START MISSION", 1, 0, 0)
    btns = {c: p.addUserDebugParameter(c.value, 1, 0, 0) for c in Cmd}
    last = {b: p.readUserDebugParameter(b) for b in [start_btn, *btns.values()]}

    m = Mission()
    pos = [0.0, 0.0, 2.0]
    prev = list(pos)
    dash_id = resp_id = None
    last_dash = 0.0
    speed, dt = 3.0, 1 / 60

    while p.isConnected():
        # 1) poll buttons (this is the "operator -> system" path)
        if p.readUserDebugParameter(start_btn) != last[start_btn]:
            last[start_btn] = p.readUserDebugParameter(start_btn)
            m.start()
        for cmd, b in btns.items():
            v = p.readUserDebugParameter(b)
            if v != last[b]:
                last[b] = v
                m.handle(cmd)

        # 2) autonomous movement toward current destination
        if m.state == "SEARCHING" and m.destination:
            dx, dy = m.destination[0] - pos[0], m.destination[1] - pos[1]
            d = math.hypot(dx, dy)
            if d < ARRIVE_DIST:
                m.arrived()
            else:
                pos[0] += dx / d * speed * dt
                pos[1] += dy / d * speed * dt
            p.resetBasePositionAndOrientation(drone, pos, [0, 0, 0, 1])
            tpos = p.getBasePositionAndOrientation(target)[0]
            if math.dist(pos, tpos) < SIGHTING_DIST:
                m.found()

        # 3) dashboard (the "system -> operator" path), ~4 updates/sec
        now = time.time()
        if now - last_dash > 0.25:
            moving = math.dist(pos, prev) > 0.001
            prev = list(pos)
            last_dash = now
            dest = m.destination if m.destination else "none"
            text = (f"x/y/z: {pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}\n"
                    f"State: {m.state}\n"
                    f"Dest: {dest}\n"
                    f"{'MOVING' if moving else 'STOPPED'}\n"
                    f"Plan: {m.plan}\n"
                    f"Possible sighting: {'YES' if m.state == 'FOUND' else 'NO'}")
            kw = dict(textColorRGB=[1, 1, 0], textSize=1.4)
            dash_id = p.addUserDebugText(
                text, [-9, 9, 5],
                **({"replaceItemUniqueId": dash_id} if dash_id else {}), **kw)
            resp_id = p.addUserDebugText(
                "System: " + m.response, [-9, 9, 1],
                **({"replaceItemUniqueId": resp_id} if resp_id else {}),
                textColorRGB=[0, 1, 0], textSize=1.4)

        p.stepSimulation()
        time.sleep(dt)


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        run_gui()
