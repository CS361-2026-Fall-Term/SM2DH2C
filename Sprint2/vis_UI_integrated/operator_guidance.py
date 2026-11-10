"""
Contributors: Helena Bianco (programmer) and Diya Deepu (spike investigator and program support)
File Description: Creates preset operator commands and track the drone's mission state.
It holds the search plan (i.e., zones still needed to be searched in a specific order),
and provides feedback from the drone to the operator.
Note: This file only decides where the drone goes next, not how to get there (which is
done with autonomous navigation and obstacle avoidance).
"""

import math
import time
from enum import Enum

# Every state a Mission can be in. IDLE -> SEARCHING -> FOUND / ABORT / COMPLETE.
STATES = ("IDLE", "SEARCHING", "FOUND", "ABORT", "COMPLETE")
# The history keeps every exchange; this cap only stops a runaway loop from growing it forever.
HISTORY_LIMIT = 2000
MAX_MESSAGE_LEN = 500

# The center of each search zone, one zone per quadrant (Cartesian plane). 
# Designed for the 26 m map; scale_zones() stretches them for other map sizes.
ZONES = {"A": (6, 6), "B": (-6, 6), "C": (-6, -6), "D": (6, -6)}


def scale_zones(k):
    """Stretch the zone centres by factor k (map size / 26 m). Updates ZONES in place so every
    module that imported it sees the new centres. Call once at startup, before the mission starts."""
    for name, (x, y) in ZONES.items():
        ZONES[name] = (round(x * k, 2), round(y * k, 2))
# Defualt order of drone navigation, prior to operator gudiance.
ORIGINAL_PLAN = ["A", "B", "C", "D"] 

# Class to hold operator commands.
class Cmd(Enum):
    """
    Initializes the preset operator commands to the drone. Each value of the intialized
    variable is what is shown on the buttons in the operator dashboard (created via
    PyBullet's built-in GUI).
    """
    INVESTIGATE_D = "Investigate sighting (Zone D)"
    SEARCH_B_FIRST = "Search Zone B first"
    AVOID_C = "Do not search Zone C"
    RETURN_PLAN = "Return to original plan"

# Class to track mission state of drone.
class Mission:
    """
    __init__: Initializes the class attributes.
    _set_dest: Set the current destination of the drone.
    _reply: Sends feedback from drone to operator.
    start: Starts the mission in simulation.
    found: Ends mission if target found.
    arrived: Marks zone as visited.
    handle: Apply preset operator commands.
    finish: Ends the mission with a state and a logged message (used by main.py).
    snapshot / restore: Save and reload the mission state (see mission_state.py, PBI-10).
    """
    def __init__(self):
        self.state = "IDLE"
        self.plan = list(ORIGINAL_PLAN) # Zones still needed to search, commands can override this.
        self.visited = []               # Zones where drone finished searching.
        self.destination = None         # (x, y) of self.plan[0] or None if nothing left to search.
        self.response = "Press START"   # Latest message to the operator.
        self.history = []               # (timestamp, command, reply) log of every exchange between operator and drone.

    # The destination is always set to the front of self.plan. Reordering the plan via operator
    # guidance means it changes the destination (i.e., where the drone goes next). If self.plan
    # is empty, the drone has nowhere left to go, therefore set self.destination to None.
    def _set_dest(self):
        self.destination = ZONES[self.plan[0]] if self.plan else None

    # Feedback from the drone to the operator, where the message is shown in the simulator UI.
    def _reply(self, cmd, msg):
        self.response = msg
        self.record(cmd, msg)                                  # Message is also logged
        return msg

    # Add one (timestamp, source, message) line to the history log without touching the HUD
    # message. Every operator command and every system response ends up here (PBI-10).
    def record(self, cmd, msg):
        self.history.append((round(time.time(), 2), str(cmd), str(msg)))
        if len(self.history) > HISTORY_LIMIT:
            del self.history[:len(self.history) - HISTORY_LIMIT]

    # End the mission from main.py (navigation aborted, plan complete, ...). Goes through the
    # same logging as everything else, so these system responses are in the history too.
    def finish(self, state, source, msg):
        if state not in STATES:
            raise ValueError(f"unknown mission state {state!r}")
        self.state = state
        return self._reply(source, msg)

    # Begin the mission, starts the simulation (only starts once, else error message).
    def start(self):
        if self.state == "SEARCHING":
            return self._reply("START", "Rejected: mission already started")
        if self.state != "IDLE":
            return self._reply("START", f"Rejected: mission is {self.state} (restart the program for a new mission)")
        self.state = "SEARCHING"
        self._set_dest()
        return self._reply("START", "Mission started")

    # Called by main.py when the person is found. Mission is ended.
    def found(self):
        if self.state != "SEARCHING":
            return self._reply("DETECT", f"Rejected: mission is {self.state}")
        self.state = "FOUND"
        self.destination = None
        return self._reply("DETECT", "Person confirmed - mission complete")

    # Called by main.py when drone has arrived to a certain zone.
    # Marks the zone at the front of the plan as visited, and moves on to the next zone in plan.
    def arrived(self):
        if self.plan:
            zone = self.plan.pop(0)
            self.visited.append(zone)
            self._reply("ARRIVED", f"Zone {zone} searched")
        self._set_dest()

    # Apply operator command by changing the current plan, and return reply message from drone.
    # Commands only accepted while drone is in "SEARCHING" state. If plan is changed (self.plan
    # is reordered or trimmed), drone re-routes.  
    def handle(self, cmd):
        # Anything that is not a preset command is rejected with a message (never an exception).
        name = cmd.name if isinstance(cmd, Cmd) else f"INVALID({str(cmd)[:40]})"
        # Check whether drone state is "SEARCHING" (before START or after the mission ended,
        # commands are rejected and logged).
        if self.state != "SEARCHING":
            return self._reply(name, f"Rejected: mission is {self.state}")
        if not isinstance(cmd, Cmd):
            return self._reply(name, "Rejected: unknown command")
        
        # Reported sighting, jump to Zone D, unless already searched.
        if cmd is Cmd.INVESTIGATE_D:
            if "D" in self.visited:
                return self._reply(cmd.name, "Rejected: Zone D already visited")
            self.plan = ["D"] + [z for z in self.plan if z != "D"]
            msg = "Received: investigating Zone D now"

        # Prioritize Zone B and move it to the front of self.plan, the rest is unchanged.
        elif cmd is Cmd.SEARCH_B_FIRST:
            if "B" in self.visited:
                return self._reply(cmd.name, "Rejected: Zone B already visited")
            self.plan = ["B"] + [z for z in self.plan if z != "B"]
            msg = "Received: Zone B moved to front of plan"

        # Exclude Zone C from the plan, not marked as visited.
        elif cmd is Cmd.AVOID_C:
            self.plan = [z for z in self.plan if z != "C"]
            msg = "Received: Zone C removed from plan"

        # Reset and build the original order, but skip the zones already visited.
        # Undoes earlier reordering and exclusion.
        elif cmd is Cmd.RETURN_PLAN:
            self.plan = [z for z in ORIGINAL_PLAN if z not in self.visited]
            msg = "Received: resuming original plan"
        else:
            return self._reply(str(cmd), "Rejected: unknown command")
        self._set_dest()
        return self._reply(cmd.name, msg)

    # ---- Saving and restoring (PBI-10) -------------------------------------------------
    def snapshot(self):
        """Plain-data copy of everything that makes up the mission state (JSON friendly)."""
        return {
            "state": self.state,
            "plan": list(self.plan),
            "visited": list(self.visited),
            "destination": list(self.destination) if self.destination else None,
            "response": self.response,
            "history": [list(h) for h in self.history],
        }

    @staticmethod
    def validate_snapshot(d):
        """Raise ValueError unless d is a well-formed snapshot (e.g. read from a damaged file)."""
        def bad(why):
            raise ValueError(f"invalid mission snapshot: {why}")
        if not isinstance(d, dict):
            bad("not an object")
        if d.get("state") not in STATES:
            bad("unknown state")
        for key in ("plan", "visited"):
            zones = d.get(key)
            if not isinstance(zones, list) or any(not isinstance(z,str) or z not in ZONES for z in zones):
                bad(f"{key} must be a list of known zones")
        if len(set(d["plan"]) | set(d["visited"])) != len(d["plan"]) + len(d["visited"]):
            bad("a zone appears twice in plan/visited")
        dest = d.get("destination")
        if dest is not None and not (isinstance(dest, list) and len(dest) == 2 and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in dest)):
            bad("destination must be null or [x, y]")
        if not isinstance(d.get("response"), str) or len(d["response"]) > MAX_MESSAGE_LEN:
            bad("response must be a short string")
        hist = d.get("history")
        if not isinstance(hist, list) or len(hist) > HISTORY_LIMIT:
            bad("history must be a list")
        for h in hist:
            if not (isinstance(h, list) and len(h) == 3
                    and isinstance(h[0], (int, float)) and not isinstance(h[0], bool)
                    and math.isfinite(h[0])
                    and isinstance(h[1], str) and isinstance(h[2], str)
                    and len(h[1]) <= MAX_MESSAGE_LEN and len(h[2]) <= MAX_MESSAGE_LEN):
                bad("history entries must be [time, source, message]")

    def restore(self, d):
        """Load a snapshot written by snapshot(). Raises ValueError (and changes nothing) if it
        is invalid. The destination is recomputed from the plan and the CURRENT zone centres;
        returns a note when that differs from the saved one (e.g. the map changed), else ''."""
        self.validate_snapshot(d)
        self.state = d["state"]
        self.plan = list(d["plan"])
        self.visited = list(d["visited"])
        self.response = d["response"]
        self.history = [(float(t), s, m) for t, s, m in d["history"]]
        self._set_dest()
        if self.state != "SEARCHING":
            self.destination = None
        note = ""
        saved = tuple(d["destination"]) if d["destination"] else None
        if self.state == "SEARCHING" and saved != self.destination:
            note = f"saved destination {saved} replaced by {self.destination} (zone centres changed)"
        self.record("RESTORE", f"Mission restored from saved state ({self.state}, "
                               f"{len(self.visited)} zone(s) searched)")
        return note

# Tests mission logic without the simulator. Only tests this file.
def selftest():
    m = Mission()
    assert "Rejected" in m.handle(Cmd.SEARCH_B_FIRST)
    m.start()
    assert m.destination == ZONES["A"]
    m.handle(Cmd.SEARCH_B_FIRST)
    assert m.plan[0] == "B"
    m.handle(Cmd.AVOID_C)
    assert "C" not in m.plan
    m.handle(Cmd.INVESTIGATE_D)
    assert m.plan[0] == "D"
    m.arrived()
    m.handle(Cmd.RETURN_PLAN)
    assert m.plan == ["A", "B", "C"]
    assert "Rejected" in m.start()
    m.found()
    assert "Rejected" in m.handle(Cmd.AVOID_C)
    assert "Rejected" in m.handle("not a command")   # odd input is rejected, never a crash
    assert "FOUND" in m.start()                      # after the end: says why it was rejected
    print("selftest passed:", len(m.history), "events logged")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Run main.py for the integrated GUI; --selftest tests the mission backend.")
