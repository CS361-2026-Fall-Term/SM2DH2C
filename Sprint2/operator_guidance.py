"""
Contributors: Helena Bianco (programmer) and Diya Deepu (spike investigator and program support)
              PBI-09 (live commands for every zone + safety rules): Sofia Monge
File Description: Creates preset operator commands and track the drone's mission state.
It holds the search plan (i.e., zones still needed to be searched in a specific order),
and provides feedback from the drone to the operator.
Note: This file only decides where the drone goes next, not how to get there (which is
done with autonomous navigation and obstacle avoidance).

PBI-09: every command now works for EVERY zone. The operator selects a zone and applies a
command to it:
    Investigate  - reported sighting: go to the zone now (and raise its probability)
    Search first - move the zone to the front of the plan
    Do not search - remove the zone from the plan (it is remembered as excluded)
    Re-include   - put an excluded zone back into the plan (searched last)
    Return to original plan - original order, minus visited zones; clears exclusions
Each command is checked against safety rules before anything changes. handle() never raises:
a bad command (wrong zone, wrong state, wrong type, ...) is rejected with a reason, and the
time spent validating is measured (target: well under 10 ms).
"""

import time
from dataclasses import dataclass
from enum import Enum

# The center of each search zone, one zone per quadrant (Cartesian plane).
# Designed for the 26 m map; scale_zones() stretches them for other map sizes.
# The rest of this file works for any number of zones: add an entry here and to ORIGINAL_PLAN.
ZONES = {"A": (6, 6), "B": (-6, 6), "C": (-6, -6), "D": (6, -6)}


def scale_zones(k):
    """Stretch the zone centres by factor k (map size / 26 m). Updates ZONES in place so every
    module that imported it sees the new centres. Call once at startup, before the mission starts."""
    for name, (x, y) in ZONES.items():
        ZONES[name] = (round(x * k, 2), round(y * k, 2))


# Default order of drone navigation, prior to operator guidance.
ORIGINAL_PLAN = ["A", "B", "C", "D"]


# Class to hold operator commands.
class Cmd(Enum):
    """
    Preset operator commands. The value is the text shown on the button in the operator
    dashboard (PyBullet's built-in GUI). Every command except RETURN_PLAN applies to the zone the
    operator has selected.
    """
    INVESTIGATE = "Investigate selected zone (sighting)"
    SEARCH_FIRST = "Search selected zone first"
    AVOID = "Do not search selected zone"
    REINCLUDE = "Re-include selected zone"
    RETURN_PLAN = "Return to original plan"


ZONE_COMMANDS = (Cmd.INVESTIGATE, Cmd.SEARCH_FIRST, Cmd.AVOID, Cmd.REINCLUDE)


@dataclass
class CommandResult:
    """What happened to one operator command (returned by Mission.handle)."""
    ok: bool                     # True = accepted and applied, False = rejected (nothing changed)
    command: str                 # command name, e.g. "AVOID"
    zone: object                 # zone letter it applied to (None for RETURN_PLAN / bad input)
    message: str                 # text shown to the operator
    validate_ms: float           # time spent checking the command against the safety rules
    reroute: bool = False        # the drone's destination changed -> navigation must re-plan
    sighting: object = None      # (x, y) of a reported sighting (Investigate), else None

    def __str__(self):
        return self.message


# Class to track mission state of drone.
class Mission:
    """
    __init__: Initializes the class attributes.
    _set_dest: Set the current destination of the drone.
    _reply: Sends feedback from drone to operator.
    start: Starts the mission in simulation.
    found: Ends mission if target found.
    arrived: Marks zone as visited.
    handle: Validate, then apply, an operator command for a zone.
    """
    def __init__(self):
        self.state = "IDLE"
        self.original_plan = list(ORIGINAL_PLAN)   # order "Return to plan" goes back to
        self.plan = list(ORIGINAL_PLAN) # Zones still needed to search, commands can override this.
        self.visited = []               # Zones where drone finished searching.
        self.excluded = []              # Zones the operator removed with "Do not search".
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
        self.history.append((round(time.time(), 2), cmd, msg)) # Message is also logged
        return msg

    def plan_text(self):
        """Readable plan for the operator, e.g. 'B -> C -> D' (or 'none')."""
        return " -> ".join(self.plan) if self.plan else "none"

    # Before START: search the zones in this order (e.g. most likely first, from the probability
    # map). It also becomes the plan that "Return to plan" restores.
    def set_plan_order(self, order):
        if self.state != "IDLE":
            return self._reply("ORDER", f"Rejected: mission is {self.state}")
        order = list(order)
        if sorted(order) != sorted(ORIGINAL_PLAN):
            return self._reply("ORDER", f"Rejected: order must list every zone once ({order})")
        self.original_plan = order
        self.plan = list(order)
        return self._reply("ORDER", f"Zones ordered by probability: {self.plan_text()}")

    # Begin the mission, starts the simulation (only starts once, else error message).
    def start(self):
        if self.state != "IDLE":
            return self._reply("START", "Rejected: mission already started")
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
            self.visited.append(self.plan.pop(0))
        self._set_dest()

    # Called by main.py in the map-guided phase (after the zone plan): the drone is heading to the
    # most likely cell of the probability map. The mission stays SEARCHING.
    def search_cell(self, xy):
        if self.state != "SEARCHING":
            return self._reply("MAP", f"Rejected: mission is {self.state}")
        return self._reply("MAP", f"Searching most likely cell ({xy[0]:.1f}, {xy[1]:.1f})")

    # Called by main.py when the time limit is reached without a find.
    def time_up(self, best_xy):
        if self.state != "SEARCHING":
            return self.response
        self.state = "COMPLETE"
        self.destination = None
        where = f"most likely remaining cell ({best_xy[0]:.1f}, {best_xy[1]:.1f})" if best_xy else \
            "no cell left on the map"
        return self._reply("TIME", f"Time limit reached - person not found; {where}")

    # ------------------------------------------------------------------ operator commands
    @staticmethod
    def _parse(cmd, zone):
        """Turn raw input into (Cmd, zone letter or None). Raises ValueError with a readable reason."""
        if isinstance(cmd, str):
            try:
                cmd = Cmd[cmd.strip().upper()]
            except KeyError:
                raise ValueError(f"unknown command '{cmd}'")
        if not isinstance(cmd, Cmd):
            raise ValueError(f"unknown command {cmd!r}")
        if cmd not in ZONE_COMMANDS:
            return cmd, None
        if zone is None or (isinstance(zone, str) and not zone.strip()):
            raise ValueError("select a zone first")
        if not isinstance(zone, str):
            raise ValueError(f"zone must be a letter ({', '.join(ZONES)}), got {zone!r}")
        z = zone.strip().upper()
        if z not in ZONES:
            raise ValueError(f"unknown zone '{zone}' (zones are {', '.join(ZONES)})")
        return cmd, z

    def _check(self, cmd, z):
        """Safety rules. Returns None if the command may run, else the reason it is rejected."""
        if self.state != "SEARCHING":
            return f"mission is {self.state}" + (" - press START first" if self.state == "IDLE" else "")
        if cmd is Cmd.RETURN_PLAN:
            if not self.excluded and self.plan == [x for x in self.original_plan if x not in self.visited]:
                return "already following the original plan"
            return None
        if cmd in (Cmd.INVESTIGATE, Cmd.SEARCH_FIRST, Cmd.AVOID) and z in self.visited:
            return f"Zone {z} has already been searched"
        if cmd in (Cmd.INVESTIGATE, Cmd.SEARCH_FIRST) and z in self.excluded:
            return f"Zone {z} is excluded - re-include it first"
        if cmd is Cmd.INVESTIGATE and self.plan[:1] == [z]:
            return f"already heading to Zone {z}"
        if cmd is Cmd.SEARCH_FIRST and self.plan[:1] == [z]:
            return f"Zone {z} is already first in the plan"
        if cmd is Cmd.AVOID:
            if z in self.excluded:
                return f"Zone {z} is already excluded"
            if self.plan == [z]:
                return f"Zone {z} is the last zone left - excluding it would end the search"
        if cmd is Cmd.REINCLUDE and z not in self.excluded:
            return f"Zone {z} is not excluded"
        return None

    def _apply(self, cmd, z):
        """Change the plan for an accepted command. Returns (message, sighting point or None)."""
        if cmd is Cmd.INVESTIGATE:
            # Reported sighting: go to the zone now.
            self.plan = [z] + [x for x in self.plan if x != z]
            return f"Investigating sighting in Zone {z} now", ZONES[z]
        if cmd is Cmd.SEARCH_FIRST:
            # Prioritize the zone and move it to the front of self.plan, the rest is unchanged.
            self.plan = [z] + [x for x in self.plan if x != z]
            return f"Zone {z} moved to the front of the plan", None
        if cmd is Cmd.AVOID:
            # Exclude the zone from the plan; it is not marked as visited.
            self.plan = [x for x in self.plan if x != z]
            self.excluded.append(z)
            return f"Zone {z} removed from the plan", None
        if cmd is Cmd.REINCLUDE:
            self.excluded.remove(z)
            self.plan.append(z)
            return f"Zone {z} re-included (searched last)", None
        # RETURN_PLAN: rebuild the original order, skipping zones already visited.
        # Undoes earlier reordering and exclusion.
        self.plan = [x for x in self.original_plan if x not in self.visited]
        self.excluded = []
        return "Resuming original plan", None

    # Apply operator command by changing the current plan, and return the result (accepted or
    # rejected with a reason). Never raises: any bad input is rejected instead.
    def handle(self, cmd, zone=None):
        t0 = time.perf_counter()
        name = getattr(cmd, "name", str(cmd))
        z = None
        try:
            cmd, z = self._parse(cmd, zone)
            name = cmd.name
            reason = self._check(cmd, z)
        except ValueError as exc:
            reason = str(exc)
        except Exception as exc:                     # last line of defence: never crash the sim
            reason = f"invalid command ({type(exc).__name__})"
        validate_ms = (time.perf_counter() - t0) * 1000.0
        label = f"{name} {z}" if z else name
        if reason:
            msg = f"Rejected: {reason}"
            self._reply(label, msg)
            return CommandResult(False, name, z, msg, validate_ms)
        old_dest = self.destination
        text, sighting = self._apply(cmd, z)
        self._set_dest()
        msg = f"Received: {text}. Plan: {self.plan_text()}"
        self._reply(label, msg)
        return CommandResult(True, name, z, msg, validate_ms,
                             reroute=self.destination != old_dest, sighting=sighting)


# Tests mission logic without the simulator. Only tests this file.
# (The full test suite is test_operator_guidance.py.)
def selftest():
    m = Mission()
    assert not m.handle(Cmd.SEARCH_FIRST, "B").ok              # mission not started
    m.start()
    assert m.destination == ZONES["A"]
    assert m.handle(Cmd.SEARCH_FIRST, "B").ok and m.plan[0] == "B"
    assert m.handle(Cmd.AVOID, "C").ok and "C" not in m.plan
    assert not m.handle(Cmd.AVOID, "C").ok                     # already excluded
    assert not m.handle(Cmd.INVESTIGATE, "C").ok               # excluded: re-include first
    assert m.handle(Cmd.REINCLUDE, "C").ok and m.plan[-1] == "C"
    r = m.handle(Cmd.INVESTIGATE, "D")
    assert r.ok and m.plan[0] == "D" and r.reroute and r.sighting == ZONES["D"]
    m.arrived()
    assert not m.handle(Cmd.SEARCH_FIRST, "D").ok              # already searched
    assert m.handle(Cmd.RETURN_PLAN).ok and m.plan == ["A", "B", "C"]
    assert not m.handle(Cmd.AVOID, "Z").ok                     # unknown zone
    assert "Rejected" in m.start()
    m.found()
    assert not m.handle(Cmd.AVOID, "A").ok                     # mission over
    print("selftest passed:", len(m.history), "events logged")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Run main.py for the integrated GUI; --selftest tests the mission backend.")
