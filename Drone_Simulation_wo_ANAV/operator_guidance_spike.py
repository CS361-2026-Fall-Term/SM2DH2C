"""Operator mission state and preset commands, reusable by main.py."""
import time
from enum import Enum

ZONES = {"A": (6, 6), "B": (-6, 6), "C": (-6, -6), "D": (6, -6)}
ORIGINAL_PLAN = ["A", "B", "C", "D"]
ARRIVE_DIST = 0.5

class Cmd(Enum):
    INVESTIGATE_D = "Investigate sighting (Zone D)"
    SEARCH_B_FIRST = "Search Zone B first"
    AVOID_C = "Do not search Zone C"
    RETURN_PLAN = "Return to original plan"

class Mission:
    def __init__(self):
        self.state = "IDLE"
        self.plan = list(ORIGINAL_PLAN)
        self.visited = []
        self.destination = None
        self.response = "Press START"
        self.history = []

    def _set_dest(self):
        self.destination = ZONES[self.plan[0]] if self.plan else None

    def _reply(self, cmd, msg):
        self.response = msg
        self.history.append((round(time.time(), 2), cmd, msg))
        return msg

    def start(self):
        if self.state != "IDLE":
            return self._reply("START", "Rejected: mission already started")
        self.state = "SEARCHING"
        self._set_dest()
        return self._reply("START", "Mission started")

    def found(self):
        if self.state != "SEARCHING":
            return self._reply("DETECT", f"Rejected: mission is {self.state}")
        self.state = "FOUND"
        self.destination = None
        return self._reply("DETECT", "Person confirmed - mission complete")

    def arrived(self):
        if self.plan:
            self.visited.append(self.plan.pop(0))
        self._set_dest()

    def handle(self, cmd):
        if self.state != "SEARCHING":
            return self._reply(cmd.name, f"Rejected: mission is {self.state}")
        if cmd is Cmd.INVESTIGATE_D:
            if "D" in self.visited:
                return self._reply(cmd.name, "Rejected: Zone D already visited")
            self.plan = ["D"] + [z for z in self.plan if z != "D"]
            msg = "Received: investigating Zone D now"
        elif cmd is Cmd.SEARCH_B_FIRST:
            if "B" in self.visited:
                return self._reply(cmd.name, "Rejected: Zone B already visited")
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
    print("selftest passed:", len(m.history), "events logged")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Run main.py for the integrated GUI; --selftest tests the mission backend.")
