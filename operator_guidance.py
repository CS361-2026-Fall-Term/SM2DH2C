"""
Contributors: Helena Bianco (programmer) and Diya Deepu (spike investigator and program support)
File Description: Creates preset operator commands and track the drone's mission state.
It holds the search plan (i.e., zones still needed to be searched in a specific order),
and provides feedback from the drone to the operator.
Note: This file only decides where the drone goes next, not how to get there (which is
done with autonomous navigation and obstacle avoidance).
"""

import time
from enum import Enum

# The center of each search zone, one zone per quadrant (Cartesian plane). 
ZONES = {"A": (6, 6), "B": (-6, 6), "C": (-6, -6), "D": (6, -6)}
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
        self.history.append((round(time.time(), 2), cmd, msg)) # Message is also logged
        return msg

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

    # Apply operator command by changing the current plan, and return reply message from drone.
    # Commands only accepted while drone is in "SEARCHING" state. If plan is changed (self.plan
    # is reordered or trimmed), drone re-routes.  
    def handle(self, cmd):
        # Check whether drone state is "SEARCHING".
        if self.state != "SEARCHING":
            return self._reply(cmd.name, f"Rejected: mission is {self.state}")
        
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
    print("selftest passed:", len(m.history), "events logged")

if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print("Run main.py for the integrated GUI; --selftest tests the mission backend.")
