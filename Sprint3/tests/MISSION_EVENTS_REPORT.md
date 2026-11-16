# PBI-14: detection and obstacle events in the mission state and on the dashboard

## Shared event structure
`operator_guidance.MissionEvent` (one structure for both modules):

| Field | Meaning |
|---|---|
| kind | SIGHTING / LOST / FOUND (person detector); DETECT / SLOW / BLOCKED / UNSAFE / REPLAN / CLIMB / DESCEND / UNREACH / RTL / ABORT (navigator) |
| severity | INFO / WARNING / CRITICAL. Obstacles: CRITICAL at or inside the 0.35 m unsafe distance (and for RTL / ABORT), WARNING at or inside the 0.9 m stop distance (and for UNSAFE / BLOCKED / UNREACH), else INFO. Person: SIGHTING = WARNING, LOST / FOUND = INFO |
| source | "target" or "obstacle" |
| drone_position, object_position | (x, y, z); the person position is the detector's estimate |
| distance | metres: camera to person, or drone body to the obstacle surface (None if nothing sensed yet) |
| timestamp | sim time (s) |
| mission_state, detection_state | the mission's state after the event |
| response | what the drone does: stopping to look, continuing the search, mission complete; obstacle mapped / whole outline mapped, slowing down, stopping to re-plan, backing away and re-planning, re-planned route, climbing, descending to cruise, skipping waypoint, returning to launch, hovering, awaiting operator |
| message, object_name | the console text; e.g. `building "Test Hall"`, `Casualty` |

## How it flows
* `SearchMonitor` (target_detection.py) and `Navigator` (object_avoidance.py) both get the shared
  `Mission` (`mission=...`, set up in main.py) and send every event to `Mission.add_event()`. The
  distance is a field (`SearchMonitor.last_distance`, `Navigator.nearest_dist` are kept per scan).
* `Mission.add_event()` keeps the latest 200 events (saved and restored with the mission, PBI-10;
  damaged entries are rejected with ValueError), sets `detection_state` (SIGHTING -> SIGHTING,
  LOST -> SEARCHING, FOUND -> FOUND and the mission ends as FOUND) and writes the important events
  (person events, UNSAFE, CLIMB, UNREACH, RTL, ABORT) to the history, i.e. the panel's message log.
  The **mission** state stays SEARCHING during a sighting, so detection keeps running and operator
  commands still work (SIGHTING is a detection state, not a mission state).
* Operator panel, new "Detections & obstacles" box: the latest event (kind, severity, time, object,
  distance, drone response, detection state; coloured by severity) and the 6 most recent events,
  newest first. It is only redrawn when a new event arrives. The 3D overlay has a "Latest event"
  line. The console output is unchanged.
* Routine events (each obstacle first seen, re-plans, slowing down) only go to the event box, not
  the history: they come in bursts and would otherwise force a durable save each time.

## Self-test without the simulator
`python operator_guidance.py --selftest`:
```
selftest passed: 15 events logged
    1.2s  DETECT 2.66 m - obstacle mapped (tree 1)                -> Mission: SEARCHING | Detection: SEARCHING
    1.9s  SLOW 1.68 m - slowing down                              -> Mission: SEARCHING | Detection: SEARCHING
    2.4s  UNSAFE 0.21 m - backing away and re-planning (tree 1)   -> Mission: SEARCHING | Detection: SEARCHING
   26.0s  SIGHTING 6.90 m - stopping to look (Casualty)           -> Mission: SEARCHING | Detection: SIGHTING
   30.0s  LOST 7.40 m - continuing the search (Casualty)          -> Mission: SEARCHING | Detection: SEARCHING
   40.0s  SIGHTING 6.10 m - stopping to look (Casualty)           -> Mission: SEARCHING | Detection: SIGHTING
   41.3s  FOUND 5.90 m - mission complete, holding position (Casualty) -> Mission: FOUND | Detection: FOUND
event selftest passed: 7 events, 6 logged
```

## Tests (`tests/test_mission_events.py`, 22 tests)

| AC | Tests |
|---|---|
| Structure | all fields present; saved and restored with the mission; old save files load; damaged events rejected cleanly (ValueError, mission unchanged); list capped at 200; self-test runs without the simulator |
| AC-01 | SIGHTING / LOST / FOUND set the detection state with the numeric distance; a sighting does not stop the mission or block commands; FOUND is logged once, followed by "mission complete"; the real detector sends SIGHTING and FOUND with the camera-to-person distance |
| AC-02 | on the PBI-15 test map the navigator sends DETECT / BLOCKED / REPLAN / SLOW with distance and response; responses match what the drone does; DETECT names the obstacle; events reach the shared mission; severity follows the distance; important responses reach the panel message log |
| AC-03 | the panel box shows the latest event and the recent list (newest first, last 6); the panel only redraws on a new event; adding an event takes well under 0.1 ms (measured ~0.3 us); dashboard status text; a full headless mission still prints the console lines and saves the events with distances |

Run from the Sprint folder: `python -m unittest tests.test_mission_events -v`.
Result: all tests pass (325 in total), the 9 PBI-06 scenarios and the PBI-15 fixture flight still
pass. Run with a stand-in for PyBullet (headless); the panel itself was tested without a window
(its formatting and redraw logic), so open it once on a machine with a display to check the layout.
