# PBI-13: detection results with the confidence level

## Where the confidence is calculated
`target_detection.py`, `SearchMonitor._process_frame()`, once per camera scan (10 per second):
1. the 7 sample points on the person that are inside the camera image and not hidden are counted (`seen`);
2. `quality` = how much a scan counts at that distance (`range_quality()`: 1.00 up to 5 m, 0.30 at 8 m);
3. `score` = seen / 7 x quality (0 if fewer than 20 % are seen);
4. `confidence` += score x 0.1 s; kept for 2 s after the last sighting, then -0.3 per second;
5. 0.25 = possible sighting, 0.08 = sighting dropped, 1.00 = FOUND; a clear view within 6 m for
   3 scans in a row is FOUND at once.

## How the results reach the user
In the terminal (`python main.py`, also with the window), controlled by `--detection-log`:

| Mode | What is printed |
|---|---|
| `scans` (default) | the explanation below once at the start; one line per scan while the person is in view or the confidence is above 0; one `person not in the camera view (N scans)` line per 5 s otherwise; SIGHTING / LOST / FOUND events; the detection summary at the end |
| `events` | only the SIGHTING / LOST / FOUND events and the `Detection: {...}` line |
| `off` | only the `Detection: {...}` line (kept for scripts such as `map_search_runs.py`) |

Operator control panel: the SIGHTING / LOST / FOUND events also go to the mission history, so they
appear in the panel's message log (and are saved with the mission state, PBI-10), e.g.
`FOUND (t = 27.3 s): Casualty confirmed (confidence, 6.9 m away). Estimated location (6.35, -5.73)
from 19 measurements`, followed by `Person confirmed - mission complete`.

How often: every scan (10 per second) while there is something to report, because that is when the
confidence changes; while the person is out of view nothing changes, so one line per 5 s is
enough and the terminal is not flooded (a 100 s mission with the person in view for 2 s prints
about 40 lines, not 1000). The column header is repeated every 25 lines.

### Explanation printed at the start
```
How the detection confidence works
----------------------------------
  The camera (70 x 70 deg, tilted 40 deg down, up to 8 m) takes 10 scans per second.
  Each scan checks 7 points on the person (centre, head, feet, 4 sides):
    seen     = points inside the image with nothing else in the way (e.g. 5/7)
    quality  = how much a scan counts at that distance: 1.00 up to 5 m,
               falling to 0.30 at 8 m
    score    = (seen points / 7) x quality, or 0 if fewer than 20% are seen
  confidence goes up by score x 1 per second while the person is seen. It is kept
  for 2 s after losing sight, then falls by 0.3 per second.
    >= 0.25  possible sighting (SIGHTING, the drone stops to look)
    <= 0.08  sighting dropped (LOST)
    >= 1.00  person FOUND
  Clear view: within 6 m, at least 4 points in the image and 75% of them unblocked,
  for 3 scans in a row (clear 3/3): FOUND straight away (confidence set to 1.00).
  Columns: t = sim time, conf = confidence after the scan, change = how much this scan
  added or removed, clear = clear-view scans in a row.
Zone order by probability: A (24%) -> B (24%) -> C (24%) -> D (24%)
[t=  0.00s] REPLAN   detour found: 1 legs, ~16.5 m to waypoint 1 at alt 2.5 m
[t=  0.63s] DETECT   tree 4 (conifer) (3.0 m tall) sensed 5.0 m away, centre (-7.5, 5.0)
[t=  3.73s] DETECT   tree 1 (deciduous) (3.0 m tall) sensed 5.0 m away, centre (1.5, 6.5)
[t=  4.10s] BLOCKED  planned route crosses a detected obstacle
[t=  4.10s] REPLAN   detour found: 2 legs, ~9.8 m to waypoint 1 at alt 2.5 m
[t=  4.63s] BLOCKED  planned route crosses a detected obstacle
[t=  4.63s] REPLAN   detour found: 2 legs, ~8.9 m to waypoint 1 at alt 2.5 m
```

### Example (headless mission, person seed 1)
```
[detect]   15.30-20.30 s  SEARCHING  conf 0.00  person not in the camera view (51 scans)
[detect]   20.40-25.30 s  SEARCHING  conf 0.00  person not in the camera view (50 scans)
[detect]   t (s)  state       conf  change  seen    dist  quality  score  clear  note
[detect]   25.40  SEARCHING   0.00   +0.00   1/7   8.0 m     0.30   0.00  0/3    too little visible (1/7, needs 2)
[detect]   25.50  SEARCHING   0.03   +0.03   7/7   7.8 m     0.34   0.34  0/3    seen
[detect]   25.60  SEARCHING   0.07   +0.04   7/7   7.6 m     0.38   0.38  0/3    seen
[detect]   25.70  SEARCHING   0.12   +0.04   7/7   7.4 m     0.43   0.43  0/3    seen
[detect]   25.80  SEARCHING   0.16   +0.05   7/7   7.2 m     0.48   0.48  0/3    seen
[detect]   25.90  SEARCHING   0.22   +0.05   7/7   7.1 m     0.52   0.52  0/3    seen
[t= 26.00s] SIGHTING  possible person, 6.9 m away (confidence 0.27 >= 0.25)
[detect]   26.00  SIGHTING    0.27   +0.06   7/7   6.9 m     0.57   0.57  0/3    seen
...
[detect]   27.20  SIGHTING    0.95   +0.06   7/7   6.9 m     0.57   0.57  0/3    seen
[t= 27.30s] FOUND     Casualty confirmed (confidence, 6.9 m away). Estimated location (6.35, -5.73) from 19 measurements
[detect]   27.30  FOUND       1.01   +0.06   7/7   6.9 m     0.57   0.57  0/3    seen
```

### Summary at the end
```
Detection summary
-----------------
  Result:           FOUND (final state FOUND, confidence 1.01 of 1.00)
  Found at:         t = 27.3 s, by confidence, 6.9 m from the person
  First sighting:   t = 26.0 s
  Sightings:        1 raised, 0 lost again
  Highest conf.:    1.01
  Camera scans:     274 total, person in view in 20, seen in 19, hidden in 1
  Location:         estimated (6.35, -5.73), true (6.33, -5.88), error 0.15 m
```

## Tests (`tests/test_detection_results.py`, 17 tests)

| AC | Tests |
|---|---|
| AC-01 confidence calculated from the detection results and returned | every scan follows the rule above exactly (score, rise, memory, decay, clear view); every printed line shows that scan's confidence; the events and the summary report it |
| AC-01 (panel) | the FOUND line with the estimated location reaches the panel's message log; `main.py` saves it in the history before "mission complete" |
| AC-02 readable, method understandable | the explanation lists every threshold from the real settings; columns line up with the header; "-" instead of numbers when the person is not in view; the summary is in plain words (also when never seen) |
| AC-03 results change every scan, readable structure | one record per scan, 0.1 s apart; every active scan printed, quiet stretches folded into one line per 5 s; values change scan to scan; header repeats; `events` / `off` modes; `main.py` order: explanation, scans, mission result, summary |

Run from the Sprint folder: `python -m unittest tests.test_detection_results -v`.
All tests pass, and the earlier tests and the 9 PBI-06 scenarios still pass (the machine-readable
`Detection: {...}` line they read is unchanged). The example above was produced with a stand-in for
PyBullet; the numbers on a machine with PyBullet will differ slightly, the format is the same.
