# Sprint 2 Folder
This folder contains the system's source/test code implemented during Sprint 2 for the course project. 

**Sprint goal**: Focus on developing and integrating the probability map to our SAR simulation system, while also adjusting
and refining preset operator instructions for better operator guidance during the simulation.

**PBIs worked on**: PBI-03, PBI-04, PBI-06, PBI-09, PBI-10

## Obstacles from geographical input (detection and avoidance)
The drone's obstacle detection and avoidance now works on obstacles built from the operator's map
data (an `.osm` export, an Overpass `.json` file, or a download around the last known position), not
only on the hand-placed trees.

- `geo_obstacles.py` reads and checks the map data, turns its key objects (buildings, single trees,
  trees scattered in woods) into obstacle records, and builds each one directly in PyBullet at its
  real position (trees as trunk + canopy, buildings as their footprint extruded to their height).
  No files are written.
- `object_avoidance.py`: the range sensor names the object it hits (`DETECT building "Kerr Library"
  ...`), puts a mapped building's whole outline into the obstacle map on first sight, and reacts to
  anything closer than `UNSAFE_DIST` (0.35 m) with `UNSAFE`, backing away and re-planning.
- Malformed input never crashes the simulation: an unusable file or position stops it with an
  `ERROR:` message; a single bad map object is skipped, named in the console and counted on the
  operator panel, and everything else is still placed.

Check a map file and list the obstacles it gives, without starting the simulation:
```
python geo_obstacles.py map.osm --lat 44.6488 --lon -123.3123 --area 200 --list
```

Tests (run from `Sprint2/`):
```
python -m unittest tests.test_geo_obstacles -v     # PBI-15: 26 tests, AC-01..AC-03 (a few seconds)
python -m unittest discover -s tests -v            # every test in tests/ (PBI-06, PBI-10, PBI-15)
python geo_avoidance_runs.py                       # flights with ground-truth clearances -> geo_avoidance_results/
```
`tests/fixtures/geo_test_area.json` is a small map with buildings and trees at predefined positions on
the default flight path (re-generate with `python tests/fixtures/make_geo_test_area.py`).
Results and analysis: `GEO_OBSTACLES_TEST_REPORT.md`.
