# Test report: obstacle detection and avoidance on obstacles from geographical input

**Type:** Feature. **Author:** Chase Vanderlip. **Date:** 2026-11-14.

Goal: extend obstacle detection and avoidance from the hand-placed trees to objects generated from
the operator's geographical input, so the drone reacts to the obstacles of the real area.

## What changed

| File | Change |
|---|---|
| `geo_obstacles.py` (new) | Reads and validates the map data (`.osm` XML or Overpass `.json`). Turns key objects (buildings, single trees, trees in woods, test walls) into `Obstacle` records, writes one URDF each (buildings also get an `.obj` mesh) to `generated_obstacles/` with `manifest.json`, and loads them at their real positions. Malformed input gives `GeoDataError` (whole input unusable) or a per-object problem line (object skipped). Also has a command-line converter. |
| `env_sim.py` | `use_osm()` validates the input and records skipped objects in `GEO_PROBLEMS`. `build_world()` places every obstacle via `geo_obstacles.spawn()` and fills `OBSTACLE_INFO` = {PyBullet body id: Obstacle}. |
| `osm_world.py` | Reading a `.osm` file skips unreadable nodes and the ways that use them, instead of crashing. `parse_scene()` skips (and reports) an object that fails to parse. Buildings keep their `name`. `building_mesh()` is split out so the same solid shape goes into the URDF. A damaged download cache is re-downloaded. |
| `object_avoidance.py` | The sensor names what it hits (`DETECT`). On first sight it puts a mapped building's whole outline in the obstacle map, with an exact 0.5 m margin from the true walls. It checks the unsafe distance (0.35 m) every scan (`UNSAFE`, then back away and re-plan). A hard stop widens the keep-out around the point that caused it. |
| `main.py`, `operator_setup.py` | Map errors are shown as `ERROR:` messages (console, setup window, operator panel) instead of tracebacks. The end-of-mission report adds detected obstacles, closest sensed surface and unsafe alerts. |
| `tests/test_geo_obstacles.py` (new) | 23 tests for AC-01 to AC-03. |
| `tests/fixtures/geo_test_area.json` (new) | Test map with obstacles at predefined positions on the default flight path. |
| `geo_avoidance_runs.py` (new) | Flight runner that checks AC-01/AC-02 against ground-truth distances. |

## Acceptance criteria

| AC | Result | Evidence |
|---|---|---|
| **AC-01** Key objects are extracted from the user's geographical data and placed in the simulated environment | **PASS** | Fixture: 4 buildings and 5 trees extracted at their predefined positions (within 5 cm). Heights come from the tags and are clamped as before. The L-shaped building keeps its inner corner (concave mesh). Real `map.osm` at 60/200/500 m: 135–137 obstacles placed. Placement error is 0.0 cm in every scenario (tree base vs map position; building bounding box vs outline). Each obstacle is a URDF loaded with `p.loadURDF`, listed in `manifest.json`. Works in both the simple and the detailed look. |
| **AC-02** The drone detects the generated obstacle within the unsafe distance (0.35 m) and avoids it by re-planning | **PASS** | Every obstacle on the route is detected and named (`DETECT building "Test Hall" (8.0 m tall) sensed 5.0 m away`). The drone re-plans around it and reaches 10/10 waypoints in all 5 scenarios. Ground-truth closest approach is 0.43–0.52 m, with 0 samples under 0.35 m and 0 contacts. Pop-up test: an obstacle appears 0.25 m from the drone. `UNSAFE` is logged after 0.004 s and the drone re-plans. It is back outside 0.35 m after 0.13 s and finishes the route without touching anything. |
| **AC-03** Malformed input prints an error message and does not crash | **PASS** | 31 bad or unusual inputs run through `main.py`, with 0 tracebacks: missing, empty, truncated, binary, HTML or invalid-JSON files; wrong top-level type; impossible latitude; nodes without coordinates; unclosed, one-corner or zero-area outlines; wrong tag/member types; unknown element types. 29 print an `ERROR:` message; the other 2 are valid but unusual (a numeric `layer` tag, `null` tags) and are accepted as they are. An unusable file or position stops with `ERROR: the map data could not be used - <reason>`. A bad object is skipped and named (e.g. `object #6 [way 6 (building=yes, "Open Shed")]: the building outline is not closed - skipped`), and everything else is still placed. On the same inputs the original code crashed with a traceback (`TypeError` in `to_xy`, `xml.etree.ElementTree.ParseError`). |

Unsafe distance stays at the default `UNSAFE_DIST = 0.35 m` (drone collision body to obstacle
surface). No change was needed.

## How to reproduce (from `Sprint2/`)

```
python -m unittest tests.test_geo_obstacles -v
python geo_avoidance_runs.py                  # writes geo_avoidance_results/summary.md + runs.csv
python main.py --headless --no-ui --osm-file map.osm --lat 44.6488 --lon -123.3123 --area 200
```

## Results

### Flights with ground truth (`geo_avoidance_runs.py`)

`fixture`/`popup` use the predefined test map (26 m map). `osm-*` use the real `map.osm` export shown
60, 200 or 500 real metres across a 60 m map. Distances are PyBullet closest points between the drone
body and every obstacle, sampled at 40 Hz. They are not the sensor's own estimate.

| Scenario | Area / map (m) | Buildings | Trees | Placed / expected | Placement err (cm) | Detected (bldg / trees) | Re-plans | Reached | Min clearance (m) | Unsafe samples | Contacts | UNSAFE alerts | AC-01 | AC-02 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| fixture | 26 / 26 | 4 | 5 | 9/9 | 0.0 | 4 / 3 | 29 | 10/10 | 0.45 | 0 | 0 | 0 | PASS | PASS |
| popup | 26 / 26 | 4 | 5 | 9/9 | 0.0 | 4 / 3 | 31 | 10/10 | 0.25* | 5* | 0 | 1 | PASS | PASS |
| osm-60 | 60 / 60 | 0 | 135 | 135/135 | 0.0 | 0 / 72 | 54 | 10/10 | 0.43 | 0 | 0 | 0 | PASS | PASS |
| osm-200 | 200 / 60 | 1 | 136 | 137/137 | 0.0 | 1 / 75 | 56 | 10/10 | 0.43 | 0 | 0 | 0 | PASS | PASS |
| osm-500 | 500 / 60 | 3 | 134 | 137/137 | 0.0 | 3 / 70 | 37 | 10/10 | 0.52 | 0 | 0 | 0 | PASS | PASS |

\* Forced by the test: the pop-up obstacle is created 0.25 m from the drone. All 5 unsafe samples
fall in the 0.13 s before the drone backed out; there were none afterwards. Slowest re-plan: 6.4 ms.
"Detected" counts the obstacles that came within sensor range (5 m) of the route; trees deep in a
wood that the drone never passed are not counted.

### Unit tests (`tests/test_geo_obstacles.py`): 23/23 pass, about 2 s

- AC-01: fixture objects at their predefined positions; real export extracted; URDF files, manifest
  and body positions; L-shaped building keeps its notch (ray test); a changed shape in the same
  process is not stale; detailed look hides the collision shapes; `env_sim` builds the map world in
  both looks.
- AC-02: `UNSAFE_DIST` default 0.35; building detected, named, and its whole outline mapped from one
  scan; footprint margin exact (no free cell within 0.5 m of a wall); unsafe alert, back-away and
  re-plan; hard stop widens the keep-out; full fixture flight; pop-up flight.
- AC-03: 10 unusable files each raise one clear error; 11 bad objects skipped and named while the rest
  is placed; `.osm` node without coordinates; a parser error in one object does not stop the rest;
  bad position rejected; unbuildable obstacle skipped; `main.py` end to end (bad file exits with an
  error message, bad objects are reported and the mission still runs).

Each fix below was also checked by undoing it and confirming the matching test fails.

### Regression (DoD: the change must not break existing components)

| Check | Original upload | With this change |
|---|---|---|
| Full test suite `pytest tests` | 225 pass, 1 fail | 248 pass, 1 fail (the same pre-existing failure, see below) |
| PBI-04 `route_safety_runs.py` (20 seeds, run twice) | AC-01..04 PASS, min clearance 0.41 m | AC-01..04 PASS, min clearance 0.41 m |
| `object_avoidance.py --headless` (hand-placed trees) | DONE, 27.1 s, 3 re-plans, 0.51 m | identical |
| `main.py` on `map.osm` 60 m, `--person-seed 3` | FOUND at 38.0 s | FOUND at 38.0 s |
| `main.py` on `map.osm` 500 m, `--person-seed 3` | FOUND at 30.3 s | FOUND at 30.3 s |
| `main.py` on test fixture, `--person-seed 3` | FOUND at 48.0 s | FOUND at 48.1 s |

## Problems found during testing (all fixed)

1. **Re-plan loop and corner cutting at map buildings.** First run of the fixture: 2220 re-plans. The
   drone was stuck about 110 s at the garage, swinging between climbing and descending, and came
   within 0.34 m of it (5 unsafe samples). Cause: a building's whole outline was stamped into the
   0.25 m grid with margins measured from cell centres, so at sharp corners the 0.5 m margin
   shrank to about 0.3 m. The forward corridor rays then stopped the drone, and re-planning on the
   unchanged map returned the same leg. Fixes: the outline margin is now measured from the true
   walls (`FOOTPRINT_REACH`), and a hard stop widens the keep-out around the point that caused it.
   After: 29 re-plans, closest approach 0.45 m, no unsafe samples. The original code on the same
   map: 43 re-plans, 0.43 m.
2. **Stale building shapes.** PyBullet caches loaded meshes by file name for the whole process, even
   after `p.disconnect()`. A second world in the same process silently got the first world's
   building shapes (a ray hit at x = 1.0 instead of 3.0). Fix: generated file names carry a hash of
   their contents.
3. **Collision shapes drawn over the detailed look.** URDF links without a `<visual>` make PyBullet
   draw the bare collision shape: bright spheres on every tree and coloured slabs over every
   building (see `geo_obstacles_render_before_after.png`, attached to the issue). Fix: an invisible
   visual on every link, the same trick `osm_world._invisible_solid` already used.
4. **Zero-area building outlines** (all corners on one line) were placed as degenerate bodies. They are
   now rejected with a message.

## Known minor issues and limitations (to file as GitHub sub-issues)

1. `test_map_search.py::AcceptanceCriteriaTests::test_ac04_targets_in_bounds_and_outside_excluded_zone`
   fails on the original upload too (`r["bounds"]` is `None`). It is unrelated to this change.
2. Climb/descend ping-pong: after a climb caused by repeated hard stops, `maybe_descend()` checks only
   the map and can drop straight back down. This exists in the original code (it caused a ~2 s stall
   at the fixture's garage there). It no longer occurs in any of the runs above, but a hold-off after a
   climb would make it impossible.
3. Map buildings get a wider effective margin (about 0.68 m to cell centres) than single sensor hits,
   so gaps narrower than about 1.4 m between two buildings count as blocked (the drone goes around or
   over).
4. OSM barriers (walls, fences, hedges) remain visual only. They are lower than the 2.5 m cruise
   altitude, so they are not flight obstacles.
5. The probability map's camera-occlusion model only knows trees. The camera detection itself
   ray-casts against buildings correctly.
6. One `DETECT` console line per detected object, about 70 on a wooded map.
7. The operator panel (Tkinter) could not be opened on the test machine (no Tkinter there; the
   original code stops at the same point). The GUI world itself was rendered and checked under a
   virtual display. A teammate should run one GUI mission as a final check.
8. The PBI description names `target_avoidance.py`. That file does not exist; sensing and avoidance
   settings are in `object_avoidance.py`, and obstacle positions are set in `env_sim.py`.
