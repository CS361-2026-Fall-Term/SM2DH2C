# PBI-02: Validate environment stability and performance

## 1. Acceptable performance metrics (team limits)
The limits live in `stability.py` (`METRICS`). Every run made with `--metrics` is checked against them.

| Metric | Limit | Why this limit |
|---|---|---|
| `frame_work_mean_ms` | <= 16.7 ms | The simulation draws at 60 Hz. On average, all work in a frame (logic, physics, drawing) has to fit in one frame. |
| `frame_work_p95_ms` | <= 33.3 ms | 95 % of frames are ready in time for at least 30 FPS, so the view doesn't stutter. |
| `fps` (window) | >= 30 FPS | Frames actually shown per second with the window. |
| `realtime_factor` (headless) | >= 1.0x | The simulation is never slower than real time. |
| `memory_growth_mb` | <= 50 MB | Process memory (RSS) growth from 5 s into the mission to the end. A leak shows up here. The steady trend (MB/min over the second half) is reported too. |
| `startup_s` | <= 10 s | Map loading, world building and the probability map are done before the mission starts. |
| `glitches` | 0 | The drone's position is a valid number, inside the map and above the ground. It never jumps further in one step than it can fly, and never ends up inside an obstacle (PyBullet ground truth). |
| `moved_objects` | 0 | Every obstacle, the floor and the person are in the same place and orientation at the end, and no body appears or disappears. |

Response times are already covered by earlier PBIs and are not measured again here: a click is shown on the panel within 50 ms (PBI-07), a command is validated within 10 ms (PBI-09), the state is saved within 1 s (PBI-10), and the dashboard updates within 300 ms (PBI-08).

## 2. How it is measured
* **`stability.py` (new): `StabilityMonitor`.** Switch it on with `python main.py --metrics`; it works with or without the window. Each frame's work time is measured before the pacing sleep. Memory is sampled every simulated second. The glitch checks run every logic step, and the "inside an obstacle" check runs every 10 steps. The environment is compared at the end. At the end, `main.py` prints a PASS / FAIL table and one `Stability: {...}` line that scripts can read.
* **`main.py`:**
  * `--metrics`: switches the measurements on.
  * `--command-script "5:AVOID:C,10:REINCLUDE:C,..."`: sends operator commands at set sim times in headless runs. They go through the same two-stage pipeline as a panel click.
* **`stability_runs.py` (new):** 16 full search missions, written to `tests/STABILITY_RUNS.md`:
  * the practice world with 5 person spots;
  * `map.osm` at 60 / 200 / 500 m, each with a short and a long search;
  * the PBI-15 test map with buildings on the route;
  * operator commands during the search, on the practice world and on the 200 m map (a 554 s mission);
  * a run that ends at the time limit;
  * a run killed at t = 15 s and resumed with `--resume`.

  Each run is checked for AC-01 (exit code 0, no traceback or error line, mission ends FOUND / COMPLETE), AC-02 (no glitches, no environment changes) and AC-03 (all metrics within the limits).

## 3. Issues found and fixed

| # | Issue | Effect | Fix |
|---|---|---|---|
| 1 | Without the window, every new sensor hit was added to `ObstacleMap.new_hits`, a list only used to draw red hit marks. Nothing emptied it. | Memory grew for the whole mission. With the window, dense areas built a backlog, so marks were drawn seconds late. | Without the window the list is cleared. With the window only the newest `MAX_HIT_MARKS` are kept, since only those can be shown. (`object_avoidance.py`) |
| 2 | Logs kept for the whole mission: `SearchMonitor.scans` (10 per second, PBI-13), and `Navigator.events` and `.mission_events` (PBI-14). | About 1 MB more for every few minutes of search. | Capped to the newest 3,000 scans and 5,000 events. The mission's own event list was already capped at 200. (`target_detection.py`, `object_avoidance.py`) |
| 3 | The obstacle map stored a Python set of height buckets for every cell. It grows with the explored area, and on the 200 m map it was ~7 MB after 150 s. | Memory grew steadily over long missions, and every "is this cell blocked?" check looped over buckets. | Each cell now stores a bit mask (one int). The check is one AND. On the PBI-15 test map the flight is identical, with the same 87 events and the same route, and the average re-plan time went from 5.9 to 4.8 ms. (`object_avoidance.py`) |
| 4 | The "jump" glitch check reported 0.013 m steps (3.2 m/s) on normal flights. | False glitches. | Not a simulation bug: the navigator snaps onto a waypoint once it is within 1.5 cm, and it can climb and fly at the same time. The check now allows `hypot(V_MAX, V_Z)` plus the 2 cm snap. Real jumps, e.g. 0.5 m in one step, are still caught (tested). |

Effect of fixes 1 to 3 on the longest run (`map.osm` 200 m with operator commands, 554 s simulated):

| | Memory growth | Trend over the second half |
|---|---|---|
| Before | 17.8 MB | (not measured yet) |
| After fixes 1 and 2 | 14.5 MB | 1.19 MB/min |
| After fixes 1 to 3 | **5.8 MB** | **0.61 MB/min** |

The remaining growth is the obstacle map filling in as new areas are explored. It is bounded by the map size and is not a leak.

**Observation, not changed.** Single frames reach 40–100 ms when the route is re-planned in dense map areas. A* can run at several flight levels since PBI-12. These are rare: the 95th percentile stays under 21 ms in every run. If they are noticeable with the window, the re-plan can be spread over several frames in a later PBI.

## 4. Results
See `tests/STABILITY_RUNS.md` for the full table. **All 16 runs pass AC-01, AC-02 and AC-03.** There were no crashes, tracebacks or error lines, no glitches, and no environment object moved. Every metric was within its limit: mean frame work 0.8–9.7 ms, 95th percentile up to 20.3 ms, headless 1.7–19.8x faster than real time, memory growth at most 5.8 MB.

> The numbers in `tests/STABILITY_RUNS.md` were made with a geometric stand-in for PyBullet during development, without the window. Before submitting:
> * regenerate them on a machine with PyBullet: `python stability_runs.py`;
> * run one mission with the window to measure FPS: `python main.py --metrics`, then look for "fps" in the table at the end.

## 5. Tests
`tests/test_stability_pbi02.py` (26 tests). Run from the Sprint folder: `python -m unittest tests.test_stability_pbi02 -v`.

* **The checks themselves:**
  * a limit for every kind of metric;
  * PASS / FAIL at the limits;
  * memory can be read;
  * every kind of glitch is caught (not a number, outside the map, altitude, jump, inside an obstacle), and normal flight and the waypoint snap are not;
  * a moved, rotated or added body is caught;
  * frame time, real-time factor and memory trend are calculated correctly;
  * the report is readable.
* **The fixes:** headless sensor hits are not kept, and the event and scan logs are capped.
* **`stability_runs.evaluate`:** flags a crash, a moved object, a slow run and an aborted mission.
* **Full missions:** the practice world, the 200 m OSM map and a run with operator commands each pass AC-01, AC-02 and AC-03. All 5 commands are applied, and a bad `--command-script` gives an error message, not a crash.

All 403 tests in `tests/` pass. `test_ui_responsiveness_pbi07.BackgroundSaves.test_ui_thread_cost_inline_vs_background` compares two timings and failed once while other heavy processes were running; it passes on its own.
