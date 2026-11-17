# Sprint 3 Folder
This folder contains the system's source/test code implemented during Sprint 3 for the course project. 

**Sprint goal**: Focus on adjusting and refining the detection mechanism, so the drone is able to better replan its navigation when it encounters an obstacle. Once this is done integrated, rigorously test and validate all components of the major capabilities and make sure it aligns with the mandatory key requirements. 

**PBIs worked on**: PBI-02, PBI-05, PBI-07, PBI-08, PBI-12, PBI-13, PBI-14, PBI-15


Tests (run from `Sprint3/`):
```
python -m unittest discover -s tests -v            # runs every test codes in tests/
```

PBI-05 (regression tests for the probability map, route planning and headless missions), see `tests/PBI05_REGRESSION_TEST_REPORT.md`:
```
python -m unittest tests.test_regression_pbi05 -v                    # AC-01..AC-04
SKIP_SLOW=1 python -m unittest tests.test_regression_pbi05 -v        # skip the 3 headless missions
BREAK_NORMALIZE=1 python -m unittest tests.test_regression_pbi05.ProbabilityMapTests -v   # AC-03: _normalize() broken on purpose, tests must fail
```
- The mission tests run `python main.py --headless --person X Y` for the default person (7, -5.5), a person under a tree (0.4, 1.3) and a person outside the zone plan (-9, -9), and check for `Mission: FOUND` within 240 s.
- `BREAK_NORMALIZE=1` swaps `_normalize()` for a do-nothing version only while the tests run; the source file is not changed.

PBI-12 (range sensor and climb-or-detour), results in `tests/RANGE_SENSOR_TEST_REPORT.md`:
```
python -m unittest tests.test_range_sensor -v      # AC-01..AC-04
python range_sensor_runs.py                        # re-runs the scenarios and rewrites the report
```

PBI-13 (detection results with the confidence level in the terminal), see `tests/DETECTION_RESULTS_REPORT.md`:
```
python main.py --detection-log scans     # default: a line per camera scan while the person is in view
python main.py --detection-log events    # only sightings and the result
python -m unittest tests.test_detection_results -v
```

PBI-14 (detection and obstacle events in the mission state and on the operator panel), see `tests/MISSION_EVENTS_REPORT.md`:
```
python operator_guidance.py --selftest           # sample events through the mission state, no simulator
python -m unittest tests.test_mission_events -v
```

PBI-07 (operator dashboard stays responsive under heavy live input: clicks shown within 50 ms), code in `ui_responsiveness.py`, see `tests/UI_RESPONSIVENESS_REPORT.md`:
```
python -m unittest tests.test_ui_responsiveness_pbi07 -v   # AC-01..AC-03 with the real Tk panel (needs a display)
python main.py --profile                                   # the final summary prints "UI response - click: ... refresh: ..."
```
- A click is shown on the panel first. The heavy work runs afterwards: the probability map, the route re-plan, and the heatmap repaint, which now runs in batches.
- Mission saves are written by a background thread. No new dependencies.

PBI-08 (live mission display: six fields together, updates within 300 ms), code in `display_status.py`:
```
python -m unittest tests.test_display_pbi08 -v     # AC-01..AC-04 + weak-network cases, no simulator needed
python main.py --profile                           # per-section timings; the final summary prints "Display latency - ..." for the HUD and the panel
```

PBI-02 (simulation stability and performance), code in `stability.py`, see `tests/STABILITY_PERFORMANCE_REPORT.md`:
```
python main.py --metrics                     # any mission: PASS/FAIL table of the performance metrics at the end
python stability_runs.py                     # 16 full missions, writes tests/STABILITY_RUNS.md (--quick: 3 runs)
python -m unittest tests.test_stability_pbi02 -v
```
