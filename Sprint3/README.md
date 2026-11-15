# Sprint 3 Folder
This folder contains the system's source/test code implemented during Sprint 3 for the course project. 

**Sprint goal**: Focus on adjusting and refining the detection mechanism, so the drone is able to better replan its navigation when it encounters an obstacle. Once this is done integrated, rigorously test and validate all components of the major capabilities and make sure it aligns with the mandatory key requirements. 

**PBIs worked on**: PBI-02, PBI-05, PBI-07, PBI-08, PBI-12, PBI-13, PBI-14, PBI-15


Tests (run from `Sprint3/`):
```
python -m unittest discover -s tests -v            # runs every test codes in tests/
```

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
