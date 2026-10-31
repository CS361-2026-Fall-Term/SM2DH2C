"""
Runs the full headless mission for several last known positions (LKPs) and reports the result.

  python tests/run_lkp_tests.py              # all cases (live cases need internet the first time,
                                             #   then they are served from osm_cache/)
  python tests/run_lkp_tests.py --offline    # only cases that use a local OSM file

Each case checks:
  - the OSM world builds without errors and the parser output is printed
  - the mission ends in FOUND or COMPLETE (not ABORT, not a timeout)
Edit CASES to add your own coordinates.
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# (name, lat, lon, area_m, osm_file or None)
CASES = [
    ("synthetic fixture (offline)", 45.5122, -122.6587, 120, os.path.join(HERE, "sample_osm.json")),
    ("synthetic fixture, true scale", 45.5122, -122.6587, 26, os.path.join(HERE, "sample_osm.json")),
    ("urban: downtown Portland", 45.5189, -122.6795, 120, None),
    ("park: Laurelhurst Park", 45.5208, -122.6252, 150, None),
    ("forest: Forest Park trail", 45.5710, -122.7590, 150, None),
    ("residential: SE Portland", 45.4965, -122.6205, 120, None),
]


def run_case(name, lat, lon, area, osm_file, max_time):
    cmd = [sys.executable, "main.py", "--headless", "--max-time", str(max_time),
           "--lat", str(lat), "--lon", str(lon), "--area", str(area)]
    if osm_file:
        cmd += ["--osm-file", osm_file]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=max_time * 4 + 120)
    out = proc.stdout + proc.stderr
    world = re.search(r"OSM world.*", out)
    mission = re.search(r"Mission: (\w+) \| Navigation: (\w+)", out)
    ok = proc.returncode == 0 and mission and mission.group(1) in ("FOUND", "COMPLETE")
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print("   ", world.group(0) if world else "(world not built)")
    print("   ", mission.group(0) if mission else "(no mission result)")
    if not ok:
        print("    --- last output ---")
        print("    " + "\n    ".join(out.strip().splitlines()[-12:]))
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--max-time", type=float, default=240)
    args = ap.parse_args()
    cases = [c for c in CASES if c[4] or not args.offline]
    results = [run_case(*c, args.max_time) for c in cases]
    print(f"\n{sum(results)}/{len(results)} cases passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
