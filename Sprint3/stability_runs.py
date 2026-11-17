"""
PBI-02: stability and performance runs. Full headless search missions on every kind of world,
with live operator commands, a long mission and a crash-and-resume, each with
`main.py --metrics`. For each run it checks:

  AC-01  no crash: exit code 0, no Python traceback, no unexpected error line; the mission ends
         properly (FOUND, or COMPLETE when the time limit is reached)
  AC-02  no glitches (drone position always valid, no jumps, never inside an obstacle) and no
         environment object moved, appeared or disappeared
  AC-03  every metric within its limit (stability.METRICS)

Run:  python stability_runs.py            (writes tests/STABILITY_PERFORMANCE_REPORT.md)
      python stability_runs.py --quick    (3 short runs, a smoke test)
"""
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time

import stability

HERE = os.path.dirname(os.path.abspath(__file__))
OSM = ["--osm-file", "map.osm", "--lat", "44.6488", "--lon", "-123.3123"]
FIXTURE = ["--osm-file", os.path.join("tests", "fixtures", "geo_test_area.json"),
           "--lat", "44.6488", "--lon", "-123.3123", "--map-size", "26"]
COMMANDS = "5:AVOID:C,10:REINCLUDE:C,15:SEARCH_FIRST:D,20:INVESTIGATE:B,25:RETURN_PLAN"

# (name, extra arguments, mission ends allowed)
SCENARIOS = [
    ("practice world, person seed 1", ["--person-seed", "1"], ("FOUND",)),
    ("practice world, person seed 2", ["--person-seed", "2"], ("FOUND",)),
    ("practice world, person seed 3", ["--person-seed", "3"], ("FOUND",)),
    ("practice world, person seed 4", ["--person-seed", "4"], ("FOUND",)),
    ("practice world, person seed 5", ["--person-seed", "5"], ("FOUND",)),
    ("OSM map, 60 m", OSM + ["--area", "60", "--person-seed", "6"], ("FOUND", "COMPLETE")),
    ("OSM map, 60 m, long search", OSM + ["--area", "60", "--person-seed", "7"], ("FOUND", "COMPLETE")),
    ("OSM map, 200 m", OSM + ["--area", "200", "--person-seed", "6"], ("FOUND", "COMPLETE")),
    ("OSM map, 200 m, long search", OSM + ["--area", "200", "--person-seed", "7"], ("FOUND", "COMPLETE")),
    ("OSM map, 500 m", OSM + ["--area", "500", "--person-seed", "6"], ("FOUND", "COMPLETE")),
    ("OSM map, 500 m, long search", OSM + ["--area", "500", "--person-seed", "7"], ("FOUND", "COMPLETE")),
    ("PBI-15 test map (buildings on the route)", FIXTURE + ["--area", "26", "--person", "9", "-9"],
     ("FOUND", "COMPLETE")),
    ("operator commands during the search", ["--person", "-9", "-9", "--command-script", COMMANDS],
     ("FOUND", "COMPLETE")),
    ("operator commands, OSM 200 m", OSM + ["--area", "200", "--person-seed", "2",
                                            "--command-script", COMMANDS], ("FOUND", "COMPLETE")),
    ("time limit reached (person in an excluded zone, 120 s)",
     ["--person", "-9", "-9", "--exclude-zones", "C", "--max-time", "120"], ("COMPLETE", "FOUND")),
]
QUICK = [0, 7, 11]
CRASH_AT_S = 15.0                 # crash-and-resume: kill the first run once this sim time is printed
UNEXPECTED = re.compile(r"Traceback|Exception|Error(?!s)\b|ERROR:|Segmentation|core dumped|invalid command name")


def run_main(extra, state_file, timeout=3600):
    cmd = [sys.executable, os.path.join(HERE, "main.py"), "--headless", "--no-ui", "--metrics",
           "--detection-log", "events", "--state-file", state_file] + extra
    t0 = time.perf_counter()
    out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=timeout)
    return out.returncode, out.stdout, out.stderr, time.perf_counter() - t0


def parse(stdout):
    """The Stability: {...} line and the mission end state of one run."""
    stab = None
    m = re.search(r"^Stability: (\{.*\})$", stdout, re.M)
    if m:
        stab = json.loads(m.group(1))
    st = re.search(r"^Mission: (\w+)", stdout, re.M)
    return stab, st.group(1) if st else None


def evaluate(name, code, stdout, stderr, wall, allowed_ends):
    stab, end = parse(stdout)
    errors = [l for l in (stdout + "\n" + stderr).splitlines() if UNEXPECTED.search(l)]
    ac01 = code == 0 and not errors and stab is not None and end in allowed_ends
    ac02 = stab is not None and stab["glitches"] == 0 and stab["moved_objects"] == 0
    failed_metrics = []
    if stab is not None:
        for metric, (limit, kind, _unit, _m) in stability.METRICS.items():
            v = stab.get(metric)
            if v is not None and not (v <= limit if kind == "max" else v >= limit):
                failed_metrics.append(f"{metric}={v}")
    ac03 = stab is not None and not failed_metrics
    return {"name": name, "exit": code, "end": end, "errors": errors[:5], "stability": stab,
            "wall_s": wall, "ac01": ac01, "ac02": ac02, "ac03": ac03, "failed_metrics": failed_metrics}


def run_scenario(name, extra, allowed_ends):
    with tempfile.TemporaryDirectory() as tmp:
        code, out, err, wall = run_main(extra, os.path.join(tmp, "mission_state.json"))
    return evaluate(name, code, out, err, wall, allowed_ends)


def crash_and_resume():
    """Kill a mission (no clean-up) once sim time CRASH_AT_S is printed, then --resume it."""
    with tempfile.TemporaryDirectory() as tmp:
        state = os.path.join(tmp, "mission_state.json")
        cmd = [sys.executable, os.path.join(HERE, "main.py"), "--headless", "--no-ui",
               "--detection-log", "events", "--state-file", state] + OSM + ["--area", "200",
                                                                           "--person", "-12", "12"]
        proc = subprocess.Popen(cmd, cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        killed_at = None
        for line in proc.stdout:
            m = re.match(r"\[t=\s*([\d.]+)s\]", line)
            if m and float(m.group(1)) >= CRASH_AT_S:
                killed_at = float(m.group(1))
                proc.kill()                           # hard kill: like a crash
                break
        proc.wait()
        if killed_at is None:
            return {"name": "crash and --resume", "exit": proc.returncode, "end": "finished before the crash",
                    "errors": [], "stability": None, "wall_s": 0, "ac01": False, "ac02": False,
                    "ac03": False, "failed_metrics": []}
        code, out, err, wall = run_main(["--resume"], state)
    r = evaluate(f"crash at t={killed_at:.0f} s, then --resume", code, out, err, wall, ("FOUND", "COMPLETE"))
    r["resumed"] = "[state] Resumed:" in out
    r["ac01"] = r["ac01"] and r["resumed"]
    return r


def fmt(v, unit=""):
    if v is None:
        return "n/a"
    return f"{v:.2f}{unit}" if isinstance(v, float) else f"{v}{unit}"


def write_report(results, path):
    engine = "stand-in" if "stand-in" in (__import__("pybullet").__doc__ or "") else "PyBullet"
    lines = ["# PBI-02: simulation stability and performance - run results", "",
             f"Generated by `python stability_runs.py` ({engine}, headless). "
             "Each run is a full `main.py --headless --metrics` search mission.", ""]
    if engine == "stand-in":
        lines += ["> These numbers come from a geometric stand-in for PyBullet used during development "
                  "(timing and memory are not the real engine's). Regenerate this file with "
                  "`python stability_runs.py` on a machine with PyBullet before using it.", ""]
    lines += ["## Metrics (limits from `stability.METRICS`)", "", "| Metric | Limit | Meaning |", "|---|---|---|"]
    for name, (limit, kind, unit, meaning) in stability.METRICS.items():
        lines.append(f"| {name} | {'<=' if kind == 'max' else '>='} {limit:g} {unit} | {meaning} |")
    lines += ["", "## Runs", "",
              "| Run | End | Frame mean / p95 / max (ms) | Real-time factor | Memory growth (MB) / trend (MB/min) | "
              "Start-up (s) | Glitches | Moved objects | AC-01 | AC-02 | AC-03 |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        s = r["stability"] or {}
        ok = lambda b: "PASS" if b else "FAIL"
        lines.append(f"| {r['name']} | {r['end']} ({fmt(s.get('sim_s'))} s) | "
                     f"{fmt(s.get('frame_work_mean_ms'))} / {fmt(s.get('frame_work_p95_ms'))} / "
                     f"{fmt(s.get('frame_work_max_ms'))} | {fmt(s.get('realtime_factor'), 'x')} | "
                     f"{fmt(s.get('memory_growth_mb'))} / {fmt(s.get('memory_trend_mb_per_min'))} | {fmt(s.get('startup_s'))} | {fmt(s.get('glitches'))} | "
                     f"{fmt(s.get('moved_objects'))} | {ok(r['ac01'])} | {ok(r['ac02'])} | {ok(r['ac03'])} |")
    problems = [r for r in results if not (r["ac01"] and r["ac02"] and r["ac03"])]
    lines += ["", f"**{len(results) - len(problems)} of {len(results)} runs pass all three acceptance criteria.**", ""]
    for r in problems:
        lines.append(f"* {r['name']}: exit {r['exit']}, end {r['end']}, errors {r['errors']}, "
                     f"metrics {r['failed_metrics']}, glitches {(r['stability'] or {}).get('glitch_list')}, "
                     f"environment {(r['stability'] or {}).get('environment_changes')}")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="3 short runs only")
    ap.add_argument("--out", default=os.path.join(HERE, "tests", "STABILITY_RUNS.md"))
    args = ap.parse_args()
    chosen = [SCENARIOS[i] for i in QUICK] if args.quick else SCENARIOS
    results = []
    for name, extra, ends in chosen:
        r = run_scenario(name, extra, ends)
        print(f"{'PASS' if r['ac01'] and r['ac02'] and r['ac03'] else 'FAIL'}  {name}  "
              f"({r['end']}, {r['wall_s']:.0f} s)", flush=True)
        results.append(r)
    if not args.quick:
        r = crash_and_resume()
        print(f"{'PASS' if r['ac01'] and r['ac02'] and r['ac03'] else 'FAIL'}  {r['name']}", flush=True)
        results.append(r)
    print(write_report(results, args.out))
    return 0 if all(r["ac01"] and r["ac02"] and r["ac03"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
