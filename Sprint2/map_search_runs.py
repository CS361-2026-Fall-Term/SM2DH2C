"""Acceptance runs for the map-guided search PBI. Needs PyBullet (runs main.py --headless).

    python map_search_runs.py            # all scenarios, prints a table + AC checks
    python map_search_runs.py --md       # same, as a Markdown table to paste into the Issue
    python map_search_runs.py --random 20   # 20 random person spots (--person-seed 1..20) instead

Each scenario runs `python main.py --headless --person X Y ...` and reads its printout:
  [map] t= ...s Searching most likely cell (x, y) (look from ...)   one line per selected target
  Mission: FOUND | Navigation: ... | sim time ...
  Final report: ...
  Detection: {... 'found_time': ...}
"""
import argparse
import ast
import math
import os
import re
import subprocess
import tempfile
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MAP_HALF = 13.0                       # default 26 m map
ZONES = {"A": (6.0, 6.0), "B": (-6.0, 6.0), "C": (-6.0, -6.0), "D": (6.0, -6.0)}

# name, person (x, y), extra args, what is expected
SCENARIOS = [
    ("AC-01 outside zone plan", (-9.0, -9.0), [], "FOUND"),
    ("Map phase: east edge", (11.5, 0.5), [], "FOUND"),
    ("Map phase: far corner", (-12.0, -12.0), [], "FOUND"),
    ("Under tree (-8, -8)", (-8.4, -8.3), [], "FOUND"),
    ("Under tree (0, 1), map centre", (0.4, 1.3), [], "FOUND"),
    ("Under tree (9, 8.5)", (9.4, 8.8), [], "FOUND"),
    ("Default spot (zone D)", (7.0, -5.5), [], "FOUND"),
    ("AC-05 time limit 45 s", (0.5, 11.5), ["--max-time", "45"], "COMPLETE"),
    # The person is inside excluded zone A: no target may be picked in A. The forward camera can
    # still see into A from outside it, so FOUND or COMPLETE are both fine here (None = any).
    ("AC-04 zone A excluded", (9.4, 8.8), ["--exclude-zones", "A", "--max-time", "120"], None),
]


def zone_rect(xy, half=MAP_HALF):
    """Same square main.zone_rect() draws and excludes."""
    h = max(1.0, min(abs(xy[0]), abs(xy[1]), half - abs(xy[0]), half - abs(xy[1])) - 0.4)
    return (xy[0] - h, xy[1] - h, xy[0] + h, xy[1] + h)


def run(person, extra):
    cmd = [sys.executable, os.path.join(HERE, "main.py"), "--headless", *extra]
    if person is not None:
        cmd += ["--person", str(person[0]), str(person[1])]
    with tempfile.TemporaryDirectory() as tmp:          # PBI-10 state file: keep it out of the repo
        cmd += ["--state-file", os.path.join(tmp, "mission_state.json")]
        out = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True, timeout=1800).stdout
    targets = [(float(m.group(1)), float(m.group(2))) for m in
               re.finditer(r"\[map\].*Searching most likely cell \((-?[\d.]+), (-?[\d.]+)\)", out)]
    state = re.search(r"^Mission: (\w+)", out, re.M)
    t_end = re.search(r"sim time ([\d.]+) s", out)
    report = re.search(r"^Final report: (.*)$", out, re.M)
    det = re.search(r"^Detection: (\{.*\})$", out, re.M)
    bounds = re.search(r"^Search area: \((.*?)\) \((.*?)\)$", out, re.M)
    plan_end = re.search(r"zone plan finished at t=([\d.]+)s", out)
    det = ast.literal_eval(det.group(1)) if det else {}
    return {"state": state.group(1) if state else "?", "t_end": float(t_end.group(1)) if t_end else None,
            "found_time": det.get("found_time"), "targets": targets,
            "report": report.group(1) if report else "",
            "plan_end": float(plan_end.group(1)) if plan_end else None,
            "bounds": [tuple(map(float, b.split(","))) for b in bounds.groups()] if bounds else None,
            "searching_shown": "Searching most likely cell" in out, "raw": out}


def check(name, person, extra, expect, r):
    notes = []
    ok = expect is None or r["state"] == expect
    if expect == "FOUND":
        ok &= r["found_time"] is not None and r["found_time"] <= 240.0
    # AC-02: once the plan ends, the drone keeps searching and says where it is going
    if r["plan_end"] is not None and not (r["state"] == "FOUND" and not r["targets"]):
        if not r["searching_shown"]:
            ok = False
            notes.append("AC-02: no 'Searching most likely cell' message")
    # AC-03: never the same cell more than 3 times in a row
    run_len, worst = 0, 0
    for i, tgt in enumerate(r["targets"]):
        run_len = run_len + 1 if i and tgt == r["targets"][i - 1] else 1
        worst = max(worst, run_len)
    if worst > 3:
        ok = False
        notes.append(f"AC-03: same cell {worst} times in a row")
    # AC-04: targets inside the search bounds and outside excluded zones
    excl = []
    if "--exclude-zones" in extra:
        excl = list(extra[extra.index("--exclude-zones") + 1].upper())
    for tgt in r["targets"]:
        if r["bounds"]:
            (x0, x1), (y0, y1) = r["bounds"]
            if not (x0 <= tgt[0] <= x1 and y0 <= tgt[1] <= y1):
                ok = False
                notes.append(f"AC-04: {tgt} outside bounds")
        for z in excl:
            a, b, c, d = zone_rect(ZONES[z])
            if a <= tgt[0] <= c and b <= tgt[1] <= d:
                ok = False
                notes.append(f"AC-04: {tgt} inside excluded zone {z}")
    # AC-05: time-out ends COMPLETE and names the most likely remaining cell
    if r["state"] == "COMPLETE" and "most likely remaining cell" not in r["report"]:
        ok = False
        notes.append("AC-05: final report has no most likely cell")
    return ok, worst, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="print a Markdown table")
    ap.add_argument("--only", type=int, nargs="*", help="scenario numbers to run (1-based)")
    ap.add_argument("--random", type=int, default=0, metavar="N",
                    help="run N random person spots (--person-seed 1..N) instead of the scenarios")
    args = ap.parse_args()
    scenarios = SCENARIOS
    if args.random:
        scenarios = [(f"Random spot, seed {k}", None, ["--person-seed", str(k)], "FOUND")
                     for k in range(1, args.random + 1)]
    rows, all_ok = [], True
    for i, (name, person, extra, expect) in enumerate(scenarios, 1):
        if args.only and i not in args.only:
            continue
        r = run(person, extra)
        ok, worst, notes = check(name, person, extra, expect, r)
        all_ok &= ok
        ft = f"{r['found_time']:.1f}" if r["found_time"] is not None else "-"
        pe = f"{r['plan_end']:.1f}" if r["plan_end"] is not None else "-"
        if person is None:
            m = re.search(r"^Person placed at \((-?[\d.]+), (-?[\d.]+)\)", r["raw"], re.M)
            person = (float(m.group(1)), float(m.group(2))) if m else (float("nan"),) * 2
        rows.append((i, name, f"({person[0]:g}, {person[1]:g})", " ".join(extra) or "-", r["state"],
                     ft, pe, str(len(r["targets"])), str(worst), "PASS" if ok else "FAIL",
                     "; ".join(notes) or r["report"]))
        print(f"[{i}] {name}: {r['state']} found_time={ft} targets={len(r['targets'])} "
              f"-> {'PASS' if ok else 'FAIL'} {'; '.join(notes)}", file=sys.stderr)
    head = ("#", "Scenario", "Person", "Extra args", "Mission", "Found at (s)", "Plan ended (s)",
            "Map targets", "Max repeats", "Result", "Notes / final report")
    if args.md:
        print("| " + " | ".join(head) + " |")
        print("|" + "---|" * len(head))
        for row in rows:
            print("| " + " | ".join(str(c) for c in row) + " |")
    else:
        for row in [head] + rows:
            print("  ".join(str(c) for c in row))
    print("ALL PASS" if all_ok else "SOME FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
