"""
PBI-02 tests: simulation stability and performance during drone search missions.

  AC-01  No crashes, glitches or unexpected errors during a search mission.
  AC-02  The environment stays stable: no glitches, and no environment object moves, appears or
         disappears during the search.
  AC-03  The simulation performs within the team's acceptable performance metrics
         (stability.METRICS: frame time / FPS, real-time factor, memory, start-up).

The checks themselves are tested first (they must catch a real glitch, a moved object, a slow
frame, a memory leak), then full headless missions are run with `main.py --metrics`.
Run from the Sprint folder (needs PyBullet; no window opens):
    python -m unittest tests.test_stability_pbi02 -v
    SKIP_SLOW=1 python -m unittest tests.test_stability_pbi02 -v      (skip the full missions)
The full set of runs (13 missions) is `python stability_runs.py`.
"""
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import pybullet as p                                    # noqa: E402

import stability                                        # noqa: E402
from stability import StabilityMonitor, METRICS         # noqa: E402

SLOW = os.environ.get("SKIP_SLOW") == "1"
STAND_IN = "stand-in" in (p.__doc__ or "")     # development stand-in: no orientations / body count
DT = 1.0 / 240.0


class FakeClock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


def world():
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    floor = p.loadURDF("plane.urdf")
    box = p.createMultiBody(0, p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1.5]), -1, [3, 0, 1.5])
    drone = p.loadURDF(os.path.join(CODE_DIR, "custom_drone.urdf"), [0, 0, 2.5], globalScaling=0.6)
    return floor, box, drone


def fly_straight(mon, steps=200, v=2.0, start=(0.0, 0.0, 2.5)):
    x, y, z = start
    for i in range(steps):
        mon.step(p, (x + v * DT * i, y, z), i * DT, map_half=13, v_max=2.0, v_z=1.0, dt=DT, max_alt=5.5)


# ====================================================================== the checks themselves
class TestMetricsDefinition(unittest.TestCase):
    def test_every_kind_of_metric_has_a_limit(self):
        # EXPECT: frame time / FPS, real-time factor, memory, start-up, glitches, environment.
        for name in ("frame_work_mean_ms", "frame_work_p95_ms", "fps", "realtime_factor",
                     "memory_growth_mb", "startup_s", "glitches", "moved_objects"):
            self.assertIn(name, METRICS)
        self.assertAlmostEqual(METRICS["frame_work_mean_ms"][0], 1000 / 60)
        self.assertEqual(METRICS["fps"][:2], (30.0, "min"))

    def test_checks_pass_and_fail_on_the_limits(self):
        mon = StabilityMonitor(True)
        good = {"frame_work_mean_ms": 5.0, "frame_work_p95_ms": 10.0, "fps": None, "realtime_factor": 3.0,
                "memory_growth_mb": 2.0, "startup_s": 1.0, "glitches": 0, "moved_objects": 0}
        self.assertTrue(all(r[-1] for r in mon.checks(good)))
        for metric, bad in (("frame_work_mean_ms", 20.0), ("frame_work_p95_ms", 40.0), ("fps", 20.0),
                            ("realtime_factor", 0.5), ("memory_growth_mb", 80.0), ("startup_s", 15.0),
                            ("glitches", 1), ("moved_objects", 2)):
            with self.subTest(metric=metric):
                rows = {r[0]: r[-1] for r in mon.checks({**good, metric: bad})}
                self.assertFalse(rows[metric])

    def test_percentile(self):
        self.assertEqual(stability.percentile(list(range(1, 101)), 95), 95)
        self.assertIsNone(stability.percentile([], 95))

    def test_memory_can_be_read(self):
        mb = stability.rss_mb()
        self.assertIsNotNone(mb, f"process memory could not be read on {sys.platform}")
        self.assertGreater(mb, 1.0)

    def test_disabled_monitor_costs_nothing_and_records_nothing(self):
        mon = StabilityMonitor(False)
        fly_straight(mon, 10)
        mon.frame_done()
        self.assertEqual((mon.frame_ms, mon.memory, mon.glitch_count), ([], [], 0))


class TestGlitchDetection(unittest.TestCase):
    def test_normal_flight_is_not_a_glitch(self):
        mon = StabilityMonitor(True)
        fly_straight(mon)
        self.assertEqual(mon.glitch_count, 0)

    def test_waypoint_snap_is_not_a_glitch(self):
        mon = StabilityMonitor(True)
        mon.step(p, (0.0, 0.0, 2.5), 0.0, v_max=2.0, v_z=1.0, dt=DT)
        mon.step(p, (0.015, 0.0, 2.5), DT, v_max=2.0, v_z=1.0, dt=DT)    # navigator snaps within 1.5 cm
        self.assertEqual(mon.glitch_count, 0)

    def test_each_kind_of_glitch_is_caught(self):
        cases = {
            "not a number": [(0, 0, 2.5), (float("nan"), 0, 2.5)],
            "outside the map": [(0, 0, 2.5), (20.0, 0, 2.5)],
            "altitude": [(0, 0, 2.5), (0, 0, -0.5)],
            "jumped": [(0, 0, 2.5), (0.5, 0, 2.5)],               # 0.5 m in 4 ms = 120 m/s
        }
        for word, path in cases.items():
            with self.subTest(glitch=word):
                mon = StabilityMonitor(True)
                for i, pos in enumerate(path):
                    mon.step(p, pos, i * DT, map_half=13, v_max=2.0, v_z=1.0, dt=DT, max_alt=5.5)
                self.assertGreaterEqual(mon.glitch_count, 1)
                self.assertTrue(any(word in g for g in mon.glitches), mon.glitches)

    def test_drone_inside_an_obstacle_is_caught(self):
        floor, box, drone = world()
        mon = StabilityMonitor(True)
        p.resetBasePositionAndOrientation(drone, [3.0, 0.0, 1.5], [0, 0, 0, 1])     # inside the box
        for i in range(stability.CONTACT_EVERY):
            mon.step(p, (3.0, 0.0, 1.5), i * DT, drone_id=drone, obstacle_ids=[box], dt=DT, v_max=2.0)
        p.disconnect()
        self.assertEqual(mon.glitch_count, 1)
        self.assertIn("inside obstacle", mon.glitches[0])

    def test_drone_clear_of_obstacles_is_fine(self):
        floor, box, drone = world()
        mon = StabilityMonitor(True)
        for i in range(3 * stability.CONTACT_EVERY):
            mon.step(p, (0.0, 0.0, 2.5), i * DT, drone_id=drone, obstacle_ids=[box], dt=DT, v_max=2.0)
        p.disconnect()
        self.assertEqual(mon.glitch_count, 0)


class TestEnvironmentChecks(unittest.TestCase):
    def setUp(self):
        self.floor, self.box, self.drone = world()
        self.before = stability.snapshot_environment(p, [self.floor, self.box])

    def tearDown(self):
        p.disconnect()

    def test_unchanged_environment(self):
        p.resetBasePositionAndOrientation(self.drone, [5, 5, 3], [0, 0, 0, 1])     # the drone may move
        self.assertEqual(stability.compare_environment(p, self.before), [])

    def test_moved_obstacle_is_caught(self):
        p.resetBasePositionAndOrientation(self.box, [3.01, 0, 1.5], [0, 0, 0, 1])
        changes = stability.compare_environment(p, self.before)
        self.assertEqual(len(changes), 1)
        self.assertIn(f"body {self.box} moved", changes[0])

    @unittest.skipIf(STAND_IN, "needs the real PyBullet")
    def test_rotated_obstacle_is_caught(self):
        p.resetBasePositionAndOrientation(self.box, [3, 0, 1.5], p.getQuaternionFromEuler([0, 0, 0.1]))
        self.assertTrue(stability.compare_environment(p, self.before))

    @unittest.skipIf(STAND_IN, "needs the real PyBullet")
    def test_new_body_is_caught(self):
        p.createMultiBody(0, p.createCollisionShape(p.GEOM_SPHERE, radius=0.2), -1, [1, 1, 1])
        self.assertTrue(any("number of bodies" in c for c in stability.compare_environment(p, self.before)))


class TestPerformanceMeasurement(unittest.TestCase):
    def test_frame_times_fps_and_realtime_factor(self):
        clock = FakeClock()
        mon = StabilityMonitor(True, gui=False, clock=clock)
        mon.t_program = clock.t - 2.0                   # 2 s start-up
        floor, box, drone = world()
        mon.started(p, [floor, box])
        for i in range(60):                             # 60 frames of 5 ms, 10 of them 40 ms
            clock.t += 0.040 if i % 6 == 0 else 0.005
            mon.frame_done()
            mon.frame_start()
        mon.finish(p, sim_t=1.0)
        p.disconnect()
        s = mon.summary()
        self.assertAlmostEqual(s["startup_s"], 2.0)
        self.assertAlmostEqual(s["frame_work_max_ms"], 40.0, places=3)
        self.assertAlmostEqual(s["frame_work_p95_ms"], 40.0, places=3)
        self.assertAlmostEqual(s["realtime_factor"], 1.0 / 0.65, places=2)
        self.assertIsNone(s["fps"])                     # headless: no frames shown
        self.assertFalse(dict((r[0], r[-1]) for r in mon.checks(s))["frame_work_p95_ms"])

    def test_memory_trend_spots_a_steady_climb(self):
        mon = StabilityMonitor(True)
        mon.memory = [(float(t), 100.0 + 0.5 * t) for t in range(0, 120)]          # +30 MB/min
        self.assertAlmostEqual(mon.summary()["memory_trend_mb_per_min"], 30.0, places=3)
        mon.memory = [(float(t), 100.0 + (5.0 if t > 10 else 0.0)) for t in range(0, 120)]
        self.assertAlmostEqual(mon.summary()["memory_trend_mb_per_min"], 0.0, places=6)

    def test_report_is_readable(self):
        mon = StabilityMonitor(True)
        fly_straight(mon, 20)
        text = mon.format_report()
        self.assertIn("Stability and performance (PBI-02)", text)
        self.assertRegex(text, r"Overall: (PASS|FAIL)")
        for name in METRICS:
            self.assertIn(name, text)


class TestBoundedMemory(unittest.TestCase):
    """Issues found by PBI-02: lists that grew for the whole mission are now capped."""

    def test_headless_sensor_hits_are_not_kept(self):
        import object_avoidance as oa
        floor, box, drone = world()
        nav = oa.Navigator(drone, floor, -1, [(0, 0, 2.5), (10, 0, 2.5)], False, lambda *_: None)
        nav.pos = [0.0, 0.0, 2.5]
        for _ in range(50):
            nav.update(oa.DT)
        p.disconnect()
        self.assertTrue(nav.map.seen)                       # it did sense the box
        self.assertEqual(nav.map.new_hits, [])              # but kept no undrawn backlog

    def test_navigator_event_logs_are_capped(self):
        import object_avoidance as oa
        old = oa.EVENT_LOG_LIMIT
        try:
            oa.EVENT_LOG_LIMIT = 50
            nav = oa.Navigator(0, 1, 2, [(0, 0, 2.5), (10, 0, 2.5)], False, lambda *_: None)
            for i in range(200):
                nav.event("REPLAN", f"r{i}", distance=1.0, point=(0, 0, 0))
            self.assertEqual(len(nav.events), 50)
            self.assertEqual(len(nav.mission_events), 50)
            self.assertEqual(nav.events[-1][2], "r199")
        finally:
            oa.EVENT_LOG_LIMIT = old

    def test_detector_scan_log_is_capped(self):
        import target_detection as td
        old = td.SCAN_LOG_LIMIT
        try:
            td.SCAN_LOG_LIMIT = 30
            floor, box, drone = world()
            person = td.MissingPerson(box)
            mon = td.SearchMonitor(person, log=lambda *_: None)
            for i in range(100):
                mon.update([-20, 0, 2.5], i * 0.1)        # person out of camera range: never found
            p.disconnect()
            self.assertEqual(len(mon.scans), 30)
            self.assertEqual(mon.frames, 100)
        finally:
            td.SCAN_LOG_LIMIT = old


class TestRunner(unittest.TestCase):
    """stability_runs.evaluate must flag every kind of failure."""

    def test_evaluate_flags_failures(self):
        import stability_runs as SR
        good = {"glitches": 0, "moved_objects": 0, "frame_work_mean_ms": 2.0, "frame_work_p95_ms": 4.0,
                "realtime_factor": 5.0, "memory_growth_mb": 1.0, "startup_s": 1.0, "fps": None}
        ok_out = f"Mission: FOUND | x\nStability: {json.dumps(good)}\n"
        r = SR.evaluate("ok", 0, ok_out, "", 1.0, ("FOUND",))
        self.assertTrue(r["ac01"] and r["ac02"] and r["ac03"])
        crash = SR.evaluate("crash", 1, ok_out + "Traceback (most recent call last):\n", "", 1.0, ("FOUND",))
        self.assertFalse(crash["ac01"])
        moved = SR.evaluate("moved", 0, ok_out.replace('"moved_objects": 0', '"moved_objects": 1'), "", 1, ("FOUND",))
        self.assertFalse(moved["ac02"])
        slow = SR.evaluate("slow", 0, ok_out.replace('"realtime_factor": 5.0', '"realtime_factor": 0.5'), "", 1, ("FOUND",))
        self.assertFalse(slow["ac03"])
        aborted = SR.evaluate("abort", 0, ok_out.replace("FOUND", "ABORT"), "", 1, ("FOUND",))
        self.assertFalse(aborted["ac01"])


# ====================================================================== full missions
def run_main(*extra, timeout=1800):
    with tempfile.TemporaryDirectory() as tmp:
        cmd = [sys.executable, os.path.join(CODE_DIR, "main.py"), "--headless", "--no-ui", "--metrics",
               "--detection-log", "off", "--state-file", os.path.join(tmp, "mission_state.json"), *extra]
        return subprocess.run(cmd, cwd=CODE_DIR, capture_output=True, text=True, timeout=timeout)


def stability_line(stdout):
    m = re.search(r"^Stability: (\{.*\})$", stdout, re.M)
    return json.loads(m.group(1)) if m else None


@unittest.skipIf(SLOW, "SKIP_SLOW=1: full missions skipped")
class TestFullMissions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = {
            "practice": run_main("--person-seed", "2"),
            "osm": run_main("--osm-file", "map.osm", "--lat", "44.6488", "--lon", "-123.3123",
                            "--area", "200", "--person", "-10", "10", "--max-time", "150"),
            "commands": run_main("--person", "-9", "-9", "--command-script",
                                 "5:AVOID:C,10:REINCLUDE:C,15:SEARCH_FIRST:D,20:INVESTIGATE:B,25:RETURN_PLAN"),
        }

    def test_ac01_no_crash_or_unexpected_error(self):
        for name, out in self.runs.items():
            with self.subTest(run=name):
                text = out.stdout + out.stderr
                self.assertEqual(out.returncode, 0, text[-1500:])
                self.assertNotIn("Traceback", text)
                self.assertRegex(out.stdout, r"Mission: (FOUND|COMPLETE)")

    def test_ac02_no_glitch_and_environment_unchanged(self):
        for name, out in self.runs.items():
            with self.subTest(run=name):
                s = stability_line(out.stdout)
                self.assertIsNotNone(s, out.stdout[-1500:])
                self.assertEqual(s["glitches"], 0, s["glitch_list"])
                self.assertEqual(s["moved_objects"], 0, s["environment_changes"])

    def test_ac03_within_the_performance_metrics(self):
        for name, out in self.runs.items():
            with self.subTest(run=name):
                s = stability_line(out.stdout)
                failed = [f"{m}={s.get(m)}" for m, (lim, kind, _u, _t) in METRICS.items()
                          if s.get(m) is not None and not (s[m] <= lim if kind == "max" else s[m] >= lim)]
                self.assertEqual(failed, [])
                self.assertTrue(s["passed"])

    def test_operator_commands_were_applied_during_the_search(self):
        out = self.runs["commands"].stdout
        self.assertEqual(len(re.findall(r"^\[operator\] t=", out, re.M)), 5)
        self.assertIn("Zone C removed from the plan", out)

    def test_bad_command_script_is_an_error_message_not_a_crash(self):
        out = run_main("--command-script", "soon:AVOID:C", timeout=120)
        self.assertEqual(out.returncode, 2)
        self.assertIn("--command-script", out.stderr)
        self.assertNotIn("Traceback", out.stderr)


if __name__ == "__main__":
    unittest.main()
