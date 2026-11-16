"""
PBI-14 tests: detection and obstacle events in the shared mission state and on the dashboard.

  AC-01  SIGHTING / LOST / FOUND events carry the numeric drone-to-person distance, and the
         mission's detection state follows them (SEARCHING / SIGHTING / FOUND).
  AC-02  Blocked routes and nearby obstacles reach the operator dashboard with the distance to the
         obstacle and the drone's response, not only the console.
  AC-03  The dashboard shows the latest event and updates live without slowing the simulation;
         the console log is still produced.

Most tests need no simulator (sample events through Mission, the panel's formatting and
bookkeeping without a window). The integration tests use PyBullet in DIRECT mode (no window).
Run from the Sprint folder:  python -m unittest tests.test_mission_events -v
"""
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import operator_guidance as og                          # noqa: E402
from operator_guidance import Mission, MissionEvent, Cmd   # noqa: E402
import operator_panel                                   # noqa: E402


def started():
    m = Mission()
    m.start()
    return m


def target_event(kind, dist=6.0, t=10.0):
    return MissionEvent(kind, {"SIGHTING": "WARNING", "LOST": "INFO", "FOUND": "INFO"}[kind],
                        "target", (0, 0, 2.5), (4, 4, 0.7), dist, t, response="x",
                        message=f"{kind} test", object_name="Casualty")


def obstacle_event(kind="BLOCKED", dist=0.8, t=5.0, severity="WARNING", response="stopping to re-plan"):
    return MissionEvent(kind, severity, "obstacle", (0, 0, 2.5), (1, 0, 2.5), dist, t,
                        response=response, message=f"{kind} test", object_name="tree 1")


# ---------------------------------------------------------------- fake Tk widgets
class FakeWidget:
    def __init__(self):
        self.config, self.text = {}, ""

    def configure(self, **kw):
        self.config.update(kw)

    def delete(self, *_a):
        self.text = ""

    def insert(self, _where, text, *_tags):
        self.text += text


def fake_panel():
    """OperatorPanel without a window: only the event bookkeeping and formatting are exercised."""
    panel = operator_panel.OperatorPanel.__new__(operator_panel.OperatorPanel)
    panel._shown, panel._events_shown = {}, -1
    panel.event_lbl, panel.event_list = FakeWidget(), FakeWidget()
    return panel


# ====================================================================== shared structure
class TestEventStructure(unittest.TestCase):
    def test_fields_of_the_shared_event(self):
        ev = obstacle_event()
        for field in ("kind", "severity", "source", "drone_position", "object_position", "distance",
                      "timestamp", "mission_state", "detection_state", "response", "message",
                      "object_name"):
            self.assertTrue(hasattr(ev, field), field)
        self.assertIsInstance(ev.distance, float)

    def test_saved_and_restored_with_the_mission(self):
        m = started()
        m.add_event(obstacle_event())
        m.add_event(target_event("SIGHTING"))
        m2 = Mission()
        m2.restore(json.loads(json.dumps(m.snapshot())))
        self.assertEqual([e.to_dict() for e in m2.events], [e.to_dict() for e in m.events])
        self.assertEqual(m2.detection_state, "SIGHTING")

    def test_old_save_files_without_events_still_load(self):
        snap = started().snapshot()
        del snap["events"], snap["detection_state"]
        m = Mission()
        m.restore(snap)
        self.assertEqual((m.events, m.detection_state), ([], "SEARCHING"))

    def test_damaged_events_are_rejected_cleanly(self):
        # EXPECT: ValueError (never TypeError / KeyError) and the mission is left unchanged.
        good = obstacle_event().to_dict()
        for bad in ({**good, "severity": "PANIC"}, {**good, "distance": -1}, {**good, "distance": "far"},
                    {k: v for k, v in good.items() if k != "kind"}, {**good, "extra": 1},
                    {**good, "drone_position": [1, 2]}, {**good, "mission_state": "SIGHTING"}, 5):
            with self.subTest(bad=str(bad)[:40]):
                snap = started().snapshot()
                snap["events"] = [bad]
                m = Mission()
                before = m.snapshot()
                with self.assertRaises(ValueError):
                    m.restore(snap)
                self.assertEqual(m.snapshot(), before)

    def test_event_list_is_capped(self):
        m = started()
        for i in range(og.EVENT_LIMIT + 50):
            m.add_event(obstacle_event("DETECT", t=float(i), severity="INFO", response="obstacle mapped"))
        self.assertEqual(len(m.events), og.EVENT_LIMIT)
        self.assertEqual(m.events_count, og.EVENT_LIMIT + 50)
        self.assertEqual(m.latest_event.timestamp, float(og.EVENT_LIMIT + 49))

    def test_selftest_runs_without_the_simulator(self):
        out = subprocess.run([sys.executable, os.path.join(CODE_DIR, "operator_guidance.py"), "--selftest"],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("event selftest passed", out.stdout)


# ====================================================================== AC-01
class TestAC01DetectionState(unittest.TestCase):
    def test_detection_state_follows_sighting_lost_found(self):
        m = started()
        for kind, want in (("SIGHTING", "SIGHTING"), ("LOST", "SEARCHING"), ("SIGHTING", "SIGHTING"),
                           ("FOUND", "FOUND")):
            ev = m.add_event(target_event(kind, dist=5.5))
            self.assertEqual(m.detection_state, want)
            self.assertEqual(ev.detection_state, want)
            self.assertEqual(ev.distance, 5.5)
        self.assertEqual(m.state, "FOUND")                 # FOUND ends the mission as found

    def test_sighting_does_not_stop_the_mission(self):
        # EXPECT: the MISSION state stays SEARCHING during a sighting, so detection keeps running
        # and operator commands still work.
        m = started()
        m.add_event(target_event("SIGHTING"))
        self.assertEqual(m.state, "SEARCHING")
        self.assertTrue(m.handle(Cmd.AVOID, "C").ok)

    def test_found_logs_once_and_mission_complete_follows(self):
        m = started()
        m.add_event(target_event("FOUND"))
        sources = [h[1] for h in m.history]
        self.assertEqual(sources[-2:], ["DETECT FOUND", "DETECT"])
        self.assertIn("6.0 m away", m.history[-2][2])
        self.assertIn("mission complete", m.history[-1][2])
        self.assertIn("Rejected", m.found())                # main.py does not call it twice

    def test_detector_sends_events_with_numeric_distance(self):
        import math
        import pybullet as p
        from target_detection import MissingPerson, SearchMonitor
        if p.isConnected():
            p.disconnect()
        p.connect(p.DIRECT)
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.85, 0.25, 0.15])
        person = MissingPerson(p.createMultiBody(0, col, -1, [0, 0, 0.15]))
        m = started()
        mon = SearchMonitor(person, log=lambda *_: None, mission=m)
        cam = [-4.5, 0.0, 2.5]
        mon.set_heading(0.0)
        t = 0.0
        while not mon.found and t < 3.0:
            mon.update(cam, t)
            t += 0.1
        p.disconnect()
        kinds = [e.kind for e in m.events]
        self.assertEqual(kinds, ["SIGHTING", "FOUND"])
        for e in m.events:
            self.assertIsInstance(e.distance, float)
            self.assertAlmostEqual(e.distance, math.dist(cam, (0, 0, 0.15)), delta=0.2)
            self.assertEqual(e.source, "target")
            self.assertEqual(e.drone_position, tuple(cam))
        self.assertEqual(m.detection_state, "FOUND")
        self.assertEqual(m.state, "FOUND")


# ====================================================================== AC-02
class TestAC02ObstacleEvents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, os.path.join(CODE_DIR, "tests"))
        import test_geo_obstacles as T
        import object_avoidance as oa
        cls.oa = oa
        floor, drone, _ids = T.fixture_world()
        cls.mission = started()
        cls.nav = oa.Navigator(drone, floor, -1, T.env_sim.FLIGHT_PATH, False, lambda *_: None,
                               obstacle_info=T.env_sim.OBSTACLE_INFO, mission=cls.mission)
        T.fly(cls.nav, drone)
        T.reset_world()

    def test_obstacle_events_carry_distance_and_response(self):
        events = [e for e in self.nav.mission_events]
        kinds = {e.kind for e in events}
        for kind in ("DETECT", "BLOCKED", "REPLAN", "SLOW"):
            self.assertIn(kind, kinds)
        for e in events:
            with self.subTest(kind=e.kind, t=e.timestamp):
                self.assertEqual(e.source, "obstacle")
                self.assertTrue(e.response)
                self.assertIn(e.severity, og.SEVERITIES)
                if e.kind in ("DETECT", "SLOW", "UNSAFE", "BLOCKED"):
                    self.assertIsNotNone(e.distance)
                    self.assertGreaterEqual(e.distance, 0.0)

    def test_responses_match_what_the_drone_does(self):
        responses = {e.kind: e.response for e in self.nav.mission_events}
        self.assertEqual(responses["SLOW"], "slowing down")
        self.assertEqual(responses["BLOCKED"], "stopping to re-plan")
        self.assertEqual(responses["REPLAN"], "re-planned route")
        self.assertIn(responses["DETECT"], ("obstacle mapped", "whole outline mapped"))

    def test_detect_names_the_obstacle_and_its_distance(self):
        detects = [e for e in self.nav.mission_events if e.kind == "DETECT"]
        named = [e for e in detects if e.object_name.startswith('building "')]
        self.assertTrue(named)
        self.assertEqual(named[0].response, "whole outline mapped")
        self.assertIsNotNone(named[0].object_position)

    def test_events_reach_the_shared_mission(self):
        self.assertEqual(self.mission.events_count, len(self.nav.mission_events))
        self.assertEqual(self.mission.latest_event, self.nav.mission_events[-1])

    def test_severity_follows_the_distance(self):
        oa = self.oa
        nav = self.nav
        before = len(nav.mission_events)
        nav.event("UNSAFE", "close", distance=0.2, point=(0, 0, 2.5))
        nav.event("REPLAN", "near", distance=0.7, point=(0, 0, 2.5))
        nav.event("REPLAN", "far", distance=2.5, point=(0, 0, 2.5))
        nav.event("RTL", "home", distance=None, point=None)
        sev = [e.severity for e in nav.mission_events[before:]]
        self.assertEqual(sev, ["CRITICAL", "WARNING", "INFO", "CRITICAL"])
        self.assertLess(oa.UNSAFE_DIST, oa.STOP_DIST)

    def test_important_responses_go_to_the_panel_message_log(self):
        m = started()
        m.add_event(obstacle_event("UNSAFE", 0.2, severity="CRITICAL", response="backing away and re-planning"))
        m.add_event(obstacle_event("REPLAN", 1.0, severity="INFO", response="re-planned route"))
        sources = [h[1] for h in m.history]
        self.assertIn("OBSTACLE UNSAFE", sources)
        self.assertNotIn("OBSTACLE REPLAN", sources)            # routine: event box only
        self.assertEqual(operator_panel.OperatorPanel._kind("OBSTACLE UNSAFE", m.history[-1][2]), "bad")


# ====================================================================== AC-03
class TestAC03Dashboard(unittest.TestCase):
    def test_latest_event_box_and_recent_list(self):
        m = started()
        panel = fake_panel()
        m.add_event(obstacle_event("BLOCKED", 0.84))
        m.add_event(target_event("SIGHTING", 6.9, t=26.0))
        panel._refresh_events(m)
        text = panel.event_lbl.config["text"]
        self.assertIn("SIGHTING", text)
        self.assertIn("Distance: 6.90 m", text)
        self.assertIn("Drone response:", text)
        self.assertEqual(panel.event_lbl.config["fg"], operator_panel.SEVERITY_FG["WARNING"])
        rows = panel.event_list.text.splitlines()
        self.assertEqual(len(rows), 2)
        self.assertIn("SIGHTING", rows[0])                     # newest first
        self.assertIn("0.84 m", rows[1])

    def test_recent_list_keeps_only_the_last_rows(self):
        m = started()
        panel = fake_panel()
        for i in range(20):
            m.add_event(obstacle_event("DETECT", 1.0 + i, t=float(i), severity="INFO", response="obstacle mapped"))
        panel._refresh_events(m)
        self.assertEqual(len(panel.event_list.text.splitlines()), operator_panel.EVENT_ROWS)

    def test_panel_only_redraws_when_a_new_event_arrives(self):
        m = started()
        panel = fake_panel()
        m.add_event(obstacle_event())
        panel._refresh_events(m)
        panel.event_list.text = "untouched"
        for _ in range(100):                                  # 100 polls, nothing new
            panel._refresh_events(m)
        self.assertEqual(panel.event_list.text, "untouched")
        m.add_event(obstacle_event("REPLAN", severity="INFO", response="re-planned route"))
        panel._refresh_events(m)
        self.assertIn("REPLAN", panel.event_list.text)

    def test_adding_events_is_cheap(self):
        # EXPECT: well under 0.1 ms per event, so a burst of events cannot slow the simulation.
        m = started()
        evs = [obstacle_event("DETECT", 1.0, t=float(i), severity="INFO", response="obstacle mapped")
               for i in range(5000)]
        t0 = time.perf_counter()
        for ev in evs:
            m.add_event(ev)
        per_event_ms = (time.perf_counter() - t0) / len(evs) * 1000
        print(f"\n  add_event: {per_event_ms * 1000:.1f} us per event")
        self.assertLess(per_event_ms, 0.1)

    def test_status_text_for_the_dashboard(self):
        m = started()
        m.add_event(target_event("SIGHTING", 6.9))
        self.assertEqual(m.status_text(), "Mission: SEARCHING | Detection: SIGHTING | "
                                          "Latest: SIGHTING 6.90 m - x (Casualty)")

    def test_main_run_keeps_console_and_saves_events(self):
        # EXPECT: the console still shows the events; the saved mission holds them with distances.
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "mission_state.json")
            out = subprocess.run([sys.executable, os.path.join(CODE_DIR, "main.py"), "--headless",
                                  "--no-ui", "--person-seed", "5", "--max-time", "60",
                                  "--detection-log", "events", "--state-file", state],
                                 cwd=tmp, capture_output=True, text=True, timeout=900).stdout
            with open(state) as f:
                mission = json.load(f)["payload"]["mission"]
        self.assertRegex(out, r"\] DETECT\s+")                  # console: obstacle detections
        self.assertRegex(out, r"\] FOUND\s+")                   # console: person found
        kinds = [e["kind"] for e in mission["events"]]
        self.assertIn("DETECT", kinds)
        self.assertEqual(kinds[-1], "FOUND")
        self.assertIsInstance(mission["events"][-1]["distance"], float)
        self.assertEqual(mission["detection_state"], "FOUND")
        self.assertEqual(mission["state"], "FOUND")


if __name__ == "__main__":
    unittest.main()
