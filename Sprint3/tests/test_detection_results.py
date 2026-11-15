"""
PBI-13 tests: detection results and the confidence level returned to the user in the terminal.

  AC-01  The confidence is calculated from the detection results of each scan and returned to
         the user (every scan line, the SIGHTING / LOST / FOUND events and the final summary).
  AC-02  The user can read the results and understand the method: the method is explained once at
         the start, columns are labelled and aligned, the summary is in plain words.
  AC-03  The results change with every scan, in a readable structure: one line per scan while the
         person is in view (or the confidence is above 0), one summary line per quiet stretch.

Run from the Sprint folder (needs PyBullet; no window opens):
    python -m unittest tests.test_detection_results -v
"""
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

import target_detection as td                           # noqa: E402
from target_detection import MissingPerson, SearchMonitor, format_scan, explain_method   # noqa: E402

ALT = 2.5
FRAME = 1.0 / td.CAM_FPS


def world():
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")


def box_person(x=0.0, y=0.0):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.85, 0.25, 0.15])
    return MissingPerson(p.createMultiBody(0, col, -1, [x, y, 0.15]))


def standoff(dist):
    h = ALT - 0.15
    return [-math.sqrt(max(dist ** 2 - h ** 2, 0.0)), 0.0, ALT]


def run(monitor, cam, start, seconds, heading):
    """Hover at cam with the camera at `heading` for `seconds`, one update per scan."""
    monitor.set_heading(heading)
    t = start
    while t < start + seconds - 1e-9 and not monitor.found:
        monitor.update(cam, t)
        t += FRAME
    return t


def scripted(log_mode="scans"):
    """7 m from the person (no clear view): 1 s looking at it, 4 s looking away, 1 s back, then
    move to 5 m (clear view, found). Returns (monitor, printed lines)."""
    world()
    person = box_person()
    lines = []
    m = SearchMonitor(person, log=lines.append, log_mode=log_mode)
    far, near = standoff(7.0), standoff(5.0)
    toward = math.atan2(-far[1], -far[0])
    t = run(m, far, 0.0, 1.0, toward)
    t = run(m, far, t, 4.0, toward + math.pi)              # looking the other way
    t = run(m, far, t, 1.0, toward)
    run(m, near, t, 2.0, toward)
    return m, lines


class TestAC01ConfidenceFromResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.lines = scripted()

    def test_every_scan_follows_the_confidence_rule(self):
        # EXPECT: score = (seen / 7) x quality (0 below 20 % seen); confidence rises by score x
        # gain x 0.1 s, is kept for SIGHTING_MEMORY_S after the last sighting, then decays.
        prev, last_seen = 0.0, None
        for s in self.m.scans:
            with self.subTest(t=s["t"]):
                frac = s["visible"] / s["points"]
                want_score = frac * s["quality"] if frac >= td.MIN_VISIBLE_FRACTION else 0.0
                self.assertAlmostEqual(s["score"], want_score, places=9)
                if s["state"] == td.FOUND_STATE and s["note"] == "clear view" and s["clear"] >= td.CLEAR_VIEW_FRAMES:
                    want = max(prev + td.CONF_GAIN * s["score"] * FRAME, td.FOUND_AT)
                elif s["score"] > 0:
                    want = prev + td.CONF_GAIN * s["score"] * FRAME
                elif last_seen is None or s["t"] - last_seen > td.SIGHTING_MEMORY_S + 1e-6:
                    want = max(0.0, prev - td.CONF_DECAY * FRAME)
                elif abs(s["t"] - last_seen - td.SIGHTING_MEMORY_S) <= 1e-6:
                    # exactly at the end of the memory time: either is right (float rounding)
                    want = s["confidence"] if s["confidence"] in (prev, max(0.0, prev - td.CONF_DECAY * FRAME)) else prev
                else:
                    want = prev
                self.assertAlmostEqual(s["confidence"], want, places=6)
                self.assertAlmostEqual(s["change"], s["confidence"] - prev, places=9)
                if s["score"] > 0:
                    last_seen = s["t"]
                prev = s["confidence"]

    def test_printed_confidence_matches_the_calculation(self):
        # EXPECT: every printed scan line shows that scan's confidence (2 decimals).
        printed = [l for l in self.lines if re.match(r"\[detect\]\s+\d+\.\d\d\s+[A-Z]", l)]
        by_t = {f"{s['t']:.2f}": s for s in self.m.scans}
        self.assertTrue(printed)
        for line in printed:
            t, state, conf = line.split()[1:4]
            self.assertEqual(conf, f"{by_t[t]['confidence']:.2f}", line)
            self.assertEqual(state, by_t[t]["state"])

    def test_events_and_summary_report_the_confidence(self):
        events = [l for l in self.lines if "SIGHTING" in l or "FOUND" in l]
        self.assertTrue(any("FOUND" in l and "clear view" in l for l in events), events)
        summary = self.m.format_report()
        self.assertIn(f"confidence {self.m.confidence:.2f} of {td.FOUND_AT:.2f}", summary)
        self.assertIn(f"Highest conf.:    {self.m.max_confidence:.2f}", summary)
        r = self.m.report()
        self.assertEqual((r["found"], r["found_by"]), (True, "clear view"))


class TestAC02Readable(unittest.TestCase):
    def test_method_explanation_names_every_threshold(self):
        # EXPECT: the explanation is built from the real settings, so it can't go out of date.
        text = explain_method()
        for value in (f"{td.SIGHTING_ON:.2f}", f"{td.SIGHTING_OFF:.2f}", f"{td.FOUND_AT:.2f}",
                      f"{td.CLEAR_VIEW_RANGE:.0f} m", f"{td.SIGHTING_MEMORY_S:.0f} s",
                      f"{td.CONF_DECAY:g} per second", f"{td.CAM_RANGE:.0f} m",
                      f"{td.MIN_VISIBLE_FRACTION:.0%}", f"{td.FULL_QUALITY_RANGE:.0f} m"):
            self.assertIn(value, text)
        for word in ("seen", "quality", "score", "confidence", "Clear view", "Columns"):
            self.assertIn(word, text)

    def test_columns_line_up_with_the_header(self):
        # EXPECT: each value ends where its column title ends (fixed-width table).
        scan = dict(t=12.3, state="SIGHTING", confidence=0.456, change=0.042, in_view=6, visible=5,
                    points=7, dist=6.234, quality=0.735, score=0.525, clear=1, note="seen")
        line = format_scan(scan)
        header = td.SCAN_HEADER
        for title, value in (("t (s)", "12.30"), ("conf", "0.46"), ("change", "+0.04"),
                             ("seen", "5/7"), ("dist", "6.2 m"), ("quality", "0.73"), ("score", "0.53")):
            self.assertEqual(header.index(title) + len(title), line.index(value) + len(value), title)
        self.assertEqual(header.index("state"), line.index("SIGHTING"))
        self.assertTrue(line.endswith("seen"))

    def test_not_in_view_scan_uses_dashes_not_numbers(self):
        line = format_scan(dict(t=1.0, state="SEARCHING", confidence=0.2, change=0.0, in_view=0,
                                visible=0, points=7, dist=9.9, quality=0.3, score=0.0, clear=0,
                                note="not in view, kept for 1.0 s"))
        self.assertNotIn("9.9", line)
        self.assertIn(" - ", line)

    def test_summary_is_plain_words(self):
        m, _ = scripted()
        text = m.format_report()
        for label in ("Result:", "Found at:", "First sighting:", "Sightings:", "Highest conf.:",
                      "Camera scans:", "Location:"):
            self.assertIn(label, text)
        self.assertNotIn("{", text)

    def test_summary_when_never_seen(self):
        world()
        m = SearchMonitor(box_person(), log=lambda *_: None)
        run(m, standoff(td.CAM_RANGE + 2), 0.0, 1.0, 0.0)
        text = m.format_report()
        self.assertIn("not found", text)
        self.assertIn("no estimate (never seen)", text)


class TestAC03EveryScan(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.m, cls.lines = scripted()

    def test_one_record_per_scan_ten_per_second(self):
        self.assertEqual(len(self.m.scans), self.m.frames)
        gaps = {round(b["t"] - a["t"], 2) for a, b in zip(self.m.scans, self.m.scans[1:])}
        self.assertEqual(gaps, {round(FRAME, 2)})

    def test_every_active_scan_is_printed_and_quiet_stretches_are_summarised(self):
        # EXPECT: a line for each scan with the person in view or confidence > 0; the rest is
        # folded into "person not in the camera view" lines, at most one per IDLE_SUMMARY_S.
        active = [s for s in self.m.scans if s["in_view"] > 0 or s["confidence"] > 0 or s["state"] == "FOUND"]
        printed = [l for l in self.lines if re.match(r"\[detect\]\s+\d+\.\d\d\s+[A-Z]", l)]
        self.assertEqual(len(printed), len(active))
        idle = [l for l in self.lines if "person not in the camera view" in l]
        quiet = len(self.m.scans) - len(active)
        self.assertLessEqual(len(idle), quiet * FRAME / td.IDLE_SUMMARY_S + 2)
        self.assertGreaterEqual(sum(int(re.search(r"\((\d+) scans\)", l).group(1)) for l in idle),
                                quiet - int(td.IDLE_SUMMARY_S / FRAME))

    def test_values_change_from_scan_to_scan(self):
        # EXPECT: while watching the person the confidence rises scan after scan.
        watching = [s for s in self.m.scans if s["score"] > 0][:5]
        confs = [s["confidence"] for s in watching]
        self.assertEqual(confs, sorted(confs))
        self.assertEqual(len(set(round(c, 4) for c in confs)), len(confs))

    def test_header_repeats_so_long_runs_stay_readable(self):
        headers = [i for i, l in enumerate(self.lines) if "t (s)" in l]
        self.assertTrue(headers)
        self.assertEqual(headers[0], min(i for i, l in enumerate(self.lines) if l.startswith("[detect]") and "t (s)" in l))

    def test_log_modes(self):
        _, events = scripted("events")
        self.assertFalse([l for l in events if l.startswith("[detect]")])
        self.assertTrue(any("FOUND" in l for l in events))
        world()
        with self.assertRaises(ValueError):
            SearchMonitor(box_person(), log_mode="loud")


class TestOperatorPanel(unittest.TestCase):
    """Detection events reach the operator panel's message log (through the mission history)."""

    def test_found_with_estimate_is_logged_for_the_panel(self):
        from operator_guidance import Mission
        import operator_panel
        world()
        mission = Mission()
        mission.start()
        m = SearchMonitor(box_person(), log=lambda *_: None,
                          on_event=lambda kind, text: mission.record(f"DETECT {kind}", text))
        run(m, standoff(5.0), 0.0, 2.0, math.atan2(0.0, -standoff(5.0)[0]))
        found = [h for h in mission.history if h[1] == "DETECT FOUND"]
        self.assertEqual(len(found), 1)
        self.assertRegex(found[0][2], r"^FOUND \(t = [\d.]+ s\): Casualty confirmed \(clear view, "
                                      r"[\d.]+ m away\)\. Estimated location \(-?[\d.]+, -?[\d.]+\)")
        self.assertTrue(any(h[1] == "DETECT SIGHTING" for h in mission.history))
        self.assertEqual(operator_panel.OperatorPanel._kind("DETECT FOUND", found[0][2]), "event")
        # the panel copies every new history line into its message log
        panel = operator_panel.OperatorPanel.__new__(operator_panel.OperatorPanel)
        panel._hist_n, panel._last_resp, panel.logged = 0, None, []
        panel._add_message = lambda ts, src, msg: panel.logged.append((src, msg))
        panel._collect_messages(mission)
        self.assertIn(found[0][2], [msg for _src, msg in panel.logged])

    def test_main_puts_detection_events_in_the_saved_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            state = os.path.join(tmp, "mission_state.json")
            subprocess.run([sys.executable, os.path.join(CODE_DIR, "main.py"), "--headless", "--no-ui",
                            "--person-seed", "5", "--max-time", "60", "--detection-log", "off",
                            "--state-file", state], cwd=tmp, capture_output=True, text=True, timeout=900)
            import json
            with open(state) as f:
                history = json.load(f)["payload"]["mission"]["history"]
        sources = [h[1] for h in history]
        self.assertIn("DETECT FOUND", sources)
        self.assertLess(sources.index("DETECT FOUND"), len(sources) - 1)   # before "mission complete"
        self.assertIn("Estimated location", history[sources.index("DETECT FOUND")][2])


class TestMainOutput(unittest.TestCase):
    """main.py prints the explanation first, a line per scan, then the readable summary."""

    def run_main(self, *extra):
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, os.path.join(CODE_DIR, "main.py"), "--headless", "--no-ui",
                   "--person-seed", "5", "--max-time", "60",
                   "--state-file", os.path.join(tmp, "mission_state.json"), *extra]
            return subprocess.run(cmd, cwd=tmp, capture_output=True, text=True, timeout=900).stdout

    def test_order_of_the_output(self):
        out = self.run_main()
        i_method = out.index("How the detection confidence works")
        i_scan = out.index("[detect]")
        i_mission = out.index("\nMission: ")
        i_summary = out.index("Detection summary")
        self.assertLess(i_method, i_scan)
        self.assertLess(i_scan, i_mission)
        self.assertLess(i_mission, i_summary)
        self.assertRegex(out, r"Detection: \{.*'found_by'")        # machine-readable line kept

    def test_off_prints_nothing_extra(self):
        out = self.run_main("--detection-log", "off")
        self.assertNotIn("[detect]", out)
        self.assertNotIn("How the detection confidence works", out)
        self.assertNotIn("Detection summary", out)
        self.assertIn("Detection: {", out)


if __name__ == "__main__":
    unittest.main()
