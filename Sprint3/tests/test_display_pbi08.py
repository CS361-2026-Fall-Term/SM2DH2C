"""
PBI-08 tests: the live mission display (display_status.py + its use in main.py).
No PyBullet or Tk needed: time is simulated with a virtual clock using the same refresh cadence
as main.py (60 Hz render tick, HUD every HUD_EVERY ticks, panel polled every BUTTON_POLL_EVERY).
Run from the Sprint folder:  python -m unittest tests.test_display_pbi08 -v

Covers
  AC-01  display updates within 300 ms (HUD, panel, operator command response)
  AC-02  all six fields visible at once
  AC-03  the system's response after an operator command is displayed
  AC-04  a full mission runs on one set of objects (no restart), display always complete
  NFR    weak network: delayed, jittered, lost and out-of-order updates never leave the
         display stale for more than 300 ms after an update arrives, and never go backwards
"""
import os
import random
import re
import sys
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import display_status as ds
from operator_guidance import Cmd, Mission

TICK = 1.0 / 60.0                       # render tick (main.py RENDER_EVERY * DT)


def _main_const(name):
    """Read a constant from main.py's source (importing main needs PyBullet)."""
    with open(os.path.join(CODE_DIR, "main.py"), encoding="utf-8") as f:
        m = re.search(rf"^{name}\s*=\s*(\d+)", f.read(), re.M)
    assert m, name
    return int(m.group(1))


HUD_EVERY = _main_const("HUD_EVERY")
# Live fields are refreshed every PANEL_REFRESH_EVERY render ticks (PBI-07: clicks are read every
# tick, but the status refresh cadence is what bounds the display latency modelled here). Since
# PBI-07 the panel also draws right after a refresh, so "drawn at the next refresh" below is a
# conservative upper bound.
POLL_EVERY = _main_const("PANEL_REFRESH_EVERY")


class FakeNav:
    def __init__(self):
        self.pos, self.state, self.horizontal_speed = [0.0, 0.0, 2.5], "IDLE", 0.0


class FakeMonitor:
    def __init__(self):
        self.state, self.confidence = "SEARCHING", 0.0


class FakeSearcher:
    active, target = False, None


def fields_of(m, nav, mon, holding=False):
    return ds.fields_from_sim(m, nav, mon, FakeSearcher(), holding)


def run_display(change_ticks, n_ticks, mutate, tracker=None):
    """Replays main.py's display cadence on a virtual clock. mutate(tick) changes the state at
    the given ticks. Returns the tracker (latencies in ms per surface)."""
    tracker = tracker or ds.LatencyTracker()
    panel_sig = None
    for tick in range(n_ticks):
        now = tick * TICK
        # panel poll (every POLL_EVERY ticks): Tk draws what refresh() set last time, then refresh
        if tick % POLL_EVERY == 0:
            if panel_sig is not None:
                tracker.shown("panel", panel_sig, now)
                panel_sig = None
        if tick in change_ticks:
            mutate(tick)
        sig = mutate.sig()
        tracker.observe(sig, now=now)
        if tick % POLL_EVERY == 0:
            panel_sig = sig
        if tick % HUD_EVERY == 0:
            tracker.shown("hud", sig, now)
    return tracker


class SixFields(unittest.TestCase):
    def setUp(self):
        self.m, self.nav, self.mon = Mission(), FakeNav(), FakeMonitor()
        self.m.start()
        self.nav.state, self.nav.horizontal_speed = "NAV", 2.0

    def test_all_six_fields_in_one_block(self):
        text = ds.panel_text(fields_of(self.m, self.nav, self.mon))
        for label in ("Position:", "Mission state:", "Destination:", "Current plan:",
                      "Moving / stopped:", "Possible sighting:"):
            self.assertIn(label, text)
        self.assertEqual(len(text.splitlines()), 6)

    def test_values(self):
        f = fields_of(self.m, self.nav, self.mon)
        self.assertEqual(f["mission"], "SEARCHING")
        self.assertTrue(f["destination"].startswith("Zone " + self.m.plan[0]))
        self.assertIn("(", f["destination"])                 # coordinates shown
        self.assertEqual(f["plan"], self.m.plan_text())
        self.assertTrue(f["motion"].startswith("MOVING"))
        self.assertTrue(f["sighting"].startswith("none"))
        self.assertIn("x 0.0", f["position"])

    def test_position_updates(self):
        a = ds.signature(fields_of(self.m, self.nav, self.mon))
        self.nav.pos[0] = 3.4
        b = ds.signature(fields_of(self.m, self.nav, self.mon))
        self.assertNotEqual(a, b)
        self.assertIn("x 3.4", fields_of(self.m, self.nav, self.mon)["position"])

    def test_invisible_position_change_is_not_an_update(self):
        a = ds.signature(fields_of(self.m, self.nav, self.mon))
        self.nav.pos[0] = 0.01                                # rounds to the same 0.1 m
        self.assertEqual(a, ds.signature(fields_of(self.m, self.nav, self.mon)))

    def test_moving_stopped_hovering(self):
        self.assertEqual(ds.motion_state("NAV", 2.0), "MOVING")
        self.assertEqual(ds.motion_state("NAV", 0.0), "STOPPED")
        self.assertEqual(ds.motion_state("DONE", 1.0), "STOPPED")
        self.assertEqual(ds.motion_state("NAV", 2.0, holding=True), "HOVERING")
        self.assertEqual(ds.motion_state("NAV", float("nan")), "STOPPED")

    def test_possible_sighting(self):
        self.mon.state, self.mon.confidence = "SIGHTING", 0.42
        self.assertEqual(fields_of(self.m, self.nav, self.mon)["sighting"],
                         "POSSIBLE SIGHTING (0.42)")
        self.mon.state = "FOUND"
        self.assertTrue(fields_of(self.m, self.nav, self.mon)["sighting"].startswith("PERSON FOUND"))

    def test_map_phase_destination(self):
        self.m.plan = []
        s = FakeSearcher()
        s.active, s.target = True, (3.0, -2.0)
        f = ds.fields_from_sim(self.m, self.nav, self.mon, s)
        self.assertEqual(f["destination"], "map cell (3.0, -2.0)")
        self.assertEqual(f["plan"], "none")

    def test_hud_line_and_clip(self):
        f = fields_of(self.m, self.nav, self.mon)
        self.assertIn("Pos:", ds.hud_line(f))
        self.assertIn("Motion:", ds.hud_line(f))
        self.assertLessEqual(len(ds.clip("x" * 500)), ds.MAX_LINE)
        self.assertEqual(ds.clip("short"), "short")


class Latency300ms(unittest.TestCase):
    def test_refresh_cadence_is_within_budget(self):
        hud_ms = HUD_EVERY * TICK * 1000
        panel_ms = POLL_EVERY * TICK * 1000 * 2              # refresh, then drawn at next poll
        print(f"\n  worst case HUD {hud_ms:.0f} ms, panel {panel_ms:.0f} ms (limit "
              f"{ds.LATENCY_LIMIT_MS:g} ms)")
        self.assertLess(hud_ms, ds.LATENCY_LIMIT_MS)
        self.assertLess(panel_ms, ds.LATENCY_LIMIT_MS)

    def test_random_state_changes_shown_within_limit(self):
        rng = random.Random(8)
        state = {"x": 0.0}
        ticks = sorted(rng.sample(range(5, 3600), 400))      # 400 changes in 60 s

        def mutate(tick):
            state["x"] += 0.5
        mutate.sig = lambda: ("pos", round(state["x"], 1))
        tr = run_display(set(ticks), 3700, mutate)
        for surface, d in tr.summary().items():
            print(f"\n  {surface}: {d['count']} updates, p95 {d['p95_ms']:.1f} ms, "
                  f"worst {d['max_ms']:.1f} ms")
            self.assertGreater(d["count"], 100)          # changes drawn together count once
            self.assertLess(d["max_ms"], ds.LATENCY_LIMIT_MS)
            self.assertEqual(d["over_limit"], 0)

    def test_tracker_flags_a_slow_draw(self):
        tr = ds.LatencyTracker(clock=lambda: 0.0)
        tr.observe("a", now=0.0)
        self.assertAlmostEqual(tr.shown("hud", "a", now=0.5), 500.0)
        self.assertEqual(tr.summary()["hud"]["over_limit"], 1)
        self.assertIn("over 300 ms: 1", tr.report())

    def test_old_state_drawn_does_not_count(self):
        tr = ds.LatencyTracker()
        tr.observe("a", now=0.0)
        tr.observe("b", now=0.1)                             # changed again before a draw
        self.assertIsNone(tr.shown("hud", "a", now=0.12))    # drew the old state: still pending
        self.assertAlmostEqual(tr.shown("hud", "b", now=0.2), 200.0)   # counted from the first change

    def test_command_response_shown_within_limit(self):
        """AC-03: an operator command and the system's reply appear on the next HUD frame
        (main.py forces a HUD refresh on every command) - timed through the real Mission."""
        m = Mission()
        m.start()
        for cmd, zone in ((Cmd.INVESTIGATE, "C"), (Cmd.AVOID, "B"), (Cmd.REINCLUDE, "B"),
                          (Cmd.RETURN_PLAN, None), (Cmd.AVOID, "Z")):
            before = m.response
            res = m.handle(cmd, zone)
            self.assertNotEqual(m.response, before)
            self.assertEqual(m.response, res.message)        # the text the HUD / panel show
            tr = ds.LatencyTracker()
            tr.observe(m.response, now=0.0)
            # forced HUD refresh = drawn on the next render tick; panel at its next two polls
            self.assertLess(tr.shown("hud", m.response, now=TICK) * 1, ds.LATENCY_LIMIT_MS)
        with open(os.path.join(CODE_DIR, "main.py"), encoding="utf-8") as f:
            src = f.read()
        self.assertIn("force_hud = True", src)
        self.assertIn("display_status.clip(mission.response", src)


class WeakNetwork(unittest.TestCase):
    """Updates reach the display late, jittered, out of order, or not at all."""

    def simulate(self, delay, jitter, loss, seed, send_hz=20, seconds=30):
        rng = random.Random(seed)
        n_send = int(send_hz * seconds)
        sent = []                                           # (seq, send_t)
        wire = []                                           # (arrival_t, seq)
        for seq in range(n_send):
            t = seq / send_hz
            sent.append((seq, t))
            if rng.random() >= loss:
                wire.append((t + delay + rng.uniform(0, jitter), seq))
        wire.sort()                                         # arrivals: may be out of order
        feed, drawn, latencies = ds.StateFeed(), -1, []
        i, shown_seq = 0, -1
        n_ticks = int((seconds + delay + jitter + 1) / TICK)
        for tick in range(n_ticks):
            now = tick * TICK
            while i < len(wire) and wire[i][0] <= now:
                feed.accept(wire[i][1], {"seq": wire[i][1]}, wire[i][0])
                i += 1
            if tick % HUD_EVERY == 0 and feed.seq != shown_seq and feed.seq >= 0:
                latencies.append((now - feed.arrived_at) * 1000)
                self.assertGreater(feed.seq, shown_seq)     # display never goes backwards
                shown_seq = feed.seq
        return feed, latencies, shown_seq

    def test_profiles(self):
        for name, delay, jitter, loss in (("good", 0.02, 0.01, 0.0),
                                          ("weak 3G", 0.3, 0.2, 0.1),
                                          ("very weak", 0.8, 0.6, 0.3)):
            feed, lat, shown = self.simulate(delay, jitter, loss, seed=3)
            print(f"\n  {name}: delay {delay * 1000:.0f}+{jitter * 1000:.0f} ms, loss {loss:.0%}: "
                  f"{len(lat)} frames, worst arrival->screen {max(lat):.1f} ms, "
                  f"stale updates ignored {feed.dropped_stale}")
            self.assertLess(max(lat), ds.LATENCY_LIMIT_MS)   # once data arrives, it is drawn fast
            self.assertEqual(shown, feed.seq)                # ends on the newest state received

    def test_out_of_order_and_duplicate_ignored(self):
        f = ds.StateFeed()
        self.assertTrue(f.accept(5, "new", 1.0))
        self.assertFalse(f.accept(3, "old", 1.1))
        self.assertFalse(f.accept(5, "dup", 1.2))
        self.assertEqual(f.state, "new")
        self.assertEqual(f.dropped_stale, 2)


class FullMission(unittest.TestCase):
    """AC-04: START to FOUND on one set of objects; every frame shows all six fields."""

    def test_full_mission_without_restart(self):
        m, nav, mon = Mission(), FakeNav(), FakeMonitor()
        mission_id = id(m)
        state = {"frame": 0}
        frames, states_seen = [], []

        def snap():
            f = fields_of(m, nav, mon)
            self.assertEqual(set(f), set(ds.FIELD_NAMES))
            self.assertTrue(all(isinstance(v, str) and v for v in f.values()), f)
            frames.append(ds.signature(f))
            states_seen.append(f["mission"])

        def run(seconds):
            for _ in range(int(seconds * 60)):
                if nav.state == "NAV":
                    nav.pos[0] += 0.03
                state["frame"] += 1
                snap()

        snap()                                              # IDLE, before START
        m.start()
        nav.state, nav.horizontal_speed = "NAV", 2.0
        for _ in range(len(m.plan)):                        # fly every zone
            run(5)
            nav.state, nav.horizontal_speed = "DONE", 0.0
            run(0.5)
            m.arrived()
            nav.state, nav.horizontal_speed = "NAV", 2.0
        mon.state, mon.confidence = "SIGHTING", 0.4
        run(1)
        mon.state, mon.confidence = "FOUND", 0.9
        m.found()
        nav.state, nav.horizontal_speed = "FOUND", 0.0
        run(1)

        self.assertEqual(id(m), mission_id)
        self.assertEqual(states_seen[0], "IDLE")
        self.assertEqual(states_seen[-1], "FOUND")
        self.assertIn("SEARCHING", states_seen)
        self.assertGreater(len(set(frames)), 10)            # the display really changed
        self.assertEqual(m.visited and len(m.visited), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
