"""
PBI-07 tests: the operator dashboard stays responsive under heavy live input.
Run from the Sprint folder:  python -m unittest tests.test_ui_responsiveness_pbi07 -v

Uses the REAL Tk operator panel (skipped if Tk / a display is not available), the real Mission,
and a large probability map (200 x 200 cells). The route re-plan is replaced by a deliberately slow
stand-in (BUSY_REPLAN_MS of CPU work, longer than the 5-27 ms measured in the test world), so the
tests show the UI does not wait for it. One test uses the real Navigator in the OSM fixture world.

Clicks are real Tk events (event_generate on the panel's buttons), handled in the same order as
main.py: poll -> stage 1 (mission) -> refresh -> flush (drawn) -> stage 2 (map + re-plan).

Covers
  AC-01  live data keeps coming (message / event bursts and a 10 s stream): no lag, no crash
  AC-02  a guidance button click is processed and drawn within 50 ms
  AC-03  under heavy input, drone-display changes reach the screen within 50 ms
"""
import os
import random
import re
import sys
import time
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import display_status as ds
import operator_panel
from operator_guidance import Cmd, Mission, MissionEvent, ZONES, ZONE_COMMANDS
from probability_map import ProbabilityMap
from ui_responsiveness import RESPONSE_LIMIT_MS, CommandPipeline, ResponseTimer

TICK = 1.0 / 60.0                       # main.py render tick
BUSY_REPLAN_MS = 40.0                   # slow stand-in for the A* re-plan
MAP_HALF = 13.0


def _main_const(name):
    """Read a constant from main.py's source (importing main needs PyBullet)."""
    with open(os.path.join(CODE_DIR, "main.py"), encoding="utf-8") as f:
        m = re.search(rf"^{name}\s*=\s*(\d+)", f.read(), re.M)
    assert m, name
    return int(m.group(1))


def busy(ms):
    """CPU work for ms milliseconds (worse than sleep: it holds the interpreter)."""
    end = time.perf_counter() + ms / 1000.0
    x = 0
    while time.perf_counter() < end:
        x += 1
    return x


def zone_rect(z):
    x, y = ZONES[z]
    return (x - 4.0, y - 4.0, x + 4.0, y + 4.0)


def apply_map(belief):
    """Same map update as main.apply_command_map (without importing PyBullet)."""
    def run(res):
        belief.apply_guidance(res.command, res.zone, zone_rect(res.zone) if res.zone in ZONES else None,
                              res.sighting, sighting_sigma=2.5, sighting_weight=0.5)
    return run


def heavy_belief():
    return ProbabilityMap(x_range=(-MAP_HALF, MAP_HALF), y_range=(-MAP_HALF, MAP_HALF),
                          cell_size=2 * MAP_HALF / 200)          # 200 x 200 = 40 000 cells


def event(i, kind="UNSAFE", severity="WARNING"):
    return MissionEvent(kind=kind, severity=severity, source="obstacle", drone_position=(i % 7, 1, 2),
                        object_position=None, distance=0.3 + (i % 10) / 10, timestamp=i * 0.1,
                        response="re-planning the route", message=f"obstacle {i}")


class FakeNav:
    def __init__(self):
        self.pos, self.state, self.horizontal_speed = [0.0, 0.0, 2.5], "NAV", 2.0


class FakeMonitor:
    state, confidence = "SEARCHING", 0.1


class FakeSearcher:
    active, target = False, None


def open_panel():
    panel, why = operator_panel.try_open(ZONES, list(Cmd), ZONE_COMMANDS,
                                         directions={z: z for z in ZONES})
    return panel, why


class PanelCase(unittest.TestCase):
    """Opens one real panel per test (skipped without Tk / a display)."""

    def setUp(self):
        self.panel, why = open_panel()
        if self.panel is None:
            self.skipTest(f"operator panel unavailable ({why})")
        self.mission = Mission()
        self.mission.start()
        self.nav, self.mon = FakeNav(), FakeMonitor()
        self.selected = "A"
        self.status = lambda z: ("searched" if z in self.mission.visited else
                                 "excluded" if z in self.mission.excluded else
                                 "next" if self.mission.plan[:1] == [z] else "planned")
        self.refresh()                                   # first refresh sizes the window (slow once)
        self.panel.flush()
        # Let the window map and paint once (as in main.py, where the panel is on screen long
        # before the mission starts): the first full paint is a one-time cost, not a refresh.
        for _ in range(3):
            self.panel.poll()

    def tearDown(self):
        if getattr(self, "panel", None) is not None:
            self.panel.close()

    def fields(self):
        return ds.fields_from_sim(self.mission, self.nav, self.mon, FakeSearcher())

    def refresh(self, last=None):
        f = self.fields()
        self.panel.refresh(self.mission, self.selected, self.status, last, "locked",
                           ds.panel_text(f))
        return ds.signature(f)

    def click(self, widget):
        """A real mouse click on a panel widget, delivered through Tk's bindings."""
        widget.event_generate("<Button-1>")

    def handle_tick(self, pipeline, timer):
        """main.py's panel block: poll, stage 1, refresh, flush, then stage 2."""
        events = self.panel.poll()
        t_input = self.panel.input_t
        last = None
        for ev in events:
            if ev[0] == "select":
                self.selected = ev[1]
            elif ev[0] == "cmd":
                cmd = ev[1]
                last = pipeline.submit(cmd, self.selected if cmd in ZONE_COMMANDS else None)
        self.refresh(last)
        self.panel.flush()
        if events:
            timer.add("click", (time.perf_counter() - t_input) * 1000.0)
        pipeline.run_effects()
        return events, last


# ------------------------------------------------------------------------------------- AC-02 --
class ClickResponse(PanelCase):
    def pipeline(self, belief, replans):
        def follow():
            replans.append(tuple(self.mission.destination or ()))
            busy(BUSY_REPLAN_MS)
        return CommandPipeline(self.mission, apply_map=apply_map(belief), follow=follow)

    def test_guidance_clicks_are_drawn_within_50ms_with_a_slow_replan(self):
        belief, replans, timer = heavy_belief(), [], ResponseTimer()
        pipe = self.pipeline(belief, replans)
        rng = random.Random(7)
        zone_cmds = [c for c in Cmd if c in ZONE_COMMANDS]
        for i in range(40):
            self.click(self.panel.zone_lbls[rng.choice(list(ZONES))])     # select a zone
            self.handle_tick(pipe, timer)
            cmd = rng.choice(zone_cmds + [Cmd.RETURN_PLAN])
            self.click(self.panel.cmd_btns[cmd])                            # press a command
            _ev, res = self.handle_tick(pipe, timer)
            self.assertIsNotNone(res)
            # the reply on screen is the one for this click (accepted or rejected)
            self.assertEqual(self.mission.response, res.message)
            self.assertEqual(self.panel.result_lbl.cget("text"), res.message)
        d = timer.summary()["click"]
        print(f"\n  {d['count']} clicks (re-plan {BUSY_REPLAN_MS:g} ms each, map 200x200): click -> "
              f"drawn mean {d['mean_ms']:.1f} ms, p95 {d['p95_ms']:.1f} ms, worst {d['max_ms']:.1f} ms"
              f"; {len(replans)} re-plans ran after the draw")
        self.assertEqual(d["over_limit"], 0)
        self.assertLess(d["max_ms"], RESPONSE_LIMIT_MS)
        self.assertGreater(len(replans), 0)                 # the heavy work still happened

    def test_reply_is_drawn_before_the_heavy_work_runs(self):
        """The order matters: if the re-plan ran first, the click would wait for it."""
        drawn_at, replan_at = [], []
        belief = heavy_belief()

        def follow():
            replan_at.append(time.perf_counter())
            busy(BUSY_REPLAN_MS)
        pipe = CommandPipeline(self.mission, apply_map=apply_map(belief), follow=follow)
        self.selected = self.mission.plan[0]
        self.click(self.panel.cmd_btns[Cmd.AVOID])          # avoid the current destination: re-plan
        events = self.panel.poll()
        pipe.submit(events[0][1], self.selected)
        self.refresh()
        self.panel.flush()
        drawn_at.append(time.perf_counter())
        self.assertIn(f"Zone {self.selected}", self.panel.result_lbl.cget("text"))
        self.assertTrue(pipe.pending)
        pipe.run_effects()
        self.assertEqual(len(replan_at), 1)
        self.assertLess(drawn_at[0], replan_at[0])

    def test_old_order_vs_new_order(self):
        """Before PBI-07: heavy work, then refresh, drawn at the next poll (up to 50 ms later).
        After: refresh + draw first. Same click, same work."""
        def measure(new_order):
            m = Mission()
            m.start()
            self.mission = m
            belief = heavy_belief()
            pipe = CommandPipeline(m, apply_map=apply_map(belief),
                                   follow=lambda: busy(BUSY_REPLAN_MS))
            self.selected = m.plan[0]
            self.click(self.panel.cmd_btns[Cmd.AVOID])
            events = self.panel.poll()
            t0 = self.panel.input_t
            pipe.submit(events[0][1], self.selected)
            if not new_order:
                pipe.run_effects()
            self.refresh()
            if not new_order:
                busy(0.5 * 3 * TICK * 1000)                 # average wait for the old 20 Hz poll
            self.panel.flush()
            ms = (time.perf_counter() - t0) * 1000.0
            pipe.run_effects()
            return ms
        old, new = measure(False), measure(True)
        print(f"\n  click -> drawn: old order {old:.1f} ms, new order {new:.1f} ms")
        self.assertGreater(old, RESPONSE_LIMIT_MS)
        self.assertLess(new, RESPONSE_LIMIT_MS)

    def test_worst_case_including_the_wait_for_the_next_poll(self):
        """A click can arrive just after a poll; it then waits one poll interval. That wait plus
        the measured handling must still be under 50 ms."""
        poll_every = _main_const("BUTTON_POLL_EVERY")
        wait_ms = poll_every * TICK * 1000.0
        belief, timer = heavy_belief(), ResponseTimer()
        pipe = CommandPipeline(self.mission, apply_map=apply_map(belief),
                               follow=lambda: busy(BUSY_REPLAN_MS))
        for cmd in [Cmd.INVESTIGATE, Cmd.AVOID, Cmd.REINCLUDE, Cmd.SEARCH_FIRST, Cmd.RETURN_PLAN] * 6:
            self.click(self.panel.cmd_btns[cmd])
            self.handle_tick(pipe, timer)
        worst = timer.summary()["click"]["max_ms"] + wait_ms
        print(f"\n  poll wait <= {wait_ms:.1f} ms + worst handling "
              f"{timer.summary()['click']['max_ms']:.1f} ms = {worst:.1f} ms")
        self.assertLess(worst, RESPONSE_LIMIT_MS)


class CommandStages(unittest.TestCase):
    """CommandPipeline on its own (no Tk): same end state as handling commands one at a time."""

    SEQ = [(Cmd.INVESTIGATE, "B"), (Cmd.AVOID, "A"), (Cmd.SEARCH_FIRST, "C"), (Cmd.AVOID, "Z"),
           (Cmd.REINCLUDE, "A"), (Cmd.AVOID, "D"), (Cmd.INVESTIGATE, "C"), (Cmd.RETURN_PLAN, None),
           ("garbage", "A"), (Cmd.AVOID, "B"), (Cmd.SEARCH_FIRST, "D")]

    def test_burst_of_clicks_shares_one_replan_and_ends_in_the_same_state(self):
        # one at a time (the old path): map + re-plan after every command
        m1, b1, routes1 = Mission(), heavy_belief(), []
        m1.start()
        for cmd, z in self.SEQ:
            res = m1.handle(cmd, z)
            if res.ok:
                apply_map(b1)(res)
                if res.reroute:
                    routes1.append(tuple(m1.destination or ()))
        # all in one poll through the pipeline
        m2, b2, routes2 = Mission(), heavy_belief(), []
        m2.start()
        pipe = CommandPipeline(m2, apply_map=apply_map(b2),
                               follow=lambda: routes2.append(tuple(m2.destination or ())))
        for cmd, z in self.SEQ:
            pipe.submit(cmd, z)
        pipe.run_effects()
        self.assertEqual(m2.snapshot()["plan"], m1.snapshot()["plan"])
        self.assertEqual(m2.excluded, m1.excluded)
        self.assertEqual([h[1:] for h in m2.history], [h[1:] for h in m1.history])
        self.assertTrue((b1.grid == b2.grid).all())                  # every map update applied
        self.assertEqual(routes2, routes1[-1:])                      # one re-plan, to the newest
        self.assertEqual(pipe.replans, 1)
        self.assertEqual(pipe.replans_saved, len(routes1) - 1)

    def test_rejected_command_queues_nothing(self):
        m = Mission()
        m.start()
        pipe = CommandPipeline(m, apply_map=lambda r: self.fail("map"), follow=lambda: self.fail("route"))
        for cmd, z in ((Cmd.AVOID, "Z"), ("nonsense", None), (Cmd.AVOID, None), (None, None)):
            res = pipe.submit(cmd, z)
            self.assertFalse(res.ok)
        self.assertFalse(pipe.pending)
        self.assertEqual(pipe.run_effects(), 0.0)

    def test_stage_one_is_fast_under_load(self):
        m = Mission()
        m.start()
        for i in range(3000):                                        # full history
            m.notify("DETECT", f"scan {i}")
        pipe = CommandPipeline(m)
        t0 = time.perf_counter()
        for cmd, z in self.SEQ * 50:
            pipe.submit(cmd, z)
        per = (time.perf_counter() - t0) * 1000.0 / (len(self.SEQ) * 50)
        print(f"\n  stage 1 (validate + mission update): {per:.3f} ms per command")
        self.assertLess(per, 1.0)

    def test_deferred_work_and_start_reroute_run_in_stage_two(self):
        m = Mission()
        ran = []
        pipe = CommandPipeline(m, follow=lambda: ran.append("route"))
        pipe.defer(lambda: ran.append("heatmap"))
        pipe.request_follow()
        pipe.request_follow()                                        # coalesced
        self.assertEqual(ran, [])
        pipe.run_effects()
        self.assertEqual(ran, ["heatmap", "route"])
        self.assertFalse(pipe.pending)


# ------------------------------------------------------------------------------------- AC-01 --
class LiveDataBursts(PanelCase):
    def timed_refresh(self):
        t0 = time.perf_counter()
        self.panel.poll()
        self.refresh()
        self.panel.flush()
        return (time.perf_counter() - t0) * 1000.0

    def test_message_bursts_refresh_within_50ms(self):
        for burst in (10, 100, 1000, 2000, 5000):
            for i in range(burst):
                self.mission.notify("DETECT", f"burst {burst} scan {i} confidence 0.{i % 100:02d}")
            ms = self.timed_refresh()
            lines = int(float(self.panel.log.index("end-1c")))
            print(f"\n  {burst:5d} new messages -> panel refresh + draw {ms:6.1f} ms "
                  f"(log holds {lines} lines)", end="")
            self.assertLess(ms, RESPONSE_LIMIT_MS, f"burst of {burst}")
            self.assertLessEqual(lines, operator_panel.LOG_KEEP)
            self.assertIn(f"scan {burst - 1} ", self.panel.log.get("end-2l", "end"))   # newest shown
        print()

    def test_event_bursts_refresh_within_50ms(self):
        for burst in (50, 200, 1000):
            for i in range(burst):
                self.mission.add_event(event(i, severity=("INFO", "WARNING", "CRITICAL")[i % 3]))
            ms = self.timed_refresh()
            rows = self.panel.event_list.get("1.0", "end").strip().splitlines()
            print(f"\n  {burst:5d} new events -> panel refresh + draw {ms:6.1f} ms", end="")
            self.assertLess(ms, RESPONSE_LIMIT_MS)
            self.assertEqual(len(rows), operator_panel.EVENT_ROWS)
        print()

    def test_odd_messages_do_not_crash_the_panel(self):
        for msg in ("", " ", "x" * 5000, "üñíçødé ✓ 🚁", "line\nbreak\ttab", "{curly} [square] $dollar",
                    "\\backslash\\", "Rejected: abort " * 100):
            self.mission.notify("DETECT", msg)
            self.mission.response = msg
            self.timed_refresh()
        self.assertFalse(self.panel.closed)

    def test_continuous_stream_does_not_lag(self):
        """10 s at 60 Hz: 20 messages + 1 event per tick (1200 messages/s), the drone moving every
        tick, a probability-map update every tick, the operator clicking twice a second."""
        belief, timer, rng = heavy_belief(), ResponseTimer(), random.Random(3)
        pipe = CommandPipeline(self.mission, apply_map=apply_map(belief),
                               follow=lambda: busy(BUSY_REPLAN_MS))
        refresh_every = _main_const("PANEL_REFRESH_EVERY")
        cmds = [Cmd.INVESTIGATE, Cmd.AVOID, Cmd.REINCLUDE, Cmd.SEARCH_FIRST, Cmd.RETURN_PLAN]
        for tick in range(600):
            for k in range(20):
                self.mission.notify("DETECT", f"t{tick} scan {k} conf 0.{rng.randrange(100):02d}")
            self.mission.add_event(event(tick))
            self.nav.pos[0] = 10 * ((tick % 120) / 120.0) - 5
            belief.add_sighting((rng.uniform(-10, 10), rng.uniform(-10, 10)), sigma=2.0, weight=0.1)
            if tick % 30 == 15:
                self.click(self.panel.cmd_btns[rng.choice(cmds)])
            t0 = time.perf_counter()
            events = self.panel.poll()
            if events or tick % refresh_every == 0:
                t_input = self.panel.input_t
                for ev in events:
                    if ev[0] == "cmd":
                        pipe.submit(ev[1], self.selected if ev[1] in ZONE_COMMANDS else None)
                self.refresh()
                self.panel.flush()
                timer.add("ui tick", (time.perf_counter() - t0) * 1000.0)
                if events:
                    timer.add("click", (time.perf_counter() - t_input) * 1000.0)
                pipe.run_effects()
            self.assertFalse(self.panel.closed)
        print("\n  " + timer.report())
        for name, d in timer.summary().items():
            self.assertEqual(d["over_limit"], 0, name)
        self.assertEqual(timer.summary()["click"]["count"], 20)
        self.assertLessEqual(len(self.mission.history), 2000)


# ------------------------------------------------------------------------------------- AC-03 --
class DroneDisplayUnderLoad(PanelCase):
    def test_drone_display_changes_reach_the_screen_within_50ms(self):
        """Real-time loop at main.py's cadence with heavy input on every tick. A LatencyTracker
        (limit 50 ms) times each change of the six drone-display fields until the panel draws it."""
        refresh_every = _main_const("PANEL_REFRESH_EVERY")
        belief, rng = heavy_belief(), random.Random(11)
        tracker = ds.LatencyTracker(limit_ms=RESPONSE_LIMIT_MS)
        next_tick = time.perf_counter()
        for tick in range(240):                                       # 4 s
            # heavy live input
            for k in range(20):
                self.mission.notify("DETECT", f"t{tick} scan {k}")
            self.mission.add_event(event(tick))
            belief.add_sighting((rng.uniform(-10, 10), rng.uniform(-10, 10)), sigma=2.0, weight=0.1)
            # the drone moves; the state / sighting change now and then
            self.nav.pos[0] += 0.12
            self.nav.pos[1] = 3 * ((tick % 40) / 40.0)
            if tick % 25 == 0:
                self.mon.state = "SIGHTING" if self.mon.state == "SEARCHING" else "SEARCHING"
            tracker.observe(ds.signature(self.fields()), surfaces=("panel",))
            self.panel.poll()
            if tick % refresh_every == 0:
                sig = self.refresh()
                self.panel.flush()
                tracker.shown("panel", sig)
                self.assertIn(f"x {self.nav.pos[0]:.1f}", self.panel.status_lbl.cget("text"))
            next_tick += TICK
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.perf_counter()
        d = tracker.summary()["panel"]
        print(f"\n  drone display: {d['count']} updates, mean {d['mean_ms']:.1f} ms, "
              f"p95 {d['p95_ms']:.1f} ms, worst {d['max_ms']:.1f} ms (limit {RESPONSE_LIMIT_MS:g} ms)")
        self.assertGreater(d["count"], 100)
        self.assertEqual(d["over_limit"], 0)

    def test_hud_and_panel_cadence_fit_50ms(self):
        hud = _main_const("HUD_EVERY") * TICK * 1000
        panel = _main_const("PANEL_REFRESH_EVERY") * TICK * 1000
        poll = _main_const("BUTTON_POLL_EVERY") * TICK * 1000
        print(f"\n  worst-case wait: HUD {hud:.1f} ms, panel status {panel:.1f} ms, click poll {poll:.1f} ms")
        for ms in (hud, panel, poll):
            self.assertLess(ms, RESPONSE_LIMIT_MS)


# ------------------------------------------------------- AC-01: saves off the UI thread --
class BackgroundSaves(unittest.TestCase):
    """mission_state.BackgroundSaver: the UI thread only copies the state; a worker writes it."""

    def setUp(self):
        import tempfile
        from mission_state import MissionStateStore
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MissionStateStore(os.path.join(self.tmp.name, "mission_state.json"))
        self.mission = Mission()
        self.mission.start()
        for i in range(2000):                                      # full history
            self.mission.notify("DETECT", f"scan {i}")
        self.belief = heavy_belief()

    def tearDown(self):
        self.tmp.cleanup()

    def stream(self, saver, ticks=120):
        """A new live message every tick (so the mission changes every tick); UI-thread ms."""
        times = []
        for i in range(ticks):
            self.mission.notify("DETECT", f"live {i}")
            t0 = time.perf_counter()
            saver.autosave(self.mission, (i * 0.1, 0, 2.5), belief=self.belief, sim_t=i / 60)
            times.append((time.perf_counter() - t0) * 1000.0)
            time.sleep(TICK / 4)
        return times

    def test_ui_thread_cost_inline_vs_background(self):
        from mission_state import BackgroundSaver
        inline = self.stream(self.store, ticks=40)
        saver = BackgroundSaver(self.store)
        try:
            bg = self.stream(saver)
        finally:
            self.assertTrue(saver.close())
        mean = lambda v: sum(v) / len(v)
        print(f"\n  autosave on the UI thread (2000-line history, 200x200 map): inline mean "
              f"{mean(inline):.1f} ms / worst {max(inline):.1f} ms -> background mean "
              f"{mean(bg):.2f} ms / worst {max(bg):.2f} ms; {saver.queued} queued, "
              f"{saver.coalesced} merged into a newer save, {self.store.saves} files written")
        # PBI-05: the worst single tick depends on when the save thread lets go of Python's GIL
        # (failed ~2 runs in 5 on a 2-core machine with a 10 ms limit). Check the typical tick
        # (95th percentile) against 10 ms and the worst one against the 50 ms PBI-07 click budget.
        p95 = sorted(bg)[int(0.95 * (len(bg) - 1))]
        self.assertLess(p95, 10.0)
        self.assertLess(max(bg), 50.0)
        self.assertLess(mean(bg), mean(inline))
        self.assertLess(self.store.saves, 40 + 120)                # bursts were merged

    def test_newest_state_is_on_disk_after_close_and_loads(self):
        from mission_state import BackgroundSaver
        saver = BackgroundSaver(self.store)
        self.stream(saver, ticks=30)
        self.mission.handle(Cmd.AVOID, "B")
        saver.autosave(self.mission, (9, 9, 2.5), force=True, belief=self.belief, sim_t=99.0)
        saver.close()
        got = self.store.load()
        self.assertEqual(got.source, "main", got.problems)
        self.assertEqual(got.payload["mission"]["excluded"], self.mission.excluded)
        self.assertEqual(got.payload["mission"]["history"][-1][2], self.mission.history[-1][2])
        self.assertEqual(got.payload["drone"]["pos"], [9.0, 9.0, 2.5])
        self.assertEqual(got.payload["belief"]["shape"], list(self.belief.grid.shape))

    def test_saved_copy_is_not_changed_by_later_updates(self):
        """The worker writes the state as it was when autosave() was called."""
        from mission_state import BackgroundSaver
        saver = BackgroundSaver(self.store)
        saver.autosave(self.mission, (1, 1, 2.5), force=True, belief=self.belief, sim_t=1.0)
        saved_grid = list(self.belief.grid.ravel()[:50])
        self.belief.grid[:] = 0.0
        self.mission.notify("DETECT", "after the save")
        saver.close()
        got = self.store.load().payload
        self.assertEqual(got["belief"]["grid"][:50], saved_grid)
        self.assertNotEqual(got["mission"]["history"][-1][2], "after the save")

    def test_disk_failure_does_not_crash_or_block(self):
        from mission_state import BackgroundSaver, MissionStateStore
        store = MissionStateStore(os.path.join(self.tmp.name, "no", "such", "dir", "s.json"))
        saver = BackgroundSaver(store)
        t0 = time.perf_counter()
        saver.autosave(self.mission, (0, 0, 2.5), belief=self.belief)
        self.assertLess((time.perf_counter() - t0) * 1000.0, 10.0)
        saver.close()
        self.assertIsNotNone(store.last_error)


# ------------------------------------------------------------------------- main.py wiring --
class MainWiring(unittest.TestCase):
    """main.py imports PyBullet, so its loop is checked from source: the panel reply is drawn
    (flush) before the heavy stage runs, commands go through the pipeline, and saves are moved
    to the background saver."""

    @classmethod
    def setUpClass(cls):
        with open(os.path.join(CODE_DIR, "main.py"), encoding="utf-8") as f:
            cls.src = f.read()

    def test_reply_drawn_before_heavy_stage(self):
        loop = self.src[self.src.index("render_tick = step_count // render_every"):]
        self.assertLess(loop.index("pipeline.submit("), loop.index("panel.refresh("))
        self.assertLess(loop.index("panel.refresh("), loop.index("panel.flush()"))
        self.assertLess(loop.index("panel.flush()"), loop.index("pipeline.run_effects()"))

    def test_commands_no_longer_replan_inside_the_click(self):
        loop = self.src[self.src.index("render_tick = step_count // render_every"):]
        self.assertNotIn("apply_operator_command(", loop)
        self.assertIn("pipeline.request_follow()", loop)             # START plans in stage 2

    def test_heatmap_toggle_and_repaint_are_batched(self):
        loop = self.src[self.src.index("render_tick = step_count // render_every"):]
        self.assertIn("set_heatmap_visible(not belief.heatmap_visible, repaint=False)", loop)
        self.assertIn("draw_heatmap(budget=None, pause_rendering=True)", loop)
        self.assertGreater(_main_const("HEATMAP_EVERY"), 1)

    def test_saves_go_through_the_background_saver(self):
        self.assertIn("BackgroundSaver(store) if gui else store", self.src)
        loop = self.src[self.src.index("next_tick = time.perf_counter()"):]
        self.assertNotIn("store.autosave(", loop)
        self.assertIn("saver.close()", loop)


if __name__ == "__main__":
    unittest.main()
