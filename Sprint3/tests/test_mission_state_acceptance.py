"""
File Description: Tests for PBI-10 (maintain mission state so a crash does not lose it).
No PyBullet needed:   python -m unittest tests.test_mission_state_acceptance

Acceptance criteria covered
  AC-01  state is saved within 1 second of any change      -> TestSaveTiming, TestAutosave
  AC-02  after stopping mid-mission, restart loads the last saved state
                                                           -> TestCrashRecovery (real subprocesses,
                                                              killed with os._exit / SIGKILL)
  AC-03  commands before start / after end are rejected with a message, no crash
                                                           -> TestCommandRejection
  AC-04  every operator command and system response is in the history log
                                                           -> TestHistory
Plus: damaged / tampered / oversized files are detected and never crash the program (TestLoadSafety).
"""
import os
import sys
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import time
import unittest

import operator_guidance as og
from operator_guidance import Mission, Cmd, ZONES, ZONE_COMMANDS
from mission_state import (MissionStateStore, SAVE_BUDGET_S, PERIODIC_SAVE_S, MAX_FILE_BYTES,
                           validate_setup, setup_from_args, apply_setup_to_args)

HERE = CODE_DIR                    # where the simulation code is
SETUP = {"lat": 44.565296, "lon": -123.276016, "area": 60.0, "map_size": None,
         "osm_file": None, "seed": 0, "simple_visuals": False, "camera": "follow", "max_time": None}
SAVE_TIMES = []


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def new_store(tmp, clock=None, name="mission_state.json"):
    store = MissionStateStore(os.path.join(tmp, name), **({"clock": clock} if clock else {}))
    store.set_context(setup=SETUP, zones=ZONES)
    return store


def send(m, cmd, zone="B"):
    """Send an operator command the way the panel does (zone commands carry the selected zone)
    and return the reply text."""
    return m.handle(cmd, zone if cmd in ZONE_COMMANDS else None).message


def started_mission():
    m = Mission()
    m.start()
    return m


def normal(snapshot):
    """Snapshot as it looks after a JSON round trip (tuples -> lists)."""
    return json.loads(json.dumps(snapshot))


# ------------------------------------------------------------------------------ AC-01 -----
class TestSaveTiming(unittest.TestCase):
    def test_save_and_reload_round_trip_has_all_required_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started_mission()
            m.handle(Cmd.SEARCH_FIRST, "B")
            m.arrived()
            store.save(store.build(m, [1.0, 2.0, 2.5]))
            got = store.load()
            self.assertEqual(got.source, "main")
            p = got.payload
            self.assertEqual(p["mission"]["state"], "SEARCHING")          # state
            self.assertEqual(p["mission"]["plan"], m.plan)                # plan
            self.assertEqual(p["mission"]["visited"], ["B"])              # visited zones
            self.assertEqual(p["mission"]["destination"], list(m.destination))
            self.assertEqual(p["mission"]["response"], m.response)        # last response
            self.assertEqual(p["setup"]["lat"], SETUP["lat"])             # last known position
            self.assertEqual(p["setup"]["lon"], SETUP["lon"])
            self.assertEqual(p["drone"]["pos"], [1.0, 2.0, 2.5])
            self.assertEqual(p["mission"]["history"], normal(m.snapshot()["history"]))

    def test_save_takes_far_less_than_one_second_even_with_a_full_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started_mission()
            for i in range(og.HISTORY_LIMIT):                      # worst case: history at its cap
                m.record("CMD", f"message {i}")
            for _ in range(60):
                took = store.save(store.build(m, [0, 0, 2.5]), durable=True)
                SAVE_TIMES.append(took)
                self.assertLess(took, SAVE_BUDGET_S)
            self.assertLess(os.path.getsize(store.path), MAX_FILE_BYTES)

    def test_any_mission_change_is_saved_immediately_and_within_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), Mission()
            changes = [lambda: m.start(), lambda: m.handle(Cmd.SEARCH_FIRST, "B"),
                       lambda: m.handle(Cmd.AVOID, "C"), lambda: m.handle(Cmd.INVESTIGATE, "D"),
                       lambda: m.arrived(), lambda: m.handle(Cmd.RETURN_PLAN),
                       lambda: m.handle("junk"), lambda: m.found()]
            for i, change in enumerate(changes):
                change()
                t0 = time.perf_counter()
                wrote = store.autosave(m, [0, 0, 2.5])
                took = time.perf_counter() - t0
                SAVE_TIMES.append(took)
                self.assertTrue(wrote, f"change {i} was not saved")
                self.assertLess(took, SAVE_BUDGET_S)
                self.assertEqual(store.load().payload["mission"], normal(m.snapshot()))


class TestAutosave(unittest.TestCase):
    def test_no_write_when_nothing_changed(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store, m = new_store(tmp, clock), Mission()
            self.assertTrue(store.autosave(m, [0, 0, 1]))
            for _ in range(100):
                clock.t += 0.001
                self.assertFalse(store.autosave(m, [0, 0, 1]))     # IDLE: nothing to refresh
            self.assertEqual(store.saves, 1)

    def test_position_refresh_only_while_searching_and_at_most_twice_a_second(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store, m = new_store(tmp, clock), started_mission()
            store.autosave(m, [0, 0, 1])
            writes = 0
            for i in range(300):                                    # 3 simulated seconds at 100 Hz
                clock.t += 0.01
                writes += store.autosave(m, [i * 0.01, 0, 2.5])
            self.assertTrue(5 <= writes <= 7, writes)               # ~every PERIODIC_SAVE_S
            saved_x = store.load().payload["drone"]["pos"][0]    # lags by at most one refresh period
            self.assertGreaterEqual(saved_x, 2.99 - PERIODIC_SAVE_S - 0.02)
            self.assertLessEqual(saved_x, 2.99)
            m.found()
            store.autosave(m, [3, 0, 2.5])
            after = store.saves
            for _ in range(200):
                clock.t += 0.05
                store.autosave(m, [3, 0, 2.5])
            self.assertEqual(store.saves, after)                    # mission over: no more writes

    def test_save_failure_never_raises_and_does_not_spam(self):
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store = MissionStateStore(os.path.join(tmp, "missing_folder", "state.json"), clock=clock)
            store.set_context(setup=SETUP, zones=ZONES)
            m = started_mission()
            self.assertFalse(store.autosave(m, [0, 0, 1]))          # must not raise
            self.assertIsNotNone(store.last_error)
            m.handle(Cmd.AVOID, "C")
            self.assertFalse(store.autosave(m, [0, 0, 1]))
            os.makedirs(os.path.join(tmp, "missing_folder"))        # problem fixed -> works again
            m.handle(Cmd.RETURN_PLAN)
            self.assertTrue(store.autosave(m, [0, 0, 1]))
            self.assertIsNone(store.last_error)

    def test_no_temp_files_left_behind(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started_mission()
            for i in range(20):
                m.record("CMD", str(i))
                store.autosave(m, [0, 0, 1])
            names = sorted(os.listdir(tmp))
            self.assertEqual(names, ["mission_state.json", "mission_state.json.bak"])


# ------------------------------------------------------------------------------ AC-02 -----
CHILD = r"""
import json, os, sys
sys.path.insert(0, sys.argv[1])
from operator_guidance import Mission, Cmd
from mission_state import MissionStateStore
path, crash_at, unsaved = sys.argv[2], int(sys.argv[3]), sys.argv[4] == "1"
store = MissionStateStore(path)
store.set_context(setup={"lat": 44.5, "lon": -123.2, "area": 60.0}, zones={"A": [6, 6], "B": [-6, 6], "C": [-6, -6], "D": [6, -6]})
m = Mission()
steps = [lambda: m.start(), lambda: m.handle(Cmd.SEARCH_FIRST, "B"), lambda: m.arrived(),
         lambda: m.handle(Cmd.AVOID, "C"), lambda: m.handle(Cmd.INVESTIGATE, "D"), lambda: m.arrived(),
         lambda: m.handle("not a command"), lambda: m.found()]
store.autosave(m, [-10, 10, 2.5], force=True)
print("ACK " + json.dumps(m.snapshot()), flush=True)
for i, step in enumerate(steps):
    step()
    if unsaved and i == crash_at:
        os._exit(137)                       # dies with a change in memory that was never saved
    store.autosave(m, [i, i, 2.5])
    print("ACK " + json.dumps(m.snapshot()), flush=True)
    if not unsaved and i == crash_at:
        os._exit(137)                       # dies right after a saved change
"""

KILL_CHILD = r"""
import sys
sys.path.insert(0, sys.argv[1])
from operator_guidance import Mission
from mission_state import MissionStateStore
store = MissionStateStore(sys.argv[2])
store.set_context(setup={"lat": 44.5, "lon": -123.2, "area": 60.0}, zones={"A": [6, 6], "B": [-6, 6], "C": [-6, -6], "D": [6, -6]})
m = Mission(); m.start()
k = 0
while True:
    k += 1
    m.response = f"step {k}"
    m.record("TICK", f"step {k}")
    store.autosave(m, [k % 10, 0, 2.5])
    print(f"ACK {k}", flush=True)
"""


class TestCrashRecovery(unittest.TestCase):
    def run_child(self, tmp, crash_at, unsaved):
        path = os.path.join(tmp, "mission_state.json")
        proc = subprocess.run([sys.executable, "-c", CHILD, HERE, path, str(crash_at),
                               "1" if unsaved else "0"], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 137, proc.stderr)           # it really did crash
        acks = [line[4:] for line in proc.stdout.splitlines() if line.startswith("ACK ")]
        return path, json.loads(acks[-1])

    def test_restart_loads_last_saved_state_for_every_crash_point(self):
        for crash_at in range(8):
            for unsaved in (False, True):
                with self.subTest(crash_after_step=crash_at, change_was_unsaved=unsaved):
                    with tempfile.TemporaryDirectory() as tmp:
                        path, last_acked = self.run_child(tmp, crash_at, unsaved)
                        store = MissionStateStore(path)                # a "restart"
                        got = store.load()
                        self.assertIsNotNone(got.payload, got.problems)
                        self.assertEqual(got.payload["mission"], last_acked)
                        self.assertEqual(got.payload["setup"]["lat"], 44.5)
                        m = Mission()                                   # and it can be restored
                        m.restore(got.payload["mission"])
                        self.assertEqual(m.state, last_acked["state"])
                        self.assertEqual(m.plan, last_acked["plan"])
                        self.assertEqual(m.visited, last_acked["visited"])
                        self.assertEqual(m.response, last_acked["response"])
                        self.assertEqual(m.history[-1][1], "RESTORE")   # restore itself is logged

    def test_restored_mission_keeps_working(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, _ = self.run_child(tmp, 3, False)    # crashed mid-mission (SEARCHING)
            m = Mission()
            m.restore(MissionStateStore(path).load().payload["mission"])
            self.assertEqual(m.state, "SEARCHING")
            self.assertIn("Received", m.handle(Cmd.RETURN_PLAN).message)
            m.arrived()
            m.found()
            self.assertEqual(m.state, "FOUND")

    def test_hard_kill_at_random_moments_never_leaves_an_unreadable_state(self):
        rng = random.Random(10)
        for trial in range(20):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "mission_state.json")
                proc = subprocess.Popen([sys.executable, "-c", KILL_CHILD, HERE, path],
                                        stdout=subprocess.PIPE, text=True)
                want, last = rng.randint(3, 60), 0
                for line in proc.stdout:
                    last = int(line.split()[1])
                    if last >= want:
                        break
                proc.kill()                                  # SIGKILL: no cleanup of any kind
                proc.wait()
                got = MissionStateStore(path).load()
                self.assertIsNotNone(got.payload, f"trial {trial}: {got.problems}")
                saved_step = int(got.payload["mission"]["response"].split()[1])
                self.assertGreaterEqual(saved_step, last, f"trial {trial}: lost an acknowledged save")


# ------------------------------------------------------------------------------ AC-03 -----
class TestCommandRejection(unittest.TestCase):
    def test_commands_before_start_are_rejected_with_a_message(self):
        m = Mission()
        for cmd in Cmd:
            reply = send(m, cmd)
            self.assertTrue(reply.startswith("Rejected"), reply)
            self.assertIn("IDLE", reply)
            self.assertEqual(m.response, reply)
        self.assertEqual(m.state, "IDLE")
        self.assertEqual(m.plan, ["A", "B", "C", "D"])

    def test_commands_after_the_mission_ended_are_rejected(self):
        for end in ("FOUND", "ABORT", "COMPLETE"):
            with self.subTest(end=end):
                m = started_mission()
                if end == "FOUND":
                    m.found()
                else:
                    m.finish(end, "NAV", "ended")
                before = m.snapshot()["plan"]
                for cmd in Cmd:
                    reply = send(m, cmd)
                    self.assertTrue(reply.startswith("Rejected") and end in reply, reply)
                self.assertIn(end, m.start())              # START after the end says why
                self.assertEqual(m.plan, before)
                self.assertEqual(m.state, end)

    def test_odd_input_never_crashes_in_any_state(self):
        junk = [None, 5, 3.5, "", "START", "x" * 10_000, [], {}, object(), Cmd, b"bytes", float("nan")]
        for prep in ("idle", "searching", "found"):
            m = Mission()
            if prep != "idle":
                m.start()
            if prep == "found":
                m.found()
            for item in junk:
                reply = m.handle(item).message              # must not raise
                self.assertTrue(reply.startswith("Rejected"))
                reply = m.handle(Cmd.AVOID, item).message   # odd zone values too
                self.assertTrue(reply.startswith("Rejected"))

    def test_a_second_start_is_rejected(self):
        m = started_mission()
        self.assertIn("already started", m.start())


# ------------------------------------------------------------------------------ AC-04 -----
class TestHistory(unittest.TestCase):
    def test_every_command_and_system_response_is_logged(self):
        m = Mission()
        expected = 0

        def step(fn, grows=True):
            nonlocal expected
            n = len(m.history)
            fn()
            expected += 1 if grows else 0
            self.assertEqual(len(m.history), n + (1 if grows else 0))

        step(lambda: m.handle(Cmd.AVOID, "C"))                 # rejected: before start
        step(lambda: m.start())                             # START
        step(lambda: m.start())                             # rejected: second START
        step(lambda: m.handle(Cmd.SEARCH_FIRST, "B"))          # accepted command
        step(lambda: m.handle("garbage"))                   # rejected: unknown command
        step(lambda: m.arrived())                           # system: zone searched
        step(lambda: m.handle(Cmd.SEARCH_FIRST, "B"))          # rejected: zone already visited
        step(lambda: m.finish("COMPLETE", "NAV", "Search plan complete"))   # system response
        step(lambda: m.handle(Cmd.AVOID, "C"))                 # rejected: after the end
        self.assertEqual(len(m.history), expected)
        for ts, source, msg in m.history:
            self.assertIsInstance(ts, float)
            self.assertTrue(source and msg)
        self.assertEqual([h[1] for h in m.history][:4], ["AVOID C", "START", "START", "SEARCH_FIRST B"])
        self.assertEqual(m.history[5][1], "ARRIVED")
        self.assertEqual(m.history[7][1], "NAV")

    def test_history_survives_save_and_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started_mission()
            m.handle(Cmd.AVOID, "C")
            m.handle("bad")
            m.arrived()
            store.save(store.build(m))
            before = [list(h) for h in m.history]
            m2 = Mission()
            m2.restore(store.load().payload["mission"])
            self.assertEqual([list(h) for h in m2.history][:len(before)], before)

    def test_history_is_capped_not_unbounded(self):
        m = started_mission()
        for i in range(og.HISTORY_LIMIT + 500):
            m.record("CMD", str(i))
        self.assertEqual(len(m.history), og.HISTORY_LIMIT)
        self.assertEqual(m.history[-1][2], str(og.HISTORY_LIMIT + 499))


# ------------------------------------------------------------------------- file safety -----
class TestLoadSafety(unittest.TestCase):
    def good(self, tmp):
        store, m = new_store(tmp), started_mission()
        store.save(store.build(m, [0, 0, 2.5]))
        m.handle(Cmd.AVOID, "C")
        store.save(store.build(m, [1, 1, 2.5]))          # now main = newer, .bak = older
        return store, m

    def test_missing_file_is_not_an_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = new_store(tmp).load()
            self.assertIsNone(got.payload)
            self.assertEqual(got.problems, [])

    def test_truncated_main_file_falls_back_to_backup(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self.good(tmp)
            with open(store.path, "rb") as f:
                data = f.read()
            with open(store.path, "wb") as f:
                f.write(data[: len(data) // 2])
            got = store.load()
            self.assertEqual(got.source, "backup")
            self.assertIsNotNone(got.payload)
            self.assertTrue(any("damaged" in p for p in got.problems))

    def test_edited_file_is_caught_by_the_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self.good(tmp)
            with open(store.path) as f:
                text = f.read().replace("SEARCHING", "COMPLETE")
            with open(store.path, "w") as f:
                f.write(text)
            got = store.load()
            self.assertEqual(got.source, "backup")
            self.assertTrue(any("checksum" in p for p in got.problems))

    def test_both_files_bad_gives_nothing_and_no_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self.good(tmp)
            for p in (store.path, store.bak_path):
                with open(p, "wb") as f:
                    f.write(b"\x00\xff garbage")
            got = store.load()
            self.assertIsNone(got.payload)
            self.assertEqual(len(got.problems), 2)

    def test_wrong_version_and_empty_file_and_oversized_file_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self.good(tmp)
            os.remove(store.bak_path)
            with open(store.path) as f:
                doc = json.load(f)
            doc["version"] = 99
            with open(store.path, "w") as f:
                json.dump(doc, f)
            self.assertIn("version", " ".join(store.load().problems))
            with open(store.path, "w") as f:
                f.write("")
            self.assertIsNone(store.load().payload)
            with open(store.path, "wb") as f:
                f.write(b" " * (MAX_FILE_BYTES + 1))
            self.assertIn("too large", " ".join(store.load().problems))

    def test_structurally_invalid_but_correctly_signed_data_is_rejected(self):
        from mission_state import _checksum
        with tempfile.TemporaryDirectory() as tmp:
            store, m = self.good(tmp)
            os.remove(store.bak_path)
            bad_edits = [lambda p: p["mission"].update(state="DANCING"),
                         lambda p: p["mission"].update(plan=["A", "Z"]),
                         lambda p: p["mission"].update(plan=["A"], visited=["A"]),
                         lambda p: p["mission"].update(destination=[1]),
                         lambda p: p["mission"].update(history=[[1, 2, 3]]),
                         lambda p: p["setup"].update(lat=123.0),
                         lambda p: p["drone"].update(pos=[1e99, 0, 0]),
                         lambda p: p.update(zones={"A": [0, 0]})]
            for i, edit in enumerate(bad_edits):
                with self.subTest(edit=i):
                    store.save(store.build(m, [0, 0, 2.5]))
                    with open(store.path) as f:
                        doc = json.load(f)
                    edit(doc["payload"])
                    doc["checksum"] = _checksum(doc["payload"])     # a valid signature on bad data
                    with open(store.path, "w") as f:
                        json.dump(doc, f)
                    os.path.exists(store.bak_path) and os.remove(store.bak_path)
                    got = store.load()
                    self.assertIsNone(got.payload)
                    self.assertTrue(got.problems)

    def test_random_corruption_never_raises(self):
        rng = random.Random(5)
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self.good(tmp)
            with open(store.path, "rb") as f:
                original = f.read()
            for _ in range(300):
                data = bytearray(original)
                for _ in range(rng.randint(1, 8)):
                    data[rng.randrange(len(data))] = rng.randrange(256)
                if rng.random() < 0.3:
                    data = data[: rng.randrange(len(data))]
                with open(store.path, "wb") as f:
                    f.write(bytes(data))
                got = store.load()                                  # must not raise
                self.assertTrue(got.payload is None or got.payload["mission"]["state"] in og.STATES)

    def test_archive_previous_keeps_the_old_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = self.good(tmp)
            self.assertTrue(store.archive_previous())
            self.assertFalse(os.path.exists(store.path))
            self.assertTrue(os.path.exists(store.prev_path))
            self.assertIsNone(store.load().payload)
            self.assertFalse(store.archive_previous())


class TestRestoreAndSetup(unittest.TestCase):
    def test_restore_rejects_bad_snapshots_and_changes_nothing(self):
        m = started_mission()
        m.handle(Cmd.AVOID, "C")
        before = m.snapshot()
        for bad in (None, {}, {"state": "SEARCHING"}, {**before, "plan": ["Q"]},
                    {**before, "response": 5}, {**before, "state": "nope"}):
            with self.assertRaises(ValueError):
                m.restore(bad)
            self.assertEqual(m.snapshot(), before)

    def test_restore_recomputes_destination_and_notes_a_changed_map(self):
        m = started_mission()
        snap = normal(m.snapshot())
        saved_zones = dict(ZONES)
        try:
            og.scale_zones(2.0)                                  # the map is bigger this run
            m2 = Mission()
            note = m2.restore(snap)
            self.assertIn("zone centres changed", note)
            self.assertEqual(m2.destination, ZONES["A"])
        finally:
            ZONES.update(saved_zones)

    def test_setup_round_trips_through_args(self):
        args = argparse.Namespace(lat=44.5, lon=-123.2, area=60.0, map_size=None, osm_file=None,
                                  seed=3, simple_visuals=True, camera="free", max_time=300.0)
        setup = validate_setup(setup_from_args(args))
        blank = argparse.Namespace()
        apply_setup_to_args(blank, setup)
        for key in ("lat", "lon", "area", "map_size", "osm_file", "seed", "simple_visuals",
                    "camera", "max_time"):
            self.assertEqual(getattr(blank, key), getattr(args, key), key)

    def test_validate_setup_rejects_bad_values(self):
        for bad in (None, [], {"lat": "x", "lon": 1}, {"lat": 91, "lon": 0}, {"lat": 1, "lon": None},
                    {"lat": 1, "lon": 1, "camera": "spin"}, {"lat": 1, "lon": 1, "seed": 1.5},
                    {"lat": float("nan"), "lon": 1}, {"lat": 1, "lon": 1, "osm_file": "x" * 600}):
            with self.assertRaises(ValueError):
                validate_setup(bad)


def _summary():
    if SAVE_TIMES:
        s = sorted(SAVE_TIMES)
        print(f"\nsave time over {len(s)} saves: mean {statistics.mean(s) * 1000:.2f} ms, "
              f"p99 {s[max(0, int(len(s) * 0.99) - 1)] * 1000:.2f} ms, max {s[-1] * 1000:.2f} ms "
              f"(budget {SAVE_BUDGET_S * 1000:.0f} ms)")


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=2)
    _summary()
    raise SystemExit(0 if result.result.wasSuccessful() else 1)
