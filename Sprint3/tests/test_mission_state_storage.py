"""
File Description: EXTRA tests for PBI-10 (mission state saving). They complement test_mission_state.py
and do not repeat it. No PyBullet needed. Put this file next to mission_state.py and run:

    python -m unittest -v tests.test_mission_state_storage

Every test has an "EXPECT:" comment saying what a passing run means.
Two tests are marked "EXPECTED TO FAIL TODAY": they target real gaps found while reading the code
(history cap not noticed by autosave; unhashable zone names crash load()). Everything else should pass.
"""
import os
import sys
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)


import argparse
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from unittest import mock

import operator_guidance as og
from operator_guidance import Mission, Cmd, ZONES
import mission_state
from mission_state import (MissionStateStore, SCHEMA_VERSION, PERIODIC_SAVE_S, validate_setup,
                           setup_from_args, apply_setup_to_args, describe, _checksum)

HERE = CODE_DIR                    # where the simulation code is
SETUP = {"lat": 44.565296, "lon": -123.276016, "area": 60.0, "map_size": None,
         "osm_file": None, "seed": 0, "simple_visuals": False, "camera": "follow", "max_time": None}
ZONES_JSON = {"A": [6, 6], "B": [-6, 6], "C": [-6, -6], "D": [6, -6]}


class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def new_store(tmp, clock=None, wall=None):
    kw = {}
    if clock:
        kw["clock"] = clock
    if wall:
        kw["wall"] = wall
    store = MissionStateStore(os.path.join(tmp, "mission_state.json"), **kw)
    store.set_context(setup=SETUP, zones=ZONES)
    return store


def started():
    m = Mission()
    m.start()
    return m


def normal(x):
    """What a snapshot looks like after a JSON round trip (tuples -> lists)."""
    return json.loads(json.dumps(x))


def two_saves(tmp):
    """main = newer state (AVOID_C applied), .bak = older state (just started)."""
    store, m = new_store(tmp), started()
    store.save(store.build(m, [0, 0, 2.5]))
    m.handle(Cmd.AVOID, "C")
    store.save(store.build(m, [1, 1, 2.5]))
    return store, m


def run_child(script, path, timeout=60):
    return subprocess.run([sys.executable, "-c", script, HERE, path],
                          capture_output=True, text=True, timeout=timeout)


# ============================================================ atomic write / backup =====
class TestAtomicWrite(unittest.TestCase):
    def test_first_save_has_no_backup_second_save_backs_up_the_first(self):
        # EXPECT: no .bak after save #1; after save #2 the .bak holds save #1's mission, main holds #2's.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), Mission()
            first = normal(m.snapshot())
            store.save(store.build(m))
            self.assertFalse(os.path.exists(store.bak_path))
            m.start()
            store.save(store.build(m))
            self.assertEqual(MissionStateStore(store.bak_path).load().payload["mission"], first)
            self.assertEqual(store.load().payload["mission"], normal(m.snapshot()))

    def test_failed_write_leaves_old_file_intact_and_no_temp_files(self):
        # EXPECT: save() raises OSError, but the previous file still loads and no stray temp file remains.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            store.save(store.build(m))
            before = normal(m.snapshot())
            m.handle(Cmd.AVOID, "C")
            with mock.patch("mission_state.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    store.save(store.build(m))
            self.assertEqual(os.listdir(tmp), ["mission_state.json"])
            got = store.load()
            self.assertEqual(got.source, "main")
            self.assertEqual(got.payload["mission"], before)

    def test_crash_between_the_two_renames_still_recovers_from_backup(self):
        # EXPECT: main is gone (moved to .bak, new file never swapped in); load() silently uses the backup
        # (source "backup", no problems reported) and returns the last fully written state.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            store.save(store.build(m))
            before = normal(m.snapshot())
            m.handle(Cmd.AVOID, "C")
            real, calls = os.replace, [0]

            def second_rename_fails(src, dst):
                calls[0] += 1
                if calls[0] == 2:
                    raise OSError("power cut")
                return real(src, dst)

            with mock.patch("mission_state.os.replace", side_effect=second_rename_fails):
                with self.assertRaises(OSError):
                    store.save(store.build(m))
            self.assertEqual(os.listdir(tmp), ["mission_state.json.bak"])
            got = store.load()
            self.assertEqual((got.source, got.problems), ("backup", []))
            self.assertEqual(got.payload["mission"], before)

    def test_fsync_only_when_durable(self):
        # EXPECT: durable=True calls fsync at least once; durable=False never does.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            with mock.patch("mission_state.os.fsync") as fs:
                store.save(store.build(m), durable=True)
                self.assertGreaterEqual(fs.call_count, 1)
                fs.reset_mock()
                store.save(store.build(m), durable=False)
                self.assertEqual(fs.call_count, 0)

    def test_file_metadata_and_counters(self):
        # EXPECT: version == SCHEMA_VERSION, saved_at comes from the injected wall clock,
        # saves counter goes up by one per save, last_save_s > 0 and max_save_s >= last_save_s.
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp, wall=lambda: 1234.5678)
            for n in (1, 2, 3):
                store.save(store.build(Mission()))
                self.assertEqual(store.saves, n)
            with open(store.path) as f:
                doc = json.load(f)
            self.assertEqual(doc["version"], SCHEMA_VERSION)
            self.assertAlmostEqual(doc["saved_at"], 1234.568, places=3)
            self.assertGreater(store.last_save_s, 0)
            self.assertGreaterEqual(store.max_save_s, store.last_save_s)

    def test_file_is_pure_ascii_and_unicode_text_round_trips(self):
        # EXPECT: raw bytes decode as ASCII (non-ASCII is escaped), and the message comes back unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            m.record("CMD", "caf\u00e9 \u2713 \u65e5\u672c")
            store.save(store.build(m))
            with open(store.path, "rb") as f:
                f.read().decode("ascii")                      # must not raise
            self.assertEqual(store.load().payload["mission"]["history"][-1][2], "caf\u00e9 \u2713 \u65e5\u672c")

    def test_checksum_ignores_key_order_but_not_values(self):
        # EXPECT: same data in a different key order -> same checksum; any changed value -> different.
        self.assertEqual(_checksum({"a": 1, "b": [1, 2]}), _checksum({"b": [1, 2], "a": 1}))
        self.assertNotEqual(_checksum({"a": 1, "b": [1, 2]}), _checksum({"a": 1, "b": [2, 1]}))


# ================================================================ autosave triggers =====
class TestAutosaveTriggers(unittest.TestCase):
    def test_each_mission_field_alone_triggers_exactly_one_save(self):
        # EXPECT: changing response / history / plan / visited / destination / state each makes the next
        # autosave() return True, and an immediate second call returns False (nothing new).
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store, m = new_store(tmp, clock), started()
            store.autosave(m, [0, 0, 1])
            self.assertFalse(store.autosave(m, [0, 0, 1]))
            changes = {"response": lambda: setattr(m, "response", "hello"),
                       "history": lambda: m.record("CMD", "x"),
                       "plan": lambda: setattr(m, "plan", ["B", "C", "D"]),
                       "visited": lambda: setattr(m, "visited", ["A"]),
                       "destination": lambda: setattr(m, "destination", ZONES["B"]),
                       "state": lambda: setattr(m, "state", "FOUND")}
            for name, change in changes.items():
                with self.subTest(field=name):
                    change()
                    self.assertTrue(store.autosave(m, [0, 0, 1]))
                    self.assertFalse(store.autosave(m, [0, 0, 1]))

    def test_drone_movement_alone_is_not_saved_when_not_searching(self):
        # EXPECT: IDLE mission, drone "moves" and 10 s pass -> no write (only SEARCHING refreshes position).
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store, m = new_store(tmp, clock), Mission()
            store.autosave(m, [0, 0, 1])
            clock.t += 10
            self.assertFalse(store.autosave(m, [5, 5, 2.5]))
            self.assertEqual(store.saves, 1)

    def test_periodic_refresh_waits_for_the_interval(self):
        # EXPECT: while SEARCHING, +0.4 s -> no write; total +0.6 s -> write; saved position is the new one.
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store, m = new_store(tmp, clock), started()
            store.autosave(m, [0, 0, 1])
            clock.t += 0.4
            self.assertFalse(store.autosave(m, [1, 0, 1]))
            clock.t += 0.2
            self.assertTrue(store.autosave(m, [2, 0, 1]))
            self.assertEqual(store.load().payload["drone"]["pos"], [2.0, 0.0, 1.0])

    def test_periodic_refresh_is_not_fsynced_but_mission_changes_are(self):
        # EXPECT: position-only refresh -> 0 fsync calls (cheap); a real mission change -> fsync called.
        with tempfile.TemporaryDirectory() as tmp:
            clock = FakeClock()
            store, m = new_store(tmp, clock), started()
            store.autosave(m, [0, 0, 1])
            with mock.patch("mission_state.os.fsync") as fs:
                clock.t += PERIODIC_SAVE_S + 0.01
                self.assertTrue(store.autosave(m, [1, 0, 1]))
                self.assertEqual(fs.call_count, 0)
                m.handle(Cmd.AVOID, "C")
                self.assertTrue(store.autosave(m, [1, 0, 1]))
                self.assertGreaterEqual(fs.call_count, 1)

    def test_force_writes_even_when_nothing_changed(self):
        # EXPECT: force=True returns True and bumps the save counter (used for first/final save in main.py).
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), Mission()
            store.autosave(m, [0, 0, 1])
            self.assertTrue(store.autosave(m, [0, 0, 1], force=True))
            self.assertEqual(store.saves, 2)

    def test_failing_disk_warns_once_and_never_raises(self):
        # EXPECT: 5 failed ticks -> all return False, exactly one "WARNING" line printed, 0 successful saves.
        with tempfile.TemporaryDirectory() as tmp:
            store = MissionStateStore(os.path.join(tmp, "nope", "state.json"))
            store.set_context(setup=SETUP, zones=ZONES)
            m, out = started(), io.StringIO()
            with contextlib.redirect_stdout(out):
                results = [store.autosave(m, [0, 0, 1]) for _ in range(5)]
            self.assertEqual(results, [False] * 5)
            self.assertEqual(out.getvalue().count("WARNING"), 1)
            self.assertEqual(store.saves, 0)

    def test_idle_ticks_are_cheap(self):
        # EXPECT: 10,000 no-change autosave() calls (~ 2.7 min of ticks at 60 Hz) take well under 2 s.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), Mission()
            store.autosave(m, [0, 0, 1])
            t0 = time.perf_counter()
            for _ in range(10_000):
                store.autosave(m, [0, 0, 1])
            self.assertLess(time.perf_counter() - t0, 2.0)
            self.assertEqual(store.saves, 1)

    def test_new_history_entry_is_saved_even_when_history_is_at_its_cap(self):
        # EXPECTED TO FAIL TODAY. At HISTORY_LIMIT the log length stops changing, and the autosave
        # fingerprint only looks at len(history), so a new entry (with an unchanged response) is not saved.
        # Fix idea: put history[-1] (or a running counter) into the fingerprint in autosave().
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), Mission()
            for i in range(og.HISTORY_LIMIT + 5):
                m.record("CMD", f"m{i}")
            store.autosave(m, [0, 0, 1])
            m.record("CMD", "late entry")
            self.assertTrue(store.autosave(m, [0, 0, 1]), "late entry was not saved")
            self.assertEqual(store.load().payload["mission"]["history"][-1][2], "late entry")


# ==================================================================== loading ===========
class TestLoadBehaviour(unittest.TestCase):
    def test_good_main_ignores_a_corrupt_backup(self):
        # EXPECT: source "main", no problems (the backup is not even looked at).
        with tempfile.TemporaryDirectory() as tmp:
            store, m = two_saves(tmp)
            with open(store.bak_path, "wb") as f:
                f.write(b"junk")
            got = store.load()
            self.assertEqual((got.source, got.problems), ("main", []))
            self.assertEqual(got.payload["mission"], normal(m.snapshot()))

    def test_missing_main_with_good_backup_loads_backup_without_problems(self):
        # EXPECT: source "backup", problems == [] (a missing file is not reported as damage).
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = two_saves(tmp)
            os.remove(store.path)
            got = store.load()
            self.assertEqual((got.source, got.problems), ("backup", []))
            self.assertEqual(got.payload["mission"]["plan"], ["A", "B", "C", "D"])   # the older state

    def test_both_files_bad_reports_each_file_by_name(self):
        # EXPECT: payload None; two problems, the first starting "mission_state.json:", the second
        # "mission_state.json.bak:".
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = two_saves(tmp)
            for p in (store.path, store.bak_path):
                with open(p, "wb") as f:
                    f.write(b"{not json")
            got = store.load()
            self.assertIsNone(got.payload)
            self.assertTrue(got.problems[0].startswith("mission_state.json:"), got.problems)
            self.assertTrue(got.problems[1].startswith("mission_state.json.bak:"), got.problems)

    def test_wrong_layout_and_non_ascii_bytes_are_rejected(self):
        # EXPECT: "[]" and '{"version": 1}' -> "wrong layout"; non-ASCII bytes -> "damaged". No exception.
        with tempfile.TemporaryDirectory() as tmp:
            store = new_store(tmp)
            store.save(store.build(started()))
            cases = [(b"[]", "wrong layout"), (b'{"version": 1}', "wrong layout"),
                     (b"\xc3\xa9\xff", "damaged")]
            for raw, word in cases:
                with self.subTest(raw=raw):
                    with open(store.path, "wb") as f:
                        f.write(raw)
                    got = store.load()
                    self.assertIsNone(got.payload)
                    self.assertIn(word, " ".join(got.problems))

    def test_load_never_modifies_files(self):
        # EXPECT: file contents and directory listing are identical before and after load(), even for bad files.
        with tempfile.TemporaryDirectory() as tmp:
            store, _ = two_saves(tmp)
            with open(store.bak_path, "wb") as f:
                f.write(b"junk")
            def snap():
                names, data = sorted(os.listdir(tmp)), []
                for n in names:
                    with open(os.path.join(tmp, n), "rb") as f:
                        data.append(f.read())
                return names, data
            store.load()
            os.remove(store.path)                      # also check the "fall back to backup" path
            before = snap()
            store.load()
            self.assertEqual(snap(), before)

    def test_store_without_context_round_trips(self):
        # EXPECT: no set_context() and no drone position -> saves and loads fine; setup/zones/drone are None.
        with tempfile.TemporaryDirectory() as tmp:
            store = MissionStateStore(os.path.join(tmp, "s.json"))
            m = started()
            store.save(store.build(m))
            p = store.load().payload
            self.assertEqual(p["mission"], normal(m.snapshot()))
            self.assertEqual((p["setup"], p["zones"], p["drone"]), (None, None, None))

    def test_non_finite_drone_position_is_rejected_on_load(self):
        # EXPECT: NaN position is written, but load() refuses it: payload None, problem mentions "drone.pos".
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            store.save(store.build(m, [float("nan"), 0, 0]))
            got = store.load()
            self.assertIsNone(got.payload)
            self.assertIn("drone.pos", " ".join(got.problems))

    def test_signed_data_with_unhashable_zone_names_does_not_raise(self):
        # EXPECTED TO FAIL TODAY (load() raises TypeError). A file whose checksum is valid but whose plan
        # contains e.g. [1] makes `z not in ZONES` raise TypeError instead of ValueError, so load() crashes.
        # Fix idea: in Mission.validate_snapshot use `not isinstance(z, str) or z not in ZONES`.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            store.save(store.build(m))
            with open(store.path) as f:
                doc = json.load(f)
            doc["payload"]["mission"]["plan"] = [[1]]
            doc["checksum"] = _checksum(doc["payload"])
            with open(store.path, "w") as f:
                json.dump(doc, f)
            got = store.load()                                  # must not raise
            self.assertIsNone(got.payload)
            self.assertTrue(got.problems)


# ====================================================== setup / helper functions ========
class TestSetupHelpers(unittest.TestCase):
    def test_validate_setup_fills_defaults_and_converts_numbers(self):
        # EXPECT: ints become floats; all unspecified keys are None; {} is valid (no OSM position).
        out = validate_setup({"lat": 44, "lon": -123})
        self.assertIsInstance(out["lat"], float)
        for key in ("area", "map_size", "osm_file", "seed", "simple_visuals", "camera", "max_time"):
            self.assertIsNone(out[key], key)
        self.assertIsNone(validate_setup({})["lat"])
        validate_setup({"lat": 90, "lon": 180})
        validate_setup({"lat": -90, "lon": -180})                # boundaries are allowed

    def test_validate_setup_rejects_wrong_types_and_lonely_coordinate(self):
        # EXPECT: ValueError for bool as a number, lon out of range, lat without lon, bool/int mix-ups, bad osm_file.
        bad = [{"lat": True, "lon": 1}, {"lat": 1, "lon": 181}, {"lat": 1}, {"lon": 1},
               {"lat": 1, "lon": 1, "seed": True}, {"lat": 1, "lon": 1, "simple_visuals": 1},
               {"lat": 1, "lon": 1, "osm_file": 5}]
        for s in bad:
            with self.subTest(setup=s):
                with self.assertRaises(ValueError):
                    validate_setup(s)

    def test_setup_from_args_normalises_empty_values(self):
        # EXPECT: osm_file "" -> None, seed None -> 0, simple_visuals 0 -> False, and the result validates.
        args = argparse.Namespace(lat=None, lon=None, area=None, map_size=None, osm_file="",
                                  seed=None, simple_visuals=0, camera="locked", max_time=None)
        setup = setup_from_args(args)
        self.assertIsNone(setup["osm_file"])
        self.assertEqual(setup["seed"], 0)
        self.assertIs(setup["simple_visuals"], False)
        validate_setup(setup)

    def test_apply_setup_handles_missing_optional_values(self):
        # EXPECT: seed None -> 0, simple_visuals None -> False, camera stays None (main.py picks the default).
        blank = argparse.Namespace()
        apply_setup_to_args(blank, validate_setup({"lat": 1, "lon": 2}))
        self.assertEqual(blank.seed, 0)
        self.assertIs(blank.simple_visuals, False)
        self.assertIsNone(blank.camera)

    def test_describe_summarises_the_saved_mission(self):
        # EXPECT: "SEARCHING, 1 zone(s) searched, plan ['B', 'C', 'D']"
        with tempfile.TemporaryDirectory() as tmp:
            store, m = new_store(tmp), started()
            m.arrived()
            self.assertEqual(describe(store.build(m)), "SEARCHING, 1 zone(s) searched, plan ['B', 'C', 'D']")


# =============================================================== Mission restore ========
class TestMissionRestore(unittest.TestCase):
    def missions(self):
        searching = started()
        searching.handle(Cmd.SEARCH_FIRST, "B")
        searching.arrived()
        found = started()
        found.found()
        aborted = started()
        aborted.finish("ABORT", "NAV", "Navigation aborted")
        complete = started()
        for _ in range(4):
            complete.arrived()
        complete.finish("COMPLETE", "NAV", "done")
        return {"SEARCHING": searching, "FOUND": found, "ABORT": aborted, "COMPLETE": complete}

    def test_every_mission_state_restores_exactly(self):
        # EXPECT: state, plan, visited, response and the old history all match; one extra RESTORE entry is appended.
        for name, m in self.missions().items():
            with self.subTest(state=name):
                snap = normal(m.snapshot())
                m2 = Mission()
                m2.restore(snap)
                for key in ("state", "plan", "visited", "response"):
                    self.assertEqual(getattr(m2, key), snap[key], key)
                self.assertEqual([list(h) for h in m2.history][:-1], snap["history"])
                self.assertEqual(m2.history[-1][1], "RESTORE")

    def test_restored_ended_mission_has_no_destination(self):
        # EXPECTED TO FAIL TODAY for FOUND/ABORT: restore() always calls _set_dest(), so a finished mission
        # whose plan is not empty gets destination = first zone instead of None (HUD would show a destination).
        # Fix idea: in restore(), set self.destination = None when self.state != "SEARCHING".
        for name in ("FOUND", "ABORT", "COMPLETE"):
            with self.subTest(state=name):
                m = self.missions()[name]
                m2 = Mission()
                m2.restore(normal(m.snapshot()))
                self.assertIsNone(m2.destination)

    def test_restore_note_only_when_searching_and_zones_changed(self):
        # EXPECT: unchanged map -> note ""; changed map + SEARCHING -> note; changed map + FOUND -> note "".
        saved = dict(ZONES)
        try:
            searching = normal(started().snapshot())
            found = normal(self.missions()["FOUND"].snapshot())
            self.assertEqual(Mission().restore(searching), "")
            og.scale_zones(2.0)
            self.assertIn("zone centres changed", Mission().restore(searching))
            self.assertEqual(Mission().restore(found), "")
        finally:
            ZONES.update(saved)

    def test_restoring_twice_logs_two_restore_entries(self):
        # EXPECT: save -> restore -> save -> restore leaves two consecutive RESTORE lines at the end.
        m2, m3 = Mission(), Mission()
        m2.restore(normal(started().snapshot()))
        m3.restore(normal(m2.snapshot()))
        self.assertEqual([h[1] for h in m3.history][-2:], ["RESTORE", "RESTORE"])

    def test_restore_with_full_history_stays_at_the_cap(self):
        # EXPECT: history length stays exactly HISTORY_LIMIT; oldest line dropped; last line is RESTORE.
        m = started()
        for i in range(og.HISTORY_LIMIT):
            m.record("CMD", f"m{i}")
        snap = normal(m.snapshot())
        m2 = Mission()
        m2.restore(snap)
        self.assertEqual(len(m2.history), og.HISTORY_LIMIT)
        self.assertEqual(m2.history[-1][1], "RESTORE")
        self.assertEqual(list(m2.history[0]), snap["history"][1])

    def test_snapshot_and_restore_do_not_share_data(self):
        # EXPECT: editing a snapshot (or the dict given to restore) never changes the live mission.
        m = started()
        s = m.snapshot()
        s["plan"].append("Z")
        s["history"][0][2] = "tampered"
        self.assertEqual(m.plan, ["A", "B", "C", "D"])
        self.assertNotEqual(m.history[0][2], "tampered")
        m2 = Mission()
        snap = normal(m.snapshot())
        m2.restore(snap)
        snap["plan"].clear()
        snap["visited"].append("A")
        self.assertEqual(m2.plan, ["A", "B", "C", "D"])
        self.assertEqual(m2.visited, [])

    def test_validate_snapshot_boundaries(self):
        # EXPECT: limits are inclusive (response 500 chars, history HISTORY_LIMIT entries are valid);
        # one past the limit, bool timestamps/coordinates, 3-D destinations and duplicate zones are ValueError.
        base = normal(started().snapshot())
        ok = [{"response": "x" * 500},
              {"history": [[1.0, "a", "b"]] * og.HISTORY_LIMIT}]
        bad = [{"response": "x" * 501},
               {"history": [[1.0, "a", "b"]] * (og.HISTORY_LIMIT + 1)},
               {"history": [[1.0, "a", "m" * 501]]},
               {"history": [[True, "a", "b"]]},
               {"destination": [True, 1]},
               {"destination": [1, 2, 3]},
               {"plan": ["A", "A"], "visited": []}]
        for edit in ok:
            with self.subTest(valid=list(edit)):
                Mission.validate_snapshot({**base, **edit})
        for edit in bad:
            with self.subTest(invalid=str(edit)[:50]):
                with self.assertRaises(ValueError):
                    Mission.validate_snapshot({**base, **edit})


# ========================================================== whole-workflow checks =======
class TestWorkflow(unittest.TestCase):
    def test_resume_cycle_continues_the_same_file(self):
        # EXPECT: a "resumed" run restores, force-saves, and the file then holds the restored mission with
        # exactly one new RESTORE line; setup and zones survive.
        with tempfile.TemporaryDirectory() as tmp:
            store, m = two_saves(tmp)
            old = normal(m.snapshot())
            store2 = MissionStateStore(store.path)                  # a fresh process
            got = store2.load()
            store2.set_context(setup=got.payload["setup"], zones=got.payload["zones"])
            m2 = Mission()
            m2.restore(got.payload["mission"])
            store2.autosave(m2, got.payload["drone"]["pos"], force=True)
            again = store2.load().payload
            self.assertEqual(again["mission"], normal(m2.snapshot()))
            self.assertEqual(len(again["mission"]["history"]), len(old["history"]) + 1)
            self.assertEqual(again["mission"]["history"][-1][1], "RESTORE")
            self.assertEqual(again["setup"]["lat"], SETUP["lat"])
            self.assertEqual(again["drone"]["pos"], [1.0, 1.0, 2.5])

    def test_fresh_run_archives_old_state_then_writes_a_new_file(self):
        # EXPECT: after archive_previous() the old mission is readable from the ".previous" file; the new run's
        # file contains only the new mission; archiving a second time replaces the ".previous" copy.
        with tempfile.TemporaryDirectory() as tmp:
            store, old_m = two_saves(tmp)
            old = normal(old_m.snapshot())
            self.assertTrue(store.archive_previous())
            self.assertEqual(MissionStateStore(store.prev_path).load().payload["mission"], old)
            new_m = Mission()
            store.autosave(new_m, [0, 0, 2.5], force=True)
            self.assertEqual(store.load().payload["mission"], normal(new_m.snapshot()))
            self.assertTrue(store.archive_previous())
            self.assertEqual(MissionStateStore(store.prev_path).load().payload["mission"],
                             normal(new_m.snapshot()))


# ============================================================== real process crashes ====
FIRST_SAVE_CRASH = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from operator_guidance import Mission
from mission_state import MissionStateStore
store = MissionStateStore(sys.argv[2])
store.set_context(setup={"lat": 44.5, "lon": -123.2, "area": 60.0}, zones={"A": [6, 6], "B": [-6, 6], "C": [-6, -6], "D": [6, -6]})
m = Mission()
store.autosave(m, [0, 0, 2.5], force=True)     # the "first save" main.py does before the loop
os._exit(137)
"""

CRASH_BETWEEN_RENAMES = r"""
import os, sys
sys.path.insert(0, sys.argv[1])
from operator_guidance import Mission, Cmd
import mission_state
from mission_state import MissionStateStore
store = MissionStateStore(sys.argv[2])
store.set_context(setup={"lat": 44.5, "lon": -123.2, "area": 60.0}, zones={"A": [6, 6], "B": [-6, 6], "C": [-6, -6], "D": [6, -6]})
m = Mission(); m.start()
store.autosave(m, [0, 0, 2.5], force=True)     # save #1 completes normally
real, calls = os.replace, [0]
def hook(src, dst):
    calls[0] += 1
    if calls[0] == 2:                          # 1st rename (main -> .bak) works, process dies before 2nd
        os._exit(9)
    return real(src, dst)
mission_state.os.replace = hook
m.handle(Cmd.AVOID, "C")
store.autosave(m, [1, 1, 2.5])
"""


class TestProcessCrashes(unittest.TestCase):
    def test_crash_right_after_the_first_save_is_recoverable(self):
        # EXPECT: child exits with 137; the restart finds an IDLE mission ("Press START") and drone [0, 0, 2.5].
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mission_state.json")
            self.assertEqual(run_child(FIRST_SAVE_CRASH, path).returncode, 137)
            got = MissionStateStore(path).load()
            self.assertEqual(got.payload["mission"]["state"], "IDLE")
            self.assertEqual(got.payload["mission"]["response"], "Press START")
            self.assertEqual(got.payload["drone"]["pos"], [0.0, 0.0, 2.5])

    def test_process_dying_inside_a_save_loses_only_that_one_change(self):
        # EXPECT: child exits with 9 mid-save; restart uses the backup (source "backup"): mission SEARCHING with the
        # full plan A-D, i.e. the AVOID_C command that was being saved is lost but nothing is corrupt.
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mission_state.json")
            self.assertEqual(run_child(CRASH_BETWEEN_RENAMES, path).returncode, 9)
            got = MissionStateStore(path).load()
            self.assertEqual(got.source, "backup")
            self.assertEqual(got.payload["mission"]["state"], "SEARCHING")
            self.assertEqual(got.payload["mission"]["plan"], ["A", "B", "C", "D"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
