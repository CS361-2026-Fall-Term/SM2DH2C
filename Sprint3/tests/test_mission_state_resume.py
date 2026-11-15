"""
PBI-10 tests for the parts added when mission saving was moved onto the latest code
(PBI-03 setup window, PBI-06 map-guided search, PBI-09 live commands):
  * excluded zones and the probability-ordered plan survive save / restore
  * files written before those fields existed still load
  * the new setup fields (person category, setup exclusions, zone order, person spot) round-trip
  * the probability map and the mission clock are saved and restored
  * zone arrivals are logged once (Mission.arrived), and the operator panel still sees new
    history lines after the history cap is reached
Run: python -m unittest tests.test_mission_state_resume -v   (no PyBullet needed)
"""
import os
import sys
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import argparse
import tempfile
import unittest

import numpy as np

from operator_guidance import Mission, Cmd, ZONES, ORIGINAL_PLAN, HISTORY_LIMIT
import mission_state
from mission_state import (MissionStateStore, validate_setup, setup_from_args,
                           apply_setup_to_args, restore_belief)
from probability_map import ProbabilityMap


def started(order=None):
    m = Mission()
    if order:
        m.set_plan_order(order)
    m.start()
    return m


class TestNewMissionFields(unittest.TestCase):
    def test_excluded_and_plan_order_survive_restore(self):
        m = started(["C", "A", "D", "B"])
        m.handle(Cmd.AVOID, "D")
        m.arrived()
        m2 = Mission()
        m2.restore(m.snapshot())
        self.assertEqual(m2.excluded, ["D"])
        self.assertEqual(m2.original_plan, ["C", "A", "D", "B"])
        self.assertEqual(m2.plan, m.plan)
        # Return to plan after a resume goes back to the saved (probability) order, not A-D.
        self.assertTrue(m2.handle(Cmd.RETURN_PLAN).ok)
        self.assertEqual(m2.plan, ["A", "D", "B"])
        self.assertEqual(m2.excluded, [])

    def test_reinclude_works_after_restore(self):
        m = started()
        m.handle(Cmd.AVOID, "B")
        m2 = Mission()
        m2.restore(m.snapshot())
        res = m2.handle(Cmd.REINCLUDE, "B")
        self.assertTrue(res.ok, res.message)
        self.assertIn("B", m2.plan)

    def test_old_snapshot_without_new_fields_still_loads(self):
        snap = started().snapshot()
        del snap["excluded"], snap["original_plan"]
        m = Mission()
        m.restore(snap)
        self.assertEqual(m.excluded, [])
        self.assertEqual(m.original_plan, list(ORIGINAL_PLAN))

    def test_bad_new_fields_are_rejected_and_change_nothing(self):
        for key, bad in (("excluded", ["Z"]), ("excluded", "C"), ("excluded", ["A"]),
                         ("original_plan", ["A", "B"]), ("original_plan", [1, 2, 3, 4])):
            with self.subTest(key=key, bad=bad):
                snap = started().snapshot()          # plan A-D, so excluded ["A"] duplicates it
                snap[key] = bad
                m = Mission()
                before = m.snapshot()
                with self.assertRaises(ValueError):
                    m.restore(snap)
                self.assertEqual(m.snapshot(), before)

    def test_zone_arrival_is_logged_once(self):
        m = started()
        m.arrived()
        lines = [h for h in m.history if h[1] == "ARRIVED"]
        self.assertEqual(len(lines), 1)
        self.assertIn("Zone A searched", lines[0][2])

    def test_map_phase_message_is_logged_and_restored(self):
        m = started()
        for _ in range(4):
            m.arrived()
        m.search_cell((1.5, -2.5))
        m2 = Mission()
        m2.restore(m.snapshot())
        self.assertEqual(m2.state, "SEARCHING")
        self.assertIsNone(m2.destination)            # main.py resumes the map-guided search
        self.assertIn("Searching most likely cell", m2.history[-2][2])

    def test_history_count_keeps_growing_past_the_cap(self):
        m = started()
        for i in range(HISTORY_LIMIT + 50):
            m.record("TEST", f"line {i}")
        self.assertEqual(len(m.history), HISTORY_LIMIT)
        self.assertEqual(m.history_count, HISTORY_LIMIT + 50 + 1)   # +1 for START


class TestPanelLog(unittest.TestCase):
    """operator_panel._collect_messages without Tk: only the bookkeeping is exercised."""

    def collector(self):
        import operator_panel
        panel = operator_panel.OperatorPanel.__new__(operator_panel.OperatorPanel)
        panel._hist_n, panel._last_resp, panel.logged = 0, None, []
        panel._add_message = lambda ts, src, msg: panel.logged.append((src, msg))
        return panel

    def test_arrival_appears_once_in_the_log(self):
        panel, m = self.collector(), started()
        m.arrived()
        panel._collect_messages(m)
        self.assertEqual(sum(1 for s, _ in panel.logged if s == "ARRIVED"), 1)

    def test_new_lines_still_reach_the_log_once_history_is_full(self):
        panel, m = self.collector(), started()
        for i in range(HISTORY_LIMIT):
            m.record("TEST", f"line {i}")
        panel._collect_messages(m)
        panel.logged.clear()
        m.record("TEST", "after the cap")
        panel._collect_messages(m)
        self.assertEqual(panel.logged, [("TEST", "after the cap")])


class TestNewSetupFields(unittest.TestCase):
    def args(self, **kw):
        base = dict(lat=None, lon=None, area=None, map_size=None, osm_file=None, seed=3,
                    simple_visuals=False, camera="follow", max_time=200.0,
                    person_category="child", exclude_zones="AC", zone_order="fixed")
        base.update(kw)
        return argparse.Namespace(**base)

    def test_round_trip_through_args(self):
        setup = validate_setup(setup_from_args(self.args(), person_xy=(-9, 4.5)))
        self.assertEqual(setup["person_xy"], [-9.0, 4.5])
        back = apply_setup_to_args(argparse.Namespace(), setup)
        self.assertEqual((back.person_category, back.exclude_zones, back.zone_order),
                         ("child", "AC", "fixed"))

    def test_old_setup_without_new_fields_gets_defaults(self):
        setup = validate_setup({"seed": 0, "max_time": 100.0})
        back = apply_setup_to_args(argparse.Namespace(), setup)
        self.assertIsNone(back.person_category)
        self.assertIsNone(back.exclude_zones)
        self.assertEqual(back.zone_order, "map")

    def test_bad_new_setup_values_are_rejected(self):
        for key, bad in (("exclude_zones", "ABCD"), ("exclude_zones", "AX"), ("exclude_zones", "AA"),
                         ("zone_order", "random"), ("person_category", ""),
                         ("person_xy", [1.0]), ("person_xy", [float("nan"), 0.0]),
                         ("person_xy", "1,2")):
            with self.subTest(key=key, bad=bad):
                with self.assertRaises(ValueError):
                    validate_setup({key: bad})


class TestBeliefAndClock(unittest.TestCase):
    def belief(self):
        return ProbabilityMap(x_range=(-13, 13), y_range=(-13, 13), cell_size=1.0)

    def test_probability_map_and_clock_round_trip_through_the_file(self):
        b = self.belief()
        b.mark_searched((5.0, 5.0), 4.0)              # some searched cells
        m = started()
        with tempfile.TemporaryDirectory() as tmp:
            store = MissionStateStore(os.path.join(tmp, "s.json"))
            store.set_context(setup={"seed": 0}, zones=ZONES)
            self.assertTrue(store.autosave(m, [1.0, 2.0, 2.5], force=True, belief=b, sim_t=42.5))
            payload = store.load().payload
        self.assertEqual(payload["drone"]["t"], 42.5)
        fresh = self.belief()
        self.assertTrue(restore_belief(fresh, payload["belief"]))
        np.testing.assert_allclose(fresh.grid, b.grid)

    def test_different_grid_size_is_ignored(self):
        b = self.belief()
        data = mission_state.belief_to_data(b)
        small = ProbabilityMap(x_range=(-6, 6), y_range=(-6, 6), cell_size=1.0)
        before = small.grid.copy()
        self.assertFalse(restore_belief(small, data))
        np.testing.assert_array_equal(small.grid, before)

    def test_damaged_belief_makes_the_file_invalid(self):
        for bel in ({"shape": [2, 2], "grid": [0.1, 0.2, 0.3]},
                    {"shape": [2, 2], "grid": [0.1, -0.2, 0.3, 0.4]},
                    {"shape": [1000, 1000], "grid": []},
                    {"shape": [2, 2], "grid": [0.1, 0.2, 0.3, float("inf")]}):
            with self.subTest(bel=str(bel)[:40]):
                p = {"mission": started().snapshot(), "belief": bel}
                with self.assertRaises(ValueError):
                    mission_state.validate_payload(p)

    def test_excluded_cells_stay_zero_after_restore(self):
        b = self.belief()
        data = mission_state.belief_to_data(b)       # saved before the zone was excluded
        fresh = self.belief()
        fresh.exclude_zone("C", (-13, -13, 0, 0))
        restore_belief(fresh, data)
        self.assertEqual(float(fresh.grid[fresh.excluded].sum()), 0.0)
        self.assertAlmostEqual(float(fresh.grid.sum()), 1.0)


if __name__ == "__main__":
    unittest.main()
