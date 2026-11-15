"""Tests for PBI-06: use the probability map to choose search destinations.

Two kinds of tests:
  * Unit tests (fast, no simulation): map_search.py, ProbabilityMap.best_target,
    Mission.search_cell / time_up, main.parse_person, zone order, random spawn.
  * AcceptanceCriteriaTests (end to end): run `python main.py --headless ...` as the PBI's
    acceptance criteria describe and check AC-01 .. AC-05 on the real printout. Each mission
    takes up to a minute or two. Needs PyBullet; set SKIP_SLOW=1 to run only the unit tests.

    python -m unittest tests.test_map_search -v                                  # everything
    python -m unittest tests.test_map_search.AcceptanceCriteriaTests -v          # only AC-01..AC-05
    SKIP_SLOW=1 python -m unittest tests.test_map_search -v                      # unit tests only
"""
import os
import sys
CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import math
import os
import random
import unittest

import numpy as np

import map_search
from map_search import MapGuidedSearch, line_of_sight_clear, standoff
from operator_guidance import Mission, Cmd
from probability_map import ProbabilityMap


def make_map(hot=None):
    """26 m x 24 m map like main.py's default; optional single hot cell at world xy."""
    b = ProbabilityMap(x_range=(-13, 13), y_range=(-11, 13), cell_size=1.0, pod=0.65)
    if hot is not None:
        m = np.full_like(b.grid, 1e-3)
        ix, iy = b.to_cell(hot)
        m[iy, ix] = 1.0
        b.set_initial(m)
    return b


def fly_and_look(s, pos, yaw=0.0, t=0.0, in_view=None, sighting=False, max_steps=100000):
    """Drive one fly -> look cycle: teleport to the viewpoint, then step until a new viewpoint
    is returned. Returns (new viewpoint, time spent looking)."""
    dt = 1.0 / 240.0
    steps = 0
    while steps < max_steps:
        vp = s.update(pos, yaw, dt, t + steps * dt, sighting=sighting, in_view=in_view)
        steps += 1
        if vp is not None:
            return vp, steps * dt
    raise AssertionError("look never finished")


class BestTargetTests(unittest.TestCase):
    def test_min_dist_skips_close_cells(self):
        b = make_map(hot=(0.5, 0.5))
        self.assertEqual(b.best_target(), (0.5, 0.5))
        t = b.best_target(from_xy=(0.5, 0.5), min_dist=2.0)
        self.assertGreaterEqual(math.dist(t, (0.5, 0.5)), 2.0)

    def test_excluded_and_skipped_never_returned(self):
        b = make_map(hot=(6.5, 6.5))
        b.exclude_zone("A", (0.4, 0.4, 11.6, 11.6))
        t = b.best_target()
        self.assertFalse(0.4 <= t[0] <= 11.6 and 0.4 <= t[1] <= 11.6)
        cell = b.best_cell()
        self.assertNotEqual(b.best_cell(skip=[cell]), cell)

    def test_none_when_everything_excluded(self):
        b = make_map()
        b.excluded[:] = True
        self.assertIsNone(b.best_target())


class ChooseTests(unittest.TestCase):
    def test_target_at_least_2m_from_drone(self):
        b = make_map(hot=(-9.5, -9.5))
        s = MapGuidedSearch(b, map_half=13)
        target, vp = s.choose((-9.5, -9.5))
        self.assertGreaterEqual(math.dist(target, (-9.5, -9.5)), map_search.MIN_TARGET_DIST)

    def test_ac03_same_cell_at_most_3_in_a_row(self):
        # A cell that never cools down (nothing observed) would be picked forever without the rule.
        b = make_map(hot=(5.5, 5.5))
        s = MapGuidedSearch(b, map_half=13)
        for _ in range(30):
            s.choose((-10.0, 10.0))
        run, worst = 1, 1
        for a, c in zip(s.selections, s.selections[1:]):
            run = run + 1 if a == c else 1
            worst = max(worst, run)
        self.assertLessEqual(worst, map_search.MAX_REPEATS)
        self.assertEqual(worst, 3)                         # the hot cell is still revisited

    def test_ac04_targets_in_bounds_and_not_excluded(self):
        rng = random.Random(1)
        b = make_map()
        b.set_initial(np.random.default_rng(1).random(b.grid.shape))
        rect = (-11.6, -11.6, -0.4, -0.4)                  # zone C square
        b.exclude_zone("C", rect)
        s = MapGuidedSearch(b, map_half=13)
        for _ in range(200):
            drone = (rng.uniform(-12, 12), rng.uniform(-10, 12))
            t, _vp = s.choose(drone)
            self.assertTrue(-13 <= t[0] <= 13 and -11 <= t[1] <= 13, t)
            self.assertFalse(rect[0] <= t[0] <= rect[2] and rect[1] <= t[1] <= rect[3], t)
            b.grid[b.to_cell(t)[1], b.to_cell(t)[0]] *= 0.3   # pretend it was looked at
            b._normalize()


class RateChoiceTests(unittest.TestCase):
    def test_nearby_patch_beats_far_single_cell(self):
        b = make_map()
        m = np.full_like(b.grid, 1e-4)
        far = b.to_cell((-11.5, 11.5)); m[far[1], far[0]] = 1.0          # lone hot cell, far away
        near = (b.cx > 2) & (b.cx < 6) & (b.cy > -2) & (b.cy < 2); m[near] = 0.4   # likely patch
        b.set_initial(m)
        s = MapGuidedSearch(b, map_half=13)
        target, _ = s.choose((0.0, 0.0))
        self.assertTrue(2 < target[0] < 6 and -2 < target[1] < 2, target)

    def test_never_targets_red_cells(self):
        b = make_map()
        m = np.ones_like(b.grid); m[b.cx > 0] = 0.0                       # east half: water
        b.set_initial(m)
        s = MapGuidedSearch(b, map_half=13)
        for _ in range(20):
            target, _ = s.choose((5.0, 5.0))
            self.assertLess(target[0], 0.0, target)


class ViewpointTests(unittest.TestCase):
    def test_standoff_puts_target_mid_image(self):
        d = standoff(2.5)
        self.assertAlmostEqual(math.degrees(math.atan2(2.5 - map_search.GROUND_Z, d)), 40.0, places=3)

    def test_viewpoint_clear_of_trees_and_in_map(self):
        trees = [(-8.0, -8.0, 0.5)]
        b = make_map(hot=(-8.5, -8.5))                     # target right beside the trunk
        s = MapGuidedSearch(b, trees=trees, map_half=13)
        target, vp = s.choose((5.0, 5.0))
        self.assertGreaterEqual(math.dist(vp, trees[0][:2]), map_search.TREE_CLEARANCE)
        self.assertTrue(abs(vp[0]) <= 13 - map_search.EDGE_MARGIN and abs(vp[1]) <= 13 - map_search.EDGE_MARGIN)
        self.assertTrue(line_of_sight_clear((vp[0], vp[1], 2.5), target, trees))

    def test_line_of_sight_blocked_by_canopy(self):
        trees = [(0.0, 0.0, 0.5)]
        self.assertFalse(line_of_sight_clear((2.6, 0.0, 2.5), (-0.4, 0.0), trees))   # through the tree
        self.assertTrue(line_of_sight_clear((-3.0, 0.0, 2.5), (-0.4, 0.0), trees))   # from its side

    def test_repeat_visit_looks_from_another_side(self):
        b = make_map(hot=(0.5, 0.5))
        s = MapGuidedSearch(b, map_half=13)
        _, vp1 = s.choose((-10.0, -10.0))
        _, vp2 = s.choose((-10.0, -10.0))
        self.assertEqual(s.streak, 2)
        self.assertNotEqual(vp1, vp2)


class CycleTests(unittest.TestCase):
    def test_fly_look_choose_and_observed(self):
        b = make_map(hot=(5.5, -5.5))
        s = MapGuidedSearch(b, map_half=13)
        vp = s.update((-5.0, 5.0, 2.5), 0.0, 1 / 240, 0.0)       # IDLE -> choose
        self.assertIsNotNone(vp)
        self.assertEqual(s.phase, "FLY")
        target = s.target
        seen = []
        in_view = lambda pt: seen.append(pt) or True
        nxt, look_s = fly_and_look(s, (vp[0], vp[1], 2.5), yaw=math.pi, in_view=in_view)
        want = math.atan2(target[1] - vp[1], target[0] - vp[0])
        self.assertLess(abs(map_search._wrap(s.heading - want)), 0.06)   # camera faced the target
        self.assertGreaterEqual(look_s, map_search.LOOK_S)
        self.assertTrue(s.log_rows[-1][3])                   # target cell was observed
        self.assertEqual(seen[0][:2], target)

    def test_sighting_extends_the_look_but_not_forever(self):
        b = make_map(hot=(5.5, -5.5))
        s = MapGuidedSearch(b, map_half=13, log=lambda *a: None)
        vp = s.update((-5.0, 5.0, 2.5), 0.0, 1 / 240, 0.0)
        _, look_s = fly_and_look(s, (vp[0], vp[1], 2.5), sighting=True)
        self.assertGreaterEqual(look_s, map_search.MAX_LOOK_S - 0.01)
        self.assertLess(look_s, map_search.MAX_LOOK_S + 5.0)

    def test_cancel_returns_to_idle(self):
        s = MapGuidedSearch(make_map(hot=(5.5, -5.5)), map_half=13)
        s.update((-5.0, 5.0, 2.5), 0.0, 1 / 240, 0.0)
        self.assertTrue(s.active)
        s.cancel()
        self.assertFalse(s.active)


class MissionTests(unittest.TestCase):
    def finished_plan(self):
        m = Mission()
        m.start()
        while m.destination:
            m.arrived()
        return m

    def test_ac02_state_stays_searching_with_status_text(self):
        m = self.finished_plan()
        msg = m.search_cell((-9.5, -7.5))
        self.assertEqual(m.state, "SEARCHING")
        self.assertEqual(msg, "Searching most likely cell (-9.5, -7.5)")
        self.assertEqual(m.response, msg)
        self.assertEqual(m.history[-1][1], "MAP")            # also in the operator panel log

    def test_ac05_time_up_reports_cell(self):
        m = self.finished_plan()
        m.time_up((-9.5, 1.5))
        self.assertEqual(m.state, "COMPLETE")
        self.assertIn("most likely remaining cell (-9.5, 1.5)", m.response)

    def test_time_up_after_found_changes_nothing(self):
        m = self.finished_plan()
        m.found()
        m.time_up((0.0, 0.0))
        self.assertEqual(m.state, "FOUND")

    def test_reinclude_after_plan_end_gives_zone_back(self):
        m = Mission()
        m.start()
        m.handle(Cmd.AVOID, "D")
        while m.destination:
            m.arrived()
        self.assertTrue(m.handle(Cmd.REINCLUDE, "D").ok)
        self.assertIsNotNone(m.destination)                # main.py cancels the map leg


class ProbabilityMapTests(unittest.TestCase):
    class Cam:                                   # minimal stand-in for SearchMonitor's camera
        range, tan_h, tan_v = 8.0, math.tan(math.radians(35)), math.tan(math.radians(35))
        def _axes(self):
            from target_detection import SearchMonitor
            return SearchMonitor._axes(self)
        pitch, yaw = math.radians(-40.0), 0.0

    def test_far_cells_lose_less_than_near_cells(self):
        b = make_map()
        before = b.grid.copy()
        b.observe_camera(self.Cam(), (0.0, 0.5, 2.5), 1.0)
        near = b.to_cell((2.5, 0.5)); far = b.to_cell((7.5, 0.5))
        drop = lambda c: b.grid[c[1], c[0]] / before[c[1], c[0]]
        self.assertLess(drop(near), drop(far))     # close look rules a cell out faster
        self.assertLess(drop(far), 1.0)            # but a far look still counts a little

    def test_heatmap_colour_scale(self):
        from probability_map import heat_colour
        self.assertEqual(heat_colour(0.0, 1)[:3], [1.0, 0.0, 0.0])          # red
        self.assertEqual(heat_colour(0.5, 1)[:3], [1.0, 1.0, 0.0])          # yellow
        self.assertEqual(heat_colour(1.0, 1)[:3][0], 0.0)                    # green
        self.assertGreater(heat_colour(1.0, 1)[1], 0.5)

    def test_zone_mass(self):
        b = make_map(hot=(6.5, 6.5))
        self.assertGreater(b.zone_mass((0.4, 0.4, 11.6, 11.6)), 0.6)   # hot cell + its share of background
        self.assertLess(b.zone_mass((-11.6, -11.6, -0.4, -0.4)), 0.1)


class ZoneOrderTests(unittest.TestCase):
    def test_search_area_square_and_centred(self):
        import main
        (x0, x1), (y0, y1) = main.search_bounds()
        self.assertEqual((x0, y0), (-x1, -y1))
        self.assertEqual(x1, y1)

    def test_uniform_map_keeps_abcd(self):
        import main
        xr, yr = main.search_bounds()
        b = ProbabilityMap(x_range=xr, y_range=yr, cell_size=max(1.0, (xr[1] - xr[0]) / 30))
        order, mass = main.zones_by_probability(b, half=13)
        self.assertEqual(order, ["A", "B", "C", "D"])
        self.assertAlmostEqual(max(mass.values()), min(mass.values()), places=3)

    def test_zones_sorted_by_probability(self):
        import main
        b = make_map(hot=(-6.5, -6.5))                 # zone C
        order, mass = main.zones_by_probability(b, half=13)
        self.assertEqual(order[0], "C")
        self.assertEqual(sorted(order), ["A", "B", "C", "D"])
        self.assertEqual([mass[z] for z in order], sorted(mass.values(), reverse=True))

    def test_mission_uses_order_and_return_to_plan_restores_it(self):
        m = Mission()
        m.set_plan_order(["C", "A", "D", "B"])
        m.start()
        from operator_guidance import ZONES
        self.assertEqual(m.destination, ZONES["C"])
        m.handle(Cmd.SEARCH_FIRST, "B")
        self.assertEqual(m.plan[0], "B")
        self.assertTrue(m.handle(Cmd.RETURN_PLAN).ok)
        self.assertEqual(m.plan, ["C", "A", "D", "B"])

    def test_order_rejected_after_start_or_when_invalid(self):
        m = Mission()
        self.assertIn("Rejected", m.set_plan_order(["A", "B"]))
        m.start()
        self.assertIn("Rejected", m.set_plan_order(["D", "C", "B", "A"]))
        self.assertEqual(m.plan[0], "A")


class PersonOptionTests(unittest.TestCase):
    def setUp(self):
        import main                                         # needs (stand-in) pybullet
        self.parse = main.parse_person

    def test_forms(self):
        self.assertEqual(self.parse(None), (None, None))
        self.assertEqual(self.parse(["child"]), ("child", None))
        self.assertEqual(self.parse(["-9", "-9"]), (None, (-9.0, -9.0)))
        self.assertEqual(self.parse(["hiker", "3.5", "-2"]), ("hiker", (3.5, -2.0)))

    def test_random_spot_rules(self):
        import main
        from env_sim import DRONE_START_POS, OBSTACLE_POSITIONS
        xr, yr = main.search_bounds()
        spots = [main.random_person_spot(seed, xr, yr) for seed in range(300)]
        self.assertEqual(spots[7], main.random_person_spot(7, xr, yr))      # same seed, same spot
        self.assertGreater(len(set(spots)), 250)                              # actually random
        for xy in spots:
            self.assertTrue(xr[0] + 1 <= xy[0] <= xr[1] - 1 and yr[0] + 1 <= xy[1] <= yr[1] - 1)
            self.assertGreaterEqual(math.dist(xy, DRONE_START_POS[:2]), main.PERSON_START_CLEARANCE)
            for t in OBSTACLE_POSITIONS:
                self.assertGreaterEqual(math.dist(xy, t[:2]), main.PERSON_TREE_CLEARANCE)

    def test_random_spot_never_red(self):
        import main
        xr, yr = main.search_bounds()
        b = ProbabilityMap(x_range=xr, y_range=yr, cell_size=1.0)
        m = np.ones_like(b.grid)
        m[b.cy < 0] = 0.0                      # south half is "water": weight 0 -> red
        b.set_initial(m)
        b.exclude_zone("B", main.zone_rect((-6.0, 6.0), 13.0))
        red = b.red_mask()
        for seed in range(300):
            xy = main.random_person_spot(seed, xr, yr, belief=b)
            ix, iy = b.to_cell(xy)
            self.assertFalse(red[iy, ix], xy)
            self.assertGreaterEqual(xy[1], 0.0)                        # never in the water
            self.assertFalse(-11.6 <= xy[0] <= -0.4 and 0.4 <= xy[1] <= 11.6, xy)   # not in B

    def test_bad_input(self):
        for bad in (["-9"], ["a", "b"], ["child", "1"], ["1", "2", "3"], ["nan", "1"]):
            with self.assertRaises(ValueError, msg=bad):
                self.parse(bad)


# ------------------------------------------------------------------ acceptance criteria
def _pybullet_available():
    try:
        import pybullet  # noqa: F401
        return True
    except ImportError:
        return False


@unittest.skipIf(os.environ.get("SKIP_SLOW") == "1", "SKIP_SLOW=1: end-to-end missions skipped")
@unittest.skipUnless(_pybullet_available(), "PyBullet is needed for the end-to-end missions")
class AcceptanceCriteriaTests(unittest.TestCase):
    """PBI-06 acceptance criteria, checked on real headless missions (main.py --headless).
    Each mission is run once and shared by the tests that need it."""

    _cache = {}

    @classmethod
    def mission(cls, person, *extra):
        """Run (once) `main.py --headless --person X Y extra...`; returns the parsed printout
        (see map_search_runs.run: state, found_time, targets, report, plan_end, bounds...)."""
        import map_search_runs
        key = (person, extra)
        if key not in cls._cache:
            cls._cache[key] = map_search_runs.run(person, list(extra))
        return cls._cache[key]

    @staticmethod
    def max_repeats(targets):
        run, worst = 0, 0
        for i, tgt in enumerate(targets):
            run = run + 1 if i and tgt == targets[i - 1] else 1
            worst = max(worst, run)
        return worst

    # AC-01: Given a person placed at (-9, -9), outside every zone,
    # `python main.py --headless --person -9 -9` ends with Mission: FOUND within 240 s.
    def test_ac01_person_outside_zones_found_within_240s(self):
        r = self.mission((-9.0, -9.0))
        self.assertEqual(r["state"], "FOUND", r["raw"][-2000:])
        self.assertIsNotNone(r["found_time"])
        self.assertLessEqual(r["found_time"], 240.0)

    # Same, but a spot only the map-guided phase can find (the zone sweep misses it).
    def test_ac01_map_phase_finds_person_the_zone_plan_misses(self):
        r = self.mission((11.5, 0.5))
        self.assertIsNotNone(r["plan_end"], "the zone plan should end before the find")
        self.assertGreater(len(r["targets"]), 0, "the map phase should have chosen targets")
        self.assertEqual(r["state"], "FOUND", r["raw"][-2000:])
        self.assertLessEqual(r["found_time"], 240.0)

    # Task: test with the person under a tree.
    def test_person_under_a_tree_is_found(self):
        r = self.mission((0.4, 1.3))                       # beside the tree at (0, 1)
        self.assertEqual(r["state"], "FOUND", r["raw"][-2000:])
        self.assertLessEqual(r["found_time"], 240.0)

    # AC-02: When the zone plan ends, the mission state stays SEARCHING and the status text shows
    # "Searching most likely cell (x, y)".
    def test_ac02_keeps_searching_after_plan_and_shows_status(self):
        r = self.mission((11.5, 0.5))
        self.assertIsNotNone(r["plan_end"])
        self.assertIn("Searching most likely cell", r["raw"])
        # the mission did not end when the plan ended: it went on to be FOUND later
        self.assertNotIn("Search plan complete", r["raw"])
        self.assertEqual(r["state"], "FOUND")
        self.assertGreater(r["found_time"], r["plan_end"])

    # AC-03: The drone never selects the same target cell more than 3 times in a row.
    def test_ac03_no_cell_more_than_3_times_in_a_row(self):
        for person, extra in (((11.5, 0.5), ()), ((0.5, 11.5), ("--max-time", "45")),
                              ((9.4, 8.8), ("--exclude-zones", "A", "--max-time", "120"))):
            r = self.mission(person, *extra)
            self.assertLessEqual(self.max_repeats(r["targets"]), 3, (person, r["targets"]))

    # AC-04: Every selected target is inside the search bounds and not inside an excluded area.
    def test_ac04_targets_in_bounds_and_outside_excluded_zone(self):
        import map_search_runs
        r = self.mission((9.4, 8.8), "--exclude-zones", "A", "--max-time", "120")
        self.assertGreater(len(r["targets"]), 0)
        (x0, x1), (y0, y1) = r["bounds"]
        a = map_search_runs.zone_rect(map_search_runs.ZONES["A"])
        for tx, ty in r["targets"]:
            self.assertTrue(x0 <= tx <= x1 and y0 <= ty <= y1, (tx, ty))
            self.assertFalse(a[0] <= tx <= a[2] and a[1] <= ty <= a[3], f"{(tx, ty)} is in zone A")

    # AC-05: If the time limit is reached without a find, the mission ends with state COMPLETE
    # and the final report lists the most likely remaining cell.
    def test_ac05_time_limit_complete_with_most_likely_cell(self):
        r = self.mission((0.5, 11.5), "--max-time", "45")
        self.assertEqual(r["state"], "COMPLETE", r["raw"][-2000:])
        self.assertIsNone(r["found_time"])
        self.assertRegex(r["report"], r"most likely remaining cell \(-?[\d.]+, -?[\d.]+\)")


if __name__ == "__main__":
    unittest.main()
