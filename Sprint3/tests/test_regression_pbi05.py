"""
PBI-05 tests: automated regression suite for the probability map, route planning and headless
end-to-end missions.

  AC-01  The suite covers ProbabilityMap (normalization, add_sighting, exclude, best_target,
         bounds) and route planning, and every test passes.
  AC-02  The headless scenarios end with `Mission: FOUND` for the default and outside-plan cases.
  AC-03  Deliberately breaking ProbabilityMap._normalize() makes at least one test fail.
  AC-04  On a small known grid, a planned route never passes through a blocked cell, and no route
         is returned when the goal is blocked, walled in or off the map.

Run from the Sprint folder:
    python -m unittest tests.test_regression_pbi05 -v                  # everything
    SKIP_SLOW=1 python -m unittest tests.test_regression_pbi05 -v      # skip the 3 headless missions
    BREAK_NORMALIZE=1 python -m unittest tests.test_regression_pbi05.ProbabilityMapTests -v
        # AC-03 demonstration: _normalize() is replaced with a do-nothing version for this run
        # (the source file is not changed) and several tests must FAIL.
The AC-03 check also runs on its own in BrokenNormalizeTests (no variable needed).
"""
import math
import os
import sys
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import numpy as np

import env_sim
from object_avoidance import ObstacleMap, CELL
from probability_map import ProbabilityMap

SLOW = os.environ.get("SKIP_SLOW") == "1"


def _broken_normalize(self):
    """AC-03: a broken _normalize(): excluded cells are not zeroed and nothing is rescaled."""
    return None


if os.environ.get("BREAK_NORMALIZE") == "1":
    ProbabilityMap._normalize = _broken_normalize
    print("\n*** BREAK_NORMALIZE=1: ProbabilityMap._normalize() is deliberately broken ***\n")


def small_map(**kw):
    """10 m x 10 m map, 1 m cells (10 x 10 = 100 cells), centred on (0, 0)."""
    return ProbabilityMap(x_range=(-5, 5), y_range=(-5, 5), cell_size=1.0, **kw)


def cell_value(pm, xy):
    ix, iy = pm.to_cell(xy)
    return float(pm.grid[iy, ix])


# ======================================================================= AC-01: probability map
class ProbabilityMapTests(unittest.TestCase):
    """Normalization, add_sighting, exclude, best_target and bounds handling."""

    def assertSumsToOne(self, pm):
        self.assertAlmostEqual(float(pm.grid.sum()), 1.0, places=9)

    # ---------------------------------------------------------------- normalization
    def test_new_map_is_uniform_and_sums_to_one(self):
        for xr, yr, cell in (((-5, 5), (-5, 5), 1.0), ((-13, 13), (-11, 13), 1.0),
                             ((-20, 20), (-20, 20), 2.0)):
            with self.subTest(x_range=xr, y_range=yr, cell=cell):
                pm = ProbabilityMap(x_range=xr, y_range=yr, cell_size=cell)
                self.assertSumsToOne(pm)
                self.assertTrue(np.allclose(pm.grid, pm.grid.flat[0]))

    def test_normalize_rescales_to_one(self):
        pm = small_map()
        pm.grid *= 7.3
        pm.grid[2, 3] += 5.0
        pm._normalize()
        self.assertSumsToOne(pm)

    def test_normalize_zeroes_excluded_cells(self):
        pm = small_map()
        pm.excluded[:, :3] = True                 # three columns marked "do not search"
        pm.grid[:] = 1.0
        pm._normalize()
        self.assertTrue(np.all(pm.grid[:, :3] == 0.0))
        self.assertSumsToOne(pm)

    def test_normalize_all_zero_resets_to_uniform_over_allowed_cells(self):
        pm = small_map()
        pm.excluded[0, :] = True
        pm.grid[:] = 0.0                          # every cell searched with no find
        pm._normalize()
        allowed = ~pm.excluded
        self.assertSumsToOne(pm)
        self.assertTrue(np.allclose(pm.grid[allowed], 1.0 / allowed.sum()))
        self.assertTrue(np.all(pm.grid[pm.excluded] == 0.0))

    def test_drone_observation_lowers_cell_and_keeps_total(self):
        pm = small_map(sensor_radius=1.5, pod=0.8)
        before = cell_value(pm, (0.5, 0.5))
        pm.observe((0.5, 0.5))
        self.assertLess(cell_value(pm, (0.5, 0.5)), before)
        self.assertSumsToOne(pm)

    # ---------------------------------------------------------------- add_sighting
    def test_sighting_raises_probability_near_the_point(self):
        pm = small_map()
        pm.add_sighting((2.5, 2.5), sigma=1.0, weight=0.6)
        self.assertGreater(cell_value(pm, (2.5, 2.5)), cell_value(pm, (-3.5, -3.5)))
        self.assertGreater(cell_value(pm, (2.5, 2.5)), 1.0 / 100)
        self.assertSumsToOne(pm)

    def test_sighting_moves_best_target_to_the_sighting(self):
        pm = small_map()
        pm.add_sighting((-3.5, 1.5), sigma=1.0, weight=0.8)
        self.assertEqual(pm.best_target(), (-3.5, 1.5))

    def test_sighting_with_zero_trust_changes_nothing(self):
        pm = small_map()
        before = pm.grid.copy()
        pm.add_sighting((1.5, 1.5), weight=0.0)
        self.assertTrue(np.allclose(pm.grid, before))

    def test_sighting_does_not_revive_excluded_cells(self):
        pm = small_map()
        pm.exclude((2.5, 2.5), radius=1.0)
        pm.add_sighting((2.5, 2.5), sigma=1.0, weight=0.9)
        self.assertEqual(cell_value(pm, (2.5, 2.5)), 0.0)
        self.assertSumsToOne(pm)

    # ---------------------------------------------------------------- exclude
    def test_exclude_sets_area_to_zero_and_renormalizes(self):
        pm = small_map()
        pm.exclude((0.0, 0.0), radius=2.0)
        inside = pm._disk((0.0, 0.0), 2.0)
        self.assertTrue(inside.any())
        self.assertTrue(np.all(pm.grid[inside] == 0.0))
        self.assertTrue(np.all(pm.grid[~inside] > 0.0))
        self.assertSumsToOne(pm)

    def test_best_target_never_inside_an_excluded_area(self):
        pm = small_map()
        pm.add_sighting((0.5, 0.5), sigma=1.0, weight=0.9)      # hottest spot ...
        pm.exclude((0.5, 0.5), radius=1.5)                      # ... is then excluded
        tx, ty = pm.best_target()
        self.assertGreater(math.hypot(tx - 0.5, ty - 0.5), 1.5)

    def test_excluding_everything_leaves_no_target(self):
        pm = small_map()
        pm.exclude((0.0, 0.0), radius=50.0)
        self.assertIsNone(pm.best_target())

    def test_excluded_zone_can_be_reincluded(self):
        pm = small_map()
        rect = (1.0, 1.0, 5.0, 5.0)
        pm.exclude_zone("A", rect)
        self.assertEqual(pm.zone_mass(rect), 0.0)
        pm.include_zone("A")
        self.assertGreater(pm.zone_mass(rect), 0.0)
        self.assertSumsToOne(pm)

    # ---------------------------------------------------------------- best_target
    def test_best_target_is_the_most_likely_cell_centre(self):
        pm = small_map()
        m = np.full(pm.grid.shape, 1e-3)
        ix, iy = pm.to_cell((3.2, -1.7))
        m[iy, ix] = 1.0
        pm.set_initial(m)
        self.assertEqual(pm.best_target(), (3.5, -1.5))

    def test_distance_weight_prefers_the_nearer_of_two_equal_cells(self):
        pm = small_map()
        m = np.full(pm.grid.shape, 1e-3)
        for xy in ((-4.5, -4.5), (4.5, 4.5)):
            ix, iy = pm.to_cell(xy)
            m[iy, ix] = 1.0
        pm.set_initial(m)
        self.assertEqual(pm.best_target(from_xy=(3.0, 3.0), distance_weight=1.0), (4.5, 4.5))
        self.assertEqual(pm.best_target(from_xy=(-3.0, -3.0), distance_weight=1.0), (-4.5, -4.5))

    def test_min_dist_and_skip_leave_cells_out(self):
        pm = small_map()
        pm.add_sighting((0.5, 0.5), sigma=1.0, weight=0.9)
        far = pm.best_target(from_xy=(0.5, 0.5), min_dist=2.0)
        self.assertGreaterEqual(math.hypot(far[0] - 0.5, far[1] - 0.5), 2.0)
        self.assertNotEqual(pm.best_target(skip=[pm.to_cell((0.5, 0.5))]), (0.5, 0.5))

    # ---------------------------------------------------------------- bounds handling
    def test_points_outside_or_on_the_edge_map_to_an_edge_cell(self):
        pm = small_map()
        for xy, cell in (((-100.0, 0.2), (0, 5)), ((100.0, 100.0), (9, 9)),
                         ((-5.0, -5.0), (0, 0)), ((5.0, 5.0), (9, 9))):
            with self.subTest(xy=xy):
                self.assertEqual(pm.to_cell(xy), cell)

    def test_sighting_outside_the_map_keeps_a_valid_map(self):
        pm = small_map()
        pm.add_sighting((30.0, 0.5), sigma=2.0, weight=0.7)
        self.assertTrue(np.all(np.isfinite(pm.grid)))
        self.assertSumsToOne(pm)
        self.assertEqual(pm.best_target()[0], 4.5)               # pulled to the nearest (east) edge

    def test_best_target_is_always_inside_the_bounds(self):
        rng = np.random.default_rng(5)
        pm = small_map()
        for _ in range(25):
            pm.add_sighting(tuple(rng.uniform(-15, 15, 2)), sigma=1.5, weight=0.5)
            tx, ty = pm.best_target(from_xy=tuple(rng.uniform(-5, 5, 2)), distance_weight=0.3)
            self.assertTrue(-5 < tx < 5 and -5 < ty < 5, (tx, ty))


# ======================================================================= AC-03: broken _normalize
class BrokenNormalizeTests(unittest.TestCase):
    """AC-03: with _normalize() broken on purpose, the ProbabilityMap tests must catch it."""

    def test_broken_normalize_makes_tests_fail(self):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ProbabilityMapTests)
        with mock.patch.object(ProbabilityMap, "_normalize", _broken_normalize):
            result = unittest.TestResult()
            suite.run(result)
        failed = [t.id().split(".")[-1] for t, _ in result.failures + result.errors]
        print(f"\n  _normalize() broken: {len(failed)} of {result.testsRun} ProbabilityMap "
              f"tests fail: {', '.join(failed)}")
        self.assertGreaterEqual(len(failed), 1)

    def test_real_normalize_is_back_after_the_check(self):
        self.assertIsNot(ProbabilityMap._normalize, _broken_normalize)


# ======================================================================= AC-01 / AC-04: route planning
ALT = 1.5          # flight altitude for the route tests


def wall(m, x0, y0, x1, y1, top=None, z=1.0):
    """Sensor hits every 0.1 m along a line (top=None: unknown height, blocked all the way up)."""
    n = int(math.hypot(x1 - x0, y1 - y0) / 0.1) + 1
    for k in range(n + 1):
        t = k / n
        m.add_hit((x0 + t * (x1 - x0), y0 + t * (y1 - y0), z), top=top)


def leg_points(start, path, step=0.05):
    """Every point along the flown route, sampled every `step` m."""
    pts, a = [], start
    for b in path:
        d = math.hypot(b[0] - a[0], b[1] - a[1])
        n = max(1, int(d / step))
        pts += [(a[0] + (b[0] - a[0]) * i / n, a[1] + (b[1] - a[1]) * i / n) for i in range(n + 1)]
        a = b
    return pts


class RoutePlanningTests(unittest.TestCase):
    """ObstacleMap.astar on a small known grid: 6 m x 6 m ([-3, 3]), 0.25 m cells -> 24 x 24."""

    def setUp(self):
        self.grid = ObstacleMap()
        self.grid.half, self.grid.n = 3.0, int(round(6.0 / CELL))

    def assertRouteClear(self, start, path):
        for x, y in leg_points(start, path):
            self.assertFalse(self.grid.blocked(x, y, ALT), f"route passes blocked point {(x, y)}")

    def test_empty_grid_gives_a_straight_route(self):
        path = self.grid.astar((-2.0, -2.0), (2.0, 2.0), ALT)
        self.assertEqual(path, [(2.0, 2.0)])

    # AC-04
    def test_route_avoids_a_wall_through_its_gap(self):
        wall(self.grid, 0.0, -3.0, 0.0, -2.3)             # wall across x = 0 with a gap
        wall(self.grid, 0.0, -0.7, 0.0, 3.0)              # between y = -2.3 and -0.7
        start, goal = (-2.0, 1.0), (2.0, 1.0)
        self.assertFalse(self.grid.line_free(start, goal, ALT))      # straight line is blocked
        path = self.grid.astar(start, goal, ALT)
        self.assertIsNotNone(path)
        self.assertEqual(path[-1], goal)
        self.assertRouteClear(start, path)
        crossing = [y for x, y in leg_points(start, path) if abs(x) < 0.05]
        self.assertTrue(crossing and all(-2.3 < y < -0.7 for y in crossing), crossing)

    # AC-04
    def test_route_around_the_end_of_a_wall_stays_in_the_grid(self):
        wall(self.grid, 0.0, -2.0, 0.0, 3.0)
        start = (-2.0, 2.0)
        path = self.grid.astar(start, (2.0, 2.0), ALT)
        self.assertIsNotNone(path)
        self.assertRouteClear(start, path)
        self.assertTrue(all(abs(x) <= 3.0 and abs(y) <= 3.0 for x, y in leg_points(start, path)))

    # AC-04
    def test_goal_inside_an_obstacle_has_no_route(self):
        wall(self.grid, 1.0, 1.0, 1.0, 1.2)
        self.assertTrue(self.grid.blocked(1.0, 1.0, ALT))
        self.assertIsNone(self.grid.astar((-2.0, -2.0), (1.0, 1.0), ALT))

    # AC-04
    def test_walled_in_goal_has_no_route(self):
        for a, b in (((1.0, 1.0), (2.6, 1.0)), ((2.6, 1.0), (2.6, 2.6)),
                     ((2.6, 2.6), (1.0, 2.6)), ((1.0, 2.6), (1.0, 1.0))):
            wall(self.grid, *a, *b)
        self.assertFalse(self.grid.blocked(1.8, 1.8, ALT))           # the goal cell itself is free
        self.assertIsNone(self.grid.astar((-2.0, -2.0), (1.8, 1.8), ALT))

    # AC-04
    def test_goal_off_the_map_has_no_route(self):
        self.assertIsNone(self.grid.astar((0.0, 0.0), (10.0, 0.0), ALT))

    def test_low_obstacle_blocks_only_low_flight_levels(self):
        wall(self.grid, 0.0, -3.0, 0.0, 3.0, top=0.5, z=0.3)        # 0.5 m wall across the map
        self.assertIsNone(self.grid.astar((-2.0, 0.0), (2.0, 0.0), 0.5))
        self.assertEqual(self.grid.astar((-2.0, 0.0), (2.0, 0.0), 2.0), [(2.0, 0.0)])


# ======================================================================= AC-02: headless missions
def _pybullet_available():
    try:
        import pybullet  # noqa: F401
        return True
    except ImportError:
        return False


DEFAULT_PERSON = tuple(env_sim.TARGET_POSITION[:2])        # (7.0, -5.5), the original target
UNDER_TREE = (0.4, 1.3)                                    # beside the tree at (0, 1)
OUTSIDE_PLAN = (-9.0, -9.0)                                # outside every zone A-D
MISSIONS = {"default": DEFAULT_PERSON, "under_tree": UNDER_TREE, "outside_plan": OUTSIDE_PLAN}
TIME_LIMIT_S = 240.0


@unittest.skipIf(SLOW, "SKIP_SLOW=1: headless missions skipped")
@unittest.skipUnless(_pybullet_available(), "PyBullet is needed for the headless missions")
class HeadlessMissionTests(unittest.TestCase):
    """Runs `python main.py --headless --person X Y` for each scenario (the three in parallel,
    once for the whole class) and checks the printout."""

    @classmethod
    def setUpClass(cls):
        import map_search_runs
        with ThreadPoolExecutor(max_workers=len(MISSIONS)) as pool:
            jobs = {name: pool.submit(map_search_runs.run, xy, []) for name, xy in MISSIONS.items()}
            cls.runs = {name: job.result() for name, job in jobs.items()}

    def assertFound(self, name):
        r = self.runs[name]
        self.assertEqual(r["state"], "FOUND", r["raw"][-2000:])
        self.assertIsNotNone(r["found_time"])
        self.assertLessEqual(r["found_time"], TIME_LIMIT_S)

    # AC-02
    def test_default_person_is_found(self):
        self.assertFound("default")

    # AC-02
    def test_person_outside_the_zone_plan_is_found(self):
        self.assertFound("outside_plan")

    def test_person_under_a_tree_is_found(self):
        self.assertFound("under_tree")

    def test_missions_end_cleanly(self):
        for name, r in self.runs.items():
            with self.subTest(mission=name):
                self.assertNotIn("Traceback", r["raw"], r["raw"][-2000:])
                self.assertTrue(r["report"], "no final report")


if __name__ == "__main__":
    unittest.main()
