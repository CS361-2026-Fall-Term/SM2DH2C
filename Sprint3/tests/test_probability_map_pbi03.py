"""
Contributors: Sofia Monge and Mylan Bui
File Description: Unit tests for the PBI-03 parts of probability_map.py: loading the initial map,
operator sightings, and zone exclusions (Do not search / Re-include / Return to original plan).
Run from the folder holding probability_map.py and tests/:  python -m unittest discover tests -v
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from probability_map import ProbabilityMap
except ImportError as exc:                         # pybullet not installed
    ProbabilityMap = None
    IMPORT_ERROR = exc

ZONE_A = (2.0, 2.0, 8.0, 8.0)        # (x0, y0, x1, y1)
ZONE_C = (-8.0, -8.0, -2.0, -2.0)


def gaussian(pm, xy, sigma):
    g = np.exp(-((pm.cx - xy[0]) ** 2 + (pm.cy - xy[1]) ** 2) / (2 * sigma ** 2))
    return g / g.sum()


@unittest.skipIf(ProbabilityMap is None, "probability_map needs pybullet")
class TestInitialMapLoading(unittest.TestCase):
    def setUp(self):
        self.pm = ProbabilityMap(x_range=(-10, 10), y_range=(-10, 10), cell_size=1.0)
        rng = np.random.default_rng(0)
        self.initial = rng.random(self.pm.grid.shape)
        self.initial /= self.initial.sum()

    def test_set_initial_keeps_values(self):
        self.pm.set_initial(self.initial)
        np.testing.assert_allclose(self.pm.grid, self.initial, atol=1e-12)
        np.testing.assert_allclose(self.pm.prior, self.initial, atol=1e-12)

    def test_set_initial_rejects_wrong_shape(self):
        with self.assertRaises(ValueError):
            self.pm.set_initial(np.ones((3, 3)))


@unittest.skipIf(ProbabilityMap is None, "probability_map needs pybullet")
class TestSighting(unittest.TestCase):
    """AC-05: after a sighting the map equals (1 - w) * P_old + w * G (excluded cells 0, normalized)."""

    def test_sighting_formula(self):
        pm = ProbabilityMap(x_range=(-10, 10), y_range=(-10, 10), cell_size=1.0)
        old = pm.grid.copy()
        pm.apply_guidance("INVESTIGATE", "A", ZONE_A, sighting=(5.5, 5.5),
                          sighting_sigma=2.5, sighting_weight=0.5)
        expected = 0.5 * old + 0.5 * gaussian(pm, (5.5, 5.5), 2.5)
        np.testing.assert_allclose(pm.grid, expected / expected.sum(), atol=1e-9)
        self.assertEqual(pm.best_target(), (5.5, 5.5))

    def test_sighting_keeps_excluded_cells_at_zero(self):
        pm = ProbabilityMap(x_range=(-10, 10), y_range=(-10, 10), cell_size=1.0)
        pm.apply_guidance("AVOID", "A", ZONE_A)
        pm.apply_guidance("INVESTIGATE", "A", ZONE_A, sighting=(5, 5))
        self.assertTrue((pm.grid[pm._box(ZONE_A)] == 0).all())
        self.assertAlmostEqual(pm.grid.sum(), 1.0, places=9)


@unittest.skipIf(ProbabilityMap is None, "probability_map needs pybullet")
class TestZoneExclusion(unittest.TestCase):
    """AC-06: an excluded zone stays at 0 until it is re-included or the plan is reset."""

    def setUp(self):
        self.pm = ProbabilityMap(x_range=(-10, 10), y_range=(-10, 10), cell_size=1.0)
        d = np.hypot(self.pm.cx, self.pm.cy)
        initial = np.exp(-d / 5.0)
        self.pm.set_initial(initial / initial.sum())
        self.a = self.pm._box(ZONE_A)

    def test_excluded_cells_are_zero_and_map_sums_to_one(self):
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        self.assertTrue((self.pm.grid[self.a] == 0).all())
        self.assertTrue(self.pm.excluded[self.a].all())
        self.assertAlmostEqual(self.pm.grid.sum(), 1.0, places=9)
        self.assertEqual(self.pm.excluded_zones(), ["A"])

    def test_exclusion_persists_through_sightings_and_drone_coverage(self):
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        for xy in ((5, 5), (-5, 5), (4, 6)):
            self.pm.add_sighting(xy)
        for x in np.linspace(-9, 9, 40):
            self.pm.observe((x, 0.0))
        self.pm.diffuse(0.2)
        self.assertTrue((self.pm.grid[self.a] == 0).all())
        bx, by = self.pm.best_target()
        self.assertFalse(ZONE_A[0] <= bx <= ZONE_A[2] and ZONE_A[1] <= by <= ZONE_A[3])

    def test_reinclude_restores_starting_values(self):
        before = self.pm.grid.copy()
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        self.pm.apply_guidance("REINCLUDE", "A")
        self.assertFalse(self.pm.excluded.any())
        self.assertEqual(self.pm.excluded_zones(), [])
        np.testing.assert_allclose(self.pm.grid, before, atol=1e-12)

    def test_reinclude_one_zone_keeps_the_other_excluded(self):
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        self.pm.apply_guidance("AVOID", "C", ZONE_C)
        self.pm.apply_guidance("REINCLUDE", "A")
        self.assertTrue((self.pm.grid[self.a] > 0).all())
        self.assertTrue((self.pm.grid[self.pm._box(ZONE_C)] == 0).all())
        self.assertEqual(self.pm.excluded_zones(), ["C"])

    def test_return_to_plan_clears_every_zone(self):
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        self.pm.apply_guidance("AVOID", "C", ZONE_C)
        self.pm.apply_guidance("RETURN_PLAN")
        self.assertFalse(self.pm.excluded.any())
        self.assertAlmostEqual(self.pm.grid.sum(), 1.0, places=9)

    def test_search_first_does_not_change_the_map(self):
        before = self.pm.grid.copy()
        self.pm.apply_guidance("SEARCH_FIRST", "A", ZONE_A)
        np.testing.assert_array_equal(self.pm.grid, before)

    def test_excluding_twice_changes_nothing(self):
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        once = self.pm.grid.copy()
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        np.testing.assert_array_equal(self.pm.grid, once)

    def test_area_exclusion_survives_zone_reinclude(self):
        self.pm.exclude((5, 5), 1.5)                 # e.g. "already searched by a ground team"
        circle = self.pm._disk((5, 5), 1.5)
        self.pm.apply_guidance("AVOID", "A", ZONE_A)
        self.pm.apply_guidance("REINCLUDE", "A")
        self.assertTrue((self.pm.grid[circle] == 0).all())


if __name__ == "__main__":
    unittest.main()