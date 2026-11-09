"""
Contributors: Sofia Monge and Mylan Bui
File Description: Unit tests for terrain_prior.py (PBI-03 Steps 1-3).
Run from the folder holding terrain_prior.py and tests/:  python -m unittest discover tests -v
"""
import math
import os
import sys
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import terrain_prior as tp  # noqa: E402


def make_grid(half=10.0, cell=1.0):
    """Same layout as ProbabilityMap: cell centres, row 0 = south, x0/y0 = lower-left corner."""
    n = int(round(2 * half / cell))
    xs = -half + (np.arange(n) + 0.5) * cell
    cx, cy = np.meshgrid(xs, xs)
    return SimpleNamespace(cx=cx, cy=cy, cell=cell, x0=-half, y0=-half)


def make_scene(**features):
    scene = SimpleNamespace(surfaces=[], water=[], buildings=[], roads=[], barriers=[], trees=[],
                            bridges=[])
    for name, value in features.items():
        setattr(scene, name, value)
    return scene


def square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def cell_at(grid, x, y):
    return tp._cell_index(grid, x, y)


def expected_density(d, radii):
    inner = 0.0
    for outer, (_f, share) in zip(radii, tp.RING_TABLE):
        if inner <= d < outer:
            return share / (math.pi * (outer ** 2 - inner ** 2))
        inner = outer
    return 0.0


class TestTerrainClassification(unittest.TestCase):
    """Step 1: every OSM feature type maps to the right terrain type (PBI-03 AC-03b)."""

    def test_empty_scene_is_all_open(self):
        terrain = tp.terrain_grid(make_scene(), make_grid())
        self.assertTrue((terrain == tp.OPEN).all())

    def test_surface_categories(self):
        cases = [("forest", tp.FOREST), ("scrub", tp.FOREST), ("rock", tp.ROCK),
                 ("parking", tp.PAVED), ("plaza", tp.PAVED), ("grass", tp.OPEN),
                 ("water", tp.WATER)]
        grid = make_grid()
        for category, expected in cases:
            with self.subTest(category=category):
                scene = make_scene(surfaces=[(square(-5, -5, 5, 5), category)])
                terrain = tp.terrain_grid(scene, grid)
                self.assertEqual(terrain[cell_at(grid, 0.5, 0.5)], expected)
                self.assertEqual(terrain[cell_at(grid, 8.5, 8.5)], tp.OPEN)

    def test_line_categories(self):
        cases = [("path", tp.TRAIL), ("foot", tp.SIDEWALK), ("cycle", tp.SIDEWALK),
                 ("service", tp.ROAD), ("minor", tp.ROAD), ("major", tp.ROAD),
                 ("crossing", tp.ROAD), ("rail", tp.RAILWAY), ("stream", tp.WATER)]
        grid = make_grid()
        for category, expected in cases:
            with self.subTest(category=category):
                scene = make_scene(roads=[((-10, 0.5), (10, 0.5), 0.5, category)])
                terrain = tp.terrain_grid(scene, grid)
                self.assertEqual(terrain[cell_at(grid, -6.5, 0.5)], expected)
                self.assertEqual(terrain[cell_at(grid, -6.5, 6.5)], tp.OPEN)

    def test_water_building_barrier_tree(self):
        grid = make_grid()
        scene = make_scene(water=[square(-9, -9, -6, -6)],
                           buildings=[(square(6, 6, 9, 9), 4.0, {})],
                           barriers=[((-10, 4.5), (10, 4.5), "fence")],
                           trees=[(-7.5, 7.5)])
        terrain = tp.terrain_grid(scene, grid)
        self.assertEqual(terrain[cell_at(grid, -7.5, -7.5)], tp.WATER)
        self.assertEqual(terrain[cell_at(grid, 7.5, 7.5)], tp.BUILDING)
        self.assertEqual(terrain[cell_at(grid, 0.5, 4.5)], tp.BARRIER)
        self.assertEqual(terrain[cell_at(grid, -7.5, 7.5)], tp.FOREST)

    def test_small_building_still_marks_its_cell(self):
        grid = make_grid(cell=2.0)
        tiny = square(0.2, 0.2, 0.8, 0.8)       # smaller than a cell, misses every cell centre
        terrain = tp.terrain_grid(make_scene(buildings=[(tiny, 4.0, {})]), grid)
        self.assertEqual(terrain[cell_at(grid, 0.5, 0.5)], tp.BUILDING)

    def test_priority_order(self):
        grid = make_grid()
        scene = make_scene(
            surfaces=[(square(-10, -10, 10, 10), "forest")],
            water=[square(-10, -1, 10, 1)],                       # river across the middle
            roads=[((0.5, -10), (0.5, 10), 0.5, "major"),          # road crossing the river
                   ((3.5, -10), (3.5, 10), 0.5, "foot"),           # sidewalk ...
                   ((3.5, -10), (3.5, 10), 0.5, "minor")])         # ... on top of a road
        terrain = tp.terrain_grid(scene, grid)
        self.assertEqual(terrain[cell_at(grid, 0.5, 0.5)], tp.WATER)      # water beats road
        self.assertEqual(terrain[cell_at(grid, 0.5, 6.5)], tp.ROAD)       # road beats forest
        self.assertEqual(terrain[cell_at(grid, 3.5, 6.5)], tp.SIDEWALK)   # sidewalk beats road
        self.assertEqual(terrain[cell_at(grid, -6.5, 6.5)], tp.FOREST)

    def test_bridge_over_water_is_walkable(self):
        grid = make_grid()
        scene = make_scene(water=[square(-10, -2, 10, 2)],                        # river
                           bridges=[([(0.5, -10), (0.5, 10)], 0.5, "path")])     # footbridge
        terrain = tp.terrain_grid(scene, grid)
        self.assertEqual(terrain[cell_at(grid, 0.5, 0.5)], tp.TRAIL)     # on the bridge
        self.assertEqual(terrain[cell_at(grid, 6.5, 0.5)], tp.WATER)     # river beside it

    def test_unknown_categories_are_ignored(self):
        grid = make_grid()
        scene = make_scene(surfaces=[(square(-5, -5, 5, 5), "unknown")],
                           roads=[((-10, 0.5), (10, 0.5), 1.0, "unknown")])
        self.assertTrue((tp.terrain_grid(scene, grid) == tp.OPEN).all())


class TestTerrainWeights(unittest.TestCase):
    """Step 2: each cell gets the weight from the table."""

    def test_weights_follow_table(self):
        codes = list(range(len(tp.TERRAIN_NAMES)))
        weights = tp.terrain_weights(np.array([codes]))
        for code in codes:
            self.assertEqual(weights[0, code], tp.TERRAIN_WEIGHTS[code])
        self.assertEqual(tp.TERRAIN_WEIGHTS[tp.WATER], 0.0)
        self.assertEqual(tp.TERRAIN_WEIGHTS[tp.BUILDING], 0.0)

    def test_weight_ratio(self):
        self.assertAlmostEqual(tp.TERRAIN_WEIGHTS[tp.TRAIL] / tp.TERRAIN_WEIGHTS[tp.FOREST],
                               2.0 / 0.6, places=9)


class TestRingDensity(unittest.TestCase):
    """Step 3: distance-ring density around the LKP, scaled by person category."""

    def test_matches_formula_every_cell(self):
        grid = make_grid(half=10.0)
        density = tp.ring_density(grid, (0, 0), "hiker", map_half=10.0)
        radii = tp.ring_radii("hiker", 10.0)
        for iy in range(grid.cx.shape[0]):
            for ix in range(grid.cx.shape[1]):
                d = math.hypot(grid.cx[iy, ix], grid.cy[iy, ix])
                self.assertAlmostEqual(density[iy, ix], expected_density(d, radii), places=12)

    def test_inner_ring_denser_than_outer(self):
        grid = make_grid(half=10.0)
        density = tp.ring_density(grid, (0, 0), "hiker", map_half=10.0)
        self.assertGreater(density[cell_at(grid, 0.5, 0.5)], density[cell_at(grid, 6.5, 0.5)])

    def test_category_scales_radii(self):
        hiker = tp.ring_radii("hiker", 10.0)
        child = tp.ring_radii("child", 10.0)
        for h, c in zip(hiker, child):
            self.assertAlmostEqual(c, h * tp.PERSON_CATEGORIES["child"], places=12)

    def test_unknown_category_raises(self):
        with self.assertRaisesRegex(ValueError, "astronaut"):
            tp.ring_density(make_grid(), (0, 0), "astronaut", map_half=10.0)

    def test_rings_follow_lkp(self):
        grid = make_grid(half=10.0)
        density = tp.ring_density(grid, (5.0, -5.0), "hiker", map_half=10.0)
        self.assertEqual(density[cell_at(grid, 5.5, -5.5)], density.max())

    def test_lkp_centered_map_has_no_zero_cells(self):
        grid = make_grid(half=10.0)
        density = tp.ring_density(grid, (0, 0), "hiker", map_half=10.0)
        self.assertTrue((density > 0).all())     # outer ring (2.46 h) covers corners (1.41 h)


class TestWithRealOsmScene(unittest.TestCase):
    """End to end on osm_world's parser, using a tiny inline .osm export."""

    def test_parsed_scene_classifies(self):
        try:
            import osm_world
            import sim_config
        except ImportError:
            self.skipTest("osm_world / sim_config not found next to terrain_prior.py")
        sim_config.set_map_size(26.0)
        lat0, lon0 = 44.6488, -123.3046
        m_lat, m_lon = 111132.0, 111320.0 * math.cos(math.radians(lat0))

        def ll(x, y):
            return {"lat": lat0 + y / m_lat, "lon": lon0 + x / m_lon}

        data = {"elements": [
            {"type": "way", "tags": {"highway": "path"},
             "geometry": [ll(-13, 0.5), ll(13, 0.5)]},
            {"type": "way", "tags": {"highway": "footway"},
             "geometry": [ll(-13, 6.5), ll(13, 6.5)]},
            {"type": "way", "tags": {"natural": "water"},
             "geometry": [ll(-10, -10), ll(-6, -10), ll(-6, -6), ll(-10, -6), ll(-10, -10)]},
        ]}
        scene = osm_world.parse_scene(data, lat0, lon0, 26.0)
        grid = make_grid(half=13.0)
        terrain = tp.terrain_grid(scene, grid)
        self.assertEqual(terrain[cell_at(grid, 3.5, 0.5)], tp.TRAIL)
        self.assertEqual(terrain[cell_at(grid, 3.5, 6.5)], tp.SIDEWALK)
        self.assertEqual(terrain[cell_at(grid, -8.5, -8.5)], tp.WATER)


class TestInitialMap(unittest.TestCase):
    """Steps 4 and 6: ring density x terrain weight, normalized (PBI-03 AC-01 to AC-04, AC-08)."""

    def setUp(self):
        self.grid = make_grid(half=10.0)
        self.scene = make_scene(
            surfaces=[(square(-10, 2, 10, 10), "forest")],             # forest in the north
            water=[square(-10, -4, 10, -2)],                              # river in the south
            roads=[((-10, 0.5), (10, 0.5), 0.5, "path")],                 # trail through the middle
            buildings=[(square(6, -9, 9, -6), 4.0, {})])

    def test_ac01_every_cell_matches_formula(self):
        result, warning = tp.build_initial_map(self.scene, self.grid, (0, 0), "hiker", 10.0)
        self.assertIsNone(warning)
        raw = (tp.ring_density(self.grid, (0, 0), "hiker", 10.0)
               * tp.terrain_weights(tp.terrain_grid(self.scene, self.grid)))
        expected = raw / raw.sum()
        np.testing.assert_allclose(result, expected, rtol=0, atol=1e-9)

    def test_ac02_valid_probability_map(self):
        for lkp in ((0, 0), (5, 5), (-7, 3)):
            with self.subTest(lkp=lkp):
                result, _ = tp.build_initial_map(self.scene, self.grid, lkp, "hiker", 10.0)
                self.assertTrue((result >= 0).all())
                self.assertAlmostEqual(result.sum(), 1.0, places=9)

    def test_ac03_terrain_ratio_and_zero_cells(self):
        result, _ = tp.build_initial_map(self.scene, self.grid, (0, 0), "hiker", 10.0)
        terrain = tp.terrain_grid(self.scene, self.grid)
        trail, forest = cell_at(self.grid, -2.5, 0.5), cell_at(self.grid, -0.5, 2.5)
        d_trail = math.hypot(-2.5, 0.5)
        d_forest = math.hypot(-0.5, 2.5)
        radii = tp.ring_radii("hiker", 10.0)
        self.assertEqual(expected_density(d_trail, radii), expected_density(d_forest, radii))  # same ring
        self.assertAlmostEqual(result[trail] / result[forest], 2.0 / 0.6, places=6)
        self.assertTrue((result[terrain == tp.WATER] == 0).all())
        self.assertTrue((result[terrain == tp.BUILDING] == 0).all())

    def test_ac04_inner_ring_higher_for_same_terrain(self):
        result, _ = tp.build_initial_map(make_scene(), self.grid, (0, 0), "hiker", 10.0)
        self.assertGreater(result[cell_at(self.grid, 0.5, 0.5)], result[cell_at(self.grid, 7.5, 0.5)])

    def test_practice_world_without_scene(self):
        result, warning = tp.build_initial_map(None, self.grid, (0, 0), "hiker", 10.0)
        self.assertIsNone(warning)
        self.assertAlmostEqual(result.sum(), 1.0, places=9)

    def test_fallback_when_nothing_is_walkable(self):
        all_water = make_scene(water=[square(-10, -10, 10, 10)])
        result, warning = tp.build_initial_map(all_water, self.grid, (0, 0), "hiker", 10.0)
        self.assertIn("distance only", warning)
        self.assertAlmostEqual(result.sum(), 1.0, places=9)

    def test_unknown_category_raises(self):
        with self.assertRaises(ValueError):
            tp.build_initial_map(self.scene, self.grid, (0, 0), "astronaut", 10.0)

    def test_ac08_largest_map_under_one_second(self):
        import time
        grid = make_grid(half=30.0, cell=2.0)                 # 60 m map, 30 x 30 cells (main.py)
        roads = [((-30, y), (30, y + 3), 1.5, "path") for y in range(-28, 28, 4)]
        water = [square(x, -30, x + 2, 30) for x in range(-25, 25, 10)]
        buildings = [(square(x, x, x + 3, x + 3), 4.0, {}) for x in range(-27, 27, 6)]
        scene = make_scene(roads=roads, water=water, buildings=buildings,
                           surfaces=[(square(-30, -30, 0, 0), "forest")])
        t0 = time.perf_counter()
        tp.build_initial_map(scene, grid, (0, 0), "hiker", 30.0)
        self.assertLess(time.perf_counter() - t0, 1.0)


if __name__ == "__main__":
    unittest.main()
    