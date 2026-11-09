"""
Contributors: Sofia Monge and Mylan Bui
File Description: Integration tests for PBI-03 in main.py: the initial map is built from the LKP and
terrain, zones excluded in the setup window are applied at START, and operator commands update the
probability map. Uses the real Mission and ProbabilityMap; navigation is replaced by a stub, so no
PyBullet window is opened.
Run from the folder holding main.py and tests/:  python -m unittest discover tests -v
"""
import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    import main
    import sim_config
    from operator_guidance import Mission, Cmd, ZONES
    from probability_map import ProbabilityMap
except ImportError as exc:                         # pybullet / project modules missing
    main = None
    IMPORT_ERROR = exc


def make_belief():
    half = sim_config.MAP_HALF
    return ProbabilityMap(x_range=(-half, half), y_range=(-half, half), cell_size=1.0)


class RerouteSpy:
    def __init__(self):
        self.calls = []

    def __call__(self, nav, xy):
        self.calls.append(tuple(xy))


@unittest.skipIf(main is None, "main.py dependencies not available")
class TestInitialBelief(unittest.TestCase):
    def test_practice_world_map_is_valid(self):
        belief = make_belief()
        warning = main.build_initial_belief(belief, None, "hiker")
        self.assertIsNone(warning)
        self.assertAlmostEqual(belief.grid.sum(), 1.0, places=9)
        import math
        import terrain_prior
        inner_ring = terrain_prior.ring_radii("hiker", sim_config.MAP_HALF)[0]
        self.assertLess(math.hypot(*belief.best_target()), inner_ring)   # peak in the first ring

    def test_zone_rect_is_the_drawn_square(self):
        half = sim_config.MAP_HALF
        for z, xy in ZONES.items():
            with self.subTest(zone=z):
                h = main.zone_box_half(xy, half)
                self.assertEqual(main.zone_rect(xy, half),
                                 (xy[0] - h, xy[1] - h, xy[0] + h, xy[1] + h))


@unittest.skipIf(main is None, "main.py dependencies not available")
class TestStartWithExcludedZones(unittest.TestCase):
    def test_setup_exclusions_applied_at_start(self):
        mission, belief, spy = Mission(), make_belief(), RerouteSpy()
        main.start_mission(mission, SimpleNamespace(), belief, ["A"], reroute=spy)
        self.assertEqual(mission.state, "SEARCHING")
        self.assertEqual(mission.excluded, ["A"])
        self.assertNotIn("A", mission.plan)
        self.assertEqual(belief.excluded_zones(), ["A"])
        self.assertEqual(spy.calls[-1], tuple(ZONES[mission.plan[0]]))     # heading to the next zone

    def test_second_start_does_nothing(self):
        mission, belief, spy = Mission(), make_belief(), RerouteSpy()
        main.start_mission(mission, SimpleNamespace(), belief, [], reroute=spy)
        calls = len(spy.calls)
        msg = main.start_mission(mission, SimpleNamespace(), belief, ["A"], reroute=spy)
        self.assertIn("Rejected", msg)
        self.assertEqual(len(spy.calls), calls)
        self.assertEqual(mission.excluded, [])


@unittest.skipIf(main is None, "main.py dependencies not available")
class TestCommandsUpdateMap(unittest.TestCase):
    def setUp(self):
        self.mission, self.belief, self.spy = Mission(), make_belief(), RerouteSpy()
        main.start_mission(self.mission, SimpleNamespace(), self.belief, reroute=self.spy)

    def command(self, cmd, zone=None):
        return main.apply_operator_command(self.mission, cmd, zone, SimpleNamespace(),
                                           self.belief, reroute=self.spy)

    def test_do_not_search_excludes_zone_cells(self):
        self.assertTrue(self.command(Cmd.AVOID, "C").ok)
        rect = main.zone_rect(ZONES["C"], sim_config.MAP_HALF)
        self.assertTrue((self.belief.grid[self.belief._box(rect)] == 0).all())

    def test_reinclude_and_return_to_plan_restore_cells(self):
        self.command(Cmd.AVOID, "C")
        self.assertTrue(self.command(Cmd.REINCLUDE, "C").ok)
        self.assertEqual(self.belief.excluded_zones(), [])
        self.command(Cmd.AVOID, "B")
        self.command(Cmd.AVOID, "C")
        self.assertTrue(self.command(Cmd.RETURN_PLAN).ok)
        self.assertEqual(self.belief.excluded_zones(), [])

    def test_rejected_command_leaves_map_unchanged(self):
        before = self.belief.grid.copy()
        res = self.command(Cmd.REINCLUDE, "C")            # C is not excluded -> rejected
        self.assertFalse(res.ok)
        self.assertTrue((self.belief.grid == before).all())

    def test_investigate_raises_probability_in_zone(self):
        rect = main.zone_rect(ZONES["D"], sim_config.MAP_HALF)
        before = self.belief.grid[self.belief._box(rect)].sum()
        self.assertTrue(self.command(Cmd.INVESTIGATE, "D").ok)
        self.assertGreater(self.belief.grid[self.belief._box(rect)].sum(), before)


if __name__ == "__main__":
    unittest.main()
