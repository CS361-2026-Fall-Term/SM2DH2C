"""
Contributors: Mylan Bui
PBI-04 tests: obstacle-safe A* routes in generated environments.
No running simulator needed (sensor hits are added to the obstacle map directly), but
object_avoidance.py imports PyBullet, so these are skipped if it isn't installed.
Run from the Sprint folder:  python -m unittest tests.test_route_safety -v
Full 20-seed check of the ACs (with physics): python route_safety_runs.py

Covers
  AC-01  A* routes through a gap without crossing the obstacle margin;
         generate_layout keeps trees off the start/waypoints and apart from each other
  AC-02  a walled-in waypoint is skipped and logged by name; the drone heads to the next one
  AC-03  every replan is timed; a walled-in goal is rejected in well under 50 ms
  AC-04  the same seed gives the same layout; the same map gives the same route
"""
import math
import os
import sys
import time
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

try:
    import env_sim
    import object_avoidance as oa
    from object_avoidance import Navigator, ObstacleMap
    HAVE_SIM = True
except ImportError:                     # PyBullet not installed
    HAVE_SIM = False

ALT = 2.5                               # cruise altitude used by FLIGHT_PATH
REPLAN_LIMIT_MS = 50.0


def add_wall_line(m, x, y0, y1, z=ALT, step=0.1, gap=None, top=None):
    """Sensor hits along the line x = const from y0 to y1, leaving out |y - gap[0]| < gap[1].
    top: known top of the wall (None = unknown, blocked all the way up; PBI-12)."""
    y = y0
    while y <= y1 + 1e-9:
        if gap is None or abs(y - gap[0]) >= gap[1]:
            m.add_hit((x, y, z), top=top)
        y += step


def add_pen(m, cx, cy, half=1.5, z=ALT, step=0.1, top=None):
    """Sensor hits on all 4 sides of a square pen around (cx, cy), like generate_layout's walls."""
    k = int(round(2 * half / step))
    for i in range(k + 1):
        t = -half + i * step
        for pt in ((cx + t, cy + half), (cx + t, cy - half), (cx + half, cy + t), (cx - half, cy + t)):
            m.add_hit((pt[0], pt[1], z), top=top)


@unittest.skipUnless(HAVE_SIM, "PyBullet not installed")
class TestGenerateLayout(unittest.TestCase):
    """env_sim.generate_layout(): seeded, repeatable obstacle worlds."""

    def setUp(self):
        # generate_layout changes these lists in place; put them back after each test
        self.saved = (list(env_sim.OBSTACLE_POSITIONS), list(env_sim.TREE_KINDS), list(env_sim.WALLS))
        self.keep = [tuple(w[:2]) for w in env_sim.FLIGHT_PATH[1:]]

    def tearDown(self):
        env_sim.OBSTACLE_POSITIONS[:], env_sim.TREE_KINDS[:], env_sim.WALLS[:] = self.saved

    def layout(self, seed, count, enclose=None):
        env_sim.generate_layout(seed, count, keep_clear=self.keep, enclose=enclose)
        return [list(o) for o in env_sim.OBSTACLE_POSITIONS], list(env_sim.TREE_KINDS), list(env_sim.WALLS)

    def test_same_seed_same_layout(self):                                  # AC-04
        for seed in (0, 7, 19):
            self.assertEqual(self.layout(seed, 12), self.layout(seed, 12), f"seed {seed}")

    def test_different_seed_different_layout(self):
        self.assertNotEqual(self.layout(1, 12)[0], self.layout(2, 12)[0])

    def test_obstacle_count_8_to_20(self):                                 # AC-01
        for count in (8, 14, 20):
            trees, kinds, walls = self.layout(count, count)
            self.assertEqual(len(trees) + len(walls), count)
            self.assertEqual(len(kinds), len(trees))

    def test_trees_keep_clear_of_start_and_waypoints(self):                # AC-01
        trees, _, _ = self.layout(3, 20)
        start = tuple(env_sim.DRONE_START_POS[:2])
        for x, y, _ in trees:
            for px, py in [start] + self.keep:
                self.assertGreaterEqual(math.hypot(x - px, y - py), 1.5,
                                        f"tree ({x}, {y}) too close to ({px}, {py})")

    def test_trees_spaced_apart(self):
        trees, _, _ = self.layout(5, 20)
        for i, (x1, y1, _) in enumerate(trees):
            for x2, y2, _ in trees[i + 1:]:
                self.assertGreaterEqual(math.hypot(x1 - x2, y1 - y2), 1.2)

    def test_enclose_builds_4_walls_too_tall_to_climb(self):               # AC-02
        ex, ey = self.keep[1]
        trees, _, walls = self.layout(11, 12, enclose=(ex, ey))
        self.assertEqual(len(walls), 4)
        self.assertEqual(len(trees), 12 - 4)
        for wall in walls:
            self.assertGreater(wall[4], oa.MAX_ALT + oa.VERT_CLEAR, "drone could climb over this wall")
        for x, y, _ in trees:                                  # no tree inside or on the pen
            self.assertTrue(abs(x - ex) >= 3.0 or abs(y - ey) >= 3.0, f"tree ({x}, {y}) in the pen")

    def test_no_enclose_means_no_walls(self):
        self.assertEqual(self.layout(4, 10)[2], [])


@unittest.skipUnless(HAVE_SIM, "PyBullet not installed")
class TestAStar(unittest.TestCase):
    """ObstacleMap.astar(): routes around sensed obstacles, rejects walled-in goals fast."""

    def gap_map(self):
        # A wall across the whole map at x = 0 with a 1.6 m opening around y = 0.
        m = ObstacleMap()
        add_wall_line(m, 0.0, -m.half, m.half, gap=(0.0, 0.8))
        return m

    def test_routes_through_gap(self):                                     # AC-01
        m = self.gap_map()
        start, goal = (-5.0, 5.0), (5.0, -5.0)
        path = m.astar(start, goal, ALT)
        self.assertIsNotNone(path, "no route found through the opening")
        self.assertEqual(path[-1], goal)
        pts = [start] + path
        crossed = False
        for a, b in zip(pts, pts[1:]):
            self.assertTrue(m.line_free(a, b, ALT), f"leg {a} -> {b} crosses the obstacle margin")
            if (a[0] < 0) != (b[0] < 0):                       # this leg crosses the wall line
                y = a[1] + (b[1] - a[1]) * (0 - a[0]) / (b[0] - a[0])
                self.assertLess(abs(y), 0.8, f"route crosses the wall at y = {y:.2f}, not the gap")
                crossed = True
        self.assertTrue(crossed)

    def test_no_route_when_gap_closed(self):
        m = ObstacleMap()
        add_wall_line(m, 0.0, -m.half, m.half)                 # no opening
        self.assertIsNone(m.astar((-5.0, 5.0), (5.0, -5.0), ALT))

    def test_walled_goal_rejected_fast(self):                              # AC-02 / AC-03
        m = ObstacleMap()
        add_pen(m, 5.0, 5.0)
        t0 = time.perf_counter()
        path = m.astar((-10.0, -10.0), (5.0, 5.0), ALT)
        ms = (time.perf_counter() - t0) * 1000.0
        self.assertIsNone(path)
        self.assertLess(ms, REPLAN_LIMIT_MS, f"walled-goal check took {ms:.1f} ms")

    def test_open_goal_not_mistaken_for_walled(self):
        # The quick "walled in?" check must not reject a goal that is only partly surrounded.
        m = ObstacleMap()
        add_pen(m, 5.0, 5.0)
        add_wall_line(m, 0.0, -m.half, m.half, gap=(0.0, 0.8))
        self.assertIsNotNone(m.astar((-5.0, 5.0), (8.0, -8.0), ALT))

    def test_pen_only_blocks_at_its_height(self):
        # A pen whose top is known (3 m): well above it the same goal is reachable again.
        m = ObstacleMap()
        add_pen(m, 5.0, 5.0, top=3.0)
        self.assertIsNotNone(m.astar((-10.0, -10.0), (5.0, 5.0), oa.MAX_ALT))

    def test_pen_of_unknown_height_blocks_every_altitude(self):
        # PBI-12: hits on something whose top is unknown never leave open space above them.
        m = ObstacleMap()
        add_pen(m, 5.0, 5.0)
        self.assertIsNone(m.astar((-10.0, -10.0), (5.0, 5.0), oa.MAX_ALT))

    def test_same_map_same_route(self):                                    # AC-04
        a = self.gap_map().astar((-5.0, 5.0), (5.0, -5.0), ALT)
        b = self.gap_map().astar((-5.0, 5.0), (5.0, -5.0), ALT)
        self.assertEqual(a, b)


@unittest.skipUnless(HAVE_SIM, "PyBullet not installed")
class TestReplan(unittest.TestCase):
    """Navigator.replan(): timing and named unreachable waypoints."""

    def nav(self):
        # Body ids are dummies: replan() only uses the obstacle map (gui=False skips drawing).
        n = Navigator(0, 1, 2, env_sim.FLIGHT_PATH, gui=False, log=lambda msg: None)
        n.pos = [env_sim.FLIGHT_PATH[0][0], env_sim.FLIGHT_PATH[0][1], ALT]
        return n

    def test_replan_is_timed(self):                                        # AC-03
        n = self.nav()
        self.assertTrue(n.replan())
        self.assertTrue(n.replan())
        self.assertEqual(len(n.replan_ms), 2)
        for ms in n.replan_ms:
            self.assertGreaterEqual(ms, 0.0)
            self.assertLess(ms, REPLAN_LIMIT_MS)

    def test_walled_waypoint_skipped_and_named(self):                      # AC-02
        n = self.nav()
        wx, wy = n.wps[1]
        add_pen(n.map, wx, wy)
        self.assertTrue(n.replan(), "should re-route to the next waypoint")
        self.assertEqual(n.skipped, [(n.t, 1)])
        unreach = [msg for _, kind, msg in n.events if kind == "UNREACH"]
        self.assertEqual(len(unreach), 1)
        self.assertIn(f"waypoint 1 at ({wx:.1f}, {wy:.1f})", unreach[0])
        self.assertIn("heading to waypoint 2", unreach[0])
        # The new route ends its detour at waypoint 2, and waypoint 1 is gone from it.
        tags = [r[2] for r in n.route if r[2] is not None]
        self.assertEqual(tags[0], 2)
        self.assertNotIn(1, tags)
        self.assertLess(max(n.replan_ms), REPLAN_LIMIT_MS)


if __name__ == "__main__":
    unittest.main()
