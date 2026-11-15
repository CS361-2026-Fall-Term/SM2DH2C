"""
PBI-15 tests: obstacle detection and avoidance for obstacles generated from the user's
geographical input (map data).

  AC-01  On the first sighting of a map obstacle the system names it (type, name, height) and puts
         its WHOLE outline in the obstacle map, so the route goes around the whole object.
  AC-02  An obstacle inside the unsafe distance (0.35 m) is detected (UNSAFE); the drone backs
         away and re-plans its route around it.
  AC-03  Malformed input prints an error message and does not crash; a single bad map object is
         skipped and named, and the rest of the map is still placed.

Obstacles are at predefined positions: tests/fixtures/geo_test_area.json (see
tests/fixtures/make_geo_test_area.py), a 26 m map at 1 sim metre = 1 real metre.

Run from the Sprint folder (needs PyBullet; no window opens):
    python -m unittest tests.test_geo_obstacles -v
    python -m unittest discover -s tests -v          (all tests)
"""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import pybullet as p                                    # noqa: E402

import env_sim                                          # noqa: E402
import geo_obstacles                                    # noqa: E402
import object_avoidance as oa                           # noqa: E402
import osm_world                                        # noqa: E402
import sim_config                                       # noqa: E402
from geo_obstacles import GeoDataError, Obstacle       # noqa: E402

FIXTURE = os.path.join(CODE_DIR, "tests", "fixtures", "geo_test_area.json")
LKP = (44.6488, -123.3123)
MAP_SIZE = 26.0
ON_ROUTE = ('building "Test Hall"', 'building "L Annex"', 'building "Garage"')
# Predefined positions in the fixture (sim metres), see make_geo_test_area.py
BUILDING_CENTRES = {'building "Test Hall"': (-5.0, 2.5), 'building "L Annex"': (2.7, 0.0),
                    'building "Garage"': (10.0, 1.25), 'building "Library"': (-9.5, -9.5)}
TREE_XY = [(-7.0, 10.2), (7.5, 5.0), (7.5, -5.2), (0.0, 9.0), (-2.0, -9.0)]
# The real engine is needed for exact mesh shapes; a geometric stand-in used for development only
# approximates a building mesh by its bounding box.
STAND_IN = "stand-in" in (p.__doc__ or "")

# env_sim's world lists are changed IN PLACE by use_osm(); keep the originals so every test
# starts from the same hand-placed layout.
_ORIGINAL = {name: copy.deepcopy(getattr(env_sim, name)) for name in
             ("DRONE_START_POS", "TARGET_POSITION", "OBSTACLE_POSITIONS", "FLIGHT_PATH",
              "TREE_KINDS", "WALLS")}


def reset_world():
    if p.isConnected():
        p.disconnect()
    for name, value in _ORIGINAL.items():
        getattr(env_sim, name)[:] = copy.deepcopy(value)
    env_sim.OSM_SCENE = None
    env_sim.OBSTACLE_INFO.clear()
    env_sim.GEO_PROBLEMS.clear()
    sim_config.set_map_size(MAP_SIZE)
    env_sim.scale_layout(sim_config.layout_scale())


def fixture_world():
    """Build the fixture world (simple look, no window). Returns (floor, drone, obstacle ids)."""
    reset_world()
    env_sim.DETAILED_VISUALS = False
    env_sim.use_osm(*LKP, area_m=MAP_SIZE, osm_file=FIXTURE)
    return env_sim.build_world(gui=False)


def by_name(table):
    return {ob.name: uid for uid, ob in table.items()}


def quiet(*_a, **_k):
    pass


def point_in_poly(x, y, poly):
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def fly(nav, drone_id, max_t=240.0, on_step=None):
    """Run the Navigator until it finishes (or max_t). Returns the drone positions (every step)."""
    quat = p.getQuaternionFromEuler([0, 0, 0])
    track = []
    while nav.t < max_t and nav.state not in ("DONE", "RTL_DONE", "ABORT"):
        nav.update(oa.DT)
        p.resetBasePositionAndOrientation(drone_id, nav.pos, quat)
        track.append(tuple(nav.pos))
        if on_step:
            on_step(nav)
    return track


def write_tmp(test, name, content, binary=False):
    path = os.path.join(test.tmp.name, name)
    with open(path, "wb" if binary else "w", **({} if binary else {"encoding": "utf-8"})) as f:
        f.write(content)
    return path


# ====================================================================== AC-01
class TestAC01IdentifyAndMapOutline(unittest.TestCase):
    """Map obstacles are known to the sensor: named on first sight, whole outline mapped."""

    def setUp(self):
        self.floor, self.drone, self.ids = fixture_world()
        self.table = dict(env_sim.OBSTACLE_INFO)
        self.names = by_name(self.table)

    def tearDown(self):
        reset_world()

    def nav(self):
        return oa.Navigator(self.drone, self.floor, -1, env_sim.FLIGHT_PATH, False, quiet,
                            obstacle_info=env_sim.OBSTACLE_INFO)

    def test_fixture_objects_extracted_at_predefined_positions(self):
        # EXPECT: the 4 buildings and 5 trees of the user's map, each within 5 cm of its position.
        kinds = [ob.kind for ob in self.table.values()]
        self.assertEqual((kinds.count("building"), kinds.count("tree")), (4, 5))
        for name, (cx, cy) in BUILDING_CENTRES.items():
            ob = self.table[self.names[name]]
            xs, ys = [q[0] for q in ob.footprint], [q[1] for q in ob.footprint]
            self.assertLess(math.dist(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2), (cx, cy)), 0.05, name)
        trees = [(ob.x, ob.y) for ob in self.table.values() if ob.kind == "tree"]
        for xy in TREE_XY:
            self.assertTrue(any(math.dist(xy, t) < 0.05 for t in trees), xy)

    def test_every_obstacle_is_one_body_in_the_table(self):
        # EXPECT: one PyBullet body per obstacle, and the table says what each body is.
        self.assertEqual(len(self.ids), len(self.table))
        self.assertEqual(set(self.ids), set(self.table))
        for uid, ob in self.table.items():
            self.assertIn(ob.kind, ("tree", "building", "wall"))
            self.assertTrue(ob.name)
            self.assertGreater(ob.height, 0)

    def test_bodies_are_at_the_real_positions(self):
        # EXPECT: tree bases at their map positions; building boxes match their outlines.
        for uid, ob in self.table.items():
            if ob.kind == "tree":
                pos, _ = p.getBasePositionAndOrientation(uid)
                self.assertLess(math.dist(pos, (ob.x, ob.y, ob.z)), 0.05, ob.name)
            else:
                lo, hi = p.getAABB(uid)
                xs, ys = [q[0] for q in ob.footprint], [q[1] for q in ob.footprint]
                for got, want in zip((lo[0], lo[1], hi[0], hi[1], hi[2]),
                                     (min(xs), min(ys), max(xs), max(ys), ob.height)):
                    self.assertAlmostEqual(got, want, delta=0.06, msg=ob.name)

    def test_no_obstacle_files_are_written(self):
        # EXPECT: obstacles are built in memory (no URDF / OBJ files, no generated_obstacles/).
        reset_world()
        before = set(os.listdir(CODE_DIR))
        fixture_world()
        self.assertEqual(set(os.listdir(CODE_DIR)) - before, set())
        self.assertFalse(os.path.exists(os.path.join(CODE_DIR, "generated_obstacles", "manifest.json")))

    @unittest.skipIf(STAND_IN, "needs the real PyBullet mesh")
    def test_l_shaped_building_keeps_its_inner_corner(self):
        # EXPECT: a ray through the notch of "L Annex" (x 2.8..3.6, y > -0.2) hits nothing.
        uid = self.names['building "L Annex"']
        hit = p.rayTest([3.2, 3.0, 2.0], [3.2, 0.0, 2.0])[0]
        self.assertNotEqual(hit[0], uid)
        hit = p.rayTest([3.2, 3.0, 2.0], [3.2, -1.0, 2.0])[0]       # ...but the wall below it is hit
        self.assertEqual(hit[0], uid)

    def test_first_sighting_is_named_once(self):
        # EXPECT: DETECT with the building's name and height on the first hit only.
        nav, uid = self.nav(), self.names['building "Test Hall"']
        nav._identify(uid, (-5.0, 3.2, 2.5), 4.0)
        nav._identify(uid, (-5.0, 3.2, 2.5), 3.0)
        detects = [m for _t, k, m in nav.events if k == "DETECT"]
        self.assertEqual(len(detects), 1)
        self.assertIn('building "Test Hall"', detects[0])
        self.assertIn("8.0 m tall", detects[0])
        self.assertEqual(nav.detected[uid][1], 3.0)                # closest sensed range kept

    def test_whole_outline_is_mapped_from_the_first_hit(self):
        # EXPECT: one hit on the north wall blocks the far (south) wall and the inside too.
        nav, uid = self.nav(), self.names['building "Test Hall"']
        alt = 2.5
        far_side = (-5.0, 1.9)                                     # south wall, never sensed
        self.assertFalse(nav.map.blocked(*far_side, alt))
        nav._identify(uid, (-5.0, 3.2, alt), 4.0)
        self.assertTrue(nav.map.blocked(*far_side, alt))
        self.assertTrue(nav.map.blocked(-5.0, 2.5, alt))           # inside
        self.assertTrue(nav.map.blocked(-6.4, 1.9, alt))           # far corner

    def test_outline_margin_is_measured_from_the_walls(self):
        # EXPECT: every point closer than SAFETY_RADIUS to a wall is blocked; 1 m out is free.
        nav, uid = self.nav(), self.names['building "Test Hall"']
        nav._identify(uid, (-5.0, 3.2, 2.5), 4.0)
        r = oa.SAFETY_RADIUS
        for x, y in ((-5.0, 3.2 + r - 0.02), (-5.0, 1.8 - r + 0.02), (-3.5 + r - 0.02, 2.5),
                     (-6.5 - r + 0.02, 2.5), (-3.5 + (r - 0.02) * 0.7, 3.2 + (r - 0.02) * 0.7)):
            self.assertTrue(nav.map.blocked(x, y, 2.5), (x, y))
        for x, y in ((-5.0, 4.2), (-5.0, 0.8), (-2.5, 2.5)):
            self.assertFalse(nav.map.blocked(x, y, 2.5), (x, y))

    def test_outline_only_blocks_up_to_the_building_height(self):
        # EXPECT: "Garage" is 4 m tall: blocked at cruise height, free well above its roof.
        nav, uid = self.nav(), self.names['building "Garage"']
        nav._identify(uid, (10.0, 2.0, 2.5), 3.0)
        self.assertTrue(nav.map.blocked(10.0, 1.25, 2.5))
        self.assertFalse(nav.map.blocked(10.0, 1.25, 4.0 + oa.VERT_CLEAR + 0.5))

    def test_tree_is_named_without_an_outline(self):
        # EXPECT: a tree is named; it has no outline, so nothing beyond the ray hits is mapped.
        nav = self.nav()
        uid = next(u for u, ob in self.table.items() if ob.kind == "tree")
        version = nav.map.version
        nav._identify(uid, (0, 0, 2.5), 3.0)
        self.assertIn(self.table[uid].name, nav.events[-1][2])
        self.assertEqual(nav.map.version, version)

    def test_obstacle_not_from_the_map_is_still_detected(self):
        # EXPECT: something not in the table (appeared mid-mission) is still reported.
        nav = self.nav()
        nav._identify(99999, (1.0, 2.0, 2.5), 2.0)
        self.assertEqual(nav.events[-1][1], "DETECT")
        self.assertIn("unmapped obstacle", nav.events[-1][2])

    def test_full_flight_goes_around_the_buildings(self):
        # EXPECT: the 3 buildings on the route are named, the drone re-plans around them, never
        # enters a building outline, and reaches every waypoint.
        nav = self.nav()
        track = fly(nav, self.drone)
        self.assertEqual(nav.state, "DONE")
        detected = {self.table[u].name for u in nav.detected if u in self.table}
        for name in ON_ROUTE:
            self.assertIn(name, detected)
        self.assertGreater(nav.replans, 0)
        reached = {m for _t, k, m in nav.events if k == "WAYPT"}
        self.assertEqual(len(reached), len(env_sim.FLIGHT_PATH) - 1)
        buildings = [ob for ob in self.table.values() if ob.kind == "building"]
        for x, y, z in track[::10]:
            for ob in buildings:
                if z < ob.height:
                    self.assertFalse(point_in_poly(x, y, ob.footprint), f"inside {ob.name} at ({x:.2f}, {y:.2f})")


# ====================================================================== AC-02
class TestAC02UnsafeDistance(unittest.TestCase):
    """An obstacle inside 0.35 m is detected; the drone backs away and re-plans."""

    def tearDown(self):
        reset_world()

    def nav_in_fixture(self):
        floor, drone, _ids = fixture_world()
        return oa.Navigator(drone, floor, -1, env_sim.FLIGHT_PATH, False, quiet,
                            obstacle_info=env_sim.OBSTACLE_INFO)

    def test_unsafe_distance_default(self):
        self.assertEqual(oa.UNSAFE_DIST, 0.35)

    def test_close_surface_logs_unsafe_once_and_asks_for_a_replan(self):
        # EXPECT: one UNSAFE line (with the obstacle's name) per approach, and a re-plan request.
        nav = self.nav_in_fixture()
        uid = by_name(env_sim.OBSTACLE_INFO)['building "Test Hall"']
        nav._check_unsafe(0.20, uid, [(-5.0, 3.2, 2.5)])
        nav._check_unsafe(0.18, uid, [(-5.0, 3.2, 2.5)])
        unsafe = [m for _t, k, m in nav.events if k == "UNSAFE"]
        self.assertEqual(len(unsafe), 1)
        self.assertIn('building "Test Hall"', unsafe[0])
        self.assertEqual(nav.unsafe_alerts, 1)
        self.assertTrue(nav._unsafe_replan)

    def test_close_surface_widens_the_keep_out(self):
        # EXPECT: the map is blocked further around a surface that came too close.
        nav = self.nav_in_fixture()
        hit = (0.0, 6.0, 2.5)
        probe = (0.0, 6.0 + oa.SAFETY_RADIUS + oa.KEEPOUT_EXTRA + 0.1)   # past a normal hit's margin
        nav.map.add_hit(hit)
        self.assertFalse(nav.map.blocked(*probe, 2.5))
        nav._check_unsafe(0.2, None, [hit])
        self.assertTrue(nav.map.blocked(*probe, 2.5))

    def test_alert_resets_once_clear(self):
        # EXPECT: after moving clear (> 0.35 + 0.05 m), a new approach gives a new alert.
        nav = self.nav_in_fixture()
        nav._check_unsafe(0.2, None, [])
        nav._check_unsafe(0.37, None, [])           # still inside the hysteresis band
        nav._check_unsafe(0.2, None, [])
        self.assertEqual(nav.unsafe_alerts, 1)
        nav._check_unsafe(0.5, None, [])
        nav._check_unsafe(0.2, None, [])
        self.assertEqual(nav.unsafe_alerts, 2)

    def test_drone_backs_away_and_replans(self):
        # A building is placed 0.25 m from the drone's body beside its first leg.
        # EXPECT: UNSAFE on the first scan, a re-plan, back outside 0.35 m within 2 s, no contact,
        # and the route is still completed.
        reset_world()
        p.connect(p.DIRECT)
        import pybullet_data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        floor = p.loadURDF("plane.urdf")
        start = env_sim.FLIGHT_PATH[0]
        drone = p.loadURDF(os.path.join(CODE_DIR, "custom_drone.urdf"), start,
                           p.getQuaternionFromEuler([0, 0, 0]), globalScaling=oa.DRONE_SCALE)
        y0 = start[1] + oa.DRONE_RADIUS + 0.25                   # near wall 0.25 m from the body
        box = [(start[0] - 1.0, y0), (start[0] + 2.0, y0), (start[0] + 2.0, y0 + 1.0),
               (start[0] - 1.0, y0 + 1.0)]
        kiosk = Obstacle("building", 'building "Pop-up Kiosk"', 0.0, 0.0, 0.0, height=6.0,
                         radius=2.0, cx=start[0] + 0.5, cy=y0 + 0.5, footprint=box, source="map")
        ids, table, problems = geo_obstacles.spawn(p, [kiosk])
        self.assertEqual(problems, [])
        nav = oa.Navigator(drone, floor, -1, env_sim.FLIGHT_PATH, False, quiet, obstacle_info=table)
        nav.pos = list(start)
        p.resetBasePositionAndOrientation(drone, start, [0, 0, 0, 1])

        clear_at, contacts, track = [None], [0], []

        def ground_truth(n):
            pts = p.getClosestPoints(drone, ids[0], 1.0)
            d = min((c[8] for c in pts), default=1.0)
            contacts[0] += d <= 0.0
            if clear_at[0] is None and d >= oa.UNSAFE_DIST:
                clear_at[0] = n.t

        fly(nav, drone, max_t=240.0, on_step=ground_truth)
        unsafe = [(t, m) for t, k, m in nav.events if k == "UNSAFE"]
        self.assertTrue(unsafe, "no UNSAFE alert")
        self.assertLessEqual(unsafe[0][0], oa.DT * 10)            # on the first scans
        self.assertIn("Pop-up Kiosk", unsafe[0][1])
        self.assertGreaterEqual(nav.replans, 1)
        self.assertIsNotNone(clear_at[0])
        self.assertLessEqual(clear_at[0], 2.0)
        self.assertEqual(contacts[0], 0)
        self.assertEqual(nav.state, "DONE")


# ====================================================================== AC-03
class TestAC03MalformedInput(unittest.TestCase):
    """Bad input gives a clear error message, never a crash."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()
        reset_world()

    def load(self, path):
        return geo_obstacles.validate(geo_obstacles.load_geo_file(path, []), [])

    def test_unusable_files_raise_one_clear_error(self):
        # EXPECT: GeoDataError (never another exception) with a message for each unusable file.
        cases = {
            "empty.osm": "",
            "cut_off.osm": '<osm><node id="1" lat="44.6"',
            "webpage.osm": "<html><body>Not found</body></html>",
            "broken.json": '{"elements": [',
            "list.json": "[1, 2, 3]",
            "no_elements.json": '{"version": 0.6}',
            "elements_not_list.json": '{"elements": 5}',
            "nothing_usable.json": '{"elements": [{"type": "node", "id": 1}]}',
        }
        for name, text in cases.items():
            with self.subTest(file=name):
                with self.assertRaises(GeoDataError) as ctx:
                    self.load(write_tmp(self, name, text))
                self.assertGreater(len(str(ctx.exception)), 10)
        with self.subTest(file="binary.osm"):
            with self.assertRaises(GeoDataError):
                self.load(write_tmp(self, "binary.osm", bytes(range(256)), binary=True))

    def test_missing_file(self):
        with self.assertRaises(GeoDataError) as ctx:
            self.load(os.path.join(self.tmp.name, "nope.osm"))
        self.assertIn("not found", str(ctx.exception))

    def test_bad_position_or_area_is_rejected(self):
        # EXPECT: impossible latitude / longitude / area -> GeoDataError naming the bad value.
        for lat, lon, area in ((144.0, -123.3, 100), (44.6, -200.0, 100), (44.6, -123.3, -5),
                               (float("nan"), -123.3, 100), ("north", -123.3, 100)):
            with self.subTest(lat=lat, lon=lon, area=area):
                with self.assertRaises(GeoDataError):
                    geo_obstacles.check_position(lat, lon, area)
        geo_obstacles.check_position(44.6488, -123.3123, 200)          # a good one passes

    def test_bad_objects_are_skipped_and_named_the_rest_is_kept(self):
        # EXPECT: each bad object gives one problem line naming it; the good building survives.
        with open(FIXTURE, encoding="utf-8") as f:
            good = json.load(f)["elements"][0]
        g = good["geometry"]
        bad = [
            {"type": "way", "id": 7, "tags": {"building": "yes", "name": "Open Shed"}, "geometry": g[:3]},
            {"type": "node", "id": 8, "tags": {"natural": "tree"}},
            {"type": "way", "id": 9, "tags": {"building": "yes"}, "geometry": g[:2]},
            5,
            {"type": "way", "id": 11, "tags": "oops", "geometry": g},
            {"type": "spaceship", "id": 12},
        ]
        problems = []
        data = geo_obstacles.validate({"elements": [good] + bad}, problems)
        self.assertEqual(len(data["elements"]), 1)
        self.assertEqual(len(problems), len(bad))
        self.assertTrue(any("Open Shed" in line for line in problems))
        self.assertTrue(all("skipped" in line for line in problems))
        scene = osm_world.parse_scene(data, *LKP, MAP_SIZE)
        self.assertEqual([info.get("name") for _poly, _h, info in scene.buildings], ["Test Hall"])

    def test_osm_node_without_coordinates_skips_its_building(self):
        # EXPECT: .osm XML with a node missing 'lat' -> that node and its way skipped, rest kept.
        text = ('<osm version="0.6">'
                '<node id="1" lat="44.6488" lon="-123.3123"/><node id="2" lat="44.64883" lon="-123.3123"/>'
                '<node id="3" lat="44.64883" lon="-123.31226"/><node id="4" lon="-123.3122"/>'
                '<way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/>'
                '<tag k="building" v="yes"/><tag k="name" v="Good"/></way>'
                '<way id="11"><nd ref="1"/><nd ref="4"/><nd ref="3"/><nd ref="1"/>'
                '<tag k="building" v="yes"/><tag k="name" v="Bad"/></way></osm>')
        problems = []
        data = geo_obstacles.validate(
            geo_obstacles.load_geo_file(write_tmp(self, "mixed.osm", text), problems), problems)
        names = [e.get("tags", {}).get("name") for e in data["elements"]]
        self.assertEqual(names, ["Good"])
        self.assertTrue(any("node 4" in line for line in problems))

    def test_bad_layout_entries_are_skipped(self):
        # EXPECT: a bad tree position / wall size is skipped with a message; the rest is kept.
        obstacles, problems = geo_obstacles.obstacles_from_world(
            tree_positions=[(1.0, 2.0, 0.5), ("x", 2.0, 0.5), (float("inf"), 0, 0.5)],
            walls=[(0, 0, 1, 1, 3), (0, 0, -1, 1, 3)])
        self.assertEqual([o.kind for o in obstacles], ["tree", "wall"])
        self.assertEqual(len(problems), 3)

    def test_unbuildable_obstacle_is_skipped_others_placed(self):
        # EXPECT: spawn() reports an obstacle it cannot build and still places the others.
        reset_world()
        p.connect(p.DIRECT)
        flat = Obstacle("building", 'building "Flat"', 0, 0, 0, height=5.0, radius=1.0,
                        footprint=[(0, 0), (1, 1), (2, 2)], source="map")
        odd = Obstacle("statue", "statue 1", 0, 0, 0, height=2.0, radius=0.5)
        tree, _ = geo_obstacles.obstacles_from_world(tree_positions=[(3.0, 3.0, 0.5)])
        ids, table, problems = geo_obstacles.spawn(p, [flat, odd] + tree)
        self.assertEqual(len(ids), 1)
        self.assertEqual(table[ids[0]].kind, "tree")
        self.assertEqual(len(problems), 2)
        self.assertTrue(any('"Flat"' in line for line in problems))

    def run_main(self, osm_file, *extra):
        cmd = [sys.executable, os.path.join(CODE_DIR, "main.py"), "--headless", "--no-ui",
               "--osm-file", osm_file, "--lat", str(LKP[0]), "--lon", str(LKP[1]),
               "--area", str(MAP_SIZE), "--map-size", str(MAP_SIZE), "--max-time", "20",
               "--state-file", os.path.join(self.tmp.name, "mission_state.json"), *extra]
        return subprocess.run(cmd, cwd=self.tmp.name, capture_output=True, text=True, timeout=600)

    def test_main_stops_with_an_error_message_on_a_bad_file(self):
        # EXPECT: main.py prints "ERROR: ..." and exits without a traceback.
        for name, text in (("cut_off.osm", "<osm><node"), ("empty.json", "")):
            with self.subTest(file=name):
                out = self.run_main(write_tmp(self, name, text))
                text_out = out.stdout + out.stderr
                self.assertIn("ERROR:", text_out)
                self.assertNotIn("Traceback", text_out)

    def test_main_reports_bad_objects_and_still_runs_the_mission(self):
        # EXPECT: the bad object is named, everything else is placed, and the mission runs.
        with open(FIXTURE, encoding="utf-8") as f:
            data = json.load(f)
        data["elements"].append({"type": "way", "id": 900, "tags": {"building": "yes", "name": "Open Shed"},
                                 "geometry": data["elements"][0]["geometry"][:2]})
        out = self.run_main(write_tmp(self, "with_bad.json", json.dumps(data)))
        text_out = out.stdout + out.stderr
        self.assertNotIn("Traceback", text_out)
        self.assertIn("Open Shed", text_out)
        self.assertIn("ERROR:", text_out)
        self.assertIn("Mission:", text_out)
        self.assertIn("generated obstacles detected", text_out)   # end-of-mission obstacle report


if __name__ == "__main__":
    unittest.main()
