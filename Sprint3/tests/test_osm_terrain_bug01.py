"""
Contributors: Mylan Bui
File Description: BUG-01 tests - a live OSM download builds the correct terrain.
No network or PyBullet needed: the Overpass download is replaced with a fake.
Run from the Sprint folder:  python -m unittest discover -s tests -v

Covers
  AC-01  areas that CONTAIN the map (forest/lake around the LKP) are requested and drawn,
         and a forest that covers the whole map gets trees scattered in it
  AC-02  every supported terrain type is requested by the query and ends up in the scene
  AC-03  tall buildings: collision height clamped, drawn (visual) height full scale
  AC-04  short buildings unchanged; old v3 cache files are not reused
"""
import math
import os
import shutil
import sys
import tempfile
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)
import osm_world as ow  # noqa: E402
import sim_config  # noqa: E402

LAT0, LON0 = 44.6, -123.3                  # any LKP works; the tests only use offsets from it


def latlon(x, y):
    """Sim metres (scale 1) -> {'lat', 'lon'}, the inverse of osm_world.make_projector."""
    return {"lat": LAT0 + y / 111_132.0,
            "lon": LON0 + x / (111_320.0 * math.cos(math.radians(LAT0)))}


def square_way(cx, cy, half, tags):
    """A closed OSM way: a square of side 2*half centred on (cx, cy) in sim metres."""
    corners = [(cx - half, cy - half), (cx + half, cy - half), (cx + half, cy + half),
               (cx - half, cy + half), (cx - half, cy - half)]
    return {"type": "way", "tags": tags, "geometry": [latlon(x, y) for x, y in corners]}


class MapSizeMixin:
    def setUp(self):
        self._half = sim_config.MAP_HALF
        sim_config.MAP_HALF = 13.0         # the standard 26 m map, scale 1 when area_m = 26

    def tearDown(self):
        sim_config.MAP_HALF = self._half


class TestQuery(MapSizeMixin, unittest.TestCase):
    """fetch_overpass(): what gets asked for (the download itself is faked)."""

    def setUp(self):
        super().setUp()
        self.queries = []
        self._download, self._cache = ow._download, ow.CACHE_DIR
        ow.CACHE_DIR = tempfile.mkdtemp()
        ow._download = lambda query, timeout: self.queries.append(query) or {"elements": []}

    def tearDown(self):
        ow._download = self._download
        shutil.rmtree(ow.CACHE_DIR, ignore_errors=True)
        ow.CACHE_DIR = self._cache
        super().tearDown()

    def query(self):
        ow.fetch_overpass(LAT0, LON0, area_m=200)
        return self.queries[-1]

    def test_areas_containing_the_map_are_requested(self):                 # AC-01
        q = self.query()
        self.assertEqual(q.count("is_in("), 5, "LKP + 4 corners")
        for k in range(5):
            self.assertIn(f"way(pivot.a{k})", q)
            self.assertIn(f"relation(pivot.a{k})", q)

    def test_every_terrain_type_is_requested(self):                        # AC-02
        q = self.query()
        for value in ("wood", "water", "scrub", "grassland", "sand", "beach", "bare_rock", "scree",
                      "forest", "meadow", "farmland", "orchard", "construction", "reservoir",
                      "basin", "park", "pitch", "track", "playground"):
            self.assertIn(value, q, f"{value} missing from the query")
        for line in ('way["place"="square"]', 'relation["waterway"="riverbank"]',
                     'way["amenity"~"^(parking|marketplace)$"]', 'way["building"]',
                     'way["building:part"]', 'node["natural"="tree"]', 'way["highway"]',
                     'way["waterway"]', 'way["barrier"'):
            self.assertIn(line, q)

    def test_cache_is_v4_and_reused(self):                                 # AC-04
        self.query()
        files = os.listdir(ow.CACHE_DIR)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith("_v4.json"), files[0])
        ow.fetch_overpass(LAT0, LON0, area_m=200)                          # second call: cached
        self.assertEqual(len(self.queries), 1)

    def test_old_v3_cache_not_reused(self):                                # AC-04
        radius = math.ceil(200 / 2 * math.sqrt(2)) + 5
        old = os.path.join(ow.CACHE_DIR, f"{LAT0:.6f}_{LON0:.6f}_{radius}_v3.json")
        with open(old, "w") as f:
            f.write('{"elements": []}')
        self.query()
        self.assertEqual(len(self.queries), 1, "should download again, not load the v3 file")


class TestTerrainParsing(MapSizeMixin, unittest.TestCase):
    """parse_scene(): each supported terrain type ends up in the scene."""

    def parse(self, elements):
        return ow.parse_scene({"elements": elements}, LAT0, LON0, area_m=26)

    def test_forest_covering_the_whole_map(self):                          # AC-01
        scene = self.parse([square_way(0, 0, 40, {"landuse": "forest"})])  # much bigger than the map
        cats = [c for _, c in scene.surfaces]
        self.assertIn("forest", cats)
        self.assertGreater(len(scene.trees), 10, "forest should get scattered trees")

    def test_lake_covering_the_whole_map(self):                            # AC-01
        scene = self.parse([square_way(0, 0, 40, {"natural": "water"})])
        self.assertEqual(len(scene.water), 1)

    def test_every_surface_type(self):                                     # AC-02
        cases = [({"landuse": "forest"}, "forest"), ({"natural": "wood"}, "forest"),
                 ({"natural": "scrub"}, "scrub"), ({"leisure": "park"}, "grass"),
                 ({"landuse": "meadow"}, "grass"), ({"landuse": "farmland"}, "farm"),
                 ({"landuse": "construction"}, "dirt"), ({"natural": "beach"}, "sand"),
                 ({"natural": "bare_rock"}, "rock"), ({"leisure": "pitch"}, "pitch"),
                 ({"leisure": "track"}, "track"), ({"amenity": "parking"}, "parking"),
                 ({"place": "square"}, "plaza")]
        for i, (tags, want) in enumerate(cases):
            with self.subTest(tags=tags):
                x, y = -9 + 4.5 * (i % 5), -9 + 4.5 * (i // 5)
                scene = self.parse([square_way(x, y, 1.5, tags)])
                self.assertIn(want, [c for _, c in scene.surfaces])

    def test_water_types(self):                                            # AC-02
        for tags in ({"natural": "water"}, {"landuse": "reservoir"}, {"waterway": "riverbank"}):
            with self.subTest(tags=tags):
                scene = self.parse([square_way(0, 0, 3, tags)])
                self.assertEqual(len(scene.water), 1)

    def test_riverbank_is_an_area_not_a_line(self):
        scene = self.parse([square_way(0, 0, 3, {"waterway": "riverbank"})])
        self.assertEqual(len(scene.water), 1)
        self.assertEqual(scene.roads, [], "riverbank should not be drawn as a stream line")

    def test_stream_is_still_a_line(self):
        way = {"type": "way", "tags": {"waterway": "stream"},
               "geometry": [latlon(-5, 0), latlon(5, 0)]}
        scene = self.parse([way])
        self.assertTrue(scene.roads)
        self.assertEqual(scene.water, [])


class TestBuildingHeight(unittest.TestCase):
    """building_height() / building_info(): clamped collision, full-scale visual."""
    SCALE = 0.13                               # about --area 200 on the 26 m map

    def test_tall_building(self):                                          # AC-03
        h, info = ow.building_info({"building": "tower", "height": "330"}, self.SCALE)
        self.assertEqual(h, ow.MAX_BUILDING_HEIGHT, "collision must stay clamped")
        self.assertAlmostEqual(info["visual_height"], 330 * self.SCALE, places=6)

    def test_clamp_flag(self):
        tags = {"height": "330"}
        self.assertEqual(ow.building_height(tags, self.SCALE), ow.MAX_BUILDING_HEIGHT)
        self.assertAlmostEqual(ow.building_height(tags, self.SCALE, clamp=False), 330 * self.SCALE)

    def test_short_building_unchanged(self):                               # AC-04
        h, info = ow.building_info({"building": "yes", "height": "30"}, self.SCALE)
        self.assertAlmostEqual(h, 30 * self.SCALE, places=6)
        self.assertEqual(h, info["visual_height"])

    def test_minimum_height_still_applies(self):
        h, info = ow.building_info({"building": "shed", "height": "2"}, self.SCALE)
        self.assertEqual(h, ow.MIN_BUILDING_HEIGHT)
        self.assertEqual(info["visual_height"], ow.MIN_BUILDING_HEIGHT)


if __name__ == "__main__":
    unittest.main()
