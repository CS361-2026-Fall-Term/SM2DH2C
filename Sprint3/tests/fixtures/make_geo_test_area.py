"""
Writes geo_test_area.json: a small Overpass-style map used by tests/test_geo_obstacles.py and
geo_avoidance_runs.py. Obstacles are at PREDEFINED positions (sim metres, x east / y north) for a
26 m map showing 26 real metres around LKP (44.6488, -123.3123), i.e. 1 sim metre = 1 real metre.

On the default flight path (-10,10) -> (-5,10) -> (-5,5) -> (-5,0) -> (0,0) -> (5,0) -> (5,5)
-> (10,5) -> (10,-5) -> (5,-5) -> (6,-5):
  building "Test Hall"  8 m tall, across the leg (-5,5) -> (-5,0)
  building "L Annex"    7 m tall, L-shaped, across the leg (0,0) -> (5,0)
  building "Garage"     4 m tall, across the leg (10,5) -> (10,-5)
  trees at (-7.0, 10.2), (7.5, 5.0), (7.5, -5.2)
Away from the route: building "Library" (10 m) in the south-west corner, trees at (0, 9) and (-2, -9).

Run: python tests/fixtures/make_geo_test_area.py
"""
import json
import math
import os

LAT0, LON0 = 44.6488, -123.3123
M_PER_DEG_LAT = 111_132.0                                   # same projection as osm_world.py
M_PER_DEG_LON = 111_320.0 * math.cos(math.radians(LAT0))

BUILDINGS = [
    ("Test Hall", 8, [(-6.5, 1.8), (-3.5, 1.8), (-3.5, 3.2), (-6.5, 3.2)]),
    ("L Annex", 7, [(1.8, -1.5), (3.6, -1.5), (3.6, -0.2), (2.8, -0.2), (2.8, 1.5), (1.8, 1.5)]),
    ("Garage", 4, [(9.2, 0.5), (10.8, 0.5), (10.8, 2.0), (9.2, 2.0)]),
    ("Library", 10, [(-11.0, -11.0), (-8.0, -11.0), (-8.0, -8.0), (-11.0, -8.0)]),
]
TREES = [(-7.0, 10.2), (7.5, 5.0), (7.5, -5.2), (0.0, 9.0), (-2.0, -9.0)]


def latlon(x, y):
    return {"lat": round(LAT0 + y / M_PER_DEG_LAT, 9), "lon": round(LON0 + x / M_PER_DEG_LON, 9)}


def build():
    elements = []
    for i, (name, h, pts) in enumerate(BUILDINGS):
        ring = [latlon(x, y) for x, y in pts]
        elements.append({"type": "way", "id": 100 + i, "geometry": ring + [ring[0]],
                         "tags": {"building": "yes", "name": name, "height": str(h)}})
    for i, (x, y) in enumerate(TREES):
        elements.append({"type": "node", "id": 500 + i, **latlon(x, y),
                         "tags": {"natural": "tree", "leaf_type": "broadleaved"}})
    return {"version": 0.6, "generator": "make_geo_test_area.py", "elements": elements}


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "geo_test_area.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(build(), f, indent=1)
    print("wrote", out)
