"""Builds a small synthetic Overpass JSON (tests/sample_osm.json) so the parser can be tested offline."""
import json, math
LAT, LON = 45.5122, -122.6587
mlat, mlon = 111132.0, 111320.0 * math.cos(math.radians(LAT))
def ll(e, n): return {"lat": LAT + n / mlat, "lon": LON + e / mlon}
def ring(pts): return [ll(*q) for q in pts + [pts[0]]]
els = [
 {"type": "way", "tags": {"building": "yes", "building:levels": "3"},
  "geometry": ring([(-60, -10), (-35, -10), (-35, 0), (-50, 0), (-50, 15), (-60, 15)])},   # L-shape
 {"type": "way", "tags": {"building": "house"}, "geometry": ring([(20, 20), (35, 20), (35, 35), (20, 35)])},  # covers zone A
 {"type": "way", "tags": {"building": "yes", "height": "40"}, "geometry": ring([(50, -20), (90, -20), (90, 0), (50, 0)])},  # crosses map edge
 {"type": "node", "lat": ll(5, -10)["lat"], "lon": ll(5, -10)["lon"], "tags": {"natural": "tree"}},
 {"type": "node", "lat": ll(-10, 15)["lat"], "lon": ll(-10, 15)["lon"], "tags": {"natural": "tree"}},
 {"type": "node", "lat": ll(-35, -35)["lat"], "lon": ll(-35, -35)["lon"], "tags": {"natural": "tree"}},  # inside building -> dropped
 {"type": "way", "tags": {"natural": "wood"}, "geometry": ring([(10, -55), (50, -55), (50, -25), (10, -25)])},
 {"type": "way", "tags": {"highway": "residential"}, "geometry": [ll(-70, 0), ll(70, 0)]},
 {"type": "way", "tags": {"highway": "footway"}, "geometry": [ll(0, -70), ll(0, 70)]},
 {"type": "way", "tags": {"natural": "water"}, "geometry": ring([(-50, 20), (-25, 20), (-25, 40), (-50, 40)])},
]
json.dump({"elements": els}, open("tests/sample_osm.json", "w"), indent=1)
print("wrote tests/sample_osm.json")
