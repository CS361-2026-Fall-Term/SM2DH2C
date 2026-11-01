"""
File Description: Turns OpenStreetMap (OSM) data around a last known position (LKP) into
obstacles for the existing PyBullet world, WITHOUT changing the size of the simulation map.

The simulation map is [-MAP_HALF, MAP_HALF] in x and y, where MAP_HALF comes from sim_config
(set once by main.py; 26 m x 26 m by default). The obstacle map (object_avoidance.py), the search
zones (operator_guidance.py) and the probability map all use the same value. The LKP is placed at
the centre (0, 0).

  area_m  = how many REAL metres of OSM data to show across the map.
  scale   = (2 * MAP_HALF) / area_m   (sim metres per real metre)

  map 26 m, area_m = 26  -> scale 1.0  : true size, but only a small patch around the LKP
  map 26 m, area_m = 200 -> scale 0.13 : a whole block squeezed into the same 26 m map
  map 60 m, area_m = 60  -> scale 1.0  : true size, bigger patch

What gets converted (ways AND multipolygon relations):
  buildings (building=*)                 -> extruded, solid footprint meshes (drone senses/avoids them),
                                            textured walls by building type/material/colour + roof colour
  single trees (natural=tree)            -> trunk + canopy trees (conifer or deciduous from leaf_type/genus)
  woods / forests (natural=wood, ...)    -> trees scattered inside the polygon + dark forest floor
  roads / paths / rails (highway=*, ...) -> asphalt, sidewalk, dirt-path, rail strips; yellow centre
                                            dashes on major roads (visual only, no collision)
  water (natural=water, waterway=*)      -> blue surfaces and streams (visual only, no collision)
  ground (grass, park, parking, pitch,   -> coloured ground areas (visual only, no collision)
          sand, farmland, ...)
Only collision shapes affect sensing/navigation, and those are the same as the simple look.

Data sources (pick one):
  1. Live: lat/lon -> Overpass API (cached to osm_cache/ so repeat runs work offline)
  2. File: an Overpass JSON file, or a .osm XML file from openstreetmap.org -> Export
"""
import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import sim_config

# Map size lives in sim_config.MAP_HALF. Read it at call time (never as a default argument,
# which would freeze the value at import time).
DEFAULT_AREA_M = None           # None = same as the map size, i.e. true scale (1 sim m = 1 real m).

# Places the drone has to reach (start + zone centres from operator_guidance.py) are kept clear,
# otherwise the drone could spawn inside a building or be sent to a goal it can never reach.
KEEP_CLEAR_RADIUS = 1.6         # sim metres
TREE_CLEAR_RADIUS = 1.2         # extra gap between a tree trunk and a kept-clear point

BUILDING_LEVEL_HEIGHT = 3.0     # real metres per storey, when OSM gives building:levels
DEFAULT_BUILDING_HEIGHT = 6.0   # real metres, when OSM gives no height at all
MIN_BUILDING_HEIGHT = 3.5       # sim metres: always taller than cruise altitude (2.5 m), so it is a real obstacle
MAX_BUILDING_HEIGHT = 8.0       # sim metres: taller than MAX_ALT (5.5 m), so the drone routes around, not over

WOOD_TREE_SPACING = 3.0         # sim metres between scattered trees in a wood polygon
MAX_TREES = 40                  # cap for the 26 m map; each tree = 2 bodies + ray hits
MAX_TREES_LIMIT = 150           # cap never grows past this, however big the map is
MIN_TREE_SPACING = 1.0          # sim metres; closer trees are merged

ROAD_WIDTH_M = {"motorway": 12, "trunk": 10, "primary": 9, "secondary": 8, "tertiary": 7,
                "residential": 6, "service": 4, "unclassified": 5, "living_street": 5,
                "footway": 2, "steps": 2, "cycleway": 2, "path": 1.5, "track": 3}
DEFAULT_ROAD_WIDTH_M = 2.0      # footways, paths, tracks
MIN_ROAD_WIDTH = 0.3            # sim metres, so paths stay visible at small scales

# Public Overpass servers, tried in order. The main one often answers 502/504/429 when it is busy,
# so the download retries and then moves on to the next server.
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass.private.coffee/api/interpreter",
]
OVERPASS_URL = OVERPASS_URLS[0]          # kept for anything that still imports the old name
RETRY_STATUS = {429, 500, 502, 503, 504}  # "busy / try later" answers
TRIES_PER_SERVER = 2
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osm_cache")


# ------------------------------------------------------------------ loading OSM data
CACHE_VERSION = "v3"            # bump when the query changes, so old cached downloads are not reused


def fetch_overpass(lat, lon, area_m=None, timeout=90):
    """Download OSM features around (lat, lon) from Overpass, cached as JSON."""
    area_m = sim_config.map_size() if area_m is None else area_m
    radius = math.ceil(area_m / 2 * math.sqrt(2)) + 5      # circle that covers the whole square
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{lat:.6f}_{lon:.6f}_{radius}_{CACHE_VERSION}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    around = f"(around:{radius},{lat},{lon})"
    # ways AND multipolygon relations: big or complex buildings, parks and forests are often relations
    query = f"""[out:json][timeout:{timeout}];
(
  way["building"]{around};
  relation["building"]{around};
  way["building:part"]{around};
  relation["building:part"]{around};
  way["barrier"~"^(hedge|fence|wall|retaining_wall|city_wall|guard_rail)$"]{around};
  node["natural"="tree"]{around};
  way["natural"]{around};
  relation["natural"~"^(wood|water|scrub|grassland|heath|sand|beach|bare_rock)$"]{around};
  way["landuse"]{around};
  relation["landuse"~"^(forest|grass|meadow|recreation_ground|village_green|cemetery|farmland|construction)$"]{around};
  way["leisure"]{around};
  relation["leisure"~"^(park|garden|pitch|playground|track|golf_course)$"]{around};
  way["amenity"="parking"]{around};
  relation["amenity"="parking"]{around};
  way["highway"]{around};
  way["waterway"]{around};
  way["railway"~"^(rail|light_rail|tram|narrow_gauge)$"]{around};
);
out geom;"""
    result = _download(query, timeout)
    with open(cache, "w") as f:
        json.dump(result, f)
    return result


class OverpassUnavailable(RuntimeError):
    """Every Overpass server failed; the message says what to do instead."""


def _download(query, timeout, sleep=time.sleep):
    data = urllib.parse.urlencode({"data": query}).encode()
    errors = []
    for url in OVERPASS_URLS:
        host = urllib.parse.urlparse(url).netloc
        for attempt in range(TRIES_PER_SERVER):
            req = urllib.request.Request(url, data=data,
                                         headers={"User-Agent": "SM2DH2C-drone-sim/1.0"})
            try:
                print(f"Downloading OSM data from {host}" + (" (retry)" if attempt else "") + " ...")
                with urllib.request.urlopen(req, timeout=timeout + 15) as resp:
                    result = json.load(resp)
                if "elements" not in result:
                    raise ValueError(result.get("remark", "response has no elements"))
                if result.get("remark") and "error" in result["remark"].lower():
                    raise ValueError(result["remark"])          # e.g. server-side timeout
                return result
            except urllib.error.HTTPError as exc:
                errors.append(f"{host}: HTTP {exc.code} {exc.reason}")
                if exc.code not in RETRY_STATUS:
                    break                                       # e.g. 400 = bad query, don't retry
            except (urllib.error.URLError, TimeoutError, ValueError, OSError) as exc:
                errors.append(f"{host}: {getattr(exc, 'reason', exc)}")
            if attempt + 1 < TRIES_PER_SERVER:
                sleep(5 * (attempt + 1))                        # short back-off before retrying
    raise OverpassUnavailable(
        "Could not download OpenStreetMap data (all Overpass servers busy or unreachable):\n  "
        + "\n  ".join(errors)
        + "\nTry again in a minute, or use an offline export: openstreetmap.org -> Export -> "
          "save map.osm, then add --osm-file map.osm to the same command.")


def load_osm_file(path):
    """Read an Overpass JSON file or a .osm XML export into the same element format
    (nodes with tags, ways with geometry, relations with member way geometry)."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            return json.load(f)
    root = ET.parse(path).getroot()
    nodes, ways, elements = {}, {}, []
    for n in root.iter("node"):
        nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")))
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        if tags:
            elements.append({"type": "node", "lat": float(n.get("lat")),
                             "lon": float(n.get("lon")), "tags": tags})
    for w in root.iter("way"):
        geom = [{"lat": nodes[nd.get("ref")][0], "lon": nodes[nd.get("ref")][1]}
                for nd in w.findall("nd") if nd.get("ref") in nodes]
        ways[w.get("id")] = geom
        elements.append({"type": "way", "tags": {t.get("k"): t.get("v") for t in w.findall("tag")},
                         "geometry": geom})
    for r in root.iter("relation"):
        tags = {t.get("k"): t.get("v") for t in r.findall("tag")}
        members = [{"type": "way", "role": m.get("role", ""), "geometry": ways[m.get("ref")]}
                   for m in r.findall("member") if m.get("type") == "way" and m.get("ref") in ways]
        if members:
            elements.append({"type": "relation", "tags": tags, "members": members})
    return {"elements": elements}


def stitch_rings(segments):
    """Join way pieces (lists of (lat, lon)) end-to-end into closed rings. Multipolygon
    relations often split one outline across several ways."""
    segs = [list(s) for s in segments if len(s) >= 2]
    rings = []
    while segs:
        cur = segs.pop(0)
        joined = True
        while cur[0] != cur[-1] and joined:
            joined = False
            for i, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur = cur + s[1:]
                elif s[-1] == cur[-1]:
                    cur = cur + s[::-1][1:]
                elif s[-1] == cur[0]:
                    cur = s[:-1] + cur
                elif s[0] == cur[0]:
                    cur = s[::-1][:-1] + cur
                else:
                    continue
                segs.pop(i)
                joined = True
                break
        if len(cur) >= 4 and cur[0] == cur[-1]:
            rings.append(cur)
    return rings


def relation_outer_rings(el):
    """Outer rings of a multipolygon relation, in the same {'lat','lon'} format as way geometry.
    Inner rings (holes such as courtyards) are ignored: the area is drawn/solid without the hole."""
    pieces = [[(g["lat"], g["lon"]) for g in m.get("geometry") or [] if g]
              for m in el.get("members", [])
              if m.get("type") == "way" and m.get("role", "") in ("outer", "")]
    return [[{"lat": a, "lon": b} for a, b in ring] for ring in stitch_rings(pieces)]


# ------------------------------------------------------------------ geometry helpers
def make_projector(lat0, lon0, scale):
    """lat/lon -> sim (x, y). x = east, y = north, LKP at (0, 0). Accurate to well under
    1% over a few km, which is far more than this map ever shows."""
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    def to_xy(lat, lon):
        return ((lon - lon0) * m_per_deg_lon * scale, (lat - lat0) * m_per_deg_lat * scale)
    return to_xy


def clip_polygon(poly, half=None):
    """Sutherland-Hodgman clip of a polygon to the square map."""
    half = sim_config.MAP_HALF if half is None else half
    def clip(pts, inside, cross):
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(cross(prev, cur))
                out.append(cur)
            elif inside(prev):
                out.append(cross(prev, cur))
        return out

    def x_cross(xv):
        return lambda a, b: (xv, a[1] + (b[1] - a[1]) * (xv - a[0]) / (b[0] - a[0]))

    def y_cross(yv):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (yv - a[1]) / (b[1] - a[1]), yv)

    pts = list(poly)
    for inside, cross in ((lambda q: q[0] >= -half, x_cross(-half)),
                          (lambda q: q[0] <= half, x_cross(half)),
                          (lambda q: q[1] >= -half, y_cross(-half)),
                          (lambda q: q[1] <= half, y_cross(half))):
        if not pts:
            break
        pts = clip(pts, inside, cross)
    return pts


def clip_segment(a, b, half=None):
    """Liang-Barsky clip of a line segment to the square map. Returns None if fully outside."""
    half = sim_config.MAP_HALF if half is None else half
    t0, t1 = 0.0, 1.0
    dx, dy = b[0] - a[0], b[1] - a[1]
    for pp, q in ((-dx, a[0] + half), (dx, half - a[0]), (-dy, a[1] + half), (dy, half - a[1])):
        if pp == 0:
            if q < 0:
                return None
            continue
        t = q / pp
        if pp < 0:
            t0 = max(t0, t)
        else:
            t1 = min(t1, t)
        if t0 > t1:
            return None
    return (a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy)


def polygon_area(poly):
    return 0.5 * sum(poly[i - 1][0] * poly[i][1] - poly[i][0] * poly[i - 1][1] for i in range(len(poly)))


def point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i - 1], poly[i]
        if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
            inside = not inside
    return inside


def distance_to_polygon(pt, poly):
    """0 if inside, else distance to the nearest edge."""
    if point_in_polygon(pt, poly):
        return 0.0
    best = float("inf")
    for i in range(len(poly)):
        a, b = poly[i - 1], poly[i]
        dx, dy = b[0] - a[0], b[1] - a[1]
        L2 = dx * dx + dy * dy
        t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / L2))
        best = min(best, math.hypot(pt[0] - a[0] - t * dx, pt[1] - a[1] - t * dy))
    return best


def triangulate(poly):
    """Ear-clipping triangulation of a simple polygon (handles L-shaped / concave buildings).
    Returns index triples into poly, counter-clockwise."""
    idx = list(range(len(poly)))
    if polygon_area(poly) < 0:
        idx.reverse()

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    tris, guard = [], 0
    while len(idx) > 3 and guard < 10 * len(poly):
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if cross(a, b, c) <= 1e-12:                  # reflex corner, not an ear
                continue
            if any(point_in_polygon(poly[j], (a, b, c)) for j in idx if j not in (i0, i1, i2)):
                continue
            tris.append((i0, i1, i2))
            idx.pop(k)
            break
        else:
            break                                        # degenerate polygon; stop safely
    if len(idx) == 3:
        tris.append(tuple(idx))
    return tris


def ring_to_xy(geometry, to_xy):
    pts = [to_xy(g["lat"], g["lon"]) for g in geometry]
    if len(pts) > 1 and math.isclose(pts[0][0], pts[-1][0]) and math.isclose(pts[0][1], pts[-1][1]):
        pts = pts[:-1]                                   # OSM repeats the first node to close a ring
    cleaned = []                                         # drop duplicate consecutive points
    for q in pts:
        if not cleaned or math.hypot(q[0] - cleaned[-1][0], q[1] - cleaned[-1][1]) > 1e-4:
            cleaned.append(q)
    return cleaned


def building_height(tags, scale):
    """Real height from OSM tags, scaled, then clamped so it is always a meaningful obstacle."""
    h = None
    for key in ("height", "building:height"):
        try:
            h = float(str(tags.get(key, "")).replace("m", "").strip())
            break
        except ValueError:
            pass
    if h is None and tags.get("building:levels"):
        try:
            h = float(tags["building:levels"]) * BUILDING_LEVEL_HEIGHT
        except ValueError:
            pass
    h = DEFAULT_BUILDING_HEIGHT if h is None else h
    return max(MIN_BUILDING_HEIGHT, min(MAX_BUILDING_HEIGHT, h * scale))


def building_info(tags, scale):
    """Height (sim m, clamped like before) plus what the building should look like."""
    h = building_height(tags, scale)
    # storeys: from the tag if present, else from the real height; used for the window grid
    try:
        storeys = max(1, int(round(float(tags.get("building:levels")))))
    except (TypeError, ValueError):
        storeys = max(1, int(round(h / scale / BUILDING_LEVEL_HEIGHT)))
    kind = tags.get("building", "yes")
    material = str(tags.get("building:material", "")).lower()
    if material in MATERIAL_STYLE:
        style = MATERIAL_STYLE[material]
    else:
        style = next((s for s, kinds in BUILDING_STYLE_KINDS.items() if kind in kinds), "stucco")
    return h, {"style": style, "storeys": storeys, "kind": kind,
               "wall": tags.get("building:colour"), "roof": tags.get("roof:colour"),
               "roof_shape": roof_shape(tags, kind), "roof_height": _metres(tags.get("roof:height")),
               "material_set": material in MATERIAL_STYLE}


def _metres(value):
    """'12', '12 m', '12m' -> 12.0; anything else -> None."""
    try:
        return float(str(value).replace("m", "").strip())
    except (TypeError, ValueError):
        return None


PITCHED_BY_DEFAULT = {"house", "detached", "semidetached_house", "terrace", "bungalow", "cabin",
                      "farm", "barn", "garage", "shed", "church", "chapel", "residential"}
ROOF_SHAPES = {"gabled", "hipped", "pyramidal", "skillion", "dome", "onion", "flat",
               "half-hipped", "gambrel", "mansard", "saltbox", "round"}


def roof_shape(tags, kind):
    """roof:shape tag, mapped to what we can draw; untagged houses get a pitched roof."""
    shape = str(tags.get("roof:shape", "")).lower()
    if shape in ("half-hipped", "mansard"):
        return "hipped"
    if shape in ("gambrel", "saltbox", "round"):
        return "gabled"
    if shape == "onion":
        return "dome"
    if shape in ROOF_SHAPES:
        return shape
    return "gabled" if kind in PITCHED_BY_DEFAULT else "flat"


def part_info(tags):
    """Real-world vertical extent of a building:part (scaled later to fit its parent)."""
    h = _metres(tags.get("height")) or _metres(tags.get("building:height"))
    if h is None and tags.get("building:levels"):
        lv = _metres(tags.get("building:levels"))
        h = None if lv is None else lv * BUILDING_LEVEL_HEIGHT + (_metres(tags.get("roof:height")) or 0)
    h = DEFAULT_BUILDING_HEIGHT if h is None else h
    z0 = _metres(tags.get("min_height"))
    if z0 is None and tags.get("building:min_level"):
        ml = _metres(tags.get("building:min_level"))
        z0 = None if ml is None else ml * BUILDING_LEVEL_HEIGHT
    z0 = min(z0 or 0.0, h - 0.5)
    material = str(tags.get("building:material", "")).lower()
    return {"top": h, "bottom": max(0.0, z0), "wall": tags.get("building:colour"),
            "roof": tags.get("roof:colour"), "roof_shape": roof_shape(tags, "part"),
            "style": MATERIAL_STYLE.get(material),
            "levels": _metres(tags.get("building:levels"))}


# Building type -> wall style (see visual_assets.facade_texture).
BUILDING_STYLE_KINDS = {
    "brick": {"university", "college", "school", "public", "civic", "government", "library",
              "church", "chapel", "apartments", "dormitory", "train_station", "townhall"},
    "siding": {"house", "residential", "detached", "semidetached_house", "terrace", "bungalow",
               "cabin", "farm"},
    "glass": {"commercial", "office", "retail", "hotel", "hospital", "supermarket"},
    "metal": {"industrial", "warehouse", "garage", "garages", "shed", "barn", "hangar", "service",
              "carport", "greenhouse", "roof", "storage_tank", "transportation", "parking"},
}
MATERIAL_STYLE = {"brick": "brick", "glass": "glass", "metal": "metal", "steel": "metal",
                  "wood": "siding", "timber_framing": "siding", "concrete": "stucco",
                  "plaster": "stucco", "stone": "stucco", "sandstone": "stucco"}

# Flat ground surfaces: category -> (height above ground in m, RGB). Higher layers draw over lower
# ones (4 mm apart so the GUI depth buffer doesn't flicker); all stay below the heatmap.
SURFACE_LAYERS = {
    "farm": (0.004, (0.66, 0.62, 0.42)),
    "grass": (0.008, (0.40, 0.58, 0.28)),
    "scrub": (0.012, (0.45, 0.52, 0.30)),
    "forest": (0.016, (0.27, 0.38, 0.20)),
    "dirt": (0.020, (0.55, 0.47, 0.36)),
    "pitch": (0.024, (0.33, 0.60, 0.30)),
    "track": (0.024, (0.66, 0.36, 0.28)),
    "sand": (0.028, (0.84, 0.77, 0.58)),
    "rock": (0.028, (0.55, 0.53, 0.50)),
    "parking": (0.032, (0.31, 0.31, 0.33)),
    "plaza": (0.036, (0.72, 0.70, 0.66)),
    "water": (0.040, (0.22, 0.43, 0.70)),
}
# Line features: category -> (height, RGB, default real width in m)
LINE_LAYERS = {
    "stream": (0.044, (0.24, 0.45, 0.72), 2.0),
    "path": (0.048, (0.60, 0.50, 0.36), 1.5),
    "cycle": (0.052, (0.62, 0.42, 0.38), 2.0),
    "foot": (0.056, (0.76, 0.74, 0.70), 2.0),
    "rail": (0.060, (0.38, 0.34, 0.31), 3.0),
    "service": (0.064, (0.30, 0.30, 0.32), 4.0),
    "minor": (0.068, (0.26, 0.26, 0.28), 6.0),
    "major": (0.072, (0.22, 0.22, 0.24), 8.0),
    "crossing": (0.074, (0.93, 0.93, 0.90), 3.0),     # drawn as zebra stripes over the road
}
MARKING_Z, MARKING_RGB = 0.076, (0.92, 0.78, 0.20)      # yellow centre dashes on major roads

HIGHWAY_CATEGORY = {
    **{k: "major" for k in ("motorway", "trunk", "primary", "secondary", "tertiary", "motorway_link",
                            "trunk_link", "primary_link", "secondary_link", "tertiary_link")},
    **{k: "minor" for k in ("residential", "unclassified", "living_street", "road")},
    "service": "service",
    **{k: "foot" for k in ("footway", "pedestrian", "steps", "corridor", "sidewalk", "crossing")},
    "cycleway": "cycle",
    **{k: "path" for k in ("path", "track", "bridleway")},
}
WATERWAY_WIDTH_M = {"river": 10.0, "canal": 6.0, "stream": 2.0, "ditch": 1.0, "drain": 1.0}
CONIFER_GENUS = {"pinus", "picea", "abies", "pseudotsuga", "thuja", "tsuga", "sequoia",
                 "sequoiadendron", "cedrus", "juniperus", "larix", "taxus", "calocedrus", "chamaecyparis"}


def surface_category(tags):
    """Which flat ground surface an area represents, or None (e.g. landuse=residential is a zoning
    area, not a surface, so it is not drawn)."""
    lu, le, na, am = (tags.get(k) for k in ("landuse", "leisure", "natural", "amenity"))
    if na == "water" or lu in ("reservoir", "basin") or tags.get("water"):
        return "water"
    if am == "parking":
        return "parking"
    if le == "track":
        return "track"
    if le == "pitch":
        return "pitch"
    if le == "playground" or na in ("sand", "beach"):
        return "sand"
    if na in ("bare_rock", "scree"):
        return "rock"
    if lu == "forest" or na == "wood":
        return "forest"
    if na in ("scrub", "heath"):
        return "scrub"
    if lu in ("farmland", "farmyard", "orchard", "vineyard", "allotments"):
        return "farm"
    if (lu in ("grass", "meadow", "village_green", "recreation_ground", "cemetery")
            or le in ("park", "garden", "golf_course", "common", "dog_park") or na == "grassland"):
        return "grass"
    if lu in ("construction", "brownfield", "landfill"):
        return "dirt"
    if (tags.get("highway") in ("pedestrian", "footway") and tags.get("area") == "yes") \
            or tags.get("place") == "square" or am == "marketplace":
        return "plaza"
    return None


def tree_kind(tags, rng=None, mixed=0.5):
    leaf = tags.get("leaf_type")
    if leaf == "needleleaved" or str(tags.get("genus", "")).lower() in CONIFER_GENUS:
        return "conifer"
    if leaf == "broadleaved" or rng is None:
        return "deciduous"
    return "conifer" if rng.random() < mixed else "deciduous"


def _point_segment_dist(pt, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((pt[0] - a[0]) * dx + (pt[1] - a[1]) * dy) / L2))
    return math.hypot(pt[0] - a[0] - t * dx, pt[1] - a[1] - t * dy)


# ------------------------------------------------------------------ the parsed scene
class OSMScene:
    """Everything parsed from OSM, already in sim coordinates and clipped to the map."""

    def __init__(self):
        self.scale = 1.0
        self.buildings = []     # [(polygon [(x, y), ...], height, info dict)]
        self.trees = []         # [(x, y)]
        self.tree_kinds = []    # ["deciduous" | "conifer"], parallel to trees
        self.roads = []         # [((x1, y1), (x2, y2), width, category)]
        self.water = []         # [polygon]
        self.surfaces = []      # [(polygon, category)]  grass, parking, sand, forest floor, ...
        self.parts = []         # [(polygon, part info)]  building:part pieces (visual only)
        self.barriers = []      # [((x1, y1), (x2, y2), kind)]  hedge / fence / wall (visual only, low)
        self.parking_lots = []  # [polygon]  surface parking where parked cars are drawn
        self.pitches = []       # [(polygon, sport)]  sports fields that get white line markings
        self.dropped = []       # human-readable notes about features removed to keep the mission valid
        self.moved = {}         # keep_clear point -> where it was moved (it was on a building)

    def summary(self):
        return (f"{len(self.buildings)} buildings, {len(self.trees)} trees, "
                f"{len(self.roads)} road/path segments, {len(self.water)} water areas, "
                f"{len(self.surfaces)} ground areas")


def parse_scene(osm_json, lat, lon, area_m=DEFAULT_AREA_M, keep_clear=(), seed=0):
    """Convert OSM elements into an OSMScene that fits the fixed-size map.
    keep_clear: (x, y) points (drone start, zone centres) that must stay reachable."""
    MAP_HALF = sim_config.MAP_HALF
    area_m = 2 * MAP_HALF if area_m is None else area_m
    scale = (2 * MAP_HALF) / area_m
    to_xy = make_projector(lat, lon, scale)
    scene = OSMScene()
    scene.scale = scale
    woods = []                  # (polygon, tags) for tree scattering
    trees = []                  # (x, y, kind, scattered?)
    rng = random.Random(seed)

    def add_area(pts, tags):
        """A closed outline (from a way or a relation ring)."""
        if len(pts) < 3:
            return
        if tags.get("building:part", "no") != "no" and tags.get("building", "no") == "no":
            poly = clip_polygon(pts)
            if len(poly) >= 3 and abs(polygon_area(poly)) > 0.02:
                scene.parts.append((poly, part_info(tags)))
            return
        if tags.get("building", "no") != "no":
            poly = clip_polygon(pts)
            if len(poly) >= 3 and abs(polygon_area(poly)) > 0.05:
                h, info = building_info(tags, scale)
                scene.buildings.append((poly, h, info))
            return
        cat = surface_category(tags)
        if cat is None:
            return
        if cat == "forest" or tags.get("natural") == "scrub":
            woods.append((pts, tags))
        poly = clip_polygon(pts)
        if len(poly) >= 3 and abs(polygon_area(poly)) > 0.01:
            if cat == "water":
                scene.water.append(poly)
            else:
                scene.surfaces.append((poly, cat))
            if cat == "parking" and tags.get("parking", "surface") in ("surface", "lane", "street_side"):
                scene.parking_lots.append(poly)
            if cat in ("pitch", "track"):
                scene.pitches.append((poly, tags.get("sport", "")))

    def add_line(pts, tags):
        if tags.get("tunnel") in ("yes", "culvert") or tags.get("layer", "0").startswith("-"):
            return                          # underground: not visible from the air
        if "barrier" in tags:
            kind = "hedge" if tags["barrier"] == "hedge" else (
                "fence" if tags["barrier"] in ("fence", "guard_rail") else "wall")
            for a, b in zip(pts, pts[1:]):
                seg = clip_segment(a, b)
                if seg and math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) > 1e-3:
                    scene.barriers.append((seg[0], seg[1], kind))
            return
        if "highway" in tags:
            cat = HIGHWAY_CATEGORY.get(tags["highway"])
            if cat is None:
                return                      # proposed / construction / bus_stop / ...
            w = ROAD_WIDTH_M.get(tags["highway"], LINE_LAYERS[cat][2])
            if tags.get("highway") == "pedestrian":
                w = 4.0
            lanes = _metres(tags.get("lanes"))
            if lanes and cat in ("major", "minor", "service"):
                w = max(lanes * 3.3, 3.0)                            # real width from lane count
            if tags.get("footway") == "crossing" or tags.get("cycleway") == "crossing":
                cat, w = "crossing", 3.0
        elif "waterway" in tags:
            cat, w = "stream", WATERWAY_WIDTH_M.get(tags["waterway"], 1.0)
        elif "railway" in tags:
            cat, w = "rail", LINE_LAYERS["rail"][2]
        else:
            return
        try:
            w = float(str(tags.get("width", "")).replace("m", "").strip())
        except ValueError:
            pass
        width = max(MIN_ROAD_WIDTH, w * scale)
        for a, b in zip(pts, pts[1:]):
            seg = clip_segment(a, b)
            if seg and math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) > 1e-3:
                scene.roads.append((seg[0], seg[1], width, cat))

    for el in osm_json.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            if tags.get("natural") == "tree":
                x, y = to_xy(el["lat"], el["lon"])
                if abs(x) <= MAP_HALF and abs(y) <= MAP_HALF:
                    trees.append((x, y, tree_kind(tags), False))
            continue
        if el["type"] == "relation":
            if tags.get("type") not in ("multipolygon", "building", None):
                continue
            for ring in relation_outer_rings(el):
                add_area(ring_to_xy(ring, to_xy), tags)
            continue
        geom = [g for g in (el.get("geometry") or []) if g]
        if len(geom) < 2:
            continue
        closed = len(geom) >= 4 and geom[0] == geom[-1]
        pts = ring_to_xy(geom, to_xy)
        is_line = ("highway" in tags and tags.get("area") != "yes") or "waterway" in tags \
            or "railway" in tags or ("barrier" in tags and "building" not in tags)
        if is_line:
            add_line(pts, tags)
        elif closed:
            add_area(pts, tags)

    # Scatter trees inside wood/forest polygons on a jittered grid (deterministic per seed).
    s = WOOD_TREE_SPACING
    for poly, wtags in woods:
        xs, ys = [q[0] for q in poly], [q[1] for q in poly]
        x = max(min(xs), -MAP_HALF)
        while x <= min(max(xs), MAP_HALF):
            y = max(min(ys), -MAP_HALF)
            while y <= min(max(ys), MAP_HALF):
                q = (x + rng.uniform(-0.3, 0.3) * s, y + rng.uniform(-0.3, 0.3) * s)
                if abs(q[0]) <= MAP_HALF and abs(q[1]) <= MAP_HALF and point_in_polygon(q, poly):
                    trees.append((q[0], q[1], tree_kind(wtags, rng), True))
                y += s
            x += s

    # Scattered trees should not stand in the middle of a road or path.
    if scene.roads:
        trees = [t for t in trees if not t[3] or all(
            _point_segment_dist(t, a, b) > w / 2 + 0.3 for a, b, w, _ in scene.roads)]

    # Trees must not sit inside buildings, and very close trees are merged.
    merged = []
    for t in trees:
        if any(point_in_polygon(t[:2], b) for b, *_ in scene.buildings):
            continue
        if all(math.hypot(t[0] - m[0], t[1] - m[1]) >= MIN_TREE_SPACING for m in merged):
            merged.append(t)
    # Same tree density as the 26 m map: the cap grows with map area, up to MAX_TREES_LIMIT.
    max_trees = min(MAX_TREES_LIMIT, int(MAX_TREES * max(1.0, sim_config.layout_scale()) ** 2))
    if len(merged) > max_trees:
        scene.dropped.append(f"{len(merged) - max_trees} trees over the {max_trees}-tree cap")
        merged.sort(key=lambda t: math.hypot(t[0], t[1]))      # keep the ones nearest the LKP
        merged = merged[:max_trees]
    trees = merged

    # Keep the drone start and every search-zone centre reachable. If a point lands on a building
    # (e.g. a big library in the middle of the map), MOVE THE POINT to the nearest open ground and
    # keep the building. A building is only removed when there is no open ground left to move to.
    for c0 in keep_clear:
        c = tuple(c0)
        if not _clear_of_buildings(scene, c):
            moved = _nearest_clear_point(scene, c)
            if moved is not None:
                scene.moved[tuple(c0)] = moved
                scene.dropped.append(f"point ({c[0]:.0f}, {c[1]:.0f}) is on a building; "
                                     f"moved to ({moved[0]:.1f}, {moved[1]:.1f})")
                c = moved
            else:
                kept = []
                for b in scene.buildings:
                    if distance_to_polygon(c, b[0]) < KEEP_CLEAR_RADIUS:
                        scene.dropped.append(f"building covering ({c[0]:.0f}, {c[1]:.0f}) "
                                             f"(no open ground nearby)")
                    else:
                        kept.append(b)
                scene.buildings = kept
        before = len(trees)
        trees = [t for t in trees
                 if math.hypot(t[0] - c[0], t[1] - c[1]) >= KEEP_CLEAR_RADIUS + TREE_CLEAR_RADIUS]
        if len(trees) < before:
            scene.dropped.append(f"{before - len(trees)} tree(s) near ({c[0]:.0f}, {c[1]:.0f})")
    scene.trees = [(t[0], t[1]) for t in trees]
    scene.tree_kinds = [t[2] for t in trees]
    return scene


def _clear_of_buildings(scene, xy, margin=KEEP_CLEAR_RADIUS):
    return all(distance_to_polygon(xy, b[0]) >= margin for b in scene.buildings)


def _nearest_clear_point(scene, xy, step=0.5):
    """Spiral outward from xy to the closest point inside the map that is KEEP_CLEAR_RADIUS away
    from every building. None if the whole map is covered."""
    lim = sim_config.MAP_HALF - 1.0
    r = step
    while r <= 2 * sim_config.MAP_HALF:
        n = max(8, int(2 * math.pi * r / step))
        for k in range(n):
            a = 2 * math.pi * k / n
            q = (xy[0] + r * math.cos(a), xy[1] + r * math.sin(a))
            if abs(q[0]) <= lim and abs(q[1]) <= lim and _clear_of_buildings(scene, q):
                return (round(q[0], 2), round(q[1], 2))
        r += step
    return None


def is_free(scene, xy, margin=1.0):
    """True if xy is outside every building (by margin) and away from every tree trunk."""
    return (all(distance_to_polygon(xy, b[0]) >= margin for b in scene.buildings)
            and all(math.hypot(xy[0] - t[0], xy[1] - t[1]) >= margin for t in scene.trees))


def nearest_free_point(scene, xy, margin=1.0, step=0.5, max_r=None):
    """Spiral outward from xy until a free spot inside the map is found."""
    MAP_HALF = sim_config.MAP_HALF
    max_r = MAP_HALF - 1 if max_r is None else max_r
    if is_free(scene, xy, margin):
        return xy
    r = step
    while r <= max_r:
        for k in range(max(8, int(2 * math.pi * r / step))):
            a = 2 * math.pi * k / max(8, int(2 * math.pi * r / step))
            q = (xy[0] + r * math.cos(a), xy[1] + r * math.sin(a))
            if abs(q[0]) <= MAP_HALF - 1 and abs(q[1]) <= MAP_HALF - 1 and is_free(scene, q, margin):
                return (round(q[0], 2), round(q[1], 2))
        r += step
    return xy


# ------------------------------------------------------------------ PyBullet bodies
LOW_SPECULAR = [0.04, 0.04, 0.04]   # PyBullet's default specular makes everything look like plastic


def _mesh_bodies(p, builder, rgb, alpha=1.0):
    """One visual-only body per mesh chunk (no collision: rays and sensors ignore them)."""
    for v, n, uv, i in builder.meshes():
        vis = p.createVisualShape(p.GEOM_MESH, vertices=v, indices=i, normals=n, uvs=uv,
                                  rgbaColor=list(rgb) + [alpha], specularColor=LOW_SPECULAR)
        p.createMultiBody(0, -1, vis, [0, 0, 0])


def _building_collision(p, poly, h):
    """Solid extruded footprint (same as before): this is what the LIDAR and A* see."""
    tris = triangulate(poly)
    if not tris:
        return None, None
    n = len(poly)
    verts = [[x, y, 0.0] for x, y in poly] + [[x, y, h] for x, y in poly]
    idx = []
    for a, b, c in tris:
        idx += [a + n, b + n, c + n]                 # roof (CCW from above)
        idx += [c, b, a]                             # floor (CW from above = facing down)
    ccw = polygon_area(poly) > 0
    for i in range(n):
        j = (i + 1) % n
        quad = (i, j, j + n, i, j + n, i + n) if ccw else (j, i, i + n, j, i + n, j + n)
        idx += list(quad)                            # walls, facing outward
    col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx,
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    return col, (verts, idx)


def _build_simple(p, scene):
    """The original flat-colour look (also the fallback if anything in the detailed look fails)."""
    solid_ids = []
    for poly, h, *_ in scene.buildings:
        col, mesh = _building_collision(p, poly, h)
        if col is None:
            continue
        vis = p.createVisualShape(p.GEOM_MESH, vertices=mesh[0], indices=mesh[1],
                                  rgbaColor=[0.62, 0.58, 0.55, 1])
        solid_ids.append(p.createMultiBody(0, col, vis, [0, 0, 0]))

    for a, b, width, *_ in scene.roads:
        length = math.hypot(b[0] - a[0], b[1] - a[1])
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[length / 2, width / 2, 0.005],
                                  rgbaColor=[0.35, 0.35, 0.35, 1])
        yaw = math.atan2(b[1] - a[1], b[0] - a[0])
        p.createMultiBody(0, -1, vis, [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.006],
                          p.getQuaternionFromEuler([0, 0, yaw]))

    for poly in scene.water:
        tris = triangulate(poly)
        if not tris:
            continue
        verts = [[x, y, 0.004] for x, y in poly]
        idx = [i for t in tris for i in t]
        vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx,
                                  rgbaColor=[0.2, 0.45, 0.85, 0.85])
        p.createMultiBody(0, -1, vis, [0, 0, 0])
    return solid_ids


def _build_detailed(p, scene, avoid=()):
    import visual_assets as va
    solid_ids = []

    # --- ground surfaces (grass, parking, sand, forest floor, water ...), merged per category
    try:
        _detailed_ground(p, va, scene)
    except Exception as exc:
        print("  ground areas drawn simple:", exc)
        _build_simple(p, _only(scene, water=True))
    try:
        _detailed_lines(p, va, scene)
    except Exception as exc:
        print("  roads drawn simple:", exc)
        _build_simple(p, _only(scene, roads=True))
    try:
        _detailed_props(p, va, scene, avoid)
    except Exception as exc:
        print("  cars / fences / field lines skipped:", exc)
    _detailed_buildings(p, va, scene, solid_ids)
    return solid_ids


def _only(scene, water=False, roads=False):
    """Copy of the scene holding just one feature type (for the simple-look fallback)."""
    part = OSMScene()
    part.water = list(scene.water) if water else []
    part.roads = list(scene.roads) if roads else []
    return part


def _detailed_ground(p, va, scene):
    by_cat = {}
    for poly, cat in scene.surfaces:
        by_cat.setdefault(cat, []).append(poly)
    by_cat.setdefault("water", []).extend(scene.water)
    for cat, polys in by_cat.items():
        z, rgb = SURFACE_LAYERS[cat]
        mb = va.MeshBuilder()
        for poly in polys:
            mb.add_flat_polygon(va.simplify(poly, 0.03), z, triangulate)
        _mesh_bodies(p, mb, rgb, alpha=0.92 if cat == "water" else 1.0)


def _offset_segment(a, b, d):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy) or 1e-9
    nx, ny = -dy / L * d, dx / L * d
    return (a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny)


def _detailed_lines(p, va, scene):
    """Roads, paths, rails, streams: merged strips with round joints. Major roads get yellow centre
    dashes and white edge lines; crossings are drawn as white zebra stripes."""
    by_cat = {}
    yellow, white = va.MeshBuilder(), va.MeshBuilder()
    s = min(scene.scale, 1.2)
    dash = max(0.25, 3.0 * s)
    line_w = max(0.05, 0.15 * s)
    # Round joints only where two pieces of road meet (not at dead ends or the map edge).
    ends = {}
    for a, b, width, cat in scene.roads:
        for q in (a, b):
            key = (cat, round(q[0], 3), round(q[1], 3))
            ends[key] = ends.get(key, 0) + 1
    joint = lambda cat, q: ends.get((cat, round(q[0], 3), round(q[1], 3)), 0) > 1
    for a, b, width, cat in scene.roads:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if cat == "crossing":
            ux, uy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            half = max(0.4, 1.5 * s)
            d = 0.45 * s
            while d < L:
                c = (a[0] + ux * d, a[1] + uy * d)
                white.add_strip((c[0] - uy * half, c[1] + ux * half),
                                (c[0] + uy * half, c[1] - ux * half), max(0.06, 0.45 * s), MARKING_Z)
                d += max(0.15, 0.9 * s)
            continue
        by_cat.setdefault(cat, va.MeshBuilder()).add_strip(a, b, width, LINE_LAYERS[cat][0],
                                                           caps=(joint(cat, a), joint(cat, b)))
        if cat == "major" and width >= 6 * line_w:
            d = 0.0
            while d + dash <= L:
                p0 = (a[0] + (b[0] - a[0]) * d / L, a[1] + (b[1] - a[1]) * d / L)
                p1 = (a[0] + (b[0] - a[0]) * (d + dash) / L, a[1] + (b[1] - a[1]) * (d + dash) / L)
                yellow.add_strip(p0, p1, line_w, MARKING_Z)
                d += 3 * dash
            for side in (1, -1):                                   # white edge lines
                e0, e1 = _offset_segment(a, b, side * (width / 2 - max(line_w, 0.4 * s)))
                white.add_strip(e0, e1, line_w * 0.8, MARKING_Z)
    for cat, mb in by_cat.items():
        _mesh_bodies(p, mb, LINE_LAYERS[cat][1])
    _mesh_bodies(p, yellow, MARKING_RGB)
    _mesh_bodies(p, white, (0.93, 0.93, 0.90))


def _edge_distance(pt, poly):
    return min(_point_segment_dist(pt, poly[i - 1], poly[i]) for i in range(len(poly)))


MAX_CARS = 400


def _detailed_props(p, va, scene, avoid):
    """Low, visual-only details. All stay below 1.8 m, i.e. under the 2.5 m cruise altitude, so the
    drone never visibly flies through them: parked cars, hedges, fences, walls, field lines."""
    s = min(scene.scale, 1.2)
    rng = random.Random(7)

    # parked cars in surface parking lots, in rows of stalls along the lot's long side
    bodies, glass = {}, va.MeshBuilder()
    n_cars = 0
    stall, depth, aisle = 2.7 * s, 5.5 * s, 7.0 * s
    for poly in scene.parking_lots:
        rect = va.min_area_rect(poly)
        if rect is None or rect[3] < depth:
            continue
        centre, ang, L, W = rect
        P = va._local(centre, ang)
        v = -W / 2 + depth / 2
        row = 0
        while v <= W / 2 - depth / 2 and n_cars < MAX_CARS:
            u = -L / 2 + stall / 2
            while u <= L / 2 - stall / 2 and n_cars < MAX_CARS:
                x, y, _ = P(u, v, 0)
                if (point_in_polygon((x, y), poly) and _edge_distance((x, y), poly) > 1.2 * s
                        and rng.random() < 0.7
                        and not any(point_in_polygon((x, y), b[0]) for b in scene.buildings)
                        and all(math.hypot(x - q[0], y - q[1]) > 2.5 for q in avoid)):
                    car_ang = ang + math.pi / 2 + (math.pi if row % 2 else 0) + rng.uniform(-0.05, 0.05)
                    body, gl = va.car_tris((x, y), car_ang, s)
                    colour = rng.choice(va.CAR_COLOURS)
                    mid = (x, y, 0.8 * s)
                    va.add_faceted(bodies.setdefault(colour, va.MeshBuilder()), body, outward_from=mid)
                    va.add_faceted(glass, gl, outward_from=mid)
                    n_cars += 1
                u += stall
            v += depth if row % 2 == 0 else depth + aisle            # two facing rows, then an aisle
            row += 1
    for colour, mb in bodies.items():
        _mesh_bodies(p, mb, colour)
    _mesh_bodies(p, glass, va.CAR_GLASS)

    # hedges (green boxes), fences (thin panels), walls (thin stone boxes)
    hedge, fence, wall = va.MeshBuilder(), va.MeshBuilder(), va.MeshBuilder()
    for a, b, kind in scene.barriers:
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        ang = math.atan2(b[1] - a[1], b[0] - a[0])
        if kind == "hedge":
            hh = min(1.6, max(0.3, 1.4 * s))
            va.add_faceted(hedge, va.box_tris(c, ang, L + 0.4 * s, max(0.15, 0.9 * s), 0, hh),
                           outward_from=(c[0], c[1], hh / 2))
        elif kind == "fence":
            va.add_faceted(fence, va.vertical_strip_tris(a, b, 0, min(1.5, max(0.25, 1.4 * s))))
        else:
            wh = min(1.8, max(0.3, 1.6 * s))
            va.add_faceted(wall, va.box_tris(c, ang, L, max(0.08, 0.3 * s), 0, wh),
                           outward_from=(c[0], c[1], wh / 2))
    _mesh_bodies(p, hedge, (0.20, 0.38, 0.17))
    _mesh_bodies(p, fence, (0.50, 0.45, 0.40))
    _mesh_bodies(p, wall, (0.62, 0.60, 0.56))

    # white boundary + halfway line on sports fields
    lines = va.MeshBuilder()
    w = max(0.04, 0.12 * s)
    for poly, sport in scene.pitches:
        z = SURFACE_LAYERS["pitch"][0] + 0.003
        vp = va.simplify(poly, 0.05)
        for i in range(len(vp)):
            lines.add_strip(vp[i - 1], vp[i], w, z)
        rect = va.min_area_rect(vp)
        if rect and abs(polygon_area(vp)) > 0.85 * rect[2] * rect[3] and sport not in ("running", "athletics"):
            P = va._local(rect[0], rect[1])
            a, b = P(0, -rect[3] / 2, 0), P(0, rect[3] / 2, 0)
            lines.add_strip(a[:2], b[:2], w, z)
            lines.add_disc(rect[0], max(0.1, 0.15 * min(rect[2], rect[3])) , z - 0.001, sides=16)
    _mesh_bodies(p, lines, (0.95, 0.95, 0.93))


# ---------------------------------------------------------------- buildings
PITCHED = ("gabled", "hipped", "pyramidal", "skillion")


class _BuildingGroups:
    """Collects building geometry from every building into a few merged meshes:
    textured walls per (style, colour), plain triangles per colour (roofs, gables, parapets...)."""

    def __init__(self, va):
        self.va = va
        self.textured = {}
        self.plain = {}

    def walls(self, style, rgb, mesh):
        v, n, uv, t = mesh
        if t:
            self.textured.setdefault((style, tuple(round(c, 3) for c in rgb)),
                                     self.va.MeshBuilder()).add(v, t, normals=n, uvs=uv)

    def tris(self, rgb, tris, **kw):
        self.va.add_faceted(self.plain.setdefault(tuple(round(c, 3) for c in rgb),
                                                  self.va.MeshBuilder()), tris, **kw)

    def flush(self, p):
        for (style, rgb), mb in self.textured.items():
            try:
                tex = p.loadTexture(self.va.facade_texture(style, rgb))
                for v, n, uv, i in mb.meshes():
                    vis = p.createVisualShape(p.GEOM_MESH, vertices=v, indices=i, normals=n, uvs=uv,
                                              rgbaColor=[1, 1, 1, 1], specularColor=LOW_SPECULAR)
                    body = p.createMultiBody(0, -1, vis, [0, 0, 0])
                    p.changeVisualShape(body, -1, textureUniqueId=tex)
            except Exception as exc:                               # plain wall colour instead
                print("  wall texture not applied:", exc)
                _mesh_bodies(p, mb, rgb)
        for rgb, mb in self.plain.items():
            _mesh_bodies(p, mb, rgb)


def _building_colours(va, info, poly, shape):
    style = info.get("style") or "stucco"
    x, y = poly[0]
    wall = va.parse_colour(info.get("wall"))
    if wall is None:
        wall = (va.pick(va.SIDING_VARIANTS, x, y) if style == "siding" else
                va.pick(va.BRICK_VARIANTS, x, y) if style == "brick" else va.STYLE_WALL[style])
    roof = va.parse_colour(info.get("roof"))
    if roof is None:
        roof = (va.pick(va.HOUSE_ROOF_VARIANTS, y, x) if shape in PITCHED else
                (0.47, 0.63, 0.57) if shape == "dome" else va.STYLE_ROOF[style])   # copper patina
    return style, wall, roof


def _solid_visual(va, groups, poly, z0, z1, style, wall, roof, shape, storey_h, scale,
                  roof_height=None, parapet=True):
    """Draw one building (or building part) whose collision volume is poly x [z0, z1].
    Everything drawn stays inside that volume: walls, roof shape, parapet, rooftop units."""
    h = z1 - z0
    rect = va.min_area_rect(poly)
    # only truly rectangular outlines get gables/hips drawn on their rectangle, so the visual never
    # sticks out past the real (collision) footprint by more than a sliver
    rect_like = (rect is not None and abs(polygon_area(poly)) >= 0.95 * rect[2] * rect[3]
                 and all(distance_to_polygon(c, poly) <= 0.15 for c in va.rect_corners(*rect)))
    centroid = (sum(q[0] for q in poly) / len(poly), sum(q[1] for q in poly) / len(poly))
    if shape in PITCHED and not rect_like:
        convex = len(va.convex_hull(poly)) == len(poly)
        shape = "pyramidal" if convex and point_in_polygon(centroid, poly) and len(poly) <= 8 else "flat"

    if shape in PITCHED:
        span = rect[3] if rect_like else 2 * min(_edge_distance(centroid, poly), rect[3] / 2)
        rh = roof_height * scale if roof_height else 0.32 * span
        rh = max(0.1, min(rh, 0.5 * h))
        eave = z1 - rh
        footprint = va.rect_corners(*rect) if rect_like else poly
        groups.walls(style, wall, va.wall_band(footprint, z0, eave, storey_h, storey_h))
        if rect_like:
            roof_tris, gables = va.pitched_roof(rect, eave, z1, shape, overhang=min(0.35 * scale, 0.1 * span))
        else:
            roof_tris, gables = va.pyramid_on_polygon(va.ensure_ccw(poly), eave, z1), []
        inside = (rect[0][0], rect[0][1], eave) if rect_like else (centroid[0], centroid[1], eave)
        groups.tris(roof, roof_tris, outward_from=inside)
        groups.tris(tuple(c * 0.93 for c in wall), gables, outward_from=inside)
        return

    if shape == "dome":
        inr = _edge_distance(centroid, poly) if point_in_polygon(centroid, poly) else 0
        dh = min(0.45 * h, 0.9 * inr)
        if dh > 0.15:
            groups.walls(style, wall, va.wall_band(poly, z0, z1 - dh, storey_h, storey_h))
            groups.tris(roof, va.dome(centroid, inr * 0.95, z1 - dh, dh), outward_from=(*centroid, z1 - dh))
            deck = va.ensure_ccw(poly)
            tris = triangulate(deck)
            groups.tris(tuple(c * 0.9 for c in roof),
                        [tuple((deck[i][0], deck[i][1], z1 - dh) for i in t) for t in tris], up=True)
            return
        shape = "flat"

    # flat roof: walls to the top, deck a little lower behind a parapet, rooftop units on big roofs
    groups.walls(style, wall, va.wall_band(poly, z0, z1, storey_h, storey_h))
    pp = min(0.6, 0.1 * h, max(0.15, 0.8 * scale)) if parapet else 0.0
    deck_z = z1 - pp
    deck = va.ensure_ccw(poly)
    tris = triangulate(deck)
    groups.tris(roof, [tuple((deck[i][0], deck[i][1], deck_z + 0.002) for i in t) for t in tris], up=True)
    if pp > 0:
        groups.tris(tuple(c * 0.82 for c in wall), va.inner_parapet_tris(deck, deck_z, z1))
        area = abs(polygon_area(poly))
        if rect is not None and area > 60 * scale * scale:
            rng = random.Random(int(abs(centroid[0] * 1000 + centroid[1] * 7)))
            for _ in range(rng.randint(1, 3)):
                bw = rng.uniform(1.2, 2.6) * scale
                bl = bw * rng.uniform(1.0, 1.8)
                P = va._local(rect[0], rect[1])
                bx, by, _ = P(rng.uniform(-0.3, 0.3) * rect[2], rng.uniform(-0.3, 0.3) * rect[3], 0)
                corners = va.rect_corners((bx, by), rect[1], bl, bw)
                if all(point_in_polygon(c, poly) and _edge_distance(c, poly) > 0.3 * scale for c in corners):
                    top = deck_z + min(pp * 0.95, 1.4 * scale)
                    groups.tris((0.70, 0.71, 0.72), va.box_tris((bx, by), rect[1], bl, bw, deck_z, top),
                                outward_from=(bx, by, (deck_z + top) / 2))


def _invisible_solid(p, col, poly):
    """Building collision body whose own visual is a speck hidden inside the building.
    (A body with no visual shape would make PyBullet draw the bare collision mesh.)"""
    cx = sum(q[0] for q in poly) / len(poly)
    cy = sum(q[1] for q in poly) / len(poly)
    vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.001, rgbaColor=[0, 0, 0, 0],
                              visualFramePosition=[cx, cy, 0.5])
    return p.createMultiBody(0, col, vis, [0, 0, 0])


def _detailed_buildings(p, va, scene, solid_ids):
    """Same solid collision mesh per building as before (what the LIDAR and A* see), drawn with
    textured walls, real roof shapes, parapets, rooftop units, and building:part massing."""
    s = scene.scale
    # building:part pieces belong to the building that contains their centre
    parts_of = {}
    for part_poly, pinfo in scene.parts:
        c = (sum(q[0] for q in part_poly) / len(part_poly), sum(q[1] for q in part_poly) / len(part_poly))
        for bi, (poly, *_rest) in enumerate(scene.buildings):
            if point_in_polygon(c, poly):
                parts_of.setdefault(bi, []).append((part_poly, pinfo))
                break

    groups = _BuildingGroups(va)
    plain_needed = []
    for bi, (poly, h, info) in enumerate(scene.buildings):
        col, mesh = _building_collision(p, poly, h)
        if col is None:
            continue
        try:
            vpoly, tol = poly, 0.02           # visual outline: simplify very detailed footprints
            while 4 * len(vpoly) > va.MAX_VERTS and tol < 2.0:
                vpoly, tol = va.simplify(poly, tol), tol * 2
            shape = info.get("roof_shape", "flat")
            style, wall, roof = _building_colours(va, info, vpoly, shape)
            parts = parts_of.get(bi)
            if parts:
                top = max(pi["top"] for _, pi in parts)
                k = h / max(top, 1e-6)        # tallest part reaches the building's (clamped) height
                for part_poly, pi in parts:
                    pstyle = pi.get("style") or style
                    pwall = va.parse_colour(pi.get("wall")) or wall
                    proof = va.parse_colour(pi.get("roof")) or roof
                    z0, z1 = pi["bottom"] * k, pi["top"] * k
                    _solid_visual(va, groups, va.simplify(part_poly, 0.02), z0, z1, pstyle, pwall,
                                  proof, pi["roof_shape"], BUILDING_LEVEL_HEIGHT * k, s, parapet=True)
                    if z0 > 0.05:             # overhanging part: draw its underside
                        under = va.ensure_ccw(part_poly)
                        groups.tris(tuple(c * 0.6 for c in pwall),
                                    [tuple((under[i][0], under[i][1], z0) for i in t)
                                     for t in triangulate(under)], up=True)
            else:
                storey_h = h / max(1, info.get("storeys", 1))
                _solid_visual(va, groups, vpoly, 0.0, h, style, wall, roof, shape, storey_h, s,
                              roof_height=info.get("roof_height"))
            solid_ids.append(_invisible_solid(p, col, poly))
        except Exception as exc:
            print("  building drawn plain:", exc)
            plain_needed.append((col, mesh))
    try:
        groups.flush(p)
    except Exception as exc:
        print("  detailed building visuals failed, drawing buildings plain:", exc)
    for col, mesh in plain_needed:
        vis = p.createVisualShape(p.GEOM_MESH, vertices=mesh[0], indices=mesh[1],
                                  rgbaColor=[0.62, 0.58, 0.55, 1])
        solid_ids.append(p.createMultiBody(0, col, vis, [0, 0, 0]))


def build_bodies(scene, detailed=True, avoid=()):
    """Create PyBullet bodies for buildings (solid) and ground features (visual only).
    Trees are NOT built here: env_sim.build_world builds them from OBSTACLE_POSITIONS.
    detailed=False gives the original flat-colour look. avoid: (x, y) points to keep props
    (parked cars) away from, e.g. the missing person. Returns the solid body ids.
    Collision shapes are identical in both modes, so sensing and navigation do not change."""
    import pybullet as p
    if detailed:
        try:
            return _build_detailed(p, scene, avoid)
        except ImportError as exc:                     # visual_assets.py missing / numpy missing
            print("Detailed OSM visuals unavailable, using the simple look:", exc)
    return _build_simple(p, scene)
