"""
File Description: Turns OpenStreetMap (OSM) data around a last known position (LKP) into
obstacles for the existing PyBullet world, WITHOUT changing the size of the simulation map.

The simulation map stays exactly as it is: [-MAP_HALF, MAP_HALF] in x and y (26 m x 26 m),
which is what the obstacle map (object_avoidance.py), the search zones (operator_guidance.py)
and the probability map already assume. The LKP is placed at the centre (0, 0).

  area_m  = how many REAL metres of OSM data to show across the map.
  scale   = (2 * MAP_HALF) / area_m   (sim metres per real metre)

  area_m = 26  -> scale 1.0  : true size, but only a small patch around the LKP
  area_m = 200 -> scale 0.13 : a whole block squeezed into the same 26 m map

What gets converted:
  buildings (building=*)                 -> extruded, solid footprint meshes (drone senses/avoids them)
  single trees (natural=tree)            -> the existing trunk + canopy trees
  woods / forests (natural=wood, ...)    -> trees scattered inside the polygon
  roads / paths (highway=*)              -> flat grey strips (visual only, no collision)
  water (natural=water, waterway=*)      -> flat blue surface (visual only, no collision)

Data sources (pick one):
  1. Live: lat/lon -> Overpass API (cached to osm_cache/ so repeat runs work offline)
  2. File: an Overpass JSON file, or a .osm XML file from openstreetmap.org -> Export
"""
import json
import math
import os
import random
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

MAP_HALF = 13.0                 # Must match MAP_HALF in object_avoidance.py.
DEFAULT_AREA_M = 2 * MAP_HALF   # Default: true scale (1 sim metre = 1 real metre).

# Places the drone has to reach (start + zone centres from operator_guidance.py) are kept clear,
# otherwise the drone could spawn inside a building or be sent to a goal it can never reach.
KEEP_CLEAR_RADIUS = 1.6         # sim metres
TREE_CLEAR_RADIUS = 1.2         # extra gap between a tree trunk and a kept-clear point

BUILDING_LEVEL_HEIGHT = 3.0     # real metres per storey, when OSM gives building:levels
DEFAULT_BUILDING_HEIGHT = 6.0   # real metres, when OSM gives no height at all
MIN_BUILDING_HEIGHT = 3.5       # sim metres: always taller than cruise altitude (2.5 m), so it is a real obstacle
MAX_BUILDING_HEIGHT = 8.0       # sim metres: taller than MAX_ALT (5.5 m), so the drone routes around, not over

WOOD_TREE_SPACING = 3.0         # sim metres between scattered trees in a wood polygon
MAX_TREES = 40                  # each tree = 2 bodies + ray hits; caps cost in dense forests
MIN_TREE_SPACING = 1.0          # sim metres; closer trees are merged

ROAD_WIDTH_M = {"motorway": 12, "trunk": 10, "primary": 9, "secondary": 8, "tertiary": 7,
                "residential": 6, "service": 4, "unclassified": 5}
DEFAULT_ROAD_WIDTH_M = 2.0      # footways, paths, tracks
MIN_ROAD_WIDTH = 0.3            # sim metres, so paths stay visible at small scales

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "osm_cache")


# ------------------------------------------------------------------ loading OSM data
def fetch_overpass(lat, lon, area_m, timeout=60):
    """Download OSM features around (lat, lon) from Overpass, cached as JSON."""
    radius = math.ceil(area_m / 2 * math.sqrt(2)) + 5      # circle that covers the whole square
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache = os.path.join(CACHE_DIR, f"{lat:.6f}_{lon:.6f}_{radius}.json")
    if os.path.exists(cache):
        with open(cache) as f:
            return json.load(f)
    around = f"(around:{radius},{lat},{lon})"
    query = f"""[out:json][timeout:{timeout}];
(
  way["building"]{around};
  node["natural"="tree"]{around};
  way["natural"~"wood|scrub"]{around};
  way["landuse"="forest"]{around};
  way["highway"]{around};
  way["natural"="water"]{around};
  way["waterway"]{around};
);
out geom;"""
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(OVERPASS_URL, data=data,
                                 headers={"User-Agent": "SM2DH2C-drone-sim/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = json.load(resp)
    with open(cache, "w") as f:
        json.dump(result, f)
    return result


def load_osm_file(path):
    """Read an Overpass JSON file or a .osm XML export into the same element format."""
    if path.lower().endswith(".json"):
        with open(path) as f:
            return json.load(f)
    root = ET.parse(path).getroot()
    nodes, elements = {}, []
    for n in root.iter("node"):
        nodes[n.get("id")] = (float(n.get("lat")), float(n.get("lon")))
        tags = {t.get("k"): t.get("v") for t in n.findall("tag")}
        if tags:
            elements.append({"type": "node", "lat": float(n.get("lat")),
                             "lon": float(n.get("lon")), "tags": tags})
    for w in root.iter("way"):
        geom = [nodes[nd.get("ref")] for nd in w.findall("nd") if nd.get("ref") in nodes]
        elements.append({"type": "way", "tags": {t.get("k"): t.get("v") for t in w.findall("tag")},
                         "geometry": [{"lat": a, "lon": b} for a, b in geom]})
    return {"elements": elements}


# ------------------------------------------------------------------ geometry helpers
def make_projector(lat0, lon0, scale):
    """lat/lon -> sim (x, y). x = east, y = north, LKP at (0, 0). Accurate to well under
    1% over a few km, which is far more than this map ever shows."""
    m_per_deg_lat = 111_132.0
    m_per_deg_lon = 111_320.0 * math.cos(math.radians(lat0))

    def to_xy(lat, lon):
        return ((lon - lon0) * m_per_deg_lon * scale, (lat - lat0) * m_per_deg_lat * scale)
    return to_xy


def clip_polygon(poly, half=MAP_HALF):
    """Sutherland-Hodgman clip of a polygon to the square map."""
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


def clip_segment(a, b, half=MAP_HALF):
    """Liang-Barsky clip of a line segment to the square map. Returns None if fully outside."""
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


# ------------------------------------------------------------------ the parsed scene
class OSMScene:
    """Everything parsed from OSM, already in sim coordinates and clipped to the map."""

    def __init__(self):
        self.buildings = []     # [(polygon [(x, y), ...], height)]
        self.trees = []         # [(x, y)]
        self.roads = []         # [((x1, y1), (x2, y2), width)]
        self.water = []         # [polygon]
        self.dropped = []       # human-readable notes about features removed to keep the mission valid

    def summary(self):
        return (f"{len(self.buildings)} buildings, {len(self.trees)} trees, "
                f"{len(self.roads)} road segments, {len(self.water)} water areas")


def parse_scene(osm_json, lat, lon, area_m=DEFAULT_AREA_M, keep_clear=(), seed=0):
    """Convert OSM elements into an OSMScene that fits the fixed-size map.
    keep_clear: (x, y) points (drone start, zone centres) that must stay reachable."""
    scale = (2 * MAP_HALF) / area_m
    to_xy = make_projector(lat, lon, scale)
    scene = OSMScene()
    woods = []
    rng = random.Random(seed)

    for el in osm_json.get("elements", []):
        tags = el.get("tags", {})
        if el["type"] == "node":
            if tags.get("natural") == "tree":
                x, y = to_xy(el["lat"], el["lon"])
                if abs(x) <= MAP_HALF and abs(y) <= MAP_HALF:
                    scene.trees.append((x, y))
            continue
        geom = el.get("geometry") or []
        if len(geom) < 2:
            continue
        pts = ring_to_xy(geom, to_xy)

        if "building" in tags and len(pts) >= 3:
            poly = clip_polygon(pts)
            if len(poly) >= 3 and abs(polygon_area(poly)) > 0.05:
                scene.buildings.append((poly, building_height(tags, scale)))
        elif (tags.get("natural") in ("wood", "scrub") or tags.get("landuse") == "forest") and len(pts) >= 3:
            woods.append(pts)
        elif tags.get("natural") == "water" and len(pts) >= 3:
            poly = clip_polygon(pts)
            if len(poly) >= 3:
                scene.water.append(poly)
        elif "highway" in tags or "waterway" in tags:
            if "highway" in tags:
                w = ROAD_WIDTH_M.get(tags["highway"], DEFAULT_ROAD_WIDTH_M)
            else:
                w = 3.0 if tags["waterway"] in ("river", "canal") else 1.0
            width = max(MIN_ROAD_WIDTH, w * scale)
            for a, b in zip(pts, pts[1:]):
                seg = clip_segment(a, b)
                if seg and math.hypot(seg[1][0] - seg[0][0], seg[1][1] - seg[0][1]) > 1e-3:
                    scene.roads.append((seg[0], seg[1], width))

    # Scatter trees inside wood/forest polygons on a jittered grid (deterministic per seed).
    s = WOOD_TREE_SPACING
    for poly in woods:
        xs, ys = [q[0] for q in poly], [q[1] for q in poly]
        x = max(min(xs), -MAP_HALF)
        while x <= min(max(xs), MAP_HALF):
            y = max(min(ys), -MAP_HALF)
            while y <= min(max(ys), MAP_HALF):
                q = (x + rng.uniform(-0.3, 0.3) * s, y + rng.uniform(-0.3, 0.3) * s)
                if abs(q[0]) <= MAP_HALF and abs(q[1]) <= MAP_HALF and point_in_polygon(q, poly):
                    scene.trees.append(q)
                y += s
            x += s

    # Trees must not sit inside buildings, and very close trees are merged.
    merged = []
    for t in scene.trees:
        if any(point_in_polygon(t, b) for b, _ in scene.buildings):
            continue
        if all(math.hypot(t[0] - m[0], t[1] - m[1]) >= MIN_TREE_SPACING for m in merged):
            merged.append(t)
    if len(merged) > MAX_TREES:
        scene.dropped.append(f"{len(merged) - MAX_TREES} trees over the MAX_TREES={MAX_TREES} cap")
        merged.sort(key=lambda t: math.hypot(*t))        # keep the ones nearest the LKP
        merged = merged[:MAX_TREES]
    scene.trees = merged

    # Keep the drone start and every search-zone centre reachable.
    for c in keep_clear:
        kept = []
        for poly, h in scene.buildings:
            if distance_to_polygon(c, poly) < KEEP_CLEAR_RADIUS:
                scene.dropped.append(f"building covering ({c[0]:.0f}, {c[1]:.0f})")
            else:
                kept.append((poly, h))
        scene.buildings = kept
        before = len(scene.trees)
        scene.trees = [t for t in scene.trees
                       if math.hypot(t[0] - c[0], t[1] - c[1]) >= KEEP_CLEAR_RADIUS + TREE_CLEAR_RADIUS]
        if len(scene.trees) < before:
            scene.dropped.append(f"{before - len(scene.trees)} tree(s) near ({c[0]:.0f}, {c[1]:.0f})")
    return scene


def is_free(scene, xy, margin=1.0):
    """True if xy is outside every building (by margin) and away from every tree trunk."""
    return (all(distance_to_polygon(xy, b) >= margin for b, _ in scene.buildings)
            and all(math.hypot(xy[0] - t[0], xy[1] - t[1]) >= margin for t in scene.trees))


def nearest_free_point(scene, xy, margin=1.0, step=0.5, max_r=MAP_HALF - 1):
    """Spiral outward from xy until a free spot inside the map is found."""
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
def build_bodies(scene):
    """Create PyBullet bodies for buildings (solid), roads and water (visual only).
    Trees are NOT built here: env_sim.build_world builds them from OBSTACLE_POSITIONS
    exactly like the original hand-placed trees. Returns the solid body ids."""
    import pybullet as p

    solid_ids = []
    for poly, h in scene.buildings:
        tris = triangulate(poly)
        if not tris:
            continue
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
        vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx,
                                  rgbaColor=[0.62, 0.58, 0.55, 1])
        solid_ids.append(p.createMultiBody(0, col, vis, [0, 0, 0]))

    for a, b, width in scene.roads:
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
