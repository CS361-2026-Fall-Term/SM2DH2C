"""Geographic/legacy PyBullet world builder for the SM2DH2C SAR prototype.

Contributors to original env_sim.py: Mari Tokuda and Heeseo Kim.
Geographic extension: generated with assistance from ChatGPT (review before merging).

Usage (put this file beside custom_drone.urdf):
    pip install pybullet osmnx shapely pyproj
    python env_sim.py --lat 44.5646 --lon -123.2620 --radius 50
    python env_sim.py --lat 44.5646 --lon -123.2620 --radius 50 --no-gui
    python env_sim.py --lat 44.5646 --lon -123.2620 --geojson region.geojson
    python env_sim.py --prompt
    python env_sim.py --demo

From existing main.py (which calls build_world(gui) without coordinates):
    Set SAR_LAT, SAR_LON, SAR_RADIUS_M environment variables, then run main.py.
    Example: SAR_LAT=44.5646 SAR_LON=-123.2620 SAR_RADIUS_M=10 python main.py
    The current navigator is limited to +/- 13 m and operator zones are hardcoded;
    environments larger than that require changes to those OTHER modules.

This module creates a FLAT 3D world with OSM-based buildings, roads, mapped
individual trees, approximate woodland trees, and water/land-use surfaces.
It does NOT infer actual tree locations within forests, create elevation from
OpenStreetMap, or guarantee complete map coverage. For hills, add a separately
sourced DEM/heightfield and adapt navigation/target altitudes.

The geographic scene is generated directly from OSMnx + Shapely, without
requiring Blender, Unity, Java, or OSM2World. OSM data: (c) OpenStreetMap
contributors, available under the ODbL. https://www.openstreetmap.org/copyright
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Iterable

import pybullet as p
import pybullet_data


# Mutable lists are intentional: other prototype files use
#     from env_sim import DRONE_START_POS, OBSTACLE_POSITIONS, ...
# Mutating in-place keeps those imported references synchronized after build_world.
DRONE_START_POS = [-10.0, 10.0, 3.5]
DRONE_SCALE = 1.5
OBSTACLE_POSITIONS = [
    [1.5, 6.5, 0.5], [-4.0, -1.5, 0.5], [8.0, -2.0, 0.5],
    [-7.5, 5.0, 0.5], [-8.0, -8.0, 0.5], [9.0, 8.5, 0.5],
    [-1.0, -7.5, 0.5], [0.0, 1.0, 0.5],
]
TARGET_POSITION = [7.0, -5.5, 0.875]
TARGET_ORIENTATION_EULER = [1.5708, 0, 0]
TARGET_SCALE = 0.25
FLIGHT_PATH = [
    [-10, 10, 2.5], [-5, 10, 2.5], [-5, 5, 2.5], [-5, 0, 2.5],
    [0, 0, 2.5], [5, 0, 2.5], [5, 5, 2.5], [10, 5, 2.5],
    [10, -5, 2.5], [5, -5, 2.5], [6, -5, 2.5],
]

# Summary available after any call to build_world().
LAST_WORLD_INFO: dict[str, Any] = {}

# Snapshot of original positions. Re-entering legacy mode after geographic
# mode must restore the original scene instead of reusing generated tree points.
_ORIGINAL_START = DRONE_START_POS.copy()
_ORIGINAL_TREES = [point.copy() for point in OBSTACLE_POSITIONS]
_ORIGINAL_TARGET = TARGET_POSITION.copy()
_ORIGINAL_PATH = [point.copy() for point in FLIGHT_PATH]

OSM_TAGS = {
    "building": True,
    "highway": True,
    "natural": ["tree", "wood", "water", "scrub"],
    "landuse": ["forest", "grass", "meadow"],
    "leisure": ["park"],
    "waterway": ["riverbank"],
}
BUILDING_COLOR = [0.66, 0.68, 0.72, 1]
ROAD_COLOR = [0.34, 0.36, 0.38, 1]
PATH_COLOR = [0.65, 0.58, 0.47, 1]
WATER_COLOR = [0.18, 0.44, 0.72, 1]
GREEN_COLOR = [0.36, 0.55, 0.31, 1]
FOREST_COLOR = [0.20, 0.41, 0.23, 1]
GROUND_COLOR = [0.54, 0.65, 0.47, 1]


def _tag(properties: dict, key: str) -> str:
    """Cope with pandas missing values and non-string OSM tag values."""
    value = properties.get(key)
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return str(value)
    return ""


def _building_height(props: dict) -> float:
    """OSM height (m, sometimes '12 m'/'40 ft') or building:levels fallback."""
    pre = props.get("height_m")
    if isinstance(pre, (int, float)) and math.isfinite(float(pre)):
        return float(pre)  # already parsed by osm_parser.py
    value = _tag(props, "height")
    match = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(m|meters?|ft|feet|')?\s*$", value, re.I)
    if match:
        height = float(match.group(1))
        if match.group(2) and match.group(2).lower() in {"ft", "feet", "'"}:
            height *= 0.3048
        return max(2.0, min(height, 100.0))
    levels = _tag(props, "building:levels")
    try:
        n = float(levels)
        if 0 < n < 80:
            return min(100.0, max(2.5, 3.0 * n))
    except ValueError:
        pass
    return 9.0


def _polygon_parts(geometry):
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection
    if isinstance(geometry, Polygon):
        if not geometry.is_empty and geometry.area > 0.2:
            yield geometry
    elif isinstance(geometry, (MultiPolygon, GeometryCollection)):
        for part in geometry.geoms:
            yield from _polygon_parts(part)


def _line_parts(geometry):
    from shapely.geometry import LineString, MultiLineString, GeometryCollection
    if isinstance(geometry, LineString):
        if geometry.length > 0.05:
            yield geometry
    elif isinstance(geometry, (MultiLineString, GeometryCollection)):
        for part in geometry.geoms:
            yield from _line_parts(part)


def _point_parts(geometry):
    from shapely.geometry import Point, MultiPoint, GeometryCollection
    if isinstance(geometry, Point):
        yield geometry
    elif isinstance(geometry, (MultiPoint, GeometryCollection)):
        for part in geometry.geoms:
            yield from _point_parts(part)


def _triangles(poly):
    """Triangulate a footprint, respecting holes and concave boundaries.

    Shapely 2.1+ has constrained_delaunay_triangles. Older versions use
    unconstrained Delaunay with interior filtering (less exact for concave lots).
    """
    from shapely.ops import triangulate
    try:
        from shapely import constrained_delaunay_triangles
        return [t for t in constrained_delaunay_triangles(poly).geoms
                if t.area > 1e-6 and poly.covers(t.representative_point())]
    except (ImportError, AttributeError, NotImplementedError):
        return [t for t in triangulate(poly) if poly.covers(t) and t.area > 1e-6]


def _building_mesh(polygon, height: float):
    """Closed triangle-mesh building: roof, base, exterior and courtyard walls."""
    vertices: list[list[float]] = []
    indices: list[int] = []

    def tri(a, b, c, two_sided=False):
        start = len(vertices)
        vertices.extend([list(a), list(b), list(c)])
        indices.extend([start, start + 1, start + 2])
        if two_sided:
            indices.extend([start, start + 2, start + 1])

    for roof_tri in _triangles(polygon):
        a, b, c = list(roof_tri.exterior.coords)[:3]
        tri((a[0], a[1], height), (b[0], b[1], height),
            (c[0], c[1], height), two_sided=True)
        tri((a[0], a[1], 0), (c[0], c[1], 0),
            (b[0], b[1], 0), two_sided=True)

    for ring in [polygon.exterior, *polygon.interiors]:
        xy = list(ring.coords)
        for a, b in zip(xy, xy[1:]):
            if math.dist(a[:2], b[:2]) < 1e-5:
                continue
            lo_a, lo_b = (a[0], a[1], 0), (b[0], b[1], 0)
            hi_a, hi_b = (a[0], a[1], height), (b[0], b[1], height)
            tri(lo_a, lo_b, hi_b, two_sided=True)
            tri(lo_a, hi_b, hi_a, two_sided=True)
    return vertices, indices


def _create_building(poly, height: float):
    """Buildings are STATIC concave collision meshes so raycasts hit walls."""
    vertices, indices = _building_mesh(poly, height)
    if not indices:
        return None
    collision = p.createCollisionShape(
        p.GEOM_MESH, vertices=vertices, indices=indices,
        flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
    )
    visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices, indices=indices,
                                 rgbaColor=BUILDING_COLOR)
    if collision < 0 or visual < 0:
        raise RuntimeError("PyBullet rejected the building mesh. Check mesh compatibility.")
    return p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision,
                             baseVisualShapeIndex=visual)


def _paint_polygon(poly, color, z=0.025):
    """Paint a flat mapped land-use/water polygon, with NO collision."""
    vertices = []
    indices = []
    for triangle in _triangles(poly):
        coords = list(triangle.exterior.coords)[:3]
        base = len(vertices)
        vertices.extend([[xy[0], xy[1], z] for xy in coords])
        indices.extend([base, base + 1, base + 2, base, base + 2, base + 1])
    if not indices:
        return None
    visual = p.createVisualShape(p.GEOM_MESH, vertices=vertices,
                                 indices=indices, rgbaColor=color)
    return p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual)


def _draw_roads(line, props, max_segments: int) -> int:
    """Represent mapped road centerlines as thin, rotated visual-only boxes."""
    highway = _tag(props, "highway").lower()
    if highway in {"footway", "path", "cycleway", "pedestrian", "steps", "track"}:
        width, color = 1.7, PATH_COLOR
    elif highway in {"motorway", "trunk", "primary"}:
        width, color = 7.0, ROAD_COLOR
    elif highway in {"secondary", "tertiary"}:
        width, color = 5.0, ROAD_COLOR
    else:
        width, color = 3.5, ROAD_COLOR
    count = 0
    points = list(line.coords)
    for a, b in zip(points, points[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        if length < 0.1:
            continue
        if count >= max_segments:
            break
        visual = p.createVisualShape(p.GEOM_BOX,
                                     halfExtents=[length / 2, width / 2, 0.012],
                                     rgbaColor=color)
        p.createMultiBody(baseMass=0, baseVisualShapeIndex=visual,
                          basePosition=[(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, 0.038],
                          baseOrientation=p.getQuaternionFromEuler([0, 0, math.atan2(dy, dx)]))
        count += 1
    return count


def _create_tree(x: float, y: float, size: float = 1.0):
    """Return collision-capable trunk and canopy body IDs."""
    trunk_h = 2.8 * size
    trunk_r = 0.16 * size
    crown_r = 0.82 * size
    tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=trunk_r, height=trunk_h)
    tv = p.createVisualShape(p.GEOM_CYLINDER, radius=trunk_r, length=trunk_h,
                             rgbaColor=[0.39, 0.22, 0.10, 1])
    trunk_id = p.createMultiBody(0, tc, tv, [x, y, trunk_h / 2])
    cc = p.createCollisionShape(p.GEOM_SPHERE, radius=crown_r)
    cv = p.createVisualShape(p.GEOM_SPHERE, radius=crown_r,
                             rgbaColor=[0.10, 0.39, 0.13, 1])
    canopy_id = p.createMultiBody(0, cc, cv, [x, y, trunk_h + crown_r * 0.22])
    return trunk_id, canopy_id


def _load_features(lat: float, lon: float, radius_m: float, geojson_path: str | None,
                   json_in: str | None = None, json_out: str | None = "environment.json"):
    """Return [(shapely geometry in local meters, props dict), ...].

    All geographic work (download -> reproject -> clip -> classify -> JSON) is done
    with GeoPandas in osm_parser.py. Sources, in priority order:
      json_in      previously exported environment JSON (offline, no re-download)
      geojson_path offline WGS84 GeoJSON FeatureCollection
      otherwise    live OpenStreetMap download around (lat, lon)
    """
    import osm_parser
    if json_in:
        env, meta = osm_parser.load_json(json_in)
        print(f"[env_sim] Loaded environment JSON {json_in} ({len(env)} features).", flush=True)
    else:
        env = osm_parser.build_environment(lat, lon, radius_m,
                                           geojson_path=geojson_path, json_path=json_out)
    props = env.drop(columns="geometry").to_dict("records")
    return list(zip(env.geometry, props))


def _choose_position(building_footprints, tree_xy, radius_m, other_xy=None):
    """Pick a safe XY inside the scene without placing a drone/person in walls."""
    from shapely.geometry import Point
    search_r = max(2.0, radius_m - 1.0)
    candidates = [(0.0, 0.0)]
    for r in [3, 6, 9, 12, 18, 25, 35, 50, 75, 100, 150]:
        if r > search_r:
            continue
        for k in range(16):
            angle = math.tau * (k + 0.5) / 16
            candidates.append((r * math.cos(angle), r * math.sin(angle)))
    for x, y in candidates:
        if other_xy is not None and math.dist([x, y], other_xy) < min(8.0, radius_m * 0.25):
            continue
        pt = Point(x, y)
        if any(poly.distance(pt) < 1.5 for poly in building_footprints):
            continue
        if any(math.hypot(x - tx, y - ty) < 1.4 for tx, ty in tree_xy):
            continue
        return x, y
    raise RuntimeError("No free spawn location found. Try a larger radius or a nearby LKP.")


def _update_legacy_positions(start_xy, person_xy, tree_xy, radius_m):
    """Keep imported mutable constants in sync for existing project modules."""
    sx, sy = start_xy
    tx, ty = person_xy
    DRONE_START_POS[:] = [sx, sy, 3.5]
    TARGET_POSITION[:] = [tx, ty, 0.875]
    # Legacy tree positions include an unused z=0.5 expected by old helpers.
    OBSTACLE_POSITIONS[:] = [[x, y, 0.5] for x, y in tree_xy]
    span = min(10.0, max(2.0, radius_m * 0.35))
    FLIGHT_PATH[:] = [
        [sx, sy, 2.5], [sx - span, sy, 2.5], [sx - span, sy - span, 2.5],
        [sx + span, sy - span, 2.5], [sx + span, sy + span, 2.5],
        [sx - span, sy + span, 2.5], [tx, ty, 2.5],
    ]


def _legacy_world(assets: str):
    """Original hand-built scene, used by default to avoid breaking main.py."""
    DRONE_START_POS[:] = _ORIGINAL_START
    TARGET_POSITION[:] = _ORIGINAL_TARGET
    OBSTACLE_POSITIONS[:] = [position.copy() for position in _ORIGINAL_TREES]
    FLIGHT_PATH[:] = [point.copy() for point in _ORIGINAL_PATH]
    floor_id = p.loadURDF("plane.urdf")
    texture = os.path.join(assets, "ground_texture.jpg")
    if os.path.isfile(texture):
        try:
            p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(texture))
        except Exception:
            pass
    obstacles = []
    # Match the original tree/canopy collision geometry and positions.
    for x, y, z in OBSTACLE_POSITIONS:
        tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.5)
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.5,
                                 rgbaColor=[0.4, 0.2, 0.05, 1])
        obstacles.append(p.createMultiBody(0, tc, tv, [x, y, z]))
        cc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.75)
        cv = p.createVisualShape(p.GEOM_SPHERE, radius=0.75,
                                 rgbaColor=[0.1, 0.45, 0.1, 1])
        obstacles.append(p.createMultiBody(0, cc, cv, [x, y, z + 1.75]))
    return floor_id, obstacles


def _geo_world(lat: float, lon: float, radius_m: float, geojson_path: str | None,
               max_buildings: int, max_road_segments: int, max_trees: int,
               json_in: str | None = None, json_out: str | None = "environment.json"):
    """Build a geographical scene; return (floor_id, obstacle_ids, stats)."""
    from shapely.geometry import Point

    print("[env_sim] Loading geographical features...", flush=True)
    features = _load_features(lat, lon, radius_m, geojson_path, json_in, json_out)
    print(f"[env_sim] Generating PyBullet geometry from {len(features)} features...", flush=True)
    # Truncate all geometry to a circle around LKP. OSMnx fetches bounding boxes.
    area = Point(0, 0).buffer(radius_m, resolution=48)
    floor_collision = p.createCollisionShape(p.GEOM_BOX,
                                             halfExtents=[radius_m + 2, radius_m + 2, 0.1])
    floor_visual = p.createVisualShape(p.GEOM_BOX,
                                       halfExtents=[radius_m + 2, radius_m + 2, 0.1],
                                       rgbaColor=GROUND_COLOR)
    floor_id = p.createMultiBody(0, floor_collision, floor_visual, [0, 0, -0.1])

    obstacles = []
    footprints = []
    forest_polygons = []
    tree_xy = []
    road_segments = 0
    paintings = 0
    building_count = 0
    painted_polygons = set()
    seen_tree_xy = set()
    rng = random.Random(271361)

    # Geographic features can have multiple tags; handle each feature type once.
    # Draw paint and roads first, then buildings and trees.
    for geometry, props in features:
        natural = _tag(props, "natural").lower()
        landuse = _tag(props, "landuse").lower()
        leisure = _tag(props, "leisure").lower()
        is_water = natural == "water" or _tag(props, "waterway") == "riverbank"
        is_forest = natural == "wood" or landuse == "forest"
        is_grass = landuse in {"grass", "meadow"} or leisure == "park"
        if not (is_water or is_forest or is_grass):
            continue
        for poly in _polygon_parts(geometry.intersection(area)):
            if paintings >= 100:
                break
            # Deduplicate overlapping tagging within the same feature pass.
            k = poly.wkb
            if k in painted_polygons:
                continue
            painted_polygons.add(k)
            color = WATER_COLOR if is_water else (FOREST_COLOR if is_forest else GREEN_COLOR)
            if _paint_polygon(poly, color, z=0.029 if is_water else 0.021) is not None:
                paintings += 1
            if is_forest:
                forest_polygons.append(poly)

    for geometry, props in features:
        if not _tag(props, "highway"):
            continue
        for line in _line_parts(geometry.intersection(area)):
            if road_segments >= max_road_segments:
                break
            road_segments += _draw_roads(line, props, max_road_segments - road_segments)

    for geometry, props in features:
        if not _tag(props, "building") or building_count >= max_buildings:
            continue
        for poly in _polygon_parts(geometry.intersection(area)):
            if building_count >= max_buildings:
                break
            if poly.area < 2.0:
                continue
            # Only add as a collision obstacle if it produced a valid mesh.
            body = _create_building(poly, _building_height(props))
            if body is not None:
                obstacles.append(body)
                footprints.append(poly)
                building_count += 1

    # OSM individual tree points are real mapped points, not a procedural estimate.
    for geometry, props in features:
        if _tag(props, "natural").lower() != "tree":
            continue
        for pt in _point_parts(geometry.intersection(area)):
            key = (round(pt.x, 1), round(pt.y, 1))
            if key in seen_tree_xy or len(tree_xy) >= max_trees:
                continue
            seen_tree_xy.add(key)
            if any(poly.buffer(1).covers(pt) for poly in footprints):
                continue
            tree_xy.append((float(pt.x), float(pt.y)))

    mapped_trees = len(tree_xy)
    # A forest polygon records an AREA, not precise tree coordinates. The points
    # below are synthetic and should never be reported as real mapped trees.
    for forest in forest_polygons:
        if len(tree_xy) >= max_trees:
            break
        desired = min(30, int(forest.area / 55.0))
        minx, miny, maxx, maxy = forest.bounds
        added = 0
        for _ in range(max(30, desired * 20)):
            if added >= desired or len(tree_xy) >= max_trees:
                break
            x, y = rng.uniform(minx, maxx), rng.uniform(miny, maxy)
            pt = Point(x, y)
            if not forest.contains(pt):
                continue
            if any(poly.buffer(1.1).covers(pt) for poly in footprints):
                continue
            if any(math.hypot(x - tx, y - ty) < 2.3 for tx, ty in tree_xy):
                continue
            tree_xy.append((x, y))
            added += 1

    for x, y in tree_xy:
        obstacles.extend(_create_tree(x, y, size=rng.uniform(0.8, 1.25)))

    print(f"[env_sim] Created {building_count} buildings, {road_segments} road segments, "
          f"{len(tree_xy)} trees, {paintings} land/water surfaces.", flush=True)
    if not building_count and not road_segments and not tree_xy and not paintings:
        print("[env_sim] WARNING: No displayable OSM objects found in this circle. "
              "Try a populated area or a larger radius (e.g., 80m).", flush=True)
    spawn_xy = _choose_position(footprints, tree_xy, radius_m)
    # Pick farthest available safe candidate for target, using deterministic order.
    person_xy = _choose_position(footprints, tree_xy, radius_m, other_xy=spawn_xy)
    _update_legacy_positions(spawn_xy, person_xy, tree_xy, radius_m)
    stats = {
        "mode": "geographic", "latitude": lat, "longitude": lon,
        "radius_m": radius_m, "features_received": len(features),
        "buildings": building_count, "road_segments": road_segments,
        "land_water_polygons": paintings,
        "mapped_trees": mapped_trees,
        "approximate_forest_trees": len(tree_xy) - mapped_trees,
        "drone_start_xyz": tuple(DRONE_START_POS),
        "target_xyz": tuple(TARGET_POSITION),
        "collision_obstacles": len(obstacles),
    }
    return floor_id, obstacles, stats


def build_world(gui=True, assets=None, *, lat=None, lon=None, radius_m=50.0,
                geojson_path=None, max_buildings=150, max_road_segments=450,
                max_trees=110, json_in=None, json_out="environment.json"):
    """Create PyBullet scene + drone; humanoid spawned by target_detection.py.

    Returns exactly (floor_id, drone_id, obstacle_ids), preserving main.py.

    Coordinate-based mode:
        build_world(gui=True, lat=44.5646, lon=-123.262, radius_m=50)
    Offline mode (WGS84 GeoJSON):
        build_world(gui=True, lat=44.5646, lon=-123.262,
                    geojson_path='area.geojson')
    Legacy mode:
        build_world(gui=True)

    Environment variable route (without changing main.py):
        SAR_LAT, SAR_LON, SAR_RADIUS_M, optional SAR_GEOJSON
    """
    assets = str(Path(assets or Path(__file__).resolve().parent).resolve())
    if lat is None and os.getenv("SAR_LAT"):
        lat = float(os.environ["SAR_LAT"])
    if lon is None and os.getenv("SAR_LON"):
        lon = float(os.environ["SAR_LON"])
    if os.getenv("SAR_RADIUS_M") and radius_m == 50.0:
        radius_m = float(os.environ["SAR_RADIUS_M"])
    if geojson_path is None:
        geojson_path = os.getenv("SAR_GEOJSON") or None
    if json_in is None:
        json_in = os.getenv("SAR_JSON_IN") or None
    if (lat is None) != (lon is None):
        raise ValueError("Provide BOTH latitude and longitude, not only one.")
    if json_in and lat is None:
        import osm_parser
        meta = osm_parser.load_json(json_in)[1]
        lat, lon, radius_m = meta["lkp"]["lat"], meta["lkp"]["lon"], meta["radius_m"]
    if geojson_path and lat is None:
        raise ValueError("With --geojson, also provide LKP --lat and --lon.")
    if lat is not None and (not -90 <= lat <= 90 or not -180 <= lon <= 180):
        raise ValueError("Latitude must be [-90,90] and longitude [-180,180].")
    if radius_m < 5 or radius_m > 500:
        raise ValueError("radius_m must be between 5 and 500 meters.")
    if min(max_buildings, max_road_segments, max_trees) < 0:
        raise ValueError("Geometry limits must be nonnegative.")

    drone_urdf = os.path.join(assets, "custom_drone.urdf")
    if not os.path.isfile(drone_urdf):
        raise FileNotFoundError(f"Missing drone URDF: {drone_urdf}. Place env_sim.py beside project assets.")
    if p.isConnected():
        raise RuntimeError("PyBullet is already connected. Call p.disconnect() before build_world().")
    client = p.connect(p.GUI if gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Unable to connect to PyBullet. Use gui=False on a headless machine.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        # Speed up GUI startup by avoiding thousands of intermediate redraws.
        if gui:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        if lat is None:
            floor_id, obstacles = _legacy_world(assets)
            stats = {"mode": "legacy", "collision_obstacles": len(obstacles)}
        else:
            floor_id, obstacles, stats = _geo_world(float(lat), float(lon),
                                                   float(radius_m), geojson_path,
                                                   max_buildings, max_road_segments,
                                                   max_trees, json_in, json_out)
            print("Geographic environment generated from OpenStreetMap data "
                  "(c) OpenStreetMap contributors, ODbL.")
        drone_id = p.loadURDF(drone_urdf, DRONE_START_POS,
                              p.getQuaternionFromEuler([0, 0, 0]),
                              globalScaling=DRONE_SCALE)
        p.changeDynamics(drone_id, -1, mass=0)  # Navigator owns position updates.
        LAST_WORLD_INFO.clear()
        LAST_WORLD_INFO.update(stats)
        print("Environment summary:", stats)
        if gui:
            cam_dist = max(18.0, min(140.0, float(radius_m) * 1.45)) if lat is not None else 24.0
            target = [0.0, 0.0, 0.5] if lat is not None else DRONE_START_POS
            p.resetDebugVisualizerCamera(cam_dist, 35, -50 if lat is not None else -45, target)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        return floor_id, drone_id, obstacles
    except Exception:
        if gui:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            except Exception:
                pass
        p.disconnect()
        raise


def _main():
    parser = argparse.ArgumentParser(description="OSM geographical environment for PyBullet SAR prototype")
    parser.add_argument("--lat", type=float, help="Latitude of last known position (WGS84)")
    parser.add_argument("--lon", type=float, help="Longitude of last known position (WGS84)")
    parser.add_argument("--radius", type=float, default=50, help="Search radius in meters (5-500)")
    parser.add_argument("--geojson", help="Offline WGS84 GeoJSON FeatureCollection (no OSM download)")
    parser.add_argument("--json-out", default="environment.json",
                        help="Where to save the parsed environment JSON (default environment.json)")
    parser.add_argument("--json-in", help="Rebuild from a saved environment JSON (offline)")
    parser.add_argument("--assets", help="Folder containing custom_drone.urdf")
    parser.add_argument("--no-gui", action="store_true", help="Build/validate world without opening a window")
    parser.add_argument("--prompt", action="store_true", help="Ask user for latitude/longitude/radius")
    parser.add_argument("--demo", action="store_true", help="Run original manually created environment")
    args = parser.parse_args()
    if args.prompt:
        import osm_parser
        args.lat, args.lon, args.radius = osm_parser.prompt_for_lkp()
    if args.demo:
        args.lat = args.lon = None
        os.environ.pop("SAR_LAT", None)
        os.environ.pop("SAR_LON", None)
        os.environ.pop("SAR_GEOJSON", None)
    floor, drone, obstacles = build_world(
        gui=not args.no_gui, assets=args.assets,
        lat=args.lat, lon=args.lon, radius_m=args.radius,
        geojson_path=args.geojson, json_in=args.json_in, json_out=args.json_out,
    )
    if args.no_gui:
        print(f"Created floor={floor}, drone={drone}, {len(obstacles)} obstacle bodies.")
        p.disconnect()
        return
    print("World loaded. Drag the mouse in PyBullet to inspect it; Ctrl+C to exit.")
    try:
        while p.isConnected():
            p.stepSimulation()
            time.sleep(1.0 / 120.0)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    _main()
