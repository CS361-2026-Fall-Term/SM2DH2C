"""
Contributor: Chase Vanderlip
File Description: Turns the operator's geographical input into the obstacles the drone has to
detect and avoid (extends obstacle detection/avoidance from hand-placed trees to map objects).

  geographical input (.osm / Overpass .json file, or a download around the last known position)
      |  load_geo_file()   read the file; a file that cannot be read gives one clear GeoDataError
      |  validate()        check every map object; a malformed object is skipped and reported
      v
  osm_world.parse_scene()  lat/lon -> sim x/y, clipped to the map (existing code)
      |  obstacles_from_world()   key objects: trees, buildings, walls -> Obstacle records
      v
  spawn()                  builds each obstacle directly in PyBullet (collision + visual shapes,
                           createMultiBody) at its real location: a tree is one body (trunk +
                           canopy), a wall is a box, a building is its footprint extruded to its
                           height (a concave mesh, so an L-shaped building keeps its inner
                           corner). Returns the PyBullet body ids and {body id: Obstacle}. The
                           Navigator (object_avoidance.py) uses that table to know WHAT its range
                           sensor hit: name, kind, height, outline.
                           (No URDF/OBJ files are written: building the shapes in memory gives
                           the same collision shapes without files on disk.)

Nothing here raises for a single bad object: it is skipped, a message says which one and why, and
the rest of the world is still built (AC-03). Only input that is unusable as a whole (missing
file, not JSON/XML, no map objects, impossible latitude) raises GeoDataError, whose text is meant
to be shown to the operator as-is.

Run as a script to check a map file and list the obstacles it gives, without starting the simulation:
    python geo_obstacles.py map.osm --lat 44.6488 --lon -123.3123 --area 60
"""
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import osm_world
import sim_config

# Tree shape. Every tree is the same trunk + canopy the hand-placed trees always had, so the
# camera occlusion and probability-map code (which assume this shape) stay correct.
TRUNK_RADIUS = 0.2
TRUNK_HEIGHT = 2.5
CANOPY_RADIUS = 0.75
CANOPY_OFFSET_Z = 1.75          # canopy centre above the trunk centre
TREE_BASE_Z = 0.5               # z of the trunk centre (third value of env_sim.OBSTACLE_POSITIONS)
TREE_TOP = TREE_BASE_Z + CANOPY_OFFSET_Z + CANOPY_RADIUS      # 3.0 m

MAX_PROBLEMS_SHOWN = 10         # the rest are summarised as "... and N more"
MIN_FOOTPRINT_AREA = 1e-4       # m^2: a building outline smaller than this is not a solid


class GeoDataError(ValueError):
    """The geographical input cannot be used at all. str(exc) is a message for the operator."""


# ------------------------------------------------------------------ 1. reading the input
def load_geo_file(path, problems=None):
    """Read an OpenStreetMap .osm (XML) export or an Overpass .json file.
    Returns the raw {"elements": [...]} data; pass it through validate() next.
    problems: list that receives a line for every object in the file that had to be skipped.
    Raises GeoDataError (never a bare parser error) if the file as a whole is unusable."""
    problems = [] if problems is None else problems
    if not isinstance(path, (str, os.PathLike)) or not str(path).strip():
        raise GeoDataError("No map file was given.")
    path = str(path)
    name = os.path.basename(path)
    if not os.path.isfile(path):
        raise GeoDataError(f"Map file not found: {path}")
    try:
        if os.path.getsize(path) == 0:
            raise GeoDataError(f"Map file {name} is empty.")
        data = osm_world.load_osm_file(path, problems)
    except GeoDataError:
        raise
    except json.JSONDecodeError as exc:
        raise GeoDataError(f"Map file {name} is not valid JSON (line {exc.lineno}, column "
                           f"{exc.colno}: {exc.msg}). Export the area again from Overpass or "
                           f"openstreetmap.org.") from exc
    except ET.ParseError as exc:
        raise GeoDataError(f"Map file {name} is not valid OpenStreetMap XML ({exc}). The file may "
                           f"be cut off or not an .osm export; export the area again from "
                           f"openstreetmap.org.") from exc
    except UnicodeDecodeError as exc:
        raise GeoDataError(f"Map file {name} is not a text file (it could not be decoded: "
                           f"{exc.reason}).") from exc
    except OSError as exc:
        raise GeoDataError(f"Map file {name} could not be read ({exc.strerror or exc}).") from exc
    except Exception as exc:                      # anything unexpected in the reader
        raise GeoDataError(f"Map file {name} could not be read ({type(exc).__name__}: {exc}).") from exc
    if not path.lower().endswith(".json"):
        root_tag = _xml_root_tag(path)
        if root_tag != "osm":
            raise GeoDataError(f"Map file {name} is not an OpenStreetMap export (it starts with "
                               f"<{root_tag}>, expected <osm>).")
    return data


def _xml_root_tag(path):
    try:
        for _event, el in ET.iterparse(path, events=("start",)):
            return el.tag
    except ET.ParseError:
        pass
    return None


def _number(v):
    """True for a real, finite int/float (bool, str, None, nan and inf are not coordinates)."""
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _bad_point(g):
    """None if g is a usable {'lat', 'lon'} point, else a short reason."""
    if not isinstance(g, dict):
        return f"a point is {type(g).__name__}, not a lat/lon pair"
    for key, lim in (("lat", 90.0), ("lon", 180.0)):
        if key not in g:
            return f"a point has no '{key}'"
        if not _number(g[key]):
            return f"a point has {key}={g[key]!r}, not a number"
        if abs(g[key]) > lim:
            return f"a point has {key}={g[key]}, outside +-{lim:g}"
    return None


def _clean_geometry(geom, what="geometry"):
    """(points, None) or (None, reason). Overpass uses null for points outside its search circle;
    those are dropped, exactly as the parser always did."""
    if geom is None:
        return [], None
    if not isinstance(geom, list):
        return None, f"{what} is {type(geom).__name__}, not a list of points"
    out = []
    for g in geom:
        if g is None:
            continue
        reason = _bad_point(g)
        if reason:
            return None, reason
        out.append({"lat": g["lat"], "lon": g["lon"]})
    return out, None


def _ring_is_degenerate(points):
    """A closed outline with (almost) no area: every point on one line, or a figure-eight whose
    two halves cancel out. Such a building has no footprint to place."""
    pts = [(g["lon"], g["lat"]) for g in points]
    if len(pts) < 4 or pts[0] != pts[-1]:
        return False
    xs, ys = [q[0] for q in pts], [q[1] for q in pts]
    box = (max(xs) - min(xs)) * (max(ys) - min(ys))
    area = abs(osm_world.polygon_area(pts[:-1]))
    return area <= 1e-6 * box or box == 0.0


def validate(data, problems=None):
    """Check the loaded map data object by object.
    Returns a cleaned copy {"elements": [...]} holding only objects that are safe to parse; each
    rejected object adds one line to `problems` saying which object and what is wrong with it.
    Raises GeoDataError if the data is not map data at all, or if no object in it is usable."""
    problems = [] if problems is None else problems
    if not isinstance(data, dict):
        raise GeoDataError(f"The map data is not an OpenStreetMap/Overpass document (top level is "
                           f"{type(data).__name__}, expected an object with an 'elements' list).")
    if "elements" not in data:
        raise GeoDataError("The map data has no 'elements' list, so it contains no map objects. "
                           "Is this an Overpass JSON or .osm export?")
    elements = data["elements"]
    if not isinstance(elements, list):
        raise GeoDataError(f"The map data's 'elements' entry is {type(elements).__name__}, not a list "
                           f"of map objects.")
    clean = []
    for i, el in enumerate(elements):
        label = f"object #{i + 1} [{osm_world.describe_element(el)}]"
        if not isinstance(el, dict):
            problems.append(f"{label}: not a map object - skipped")
            continue
        kind = el.get("type")
        if kind not in ("node", "way", "relation"):
            problems.append(f"{label}: unknown or missing type {kind!r} (expected node, way or "
                            f"relation) - skipped")
            continue
        tags = el.get("tags")
        if tags is None:
            tags = {}
        if not isinstance(tags, dict):
            problems.append(f"{label}: its tags are {type(tags).__name__}, not key/value pairs - skipped")
            continue
        # OSM tag values are text; a number written without quotes is accepted as text
        out = {"type": kind, "tags": {str(k): ("" if v is None else str(v)) for k, v in tags.items()}}
        if el.get("id") is not None:
            out["id"] = el["id"]
        if kind == "node":
            reason = _bad_point(el)
            if reason:
                problems.append(f"{label}: {reason.replace('a point', 'it')} - skipped")
                continue
            out["lat"], out["lon"] = el["lat"], el["lon"]
        elif kind == "way":
            geom, reason = _clean_geometry(el.get("geometry"))
            if reason:
                problems.append(f"{label}: {reason} - skipped")
                continue
            if "building" in out["tags"] and out["tags"]["building"] != "no":
                if len(geom) < 4:
                    problems.append(f"{label}: a building outline needs at least 3 corners, this one "
                                    f"has {max(0, len(geom) - 1)} - skipped")
                    continue
                if geom[0] != geom[-1]:
                    problems.append(f"{label}: the building outline is not closed (first and last "
                                    f"point differ) - skipped")
                    continue
                if _ring_is_degenerate(geom):
                    problems.append(f"{label}: the building outline has no area (points on one line, "
                                    f"or the outline crosses itself) - skipped")
                    continue
            out["geometry"] = geom
        else:
            members = el.get("members")
            if members is None:
                members = []
            if not isinstance(members, list):
                problems.append(f"{label}: its members are {type(members).__name__}, not a list - skipped")
                continue
            kept, reason = [], None
            for m in members:
                if not isinstance(m, dict):
                    reason = f"a member is {type(m).__name__}, not a map object"
                    break
                if m.get("type") != "way":
                    continue                              # node / sub-relation members are not used
                geom, reason = _clean_geometry(m.get("geometry"), "a member's geometry")
                if reason:
                    break
                kept.append({"type": "way", "role": str(m.get("role") or ""), "geometry": geom})
            if reason:
                problems.append(f"{label}: {reason} - skipped")
                continue
            out["members"] = kept
        clean.append(out)
    if elements and not clean:
        raise GeoDataError(f"None of the {len(elements)} map objects could be read "
                           f"(first problem: {problems[0] if problems else 'unknown'}).")
    return {"elements": clean}


def check_position(lat, lon, area_m=None):
    """The last known position and search width typed by the operator. Raises GeoDataError."""
    if not _number(lat) or abs(lat) > 90:
        raise GeoDataError(f"Latitude must be a number between -90 and 90 (got {lat!r}).")
    if not _number(lon) or abs(lon) > 180:
        raise GeoDataError(f"Longitude must be a number between -180 and 180 (got {lon!r}).")
    if area_m is not None and (not _number(area_m) or area_m <= 0):
        raise GeoDataError(f"The search area width must be a positive number of metres (got {area_m!r}).")


def format_problems(problems, limit=MAX_PROBLEMS_SHOWN):
    """Lines to print for the operator: the first `limit` problems, then a count of the rest."""
    lines = list(problems[:limit])
    if len(problems) > limit:
        lines.append(f"... and {len(problems) - limit} more")
    return lines


# ------------------------------------------------------------------ 2. key objects
@dataclass
class Obstacle:
    """One solid object the drone must detect and avoid, in sim coordinates (x east, y north)."""
    kind: str                       # "tree" | "building" | "wall"
    name: str                       # shown to the operator, e.g. 'building "Kerr Library"', "tree 12"
    x: float                        # where its body is placed (tree: trunk; wall: centre; building: 0)
    y: float
    z: float                        # z of the body origin
    height: float                   # top of the object above the ground (m)
    radius: float                   # farthest part of the object from (cx, cy), horizontally
    cx: float = 0.0                 # centre of the object on the map
    cy: float = 0.0
    footprint: list = None          # outline [(x, y), ...] of a vertical-sided object, else None
    source: str = "layout"          # "map" = from the geographical input, "layout" = built-in / generated
    meta: dict = field(default_factory=dict)

    def describe(self):
        return f"{self.name} ({self.height:.1f} m tall) at ({self.cx:.1f}, {self.cy:.1f})"


def obstacles_from_world(tree_positions=(), tree_kinds=(), walls=(), scene=None, tree_source=None):
    """The key objects of the world as Obstacle records, plus problems for any that are unusable.
    tree_positions: env_sim.OBSTACLE_POSITIONS  [(x, y, z of trunk centre)]
    walls:          env_sim.WALLS               [(x, y, half_x, half_y, height)]
    scene:          osm_world.OSMScene or None  (its buildings become obstacles; its trees are
                    already in tree_positions, put there by env_sim.use_osm)
    Returns (obstacles, problems)."""
    obstacles, problems = [], []
    tree_source = tree_source or ("map" if scene is not None else "layout")
    for i, pos in enumerate(tree_positions):
        try:
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2]) if len(pos) > 2 else TREE_BASE_Z
            if not all(math.isfinite(v) for v in (x, y, z)):
                raise ValueError("position is not finite")
        except (TypeError, ValueError, IndexError) as exc:
            problems.append(f"tree {i + 1}: bad position {pos!r} ({exc}) - skipped")
            continue
        kind = tree_kinds[i] if i < len(tree_kinds) else "deciduous"
        obstacles.append(Obstacle("tree", f"tree {i + 1} ({kind})", x, y, z,
                                  height=z + CANOPY_OFFSET_Z + CANOPY_RADIUS, radius=CANOPY_RADIUS,
                                  cx=x, cy=y, source=tree_source, meta={"tree_kind": kind, "index": i}))
    for i, wall in enumerate(walls):
        try:
            x, y, hx, hy, h = (float(v) for v in wall)
            if not all(math.isfinite(v) for v in (x, y, hx, hy, h)) or min(hx, hy, h) <= 0:
                raise ValueError("size must be positive and finite")
        except (TypeError, ValueError) as exc:
            problems.append(f"wall {i + 1}: bad definition {wall!r} ({exc}) - skipped")
            continue
        outline = [(x - hx, y - hy), (x + hx, y - hy), (x + hx, y + hy), (x - hx, y + hy)]
        obstacles.append(Obstacle("wall", f"wall {i + 1}", x, y, h / 2, height=h,
                                  radius=math.hypot(hx, hy), cx=x, cy=y, footprint=outline,
                                  meta={"half_extents": (hx, hy, h / 2)}))
    for i, (poly, h, info) in enumerate(scene.buildings if scene is not None else []):
        label = f'building "{info["name"]}"' if info.get("name") else \
            f"building {i + 1} ({info.get('kind', 'yes')})"
        try:
            poly = [(float(q[0]), float(q[1])) for q in poly]
            h = float(h)
            if len(poly) < 3 or not math.isfinite(h) or h <= 0 or \
                    not all(math.isfinite(c) for q in poly for c in q):
                raise ValueError("outline needs 3+ finite corners and a positive height")
            if abs(osm_world.polygon_area(poly)) < MIN_FOOTPRINT_AREA:
                raise ValueError("outline has no area (its corners are on one line)")
        except (TypeError, ValueError, IndexError) as exc:
            problems.append(f"{label}: unusable outline ({exc}) - skipped")
            continue
        cx = sum(q[0] for q in poly) / len(poly)
        cy = sum(q[1] for q in poly) / len(poly)
        obstacles.append(Obstacle("building", label, 0.0, 0.0, 0.0, height=h,
                                  radius=max(math.hypot(q[0] - cx, q[1] - cy) for q in poly),
                                  cx=cx, cy=cy, footprint=poly, source="map",
                                  meta={"index": i, "building": info.get("kind", "yes")}))
    return obstacles, problems


# ------------------------------------------------------------------ 3. shapes
def _building_shape(ob):
    """(vertices, triangle indices) of a building's extruded footprint. Raises ValueError for an
    outline that cannot be a solid (no area, or it cannot be triangulated)."""
    if len(ob.footprint or []) < 3 or abs(osm_world.polygon_area(ob.footprint)) < MIN_FOOTPRINT_AREA:
        raise ValueError("the outline has no area (its corners are on one line)")
    mesh = osm_world.building_mesh(ob.footprint, ob.height)
    if mesh is None:
        raise ValueError("the outline could not be turned into a solid shape "
                         "(it probably crosses itself)")
    return mesh


# ------------------------------------------------------------------ 4. into PyBullet
HIDDEN = [0, 0, 0, 0]            # fully transparent: the shape exists for the sensor only


def _body(p, ob, visual):
    """Create one obstacle in the connected PyBullet world and return its body id.
    visual=False: the shapes are invisible (the detailed look draws its own tree models / textured
    buildings on top). A body always gets SOME visual shape: one without makes PyBullet draw the
    bare collision shape (grey cylinders / slabs over the detailed models)."""
    def vis(shape, rgba, **kw):
        return p.createVisualShape(shape, rgbaColor=rgba if visual else HIDDEN, **kw)
    if ob.kind == "tree":
        # One body, two parts (like the hand-placed trees): the trunk is the base and the canopy
        # a fixed link above it, so every ray that hits either part reports the same body id.
        trunk_c = p.createCollisionShape(p.GEOM_CYLINDER, radius=TRUNK_RADIUS, height=TRUNK_HEIGHT)
        canopy_c = p.createCollisionShape(p.GEOM_SPHERE, radius=CANOPY_RADIUS)
        trunk_v = vis(p.GEOM_CYLINDER, [0.4, 0.2, 0.05, 1], radius=TRUNK_RADIUS, length=TRUNK_HEIGHT)
        canopy_v = vis(p.GEOM_SPHERE, [0.1, 0.45, 0.1, 1], radius=CANOPY_RADIUS)
        return p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=trunk_c, baseVisualShapeIndex=trunk_v,
            basePosition=[ob.x, ob.y, ob.z],
            linkMasses=[0], linkCollisionShapeIndices=[canopy_c], linkVisualShapeIndices=[canopy_v],
            linkPositions=[[0, 0, CANOPY_OFFSET_Z]], linkOrientations=[[0, 0, 0, 1]],
            linkInertialFramePositions=[[0, 0, 0]], linkInertialFrameOrientations=[[0, 0, 0, 1]],
            linkParentIndices=[0], linkJointTypes=[p.JOINT_FIXED], linkJointAxis=[[0, 0, 1]])
    if ob.kind == "wall":
        half = list(ob.meta["half_extents"])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        # walls have no separate detailed model, so they are always drawn
        wall_v = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=[0.6, 0.6, 0.62, 1])
        return p.createMultiBody(0, col, wall_v, [ob.x, ob.y, ob.z])
    if ob.kind == "building":
        verts, idx = _building_shape(ob)
        # Concave triangle mesh: the footprint is used as-is, not replaced by its convex hull.
        col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx,
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
        if visual:
            bv = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=idx,
                                     rgbaColor=[0.62, 0.58, 0.55, 1])
        else:                                     # invisible speck inside the building
            bv = p.createVisualShape(p.GEOM_SPHERE, radius=0.001, rgbaColor=HIDDEN,
                                     visualFramePosition=[ob.cx, ob.cy, min(0.5, ob.height / 2)])
        return p.createMultiBody(0, col, bv, [0, 0, 0])
    raise ValueError(f"unknown obstacle kind {ob.kind!r}")


def spawn(p, obstacles, visual=True):
    """Build every obstacle in the connected PyBullet world at its real position.
    Returns (body ids, {body id: Obstacle}, problems). An obstacle that cannot be built (e.g. a
    building outline that crosses itself) is left out and reported; the others are still placed."""
    ids, table, problems = [], {}, []
    for ob in obstacles:
        try:
            body = _body(p, ob, visual)
        except Exception as exc:                  # p.error, ValueError
            problems.append(f"{ob.name}: could not be placed ({type(exc).__name__}: {exc}) - skipped")
            continue
        if body is None or body < 0:
            problems.append(f"{ob.name}: PyBullet did not create a body - skipped")
            continue
        ids.append(body)
        table[body] = ob
    return ids, table, problems


# ------------------------------------------------------------------ command line
def main():
    """Check a map file and list the obstacles it gives, without starting the simulation."""
    import argparse
    ap = argparse.ArgumentParser(description="Check a map file and list the obstacles the "
                                             "simulation would build from it.")
    ap.add_argument("map_file", help=".osm export or Overpass .json")
    ap.add_argument("--lat", type=float, required=True, help="last known position latitude")
    ap.add_argument("--lon", type=float, required=True, help="last known position longitude")
    ap.add_argument("--area", type=float, default=None, help="real metres shown across the map")
    ap.add_argument("--map-size", type=float, default=None, help="sim map width in metres")
    ap.add_argument("--list", action="store_true", help="print every obstacle, not just the counts")
    args = ap.parse_args()
    problems = []
    try:
        check_position(args.lat, args.lon, args.area)
        area = sim_config.BASE_MAP_SIZE if args.area is None else args.area
        sim_config.set_map_size(args.map_size if args.map_size is not None
                                else sim_config.clamp_map_size(area))
        data = validate(load_geo_file(args.map_file, problems), problems)
    except (GeoDataError, ValueError) as exc:
        raise SystemExit(f"ERROR: {exc}")
    scene = osm_world.parse_scene(data, args.lat, args.lon, area)
    problems += scene.problems
    obstacles, more = obstacles_from_world([(x, y, TREE_BASE_Z) for x, y in scene.trees],
                                           scene.tree_kinds, (), scene)
    problems += more
    usable = []
    for ob in obstacles:                          # same shape check spawn() relies on
        try:
            if ob.kind == "building":
                _building_shape(ob)
            usable.append(ob)
        except ValueError as exc:
            problems.append(f"{ob.name}: {exc} - skipped")
    kinds = {k: sum(1 for o in usable if o.kind == k) for k in ("building", "tree")}
    print(f"{kinds['building']} buildings and {kinds['tree']} trees would be placed")
    if args.list:
        for ob in usable:
            print("  ", ob.describe())
    for line in format_problems(problems):
        print("  problem:", line)


if __name__ == "__main__":
    main()
