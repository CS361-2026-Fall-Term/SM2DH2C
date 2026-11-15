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
  write_urdf()             one URDF per obstacle (buildings also get an .obj mesh), in
      |                    generated_obstacles/ together with manifest.json. File names carry a
      |                    hash of their contents (building_003_1a2b3c4d5e.urdf): PyBullet caches
      |                    loaded files by name for the whole process, so a changed shape must
      |                    get a new name or the OLD shape would be used.
      v
  spawn()                  p.loadURDF() at the real location; returns the PyBullet body ids and
                           {body id: Obstacle}. The Navigator (object_avoidance.py) uses that
                           table to know WHAT its range sensor hit: name, kind, height, outline.

Nothing here raises for a single bad object: it is skipped, a message says which one and why, and
the rest of the world is still built (AC-03). Only input that is unusable as a whole (missing
file, not JSON/XML, no map objects, impossible latitude) raises GeoDataError, whose text is meant
to be shown to the operator as-is.

Run as a script to convert a map file without starting the simulation:
    python geo_obstacles.py map.osm --lat 44.6488 --lon -123.3123 --area 60 --out my_obstacles
"""
import hashlib
import json
import math
import os
import tempfile
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

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generated_obstacles")
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
    x: float                        # where its URDF is placed (tree: trunk; wall: centre; building: 0)
    y: float
    z: float                        # z of the URDF origin
    height: float                   # top of the object above the ground (m)
    radius: float                   # farthest part of the object from (cx, cy), horizontally
    cx: float = 0.0                 # centre of the object on the map
    cy: float = 0.0
    footprint: list = None          # outline [(x, y), ...] of a vertical-sided object, else None
    source: str = "layout"          # "map" = from the geographical input, "layout" = built-in / generated
    meta: dict = field(default_factory=dict)
    urdf: str = None                # file it was loaded from (None = primitive fallback)

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


# ------------------------------------------------------------------ 3. URDF files
QUOTE = '"'


def _fmt(v):
    """Number for a URDF attribute: 5 decimals, without trailing zeros."""
    return f"{v:.5f}".rstrip("0").rstrip(".")


def _urdf(name, links, hide_at=(0.0, 0.0, 0.0)):
    """URDF text for a static (mass 0) body.
    links: [(link name, geometry xml, origin xyz, rgba or None for no visible shape, concave?)].
    The first link is the base; the others hang off it with fixed joints.
    rgba None: the link gets an invisible speck as its visual, placed at `hide_at` (link frame).
    A link with NO <visual> would make PyBullet draw its bare collision shape instead (grey
    cylinders/slabs on top of the detailed tree and building models); osm_world._invisible_solid
    uses the same trick for the same reason."""
    out = ['<?xml version="1.0"?>',
           "<!-- Generated by geo_obstacles.py from the operator's geographical input. Do not edit: "
           "it is rewritten every time the simulation starts. -->",
           f'<robot name="{name}">']
    for i, (lname, geom, xyz, rgba, concave) in enumerate(links):
        origin = f'<origin xyz="{_fmt(xyz[0])} {_fmt(xyz[1])} {_fmt(xyz[2])}" rpy="0 0 0"/>'
        out.append(f'  <link name="{lname}">')
        out.append('    <inertial><origin xyz="0 0 0"/><mass value="0"/>'
                   '<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/></inertial>')
        if rgba is not None:
            colour = " ".join(_fmt(c) for c in rgba)
            out.append(f'    <visual>{origin}<geometry>{geom}</geometry>'
                       f'<material name="{lname}_colour"><color rgba="{colour}"/></material></visual>')
        else:
            spot = f'<origin xyz="{_fmt(hide_at[0])} {_fmt(hide_at[1])} {_fmt(hide_at[2])}" rpy="0 0 0"/>'
            out.append(f'    <visual>{spot}<geometry><sphere radius="0.001"/></geometry>'
                       f'<material name="{lname}_hidden"><color rgba="0 0 0 0"/></material></visual>')
        # concave="yes" on <collision> makes PyBullet keep the mesh exactly as written (a static
        # triangle mesh) instead of replacing it with its convex hull.
        out.append(f'    <collision{" concave=" + QUOTE + "yes" + QUOTE if concave else ""}>{origin}'
                   f'<geometry>{geom}</geometry></collision>')
        out.append("  </link>")
        if i:
            out.append(f'  <joint name="{links[0][0]}_to_{lname}" type="fixed">'
                       f'<parent link="{links[0][0]}"/><child link="{lname}"/>'
                       f'<origin xyz="0 0 0"/></joint>')
    out.append("</robot>")
    return "\n".join(out) + "\n"


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


def _unique_name(stem, text, ext):
    """File name with a short hash of the file's contents, e.g. building_003_1a2b3c4d5e.obj.
    PyBullet caches every mesh and URDF it loads BY FILE NAME for the rest of the process (even
    after p.disconnect()). Reusing building_003.obj for a different map in the same process would
    silently give the OLD shape; a content hash makes a changed shape a new file."""
    return f"{stem}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}{ext}"


def _write(out_dir, name, text):
    path = os.path.join(out_dir, name)
    if not os.path.exists(path):                  # same name = same contents, already written
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return path


def write_urdf(ob, out_dir, visual=True):
    """Write the URDF (and, for a building, its .obj mesh) for one obstacle. Returns the URDF path.
    visual=False gives the shapes an invisible visual: used with the detailed look, where separate
    textured/tree-model bodies are drawn instead. The collision shape is the same either way.
    Trees share one file (they all have the same shape); it is loaded once per tree position."""
    if ob.kind == "tree":
        stem = "tree" if visual else "tree_collision_only"
        text = _urdf(stem, [
            ("trunk", f'<cylinder radius="{TRUNK_RADIUS}" length="{TRUNK_HEIGHT}"/>', (0, 0, 0),
             (0.4, 0.2, 0.05, 1) if visual else None, False),
            ("canopy", f'<sphere radius="{CANOPY_RADIUS}"/>', (0, 0, CANOPY_OFFSET_Z),
             (0.1, 0.45, 0.1, 1) if visual else None, False)])
    elif ob.kind == "wall":
        stem = ob.name.replace(" ", "_")
        hx, hy, hz = ob.meta["half_extents"]
        geom = f'<box size="{_fmt(2 * hx)} {_fmt(2 * hy)} {_fmt(2 * hz)}"/>'
        text = _urdf(stem, [("wall", geom, (0, 0, 0), (0.6, 0.6, 0.62, 1) if visual else None, False)])
    elif ob.kind == "building":
        stem = f"building_{ob.meta.get('index', 0):03d}"
        verts, idx = _building_shape(ob)
        lines = [f"# {ob.name}: extruded footprint, {ob.height:.2f} m tall (sim metres)"]
        lines += [f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}" for v in verts]
        lines += [f"f {idx[k] + 1} {idx[k + 1] + 1} {idx[k + 2] + 1}" for k in range(0, len(idx), 3)]
        obj_text = "\n".join(lines) + "\n"
        obj_name = _write(out_dir, _unique_name(stem, obj_text, ".obj"), obj_text)
        geom = f'<mesh filename="{os.path.basename(obj_name)}" scale="1 1 1"/>'
        # concave: the footprint is used as-is (an L-shaped building keeps its inner corner).
        text = _urdf(stem, [("building", geom, (0, 0, 0),
                             (0.62, 0.58, 0.55, 1) if visual else None, True)],
                     hide_at=(ob.cx, ob.cy, min(0.5, ob.height / 2)))     # speck inside the building
    else:
        raise ValueError(f"unknown obstacle kind {ob.kind!r}")
    return _write(out_dir, _unique_name(stem, text, ".urdf"), text)


def prepare_dir(out_dir=None):
    """Empty folder for this run's obstacle files. Old generated files are removed (only .urdf,
    .obj and manifest.json, nothing else). Falls back to a temp folder if it cannot be written."""
    out_dir = out_dir or OUT_DIR
    try:
        os.makedirs(out_dir, exist_ok=True)
        for name in os.listdir(out_dir):
            if name.endswith((".urdf", ".obj")) or name == "manifest.json":
                os.remove(os.path.join(out_dir, name))
        return out_dir
    except OSError as exc:
        tmp = tempfile.mkdtemp(prefix="generated_obstacles_")
        print(f"Note: could not use {out_dir} ({exc}); obstacle files are written to {tmp}")
        return tmp


def write_manifest(obstacles, out_dir):
    """manifest.json: what was generated and where each object was placed."""
    rows = [{"kind": o.kind, "name": o.name, "source": o.source,
             "centre_xy": [round(o.cx, 3), round(o.cy, 3)], "height_m": round(o.height, 3),
             "radius_m": round(o.radius, 3),
             "urdf": os.path.basename(o.urdf) if o.urdf else None,
             "urdf_position": [round(o.x, 3), round(o.y, 3), round(o.z, 3)]} for o in obstacles]
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"map_size_m": sim_config.map_size(), "obstacles": rows}, f, indent=1)


# ------------------------------------------------------------------ 4. into PyBullet
def _primitive_body(p, ob, visual):
    """The same collision shape built directly (no file). Used only if a URDF cannot be
    written or loaded, so one bad file never leaves a hole in the world."""
    def vis(shape, rgba, **kw):
        return p.createVisualShape(shape, rgbaColor=rgba, **kw) if visual else \
            p.createVisualShape(p.GEOM_SPHERE, radius=0.001, rgbaColor=[0, 0, 0, 0])
    if ob.kind == "tree":
        col = p.createCollisionShapeArray(
            shapeTypes=[p.GEOM_CYLINDER, p.GEOM_SPHERE], radii=[TRUNK_RADIUS, CANOPY_RADIUS],
            lengths=[TRUNK_HEIGHT, 0],
            collisionFramePositions=[[0, 0, 0], [0, 0, CANOPY_OFFSET_Z]])
        return p.createMultiBody(0, col, vis(p.GEOM_CYLINDER, [0.4, 0.2, 0.05, 1], radius=TRUNK_RADIUS,
                                             length=TRUNK_HEIGHT), [ob.x, ob.y, ob.z])
    if ob.kind == "wall":
        half = list(ob.meta["half_extents"])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        return p.createMultiBody(0, col, vis(p.GEOM_BOX, [0.6, 0.6, 0.62, 1], halfExtents=half),
                                 [ob.x, ob.y, ob.z])
    verts, idx = _building_shape(ob)
    col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=idx,
                                 flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    return p.createMultiBody(0, col, vis(p.GEOM_MESH, [0.62, 0.58, 0.55, 1], vertices=verts,
                                         indices=idx), [0, 0, 0])


def spawn(p, obstacles, out_dir=None, visual=True):
    """Write every obstacle as a URDF file and load it into the connected PyBullet world at its
    real position. Returns (body ids, {body id: Obstacle}, problems).
    An obstacle whose file cannot be written or loaded is built directly instead; if that fails
    too it is left out and reported. Neither case stops the others from being placed."""
    out_dir = prepare_dir(out_dir)
    ids, table, problems, placed = [], {}, [], []
    for ob in obstacles:
        body = -1
        try:
            # walls have no separate detailed model, so they always carry their own visual
            ob.urdf = write_urdf(ob, out_dir, visual or ob.kind == "wall")
            body = p.loadURDF(ob.urdf, [ob.x, ob.y, ob.z], useFixedBase=True)
        except Exception as exc:                  # p.error, OSError, ValueError
            ob.urdf = None
            first = f"{type(exc).__name__}: {exc}"
            try:
                body = _primitive_body(p, ob, visual)
                problems.append(f"{ob.name}: URDF not used ({first}); built directly instead")
            except Exception as exc2:
                problems.append(f"{ob.name}: could not be placed ({first}; then "
                                f"{type(exc2).__name__}: {exc2}) - skipped")
                continue
        if body is None or body < 0:
            problems.append(f"{ob.name}: PyBullet did not create a body - skipped")
            continue
        ids.append(body)
        table[body] = ob
        placed.append(ob)
    try:
        write_manifest(placed, out_dir)
    except OSError as exc:
        problems.append(f"manifest.json not written ({exc})")
    return ids, table, problems


# ------------------------------------------------------------------ command line
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Convert a map file into URDF obstacle files "
                                             "(the same ones the simulation generates).")
    ap.add_argument("map_file", help=".osm export or Overpass .json")
    ap.add_argument("--lat", type=float, required=True, help="last known position latitude")
    ap.add_argument("--lon", type=float, required=True, help="last known position longitude")
    ap.add_argument("--area", type=float, default=None, help="real metres shown across the map")
    ap.add_argument("--map-size", type=float, default=None, help="sim map width in metres")
    ap.add_argument("--out", default=OUT_DIR, help="folder for the .urdf / .obj files")
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
    out_dir = prepare_dir(args.out)
    written = []
    for ob in obstacles:
        try:
            ob.urdf = write_urdf(ob, out_dir)
            written.append(ob)
        except (OSError, ValueError) as exc:
            problems.append(f"{ob.name}: no URDF written ({exc})")
    write_manifest(written, out_dir)
    kinds = {k: sum(1 for o in written if o.kind == k) for k in ("building", "tree")}
    print(f"{kinds['building']} buildings and {kinds['tree']} trees written to {out_dir} "
          f"(see manifest.json)")
    for line in format_problems(problems):
        print("  problem:", line)


if __name__ == "__main__":
    main()
