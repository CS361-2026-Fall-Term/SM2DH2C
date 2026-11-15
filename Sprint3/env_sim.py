"""
Contributors: Mari Tokuda, Heeseo Kim and Mylan Bui
File Description: Creates shared PyBullet environment. Builds the physical world where
the search happens, including the visualization of the drone body, trees, and ground.
While the target position is set here, the humaniod target is created and spawned 
separately in target_detection.py.
"""
import math
import os
import random
import pybullet as p
import pybullet_data
import geo_obstacles
import osm_world
import sim_config

# Static positions of drone, obstacle, and target in the simulated environment.
# Positions are (x , y, z) in the PyBullet world.
DRONE_START_POS = [-10, 10, 3.5]
DRONE_SCALE = 1.5
OBSTACLE_POSITIONS = [
    [1.5, 6.5, 0.5], [-4.0, -1.5, 0.5], [8.0, -2.0, 0.5], [-7.5, 5, 0.5],
    [-8.0, -8.0, 0.5], [9.0, 8.5, 0.5], [-1.0, -7.5, 0.5], [0.0, 1.0, 0.5],
]
TARGET_POSITION = [7.0, -5.5, 0.875]
TARGET_ORIENTATION_EULER = [1.5708, 0, 0]
TARGET_SCALE = 0.25
FLIGHT_PATH = [
    [-10, 10, 2.5], [-5, 10, 2.5], [-5, 5, 2.5], [-5, 0, 2.5], [0, 0, 2.5],
    [5, 0, 2.5], [5, 5, 2.5], [10, 5, 2.5], [10, -5, 2.5], [5, -5, 2.5],
    [6, -5, 2.5],
]


# Set by use_osm(); None means the original hand-placed world.
OSM_SCENE = None

# Filled by build_world(): PyBullet body id -> geo_obstacles.Obstacle (kind, name, height, outline)
# for every solid obstacle. The Navigator uses it to know what its range sensor hit. Updated IN
# PLACE, so modules that imported it see the current world.
OBSTACLE_INFO = {}
# Map objects that could not be read / placed and were skipped (one line each, for the operator).
GEO_PROBLEMS = []

# Tall walls added by generate_layout() (PBI-04): (x, y, half_x, half_y, height). Empty = none.
# build_world() turns each into a solid box the drone senses and must route around.
WALLS = []
WALL_HEIGHT = 7.0              # taller than the drone can climb (MAX_ALT 5.5 m + clearance)
WALL_THICKNESS = 0.3

# Detailed visuals (textured buildings, tree models, coloured ground, sky, shadows). main.py turns
# this off with --simple-visuals. Visual only: collision shapes are identical either way.
DETAILED_VISUALS = True
# Tree model per entry of OBSTACLE_POSITIONS ("deciduous" or "conifer"); use_osm() replaces it.
TREE_KINDS = ["deciduous", "conifer", "deciduous", "conifer",
              "conifer", "deciduous", "deciduous", "conifer"]
SKY_OPTIONS = "--background_color_red=0.62 --background_color_green=0.76 --background_color_blue=0.92"
LOW_SPECULAR = [0.04, 0.04, 0.04]


def scale_layout(k):
    """
    Stretch the hand-placed layout (drone start, trees, target, flight path) by factor k in x and y,
    so it fills a map that is k times the original 26 m. Heights (z) are unchanged.
    Call ONCE, after sim_config.set_map_size() and BEFORE use_osm() / build_world().

    Lists are updated IN PLACE, so modules that imported them (object_avoidance.py,
    target_detection.py, main.py) see the scaled values, the same way use_osm() works.
    """
    if k == 1.0:
        return
    for pt in [DRONE_START_POS, TARGET_POSITION, *OBSTACLE_POSITIONS, *FLIGHT_PATH]:
        pt[0], pt[1] = round(pt[0] * k, 3), round(pt[1] * k, 3)


def use_osm(lat, lon, area_m=None, osm_file=None, keep_clear=(), seed=0):
    """
    Replace the hand-placed trees with real OSM features around the last known position (lat, lon).
    area_m real metres are fitted into the sim map (sim_config.map_size(), 26 m by default).
    area_m=None means true scale (area_m = map size).
    Call this BEFORE build_world() and MissingPerson.spawn().
    keep_clear: (x, y) points that must stay reachable (search-zone centres); the drone start is always added.

    OBSTACLE_POSITIONS and TARGET_POSITION are updated IN PLACE, so object_avoidance.py,
    target_detection.py and probability_map.py (which imported these lists) see the new world
    without any changes.
    """
    global OSM_SCENE
    area_m = sim_config.map_size() if area_m is None else area_m
    # Malformed input (AC-03): a bad position or an unreadable file raises geo_obstacles.GeoDataError
    # with a message for the operator; a single bad map object is skipped and listed in GEO_PROBLEMS.
    geo_obstacles.check_position(lat, lon, area_m)
    problems = []
    data = (geo_obstacles.load_geo_file(osm_file, problems) if osm_file
            else osm_world.fetch_overpass(lat, lon, area_m))
    data = geo_obstacles.validate(data, problems)
    clear = [tuple(DRONE_START_POS[:2])] + [tuple(c) for c in keep_clear]
    OSM_SCENE = osm_world.parse_scene(data, lat, lon, area_m, keep_clear=clear, seed=seed)
    problems += OSM_SCENE.problems
    GEO_PROBLEMS[:] = problems
    # If the start point was on a building it was moved to open ground; follow it.
    start = OSM_SCENE.moved.get(clear[0])
    if start is not None:
        DRONE_START_POS[0], DRONE_START_POS[1] = start

    OBSTACLE_POSITIONS[:] = [[x, y, 0.5] for x, y in OSM_SCENE.trees]
    TREE_KINDS[:] = OSM_SCENE.tree_kinds
    tx, ty = osm_world.nearest_free_point(OSM_SCENE, tuple(TARGET_POSITION[:2]), margin=1.2)
    TARGET_POSITION[0], TARGET_POSITION[1] = tx, ty

    print(f"OSM world at LKP ({lat}, {lon}), {area_m:.0f} m shown across the "
          f"{sim_config.map_size():.0f} m map (scale {sim_config.map_size() / area_m:.3f}): "
          f"{OSM_SCENE.summary()}")
    for note in OSM_SCENE.dropped:
        print("  adjusted to keep mission valid:", note)
    if problems:
        print(f"ERROR: {len(problems)} map object(s) could not be read and were left out of the "
              f"world (everything else was placed):")
        for line in geo_obstacles.format_problems(problems):
            print("  -", line)
    print(f"  target placed at ({tx}, {ty})")
    return OSM_SCENE




def generate_layout(seed, count, keep_clear=(), enclose=None, margin=1.5, min_spacing=1.2,
                    pen_half=1.5):
    """
    PBI-04: replace the hand-placed trees with a seeded random layout of `count` obstacles.
    The same seed always gives the same layout (AC-04).
    Call ONCE, after scale_layout() and BEFORE build_world() (same as use_osm()).

    keep_clear: (x, y) points (e.g. FLIGHT_PATH waypoints) that no tree may be within `margin` m of;
                the drone start is always kept clear.
    enclose:    optional (x, y) point to wall in with a 4-wall pen of WALL_HEIGHT, so that
                destination is truly unreachable (AC-02). The 4 walls count toward `count`.
    min_spacing: minimum distance between tree trunks (m), so trees don't merge into one blob.

    OBSTACLE_POSITIONS, TREE_KINDS and WALLS are updated IN PLACE, so modules that imported
    them see the new world. Returns a short summary dict.
    """
    rng = random.Random(seed)
    half = sim_config.MAP_HALF - 1.0
    clear = [tuple(DRONE_START_POS[:2])] + [tuple(c[:2]) for c in keep_clear]

    WALLS.clear()
    if enclose is not None:
        ex, ey = enclose[0], enclose[1]
        t = WALL_THICKNESS / 2
        WALLS.extend([
            (ex, ey + pen_half, pen_half + t, t, WALL_HEIGHT),     # north
            (ex, ey - pen_half, pen_half + t, t, WALL_HEIGHT),     # south
            (ex + pen_half, ey, t, pen_half + t, WALL_HEIGHT),     # east
            (ex - pen_half, ey, t, pen_half + t, WALL_HEIGHT),     # west
        ])

    n_trees = max(0, count - len(WALLS))
    trees, tries = [], 0
    while len(trees) < n_trees and tries < 20000:
        tries += 1
        x, y = round(rng.uniform(-half, half), 2), round(rng.uniform(-half, half), 2)
        if any(math.hypot(x - cx, y - cy) < margin for cx, cy in clear):
            continue
        if enclose is not None and (abs(x - enclose[0]) < pen_half + margin
                                    and abs(y - enclose[1]) < pen_half + margin):
            continue                                   # keep trees out of / off the pen walls
        if any(math.hypot(x - tx, y - ty) < min_spacing for tx, ty, _ in trees):
            continue
        trees.append([x, y, 0.5])
    if len(trees) < n_trees:
        raise ValueError(f"could only place {len(trees)} of {n_trees} trees (seed {seed}); "
                         f"lower count or margin")

    OBSTACLE_POSITIONS[:] = trees
    TREE_KINDS[:] = [rng.choice(("deciduous", "conifer")) for _ in trees]
    return {"seed": seed, "trees": len(trees), "walls": len(WALLS)}


def build_world(gui=True, assets=None):
    """
    Create exactly one world, drone and set of trees; target spawned separately.
    gui: True opens PyBullet window, False runs with no window (headless).
    assets: Folder holding custom_drone.urdf and ground_texture.jpg (defaults to the folder this file is in).
    Returns (floor_id, drone_id, obstacle_ids): PyBullet body ids for the ground, the drone and
    every solid obstacle (one body per tree, wall and map building). What each obstacle is can be
    looked up in OBSTACLE_INFO[body id].
    """
    assets = assets or os.path.dirname(os.path.abspath(__file__))
    # Start PyBullet: GUI = window to watch simulation, DIRECT = no window
    if gui:
        p.connect(p.GUI, options=SKY_OPTIONS if DETAILED_VISUALS else "")
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)   # build faster, no flicker
    else:
        p.connect(p.DIRECT)
    # PyBullet searchs built-in models, including plane.urdf used for the floor_id.
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81) 

    # Initialize floor (ground) of simulated environment.
    floor_id = p.loadURDF("plane.urdf")
    texture = os.path.join(assets, "ground_texture.jpg")
    if gui and os.path.exists(texture):
        try:
            p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(texture))
        except Exception as exc:
            print("Texture not applied:", exc)

    # Intialize drone body, loaded from custom URDF file. 
    drone_id = p.loadURDF(os.path.join(assets, "custom_drone.urdf"), DRONE_START_POS,
                          p.getQuaternionFromEuler([0, 0, 0]), globalScaling=DRONE_SCALE)
    # Navigator moves the drone kinematically. Prevent gravity/contacts from moving it a second time between position updates.
    p.changeDynamics(drone_id, -1, mass=0)

    # Obstacles (trees, tall walls, map buildings). Each one is described by an Obstacle record and
    # built directly in PyBullet at its real position. Its collision shape is what the LIDAR and
    # camera rays hit; it is the same in every visual mode.
    obstacles, problems = geo_obstacles.obstacles_from_world(OBSTACLE_POSITIONS, TREE_KINDS, WALLS,
                                                             OSM_SCENE)
    # Detailed look: the obstacle bodies are invisible, the tree models / textured buildings are
    # drawn as separate visual-only bodies below. Simple look: their own flat-colour shapes are shown.
    detailed = DETAILED_VISUALS
    obstacle_ids, table, spawn_problems = geo_obstacles.spawn(p, obstacles, visual=not detailed)
    OBSTACLE_INFO.clear()
    OBSTACLE_INFO.update(table)
    problems += spawn_problems
    if detailed:
        for ob in table.values():
            if ob.kind == "tree":
                _draw_tree(ob)

    # Map roads/water/ground (visual only) and, in the detailed look, the building visuals. The
    # solid building bodies were already built above (solids=False).
    if OSM_SCENE is not None:
        try:
            osm_world.build_bodies(OSM_SCENE, detailed=detailed, avoid=[tuple(TARGET_POSITION[:2])],
                                   solids=False)
        except Exception as exc:                    # drawing only: obstacles are already in place
            problems.append(f"map ground/building visuals not drawn ({type(exc).__name__}: {exc})")
    if problems:
        GEO_PROBLEMS.extend(problems)
        print(f"ERROR: {len(problems)} obstacle(s) could not be placed as planned:")
        for line in geo_obstacles.format_problems(problems):
            print("  -", line)

    if gui:
        if DETAILED_VISUALS:
            _scene_look(floor_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    return floor_id, drone_id, obstacle_ids


def _draw_tree(ob):
    """Detailed look of one tree: trunk + low-poly canopy as visual-only bodies on top of the
    tree's (invisible) collision body. Falls back to the plain cylinder + sphere."""
    k = ob.meta.get("index", 0)
    tv, lv = _tree_visuals(ob.meta.get("tree_kind", "deciduous"), k)
    if tv is None:
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=geo_obstacles.TRUNK_RADIUS,
                                 length=geo_obstacles.TRUNK_HEIGHT, rgbaColor=[0.4, 0.2, 0.05, 1])
        lv = p.createVisualShape(p.GEOM_SPHERE, radius=geo_obstacles.CANOPY_RADIUS,
                                 rgbaColor=[0.1, 0.45, 0.1, 1])
    p.createMultiBody(0, -1, tv, [ob.x, ob.y, ob.z])
    canopy_orn = p.getQuaternionFromEuler([0, 0, (k * 2.399) % (2 * 3.14159)])   # vary each tree
    p.createMultiBody(0, -1, lv, [ob.x, ob.y, ob.z + geo_obstacles.CANOPY_OFFSET_Z], canopy_orn)


def _tree_visuals(kind, seed):
    """Low-poly trunk + canopy visual shapes, or (None, None) if they can't be made."""
    try:
        import visual_assets as va
        leaf_rgba, trunk_rgba = va.tree_colours(kind, seed)
        trunk = p.createVisualShape(p.GEOM_CYLINDER, radius=0.13, length=2.5, rgbaColor=trunk_rgba,
                                    specularColor=LOW_SPECULAR)
        v, n, t = va.tree_canopy_mesh(kind, seed)
        canopy = p.createVisualShape(p.GEOM_MESH, vertices=[list(q) for q in v],
                                     indices=[i for tri in t for i in tri],
                                     normals=[list(q) for q in n], rgbaColor=leaf_rgba,
                                     specularColor=LOW_SPECULAR)
        if trunk < 0 or canopy < 0:
            return None, None
        return trunk, canopy
    except Exception as exc:
        print("Tree model not available, using simple tree:", exc)
        return None, None


def _scene_look(floor_id):
    """Shadows, sun position, matte ground, and hide the camera-preview panels (GUI only)."""
    for name, value in (("COV_ENABLE_SHADOWS", 1), ("COV_ENABLE_RGB_BUFFER_PREVIEW", 0),
                        ("COV_ENABLE_DEPTH_BUFFER_PREVIEW", 0),
                        ("COV_ENABLE_SEGMENTATION_MARK_PREVIEW", 0)):
        if hasattr(p, name):
            try:
                p.configureDebugVisualizer(getattr(p, name), value)
            except Exception:
                pass
    half = sim_config.MAP_HALF
    try:   # newer PyBullet: shadow map sized to the map so every building casts a shadow
        p.configureDebugVisualizer(lightPosition=[-0.8 * half, 0.5 * half, 2.5 * half],
                                   shadowMapResolution=4096, shadowMapWorldSize=int(3 * half))
    except Exception:
        try:
            p.configureDebugVisualizer(lightPosition=[-0.8 * half, 0.5 * half, 2.5 * half])
        except Exception:
            pass
    try:
        p.changeVisualShape(floor_id, -1, specularColor=LOW_SPECULAR)
    except Exception:
        pass


if __name__ == "__main__":
    print("Run main.py to launch the integrated simulation.")
