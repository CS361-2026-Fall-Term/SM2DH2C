"""Integrated single-drone PyBullet SAR demo.
Run: python main.py                  (mission setup window, then GUI; press START)
     python main.py --no-ui          (skip the setup window; use command-line options / defaults)
     python main.py --headless       (auto-start; no GUI, no setup window)
     python main.py --profile        (GUI; prints per-section timings every 5 s)
     python main.py --lat 45.5122 --lon -122.6587 --area 120   (OSM world around a last known position)
     python main.py --osm-file map.osm --lat 45.5122 --lon -122.6587  (offline, from an OSM export)
     python main.py --area 40                  (40 m x 40 m search map; layout and zones stretch to fit)
     python main.py --lat 45.5122 --lon -122.6587 --area 300 --map-size 60
                                               (300 real m of OSM squeezed into a 60 m map)
     python main.py --camera follow            (camera: locked | follow | free - see CAMERA_MODES)
     python main.py --simple-visuals           (flat colours, no textures/tree models/shadows: faster)
     python main.py --person child --exclude-zones AC   (PBI-03: missing person category, zones not searched)
     python main.py --headless --person -9 -9  (place the person at x=-9, y=-9 sim metres)
     python main.py --person-seed 42           (repeat a random person location; default: new random spot each run)
     python main.py --person child -9 -9       (category and position together)
     python main.py --resume                   (PBI-10: continue the last saved mission after a crash;
                                                see mission_state.py / MISSION_STATE.md)

After the zone plan ends, the drone keeps searching the most likely cells of the probability map
(map_search.py) until the person is found or the time limit is reached.

Search-area size:
  --area      how much ground the search covers, in real metres (MIN_AREA_M..MAX_AREA_M in sim_config).
              The sim map follows it 1:1 (true scale) up to MAX_MAP_SIZE; bigger areas are squeezed.
  --map-size  optional override for the sim map width (MIN_MAP_SIZE..MAX_MAP_SIZE).
"""
import argparse
import numpy as np
import random
import contextlib
import math
import time
import pybullet as p
import env_sim
import geo_obstacles
import operator_panel
import operator_setup
import osm_world
import sim_config
import terrain_prior
import map_search
from env_sim import build_world, FLIGHT_PATH, DRONE_START_POS, OBSTACLE_POSITIONS
from object_avoidance import Navigator, DT, V_MAX
from target_detection import MissingPerson, SearchMonitor, explain_method, LOG_MODES
from operator_guidance import Mission, Cmd, ZONES, ZONE_COMMANDS, scale_zones
from map_search import MapGuidedSearch, SightingConfirm
from probability_map import ProbabilityMap
from mission_state import (MissionStateStore, DEFAULT_PATH, setup_from_args,
                           apply_setup_to_args, describe, restore_belief)

# Mission logic (navigation, detection, probability map) still advances every DT (240 Hz).
# Everything that talks to the PyBullet GUI (button reads, pose push, stepSimulation, camera,
# heatmap, HUD text, sleeping) happens once per RENDER_EVERY logic steps: 4 -> 60 Hz.
RENDER_EVERY = 4
HUD_EVERY = 8                 # HUD text refresh, in render ticks (60 / 8 = 7.5 Hz); only re-sent if changed
HUD_LINE_SPACING = 0.045       # HUD line gap per metre of camera distance (debug text has no line breaks)
BUTTON_POLL_EVERY = 3         # operator buttons are polled every 3rd render tick (20 Hz)
FOLLOW_VIEW = (8.0, 45.0, -35.0)  # locked chase camera: distance (m), yaw (deg), pitch (deg)
# Camera modes, in the order the in-sim "Camera" button cycles through them:
#   locked - follows the drone at FOLLOW_VIEW; the user cannot move the camera
#   follow - follows the drone; the user can orbit (Ctrl/Alt + drag) and zoom (wheel) around it
#   free   - never moved by the program; the user has full mouse control
CAMERA_MODES = ("locked", "follow", "free")
CAMERA_HELP = {"locked": "following the drone, locked view (no user control)",
               "follow": "following the drone; Ctrl/Alt + drag to orbit, wheel to zoom",
               "free": "free; arrow keys / PageUp / PageDown to move, Ctrl/Alt + drag to rotate, "
                       "wheel to zoom (Ctrl/Alt + middle-drag also pans)"}
PAN_SPEED = 0.8               # free camera: arrow-key speed, in camera distances per second
HEATMAP_BUDGET = 8            # max heatmap tiles recolored per render tick


class Profiler:
    """Wall-time accounting per render tick (mean / worst ms per section). No-op unless enabled."""

    def __init__(self, enabled):
        self.enabled = enabled
        self.cur = {}
        self.stats = {}
        self.t_report = time.perf_counter()

    def section(self, name):
        return self._timer(name) if self.enabled else contextlib.nullcontext()

    @contextlib.contextmanager
    def _timer(self, name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.cur[name] = self.cur.get(name, 0.0) + time.perf_counter() - t0

    def end_tick(self):
        if not self.enabled:
            return
        for name, v in self.cur.items():
            s = self.stats.setdefault(name, [0.0, 0.0, 0])
            s[0] += v
            s[1] = max(s[1], v)
            s[2] += 1
        self.cur = {}
        if time.perf_counter() - self.t_report >= 5.0:
            self.report()

    def report(self):
        if not self.enabled or not self.stats:
            return
        parts = [f"{n} {s[0] / s[2] * 1000:.2f}/{s[1] * 1000:.1f}" for n, s in self.stats.items()]
        print("[profile, mean/worst ms per 60 Hz tick (tick budget 16.7)] " + " | ".join(parts))
        self.stats = {}
        self.t_report = time.perf_counter()


# Figures out how big the zone is from where everything in the env is palced, then sizes the probability map off that

def hud_line_step():
    """World-space offset between two HUD text lines for the current camera view
    (rounded so small camera moves don't cause re-sends)."""
    try:
        cam = p.getDebugVisualizerCamera()
        up, dist = cam[4], cam[10]
        n = math.sqrt(sum(c * c for c in up)) or 1.0
        gap = max(0.3, dist * HUD_LINE_SPACING)
        step = tuple(round(c / n * gap, 2) for c in up)
        return step if any(step) else (0.0, 0.0, round(gap, 2))
    except Exception:
        return (0.0, 0.0, 0.45)

def search_bounds(margins = 3.0):
    """Search area: a square centred on the LKP (0, 0) that covers the layout (start, trees,
    flight path) plus `margins`, clamped to the map. It is kept SQUARE AND CENTRED so every zone
    gets the same amount of area: an off-centre area cut the southern zone squares short, which
    gave them less probability and always ordered the northern zones first."""
    pts = [DRONE_START_POS] + list(OBSTACLE_POSITIONS) + list(FLIGHT_PATH)
    reach = max(max(abs(pt[0]), abs(pt[1])) for pt in pts) + margins
    # Clamped to the map, so OSM trees near the edge never grow the search area past it.
    h = float(min(reach, sim_config.MAP_HALF))   # plain float: numpy 2 prints np.float64(...)
    return ((-h, h), (-h, h))


# Zone label colours in the 3D view (PBI-09 operator feedback)
ZONE_COLOURS = {"next": [0.1, 0.85, 0.1], "planned": [0.3, 0.7, 1.0], "searched": [0.55, 0.55, 0.55],
                "excluded": [0.95, 0.2, 0.2]}


def zone_rect(xy, half):
    """(x0, y0, x1, y1) of the square drawn for a zone (draw_zone_markers). The probability map
    excludes exactly this square, so what the operator sees outlined is what is excluded."""
    h = zone_box_half(xy, half)
    return (xy[0] - h, xy[1] - h, xy[0] + h, xy[1] + h)


def apply_operator_command(mission, cmd, zone, nav, belief, reroute=None):
    """
    Validate and apply one live operator command (PBI-09), then make the drone follow it.
    Returns the CommandResult (accepted / rejected + message + validation time).
    A rejected command changes nothing; an accepted one re-plans the route only if the drone's
    destination changed, and an Investigate (reported sighting) raises the probability map there.
    reroute: function(nav, xy) used to re-plan; defaults to set_destination (tests pass a stub).
    """
    reroute = reroute or set_destination
    res = mission.handle(cmd, zone)
    if res.ok:
        if belief is not None:
            sigma = max(2.5, 0.15 * sim_config.MAP_HALF)
            if hasattr(belief, "apply_guidance"):
                # PBI-03: sightings raise the map, Do not search / Re-include / Return to plan
                # exclude and restore the zone's cells.
                rect = zone_rect(ZONES[res.zone], sim_config.MAP_HALF) if res.zone in ZONES else None
                belief.apply_guidance(res.command, res.zone, rect, res.sighting,
                                      sighting_sigma=sigma, sighting_weight=0.5)
            elif res.sighting is not None:
                belief.add_sighting(res.sighting, sigma=sigma, weight=0.5)
        if res.reroute:
            if mission.destination:
                reroute(nav, mission.destination)
            else:                                   # nothing left to search
                nav.route = []
                nav.state = "DONE"
    return res


# Compass direction of each zone from the map centre (x = east, y = north), for button labels.
def zone_direction(xy):
    ns = "north" if xy[1] > 0 else "south" if xy[1] < 0 else ""
    ew = "east" if xy[0] > 0 else "west" if xy[0] < 0 else ""
    return "-".join(d for d in (ns, ew) if d) or "centre"


def zone_box_half(xy, half):
    """Half-size of the square drawn for a zone: as big as fits around its centre
    without crossing the map edge or the centre lines (so zones never overlap)."""
    return max(1.0, min(abs(xy[0]), abs(xy[1]), half - abs(xy[0]), half - abs(xy[1])) - 0.4)


def draw_zone_markers(ids, mission, selected, half):
    """Ground outline + centre pole for every zone, coloured by status; the selected zone is
    drawn with thicker lines. Debug *lines* only (debug points break PyBullet's GUI font).
    Item ids are reused, so redrawing never adds new debug items."""
    for z, (zx, zy) in ZONES.items():
        colour = ZONE_COLOURS[zone_status(mission, z)]
        width = 7.0 if z == selected else 2.5
        h, zg = zone_box_half((zx, zy), half), 0.12
        c = [(zx - h, zy - h), (zx + h, zy - h), (zx + h, zy + h), (zx - h, zy + h)]
        segs = [([*c[i], zg], [*c[(i + 1) % 4], zg]) for i in range(4)]
        segs.append(([zx, zy, 0.0], [zx, zy, 3.0]))                 # centre pole
        old = ids.get(z, [])
        new = []
        for i, (a, b) in enumerate(segs):
            kw = {"replaceItemUniqueId": old[i]} if i < len(old) else {}
            new.append(p.addUserDebugLine(a, b, colour, width, **kw))
        ids[z] = new


def zone_status(mission, name):
    if name in mission.visited:
        return "searched"
    if name in mission.excluded:
        return "excluded"
    return "next" if mission.plan[:1] == [name] else "planned"


def map_status(searcher, mission):
    """Extra panel status text for the map-guided phase."""
    if mission.state == "SEARCHING" and searcher.active and searcher.target:
        return f"   |   Searching most likely cell {map_search.fmt(searcher.target)}"
    return ""


def set_destination(nav, xy):
    """Change high-level destination; Navigator still owns obstacle-safe routing."""
    nav.wps = [(nav.pos[0], nav.pos[1]), tuple(xy)]
    nav.wp_next = 1
    nav.route = [(xy[0], xy[1], 1)]
    nav.state = "NAV"
    nav.clear_ahead = float("inf")
    # Preserve horizontal speed across destination changes; Navigator's
    # acceleration limiter will smoothly adapt to the new route.
    nav.emergency_stops = 0
    nav.last_replan_t = -99
    nav.sense()      # Scan surroundings so the obstacle map knows what's nearby.
    nav.replan()     # A* from the current position to xy; falls back to a straight line if no path.
    nav.draw_route()


def start_mission(mission, nav, belief, exclude_zones=(), reroute=None):
    """Start the mission, then apply the zones the operator excluded in the setup window (PBI-03).
    They go through Mission.handle like a live "Do not search", so the plan, the zone colours and the
    probability map stay in sync and the same safety rules apply. Returns the start message."""
    reroute = reroute or set_destination
    first_start = mission.state == "IDLE"
    msg = mission.start()
    if not first_start:
        return msg
    if mission.destination:
        reroute(nav, mission.destination)
    for z in exclude_zones:
        res = apply_operator_command(mission, Cmd.AVOID, z, nav, belief, reroute)
        print(f"[setup] {res.message}")
    return msg


def build_initial_belief(belief, scene, category):
    """PBI-03: load the starting map (distance from the LKP x terrain) into the probability map.
    The LKP is the map centre (0, 0). Returns a warning for the operator, or None."""
    t0 = time.perf_counter()
    initial, warning = terrain_prior.build_initial_map(scene, belief, (0.0, 0.0), category,
                                                       sim_config.MAP_HALF)
    belief.set_initial(initial)
    print(f"Initial probability map: {category}, {belief.nx} x {belief.ny} cells, "
          f"built in {(time.perf_counter() - t0) * 1000:.1f} ms" + (f" ({warning})" if warning else ""))
    return warning


def _free_camera_keys(dt):
    """Free camera: arrow keys pan sideways / forward relative to where the camera looks,
    PageUp / PageDown move it up / down. (Mouse panning is Ctrl/Alt + middle-drag, which
    laptops and trackpads often can't do.)"""
    keys = p.getKeyboardEvents()
    if not isinstance(keys, dict):                # defensive: no keyboard data this frame
        return
    held = lambda k: k in keys and keys[k] & p.KEY_IS_DOWN
    right = held(p.B3G_RIGHT_ARROW) - held(p.B3G_LEFT_ARROW)
    ahead = held(p.B3G_UP_ARROW) - held(p.B3G_DOWN_ARROW)
    up = held(p.B3G_PAGE_UP) - held(p.B3G_PAGE_DOWN)
    if not (right or ahead or up):
        return
    cam = p.getDebugVisualizerCamera()
    yaw, pitch, dist, target = cam[8], cam[9], cam[10], cam[11]
    fx, fy = cam[5][0], cam[5][1]                 # camera forward direction, flattened to the ground
    n = math.hypot(fx, fy)
    if n < 1e-6:                                  # looking straight down: use the yaw instead
        fx, fy = -math.sin(math.radians(yaw)), math.cos(math.radians(yaw))
        n = 1.0
    fx, fy = fx / n, fy / n
    step = PAN_SPEED * max(dist, 1.0) * dt
    new_target = [target[0] + (fx * ahead + fy * right) * step,
                  target[1] + (fy * ahead - fx * right) * step,
                  max(0.0, target[2] + up * step)]
    p.resetDebugVisualizerCamera(cameraDistance=dist, cameraYaw=yaw, cameraPitch=pitch,
                                 cameraTargetPosition=new_target)


def parse_person(values):
    """--person accepts a category, a position X Y (sim metres), or both: 'child', '-9 -9',
    'child -9 -9'. Returns (category or None, (x, y) or None); raises ValueError."""
    if not values:
        return None, None
    values = list(values)
    category = None
    if values[0].lower() in terrain_prior.PERSON_CATEGORIES:
        category = values.pop(0).lower()
    if not values:
        return category, None
    if len(values) != 2:
        raise ValueError("give a category, a position X Y, or both (e.g. 'child -9 -9'); "
                         f"categories: {', '.join(terrain_prior.PERSON_CATEGORIES)}")
    try:
        xy = (float(values[0]), float(values[1]))
    except ValueError:
        raise ValueError(f"position must be two numbers, got {' '.join(values)}")
    if not all(math.isfinite(c) for c in xy):
        raise ValueError("position must be finite numbers")
    return category, xy


PERSON_START_CLEARANCE = 5.0   # m, a random person spot is at least this far from the drone start
PERSON_TREE_CLEARANCE = 1.0    # m from a trunk (closer = right under the canopy, never visible)


def random_person_spot(seed, x_range, y_range, scene=None, margin=1.0, belief=None):
    """Random (x, y) for the missing person inside the search area: clear of tree trunks,
    walls, OSM buildings and water, never in a cell the starting heatmap shows red (water,
    buildings, excluded zones), and not right next to the drone's start. Same seed = same spot.
    Uniform over the allowed area (not weighted by the map), so tests aren't biased toward it."""
    rng = random.Random(seed)
    red = belief.red_mask() if belief is not None else None
    for _ in range(20000):
        xy = (round(rng.uniform(x_range[0] + margin, x_range[1] - margin), 2),
              round(rng.uniform(y_range[0] + margin, y_range[1] - margin), 2))
        if math.dist(xy, DRONE_START_POS[:2]) < PERSON_START_CLEARANCE:
            continue
        if any(math.dist(xy, t[:2]) < PERSON_TREE_CLEARANCE for t in OBSTACLE_POSITIONS):
            continue
        if any(abs(xy[0] - wx) < hx + 0.5 and abs(xy[1] - wy) < hy + 0.5
               for wx, wy, hx, hy, _h in env_sim.WALLS):
            continue
        if red is not None:
            ix, iy = belief.to_cell(xy)
            if red[iy, ix]:
                continue
        if scene is not None and (not osm_world.is_free(scene, xy, margin=1.2) or
                                  any(osm_world.point_in_polygon(xy, w) for w in scene.water)):
            continue
        return xy
    return tuple(env_sim.TARGET_POSITION[:2])        # nowhere free: the original spot


def zones_by_probability(belief, half=None):
    """Zone names, most likely first: probability-map mass inside each zone's square.
    Ties keep the A-B-C-D order. Excluded zones have no mass, so they come last."""
    half = sim_config.MAP_HALF if half is None else half
    mass = {z: belief.zone_mass(zone_rect(c, half)) for z, c in ZONES.items()}
    # rounded so floating-point noise doesn't reorder zones that really tie (keeps A-B-C-D)
    return sorted(ZONES, key=lambda z: -round(mass[z], 3)), mass


def zone_terrain_lines(scene, belief, top=4):
    """One line per zone: what its terrain is made of (share of cells), to explain the order."""
    terrain = terrain_prior.terrain_grid(scene, belief)
    lines = []
    for z, c in ZONES.items():
        x0, y0, x1, y1 = zone_rect(c, sim_config.MAP_HALF)
        inside = (belief.cx >= x0) & (belief.cx <= x1) & (belief.cy >= y0) & (belief.cy <= y1)
        if not inside.any():
            continue
        codes, counts = np.unique(terrain[inside], return_counts=True)
        parts = sorted(zip(counts, codes), reverse=True)[:top]
        lines.append(f"Zone {z} ({zone_direction(c)}): " + ", ".join(
            f"{terrain_prior.TERRAIN_NAMES[int(k)]} {n / inside.sum():.0%}" for n, k in parts))
    return lines


def nearest_zone(xy):
    """(zone name, distance in m) of the zone centre closest to xy."""
    return min(((z, math.dist(c, xy)) for z, c in ZONES.items()), key=lambda zc: zc[1])


def _setup_prefill(args):
    """Command-line values the operator already gave, used to pre-fill the setup window."""
    pre = {}
    if args.lat is not None:
        pre.update(mode="osm", lat=args.lat, lon=args.lon)
    for key in ("area", "map_size", "max_time"):
        if getattr(args, key) is not None:
            pre[key] = getattr(args, key)
    if args.osm_file:
        pre.update(source="file", osm_file=args.osm_file)
    if args.seed:
        pre["seed"] = args.seed
    if args.simple_visuals:
        pre["visuals"] = "simple"
    if args.camera or args.free_camera:
        pre["camera"] = args.camera or "free"
    if args.person_category:
        pre["person_category"] = args.person_category
    if args.exclude_zones:
        pre["exclude_zones"] = args.exclude_zones
    return pre


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-time", type=float, default=None,
                        help="sim time limit in s (default 240, scaled up with map size)")
    parser.add_argument("--profile", action="store_true",
                        help="print per-section timings every 5 s (GUI mode)")
    parser.add_argument("--lat", type=float, help="last known position latitude (enables OSM world)")
    parser.add_argument("--lon", type=float, help="last known position longitude")
    parser.add_argument("--area", type=float, default=None,
                        help=f"width of the search area in real metres "
                             f"({sim_config.MIN_AREA_M:g}-{sim_config.MAX_AREA_M:g}, default "
                             f"{sim_config.BASE_MAP_SIZE:g}); sets the map size and the OSM area")
    parser.add_argument("--map-size", type=float, default=None,
                        help=f"override the sim map width in metres "
                             f"({sim_config.MIN_MAP_SIZE:g}-{sim_config.MAX_MAP_SIZE:g})")
    parser.add_argument("--osm-file", help="Overpass .json or .osm export to use instead of downloading")
    parser.add_argument("--seed", type=int, default=0, help="seed for trees scattered in woods")
    parser.add_argument("--camera", choices=CAMERA_MODES, default=None,
                        help="locked = follow the drone, no user control (default); follow = follow "
                             "the drone, user can orbit/zoom; free = user controls the camera")
    parser.add_argument("--free-camera", action="store_true", help="same as --camera free")
    parser.add_argument("--simple-visuals", action="store_true",
                        help="flat colours instead of textured buildings, tree models, sky and "
                             "shadows (faster on slow computers; sensing/navigation identical)")
    parser.add_argument("--no-dock", action="store_true",
                        help="don't dock the operator panel beside the 3D window")
    parser.add_argument("--no-ui", action="store_true",
                        help="skip the mission setup window and use the command-line options")
    parser.add_argument("--person", nargs="+", default=None, metavar="CATEGORY | X Y",
                        help="missing person category (" + ", ".join(terrain_prior.PERSON_CATEGORIES)
                             + "; sets the distance rings of the initial probability map, default "
                             f"{terrain_prior.DEFAULT_CATEGORY}) and/or where to place the person, "
                             "X Y in sim metres, e.g. --person -9 -9 or --person child -9 -9")
    parser.add_argument("--person-seed", type=int, default=None,
                        help="seed for the person's random location (default: a new random seed "
                             "each run, printed so the run can be repeated); ignored with --person X Y")
    parser.add_argument("--zone-order", choices=("map", "fixed"), default="map",
                        help="map = search the zones most likely first (probability map mass, "
                             "default); fixed = always A -> B -> C -> D")
    parser.add_argument("--exclude-zones", default=None,
                        help="zones not to search, set before the mission, e.g. AC")
    parser.add_argument("--detection-log", choices=LOG_MODES, default="scans",
                        help="detection results in the terminal (PBI-13): 'scans' = a line per camera "
                             "scan with the confidence while the person is in view (default), "
                             "'events' = only sightings and the result, 'off' = nothing")
    parser.add_argument("--resume", action="store_true",
                        help="continue the last saved mission (state, plan, history, drone position, "
                             "probability map) instead of starting a new one")
    parser.add_argument("--state-file", default=DEFAULT_PATH,
                        help="where the mission state is saved (default: mission_state.json here)")
    args = parser.parse_args()
    try:
        args.person_category, person_xy = parse_person(args.person)
    except ValueError as exc:
        parser.error(f"--person: {exc}")

    # ---- Saved mission state (PBI-10). With --resume the last saved state is loaded and the
    # operator's starting information is reused, so the same world (and person spot) is rebuilt.
    # Without it, any old state file is moved aside (mission_state.json.previous).
    store = MissionStateStore(args.state_file)
    resumed = None
    if args.resume:
        loaded = store.load()
        for problem in loaded.problems:
            print(f"[state] Skipped saved file - {problem}")
        if loaded.payload is None:
            print("[state] No usable saved mission found; starting a new mission.")
        elif loaded.payload.get("setup") is None:
            print("[state] The saved mission has no starting information; starting a new mission.")
        else:
            resumed = loaded.payload
            apply_setup_to_args(args, resumed["setup"])
            if resumed["setup"].get("person_xy") is not None:
                person_xy = tuple(resumed["setup"]["person_xy"])
            print(f"[state] Resuming saved mission ({loaded.source} file): {describe(resumed)}")
    else:
        old = store.load().payload
        if old is not None and old["mission"]["state"] == "SEARCHING":
            print("[state] An unfinished mission was saved earlier. Run with --resume to continue it; "
                  "starting a new mission (the old file is kept as mission_state.json.previous).")
        store.archive_previous()

    # ---- Mission setup window (issue #55): the operator enters the initial information here,
    # it is validated, and only then is the environment generated. Command-line values pre-fill it.
    setup_summary = None
    if not args.headless and not args.no_ui and resumed is None:
        try:
            cfg = operator_setup.ask(_setup_prefill(args))
        except operator_setup.SetupUnavailable as exc:
            print(f"Note: {exc}; using the command-line options instead.")
            cfg = False
        if cfg is None:
            print("Mission setup cancelled; simulation not started.")
            return
        if cfg:
            args.lat, args.lon, args.area = cfg["lat"], cfg["lon"], cfg["area"]
            args.map_size, args.osm_file, args.seed = cfg["map_size"], cfg["osm_file"], cfg["seed"]
            args.simple_visuals = cfg["simple_visuals"]
            args.camera = cfg.get("camera") or ("free" if cfg.get("free_camera") else "locked")
            args.max_time = cfg["max_time"]
            args.person_category = cfg.get("person_category") or args.person_category
            args.exclude_zones = cfg.get("exclude_zones") or args.exclude_zones
            setup_summary = cfg.get("summary", "")
            print("Mission setup accepted:", setup_summary)
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be given together")
    if args.osm_file and args.lat is None:
        parser.error("--osm-file needs --lat and --lon (the last known position)")
    person_category = args.person_category or terrain_prior.DEFAULT_CATEGORY
    try:
        exclude_zones = operator_setup.parse_zones(args.exclude_zones or "")
    except ValueError as exc:
        parser.error(f"--exclude-zones: {exc}")
    if len(exclude_zones) >= len(ZONES):
        parser.error("--exclude-zones: at least one zone must stay in the search")
    args.exclude_zones = "".join(exclude_zones) or None     # saved in this form (PBI-10)

    # ---- Search-area size: validate, then size everything BEFORE any world object is built.
    area = sim_config.BASE_MAP_SIZE if args.area is None else args.area
    if not sim_config.MIN_AREA_M <= area <= sim_config.MAX_AREA_M:
        parser.error(f"--area must be between {sim_config.MIN_AREA_M:g} and "
                     f"{sim_config.MAX_AREA_M:g} m (got {area:g})")
    if args.map_size is not None:
        if not sim_config.MIN_MAP_SIZE <= args.map_size <= sim_config.MAX_MAP_SIZE:
            parser.error(f"--map-size must be between {sim_config.MIN_MAP_SIZE:g} and "
                         f"{sim_config.MAX_MAP_SIZE:g} m (got {args.map_size:g})")
        map_size = args.map_size
    else:
        map_size = sim_config.clamp_map_size(area)      # true scale until MAX_MAP_SIZE
    sim_config.set_map_size(map_size)
    k = sim_config.layout_scale()
    env_sim.scale_layout(k)                             # start, trees, target, flight path
    scale_zones(k)                                      # zone centres A-D
    max_time = args.max_time if args.max_time is not None else 240.0 * max(1.0, k)
    if args.lat is not None:
        print(f"Map {map_size:g} m x {map_size:g} m showing {area:g} real m of OSM data "
              f"(scale {map_size / area:.3f}); zones {ZONES}")
    else:
        print(f"Map {map_size:g} m x {map_size:g} m (hand-placed layout x{k:.2f}); zones {ZONES}")

    env_sim.DETAILED_VISUALS = not args.simple_visuals
    scene = None                                        # OSM scene (None = practice world)
    if args.lat is not None:
        try:
            scene = env_sim.use_osm(args.lat, args.lon, area, args.osm_file,
                                    keep_clear=list(ZONES.values()), seed=args.seed)
        except osm_world.OverpassUnavailable as exc:
            raise SystemExit(str(exc))
        except geo_obstacles.GeoDataError as exc:
            # Malformed geographical input that cannot be used at all (unreadable file, impossible
            # position, ...): tell the operator what is wrong and stop cleanly, without a traceback.
            raise SystemExit(f"ERROR: the map data could not be used - {exc}\n"
                             "The simulation was not started. Fix the map file or the last known "
                             "position and run again, or leave out --lat/--lon/--osm-file to use "
                             "the practice world.")
        # Zone centres that landed on a building were moved to open ground; use the new spots.
        for name, centre in list(ZONES.items()):
            if tuple(centre) in scene.moved:
                ZONES[name] = scene.moved[tuple(centre)]
        print("Zones:", ZONES)
        if not (scene.buildings or scene.trees or scene.roads or scene.water or scene.surfaces):
            osm_empty = ("no map data around this LKP (0 buildings, trees, roads, water or ground "
                         "areas). If you picked 'Exported file' in the setup window, check that the "
                         "file covers this latitude/longitude; otherwise the terrain is all 'open'")
            print("WARNING: " + osm_empty)
    gui = not args.headless
    render_every = RENDER_EVERY if gui else 1       # headless: no GUI calls, nothing to decimate
    prof = Profiler(args.profile and gui)
    floor_id, drone_id, obstacle_ids = build_world(gui)
    # Fixed-step simulation; stepSimulation runs once per render tick, so give it the matching
    # step. Nothing here is dynamic (kinematic drone, static trees), so this only keeps
    # PyBullet's clock equal to wall time.
    p.setTimeStep(DT * render_every)
    p.setRealTimeSimulation(0)
    # Sanity-check for probability map
    x_range, y_range = search_bounds()
    print("Search area:", x_range, y_range)
    # pod is a per-second detection chance while a cell is in the camera view.
    # Keep the heatmap around 30 x 30 tiles: each tile is a PyBullet body, so 1 m cells on a
    # 60 m map (3600 bodies) would make startup and the GUI slow.
    cell = max(1.0, max(x_range[1] - x_range[0], y_range[1] - y_range[0]) / 30.0)
    belief = ProbabilityMap(x_range=x_range, y_range=y_range, cell_size=cell, pod=0.65)
    # PBI-03: start from the LKP + terrain map instead of a uniform one, with the zones excluded
    # in the setup window already shown in red before START.
    map_warning = build_initial_belief(belief, scene, person_category)
    for z in exclude_zones:
        belief.exclude_zone(z, zone_rect(ZONES[z], sim_config.MAP_HALF))
    # The person is placed after the map is built, so a random spot can avoid red cells.
    if person_xy is None:
        # No position given: hide the person at a random spot in the search area.
        person_seed = args.person_seed if args.person_seed is not None else random.randrange(10 ** 6)
        person_xy = random_person_spot(person_seed, x_range, y_range, scene=scene, belief=belief)
        print(f"Person placed at random (--person-seed {person_seed}); "
              f"repeat with --person-seed {person_seed} or --person {person_xy[0]:g} {person_xy[1]:g}")
    if person_xy is not None:
        half = sim_config.MAP_HALF
        if not (-half < person_xy[0] < half and -half < person_xy[1] < half):
            parser.error(f"--person: ({person_xy[0]:g}, {person_xy[1]:g}) is outside the "
                         f"{2 * half:g} m map (x and y must be between {-half:g} and {half:g})")
        if scene is not None:                       # OSM world: not inside a building
            free = osm_world.nearest_free_point(scene, person_xy, margin=1.2)
            if math.dist(free, person_xy) > 1e-6:
                print(f"Person moved from {map_search.fmt(person_xy)} to {map_search.fmt(free)} "
                      "(was inside a building)")
            person_xy = tuple(free)
        if belief.red_mask()[belief.to_cell(person_xy)[::-1]]:
            print(f"Warning: the person is in a red cell (water, building or excluded zone): "
                  "the map rates it as almost impossible, so the drone will search it last")
        z, d = nearest_zone(person_xy)
        print(f"Person placed at {map_search.fmt(person_xy)}, {d:.1f} m from the nearest zone "
              f"centre (zone {z})")
    person = MissingPerson.spawn(person_xy)
    # PBI-10: every save carries the starting information, so --resume rebuilds this same search.
    store.set_context(setup=setup_from_args(args, person_xy), zones=ZONES)
    # label_dz lifts the detection label above the Navigator's label so they don't overprint.
    monitor = SearchMonitor(person, gui=gui, label_dz=1.4, log_mode=args.detection_log)
    if args.detection_log != "off":
        print(explain_method())                         # PBI-13: what the numbers mean

    if person_xy is not None and not (x_range[0] <= person_xy[0] <= x_range[1]
                                      and y_range[0] <= person_xy[1] <= y_range[1]):
        print(f"Warning: the person is outside the search area {x_range} x {y_range}; "
              "the probability map will never send the drone there")
    if gui:
        # Build all heatmap tiles once, up front, with rendering off so there is no
        # mid-mission freeze when the first tiles are created.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        belief.draw_heatmap(budget=belief.grid.size)          # paint every tile once
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    mission = Mission()
    # PBI-13/14: the detector sends SIGHTING / LOST / FOUND to the shared mission as MissionEvents
    # (numeric distance, positions); the mission updates its detection state, logs them (panel
    # message log, save file) and the operator panel shows them.
    monitor.mission = mission
    if args.zone_order == "map" and resumed is None:   # a resumed mission keeps its saved order
        order, mass = zones_by_probability(belief)
        print("Zone order by probability: " + " -> ".join(f"{z} ({mass[z]:.0%})" for z in order))
        if scene is not None:
            for line in zone_terrain_lines(scene, belief):
                print("  " + line)
        mission.set_plan_order(order)
    if setup_summary is not None:
        mission.notify("SETUP", "Setup accepted - environment generated. Press START MISSION")
    if map_warning and resumed is None:
        mission.notify("MAP", f"Warning: {map_warning}. Press START MISSION")
    if env_sim.GEO_PROBLEMS and resumed is None:
        # Map objects that could not be read or placed were skipped (details printed above).
        mission.notify("MAP", f"Warning: {len(env_sim.GEO_PROBLEMS)} map object(s) could not be "
                              "read and were left out - see the console. Press START MISSION")
    # Two-point route is replaced when the mission starts.
    # obstacle_info tells the Navigator what each generated obstacle is (name, height, outline).
    nav = Navigator(drone_id, floor_id, person.body_id,
                    [DRONE_START_POS, [DRONE_START_POS[0], DRONE_START_POS[1], 2.5]], gui, print,
                    obstacle_info=env_sim.OBSTACLE_INFO, mission=mission)
    nav.route = []
    nav.state = "IDLE"
    nav.alt = 2.5
    if resumed is not None:
        # PBI-10: put the mission back as it was saved and send the drone on its way again.
        try:
            note = mission.restore(resumed["mission"])
        except ValueError as exc:                    # cannot happen after load(); belt and braces
            print(f"[state] Saved mission could not be restored ({exc}); starting a new one.")
        else:
            if note:
                print("[state] Note:", note)
            # Probability map: the zones excluded at the moment of the crash (or, before START,
            # in the setup window), then the saved cell values, so searched cells stay searched.
            belief.include_all_zones()
            for z in (mission.excluded if mission.state != "IDLE" else exclude_zones):
                belief.exclude_zone(z, zone_rect(ZONES[z], sim_config.MAP_HALF))
            if restore_belief(belief, resumed.get("belief")):
                print("[state] Probability map restored")
            drone = resumed.get("drone") or {}
            pos = drone.get("pos")
            half = sim_config.MAP_HALF
            if pos and abs(pos[0]) <= half and abs(pos[1]) <= half and 0.0 <= pos[2] <= 20.0:
                nav.pos[:] = pos                     # continue from where the drone was
            if drone.get("t") is not None:
                nav.t = min(float(drone["t"]), max_time)   # the time limit keeps counting
            if mission.state == "SEARCHING":
                if mission.destination:
                    set_destination(nav, mission.destination)
                else:                                # zone plan done: map-guided search resumes
                    nav.route = []
                    nav.state = "DONE"
            print(f"[state] Resumed: {mission.state}, plan {mission.plan}, searched "
                  f"{mission.visited}, excluded {mission.excluded}, sim time {nav.t:.1f} s")
    # Map-guided search (runs once the zone plan is finished). A viewpoint must be clear of
    # obstacles the drone has already sensed.
    def viewpoint_blocked(x, y):
        """A viewpoint must be clear of obstacles already sensed and of OSM buildings (even ones
        not sensed yet), or the drone flies into a building and has to climb / abort."""
        if nav.map.blocked(x, y, nav.alt):
            return True
        return scene is not None and not osm_world.is_free(scene, (x, y), margin=1.0)
    searcher = MapGuidedSearch(belief, OBSTACLE_POSITIONS, sim_config.MAP_HALF,
                               blocked=viewpoint_blocked, log=print, speed=V_MAX)
    confirmer = SightingConfirm()        # stop and look when the detector has a possible sighting
    yaw = 0.0
    previous_xy = nav.pos[:2]
    zone_line_ids = {}                   # zone name -> debug line ids of its outline + pole
    last_result = None                   # latest CommandResult, for the operator panel
    panel = None                         # operator_panel.OperatorPanel (GUI runs only)
    zone_names = list(ZONES)
    selected_zone = zone_names[0]
    zone_label_ids = {}
    last_zone_sig = None
    force_hud = False                    # re-send the HUD right away (operator command feedback)
    pending_confirm = None               # (time the command was read, result) until it is shown
    confirm_times_ms = []                # input -> on-screen confirmation, for every command
    status_id = None
    hud_pos = [-sim_config.MAP_HALF + 2, sim_config.MAP_HALF - 2, 5]   # top-left corner of the map
    last_info = None
    last_hud_step = None
    step_count = 0
    cam_pos = list(nav.pos)              # smoothed chase-camera target
    cam_mode = args.camera or ("free" if args.free_camera else "locked")    # see CAMERA_MODES
    cam_view = list(FOLLOW_VIEW)         # current (distance, yaw, pitch), eased toward FOLLOW_VIEW
    if gui:
        # PBI-09 operator controls live only in the Tk operator panel (operator_panel.py).
        # PyBullet's own side panel is hidden, so there is a single set of controls.
        panel, why = operator_panel.try_open(
            ZONES, list(Cmd), ZONE_COMMANDS,
            directions={z: zone_direction(xy) for z, xy in ZONES.items()})
        if panel is None:
            p.disconnect()
            raise SystemExit("The operator panel could not open (" + why + ").\n"
                             "It needs Tkinter, which comes with the standard Python installer "
                             "from python.org.")
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        panel.refresh(mission, selected_zone, lambda z: zone_status(mission, z), None, cam_mode)
        if not args.no_dock:
            print("Operator panel", panel.dock())
        # Starting view: the locked chase view, centred on the drone.
        p.resetDebugVisualizerCamera(cameraDistance=FOLLOW_VIEW[0], cameraYaw=FOLLOW_VIEW[1],
                                     cameraPitch=FOLLOW_VIEW[2], cameraTargetPosition=cam_pos)
    elif mission.state == "IDLE":
        start_mission(mission, nav, belief, exclude_zones)
    store.autosave(mission, nav.pos, force=True, belief=belief, sim_t=nav.t)   # first save

    next_tick = time.perf_counter()      # deadline for the next render tick (GUI pacing)
    tick_t0 = next_tick                  # start of the current tick, for the profiler
    try:
        while p.isConnected() and nav.t < max_time:
            render = step_count % render_every == 0

            # Operator panel, polled at 20 Hz: button presses come back as events, handled below.
            if gui and render and (step_count // render_every) % BUTTON_POLL_EVERY == 0:
                with prof.section("buttons"):
                    events = panel.poll()
                    if panel.closed:         # operator closed the panel (and confirmed): end
                        print("Operator panel closed - ending the simulation.")
                        break
                    for ev in events:
                        kind = ev[0]
                        if kind == "start":
                            start_mission(mission, nav, belief, exclude_zones)
                            force_hud = True
                        elif kind == "heat":
                            belief.set_heatmap_visible(not belief.heatmap_visible)
                        elif kind == "cam":
                            cam_mode = CAMERA_MODES[(CAMERA_MODES.index(cam_mode) + 1) % len(CAMERA_MODES)]
                            # glide from wherever the camera is now (no jump when the mode changes)
                            cam = p.getDebugVisualizerCamera()
                            cam_view[:] = [cam[10], cam[8], cam[9]]
                            cam_pos[:] = list(cam[11])
                            print("Camera:", CAMERA_HELP[cam_mode])
                        elif kind == "select":
                            if ev[1] != selected_zone:
                                selected_zone = ev[1]
                                last_result = None       # old result was for another zone
                                print(f"[operator] Zone {selected_zone} selected")
                            force_hud = True
                        elif kind == "cmd":
                            cmd = ev[1]
                            t_cmd = time.perf_counter()
                            res = apply_operator_command(
                                mission, cmd, selected_zone if cmd in ZONE_COMMANDS else None,
                                nav, belief)
                            print(f"[operator] {res.message} (validated in {res.validate_ms:.3f} ms)")
                            force_hud = True
                            pending_confirm = (t_cmd, res)
                            last_result = res
                    if not panel.closed:
                        # cheap: only widgets whose text/colour changed are touched
                        panel.refresh(mission, selected_zone, lambda z: zone_status(mission, z),
                                      last_result, cam_mode,
                                      f"Navigation: {nav.state}   |   Detection: {monitor.state} "
                                      f"({monitor.confidence:.2f})" + map_status(searcher, mission))

            with prof.section("logic"):
                holding = False
                if mission.state == "SEARCHING":
                    was = confirmer.active
                    holding = confirmer.update(monitor.state, monitor._current_estimate(),
                                               nav.pos, yaw, DT, nav.t)
                    if holding and not was:
                        print(f"[confirm] t={nav.t:.1f}s possible sighting - holding position "
                              "to confirm")
                    if holding:
                        nav.t += DT                    # hover: the route is kept, resumed after
                        nav.horizontal_speed = 0.0
                    elif nav.state in ("NAV", "CLIMB", "RTL"):
                        nav.update(DT)
                    else:
                        nav.t += DT
                    # A reached zone is marked visited only when the navigator completes its leg.
                    if nav.state == "DONE" and mission.destination:
                        mission.arrived()
                        if mission.destination:
                            set_destination(nav, mission.destination)
                    # The operator put a zone back in the plan: zones come first again.
                    if mission.destination and searcher.active:
                        searcher.cancel()
                    if nav.state == "ABORT":
                        mission.finish("ABORT", "NAV", "Navigation aborted: awaiting operator")
                    elif nav.state == "RTL_DONE":
                        mission.finish("COMPLETE", "NAV", "Returned to launch")
                    elif nav.state == "DONE" and not mission.destination and not holding:
                        # Map-guided phase: the zone plan is finished, keep searching the most
                        # likely cells of the probability map. The mission stays SEARCHING.
                        if not searcher.active:
                            print(f"[map] zone plan finished at t={nav.t:.1f}s - "
                                  "searching the probability map")
                        vp = searcher.update(nav.pos, yaw, DT, nav.t,
                                             sighting=monitor.state == "SIGHTING",
                                             in_view=lambda pt: monitor.in_frustum(nav.pos, pt))
                        if vp is not None:
                            set_destination(nav, vp)
                            msg = mission.search_cell(searcher.target)
                            print(f"[map] t={nav.t:6.1f}s {msg} (look from {map_search.fmt(vp)})")
                            force_hud = True

                else:
                    nav.t += DT

                # Point the nose along the actual movement, with gradual turns.
                dx = nav.pos[0] - previous_xy[0]
                dy = nav.pos[1] - previous_xy[1]
                if math.hypot(dx, dy) > 1e-6:
                    desired_yaw = math.atan2(dy, dx)
                    yaw_error = (desired_yaw - yaw + math.pi) % (2 * math.pi) - math.pi
                    max_turn = 1.5 * DT  # rad/s
                    yaw += max(-max_turn, min(max_turn, yaw_error))
                elif holding:
                    # confirming a possible sighting: face the detector's estimate
                    yaw = confirmer.heading
                    monitor.set_heading(yaw)
                elif searcher.phase == "LOOK" and mission.state == "SEARCHING":
                    # hovering at a map-guided viewpoint: turn the nose and camera to the target
                    yaw = searcher.heading
                    monitor.set_heading(yaw)
                previous_xy = nav.pos[:2]

            # Push the drone pose and step the world once per render tick, not every logic step.
            if render:
                with prof.section("pose+step"):
                    quat = p.getQuaternionFromEuler([0, 0, yaw])
                    p.resetBasePositionAndOrientation(drone_id, nav.pos, quat)
                    # This is a kinematic drone; don't repeatedly zero its velocity.
                    p.stepSimulation()

            with prof.section("logic"):
                if mission.state == "SEARCHING":
                    monitor.update(nav.pos, nav.t)
                    # Continuous, vectorized update every step (cheap enough now).
                    belief.observe_camera(monitor, nav.pos, DT, trees=OBSTACLE_POSITIONS)
                    if monitor.found:
                        if mission.state == "SEARCHING":   # the FOUND event normally did this already
                            mission.found()
                        nav.route = []
                        nav.state = "FOUND"

            if gui and render:
                # Heatmap recolors at most HEATMAP_BUDGET tiles per tick: each recolor is a
                # call into the GUI, so a small steady budget avoids frame-time spikes.
                with prof.section("heatmap"):
                    belief.draw_heatmap(budget=HEATMAP_BUDGET)

                # HUD text: only re-sent when the text actually changed. An operator command
                # forces an immediate refresh, so its confirmation appears on the next frame.
                if (step_count // render_every) % HUD_EVERY == 0 or force_hud:
                    force_hud = False
                    with prof.section("hud"):
                        nxt = (f"Zone {mission.plan[0]}" if mission.plan else
                               f"map cell {map_search.fmt(searcher.target)}" if searcher.active
                               else "none")
                        # PyBullet debug text ignores "\n", so each HUD line is its own
                        # text item, stacked downward. Only changed lines are re-sent.
                        info = [f"Mission: {mission.state} | Nav: {nav.state} | "
                                f"Detection: {monitor.state} ({monitor.confidence:.2f})",
                                f"Selected zone: {selected_zone} | Next: {nxt} | "
                                f"Plan: {mission.plan_text()}",
                                f"Excluded: {', '.join(mission.excluded) or 'none'} | "
                                f"Searched: {', '.join(mission.visited) or 'none'}",
                                mission.response or "",
                                # PBI-14: latest detection / obstacle event
                                (f"Latest event: {mission.latest_event.summary()}"
                                 if mission.latest_event else "Latest event: none")]
                        if last_info is None:
                            last_info = [None] * len(info)
                        if status_id is None:
                            status_id = [None] * len(info)
                        # Text is drawn at a fixed pixel size, so the gap between lines has
                        # to follow the camera: step "down the screen" (-camera up vector),
                        # scaled by camera distance. Re-place all lines when the view moves.
                        hud_step = hud_line_step()
                        if hud_step != last_hud_step:
                            last_hud_step = hud_step
                            last_info = [None] * len(info)
                        for i, line in enumerate(info):
                            if line == last_info[i]:
                                continue
                            last_info[i] = line
                            pos = [hud_pos[k] - i * hud_step[k] for k in range(3)]
                            kw = {"textColorRGB": [1, 1, 0], "textSize": 1.3}
                            if status_id[i] is not None:
                                kw["replaceItemUniqueId"] = status_id[i]
                            status_id[i] = p.addUserDebugText(line or " ", pos, **kw)
                        # zones in the 3D view: ground outline + pole + label, colour = status,
                        # thick outline and >> << = selected zone
                        sig = (tuple(mission.plan), tuple(mission.visited), tuple(mission.excluded),
                               selected_zone)
                        if sig != last_zone_sig:
                            last_zone_sig = sig
                            for z, (zx, zy) in ZONES.items():
                                st = zone_status(mission, z)
                                text = f"ZONE {z} - {st.upper()}"
                                if z == selected_zone:
                                    text = f">> {text} (selected) <<"
                                kw = {"textColorRGB": ZONE_COLOURS[st], "textSize": 1.8}
                                if z in zone_label_ids:
                                    kw["replaceItemUniqueId"] = zone_label_ids[z]
                                # above the pole and cruise height, so buildings don't hide it
                                zone_label_ids[z] = p.addUserDebugText(text, [zx, zy, 3.3], **kw)
                            draw_zone_markers(zone_line_ids, mission, selected_zone,
                                              sim_config.MAP_HALF)
                    if pending_confirm is not None:
                        t_cmd, res = pending_confirm
                        pending_confirm = None
                        ms = (time.perf_counter() - t_cmd) * 1000.0
                        confirm_times_ms.append(ms)
                        print(f"[operator] {'accepted' if res.ok else 'rejected'} - shown to operator "
                              f"{ms:.1f} ms after input")

                # Camera (see CAMERA_MODES). The target always eases toward the drone (low-passed so
                # tiny speed/direction changes don't read as stutter).
                #   locked: the program sets distance, angle and target every frame, so mouse drags
                #           and scrolls have no effect; after a mode change it glides to FOLLOW_VIEW.
                #   follow: only the target is set; distance and angle are read back from the GUI
                #           each frame, so the user can orbit and zoom around the drone.
                #   free:   the program only moves the camera when the user holds the arrow keys
                #           (pan) or PageUp/PageDown (up/down); the mouse controls everything else.
                if cam_mode == "free":
                    with prof.section("camera"):
                        _free_camera_keys(render_every * DT)
                else:
                    with prof.section("camera"):
                        for i in range(3):
                            cam_pos[i] += 0.2 * (nav.pos[i] - cam_pos[i])
                        if cam_mode == "locked":
                            cam_view[0] += 0.15 * (FOLLOW_VIEW[0] - cam_view[0])
                            cam_view[1] += 0.15 * ((FOLLOW_VIEW[1] - cam_view[1] + 180) % 360 - 180)
                            cam_view[2] += 0.15 * (FOLLOW_VIEW[2] - cam_view[2])
                        else:
                            cam = p.getDebugVisualizerCamera()
                            cam_view[:] = [cam[10], cam[8], cam[9]]
                        p.resetDebugVisualizerCamera(
                            cameraDistance=cam_view[0],
                            cameraYaw=cam_view[1],
                            cameraPitch=cam_view[2],
                            cameraTargetPosition=cam_pos
                        )

            # PBI-10: save right away when the mission changed (state, plan, zones, a new message),
            # and refresh the drone position + probability map every 0.5 s while searching.
            if render:
                with prof.section("save"):
                    store.autosave(mission, nav.pos, belief=belief, sim_t=nav.t)

            step_count += 1
            if not gui and mission.state in ("FOUND", "ABORT", "COMPLETE"):
                break
            # Pace once per render tick, at the end of the group of logic steps, so the next
            # render step (pose push, camera) lands on a steady 60 Hz deadline.
            if gui and step_count % render_every == 0:
                if prof.enabled:
                    prof.cur["TOTAL work"] = time.perf_counter() - tick_t0
                next_tick += render_every * DT
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()   # fell behind; don't try to catch up
                prof.end_tick()
                tick_t0 = time.perf_counter()
        # Time limit reached without a find: end the search and report where to look next.
        if p.isConnected() and nav.t >= max_time and mission.state == "SEARCHING":
            mission.time_up(searcher.most_likely())
    except KeyboardInterrupt:
        pass
    finally:
        # Final save on a normal exit / Ctrl+C / error (a hard crash is covered by the autosaves).
        store.autosave(mission, nav.pos, force=True, belief=belief, sim_t=nav.t)
        if store.saves:
            print(f"[state] Mission state saved to {store.path} ({store.saves} saves, slowest "
                  f"{store.max_save_s * 1000:.1f} ms)")
        if panel is not None:
            panel.close()
        prof.report()
        print("Mission:", mission.state, "| Navigation:", nav.state, f"| sim time {nav.t:.1f} s")
        print("Final report:", mission.response)
        if searcher.selections:
            print(f"Map-guided search: {len(searcher.selections)} targets selected, "
                  f"{sum(1 for r in searcher.log_rows if r[3])}/{len(searcher.log_rows)} looks "
                  "had the target cell in the camera view")
        print("Detection:", monitor.report())
        if args.detection_log != "off":
            print(monitor.format_report())             # PBI-13: readable detection summary
        closest = ("nothing sensed" if nav.min_sensed_clearance == float("inf")
                   else f"{nav.min_sensed_clearance:.2f} m")
        print(f"Obstacles: {len(nav.detected)} of {len(obstacle_ids)} generated obstacles detected | "
              f"closest sensed surface {closest} | unsafe-distance alerts {nav.unsafe_alerts} | "
              f"re-plans {nav.replans}")
        print("Events:", mission.history)
        if confirm_times_ms:
            print(f"Operator commands: {len(confirm_times_ms)} | slowest confirmation "
                  f"{max(confirm_times_ms):.1f} ms (target < 1000 ms)")
        p.disconnect()


if __name__ == "__main__":
    main()