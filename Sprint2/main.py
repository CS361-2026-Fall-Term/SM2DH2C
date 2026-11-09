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

Search-area size:
  --area      how much ground the search covers, in real metres (MIN_AREA_M..MAX_AREA_M in sim_config).
              The sim map follows it 1:1 (true scale) up to MAX_MAP_SIZE; bigger areas are squeezed.
  --map-size  optional override for the sim map width (MIN_MAP_SIZE..MAX_MAP_SIZE).
"""
import argparse
import contextlib
import math
import time
import pybullet as p
import env_sim
import operator_panel
import operator_setup
import osm_world
import sim_config
import terrain_prior
from env_sim import build_world, FLIGHT_PATH, DRONE_START_POS, OBSTACLE_POSITIONS
from object_avoidance import Navigator, DT
from target_detection import MissingPerson, SearchMonitor
from operator_guidance import Mission, Cmd, ZONES, ZONE_COMMANDS, scale_zones
from probability_map import ProbabilityMap

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
    pts = [DRONE_START_POS] + list(OBSTACLE_POSITIONS) + list(FLIGHT_PATH)
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    # Clamped to the map, so OSM trees near the edge never grow the search area past it.
    half = sim_config.MAP_HALF
    return ((max(min(xs) - margins, -half), min(max(xs) + margins, half)),
            (max(min(ys) - margins, -half), min(max(ys) + margins, half)))
    # Creates a border w/ padding (3m)

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
    if args.person:
        pre["person_category"] = args.person
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
    parser.add_argument("--person", choices=list(terrain_prior.PERSON_CATEGORIES), default=None,
                        help="missing person category; sets the distance rings of the initial "
                             f"probability map (default {terrain_prior.DEFAULT_CATEGORY})")
    parser.add_argument("--exclude-zones", default=None,
                        help="zones not to search, set before the mission, e.g. AC")
    args = parser.parse_args()

    # ---- Mission setup window (issue #55): the operator enters the initial information here,
    # it is validated, and only then is the environment generated. Command-line values pre-fill it.
    setup_summary = None
    if not args.headless and not args.no_ui:
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
            args.person = cfg.get("person_category") or args.person
            args.exclude_zones = cfg.get("exclude_zones") or args.exclude_zones
            setup_summary = cfg.get("summary", "")
            print("Mission setup accepted:", setup_summary)
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be given together")
    if args.osm_file and args.lat is None:
        parser.error("--osm-file needs --lat and --lon (the last known position)")
    person_category = args.person or terrain_prior.DEFAULT_CATEGORY
    try:
        exclude_zones = operator_setup.parse_zones(args.exclude_zones or "")
    except ValueError as exc:
        parser.error(f"--exclude-zones: {exc}")
    if len(exclude_zones) >= len(ZONES):
        parser.error("--exclude-zones: at least one zone must stay in the search")

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
        # Zone centres that landed on a building were moved to open ground; use the new spots.
        for name, centre in list(ZONES.items()):
            if tuple(centre) in scene.moved:
                ZONES[name] = scene.moved[tuple(centre)]
        print("Zones:", ZONES)
    gui = not args.headless
    render_every = RENDER_EVERY if gui else 1       # headless: no GUI calls, nothing to decimate
    prof = Profiler(args.profile and gui)
    floor_id, drone_id, obstacle_ids = build_world(gui)
    # Fixed-step simulation; stepSimulation runs once per render tick, so give it the matching
    # step. Nothing here is dynamic (kinematic drone, static trees), so this only keeps
    # PyBullet's clock equal to wall time.
    p.setTimeStep(DT * render_every)
    p.setRealTimeSimulation(0)
    person = MissingPerson.spawn()
    # label_dz lifts the detection label above the Navigator's label so they don't overprint.
    monitor = SearchMonitor(person, gui=gui, label_dz=1.4)

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
    if gui:
        # Build all heatmap tiles once, up front, with rendering off so there is no
        # mid-mission freeze when the first tiles are created.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        belief.draw_heatmap()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    mission = Mission()
    if setup_summary is not None:
        mission.response = "Setup accepted - environment generated. Press START MISSION"
    if map_warning:
        mission.response = f"Warning: {map_warning}. Press START MISSION"
    # Two-point route is replaced when the mission starts.
    nav = Navigator(drone_id, floor_id, person.body_id,
                    [DRONE_START_POS, [DRONE_START_POS[0], DRONE_START_POS[1], 2.5]], gui, print)
    nav.route = []
    nav.state = "IDLE"
    nav.alt = 2.5
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
    else:
        start_mission(mission, nav, belief, exclude_zones)

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
                                      f"({monitor.confidence:.2f})")

            with prof.section("logic"):
                if mission.state == "SEARCHING":
                    if nav.state in ("NAV", "CLIMB", "RTL"):
                        nav.update(DT)
                    else:
                        nav.t += DT
                    # A reached zone is marked visited only when the navigator completes its leg.
                    if nav.state == "DONE" and mission.destination:
                        mission.arrived()
                        if mission.destination:
                            set_destination(nav, mission.destination)
                    # A completed plan remains completed; don't restart it.
                    if nav.state == "ABORT":
                        mission.state = "ABORT"
                        mission.response = "Navigation aborted: awaiting operator"
                    elif nav.state == "RTL_DONE":
                        mission.state = "COMPLETE"
                        mission.response = "Returned to launch"
                    elif nav.state == "DONE" and not mission.destination:
                        target = belief.best_target(nav.pos[:2], distance_weight=0.1)
                        set_destination(nav, target)
                        mission.state = "COMPLETE"
                        mission.response = f'Search plan complete; person not confirmed, but most likely in cell {target}'

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
                        nxt = f"Zone {mission.plan[0]}" if mission.plan else "none"
                        # PyBullet debug text ignores "\n", so each HUD line is its own
                        # text item, stacked downward. Only changed lines are re-sent.
                        info = [f"Mission: {mission.state} | Nav: {nav.state} | "
                                f"Detection: {monitor.state} ({monitor.confidence:.2f})",
                                f"Selected zone: {selected_zone} | Next: {nxt} | "
                                f"Plan: {mission.plan_text()}",
                                f"Excluded: {', '.join(mission.excluded) or 'none'} | "
                                f"Searched: {', '.join(mission.visited) or 'none'}",
                                mission.response or ""]
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
    except KeyboardInterrupt:
        pass
    finally:
        if panel is not None:
            panel.close()
        prof.report()
        print("Mission:", mission.state, "| Navigation:", nav.state)
        print("Detection:", monitor.report())
        print("Events:", mission.history)
        if confirm_times_ms:
            print(f"Operator commands: {len(confirm_times_ms)} | slowest confirmation "
                  f"{max(confirm_times_ms):.1f} ms (target < 1000 ms)")
        p.disconnect()


if __name__ == "__main__":
    main()