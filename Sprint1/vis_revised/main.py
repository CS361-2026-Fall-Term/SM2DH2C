"""Integrated single-drone PyBullet SAR demo.
Run: python main.py                  (GUI; press START)
     python main.py --headless       (auto-start; no GUI)
     python main.py --profile        (GUI; prints per-section timings every 5 s)
     python main.py --lat 45.5122 --lon -122.6587 --area 120   (OSM world around a last known position)
     python main.py --osm-file map.osm --lat 45.5122 --lon -122.6587  (offline, from an OSM export)
     python main.py --area 40                  (40 m x 40 m search map; layout and zones stretch to fit)
     python main.py --lat 45.5122 --lon -122.6587 --area 300 --map-size 60
                                               (300 real m of OSM squeezed into a 60 m map)
     python main.py --free-camera              (camera does not follow the drone; move it with the mouse)
     python main.py --simple-visuals           (flat colours, no textures/tree models/shadows: faster)

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
import osm_world
import sim_config
from env_sim import build_world, FLIGHT_PATH, DRONE_START_POS, OBSTACLE_POSITIONS
from object_avoidance import Navigator, DT
from target_detection import MissingPerson, SearchMonitor
from operator_guidance import Mission, Cmd, ZONES, scale_zones
from probability_map import ProbabilityMap

# Mission logic (navigation, detection, probability map) still advances every DT (240 Hz).
# Everything that talks to the PyBullet GUI (button reads, pose push, stepSimulation, camera,
# heatmap, HUD text, sleeping) happens once per RENDER_EVERY logic steps: 4 -> 60 Hz.
RENDER_EVERY = 4
HUD_EVERY = 8                 # HUD text refresh, in render ticks (60 / 8 = 7.5 Hz); only re-sent if changed
BUTTON_POLL_EVERY = 3         # operator buttons are polled every 3rd render tick (20 Hz)
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
def search_bounds(margins = 3.0):
    pts = [DRONE_START_POS] + list(OBSTACLE_POSITIONS) + list(FLIGHT_PATH)
    xs = [pt[0] for pt in pts]
    ys = [pt[1] for pt in pts]
    # Clamped to the map, so OSM trees near the edge never grow the search area past it.
    half = sim_config.MAP_HALF
    return ((max(min(xs) - margins, -half), min(max(xs) + margins, half)),
            (max(min(ys) - margins, -half), min(max(ys) + margins, half)))
    # Creates a border w/ padding (3m)

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
    parser.add_argument("--free-camera", action="store_true",
                        help="start with the camera NOT following the drone (toggle in the GUI)")
    parser.add_argument("--simple-visuals", action="store_true",
                        help="flat colours instead of textured buildings, tree models, sky and "
                             "shadows (faster on slow computers; sensing/navigation identical)")
    args = parser.parse_args()
    if (args.lat is None) != (args.lon is None):
        parser.error("--lat and --lon must be given together")
    if args.osm_file and args.lat is None:
        parser.error("--osm-file needs --lat and --lon (the last known position)")

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
    if gui:
        # Build all heatmap tiles once, up front, with rendering off so there is no
        # mid-mission freeze when the first tiles are created.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        belief.draw_heatmap()
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    mission = Mission()
    # Two-point route is replaced when the mission starts.
    nav = Navigator(drone_id, floor_id, person.body_id,
                    [DRONE_START_POS, [DRONE_START_POS[0], DRONE_START_POS[1], 2.5]], gui, print)
    nav.route = []
    nav.state = "IDLE"
    nav.alt = 2.5
    yaw = 0.0
    previous_xy = nav.pos[:2]
    start_button = None
    heat_button = None
    buttons = {}
    last = {}
    status_id = None
    hud_pos = [-sim_config.MAP_HALF + 2, sim_config.MAP_HALF - 2, 5]   # top-left corner of the map
    last_info = None
    step_count = 0
    cam_pos = list(nav.pos)              # smoothed chase-camera target
    follow_cam = not args.free_camera    # True: camera tracks the drone; False: user moves it freely
    cam_button = None
    if gui:
        start_button = p.addUserDebugParameter("START MISSION", 1, 0, 0)
        buttons = {cmd: p.addUserDebugParameter(cmd.value, 1, 0, 0) for cmd in Cmd}
        heat_button = p.addUserDebugParameter("Show / hide heatmap", 1, 0, 0)
        cam_button = p.addUserDebugParameter("Camera: follow drone / free", 1, 0, 0)
        last = {button: p.readUserDebugParameter(button)
                for button in [start_button, heat_button, cam_button, *buttons.values()]}
        # Starting view: same chase view as before, centred on the drone.
        p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=45, cameraPitch=-35,
                                     cameraTargetPosition=cam_pos)
    else:
        mission.start()
        set_destination(nav, mission.destination)

    next_tick = time.perf_counter()      # deadline for the next render tick (GUI pacing)
    tick_t0 = next_tick                  # start of the current tick, for the profiler
    try:
        while p.isConnected() and nav.t < max_time:
            render = step_count % render_every == 0

            # Operator buttons. A slider keeps its value, so polling at 20 Hz loses nothing
            # (this used to be ~1200 reads per second).
            if gui and render and (step_count // render_every) % BUTTON_POLL_EVERY == 0:
                with prof.section("buttons"):
                    start_value = p.readUserDebugParameter(start_button)
                    if start_value != last[start_button]:
                        last[start_button] = start_value
                        if mission.state == "IDLE":
                            mission.start()
                            set_destination(nav, mission.destination)
                        else:
                            mission.start()
                    heat_value = p.readUserDebugParameter(heat_button)
                    if heat_value != last[heat_button]:
                        last[heat_button] = heat_value
                        belief.set_heatmap_visible(not belief.heatmap_visible)
                    cam_value = p.readUserDebugParameter(cam_button)
                    if cam_value != last[cam_button]:
                        last[cam_button] = cam_value
                        follow_cam = not follow_cam
                        print("Camera:", "following the drone" if follow_cam else
                              "free (mouse: Ctrl/Alt + drag to rotate, middle-drag to pan, wheel to zoom)")
                    for cmd, button in buttons.items():
                        value = p.readUserDebugParameter(button)
                        if value != last[button]:
                            last[button] = value
                            old_plan = list(mission.plan)
                            mission.handle(cmd)
                            if mission.state == "SEARCHING" and mission.plan != old_plan:
                                if mission.destination:
                                    set_destination(nav, mission.destination)
                                else:
                                    nav.route = []
                                    nav.state = "DONE"

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

                # HUD text: only re-sent when the text actually changed.
                if (step_count // render_every) % HUD_EVERY == 0:
                    with prof.section("hud"):
                        info = (f"Mission: {mission.state} | Nav: {nav.state} | "
                                f"Detection: {monitor.state} ({monitor.confidence:.2f})\n"
                                f"Destination: {mission.destination} | Plan: {mission.plan}\n"
                                f"{mission.response}")
                        if info != last_info:
                            last_info = info
                            kw = {"textColorRGB": [1, 1, 0], "textSize": 1.3}
                            if status_id is not None:
                                kw["replaceItemUniqueId"] = status_id
                            status_id = p.addUserDebugText(info, hud_pos, **kw)

                # Follow mode: only the camera's TARGET tracks the drone (low-passed so tiny changes
                # in speed or direction don't read as stutter). Zoom, yaw and pitch are read back
                # from the GUI each frame, so the user can still orbit and zoom with the mouse.
                # Free mode: the camera is never touched, so the mouse has full control.
                if follow_cam:
                    with prof.section("camera"):
                        for i in range(3):
                            cam_pos[i] += 0.2 * (nav.pos[i] - cam_pos[i])
                        cam = p.getDebugVisualizerCamera()
                        p.resetDebugVisualizerCamera(
                            cameraDistance=cam[10],
                            cameraYaw=cam[8],
                            cameraPitch=cam[9],
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
        prof.report()
        print("Mission:", mission.state, "| Navigation:", nav.state)
        print("Detection:", monitor.report())
        print("Events:", mission.history)
        p.disconnect()


if __name__ == "__main__":
    main()
