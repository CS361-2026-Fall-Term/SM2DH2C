"""Integrated single-drone PyBullet SAR demo.
Run: python main.py                  (GUI; press START)
     python main.py --headless       (auto-start; no GUI)
     python main.py --profile        (GUI; prints per-section timings every 5 s)
"""
import argparse
import contextlib
import math
import time
import pybullet as p
from env_sim import build_world, FLIGHT_PATH, DRONE_START_POS, OBSTACLE_POSITIONS, LAST_WORLD_INFO
from object_avoidance import Navigator, DT
from target_detection import MissingPerson, SearchMonitor
from operator_guidance import Mission, Cmd
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
    return ((min(xs) - margins, max(xs) + margins), (min(ys) - margins, max(ys) + margins))
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
    parser.add_argument("--max-time", type=float, default=240)
    parser.add_argument("--profile", action="store_true",
                        help="print per-section timings every 5 s (GUI mode)")
    parser.add_argument("--prompt", action="store_true",
                        help="ask for last known position (lat/lon) and search radius, "
                             "then build the environment from OpenStreetMap")
    parser.add_argument("--json-in", help="rebuild environment from a saved environment.json")
    args = parser.parse_args()
    gui = not args.headless
    render_every = RENDER_EVERY if gui else 1       # headless: no GUI calls, nothing to decimate
    prof = Profiler(args.profile and gui)
    if args.prompt:
        from osm_parser import prompt_for_lkp
        lat, lon, radius = prompt_for_lkp()
        floor_id, drone_id, obstacle_ids = build_world(gui, lat=lat, lon=lon, radius_m=radius)
    else:
        floor_id, drone_id, obstacle_ids = build_world(gui, json_in=args.json_in)
    geo_mode = LAST_WORLD_INFO.get("mode") == "geographic"
    if geo_mode:
        print("[main] Geographic mode activated:", LAST_WORLD_INFO, flush=True)
        if LAST_WORLD_INFO.get("buildings", 0) + LAST_WORLD_INFO.get("road_segments", 0) == 0:
            print("[main] WARNING: No buildings or roads were generated from OSM for this area. "
                  "Try a different location or larger radius.", flush=True)
        if LAST_WORLD_INFO.get("radius_m", 0) > 13:
            print("[main] NOTE: The current Navigator's fixed +/-13 m A* grid does not cover "
                  "the entire geographical environment.", flush=True)
    else:
        print("[main] Running original scene. To activate OSM, set SAR_LAT and SAR_LON.", flush=True)
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
    # Creating a 1 m tile for every cell in a 100+ m OSM area can freeze the GUI.
    # Cap the grid to ~45 cells on its widest axis (roughly <= 2,025 visual tiles).
    span_x, span_y = x_range[1] - x_range[0], y_range[1] - y_range[0]
    cell_size = max(1.0, math.ceil(max(span_x, span_y) / 45.0)) if geo_mode else 1.0
    print(f"[main] Probability-map cell size: {cell_size:.0f} m; "
          f"approx. {math.ceil(span_x/cell_size) * math.ceil(span_y/cell_size)} heatmap tiles.",
          flush=True)
    belief = ProbabilityMap(x_range=x_range, y_range=y_range, cell_size=cell_size, pod=0.65)
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
    buttons = {}
    last = {}
    status_id = None
    last_info = None
    step_count = 0
    cam_pos = list(nav.pos)              # smoothed chase-camera target
    if gui:
        start_button = p.addUserDebugParameter("START MISSION", 1, 0, 0)
        buttons = {cmd: p.addUserDebugParameter(cmd.value, 1, 0, 0) for cmd in Cmd}
        last = {button: p.readUserDebugParameter(button)
                for button in [start_button, *buttons.values()]}
    else:
        mission.start()
        set_destination(nav, mission.destination)

    next_tick = time.perf_counter()      # deadline for the next render tick (GUI pacing)
    tick_t0 = next_tick                  # start of the current tick, for the profiler
    try:
        while p.isConnected() and nav.t < args.max_time:
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
                            status_id = p.addUserDebugText(info, [-11, 11, 5], **kw)

                # Chase camera, low-passed so tiny per-step changes in speed or direction
                # don't read as drone stutter.
                with prof.section("camera"):
                    for i in range(3):
                        cam_pos[i] += 0.2 * (nav.pos[i] - cam_pos[i])
                    # Do not overwrite the environment builder's wide map view
                    # with the old 8-meter close-up camera in geographic mode.
                    distance = (max(18.0, min(250.0, LAST_WORLD_INFO["radius_m"] * 1.5))
                                if geo_mode else 8.0)
                    target = [0, 0, 0.5] if geo_mode else cam_pos
                    p.resetDebugVisualizerCamera(
                        cameraDistance=distance,
                        cameraYaw=45,
                        cameraPitch=-50 if geo_mode else -35,
                        cameraTargetPosition=target
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
