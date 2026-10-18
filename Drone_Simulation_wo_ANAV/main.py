"""Integrated single-drone PyBullet SAR demo.
Run: python main.py                  (GUI; press START)
     python main.py --headless       (auto-start; no GUI)
"""
import argparse
import math
import time
import pybullet as p
from env_sim import build_world
from ObjectAvoidance import Navigator, DT, DRONE_START_POS, OBSTACLE_POSITIONS, FLIGHT_PATH
from TargetDetection import MissingPerson, SearchMonitor
from operator_guidance_spike import Mission, Cmd, ZONES, ARRIVE_DIST
from probability_map import ProbabilityMap
from DSSE import DroneSwarmSearch

# CONSTANTS
MAX_BELIEF_LEGS = 20

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
    nav.sense()     # scans so that the obstacle map knows what's around
    nav.replan()   # A* from the current position to xy; falls back to a straight line if theres no path
    nav.draw_route() # reroute

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-time", type=float, default=240)
    args = parser.parse_args()
    gui = not args.headless
    floor_id, drone_id, obstacles = build_world(gui)
    # Use a deterministic fixed-step simulation; the controller advances by DT.
    p.setTimeStep(DT)
    p.setRealTimeSimulation(0)
    person = MissingPerson.spawn()
    monitor = SearchMonitor(person, gui=gui)
    
    # Sanity-check for probability map    
    x_range, y_range = search_bounds()
    print("Search area:", x_range, y_range)
    belief = ProbabilityMap(x_range=x_range, y_range=y_range, pod=0.65)

    # Seed the belief map with DSSE's probability model around the last known position
    GRID = 40
    lkp = (person.position[0] - 3, person.position[1] + 2)   # last known position: a few m off the truth
    col = int((lkp[0] - x_range[0]) / (x_range[1] - x_range[0]) * GRID)
    row = int((lkp[1] - y_range[0]) / (y_range[1] - y_range[0]) * GRID)
    dsse = DroneSwarmSearch(grid_size=GRID, render_mode="ansi", drone_amount=1,
                            person_initial_position=(col, row), vector=(0.3, 0.3),
                            timestep_limit=10, pre_render_time=0)
    obs, _ = dsse.reset(options={"drones_positions": [(0, 0)]})
    belief.set_prior(obs[dsse.get_agents()[0]][1])
    dsse.close()
    
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
    step_count = 0
    belief_legs = 0

    if gui:
        start_button = p.addUserDebugParameter("START MISSION", 1, 0, 0)
        buttons = {cmd: p.addUserDebugParameter(cmd.value, 1, 0, 0) for cmd in Cmd}
        last = {button: p.readUserDebugParameter(button)
                for button in [start_button, *buttons.values()]}
    else:
        mission.start()
        set_destination(nav, mission.destination)

    try:
        while p.isConnected() and nav.t < args.max_time:
            if gui:
                if p.readUserDebugParameter(start_button) != last[start_button]:
                    last[start_button] = p.readUserDebugParameter(start_button)
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
                    belief.mark_searched(nav.pos[:2], belief.sensor_radius)
                    belief_legs += 1
                    if belief_legs > MAX_BELIEF_LEGS:
                        target = belief.best_target(nav.pos[:2])
                        mission.state = "COMPLETE"
                        mission.response = f'Search plan complete; person not confirmed, but most likely in cell {target}'
                    else:
                        target = belief.best_target(nav.pos[:2], distance_weight=0.1)
                        set_destination(nav, target)                # A* plans the route
                        mission.response = f"Belief search leg {belief_legs}: heading to {target}"
                    
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
            quat = p.getQuaternionFromEuler([0, 0, yaw])
            p.resetBasePositionAndOrientation(drone_id, nav.pos, quat)
            # This is a kinematic drone; don't repeatedly zero its velocity.
            p.stepSimulation()
            if mission.state == "SEARCHING":
                monitor.update(nav.pos, nav.t)
                monitor.update(nav.pos, nav.t)
                belief.observe_camera(monitor, nav.pos, trees = OBSTACLE_POSITIONS)
                if monitor.found:
                    mission.found()
                    nav.route = []
                    nav.state = "FOUND"

            if gui and step_count % 30 == 0:
                belief.draw()
                info = (f"Mission: {mission.state} | Nav: {nav.state} | "
                        f"Detection: {monitor.state} ({monitor.confidence:.2f})\n"
                        f"Destination: {mission.destination} | Plan: {mission.plan}\n"
                        f"{mission.response}")
                kw = {"textColorRGB": [1, 1, 0], "textSize": 1.3}
                if status_id is not None:
                    kw["replaceItemUniqueId"] = status_id
                status_id = p.addUserDebugText(info, [-11, 11, 5], **kw)

            # Update the chase camera more frequently so its target doesn't
            # jump every 0.125 s. A slightly wider view also shows the drone.
            if gui and step_count % 4 == 0:
                p.resetDebugVisualizerCamera(
                    cameraDistance=8,
                    cameraYaw=45,
                    cameraPitch=-35,
                    cameraTargetPosition=nav.pos
                )

            step_count += 1
            if not gui and mission.state in ("FOUND", "ABORT", "COMPLETE"):
                break
            if gui:
                time.sleep(DT)
    except KeyboardInterrupt:
        pass
    finally:
        print("Mission:", mission.state, "| Navigation:", nav.state)
        print("Detection:", monitor.report())
        print("Events:", mission.history)
        p.disconnect()


if __name__ == "__main__":
    main()
