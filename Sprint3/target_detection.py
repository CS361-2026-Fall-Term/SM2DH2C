"""
Contributor: Chase Vanderlip, Diya Deepu
File Description: Creates missing person and supports "Found" detection.
Two parts, both usable on their own or inside another script: 
MissingPerson - The casualty inside the simulation: a humanoid body (same model, scale and 
                pose as env_sim.py), with helpers for placement (fixed, random, or lying 
                under a tree canopy) and visibility sample points.
SearchMonitor - The mechanism that decides when the person has been FOUND. A simulated downward-
                tilted camera (field of view, max range) looks for the person. 
                For every camera frame it:
                1. Checks which of the person's sample points (head / torso / feet ...)
                   are inside the camera frustum.
                2. Ray-casts to each one, so trees and canopies hide the person (occlusion).
                   A point counts as seen when nothing else is in the way (the ray hits the
                   person, or gets to the point without hitting anything): the humanoid's thin
                   limbs leave gaps, and a ray through such a gap is not hidden by anything.
                3. Turns "how much of the person is visible, from how far" into a score.
                4. Accumulates the score into a confidence value that also decays when the
                   person is out of view (so one lucky glimpse is not enough).
                   A CLEAR view (most of the person visible, nothing in front, within
                   CLEAR_VIEW_RANGE) for CLEAR_VIEW_FRAMES frames in a row is enough on its own:
                   the person is found right away instead of waiting for the confidence.
                5. Fuses noisy position measurements into a location estimate.
                States:  SEARCHING -> SIGHTING (possible) -> FOUND (latched).
                When FOUND, it records time / drone position / estimated location and calls
                on_found(monitor) callback.
"""
import argparse
import math
import os
import random
import sys
import time
import pybullet as p
import pybullet_data
import sim_config
from env_sim import DRONE_SCALE, OBSTACLE_POSITIONS, TARGET_POSITION, TARGET_ORIENTATION_EULER, TARGET_SCALE, FLIGHT_PATH



DT = 1.0 / 240.0
FLIGHT_SPEED = 2.0             # m/s for the standalone demo

# camera model
CAM_HFOV_DEG = 70.0
CAM_VFOV_DEG = 70.0
CAM_PITCH_DEG = -40.0          # tilted down from the direction of travel
CAM_RANGE = 8.0                # m, beyond this the person is too small to recognise
CAM_FPS = 10.0
RAY_START_OFFSET = 0.30        # rays start outside the drone's own body
RAY_END_PAD = 0.25             # rays end slightly behind the sample point so they hit the person

# detection / confirmation
MIN_VISIBLE_FRACTION = 0.20    # at least this share of sample points must be seen for a frame to count
CONF_GAIN = 1.0                # confidence per second at a perfect score
# Range quality: how much one frame counts at a given distance. Full up to FULL_QUALITY_RANGE,
# then falling linearly to MIN_QUALITY at CAM_RANGE (the person gets smaller in the image).
# (It used to fall from the drone outwards: 1 - (d / 8)^2, so at 5 m a frame only counted 0.6.)
FULL_QUALITY_RANGE = 5.0       # m
MIN_QUALITY = 0.3
# Clear view: a person seen unobstructed and close enough is found without waiting for the
# confidence to build up. This is the detection distance threshold for open areas.
CLEAR_VIEW_RANGE = 6.0         # m, camera to person
CLEAR_VIEW_MIN_POINTS = 4      # of the 7 sample points, at least this many inside the image ...
CLEAR_VIEW_FRACTION = 0.75     # ... and at least this share of those not hidden by anything
CLEAR_VIEW_FRAMES = 3          # frames in a row (0.3 s at 10 fps)
# Losing sight: the confidence is kept for SIGHTING_MEMORY_S after the last frame that saw the
# person (a turn or a branch in front for a moment does not throw the sighting away), then
# decays at CONF_DECAY per second.
SIGHTING_MEMORY_S = 2.0
CONF_DECAY = 0.3               # confidence lost per second while the person is not seen
SIGHTING_ON = 0.25             # confidence at which a "possible sighting" is raised
SIGHTING_OFF = 0.08            # ... and cleared (hysteresis)
FOUND_AT = 1.0                 # confidence at which the person is declared FOUND (latched)
MEAS_SIGMA_BASE = 0.10         # position-measurement noise (m) = base + per_m * range
MEAS_SIGMA_PER_M = 0.05

SEARCHING, SIGHTING, FOUND_STATE = "SEARCHING", "SIGHTING", "FOUND"


def range_quality(dist, cam_range=CAM_RANGE):
    """How much one camera frame counts at this distance (1 = fully, MIN_QUALITY at cam_range)."""
    if dist <= FULL_QUALITY_RANGE:
        return 1.0
    if dist >= cam_range:
        return MIN_QUALITY
    return 1.0 - (1.0 - MIN_QUALITY) * (dist - FULL_QUALITY_RANGE) / (cam_range - FULL_QUALITY_RANGE)



class MissingPerson:
    """The casualty in the simulation."""

    def __init__(self, body_id, name="Casualty"):
        self.body_id = body_id
        self.name = name
        self.marker_id = None

    @classmethod
    def spawn(cls, xy=None, name="Casualty", debug_marker=False):
        """Load the humanoid (same model / scale / pose as env_sim.py) at xy (default: original spot)."""
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        pos = list(TARGET_POSITION)
        if xy is not None:
            pos[0], pos[1] = xy[0], xy[1]
        body = p.loadURDF("humanoid/humanoid.urdf", pos, p.getQuaternionFromEuler(TARGET_ORIENTATION_EULER),
                          globalScaling=TARGET_SCALE, useFixedBase=True)
        person = cls(body, name)
        if debug_marker:
            vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.08, length=3.0, rgbaColor=[1, 0, 0, 0.45])
            person.marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                                 basePosition=[pos[0], pos[1], 1.5])
        return person

    # --- placement helpers (return an (x, y) for spawn) ---
    @staticmethod
    def random_open_position(seed, margin=1.8, extent=None):
        """Random spot in the open: at least `margin` m from every tree trunk.
        extent defaults to ~70% of the map half-width (9 m on the 26 m map)."""
        extent = 0.7 * sim_config.MAP_HALF if extent is None else extent
        rng = random.Random(seed)
        while True:
            x, y = rng.uniform(-extent, extent), rng.uniform(-extent, extent)
            if all(math.hypot(x - t[0], y - t[1]) >= margin for t in OBSTACLE_POSITIONS):
                return (round(x, 2), round(y, 2))

    @staticmethod
    def position_under_tree(seed=0):
        """Spot just beside a trunk, i.e. underneath a canopy (hard to see from above)."""
        rng = random.Random(seed)
        t = OBSTACLE_POSITIONS[rng.randrange(len(OBSTACLE_POSITIONS))]
        ang = rng.uniform(0, 2 * math.pi)
        return (round(t[0] + 0.5 * math.cos(ang), 2), round(t[1] + 0.5 * math.sin(ang), 2))

    # --- state ---
    @property
    def position(self):
        return p.getBasePositionAndOrientation(self.body_id)[0]

    def aabb(self):
        lo, hi = list(p.getAABB(self.body_id, -1)[0]), list(p.getAABB(self.body_id, -1)[1])
        for link in range(p.getNumJoints(self.body_id)):
            a, b = p.getAABB(self.body_id, link)
            lo = [min(lo[i], a[i]) for i in range(3)]
            hi = [max(hi[i], b[i]) for i in range(3)]
        return lo, hi

    def sample_points(self):
        """Points on/in the body used for line-of-sight tests: centre, top, bottom, and 4 sides."""
        lo, hi = self.aabb()
        cx, cy, cz = [(lo[i] + hi[i]) / 2 for i in range(3)]
        hx, hy = (hi[0] - lo[0]) / 2 * 0.8, (hi[1] - lo[1]) / 2 * 0.8
        return [[cx, cy, cz], [cx, cy, hi[2] - 0.02], [cx, cy, lo[2] + 0.02],
                [cx + hx, cy, cz], [cx - hx, cy, cz], [cx, cy + hy, cz], [cx, cy - hy, cz]]

    def mark_found(self):
        """Visual cue: green translucent pole at the person's location."""
        pos = self.position
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=0.12, length=3.0, rgbaColor=[0.1, 1, 0.2, 0.55])
        if self.marker_id is not None:
            p.changeVisualShape(self.marker_id, -1, rgbaColor=[0.1, 1, 0.2, 0.55])
        else:
            self.marker_id = p.createMultiBody(baseMass=0, baseVisualShapeIndex=vis,
                                               basePosition=[pos[0], pos[1], 1.5])



class SearchMonitor:
    """Decides when the person has been found. Call update() every physics step."""

    def __init__(self, person, gui=False, on_found=None, log=print, seed=0,
                 hfov=CAM_HFOV_DEG, vfov=CAM_VFOV_DEG, pitch=CAM_PITCH_DEG, cam_range=CAM_RANGE,
                 label_dz=0.8, mission=None):
        self.person, self.gui, self.on_found, self.log = person, gui, on_found, log
        self.mission = mission
        self.label_dz = label_dz      # height of the status label above the drone (m)
        self.tan_h, self.tan_v = math.tan(math.radians(hfov / 2)), math.tan(math.radians(vfov / 2))
        self.pitch, self.range = math.radians(pitch), cam_range
        self.rng = random.Random(seed)

        self.state = SEARCHING
        self.confidence = 0.0
        self.yaw = 0.0
        self._last_xy = None
        self._next_frame_t = 0.0
        self.t = 0.0

        # fused location estimate (running mean of noisy measurements)
        self._sum = [0.0, 0.0, 0.0]
        self._n = 0

        # result once found
        self.found = False
        self.found_time = None
        self.found_drone_pos = None
        self.estimate = None

        # statistics
        self.frames = self.frames_in_view = self.frames_seen = self.frames_occluded = 0
        self.clear_frames = 0         # clear-view frames in a row (see CLEAR_VIEW_*)
        self.last_seen_t = None       # sim time of the last frame that saw the person
        self.found_by = None          # "clear view" or "confidence"
        self.first_sighting_time = None
        self.events = []
        self._line_id = None
        self._text_id = None

    # ------------------------------------------------------------------ log --
    def _event(self, kind, msg, drone_pos=None, obj_pos=None, dist=None):
        self.events.append((self.t, kind, msg))
        self.log(f"[t={self.t:6.2f}s] {kind:<9} {msg}")
        
        if self.mission and drone_pos and obj_pos and dist is not None:
            from operator_guidance import MissionEvent
            me = MissionEvent(
                kind=kind,
                severity="INFO" if kind == "FOUND" else "WARNING",
                drone_position=tuple(drone_pos),
                object_position=tuple(obj_pos),
                distance=dist,
                timestamp=self.t,
                mission_state=self.state,
                response="N/A"
            )
            self.mission.add_mission_event(me)

    # --------------------------------------------------------------- camera --
    def set_heading(self, yaw_rad):
        self.yaw = yaw_rad

    def _axes(self):
        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        f = (cp * math.cos(self.yaw), cp * math.sin(self.yaw), sp)
        right = (math.sin(self.yaw), -math.cos(self.yaw), 0.0)
        up = (right[1] * f[2] - right[2] * f[1], right[2] * f[0] - right[0] * f[2],
              right[0] * f[1] - right[1] * f[0])
        return f, right, up

    def in_frustum(self, cam, pt):
        """Pinhole field-of-view test plus maximum range."""
        f, r, u = self._axes()
        v = (pt[0] - cam[0], pt[1] - cam[1], pt[2] - cam[2])
        dist = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
        if dist > self.range or dist < 1e-6:
            return False
        xf = v[0] * f[0] + v[1] * f[1] + v[2] * f[2]
        if xf <= 0.05:
            return False
        xr = v[0] * r[0] + v[1] * r[1] + v[2] * r[2]
        xu = v[0] * u[0] + v[1] * u[1] + v[2] * u[2]
        return abs(xr / xf) <= self.tan_h and abs(xu / xf) <= self.tan_v

    # --------------------------------------------------------------- update --
    def update(self, drone_pos, t):
        """Call every physics step with the drone position (x, y, z) and the sim time (s)."""
        self.t = t
        if self.found:
            return
        if t + 1e-9 < self._next_frame_t:
            return
        self._next_frame_t = t + 1.0 / CAM_FPS
        # heading follows the direction of travel
        xy = (drone_pos[0], drone_pos[1])
        if self._last_xy is not None and math.hypot(xy[0] - self._last_xy[0], xy[1] - self._last_xy[1]) > 0.05:
            self.yaw = math.atan2(xy[1] - self._last_xy[1], xy[0] - self._last_xy[0])
            self._last_xy = xy
        elif self._last_xy is None:
            self._last_xy = xy
        self._process_frame(list(drone_pos))

    def _process_frame(self, cam):
        self.frames += 1
        pts = self.person.sample_points()
        in_view = [pt for pt in pts if self.in_frustum(cam, pt)]
        visible = 0
        if in_view:
            self.frames_in_view += 1
            froms, tos, reach = [], [], []
            for pt in in_view:
                d = [pt[i] - cam[i] for i in range(3)]
                n = math.sqrt(sum(c * c for c in d))
                d = [c / n for c in d]
                froms.append([cam[i] + d[i] * RAY_START_OFFSET for i in range(3)])
                tos.append([pt[i] + d[i] * RAY_END_PAD for i in range(3)])
                # fraction of the ray at which it reaches the sample point
                reach.append((n - RAY_START_OFFSET) / (n - RAY_START_OFFSET + RAY_END_PAD))
            for res, at_point in zip(p.rayTestBatch(froms, tos), reach):
                # visible: the first thing hit is the person, or nothing is hit before the point
                if res[0] == self.person.body_id or res[0] < 0 or res[2] >= at_point - 1e-3:
                    visible += 1
        frac = visible / len(pts)
        centre = self.person.position
        dist = math.dist(cam, centre)
        quality = range_quality(dist, self.range)
        score = frac * quality if frac >= MIN_VISIBLE_FRACTION else 0.0

        # Clear view: enough of the person inside the image and (almost) nothing in front of it.
        clear = (dist <= CLEAR_VIEW_RANGE and len(in_view) >= CLEAR_VIEW_MIN_POINTS
                 and visible >= CLEAR_VIEW_FRACTION * len(in_view))
        self.clear_frames = self.clear_frames + 1 if clear else 0

        if score > 0:
            self.frames_seen += 1
            self.last_seen_t = self.t
            self._add_measurement(centre, dist)
        elif in_view:
            self.frames_occluded += 1

        dt = 1.0 / CAM_FPS
        if score > 0:
            self.confidence += CONF_GAIN * score * dt
        elif self.last_seen_t is None or self.t - self.last_seen_t > SIGHTING_MEMORY_S:
            self.confidence = max(0.0, self.confidence - CONF_DECAY * dt)
        self._update_state(cam, score, dist)
        self._draw(cam, centre, score)

    # ---------------------------------------------------- location estimate --
    def _add_measurement(self, truth, dist):
        sigma = MEAS_SIGMA_BASE + MEAS_SIGMA_PER_M * dist
        for i in range(3):
            self._sum[i] += truth[i] + self.rng.gauss(0, sigma)
        self._n += 1

    def _current_estimate(self):
        return [s / self._n for s in self._sum] if self._n else None

    # ---------------------------------------------------------------- state --
    def _update_state(self, cam, score, dist):
        if self.clear_frames >= CLEAR_VIEW_FRAMES:
            # Clear view for long enough: found now (a sighting is logged first if there was none).
            if self.state == SEARCHING:
                self.state = SIGHTING
                if self.first_sighting_time is None:
                    self.first_sighting_time = self.t
                self._event("SIGHTING", f"person in clear view, {dist:.1f} m away", cam, self.person.position, dist)
            self.confidence = max(self.confidence, FOUND_AT)
            self.found_by = "clear view"
            self._declare_found(cam)
            return
        if self.state == SEARCHING and self.confidence >= SIGHTING_ON:
            self.state = SIGHTING
            if self.first_sighting_time is None:
                self.first_sighting_time = self.t
            self._event("SIGHTING", f"possible person, {dist:.1f} m away (confidence {self.confidence:.2f})", cam, self.person.position, dist)
        elif self.state == SIGHTING and self.confidence <= SIGHTING_OFF:
            self.state = SEARCHING
            self._event("LOST", "sighting not confirmed, confidence dropped", cam, self.person.position, dist)
        if self.state == SIGHTING and self.confidence >= FOUND_AT:
            self.found_by = "confidence"
            self._declare_found(cam)

    def _declare_found(self, cam):
        self.state = FOUND_STATE
        self.found = True
        self.found_time = self.t
        self.found_drone_pos = list(cam)
        self.estimate = self._current_estimate()
        e = self.estimate
        dist = math.dist(cam, self.person.position)
        self._event("FOUND", f"{self.person.name} confirmed ({self.found_by}, "
                             f"{dist:.1f} m away). Estimated location "
                             f"({e[0]:.2f}, {e[1]:.2f}) from {self._n} measurements", cam, self.person.position, dist)
        if self.gui:
            self.person.mark_found()
        if self.on_found:
            self.on_found(self)

    # ------------------------------------------------------------------ GUI --
    def _draw(self, cam, centre, score):
        if not self.gui:
            return
        color = [0.1, 1, 0.2] if self.state == FOUND_STATE else ([1, 0.6, 0] if score > 0 else [0.5, 0.5, 0.5])
        kw = dict(lineColorRGB=color, lineWidth=2.0)
        if score > 0 or self.state == FOUND_STATE:
            if self._line_id is None:
                self._line_id = p.addUserDebugLine(cam, centre, **kw)
            else:
                self._line_id = p.addUserDebugLine(cam, centre, replaceItemUniqueId=self._line_id, **kw)
        elif self._line_id is not None:
            # park the line out of sight instead of deleting it (re-adding debug items over and
            # over can crash PyBullet's GUI; the map-guided search toggles this often)
            self._line_id = p.addUserDebugLine([0, 0, -50], [0, 0, -50.01],
                                               replaceItemUniqueId=self._line_id, **kw)
        txt = f"{self.state}  conf {self.confidence:.2f}/{FOUND_AT:.1f}"
        tk = dict(textColorRGB=color, textSize=1.4)
        pos = [cam[0], cam[1], cam[2] + self.label_dz]
        if self._text_id is None:
            self._text_id = p.addUserDebugText(txt, pos, **tk)
        else:
            self._text_id = p.addUserDebugText(txt, pos, replaceItemUniqueId=self._text_id, **tk)

    # --------------------------------------------------------------- report --
    def report(self):
        truth = self.person.position
        r = {"found": self.found, "state": self.state, "confidence": round(self.confidence, 2),
             "frames": self.frames, "frames_in_view": self.frames_in_view,
             "frames_seen": self.frames_seen, "frames_occluded": self.frames_occluded,
             "first_sighting_time": self.first_sighting_time, "found_time": self.found_time,
             "found_by": self.found_by,
             "true_position": [round(truth[0], 2), round(truth[1], 2)]}
        est = self.estimate or self._current_estimate()
        if est:
            r["estimate"] = [round(est[0], 2), round(est[1], 2)]
            r["estimate_error_m"] = round(math.hypot(est[0] - truth[0], est[1] - truth[1]), 2)
        return r


# ---------------------------------------------------------- standalone demo --
def find_assets(user_dir):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [user_dir] if user_dir else []
    cands += [here, os.path.join(here, "..", "Environment3D_Spike"),
              os.path.join(here, "..", "..", "Environment3D_Spike"), os.getcwd()]
    for c in cands:
        if c and os.path.exists(os.path.join(c, "custom_drone.urdf")):
            return os.path.abspath(c)
    sys.exit("Could not find custom_drone.urdf. Use --assets <Environment3D_Spike folder>.")


def build_world(gui, assets):
    p.connect(p.GUI if gui else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    floor_id = p.loadURDF("plane.urdf")
    tex = os.path.join(assets, "ground_texture.jpg")
    if gui and os.path.exists(tex):
        try:
            p.changeVisualShape(floor_id, -1, textureUniqueId=p.loadTexture(tex))
        except Exception as e:
            print("Texture not applied:", e)
    drone_id = p.loadURDF(os.path.join(assets, "custom_drone.urdf"), FLIGHT_PATH[0],
                          p.getQuaternionFromEuler([0, 0, 0]), globalScaling=DRONE_SCALE)
    for x, y, z in OBSTACLE_POSITIONS:                 # trees: these also hide the person from the camera
        tc = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.2, height=2.5)
        tv = p.createVisualShape(p.GEOM_CYLINDER, radius=0.2, length=2.5, rgbaColor=[0.4, 0.2, 0.05, 1])
        p.createMultiBody(0, tc, tv, [x, y, z])
        lc = p.createCollisionShape(p.GEOM_SPHERE, radius=0.75)
        lv = p.createVisualShape(p.GEOM_SPHERE, radius=0.75, rgbaColor=[0.1, 0.45, 0.1, 1])
        p.createMultiBody(0, lc, lv, [x, y, z + 1.75])
    return drone_id


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--seed", type=int, default=None, help="random open-field person location")
    ap.add_argument("--under-tree", action="store_true", help="put the person under a tree canopy")
    ap.add_argument("--person", type=float, nargs=2, metavar=("X", "Y"), help="explicit person location")
    ap.add_argument("--debug-marker", action="store_true", help="red pole over the person (for the operator view)")
    ap.add_argument("--cam-pitch", type=float, default=CAM_PITCH_DEG, help="camera pitch in deg (negative = down)")
    ap.add_argument("--continue-after-found", action="store_true", help="keep flying after FOUND")
    ap.add_argument("--assets", default=None)
    args = ap.parse_args()

    gui = not args.headless
    assets = find_assets(args.assets)
    drone_id = build_world(gui, assets)

    if args.person:
        xy = tuple(args.person)
    elif args.under_tree:
        xy = MissingPerson.position_under_tree(args.seed or 0)
    elif args.seed is not None:
        xy = MissingPerson.random_open_position(args.seed)
    else:
        xy = None
    person = MissingPerson.spawn(xy, debug_marker=args.debug_marker)
    monitor = SearchMonitor(person, gui=gui, pitch=args.cam_pitch)
    pos0 = person.position
    print(f"Person placed at ({pos0[0]:.2f}, {pos0[1]:.2f}). Search started.")

    quat = p.getQuaternionFromEuler([0, 0, 0])
    pos = list(FLIGHT_PATH[0])
    segment = 0
    sim_t = 0.0
    try:
        while segment < len(FLIGHT_PATH) - 1 and not (monitor.found and not args.continue_after_found):
            dest = FLIGHT_PATH[segment + 1]
            delta = [dest[i] - pos[i] for i in range(3)]
            dist = math.sqrt(sum(d * d for d in delta))
            if dist <= FLIGHT_SPEED * DT:
                pos = list(dest)
                segment += 1
            else:
                pos = [pos[i] + delta[i] / dist * FLIGHT_SPEED * DT for i in range(3)]
            p.resetBasePositionAndOrientation(drone_id, pos, quat)
            p.resetBaseVelocity(drone_id, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
            sim_t += DT
            monitor.update(pos, sim_t)
            if gui:
                p.resetDebugVisualizerCamera(5, 45, -30, pos)
                time.sleep(DT)
    except KeyboardInterrupt:
        pass
    print("Detection report:", monitor.report())
    p.disconnect()

if __name__ == "__main__":
    main()
