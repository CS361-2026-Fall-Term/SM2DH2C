"""
Contributor: Chase Vanderlip
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

PBI-13 (detection results with the confidence level, in the terminal):
  * explain_method()  printed once at the start: how the confidence is calculated and what the
                      numbers mean.
  * one line per camera scan (10 per second) while the person is in the camera's view or the
    confidence is above 0, with the confidence, its change, how much of the person was seen, the
    distance, the range quality, the score of that scan and the clear-view counter. While nothing
    is in view, one short summary line every IDLE_SUMMARY_S seconds instead of 10 identical lines
    a second. Every scan is also kept in SearchMonitor.scans (a list of dicts).
  * format_report()   a readable detection summary at the end of the mission.
  log_mode: "scans" (all of the above, default), "events" (only SIGHTING / LOST / FOUND and the
  summary) or "off".

PBI-14: every SIGHTING / LOST / FOUND is also sent to the shared Mission (mission=...) as a
MissionEvent with the numeric drone-to-person distance, the drone and person positions and the
time, so the mission's detection state and the operator panel update from it.
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

# terminal output (PBI-13)
LOG_MODES = ("scans", "events", "off")
IDLE_SUMMARY_S = 5.0           # one summary line this often while the person is not in view
HEADER_EVERY = 25              # repeat the column header every this many scan lines
SCAN_HEADER = (f"{'t (s)':>7}  {'state':<9}  {'conf':>5}  {'change':>6}  {'seen':>4}  "
               f"{'dist':>6}  {'quality':>7}  {'score':>5}  {'clear':>5}  note")


def format_scan(s):
    """One scan as a fixed-width line matching SCAN_HEADER (PBI-13)."""
    seen = f"{s['visible']}/{s['points']}" if s["in_view"] else "-"
    dist = f"{s['dist']:.1f} m" if s["in_view"] else "-"
    quality = f"{s['quality']:.2f}" if s["in_view"] else "-"
    return (f"{s['t']:7.2f}  {s['state']:<9}  {s['confidence']:5.2f}  {s['change']:+6.2f}  {seen:>4}  "
            f"{dist:>6}  {quality:>7}  {s['score']:5.2f}  {s['clear']}/{CLEAR_VIEW_FRAMES:<3}  {s['note']}")


def explain_method():
    """How the confidence is calculated, printed once at the start of a mission (PBI-13)."""
    return "\n".join([
        "How the detection confidence works",
        "----------------------------------",
        f"  The camera ({CAM_HFOV_DEG:.0f} x {CAM_VFOV_DEG:.0f} deg, tilted {abs(CAM_PITCH_DEG):.0f} deg down, "
        f"up to {CAM_RANGE:.0f} m) takes {CAM_FPS:.0f} scans per second.",
        "  Each scan checks 7 points on the person (centre, head, feet, 4 sides):",
        "    seen     = points inside the image with nothing else in the way (e.g. 5/7)",
        f"    quality  = how much a scan counts at that distance: 1.00 up to {FULL_QUALITY_RANGE:.0f} m,",
        f"               falling to {MIN_QUALITY:.2f} at {CAM_RANGE:.0f} m",
        f"    score    = (seen points / 7) x quality, or 0 if fewer than {MIN_VISIBLE_FRACTION:.0%} are seen",
        f"  confidence goes up by score x {CONF_GAIN:g} per second while the person is seen. It is kept",
        f"  for {SIGHTING_MEMORY_S:.0f} s after losing sight, then falls by {CONF_DECAY:g} per second.",
        f"    >= {SIGHTING_ON:.2f}  possible sighting (SIGHTING, the drone stops to look)",
        f"    <= {SIGHTING_OFF:.2f}  sighting dropped (LOST)",
        f"    >= {FOUND_AT:.2f}  person FOUND",
        f"  Clear view: within {CLEAR_VIEW_RANGE:.0f} m, at least {CLEAR_VIEW_MIN_POINTS} points in the image and "
        f"{CLEAR_VIEW_FRACTION:.0%} of them unblocked,",
        f"  for {CLEAR_VIEW_FRAMES} scans in a row (clear 3/3): FOUND straight away (confidence set to "
        f"{FOUND_AT:.2f}).",
        "  Columns: t = sim time, conf = confidence after the scan, change = how much this scan",
        "  added or removed, clear = clear-view scans in a row."])


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
                 label_dz=0.8, log_mode="events", on_event=None, mission=None):
        self.person, self.gui, self.on_found, self.log = person, gui, on_found, log
        self.mission = mission        # PBI-14: shared Mission; events go to mission.add_event()
        self.mission_events = []      # PBI-14: every MissionEvent this detector made
        self.last_distance = None     # drone (camera) to person, metres, at the last scan
        self.on_event = on_event      # on_event(kind, text): SIGHTING / LOST / FOUND, e.g. for the panel
        if log_mode not in LOG_MODES:
            raise ValueError(f"log_mode must be one of {LOG_MODES}")
        self.log_mode = log_mode      # PBI-13: what goes to the terminal (see LOG_MODES)
        self.scans = []               # every camera scan: dict(t, state, confidence, ...), PBI-13
        self._lines = 0               # scan lines printed (header repeated every HEADER_EVERY)
        self._idle_from = None        # first scan of the current stretch without the person in view
        self._idle_scans = 0
        self._last_summary_t = None
        self.sightings = self.losses = 0
        self.max_confidence = 0.0
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
    def _event(self, kind, msg, cam=None, dist=None):
        self.events.append((self.t, kind, msg))
        self.log(f"[t={self.t:6.2f}s] {kind:<9} {msg}")
        if self.on_event is not None:
            self.on_event(kind, f"{kind} (t = {self.t:.1f} s): {msg}")
        if kind in ("SIGHTING", "LOST", "FOUND"):
            self._mission_event(kind, msg, cam, dist)

    def _mission_event(self, kind, msg, cam, dist):
        """PBI-14: the event as a MissionEvent (numeric distance, positions), sent to the Mission."""
        from operator_guidance import MissionEvent
        est = self.estimate if self.estimate is not None else self._current_estimate()
        ev = MissionEvent(
            kind=kind, severity={"SIGHTING": "WARNING", "LOST": "INFO", "FOUND": "INFO"}[kind],
            source="target", drone_position=tuple(cam) if cam is not None else None,
            object_position=tuple(est) if est is not None else None, distance=dist,
            timestamp=self.t, response={"SIGHTING": "stopping to look",
                                        "LOST": "continuing the search",
                                        "FOUND": "mission complete, holding position"}[kind],
            message=msg, object_name=self.person.name)
        self.mission_events.append(ev)
        if self.mission is not None:
            self.mission.add_event(ev)

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
        self.last_distance = dist
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
        before = self.confidence
        decaying = False
        if score > 0:
            self.confidence += CONF_GAIN * score * dt
        elif self.last_seen_t is None or self.t - self.last_seen_t > SIGHTING_MEMORY_S:
            decaying = self.confidence > 0
            self.confidence = max(0.0, self.confidence - CONF_DECAY * dt)
        self._update_state(cam, score, dist)
        self.max_confidence = max(self.max_confidence, self.confidence)
        if score > 0:
            note = "clear view" if clear else "seen"
        elif in_view and visible == 0:
            note = "in view but hidden (something in front)"
        elif in_view:
            note = f"too little visible ({visible}/{len(pts)}, needs {math.ceil(MIN_VISIBLE_FRACTION * len(pts))})"
        elif decaying:
            note = "not in view, confidence decaying"
        elif self.confidence > 0:
            note = f"not in view, kept for {SIGHTING_MEMORY_S - (self.t - self.last_seen_t):.1f} s"
        else:
            note = "not in view"
        self._record_scan(dict(t=round(self.t, 2), state=self.state, confidence=self.confidence,
                               change=self.confidence - before, in_view=len(in_view),
                               visible=visible, points=len(pts), dist=dist, quality=quality,
                               score=score, clear=min(self.clear_frames, CLEAR_VIEW_FRAMES),
                               note=note))
        self._draw(cam, centre, score)

    # ------------------------------------------------------- terminal (PBI-13) --
    def _record_scan(self, scan):
        """Keep the scan and print it (log_mode "scans"): a line per scan while the person is in
        view or the confidence is above 0, else a summary line every IDLE_SUMMARY_S seconds."""
        self.scans.append(scan)
        if self.log_mode != "scans":
            return
        active = scan["in_view"] > 0 or scan["confidence"] > 0 or self.found
        if active:
            if self._idle_scans:                          # close the quiet stretch first
                self._print_idle(self.scans[-2]["t"])
            if self._lines % HEADER_EVERY == 0:
                self.log(f"[detect] {SCAN_HEADER}")
            self._lines += 1
            self.log("[detect] " + format_scan(scan))
            return
        if self._idle_from is None:
            self._idle_from = scan["t"]
        self._idle_scans += 1
        self._idle_conf = scan["confidence"]
        if scan["t"] - self._idle_from >= IDLE_SUMMARY_S:
            self._print_idle(scan["t"])

    def _print_idle(self, t):
        self.log(f"[detect] {self._idle_from:7.2f}-{t:.2f} s  {self.state:<9}  conf "
                 f"{self._idle_conf:.2f}  person not in the camera view ({self._idle_scans} scans)")
        self._idle_from, self._idle_scans = None, 0

    def format_report(self):
        """Readable end-of-mission detection summary (PBI-13)."""
        r = self.report()
        lines = ["Detection summary", "-----------------",
                 f"  Result:           {'FOUND' if r['found'] else 'not found'} "
                 f"(final state {r['state']}, confidence {r['confidence']:.2f} of {FOUND_AT:.2f})"]
        if r["found"]:
            lines.append(f"  Found at:         t = {r['found_time']:.1f} s, by {r['found_by']}"
                         + (f", {r['found_distance_m']:.1f} m from the person"
                            if r.get("found_distance_m") is not None else ""))
        if r["first_sighting_time"] is not None:
            lines.append(f"  First sighting:   t = {r['first_sighting_time']:.1f} s")
        lines += [f"  Sightings:        {r['sightings']} raised, {r['losses']} lost again",
                  f"  Highest conf.:    {r['max_confidence']:.2f}",
                  f"  Camera scans:     {r['frames']} total, person in view in {r['frames_in_view']}, "
                  f"seen in {r['frames_seen']}, hidden in {r['frames_occluded']}"]
        if "estimate" in r:
            lines.append(f"  Location:         estimated ({r['estimate'][0]:.2f}, {r['estimate'][1]:.2f}), "
                         f"true ({r['true_position'][0]:.2f}, {r['true_position'][1]:.2f}), "
                         f"error {r['estimate_error_m']:.2f} m")
        else:
            lines.append(f"  Location:         no estimate (never seen); true position "
                         f"({r['true_position'][0]:.2f}, {r['true_position'][1]:.2f})")
        return "\n".join(lines)

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
                self.sightings += 1
                if self.first_sighting_time is None:
                    self.first_sighting_time = self.t
                self._event("SIGHTING", f"person in clear view, {dist:.1f} m away", cam, dist)
            self.confidence = max(self.confidence, FOUND_AT)
            self.found_by = "clear view"
            self._declare_found(cam)
            return
        if self.state == SEARCHING and self.confidence >= SIGHTING_ON:
            self.state = SIGHTING
            if self.first_sighting_time is None:
                self.first_sighting_time = self.t
            self.sightings += 1
            self._event("SIGHTING", f"possible person, {dist:.1f} m away (confidence {self.confidence:.2f} "
                                    f">= {SIGHTING_ON:.2f})", cam, dist)
        elif self.state == SIGHTING and self.confidence <= SIGHTING_OFF:
            self.state = SEARCHING
            self.losses += 1
            self._event("LOST", f"sighting not confirmed, confidence dropped to {self.confidence:.2f} "
                                f"(<= {SIGHTING_OFF:.2f})", cam, dist)
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
                             f"({e[0]:.2f}, {e[1]:.2f}) from {self._n} measurements", cam, dist)
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
             "found_distance_m": (round(math.dist(self.found_drone_pos, truth), 2)
                                  if self.found_drone_pos else None),
             "sightings": self.sightings, "losses": self.losses,
             "max_confidence": round(self.max_confidence, 2),
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
