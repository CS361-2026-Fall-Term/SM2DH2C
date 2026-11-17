"""
File Description: PBI-10 - keeps the drone's mission state saved on disk so that if the program
(or the window) crashes, the last known state can be loaded again with `python main.py --resume`.

What counts as mission state (saved in mission_state.json, next to this file):
  mission  state, plan (zones still to search), visited zones, excluded zones, the plan order,
           destination, last response, and the full history log of operator commands and
           system responses
  setup    the operator's starting information: last known position (lat/lon), search area,
           map size, OSM file, seed, visuals, camera mode, time limit, missing-person category,
           zones excluded in the setup window, zone order, and where the person was placed
  zones    zone centres A-D as used in that run
  drone    last known position [x, y, z] and the mission clock (sim seconds used so far)
  belief   the probability map (PBI-06), so cells already searched stay searched after a resume
Not saved: detection confidence (it starts fresh after a resume).

Why a plain JSON file (the "safest option" research for this PBI):
  * JSON is text: easy to inspect and no code can run when it is loaded. (pickle can execute code
    from a tampered file; SQLite is more than this small state needs and adds nothing here.)
  * Writes are atomic: data goes to a temporary file in the same folder, is flushed (and fsync'ed
    for state changes), then os.replace() swaps it in. A crash mid-write leaves the old file intact.
  * The previous good copy is kept as mission_state.json.bak, and every file carries a SHA-256
    checksum, so a truncated or edited file is detected and the backup is used instead.
  * Loading never raises: bad data is reported as "problems" and the program starts fresh.

Saving rules (MissionStateStore.autosave):
  * Any change to the mission (state, plan, visited, destination, response, new history line)
    is written immediately - far inside the 1 second limit (PBI-10 acceptance criterion).
  * While SEARCHING the drone position is also refreshed every PERIODIC_SAVE_S seconds.
  * PBI-07: main.py wraps the store in BackgroundSaver, so the slow part of a save (JSON,
    checksum, write, fsync) runs on a worker thread and the operator UI never waits for the disk.
"""

import hashlib
import json
import math
import os
import tempfile
import time

from operator_guidance import Mission, ZONES

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PATH = os.path.join(HERE, "mission_state.json")
SCHEMA_VERSION = 1
MAX_FILE_BYTES = 2_000_000       # a state file bigger than this is treated as damaged
PERIODIC_SAVE_S = 0.5            # drone-position refresh while SEARCHING
SAVE_BUDGET_S = 1.0              # acceptance criterion: saved within 1 second of a change
CAMERA_NAMES = ("locked", "follow", "free")
MAX_BELIEF_CELLS = 40_000         # far more than the ~30 x 30 heatmap main.py uses
SETUP_KEYS = ("lat", "lon", "area", "map_size", "osm_file", "seed",
              "simple_visuals", "camera", "max_time",
              # PBI-03 / PBI-06 setup, so --resume rebuilds the same search (missing in old files)
              "person_category", "exclude_zones", "zone_order", "person_xy")
ZONE_ORDERS = ("map", "fixed")


# ------------------------------------------------------------------ validation helpers --
def _finite(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def validate_setup(s):
    """Return a cleaned copy of the operator's starting information or raise ValueError."""
    if not isinstance(s, dict):
        raise ValueError("setup must be an object")
    out = {}
    for key in SETUP_KEYS:
        v = s.get(key)
        if v is None:
            out[key] = None
        elif key in ("lat", "lon", "area", "map_size", "max_time"):
            if not _finite(v):
                raise ValueError(f"setup.{key} must be a number")
            out[key] = float(v)
        elif key == "seed":
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError("setup.seed must be an integer")
            out[key] = v
        elif key == "simple_visuals":
            if not isinstance(v, bool):
                raise ValueError("setup.simple_visuals must be true/false")
            out[key] = v
        elif key == "camera":
            if v not in CAMERA_NAMES:
                raise ValueError("setup.camera is not a known camera mode")
            out[key] = v
        elif key == "person_category":
            if not isinstance(v, str) or not 0 < len(v) <= 50:
                raise ValueError("setup.person_category must be a short string")
            out[key] = v
        elif key == "exclude_zones":
            if not isinstance(v, str) or any(z not in ZONES for z in v) or len(set(v)) != len(v) \
                    or len(v) >= len(ZONES):
                raise ValueError("setup.exclude_zones must be distinct known zones, not all")
            out[key] = v
        elif key == "zone_order":
            if v not in ZONE_ORDERS:
                raise ValueError("setup.zone_order must be 'map' or 'fixed'")
            out[key] = v
        elif key == "person_xy":
            if not (isinstance(v, list) and len(v) == 2 and all(_finite(c) and abs(c) < 10_000
                                                                for c in v)):
                raise ValueError("setup.person_xy must be [x, y]")
            out[key] = [float(c) for c in v]
        else:                                                     # osm_file
            if not isinstance(v, str) or len(v) > 500:
                raise ValueError("setup.osm_file must be a short string")
            out[key] = v
    if out["lat"] is not None and not -90 <= out["lat"] <= 90:
        raise ValueError("setup.lat out of range")
    if out["lon"] is not None and not -180 <= out["lon"] <= 180:
        raise ValueError("setup.lon out of range")
    if (out["lat"] is None) != (out["lon"] is None):
        raise ValueError("setup.lat and setup.lon must be given together")
    return out


def validate_payload(p):
    """Check the whole saved payload. Raises ValueError with the reason."""
    if not isinstance(p, dict):
        raise ValueError("payload is not an object")
    Mission.validate_snapshot(p.get("mission"))
    if p.get("setup") is not None:
        p["setup"] = validate_setup(p["setup"])
    zones = p.get("zones")
    if zones is not None:
        if not isinstance(zones, dict) or set(zones) != set(ZONES) or any(
                not (isinstance(c, list) and len(c) == 2 and all(_finite(v) for v in c))
                for c in zones.values()):
            raise ValueError("zones must map every zone to [x, y]")
    bel = p.get("belief")
    if bel is not None:
        shape = bel.get("shape") if isinstance(bel, dict) else None
        grid = bel.get("grid") if isinstance(bel, dict) else None
        if not (isinstance(shape, list) and len(shape) == 2
                and all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in shape)
                and shape[0] * shape[1] <= MAX_BELIEF_CELLS
                and isinstance(grid, list) and len(grid) == shape[0] * shape[1]
                and all(_finite(v) and v >= 0 for v in grid)):
            raise ValueError("belief must be {shape: [ny, nx], grid: [ny*nx numbers >= 0]}")
    drone = p.get("drone")
    if drone is not None:
        pos = drone.get("pos") if isinstance(drone, dict) else None
        if not (isinstance(pos, list) and len(pos) == 3 and all(_finite(v) for v in pos)
                and all(abs(v) < 10_000 for v in pos)):
            raise ValueError("drone.pos must be [x, y, z]")
        t = drone.get("t")
        if t is not None and not (_finite(t) and t >= 0):
            raise ValueError("drone.t must be a time in seconds")
    return p


def _payload_json(payload):
    """The payload as canonical JSON (sorted keys, no spaces). The checksum is taken over this
    text, and (PBI-07) the same text is written into the file, so a save serialises the payload
    once instead of twice. Loading re-parses it and gets the same checksum back."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _checksum(payload, blob=None):
    blob = _payload_json(payload) if blob is None else blob
    return hashlib.sha256(blob.encode("ascii")).hexdigest()


# ------------------------------------------------------------------------------- the store --
class LoadResult:
    """payload: the validated saved data, or None. source: 'main' | 'backup' | None.
    problems: why a file was skipped (shown to the operator, never raised)."""

    def __init__(self, payload=None, source=None, problems=None):
        self.payload = payload
        self.source = source
        self.problems = problems or []


class MissionStateStore:
    def __init__(self, path=DEFAULT_PATH, clock=time.monotonic, wall=time.time):
        self.path = path
        self.bak_path = path + ".bak"
        self.prev_path = path + ".previous"      # the last run's file when a new run starts
        self.clock = clock
        self.wall = wall
        self.setup = None                        # operator's starting information
        self.zones = None
        self.last_error = None                   # last save failure (never raised)
        self.last_save_s = 0.0                   # how long the most recent save took
        self.max_save_s = 0.0
        self.saves = 0
        self._fingerprint = None
        self._last_save_t = -1e9
        self._warned = False

    # -- what gets saved ------------------------------------------------------------------
    def set_context(self, setup=None, zones=None):
        """Remember the operator's starting information and the zone centres for every save."""
        self.setup = validate_setup(setup) if setup is not None else None
        self.zones = ({k: [float(v[0]), float(v[1])] for k, v in zones.items()}
                      if zones else None)

    def build(self, mission, drone_pos=None, belief=None, sim_t=None):
        return self.build_from(mission.snapshot(), drone_pos,
                               belief.grid if belief is not None else None, sim_t)

    def build_from(self, mission_snapshot, drone_pos=None, belief_grid=None, sim_t=None):
        """build() from data already copied out of the live objects (a Mission.snapshot() and a
        copy of the probability map's grid). BackgroundSaver uses it on its worker thread."""
        return {
            "mission": mission_snapshot,
            "setup": self.setup,
            "zones": self.zones,
            "drone": ({"pos": [float(v) for v in drone_pos],
                       "t": float(sim_t) if sim_t is not None else None}
                      if drone_pos is not None else None),
            "belief": grid_to_data(belief_grid) if belief_grid is not None else None,
        }

    # -- writing --------------------------------------------------------------------------
    def save(self, payload, durable=True):
        """Atomically write payload. Returns the seconds it took. Raises OSError on failure
        (autosave() catches it, so the simulation never crashes because of a full disk)."""
        t0 = time.perf_counter()
        blob = _payload_json(payload)                  # serialised once (PBI-07)
        head = json.dumps({"version": SCHEMA_VERSION, "saved_at": round(self.wall(), 3),
                           "checksum": _checksum(payload, blob)}, ensure_ascii=True)
        data = (head[:-1] + ', "payload": ' + blob + "}").encode("ascii")
        folder = os.path.dirname(os.path.abspath(self.path))
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".mission_state.", suffix=".tmp")
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
                f.flush()
                if durable:
                    os.fsync(f.fileno())
            if os.path.exists(self.path):
                os.replace(self.path, self.bak_path)     # keep the previous good copy
            os.replace(tmp, self.path)                   # atomic swap
            if durable:
                self._fsync_dir(folder)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        self.last_save_s = time.perf_counter() - t0
        self.max_save_s = max(self.max_save_s, self.last_save_s)
        self.saves += 1
        return self.last_save_s

    @staticmethod
    def _fsync_dir(folder):
        try:                                             # not possible on every OS; best effort
            fd = os.open(folder, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            pass

    def autosave(self, mission, drone_pos=None, force=False, belief=None, sim_t=None):
        """Call once per render tick. Saves right away when the mission changed; otherwise
        refreshes the drone position (and probability map) every PERIODIC_SAVE_S while
        SEARCHING. Never raises. Returns True when a file was written."""
        due = self.due(mission, force)
        if due is None:
            return False
        fp, durable, now = due
        try:
            self.save(self.build(mission, drone_pos, belief, sim_t), durable=durable)
        except Exception as exc:                         # disk full, read-only folder, ...
            self.save_failed(exc, now)
            return False
        self.saved(fp, now)
        return True

    # The three steps of autosave(), shared with BackgroundSaver (PBI-07).
    @staticmethod
    def fingerprint(mission):
        """What counts as 'the mission changed' (a new history line included)."""
        return (mission.state, tuple(mission.plan), tuple(mission.visited),
                tuple(mission.excluded), tuple(mission.original_plan),
                tuple(mission.destination) if mission.destination else None,
                mission.response, getattr(mission, "history_count", len(mission.history)),
                mission.history[-1] if mission.history else None)

    def due(self, mission, force=False):
        """None if nothing needs saving now, else (fingerprint, durable, now)."""
        fp = self.fingerprint(mission)
        now = self.clock()
        changed = fp != self._fingerprint
        periodic = mission.state == "SEARCHING" and now - self._last_save_t >= PERIODIC_SAVE_S
        if not (force or changed or periodic):
            return None
        return fp, force or changed, now

    def saved(self, fp, now):
        self._fingerprint = fp
        self._last_save_t = now
        self.last_error = None
        self._warned = False

    def save_failed(self, exc, now):
        self.last_error = f"{type(exc).__name__}: {exc}"
        if not self._warned:
            self._warned = True
            print(f"[state] WARNING: could not save mission state ({self.last_error}); "
                  "the mission continues without saving.", flush=True)
        self._last_save_t = now                          # do not retry on every tick

    # -- reading --------------------------------------------------------------------------
    def _read(self, path):
        """Parse and fully validate one file. Returns (payload, None) or (None, reason)."""
        try:
            size = os.path.getsize(path)
        except OSError:
            return None, "file not found"
        if size > MAX_FILE_BYTES:
            return None, "file is too large to be a mission state"
        try:
            with open(path, "rb") as f:
                doc = json.loads(f.read().decode("ascii"))
        except (OSError, ValueError):                    # JSONDecodeError and bad bytes
            return None, "file is damaged (not valid JSON)"
        if not isinstance(doc, dict) or "payload" not in doc:
            return None, "file has the wrong layout"
        if doc.get("version") != SCHEMA_VERSION:
            return None, f"unsupported file version {doc.get('version')!r}"
        if doc.get("checksum") != _checksum(doc["payload"]):
            return None, "checksum mismatch (file was cut off or edited)"
        try:
            return validate_payload(doc["payload"]), None
        except ValueError as exc:
            return None, str(exc)

    def load(self):
        """Return a LoadResult: the saved file if it is good, else the backup, else nothing."""
        problems = []
        for source, path in (("main", self.path), ("backup", self.bak_path)):
            if not os.path.exists(path):
                continue
            payload, why = self._read(path)
            if payload is not None:
                return LoadResult(payload, source, problems)
            problems.append(f"{os.path.basename(path)}: {why}")
        return LoadResult(None, None, problems)

    def archive_previous(self):
        """Fresh run (no --resume): move the old state aside so nothing is lost. Returns True
        if there was a file to move."""
        moved = False
        for src, dst in ((self.path, self.prev_path), (self.bak_path, self.prev_path + ".bak")):
            try:
                if os.path.exists(src):
                    os.replace(src, dst)
                    moved = True
            except OSError:
                pass
        return moved


class BackgroundSaver:
    """PBI-07: MissionStateStore.autosave() with the slow part moved off the UI thread.

    A save used to cost the main loop 6-34 ms per change (serialising the whole history and the
    probability map, then fsync), and with live data the mission changes on almost every frame.
    Now autosave() here only decides whether a save is due and copies the state (about 0.3 ms:
    Mission.snapshot() plus a copy of the probability grid). A worker thread does the rest:
    JSON, checksum, atomic write, fsync.

    Newest wins: while the worker is writing, further changes replace the one waiting snapshot,
    so a burst of 100 live messages becomes a couple of saves, never a queue that grows. Saves
    are at most MIN_GAP_S apart (still far inside PBI-10's 1 second limit). close() writes the
    final state and waits for it, so nothing is lost on a normal exit.

    Same call as the store: autosave(mission, drone_pos, force, belief, sim_t) -> True when a
    save was queued. store.saves / max_save_s / last_error keep working (set by the worker)."""

    MIN_GAP_S = 0.1

    def __init__(self, store):
        import threading
        self.store = store
        self.queued = 0                           # snapshots handed to the worker
        self.coalesced = 0                        # snapshots replaced by a newer one before writing
        self._cv = threading.Condition()
        self._job = None                          # (payload parts, fp, durable, now) waiting
        self._busy = False
        self._stop = False
        self._last_write = -1e9
        self._thread = threading.Thread(target=self._run, name="mission-state-saver", daemon=True)
        self._thread.start()

    def autosave(self, mission, drone_pos=None, force=False, belief=None, sim_t=None):
        """Called once per render tick from the UI thread; never blocks on disk."""
        st = self.store
        due = st.due(mission, force)
        if due is None:
            return False
        fp, durable, now = due
        parts = (mission.snapshot(), list(drone_pos) if drone_pos is not None else None,
                 belief.grid.copy() if belief is not None else None, sim_t)
        # Mark it saved now so the next tick does not queue the same state again; a failure on
        # the worker is reported through store.last_error, as before.
        st.saved(fp, now)
        with self._cv:
            if self._job is not None:
                self.coalesced += 1
                durable = durable or self._job[2]
            self._job = (parts, fp, durable, now)
            self.queued += 1
            self._cv.notify()
        if force:
            self.flush()
        return True

    def _run(self):
        while True:
            with self._cv:
                while self._job is None and not self._stop:
                    self._cv.wait()
                if self._job is None:
                    return
                wait = self._last_write + self.MIN_GAP_S - time.monotonic()
                if wait > 0 and not self._stop:
                    self._cv.wait(wait)               # let more changes coalesce into one save
                    continue
                job, self._job, self._busy = self._job, None, True
            (snap, pos, grid, sim_t), _fp, durable, now = job
            try:
                self.store.save(self.store.build_from(snap, pos, grid, sim_t), durable=durable)
            except Exception as exc:                  # disk full, read-only folder, ...
                self.store.save_failed(exc, now)
            with self._cv:
                self._busy = False
                self._last_write = time.monotonic()
                self._cv.notify_all()

    def flush(self, timeout=5.0):
        """Wait until everything queued so far is on disk. Returns False on timeout."""
        end = time.monotonic() + timeout
        with self._cv:
            self._last_write = -1e9                   # write now, skip the MIN_GAP_S wait
            self._cv.notify_all()
            while self._job is not None or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self, timeout=5.0):
        """Write what is left and stop the worker."""
        ok = self.flush(timeout)
        with self._cv:
            self._stop = True
            self._cv.notify_all()
        self._thread.join(timeout)
        return ok

    # the store's counters, so main.py's end-of-run report reads the same either way
    def __getattr__(self, name):
        return getattr(self.store, name)


# ----------------------------------------------------- bridging the saved setup and main.py --
def setup_from_args(args, person_xy=None):
    """The operator's starting information, taken from main.py's parsed arguments.
    person_xy: where the missing person was placed, so a resumed run uses the same spot."""
    return {
        "lat": args.lat, "lon": args.lon, "area": args.area, "map_size": args.map_size,
        "osm_file": args.osm_file or None, "seed": int(args.seed or 0),
        "simple_visuals": bool(args.simple_visuals), "camera": args.camera,
        "max_time": args.max_time,
        "person_category": getattr(args, "person_category", None) or None,
        "exclude_zones": getattr(args, "exclude_zones", None) or None,
        "zone_order": getattr(args, "zone_order", None) or None,
        "person_xy": [float(person_xy[0]), float(person_xy[1])] if person_xy is not None else None,
    }


def apply_setup_to_args(args, setup):
    """Copy a saved setup back into main.py's arguments so --resume rebuilds the same world."""
    args.lat, args.lon = setup["lat"], setup["lon"]
    args.area, args.map_size = setup["area"], setup["map_size"]
    args.osm_file, args.seed = setup["osm_file"], setup["seed"] or 0
    args.simple_visuals = bool(setup["simple_visuals"])
    args.camera, args.max_time = setup["camera"], setup["max_time"]
    args.person_category = setup.get("person_category")
    args.exclude_zones = setup.get("exclude_zones")
    args.zone_order = setup.get("zone_order") or "map"
    return args


def belief_to_data(belief):
    """The probability map's cell values as plain JSON data."""
    return grid_to_data(belief.grid)


def grid_to_data(grid):
    """A probability-map grid (numpy array, ny x nx) as plain JSON data."""
    ny, nx = grid.shape
    return {"shape": [int(ny), int(nx)], "grid": grid.ravel().tolist()}


def restore_belief(belief, data):
    """Put saved cell values back into the probability map. Returns False (and changes nothing)
    when there is nothing saved or the grid size differs (the map was built differently).
    Cells that are excluded now stay at 0."""
    if not data or list(belief.grid.shape) != data["shape"]:
        return False
    import numpy as np
    grid = np.asarray(data["grid"], dtype=float).reshape(belief.grid.shape)
    if grid.sum() <= 0:
        return False
    belief.grid = grid
    belief._normalize()
    return True


def describe(payload):
    """One line for the console, e.g. 'SEARCHING, 2 zone(s) searched, saved 3.1 s ago'."""
    m = payload["mission"]
    return f"{m['state']}, {len(m['visited'])} zone(s) searched, plan {m['plan']}"
