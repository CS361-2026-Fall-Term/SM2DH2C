"""
File Description: PBI-10 - keeps the drone's mission state saved on disk so that if the program
(or the window) crashes, the last known state can be loaded again with `python main.py --resume`.

What counts as mission state (saved in mission_state.json, next to this file):
  mission  state, plan (zones still to search), visited zones, destination, last response,
           and the full history log of operator commands and system responses
  setup    the operator's starting information: last known position (lat/lon), search area,
           map size, OSM file, seed, visuals, camera mode, time limit
  zones    zone centres A-D as used in that run
  drone    last known position [x, y, z]
Not saved: the probability map / detection confidence (they start fresh after a resume).

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
SETUP_KEYS = ("lat", "lon", "area", "map_size", "osm_file", "seed",
              "simple_visuals", "camera", "max_time")


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
    drone = p.get("drone")
    if drone is not None:
        pos = drone.get("pos") if isinstance(drone, dict) else None
        if not (isinstance(pos, list) and len(pos) == 3 and all(_finite(v) for v in pos)
                and all(abs(v) < 10_000 for v in pos)):
            raise ValueError("drone.pos must be [x, y, z]")
    return p


def _checksum(payload):
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
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

    def build(self, mission, drone_pos=None):
        return {
            "mission": mission.snapshot(),
            "setup": self.setup,
            "zones": self.zones,
            "drone": {"pos": [float(v) for v in drone_pos]} if drone_pos is not None else None,
        }

    # -- writing --------------------------------------------------------------------------
    def save(self, payload, durable=True):
        """Atomically write payload. Returns the seconds it took. Raises OSError on failure
        (autosave() catches it, so the simulation never crashes because of a full disk)."""
        t0 = time.perf_counter()
        doc = {"version": SCHEMA_VERSION, "saved_at": round(self.wall(), 3),
               "checksum": _checksum(payload), "payload": payload}
        data = json.dumps(doc, indent=1, ensure_ascii=True).encode("ascii")
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

    def autosave(self, mission, drone_pos=None, force=False):
        """Call once per render tick. Saves right away when the mission changed; otherwise
        refreshes the drone position every PERIODIC_SAVE_S while SEARCHING. Never raises.
        Returns True when a file was written."""
        fp = (mission.state, tuple(mission.plan), tuple(mission.visited),
              tuple(mission.destination) if mission.destination else None,
              mission.response, len(mission.history))
        now = self.clock()
        changed = fp != self._fingerprint
        due = mission.state == "SEARCHING" and now - self._last_save_t >= PERIODIC_SAVE_S
        if not (force or changed or due):
            return False
        try:
            self.save(self.build(mission, drone_pos), durable=force or changed)
        except Exception as exc:                         # disk full, read-only folder, ...
            self.last_error = f"{type(exc).__name__}: {exc}"
            if not self._warned:
                self._warned = True
                print(f"[state] WARNING: could not save mission state ({self.last_error}); "
                      "the mission continues without saving.", flush=True)
            self._last_save_t = now                      # do not retry on every tick
            return False
        self._fingerprint = fp
        self._last_save_t = now
        self.last_error = None
        self._warned = False
        return True

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


# ----------------------------------------------------- bridging the saved setup and main.py --
def setup_from_args(args):
    """The operator's starting information, taken from main.py's parsed arguments."""
    return {
        "lat": args.lat, "lon": args.lon, "area": args.area, "map_size": args.map_size,
        "osm_file": args.osm_file or None, "seed": int(args.seed or 0),
        "simple_visuals": bool(args.simple_visuals), "camera": args.camera,
        "max_time": args.max_time,
    }


def apply_setup_to_args(args, setup):
    """Copy a saved setup back into main.py's arguments so --resume rebuilds the same world."""
    args.lat, args.lon = setup["lat"], setup["lon"]
    args.area, args.map_size = setup["area"], setup["map_size"]
    args.osm_file, args.seed = setup["osm_file"], setup["seed"] or 0
    args.simple_visuals = bool(setup["simple_visuals"])
    args.camera, args.max_time = setup["camera"], setup["max_time"]
    return args


def describe(payload):
    """One line for the console, e.g. 'SEARCHING, 2 zone(s) searched, saved 3.1 s ago'."""
    m = payload["mission"]
    return f"{m['state']}, {len(m['visited'])} zone(s) searched, plan {m['plan']}"
