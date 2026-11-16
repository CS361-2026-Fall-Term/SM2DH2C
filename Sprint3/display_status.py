"""
Live mission display (PBI-08): the six fields the operator needs at once, plus latency timing.

    position | mission state | destination | current plan | moving / stopped | possible sighting

Everything here is plain Python (no PyBullet, no Tk), so it can be unit-tested and the same text
is used by the 3D overlay (main.py HUD) and by the operator panel.

    build_fields(...)      the six fields as short strings
    signature(fields)      what "the display changed" means (position rounded to 0.1 m)
    hud_line(fields)       the extra HUD line (position + motion)
    panel_text(fields)     the six-line block shown in the operator panel
    LatencyTracker         time from "state changed" to "drawn on screen", per surface
    StateFeed              keeps only the newest state update (out-of-order / dropped / late safe)

Latency limit: LATENCY_LIMIT_MS (300 ms, PBI-08).
"""
import math
import time

LATENCY_LIMIT_MS = 300.0
MOVING_MIN_SPEED = 0.05             # m/s horizontal; below this the drone counts as stopped
STOPPED_NAV_STATES = ("IDLE", "DONE", "FOUND", "RTL_DONE", "ABORT")
FIELD_NAMES = ("position", "mission", "destination", "plan", "motion", "sighting")
MAX_LINE = 110                      # HUD lines are cut at this many characters (readability)


def motion_state(nav_state, speed, holding=False):
    """MOVING / STOPPED / HOVERING (holding still to confirm a possible sighting)."""
    if holding:
        return "HOVERING"
    if nav_state in STOPPED_NAV_STATES or not math.isfinite(speed) or speed < MOVING_MIN_SPEED:
        return "STOPPED"
    return "MOVING"


def sighting_text(detection_state, confidence):
    if detection_state == "SIGHTING":
        return f"POSSIBLE SIGHTING ({confidence:.2f})"
    if detection_state == "FOUND":
        return f"PERSON FOUND ({confidence:.2f})"
    return f"none ({confidence:.2f})"


def build_fields(pos, mission_state, nav_state, speed, plan, destination, detection_state,
                 confidence, holding=False, next_label=None):
    """The six display fields. destination: (x, y) or None; next_label: e.g. 'Zone B' or
    'map cell (3.0, -2.0)' (shown with the coordinates)."""
    if destination is not None:
        where = f"({destination[0]:.1f}, {destination[1]:.1f})"
        dest = f"{next_label} {where}" if next_label else where
    else:
        dest = next_label or "none"
    return {
        "position": f"x {pos[0]:.1f}  y {pos[1]:.1f}  z {pos[2]:.1f} m",
        "mission": str(mission_state),
        "destination": dest,
        "plan": " -> ".join(plan) if plan else "none",
        "motion": f"{motion_state(nav_state, speed, holding)} ({nav_state})",
        "sighting": sighting_text(detection_state, confidence),
    }


def fields_from_sim(mission, nav, monitor, searcher=None, holding=False):
    """Collect the six fields from the live objects of main.py."""
    label = None
    dest = mission.destination
    if mission.plan:
        label = f"Zone {mission.plan[0]}"
    elif searcher is not None and getattr(searcher, "active", False) and searcher.target:
        label, dest = "map cell", tuple(searcher.target)
    return build_fields(nav.pos, mission.state, nav.state, nav.horizontal_speed, mission.plan,
                        dest, monitor.state, monitor.confidence, holding, label)


def signature(fields):
    """Display content as a comparable value. Position is rounded to 0.1 m (what is shown), so
    a change that is not visible does not count as a display update."""
    return tuple(fields[k] for k in FIELD_NAMES)


def hud_line(fields):
    """Position + motion line for the 3D overlay (the other fields are in the HUD already)."""
    return f"Pos: {fields['position']} | Motion: {fields['motion']}"


def panel_text(fields):
    """All six fields, one per line, for the operator panel's status block."""
    return (f"Position: {fields['position']}\n"
            f"Mission state: {fields['mission']}\n"
            f"Destination: {fields['destination']}\n"
            f"Current plan: {fields['plan']}\n"
            f"Moving / stopped: {fields['motion']}\n"
            f"Possible sighting: {fields['sighting']}")


def clip(text, limit=MAX_LINE):
    """Keep a HUD line readable: very long messages are cut with '...'."""
    text = str(text)
    return text if len(text) <= limit else text[:limit - 3] + "..."


class LatencyTracker:
    """Time from 'the displayed state changed' to 'it was drawn', for each surface ('hud',
    'panel'). observe() is called every tick with the current signature; shown() when a surface
    has drawn that signature. If the state changes again before a draw, the earlier change
    keeps counting (that is the real delay the operator sees). `clock` is injectable for tests."""

    def __init__(self, clock=time.perf_counter, limit_ms=LATENCY_LIMIT_MS):
        self.clock = clock
        self.limit_ms = limit_ms
        self.sig = None
        self.pending = {}               # surface -> time of the oldest undrawn change
        self.samples = {}               # surface -> [latency ms, ...]

    def observe(self, sig, surfaces=("hud", "panel"), now=None):
        now = self.clock() if now is None else now
        if sig != self.sig:
            self.sig = sig
            for s in surfaces:
                self.pending.setdefault(s, now)

    def shown(self, surface, sig, now=None):
        """Returns the latency in ms if this draw completed a pending change, else None."""
        now = self.clock() if now is None else now
        if sig != self.sig or surface not in self.pending:
            return None                 # drew an older state: still pending
        ms = (now - self.pending.pop(surface)) * 1000.0
        self.samples.setdefault(surface, []).append(ms)
        return ms

    @staticmethod
    def _p95(values):
        v = sorted(values)
        return v[min(len(v) - 1, math.ceil(0.95 * len(v)) - 1)]

    def summary(self):
        """{surface: dict(count, mean_ms, p95_ms, max_ms, over_limit)}"""
        out = {}
        for s, v in self.samples.items():
            if v:
                out[s] = dict(count=len(v), mean_ms=sum(v) / len(v), p95_ms=self._p95(v),
                              max_ms=max(v), over_limit=sum(1 for x in v if x > self.limit_ms))
        return out

    def report(self):
        rows = [f"{s}: {d['count']} updates, mean {d['mean_ms']:.1f} ms, p95 {d['p95_ms']:.1f} ms, "
                f"worst {d['max_ms']:.1f} ms, over {self.limit_ms:g} ms: {d['over_limit']}"
                for s, d in self.summary().items()]
        return "Display latency - " + (" | ".join(rows) if rows else "no updates recorded")


class StateFeed:
    """Newest-wins buffer between whatever produces mission updates and the display. Updates
    carry a sequence number; late, duplicate or out-of-order ones (weak network) are ignored,
    so the display never goes back to an older state."""

    def __init__(self):
        self.seq = -1
        self.state = None
        self.arrived_at = None
        self.dropped_stale = 0

    def accept(self, seq, state, now):
        if seq <= self.seq:
            self.dropped_stale += 1
            return False
        self.seq, self.state, self.arrived_at = seq, state, now
        return True
