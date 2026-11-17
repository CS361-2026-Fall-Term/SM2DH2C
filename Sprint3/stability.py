"""
PBI-02: simulation stability and performance during drone search missions.

StabilityMonitor is switched on with `python main.py --metrics` (with or without the window).
During the mission it records:

  performance  work time per 60 Hz frame (logic + physics + drawing, without the pacing sleep),
               frames per second shown (window only), how much faster than real time the
               headless simulation runs, start-up time, process memory (RSS) over the mission
  glitches     the drone's position is a real number, inside the map and above the ground; it
               never jumps further in one step than it can fly; it never ends up inside an
               obstacle (ground truth from PyBullet, checked every CONTACT_EVERY steps)
  environment  every obstacle, the floor and the person are in the same place, with the same
               orientation, at the end of the mission as at the start, and no body appeared or
               disappeared (AC-02)

At the end main.py prints a readable table (metric, value, limit, PASS / FAIL) and one
machine-readable line `Stability: {...}` used by stability_runs.py and the tests.
The limits (METRICS) are the team's acceptable performance metrics, see
tests/STABILITY_PERFORMANCE_REPORT.md.
"""
import math
import os
import sys
import time

# ------------------------------------------------------------------ acceptable performance metrics
FRAME_BUDGET_MS = 1000.0 / 60.0          # the simulation draws at 60 Hz
METRICS = {
    # name: (limit, "max" = value must be <= limit / "min" = value must be >= limit, unit, meaning)
    "frame_work_mean_ms": (FRAME_BUDGET_MS, "max", "ms", "average work per 60 Hz frame (fits the frame budget)"),
    "frame_work_p95_ms": (2 * FRAME_BUDGET_MS, "max", "ms", "95 % of frames done in time for >= 30 FPS"),
    "fps": (30.0, "min", "FPS", "frames shown per second (window only)"),
    "realtime_factor": (1.0, "min", "x", "headless: simulated seconds per real second"),
    "memory_growth_mb": (50.0, "max", "MB", "process memory growth over the mission (after start-up)"),
    "startup_s": (10.0, "max", "s", "start-up: map, world and probability map built"),
    "glitches": (0, "max", "", "drone NaN / outside the map / below ground / jumps / inside an obstacle"),
    "moved_objects": (0, "max", "", "obstacles, floor or person moved, appeared or disappeared"),
}
WARMUP_S = 5.0                 # memory growth is measured from this sim time (after start-up allocations)
MEMORY_EVERY_S = 1.0           # memory sample interval (sim seconds)
CONTACT_EVERY = 10             # ground-truth "inside an obstacle" check every this many logic steps
JUMP_SLACK = 1.5               # a step may move at most this x the fastest the drone can fly ...
JUMP_SNAP_M = 0.02             # ... plus this: the navigator snaps onto a waypoint within 1.5 cm
MOVE_TOLERANCE = 1e-3          # m / quaternion component: anything more counts as moved
MAX_GLITCH_REPORTS = 20


def _windows_rss_mb():
    """Working set of this process on Windows, through the Windows API (ctypes, no extra package).
    The argument and return types must be declared: without them ctypes passes the process handle
    (-1) as a 32-bit int, which on 64-bit Python is not the same handle, and the call fails."""
    try:
        import ctypes
        from ctypes import wintypes

        class PMC(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.GetCurrentProcess.argtypes = []
        handle = kernel32.GetCurrentProcess()
        # Windows 7+ has it in kernel32 (K32...); older systems only in psapi.dll
        candidates = []
        if hasattr(kernel32, "K32GetProcessMemoryInfo"):
            candidates.append(kernel32.K32GetProcessMemoryInfo)
        try:
            candidates.append(ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo)
        except OSError:
            pass
        for fn in candidates:
            fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            fn.restype = wintypes.BOOL
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            if fn(handle, ctypes.byref(pmc), pmc.cb):
                return pmc.WorkingSetSize / 1e6
    except Exception:
        pass
    return None


def rss_mb():
    """Resident memory of this process in MB, or None if it cannot be read (no new dependencies:
    psutil if installed, else /proc on Linux, the Windows API, or ru_maxrss on macOS)."""
    try:
        import psutil                                     # optional
        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as f:               # Linux: pages
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e6
    except Exception:
        pass
    if sys.platform == "win32":
        mb = _windows_rss_mb()
        if mb is not None:
            return mb
    try:
        import resource                                    # macOS: peak only (bytes)
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / 1e6 if sys.platform == "darwin" else peak / 1e3
    except Exception:
        return None


def percentile(values, q):
    if not values:
        return None
    s = sorted(values)
    k = min(len(s) - 1, max(0, int(math.ceil(q / 100.0 * len(s))) - 1))
    return s[k]


def body_pose(p, uid):
    pos, orn = p.getBasePositionAndOrientation(uid)
    return tuple(pos), tuple(orn)


def snapshot_environment(p, body_ids):
    """Pose of every environment body (obstacles, floor, person) and the number of bodies."""
    return {"poses": {int(u): body_pose(p, u) for u in body_ids}, "num_bodies": p.getNumBodies()}


def compare_environment(p, before, tolerance=MOVE_TOLERANCE):
    """List of what changed since snapshot_environment(): moved / rotated / missing bodies and a
    changed body count (bodies drawn for the operator, e.g. markers, are added at start-up only)."""
    changes = []
    for uid, (pos0, orn0) in before["poses"].items():
        try:
            pos1, orn1 = body_pose(p, uid)
        except Exception:
            changes.append(f"body {uid} disappeared")
            continue
        moved = max(abs(a - b) for a, b in zip(pos0, pos1))
        turned = max(abs(a - b) for a, b in zip(orn0, orn1))
        if moved > tolerance or turned > tolerance:
            changes.append(f"body {uid} moved {moved:.4f} m / rotated {turned:.4f} "
                           f"(from {tuple(round(c, 3) for c in pos0)} to {tuple(round(c, 3) for c in pos1)})")
    n = p.getNumBodies()
    if n != before["num_bodies"]:
        changes.append(f"number of bodies changed from {before['num_bodies']} to {n}")
    return changes


class StabilityMonitor:
    """Collects the PBI-02 measurements during one mission. All methods are cheap no-ops when
    enabled=False, so main.py can always call them."""

    def __init__(self, enabled, gui=False, clock=time.perf_counter):
        self.enabled, self.gui, self.clock = enabled, gui, clock
        self.t_program = clock()
        self.startup_s = None
        self.frame_ms = []
        self.frames_shown = 0
        self.loop_wall_s = 0.0
        self.sim_s = 0.0
        self.memory = []                  # (sim time, MB)
        self.glitches = []                # text of each glitch (at most MAX_GLITCH_REPORTS kept)
        self.glitch_count = 0
        self.env_before = None
        self.env_changes = []
        self._frame_t0 = None
        self._loop_t0 = None
        self._last_pos = None
        self._last_t = None
        self._next_mem = 0.0
        self._steps = 0

    # ---- start-up and environment
    def started(self, p, environment_ids):
        """Call once the world is built, just before the mission loop."""
        if not self.enabled:
            return
        self.startup_s = self.clock() - self.t_program
        self.env_before = snapshot_environment(p, environment_ids)
        self._loop_t0 = self._frame_t0 = self.clock()

    # ---- per logic step
    def step(self, p, pos, t, drone_id=None, obstacle_ids=(), map_half=None, v_max=None, dt=None,
             max_alt=None, v_z=0.0):
        if not self.enabled:
            return
        self._steps += 1
        x, y, z = pos
        if not all(math.isfinite(c) for c in pos):
            self._glitch(t, f"drone position is not a number: {tuple(pos)}")
        else:
            if map_half is not None and (abs(x) > map_half + 1.0 or abs(y) > map_half + 1.0):
                self._glitch(t, f"drone outside the map at ({x:.2f}, {y:.2f})")
            if z < 0.0 or (max_alt is not None and z > max_alt + 1.0):
                self._glitch(t, f"drone altitude {z:.2f} m out of range")
            if self._last_pos is not None and v_max and dt and self._last_t is not None:
                elapsed = max(t - self._last_t, dt)
                jump = math.dist(pos, self._last_pos)
                if jump > JUMP_SLACK * math.hypot(v_max, v_z) * elapsed + JUMP_SNAP_M:
                    self._glitch(t, f"drone jumped {jump:.3f} m in {elapsed * 1000:.1f} ms ({jump / elapsed:.2f} m/s)")
            self._last_pos, self._last_t = tuple(pos), t
            if drone_id is not None and obstacle_ids and self._steps % CONTACT_EVERY == 0:
                self._check_contact(p, drone_id, obstacle_ids, t)
        if t >= self._next_mem:
            mb = rss_mb()
            if mb is not None:
                self.memory.append((round(t, 2), round(mb, 1)))
            self._next_mem = t + MEMORY_EVERY_S

    def _check_contact(self, p, drone_id, obstacle_ids, t):
        lo, hi = p.getAABB(drone_id)
        near = {o[0] for o in (p.getOverlappingObjects(lo, hi) or [])}
        for uid in near & set(obstacle_ids):
            pts = p.getClosestPoints(drone_id, uid, 0.0)
            if pts and min(c[8] for c in pts) < -0.01:     # 1 cm tolerance for contact noise
                self._glitch(t, f"drone inside obstacle body {uid} ({min(c[8] for c in pts):.3f} m)")

    def _glitch(self, t, text):
        self.glitch_count += 1
        if len(self.glitches) < MAX_GLITCH_REPORTS:
            self.glitches.append(f"t={t:.2f}s {text}")

    # ---- per 60 Hz frame
    def frame_done(self, shown=False):
        """Call at the end of each 60 Hz frame (before any pacing sleep). shown: it was drawn."""
        if not self.enabled:
            return
        now = self.clock()
        if self._frame_t0 is not None:
            self.frame_ms.append((now - self._frame_t0) * 1000.0)
        self.frames_shown += 1 if shown else 0

    def frame_start(self):
        """Start timing the next frame (after the pacing sleep)."""
        if self.enabled:
            self._frame_t0 = self.clock()

    # ---- end
    def finish(self, p, sim_t):
        if not self.enabled:
            return
        if self._loop_t0 is not None:
            self.loop_wall_s = self.clock() - self._loop_t0
        self.sim_s = sim_t
        if self.env_before is not None and p.isConnected():
            self.env_changes = compare_environment(p, self.env_before)

    def summary(self):
        mem_after = [mb for t, mb in self.memory if t >= WARMUP_S] or [mb for _t, mb in self.memory]
        values = {
            "frame_work_mean_ms": (sum(self.frame_ms) / len(self.frame_ms)) if self.frame_ms else None,
            "frame_work_p95_ms": percentile(self.frame_ms, 95),
            "frame_work_max_ms": max(self.frame_ms) if self.frame_ms else None,
            "fps": (self.frames_shown / self.loop_wall_s) if (self.gui and self.loop_wall_s > 0) else None,
            "realtime_factor": (self.sim_s / self.loop_wall_s) if (not self.gui and self.loop_wall_s > 0) else None,
            "memory_start_mb": mem_after[0] if mem_after else None,
            "memory_end_mb": mem_after[-1] if mem_after else None,
            "memory_peak_mb": max(mem_after) if mem_after else None,
            "memory_growth_mb": (mem_after[-1] - mem_after[0]) if mem_after else None,
            "memory_trend_mb_per_min": self._trend(),
            "startup_s": self.startup_s,
            "glitches": self.glitch_count,
            "moved_objects": len(self.env_changes),
            "frames": len(self.frame_ms),
            "sim_s": self.sim_s,
            "wall_s": self.loop_wall_s,
            "gui": self.gui,
        }
        return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in values.items()}

    def _trend(self):
        """Memory slope over the second half of the mission (MB per sim minute): about 0 when
        memory has levelled off, positive and steady when something keeps growing (a leak)."""
        pts = [(t, mb) for t, mb in self.memory if t >= WARMUP_S]
        pts = pts[len(pts) // 2:]
        if len(pts) < 5:
            return None
        n = len(pts)
        mt = sum(t for t, _ in pts) / n
        mm = sum(mb for _, mb in pts) / n
        var = sum((t - mt) ** 2 for t, _ in pts)
        return 60.0 * sum((t - mt) * (mb - mm) for t, mb in pts) / var if var else None

    def checks(self, summary=None):
        """[(metric, value, limit, kind, unit, meaning, passed)]; a metric that does not apply to
        this run (fps headless, realtime_factor with the window) has value None and passes."""
        s = summary or self.summary()
        rows = []
        for name, (limit, kind, unit, meaning) in METRICS.items():
            v = s.get(name)
            ok = True if v is None else (v <= limit if kind == "max" else v >= limit)
            rows.append((name, v, limit, kind, unit, meaning, ok))
        return rows

    def passed(self):
        return all(row[-1] for row in self.checks())

    def format_report(self):
        s = self.summary()
        lines = ["Stability and performance (PBI-02)", "----------------------------------",
                 f"  {'metric':<20} {'value':>10}  {'limit':>12}  result"]
        for name, v, limit, kind, unit, meaning, ok in self.checks(s):
            shown = "n/a" if v is None else (f"{v:.2f} {unit}".strip() if isinstance(v, float) else f"{v} {unit}".strip())
            lim = f"{'<=' if kind == 'max' else '>='} {limit:g} {unit}".strip()
            lines.append(f"  {name:<20} {shown:>10}  {lim:>12}  {'PASS' if ok else 'FAIL'}"
                         + ("" if v is not None else " (not measured in this mode)"))
        lines.append(f"  {s['frames']} frames, {s['sim_s']:.1f} s simulated in {s['wall_s']:.1f} s; "
                     f"memory {s['memory_start_mb']} -> {s['memory_end_mb']} MB (peak {s['memory_peak_mb']}, "
                     f"trend {s['memory_trend_mb_per_min']} MB/min over the second half)")
        for g in self.glitches:
            lines.append(f"  glitch: {g}")
        if self.glitch_count > len(self.glitches):
            lines.append(f"  ... {self.glitch_count - len(self.glitches)} more glitches")
        for c in self.env_changes:
            lines.append(f"  environment changed: {c}")
        lines.append(f"  Overall: {'PASS' if self.passed() else 'FAIL'}")
        return "\n".join(lines)
