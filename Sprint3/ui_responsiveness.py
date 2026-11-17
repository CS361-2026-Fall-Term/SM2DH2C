"""
PBI-07: keep the operator dashboard responsive under heavy live input.

Goal (acceptance criteria)
  AC-01  the dashboard does not lag while live data keeps coming in
  AC-02  an operator guidance button click is processed and shown within RESPONSE_LIMIT_MS (50 ms)
  AC-03  heavy input still updates the drone display within 50 ms, without crashes

What was slow (measured before this PBI, see tests/UI_RESPONSIVENESS_REPORT.md)
  1. A click waited for the next panel poll (20 Hz -> up to 50 ms) before it was even read.
  2. The click's heavy side effects (probability-map update + A* route re-plan, 5-27 ms in the
     test world, more in big OSM worlds) ran BEFORE the panel was updated.
  3. The panel then only drew the reply at its next poll - another 50 ms later.
  4. The message log wrote live messages one Tk call at a time: a burst of 1000 froze it ~440 ms.
  5. Every mission change was saved to disk on the UI thread: 6-34 ms per frame with live data.

What this module provides (main.py / operator_panel.py / mission_state.py use it)
  CommandPipeline  splits a guidance command in two stages:
                     submit()      fast: validate it and update the mission (< 0.1 ms), so the
                                   panel can show the reply straight away
                     run_effects() heavy: probability map + route re-plan, run AFTER the panel
                                   has drawn. Several clicks before it runs share ONE re-plan to
                                   the newest destination (the end result is the same).
  ResponseTimer    collects click -> on-screen and refresh timings and checks them against 50 ms.

Why not a thread for the re-plan: it reads and draws through PyBullet and changes the navigator
the flight loop is using, and neither PyBullet nor Tk may be used from two threads. So the heavy
command work is moved behind the UI update instead; the disk saves, which only need a copy of the
data, do run on a worker thread (mission_state.BackgroundSaver).
"""
import math
import time

RESPONSE_LIMIT_MS = 50.0        # AC-02 / AC-03


class ResponseTimer:
    """Timings in ms, per name (e.g. 'click', 'refresh'), checked against limit_ms."""

    def __init__(self, limit_ms=RESPONSE_LIMIT_MS, keep=10000):
        self.limit_ms = limit_ms
        self.keep = keep
        self.samples = {}

    def add(self, name, ms):
        v = self.samples.setdefault(name, [])
        v.append(float(ms))
        if len(v) > self.keep:
            del v[:len(v) - self.keep]
        return ms

    @staticmethod
    def _p95(values):
        v = sorted(values)
        return v[min(len(v) - 1, math.ceil(0.95 * len(v)) - 1)]

    def summary(self):
        """{name: dict(count, mean_ms, p95_ms, max_ms, over_limit)}"""
        out = {}
        for name, v in self.samples.items():
            if v:
                out[name] = dict(count=len(v), mean_ms=sum(v) / len(v), p95_ms=self._p95(v),
                                 max_ms=max(v), over_limit=sum(1 for x in v if x > self.limit_ms))
        return out

    def report(self):
        rows = [f"{n}: {d['count']} x, mean {d['mean_ms']:.1f} ms, p95 {d['p95_ms']:.1f} ms, "
                f"worst {d['max_ms']:.1f} ms, over {self.limit_ms:g} ms: {d['over_limit']}"
                for n, d in self.summary().items()]
        return "UI response - " + (" | ".join(rows) if rows else "no operator input recorded")


class CommandPipeline:
    """Operator guidance commands in two stages (see the module docstring).

    mission    operator_guidance.Mission (validates and applies the command)
    apply_map  function(CommandResult): probability-map update for an accepted command
    follow     function(): make the drone follow mission.destination (re-plan the route)
    clock      injectable for tests
    """

    def __init__(self, mission, apply_map=None, follow=None, clock=time.perf_counter):
        self.mission = mission
        self.apply_map = apply_map
        self.follow = follow
        self.clock = clock
        self._map_jobs = []             # accepted results whose map update has not run yet
        self._reroute = False           # a re-plan is owed (coalesced)
        self._later = []                # other deferred work (functions)
        self.submitted = 0
        self.replans = 0
        self.replans_saved = 0          # re-plans skipped because a newer one replaced them
        self.last_effects_ms = 0.0

    def submit(self, cmd, zone=None):
        """Stage 1 (fast, before the UI update): validate and apply the command to the mission.
        Returns the CommandResult; a rejected command queues nothing."""
        res = self.mission.handle(cmd, zone)
        self.submitted += 1
        if res.ok:
            self._map_jobs.append(res)
            if res.reroute:
                if self._reroute:
                    self.replans_saved += 1
                self._reroute = True
        return res

    def request_follow(self):
        """Ask for a re-plan to the mission's destination in stage 2 (e.g. after START)."""
        if self._reroute:
            self.replans_saved += 1
        self._reroute = True

    def defer(self, fn):
        """Any other slow work a click causes (e.g. recolouring every heatmap tile), run in
        stage 2 after the panel has drawn."""
        self._later.append(fn)

    @property
    def pending(self):
        return bool(self._map_jobs) or bool(self._later) or self._reroute

    def run_effects(self):
        """Stage 2 (heavy, after the UI has drawn): every queued map update in order, other
        deferred work, then at most one re-plan to the mission's current destination.
        Returns the time it took (ms)."""
        if not self.pending:
            return 0.0
        t0 = self.clock()
        jobs, self._map_jobs = self._map_jobs, []
        if self.apply_map is not None:
            for res in jobs:
                self.apply_map(res)
        later, self._later = self._later, []
        for fn in later:
            fn()
        if self._reroute:
            self._reroute = False
            self.replans += 1
            if self.follow is not None:
                self.follow()
        self.last_effects_ms = (self.clock() - t0) * 1000.0
        return self.last_effects_ms
