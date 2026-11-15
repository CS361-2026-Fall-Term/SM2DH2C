"""
PBI-09 tests: live operator commands for every zone, safety rules, timing.
No simulator needed.   Run:  python -m unittest tests.test_operator_guidance -v

Covers
  AC-01  every command works for every zone and changes the plan correctly
  AC-02  every safety rule rejects bad commands; validation < 10 ms; rejected = nothing changes
  AC-04  bad input of any kind returns an error message and never raises (no crash)
"""
import random
import time
import unittest

import operator_guidance as og
from operator_guidance import Cmd, Mission, ZONES, ZONE_COMMANDS, ORIGINAL_PLAN

VALIDATION_LIMIT_MS = 10.0


def started():
    m = Mission()
    m.start()
    return m


def snapshot(m):
    return (list(m.plan), list(m.excluded), list(m.visited), m.destination, m.state)


class EveryCommandEveryZone(unittest.TestCase):
    """AC-01: each command x each zone, starting from a fresh mission (plan A, B, C, D)."""

    def test_investigate_each_zone(self):
        for z in ZONES:
            with self.subTest(zone=z):
                m = started()
                r = m.handle(Cmd.INVESTIGATE, z)
                if z == ORIGINAL_PLAN[0]:
                    self.assertFalse(r.ok)                       # already heading there
                    continue
                self.assertTrue(r.ok, r.message)
                self.assertEqual(m.plan[0], z)
                self.assertEqual(m.destination, ZONES[z])
                self.assertTrue(r.reroute)
                self.assertEqual(r.sighting, ZONES[z])           # sighting raises probability there
                self.assertEqual(sorted(m.plan), sorted(ORIGINAL_PLAN))

    def test_search_first_each_zone(self):
        for z in ZONES:
            with self.subTest(zone=z):
                m = started()
                r = m.handle(Cmd.SEARCH_FIRST, z)
                if z == ORIGINAL_PLAN[0]:
                    self.assertFalse(r.ok)                       # already first
                    continue
                self.assertTrue(r.ok, r.message)
                self.assertEqual(m.plan[0], z)
                self.assertIsNone(r.sighting)
                rest = [x for x in ORIGINAL_PLAN if x != z]
                self.assertEqual(m.plan[1:], rest)               # the rest keeps its order

    def test_avoid_each_zone(self):
        for z in ZONES:
            with self.subTest(zone=z):
                m = started()
                r = m.handle(Cmd.AVOID, z)
                self.assertTrue(r.ok, r.message)
                self.assertNotIn(z, m.plan)
                self.assertIn(z, m.excluded)
                self.assertNotIn(z, m.visited)                   # excluded is not "searched"
                self.assertEqual(r.reroute, z == ORIGINAL_PLAN[0])

    def test_reinclude_each_zone(self):
        for z in ZONES:
            with self.subTest(zone=z):
                m = started()
                self.assertTrue(m.handle(Cmd.AVOID, z).ok)
                r = m.handle(Cmd.REINCLUDE, z)
                self.assertTrue(r.ok, r.message)
                self.assertEqual(m.plan[-1], z)                  # comes back at the end
                self.assertNotIn(z, m.excluded)

    def test_return_to_original_plan(self):
        m = started()
        m.handle(Cmd.SEARCH_FIRST, "D")
        m.handle(Cmd.AVOID, "B")
        m.arrived()                                           # D searched
        r = m.handle(Cmd.RETURN_PLAN)
        self.assertTrue(r.ok, r.message)
        self.assertEqual(m.plan, [z for z in ORIGINAL_PLAN if z != "D"])
        self.assertEqual(m.excluded, [])

    def test_lowercase_and_spaces_accepted(self):
        m = started()
        self.assertTrue(m.handle(Cmd.AVOID, " c ").ok)
        self.assertTrue(m.handle("reinclude", "C").ok)        # command by name also works


class SafetyRules(unittest.TestCase):
    """AC-02 / AC-04: every rule rejects the command, with a reason, and changes nothing."""

    def assertRejected(self, m, cmd, zone=None, contains=""):
        before = snapshot(m)
        r = m.handle(cmd, zone)
        self.assertFalse(r.ok, f"{cmd} {zone} should be rejected")
        self.assertTrue(r.message.startswith("Rejected:"), r.message)
        self.assertIn(contains, r.message)
        self.assertEqual(snapshot(m), before, "a rejected command must not change the mission")
        self.assertFalse(r.reroute)
        return r

    def test_no_mission_running(self):
        m = Mission()                                         # IDLE
        for cmd in Cmd:
            self.assertRejected(m, cmd, "B", "IDLE")
        m.start()
        m.found()                                             # FOUND
        for cmd in Cmd:
            self.assertRejected(m, cmd, "B", "FOUND")

    def test_excluding_an_excluded_zone(self):
        m = started()
        m.handle(Cmd.AVOID, "C")
        self.assertRejected(m, Cmd.AVOID, "C", "already excluded")

    def test_reinclude_zone_not_excluded(self):
        m = started()
        for z in ZONES:
            self.assertRejected(m, Cmd.REINCLUDE, z, "not excluded")

    def test_commands_on_searched_zone(self):
        m = started()
        m.arrived()                                           # A searched
        for cmd in (Cmd.INVESTIGATE, Cmd.SEARCH_FIRST, Cmd.AVOID):
            self.assertRejected(m, cmd, "A", "already been searched")

    def test_investigate_or_prioritise_excluded_zone(self):
        m = started()
        m.handle(Cmd.AVOID, "D")
        self.assertRejected(m, Cmd.INVESTIGATE, "D", "re-include it first")
        self.assertRejected(m, Cmd.SEARCH_FIRST, "D", "re-include it first")

    def test_excluding_last_remaining_zone(self):
        m = started()
        for z in ("A", "B", "C"):
            m.handle(Cmd.AVOID, z)
        self.assertRejected(m, Cmd.AVOID, "D", "last zone")

    def test_return_when_already_on_original_plan(self):
        self.assertRejected(started(), Cmd.RETURN_PLAN, None, "already following")

    def test_bad_inputs_never_crash(self):
        m = started()
        bad = [(Cmd.AVOID, None), (Cmd.AVOID, ""), (Cmd.AVOID, "Z"), (Cmd.AVOID, "AB"),
               (Cmd.AVOID, 3), (Cmd.AVOID, 2.5), (Cmd.AVOID, ["A"]), (Cmd.AVOID, object()),
               ("FLY_AWAY", "A"), (None, "A"), (42, "A"), ("", None), (b"AVOID", "A")]
        for cmd, zone in bad:
            with self.subTest(cmd=cmd, zone=zone):
                self.assertRejected(m, cmd, zone)


class Timing(unittest.TestCase):
    """AC-02: validation in under 10 ms (worst case over many commands); AC-03 backend part."""

    def test_validation_under_10_ms(self):
        rng = random.Random(1)
        worst, total, n = 0.0, 0.0, 0
        for _ in range(500):
            m = started()
            for _ in range(20):
                cmd = rng.choice(list(Cmd))
                r = m.handle(cmd, rng.choice(list(ZONES) + ["Z", None]))
                worst = max(worst, r.validate_ms)
                total += r.validate_ms
                n += 1
                if rng.random() < 0.1:
                    m.arrived()
        print(f"\n  validation: {n} commands, mean {total / n:.4f} ms, worst {worst:.4f} ms")
        self.assertLess(worst, VALIDATION_LIMIT_MS)

    def test_whole_command_under_1_second(self):
        """Validation + applying the change + building the confirmation message."""
        m = started()
        t0 = time.perf_counter()
        r = m.handle(Cmd.INVESTIGATE, "C")
        ms = (time.perf_counter() - t0) * 1000
        self.assertTrue(r.ok)
        self.assertIn("Received", m.response)
        self.assertLess(ms, 1000)


class RandomStress(unittest.TestCase):
    """Thousands of random (valid and invalid) commands: never crashes, plan stays consistent."""

    def test_invariants_hold(self):
        rng = random.Random(7)
        junk = ["Z", None, "", 5, "a", "d "]
        for trial in range(200):
            m = started()
            for _ in range(40):
                cmd = rng.choice(list(Cmd) + ["AVOID", "bogus", None])
                m.handle(cmd, rng.choice(list(ZONES) + junk))
                if rng.random() < 0.15 and m.plan:
                    m.arrived()
                # every zone is in exactly one of: plan, excluded, searched
                groups = m.plan + m.excluded + m.visited
                self.assertEqual(sorted(groups), sorted(ZONES), (m.plan, m.excluded, m.visited))
                self.assertEqual(m.destination, ZONES[m.plan[0]] if m.plan else None)


class HistoryLog(unittest.TestCase):
    def test_every_command_is_logged(self):
        m = started()
        m.handle(Cmd.AVOID, "B")
        m.handle(Cmd.AVOID, "B")
        self.assertEqual(m.history[-2][1], "AVOID B")
        self.assertIn("Received", m.history[-2][2])
        self.assertIn("Rejected", m.history[-1][2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
