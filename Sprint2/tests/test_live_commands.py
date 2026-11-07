"""
PBI-09 tests: live commands reaching the drone and the operator (main.apply_operator_command).
Needs PyBullet installed (main.py imports it); skipped otherwise.
Run:  python -m unittest test_live_commands -v

Covers
  AC-01  accepted commands re-route the drone only when its destination changes;
         Investigate raises the probability map at the zone
  AC-03  input -> applied + confirmation message in well under 1 second
  AC-04  rejected commands leave the drone's route untouched
"""
import time
import unittest

try:
    import main
    HAVE_SIM = True
except ImportError:                     # PyBullet / numpy not installed
    HAVE_SIM = False

from operator_guidance import Cmd, Mission, ZONES


class FakeNav:
    def __init__(self):
        self.route, self.state, self.reroutes = [(0, 0, 1)], "NAV", []


class FakeBelief:
    def __init__(self):
        self.sightings = []

    def add_sighting(self, xy, sigma, weight):
        self.sightings.append(xy)


@unittest.skipUnless(HAVE_SIM, "PyBullet not installed")
class LiveCommands(unittest.TestCase):
    def setUp(self):
        self.m = Mission()
        self.m.start()
        self.nav, self.belief = FakeNav(), FakeBelief()

    def send(self, cmd, zone=None):
        return main.apply_operator_command(self.m, cmd, zone, self.nav, self.belief,
                                           reroute=lambda nav, xy: nav.reroutes.append(xy))

    def test_investigate_reroutes_and_marks_sighting(self):
        r = self.send(Cmd.INVESTIGATE, "C")
        self.assertTrue(r.ok)
        self.assertEqual(self.nav.reroutes, [ZONES["C"]])
        self.assertEqual(self.belief.sightings, [ZONES["C"]])

    def test_reorder_without_destination_change_does_not_reroute(self):
        self.send(Cmd.AVOID, "D")
        self.send(Cmd.REINCLUDE, "D")                        # A is still next
        self.assertEqual(self.nav.reroutes, [])

    def test_avoiding_current_destination_reroutes_to_next(self):
        self.send(Cmd.AVOID, "A")
        self.assertEqual(self.nav.reroutes, [ZONES["B"]])

    def test_rejected_command_leaves_route_alone(self):
        before = list(self.nav.route)
        r = self.send(Cmd.REINCLUDE, "B")                    # not excluded -> rejected
        self.assertFalse(r.ok)
        self.assertEqual(self.nav.route, before)
        self.assertEqual(self.nav.reroutes, [])
        self.assertEqual(self.belief.sightings, [])

    def test_bad_input_does_not_crash(self):
        for cmd, zone in ((Cmd.AVOID, "Q"), ("nonsense", None), (None, None), (Cmd.INVESTIGATE, 7)):
            r = self.send(cmd, zone)
            self.assertFalse(r.ok)
            self.assertIn("Rejected", self.m.response)

    def test_confirmation_well_under_one_second(self):
        worst = 0.0
        for cmd, zone in ((Cmd.INVESTIGATE, "D"), (Cmd.AVOID, "B"), (Cmd.REINCLUDE, "B"),
                          (Cmd.SEARCH_FIRST, "C"), (Cmd.RETURN_PLAN, None), (Cmd.AVOID, "Z")):
            t0 = time.perf_counter()
            self.send(cmd, zone)
            _ = self.m.response                               # the text the HUD shows
            worst = max(worst, (time.perf_counter() - t0) * 1000)
        print(f"\n  apply + confirmation message: worst {worst:.3f} ms")
        self.assertLess(worst, 1000)


if __name__ == "__main__":
    unittest.main(verbosity=2)
