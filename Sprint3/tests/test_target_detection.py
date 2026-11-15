"""
Tests for the person detector (target_detection.SearchMonitor): clear-view detection distance,
range quality, occlusion, and keeping a sighting through a short loss of view.
Run from the Sprint folder (needs PyBullet; no window opens):
    python -m unittest tests.test_target_detection -v
"""
import math
import os
import sys
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

import pybullet as p                                    # noqa: E402

import target_detection as td                           # noqa: E402
from target_detection import MissingPerson, SearchMonitor   # noqa: E402

STAND_IN = "stand-in" in (p.__doc__ or "")
ALT = 2.5
FRAME = 1.0 / td.CAM_FPS


def world():
    if p.isConnected():
        p.disconnect()
    p.connect(p.DIRECT)
    import pybullet_data
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf")


def box_person(x, y):
    """A simple 1.7 m x 0.5 m x 0.3 m lying person (a box), at (x, y)."""
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.85, 0.25, 0.15])
    return MissingPerson(p.createMultiBody(0, col, -1, [x, y, 0.15]))


def wall(x, y, half):
    col = p.createCollisionShape(p.GEOM_BOX, halfExtents=list(half))
    return p.createMultiBody(0, col, -1, [x, y, half[2]])


def watch(monitor, cam, seconds, start=0.0):
    """Hover at cam looking at the person for `seconds`. Returns the time it was found (or None)."""
    target = monitor.person.position
    monitor.set_heading(math.atan2(target[1] - cam[1], target[0] - cam[0]))
    t = start
    while t < start + seconds:
        monitor.update(cam, t)
        if monitor.found:
            return t
        t += FRAME
    return None


def standoff(dist):
    """Camera position `dist` m (straight line) from a person at (0, 0), at the cruise altitude."""
    h = ALT - 0.15
    return [-math.sqrt(max(dist ** 2 - h ** 2, 0.0)), 0.0, ALT]


def quiet(*_a, **_k):
    pass


class TestClearView(unittest.TestCase):
    def setUp(self):
        world()
        self.person = box_person(0.0, 0.0)

    def tearDown(self):
        p.disconnect()

    def test_person_in_clear_view_is_found_within_a_few_frames(self):
        # EXPECT: 5 m away, nothing in front: found by "clear view" after CLEAR_VIEW_FRAMES frames.
        m = SearchMonitor(self.person, log=quiet)
        t = watch(m, standoff(5.0), 2.0)
        self.assertIsNotNone(t)
        self.assertEqual(m.found_by, "clear view")
        self.assertLessEqual(t, (td.CLEAR_VIEW_FRAMES - 1) * FRAME + 1e-6)

    def test_at_the_clear_view_range_still_found_instantly(self):
        m = SearchMonitor(self.person, log=quiet)
        t = watch(m, standoff(td.CLEAR_VIEW_RANGE - 0.1), 2.0)
        self.assertEqual(m.found_by, "clear view")
        self.assertLess(t, 0.5)

    def test_beyond_the_clear_view_range_found_by_confidence_while_watching(self):
        # EXPECT: 7 m away (beyond 6 m): not instant, but confidence builds up and finds the person.
        m = SearchMonitor(self.person, log=quiet)
        t = watch(m, standoff(7.0), 10.0)
        self.assertIsNotNone(t)
        self.assertEqual(m.found_by, "confidence")
        self.assertGreater(t, 0.5)

    def test_out_of_camera_range_never_found(self):
        m = SearchMonitor(self.person, log=quiet)
        self.assertIsNone(watch(m, standoff(td.CAM_RANGE + 1.0), 5.0))
        self.assertEqual(m.frames_seen, 0)

    def test_person_behind_a_wall_is_not_found(self):
        wall(-1.5, 0.0, (0.1, 2.0, 2.5))                     # between the drone and the person
        m = SearchMonitor(self.person, log=quiet)
        self.assertIsNone(watch(m, standoff(5.0), 5.0))
        self.assertGreater(m.frames_occluded, 0)

    def test_clear_view_threshold_can_be_changed(self):
        # EXPECT: with CLEAR_VIEW_RANGE = 4 m, a person 5 m away is no longer found instantly.
        old = td.CLEAR_VIEW_RANGE
        try:
            td.CLEAR_VIEW_RANGE = 4.0
            m = SearchMonitor(self.person, log=quiet)
            watch(m, standoff(5.0), 10.0)
            self.assertEqual(m.found_by, "confidence")
        finally:
            td.CLEAR_VIEW_RANGE = old

    def test_found_event_says_how_and_from_how_far(self):
        events = []
        m = SearchMonitor(self.person, log=events.append)
        watch(m, standoff(5.0), 2.0)
        self.assertTrue(any("clear view" in e and "5.0 m away" in e for e in events), events)
        self.assertEqual(m.report()["found_by"], "clear view")


class TestSightingMemory(unittest.TestCase):
    def setUp(self):
        world()
        self.person = box_person(0.0, 0.0)

    def tearDown(self):
        p.disconnect()

    def test_confidence_kept_through_a_short_loss_of_view_then_decays(self):
        # 7 m away (no clear view) for 1 s, then the camera turns away.
        m = SearchMonitor(self.person, log=quiet)
        cam = standoff(7.0)
        watch(m, cam, 1.0)
        built = m.confidence
        self.assertGreater(built, 0)
        m.set_heading(math.pi)                               # looking the other way
        t = 1.0
        while t < 1.0 + td.SIGHTING_MEMORY_S - 0.2:
            m.update(cam, t)
            t += FRAME
        self.assertAlmostEqual(m.confidence, built)          # kept during the memory time
        while t < 1.0 + td.SIGHTING_MEMORY_S + 2.0:
            m.update(cam, t)
            t += FRAME
        self.assertLess(m.confidence, built)                 # then it decays


class TestRangeQuality(unittest.TestCase):
    def test_full_up_to_five_metres_then_falls_to_the_minimum(self):
        self.assertEqual(td.range_quality(1.0), 1.0)
        self.assertEqual(td.range_quality(td.FULL_QUALITY_RANGE), 1.0)
        self.assertAlmostEqual(td.range_quality(td.CAM_RANGE), td.MIN_QUALITY)
        mid = td.range_quality((td.FULL_QUALITY_RANGE + td.CAM_RANGE) / 2)
        self.assertTrue(td.MIN_QUALITY < mid < 1.0)


@unittest.skipIf(STAND_IN, "needs the real humanoid model")
class TestHumanoid(unittest.TestCase):
    """The real casualty model (thin limbs): rays through the gaps between limbs are not hidden."""

    def test_lying_humanoid_in_the_open_is_found_by_clear_view(self):
        world()
        person = MissingPerson.spawn((0.0, 0.0))
        m = SearchMonitor(person, log=quiet)
        t = watch(m, standoff(5.0), 3.0)
        self.assertIsNotNone(t)
        self.assertEqual(m.found_by, "clear view")
        p.disconnect()


if __name__ == "__main__":
    unittest.main()
