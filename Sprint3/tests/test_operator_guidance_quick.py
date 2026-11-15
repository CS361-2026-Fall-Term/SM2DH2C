"""
Quick smoke tests for PBI-09 operator commands (operator_guidance.Mission). No simulator needed.
Run from the Sprint folder:  python -m unittest tests.test_operator_guidance_quick -v
(also works with pytest, or directly: python tests/test_operator_guidance_quick.py)
"""
import os
import sys
import unittest

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the Sprint folder
if CODE_DIR not in sys.path:
    sys.path.insert(0, CODE_DIR)

from operator_guidance import Mission, Cmd  # noqa: E402


class QuickCommandTests(unittest.TestCase):
    def test_search_first(self):
        mission = Mission()
        mission.start()
        result = mission.handle(Cmd.SEARCH_FIRST, "C")
        self.assertTrue(result.ok)
        self.assertEqual(mission.plan[0], "C")

    def test_avoid_zone(self):
        mission = Mission()
        mission.start()
        result = mission.handle(Cmd.AVOID, "B")
        self.assertTrue(result.ok)
        self.assertNotIn("B", mission.plan)

    def test_reinclude_zone(self):
        mission = Mission()
        mission.start()
        mission.handle(Cmd.AVOID, "B")
        result = mission.handle(Cmd.REINCLUDE, "B")
        self.assertTrue(result.ok)
        self.assertEqual(mission.plan[-1], "B")

    def test_invalid_zone(self):
        mission = Mission()
        mission.start()
        result = mission.handle(Cmd.AVOID, "Z")
        self.assertFalse(result.ok)
        self.assertEqual(mission.plan, ["A", "B", "C", "D"])

    def test_before_start(self):
        mission = Mission()
        result = mission.handle(Cmd.SEARCH_FIRST, "B")
        self.assertFalse(result.ok)


if __name__ == "__main__":
    unittest.main()
