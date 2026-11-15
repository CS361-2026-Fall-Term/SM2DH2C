"""
Contributors: Sofia Monge and Mylan Bui
File Description: Unit tests for the PBI-03 inputs of the mission setup window (person category and
zones to exclude). Tests the validation only; no window is opened.
Run from the folder holding operator_setup.py and tests/:  python -m unittest discover tests -v
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import operator_setup as setup  # noqa: E402


def raw(**changes):
    values = dict(setup.DEFAULTS, mode="practice", area="26")
    values.update(changes)
    return values


class TestPersonCategory(unittest.TestCase):
    def test_default_is_hiker(self):
        cfg, _ = setup.validate_inputs(raw())
        self.assertEqual(cfg["person_category"], "hiker")

    def test_every_listed_category_is_accepted(self):
        for category in setup.CATEGORIES:
            with self.subTest(category=category):
                cfg, msgs = setup.validate_inputs(raw(person_category=category))
                self.assertEqual(cfg["person_category"], category)
                self.assertTrue(msgs["person_category"][0])

    def test_unknown_category_is_rejected_with_a_message(self):
        cfg, msgs = setup.validate_inputs(raw(person_category="astronaut"))
        self.assertIsNone(cfg)
        ok, text = msgs["person_category"]
        self.assertFalse(ok)
        self.assertIn("hiker", text)


class TestExcludeZones(unittest.TestCase):
    def test_none_by_default(self):
        cfg, _ = setup.validate_inputs(raw())
        self.assertEqual(cfg["exclude_zones"], [])

    def test_letters_are_parsed_sorted_and_deduplicated(self):
        for text in ("CA", "a, c", "C A A"):
            with self.subTest(text=text):
                cfg, _ = setup.validate_inputs(raw(exclude_zones=text))
                self.assertEqual(cfg["exclude_zones"], ["A", "C"])

    def test_unknown_zone_is_rejected(self):
        cfg, msgs = setup.validate_inputs(raw(exclude_zones="AZ"))
        self.assertIsNone(cfg)
        self.assertIn("Z", msgs["exclude_zones"][1])

    def test_cannot_exclude_every_zone(self):
        cfg, msgs = setup.validate_inputs(raw(exclude_zones="".join(setup.ZONES)))
        self.assertIsNone(cfg)
        self.assertIn("At least one zone", msgs["exclude_zones"][1])

    def test_summary_and_prefill_round_trip(self):
        cfg, _ = setup.validate_inputs(raw(person_category="child", exclude_zones="B"))
        self.assertIn("child", setup.describe(cfg))
        self.assertIn("Excluded zones: B", setup.describe(cfg))
        again = setup.cfg_to_raw(cfg)
        self.assertEqual((again["person_category"], again["exclude_zones"]), ("child", "B"))


class TestLastKnownPositionAndArea(unittest.TestCase):
    """AC-07: an LKP outside the valid range or a search area outside 20-500 m is rejected with an
    error naming the problem, so the mission cannot start."""

    def osm_raw(self, **changes):
        values = dict(setup.DEFAULTS, mode="osm", lat="44.6497", lon="-123.3133", area="100")
        values.update(changes)
        return values

    def test_valid_lkp_and_area_accepted(self):
        cfg, _ = setup.validate_inputs(self.osm_raw())
        self.assertIsNotNone(cfg)

    def test_latitude_out_of_range(self):
        cfg, msgs = setup.validate_inputs(self.osm_raw(lat="95"))
        self.assertIsNone(cfg)
        self.assertFalse(msgs["lat"][0])
        self.assertIn("Latitude", msgs["lat"][1])

    def test_longitude_out_of_range(self):
        cfg, msgs = setup.validate_inputs(self.osm_raw(lon="-200"))
        self.assertIsNone(cfg)
        self.assertIn("Longitude", msgs["lon"][1])

    def test_lkp_not_a_number(self):
        cfg, msgs = setup.validate_inputs(self.osm_raw(lat="north"))
        self.assertIsNone(cfg)
        self.assertFalse(msgs["lat"][0])

    def test_search_area_outside_limits(self):
        for area in ("10", "600"):
            with self.subTest(area=area):
                cfg, msgs = setup.validate_inputs(self.osm_raw(area=area))
                self.assertIsNone(cfg)
                self.assertFalse(msgs["area"][0])


if __name__ == "__main__":
    unittest.main()