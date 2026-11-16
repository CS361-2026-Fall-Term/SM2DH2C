import unittest
import time
from operator_guidance import Mission, MissionEvent
from target_detection import SearchMonitor
from object_avoidance import Navigator

class DummyPerson:
    def __init__(self):
        self.position = [10, 10, 0]
        self.name = "Test Person"

class TestMissionEvents(unittest.TestCase):
    def setUp(self):
        self.mission = Mission()
        self.mission.start()
        
        self.search_monitor = SearchMonitor(DummyPerson(), mission=self.mission)
        self.search_monitor.state = "SEARCHING"
        self.search_monitor.confidence = 0.0
        self.search_monitor.t = 1.0

        self.navigator = Navigator(drone_id=-1, floor_id=-1, target_id=-1, waypoints=[(0,0,5), (20,20,5)], gui=False, log=lambda x: None, mission=self.mission)
        self.navigator.t = 1.0
        self.navigator.pos = [0, 0, 5]

    def test_target_detection_updates_mission_state(self):
        # AC-01
        drone_pos = [0, 0, 5]
        obj_pos = [10, 10, 0]
        dist = 14.14
        
        # Trigger SIGHTING
        self.search_monitor._event("SIGHTING", "test sighting", drone_pos, obj_pos, dist)
        
        self.assertEqual(len(self.mission.mission_events), 1)
        latest = self.mission.latest_event
        self.assertEqual(latest.kind, "SIGHTING")
        self.assertEqual(latest.distance, dist)
        self.assertEqual(self.mission.state, "SIGHTING")

    def test_obstacle_event_includes_distance_and_response(self):
        # AC-02
        drone_pos = [0, 0, 5]
        obs_pos = [1, 1, 5]
        dist = 1.41
        
        # Trigger BLOCKED
        self.navigator.event("BLOCKED", "test blocked", drone_pos, obs_pos, dist, response="re-plan")
        
        self.assertEqual(len(self.mission.mission_events), 1)
        latest = self.mission.latest_event
        self.assertEqual(latest.kind, "BLOCKED")
        self.assertEqual(latest.distance, dist)
        self.assertEqual(latest.response, "re-plan")
        self.assertEqual(latest.severity, "CRITICAL")
        # Ensure core mission state is preserved
        self.assertEqual(self.mission.state, "SEARCHING")

if __name__ == '__main__':
    unittest.main()
