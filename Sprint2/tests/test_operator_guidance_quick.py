
from operator_guidance import Mission, Cmd


def test_search_first():
    mission = Mission()
    mission.start()
    result = mission.handle(Cmd.SEARCH_FIRST, "C")
    assert result.ok
    assert mission.plan[0] == "C"


def test_avoid_zone():
    mission = Mission()
    mission.start()
    result = mission.handle(Cmd.AVOID, "B")
    assert result.ok
    assert "B" not in mission.plan


def test_reinclude_zone():
    mission = Mission()
    mission.start()
    mission.handle(Cmd.AVOID, "B")
    result = mission.handle(Cmd.REINCLUDE, "B")
    assert result.ok
    assert mission.plan[-1] == "B"


def test_invalid_zone():
    mission = Mission()
    mission.start()
    result = mission.handle(Cmd.AVOID, "Z")
    assert not result.ok
    assert mission.plan == ["A", "B", "C", "D"]


def test_before_start():
    mission = Mission()
    result = mission.handle(Cmd.SEARCH_FIRST, "B")
    assert not result.ok


if __name__ == "__main__":
    test_search_first()
    test_avoid_zone()
    test_reinclude_zone()
    test_invalid_zone()
    test_before_start()
    print("All 5 tests passed")
