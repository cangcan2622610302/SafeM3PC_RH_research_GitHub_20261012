import pytest

from hsr_ttr.data import build_small_instance, build_standalone_recovery_scenarios


def test_four_standalone_recovery_cases_match_declared_durations():
    instance = build_small_instance()
    scenarios = build_standalone_recovery_scenarios(
        instance.disruption_start,
        extreme_s1_duration=210,
    )

    assert [item.scenario_id for item in scenarios] == [
        "A_fast",
        "B_base",
        "C_slow",
        "D_extreme",
    ]
    assert [item.t12 - instance.disruption_start for item in scenarios] == [
        90,
        120,
        150,
        210,
    ]
    assert [item.t23 - item.t12 for item in scenarios] == [30, 45, 60, 60]
    assert [item.s2_speed_kmh for item in scenarios] == [200, 160, 120, 120]
    assert all(item.probability == 1.0 for item in scenarios)


def test_extreme_case_accepts_only_the_declared_stress_durations():
    scenarios = build_standalone_recovery_scenarios(extreme_s1_duration=180)
    assert scenarios[-1].t12 - 60 == 180

    with pytest.raises(ValueError, match="180 or 210"):
        build_standalone_recovery_scenarios(extreme_s1_duration=195)
