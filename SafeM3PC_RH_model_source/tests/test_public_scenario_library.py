from hsr_ttr.paths import generate_path_pool
from hsr_ttr.public_data import (
    CORRIDOR_CODES,
    DISRUPTION_SECTION,
    build_public_macro_cases,
    scenario_catalog_rows,
)


def test_public_scenario_library_has_six_macros_and_eighteen_branches():
    cases = build_public_macro_cases()
    rows = scenario_catalog_rows(cases)

    assert len(cases) == 6
    assert [case.train_count for case in cases] == [15, 25, 31, 33, 35, 37]
    assert len(rows) == 18
    assert len({row["scenario_id"] for row in rows}) == 18
    assert {row["recovery_type"] for row in rows} == {"fast", "medium", "slow"}
    assert all(case.instance.disruption_section == DISRUPTION_SECTION for case in cases)
    assert all(tuple(station.name for station in case.instance.stations) == CORRIDOR_CODES for case in cases)


def test_future_corridor_entry_does_not_occupy_boundary_before_available_time():
    case = build_public_macro_cases()[0]
    future = next(
        state
        for state in case.context.train_states.values()
        if state.current_station == CORRIDOR_CODES[0]
        and state.available_from > case.context.now
    )
    pool = generate_path_pool(case.instance, case.context)

    for scenario in case.context.scenarios:
        paths = [
            path
            for path in pool.paths(scenario.scenario_id, future.train_id)
            if not path.cancelled
        ]
        assert paths
        for path in paths[:10]:
            occupied = [
                minute
                for station, minute in path.station_occupancy
                if station == future.current_station
            ]
            assert not occupied or min(occupied) >= future.available_from
