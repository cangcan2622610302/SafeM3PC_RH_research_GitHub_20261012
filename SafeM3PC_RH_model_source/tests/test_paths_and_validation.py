                                                                    

                                                                             
                                                                              
                                                                            
           
   

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Tuple

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from hsr_ttr.data import build_small_context, build_small_instance              
from hsr_ttr.domain import ObjectiveVector, PathColumn, ScheduleSolution              
from hsr_ttr.paths import PathPool, generate_path_pool              
from hsr_ttr.validation import (              
    ValidationConfig,
    recompute_objective,
    validate_selection,
    validate_solution,
)


Selection = Dict[Tuple[str, str], PathColumn]


@pytest.fixture(scope="module")
def pilot():
    instance = build_small_instance()
    context = build_small_context(instance)
    pool = generate_path_pool(instance, context)
    return instance, context, pool


def _path(
    pool: PathPool,
    context,
    scenario_id: str,
    train_id: str,
    release_time: int,
    hold_station: str = None,
) -> PathColumn:
    station = hold_station or context.train_states[train_id].current_station
    return next(
        candidate
        for candidate in pool.paths(scenario_id, train_id)
        if not candidate.cancelled
        and candidate.hold_station == station
        and candidate.release_time == release_time
    )


def _cancel_path(pool: PathPool, scenario_id: str, train_id: str) -> PathColumn:
    return next(path for path in pool.paths(scenario_id, train_id) if path.cancelled)


def _valid_selection(instance, context, pool: PathPool) -> Selection:
                                                               

                                                                         
                                                                               
                                                                              
                                                                  
       

    operating_ids = ["%03d" % number for number in range(3, 14)]
    releases = {
        "fast": [150, 160, 170, 185, 190, 195, 200, 205, 210, 215, 220],
        "base": [180, 190, 200, 210, 220, 240, 245, 250, 255, 260, 265],
        "slow": [210, 220, 230, 240, 250, 260, 285, 290, 295, 300, 305],
    }
    selected: Selection = {}
    for scenario in context.scenarios:
        for train_id in ("001", "002"):
            selected[(scenario.scenario_id, train_id)] = pool.paths(
                scenario.scenario_id, train_id
            )[0]
        for train_id, release in zip(operating_ids, releases[scenario.scenario_id]):
            selected[(scenario.scenario_id, train_id)] = _path(
                pool,
                context,
                scenario.scenario_id,
                train_id,
                release,
            )
        for train_id in ("014", "015"):
            selected[(scenario.scenario_id, train_id)] = _cancel_path(
                pool, scenario.scenario_id, train_id
            )
    assert len(selected) == len(context.scenarios) * len(instance.trains)
    return selected


def test_valid_pathpool_ids_and_manual_columns_recompute_identically(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)

                                                                          
    manual_report = validate_selection(instance, context, selected)
    assert manual_report.valid, manual_report.messages()
    assert manual_report.objective == ObjectiveVector(2250.0, 3.0, 9.0, 2.5)
                                                                           
                                                                         
    assert manual_report.objective.workload_peak < manual_report.objective.workload_total

                                                                             
                                                 
    by_id = {key: path.path_id for key, path in selected.items()}
    objective, _ = recompute_objective(instance, context, selected)
    solution = ScheduleSolution(
        algorithm="hand-audited",
        status="optimal",
        selected_paths=by_id,
        objective=objective,
        cancelled_trains=("014", "015"),
        service_optimum=objective.service_loss,
        cancellation_bound=2,
        stage_objectives={
            "service_optimum": objective.service_loss,
            "service_final": objective.service_loss,
            "cancellation_optimum": 2.0,
            "command_total_optimum": objective.workload_total,
            "command_peak_optimum": objective.workload_peak,
        },
        runtime_seconds=0.0,
    )
    solution_report = validate_solution(instance, context, pool, solution)
    assert solution_report.valid, solution_report.messages()
    assert solution_report.objective == manual_report.objective


def test_requires_exactly_one_owned_path_per_scenario_and_train(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)
    selected.pop(("fast", "003"))

    report = validate_selection(instance, context, selected)
    assert "missing_path" in report.issue_codes

                                                                             
    selected[("fast", "003")] = selected[("fast", "004")]
    report = validate_selection(instance, context, selected)
    assert "path_owner_mismatch" in report.issue_codes


def test_cancellation_is_scenario_consistent_and_within_budget(pilot):
    instance, context, pool = pilot
    inconsistent = _valid_selection(instance, context, pool)
    inconsistent[("fast", "013")] = _cancel_path(pool, "fast", "013")
    report = validate_selection(instance, context, inconsistent)
    assert "cancellation_nonanticipativity" in report.issue_codes

    over_budget = _valid_selection(instance, context, pool)
    for scenario in context.scenarios:
        for train_id in ("012", "013"):
            over_budget[(scenario.scenario_id, train_id)] = _cancel_path(
                pool, scenario.scenario_id, train_id
            )
    report = validate_selection(instance, context, over_budget)
    assert "cancellation_budget" in report.issue_codes


def test_detects_s1_entry_and_wrong_state_runtime(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)

    slow = selected[("slow", "003")]
    slow_events = dict(slow.section_events)
                                                                              
                                                                              
    slow_events["DE"] = (205, 230, 10)
    selected[("slow", "003")] = replace(slow, section_events=slow_events)
    report = validate_selection(instance, context, selected)
    assert "s1_disruption_entry" in report.issue_codes

    selected = _valid_selection(instance, context, pool)
    base = selected[("base", "003")]
    base_events = dict(base.section_events)
    departure, arrival, headway = base_events["DE"]
    base_events["DE"] = (departure, arrival + 5, headway)
    selected[("base", "003")] = replace(base, section_events=base_events)
    report = validate_selection(instance, context, selected)
    assert "state_runtime" in report.issue_codes


def test_rebuilds_station_capacity_and_both_section_headways(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)

                                                                            
                                                                              
                                                             
    tighter_stations = tuple(
        replace(station, hold_capacity=2) if station.name == "D" else station
        for station in instance.stations
    )
    tighter_instance = replace(instance, stations=tighter_stations)
    report = validate_selection(tighter_instance, context, selected)
    assert "station_capacity" in report.issue_codes

                                                                             
                                                      
    selected = _valid_selection(instance, context, pool)
    selected[("fast", "004")] = _path(pool, context, "fast", "004", 150)
    report = validate_selection(instance, context, selected)
    assert "departure_headway" in report.issue_codes
    assert "arrival_headway" in report.issue_codes


def test_b1_near_term_signature_nonanticipativity(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)

                                                                                 
                                                                                 
                                                               
    selected[("fast", "010")] = _path(
        pool, context, "fast", "010", 205, hold_station="B"
    )
    report = validate_selection(instance, context, selected)
    assert "near_term_nonanticipativity" in report.issue_codes


def test_all_four_recorded_objectives_are_checked_independently(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)
    correct, _ = recompute_objective(instance, context, selected)
    wrong = ObjectiveVector(
        service_loss=correct.service_loss + 1,
        hdfs_violations=correct.hdfs_violations + 1,
        workload_total=correct.workload_total + 1,
        workload_peak=correct.workload_peak + 1,
    )

    report = validate_selection(
        instance,
        context,
        selected,
        recorded_objective=wrong,
        recorded_cancelled_trains=("014", "015"),
        cancellation_bound=2,
        config=ValidationConfig(),
    )
    mismatched_fields = {
        issue.resource for issue in report.issues if issue.code == "objective_mismatch"
    }
    assert mismatched_fields == {
        "service_loss",
        "hdfs_violations",
        "workload_total",
        "workload_peak",
    }


def test_deterministic_b0_subset_uses_probability_one(pilot):
    instance, context, pool = pilot
    full = _valid_selection(instance, context, pool)
    base_selection = {
        key: path for key, path in full.items() if key[0] == "base"
    }
    objective, _ = recompute_objective(
        instance,
        context,
        base_selection,
        scenarios=("base",),
    )
    solution = ScheduleSolution(
        algorithm="B0",
        status="optimal",
        selected_paths={key: path.path_id for key, path in base_selection.items()},
        objective=objective,
        cancelled_trains=("014", "015"),
        service_optimum=objective.service_loss,
        cancellation_bound=2,
        stage_objectives={
            "service_optimum": objective.service_loss,
            "service_final": objective.service_loss,
            "hdfs_optimum": objective.hdfs_violations,
            "workload_optimum": objective.workload_total,
            "peak_optimum": objective.workload_peak,
        },
        runtime_seconds=0.0,
        metadata={"scenario_ids": ["base"]},
    )
    report = validate_solution(instance, context, pool, solution)
    assert report.valid, report.messages()
                                                                              
                                                               
    assert report.details["by_scenario"]["base"]["probability"] == 1.0


def test_solution_metadata_epsilon_overrides_context_default(pilot):
    instance, context, pool = pilot
    selected = _valid_selection(instance, context, pool)
    objective, _ = recompute_objective(instance, context, selected)
                                                                             
                                                                             
    service_optimum = objective.service_loss / 1.02
    solution = ScheduleSolution(
        algorithm="B1-epsilon-3pct",
        status="optimal",
        selected_paths={key: path.path_id for key, path in selected.items()},
        objective=objective,
        cancelled_trains=("014", "015"),
        service_optimum=service_optimum,
        cancellation_bound=2,
        stage_objectives={
            "service_optimum": service_optimum,
            "service_final": objective.service_loss,
            "hdfs_optimum": objective.hdfs_violations,
            "workload_optimum": objective.workload_total,
            "peak_optimum": objective.workload_peak,
        },
        runtime_seconds=0.0,
        metadata={
            "scenario_ids": [scenario.scenario_id for scenario in context.scenarios],
            "epsilon": 0.03,
        },
    )
    report = validate_solution(instance, context, pool, solution)
    assert report.valid, report.messages()
