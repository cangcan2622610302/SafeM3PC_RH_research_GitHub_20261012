                                                        

                                                                        
                                                                      
                                                                             
                                                          
   

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from .domain import (
    RailwayInstance,
    RecoveryScenario,
    RollingContext,
    Section,
    Station,
    TrainPlan,
    TrainState,
)


STATION_NAMES = tuple("ABCDEFGH")


def _build_train_plan(
    train_number: int,
    departure_a: int,
    station_names: Tuple[str, ...] = STATION_NAMES,
) -> TrainPlan:
                                                                      

                                                                     
                                                                             
                                                                          
                                                                          
       

    arrivals: Dict[str, int] = {"A": departure_a}
    departures: Dict[str, int] = {"A": departure_a}
    current = departure_a
    for index in range(len(station_names) - 1):
        current += 10
        destination = station_names[index + 1]
        arrivals[destination] = current
        if destination != station_names[-1]:
            current += 5
        departures[destination] = current
    train_id = "%03d" % train_number
    return TrainPlan(
        train_id=train_id,
        planned_arrival=arrivals,
        planned_departure=departures,
        planned_stops=tuple(station_names[1:-1]),
        original_order=train_number,
    )


def build_small_instance() -> RailwayInstance:
                                                                           

    capacities = {
        "A": 8,
        "B": 2,
        "C": 2,
        "D": 3,
        "E": 3,
        "F": 3,
        "G": 3,
        "H": 8,
    }
    stations = tuple(
        Station(name=name, index=index, hold_capacity=capacities[name])
        for index, name in enumerate(STATION_NAMES)
    )
    sections = tuple(
        Section(
            name=STATION_NAMES[index] + STATION_NAMES[index + 1],
            origin=STATION_NAMES[index],
            destination=STATION_NAMES[index + 1],
            normal_runtime=10,
            normal_headway=5,
        )
        for index in range(len(STATION_NAMES) - 1)
    )
    trains = tuple(_build_train_plan(number, (number - 1) * 10) for number in range(1, 16))
    return RailwayInstance(
        stations=stations,
        sections=sections,
        trains=trains,
        disruption_section="DE",
        disruption_start=60,
        time_step=5,
        horizon_end=360,
        minimum_dwell=5,
        command_change_threshold=5,
        cancellation_penalty=480,
        max_cancellations=3,
    )


def build_scalable_instance(
    station_count: int,
    train_count: int,
) -> RailwayInstance:
                                                                           

                                                                          
                                                                             
                                                                  
       

    if station_count < 5 or station_count > 20:
        raise ValueError("station_count must lie between 5 and 20.")
    if train_count < 4 or train_count > 120:
        raise ValueError("train_count must lie between 4 and 120.")
    station_names = tuple(chr(ord("A") + index) for index in range(station_count))
    disruption_origin_index = station_count // 2 - 1
    disruption_origin = station_names[disruption_origin_index]
    disruption_destination = station_names[disruption_origin_index + 1]
    capacities = {
        name: (
            train_count
            if index in (0, station_count - 1)
            else (3 if index == disruption_origin_index else 2)
        )
        for index, name in enumerate(station_names)
    }
    stations = tuple(
        Station(name=name, index=index, hold_capacity=capacities[name])
        for index, name in enumerate(station_names)
    )
    sections = tuple(
        Section(
            name=station_names[index] + station_names[index + 1],
            origin=station_names[index],
            destination=station_names[index + 1],
            normal_runtime=10,
            normal_headway=5,
        )
        for index in range(station_count - 1)
    )
                                                                             
                                                                             
                                                                     
    trains = tuple(
        _build_train_plan(number, (number - 1) * 5, station_names)
        for number in range(1, train_count + 1)
    )
                                                                             
                                                                              
                                                                           
    horizon_end = max(
        600,
        300 + (train_count - 1) * 5 + 15 * (station_count - 1),
    )
    return RailwayInstance(
        stations=stations,
        sections=sections,
        trains=trains,
        disruption_section=disruption_origin + disruption_destination,
        disruption_start=60,
        time_step=5,
        horizon_end=horizon_end,
        minimum_dwell=5,
        command_change_threshold=5,
        cancellation_penalty=480,
        max_cancellations=max(1, min(train_count - 1, train_count // 5)),
    )


def build_recovery_scenarios(disruption_start: int = 60) -> Tuple[RecoveryScenario, ...]:
                                                                      

                                                                     
                                                                       
                                                                           
                                                             
       

    return (
        RecoveryScenario(
            scenario_id="fast",
            probability=0.25,
            t12=disruption_start + 90,
            t23=disruption_start + 90 + 30,
            s2_speed_kmh=200,
            s2_runtime=15,
            s2_headway=10,
        ),
        RecoveryScenario(
            scenario_id="base",
            probability=0.50,
            t12=disruption_start + 120,
            t23=disruption_start + 120 + 45,
            s2_speed_kmh=160,
            s2_runtime=20,
            s2_headway=10,
        ),
        RecoveryScenario(
            scenario_id="slow",
            probability=0.25,
            t12=disruption_start + 150,
            t23=disruption_start + 150 + 60,
            s2_speed_kmh=120,
            s2_runtime=25,
            s2_headway=10,
        ),
    )


def build_standalone_recovery_scenarios(
    disruption_start: int = 60,
    extreme_s1_duration: int = 210,
) -> Tuple[RecoveryScenario, ...]:
                                                                        

                                                                             
                                                                         
                                                                              
                                                                              
                                                                    
                                       
       

    if extreme_s1_duration not in (180, 210):
        raise ValueError("extreme_s1_duration must be either 180 or 210 minutes.")
    specifications = (
        ("A_fast", 90, 30, 200, 15),
        ("B_base", 120, 45, 160, 20),
        ("C_slow", 150, 60, 120, 25),
        ("D_extreme", extreme_s1_duration, 60, 120, 25),
    )
    return tuple(
        RecoveryScenario(
            scenario_id=scenario_id,
            probability=1.0,
            t12=disruption_start + s1_duration,
            t23=disruption_start + s1_duration + s2_duration,
            s2_speed_kmh=speed,
            s2_runtime=runtime,
            s2_headway=10,
        )
        for scenario_id, s1_duration, s2_duration, speed, runtime in specifications
    )


def build_train_states(instance: RailwayInstance, now: int = 120) -> Dict[str, TrainState]:
                                                                    

                                                                            
                                                                            
                                                                            
                                                                          
                                                                      
       

    location = {
        "003": "D",
        "004": "D",
        "005": "D",
        "006": "C",
        "007": "C",
        "008": "B",
        "009": "B",
        "010": "A",
        "011": "A",
        "012": "A",
        "013": "A",
        "014": "A",
        "015": "A",
    }
    states: Dict[str, TrainState] = {}
    queue_order = ["003", "004", "005", "006", "007", "008", "009", "010", "011", "012", "013", "014", "015"]
    issued_release = {train_id: 180 + 10 * index for index, train_id in enumerate(queue_order)}
    for train in instance.trains:
        if train.train_id in ("001", "002"):
            states[train.train_id] = TrainState(
                train_id=train.train_id,
                current_station="H",
                available_from=now,
                current_delay=0,
                issued_hold_station=None,
                issued_release_time=None,
                cancel_allowed=False,
                already_crossed_disruption=True,
                completed=True,
            )
            continue
        station = location[train.train_id]
        planned_reference = int(train.planned_departure[station])
        states[train.train_id] = TrainState(
            train_id=train.train_id,
            current_station=station,
            available_from=now,
            current_delay=max(0, now - planned_reference),
            issued_hold_station=station,
            issued_release_time=issued_release[train.train_id],
            cancel_allowed=train.train_id in {"010", "011", "012", "013", "014", "015"},
            already_crossed_disruption=False,
            completed=False,
        )
    return states


def build_scalable_train_states(
    instance: RailwayInstance,
    now: int = 120,
) -> Dict[str, TrainState]:
                                                                          

    station_names = tuple(station.name for station in instance.stations)
    terminal = station_names[-1]
    disruption_origin = instance.section(instance.disruption_section).origin
    origin_index = station_names.index(disruption_origin)
    completed_count = max(1, min(2, len(instance.trains) // 8))
    active = list(instance.trains[completed_count:])
    holding_slots: List[str] = []
    for index in range(origin_index, 0, -1):
        station = instance.stations[index]
        holding_slots.extend([station.name] * int(station.hold_capacity))
    if len(holding_slots) < len(active):
        holding_slots.extend([station_names[0]] * (len(active) - len(holding_slots)))
    cancellable_count = max(1, min(len(active), int(round(0.4 * len(active)))))
    cancellable_ids = {train.train_id for train in active[-cancellable_count:]}

    states: Dict[str, TrainState] = {}
    for train in instance.trains[:completed_count]:
        states[train.train_id] = TrainState(
            train_id=train.train_id,
            current_station=terminal,
            available_from=now,
            current_delay=0,
            issued_hold_station=None,
            issued_release_time=None,
            cancel_allowed=False,
            already_crossed_disruption=True,
            completed=True,
        )
    for queue_index, (train, station) in enumerate(zip(active, holding_slots)):
        available_from = max(now, int(train.planned_departure[station]))
        command_already_issued = available_from <= now
        states[train.train_id] = TrainState(
            train_id=train.train_id,
            current_station=station,
            available_from=available_from,
            current_delay=max(0, now - int(train.planned_departure[station])),
            issued_hold_station=station if command_already_issued else None,
            issued_release_time=(
                180 + 10 * queue_index if command_already_issued else None
            ),
            cancel_allowed=train.train_id in cancellable_ids,
            already_crossed_disruption=False,
            completed=False,
        )
    return states


def build_scalable_context(
    instance: RailwayInstance,
    epsilon: float = 0.01,
    now: int = 120,
) -> RollingContext:
                                                                      

    scenario = build_standalone_recovery_scenarios(instance.disruption_start)[1]
    return RollingContext(
        now=now,
        freeze_until=now + 10,
        next_observation=now + 10,
        train_states=build_scalable_train_states(instance, now),
        scenarios=(scenario,),
        epsilon=epsilon,
        random_seed=20260827,
    )


def build_scalable_uncertain_context(
    instance: RailwayInstance,
    epsilon: float = 0.01,
    now: int = 120,
    random_seed: int = 20260827,
) -> RollingContext:
                                                                      

                                                    
                                                                           
                                                                    
                                                                     
                        
       

    return RollingContext(
        now=now,
        freeze_until=now + 10,
        next_observation=now + 10,
        train_states=build_scalable_train_states(instance, now),
        scenarios=build_recovery_scenarios(instance.disruption_start),
        epsilon=epsilon,
        random_seed=int(random_seed),
    )


def build_small_context(
    instance: RailwayInstance,
    now: int = 120,
    epsilon: float = 0.01,
    scenarios: Optional[Tuple[RecoveryScenario, ...]] = None,
) -> RollingContext:
    selected_scenarios = (
        build_recovery_scenarios(instance.disruption_start)
        if scenarios is None
        else tuple(scenarios)
    )
    if not selected_scenarios:
        raise ValueError("At least one recovery scenario is required.")
    return RollingContext(
        now=now,
        freeze_until=now + 10,
        next_observation=now + 10,
        train_states=build_train_states(instance, now=now),
        scenarios=selected_scenarios,
        epsilon=epsilon,
        random_seed=20260826,
    )


def validate_small_instance(instance: RailwayInstance) -> List[str]:
                                                                             

    errors: List[str] = []
    if tuple(station.name for station in instance.stations) != STATION_NAMES:
        errors.append("Stations must be named A through H in order.")
    expected_trains = tuple("%03d" % index for index in range(1, 16))
    if tuple(train.train_id for train in instance.trains) != expected_trains:
        errors.append("Trains must be named 001 through 015 in order.")
    if abs(sum(s.probability for s in build_recovery_scenarios(instance.disruption_start)) - 1.0) > 1e-9:
        errors.append("Recovery scenario probabilities must sum to one.")
    for train in instance.trains:
        previous = -1
        for station in STATION_NAMES:
            arrival = int(train.planned_arrival[station])
            departure = int(train.planned_departure[station])
            if arrival < previous or departure < arrival:
                errors.append("Non-monotone timetable for train %s at %s." % (train.train_id, station))
            previous = departure
    if instance.max_cancellations >= len(instance.trains):
        errors.append("The cancellation budget must prohibit cancelling every train.")
    return errors
