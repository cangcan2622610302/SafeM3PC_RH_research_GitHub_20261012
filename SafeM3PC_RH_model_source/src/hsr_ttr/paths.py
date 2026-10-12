                                                          

                                                                           
                                                                              
                                                                               
                                                                            
                                                                             
   

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .domain import (
    CommandEvent,
    PathColumn,
    RailwayInstance,
    RecoveryScenario,
    RollingContext,
    TrainPlan,
    TrainState,
)


COMMAND_WEIGHTS = {
    "hold_change": 1,
    "order_change": 1,
    "station_change": 1,
    "cancel": 1,
}


@dataclass
class PathPool:
                                                                        

    by_scenario_train: Dict[str, Dict[str, List[PathColumn]]]
    by_id: Dict[str, PathColumn]

    def paths(self, scenario_id: str, train_id: str) -> List[PathColumn]:
        return self.by_scenario_train[scenario_id][train_id]

    def all_paths(self) -> Iterable[PathColumn]:
        return self.by_id.values()


def _ceil_grid(value: int, step: int) -> int:
    return ((int(value) + step - 1) // step) * step


def _station_service_time(
    instance: RailwayInstance,
    train: TrainPlan,
    station_name: str,
) -> int:
                                                                    

    if station_name == instance.stations[-1].name:
        return 0
    if station_name in train.planned_stops:
        return int(instance.minimum_dwell)
    return int(instance.passing_time)


def _make_signature(
    cancelled: bool,
    completed: bool,
    state: TrainState,
    departures: Mapping[str, int],
    section_events: Mapping[str, Tuple[int, int, int]],
    next_observation: int,
    command_events: Sequence[CommandEvent] = (),
    hold_station: Optional[str] = None,
    release_time: Optional[int] = None,
) -> str:
                                                                         

                                                                        
                                                                              
                                                                             
                                                                    
       

    if completed:
        return "COMPLETE"
    if cancelled:
        return "CANCEL"
    tokens: List[str] = []
    for station, minute in sorted(departures.items(), key=lambda item: (item[1], item[0])):
        if minute <= next_observation:
            tokens.append("D%s@%d" % (station, minute))
    for section, event in sorted(section_events.items(), key=lambda item: (item[1][0], item[0])):
        if event[0] <= next_observation:
            tokens.append("R%s@%d" % (section, event[0]))
    for event in sorted(command_events, key=lambda item: (item.issue_time, item.command_type)):
        if event.issue_time > next_observation:
            continue
        if event.command_type == "station_change":
            payload = "to%s" % (hold_station or "NONE")
        elif event.command_type == "hold_change":
            payload = "at%s-r%s" % (
                hold_station or state.current_station,
                "NONE" if release_time is None else int(release_time),
            )
        else:
            payload = ""
        tokens.append(
            "CMD%s:%s@%d" % (event.command_type, payload, int(event.issue_time))
        )
    if not tokens:
        tokens.append("WAIT@%s" % state.current_station)
    return ";".join(tokens)


def _completed_path(
    instance: RailwayInstance,
    train: TrainPlan,
    state: TrainState,
    scenario: RecoveryScenario,
) -> PathColumn:
    terminal = instance.stations[-1].name
    disruption_origin = instance.section(instance.disruption_section).origin
    path_id = "%s|%s|complete" % (scenario.scenario_id, train.train_id)
    return PathColumn(
        path_id=path_id,
        train_id=train.train_id,
        scenario_id=scenario.scenario_id,
        cancelled=False,
        hold_station=None,
        release_time=int(train.planned_departure[disruption_origin]),
        arrivals={terminal: int(train.planned_arrival[terminal])},
        departures={terminal: int(train.planned_arrival[terminal])},
        section_events={},
        station_occupancy=(),
        terminal_arrival=int(train.planned_arrival[terminal]),
        terminal_delay=0,
        command_events=(),
        near_term_signature="COMPLETE",
    )


def _cancel_path(
    train: TrainPlan,
    state: TrainState,
    scenario: RecoveryScenario,
    context: RollingContext,
) -> PathColumn:
    path_id = "%s|%s|cancel" % (scenario.scenario_id, train.train_id)
    cancel_event = CommandEvent("cancel", context.now, COMMAND_WEIGHTS["cancel"])
    signature = _make_signature(
        True,
        False,
        state,
        {},
        {},
        context.next_observation,
        command_events=(cancel_event,),
    )
    return PathColumn(
        path_id=path_id,
        train_id=train.train_id,
        scenario_id=scenario.scenario_id,
        cancelled=True,
        hold_station=None,
        release_time=None,
        arrivals={},
        departures={},
        section_events={},
        station_occupancy=(),
        terminal_arrival=None,
        terminal_delay=0,
        command_events=(cancel_event,),
        near_term_signature=signature,
    )


def _build_operating_path(
    instance: RailwayInstance,
    context: RollingContext,
    scenario: RecoveryScenario,
    train: TrainPlan,
    state: TrainState,
    hold_station: str,
    release_time: int,
    serial: int,
) -> Optional[PathColumn]:
                                                       

                                                                          
                                                                            
                                                                           
       

    step = instance.time_step
    station_names = tuple(station.name for station in instance.stations)
    disrupted_section = instance.section(instance.disruption_section)
    disruption_origin = disrupted_section.origin
    disruption_destination = disrupted_section.destination
    terminal = station_names[-1]
    start_index = station_names.index(state.current_station)
    hold_index = station_names.index(hold_station)
    disruption_origin_index = station_names.index(disruption_origin)
    if hold_index < start_index or hold_index > disruption_origin_index:
        return None
    if release_time % step != 0 or release_time < scenario.t12:
        return None

    disrupted_runtime = scenario.disrupted_runtime(release_time, disrupted_section.normal_runtime)
    if disrupted_runtime is None:
        return None

                                                                        
                                                                            
                                                                             
                                                                           
                                                                             
    current_arrival = max(context.now, state.available_from)
    arrivals: Dict[str, int] = {state.current_station: current_arrival}
    departures: Dict[str, int] = {}
    section_events: Dict[str, Tuple[int, int, int]] = {}

                                                                             
                                                       
    cursor_station = state.current_station
    cursor_time = max(context.freeze_until, state.available_from)
    if hold_station == cursor_station:
        hold_arrival = context.now
        earliest_hold_departure = max(context.freeze_until, state.available_from)
    else:
        departures[cursor_station] = cursor_time
        cursor_index = station_names.index(cursor_station)
        while cursor_index < hold_index:
            section_name = station_names[cursor_index] + station_names[cursor_index + 1]
            section = instance.section(section_name)
            arrival_time = cursor_time + section.normal_runtime
            destination = section.destination
            arrivals[destination] = arrival_time
            section_events[section_name] = (cursor_time, arrival_time, section.normal_headway)
            cursor_index += 1
            cursor_station = destination
            if cursor_station == hold_station:
                hold_arrival = arrival_time
                earliest_hold_departure = arrival_time + instance.minimum_dwell
                break
            cursor_time = arrival_time + _station_service_time(
                instance, train, cursor_station
            )
            departures[cursor_station] = cursor_time

                                                                             
                                                                              
    lag_to_release = sum(
        instance.section(station_names[index] + station_names[index + 1]).normal_runtime
        + _station_service_time(instance, train, station_names[index + 1])
        for index in range(hold_index, disruption_origin_index)
    )
    hold_departure = release_time - lag_to_release
    if hold_departure < earliest_hold_departure:
        return None
    departures[hold_station] = hold_departure

                                              
    cursor_index = hold_index
    cursor_time = hold_departure
    while cursor_index < disruption_origin_index:
        section_name = station_names[cursor_index] + station_names[cursor_index + 1]
        section = instance.section(section_name)
        arrival_time = cursor_time + section.normal_runtime
        destination = section.destination
        arrivals[destination] = arrival_time
        section_events[section_name] = (cursor_time, arrival_time, section.normal_headway)
        cursor_index += 1
        cursor_time = arrival_time + _station_service_time(
            instance, train, destination
        )
        departures[destination] = cursor_time
    if departures.get(disruption_origin, release_time) != release_time:
        return None
    departures[disruption_origin] = release_time

                                                                               
                                                                             
                                                               
    arrival_e = release_time + disrupted_runtime
    arrivals[disruption_destination] = arrival_e
    section_events[instance.disruption_section] = (
        release_time,
        arrival_e,
        scenario.disrupted_headway(release_time, disrupted_section.normal_headway),
    )
    cursor_time = arrival_e + _station_service_time(
        instance, train, disruption_destination
    )
    departures[disruption_destination] = cursor_time

                                                                        
    cursor_index = station_names.index(disruption_destination)
    while cursor_index < len(station_names) - 1:
        section_name = station_names[cursor_index] + station_names[cursor_index + 1]
        section = instance.section(section_name)
        arrival_time = cursor_time + section.normal_runtime
        destination = section.destination
        arrivals[destination] = arrival_time
        section_events[section_name] = (cursor_time, arrival_time, section.normal_headway)
        cursor_index += 1
        if destination != terminal:
            cursor_time = arrival_time + _station_service_time(
                instance, train, destination
            )
            departures[destination] = cursor_time
        else:
            departures[destination] = arrival_time

    terminal_arrival = arrivals[terminal]
    if terminal_arrival > instance.horizon_end:
        return None

    occupancy: List[Tuple[str, int]] = []
    for station, arrival_time in arrivals.items():
        if station == terminal:
            continue
        departure_time = departures.get(station, arrival_time)
        for minute in range(max(context.now, arrival_time), departure_time, step):
            occupancy.append((station, minute))

    command_events: List[CommandEvent] = []
    command_lead = 2 * instance.time_step
    if state.issued_hold_station is None:
        if hold_departure > earliest_hold_departure + instance.command_change_threshold:
            command_events.append(
                CommandEvent(
                    "hold_change",
                    max(context.now, hold_departure - command_lead),
                    COMMAND_WEIGHTS["hold_change"],
                )
            )
    elif state.issued_hold_station != hold_station:
                                                                            
                                                   
        first_departure = departures.get(state.current_station, context.freeze_until)
        command_events.append(
            CommandEvent(
                "station_change",
                max(context.now, int(first_departure) - command_lead),
                COMMAND_WEIGHTS["station_change"],
            )
        )
    elif state.issued_release_time is not None:
        if abs(release_time - state.issued_release_time) > instance.command_change_threshold:
            command_events.append(
                CommandEvent(
                    "hold_change",
                    max(context.now, hold_departure - command_lead),
                    COMMAND_WEIGHTS["hold_change"],
                )
            )

    signature = _make_signature(
        False,
        False,
        state,
        departures,
        section_events,
        context.next_observation,
        command_events=command_events,
        hold_station=hold_station,
        release_time=release_time,
    )
    terminal_delay = max(0, terminal_arrival - train.planned_terminal_arrival)
    path_id = "%s|%s|h%s|r%d|%04d" % (
        scenario.scenario_id,
        train.train_id,
        hold_station,
        release_time,
        serial,
    )
    return PathColumn(
        path_id=path_id,
        train_id=train.train_id,
        scenario_id=scenario.scenario_id,
        cancelled=False,
        hold_station=hold_station,
        release_time=release_time,
        arrivals=arrivals,
        departures=departures,
        section_events=section_events,
        station_occupancy=tuple(sorted(set(occupancy))),
        terminal_arrival=terminal_arrival,
        terminal_delay=terminal_delay,
        command_events=tuple(command_events),
        near_term_signature=signature,
    )


def generate_path_pool(
    instance: RailwayInstance,
    context: RollingContext,
    scenarios: Optional[Sequence[RecoveryScenario]] = None,
) -> PathPool:
                                                                   

    selected_scenarios = tuple(scenarios or context.scenarios)
    by_scenario_train: Dict[str, Dict[str, List[PathColumn]]] = {}
    by_id: Dict[str, PathColumn] = {}
    for scenario in selected_scenarios:
        train_map: Dict[str, List[PathColumn]] = {}
        for train in instance.trains:
            state = context.train_states[train.train_id]
            paths: List[PathColumn] = []
            if state.completed:
                paths = [_completed_path(instance, train, state, scenario)]
            else:
                if state.cancel_allowed:
                    paths.append(_cancel_path(train, state, scenario, context))
                station_names = tuple(station.name for station in instance.stations)
                disruption_origin = instance.section(instance.disruption_section).origin
                start_index = station_names.index(state.current_station)
                end_index = station_names.index(disruption_origin)
                                                                         
                                                                             
                                                                      
                                                                           
                                                                  
                if len(instance.trains) <= 15 and len(station_names) <= 8:
                    max_release = min(
                        instance.horizon_end - 55,
                        scenario.t23 + 60,
                    )
                else:
                    disruption_origin_index = station_names.index(
                        disruption_origin
                    )
                    tail_minutes = instance.section(
                        instance.disruption_section
                    ).normal_runtime
                    for section_index in range(
                        disruption_origin_index + 1,
                        len(station_names) - 1,
                    ):
                        tail_minutes += _station_service_time(
                            instance,
                            train,
                            station_names[section_index],
                        )
                        tail_minutes += instance.section(
                            station_names[section_index]
                            + station_names[section_index + 1]
                        ).normal_runtime
                    active_count = sum(
                        not item.completed
                        for item in context.train_states.values()
                    )
                    release_extension = max(
                        60,
                        active_count
                        * instance.section(
                            instance.disruption_section
                        ).normal_headway
                        + 30,
                    )
                    latest_planned_release = max(
                        int(item.planned_departure[disruption_origin])
                        for item in instance.trains
                    )
                    max_release = min(
                        instance.horizon_end - tail_minutes,
                        max(
                            scenario.t23 + release_extension,
                            latest_planned_release + 30,
                        ),
                    )
                first_release = _ceil_grid(
                    max(scenario.t12, context.freeze_until),
                    instance.time_step,
                )
                serial = 0
                for hold_index in range(start_index, end_index + 1):
                    hold_station = station_names[hold_index]
                    if not instance.station(hold_station).safe_holding:
                        continue
                    for release_time in range(first_release, max_release + 1, instance.time_step):
                        path = _build_operating_path(
                            instance,
                            context,
                            scenario,
                            train,
                            state,
                            hold_station,
                            release_time,
                            serial,
                        )
                        serial += 1
                        if path is not None:
                            paths.append(path)
            if not paths:
                raise ValueError(
                    "No candidate path for train %s in scenario %s"
                    % (train.train_id, scenario.scenario_id)
                )
            train_map[train.train_id] = paths
            for path in paths:
                if path.path_id in by_id:
                    raise ValueError("Duplicate path id: %s" % path.path_id)
                by_id[path.path_id] = path
        by_scenario_train[scenario.scenario_id] = train_map
    return PathPool(by_scenario_train=by_scenario_train, by_id=by_id)


def path_pool_summary(pool: PathPool) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for scenario_id, train_map in pool.by_scenario_train.items():
        summary[scenario_id] = sum(len(paths) for paths in train_map.values())
    return summary
