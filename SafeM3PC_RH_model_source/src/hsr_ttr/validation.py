                                                                        

                                                                          
                                                                             
                                                                              
                                                                            
                                                                             
                       

                                                                      

                                                             
                                                                           
                                                                              
                   
                                                                               
                                               
                                                                            
                                                                           

                                                                            
                                                                              
                                                                     
   

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from .domain import (
    CommandEvent,
    ObjectiveVector,
    PathColumn,
    RailwayInstance,
    RecoveryScenario,
    RollingContext,
    ScheduleSolution,
)
from .paths import COMMAND_WEIGHTS, PathPool


SelectionKey = Tuple[str, str]
SelectionValue = Union[str, PathColumn]
PathSelection = Mapping[SelectionKey, SelectionValue]


@dataclass(frozen=True)
class ValidationConfig:
                                                                           

                                                                           
                                                                         
                                                                           
                                                                            
                                                                             
                                             
       

    tolerance: float = 1e-6
    hdfs_delay_threshold: int = 2
    peak_window_minutes: Optional[int] = None
    check_path_command_semantics: bool = True


@dataclass(frozen=True)
class ValidationIssue:
                                                  

    code: str
    message: str
    scenario_id: Optional[str] = None
    train_id: Optional[str] = None
    resource: Optional[str] = None


@dataclass
class ValidationReport:
                                                 

    issues: List[ValidationIssue] = field(default_factory=list)
    objective: ObjectiveVector = field(
        default_factory=lambda: ObjectiveVector(0.0, 0.0, 0.0, 0.0)
    )
    resolved_paths: Dict[SelectionKey, PathColumn] = field(default_factory=dict)
    details: Dict[str, object] = field(default_factory=dict)

    @property
    def valid(self) -> bool:
        return not self.issues

    @property
    def issue_codes(self) -> Set[str]:
        return {issue.code for issue in self.issues}

    def messages(self) -> List[str]:
        return [issue.message for issue in self.issues]

    def raise_if_invalid(self) -> None:
                                                                              

        if self.valid:
            return
        raise ValueError("Schedule validation failed:\n- " + "\n- ".join(self.messages()))


def _issue(
    report: ValidationReport,
    code: str,
    message: str,
    scenario_id: Optional[str] = None,
    train_id: Optional[str] = None,
    resource: Optional[str] = None,
) -> None:
    report.issues.append(
        ValidationIssue(
            code=code,
            message=message,
            scenario_id=scenario_id,
            train_id=train_id,
            resource=resource,
        )
    )


def _resolve_validation_scenarios(
    context: RollingContext,
    scenarios: Optional[Sequence[Union[RecoveryScenario, str]]],
) -> Tuple[RecoveryScenario, ...]:
                                                                         

    lookup = {scenario.scenario_id: scenario for scenario in context.scenarios}
    if scenarios is None:
        resolved = tuple(context.scenarios)
    else:
        items: List[RecoveryScenario] = []
        for item in scenarios:
            if isinstance(item, RecoveryScenario):
                candidate = item
            else:
                scenario_id = str(item)
                if scenario_id not in lookup:
                    raise ValueError("Unknown recovery scenario: %s" % scenario_id)
                candidate = lookup[scenario_id]
            items.append(candidate)
        resolved = tuple(items)
    if not resolved:
        raise ValueError("At least one recovery scenario is required for validation")
    if len({scenario.scenario_id for scenario in resolved}) != len(resolved):
        raise ValueError("Recovery scenario ids must be unique")
    return resolved


def _scenario_lookup(
    scenarios: Sequence[RecoveryScenario],
) -> Dict[str, RecoveryScenario]:
    return {scenario.scenario_id: scenario for scenario in scenarios}


def _resolve_selection(
    instance: RailwayInstance,
    scenarios: Sequence[RecoveryScenario],
    selection: PathSelection,
    pool: Optional[PathPool],
    report: ValidationReport,
) -> Dict[SelectionKey, PathColumn]:
                                                                              

    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    train_ids = {train.train_id for train in instance.trains}
    expected = {(scenario_id, train_id) for scenario_id in scenario_ids for train_id in train_ids}
    supplied = set(selection)

    for scenario_id, train_id in sorted(expected - supplied):
        _issue(
            report,
            "missing_path",
            "No path selected for scenario %s and train %s." % (scenario_id, train_id),
            scenario_id,
            train_id,
        )
    for scenario_id, train_id in sorted(supplied - expected):
        _issue(
            report,
            "unexpected_selection_key",
            "Unexpected selected-path key (%s, %s)." % (scenario_id, train_id),
            scenario_id,
            train_id,
        )

    resolved: Dict[SelectionKey, PathColumn] = {}
    for key in sorted(expected & supplied):
        scenario_id, train_id = key
        value = selection[key]
        path: Optional[PathColumn]
        if isinstance(value, PathColumn):
            path = value
        elif isinstance(value, str):
            if pool is None:
                _issue(
                    report,
                    "path_pool_required",
                    "Path id %s cannot be resolved without a PathPool." % value,
                    scenario_id,
                    train_id,
                )
                continue
            path = pool.by_id.get(value)
            if path is None:
                _issue(
                    report,
                    "unknown_path_id",
                    "Selected path id %s is not present in the PathPool." % value,
                    scenario_id,
                    train_id,
                )
                continue
                                                                          
                                                                    
            indexed_ids = {
                candidate.path_id for candidate in pool.paths(scenario_id, train_id)
            }
            if value not in indexed_ids:
                _issue(
                    report,
                    "path_pool_index_mismatch",
                    "Path %s is not indexed under (%s, %s)."
                    % (value, scenario_id, train_id),
                    scenario_id,
                    train_id,
                )
        else:
            _issue(
                report,
                "invalid_selection_value",
                "Selection for (%s, %s) must be a path id or PathColumn."
                % (scenario_id, train_id),
                scenario_id,
                train_id,
            )
            continue

        if path.scenario_id != scenario_id or path.train_id != train_id:
            _issue(
                report,
                "path_owner_mismatch",
                "Path %s belongs to (%s, %s), not (%s, %s)."
                % (
                    path.path_id,
                    path.scenario_id,
                    path.train_id,
                    scenario_id,
                    train_id,
                ),
                scenario_id,
                train_id,
            )
            continue
        resolved[key] = path
    return resolved


def _expected_station_occupancy(
    instance: RailwayInstance,
    context: RollingContext,
    path: PathColumn,
) -> Set[Tuple[str, int]]:
                                                                                   

    occupancy: Set[Tuple[str, int]] = set()
    if path.cancelled:
        return occupancy
    terminal = instance.stations[-1].name
    for station, arrival in path.arrivals.items():
        if station == terminal:
            continue
        departure = path.departures.get(station, arrival)
        for minute in range(max(context.now, int(arrival)), int(departure), instance.time_step):
            occupancy.add((station, minute))
    return occupancy


def _expected_path_commands(
    instance: RailwayInstance,
    context: RollingContext,
    path: PathColumn,
) -> Tuple[CommandEvent, ...]:
                                                                           

                                                                             
                                                                             
       

    state = context.train_states[path.train_id]
    if state.completed:
        return ()
    if path.cancelled:
        return (
            CommandEvent("cancel", context.now, COMMAND_WEIGHTS["cancel"]),
        )

    commands: List[CommandEvent] = []
    hold_station = path.hold_station
    release_time = path.release_time
    if hold_station is None or release_time is None:
        return tuple(commands)
    command_lead = 2 * instance.time_step
    hold_departure = int(path.departures.get(hold_station, release_time))

    if state.issued_hold_station is None:
        hold_arrival = int(path.arrivals.get(hold_station, context.now))
        if hold_station == state.current_station:
            earliest_departure = context.freeze_until
        else:
            earliest_departure = hold_arrival + instance.minimum_dwell
        if hold_departure > earliest_departure + instance.command_change_threshold:
            commands.append(
                CommandEvent(
                    "hold_change",
                    max(context.now, hold_departure - command_lead),
                    COMMAND_WEIGHTS["hold_change"],
                )
            )
    elif state.issued_hold_station != hold_station:
                                                                             
                                                                               
        commands.append(
            CommandEvent(
                "station_change",
                max(
                    context.now,
                    int(path.departures.get(state.current_station, context.freeze_until))
                    - command_lead,
                ),
                COMMAND_WEIGHTS["station_change"],
            )
        )
    elif state.issued_release_time is not None:
        if abs(int(release_time) - int(state.issued_release_time)) > instance.command_change_threshold:
            commands.append(
                CommandEvent(
                    "hold_change",
                    max(context.now, hold_departure - command_lead),
                    COMMAND_WEIGHTS["hold_change"],
                )
            )
    return tuple(commands)


def _command_signature(events: Iterable[CommandEvent]) -> List[Tuple[str, int, int]]:
    return sorted((event.command_type, int(event.issue_time), int(event.weight)) for event in events)


def _validate_path_local_rules(
    instance: RailwayInstance,
    context: RollingContext,
    scenario: RecoveryScenario,
    path: PathColumn,
    config: ValidationConfig,
    report: ValidationReport,
) -> Set[Tuple[str, int]]:
                                                                               

    train = instance.train(path.train_id)
    state = context.train_states[path.train_id]
    tolerance = config.tolerance

    if path.cancelled:
        if state.completed or not state.cancel_allowed:
            _issue(
                report,
                "cancellation_not_allowed",
                "Train %s is cancelled although cancellation is not allowed."
                % path.train_id,
                scenario.scenario_id,
                path.train_id,
            )
        if path.section_events or path.arrivals or path.departures or path.terminal_arrival is not None:
            _issue(
                report,
                "cancelled_path_has_movements",
                "Cancelled path %s still contains physical movements." % path.path_id,
                scenario.scenario_id,
                path.train_id,
            )
    else:
        if path.terminal_arrival is None:
            _issue(
                report,
                "missing_terminal_arrival",
                "Operating path %s has no terminal arrival." % path.path_id,
                scenario.scenario_id,
                path.train_id,
            )
        terminal = instance.stations[-1].name
        if path.terminal_arrival is not None and (
            terminal not in path.arrivals
            or int(path.arrivals[terminal]) != int(path.terminal_arrival)
        ):
            _issue(
                report,
                "terminal_arrival_mismatch",
                "Path %s terminal_arrival does not match its terminal arrival."
                % path.path_id,
                scenario.scenario_id,
                path.train_id,
            )
        if path.terminal_arrival is not None:
            expected_delay = max(0, int(path.terminal_arrival) - train.planned_terminal_arrival)
            if abs(float(path.terminal_delay) - float(expected_delay)) > tolerance:
                _issue(
                    report,
                    "terminal_delay_mismatch",
                    "Path %s records terminal delay %s, independently recomputed as %s."
                    % (path.path_id, path.terminal_delay, expected_delay),
                    scenario.scenario_id,
                    path.train_id,
                )

    known_sections = {section.name: section for section in instance.sections}
    for section_name, event in path.section_events.items():
        if section_name not in known_sections:
            _issue(
                report,
                "unknown_section",
                "Path %s uses unknown section %s." % (path.path_id, section_name),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )
            continue
        if len(event) != 3:
            _issue(
                report,
                "invalid_section_event",
                "Section event %s on path %s must contain departure, arrival and headway."
                % (section_name, path.path_id),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )
            continue
        departure, arrival, stored_headway = (int(event[0]), int(event[1]), int(event[2]))
        section = known_sections[section_name]
        if arrival <= departure:
            _issue(
                report,
                "nonpositive_runtime",
                "Train %s has non-positive running time on %s."
                % (path.train_id, section_name),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )
        if departure % instance.time_step or arrival % instance.time_step:
            _issue(
                report,
                "off_grid_event",
                "Train %s event on %s is not on the %d-minute grid."
                % (path.train_id, section_name, instance.time_step),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )

        if section_name == instance.disruption_section:
            state_name = scenario.state_at(departure)
            expected_runtime = scenario.disrupted_runtime(departure, section.normal_runtime)
            expected_headway = scenario.disrupted_headway(departure, section.normal_headway)
            if state_name == "S1" or expected_runtime is None:
                _issue(
                    report,
                    "s1_disruption_entry",
                    "Train %s enters %s at %d while scenario %s is in S1."
                    % (path.train_id, section_name, departure, scenario.scenario_id),
                    scenario.scenario_id,
                    path.train_id,
                    section_name,
                )
            elif arrival - departure != int(expected_runtime):
                _issue(
                    report,
                    "state_runtime",
                    "Train %s uses runtime %d on %s in %s; required runtime is %d."
                    % (
                        path.train_id,
                        arrival - departure,
                        section_name,
                        state_name,
                        expected_runtime,
                    ),
                    scenario.scenario_id,
                    path.train_id,
                    section_name,
                )
        else:
            expected_runtime = section.normal_runtime
            expected_headway = section.normal_headway
            if arrival - departure != int(expected_runtime):
                _issue(
                    report,
                    "section_runtime",
                    "Train %s uses runtime %d on %s; required runtime is %d."
                    % (path.train_id, arrival - departure, section_name, expected_runtime),
                    scenario.scenario_id,
                    path.train_id,
                    section_name,
                )

        if stored_headway != int(expected_headway):
            _issue(
                report,
                "stored_headway_mismatch",
                "Train %s stores headway %d on %s; state requires %d."
                % (path.train_id, stored_headway, section_name, expected_headway),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )
        if section.origin in path.departures and int(path.departures[section.origin]) != departure:
            _issue(
                report,
                "section_departure_mismatch",
                "Section %s departure on path %s disagrees with the station departure."
                % (section_name, path.path_id),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )
        if section.destination in path.arrivals and int(path.arrivals[section.destination]) != arrival:
            _issue(
                report,
                "section_arrival_mismatch",
                "Section %s arrival on path %s disagrees with the station arrival."
                % (section_name, path.path_id),
                scenario.scenario_id,
                path.train_id,
                section_name,
            )

    if (
        not path.cancelled
        and not state.completed
        and not state.already_crossed_disruption
        and instance.disruption_section not in path.section_events
    ):
        _issue(
            report,
            "missing_disruption_crossing",
            "Operating train %s has no event for disruption section %s."
            % (path.train_id, instance.disruption_section),
            scenario.scenario_id,
            path.train_id,
            instance.disruption_section,
        )

    expected_occupancy = _expected_station_occupancy(instance, context, path)
    stored_occupancy = set(path.station_occupancy)
    if len(stored_occupancy) != len(path.station_occupancy):
        _issue(
            report,
            "duplicate_station_occupancy",
            "Path %s contains duplicate station occupancy slots." % path.path_id,
            scenario.scenario_id,
            path.train_id,
        )
    if stored_occupancy != expected_occupancy:
        _issue(
            report,
            "station_occupancy_mismatch",
            "Path %s station occupancy differs from arrivals/departures."
            % path.path_id,
            scenario.scenario_id,
            path.train_id,
        )

    if config.check_path_command_semantics:
        expected_commands = _expected_path_commands(instance, context, path)
        if _command_signature(path.command_events) != _command_signature(expected_commands):
            _issue(
                report,
                "path_command_mismatch",
                "Path %s command events do not match the previously issued plan."
                % path.path_id,
                scenario.scenario_id,
                path.train_id,
            )
    for event in path.command_events:
        expected_weight = COMMAND_WEIGHTS.get(event.command_type)
        if expected_weight is None:
            _issue(
                report,
                "unknown_command_type",
                "Path %s contains unknown command type %s."
                % (path.path_id, event.command_type),
                scenario.scenario_id,
                path.train_id,
            )
        elif int(event.weight) != int(expected_weight):
            _issue(
                report,
                "command_weight_mismatch",
                "Command %s on path %s has weight %d; required weight is %d."
                % (event.command_type, path.path_id, event.weight, expected_weight),
                scenario.scenario_id,
                path.train_id,
            )
    return expected_occupancy


def _validate_cancellations(
    instance: RailwayInstance,
    context: RollingContext,
    scenarios: Sequence[RecoveryScenario],
    resolved: Mapping[SelectionKey, PathColumn],
    report: ValidationReport,
) -> Tuple[Set[str], Dict[str, Set[str]]]:
                                                                                

    cancelled_by_scenario: Dict[str, Set[str]] = {}
    for scenario in scenarios:
        cancelled_by_scenario[scenario.scenario_id] = {
            train.train_id
            for train in instance.trains
            if (scenario.scenario_id, train.train_id) in resolved
            and resolved[(scenario.scenario_id, train.train_id)].cancelled
        }
        count = len(cancelled_by_scenario[scenario.scenario_id])
        if count > instance.max_cancellations:
            _issue(
                report,
                "cancellation_budget",
                "Scenario %s cancels %d trains; budget is %d."
                % (scenario.scenario_id, count, instance.max_cancellations),
                scenario.scenario_id,
            )
        if count >= len(instance.trains):
            _issue(
                report,
                "all_trains_cancelled",
                "Scenario %s cancels every train." % scenario.scenario_id,
                scenario.scenario_id,
            )

    common: Set[str] = set()
    if cancelled_by_scenario:
        scenario_sets = list(cancelled_by_scenario.values())
        common = set(scenario_sets[0])
        for train in instance.trains:
            statuses = {
                train.train_id in cancelled_by_scenario[scenario.scenario_id]
                for scenario in scenarios
            }
            if len(statuses) > 1:
                _issue(
                    report,
                    "cancellation_nonanticipativity",
                    "Train %s cancellation differs across recovery scenarios."
                    % train.train_id,
                    train_id=train.train_id,
                )
        common = set.intersection(*scenario_sets) if scenario_sets else set()
    return common, cancelled_by_scenario


def _validate_near_term_signatures(
    instance: RailwayInstance,
    scenarios: Sequence[RecoveryScenario],
    resolved: Mapping[SelectionKey, PathColumn],
    report: ValidationReport,
) -> None:
                                                                             

    for train in instance.trains:
        signatures: Dict[str, str] = {}
        for scenario in scenarios:
            key = (scenario.scenario_id, train.train_id)
            if key in resolved:
                signatures[scenario.scenario_id] = resolved[key].near_term_signature
        if len(set(signatures.values())) > 1:
            _issue(
                report,
                "near_term_nonanticipativity",
                "Train %s has different near-term signatures across scenarios: %s."
                % (train.train_id, signatures),
                train_id=train.train_id,
            )


def _validate_station_capacity_and_headways(
    instance: RailwayInstance,
    context: RollingContext,
    scenarios: Sequence[RecoveryScenario],
    resolved: Mapping[SelectionKey, PathColumn],
    occupancy_by_path: Mapping[SelectionKey, Set[Tuple[str, int]]],
    report: ValidationReport,
) -> None:
                                                                         

    scenario_lookup = _scenario_lookup(scenarios)
    for scenario_id in scenario_lookup:
        station_slots: Dict[Tuple[str, int], List[str]] = {}
        section_events: Dict[str, List[Tuple[str, int, int, int]]] = {
            section.name: [] for section in instance.sections
        }
        for train in instance.trains:
            key = (scenario_id, train.train_id)
            path = resolved.get(key)
            if path is None or path.cancelled:
                continue
            for slot in occupancy_by_path.get(key, set()):
                station_slots.setdefault(slot, []).append(train.train_id)
            for section_name, event in path.section_events.items():
                if section_name not in section_events or len(event) != 3:
                    continue
                departure, arrival = int(event[0]), int(event[1])
                section = instance.section(section_name)
                if section_name == instance.disruption_section:
                    required = scenario_lookup[scenario_id].disrupted_headway(
                        departure, section.normal_headway
                    )
                else:
                    required = section.normal_headway
                section_events[section_name].append(
                    (train.train_id, departure, arrival, int(required))
                )

        for (station_name, minute), trains in sorted(station_slots.items()):
            try:
                capacity = instance.station(station_name).hold_capacity
            except StopIteration:
                _issue(
                    report,
                    "unknown_station_occupancy",
                    "Unknown station %s occurs in occupancy data." % station_name,
                    scenario_id,
                    resource=station_name,
                )
                continue
            if len(trains) > capacity:
                _issue(
                    report,
                    "station_capacity",
                    "Scenario %s station %s at %d holds %d trains %s; capacity is %d."
                    % (scenario_id, station_name, minute, len(trains), sorted(trains), capacity),
                    scenario_id,
                    resource=station_name,
                )

                                                                               
                                                                         
                                                                               
        for section_name, movements in section_events.items():
            movements.sort(key=lambda item: (item[1], item[0]))
            for previous, following in zip(movements, movements[1:]):
                prev_train, prev_dep, prev_arr, prev_headway = previous
                next_train, next_dep, next_arr, next_headway = following
                required = max(prev_headway, next_headway)
                if next_dep - prev_dep < required:
                    _issue(
                        report,
                        "departure_headway",
                        "Scenario %s section %s trains %s/%s depart %d minutes apart; required %d."
                        % (
                            scenario_id,
                            section_name,
                            prev_train,
                            next_train,
                            next_dep - prev_dep,
                            required,
                        ),
                        scenario_id,
                        resource=section_name,
                    )
                if next_arr < prev_arr:
                    _issue(
                        report,
                        "section_overtaking",
                        "Scenario %s section %s reverses order between trains %s and %s."
                        % (scenario_id, section_name, prev_train, next_train),
                        scenario_id,
                        resource=section_name,
                    )
                if next_arr - prev_arr < required:
                    _issue(
                        report,
                        "arrival_headway",
                        "Scenario %s section %s trains %s/%s arrive %d minutes apart; required %d."
                        % (
                            scenario_id,
                            section_name,
                            prev_train,
                            next_train,
                            next_arr - prev_arr,
                            required,
                        ),
                        scenario_id,
                        resource=section_name,
                    )


def _active_disruption_entries(
    instance: RailwayInstance,
    context: RollingContext,
    scenario_id: str,
    resolved: Mapping[SelectionKey, PathColumn],
) -> Dict[str, int]:
    entries: Dict[str, int] = {}
    for train in instance.trains:
        state = context.train_states[train.train_id]
        path = resolved.get((scenario_id, train.train_id))
        if (
            path is None
            or path.cancelled
            or state.completed
            or state.already_crossed_disruption
        ):
            continue
        event = path.section_events.get(instance.disruption_section)
        if event is not None and len(event) == 3:
            entries[train.train_id] = int(event[0])
    return entries


def _scenario_objectives(
    instance: RailwayInstance,
    context: RollingContext,
    scenario_id: str,
    resolved: Mapping[SelectionKey, PathColumn],
    config: ValidationConfig,
) -> Tuple[float, float, float, float, int]:
                                                                         

    service = 0.0
    bins: Dict[int, float] = {}
    window = int(config.peak_window_minutes or instance.time_step)
    if window <= 0:
        raise ValueError("peak_window_minutes must be positive")

    for train in instance.trains:
        path = resolved.get((scenario_id, train.train_id))
        if path is None:
            continue
        if path.cancelled:
            service += float(instance.cancellation_penalty)
        elif path.terminal_arrival is not None:
            service += float(max(0, int(path.terminal_arrival) - train.planned_terminal_arrival))
        for event in path.command_events:
                                                                         
                                                                             
                                                                
            if event.command_type == "order_change":
                continue
            bucket = (int(event.issue_time) // window) * window
            bins[bucket] = bins.get(bucket, 0.0) + float(event.weight)

    entries = _active_disruption_entries(instance, context, scenario_id, resolved)
    active_ids = sorted(entries)
    hdfs = 0.0
    order_changes = 0
    for left_index, left_id in enumerate(active_ids):
        for right_id in active_ids[left_index + 1 :]:
            left_state = context.train_states[left_id]
            right_state = context.train_states[right_id]
            delay_difference = left_state.current_delay - right_state.current_delay
            if abs(delay_difference) >= config.hdfs_delay_threshold:
                if delay_difference > 0:
                    longer_id, shorter_id = left_id, right_id
                else:
                    longer_id, shorter_id = right_id, left_id
                if entries[longer_id] >= entries[shorter_id]:
                    hdfs += 1.0

            left_issued = left_state.issued_release_time
            right_issued = right_state.issued_release_time
            left_original = instance.train(left_id).original_order
            right_original = instance.train(right_id).original_order
            if (
                left_issued is not None
                and right_issued is not None
                and int(left_issued) != int(right_issued)
            ):
                issued_left_first = int(left_issued) < int(right_issued)
            else:
                                                                           
                                                                
                issued_left_first = left_original < right_original
            new_left_first = (entries[left_id], left_original) < (
                entries[right_id],
                right_original,
            )
            if issued_left_first != new_left_first:
                order_changes += 1

    order_weight = float(order_changes * COMMAND_WEIGHTS["order_change"])
    scenario = next(
        item for item in context.scenarios if item.scenario_id == scenario_id
    )
    order_issue_time = max(
        context.now,
        int(scenario.t12) - 2 * instance.time_step,
    )
    order_bucket = (order_issue_time // window) * window
    bins[order_bucket] = bins.get(order_bucket, 0.0) + order_weight
    workload_total = float(sum(bins.values()))
    workload_peak = float(max(bins.values())) if bins else 0.0
    return service, hdfs, workload_total, workload_peak, order_changes


def recompute_objective(
    instance: RailwayInstance,
    context: RollingContext,
    selected_paths: Mapping[SelectionKey, PathColumn],
    config: Optional[ValidationConfig] = None,
    scenarios: Optional[Sequence[Union[RecoveryScenario, str]]] = None,
) -> Tuple[ObjectiveVector, Dict[str, object]]:
                                                                            

                                                                  
                                                                             
                                                                       
                                      
       

    cfg = config or ValidationConfig()
    selected_scenarios = _resolve_validation_scenarios(context, scenarios)
    probability_total = sum(float(scenario.probability) for scenario in selected_scenarios)
    if probability_total <= 0.0:
        raise ValueError("Selected recovery-scenario probabilities must be positive")
    service_expected = 0.0
    hdfs_expected = 0.0
    workload_expected = 0.0
    peak_expected = 0.0
    by_scenario: Dict[str, Dict[str, float]] = {}
    for scenario in selected_scenarios:
        service, hdfs, workload, peak, order_changes = _scenario_objectives(
            instance,
            context,
            scenario.scenario_id,
            selected_paths,
            cfg,
        )
                                                                             
                                                                            
                                  
        probability = float(scenario.probability) / probability_total
        service_expected += probability * service
        hdfs_expected += probability * hdfs
        workload_expected += probability * workload
        peak_expected += probability * peak
        by_scenario[scenario.scenario_id] = {
            "probability": probability,
            "service_loss": service,
            "hdfs_violations": hdfs,
            "workload_total": workload,
            "workload_peak": peak,
            "order_changes": float(order_changes),
        }
    objective = ObjectiveVector(
        service_loss=service_expected,
        hdfs_violations=hdfs_expected,
        workload_total=workload_expected,
        workload_peak=peak_expected,
    )
    return objective, {"by_scenario": by_scenario}


def _compare_objectives(
    recorded: ObjectiveVector,
    recomputed: ObjectiveVector,
    tolerance: float,
    report: ValidationReport,
) -> None:
    for field_name in (
        "service_loss",
        "hdfs_violations",
        "workload_total",
        "workload_peak",
    ):
        recorded_value = float(getattr(recorded, field_name))
        recomputed_value = float(getattr(recomputed, field_name))
        if abs(recorded_value - recomputed_value) > tolerance:
            _issue(
                report,
                "objective_mismatch",
                "Recorded %s is %.9g; independently recomputed value is %.9g."
                % (field_name, recorded_value, recomputed_value),
                resource=field_name,
            )


def validate_selection(
    instance: RailwayInstance,
    context: RollingContext,
    selection: PathSelection,
    pool: Optional[PathPool] = None,
    recorded_objective: Optional[ObjectiveVector] = None,
    recorded_cancelled_trains: Optional[Sequence[str]] = None,
    cancellation_bound: Optional[int] = None,
    config: Optional[ValidationConfig] = None,
    scenarios: Optional[Sequence[Union[RecoveryScenario, str]]] = None,
) -> ValidationReport:
                                                           

                                                         
                                                                          
                                                                         
                                                        
       

    cfg = config or ValidationConfig()
    report = ValidationReport()
    selected_scenarios = _resolve_validation_scenarios(context, scenarios)
    probability_sum = sum(float(scenario.probability) for scenario in selected_scenarios)
    if probability_sum <= cfg.tolerance:
        _issue(
            report,
            "scenario_probability_sum",
            "Selected recovery scenario probabilities must sum to a positive value.",
        )

    resolved = _resolve_selection(instance, selected_scenarios, selection, pool, report)
    report.resolved_paths = resolved
    scenario_lookup = _scenario_lookup(selected_scenarios)
    occupancy_by_path: Dict[SelectionKey, Set[Tuple[str, int]]] = {}
    for key, path in resolved.items():
        scenario_id, _ = key
        occupancy_by_path[key] = _validate_path_local_rules(
            instance,
            context,
            scenario_lookup[scenario_id],
            path,
            cfg,
            report,
        )

    common_cancelled, cancelled_by_scenario = _validate_cancellations(
        instance, context, selected_scenarios, resolved, report
    )
    _validate_near_term_signatures(instance, selected_scenarios, resolved, report)
    _validate_station_capacity_and_headways(
        instance,
        context,
        selected_scenarios,
        resolved,
        occupancy_by_path,
        report,
    )

    objective, details = recompute_objective(
        instance, context, resolved, cfg, selected_scenarios
    )
    report.objective = objective
    details["cancelled_by_scenario"] = {
        scenario_id: sorted(cancelled)
        for scenario_id, cancelled in cancelled_by_scenario.items()
    }
    details["common_cancelled_trains"] = sorted(common_cancelled)
    report.details = details

    if recorded_objective is not None:
        _compare_objectives(recorded_objective, objective, cfg.tolerance, report)
    if recorded_cancelled_trains is not None:
        recorded_set = set(recorded_cancelled_trains)
        if recorded_set != common_cancelled:
            _issue(
                report,
                "cancelled_trains_mismatch",
                "Recorded cancelled trains %s; independently recomputed common set is %s."
                % (sorted(recorded_set), sorted(common_cancelled)),
            )
    if cancellation_bound is not None:
        if len(common_cancelled) > int(cancellation_bound):
            _issue(
                report,
                "solution_cancellation_bound",
                "Solution cancels %d trains; recorded stage-one bound is %d."
                % (len(common_cancelled), cancellation_bound),
            )
        if int(cancellation_bound) > instance.max_cancellations:
            _issue(
                report,
                "invalid_cancellation_bound",
                "Recorded cancellation bound %d exceeds instance cap %d."
                % (cancellation_bound, instance.max_cancellations),
            )
    return report


def validate_solution(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    solution: ScheduleSolution,
    config: Optional[ValidationConfig] = None,
) -> ValidationReport:
                                                                     

    cfg = config or ValidationConfig()
    metadata_scenarios = solution.metadata.get("scenario_ids")
    if isinstance(metadata_scenarios, (list, tuple)) and metadata_scenarios:
        selected_scenarios: Sequence[Union[RecoveryScenario, str]] = tuple(
            str(item) for item in metadata_scenarios
        )
    else:
                                                                            
                                                                                
                                                                            
        selected_scenarios = tuple(context.scenarios)
    report = validate_selection(
        instance=instance,
        context=context,
        selection=solution.selected_paths,
        pool=pool,
        recorded_objective=solution.objective,
        recorded_cancelled_trains=solution.cancelled_trains,
        cancellation_bound=solution.cancellation_bound,
        config=cfg,
        scenarios=selected_scenarios,
    )

    if solution.service_optimum < -cfg.tolerance:
        _issue(
            report,
            "negative_service_optimum",
            "Recorded service optimum must be non-negative.",
        )
    epsilon_value = float(solution.metadata.get("epsilon", context.epsilon))
    if epsilon_value < -cfg.tolerance:
        _issue(report, "negative_epsilon", "Service-loss tolerance epsilon is negative.")
    permitted = (1.0 + epsilon_value) * float(solution.service_optimum)
    if report.objective.service_loss > permitted + cfg.tolerance:
        _issue(
            report,
            "service_tolerance",
            "Final service loss %.9g exceeds (1+epsilon)J*=%.9g."
            % (report.objective.service_loss, permitted),
        )

                                                                         
                                                                              
                                               
    expected_stage_values = {
        "service_optimum": float(solution.service_optimum),
        "service_final": float(report.objective.service_loss),
        "hdfs_optimum": float(report.objective.hdfs_violations),
        "workload_optimum": float(report.objective.workload_total),
        "peak_optimum": float(report.objective.workload_peak),
    }
    for name, expected in expected_stage_values.items():
        if name not in solution.stage_objectives:
            continue
        recorded = float(solution.stage_objectives[name])
        if abs(recorded - expected) > cfg.tolerance:
            _issue(
                report,
                "stage_objective_mismatch",
                "Stage objective %s is %.9g; expected %.9g."
                % (name, recorded, expected),
                resource=name,
            )
    return report
