                                                                   

                                                                          
                                                                         
                                                                             
                                                                             
                             

                                                         
                                                                           
                                   
                                                                            
                                                 
                                                                   

                                                                           
                                                                           
                                                                             
                                                                             
                                                                             
                                                                           

                                                                       
                                                                      
                                                                     
                                 
                                                                            
   

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import (
    Dict,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, milp
from scipy.sparse import coo_matrix

from .domain import (
    ObjectiveVector,
    PathColumn,
    RailwayInstance,
    RecoveryScenario,
    RollingContext,
    ScheduleSolution,
)
from .paths import COMMAND_WEIGHTS, PathPool, generate_path_pool


                                                                             
                                                                        
                                                                     
HDFS_DELAY_THRESHOLD_MIN = 2

                                                                           
                                                                              
                                                                    
DEFAULT_OBJECTIVE_TOLERANCE = 1.0e-7


class MilpBuildError(ValueError):
    pass                                                                   


class MilpSolveError(RuntimeError):
    pass                                                                           


def _gurobi_milp(
    objective: np.ndarray,
    variables: "_VariableStore",
    rows: "_RowStore",
    time_limit: Optional[float],
) -> OptimizeResult:
                                                                            

                                                                             
                                                             
       

    try:
        import gurobipy as gp
    except ImportError as exc:                                                      
        raise MilpSolveError(
            "Gurobi backend requested but gurobipy is not installed."
        ) from exc

    matrix = rows.linear_constraint(variables.size).A
    model = gp.Model("hsr_ttr_lex_stage")
    model.Params.OutputFlag = 0
    model.Params.Presolve = 2
    model.Params.MIPGap = 0.0
                                                                             
    model.Params.Threads = 1
    if time_limit is not None:
        model.Params.TimeLimit = max(0.01, float(time_limit))

    variable_types = np.asarray(
        [gp.GRB.INTEGER if flag else gp.GRB.CONTINUOUS for flag in variables.integrality],
        dtype="<U1",
    )
    x = model.addMVar(
        variables.size,
        lb=np.asarray(variables.lower, dtype=float),
        ub=np.asarray(variables.upper, dtype=float),
        vtype=variable_types,
        name="z",
    )
    lower = np.asarray(rows.lower, dtype=float)
    upper = np.asarray(rows.upper, dtype=float)
    equality = np.isfinite(lower) & np.isfinite(upper) & (
        np.abs(lower - upper) <= 1.0e-10
    )
    lower_only = np.isfinite(lower) & ~equality
    upper_only = np.isfinite(upper) & ~equality
    if np.any(equality):
        model.addMConstr(matrix[equality], x, "=", upper[equality], name="eq")
    if np.any(lower_only):
        model.addMConstr(matrix[lower_only], x, ">", lower[lower_only], name="lb")
    if np.any(upper_only):
        model.addMConstr(matrix[upper_only], x, "<", upper[upper_only], name="ub")
    model.setObjective(objective @ x, gp.GRB.MINIMIZE)
    model.optimize()

    if model.SolCount <= 0:
        raise MilpSolveError(
            "Gurobi stage has no feasible incumbent (status=%s)." % model.Status
        )
    status = 0 if model.Status == gp.GRB.OPTIMAL else 1
    gap = float(model.MIPGap) if model.IsMIP else 0.0
    return OptimizeResult(
        x=np.asarray(x.X, dtype=float),
        fun=float(model.ObjVal),
        status=status,
        message="Gurobi status %s" % model.Status,
        mip_gap=gap,
        mip_dual_bound=float(model.ObjBound),
        mip_node_count=int(model.NodeCount),
    )


@dataclass
class _VariableStore:
                                                             

                                                                               
                                                                              
                                                                 
       

    names: List[str] = field(default_factory=list)
    lower: List[float] = field(default_factory=list)
    upper: List[float] = field(default_factory=list)
    integrality: List[int] = field(default_factory=list)

    def add(
        self,
        name: str,
        lower: float = 0.0,
        upper: float = 1.0,
        integer: bool = True,
    ) -> int:
        index = len(self.names)
        self.names.append(name)
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.integrality.append(1 if integer else 0)
        return index

    def fix(self, index: int, value: float) -> None:
        self.lower[index] = float(value)
        self.upper[index] = float(value)

    @property
    def size(self) -> int:
        return len(self.names)

    def bounds(self) -> Bounds:
        return Bounds(np.asarray(self.lower), np.asarray(self.upper))


@dataclass
class _RowStore:
                                                                            

    names: List[str] = field(default_factory=list)
    coefficients: List[Dict[int, float]] = field(default_factory=list)
    lower: List[float] = field(default_factory=list)
    upper: List[float] = field(default_factory=list)

    def add(
        self,
        name: str,
        coefficients: Mapping[int, float],
        lower: float = -np.inf,
        upper: float = np.inf,
    ) -> None:
        compact = {
            int(index): float(value)
            for index, value in coefficients.items()
            if abs(float(value)) > 1.0e-12
        }
        self.names.append(name)
        self.coefficients.append(compact)
        self.lower.append(float(lower))
        self.upper.append(float(upper))

    @property
    def size(self) -> int:
        return len(self.names)

    def linear_constraint(self, variable_count: int) -> LinearConstraint:
        row_indices: List[int] = []
        column_indices: List[int] = []
        values: List[float] = []
        for row_index, row in enumerate(self.coefficients):
            for column_index, coefficient in row.items():
                row_indices.append(row_index)
                column_indices.append(column_index)
                values.append(coefficient)
        matrix = coo_matrix(
            (values, (row_indices, column_indices)),
            shape=(self.size, variable_count),
            dtype=float,
        ).tocsr()
        return LinearConstraint(
            matrix,
            np.asarray(self.lower, dtype=float),
            np.asarray(self.upper, dtype=float),
        )


@dataclass
class _ModelArtifacts:
                                                                              

    variables: _VariableStore
    rows: _RowStore
    scenarios: Tuple[RecoveryScenario, ...]
    probability: Dict[str, float]
    paths_by_group: Dict[Tuple[str, str], Tuple[PathColumn, ...]]
    x_by_path: Dict[str, int]
    order_by_scenario_pair: Dict[Tuple[str, str, str], int]
    reversal_by_scenario_pair: Dict[Tuple[str, str, str], int]
    hdfs_by_scenario_pair: Dict[Tuple[str, str, str], int]
    peak_by_scenario: Dict[str, int]
    service_objective: np.ndarray
    hdfs_objective: np.ndarray
    workload_objective: np.ndarray
    peak_objective: np.ndarray
    cancellation_expression: Dict[Tuple[str, str], Dict[int, float]]
    conflict_counts: Dict[str, int]


def _add_to(row: MutableMapping[int, float], index: int, value: float) -> None:
                                                                            

    new_value = float(row.get(index, 0.0)) + float(value)
    if abs(new_value) <= 1.0e-12:
        row.pop(index, None)
    else:
        row[index] = new_value


def _add_expression(
    target: MutableMapping[int, float],
    source: Mapping[int, float],
    multiplier: float = 1.0,
) -> None:
    for index, value in source.items():
        _add_to(target, index, multiplier * value)


def _normalise_probabilities(
    scenarios: Sequence[RecoveryScenario],
) -> Dict[str, float]:
    total = sum(float(scenario.probability) for scenario in scenarios)
    if total <= 0.0:
        raise MilpBuildError("Selected recovery-scenario probabilities must sum to a positive value.")
    return {
        scenario.scenario_id: float(scenario.probability) / total
        for scenario in scenarios
    }


def _resolve_scenarios(
    context: RollingContext,
    algorithm: str,
    scenarios: Optional[Sequence[Union[RecoveryScenario, str]]],
) -> Tuple[RecoveryScenario, ...]:
                                                                              

    lookup = {scenario.scenario_id: scenario for scenario in context.scenarios}
    if scenarios is None:
        candidates = tuple(context.scenarios)
    else:
        resolved: List[RecoveryScenario] = []
        for item in scenarios:
            if isinstance(item, RecoveryScenario):
                resolved.append(item)
            else:
                scenario_id = str(item)
                if scenario_id not in lookup:
                    raise MilpBuildError("Unknown recovery scenario: %s" % scenario_id)
                resolved.append(lookup[scenario_id])
        candidates = tuple(resolved)
    if not candidates:
        raise MilpBuildError("At least one recovery scenario is required.")
    if len({item.scenario_id for item in candidates}) != len(candidates):
        raise MilpBuildError("Recovery scenario identifiers must be unique.")

                                                                             
                                                                            
                                                 
    if algorithm.upper() == "B0":
        selected = max(
            candidates,
            key=lambda item: (float(item.probability), item.scenario_id),
        )
        return (selected,)
    return candidates


def _normalise_fixed_paths(
    fixed_selected_paths: Optional[Mapping[object, str]],
    selected_scenarios: Sequence[RecoveryScenario],
) -> Dict[Tuple[str, str], str]:
                                                                    

                                                                          
                                                                           
                                                                          
                                                                           
                                                                         
       

    if not fixed_selected_paths:
        return {}
    scenario_ids = {scenario.scenario_id for scenario in selected_scenarios}
    only_scenario = next(iter(scenario_ids)) if len(scenario_ids) == 1 else None
    normalised: Dict[Tuple[str, str], str] = {}
    for raw_key, path_id in fixed_selected_paths.items():
        if isinstance(raw_key, tuple) and len(raw_key) == 2:
            scenario_id, train_id = str(raw_key[0]), str(raw_key[1])
        elif isinstance(raw_key, str) and "|" in raw_key:
            scenario_id, train_id = raw_key.split("|", 1)
        elif isinstance(raw_key, str) and only_scenario is not None:
            scenario_id, train_id = only_scenario, raw_key
        else:
            raise MilpBuildError(
                "Fixed path keys must be (scenario_id, train_id), "
                "'scenario|train', or a train id in a one-scenario solve."
            )
        if scenario_id not in scenario_ids:
            continue
        key = (scenario_id, train_id)
        if key in normalised and normalised[key] != str(path_id):
            raise MilpBuildError("Two different fixed paths were supplied for %r." % (key,))
        normalised[key] = str(path_id)
    return normalised


def _commitment_signature(path: PathColumn, next_observation: int) -> str:
                                                                             

                                                                              
                                                                              
                                                                               
                                                                              
                                                                           
                                                                              
                                                                             
                                                                           
       

    del next_observation                                                  
    return path.near_term_signature


def _time_bucket(minute: int, step: int) -> int:
                                                                                

    return (int(minute) // int(step)) * int(step)


def _issued_precedence(
    instance: RailwayInstance,
    context: RollingContext,
    first_train: str,
    second_train: str,
) -> int:
                                                                               

    first_state = context.train_states[first_train]
    second_state = context.train_states[second_train]
    first_release = first_state.issued_release_time
    second_release = second_state.issued_release_time
    if (
        first_release is not None
        and second_release is not None
        and int(first_release) != int(second_release)
    ):
        return 1 if int(first_release) < int(second_release) else 0
    first_order = instance.train(first_train).original_order
    second_order = instance.train(second_train).original_order
    return 1 if first_order < second_order else 0


def _add_active_mismatch_constraints(
    rows: _RowStore,
    name_prefix: str,
    mismatch_index: int,
    order_index: int,
    desired_order_value: int,
    first_cancel: Mapping[int, float],
    second_cancel: Mapping[int, float],
) -> None:
                                                                               

                                                                           
                                                                          
                                                                           
       

    if desired_order_value == 1:
                                    
        lower_row: Dict[int, float] = {mismatch_index: 1.0, order_index: 1.0}
        _add_expression(lower_row, first_cancel, 1.0)
        _add_expression(lower_row, second_cancel, 1.0)
        rows.add(name_prefix + "_lb", lower_row, lower=1.0)
        rows.add(
            name_prefix + "_q_ub",
            {mismatch_index: 1.0, order_index: 1.0},
            upper=1.0,
        )
    else:
                                
        lower_row = {mismatch_index: 1.0, order_index: -1.0}
        _add_expression(lower_row, first_cancel, 1.0)
        _add_expression(lower_row, second_cancel, 1.0)
        rows.add(name_prefix + "_lb", lower_row, lower=0.0)
        rows.add(
            name_prefix + "_q_ub",
            {mismatch_index: 1.0, order_index: -1.0},
            upper=0.0,
        )

    first_active_ub: Dict[int, float] = {mismatch_index: 1.0}
    _add_expression(first_active_ub, first_cancel, 1.0)
    rows.add(name_prefix + "_first_active", first_active_ub, upper=1.0)
    second_active_ub: Dict[int, float] = {mismatch_index: 1.0}
    _add_expression(second_active_ub, second_cancel, 1.0)
    rows.add(name_prefix + "_second_active", second_active_ub, upper=1.0)


def _section_conflict_sets(
    paths_by_group: Mapping[Tuple[str, str], Tuple[PathColumn, ...]],
    x_by_path: Mapping[str, int],
    scenarios: Sequence[RecoveryScenario],
    section_names: Iterable[str],
) -> Tuple[Set[Tuple[int, int]], Dict[str, int]]:
                                                                        

                                                                          
                                                                       
                                                                        
                                                                          
                                                                            
                                                                      
       

                                                                              
                                                                              
                            
    section_name_list = tuple(section_names)

    all_conflicts: Set[Tuple[int, int]] = set()
    departure_conflicts: Set[Tuple[int, int]] = set()
    arrival_conflicts: Set[Tuple[int, int]] = set()
    order_conflicts: Set[Tuple[int, int]] = set()

    for scenario in scenarios:
        scenario_id = scenario.scenario_id
        scenario_paths: List[PathColumn] = []
        for (group_scenario, _train_id), paths in paths_by_group.items():
            if group_scenario == scenario_id:
                scenario_paths.extend(paths)
        for section_name in section_name_list:
            records: List[Tuple[int, int, int, int, str]] = []
            for path in scenario_paths:
                event = path.section_events.get(section_name)
                if event is None:
                    continue
                departure, arrival, required_headway = event
                records.append(
                    (
                        int(departure),
                        int(arrival),
                        int(required_headway),
                        x_by_path[path.path_id],
                        path.train_id,
                    )
                )
            if len(records) < 2:
                continue
            maximum_headway = max(record[2] for record in records)

                                                                             
                                                                          
            by_departure = sorted(records, key=lambda item: (item[0], item[1], item[3]))
            for left_index, left in enumerate(by_departure):
                right_index = left_index + 1
                while right_index < len(by_departure):
                    right = by_departure[right_index]
                    difference = right[0] - left[0]
                    if difference >= maximum_headway:
                        break
                    if left[4] != right[4] and difference < max(left[2], right[2]):
                        pair = tuple(sorted((left[3], right[3])))
                        departure_conflicts.add(pair)
                    right_index += 1

            by_arrival = sorted(records, key=lambda item: (item[1], item[0], item[3]))
            for left_index, left in enumerate(by_arrival):
                right_index = left_index + 1
                while right_index < len(by_arrival):
                    right = by_arrival[right_index]
                    difference = right[1] - left[1]
                    if difference >= maximum_headway:
                        break
                    if left[4] != right[4] and difference < max(left[2], right[2]):
                        pair = tuple(sorted((left[3], right[3])))
                        arrival_conflicts.add(pair)
                    right_index += 1

                                                                              
                                                                              
                                                                       
            records_by_train: Dict[str, List[Tuple[int, int, int, int, str]]] = {}
            for record in records:
                records_by_train.setdefault(record[4], []).append(record)
            train_ids = sorted(records_by_train)
            for first_position, first_train in enumerate(train_ids):
                for second_train in train_ids[first_position + 1 :]:
                    for left in records_by_train[first_train]:
                        for right in records_by_train[second_train]:
                            departure_difference = left[0] - right[0]
                            arrival_difference = left[1] - right[1]
                            if departure_difference * arrival_difference < 0:
                                pair = tuple(sorted((left[3], right[3])))
                                order_conflicts.add(pair)

    all_conflicts.update(departure_conflicts)
    all_conflicts.update(arrival_conflicts)
    all_conflicts.update(order_conflicts)
    return all_conflicts, {
        "departure": len(departure_conflicts),
        "arrival": len(arrival_conflicts),
        "section_order": len(order_conflicts),
        "unique_section": len(all_conflicts),
    }


def _add_section_safety_rows(
    rows: _RowStore,
    paths_by_group: Mapping[Tuple[str, str], Tuple[PathColumn, ...]],
    x_by_path: Mapping[str, int],
    scenarios: Sequence[RecoveryScenario],
    section_names: Iterable[str],
) -> Dict[str, int]:
                                                                   

                                                                       
                                                                             
                                                                           
                                                                             
                                                       
       

    counts = {
        "departure_cliques": 0,
        "arrival_cliques": 0,
        "section_order_cliques": 0,
        "unique_section_rows": 0,
    }
    seen_rows: Set[Tuple[int, ...]] = set()

    def add_unique(name: str, indices: Iterable[int], kind: str) -> None:
        signature = tuple(sorted(set(int(index) for index in indices)))
        if len(signature) < 2 or signature in seen_rows:
            return
        seen_rows.add(signature)
        rows.add(name, {index: 1.0 for index in signature}, upper=1.0)
        counts[kind] += 1
        counts["unique_section_rows"] += 1

    section_name_list = tuple(section_names)
    for scenario in scenarios:
        scenario_paths: List[PathColumn] = []
        for (scenario_id, _train_id), paths in paths_by_group.items():
            if scenario_id == scenario.scenario_id:
                scenario_paths.extend(paths)

        for section_name in section_name_list:
            records: List[Tuple[int, int, int, int]] = []
            for path in scenario_paths:
                event = path.section_events.get(section_name)
                if event is None:
                    continue
                records.append(
                    (
                        int(event[0]),
                        int(event[1]),
                        int(event[2]),
                        x_by_path[path.path_id],
                    )
                )
            if len(records) < 2:
                continue

            for endpoint, time_position, count_key in (
                ("dep", 0, "departure_cliques"),
                ("arr", 1, "arrival_cliques"),
            ):
                by_time: Dict[int, List[Tuple[int, int]]] = {}
                for record in records:
                    by_time.setdefault(record[time_position], []).append(
                        (record[3], record[2])
                    )
                times = sorted(by_time)
                for time_value in times:
                    add_unique(
                        "section_%s_same[%s,%s,%d]"
                        % (endpoint, scenario.scenario_id, section_name, time_value),
                        (index for index, _headway in by_time[time_value]),
                        count_key,
                    )

                maximum_headway = max(record[2] for record in records)
                for left_position, left_time in enumerate(times):
                    for right_time in times[left_position + 1 :]:
                        difference = right_time - left_time
                        if difference >= maximum_headway:
                            break
                        left = by_time[left_time]
                        right = by_time[right_time]
                        left_all = [index for index, _headway in left]
                        right_all = [index for index, _headway in right]
                        for headway in sorted({value for _index, value in left}):
                            if difference < headway:
                                add_unique(
                                    "section_%s_left_h%d[%s,%s,%d,%d]"
                                    % (
                                        endpoint,
                                        headway,
                                        scenario.scenario_id,
                                        section_name,
                                        left_time,
                                        right_time,
                                    ),
                                    [
                                        index
                                        for index, value in left
                                        if value == headway
                                    ]
                                    + right_all,
                                    count_key,
                                )
                        for headway in sorted({value for _index, value in right}):
                            if difference < headway:
                                add_unique(
                                    "section_%s_right_h%d[%s,%s,%d,%d]"
                                    % (
                                        endpoint,
                                        headway,
                                        scenario.scenario_id,
                                        section_name,
                                        left_time,
                                        right_time,
                                    ),
                                    left_all
                                    + [
                                        index
                                        for index, value in right
                                        if value == headway
                                    ],
                                    count_key,
                                )

                                                                           
                                                                         
            by_event: Dict[Tuple[int, int], List[int]] = {}
            for departure, arrival, _headway, index in records:
                by_event.setdefault((departure, arrival), []).append(index)
            event_groups = sorted(by_event)
            for left_position, left_event in enumerate(event_groups):
                for right_event in event_groups[left_position + 1 :]:
                    if left_event[0] < right_event[0] and left_event[1] > right_event[1]:
                        add_unique(
                            "section_order[%s,%s,%d,%d,%d,%d]"
                            % (
                                scenario.scenario_id,
                                section_name,
                                left_event[0],
                                left_event[1],
                                right_event[0],
                                right_event[1],
                            ),
                            by_event[left_event] + by_event[right_event],
                            "section_order_cliques",
                        )
    return counts


class PathMasterMILP:
                                                                

    def __init__(
        self,
        instance: RailwayInstance,
        context: RollingContext,
        pool: PathPool,
    ) -> None:
        self.instance = instance
        self.context = context
        self.pool = pool

    def _build(
        self,
        scenarios: Sequence[RecoveryScenario],
        allowed_path_ids: Optional[Set[str]],
        fixed_selected_paths: Mapping[Tuple[str, str], str],
        enforce_nonanticipativity: bool,
        include_command_model: bool = True,
    ) -> _ModelArtifacts:
        variables = _VariableStore()
        rows = _RowStore()
        probability = _normalise_probabilities(scenarios)
        scenario_ids = {scenario.scenario_id for scenario in scenarios}

        paths_by_group: Dict[Tuple[str, str], Tuple[PathColumn, ...]] = {}
        x_by_path: Dict[str, int] = {}
        for scenario in scenarios:
            scenario_id = scenario.scenario_id
            if scenario_id not in self.pool.by_scenario_train:
                raise MilpBuildError(
                    "Path pool does not contain scenario %s." % scenario_id
                )
            for train in self.instance.trains:
                train_id = train.train_id
                try:
                    candidates = self.pool.paths(scenario_id, train_id)
                except KeyError:
                    raise MilpBuildError(
                        "Path pool has no group for scenario %s and train %s."
                        % (scenario_id, train_id)
                    )
                if allowed_path_ids is None:
                    selected_candidates = tuple(candidates)
                else:
                    selected_candidates = tuple(
                        path for path in candidates if path.path_id in allowed_path_ids
                    )
                if not selected_candidates:
                    raise MilpBuildError(
                        "Path filtering leaves no candidate for scenario %s, train %s."
                        % (scenario_id, train_id)
                    )
                key = (scenario_id, train_id)
                paths_by_group[key] = selected_candidates
                for path in selected_candidates:
                    if path.path_id in x_by_path:
                        raise MilpBuildError("Duplicate selected path id: %s" % path.path_id)
                    x_by_path[path.path_id] = variables.add(
                        "x[%s]" % path.path_id,
                        lower=0.0,
                        upper=1.0,
                        integer=True,
                    )

                                                                                
                                                                            
        for key, path_id in fixed_selected_paths.items():
            if key[0] not in scenario_ids:
                continue
            if key not in paths_by_group:
                raise MilpBuildError("Unknown fixed scenario/train group: %r" % (key,))
            valid_ids = {path.path_id for path in paths_by_group[key]}
            if path_id not in valid_ids:
                raise MilpBuildError(
                    "Fixed path %s is not allowed for %s/%s."
                    % (path_id, key[0], key[1])
                )
            variables.fix(x_by_path[path_id], 1.0)

                                                                               
                                                                           
        for (scenario_id, train_id), paths in paths_by_group.items():
            rows.add(
                "choose_one[%s,%s]" % (scenario_id, train_id),
                {x_by_path[path.path_id]: 1.0 for path in paths},
                lower=1.0,
                upper=1.0,
            )

        cancellation_expression: Dict[Tuple[str, str], Dict[int, float]] = {}
        for key, paths in paths_by_group.items():
            cancellation_expression[key] = {
                x_by_path[path.path_id]: 1.0
                for path in paths
                if path.cancelled
            }

                                                                         
                                                                             
        for scenario in scenarios:
            row: Dict[int, float] = {}
            for train in self.instance.trains:
                _add_expression(
                    row,
                    cancellation_expression[(scenario.scenario_id, train.train_id)],
                )
            rows.add(
                "cancel_budget[%s]" % scenario.scenario_id,
                row,
                upper=float(self.instance.max_cancellations),
            )

                                                                            
                                                                           
                                                                             
        if len(scenarios) > 1:
            reference_id = scenarios[0].scenario_id
            for scenario in scenarios[1:]:
                for train in self.instance.trains:
                    row = {}
                    _add_expression(
                        row,
                        cancellation_expression[(reference_id, train.train_id)],
                        1.0,
                    )
                    _add_expression(
                        row,
                        cancellation_expression[(scenario.scenario_id, train.train_id)],
                        -1.0,
                    )
                    rows.add(
                        "cancel_consistency[%s,%s,%s]"
                        % (reference_id, scenario.scenario_id, train.train_id),
                        row,
                        lower=0.0,
                        upper=0.0,
                    )

                                                                        
                                                                             
                                                                           
        if enforce_nonanticipativity and len(scenarios) > 1:
            reference_id = scenarios[0].scenario_id
            for train in self.instance.trains:
                signature_union: Set[str] = set()
                signature_paths: Dict[Tuple[str, str], List[PathColumn]] = {}
                for scenario in scenarios:
                    group = paths_by_group[(scenario.scenario_id, train.train_id)]
                    for path in group:
                        signature = _commitment_signature(
                            path, self.context.next_observation
                        )
                        signature_union.add(signature)
                        signature_paths.setdefault(
                            (scenario.scenario_id, signature), []
                        ).append(path)
                for scenario in scenarios[1:]:
                    for signature in sorted(signature_union):
                        row = {}
                        for path in signature_paths.get((reference_id, signature), []):
                            _add_to(row, x_by_path[path.path_id], 1.0)
                        for path in signature_paths.get(
                            (scenario.scenario_id, signature), []
                        ):
                            _add_to(row, x_by_path[path.path_id], -1.0)
                        if row:
                            rows.add(
                                "nonanticipativity[%s,%s,%s,%s]"
                                % (
                                    reference_id,
                                    scenario.scenario_id,
                                    train.train_id,
                                    len(rows.names),
                                ),
                                row,
                                lower=0.0,
                                upper=0.0,
                            )

                                                                      
                                                                             
                                                                             
        occupancy_rows: Dict[Tuple[str, str, int], Dict[int, float]] = {}
        for (scenario_id, _train_id), paths in paths_by_group.items():
            for path in paths:
                index = x_by_path[path.path_id]
                for station_name, minute in set(path.station_occupancy):
                    key = (scenario_id, station_name, int(minute))
                    _add_to(occupancy_rows.setdefault(key, {}), index, 1.0)
        for (scenario_id, station_name, minute), row in occupancy_rows.items():
            rows.add(
                "station_capacity[%s,%s,%d]"
                % (scenario_id, station_name, minute),
                row,
                upper=float(self.instance.station(station_name).hold_capacity),
            )

        conflict_counts = _add_section_safety_rows(
            rows,
            paths_by_group,
            x_by_path,
            scenarios,
            tuple(section.name for section in self.instance.sections),
        )

        order_by_scenario_pair: Dict[Tuple[str, str, str], int] = {}
        reversal_by_scenario_pair: Dict[Tuple[str, str, str], int] = {}
        hdfs_by_scenario_pair: Dict[Tuple[str, str, str], int] = {}

        train_ids_in_original_order = [
            train.train_id
            for train in sorted(
                self.instance.trains, key=lambda item: item.original_order
            )
        ]
        big_m = float(self.instance.horizon_end + 2 * self.instance.time_step)
        order_separation = float(self.instance.time_step)

        for scenario in scenarios:
            scenario_id = scenario.scenario_id
            operating_de_paths: Dict[str, Tuple[PathColumn, ...]] = {}
            for train_id in train_ids_in_original_order:
                state = self.context.train_states[train_id]
                if state.completed or state.already_crossed_disruption:
                    continue
                candidates = tuple(
                    path
                    for path in paths_by_group[(scenario_id, train_id)]
                    if not path.cancelled
                    and self.instance.disruption_section in path.section_events
                )
                if candidates:
                    operating_de_paths[train_id] = candidates

            eligible_trains = (
                [
                    train_id
                    for train_id in train_ids_in_original_order
                    if train_id in operating_de_paths
                ]
                if include_command_model
                else []
            )
            for first_position, first_train in enumerate(eligible_trains):
                for second_train in eligible_trains[first_position + 1 :]:
                    pair_key = (scenario_id, first_train, second_train)
                    order_index = variables.add(
                        "qDE[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        integer=True,
                    )
                    order_by_scenario_pair[pair_key] = order_index

                    first_time: Dict[int, float] = {}
                    for path in operating_de_paths[first_train]:
                        departure = path.section_events[
                            self.instance.disruption_section
                        ][0]
                        _add_to(
                            first_time,
                            x_by_path[path.path_id],
                            float(departure),
                        )
                    second_time: Dict[int, float] = {}
                    for path in operating_de_paths[second_train]:
                        departure = path.section_events[
                            self.instance.disruption_section
                        ][0]
                        _add_to(
                            second_time,
                            x_by_path[path.path_id],
                            float(departure),
                        )
                    first_cancel = cancellation_expression[
                        (scenario_id, first_train)
                    ]
                    second_cancel = cancellation_expression[
                        (scenario_id, second_train)
                    ]

                                                                             
                                                                             
                                                                         
                                                                        
                    first_active_row: Dict[int, float] = {order_index: 1.0}
                    _add_expression(first_active_row, first_cancel, 1.0)
                    rows.add(
                        "de_order_active_first[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        first_active_row,
                        upper=1.0,
                    )
                    second_active_row: Dict[int, float] = {order_index: 1.0}
                    _add_expression(second_active_row, second_cancel, 1.0)
                    rows.add(
                        "de_order_active_second[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        second_active_row,
                        upper=1.0,
                    )

                                                                        
                                                                      
                    first_before_row: Dict[int, float] = {}
                    _add_expression(first_before_row, first_time, 1.0)
                    _add_expression(first_before_row, second_time, -1.0)
                    _add_to(first_before_row, order_index, big_m)
                    _add_expression(first_before_row, first_cancel, -big_m)
                    _add_expression(first_before_row, second_cancel, -big_m)
                    rows.add(
                        "de_order_first[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        first_before_row,
                        upper=big_m - order_separation,
                    )

                    second_before_row: Dict[int, float] = {}
                    _add_expression(second_before_row, second_time, 1.0)
                    _add_expression(second_before_row, first_time, -1.0)
                    _add_to(second_before_row, order_index, -big_m)
                    _add_expression(second_before_row, first_cancel, -big_m)
                    _add_expression(second_before_row, second_cancel, -big_m)
                    rows.add(
                        "de_order_second[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        second_before_row,
                        upper=-order_separation,
                    )

                    reversal_index = variables.add(
                        "reverseDE[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        integer=True,
                    )
                    reversal_by_scenario_pair[pair_key] = reversal_index
                    _add_active_mismatch_constraints(
                        rows,
                        "issued_reverse[%s,%s,%s]"
                        % (scenario_id, first_train, second_train),
                        reversal_index,
                        order_index,
                        _issued_precedence(
                            self.instance,
                            self.context,
                            first_train,
                            second_train,
                        ),
                        first_cancel,
                        second_cancel,
                    )

                                                                             
                                                                            
                                                                             
                                                                             

                                                                            
                                                                              
                                                                             
                                                                         
                                                                             

                                                                         
                                                                         
                                                                           
                                                                            
                                                                             
                                                                  
        peak_by_scenario: Dict[str, int] = {}
        order_change_weight = float(COMMAND_WEIGHTS["order_change"])
        for scenario in scenarios if include_command_model else ():
            scenario_id = scenario.scenario_id
            peak_index = variables.add(
                "peak[%s]" % scenario_id,
                lower=0.0,
                upper=np.inf,
                integer=False,
            )
            peak_by_scenario[scenario_id] = peak_index
            bucket_loads: Dict[int, Dict[int, float]] = {}
            for (group_scenario, _train_id), paths in paths_by_group.items():
                if group_scenario != scenario_id:
                    continue
                for path in paths:
                    x_index = x_by_path[path.path_id]
                    for event in path.command_events:
                        if event.command_type == "order_change":
                            continue
                        bucket = _time_bucket(
                            int(event.issue_time), self.instance.time_step
                        )
                        _add_to(
                            bucket_loads.setdefault(bucket, {}),
                            x_index,
                            float(event.weight),
                        )
            order_issue_bucket = _time_bucket(
                max(
                    self.context.now,
                    int(scenario.t12) - 2 * self.instance.time_step,
                ),
                self.instance.time_step,
            )
            for (pair_scenario, _first, _second), reverse_index in (
                reversal_by_scenario_pair.items()
            ):
                if pair_scenario == scenario_id:
                    _add_to(
                        bucket_loads.setdefault(order_issue_bucket, {}),
                        reverse_index,
                        order_change_weight,
                    )
            for bucket, load_expression in sorted(bucket_loads.items()):
                row: Dict[int, float] = {peak_index: 1.0}
                _add_expression(row, load_expression, -1.0)
                rows.add(
                    "peak_dominates[%s,%d]" % (scenario_id, bucket),
                    row,
                    lower=0.0,
                )

                                                                              
                                              
        service_objective = np.zeros(variables.size, dtype=float)
        hdfs_objective = np.zeros(variables.size, dtype=float)
        workload_objective = np.zeros(variables.size, dtype=float)
        peak_objective = np.zeros(variables.size, dtype=float)

        for (scenario_id, _train_id), paths in paths_by_group.items():
            scenario_probability = probability[scenario_id]
            for path in paths:
                index = x_by_path[path.path_id]
                service_cost = (
                    self.instance.cancellation_penalty
                    if path.cancelled
                    else path.terminal_delay
                )
                service_objective[index] += scenario_probability * float(
                    service_cost
                )
                path_workload = sum(
                    float(event.weight)
                    for event in path.command_events
                    if event.command_type != "order_change"
                )
                workload_objective[index] += (
                    scenario_probability * path_workload
                )
        for (scenario_id, _first, _second), index in (
            hdfs_by_scenario_pair.items()
        ):
            hdfs_objective[index] = probability[scenario_id]
        for (scenario_id, _first, _second), index in (
            reversal_by_scenario_pair.items()
        ):
            workload_objective[index] = (
                probability[scenario_id] * order_change_weight
            )
        for scenario_id, index in peak_by_scenario.items():
            peak_objective[index] = probability[scenario_id]

        return _ModelArtifacts(
            variables=variables,
            rows=rows,
            scenarios=tuple(scenarios),
            probability=probability,
            paths_by_group=paths_by_group,
            x_by_path=x_by_path,
            order_by_scenario_pair=order_by_scenario_pair,
            reversal_by_scenario_pair=reversal_by_scenario_pair,
            hdfs_by_scenario_pair=hdfs_by_scenario_pair,
            peak_by_scenario=peak_by_scenario,
            service_objective=service_objective,
            hdfs_objective=hdfs_objective,
            workload_objective=workload_objective,
            peak_objective=peak_objective,
            cancellation_expression=cancellation_expression,
            conflict_counts=conflict_counts,
        )

    def solve(
        self,
        algorithm: str,
        scenarios: Optional[Sequence[Union[RecoveryScenario, str]]] = None,
        allowed_path_ids: Optional[Iterable[str]] = None,
        fixed_selected_paths: Optional[Mapping[object, str]] = None,
        enforce_nonanticipativity: bool = True,
        epsilon: Optional[float] = None,
        service_reference: Optional[float] = None,
        cancellation_reference: Optional[int] = None,
        time_limit: Optional[float] = None,
        backend: str = "highs",
        objective_mode: str = "full",
        stabilize_cancellations: bool = True,
    ) -> ScheduleSolution:
                                                                   

                                                                              
                                                                                     
                                                                              
                                                        
           

        start_time = time.perf_counter()
        backend_name = str(backend).strip().lower()
        if backend_name not in {"highs", "gurobi"}:
            raise MilpBuildError("backend must be either 'highs' or 'gurobi'.")
        mode = str(objective_mode).strip().lower()
        if mode not in {"full", "service_only", "command_only"}:
            raise MilpBuildError("objective_mode must be 'full', 'service_only' or 'command_only'.")
        if mode == "command_only" and (service_reference is None or cancellation_reference is None):
            raise MilpBuildError("command_only requires common service and cancellation references.")
        selected_scenarios = _resolve_scenarios(
            self.context, algorithm, scenarios
        )
        allowed_set = (
            None
            if allowed_path_ids is None
            else {str(path_id) for path_id in allowed_path_ids}
        )
        fixed = _normalise_fixed_paths(
            fixed_selected_paths, selected_scenarios
        )
        artifacts = self._build(
            selected_scenarios,
            allowed_set,
            fixed,
            enforce_nonanticipativity=(
                bool(enforce_nonanticipativity)
                and len(selected_scenarios) > 1
            ),
            include_command_model=(mode != "service_only"),
        )
        build_seconds = time.perf_counter() - start_time

        if epsilon is None:
            epsilon_value = float(self.context.epsilon)
        else:
            epsilon_value = float(epsilon)
        if epsilon_value < 0.0:
            raise MilpBuildError("epsilon must be non-negative.")

        stage_results: Dict[str, OptimizeResult] = {}
        stage_values: Dict[str, float] = {}
        stage_seconds: Dict[str, float] = {}

        def remaining_seconds() -> Optional[float]:
            if time_limit is None:
                return None
            return max(0.01, float(time_limit) - (time.perf_counter() - start_time))

        def run_stage(stage_name: str, objective: np.ndarray) -> OptimizeResult:
            stage_start = time.perf_counter()
            remaining = remaining_seconds()
            if time_limit is not None and remaining is not None and remaining <= 0.011:
                raise MilpSolveError(
                    "Total time limit expired before stage %s." % stage_name
                )
            options: Dict[str, object] = {
                "disp": False,
                "presolve": True,
                                                                           
                                                                            
                "mip_rel_gap": 0.0,
            }
            if remaining is not None:
                options["time_limit"] = float(remaining)
            if backend_name == "gurobi":
                result = _gurobi_milp(
                    objective,
                    artifacts.variables,
                    artifacts.rows,
                    remaining,
                )
            else:
                result = milp(
                    c=objective,
                    integrality=np.asarray(
                        artifacts.variables.integrality, dtype=np.int32
                    ),
                    bounds=artifacts.variables.bounds(),
                    constraints=artifacts.rows.linear_constraint(
                        artifacts.variables.size
                    ),
                    options=options,
                )
            if getattr(result, "x", None) is None:
                raise MilpSolveError(
                    "MILP stage %s has no feasible incumbent (status=%s): %s"
                    % (stage_name, result.status, result.message)
                )
                                                                         
                                                                         
                                                                             
                                                                          
            cleaned_x = np.asarray(result.x, dtype=float).copy()
            integer_mask = np.asarray(
                artifacts.variables.integrality, dtype=bool
            )
            cleaned_x[integer_mask] = np.rint(cleaned_x[integer_mask])
            result.x = cleaned_x
            stage_results[stage_name] = result
            stage_values[stage_name] = float(np.dot(objective, result.x))
            stage_seconds[stage_name] = time.perf_counter() - stage_start
            return result

                                                                           
                                                                             
                                                                           
                                                                               
        if mode == "command_only":
            if float(service_reference) < 0 or not 0 <= int(cancellation_reference) <= self.instance.max_cancellations:
                raise MilpBuildError("Invalid common references.")
            artifacts.rows.add(
                "service_epsilon_cap",
                {i: float(v) for i, v in enumerate(artifacts.service_objective) if abs(float(v)) > 1e-12},
                upper=(1.0 + epsilon_value) * float(service_reference) + max(DEFAULT_OBJECTIVE_TOLERANCE, abs(float(service_reference)) * 1e-9),
            )
            for scenario in selected_scenarios:
                row: Dict[int, float] = {}
                for train in self.instance.trains:
                    _add_expression(row, artifacts.cancellation_expression[(scenario.scenario_id, train.train_id)])
                artifacts.rows.add("stage1_cancel_cap[%s]" % scenario.scenario_id, row, upper=float(cancellation_reference))
            service_result = run_stage("command_total", artifacts.workload_objective)
            restricted_service_optimum = float(np.dot(artifacts.service_objective, service_result.x))
        else:
            service_result = run_stage("service", artifacts.service_objective)
            restricted_service_optimum = stage_values["service"]
        service_optimum = (
            restricted_service_optimum
            if service_reference is None
            else float(service_reference)
        )
        if service_optimum < 0.0:
            raise MilpBuildError("service_reference must be non-negative.")
        tolerance = max(
            DEFAULT_OBJECTIVE_TOLERANCE,
            abs(service_optimum) * 1.0e-9,
        )

                                                                     
                                                                      
                                                                            
                                                                      
                                                                            
                                                                              
        reference_scenario = selected_scenarios[0].scenario_id
        cancellation_objective = np.zeros(artifacts.variables.size, dtype=float)
        for train in self.instance.trains:
            expression = artifacts.cancellation_expression[
                (reference_scenario, train.train_id)
            ]
            for index, coefficient in expression.items():
                cancellation_objective[index] += float(coefficient)

        cancellation_stage_complete = True
        cancellation_stage_skipped = (
            not bool(stabilize_cancellations)
            or cancellation_reference is not None
        )
        if cancellation_stage_skipped:
            cancellation_result = service_result
        else:
            artifacts.rows.add(
                "service_optimality_cap_for_cancellation",
                {
                    index: float(value)
                    for index, value in enumerate(artifacts.service_objective)
                    if abs(float(value)) > 1.0e-12
                },
                upper=restricted_service_optimum
                + max(
                    DEFAULT_OBJECTIVE_TOLERANCE,
                    abs(restricted_service_optimum) * 1.0e-9,
                ),
            )
            try:
                cancellation_result = run_stage(
                    "cancellation", cancellation_objective
                )
            except MilpSolveError:
                                                                      
                                                                          
                                                                    
                                                            
                cancellation_result = service_result
                cancellation_stage_complete = False
                                                                    
            artifacts.rows.names.pop()
            artifacts.rows.coefficients.pop()
            artifacts.rows.lower.pop()
            artifacts.rows.upper.pop()
        restricted_cancellation_count = int(
            round(float(np.dot(cancellation_objective, cancellation_result.x)))
        )
        cancellation_count = (
            restricted_cancellation_count
            if cancellation_reference is None
            else int(cancellation_reference)
        )
        if cancellation_count < 0 or cancellation_count > self.instance.max_cancellations:
            raise MilpBuildError(
                "cancellation_reference must lie within the cancellation budget."
            )

        workload_optimum: Optional[float] = None
        if mode != "service_only":
            if mode != "command_only":
                artifacts.rows.add(
                    "service_epsilon_cap",
                    {
                        index: float(value)
                        for index, value in enumerate(artifacts.service_objective)
                        if abs(float(value)) > 1.0e-12
                    },
                    upper=(1.0 + epsilon_value) * service_optimum + tolerance,
                )
                                                                            
                                                                              
                                                                     
            for scenario in (() if mode == "command_only" else selected_scenarios):
                row: Dict[int, float] = {}
                for train in self.instance.trains:
                    _add_expression(
                        row,
                        artifacts.cancellation_expression[
                            (scenario.scenario_id, train.train_id)
                        ],
                    )
                artifacts.rows.add(
                    "stage1_cancel_cap[%s]" % scenario.scenario_id,
                    row,
                    upper=float(cancellation_count),
                )

            try:
                command_result = service_result if mode == "command_only" else run_stage(
                    "command_total", artifacts.workload_objective
                )
            except MilpSolveError:
                final_result = cancellation_result
            else:
                workload_optimum = stage_values["command_total"]
                artifacts.rows.add(
                    "command_total_lex_cap",
                    {
                        index: float(value)
                        for index, value in enumerate(artifacts.workload_objective)
                        if abs(float(value)) > 1.0e-12
                    },
                    upper=workload_optimum
                    + max(
                        DEFAULT_OBJECTIVE_TOLERANCE,
                        abs(workload_optimum) * 1.0e-9,
                    ),
                )
                try:
                    peak_result = run_stage(
                        "command_peak", artifacts.peak_objective
                    )
                except MilpSolveError:
                    final_result = command_result
                else:
                                                                          
                                                                            
                                                                            
                                                                           
                                                                   
                    peak_optimum = stage_values["command_peak"]
                    artifacts.rows.add(
                        "command_peak_lex_cap",
                        {
                            index: float(value)
                            for index, value in enumerate(artifacts.peak_objective)
                            if abs(float(value)) > 1.0e-12
                        },
                        upper=peak_optimum
                        + max(
                            DEFAULT_OBJECTIVE_TOLERANCE,
                            abs(peak_optimum) * 1.0e-9,
                        ),
                    )
                    try:
                        final_result = run_stage(
                            "service_tie_break", artifacts.service_objective
                        )
                    except MilpSolveError:
                        final_result = peak_result
        else:
                                                                          
                                                         
            final_result = cancellation_result
        final_x = final_result.x

        selected_paths: Dict[Tuple[str, str], str] = {}
        for key, paths in artifacts.paths_by_group.items():
            best_path = max(
                paths,
                key=lambda path: final_x[artifacts.x_by_path[path.path_id]],
            )
            best_value = final_x[artifacts.x_by_path[best_path.path_id]]
            if best_value < 0.5 - 1.0e-6:
                raise MilpSolveError(
                    "Decoded group %r has no selected integer path." % (key,)
                )
            selected_paths[key] = best_path.path_id

        cancelled_trains = tuple(
            sorted(
                train.train_id
                for train in self.instance.trains
                if self.pool.by_id[selected_paths[
                    (reference_scenario, train.train_id)
                ]].cancelled
            )
        )
                                                                
                                                                           
                                                           
        from .validation import recompute_objective

        resolved_paths = {
            key: self.pool.by_id[path_id]
            for key, path_id in selected_paths.items()
        }
        recomputed_objective, recomputed_details = recompute_objective(
            self.instance,
            self.context,
            resolved_paths,
            scenarios=selected_scenarios,
        )
        final_service = recomputed_objective.service_loss
        final_hdfs = recomputed_objective.hdfs_violations
        final_workload = recomputed_objective.workload_total
        final_peak = recomputed_objective.workload_peak

        stage_status = {
            name: int(result.status) for name, result in stage_results.items()
        }
        required_stages_complete = cancellation_stage_complete and (
            mode == "service_only" or "command_peak" in stage_results
        )
        all_optimal = required_stages_complete and all(
            status == 0 for status in stage_status.values()
        )
        status = "optimal" if all_optimal else "feasible_limit"
        gaps: List[float] = []
        for result in stage_results.values():
            gap = getattr(result, "mip_gap", None)
            if gap is not None and math.isfinite(float(gap)):
                gaps.append(float(gap))

        de_orders: Dict[str, List[str]] = {}
        de_reversals: Dict[str, List[str]] = {}
        hdfs_violations: Dict[str, List[str]] = {}
        for (scenario_id, first, second), index in (
            artifacts.order_by_scenario_pair.items()
        ):
            if (
                self.pool.by_id[selected_paths[(scenario_id, first)]].cancelled
                or self.pool.by_id[selected_paths[(scenario_id, second)]].cancelled
            ):
                continue
            if final_x[index] >= 0.5:
                order_text = "%s<%s" % (first, second)
            else:
                order_text = "%s<%s" % (second, first)
            de_orders.setdefault(scenario_id, []).append(order_text)
        for (scenario_id, first, second), index in (
            artifacts.reversal_by_scenario_pair.items()
        ):
            if (
                self.pool.by_id[selected_paths[(scenario_id, first)]].cancelled
                or self.pool.by_id[selected_paths[(scenario_id, second)]].cancelled
            ):
                continue
            if final_x[index] >= 0.5:
                de_reversals.setdefault(scenario_id, []).append(
                    "%s/%s" % (first, second)
                )
        for scenario_id, values in recomputed_details["by_scenario"].items():
            if float(values["hdfs_violations"]) > 0.0:
                hdfs_violations[scenario_id] = [
                    "count=%g" % float(values["hdfs_violations"])
                ]

        runtime_seconds = time.perf_counter() - start_time
        return ScheduleSolution(
            algorithm=str(algorithm),
            status=status,
            selected_paths=selected_paths,
            objective=ObjectiveVector(
                service_loss=final_service,
                hdfs_violations=final_hdfs,
                workload_total=final_workload,
                workload_peak=final_peak,
            ),
            cancelled_trains=cancelled_trains,
            service_optimum=service_optimum,
            cancellation_bound=cancellation_count,
            stage_objectives={
                "service_optimum": service_optimum,
                "service_final": final_service,
                "cancellation_optimum": float(cancellation_count),
                "command_total_optimum": (
                    final_workload if workload_optimum is None else workload_optimum
                ),
                "command_peak_optimum": final_peak,
                "service_tie_break": final_service,
            },
            runtime_seconds=runtime_seconds,
            mip_gap=max(gaps) if gaps else None,
            metadata={
                "scenario_ids": [
                    scenario.scenario_id for scenario in selected_scenarios
                ],
                "solver_backend": backend_name,
                "scenario_probability": dict(artifacts.probability),
                "epsilon": epsilon_value,
                "objective_mode": mode,
                "restricted_service_optimum": restricted_service_optimum,
                "service_reference_external": service_reference is not None,
                "restricted_cancellation_count": restricted_cancellation_count,
                "cancellation_stage_complete": cancellation_stage_complete,
                "cancellation_stage_skipped": cancellation_stage_skipped,
                "stabilize_cancellations": bool(stabilize_cancellations),
                "cancellation_reference_external": cancellation_reference is not None,
                "nonanticipativity": bool(enforce_nonanticipativity)
                and len(selected_scenarios) > 1,
                "variable_count": artifacts.variables.size,
                "constraint_count": artifacts.rows.size,
                "path_variable_count": len(artifacts.x_by_path),
                "build_seconds": build_seconds,
                "stage_status": stage_status,
                "stage_seconds": stage_seconds,
                "stage_bounds": {
                    name: {
                        "incumbent": stage_values[name],
                        "dual_bound": float(result.mip_dual_bound) if getattr(result, "mip_dual_bound", None) is not None and math.isfinite(float(result.mip_dual_bound)) else None,
                        "relative_gap": float(result.mip_gap) if getattr(result, "mip_gap", None) is not None and math.isfinite(float(result.mip_gap)) else None,
                        "node_count": int(result.mip_node_count) if getattr(result, "mip_node_count", None) is not None else None,
                        "scope": "full_path_pool" if allowed_set is None and not fixed else "restricted_path_domain",
                    }
                    for name, result in stage_results.items()
                },
                "stage_messages": {
                    name: str(result.message)
                    for name, result in stage_results.items()
                },
                "conflict_counts": dict(artifacts.conflict_counts),
                "station_clearance_minutes": 0,
                "de_orders": de_orders,
                "de_reversals": de_reversals,
                "hdfs_violation_pairs": hdfs_violations,
                "fixed_path_count": len(fixed),
                "allowed_path_count": (
                    None if allowed_set is None else len(allowed_set)
                ),
            },
        )


def solve_lexicographic(
    instance: RailwayInstance,
    context: RollingContext,
    pool: Optional[PathPool],
    algorithm: str,
    scenarios: Optional[Sequence[Union[RecoveryScenario, str]]] = None,
    allowed_path_ids: Optional[Iterable[str]] = None,
    fixed_selected_paths: Optional[Mapping[object, str]] = None,
    enforce_nonanticipativity: bool = True,
    epsilon: Optional[float] = None,
    service_reference: Optional[float] = None,
    cancellation_reference: Optional[int] = None,
    time_limit: Optional[float] = None,
    backend: str = "highs",
    objective_mode: str = "full",
    stabilize_cancellations: bool = True,
) -> ScheduleSolution:
                                                                        

              
              
                     
                                                                           
        
                                                                          
                                       
             
                                                                        
                                                                          
                                                                             
             
                                                                             
                    
                                                                            
                                 
                        
                                                                           
                             
                                                                         
                                                                           
                    
           
                                                                  
                            
                                             
                                                                              
                                                                           
                                                                        
                             
              
                                                                        
                                         
                  
                                                                    
                                                                           
                                              
                           
                                                                         
                                                                         
       

    selected_scenarios = _resolve_scenarios(context, algorithm, scenarios)
    actual_pool = pool
    if actual_pool is None:
        actual_pool = generate_path_pool(
            instance, context, scenarios=selected_scenarios
        )
    return PathMasterMILP(instance, context, actual_pool).solve(
        algorithm=algorithm,
        scenarios=selected_scenarios,
        allowed_path_ids=allowed_path_ids,
        fixed_selected_paths=fixed_selected_paths,
        enforce_nonanticipativity=enforce_nonanticipativity,
        epsilon=epsilon,
        service_reference=service_reference,
        cancellation_reference=cancellation_reference,
        time_limit=time_limit,
        backend=backend,
        objective_mode=objective_mode,
        stabilize_cancellations=stabilize_cancellations,
    )


__all__ = [
    "HDFS_DELAY_THRESHOLD_MIN",
    "MilpBuildError",
    "MilpSolveError",
    "PathMasterMILP",
    "solve_lexicographic",
]
