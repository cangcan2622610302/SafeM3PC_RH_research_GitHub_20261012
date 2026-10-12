                                                                

                                                                           
                                                                              
                                                                             
        

                                                                      
                                                  
                                                                            
                                                                  
                                                                             
                       
                                                                          

                                                                         
                                             
   

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..domain import PathColumn, RailwayInstance, RecoveryScenario, RollingContext
from ..paths import PathPool

try:                                                       
    import torch
    from torch import Tensor, nn
except ImportError:                                                          
    torch = None
    Tensor = object                
    nn = None                


FEATURE_DIM = 12


@dataclass(frozen=True)
class SafeM3PCConfig:
                                                                    

    embedding_dim: int = 32
    attention_heads: int = 4
    transformer_layers: int = 2
    training_epochs: int = 25
    batch_size: int = 128
    learning_rate: float = 2.0e-3
                                                                         
                                                                         
                                                                         
                                                                           
                                                                  
    max_training_paths: Optional[int] = 12000
    inference_batch_size: int = 4096
                                                                          
                                                                            
                                       
    candidate_widths: Tuple[Optional[int], ...] = (8, 16, None)
    random_seed: int = 20260826
    use_hdfs_backbone: bool = True
    use_constructive_backbone: bool = False
    stop_after_first_feasible: bool = False
    command_aware_ranking_for_service: bool = False


if nn is not None:

    class MaskedTrajectoryTransformer(nn.Module):
                                                                            

                                                                        
                                                                              
                                                                             
                                                                         
           

        def __init__(
            self,
            feature_dim: int = FEATURE_DIM,
            embedding_dim: int = 32,
            attention_heads: int = 4,
            layers: int = 2,
        ) -> None:
            super().__init__()
            self.feature_dim = feature_dim
            self.input_projection = nn.Linear(feature_dim, embedding_dim)
            self.position_embedding = nn.Parameter(torch.zeros(1, 3, embedding_dim))
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=attention_heads,
                dim_feedforward=embedding_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.output_projection = nn.Linear(embedding_dim, feature_dim)
            self.mask_token = nn.Parameter(torch.zeros(1, 1, feature_dim))
            nn.init.normal_(self.position_embedding, std=0.02)
            nn.init.normal_(self.mask_token, std=0.02)

        def forward(self, trajectory: Tensor, mask: Tensor) -> Tensor:
                                                                               

            masked = torch.where(mask.unsqueeze(-1), self.mask_token, trajectory)
            hidden = self.input_projection(masked) + self.position_embedding
            return self.output_projection(self.encoder(hidden))


def _scenario_by_id(context: RollingContext) -> Dict[str, RecoveryScenario]:
    return {item.scenario_id: item for item in context.scenarios}


def encode_path_trajectory(
    instance: RailwayInstance,
    context: RollingContext,
    path: PathColumn,
) -> np.ndarray:
                                                                     

                                                                         
                                                                             
                                              
       

    state = context.train_states[path.train_id]
    scenario = _scenario_by_id(context)[path.scenario_id]
    station_names = tuple(station.name for station in instance.stations)
    station_index = {name: index for index, name in enumerate(station_names)}
    station_scale = float(max(len(station_names) - 1, 1))
    context_token = np.zeros(FEATURE_DIM, dtype=np.float32)
    context_token[0] = 0.0                       
    context_token[1] = station_index[state.current_station] / station_scale
    context_token[2] = min(state.current_delay, 240) / 240.0
    context_token[3] = (scenario.t12 - context.now) / 180.0
    context_token[4] = (scenario.t23 - context.now) / 240.0
    context_token[5] = scenario.s2_speed_kmh / 300.0
    context_token[6] = scenario.probability
    context_token[7] = float(state.cancel_allowed)
    context_token[8] = float(state.completed)
    if state.issued_release_time is not None:
        context_token[9] = (state.issued_release_time - context.now) / 240.0

    action_token = np.zeros(FEATURE_DIM, dtype=np.float32)
    action_token[0] = 0.5                            
    action_token[1] = (
        station_index[path.hold_station] / station_scale
        if path.hold_station is not None
        else -1.0
    )
    action_token[2] = (
        (path.release_time - context.now) / 240.0
        if path.release_time is not None
        else -1.0
    )
    action_token[3] = float(path.cancelled)
    action_token[4] = float(path.near_term_signature == "COMPLETE")
    if path.release_time is not None:
        action_token[5] = float(scenario.state_at(path.release_time) == "S2")
        action_token[6] = float(scenario.state_at(path.release_time) == "S3")
    action_token[7] = path.path_workload / 4.0

    outcome_token = np.zeros(FEATURE_DIM, dtype=np.float32)
    outcome_token[0] = 1.0                             
    outcome_token[1] = min(path.terminal_delay, 480) / 480.0
    outcome_token[2] = path.path_workload / 4.0
    outcome_token[3] = float(path.cancelled)
                                                                          
                                                                         
    service = path.terminal_delay + instance.cancellation_penalty * int(path.cancelled)
    outcome_token[4] = min(service, 960) / 960.0
    outcome_token[5] = (
        max(0, (path.release_time or context.now) - scenario.t12) / 240.0
    )
    outcome_token[6] = len(path.station_occupancy) / 100.0
    return np.stack((context_token, action_token, outcome_token), axis=0)


def build_offline_trajectory_array(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
) -> np.ndarray:
                                                                                 

    return np.stack(
        [encode_path_trajectory(instance, context, path) for path in pool.all_paths()],
        axis=0,
    )


def _balanced_training_paths(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    maximum: Optional[int],
) -> List[PathColumn]:
                                                                           

                                                                             
                                                                          
                                                                       
                                                                             
                                                        
       

    all_paths = list(pool.all_paths())
    if maximum is None or maximum <= 0 or len(all_paths) <= maximum:
        return all_paths

    group_count = max(1, len(context.scenarios) * len(instance.trains))
    quota = max(8, int(maximum) // group_count)
    selected: Dict[str, PathColumn] = {}

    def retain(path: PathColumn) -> None:
        selected[path.path_id] = path

    for scenario in context.scenarios:
        for train in instance.trains:
            group = list(pool.paths(scenario.scenario_id, train.train_id))
            for path in group:
                if path.cancelled or path.near_term_signature == "COMPLETE":
                    retain(path)

            ordered_service = sorted(
                group,
                key=lambda path: (
                    instance.cancellation_penalty * int(path.cancelled)
                    + path.terminal_delay,
                    path.path_workload,
                    path.path_id,
                ),
            )
            ordered_command = sorted(
                group,
                key=lambda path: (
                    path.path_workload,
                    instance.cancellation_penalty * int(path.cancelled)
                    + path.terminal_delay,
                    path.path_id,
                ),
            )
            extreme_count = max(2, min(8, quota // 6))
            for path in ordered_service[:extreme_count]:
                retain(path)
            for path in ordered_command[:extreme_count]:
                retain(path)

            by_hold: Dict[str, List[PathColumn]] = {}
            for path in group:
                if path.release_time is None or path.cancelled:
                    continue
                by_hold.setdefault(path.hold_station or "", []).append(path)
            remaining = max(1, quota - 2 * extreme_count)
            per_hold = max(1, remaining // max(len(by_hold), 1))
            for hold_paths in by_hold.values():
                hold_paths.sort(key=lambda path: (int(path.release_time or 0), path.path_id))
                if len(hold_paths) <= per_hold:
                    chosen = hold_paths
                elif per_hold == 1:
                    chosen = [hold_paths[len(hold_paths) // 2]]
                else:
                    positions = np.linspace(0, len(hold_paths) - 1, per_hold)
                    chosen = [hold_paths[int(round(position))] for position in positions]
                for path in chosen:
                    retain(path)

    ordered = sorted(selected.values(), key=lambda path: path.path_id)
    if len(ordered) <= maximum:
        return ordered

                                                                             
                                                                           
                                                               
    required = [
        path
        for path in ordered
        if path.cancelled or path.near_term_signature == "COMPLETE"
    ]
    optional = [path for path in ordered if path not in required]
    slots = max(0, int(maximum) - len(required))
    if slots >= len(optional):
        return required + optional
    positions = np.linspace(0, len(optional) - 1, slots) if slots else []
    return required + [optional[int(round(position))] for position in positions]


def train_masked_model(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    config: SafeM3PCConfig = SafeM3PCConfig(),
) -> Tuple[object, List[float]]:
                                                                            

                                                                            
                                                                          
                                                               
       

    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for the Safe-M3PC-RH prototype.")
    torch.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)
    replay_paths = _balanced_training_paths(
        instance,
        context,
        pool,
        config.max_training_paths,
    )
    trajectories = torch.tensor(
        np.stack(
            [encode_path_trajectory(instance, context, path) for path in replay_paths]
        ),
        dtype=torch.float32,
    )
    model = MaskedTrajectoryTransformer(
        embedding_dim=config.embedding_dim,
        attention_heads=config.attention_heads,
        layers=config.transformer_layers,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.random_seed)
    losses: List[float] = []
    model.train()
    for _epoch in range(config.training_epochs):
        permutation = torch.randperm(len(trajectories), generator=generator)
        epoch_loss = 0.0
        sample_count = 0
        for start in range(0, len(trajectories), config.batch_size):
            batch = trajectories[permutation[start : start + config.batch_size]]
            mask = torch.zeros(batch.shape[:2], dtype=torch.bool)
            mask[:, 2] = True
            prediction = model(batch, mask)
            loss = torch.mean((prediction[:, 2, :] - batch[:, 2, :]) ** 2)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(batch)
            sample_count += len(batch)
        losses.append(epoch_loss / max(sample_count, 1))
                                                                              
                                                        
    model._hsr_training_sample_count = len(replay_paths)                              
    model._hsr_training_pool_count = len(pool.by_id)                              
    return model, losses


def _predict_path_scores(
    model: object,
    instance: RailwayInstance,
    context: RollingContext,
    paths: Sequence[PathColumn],
    include_command_terms: bool = True,
    batch_size: int = 4096,
) -> List[Tuple[float, str]]:
                                                                       

    if torch is None:
        raise RuntimeError("PyTorch is required for learned candidate ranking.")
    if not paths:
        return []
    model.eval()
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(paths), max(1, int(batch_size))):
            batch_paths = paths[start : start + max(1, int(batch_size))]
            trajectory = torch.tensor(
                np.stack(
                    [
                        encode_path_trajectory(instance, context, path)
                        for path in batch_paths
                    ]
                ),
                dtype=torch.float32,
            )
            mask = torch.zeros(trajectory.shape[:2], dtype=torch.bool)
            mask[:, 2] = True
            predictions.append(model(trajectory, mask)[:, 2, :].cpu().numpy())
    predicted = np.concatenate(predictions, axis=0)
                                                                           
                                                                             
                                                                           
    command_term = 10.0 * predicted[:, 2] if include_command_terms else 0.0
    scores = 1000.0 * predicted[:, 4] + command_term + predicted[:, 5]
    return sorted((float(score), path.path_id) for score, path in zip(scores, paths))


def rank_candidate_paths(
    model: object,
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    include_command_terms: bool = True,
    inference_batch_size: int = 4096,
) -> Dict[Tuple[str, str], List[str]]:
                                                                               

    ranking: Dict[Tuple[str, str], List[str]] = {}
    for scenario in context.scenarios:
        for train in instance.trains:
            paths = pool.paths(scenario.scenario_id, train.train_id)
            predicted = {
                path_id: score
                for score, path_id in _predict_path_scores(
                    model,
                    instance,
                    context,
                    paths,
                    include_command_terms=include_command_terms,
                    batch_size=inference_batch_size,
                )
            }
            if include_command_terms:
                issued = context.train_states[train.train_id].issued_release_time

                def command_aware_key(path: PathColumn):
                    release_deviation = (
                        10**9
                        if path.release_time is None and not path.cancelled
                        else 0
                        if issued is None or path.release_time is None
                        else abs(int(path.release_time) - int(issued))
                    )
                    return (
                        path.path_command_count,
                        release_deviation,
                        predicted[path.path_id],
                        instance.cancellation_penalty * int(path.cancelled)
                        + path.terminal_delay,
                        path.path_id,
                    )

                ordered = sorted(paths, key=command_aware_key)
            else:
                ordered = sorted(paths, key=lambda path: (predicted[path.path_id], path.path_id))
            ranking[(scenario.scenario_id, train.train_id)] = [
                path.path_id for path in ordered
            ]
    return ranking


def safe_allowed_path_ids(
    ranking: Mapping[Tuple[str, str], Sequence[str]],
    pool: PathPool,
    width: Optional[int],
    backbone_ids: Optional[Iterable[str]] = None,
) -> Set[str]:
                                                                         

                                                                             
                                                                           
                                                                  
       

    allowed: Set[str] = set(backbone_ids or ())
    for key, ordered_ids in ranking.items():
        selected = list(ordered_ids if width is None else ordered_ids[:width])
                                                                             
                                                                           
                                                  
        for path in pool.paths(key[0], key[1]):
            if path.cancelled or path.near_term_signature == "COMPLETE":
                selected.append(path.path_id)
        allowed.update(selected)
    return allowed


def build_hdfs_fifo_backbone_schedules(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
) -> List[Dict[Tuple[str, str], str]]:
                                                                  

                                                                             
                                                                              
                                                                             
                                                                            
                                                                            
                                                              

                                                                             
                                                                          
                                                                             
                                                                            
                                                                         
       

    from ..validation import validate_selection

    active = [
        train.train_id
        for train in instance.trains
        if not context.train_states[train.train_id].completed
        and not context.train_states[train.train_id].already_crossed_disruption
    ]
    priority_order = sorted(
        active,
        key=lambda train_id: (
            -context.train_states[train_id].current_delay,
            instance.train(train_id).original_order,
        ),
    )
    cancellation_candidates = sorted(
        [
            train_id
            for train_id in active
            if context.train_states[train_id].cancel_allowed
        ],
        key=lambda train_id: instance.train(train_id).original_order,
    )

    def maximal_release_sequence(scenario: RecoveryScenario) -> Tuple[int, ...]:
                                                                            

                                                                            
                                                                             
                                                                        
                                                                       
           

        upper = max(
            path.release_time
            for train in instance.trains
            for path in pool.paths(scenario.scenario_id, train.train_id)
            if path.release_time is not None and not path.cancelled
        )
        section = instance.section(instance.disruption_section)
        events: Dict[int, Tuple[int, int, int]] = {}
        for release in range(scenario.t12, int(upper) + 1, instance.time_step):
            runtime = scenario.disrupted_runtime(release, section.normal_runtime)
            if runtime is None:
                continue
            events[release] = (
                release,
                release + runtime,
                scenario.disrupted_headway(release, section.normal_headway),
            )

        best_ending: Dict[int, Tuple[int, ...]] = {}
        for release in sorted(events):
            best = (release,)
            current = events[release]
            for previous_release in sorted(value for value in events if value < release):
                previous = events[previous_release]
                required = max(previous[2], current[2])
                if current[0] - previous[0] < required:
                    continue
                if current[1] - previous[1] < required:
                    continue
                candidate = best_ending[previous_release] + (release,)
                if len(candidate) > len(best) or (
                    len(candidate) == len(best) and candidate < best
                ):
                    best = candidate
            best_ending[release] = best
        return min(
            best_ending.values(),
            key=lambda sequence: (-len(sequence), sequence),
        )

    release_sequences = {
        scenario.scenario_id: maximal_release_sequence(scenario)
        for scenario in context.scenarios
    }

    for cancellation_count in range(instance.max_cancellations + 1):
        valid_schedules: List[Dict[Tuple[str, str], str]] = []
        for cancellation_tuple in combinations(
            cancellation_candidates, cancellation_count
        ):
            cancelled = set(cancellation_tuple)
            selected: Dict[Tuple[str, str], PathColumn] = {}
            construction_failed = False
            for scenario in context.scenarios:
                scenario_id = scenario.scenario_id
                                                                                   
                for train in instance.trains:
                    state = context.train_states[train.train_id]
                    group = pool.paths(scenario_id, train.train_id)
                    if state.completed:
                        selected[(scenario_id, train.train_id)] = next(
                            path
                            for path in group
                            if path.near_term_signature == "COMPLETE"
                        )
                    elif train.train_id in cancelled:
                        selected[(scenario_id, train.train_id)] = next(
                            path for path in group if path.cancelled
                        )

                release_slots = release_sequences[scenario_id]
                next_slot_position = 0
                for train_id in priority_order:
                    if train_id in cancelled:
                        continue
                    state = context.train_states[train_id]
                    by_release = {
                        int(path.release_time): path
                        for path in pool.paths(scenario_id, train_id)
                        if not path.cancelled
                        and path.hold_station == state.current_station
                        and path.release_time is not None
                        and instance.disruption_section in path.section_events
                    }
                    chosen: Optional[PathColumn] = None
                    while next_slot_position < len(release_slots):
                        release = release_slots[next_slot_position]
                        next_slot_position += 1
                        if release in by_release:
                            chosen = by_release[release]
                            break
                    if chosen is None:
                        construction_failed = True
                        break
                    selected[(scenario_id, train_id)] = chosen
                if construction_failed:
                    break

            if construction_failed:
                continue
            report = validate_selection(instance, context, selected, pool=pool)
            if report.valid:
                valid_schedules.append(
                    {key: path.path_id for key, path in selected.items()}
                )
        if valid_schedules:
            return valid_schedules
    raise RuntimeError(
        "Unable to construct an independently valid HDFS/FIFO safety backbone "
        "within the cancellation budget."
    )


def build_hdfs_fifo_backbone(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
) -> Set[str]:
                                                                            

    return {
        path_id
        for schedule in build_hdfs_fifo_backbone_schedules(
            instance, context, pool
        )
        for path_id in schedule.values()
    }


def build_constructive_backbone_schedule(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    maximum_cancellation_subsets: int = 128,
) -> Dict[Tuple[str, str], str]:
                                                                      

                                                                             
                                                                               
                                                                            
                                                                             
                                                                          
                                                                            
       

    from ..validation import validate_selection

    scenarios = tuple(context.scenarios)
    scenario_ids = tuple(item.scenario_id for item in scenarios)
    probability = {item.scenario_id: float(item.probability) for item in scenarios}
    active = [
        train.train_id
        for train in instance.trains
        if not context.train_states[train.train_id].completed
    ]
    priority_orders = [
        sorted(
            active,
            key=lambda train_id: (
                -context.train_states[train_id].current_delay,
                instance.train(train_id).original_order,
            ),
        ),
        sorted(
            active,
            key=lambda train_id: (
                context.train_states[train_id].available_from,
                instance.train(train_id).original_order,
            ),
        ),
    ]
    cancellable = [
        train_id
        for train_id in active
        if context.train_states[train_id].cancel_allowed
    ]
    cancellation_subsets: List[Tuple[str, ...]] = [()]
    for count in range(1, instance.max_cancellations + 1):
        values = list(combinations(cancellable, count))
        values.sort(
            key=lambda subset: -sum(
                instance.train(train_id).original_order for train_id in subset
            )
        )
        cancellation_subsets.extend(values)
        if len(cancellation_subsets) >= maximum_cancellation_subsets:
            cancellation_subsets = cancellation_subsets[:maximum_cancellation_subsets]
            break

    signature_paths: Dict[
        Tuple[str, str, str], List[PathColumn]
    ] = {}
    common_signatures: Dict[str, List[str]] = {}
    for train_id in active:
        by_scenario: Dict[str, Dict[str, List[PathColumn]]] = {}
        for scenario_id in scenario_ids:
            groups: Dict[str, List[PathColumn]] = {}
            for path in pool.paths(scenario_id, train_id):
                if path.cancelled:
                    continue
                groups.setdefault(path.near_term_signature, []).append(path)
            by_scenario[scenario_id] = groups
            for signature, paths in groups.items():
                issued = context.train_states[train_id].issued_release_time
                paths.sort(
                    key=lambda path: (
                        path.terminal_delay,
                        0
                        if issued is None or path.release_time is None
                        else abs(int(path.release_time) - int(issued)),
                        path.path_command_count,
                        int(path.release_time or 10**9),
                        path.path_id,
                    )
                )
                signature_paths[(scenario_id, train_id, signature)] = paths
        shared = set.intersection(
            *(set(by_scenario[scenario_id]) for scenario_id in scenario_ids)
        )

        def signature_score(signature: str) -> Tuple[float, int, str]:
            expected = 0.0
            latest = 0
            for scenario_id in scenario_ids:
                path = signature_paths[(scenario_id, train_id, signature)][0]
                expected += probability[scenario_id] * path.terminal_delay
                latest = max(latest, int(path.release_time or 10**9))
            return expected, latest, signature

        common_signatures[train_id] = sorted(shared, key=signature_score)

    def compatible(
        path: PathColumn,
        occupancy: Mapping[Tuple[str, int], int],
        section_events: Mapping[str, List[Tuple[int, int, int]]],
    ) -> bool:
        for station_name, minute in path.station_occupancy:
            if occupancy.get((station_name, minute), 0) + 1 > instance.station(
                station_name
            ).hold_capacity:
                return False
        for section_name, (departure, arrival, headway) in path.section_events.items():
            for other_departure, other_arrival, other_headway in section_events.get(
                section_name, ()
            ):
                required = max(int(headway), int(other_headway))
                if abs(int(departure) - int(other_departure)) < required:
                    return False
                if abs(int(arrival) - int(other_arrival)) < required:
                    return False
                if (int(departure) - int(other_departure)) * (
                    int(arrival) - int(other_arrival)
                ) < 0:
                    return False
        return True

    def reserve(
        path: PathColumn,
        occupancy: Dict[Tuple[str, int], int],
        section_events: Dict[str, List[Tuple[int, int, int]]],
    ) -> None:
        for slot in path.station_occupancy:
            occupancy[slot] = occupancy.get(slot, 0) + 1
        for section_name, event in path.section_events.items():
            section_events.setdefault(section_name, []).append(
                (int(event[0]), int(event[1]), int(event[2]))
            )

    for train_order in priority_orders:
        for cancellation_tuple in cancellation_subsets:
            cancelled = set(cancellation_tuple)
            selected: Dict[Tuple[str, str], PathColumn] = {}
            occupancy_by_scenario: Dict[
                str, Dict[Tuple[str, int], int]
            ] = {scenario_id: {} for scenario_id in scenario_ids}
            events_by_scenario: Dict[
                str, Dict[str, List[Tuple[int, int, int]]]
            ] = {scenario_id: {} for scenario_id in scenario_ids}
            failed = False

            for train_id in train_order:
                state = context.train_states[train_id]
                if state.completed:
                    for scenario_id in scenario_ids:
                        completed = next(
                            path
                            for path in pool.paths(scenario_id, train_id)
                            if path.near_term_signature == "COMPLETE"
                        )
                        selected[(scenario_id, train_id)] = completed
                    continue
                if train_id in cancelled:
                    for scenario_id in scenario_ids:
                        selected[(scenario_id, train_id)] = next(
                            path
                            for path in pool.paths(scenario_id, train_id)
                            if path.cancelled
                        )
                    continue

                chosen_paths: Optional[Dict[str, PathColumn]] = None
                for signature in common_signatures.get(train_id, ()): 
                    tentative: Dict[str, PathColumn] = {}
                    for scenario_id in scenario_ids:
                        chosen = next(
                            (
                                path
                                for path in signature_paths[
                                    (scenario_id, train_id, signature)
                                ]
                                if compatible(
                                    path,
                                    occupancy_by_scenario[scenario_id],
                                    events_by_scenario[scenario_id],
                                )
                            ),
                            None,
                        )
                        if chosen is None:
                            tentative = {}
                            break
                        tentative[scenario_id] = chosen
                    if tentative:
                        chosen_paths = tentative
                        break
                if chosen_paths is None:
                    failed = True
                    break
                for scenario_id, path in chosen_paths.items():
                    selected[(scenario_id, train_id)] = path
                    reserve(
                        path,
                        occupancy_by_scenario[scenario_id],
                        events_by_scenario[scenario_id],
                    )

            if failed:
                continue
            report = validate_selection(instance, context, selected, pool=pool)
            if report.valid:
                return {key: path.path_id for key, path in selected.items()}
    raise RuntimeError(
        "Unable to construct a heterogeneous-timetable safety backbone within "
        "the cancellation budget."
    )


def build_constructive_backbone(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
) -> Set[str]:
    return set(
        build_constructive_backbone_schedule(instance, context, pool).values()
    )


def deterministic_rank_candidate_paths(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    include_command_terms: bool = True,
) -> Dict[Tuple[str, str], List[str]]:
                                                                   

    ranking: Dict[Tuple[str, str], List[str]] = {}
    for scenario in context.scenarios:
        for train in instance.trains:
            paths = pool.paths(scenario.scenario_id, train.train_id)
            ordered = sorted(
                paths,
                key=lambda path: (
                    instance.cancellation_penalty * int(path.cancelled)
                    + path.terminal_delay,
                    path.path_workload if include_command_terms else 0,
                    path.release_time if path.release_time is not None else 10**9,
                    path.path_id,
                ),
            )
            ranking[(scenario.scenario_id, train.train_id)] = [
                path.path_id for path in ordered
            ]
    return ranking


def solve_safe_m3pc_rh(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    config: SafeM3PCConfig = SafeM3PCConfig(),
    epsilon: Optional[float] = None,
    time_limit: Optional[float] = None,
    pretrained_model: Optional[object] = None,
    use_learned_ranking: bool = True,
    service_reference: Optional[float] = None,
    cancellation_reference: Optional[int] = None,
    objective_mode: str = "full",
    required_path_ids: Optional[Iterable[str]] = None,
    stabilize_cancellations: bool = True,
):
                                                                              

                                                                             
                                                                            
                                                                        
                                                                 
       

                                                                         
                                                                          
    from ..milp import MilpBuildError, MilpSolveError, solve_lexicographic

    start = time.perf_counter()
    objective_mode = str(objective_mode).strip().lower()
    if objective_mode not in {"full", "service_only"}:
        raise ValueError("objective_mode must be 'full' or 'service_only'.")
    training_losses: List[float] = []
    model = pretrained_model
    ranking_start = time.perf_counter()
    if use_learned_ranking:
        if model is None:
            model, training_losses = train_masked_model(
                instance, context, pool, config
            )
            ranking_start = time.perf_counter()
        ranking = rank_candidate_paths(
            model,
            instance,
            context,
            pool,
            include_command_terms=(
                objective_mode == "full"
                or config.command_aware_ranking_for_service
            ),
            inference_batch_size=config.inference_batch_size,
        )
    else:
        ranking = deterministic_rank_candidate_paths(
            instance,
            context,
            pool,
            include_command_terms=(objective_mode == "full"),
        )
    ranking_seconds = time.perf_counter() - ranking_start
    backbone_start = time.perf_counter()
    backbone_error: Optional[str] = None
    constructive_backbone_error: Optional[str] = None
    backbone_ids: Set[str] = set()
    if config.use_constructive_backbone:
        try:
            backbone_ids.update(
                build_constructive_backbone(instance, context, pool)
            )
        except RuntimeError as error:
            constructive_backbone_error = str(error)
    if not config.use_hdfs_backbone:
        backbone_error = "disabled_for_heterogeneous_real_timetable"
    else:
        try:
            backbone_ids.update(build_hdfs_fifo_backbone(instance, context, pool))
        except RuntimeError as error:
                                                                              
                                                                              
                                                                           
                                                                             
            backbone_error = str(error)
    warm_start_ids = {str(path_id) for path_id in (required_path_ids or ())}
    unknown_warm_ids = warm_start_ids.difference(pool.by_id)
    if unknown_warm_ids:
        raise ValueError(
            "required_path_ids contain paths outside the supplied pool: %s"
            % sorted(unknown_warm_ids)[:3]
        )
    backbone_ids.update(warm_start_ids)
    backbone_seconds = time.perf_counter() - backbone_start

    attempts: List[Dict[str, object]] = []
    candidates: List[Tuple[Optional[int], object]] = []
    last_error: Optional[Exception] = None
    for width in config.candidate_widths:
                                                                         
                                                                            
                                                                             
        if width is None and candidates:
            continue
        elapsed = time.perf_counter() - start
        remaining = None if time_limit is None else max(0.05, time_limit - elapsed)
        if time_limit is not None and remaining <= 0.051:
            break
        allowed = safe_allowed_path_ids(
            ranking, pool, width, backbone_ids=backbone_ids
        )
        attempt_start = time.perf_counter()
        try:
            solution = solve_lexicographic(
                instance,
                context,
                pool,
                algorithm="C1-Safe-M3PC-RH",
                allowed_path_ids=allowed,
                enforce_nonanticipativity=True,
                epsilon=epsilon,
                service_reference=service_reference,
                cancellation_reference=cancellation_reference,
                time_limit=remaining,
                objective_mode=objective_mode,
                stabilize_cancellations=stabilize_cancellations,
            )
        except (MilpBuildError, MilpSolveError) as error:
            last_error = error
            attempts.append(
                {
                    "width": width,
                    "allowed_paths": len(allowed),
                    "status": "failed",
                    "seconds": time.perf_counter() - attempt_start,
                    "error": str(error),
                }
            )
            continue
        attempts.append(
            {
                "width": width,
                "allowed_paths": len(allowed),
                "status": solution.status,
                "seconds": time.perf_counter() - attempt_start,
            }
        )
        candidates.append((width, solution))
        if config.stop_after_first_feasible and width != 0:
            break

    if candidates:
        epsilon_value = context.epsilon if epsilon is None else float(epsilon)
        best_service_reference = min(
            solution.service_optimum for _width, solution in candidates
        )
        service_cap = (1.0 + epsilon_value) * best_service_reference + 1.0e-6
        eligible = [
            (width, solution)
            for width, solution in candidates
            if solution.objective.service_loss <= service_cap
        ]
                                                                         
                                                                           
        if objective_mode == "service_only":
            selection_key = lambda item: (
                item[1].objective.service_loss,
                len(item[1].cancelled_trains),
                item[1].runtime_seconds,
            )
        else:
            selection_key = lambda item: (
                item[1].objective.workload_total,
                item[1].objective.workload_peak,
                item[1].objective.service_loss,
                item[1].runtime_seconds,
            )
        selected_width, solution = min(eligible or candidates, key=selection_key)
        solution.runtime_seconds = time.perf_counter() - start
        solution.metadata.update(
            {
                "candidate_method": (
                    "masked_trajectory_transformer"
                    if use_learned_ranking
                    else "deterministic_ablation"
                ),
                "command_aware_ranking": (
                    objective_mode == "full"
                    or config.command_aware_ranking_for_service
                ),
                "candidate_width_used": selected_width,
                "candidate_widths_solved": [width for width, _item in candidates],
                "repair_attempts": attempts,
                "training_epochs": len(training_losses),
                "training_initial_loss": (
                    training_losses[0] if training_losses else None
                ),
                "training_final_loss": (
                    training_losses[-1] if training_losses else None
                ),
                "offline_training_sample_count": getattr(
                    model, "_hsr_training_sample_count", None
                ),
                "offline_training_pool_count": getattr(
                    model, "_hsr_training_pool_count", len(pool.by_id)
                ),
                "safety_fallback_full_pool": selected_width is None,
                "feasibility_backbone_path_count": len(backbone_ids),
                "feasibility_backbone_error": backbone_error,
                "constructive_backbone_enabled": config.use_constructive_backbone,
                "constructive_backbone_error": constructive_backbone_error,
                "required_service_path_count": len(warm_start_ids),
                "candidate_ranking_seconds": ranking_seconds,
                "feasibility_backbone_seconds": backbone_seconds,
                "selection_service_reference": best_service_reference,
                "selection_service_cap": service_cap,
                "stabilize_cancellations": bool(stabilize_cancellations),
            }
        )
        return solution

    raise RuntimeError(
        "Safe-M3PC-RH found no repairable candidate within the total budget: %s"
        % (last_error or "budget exhausted")
    )


__all__ = [
    "MaskedTrajectoryTransformer",
    "SafeM3PCConfig",
    "build_offline_trajectory_array",
    "build_constructive_backbone",
    "build_constructive_backbone_schedule",
    "build_hdfs_fifo_backbone",
    "build_hdfs_fifo_backbone_schedules",
    "deterministic_rank_candidate_paths",
    "rank_candidate_paths",
    "safe_allowed_path_ids",
    "solve_safe_m3pc_rh",
    "train_masked_model",
]
