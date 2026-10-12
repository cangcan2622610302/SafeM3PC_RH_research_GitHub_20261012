                                                                    

                                                                             
                                                                           
                                                                              
                                                                             
                                                                             

                                                                          
                                                                             
                                                                              
         
   

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..domain import RailwayInstance, RollingContext, ScheduleSolution
from ..paths import PathPool

try:
    import torch
    from torch import Tensor, nn
except ImportError:                    
    torch = None
    Tensor = object                
    nn = None                


@dataclass(frozen=True)
class GraphLNSConfig:
    embedding_dim: int = 32
    attention_heads: int = 4
    transformer_layers: int = 2
    training_epochs: int = 60
    learning_rate: float = 3.0e-3
    seed_fraction: float = 0.20
    temporal_neighbourhood_min: int = 15
    random_seed: int = 20260826


if nn is not None:

    class GraphTransformerScorer(nn.Module):
                                                                           

        def __init__(
            self,
            feature_dim: int = 8,
            embedding_dim: int = 32,
            heads: int = 4,
            layers: int = 2,
        ) -> None:
            super().__init__()
            self.feature_projection = nn.Linear(feature_dim, embedding_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=embedding_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
            self.score_head = nn.Linear(embedding_dim, 1)

        def forward(self, features: Tensor, adjacency: Tensor) -> Tensor:
                                                                          

            identity = torch.eye(adjacency.shape[-1], device=adjacency.device)
            graph = adjacency + identity
            degree = torch.clamp(graph.sum(dim=-1, keepdim=True), min=1.0)
            normalized = graph / degree
            embedded = self.feature_projection(features)
            aggregated = embedded + torch.matmul(normalized, embedded)
            return self.score_head(self.encoder(aggregated)).squeeze(-1)


def _selected_path_objects(
    pool: PathPool,
    solution: ScheduleSolution,
) -> Dict[Tuple[str, str], object]:
    return {
        key: pool.by_id[path_id]
        for key, path_id in solution.selected_paths.items()
    }


def build_train_interaction_graph(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    incumbent: ScheduleSolution,
    temporal_neighbourhood_min: int = 15,
) -> Tuple[List[str], np.ndarray, np.ndarray]:
                                                                  

                                                                            
                                                                          
                                                                        
       

    train_ids = [train.train_id for train in instance.trains]
    index = {train_id: position for position, train_id in enumerate(train_ids)}
    adjacency = np.zeros((len(train_ids), len(train_ids)), dtype=np.float32)
    features = np.zeros((len(train_ids), 8), dtype=np.float32)
    selected = _selected_path_objects(pool, incumbent)

                                                       
    scenario_probability = {item.scenario_id: item.probability for item in context.scenarios}
    for train_id in train_ids:
        position = index[train_id]
        state = context.train_states[train_id]
        expected_delay = 0.0
        expected_work = 0.0
        cancelled_probability = 0.0
        release_values: List[int] = []
        for scenario in context.scenarios:
            path = selected[(scenario.scenario_id, train_id)]
            probability = scenario_probability[scenario.scenario_id]
            expected_delay += probability * path.terminal_delay
            expected_work += probability * path.path_workload
            cancelled_probability += probability * float(path.cancelled)
            if path.release_time is not None:
                release_values.append(path.release_time)
        features[position, 0] = min(state.current_delay, 240) / 240.0
        features[position, 1] = min(expected_delay, 480) / 480.0
        features[position, 2] = expected_work / 4.0
        features[position, 3] = cancelled_probability
        features[position, 4] = float(state.cancel_allowed)
        features[position, 5] = float(state.completed)
        if release_values:
            features[position, 6] = (np.mean(release_values) - context.now) / 240.0

    for scenario in context.scenarios:
        scenario_id = scenario.scenario_id
        for left_pos, left_train in enumerate(train_ids):
            left = selected[(scenario_id, left_train)]
            if left.cancelled:
                continue
            left_occupancy = set(left.station_occupancy)
            for right_pos in range(left_pos + 1, len(train_ids)):
                right_train = train_ids[right_pos]
                right = selected[(scenario_id, right_train)]
                if right.cancelled:
                    continue
                related = bool(left_occupancy.intersection(right.station_occupancy))
                if not related:
                    common_sections = set(left.section_events).intersection(right.section_events)
                    for section_name in common_sections:
                        left_event = left.section_events[section_name]
                        right_event = right.section_events[section_name]
                        if (
                            abs(left_event[0] - right_event[0]) <= temporal_neighbourhood_min
                            or abs(left_event[1] - right_event[1]) <= temporal_neighbourhood_min
                        ):
                            related = True
                            break
                if related:
                    adjacency[left_pos, right_pos] = 1.0
                    adjacency[right_pos, left_pos] = 1.0

                                                                   
    if len(train_ids):
        features[:, 7] = adjacency.sum(axis=1) / max(len(train_ids) - 1, 1)
    return train_ids, adjacency, features


def train_graph_scorer(
    features: np.ndarray,
    adjacency: np.ndarray,
    config: GraphLNSConfig = GraphLNSConfig(),
) -> Tuple[object, List[float]]:
                                                                            

                                                                           
                                                                             
                                                                        
       

    if torch is None or nn is None:
        raise RuntimeError("PyTorch is required for graph-Transformer LNS.")
    torch.manual_seed(config.random_seed)
    model = GraphTransformerScorer(
        feature_dim=features.shape[1],
        embedding_dim=config.embedding_dim,
        heads=config.attention_heads,
        layers=config.transformer_layers,
    )
    x = torch.tensor(features[None, :, :], dtype=torch.float32)
    a = torch.tensor(adjacency[None, :, :], dtype=torch.float32)
                                                                               
    target_np = 4.0 * features[:, 1] + features[:, 2] + 2.0 * features[:, 7]
    if float(np.ptp(target_np)) > 1e-9:
        target_np = (target_np - target_np.min()) / np.ptp(target_np)
    target = torch.tensor(target_np[None, :], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    losses: List[float] = []
    model.train()
    for _epoch in range(config.training_epochs):
        prediction = model(x, a)
        loss = torch.mean((prediction - target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, losses


def score_graph_nodes(
    model: object,
    features: np.ndarray,
    adjacency: np.ndarray,
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("PyTorch is required for graph-Transformer LNS.")
    model.eval()
    with torch.no_grad():
        return (
            model(
                torch.tensor(features[None, :, :], dtype=torch.float32),
                torch.tensor(adjacency[None, :, :], dtype=torch.float32),
            )[0]
            .cpu()
            .numpy()
        )


def choose_two_layer_neighbourhoods(
    train_ids: Sequence[str],
    adjacency: np.ndarray,
    scores: np.ndarray,
    seed_fraction: float = 0.20,
) -> Tuple[Set[str], Set[str]]:
                                                                    

    active = [position for position, _train in enumerate(train_ids)]
    seed_count = max(1, int(np.ceil(len(active) * seed_fraction)))
    seeds = set(np.argsort(-scores)[:seed_count].tolist())
    one_hop = set(seeds)
    for seed in seeds:
        one_hop.update(np.flatnonzero(adjacency[seed] > 0).tolist())
    two_hop = set(one_hop)
    for node in list(one_hop):
        two_hop.update(np.flatnonzero(adjacency[node] > 0).tolist())
    return (
        {train_ids[position] for position in one_hop},
        {train_ids[position] for position in two_hop},
    )


def lns_restrictions(
    pool: PathPool,
    incumbent: ScheduleSolution,
    unlocked_trains: Set[str],
) -> Tuple[Set[str], Dict[Tuple[str, str], str]]:
                                                                          

    allowed: Set[str] = set()
    fixed: Dict[Tuple[str, str], str] = {}
    for key, incumbent_path_id in incumbent.selected_paths.items():
        scenario_id, train_id = key
        if train_id in unlocked_trains:
            allowed.update(path.path_id for path in pool.paths(scenario_id, train_id))
        else:
            allowed.add(incumbent_path_id)
            fixed[key] = incumbent_path_id
    return allowed, fixed


def solve_graph_tlns(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    config: GraphLNSConfig = GraphLNSConfig(),
    epsilon: Optional[float] = None,
    time_limit: Optional[float] = None,
    initial_solution: Optional[ScheduleSolution] = None,
    service_reference: Optional[float] = None,
    cancellation_reference: Optional[int] = None,
) -> ScheduleSolution:
                                                                       

                                                                        
                                                                              
                                                                            
                                                                             
                                                                 
       

    from ..milp import MilpBuildError, MilpSolveError, solve_lexicographic
    from .safe_m3pc_rh import (
        build_hdfs_fifo_backbone,
        deterministic_rank_candidate_paths,
        safe_allowed_path_ids,
    )

    start = time.perf_counter()

    def remaining() -> Optional[float]:
        if time_limit is None:
            return None
        return max(0.05, time_limit - (time.perf_counter() - start))

    incumbent = initial_solution
    construction_attempts: List[Dict[str, object]] = []
    if incumbent is None:
        ranking = deterministic_rank_candidate_paths(instance, context, pool)
        backbone_ids = build_hdfs_fifo_backbone(instance, context, pool)
        last_error: Optional[Exception] = None
                                                                        
                                                               
        for width in (8, 16, 32, None):
            if time_limit is not None and remaining() <= 0.051:
                break
            allowed = safe_allowed_path_ids(
                ranking, pool, width, backbone_ids=backbone_ids
            )
            attempt_start = time.perf_counter()
            try:
                incumbent = solve_lexicographic(
                    instance,
                    context,
                    pool,
                    algorithm="C2-GT-TLNS",
                    allowed_path_ids=allowed,
                    enforce_nonanticipativity=True,
                    epsilon=epsilon,
                    service_reference=service_reference,
                    cancellation_reference=cancellation_reference,
                    time_limit=remaining(),
                )
            except (MilpBuildError, MilpSolveError) as error:
                last_error = error
                construction_attempts.append(
                    {
                        "width": width,
                        "status": "failed",
                        "seconds": time.perf_counter() - attempt_start,
                        "error": str(error),
                    }
                )
                continue
            construction_attempts.append(
                {
                    "width": width,
                    "status": incumbent.status,
                    "seconds": time.perf_counter() - attempt_start,
                }
            )
            break
        if incumbent is None:
            raise RuntimeError(
                "C2 could not construct a feasible incumbent: %s"
                % (last_error or "budget exhausted")
            )

    train_ids, adjacency, features = build_train_interaction_graph(
        instance,
        context,
        pool,
        incumbent,
        temporal_neighbourhood_min=config.temporal_neighbourhood_min,
    )
    graph_training_start = time.perf_counter()
    model, losses = train_graph_scorer(features, adjacency, config)
    graph_training_seconds = time.perf_counter() - graph_training_start
    scores = score_graph_nodes(model, features, adjacency)
    layer_one, layer_two = choose_two_layer_neighbourhoods(
        train_ids,
        adjacency,
        scores,
        seed_fraction=config.seed_fraction,
    )

    lns_attempts: List[Dict[str, object]] = []
    best = incumbent
    for layer_name, unlocked in (("one_hop", layer_one), ("two_hop", layer_two)):
        if time_limit is not None and remaining() <= 0.051:
            break
        allowed, fixed = lns_restrictions(pool, best, unlocked)
        attempt_start = time.perf_counter()
        try:
            candidate = solve_lexicographic(
                instance,
                context,
                pool,
                algorithm="C2-GT-TLNS",
                allowed_path_ids=allowed,
                fixed_selected_paths=fixed,
                enforce_nonanticipativity=True,
                epsilon=epsilon,
                service_reference=service_reference,
                cancellation_reference=cancellation_reference,
                time_limit=remaining(),
            )
        except (MilpBuildError, MilpSolveError) as error:
            lns_attempts.append(
                {
                    "layer": layer_name,
                    "unlocked_trains": sorted(unlocked),
                    "status": "failed",
                    "seconds": time.perf_counter() - attempt_start,
                    "error": str(error),
                }
            )
            continue
        epsilon_value = context.epsilon if epsilon is None else float(epsilon)
        common_reference = min(best.service_optimum, candidate.service_optimum)
        common_cap = (1.0 + epsilon_value) * common_reference + 1.0e-6
        both_service_admissible = (
            best.objective.service_loss <= common_cap
            and candidate.objective.service_loss <= common_cap
        )
        if both_service_admissible:
                                                                          
                                                                             
                                                                             
            improved = (
                candidate.objective.workload_total,
                candidate.objective.workload_peak,
                candidate.objective.service_loss,
            ) < (
                best.objective.workload_total,
                best.objective.workload_peak,
                best.objective.service_loss,
            )
        else:
            improved = candidate.objective.lex_better_than(best.objective)
        lns_attempts.append(
            {
                "layer": layer_name,
                "unlocked_trains": sorted(unlocked),
                "status": candidate.status,
                "improved": improved,
                "objective": candidate.objective.as_tuple(),
                "common_service_cap": common_cap,
                "seconds": time.perf_counter() - attempt_start,
            }
        )
        if improved:
            best = candidate

    best.algorithm = "C2-GT-TLNS"
    best.runtime_seconds = time.perf_counter() - start
    best.metadata.update(
        {
            "initial_construction_attempts": construction_attempts,
            "graph_node_count": len(train_ids),
            "graph_edge_count": int(adjacency.sum() / 2),
            "graph_training_epochs": len(losses),
            "graph_training_seconds": graph_training_seconds,
            "graph_training_initial_loss": losses[0] if losses else None,
            "graph_training_final_loss": losses[-1] if losses else None,
            "layer_one_trains": sorted(layer_one),
            "layer_two_trains": sorted(layer_two),
            "lns_attempts": lns_attempts,
        }
    )
    return best


__all__ = [
    "GraphLNSConfig",
    "GraphTransformerScorer",
    "build_train_interaction_graph",
    "choose_two_layer_neighbourhoods",
    "lns_restrictions",
    "score_graph_nodes",
    "solve_graph_tlns",
    "train_graph_scorer",
]
