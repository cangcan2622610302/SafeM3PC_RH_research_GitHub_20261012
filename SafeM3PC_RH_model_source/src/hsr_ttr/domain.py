                                                                  

                                                                           
                                                                           
                                                                            
                                                                            
   

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


@dataclass(frozen=True)
class Station:
                                     

                                                                          
                                                                          
                                                                              
       

    name: str
    index: int
    hold_capacity: int
    safe_holding: bool = True


@dataclass(frozen=True)
class Section:
                                                                      

    name: str
    origin: str
    destination: str
    normal_runtime: int
    normal_headway: int


@dataclass(frozen=True)
class RecoveryScenario:
                                        

                                                                            
                                                                             
                                                                
       

    scenario_id: str
    probability: float
    t12: int
    t23: int
    s2_speed_kmh: int
    s2_runtime: int
    s2_headway: int

    def state_at(self, minute: int) -> str:
        if minute < self.t12:
            return "S1"
        if minute < self.t23:
            return "S2"
        return "S3"

    def disrupted_runtime(self, entry_minute: int, normal_runtime: int) -> Optional[int]:
        state = self.state_at(entry_minute)
        if state == "S1":
            return None
        if state == "S2":
            return self.s2_runtime
        return normal_runtime

    def disrupted_headway(self, entry_minute: int, normal_headway: int) -> int:
        return self.s2_headway if self.state_at(entry_minute) == "S2" else normal_headway


@dataclass(frozen=True)
class TrainPlan:
                                                            

    train_id: str
    planned_arrival: Mapping[str, int]
    planned_departure: Mapping[str, int]
    planned_stops: Tuple[str, ...]
    original_order: int

    @property
    def planned_terminal_arrival(self) -> int:
                                                                            
                                                                            
                                                                    
        return int(next(reversed(self.planned_arrival.values())))


@dataclass(frozen=True)
class TrainState:
                                                                            

    train_id: str
    current_station: str
    available_from: int
    current_delay: int
    issued_hold_station: Optional[str]
    issued_release_time: Optional[int]
    cancel_allowed: bool
    already_crossed_disruption: bool = False
    completed: bool = False


@dataclass(frozen=True)
class RailwayInstance:
                                                                      

    stations: Tuple[Station, ...]
    sections: Tuple[Section, ...]
    trains: Tuple[TrainPlan, ...]
    disruption_section: str
    disruption_start: int
    time_step: int
    horizon_end: int
    minimum_dwell: int
    command_change_threshold: int
    cancellation_penalty: int
    max_cancellations: int
    passing_time: int = 0

    def station(self, name: str) -> Station:
        return next(item for item in self.stations if item.name == name)

    def section(self, name: str) -> Section:
        return next(item for item in self.sections if item.name == name)

    def train(self, train_id: str) -> TrainPlan:
        return next(item for item in self.trains if item.train_id == train_id)


@dataclass(frozen=True)
class RollingContext:
                                                              

    now: int
    freeze_until: int
    next_observation: int
    train_states: Mapping[str, TrainState]
    scenarios: Tuple[RecoveryScenario, ...]
    epsilon: float = 0.01
    random_seed: int = 20260826


@dataclass(frozen=True)
class CommandEvent:
                                                                  

                                                                        
                                                                             
                                              
       

    command_type: str
    issue_time: int
    weight: int


@dataclass(frozen=True)
class PathColumn:
                                                                  

                                                                        
                                                                           
                                                                            
                 
       

    path_id: str
    train_id: str
    scenario_id: str
    cancelled: bool
    hold_station: Optional[str]
    release_time: Optional[int]
    arrivals: Mapping[str, int]
    departures: Mapping[str, int]
    section_events: Mapping[str, Tuple[int, int, int]]
    station_occupancy: Tuple[Tuple[str, int], ...]
    terminal_arrival: Optional[int]
    terminal_delay: int
    command_events: Tuple[CommandEvent, ...]
    near_term_signature: str

    @property
    def path_workload(self) -> int:
                                                                        

        return sum(event.weight for event in self.command_events)

    @property
    def path_command_count(self) -> int:
        return self.path_workload


@dataclass(frozen=True)
class ObjectiveVector:
                                                    

                                                                            
                                                                            
                                                                
       

    service_loss: float
    hdfs_violations: float
    workload_total: float
    workload_peak: float

    @property
    def command_total(self) -> float:
        return float(self.workload_total)

    @property
    def command_peak(self) -> float:
        return float(self.workload_peak)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (
            float(self.service_loss),
            float(self.workload_total),
            float(self.workload_peak),
        )

    def lex_better_than(self, other: "ObjectiveVector", tolerance: float = 1e-6) -> bool:
        for left, right in zip(self.as_tuple(), other.as_tuple()):
            if left < right - tolerance:
                return True
            if left > right + tolerance:
                return False
        return False


@dataclass
class ScheduleSolution:
                                                       

    algorithm: str
    status: str
    selected_paths: Dict[Tuple[str, str], str]
    objective: ObjectiveVector
    cancelled_trains: Tuple[str, ...]
    service_optimum: float
    cancellation_bound: int
    stage_objectives: Dict[str, float]
    runtime_seconds: float
    mip_gap: Optional[float] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        selected = {
            "%s|%s" % (scenario, train): path
            for (scenario, train), path in sorted(self.selected_paths.items())
        }
        return {
            "algorithm": self.algorithm,
            "status": self.status,
            "selected_paths": selected,
            "objective": asdict(self.objective),
            "cancelled_trains": list(self.cancelled_trains),
            "service_optimum": self.service_optimum,
            "cancellation_bound": self.cancellation_bound,
            "stage_objectives": dict(self.stage_objectives),
            "runtime_seconds": self.runtime_seconds,
            "mip_gap": self.mip_gap,
            "metadata": dict(self.metadata),
        }


def scenario_map(scenarios: Iterable[RecoveryScenario]) -> Dict[str, RecoveryScenario]:
    return {scenario.scenario_id: scenario for scenario in scenarios}
