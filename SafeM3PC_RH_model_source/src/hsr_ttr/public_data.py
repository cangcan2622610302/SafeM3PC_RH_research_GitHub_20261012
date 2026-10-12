                                                                          

                                                                            
                                                                              
                                                                          
                                                                             
                                                                               
   

from __future__ import annotations

import csv
import io
import json
import math
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, TextIO, Tuple

from .domain import (
    RailwayInstance,
    RecoveryScenario,
    RollingContext,
    Section,
    Station,
    TrainPlan,
    TrainState,
)


CORRIDOR_FULL_NAMES: Tuple[str, ...] = (
    "Jinanxi Railway Station",
    "Taian Railway Station",
    "Qufudong Railway Station",
    "Tengzhoudong Railway Station",
    "Zaozhuang Railway Station",
    "Xuzhoudong Railway Station",
    "Suzhoudong Railway Station",
    "Bengbunan Railway Station",
    "Dingyuan Railway Station",
    "Chuzhou Railway Station",
    "Nanjingnan Railway Station",
)

CORRIDOR_CODES: Tuple[str, ...] = (
    "JW",
    "TA",
    "QFE",
    "TZE",
    "ZZ",
    "XZE",
    "SZE",
    "BBS",
    "DY",
    "CZ",
    "NJS",
)

FULL_TO_CODE = dict(zip(CORRIDOR_FULL_NAMES, CORRIDOR_CODES))
CODE_TO_FULL = dict(zip(CORRIDOR_CODES, CORRIDOR_FULL_NAMES))

                                                             
STATION_TRACKS: Mapping[str, int] = {
    "JW": 9,
    "TA": 3,
    "QFE": 3,
    "TZE": 2,
    "ZZ": 3,
    "XZE": 8,
    "SZE": 3,
    "BBS": 6,
    "DY": 2,
    "CZ": 3,
    "NJS": 6,
}

                                                              
SECTION_SPECS: Tuple[Tuple[str, str, str, int, int, int], ...] = (
    ("JT", "JW", "TA", 9, 14, 59),
    ("TQ", "TA", "QFE", 15, 23, 70),
    ("QT", "QFE", "TZE", 12, 18, 56),
    ("TZ", "TZE", "ZZ", 8, 12, 36),
    ("ZX", "ZZ", "XZE", 13, 21, 63),
    ("XS", "XZE", "SZE", 16, 26, 69),
    ("SB", "SZE", "BBS", 16, 25, 87),
    ("BD", "BBS", "DY", 11, 17, 53),
    ("DC", "DY", "CZ", 13, 20, 62),
    ("CN", "CZ", "NJS", 12, 19, 79),
)

SECTION_BY_CODE = {item[0]: item for item in SECTION_SPECS}
CUMULATIVE_DISTANCE = (0, 59, 129, 185, 221, 284, 353, 440, 493, 555, 634)
DISRUPTION_SECTION = "BBSDY"
DISRUPTION_START = 130                                                      
HORIZON_END = 720
TIME_STEP = 1


@dataclass(frozen=True)
class MacroScenarioSpec:
    macro_id: str
    date: str
    density_class: str


MACRO_SCENARIOS: Tuple[MacroScenarioSpec, ...] = (
    MacroScenarioSpec("M01", "2020-01-13", "low"),
    MacroScenarioSpec("M02", "2020-01-25", "low-medium"),
    MacroScenarioSpec("M03", "2019-10-09", "medium"),
    MacroScenarioSpec("M04", "2020-01-01", "medium-high"),
    MacroScenarioSpec("M05", "2019-12-30", "high"),
    MacroScenarioSpec("M06", "2020-01-10", "very-high"),
)


@dataclass(frozen=True)
class PublicMacroCase:
    spec: MacroScenarioSpec
    instance: RailwayInstance
    context: RollingContext
    planned_entries: Mapping[str, int]
    source_path: str

    @property
    def train_count(self) -> int:
        return len(self.instance.trains)


def _clock_minute(value: str) -> int:
    parsed = datetime.strptime(value, "%H:%M:%S")
    minute = parsed.hour * 60 + parsed.minute
    return minute + 24 * 60 if minute < 4 * 60 else minute


def _relative_minute(value: str) -> int:
    return _clock_minute(value) - 7 * 60


def _ceil_grid(value: float, step: int = TIME_STEP) -> int:
    return int(math.ceil(float(value) / step) * step)


def _default_candidates() -> Tuple[Path, ...]:
    project = Path(__file__).resolve().parents[2]
    workspace = project.parent
    return (
        project / "data" / "public_hsr" / "high-speed trains operation data.csv",
        workspace / "High-speed railway public dataset.zip",
    )


def resolve_public_data_source(source: Optional[Path] = None) -> Path:
    candidates = (Path(source),) if source is not None else _default_candidates()
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Public HSR data not found. Expected the extracted operation CSV under "
        "hsr_ttr_project/data/public_hsr or High-speed railway public dataset.zip "
        "in the workspace root."
    )


@contextmanager
def _open_operation_csv(source: Path) -> Iterator[TextIO]:
    if source.suffix.lower() == ".zip":
        archive = zipfile.ZipFile(source, "r")
        raw = archive.open("high-speed trains operation data.csv", "r")
        text = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")
        try:
            yield text
        finally:
            text.close()
            archive.close()
    else:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            yield handle


def _infer_bd_entry(records: Sequence[Mapping[str, str]]) -> Optional[float]:
                                                                 

    positions = {name: index for index, name in enumerate(CORRIDOR_FULL_NAMES)}
    located = [
        (positions[row["station_name"]], row)
        for row in records
        if row["station_name"] in positions
    ]
    left = [item for item in located if item[0] <= 7]
    right = [item for item in located if item[0] >= 8]
    if not left or not right:
        return None
    left_position, left_row = max(left, key=lambda item: item[0])
    right_position, right_row = min(right, key=lambda item: item[0])
    if left_position >= right_position:
        return None
    start = _relative_minute(left_row["scheduled_departure_time"])
    end = _relative_minute(right_row["scheduled_arrival_time"])
    if end <= start:
        end += 24 * 60
    span = CUMULATIVE_DISTANCE[right_position] - CUMULATIVE_DISTANCE[left_position]
    if span <= 0:
        return None
    fraction = (
        CUMULATIVE_DISTANCE[7] - CUMULATIVE_DISTANCE[left_position]
    ) / span
    return start + fraction * (end - start)


def _service_records(
    source: Path,
) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
    selected_dates = {item.date for item in MACRO_SCENARIOS}
    corridor = set(CORRIDOR_FULL_NAMES)
    grouped: Dict[str, Dict[str, List[Dict[str, str]]]] = {
        date: {} for date in selected_dates
    }
    with _open_operation_csv(source) as handle:
        for row in csv.DictReader(handle):
            date = row["date"]
            if (
                date not in selected_dates
                or row["train_direction"] != "down"
                or row["station_name"] not in corridor
            ):
                continue
            grouped[date].setdefault(row["train_number"], []).append(dict(row))
    return grouped


def _dwell_for_recorded_stops(recorded_codes: Iterable[str], station: str) -> int:
    return 3 if station in set(recorded_codes) else 1


def _consistent_plan(
    train_id: str,
    entry_bd: float,
    recorded_codes: Sequence[str],
    original_order: int,
) -> TrainPlan:
                                                                                 

    stops = tuple(
        code
        for code in CORRIDOR_CODES[1:-1]
        if code in set(recorded_codes)
    )
    arrivals: Dict[str, int] = {}
    departures: Dict[str, int] = {}
    departures["BBS"] = int(round(entry_bd))
    arrivals["BBS"] = departures["BBS"] - (
        3 if "BBS" in stops else 1
    )

                                            
    for section_index in range(6, -1, -1):
        _name, origin, destination, runtime_min, _runtime_max, _distance = SECTION_SPECS[section_index]
        departures[origin] = arrivals[destination] - runtime_min
        arrivals[origin] = departures[origin] - (3 if origin in stops else 1)
    arrivals["JW"] = departures["JW"]

                                              
    cursor = departures["BBS"]
    for section_index in range(7, len(SECTION_SPECS)):
        _name, origin, destination, runtime_min, _runtime_max, _distance = SECTION_SPECS[section_index]
        cursor += runtime_min
        arrivals[destination] = cursor
        if destination == "NJS":
            departures[destination] = cursor
        else:
            cursor += 3 if destination in stops else 1
            departures[destination] = cursor

    return TrainPlan(
        train_id=train_id,
        planned_arrival=arrivals,
        planned_departure=departures,
        planned_stops=stops,
        original_order=original_order,
    )


def prepare_public_cache(
    cache_path: Path,
    source: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, object]:
                                                                         

    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        with cache_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    resolved = resolve_public_data_source(source)
    grouped = _service_records(resolved)
    macros: List[Dict[str, object]] = []
    for spec in MACRO_SCENARIOS:
        candidates: List[Tuple[float, str, Sequence[Mapping[str, str]]]] = []
        for train_id, records in grouped[spec.date].items():
            records.sort(key=lambda row: int(row["station_order"]))
            entry = _infer_bd_entry(records)
            if entry is None or entry < DISRUPTION_START or entry > 480:
                continue
            candidates.append((entry, train_id, records))
        candidates.sort(key=lambda item: (item[0], item[1]))
        if not candidates:
            raise ValueError("No affected corridor trains found for %s" % spec.date)
        trains: List[Dict[str, object]] = []
        for order, (entry, train_id, records) in enumerate(candidates, start=1):
            recorded_codes = [
                FULL_TO_CODE[row["station_name"]]
                for row in records
                if row["station_name"] in FULL_TO_CODE
            ]
            plan = _consistent_plan(train_id, entry, recorded_codes, order)
            trains.append(
                {
                    "train_id": train_id,
                    "planned_entry_bd": float(entry),
                    "recorded_station_codes": recorded_codes,
                    "planned_arrival": dict(plan.planned_arrival),
                    "planned_departure": dict(plan.planned_departure),
                    "planned_stops": list(plan.planned_stops),
                    "original_order": order,
                }
            )
        macros.append(
            {
                "macro_id": spec.macro_id,
                "date": spec.date,
                "density_class": spec.density_class,
                "train_count": len(trains),
                "trains": trains,
            }
        )
    payload: Dict[str, object] = {
        "schema_version": 1,
        "source_path": str(resolved),
        "source_member": "high-speed trains operation data.csv",
        "corridor": list(CORRIDOR_CODES),
        "corridor_full_names": list(CORRIDOR_FULL_NAMES),
        "disruption_section": DISRUPTION_SECTION,
        "disruption_start": DISRUPTION_START,
        "selection_rule": (
            "all down-direction trains with an interpolated planned BD entry "
            "between 09:10 and 15:00"
        ),
        "macros": macros,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return payload


def build_recovery_tree(prefix: str = "") -> Tuple[RecoveryScenario, ...]:
                                                                      

    tag = (prefix + "-") if prefix else ""
    return (
        RecoveryScenario(
            scenario_id=tag + "F",
            probability=0.25,
            t12=DISRUPTION_START + 90,
            t23=DISRUPTION_START + 90 + 30,
            s2_speed_kmh=200,
            s2_runtime=16,
            s2_headway=5,
        ),
        RecoveryScenario(
            scenario_id=tag + "M",
            probability=0.50,
            t12=DISRUPTION_START + 120,
            t23=DISRUPTION_START + 120 + 45,
            s2_speed_kmh=160,
            s2_runtime=20,
            s2_headway=6,
        ),
        RecoveryScenario(
            scenario_id=tag + "S",
            probability=0.25,
            t12=DISRUPTION_START + 150,
            t23=DISRUPTION_START + 150 + 60,
            s2_speed_kmh=120,
            s2_runtime=27,
            s2_headway=8,
        ),
    )


def _instance_from_macro(macro: Mapping[str, object]) -> Tuple[RailwayInstance, Dict[str, int]]:
    stations = tuple(
        Station(
            name=code,
            index=index,
            hold_capacity=int(STATION_TRACKS[code]),
            safe_holding=True,
        )
        for index, code in enumerate(CORRIDOR_CODES)
    )
    sections = tuple(
        Section(
            name=origin + destination,
            origin=origin,
            destination=destination,
            normal_runtime=runtime_min,
            normal_headway=3,
        )
        for name, origin, destination, runtime_min, _runtime_max, _distance in SECTION_SPECS
    )
    trains: List[TrainPlan] = []
    entries: Dict[str, int] = {}
    for item in macro["trains"]:                       
        train = TrainPlan(
            train_id=str(item["train_id"]),
            planned_arrival={str(k): int(v) for k, v in item["planned_arrival"].items()},
            planned_departure={str(k): int(v) for k, v in item["planned_departure"].items()},
            planned_stops=tuple(str(value) for value in item["planned_stops"]),
            original_order=int(item["original_order"]),
        )
        trains.append(train)
        entries[train.train_id] = int(round(float(item["planned_entry_bd"])))
    train_count = len(trains)
    return (
        RailwayInstance(
            stations=stations,
            sections=sections,
            trains=tuple(trains),
            disruption_section=DISRUPTION_SECTION,
            disruption_start=DISRUPTION_START,
            time_step=TIME_STEP,
            horizon_end=HORIZON_END,
            minimum_dwell=3,
            command_change_threshold=TIME_STEP,
            cancellation_penalty=480,
            max_cancellations=max(1, min(train_count - 1, int(math.floor(0.15 * train_count)))),
            passing_time=1,
        ),
        entries,
    )


def _states_for_macro(
    instance: RailwayInstance,
    planned_entries: Mapping[str, int],
    now: int = DISRUPTION_START,
) -> Dict[str, TrainState]:
    station_names = tuple(station.name for station in instance.stations)
    disruption_index = station_names.index("BBS")
    ordered = sorted(instance.trains, key=lambda train: train.original_order)
    cancellable_start = max(1, int(math.floor(0.70 * len(ordered))))
    cancellable_ids = {train.train_id for train in ordered[cancellable_start:]}
    states: Dict[str, TrainState] = {}
    for queue_index, train in enumerate(ordered):
        candidate_indices = [
            index
            for index, station in enumerate(station_names[: disruption_index + 1])
            if int(train.planned_departure[station]) >= now
        ]
        current_index = candidate_indices[0] if candidate_indices else disruption_index
        current_station = station_names[current_index]
        available_from = max(now, int(train.planned_arrival[current_station]))
        provisional_release = _ceil_grid(
            max(
                DISRUPTION_START + 120 + queue_index * 3,
                planned_entries[train.train_id],
            )
        )
        states[train.train_id] = TrainState(
            train_id=train.train_id,
            current_station=current_station,
            available_from=available_from,
            current_delay=max(0, now - int(train.planned_departure[current_station])),
            issued_hold_station=current_station,
            issued_release_time=provisional_release,
            cancel_allowed=train.train_id in cancellable_ids,
            already_crossed_disruption=False,
            completed=False,
        )
    return states


def build_public_macro_cases(
    cache_path: Optional[Path] = None,
    source: Optional[Path] = None,
    force_prepare: bool = False,
    epsilon: float = 0.0,
) -> Tuple[PublicMacroCase, ...]:
    project = Path(__file__).resolve().parents[2]
    cache = (
        project / "data" / "processed" / "beijing_shanghai_6macros.json"
        if cache_path is None
        else Path(cache_path)
    )
    payload = prepare_public_cache(cache, source=source, force=force_prepare)
    spec_by_id = {item.macro_id: item for item in MACRO_SCENARIOS}
    result: List[PublicMacroCase] = []
    for macro in payload["macros"]:                       
        spec = spec_by_id[str(macro["macro_id"])]
        instance, entries = _instance_from_macro(macro)
        scenarios = build_recovery_tree(spec.macro_id)
        context = RollingContext(
            now=DISRUPTION_START,
            freeze_until=DISRUPTION_START + 6,
            next_observation=DISRUPTION_START + 30,
            train_states=_states_for_macro(instance, entries),
            scenarios=scenarios,
            epsilon=float(epsilon),
            random_seed=20260904 + int(spec.macro_id[1:]),
        )
        result.append(
            PublicMacroCase(
                spec=spec,
                instance=instance,
                context=context,
                planned_entries=entries,
                source_path=str(payload["source_path"]),
            )
        )
    return tuple(result)


def scenario_catalog_rows(cases: Sequence[PublicMacroCase]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    recovery_labels = {"F": "fast", "M": "medium", "S": "slow"}
    for case in cases:
        for scenario in case.context.scenarios:
            recovery_code = scenario.scenario_id.rsplit("-", 1)[-1]
            rows.append(
                {
                    "scenario_id": "%d-SRR-BD-%s-%s"
                    % (case.train_count, case.spec.macro_id, recovery_code),
                    "macro_id": case.spec.macro_id,
                    "data_date": case.spec.date,
                    "density_class": case.spec.density_class,
                    "train_count": case.train_count,
                    "network": "Beijing-Shanghai HSR: Jinan West-Nanjing South",
                    "disrupted_section": "Bengbu South-Dingyuan (BD)",
                    "disruption_start_min": DISRUPTION_START,
                    "recovery_type": recovery_labels[recovery_code],
                    "s1_duration_min": scenario.t12 - DISRUPTION_START,
                    "s2_duration_min": scenario.t23 - scenario.t12,
                    "s2_speed_kmh": scenario.s2_speed_kmh,
                    "s2_runtime_min": scenario.s2_runtime,
                    "s2_headway_min": scenario.s2_headway,
                    "prior_probability": scenario.probability,
                }
            )
    return rows


__all__ = [
    "CODE_TO_FULL",
    "CORRIDOR_CODES",
    "CORRIDOR_FULL_NAMES",
    "DISRUPTION_SECTION",
    "DISRUPTION_START",
    "MACRO_SCENARIOS",
    "PublicMacroCase",
    "SECTION_SPECS",
    "STATION_TRACKS",
    "build_public_macro_cases",
    "build_recovery_tree",
    "prepare_public_cache",
    "resolve_public_data_source",
    "scenario_catalog_rows",
]
