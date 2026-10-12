                                                                      

                                                                              
                                                                            
                                                                            
                                                                           
                                                                      
   

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from hsr_ttr.algorithms.baselines import solve_b0, solve_b1
from hsr_ttr.algorithms.commercial import solve_gurobi_exact
from hsr_ttr.algorithms.graph_lns import GraphLNSConfig, solve_graph_tlns
from hsr_ttr.algorithms.safe_m3pc_rh import (
    SafeM3PCConfig,
    solve_safe_m3pc_rh,
    train_masked_model,
)
from hsr_ttr.data import build_scalable_instance, build_scalable_uncertain_context
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.validation import validate_solution


FIELDS = [
    "case",
    "station_count",
    "train_count",
    "epsilon",
    "algorithm",
    "status",
    "independently_valid",
    "validation_issue_count",
    "service_loss",
    "command_total",
    "command_peak",
    "cancelled_count",
    "online_runtime_seconds",
    "offline_training_seconds",
    "mip_gap",
    "path_pool_count",
    "path_variable_count",
    "constraint_count",
    "solver_backend",
    "relative_gurobi_gap_pct",
    "failure_type",
    "message",
]


def _failure_type(exc: Exception) -> str:
    text = str(exc).lower()
    if "model too large" in text or "size-limited license" in text:
        return "license_size_limit"
    if "time limit" in text or "budget" in text:
        return "time_limit_no_incumbent"
    if "infeasible" in text:
        return "infeasible_or_restricted_domain"
    return "solver_error"


def _row(
    case: str,
    stations: int,
    trains: int,
    epsilon: float,
    solution,
    validation,
    pool_count: int,
    offline_seconds: float = 0.0,
) -> Dict[str, object]:
    return {
        "case": case,
        "station_count": stations,
        "train_count": trains,
        "epsilon": epsilon,
        "algorithm": solution.algorithm,
        "status": solution.status,
        "independently_valid": validation.valid,
        "validation_issue_count": len(validation.issues),
        "service_loss": solution.objective.service_loss,
        "command_total": solution.objective.command_total,
        "command_peak": solution.objective.command_peak,
        "cancelled_count": len(solution.cancelled_trains),
        "online_runtime_seconds": solution.runtime_seconds,
        "offline_training_seconds": offline_seconds,
        "mip_gap": solution.mip_gap,
        "path_pool_count": pool_count,
        "path_variable_count": solution.metadata.get("path_variable_count"),
        "constraint_count": solution.metadata.get("constraint_count"),
        "solver_backend": solution.metadata.get("solver_backend", "highs"),
        "relative_gurobi_gap_pct": "",
        "failure_type": "",
        "message": "",
    }


def _failure_row(
    case: str,
    stations: int,
    trains: int,
    epsilon: float,
    algorithm: str,
    pool_count: int,
    started: float,
    exc: Exception,
) -> Dict[str, object]:
    return {
        "case": case,
        "station_count": stations,
        "train_count": trains,
        "epsilon": epsilon,
        "algorithm": algorithm,
        "status": "not_solved",
        "independently_valid": False,
        "validation_issue_count": "",
        "service_loss": "",
        "command_total": "",
        "command_peak": "",
        "cancelled_count": "",
        "online_runtime_seconds": time.perf_counter() - started,
        "offline_training_seconds": 0.0,
        "mip_gap": "",
        "path_pool_count": pool_count,
        "path_variable_count": "",
        "constraint_count": "",
        "solver_backend": "gurobi" if algorithm == "Gurobi-Exact" else "highs",
        "relative_gurobi_gap_pct": "",
        "failure_type": _failure_type(exc),
        "message": "%s: %s" % (type(exc).__name__, exc),
    }


def _write(output_dir: Path, rows: Sequence[Mapping[str, object]], manifest: Mapping[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "five_method_scale_comparison.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "five_method_scale_comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"manifest": manifest, "rows": list(rows)}, handle, ensure_ascii=False, indent=2)


def _load_existing(output_dir: Path) -> List[Dict[str, object]]:
    path = output_dir / "five_method_scale_comparison.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def run(
    output_dir: Path,
    station_count: int,
    train_counts: Sequence[int],
    epsilon: float,
    time_limit: float,
    phase1_time_limit: float,
    c1_epochs: int,
) -> None:
    rows: List[Dict[str, object]] = _load_existing(output_dir)
    completed = {
        (str(row["case"]), str(row["algorithm"]))
        for row in rows
    }
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "station_count": station_count,
        "train_counts": list(train_counts),
        "epsilon": epsilon,
        "common_online_time_limit_seconds": time_limit,
        "c1_phase1_time_limit_seconds": phase1_time_limit,
        "c1_epochs": c1_epochs,
        "scope_note": "B0 is a MAP deterministic baseline; B1/C1/C2/Gurobi use the same three-branch non-anticipative model.",
    }

    for trains in train_counts:
        case_name = "%ds%dt-uncertain" % (station_count, trains)
        instance = build_scalable_instance(station_count, trains)
        context = build_scalable_uncertain_context(instance, epsilon=epsilon)
        pool = generate_path_pool(instance, context)
        print("\n[%s] paths=%d" % (case_name, len(pool.by_id)), flush=True)

                                       
        for label, solver in (("B0", solve_b0), ("B1", solve_b1), ("Gurobi-Exact", solve_gurobi_exact)):
            if (case_name, label) in completed:
                print("  %-20s reused" % label, flush=True)
                continue
            started = time.perf_counter()
            try:
                solution = solver(
                    instance,
                    context,
                    pool,
                    epsilon=epsilon,
                    time_limit=time_limit,
                )
                validation = validate_solution(instance, context, pool, solution)
                validation.raise_if_invalid()
                row = _row(case_name, station_count, trains, epsilon, solution, validation, len(pool.by_id))
                print(
                    "  %-20s %-14s service=%8.2f cmd=%7.2f/%6.2f time=%6.2fs"
                    % (label, solution.status, solution.objective.service_loss, solution.objective.command_total, solution.objective.command_peak, solution.runtime_seconds),
                    flush=True,
                )
            except Exception as exc:
                row = _failure_row(case_name, station_count, trains, epsilon, label, len(pool.by_id), started, exc)
                print("  %-20s not_solved: %s" % (label, row["failure_type"]), flush=True)
            rows.append(row)
            completed.add((case_name, label))
            _write(output_dir, rows, manifest)

                                                                            
                                                                  
        config = SafeM3PCConfig(
            embedding_dim=24,
            attention_heads=4,
            transformer_layers=1,
            training_epochs=c1_epochs,
            batch_size=512,
            max_training_paths=4000,
            candidate_widths=(0, max(32, 3 * trains)),
            use_hdfs_backbone=True,
            use_constructive_backbone=True,
            stop_after_first_feasible=True,
            command_aware_ranking_for_service=True,
            random_seed=context.random_seed,
        )
        training_started = time.perf_counter()
        model, _losses = train_masked_model(instance, context, pool, config)
        offline_seconds = time.perf_counter() - training_started
        c1_phase1 = None
        c1 = None
        started = time.perf_counter()
        try:
            c1_phase1 = solve_safe_m3pc_rh(
                instance,
                context,
                pool,
                config=config,
                epsilon=0.0,
                time_limit=phase1_time_limit,
                pretrained_model=model,
                objective_mode="service_only",
                stabilize_cancellations=False,
            )
            c1_remaining = max(1.0, time_limit - c1_phase1.runtime_seconds)
            c1 = solve_safe_m3pc_rh(
                instance,
                context,
                pool,
                config=config,
                epsilon=epsilon,
                time_limit=c1_remaining,
                pretrained_model=model,
                service_reference=c1_phase1.service_optimum,
                cancellation_reference=c1_phase1.cancellation_bound,
                required_path_ids=c1_phase1.selected_paths.values(),
                objective_mode="full",
            )
            validation = validate_solution(instance, context, pool, c1)
            validation.raise_if_invalid()
            row = _row(case_name, station_count, trains, epsilon, c1, validation, len(pool.by_id), offline_seconds)
            row["online_runtime_seconds"] = c1_phase1.runtime_seconds + c1.runtime_seconds
            print(
                "  %-20s %-14s service=%8.2f cmd=%7.2f/%6.2f time=%6.2fs"
                % (c1.algorithm, c1.status, c1.objective.service_loss, c1.objective.command_total, c1.objective.command_peak, c1.runtime_seconds),
                flush=True,
            )
        except Exception as exc:
            row = _failure_row(case_name, station_count, trains, epsilon, "C1-Safe-M3PC-RH", len(pool.by_id), started, exc)
        if (case_name, "C1-Safe-M3PC-RH") not in completed:
            rows.append(row)
            completed.add((case_name, "C1-Safe-M3PC-RH"))
        _write(output_dir, rows, manifest)

                                                                              
                                                                            
        started = time.perf_counter()
        try:
            if c1_phase1 is None:
                raise RuntimeError("common service-feasible seed unavailable")
            c2 = solve_graph_tlns(
                instance,
                context,
                pool,
                config=GraphLNSConfig(training_epochs=20, random_seed=context.random_seed),
                epsilon=epsilon,
                time_limit=max(1.0, time_limit - c1_phase1.runtime_seconds),
                initial_solution=c1_phase1,
                service_reference=c1_phase1.service_optimum,
                cancellation_reference=c1_phase1.cancellation_bound,
            )
            validation = validate_solution(instance, context, pool, c2)
            validation.raise_if_invalid()
            row = _row(case_name, station_count, trains, epsilon, c2, validation, len(pool.by_id), float(c2.metadata.get("graph_training_seconds", 0.0)))
            row["online_runtime_seconds"] = c1_phase1.runtime_seconds + c2.runtime_seconds
            print(
                "  %-20s %-14s service=%8.2f cmd=%7.2f/%6.2f time=%6.2fs"
                % (c2.algorithm, c2.status, c2.objective.service_loss, c2.objective.command_total, c2.objective.command_peak, c2.runtime_seconds),
                flush=True,
            )
        except Exception as exc:
            row = _failure_row(case_name, station_count, trains, epsilon, "C2-GT-TLNS", len(pool.by_id), started, exc)
        if (case_name, "C2-GT-TLNS") not in completed:
            rows.append(row)
            completed.add((case_name, "C2-GT-TLNS"))

                                                                             
                                                                      
        gurobi_rows = [
            row
            for row in rows
            if row["case"] == case_name
            and row["algorithm"] == "Gurobi-Exact"
            and row["status"] == "optimal"
        ]
        if gurobi_rows:
            reference = float(gurobi_rows[-1]["service_loss"])
            for row in rows:
                if (
                    row["case"] == case_name
                    and row["algorithm"] != "B0"
                    and row["service_loss"] != ""
                ):
                    row["relative_gurobi_gap_pct"] = 100.0 * (
                        float(row["service_loss"]) - reference
                    ) / max(abs(reference), 1.0e-9)
        _write(output_dir, rows, manifest)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-count", type=int, default=8)
    parser.add_argument("--train-counts", nargs="+", type=int, default=[8, 12, 15, 20, 25])
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--time-limit", type=float, default=60.0)
    parser.add_argument("--phase1-time-limit", type=float, default=30.0)
    parser.add_argument("--c1-epochs", type=int, default=5)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "public_scenario_study" / "method_comparison",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(args.output_dir, args.station_count, tuple(args.train_counts), args.epsilon, args.time_limit, args.phase1_time_limit, args.c1_epochs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
