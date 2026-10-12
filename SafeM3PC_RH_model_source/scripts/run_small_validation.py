                                                                 

                                                                            
                                                                           
                                                               

                                                                           
                                                                            
                               
   

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import scipy

from hsr_ttr.algorithms.baselines import solve_b0, solve_b1
from hsr_ttr.algorithms.graph_lns import GraphLNSConfig, solve_graph_tlns
from hsr_ttr.algorithms.safe_m3pc_rh import (
    SafeM3PCConfig,
    solve_safe_m3pc_rh,
    train_masked_model,
)
from hsr_ttr.data import (
    build_small_context,
    build_small_instance,
    build_standalone_recovery_scenarios,
    validate_small_instance,
)
from hsr_ttr.paths import generate_path_pool, path_pool_summary
from hsr_ttr.validation import validate_solution


def _row(case: str, epsilon: float, solution, valid: bool, issue_count: int) -> Dict[str, object]:
    return {
        "case": case,
        "epsilon": epsilon,
        "algorithm": solution.algorithm,
        "status": solution.status,
        "independently_valid": valid,
        "validation_issue_count": issue_count,
        "service_loss": solution.objective.service_loss,
        "service_optimum": solution.service_optimum,
        "hdfs_violations": solution.objective.hdfs_violations,
        "workload_total": solution.objective.workload_total,
        "workload_peak": solution.objective.workload_peak,
        "cancelled_trains": ",".join(solution.cancelled_trains),
        "cancelled_count": len(solution.cancelled_trains),
        "runtime_seconds": solution.runtime_seconds,
        "mip_gap": solution.mip_gap,
        "path_variable_count": solution.metadata.get("path_variable_count"),
        "constraint_count": solution.metadata.get("constraint_count"),
    }


def _validate_or_raise(instance, context, pool, solution) -> None:
    report = validate_solution(instance, context, pool, solution)
    if not report.valid:
        report.raise_if_invalid()


def _save_results(
    output_dir: Path,
    rows: List[Dict[str, object]],
    payload: Dict[str, object],
    stem: str = "small_validation_results",
) -> Tuple[Path, Path]:
    def write_pair(target_dir: Path) -> Tuple[Path, Path]:
        target_dir.mkdir(parents=True, exist_ok=True)
        target_csv = target_dir / (stem + ".csv")
        target_json = target_dir / (stem + ".json")
        with target_csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with target_json.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        return target_csv, target_json

    try:
        csv_path, json_path = write_pair(output_dir)
    except PermissionError:
                                                                              
                                                                           
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_dir = output_dir / "runs" / timestamp
        print(
            "WARNING: default result file is locked; redirecting to %s"
            % fallback_dir,
            flush=True,
        )
        csv_path, json_path = write_pair(fallback_dir)
    print("CSV:", csv_path)
    print("JSON:", json_path)
    return csv_path, json_path


def _epsilon_frontier(values: Optional[Iterable[float]]) -> tuple[float, ...]:
                                                           

                                                                          
                                                                             
       

    if values is None:
        config_path = Path(__file__).resolve().parents[1] / "configs" / "small_8s15t.json"
        with config_path.open("r", encoding="utf-8") as handle:
            values = json.load(handle)["epsilon_values"]

    frontier = tuple(dict.fromkeys(float(value) for value in values))
    if not frontier:
        raise ValueError("At least one epsilon value is required.")
    if any(value < 0.0 for value in frontier):
        raise ValueError("Epsilon values must be non-negative.")
    if not any(abs(value) <= 1.0e-12 for value in frontier):
        raise ValueError("The epsilon frontier must include 0.0 for the common B1 benchmark.")
    return frontier


def _solve_standalone_frontier(
    instance,
    context,
    pool,
    epsilon_values: Tuple[float, ...],
    exact_time_limit: Optional[float],
    learned_time_limit: Optional[float],
    case_label: str,
) -> Dict[str, object]:
                                                                   

                                                                            
                                                                          
                                                                             
                                                                     
       

    scenario = context.scenarios[0]
    s1_duration = scenario.t12 - instance.disruption_start
    s2_duration = scenario.t23 - scenario.t12
    scenario_fields = {
        "scenario_case": scenario.scenario_id,
        "s1_duration_min": s1_duration,
        "s2_duration_min": s2_duration,
        "s2_speed_kmh": scenario.s2_speed_kmh,
        "s2_runtime_min": scenario.s2_runtime,
    }
    rows: List[Dict[str, object]] = []
    solutions: List[Dict[str, object]] = []
    b1_service_reference: Optional[float] = None
    b1_cancellation_reference: Optional[int] = None

    for epsilon in epsilon_values:
        for name, solver in (("B0", solve_b0), ("B1", solve_b1)):
            print(
                "[%s] Solving %s epsilon=%.3f ..." % (scenario.scenario_id, name, epsilon),
                flush=True,
            )
            solution = solver(
                instance,
                context,
                pool,
                epsilon=epsilon,
                time_limit=exact_time_limit,
            )
            _validate_or_raise(instance, context, pool, solution)
            row = _row(case_label, epsilon, solution, True, 0)
            row.update(scenario_fields)
            rows.append(row)
            solutions.append(solution.to_dict())
            if name == "B1" and abs(epsilon) <= 1.0e-12:
                b1_service_reference = float(solution.service_optimum)
                b1_cancellation_reference = int(solution.cancellation_bound)

    if b1_service_reference is None or b1_cancellation_reference is None:
        raise RuntimeError("Standalone B1 epsilon=0 benchmark was not established.")

    m3pc_config = SafeM3PCConfig(
        training_epochs=25,
        batch_size=256,
        candidate_widths=(8, 16, None),
    )
    training_start = time.perf_counter()
    model, losses = train_masked_model(instance, context, pool, m3pc_config)
    offline_training_seconds = time.perf_counter() - training_start

    for epsilon in epsilon_values:
        print(
            "[%s] Solving C1 Safe-M3PC-RH epsilon=%.3f ..."
            % (scenario.scenario_id, epsilon),
            flush=True,
        )
        c1 = solve_safe_m3pc_rh(
            instance,
            context,
            pool,
            config=m3pc_config,
            epsilon=epsilon,
            time_limit=learned_time_limit,
            pretrained_model=model,
            service_reference=b1_service_reference,
            cancellation_reference=b1_cancellation_reference,
        )
        c1.metadata["offline_pretraining_seconds"] = offline_training_seconds
        c1.metadata["offline_training_initial_loss"] = losses[0]
        c1.metadata["offline_training_final_loss"] = losses[-1]
        _validate_or_raise(instance, context, pool, c1)
        row = _row(case_label, epsilon, c1, True, 0)
        row.update(scenario_fields)
        rows.append(row)
        solutions.append(c1.to_dict())

        print(
            "[%s] Solving C2 graph-Transformer TLNS epsilon=%.3f ..."
            % (scenario.scenario_id, epsilon),
            flush=True,
        )
        c2 = solve_graph_tlns(
            instance,
            context,
            pool,
            config=GraphLNSConfig(training_epochs=60),
            epsilon=epsilon,
            time_limit=learned_time_limit,
            service_reference=b1_service_reference,
            cancellation_reference=b1_cancellation_reference,
        )
        _validate_or_raise(instance, context, pool, c2)
        row = _row(case_label, epsilon, c2, True, 0)
        row.update(scenario_fields)
        rows.append(row)
        solutions.append(c2.to_dict())

    return {
        "experiment": case_label,
        "scenario": scenario_fields,
        "path_pool_summary": path_pool_summary(pool),
        "path_pool_total": len(pool.by_id),
        "common_b1_benchmark": {
            "service_reference": b1_service_reference,
            "cancellation_reference": b1_cancellation_reference,
        },
        "offline_training": {
            "algorithm": "Safe-M3PC-RH pilot",
            "seconds": offline_training_seconds,
            "epochs": len(losses),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
        "rows": rows,
        "solutions": solutions,
    }


def _run_four_scenario_outputs(
    output_dir: Path,
    epsilon_values: Tuple[float, ...],
    exact_time_limit: Optional[float],
    learned_time_limit: Optional[float],
    extreme_s1_duration: int,
    scenario_horizon_end: int,
) -> None:
                                                                        

    scenario_instance = replace(build_small_instance(), horizon_end=scenario_horizon_end)
    scenarios = build_standalone_recovery_scenarios(
        scenario_instance.disruption_start,
        extreme_s1_duration=extreme_s1_duration,
    )
    latest_recovery = max(scenario.t23 for scenario in scenarios)
    if scenario_horizon_end < latest_recovery + 60:
        raise ValueError(
            "scenario_horizon_end must be at least 60 minutes after the latest S3 recovery."
        )

    scenario_dir = output_dir / "scenario_outputs"
    combined_rows: List[Dict[str, object]] = []
    combined_payloads: List[Dict[str, object]] = []
    reference_probability = {"A_fast": 0.25, "B_base": 0.50, "C_slow": 0.25}

    print("\n=== Four standalone recovery-scenario outputs ===", flush=True)
    for scenario in scenarios:
        context = build_small_context(
            scenario_instance,
            epsilon=epsilon_values[0],
            scenarios=(scenario,),
        )
        pool = generate_path_pool(scenario_instance, context)
        print(
            "\nScenario %s: S1=%d min, S2=%d min, speed=%d km/h, paths=%d"
            % (
                scenario.scenario_id,
                scenario.t12 - scenario_instance.disruption_start,
                scenario.t23 - scenario.t12,
                scenario.s2_speed_kmh,
                len(pool.by_id),
            ),
            flush=True,
        )
        payload = _solve_standalone_frontier(
            scenario_instance,
            context,
            pool,
            epsilon_values,
            exact_time_limit,
            learned_time_limit,
            "small_8s15t_" + scenario.scenario_id,
        )
        payload["scenario"]["reference_tree_probability"] = reference_probability.get(
            scenario.scenario_id
        )
        payload["scenario"]["is_stress_test"] = scenario.scenario_id == "D_extreme"
        payload["scenario"]["analysis_horizon_end"] = scenario_horizon_end
        _save_results(
            scenario_dir / scenario.scenario_id,
            payload["rows"],
            payload,
            stem=scenario.scenario_id + "_results",
        )
        combined_rows.extend(payload["rows"])
        combined_payloads.append(payload)

    combined = {
        "experiment": "small_8s15t_four_standalone_recovery_scenarios",
        "note": (
            "A/B/C are solved independently with probability one; their reference "
            "tree probabilities are metadata only. D is a zero-probability stress "
            "test and is not inserted into the uncertain expectation model."
        ),
        "epsilon_values": list(epsilon_values),
        "extreme_s1_duration_min": extreme_s1_duration,
        "scenario_horizon_end": scenario_horizon_end,
        "rows": combined_rows,
        "scenario_payloads": combined_payloads,
    }
    _save_results(
        scenario_dir,
        combined_rows,
        combined,
        stem="four_scenario_results",
    )

    print("\nFour-scenario summary")
    for item in combined_rows:
        print(
            "{scenario_case:10s} {algorithm:20s} eps={epsilon:4.2f} "
            "service={service_loss:8.2f} work={workload_total:6.2f} "
            "peak={workload_peak:6.2f} time={runtime_seconds:7.2f}s".format(**item)
        )


def run(
    output_dir: Path,
    exact_time_limit: Optional[float],
    learned_time_limit: Optional[float],
    epsilon_values: Optional[Iterable[float]] = None,
    include_scenario_outputs: bool = True,
    extreme_s1_duration: int = 210,
    scenario_horizon_end: int = 480,
) -> None:
    epsilon_values = _epsilon_frontier(epsilon_values)
    instance = build_small_instance()
    errors = validate_small_instance(instance)
    if errors:
        raise ValueError("Invalid small instance: " + "; ".join(errors))
                                                                            
                                                              
    base_context = build_small_context(instance, epsilon=0.01)
    pool = generate_path_pool(instance, base_context)
    print("Path pool:", path_pool_summary(pool), "total=", len(pool.by_id))

    rows: List[Dict[str, object]] = []
    solutions: List[Dict[str, object]] = []
    b1_service_reference: Optional[float] = None
    b1_cancellation_reference: Optional[int] = None

                                                                             
                                                                         
    for epsilon in epsilon_values:
        for name, solver in (("B0", solve_b0), ("B1", solve_b1)):
            print("Solving %s epsilon=%.2f ..." % (name, epsilon), flush=True)
            solution = solver(
                instance,
                base_context,
                pool,
                epsilon=epsilon,
                time_limit=exact_time_limit,
            )
            report = validate_solution(instance, base_context, pool, solution)
            if not report.valid:
                report.raise_if_invalid()
            rows.append(_row("small_8s15t", epsilon, solution, True, 0))
            solutions.append(solution.to_dict())
            if name == "B1" and epsilon == 0.0:
                b1_service_reference = float(solution.service_optimum)
                b1_cancellation_reference = int(solution.cancellation_bound)

    if b1_service_reference is None or b1_cancellation_reference is None:
        raise RuntimeError("The common B1 benchmark was not established.")

                                                                             
                                                                               
                                                            
    m3pc_config = SafeM3PCConfig(
        training_epochs=25,
        batch_size=256,
        candidate_widths=(8, 16, None),
    )
    training_start = time.perf_counter()
    model, losses = train_masked_model(instance, base_context, pool, m3pc_config)
    offline_training_seconds = time.perf_counter() - training_start
    print(
        "M3PC pilot pretraining: %.3fs, loss %.6f -> %.6f"
        % (offline_training_seconds, losses[0], losses[-1]),
        flush=True,
    )

                                                                               
                                                                              
    for epsilon in epsilon_values:
        print(
            "Solving C1 Safe-M3PC-RH epsilon=%.3f ..." % epsilon,
            flush=True,
        )
        c1 = solve_safe_m3pc_rh(
            instance,
            base_context,
            pool,
            config=m3pc_config,
            epsilon=epsilon,
            time_limit=learned_time_limit,
            pretrained_model=model,
            service_reference=b1_service_reference,
            cancellation_reference=b1_cancellation_reference,
        )
        c1.metadata["offline_pretraining_seconds"] = offline_training_seconds
        c1.metadata["offline_training_initial_loss"] = losses[0]
        c1.metadata["offline_training_final_loss"] = losses[-1]
        c1_report = validate_solution(instance, base_context, pool, c1)
        if not c1_report.valid:
            c1_report.raise_if_invalid()
        rows.append(_row("small_8s15t", epsilon, c1, True, 0))
        solutions.append(c1.to_dict())

        print(
            "Solving C2 graph-Transformer TLNS epsilon=%.3f ..." % epsilon,
            flush=True,
        )
        c2 = solve_graph_tlns(
            instance,
            base_context,
            pool,
            config=GraphLNSConfig(training_epochs=60),
            epsilon=epsilon,
            time_limit=learned_time_limit,
            service_reference=b1_service_reference,
            cancellation_reference=b1_cancellation_reference,
        )
        c2_report = validate_solution(instance, base_context, pool, c2)
        if not c2_report.valid:
            c2_report.raise_if_invalid()
        rows.append(_row("small_8s15t", epsilon, c2, True, 0))
        solutions.append(c2.to_dict())

    payload: Dict[str, object] = {
        "experiment": "small_8s15t",
        "temporary_assumptions": True,
        "path_pool_summary": path_pool_summary(pool),
        "path_pool_total": len(pool.by_id),
        "run_manifest": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": getattr(__import__("torch"), "__version__", "unknown"),
            "exact_time_limit_seconds": exact_time_limit,
            "learned_time_limit_seconds": learned_time_limit,
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "random_seed": base_context.random_seed,
            "decision_time": base_context.now,
            "next_observation": base_context.next_observation,
            "experiment_scope": "single rolling-decision snapshot",
            "command_issue_proxy": (
                "hold/release: action minus 10 min; order: scenario t12 minus "
                "10 min; cancellation/current station change: now"
            ),
        },
        "common_b1_benchmark": {
            "service_reference": b1_service_reference,
            "cancellation_reference": b1_cancellation_reference,
        },
        "offline_training": {
            "algorithm": "Safe-M3PC-RH pilot",
            "seconds": offline_training_seconds,
            "epochs": len(losses),
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        },
        "rows": rows,
        "solutions": solutions,
    }
    _save_results(output_dir, rows, payload)

    print("\nSummary")
    for item in rows:
        print(
            "{algorithm:20s} eps={epsilon:4.2f} service={service_loss:8.2f} "
            "hdfs={hdfs_violations:5.2f} work={workload_total:6.2f} "
            "peak={workload_peak:6.2f} time={runtime_seconds:7.2f}s".format(**item)
        )

    if include_scenario_outputs:
        _run_four_scenario_outputs(
            output_dir,
            epsilon_values,
            exact_time_limit,
            learned_time_limit,
            extreme_s1_duration,
            scenario_horizon_end,
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs",
    )
    parser.add_argument(
        "--exact-time-limit",
        type=float,
        default=None,
        help="Optional total seconds for each B0/B1 four-stage solve.",
    )
    parser.add_argument(
        "--learned-time-limit",
        type=float,
        default=180.0,
        help="Total online seconds for each C1/C2 call.",
    )
    parser.add_argument(
        "--epsilon-values",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Optional epsilon frontier shared by B0/B1/C1/C2; include 0.0. "
            "Defaults to configs/small_8s15t.json."
        ),
    )
    parser.add_argument(
        "--skip-scenario-outputs",
        action="store_true",
        help="Run only the original three-scenario aggregate experiment.",
    )
    parser.add_argument(
        "--extreme-s1-min",
        type=int,
        choices=(180, 210),
        default=210,
        help="S1 duration for standalone D_extreme; default is the 210-min upper stress case.",
    )
    parser.add_argument(
        "--scenario-horizon-end",
        type=int,
        default=480,
        help="Planning-horizon end for the four standalone scenarios.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(
        args.output_dir,
        args.exact_time_limit,
        args.learned_time_limit,
        args.epsilon_values,
        not args.skip_scenario_outputs,
        args.extreme_s1_min,
        args.scenario_horizon_end,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
