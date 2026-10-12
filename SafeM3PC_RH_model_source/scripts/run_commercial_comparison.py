                                                            

                                                                              
                                                                             
                                               
   

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time
from typing import Dict, Iterable, List

from hsr_ttr.algorithms.baselines import solve_b0, solve_b1
from hsr_ttr.algorithms.commercial import solve_gurobi_exact
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
)
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.validation import validate_solution


FIELDS = [
    "scenario",
    "epsilon",
    "algorithm",
    "status",
    "independently_valid",
    "service_loss",
    "service_optimum",
    "hdfs_violations",
    "workload_total",
    "workload_peak",
    "cancelled_count",
    "runtime_seconds",
    "mip_gap",
    "path_variable_count",
    "constraint_count",
    "solver_backend",
    "relative_gurobi_gap_pct",
]


def _row(scenario: str, epsilon: float, solution, valid: bool) -> Dict[str, object]:
    return {
        "scenario": scenario,
        "epsilon": epsilon,
        "algorithm": solution.algorithm,
        "status": solution.status,
        "independently_valid": valid,
        "service_loss": solution.objective.service_loss,
        "service_optimum": solution.service_optimum,
        "hdfs_violations": solution.objective.hdfs_violations,
        "workload_total": solution.objective.workload_total,
        "workload_peak": solution.objective.workload_peak,
        "cancelled_count": len(solution.cancelled_trains),
        "runtime_seconds": solution.runtime_seconds,
        "mip_gap": solution.mip_gap,
        "path_variable_count": solution.metadata.get("path_variable_count"),
        "constraint_count": solution.metadata.get("constraint_count"),
        "solver_backend": solution.metadata.get("solver_backend", "highs"),
        "relative_gurobi_gap_pct": "",
    }


def _load_existing(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write(output_dir: Path, new_rows: List[Dict[str, object]], method: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "commercial_comparison.csv"
    old_rows = _load_existing(csv_path)
    labels = {
        "b0": {"B0"},
        "b1": {"B1"},
        "c1": {"C1-Safe-M3PC-RH"},
        "c2": {"C2-GT-TLNS"},
        "gurobi": {"Gurobi-Exact"},
    }[method]
    rows = [row for row in old_rows if row.get("algorithm") not in labels]
    rows.extend(new_rows)
    order = {
        "B0": 0,
        "B1": 1,
        "C1-Safe-M3PC-RH": 2,
        "C2-GT-TLNS": 3,
        "Gurobi-Exact": 4,
    }
    gurobi_service = {
        (str(row["scenario"]), float(row["epsilon"])): float(row["service_loss"])
        for row in rows
        if row.get("algorithm") == "Gurobi-Exact"
    }
    for row in rows:
        key = (str(row["scenario"]), float(row["epsilon"]))
        if key in gurobi_service:
            benchmark = gurobi_service[key]
            row["relative_gurobi_gap_pct"] = (
                (float(row["service_loss"]) - benchmark)
                / max(abs(benchmark), 1.0e-6)
                * 100.0
            )
    rows.sort(
        key=lambda row: (
            str(row["scenario"]),
            float(row["epsilon"]),
            order.get(str(row["algorithm"]), 99),
        )
    )
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "commercial_comparison.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"rows": rows}, handle, ensure_ascii=False, indent=2)
    print("CSV:", csv_path)
    print("JSON:", json_path)


def run(method: str, epsilons: Iterable[float], output_dir: Path) -> None:
    instance = replace(build_small_instance(), horizon_end=480)
    scenarios = build_standalone_recovery_scenarios(instance.disruption_start)[:3]
    rows: List[Dict[str, object]] = []
    for scenario in scenarios:
        context = build_small_context(instance, scenarios=(scenario,))
        pool = generate_path_pool(instance, context)
        reference = solve_b0(instance, context, pool, epsilon=0.0, time_limit=120)
        service_reference = float(reference.service_optimum)
        cancellation_reference = int(reference.cancellation_bound)
        model = None
        if method == "c1":
            config = SafeM3PCConfig(
                training_epochs=25,
                batch_size=256,
                candidate_widths=(8, 16, None),
            )
            started = time.perf_counter()
            model, _losses = train_masked_model(instance, context, pool, config)
            print("[%s] C1 training %.3fs" % (scenario.scenario_id, time.perf_counter() - started))

        for epsilon in epsilons:
            print("[%s] %s epsilon=%.3f" % (scenario.scenario_id, method, epsilon))
            if method == "b0":
                solution = solve_b0(
                    instance, context, pool, epsilon=epsilon, time_limit=120
                )
            elif method == "b1":
                solution = solve_b1(
                    instance, context, pool, epsilon=epsilon, time_limit=120
                )
            elif method == "gurobi":
                solution = solve_gurobi_exact(
                    instance, context, pool, epsilon=epsilon, time_limit=120
                )
            elif method == "c1":
                solution = solve_safe_m3pc_rh(
                    instance,
                    context,
                    pool,
                    config=config,
                    epsilon=epsilon,
                    time_limit=180,
                    pretrained_model=model,
                    service_reference=service_reference,
                    cancellation_reference=cancellation_reference,
                )
            else:
                solution = solve_graph_tlns(
                    instance,
                    context,
                    pool,
                    config=GraphLNSConfig(training_epochs=60),
                    epsilon=epsilon,
                    time_limit=180,
                    service_reference=service_reference,
                    cancellation_reference=cancellation_reference,
                )
            report = validate_solution(instance, context, pool, solution)
            report.raise_if_invalid()
            rows.append(_row(scenario.scenario_id, epsilon, solution, report.valid))
    _write(output_dir, rows, method)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", choices=("b0", "b1", "c1", "c2", "gurobi"), required=True
    )
    parser.add_argument(
        "--epsilon-values", nargs="+", type=float, default=(0.0, 0.01, 0.02, 0.03)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "commercial_comparison",
    )
    args = parser.parse_args()
    run(args.method, tuple(args.epsilon_values), args.output_dir)


if __name__ == "__main__":
    main()
