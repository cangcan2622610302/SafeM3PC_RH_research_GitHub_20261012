                                                                            

                                                                              
                                                                           
                                                                             
           
   

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time
from typing import Dict, List

from hsr_ttr.algorithms.commercial import solve_gurobi_exact
from hsr_ttr.data import build_scalable_context, build_scalable_instance
from hsr_ttr.milp import PathMasterMILP
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.validation import validate_solution


FIELDS = [
    "station_count",
    "train_count",
    "path_variable_count",
    "total_variable_count",
    "base_constraint_count",
    "final_constraint_count_estimate",
    "status",
    "optimal",
    "independently_valid",
    "runtime_seconds",
    "mip_gap",
    "failure_type",
    "message",
]


def _classify_failure(message: str) -> str:
    lowered = message.lower()
    if "model too large" in lowered or "size-limited license" in lowered:
        return "license_size_limit"
    if "time limit" in lowered or "feasible_limit" in lowered:
        return "time_limit"
    if "no feasible" in lowered or "infeasible" in lowered:
        return "infeasible_or_no_incumbent"
    return "solver_error"


def run(
    station_counts: List[int],
    train_counts: List[int],
    time_limit: float,
    output_dir: Path,
) -> None:
    rows: List[Dict[str, object]] = []
    for station_count in station_counts:
        for train_count in train_counts:
            print("stations=%d trains=%d" % (station_count, train_count), flush=True)
            started = time.perf_counter()
            instance = build_scalable_instance(station_count, train_count)
            context = build_scalable_context(instance, epsilon=0.0)
            pool = generate_path_pool(instance, context)
            master = PathMasterMILP(instance, context, pool)
            artifacts = master._build(                                           
                tuple(context.scenarios),
                allowed_path_ids=None,
                fixed_selected_paths={},
                enforce_nonanticipativity=False,
            )
            row: Dict[str, object] = {
                "station_count": station_count,
                "train_count": train_count,
                "path_variable_count": len(artifacts.x_by_path),
                "total_variable_count": artifacts.variables.size,
                "base_constraint_count": artifacts.rows.size,
                                                                              
                                                    
                "final_constraint_count_estimate": artifacts.rows.size + 4,
                "status": "",
                "optimal": False,
                "independently_valid": False,
                "runtime_seconds": "",
                "mip_gap": "",
                "failure_type": "",
                "message": "",
            }
            try:
                solution = solve_gurobi_exact(
                    instance,
                    context,
                    pool,
                    epsilon=0.0,
                    time_limit=time_limit,
                )
                validation = validate_solution(instance, context, pool, solution)
                row.update(
                    {
                        "status": solution.status,
                        "optimal": solution.status == "optimal",
                        "independently_valid": validation.valid,
                        "runtime_seconds": solution.runtime_seconds,
                        "mip_gap": solution.mip_gap,
                        "message": "; ".join(issue.message for issue in validation.issues),
                    }
                )
            except Exception as exc:                                   
                message = "%s: %s" % (type(exc).__name__, exc)
                row.update(
                    {
                        "status": "not_solved",
                        "runtime_seconds": time.perf_counter() - started,
                        "failure_type": _classify_failure(message),
                        "message": message,
                    }
                )
            rows.append(row)

                                                                         
                                                                              
                                                                            
            if row["failure_type"] == "license_size_limit":
                break

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "gurobi_scale_test.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "gurobi_scale_test.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "time_limit_seconds": time_limit,
                "station_counts": station_counts,
                "train_counts": train_counts,
                "rows": rows,
            },
            handle,
            ensure_ascii=False,
            indent=2,
        )
    print("CSV:", csv_path)
    print("JSON:", json_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--station-counts", nargs="+", type=int, default=[5, 6, 8, 10, 12])
    parser.add_argument("--train-counts", nargs="+", type=int, default=[8, 10, 12, 15, 18, 20, 25, 30])
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "commercial_comparison",
    )
    args = parser.parse_args()
    run(args.station_counts, args.train_counts, args.time_limit, args.output_dir)


if __name__ == "__main__":
    main()
