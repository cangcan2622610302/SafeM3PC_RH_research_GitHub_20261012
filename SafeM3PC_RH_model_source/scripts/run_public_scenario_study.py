                                                            

                                                                          
                                                                             
                                                                            
                                                                    
                                       

                                                                           
                                                                            
                                                                           
                                                                            
                 
   

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from hsr_ttr.algorithms.safe_m3pc_rh import (
    SafeM3PCConfig,
    solve_safe_m3pc_rh,
    train_masked_model,
)
from hsr_ttr.domain import ObjectiveVector, ScheduleSolution
from hsr_ttr.paths import PathPool, generate_path_pool, path_pool_summary
from hsr_ttr.public_data import (
    PublicMacroCase,
    build_public_macro_cases,
    scenario_catalog_rows,
)
from hsr_ttr.validation import recompute_objective, validate_solution


EPSILON_VALUES: Tuple[float, ...] = (0.0, 0.01, 0.02, 0.03)


def _solution_from_dict(payload: Mapping[str, object]) -> ScheduleSolution:
    objective = payload["objective"]
    selected = {
        tuple(str(key).split("|", 1)): str(value)
        for key, value in payload["selected_paths"].items()                            
    }
    return ScheduleSolution(
        algorithm=str(payload["algorithm"]),
        status=str(payload["status"]),
        selected_paths=selected,                          
        objective=ObjectiveVector(
            service_loss=float(objective["service_loss"]),                       
            hdfs_violations=float(objective["hdfs_violations"]),                       
            workload_total=float(objective["workload_total"]),                       
            workload_peak=float(objective["workload_peak"]),                       
        ),
        cancelled_trains=tuple(str(item) for item in payload["cancelled_trains"]),                          
        service_optimum=float(payload["service_optimum"]),
        cancellation_bound=int(payload["cancellation_bound"]),
        stage_objectives={
            str(key): float(value)
            for key, value in payload["stage_objectives"].items()                            
        },
        runtime_seconds=float(payload["runtime_seconds"]),
        mip_gap=(
            None if payload.get("mip_gap") is None else float(payload["mip_gap"])
        ),
        metadata=dict(payload.get("metadata", {})),
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _resolved_paths(solution: ScheduleSolution, pool: PathPool):
    return {
        key: pool.by_id[path_id]
        for key, path_id in solution.selected_paths.items()
    }


def _branch_command_counts(
    case: PublicMacroCase,
    pool: PathPool,
    solution: ScheduleSolution,
    scenario_id: str,
) -> Dict[str, float]:
    resolved = _resolved_paths(solution, pool)
    _objective, details = recompute_objective(
        case.instance,
        case.context,
        resolved,
        scenarios=(scenario_id,),
    )
    by_scenario = details["by_scenario"][scenario_id]
    counts = {
        "hold_commands": 0.0,
        "order_commands": float(by_scenario["order_changes"]),
        "station_change_commands": 0.0,
        "cancel_commands": 0.0,
    }
    for train in case.instance.trains:
        path = resolved[(scenario_id, train.train_id)]
        for event in path.command_events:
            if event.command_type == "hold_change":
                counts["hold_commands"] += 1.0
            elif event.command_type == "station_change":
                counts["station_change_commands"] += 1.0
            elif event.command_type == "cancel":
                counts["cancel_commands"] += 1.0
    return counts


def _branch_rows(
    case: PublicMacroCase,
    pool: PathPool,
    solution: ScheduleSolution,
    phase: str,
    epsilon: float,
    offline_training_seconds: float,
    branch_service_reference: Optional[Mapping[str, float]] = None,
) -> List[Dict[str, object]]:
    resolved = _resolved_paths(solution, pool)
    rows: List[Dict[str, object]] = []
    for scenario in case.context.scenarios:
        objective, details = recompute_objective(
            case.instance,
            case.context,
            resolved,
            scenarios=(scenario,),
        )
        recovery = scenario.scenario_id.rsplit("-", 1)[-1]
        scenario_label = "%d-SRR-BD-%s-%s" % (
            case.train_count,
            case.spec.macro_id,
            recovery,
        )
        commands = _branch_command_counts(
            case, pool, solution, scenario.scenario_id
        )
        reference = (
            objective.service_loss
            if branch_service_reference is None
            else float(branch_service_reference[scenario.scenario_id])
        )
        row: Dict[str, object] = {
            "scenario_id": scenario_label,
            "macro_id": case.spec.macro_id,
            "data_date": case.spec.date,
            "density_class": case.spec.density_class,
            "train_count": case.train_count,
            "recovery_code": recovery,
            "phase": phase,
            "epsilon": epsilon,
            "algorithm": solution.algorithm,
            "status": solution.status,
            "independently_valid": True,
            "service_loss": objective.service_loss,
            "service_reference_branch": reference,
            "actual_service_increase_pct": 100.0
            * (objective.service_loss - reference)
            / max(abs(reference), 1.0e-9),
            "command_total": objective.command_total,
            "command_peak": objective.command_peak,
            "hdfs_violations_diagnostic": objective.hdfs_violations,
            "cancelled_count": len(solution.cancelled_trains),
            "cancelled_trains": ";".join(solution.cancelled_trains),
            "online_runtime_seconds": solution.runtime_seconds,
            "offline_training_seconds": offline_training_seconds,
            "candidate_width": solution.metadata.get("candidate_width_used"),
            "path_variable_count": solution.metadata.get("path_variable_count"),
            "constraint_count": solution.metadata.get("constraint_count"),
            "mip_gap": solution.mip_gap,
            "s1_duration_min": scenario.t12 - case.instance.disruption_start,
            "s2_duration_min": scenario.t23 - scenario.t12,
            "s2_speed_kmh": scenario.s2_speed_kmh,
            "s2_runtime_min": scenario.s2_runtime,
            "s2_headway_min": scenario.s2_headway,
            "prior_probability": scenario.probability,
        }
        row.update(commands)
        row["command_count_check"] = sum(commands.values())
        row["order_changes_validator"] = details["by_scenario"][scenario.scenario_id][
            "order_changes"
        ]
        rows.append(row)
    return rows


def _aggregate_row(
    case: PublicMacroCase,
    solution: ScheduleSolution,
    phase: str,
    epsilon: float,
    offline_training_seconds: float,
    validation_issue_count: int,
) -> Dict[str, object]:
    return {
        "macro_id": case.spec.macro_id,
        "data_date": case.spec.date,
        "density_class": case.spec.density_class,
        "train_count": case.train_count,
        "phase": phase,
        "epsilon": epsilon,
        "algorithm": solution.algorithm,
        "status": solution.status,
        "independently_valid": validation_issue_count == 0,
        "validation_issue_count": validation_issue_count,
        "service_loss_expected": solution.objective.service_loss,
        "service_optimum_expected": solution.service_optimum,
        "actual_service_increase_pct": 100.0
        * (solution.objective.service_loss - solution.service_optimum)
        / max(abs(solution.service_optimum), 1.0e-9),
        "command_total_expected": solution.objective.command_total,
        "command_peak_expected": solution.objective.command_peak,
        "hdfs_violations_expected_diagnostic": solution.objective.hdfs_violations,
        "cancelled_count": len(solution.cancelled_trains),
        "online_runtime_seconds": solution.runtime_seconds,
        "offline_training_seconds": offline_training_seconds,
        "candidate_width": solution.metadata.get("candidate_width_used"),
        "path_variable_count": solution.metadata.get("path_variable_count"),
        "constraint_count": solution.metadata.get("constraint_count"),
        "mip_gap": solution.mip_gap,
    }


def _adaptive_phase1_width(train_count: int) -> int:
    return max(128, 6 * int(train_count))


def _checkpoint_path(output_dir: Path, macro_id: str) -> Path:
    return output_dir / "checkpoints" / (macro_id + ".json")


def _load_checkpoint(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {"phase1": None, "phase2": {}}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_checkpoint(path: Path, payload: Mapping[str, object]) -> None:
    _write_json(path, payload)


def run(
    output_dir: Path,
    macro_ids: Sequence[str],
    epsilon_values: Sequence[float],
    phase1_time_limit: float,
    phase2_time_limit: float,
    training_epochs: int,
    max_training_paths: int,
    force: bool = False,
    force_phase2: bool = False,
) -> None:
    cases = [
        case
        for case in build_public_macro_cases()
        if case.spec.macro_id in set(macro_ids)
    ]
    if not cases:
        raise ValueError("No selected macro scenarios were found.")
    output_dir.mkdir(parents=True, exist_ok=True)
    catalog = scenario_catalog_rows(cases)
    _write_csv(output_dir / "scenario_catalog_18.csv", catalog)
    _write_json(output_dir / "scenario_catalog_18.json", {"rows": catalog})

    all_branch_rows: List[Dict[str, object]] = []
    all_aggregate_rows: List[Dict[str, object]] = []
    all_solutions: Dict[str, object] = {}

    for case in cases:
        macro_started = time.perf_counter()
        print(
            "\n[%s] date=%s trains=%d" % (
                case.spec.macro_id,
                case.spec.date,
                case.train_count,
            ),
            flush=True,
        )
        pool_started = time.perf_counter()
        pool = generate_path_pool(case.instance, case.context)
        pool_seconds = time.perf_counter() - pool_started
        print(
            "[%s] path pool=%d, generation=%.2fs"
            % (case.spec.macro_id, len(pool.by_id), pool_seconds),
            flush=True,
        )

        model_config = SafeM3PCConfig(
            embedding_dim=24,
            attention_heads=4,
            transformer_layers=1,
            training_epochs=training_epochs,
            batch_size=512,
            max_training_paths=max_training_paths,
            candidate_widths=(
                0,
                _adaptive_phase1_width(case.train_count),
                2 * _adaptive_phase1_width(case.train_count),
                4 * _adaptive_phase1_width(case.train_count),
                None,
            ),
            use_hdfs_backbone=False,
            use_constructive_backbone=True,
            stop_after_first_feasible=True,
            random_seed=case.context.random_seed,
        )
        training_started = time.perf_counter()
        model, losses = train_masked_model(
            case.instance, case.context, pool, model_config
        )
        offline_seconds = time.perf_counter() - training_started
        print(
            "[%s] offline C1 training=%.2fs, samples=%s, loss %.6f -> %.6f"
            % (
                case.spec.macro_id,
                offline_seconds,
                getattr(model, "_hsr_training_sample_count", "?"),
                losses[0],
                losses[-1],
            ),
            flush=True,
        )

        checkpoint_file = _checkpoint_path(output_dir, case.spec.macro_id)
        checkpoint = {"phase1": None, "phase2": {}} if force else _load_checkpoint(checkpoint_file)
        if force_phase2:
            checkpoint["phase2"] = {}
        phase1_payload = checkpoint.get("phase1")
        if phase1_payload:
            phase1 = _solution_from_dict(phase1_payload)                          
            print("[%s] reused phase-1 checkpoint" % case.spec.macro_id, flush=True)
        else:
            phase1 = solve_safe_m3pc_rh(
                case.instance,
                case.context,
                pool,
                config=model_config,
                epsilon=0.0,
                time_limit=phase1_time_limit,
                pretrained_model=model,
                objective_mode="service_only",
                stabilize_cancellations=False,
            )
            phase1.metadata.update(
                {
                    "offline_training_seconds": offline_seconds,
                    "offline_training_epochs": training_epochs,
                    "offline_training_initial_loss": losses[0],
                    "offline_training_final_loss": losses[-1],
                    "path_pool_total": len(pool.by_id),
                    "path_pool_generation_seconds": pool_seconds,
                }
            )
            checkpoint["phase1"] = phase1.to_dict()
            _save_checkpoint(checkpoint_file, checkpoint)

                                                                             
                                                                             
                                                                             
                                                                             
                                                                          
        if force or not bool(phase1.metadata.get("phase1_refined", False)):
            refinement_config = SafeM3PCConfig(
                embedding_dim=24,
                attention_heads=4,
                transformer_layers=1,
                training_epochs=training_epochs,
                batch_size=512,
                max_training_paths=max_training_paths,
                candidate_widths=(0, 64),
                use_hdfs_backbone=False,
                use_constructive_backbone=True,
                stop_after_first_feasible=True,
                command_aware_ranking_for_service=True,
                random_seed=case.context.random_seed,
            )
            seed = phase1
            refinement = solve_safe_m3pc_rh(
                case.instance,
                case.context,
                pool,
                config=refinement_config,
                epsilon=0.0,
                time_limit=phase1_time_limit,
                pretrained_model=model,
                objective_mode="service_only",
                required_path_ids=seed.selected_paths.values(),
                stabilize_cancellations=False,
            )
            seed_online_seconds = seed.runtime_seconds
            refinement_online_seconds = refinement.runtime_seconds
            if (
                refinement.objective.service_loss,
                len(refinement.cancelled_trains),
            ) < (
                seed.objective.service_loss,
                len(seed.cancelled_trains),
            ):
                phase1 = refinement
            total_phase1_online = seed_online_seconds + refinement_online_seconds
            phase1.runtime_seconds = total_phase1_online
            phase1.metadata.update(
                {
                    "phase1_refined": True,
                    "phase1_seed_service_loss": seed.objective.service_loss,
                    "phase1_seed_status": seed.status,
                    "phase1_refinement_service_loss": refinement.objective.service_loss,
                    "phase1_refinement_status": refinement.status,
                    "phase1_seed_online_seconds": seed_online_seconds,
                    "phase1_refinement_online_seconds": refinement_online_seconds,
                    "phase1_total_online_seconds": total_phase1_online,
                    "offline_training_seconds": offline_seconds,
                    "offline_training_epochs": training_epochs,
                    "offline_training_initial_loss": losses[0],
                    "offline_training_final_loss": losses[-1],
                    "path_pool_total": len(pool.by_id),
                    "path_pool_generation_seconds": pool_seconds,
                }
            )
            checkpoint["phase1"] = phase1.to_dict()
                                                                             
                                                                 
            checkpoint["phase2"] = {}
            _save_checkpoint(checkpoint_file, checkpoint)
        validation = validate_solution(case.instance, case.context, pool, phase1)
        validation.raise_if_invalid()
        print(
            "[%s] phase 1: service=%.2f, commands=%.2f/%.2f, online=%.2fs"
            % (
                case.spec.macro_id,
                phase1.objective.service_loss,
                phase1.objective.command_total,
                phase1.objective.command_peak,
                phase1.runtime_seconds,
            ),
            flush=True,
        )
        all_branch_rows.extend(
            _branch_rows(case, pool, phase1, "phase1_service_only", 0.0, offline_seconds)
        )
        phase1_resolved = _resolved_paths(phase1, pool)
        phase1_branch_reference = {
            scenario.scenario_id: recompute_objective(
                case.instance,
                case.context,
                phase1_resolved,
                scenarios=(scenario,),
            )[0].service_loss
            for scenario in case.context.scenarios
        }
        all_aggregate_rows.append(
            _aggregate_row(
                case,
                phase1,
                "phase1_service_only",
                0.0,
                offline_seconds,
                len(validation.issues),
            )
        )

        phase2_config = SafeM3PCConfig(
            embedding_dim=24,
            attention_heads=4,
            transformer_layers=1,
            training_epochs=training_epochs,
            batch_size=512,
            max_training_paths=max_training_paths,
            candidate_widths=(0, 64),
            use_hdfs_backbone=False,
            use_constructive_backbone=True,
            stop_after_first_feasible=True,
            random_seed=case.context.random_seed,
        )
        previous_paths = set(phase1.selected_paths.values())
        phase2_store = checkpoint.setdefault("phase2", {})
        for epsilon in epsilon_values:
            epsilon_key = "%.4f" % float(epsilon)
            saved = phase2_store.get(epsilon_key)                            
            if saved:
                solution = _solution_from_dict(saved)
                print(
                    "[%s] reused phase-2 epsilon=%s checkpoint"
                    % (case.spec.macro_id, epsilon_key),
                    flush=True,
                )
            else:
                solution = solve_safe_m3pc_rh(
                    case.instance,
                    case.context,
                    pool,
                    config=phase2_config,
                    epsilon=float(epsilon),
                    time_limit=phase2_time_limit,
                    pretrained_model=model,
                    service_reference=phase1.service_optimum,
                    cancellation_reference=phase1.cancellation_bound,
                    objective_mode="full",
                    required_path_ids=previous_paths,
                )
                solution.metadata.update(
                    {
                        "offline_training_seconds": offline_seconds,
                        "offline_training_epochs": training_epochs,
                        "offline_training_initial_loss": losses[0],
                        "offline_training_final_loss": losses[-1],
                        "path_pool_total": len(pool.by_id),
                        "path_pool_generation_seconds": pool_seconds,
                    }
                )
                phase2_store[epsilon_key] = solution.to_dict()                       
                _save_checkpoint(checkpoint_file, checkpoint)
            validation = validate_solution(case.instance, case.context, pool, solution)
            validation.raise_if_invalid()
            previous_paths.update(solution.selected_paths.values())
            print(
                "[%s] phase 2 eps=%.2f: service=%.2f, commands=%.2f/%.2f, online=%.2fs"
                % (
                    case.spec.macro_id,
                    epsilon,
                    solution.objective.service_loss,
                    solution.objective.command_total,
                    solution.objective.command_peak,
                    solution.runtime_seconds,
                ),
                flush=True,
            )
            all_branch_rows.extend(
                _branch_rows(
                    case,
                    pool,
                    solution,
                    "phase2_command_lexicographic",
                    float(epsilon),
                    offline_seconds,
                    branch_service_reference=phase1_branch_reference,
                )
            )
            all_aggregate_rows.append(
                _aggregate_row(
                    case,
                    solution,
                    "phase2_command_lexicographic",
                    float(epsilon),
                    offline_seconds,
                    len(validation.issues),
                )
            )

        all_solutions[case.spec.macro_id] = checkpoint
        print(
            "[%s] complete in %.2fs" % (
                case.spec.macro_id,
                time.perf_counter() - macro_started,
            ),
            flush=True,
        )

                                                                               
                                                                        
        _write_csv(output_dir / "public_branch_results.csv", all_branch_rows)
        _write_csv(output_dir / "public_aggregate_results.csv", all_aggregate_rows)
        _write_json(
            output_dir / "public_study_results.json",
            {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "source": cases[0].source_path,
                "network": "Beijing-Shanghai HSR: Jinan West-Nanjing South",
                "disrupted_section": "Bengbu South-Dingyuan (BD)",
                "macros_requested": list(macro_ids),
                "epsilon_values": list(epsilon_values),
                "python": sys.version,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
                "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
                "phase1_time_limit_seconds": phase1_time_limit,
                "phase2_time_limit_seconds": phase2_time_limit,
                "training_epochs": training_epochs,
                "max_training_paths": max_training_paths,
                "scenario_catalog": catalog,
                "aggregate_rows": all_aggregate_rows,
                "branch_rows": all_branch_rows,
                "solutions": all_solutions,
            },
        )


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "outputs"
        / "public_scenario_study",
    )
    parser.add_argument(
        "--macros",
        nargs="+",
        default=["M01", "M02", "M03", "M04", "M05", "M06"],
    )
    parser.add_argument(
        "--epsilon-values", nargs="+", type=float, default=list(EPSILON_VALUES)
    )
    parser.add_argument("--phase1-time-limit", type=float, default=240.0)
    parser.add_argument("--phase2-time-limit", type=float, default=300.0)
    parser.add_argument("--training-epochs", type=int, default=8)
    parser.add_argument("--max-training-paths", type=int, default=8000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--force-phase2",
        action="store_true",
        help="Keep a validated phase-1 checkpoint but recompute every epsilon solution.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(
        output_dir=args.output_dir,
        macro_ids=tuple(args.macros),
        epsilon_values=tuple(dict.fromkeys(float(value) for value in args.epsilon_values)),
        phase1_time_limit=float(args.phase1_time_limit),
        phase2_time_limit=float(args.phase2_time_limit),
        training_epochs=int(args.training_epochs),
        max_training_paths=int(args.max_training_paths),
        force=bool(args.force),
        force_phase2=bool(args.force_phase2),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
