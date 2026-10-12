import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from hsr_ttr.algorithms.command_refinement import refine_commands
from hsr_ttr.algorithms.safe_m3pc_rh import SafeM3PCConfig, train_masked_model
from hsr_ttr.data import build_scalable_instance, build_scalable_uncertain_context
from hsr_ttr.milp import solve_lexicographic
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.validation import validate_solution


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--study', choices=['controlled', 'parameters'], default='controlled')
    p.add_argument('--train-counts', nargs='+', type=int, default=[8, 15, 25])
    p.add_argument('--seeds', nargs='+', type=int, default=list(range(20260905, 20260910)))
    p.add_argument('--time-limit', type=float, default=10.)
    p.add_argument('--reference-time-limit', type=float, default=60.)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs/controlled_comparison')
    args = p.parse_args()
    torch.set_num_threads(2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    counts = [15] if args.study == 'parameters' else args.train_counts
    for n in counts:
        instance = build_scalable_instance(8, n)
        context = build_scalable_uncertain_context(instance, epsilon=.01)
        pool = generate_path_pool(instance, context)
        reference = solve_lexicographic(instance, context, pool, algorithm='common-service', objective_mode='service_only', time_limit=args.reference_time_limit)
        validate_solution(instance, context, pool, reference).raise_if_invalid()
        (args.output_dir/f'reference_{n}.json').write_text(json.dumps(reference.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
        for seed in args.seeds:
            current = replace(context, random_seed=seed)
            config = SafeM3PCConfig(embedding_dim=24, attention_heads=4, transformer_layers=1, training_epochs=args.epochs, batch_size=512, max_training_paths=6000, random_seed=seed)
            model, loss = train_masked_model(instance, current, pool, config)
            if args.study == 'controlled':
                settings = [(k, k, 64, args.time_limit) for k in ('full', 'learned', 'diverse', 'rule', 'random', 'no_prior')]
            else:
                settings = [('width', str(w), w, 10.) for w in (32, 64, 128)] + [('time', str(t), 64, float(t)) for t in (2, 5, 10, 20)]
            np.random.default_rng(seed+n).shuffle(settings)
            for factor, level, width, limit in settings:
                method = level if args.study == 'controlled' else 'diverse'
                solution = refine_commands(instance, current, pool, reference, model=model, kind=method, width=width, epsilon=.01, time_limit=limit, seed=seed)
                validate_solution(instance, current, pool, solution).raise_if_invalid()
                record = dict(train_count=n, seed=seed, method=method, factor=factor, level=level, width=width, epsilon=.01, budget_seconds=limit,
                              common_reference=reference.objective.service_loss, service=solution.objective.service_loss,
                              commands=solution.objective.command_total, peak=solution.objective.command_peak, seconds=solution.runtime_seconds,
                              ranking_seconds=solution.metadata['candidate_ranking_seconds'], status=solution.status, valid=True)
                rows.append(record)
                name=f'{n}_{seed}_{factor}_{level}'
                (args.output_dir/(name+'.json')).write_text(json.dumps(solution.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
                with (args.output_dir/(args.study+'_raw.csv')).open('w', encoding='utf-8-sig', newline='') as f:
                    writer=csv.DictWriter(f, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                print(name, record['commands'], record['peak'], flush=True)


if __name__ == '__main__':
    main()
