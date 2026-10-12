                                                                                
from __future__ import annotations
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
if (ROOT / '.vendor312').exists() and sys.version_info[:2] == (3, 12):
    sys.path.insert(0, str(ROOT / '.vendor312'))
import argparse
import csv
import json
from dataclasses import replace
import time
from hsr_ttr.public_data import build_public_macro_cases
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.algorithms.safe_m3pc_rh import SafeM3PCConfig, train_masked_model
from hsr_ttr.rolling_experiment import run_closed_loop, check_shared_observation_history


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--macro', default='M01')
    parser.add_argument('--recoveries', nargs='+', default=['F', 'M', 'S'])
    parser.add_argument('--modes', nargs='+', default=['service_only', 'full'])
    parser.add_argument('--epsilon', type=float, default=.01)
    parser.add_argument('--interval', type=int, default=30)
    parser.add_argument('--time-limit', type=float, default=20.)
    parser.add_argument('--train-limit', type=int, default=0)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'c1_rolling_experiment')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    case = next(c for c in build_public_macro_cases() if c.spec.macro_id == args.macro)
    if args.train_limit:
        trains = case.instance.trains[:args.train_limit]
        ids = {t.train_id for t in trains}
        case = replace(case, instance=replace(case.instance, trains=trains),
                       context=replace(case.context, train_states={t:s for t,s in case.context.train_states.items() if t in ids}))
    config = SafeM3PCConfig(embedding_dim=24, attention_heads=4, transformer_layers=1,
                            training_epochs=4, batch_size=512, max_training_paths=4000,
                            candidate_widths=(32, 64, None), use_hdfs_backbone=False,
                            use_constructive_backbone=True, stop_after_first_feasible=True,
                            random_seed=20260908)
    started = time.perf_counter()
    pool = generate_path_pool(case.instance, case.context)
    model, losses = train_masked_model(case.instance, case.context, pool, config)
    offline = time.perf_counter() - started
    results = []
    policy_cache = {}
    for code in args.recoveries:
        truth = next(s for s in case.context.scenarios if s.scenario_id.endswith('-' + code))
        for mode in args.modes:
            path = out / ('%s_%s.json' % (truth.scenario_id, mode))
            settings = dict(macro=args.macro, train_count=len(case.instance.trains), mode=mode,
                            epsilon=args.epsilon, interval=args.interval, time_limit=args.time_limit,
                            recovery=truth.scenario_id, seed=config.random_seed)
            if path.exists() and not args.force:
                result = json.loads(path.read_text(encoding='utf-8'))
                if result.get('settings') != settings:
                    raise ValueError('Existing run settings differ; choose a new output directory or --force')
            else:
                result = run_closed_loop(case, pool, model, truth, mode, args.epsilon,
                                         args.interval, args.time_limit, config, policy_cache)
                result['settings'] = settings
                result['offline_seconds'] = offline
                result['training_losses'] = losses
                result['stations'] = [s.name for s in case.instance.stations]
                result['truth'] = dict(t12=truth.t12, t23=truth.t23)
                path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
            results.append(result)
            write_csv(out / 'rolling_summary.csv', [r['summary'] for r in results])
            write_csv(out / 'rolling_epochs.csv', [dict(recovery=r['summary']['recovery'],mode=r['summary']['mode'],**e)
                                                    for r in results for e in r['epochs']])
            write_csv(out / 'rolling_commands.csv', [dict(recovery=r['summary']['recovery'],mode=r['summary']['mode'],**c)
                                                      for r in results for c in r['commands']])
    checks = check_shared_observation_history(results)
    write_csv(out / 'rolling_nonanticipativity_checks.csv', checks)
    print('Results: %s' % out, flush=True)


if __name__ == '__main__':
    main()
