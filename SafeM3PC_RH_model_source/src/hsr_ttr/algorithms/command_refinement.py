                                                                           

                                                                           
                                                                             
                                                                         
   
from copy import deepcopy
import time
import numpy as np

from .safe_m3pc_rh import rank_candidate_paths, safe_allowed_path_ids
from ..milp import solve_lexicographic, MilpBuildError, MilpSolveError
from ..validation import validate_solution


def interleave_unique(*sequences):
                                                                                
    out, seen = [], set()
    for i in range(max((len(s) for s in sequences), default=0)):
        for seq in sequences:
            if i < len(seq) and seq[i] not in seen:
                out.append(seq[i])
                seen.add(seq[i])
    return out


def candidate_ranking(instance, context, pool, model, kind='learned', seed=0):
    if kind in ('learned', 'diverse', 'no_prior'):
        learned = rank_candidate_paths(model, instance, context, pool,
                                       include_command_terms=kind != 'no_prior')
        if kind != 'diverse':
            return learned
    rng = np.random.default_rng(seed)
    result = {}
    for scenario in context.scenarios:
        for train in instance.trains:
            key = (scenario.scenario_id, train.train_id)
            paths = list(pool.paths(*key))
            issued = context.train_states[train.train_id].issued_release_time
            def service(p):
                return instance.cancellation_penalty * int(p.cancelled) + p.terminal_delay
            def prior(p):
                deviation = 10**9 if p.release_time is None and not p.cancelled else 0 if issued is None or p.release_time is None else abs(p.release_time-issued)
                return p.path_command_count, deviation
            if kind == 'rule':
                ordered = sorted(paths, key=lambda p: (*prior(p), service(p), p.path_id))
                result[key] = [p.path_id for p in ordered]
            elif kind == 'random':
                                                                       
                                                                            
                scores = {p.path_id: float(rng.random()) for p in paths}
                result[key] = [p.path_id for p in sorted(paths, key=lambda p: (*prior(p), scores[p.path_id], p.path_id))]
            elif kind == 'diverse':
                service_ids = [p.path_id for p in sorted(paths, key=lambda p: (service(p), *prior(p), p.path_id))]
                                                                        
                                                                             
                groups = {}
                for p in paths:
                    groups.setdefault(p.hold_station or '', []).append(p)
                coverage_groups = []
                for hold in sorted(groups):
                    g = sorted(groups[hold], key=lambda p: (p.release_time or -1, p.path_id))
                    queue, indices = [(0,len(g)-1)], []
                    while queue:
                        a,b = queue.pop(0)
                        if a>b: continue
                        m=(a+b)//2
                        indices.append(m)
                        queue.extend(((a,m-1),(m+1,b)))
                    coverage_groups.append([g[i].path_id for i in indices])
                coverage = interleave_unique(*coverage_groups)
                result[key] = interleave_unique(learned[key], service_ids, coverage)
            else:
                raise ValueError('Unknown ranking kind: '+kind)
    return result


def refine_commands(instance, context, pool, reference, *, model=None,
                    kind='diverse', width=64, epsilon=0.01, time_limit=10.0,
                    seed=0, keep_seed=True, backend='highs', incumbent=None):
                                                                          

                                                                           
                                                                         
                                                           
       
    started = time.perf_counter()
    start_check = validate_solution(instance, context, pool, reference)
    start_check.raise_if_invalid()
    base = reference if incumbent is None else incumbent
    validate_solution(instance, context, pool, base).raise_if_invalid()
    cap = (1+epsilon)*reference.objective.service_loss
    if base.objective.service_loss > cap+1e-5 or len(base.cancelled_trains)>reference.cancellation_bound:
        raise ValueError('Incumbent violates the common service/cancellation cap.')
    ranking_started = time.perf_counter()
    if kind == 'full':
        allowed = None
    else:
        ranking = candidate_ranking(instance, context, pool, model, kind, seed)
        seeds = set(reference.selected_paths.values()) | set(base.selected_paths.values()) if keep_seed else set()
        allowed = safe_allowed_path_ids(ranking, pool, width, seeds)
    ranking_seconds = time.perf_counter()-ranking_started
    candidate, error = None, None
    try:
        remaining = time_limit-(time.perf_counter()-started)
        if remaining <= 0: raise MilpSolveError('Budget expired during validation/ranking.')
        candidate = solve_lexicographic(instance, context, pool,
            algorithm='C1+' if kind=='diverse' else kind,
            allowed_path_ids=allowed, epsilon=epsilon,
            service_reference=reference.objective.service_loss,
            cancellation_reference=reference.cancellation_bound,
            objective_mode='command_only', time_limit=remaining, backend=backend)
        validate_solution(instance, context, pool, candidate).raise_if_invalid()
        if candidate.objective.service_loss>cap+1e-5:
            raise ValueError('Command solve exceeds common service cap.')
    except (MilpBuildError, MilpSolveError) as exc:
        error = str(exc)
    def key(s):
        return (s.objective.command_total, s.objective.command_peak, s.objective.service_loss)
    accepted = candidate is not None and (not keep_seed or key(candidate)<=key(base))
    if not accepted and not keep_seed:
        raise MilpSolveError(error or 'No admissible candidate and seed retention disabled.')
    chosen = deepcopy(candidate if accepted else base)
    if not accepted:
        chosen.status='feasible_limit'
        chosen.mip_gap=None
        chosen.metadata={}
    chosen.algorithm='C1+' if kind=='diverse' else kind
    chosen.service_optimum=reference.objective.service_loss
    chosen.cancellation_bound=reference.cancellation_bound
    chosen.runtime_seconds=time.perf_counter()-started
    chosen.metadata.update({
        'epsilon':epsilon, 'objective_mode':'command_only',
        'service_reference_external':True, 'cancellation_reference_external':True,
        'refinement_kind':kind, 'candidate_width':width,
        'candidate_domain_count':len(pool.by_id) if allowed is None else len(allowed),
        'candidate_ranking_seconds':ranking_seconds,
        'candidate_returned':candidate is not None, 'candidate_accepted':accepted,
        'seed_retained':not accepted, 'keep_seed':keep_seed, 'refinement_error':error,
        'service_cap':cap, 'reference_service':reference.objective.service_loss,
        'reference_status':reference.status,
        'optimality_scope':'full_path_pool_conditional' if kind=='full' else 'restricted_domain_conditional',
    })
    validate_solution(instance, context, pool, chosen).raise_if_invalid()
    return chosen
