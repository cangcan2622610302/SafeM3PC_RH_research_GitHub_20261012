                                                                            

                                                                              
                                                                            
                                                                         
   
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from itertools import combinations
from copy import deepcopy
import hashlib
import json
import time

from .algorithms.safe_m3pc_rh import SafeM3PCConfig, solve_safe_m3pc_rh
from .domain import CommandEvent
from .paths import PathPool, _make_signature
from .validation import ValidationConfig, validate_selection


def check_shared_observation_history(results):
                                                                               
    checks = []
    for a, b in combinations(results, 2):
        if a['summary']['mode'] != b['summary']['mode']:
            continue
        for sa, sb in zip(a['snapshots'], b['snapshots']):
            if (sa['time'], sa['observed_state'], sa['posterior']) != (sb['time'], sb['observed_state'], sb['posterior']):
                break
            end = min(sa['next_time'], sb['next_time'])
            def prefix(s):
                return {tid:(p['cancelled'], sorted((sec, tuple(e)) for sec,e in p['section_events'].items() if e[0] < end))
                        for tid,p in s['actual_branch_plan'].items()}
            assert prefix(sa) == prefix(sb), 'Realized future affected a pre-observation movement'
            ca = sorted(json_key(c) for c in a['commands'] if sa['time'] <= c['time'] < end)
            cb = sorted(json_key(c) for c in b['commands'] if sb['time'] <= c['time'] < end)
            assert ca == cb, 'Realized future affected a pre-observation command'
            checks.append(dict(mode=a['summary']['mode'], first=a['summary']['recovery'],
                               second=b['summary']['recovery'], time=sa['time'], until=end, valid=True))
    return checks


def json_key(row):
    return repr(sorted(row.items()))


def observed_scenarios(scenarios, history):
                                                                              
    survivors = [s for s in scenarios if all(
        s.state_at(t) == state and (state != 'S2' or s.s2_speed_kmh == speed)
        for t, state, speed in history
    )]
    mass = sum(s.probability for s in survivors)
    if not mass:
        raise ValueError('Observation lies outside the specified recovery library')
    return tuple(replace(s, probability=s.probability / mass) for s in survivors)


def movement_prefix(path, until):
                                                                   
    return tuple(sorted((name, tuple(event)) for name, event in path.section_events.items()
                        if event[0] < until))


def prefix_compatible(candidate, previous, until):
    if previous.cancelled:
        return candidate.cancelled
    if candidate.cancelled:
                                                                               
                                                                              
        return False
    return movement_prefix(candidate, until) == movement_prefix(previous, until)


def state_snapshot(instance, base, previous, now, issued):
    states, rows = {}, []
    for train in instance.trains:
        tid = train.train_id
        old = base.train_states[tid]
        p = previous.get(tid)
        station, available, location = old.current_station, old.available_from, 'not_entered'
        completed = crossed = False
        if p and not p.cancelled:
            if p.terminal_arrival <= now:
                station, available, location, completed = instance.stations[-1].name, p.terminal_arrival, 'completed', True
            else:
                found = False
                for name, (dep, arr, _) in p.section_events.items():
                    if dep < now < arr:
                        station = instance.section(name).destination
                        available, location, found = arr, 'section:' + name, True
                        break
                if not found:
                    candidates = [(arr, s) for s, arr in p.arrivals.items() if arr <= now]
                    if candidates:
                        available, station = max(candidates)
                        location = 'station:' + station
            crossed = p.section_events[instance.disruption_section][0] < now
        elif p and p.cancelled:
            location = 'cancelled'
        payload = issued.get(tid, (old.issued_hold_station, old.issued_release_time))
        states[tid] = replace(old, current_station=station, available_from=max(now, available),
                              current_delay=max(0, now - train.planned_departure.get(station, now)),
                              issued_hold_station=payload[0], issued_release_time=payload[1],
                              completed=completed, already_crossed_disruption=crossed)
        rows.append(dict(train_id=tid, time=now, location=location, next_station=station,
                         available_from=available, completed=completed, crossed=crossed))
    return states, rows


def make_epoch_pool(instance, base_pool, context, previous, previous_branches, issued,
                    committed_until):
    by_group, by_id, warm = {}, {}, set()
    for scenario in context.scenarios:
        groups = by_group.setdefault(scenario.scenario_id, {})
        for train in instance.trains:
            tid = train.train_id
            state = context.train_states[tid]
            old = previous.get(tid)
            paths = []
            for p in base_pool.paths(scenario.scenario_id, tid):
                cutoff = committed_until
                if (old is not None and not old.cancelled
                        and issued.get(tid) == (old.hold_station, old.release_time)
                        and old.departures[old.hold_station] < context.now + 6):
                    cutoff = max(cutoff, old.departures[old.hold_station] + 1)
                if old is not None and not prefix_compatible(p, old, cutoff):
                    continue
                events = []
                if p.cancelled:
                    if not old:
                        events = [CommandEvent('cancel', context.now, 1)]
                elif not state.already_crossed_disruption and not state.completed:
                    hold, release = issued.get(tid, (state.issued_hold_station, state.issued_release_time))
                    lead = 2 * instance.time_step
                    if p.hold_station != hold:
                        first = min((v for v in p.departures.values() if v >= context.now), default=context.now)
                        events.append(CommandEvent('station_change', max(context.now, first - lead), 1))
                    if release is None or abs(p.release_time - release) > instance.command_change_threshold:
                        events.append(CommandEvent('hold_change', max(context.now, p.departures[p.hold_station] - lead), 1))
                signature = _make_signature(p.cancelled, False, state, p.departures, p.section_events,
                                           context.next_observation, events, p.hold_station, p.release_time)
                if (not p.cancelled and p.release_time >= context.now
                        and p.release_time - 2 * instance.time_step <= context.next_observation):
                    signature += ';ORDER_RELEASE@%d' % p.release_time
                q = replace(p, command_events=tuple(events), near_term_signature=signature,
                            station_occupancy=tuple(x for x in p.station_occupancy if x[1] >= context.now))
                paths.append(q)
                by_id[q.path_id] = q
                if previous_branches.get((scenario.scenario_id, tid)) == q.path_id:
                    warm.add(q.path_id)
            if not paths:
                raise ValueError('No prefix-compatible continuation: %s/%s at %s' %
                                 (scenario.scenario_id, tid, context.now))
            groups[tid] = paths
    return PathPool(by_group, by_id), warm


def issue_commands(instance, context, selected, issued, order_ledger, cancelled, until):
                                                                         
    rows = []
    for tid, p in selected.items():
        for event in p.command_events:
            if not context.now <= event.issue_time < until:
                continue
            payload = (p.hold_station, p.release_time)
            if event.command_type == 'cancel':
                if tid in cancelled:
                    continue
                cancelled.add(tid)
            else:
                prior = issued.get(tid, (context.train_states[tid].issued_hold_station,
                                         context.train_states[tid].issued_release_time))
                if event.command_type == 'station_change':
                    if prior[0] == p.hold_station:
                        continue
                                                                             
                                                                       
                    issued[tid] = (p.hold_station, prior[1])
                else:
                    if prior == payload:
                        continue
                    issued[tid] = payload
            rows.append(dict(time=event.issue_time, command_type=event.command_type,
                             train=tid, other_train='', hold_station=p.hold_station,
                             release_time=p.release_time if event.command_type != 'station_change' else None))
    operating = [t for t in instance.trains if not selected[t.train_id].cancelled]
    for a, b in combinations(operating, 2):
        pa, pb = selected[a.train_id], selected[b.train_id]
        ra, rb = pa.release_time, pb.release_time
        if min(ra, rb) < context.now:
            continue
        issue = max(context.now, min(ra, rb) - 2 * instance.time_step)
        if issue >= until:
            continue
        key = (a.train_id, b.train_id)
        desired = ra < rb
        if order_ledger[key] != desired:
            rows.append(dict(time=issue, command_type='order_change', train=a.train_id,
                             other_train=b.train_id, hold_station='', release_time=''))
            order_ledger[key] = desired
    return rows


def run_closed_loop(case, base_pool, model, truth, mode='service_only', epsilon=.01,
                    interval=30, time_limit=20, config=None, policy_cache=None):
    instance, base = case.instance, case.context
    config = config or SafeM3PCConfig(candidate_widths=(32, 64, None), training_epochs=4,
                                      use_hdfs_backbone=False, stop_after_first_feasible=True)
    now = base.now
    previous, previous_branches, issued, cancelled = {}, {}, {}, set()
    order_ledger = {}
    for a, b in combinations(instance.trains, 2):
        sa, sb = base.train_states[a.train_id], base.train_states[b.train_id]
        order_ledger[(a.train_id, b.train_id)] = (
            sa.issued_release_time < sb.issued_release_time
            if sa.issued_release_time != sb.issued_release_time else a.original_order < b.original_order)
    observations, epochs, snapshots, commands, all_states = [], [], [], [], []
    committed_until = now
    solve_seconds = 0.
    policy_cache = {} if policy_cache is None else policy_cache
    for k in range(100):
                                                                              
        observed_state = truth.state_at(now)
        observations.append((now, observed_state, truth.s2_speed_kmh if observed_state == 'S2' else None))
        posterior = observed_scenarios(base.scenarios, observations)
        future_events = [t for s in posterior for t in (s.t12, s.t23) if t > now]
        next_time = min([now + interval, instance.horizon_end + 1] + future_events)
        states, state_rows = state_snapshot(instance, base, previous, now, issued)
        all_states.extend(state_rows)
        ctx = replace(base, now=now, freeze_until=now + 6, next_observation=next_time - 1,
                      train_states=states, scenarios=posterior)
        started = time.perf_counter()
        pool, warm = make_epoch_pool(instance, base_pool, ctx, previous, previous_branches,
                                     issued, committed_until)
        epoch_config = replace(config, use_constructive_backbone=False) if previous else config
        cache_key = (mode, tuple(observations))
                                                                           
                                                                              
        fingerprint = hashlib.sha256(json.dumps(dict(context=asdict(ctx),
            paths=[(p.path_id, p.near_term_signature, [(e.command_type,e.issue_time) for e in p.command_events])
                   for p in pool.by_id.values()], warm=sorted(warm)), sort_keys=True).encode()).hexdigest()
        reused = cache_key in policy_cache
        if reused:
            cached_fingerprint, phase1, solution, node_seconds = deepcopy(policy_cache[cache_key])
            assert cached_fingerprint == fingerprint, 'Unobserved future changed optimizer input'
        else:
            phase1 = solve_safe_m3pc_rh(instance, ctx, pool, config=epoch_config, pretrained_model=model,
                                   required_path_ids=warm, objective_mode='service_only',
                                   epsilon=0., time_limit=time_limit)
            solution = phase1
            if mode == 'full':
                solution = solve_safe_m3pc_rh(instance, ctx, pool, config=epoch_config, pretrained_model=model,
                                         required_path_ids=set(phase1.selected_paths.values()) | warm,
                                         objective_mode='full', epsilon=epsilon, time_limit=time_limit,
                                         service_reference=phase1.objective.service_loss,
                                         cancellation_reference=phase1.cancellation_bound)
                if solution.objective.service_loss > (1 + epsilon) * phase1.objective.service_loss + 1e-5:
                    raise AssertionError('Per-epoch service allowance violated')
            node_seconds = time.perf_counter() - started
            policy_cache[cache_key] = deepcopy((fingerprint, phase1, solution, node_seconds))
                                                                                  
                                                                                          
        validation = validate_selection(instance, ctx, solution.selected_paths, pool=pool,
                                          recorded_objective=solution.objective,
                                          config=ValidationConfig(check_path_command_semantics=False))
        validation.raise_if_invalid()
        elapsed = node_seconds
        solve_seconds += elapsed
        selected = {t.train_id: pool.by_id[solution.selected_paths[(truth.scenario_id, t.train_id)]]
                    for t in instance.trains}
        if previous:
            assert all(prefix_compatible(selected[t], previous[t], committed_until) for t in previous)
                                                                              
        for tid in selected:
            assert len({pool.by_id[solution.selected_paths[(s.scenario_id, tid)]].near_term_signature
                        for s in posterior}) == 1
        changes = sum(previous[t].release_time != p.release_time or previous[t].hold_station != p.hold_station
                      for t, p in selected.items() if t in previous)
        due = issue_commands(instance, ctx, selected, issued, order_ledger, cancelled, next_time)
        commands.extend(due)
        prediction_id = max(posterior, key=lambda s: s.probability).scenario_id
        forecast = {t.train_id: pool.by_id[solution.selected_paths[(prediction_id, t.train_id)]]
                    for t in instance.trains}
        snapshots.append(dict(epoch=k, time=now, next_time=next_time, observed_state=observed_state,
                              forecast_scenario=prediction_id, posterior={s.scenario_id:s.probability for s in posterior},
                              forecast={t:asdict(p) for t,p in forecast.items()},
                              actual_branch_plan={t:asdict(p) for t,p in selected.items()}))
        epochs.append(dict(epoch=k, time=now, next_time=next_time, state=observed_state,
                           posterior_branches=len(posterior), status=solution.status,
                           service_reference=phase1.objective.service_loss,
                           planned_expected_service=solution.objective.service_loss,
                           new_commands=len(due), changed_train_plans=changes,
                           completed=sum(p.terminal_arrival is not None and p.terminal_arrival <= now for p in selected.values()),
                           prefix_valid=True, independently_valid=validation.valid,
                           runtime_seconds=elapsed, shared_policy_node_reused=reused,
                           candidate_count=len(pool.by_id)))
        print('%s %s epoch=%d t=%d %s branches=%d service=%.2f commands=%d sec=%.2f' %
              (truth.scenario_id, mode, k, now, observed_state, len(posterior),
               solution.objective.service_loss, len(due), elapsed), flush=True)
        previous, previous_branches = selected, dict(solution.selected_paths)
                                                                                  
        committed_until = max(next_time, now + 6)
        now = next_time
        if all(p.cancelled or p.terminal_arrival <= now for p in selected.values()):
            break
    else:
        raise RuntimeError('Rolling simulation did not complete')
                                                                                        
    final_context = replace(base, scenarios=(replace(truth, probability=1.),))
    final = {(truth.scenario_id, tid): replace(p, station_occupancy=base_pool.by_id[p.path_id].station_occupancy,
                                                command_events=(), near_term_signature='FINAL')
             for tid,p in previous.items()}
    final_check = validate_selection(instance, final_context, final,
                                      config=ValidationConfig(check_path_command_semantics=False))
    final_check.raise_if_invalid()
    counts = Counter(row['command_type'] for row in commands)
    buckets = Counter(row['time'] for row in commands)
    delay = sum(p.terminal_delay for p in previous.values() if not p.cancelled)
    summary = dict(macro=case.spec.macro_id, recovery=truth.scenario_id, mode=mode,
                   epsilon=epsilon if mode == 'full' else 0., train_count=len(instance.trains),
                   total_delay=delay, cancelled_count=len(cancelled),
                   service_loss=delay + instance.cancellation_penalty * len(cancelled),
                   command_total=len(commands), command_peak=max(buckets.values(), default=0),
                   hold_commands=counts['hold_change'], order_commands=counts['order_change'],
                   station_commands=counts['station_change'], cancel_commands=counts['cancel'],
                   decision_epochs=len(epochs), total_online_seconds=solve_seconds,
                   max_epoch_seconds=max(e['runtime_seconds'] for e in epochs),
                   all_independently_valid=True, all_prefix_valid=True,
                   final_arrival=max(p.terminal_arrival or 0 for p in previous.values()))
    return dict(summary=summary, epochs=epochs, snapshots=snapshots, commands=commands, states=all_states)
