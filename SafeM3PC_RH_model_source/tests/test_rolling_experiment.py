                                                                            
from dataclasses import replace
from types import SimpleNamespace
from hsr_ttr.domain import RecoveryScenario, CommandEvent, TrainState
from hsr_ttr.rolling_experiment import observed_scenarios, prefix_compatible, issue_commands, state_snapshot


def test_observation_does_not_use_future_recovery():
    fast = RecoveryScenario('F', .25, 90, 120, 200, 16, 5)
    medium = RecoveryScenario('M', .50, 120, 165, 160, 20, 6)
    slow = RecoveryScenario('S', .25, 150, 210, 120, 27, 8)
    scenarios = (fast, medium, slow)
    initial = observed_scenarios(scenarios, [(0, 'S1', None), (60, 'S1', None)])
    assert [s.probability for s in initial] == [.25, .50, .25]
    updated = observed_scenarios(scenarios, [(90, 'S1', None)])
    assert [s.scenario_id for s in updated] == ['M', 'S']
    assert abs(updated[0].probability - 2/3) < 1e-10


def test_entered_section_cannot_change_exit_or_disappear():
    old = SimpleNamespace(cancelled=False, section_events={'AB': (10, 30, 3), 'BC': (35, 45, 3)})
    changed_exit = SimpleNamespace(cancelled=False, section_events={'AB': (10, 28, 3), 'BC': (35, 45, 3)})
    changed_future = SimpleNamespace(cancelled=False, section_events={'AB': (10, 30, 3), 'BC': (38, 48, 3)})
    missing = SimpleNamespace(cancelled=False, section_events={})
    assert not prefix_compatible(changed_exit, old, 20)
    assert not prefix_compatible(missing, old, 20)
    assert prefix_compatible(changed_future, old, 20)
    assert not prefix_compatible(changed_future, old, 40)


def test_cancel_is_irreversible_and_later_operating_train_cannot_vanish():
    cancelled = SimpleNamespace(cancelled=True, section_events={})
    operating = SimpleNamespace(cancelled=False, section_events={})
    assert prefix_compatible(cancelled, cancelled, 20)
    assert not prefix_compatible(operating, cancelled, 20)
    assert not prefix_compatible(cancelled, operating, 20)


def test_only_due_commands_are_published_and_duplicate_payload_is_not_reissued():
    instance = SimpleNamespace(time_step=1, trains=[SimpleNamespace(train_id='T1')])
    context = SimpleNamespace(now=10, train_states={'T1':SimpleNamespace(issued_hold_station='A',issued_release_time=30)})
    path = SimpleNamespace(cancelled=False, hold_station='A', release_time=40,
                           command_events=(CommandEvent('hold_change', 18, 1),))
    issued = {}
    assert not issue_commands(instance, context, {'T1':path}, issued, {}, set(), 15)
    first = issue_commands(instance, context, {'T1':path}, issued, {}, set(), 20)
    assert len(first) == 1 and first[0]['time'] == 18
    assert not issue_commands(instance, context, {'T1':path}, issued, {}, set(), 20)


def test_departure_exactly_at_update_is_not_already_executed():
    train=SimpleNamespace(train_id='T1',planned_departure={'A':10,'B':30})
    instance=SimpleNamespace(trains=[train],stations=[SimpleNamespace(name='A'),SimpleNamespace(name='B')],
                             disruption_section='AB',section=lambda _:SimpleNamespace(destination='B'))
    base=SimpleNamespace(train_states={'T1':TrainState('T1','A',0,0,'A',10,False)})
    previous={'T1':SimpleNamespace(cancelled=False,terminal_arrival=30,
                                  arrivals={'A':0,'B':30},section_events={'AB':(10,30,3)})}
    states,rows=state_snapshot(instance,base,previous,10,{})
    assert rows[0]['location']=='station:A'
    assert not states['T1'].already_crossed_disruption
    states,rows=state_snapshot(instance,base,previous,11,{})
    assert rows[0]['location']=='section:AB'
    assert states['T1'].already_crossed_disruption
