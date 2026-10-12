from hsr_ttr.data import (
    build_scalable_context,
    build_scalable_instance,
    build_scalable_uncertain_context,
)
from hsr_ttr.paths import generate_path_pool


def test_scalable_context_retains_future_train_availability():
    instance = build_scalable_instance(17, 80)
    context = build_scalable_context(instance)

    assert instance.horizon_end > 600
    assert context.train_states["080"].available_from > context.now
    assert context.train_states["080"].issued_hold_station is None
    assert context.train_states["080"].issued_release_time is None


def test_scalable_path_window_is_not_artificially_capped_at_t23_plus_60():
    instance = build_scalable_instance(17, 30)
    context = build_scalable_context(instance)
    pool = generate_path_pool(instance, context)
    scenario = context.scenarios[0]

    latest_release = max(
        path.release_time
        for path in pool.all_paths()
        if path.release_time is not None and not path.cancelled
    )
    assert latest_release > scenario.t23 + 60
    assert all(pool.paths(scenario.scenario_id, train.train_id) for train in instance.trains)


def test_scalable_uncertain_context_uses_common_states_and_probability_tree():
    instance = build_scalable_instance(17, 30)
    known = build_scalable_context(instance)
    uncertain = build_scalable_uncertain_context(instance)

    assert uncertain.train_states == known.train_states
    assert len(uncertain.scenarios) == 3
    assert abs(sum(item.probability for item in uncertain.scenarios) - 1.0) < 1.0e-12
