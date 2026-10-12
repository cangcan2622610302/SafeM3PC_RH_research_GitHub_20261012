                                                                            

from hsr_ttr.algorithms.safe_m3pc_rh import (
    SafeM3PCConfig,
    build_hdfs_fifo_backbone,
    build_hdfs_fifo_backbone_schedules,
    build_offline_trajectory_array,
    deterministic_rank_candidate_paths,
    safe_allowed_path_ids,
    train_masked_model,
)
from hsr_ttr.data import build_small_context, build_small_instance
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.validation import validate_selection


def test_masked_trajectory_shapes_and_safe_domain() -> None:
    instance = build_small_instance()
    context = build_small_context(instance)
    pool = generate_path_pool(instance, context)
    array = build_offline_trajectory_array(instance, context, pool)
    assert array.shape == (len(pool.by_id), 3, 12)

    ranking = deterministic_rank_candidate_paths(instance, context, pool)
    allowed = safe_allowed_path_ids(ranking, pool, width=4)
                                                                           
    for scenario in context.scenarios:
        for train in instance.trains:
            assert any(
                path.path_id in allowed
                for path in pool.paths(scenario.scenario_id, train.train_id)
            )


def test_solver_independent_backbone_is_complete_and_feasible() -> None:
    instance = build_small_instance()
    context = build_small_context(instance)
    pool = generate_path_pool(instance, context)
    schedules = build_hdfs_fifo_backbone_schedules(instance, context, pool)
    backbone = build_hdfs_fifo_backbone(instance, context, pool)
    assert schedules
    for selected in schedules:
        report = validate_selection(instance, context, selected, pool=pool)
        assert report.valid, report.messages()
        assert len(selected) == len(context.scenarios) * len(instance.trains)
        assert set(selected.values()) <= backbone


def test_masked_model_one_epoch_smoke() -> None:
    instance = build_small_instance()
    context = build_small_context(instance)
    pool = generate_path_pool(instance, context)
    _model, losses = train_masked_model(
        instance,
        context,
        pool,
        SafeM3PCConfig(training_epochs=1, batch_size=1024),
    )
    assert len(losses) == 1
    assert losses[0] >= 0.0
