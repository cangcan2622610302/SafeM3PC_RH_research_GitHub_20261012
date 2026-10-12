                                                                       

                                                                             
                                                                           
                                                                         
   

from __future__ import annotations

from hsr_ttr.algorithms.baselines import solve_b0, solve_b1
from hsr_ttr.algorithms.graph_lns import GraphLNSConfig, solve_graph_tlns
from hsr_ttr.algorithms.safe_m3pc_rh import SafeM3PCConfig, solve_safe_m3pc_rh
from hsr_ttr.data import build_small_context, build_small_instance
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.validation import validate_solution


def test_four_algorithms_return_model_valid_solutions_with_common_b1_cap():
    instance = build_small_instance()
    context = build_small_context(instance, epsilon=0.01)
    pool = generate_path_pool(instance, context)

    b0 = solve_b0(instance, context, pool, epsilon=0.0, time_limit=30.0)
    b1 = solve_b1(instance, context, pool, epsilon=0.0, time_limit=30.0)
    assert validate_solution(instance, context, pool, b0).valid
    assert validate_solution(instance, context, pool, b1).valid

    common_service = b1.service_optimum
    common_cancellations = b1.cancellation_bound
    c1 = solve_safe_m3pc_rh(
        instance,
        context,
        pool,
        config=SafeM3PCConfig(
            training_epochs=1,
            batch_size=512,
            candidate_widths=(8, 16, None),
        ),
        epsilon=0.01,
        time_limit=30.0,
        service_reference=common_service,
        cancellation_reference=common_cancellations,
    )
    c2 = solve_graph_tlns(
        instance,
        context,
        pool,
        config=GraphLNSConfig(training_epochs=1),
        epsilon=0.01,
        time_limit=30.0,
        service_reference=common_service,
        cancellation_reference=common_cancellations,
    )

    cap = 1.01 * common_service + 1.0e-6
    for solution in (c1, c2):
        report = validate_solution(instance, context, pool, solution)
        assert report.valid, report.messages()
        assert solution.objective.service_loss <= cap
        assert len(solution.cancelled_trains) <= common_cancellations
        assert solution.metadata["service_reference_external"] is True
        assert solution.metadata["cancellation_reference_external"] is True

