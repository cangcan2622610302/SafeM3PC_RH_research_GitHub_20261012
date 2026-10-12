from copy import deepcopy
import pytest
from hsr_ttr.data import build_scalable_instance, build_scalable_uncertain_context
from hsr_ttr.paths import generate_path_pool
from hsr_ttr.milp import solve_lexicographic, MilpBuildError
from hsr_ttr.algorithms.command_refinement import refine_commands, interleave_unique
from hsr_ttr.validation import validate_solution


def test_direct_command_layer_matches_full_pipeline_under_common_caps():
    instance=build_scalable_instance(5,4)
    context=build_scalable_uncertain_context(instance)
    pool=generate_path_pool(instance,context)
    ref=solve_lexicographic(instance,context,pool,algorithm='reference',objective_mode='service_only',time_limit=20)
    kwargs=dict(algorithm='test',service_reference=ref.objective.service_loss,cancellation_reference=ref.cancellation_bound,epsilon=0.01,time_limit=20)
    old=solve_lexicographic(instance,context,pool,objective_mode='full',**kwargs)
    new=solve_lexicographic(instance,context,pool,objective_mode='command_only',**kwargs)
    assert new.status==old.status=='optimal'
    assert new.objective.as_tuple()==pytest.approx(old.objective.as_tuple())
    assert validate_solution(instance,context,pool,new).valid
    assert 'service' not in new.metadata['stage_status']
    assert new.metadata['stage_bounds']['command_total']['scope']=='full_path_pool'
    with pytest.raises(MilpBuildError):
        solve_lexicographic(instance,context,pool,algorithm='bad',objective_mode='command_only')


def test_retained_incumbent_is_not_mutated_or_misreported_as_optimal():
    instance=build_scalable_instance(5,4)
    context=build_scalable_uncertain_context(instance)
    pool=generate_path_pool(instance,context)
    ref=solve_lexicographic(instance,context,pool,algorithm='reference',objective_mode='service_only',time_limit=20)
    before=deepcopy(ref.to_dict())
    result=refine_commands(instance,context,pool,ref,kind='full',time_limit=0.0)
    assert result.metadata['seed_retained']
    assert result.status=='feasible_limit' and result.mip_gap is None
    assert ref.to_dict()==before
    assert validate_solution(instance,context,pool,result).valid


def test_interleaving_preserves_coverage_without_duplicate_budget_use():
    assert interleave_unique(['a','b','c'],['a','d'],['e','b'])==['a','e','b','d','c']
