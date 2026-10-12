                                                              

from itertools import combinations

from hsr_ttr.data import build_small_context, build_small_instance
from hsr_ttr.milp import PathMasterMILP, _section_conflict_sets
from hsr_ttr.paths import generate_path_pool


def test_compact_cliques_cover_exact_pairwise_conflict_graph() -> None:
    instance = build_small_instance()
    context = build_small_context(instance)
    pool = generate_path_pool(instance, context)
    base = (next(item for item in context.scenarios if item.scenario_id == "base"),)
    artifacts = PathMasterMILP(instance, context, pool)._build(
        base,
        allowed_path_ids=None,
        fixed_selected_paths={},
        enforce_nonanticipativity=False,
    )
    pair_conflicts, _counts = _section_conflict_sets(
        artifacts.paths_by_group,
        artifacts.x_by_path,
        base,
        tuple(section.name for section in instance.sections),
    )
    compact_rows = [
        set(coefficients)
        for name, coefficients in zip(
            artifacts.rows.names, artifacts.rows.coefficients
        )
        if name.startswith("section_")
    ]
    covered = {
        tuple(sorted(pair))
        for row in compact_rows
        for pair in combinations(row, 2)
    }
                                                                             
                                                                           
    index_to_train = {
        artifacts.x_by_path[path.path_id]: path.train_id
        for paths in artifacts.paths_by_group.values()
        for path in paths
    }
    cross_train_covered = {
        pair
        for pair in covered
        if index_to_train[pair[0]] != index_to_train[pair[1]]
    }
    assert pair_conflicts <= cross_train_covered
    assert cross_train_covered <= pair_conflicts

