                                                                       

from __future__ import annotations

from typing import Optional

from ..domain import RailwayInstance, RollingContext, ScheduleSolution
from ..milp import solve_lexicographic
from ..paths import PathPool


def solve_gurobi_exact(
    instance: RailwayInstance,
    context: RollingContext,
    pool: PathPool,
    epsilon: Optional[float] = None,
    time_limit: Optional[float] = None,
    objective_mode: str = "full",
) -> ScheduleSolution:
                                                                    

                                                                         
                                                                              
                                                                             
                                                                          
       

    return solve_lexicographic(
        instance,
        context,
        pool,
        algorithm="Gurobi-Exact",
        scenarios=context.scenarios,
        enforce_nonanticipativity=True,
        epsilon=epsilon,
        time_limit=time_limit,
        backend="gurobi",
        objective_mode=objective_mode,
    )


__all__ = ["solve_gurobi_exact"]
