                                                                      

from __future__ import annotations

from typing import Optional

from ..domain import RailwayInstance, RollingContext, ScheduleSolution
from ..milp import solve_lexicographic
from ..paths import PathPool


def solve_b0(
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
        algorithm="B0",
        enforce_nonanticipativity=False,
        epsilon=epsilon,
        time_limit=time_limit,
        objective_mode=objective_mode,
    )


def solve_b1(
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
        algorithm="B1",
        enforce_nonanticipativity=True,
        epsilon=epsilon,
        time_limit=time_limit,
        objective_mode=objective_mode,
    )


__all__ = ["solve_b0", "solve_b1"]
