# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/evolutionary/random_search.py
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective


class RandomSearch(OptimizationAlgorithm):
    """Uniform random sampling within bounds. See reference link above for full docs."""

    algorithm_str = "random_search"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=False, random_seed=random_seed)

        obj.warmup_value()
        obj.start_logging()

        while not obj.budget_exceeded:
            obj.value(obj.random_params())