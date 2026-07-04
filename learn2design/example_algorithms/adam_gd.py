# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/gradient_based/adam_gd.py
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective


class AdamGD(OptimizationAlgorithm):
    """Adam with gradient clipping. See reference link above for full docs."""

    algorithm_str = "adam_gd"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        learning_rate: float = 0.1,
        **adam_kwargs,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = init_params if init_params is not None else obj.random_params_unbounded()

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        state = optimizer.init(params)

        obj.warmup_value_and_grad()
        obj.start_logging()

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, state = optimizer.update(grads, state, params)
            params = optax.apply_updates(params, updates)