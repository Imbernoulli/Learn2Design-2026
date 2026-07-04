from collections import deque

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective


class JAXOnePlusOneES(OptimizationAlgorithm):
    """(1+1)-ES with the 1/5 success rule. See dfbench's `jax_es.py` for full docs."""

    algorithm_str = "jax_1p1es"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        sigma_min: float = 1e-10,
        success_window: int | None = None,
        max_iterations: int | None = None,
    ) -> None:
        obj = objective
        problem = obj.problem
        _, rng = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb = jnp.asarray(problem.bounds[0])
        ub = jnp.asarray(problem.bounds[1])
        n = problem.n_params

        if init_params is None:
            rng, init_rng = jax.random.split(rng)
            x = jax.random.uniform(init_rng, shape=(n,), minval=lb, maxval=ub)
        else:
            x = jnp.clip(jnp.asarray(init_params), lb, ub)

        scale = ub - lb
        sigma = float(sigma0 if sigma0 is not None else 0.3)
        window = success_window if success_window is not None else max(10, 10 * n)

        obj.warmup_value()
        obj.start_logging()

        fx = obj.value(x)
        successes = deque(maxlen=window)
        step = 0

        while not obj.budget_exceeded:
            if max_iterations is not None and step >= max_iterations:
                break
            if sigma < sigma_min:
                break

            rng, noise_rng = jax.random.split(rng)
            z = jax.random.normal(noise_rng, shape=(n,))
            y = jnp.clip(x + sigma * scale * z, lb, ub)

            fy = obj.value(y)
            success = int(float(fy) <= float(fx))
            successes.append(success)

            if success:
                x = y
                fx = fy

            if len(successes) == window and (step + 1) % window == 0:
                p_succ = sum(successes) / window
                if p_succ > 0.2:
                    sigma = sigma * float(jnp.exp(jnp.array(1.0 / n)))
                elif p_succ < 0.2:
                    sigma = sigma * float(jnp.exp(jnp.array(-1.0 / (5.0 * n))))

            step += 1