# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/gradient_based/lbfgs_gd.py
import jax
import jax.numpy as jnp
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective


class LBFGSGD(OptimizationAlgorithm):
    """L-BFGS via optax with a JIT-compiled loop. See reference link above for full docs."""

    algorithm_str = "lbfgs_gd"

    def __init__(self) -> None:
        pass

    @staticmethod
    def _linesearch_eval_count(opt_state) -> int:
        states = opt_state if isinstance(opt_state, tuple) else (opt_state,)
        for state in reversed(states):
            info = getattr(state, "info", None)
            num_steps = getattr(info, "num_linesearch_steps", None) if info else None
            if num_steps is not None:
                return int(jax.device_get(num_steps))
        return 0

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        **lbfgs_kwargs,
    ) -> None:
        obj = objective
        self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = init_params if init_params is not None else obj.random_params_unbounded()

        value_fn = obj.value_function(unbounded=True)
        value_and_grad_fn = jax.value_and_grad(value_fn)

        optimizer = optax.lbfgs(**lbfgs_kwargs)
        opt_state = optimizer.init(params)

        @jax.jit
        def _step(params, opt_state):
            loss, grads = value_and_grad_fn(params)
            updates, new_state = optimizer.update(
                grads, opt_state, params, value=loss, grad=grads, value_fn=value_fn,
            )
            new_params = jnp.asarray(optax.apply_updates(params, updates))
            return new_params, new_state, loss, grads

        warmup_state = optimizer.init(obj.random_params_unbounded())
        _, warmup_state, _, _ = _step(obj.random_params_unbounded(), warmup_state)
        _ = _step(obj.random_params_unbounded(), warmup_state)

        obj.start_logging()

        while not obj.budget_exceeded:
            prior_params = params
            params, opt_state, loss, grads = _step(params, opt_state)

            obj.log_evaluation(prior_params, loss, grads)

            for _ in range(self._linesearch_eval_count(opt_state)):
                if obj.budget_exceeded:
                    break
                obj.log_evaluation(prior_params, loss, grads)

            if patience is not None and obj.evals_since_improvement > patience:
                break