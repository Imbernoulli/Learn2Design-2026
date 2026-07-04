import math

import jax
import optax
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective


class SAGD(OptimizationAlgorithm):
    """Simulated-Annealing Gradient Descent (arXiv:2107.07558). See dfbench's `sa_gd.py`."""

    algorithm_str = "sa_gd"

    def __init__(self) -> None:
        pass

    def _transition_probability(self, delta_e, epoch, T0, learning_rate,
                               use_double_annealing, lr_decay, initial_lr):
        n = max(epoch, 0)
        eps = 1e-10
        delta_e = max(abs(delta_e), eps)

        if use_double_annealing:
            alpha = math.e
            beta = 0.5772
            frac_power = math.log(n + 2) ** (-1.0 / alpha)
            current_lr = initial_lr * (lr_decay ** n)
            temperature = T0 * current_lr * math.log(n + 2)
            numerator = delta_e ** frac_power
            ratio = numerator / max(temperature, eps)
            outer_exp = beta * math.log(n + 2)
            exponent = -(ratio ** outer_exp)
        else:
            temperature = T0 * learning_rate * math.log(n + 2)
            exponent = -delta_e / max(temperature, eps)

        return min(max(math.exp(max(exponent, -100)), 0.0), 1.0)

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        patience: int | None = None,
        T0: float = 15.0,
        sigma: float = 1.0,
        max_ascent_prob: float = 0.33,
        use_double_annealing: bool = False,
        lr_decay: float = 1.0,
        **adam_kwargs,
    ) -> None:
        obj = objective
        _, rng_key = self.prepare(obj, unbounded=True, random_seed=random_seed)

        params = init_params if init_params is not None else obj.random_params_unbounded()

        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(learning_rate, **adam_kwargs)
        )
        state = optimizer.init(params)

        obj.warmup_value_and_grad()
        obj.start_logging()

        prev_loss = 0.0
        iteration = 0

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            delta_e = abs(float(loss) - prev_loss)
            current_lr = learning_rate * (lr_decay ** iteration)
            trans_prob = self._transition_probability(
                delta_e, iteration, T0, current_lr,
                use_double_annealing, lr_decay, learning_rate,
            )
            ascent_prob = min(1.0 - trans_prob, max_ascent_prob)

            rng_key, subkey = jax.random.split(rng_key)
            random_val = float(jax.random.uniform(subkey))

            updates, state = optimizer.update(grads, state, params)

            if random_val < ascent_prob:
                updates = jax.tree.map(lambda x: -sigma * x, updates)

            params = optax.apply_updates(params, updates)
            prev_loss = float(loss)
            iteration += 1