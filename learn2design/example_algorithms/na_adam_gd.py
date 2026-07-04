# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/gradient_based/na_adam_gd.py
import jax
import jax.numpy as jnp
import optax
from typing import Literal
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective

NoiseSchedule = Literal["linear", "exponential"]
NoiseInjection = Literal["update", "params"]


def _anneal_sigma(progress, sigma_start, sigma_end, schedule):
    progress = float(jnp.clip(progress, 0.0, 1.0))
    if schedule == "linear":
        return sigma_start + (sigma_end - sigma_start) * progress
    eps = 1e-12
    s0 = max(float(sigma_start), 0.0)
    s1 = max(float(sigma_end), 0.0)
    if s0 <= 0.0:
        return 0.0
    return s0 * (max(s1, eps) / max(s0, eps)) ** progress


def _clip_step_by_global_norm(step, max_norm):
    norm = jnp.linalg.norm(step)
    return step * jnp.minimum(1.0, max_norm / (norm + 1e-12))


def _cap_step_relative(step, reference, ratio):
    max_norm = ratio * jnp.linalg.norm(reference)
    return step * jnp.minimum(1.0, max_norm / (jnp.linalg.norm(step) + 1e-12))


class NAAdamGD(OptimizationAlgorithm):
    """Noisy-Annealing Adam. See reference link above for full docs."""

    algorithm_str = "na_adam_gd"

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        learning_rate: float = 0.1,
        patience: int | None = None,
        noise_std_start: float = 0.3,
        noise_std_end: float = 0.0,
        noise_schedule: NoiseSchedule = "exponential",
        noise_injection: NoiseInjection = "update",
        noise_clip_norm: float | None = None,
        noise_anneal_iters: int = 5000,
        noise_anneal_budget_fraction: float | None = None,
        noise_cap_relative_to_update: float | None = 0.25,
        noise_cap_start_iter: int = 500,
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

        iteration = 0
        while not obj.budget_exceeded:
            if noise_anneal_budget_fraction is not None:
                progress = obj.budget_progress_fraction / noise_anneal_budget_fraction
            else:
                progress = iteration / max(1, noise_anneal_iters)
            sigma_t = _anneal_sigma(progress, noise_std_start, noise_std_end, noise_schedule)

            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            updates, state = optimizer.update(grads, state, params)

            if sigma_t > 0:
                rng_key, subkey = jax.random.split(rng_key)
                noise_step = jax.random.normal(subkey, shape=params.shape) * sigma_t
                if noise_clip_norm is not None:
                    noise_step = _clip_step_by_global_norm(noise_step, noise_clip_norm)
                if (
                    noise_cap_relative_to_update is not None
                    and iteration >= noise_cap_start_iter
                ):
                    noise_step = _cap_step_relative(
                        noise_step, updates, float(noise_cap_relative_to_update)
                    )
                if noise_injection == "update":
                    updates = updates + noise_step

            params = optax.apply_updates(params, updates)

            if sigma_t > 0 and noise_injection == "params":
                params = params + noise_step

            iteration += 1