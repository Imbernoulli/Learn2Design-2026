# Reference: https://github.com/artificial-scientist-lab/Differometor-Benchmark/blob/main/src/dfbench/algorithms/evolutionary/jax_es.py
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench.core.algorithm import OptimizationAlgorithm
from dfbench.core.objective import Objective


class JAXMuLambdaES(OptimizationAlgorithm):
    """(μ,λ)-ES with truncation selection. See reference link above for full docs."""

    algorithm_str = "jax_mu_lambda_es"

    def __init__(self, batch_size: int = 1) -> None:
        self.batch_size = batch_size

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "n_params"] | None = None,
        random_seed: int | None = None,
        sigma0: float | None = None,
        sigma_min: float = 1e-10,
        mu: int = 10,
        lam: int = 50,
        max_iterations: int | None = None,
    ) -> None:
        if mu >= lam:
            raise ValueError(f"(μ,λ)-ES requires mu < lam, got mu={mu}, lam={lam}.")

        obj = objective
        problem = obj.problem
        _, rng = self.prepare(obj, unbounded=False, random_seed=random_seed)

        lb = jnp.asarray(problem.bounds[0])
        ub = jnp.asarray(problem.bounds[1])
        n = problem.n_params

        if init_params is None:
            rng, init_rng = jax.random.split(rng)
            mean = jax.random.uniform(init_rng, shape=(n,), minval=lb, maxval=ub)
        else:
            mean = jnp.clip(jnp.asarray(init_params), lb, ub)

        scale = ub - lb
        sigma = float(sigma0 if sigma0 is not None else 0.3)

        target_succ = mu / lam
        c_s = 1.0 / max(n, 1)
        d_sigma = 1.0
        p_succ = target_succ
        prev_mean_loss = float("inf")

        batch_size = self.batch_size
        obj.warmup_vmap_value(batch_size=min(batch_size, lam))
        obj.start_logging()

        iteration = 0
        while not obj.budget_exceeded:
            if max_iterations is not None and iteration >= max_iterations:
                break
            if sigma < sigma_min:
                break

            rng, noise_rng = jax.random.split(rng)
            noise = jax.random.normal(noise_rng, shape=(lam, n))
            offspring = jnp.clip(mean[None, :] + sigma * scale[None, :] * noise, lb, ub)

            chunks = [
                obj.vmap_value(offspring[i : i + batch_size])
                for i in range(0, lam, batch_size)
            ]
            losses = jnp.concatenate(chunks)

            if obj.budget_exceeded:
                break

            losses = jnp.where(jnp.isfinite(losses), losses, jnp.float32(1e30))

            # Truncation selection: keep best μ (comma semantics)
            sorted_idx = jnp.argsort(losses)[:mu]
            selected = offspring[sorted_idx]
            mean = jnp.clip(jnp.mean(selected, axis=0), lb, ub)

            # Cumulative step-size adaptation
            mean_selected_loss = float(jnp.mean(losses[sorted_idx]))
            improved = 1.0 if mean_selected_loss < prev_mean_loss else 0.0
            prev_mean_loss = mean_selected_loss

            p_succ = (1.0 - c_s) * p_succ + c_s * improved
            sigma = sigma * float(jnp.exp(jnp.array((p_succ - target_succ) / d_sigma)))
            sigma = max(sigma, sigma_min)

            iteration += 1