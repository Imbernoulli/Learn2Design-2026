"""Standalone SGD with momentum (SGDM) optimizer via Optax.

Self-contained version of dfbench's `OptaxSGDM` that does not depend on the
shared `OptaxAlgorithm` base class. Operates in unbounded (sigmoid-transformed)
space by default.
"""

from __future__ import annotations

import jax
import optax
import jax.numpy as jnp
from jaxtyping import Array, Float

from dfbench import AlgorithmType, Objective, OptimizationAlgorithm


# Maximum consecutive NaN/Inf evaluations before falling back to the
# best-known point (or a fresh random start).
_MAX_NAN_STREAK: int = 20

# NaN perturbation base scale. Doubles each consecutive miss.
_NAN_PERTURB_BASE: float = 1e-10


def _is_nonfinite(loss, grads) -> bool:
    """Return True if loss or any gradient entry is NaN or Inf."""
    return bool(not jnp.isfinite(loss) or not jnp.all(jnp.isfinite(grads)))


class OptaxSGDM(OptimizationAlgorithm):
    """SGD with momentum via Optax."""

    algorithm_str: str = "optax_sgdm"
    algorithm_type: AlgorithmType = AlgorithmType.GRADIENT_BASED

    def __init__(self) -> None:
        pass

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        patience: int | None = None,
        learning_rate: float = 0.1,
        grad_clip_norm: float | None = 1.0,
        momentum: float = 0.9,
        **kwargs,
    ) -> None:
        obj = objective

        self.prepare(obj, unbounded=True, random_seed=random_seed)

        if init_params is None:
            params = obj.random_params_unbounded() * (1 + 1e-8)
        else:
            params = init_params

        # Build optimizer chain: optional gradient clipping + momentum SGD.
        parts: list[optax.GradientTransformation] = []
        if grad_clip_norm is not None:
            parts.append(optax.clip_by_global_norm(grad_clip_norm))
        parts.append(optax.sgd(learning_rate, momentum=momentum, nesterov=False))
        optimizer = optax.chain(*parts) if len(parts) > 1 else parts[0]
        opt_state = optimizer.init(params)

        obj.warmup_value_and_grad()
        obj.start_logging()

        nan_streak = 0
        rng_key = jax.random.PRNGKey(random_seed if random_seed is not None else 0)

        while not obj.budget_exceeded:
            loss, grads = obj.value_and_grad(params)

            if patience is not None and obj.evals_since_improvement > patience:
                break

            # NaN / Inf guard: if the loss or gradient is non-finite, perturb the
            # current point with escalating noise. After _MAX_NAN_STREAK misses,
            # jump to the best-known point (or a fresh random start) and reset.
            if _is_nonfinite(loss, grads):
                nan_streak += 1
                rng_key, sub_key = jax.random.split(rng_key)

                if nan_streak > _MAX_NAN_STREAK:
                    best = obj.best_params
                    if best is not None:
                        params = (
                            best
                            + jax.random.normal(sub_key, best.shape)
                            * _NAN_PERTURB_BASE
                        )
                    else:
                        params = obj.random_params_unbounded()
                    opt_state = optimizer.init(params)
                    nan_streak = 0
                else:
                    # Escalating perturbation: doubles each consecutive miss.
                    scale = _NAN_PERTURB_BASE * (2 ** min(nan_streak, 30))
                    params = params + jax.random.normal(sub_key, params.shape) * scale
                continue

            nan_streak = 0
            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)