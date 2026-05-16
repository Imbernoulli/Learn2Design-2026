# Learn2Design-2026

A black-box optimization competition for gravitational-wave detector design.

You are given a fixed quasi-universal interferometer with a fixed topology and
continuous parameters (laser powers, mirror reflectivities, grid distances, ...).
Your job is to find the parameter vector that maximises detector sensitivity,
within a fixed compute budget.

You submit algorithms. The best algorithm over 10 problems wins.

> **Status:** Pre-launch. The starting kit is being finalised. Expect
> breaking changes until the first month's problems are released.

---

## How it works

- Each month we publish 10 new public topologies. Submissions are scored on the 
  average best loss across all 10 of them.
- Your score will get published to that month's leaderboard.
- Every topology run gets 4 hours of wall-clock time on a single A100 GPU with 
  an AMD EPYC 7302 CPU.
- The final leaderboard is decided on 10 held-out private topologies that
  are never published.
- Your final score will be the mean of the best loss reached on the 10 private 
  problems. Lower is better.

A "topology" fixes the choice of optical components for a UIFO; you only optimize 
the continuous parameters attached to it. These could be laser power, mirror 
reflectivity, grid distance, etc..

---

## Install

```bash
git clone https://github.com/artificial-scientist-lab/Learn2Design-2026
cd Learn2Design-2026
pip install -e .
```

This pulls in [`dfbench`](https://pypi.org/project/dfbench/) (the benchmark
framework) and [`differometor`](https://pypi.org/project/differometor/) (the
JAX-based interferometer simulator).

Smoke-test:

```bash
python -m learn2design.scripts.uifo_random_search -s 0
```

---

## The contract

A submission is one class subclassing `OptimizationAlgorithm`:

```python
from dfbench.core import Objective, OptimizationAlgorithm
from jaxtyping import Array, Float


class MyAlgorithm(OptimizationAlgorithm):

    algorithm_str = "my_algo"

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."],
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        # 1. Warm up JIT (compilation is free, not counted against the budget)
        objective.warmup_value()

        # 2. Start the clock
        objective.start_logging()

        # 3. Optimization loop
        params = init_params
        while not objective.budget_exceeded:
            # ... your update logic here, producing `params` ...
            loss = objective.value(params)  # automatically logged
```

That is the entire contract. The `Objective` handles seeding, history,
checkpointing, and budget enforcement. You write the loop.

---

## What `Objective` gives you

| Method / attribute | Purpose |
|---|---|
| `objective.value(params)` | Forward pass; logged automatically. |
| `objective.value_and_grad(params)` | Loss and exact JAX gradient in one call. |
| `objective.warmup_value()` | JIT-compile before the timer starts. |
| `objective.start_logging()` | Begin the budget clock. |
| `objective.budget_exceeded` | Stop condition (max evals or wall time). |
| `objective.evals_since_improvement` | For your own early-stopping logic. |
| `objective.bounds` | Per-parameter `(low, high)` arrays. |
| `objective.random_params()` | Sample uniformly within bounds. |
| `objective.best_loss`, `objective.best_params_bounded` | Final results. |

Full reference: [`dfbench` docs](https://pypi.org/project/dfbench/).

---

## Repository layout

```
Learn2Design-2026/
├── learn2design/
│   ├── example_algorithms/     # Reference algorithms — read these first
│   │   ├── random_search.py
│   │   └── adam_gd.py
│   └── scripts/                # One-file runners per baseline
│       ├── uifo_random_search.py
│       └── uifo_adam_gd.py
├── pyproject.toml
└── README.md
```

The repository is intentionally small. We tried to condese relevant content for the user to see.

---

## Baselines

The starting kit ships with reference baselines so you can sanity-check your
environment and have something to beat:

| Baseline | File | Notes |
|---|---|---|
| Random search | [learn2design/example_algorithms/random_search.py](learn2design/example_algorithms/random_search.py) | The floor. |
| Adam | [learn2design/example_algorithms/adam_gd.py](learn2design/example_algorithms/adam_gd.py) | Uses exact JAX gradients via `value_and_grad`. |

Each has a matching one-file runner under `learn2design/scripts/`.
Run any of them with `python -m learn2design.scripts.<name> -s <seed>`.

---

## Submitting

1. Fork the repo. Place a single `.py` file in `submit/<your-handle>/`.
2. Add a `metadata.yaml` next to it with your handle, team members, and a
   one-line description.
3. If you need extra packages, add a `requirements.txt` in the same directory.
4. Open a pull request titled `[submission] <your-handle>: <algo name>`.

CI runs a smoke test; once it passes, your submission is queued for the next
evaluation batch. The submission present in `main` at the monthly deadline
counts for that month. In the meantime, iterate locally:

```bash
python -m learn2design.scripts.uifo_adam_gd -s 42
```

- **Time budget:** 4 h of `Objective.value` time + 30 min overhead per topology.
  JIT compilation before `start_logging()` is free.
- **Dependencies:** any pip-installable package; only `import differometor`
  directly is banned.
- **Scoring:** mean best-loss across 10 topologies. `NaN` / crashes → `+inf`.

Full rules: [docs/submission.md](docs/submission.md) ·
Scoring formula: [docs/scoring.md](docs/scoring.md) ·
FAQ: [docs/faq.md](docs/faq.md)

---

## Timeline

| Date | Event |
|---|---|
| TBA | Starting kit + month 1 problems released |
| TBA | Monthly public leaderboards open |
| 15.10.2026 | Final submission deadline |
| Before workshop | Private leaderboard announced |

---

## Resources

- **Repository:** <https://github.com/artificial-scientist-lab/Learn2Design-2026>
- **Issues / questions:** <https://github.com/artificial-scientist-lab/Learn2Design-2026/issues>
- **Simulator:** [`differometor`](https://pypi.org/project/differometor/)
- **Benchmark framework:** [`dfbench`](https://pypi.org/project/dfbench/)
- **Group:** [Artificial Scientist Lab](https://github.com/artificial-scientist-lab)

A website, FAQ page, and contact email will be added before the competition
opens.

---

## Citing

```bibtex
@misc{learn2design2026,
  title  = {Learn2Design 2026: Black-box Optimization for Gravitational-Wave Detector Design},
  author = {Artificial Scientist Lab},
  year   = {2026},
  url    = {https://github.com/artificial-scientist-lab/Learn2Design-2026},
}
```

## License

MIT — see [LICENSE](LICENSE).
