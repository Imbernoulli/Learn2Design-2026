from dfbench import Objective
from dfbench.problems import UIFOProblem
from learn2design.example_algorithms import (
    AdamGD, NAAdamGD, OptaxSGDM, BFGS, LBFGSGD, PyCMACMAES, RandomSearch,
)

problem = UIFOProblem(topology_seed=0)
objective = Objective(problem, max_evals=1, max_time=60, verbose=0)
objective.warmup_value()
objective.start_logging()
loss = objective.value(objective.random_params())

print(f"loss={float(loss):.6f}")
print(
    "algorithms="
    + ", ".join(
        cls.algorithm_str
        for cls in (AdamGD, NAAdamGD, OptaxSGDM, BFGS, LBFGSGD, PyCMACMAES, RandomSearch)
    )
)