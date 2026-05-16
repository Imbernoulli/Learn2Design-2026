"""Test script for RandomSearch optimizer."""

import argparse

from dfbench.problems import UIFOProblem
from dfbench import Objective

from learn2design.example_algorithms import RandomSearch

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--seed", type=int, default=None)
seed = parser.parse_args().seed

# Optimization workflow with RandomSearch
problem = UIFOProblem(topology_seed=seed)
obj = Objective(
    problem,
    verbose=1,
    max_time= 60*60*24,
    print_every=100,
    save_params_history=True,
    save_to_file_every=100,
    display_mode="log",
)

optimizer = RandomSearch()


# Run optimization - returns Objective instance
optimizer.optimize(
    obj,
    random_seed=seed,
)

obj.save_run_data()

print("Best loss:")
print(f"    {obj.best_loss:.6f}")
print("Total evaluations:")
print(f"    {obj.eval_count}")
print("First parameters:")
print(f"    {obj.params_history_bounded[0]}")
print("Best parameters:")
print(f"    {obj.best_params_bounded}")
print("Seed:")
print(f"    {seed}")
