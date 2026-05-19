"""Test script for AdamGD optimizer."""

from dfbench.problems import UIFOProblem
from dfbench import Objective

from learn2design.example_algorithms import AdamGD

SEED = 42

problem = UIFOProblem(topology_seed=SEED)  # Create UIFO instance with a random topology
obj = Objective(
    problem,
    verbose=1,
    max_time= 60*60*4,  # 4 hours
    print_every=1,
    save_params_history=True,
    save_to_file_every=100,
    display_mode="live",  # Change to "log" when running without a live display (e.g., on a cluster)
)

optimizer = AdamGD()


# Run optimization
optimizer.optimize(
    obj,
    learning_rate=0.1,
    random_seed=SEED,  # This is only for random param generation in AdamGD
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
print(f"    {SEED}")
