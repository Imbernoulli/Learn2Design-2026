"""Test script for PyCMACMAES optimizer."""

from dfbench.problems import UIFOProblem
from dfbench import Objective

from learn2design.example_algorithms import PyCMACMAES

SEED = 42

problem = UIFOProblem(topology_seed=SEED)  # Create UIFO instance with a random topology
obj = Objective(
    problem,
    verbose=1,
    max_time=60 * 60 * 4,  # 4 hours
    print_every=5,
    save_params_history=True,
    save_batched_params_history=True,
    save_to_file_every=100,
    display_mode="log",  # Use "live" for a live display on a local machine
)

optimizer = PyCMACMAES(batch_size=20)


# Run optimization
optimizer.optimize(
    obj,
    pop_size=50,
    sigma0=0.3,
    random_seed=SEED,  # This is only for random param generation in PyCMACMAES
)

# Save run data
obj.save_run_data()
# Output loss plot, sensitivity plot, final parameters(JSON), and loss history(JSON) which can be toggled by arguments
obj.output_to_files()

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