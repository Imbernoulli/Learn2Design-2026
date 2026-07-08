"""Test script for NAAdamGD optimizer."""

from dfbench.problems import UIFOProblem
from dfbench import Objective

from learn2design.example_algorithms import NAAdamGD

SEED = 42

problem = UIFOProblem(topology_seed=SEED)  # Create UIFO instance with a random topology
obj = Objective(
    problem,
    verbose=1,
    max_time=60 * 60 * 4,  # 4 hours
    print_every=1,
    save_params_history=True,
    save_to_file_every=100,
    display_mode="log",  # Use "live" for a live display on a local machine
)

optimizer = NAAdamGD()

# Run optimization
optimizer.optimize(
    obj,
    learning_rate=0.1,
    noise_schedule="exponential",
    noise_anneal_budget_fraction=0.5,
    random_seed=SEED,  # This is only for random param generation in NAAdamGD
)

# Save run data
obj.save_run_data(hyper_param_str="lr0.1_0.5budget")  # Optional hyperparameter string will be included in the filename
# Output loss plot, sensitivity plot, final parameters(JSON), and loss history(JSON) which can be toggled by arguments
obj.output_to_files(hyper_param_str="lr0.1_0.5budget")   

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