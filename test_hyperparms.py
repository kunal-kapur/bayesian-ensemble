import itertools
import subprocess
import torch
import math

# Define hyperparameter values
dropout_hyperparms = [0.7, 0.4]
dropout_hyperparms = list(itertools.product(dropout_hyperparms, dropout_hyperparms))  # Pairs of dropout values
# num_masks_hyperparms = [3, 5, 10, 20]
num_masks_hyperparms = [1]

num_groups_hyperparms = [0.1]
increment_amt_hyperparams = [1]
# num_groups_hyperparms = [0.1, 0.3, 0.5]
# increment_amt_hyperparams = [1, 10, 50]

# Get all possible combinations of hyperparameters
hyperparameter_combinations = itertools.product(
    dropout_hyperparms, num_masks_hyperparms, num_groups_hyperparms, increment_amt_hyperparams
)

PATH = "experiments"

# Iterate through all combinations and train
seen = set()
for dropout, num_masks, group_prop, increment_amt in hyperparameter_combinations:
    dropout1, dropout2 = dropout  # Unpacking the dropout tuple
    num_groups = math.ceil(group_prop / num_masks)
    vals = (num_masks, dropout1, dropout2, num_groups, increment_amt)
    if vals in seen:
        continue
    seen.add(vals)
    command = [
        "python", "train.py",
        "--num_masks", str(num_masks),
        "--dropout_probs", str(dropout1), str(dropout2),
        "--num_groups", str(num_groups),  # Convert proportion to integer
        "--increment_amt", str(increment_amt),
        "--path", PATH
    ]
    
    print(f"Running: {' '.join(command)}")
    subprocess.run(command)
