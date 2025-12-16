# examples/hybrid_mnist_example.py
"""
Example: Hybrid Neuro-Fuzzy Evolution for MNIST Classification
Demonstrates Phase 1 capabilities with a simple neural network.
"""

import torch
import torch.nn as nn


class SimpleMNISTNet(nn.Module):
    """
    Simple feedforward network for MNIST classification.
    Hyperparameters: dropout rate (evolvable)
    """
    
    def __init__(self, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        return self.fc3(x)


# Configuration for hybrid evolution
HYBRID_CONFIG = {
    # Evolution parameters
    "generations": 20,
    "population_size": 15,
    "elitism_count": 2,
    "tournament_size": 3,
    
    # Model configuration
    "model_class": "SimpleMNISTNet",
    "model_args": [],
    "model_kwargs": {},
    
    # Fuzzy system configuration
    "use_fuzzy": True,
    "fuzzy_num_inputs": 2,   # e.g., training loss and validation accuracy
    "fuzzy_num_outputs": 1,  # e.g., learning rate adjustment factor
    
    # Mutation rates (per segment)
    "hyperparam_mutation_rate": 0.15,
    "weight_mutation_rate": 0.10,
    "fuzzy_mutation_rate": 0.08,
    
    # Mutation strengths
    "hyperparam_mutation_strength": 0.03,
    "weight_mutation_strength": 0.05,
    "fuzzy_mutation_strength": 0.02,
    
    # Evolvable hyperparameters
    "evolvable_hyperparams": {
        "dropout": {
            "range": [0.1, 0.5],
            "type": "float",
            "description": "Dropout rate for regularization"
        }
    },
    
    # Evaluation configuration
    "eval_config": {
        "dataset": "mnist",
        "batch_size": 128,
        "num_epochs": 1,
        "validation_split": 0.2,
        "device": "cuda"  # or "cpu"
    }
}


# Example usage with curl:
"""
# 1. Save this file as model_definition.py

# 2. Create config.json with HYBRID_CONFIG

# 3. Start evolution:
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@model_definition.py" \
  -F "config=@config.json" \
  -F "use_standard_eval=true"

# 4. Check status:
curl http://localhost:8000/api/v1/hybrid/status/{task_id}

# 5. Monitor progress:
watch -n 5 'curl -s http://localhost:8000/api/v1/hybrid/status/{task_id} | jq'
"""


# Example usage with Python requests:
"""
import requests
import json

# Start evolution
with open('model_definition.py', 'rb') as f:
    files = {'model_definition': f}
    data = {
        'config': json.dumps(HYBRID_CONFIG),
        'use_standard_eval': 'true'
    }
    response = requests.post(
        'http://localhost:8000/api/v1/hybrid/start',
        files=files,
        data=data
    )
    task_id = response.json()['task_id']
    print(f"Task started: {task_id}")

# Poll status
import time
while True:
    response = requests.get(f'http://localhost:8000/api/v1/hybrid/status/{task_id}')
    status = response.json()
    print(f"Progress: {status.get('progress', 0)*100:.1f}% - {status.get('message', '')}")
    
    if status['state'] in ['SUCCESS', 'FAILURE']:
        print(f"Final result: {status}")
        break
    
    time.sleep(5)
"""


# Fuzzy system interpretation example:
"""
After evolution completes, the fuzzy system can be interpreted:

1. Load the fuzzy parameters:
   fuzzy_params = np.load('results/fuzzy_system_{task_id}.npy')

2. Decode into FIS:
   from app.core.fuzzy import create_default_fis
   fis = create_default_fis(num_inputs=2, num_outputs=1)
   fis.decode_from_chromosome(fuzzy_params)

3. Inspect rules:
   for i, rule in enumerate(fis.rules):
       print(f"Rule {i}: IF {rule.antecedents} THEN {rule.consequent} (weight: {rule.weight:.3f})")

4. Visualize membership functions:
   import matplotlib.pyplot as plt
   for var_name, var in fis.input_variables.items():
       x = np.linspace(var.range[0], var.range[1], 100)
       for mf in var.membership_functions:
           y = [mf.evaluate(xi) for xi in x]
           plt.plot(x, y, label=mf.name)
       plt.title(f"Input: {var_name}")
       plt.legend()
       plt.show()

5. Test inference:
   inputs = {"input_0": 0.3, "input_1": 0.7}
   outputs = fis.infer(inputs)
   print(f"Fuzzy inference: {inputs} -> {outputs}")
"""
