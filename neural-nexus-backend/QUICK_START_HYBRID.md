# Quick Start: Hybrid Neuro-Fuzzy Evolution

## 5-Minute Setup

### 1. Start the Backend

```bash
cd NeuroForge_V2.2
docker-compose up -d
```

### 2. Create Your Model

```python
# my_model.py
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(-1, 784)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### 3. Create Configuration

```json
{
  "generations": 20,
  "population_size": 15,
  "model_class": "MyModel",
  "use_fuzzy": true,
  "fuzzy_num_inputs": 2,
  "fuzzy_num_outputs": 1,
  "evolvable_hyperparams": {
    "dropout": {
      "range": [0.1, 0.5],
      "type": "float"
    }
  },
  "eval_config": {
    "dataset": "mnist",
    "batch_size": 128,
    "num_epochs": 1
  }
}
```

### 4. Start Evolution

```bash
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@my_model.py" \
  -F "config=@config.json" \
  -F "use_standard_eval=true"
```

Response:
```json
{
  "task_id": "abc-123-def-456",
  "message": "Hybrid evolution task started successfully",
  "hybrid_mode": "neuro-fuzzy"
}
```

### 5. Monitor Progress

```bash
# Check status
curl http://localhost:8000/api/v1/hybrid/status/abc-123-def-456

# Watch progress (Linux/Mac)
watch -n 5 'curl -s http://localhost:8000/api/v1/hybrid/status/abc-123-def-456 | jq'

# Watch progress (Windows PowerShell)
while($true) { 
  curl http://localhost:8000/api/v1/hybrid/status/abc-123-def-456 | ConvertFrom-Json | ConvertTo-Json
  Start-Sleep -Seconds 5
}
```

## Configuration Options

### Basic Settings

```json
{
  "generations": 20,           // Number of evolution generations
  "population_size": 15,       // Population size
  "model_class": "MyModel",    // Your model class name
  "use_fuzzy": true           // Enable fuzzy component
}
```

### Fuzzy System

```json
{
  "fuzzy_num_inputs": 2,       // Number of fuzzy inputs
  "fuzzy_num_outputs": 1,      // Number of fuzzy outputs
  "fuzzy_mutation_rate": 0.05, // Mutation rate for fuzzy params
  "fuzzy_mutation_strength": 0.01  // Mutation strength
}
```

### Evolution Parameters

```json
{
  "elitism_count": 2,          // Number of elite individuals
  "tournament_size": 3,        // Tournament selection size
  "hyperparam_mutation_rate": 0.1,
  "weight_mutation_rate": 0.1,
  "hyperparam_mutation_strength": 0.02,
  "weight_mutation_strength": 0.05
}
```

### Hyperparameter Evolution

```json
{
  "evolvable_hyperparams": {
    "learning_rate": {
      "range": [0.0001, 0.1],
      "type": "float"
    },
    "dropout": {
      "range": [0.0, 0.5],
      "type": "float"
    },
    "hidden_size": {
      "range": [64, 256],
      "type": "int"
    }
  }
}
```

## Common Use Cases

### 1. Pure Neural Network (No Fuzzy)

```json
{
  "use_fuzzy": false,
  "generations": 30,
  "population_size": 20
}
```

### 2. Fuzzy-Guided Evolution

```json
{
  "use_fuzzy": true,
  "fuzzy_num_inputs": 2,    // e.g., loss and accuracy
  "fuzzy_num_outputs": 1,   // e.g., learning rate adjustment
  "generations": 40
}
```

### 3. Multi-Hyperparameter Optimization

```json
{
  "evolvable_hyperparams": {
    "learning_rate": {"range": [0.0001, 0.1]},
    "dropout": {"range": [0.1, 0.5]},
    "weight_decay": {"range": [0.0, 0.01]},
    "momentum": {"range": [0.8, 0.99]}
  }
}
```

### 4. Fast Prototyping

```json
{
  "generations": 10,
  "population_size": 10,
  "eval_config": {
    "num_epochs": 1,
    "batch_size": 256
  }
}
```

### 5. Production Run

```json
{
  "generations": 100,
  "population_size": 50,
  "elitism_count": 5,
  "eval_config": {
    "num_epochs": 3,
    "validation_split": 0.2
  }
}
```

## Python Client Example

```python
import requests
import json
import time

# Configuration
config = {
    "generations": 20,
    "population_size": 15,
    "model_class": "MyModel",
    "use_fuzzy": True,
    "fuzzy_num_inputs": 2,
    "fuzzy_num_outputs": 1,
    "evolvable_hyperparams": {
        "dropout": {"range": [0.1, 0.5], "type": "float"}
    }
}

# Start evolution
with open('my_model.py', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/api/v1/hybrid/start',
        files={'model_definition': f},
        data={
            'config': json.dumps(config),
            'use_standard_eval': 'true'
        }
    )

task_id = response.json()['task_id']
print(f"Task started: {task_id}")

# Monitor progress
while True:
    response = requests.get(
        f'http://localhost:8000/api/v1/hybrid/status/{task_id}'
    )
    status = response.json()
    
    state = status['state']
    progress = status.get('progress', 0) * 100
    message = status.get('message', '')
    
    print(f"[{state}] {progress:.1f}% - {message}")
    
    if state in ['SUCCESS', 'FAILURE']:
        print("\nFinal result:")
        print(json.dumps(status, indent=2))
        break
    
    time.sleep(5)
```

## Interpreting Results

### Success Response

```json
{
  "state": "SUCCESS",
  "progress": 1.0,
  "message": "Hybrid evolution completed. Best fitness: 0.9234",
  "final_model_path": "results/hybrid_evolved_abc123.pth",
  "fuzzy_system_path": "results/fuzzy_system_abc123.npy",
  "best_fitness": 0.9234,
  "best_hyperparameters": {
    "dropout": 0.23
  },
  "fitness_history": [0.85, 0.87, 0.89, 0.91, 0.92],
  "avg_fitness_history": [0.75, 0.78, 0.81, 0.84, 0.86],
  "diversity_history": [12.3, 11.8, 10.5, 9.2, 8.1],
  "hybrid_info": {
    "num_hyperparams": 1,
    "num_nn_weights": 7850,
    "num_fuzzy_params": 27,
    "used_fuzzy": true
  }
}
```

### Loading Results

```python
import torch
import numpy as np
from app.core.fuzzy import create_default_fis

# Load model
model = MyModel(dropout=0.23)
model.load_state_dict(torch.load('results/hybrid_evolved_abc123.pth'))
model.eval()

# Load fuzzy system
fuzzy_params = np.load('results/fuzzy_system_abc123.npy')
fis = create_default_fis(num_inputs=2, num_outputs=1)
fis.decode_from_chromosome(fuzzy_params)

# Use model
with torch.no_grad():
    output = model(input_tensor)

# Use fuzzy system
fuzzy_output = fis.infer({"input_0": 0.5, "input_1": 0.7})
```

## Troubleshooting

### Issue: Task stuck in PENDING

**Check:**
```bash
# Verify Celery worker is running
docker-compose logs celery_worker

# Restart if needed
docker-compose restart celery_worker
```

### Issue: Low fitness scores

**Try:**
- Increase `num_epochs` in eval_config
- Increase `population_size`
- Adjust mutation rates
- Check model architecture

### Issue: Slow evolution

**Optimize:**
- Reduce `population_size`
- Reduce `num_epochs`
- Increase `batch_size`
- Use GPU if available

### Issue: Out of memory

**Solutions:**
- Reduce `population_size`
- Reduce `batch_size`
- Use smaller model
- Enable gradient checkpointing

## Advanced Features

### Custom Evaluation

```python
# custom_eval.py
def evaluate_model(model, eval_config):
    # Your custom evaluation logic
    accuracy = train_and_evaluate(model, eval_config)
    return accuracy
```

```bash
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@my_model.py" \
  -F "task_evaluation=@custom_eval.py" \
  -F "config=@config.json" \
  -F "use_standard_eval=false"
```

### Pretrained Weights

```bash
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@my_model.py" \
  -F "initial_weights=@pretrained.pth" \
  -F "config=@config.json"
```

### Fuzzy System Visualization

```python
import matplotlib.pyplot as plt
import numpy as np
from app.core.fuzzy import create_default_fis

# Load fuzzy system
fuzzy_params = np.load('results/fuzzy_system_abc123.npy')
fis = create_default_fis(2, 1)
fis.decode_from_chromosome(fuzzy_params)

# Plot membership functions
for var_name, var in fis.input_variables.items():
    x = np.linspace(var.range[0], var.range[1], 100)
    plt.figure(figsize=(10, 6))
    
    for mf in var.membership_functions:
        y = [mf.evaluate(xi) for xi in x]
        plt.plot(x, y, label=mf.name, linewidth=2)
    
    plt.xlabel('Input Value')
    plt.ylabel('Membership Degree')
    plt.title(f'Membership Functions: {var_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Print rules
print("\nFuzzy Rules:")
for i, rule in enumerate(fis.rules):
    print(f"Rule {i+1}: IF {rule.antecedents} THEN {rule.consequent} (weight: {rule.weight:.3f})")
```

## Next Steps

1. **Read Full Documentation:** `PHASE1_HYBRID_INTELLIGENCE.md`
2. **Explore Examples:** `examples/hybrid_mnist_example.py`
3. **Check Roadmap:** `ROADMAP.md`
4. **Try Advanced Features:** Multi-objective optimization (Phase 2)

## Support

- **Documentation:** See `PHASE1_HYBRID_INTELLIGENCE.md`
- **Examples:** `examples/` directory
- **API Reference:** `http://localhost:8000/docs`
- **System Info:** `http://localhost:8000/api/v1/hybrid/info`

---

**Happy Evolving! 🧬🤖**
