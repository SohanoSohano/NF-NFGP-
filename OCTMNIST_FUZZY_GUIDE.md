# OCTMNIST Hybrid Neuro-Fuzzy Evolution Guide

## ✅ Compatibility Status: READY!

Your `octmnist_cnn_hp_ready_evaluation.py` file is **fully compatible** with hybrid neuro-fuzzy evolution after the minor fix applied.

---

## What Makes It Compatible?

### ✅ Correct Function Signature
```python
def evaluate_network_on_task(model_instance: torch.nn.Module, config: dict) -> float:
```
This matches exactly what the hybrid evolution system expects!

### ✅ Proper Config Handling
```python
device = config.get('device')  # Gets device from config
```
The hybrid system will pass additional config parameters, and your code handles them gracefully.

### ✅ Returns Fitness Value
```python
return float(accuracy)  # Returns 0.0 to 1.0
```
Perfect for evolution - higher is better!

### ✅ Error Handling
```python
except Exception as e:
    return -float('inf')  # Bad individuals get very low fitness
```
This ensures evolution continues even if some individuals fail.

---

## How Fuzzy Evolution Will Work with OCTMNIST

### 1. Hybrid Chromosome Structure

```
┌─────────────────┬──────────────────┬─────────────────┐
│  Hyperparams    │   NN Weights     │  Fuzzy Params   │
│  (3 params)     │   (~50K params)  │  (27 params)    │
├─────────────────┼──────────────────┼─────────────────┤
│ dropout_rate    │ Conv1 weights    │ MF shapes       │
│ conv1_filters   │ Conv2 weights    │ Rule weights    │
│ conv2_filters   │ FC weights       │                 │
└─────────────────┴──────────────────┴─────────────────┘
```

### 2. What Gets Evolved

**Neural Network:**
- Convolutional layer weights
- Fully connected layer weights
- Batch norm parameters

**Hyperparameters:**
- `dropout_rate`: 0.1 to 0.5
- `conv1_filters`: 16 to 64
- `conv2_filters`: 32 to 128

**Fuzzy System:**
- 2 input variables (e.g., training loss, validation accuracy)
- 1 output variable (e.g., learning rate multiplier)
- 9 fuzzy rules with evolvable weights
- 18 membership function parameters

### 3. Evaluation Process

```python
# For each individual in population:

# 1. Extract evolved parameters
dropout = individual.hyperparams[0]      # e.g., 0.23
conv1_f = int(individual.hyperparams[1]) # e.g., 48
conv2_f = int(individual.hyperparams[2]) # e.g., 96

# 2. Build model with evolved hyperparams
model = OCTMNIST_CNN(
    input_channels=1,
    num_classes=4,
    dropout_rate=dropout,
    conv1_filters=conv1_f,
    conv2_filters=conv2_f
)

# 3. Load evolved weights
load_weights_from_flat(model, individual.nn_weights)

# 4. Decode fuzzy system
fis = create_default_fis(2, 1)
fis.decode_from_chromosome(individual.fuzzy_params)

# 5. Evaluate on OCTMNIST
config = {
    'device': torch.device('cuda'),
    'fis': fis,  # Fuzzy system available if needed
    'batch_size': 256
}
fitness = evaluate_network_on_task(model, config)

# 6. Store fitness
individual.fitness = fitness  # e.g., 0.87 (87% accuracy)
```

---

## How to Use with Hybrid Evolution

### Option 1: Via Frontend (Easiest)

1. **Open Frontend:**
   ```
   http://localhost:3000
   ```

2. **Go to "Hybrid Neuro-Fuzzy" Tab**

3. **Upload Files:**
   - Model Definition: `octmnist_cnn_hp_ready.py`
   - Evaluation: `octmnist_cnn_hp_ready_evaluation.py`

4. **Configure:**
   - Enable "Fuzzy Component" ✓
   - Fuzzy Inputs: 2
   - Fuzzy Outputs: 1
   - Generations: 20
   - Population Size: 15

5. **Start Evolution!**

### Option 2: Via API (More Control)

```bash
# 1. Prepare files
# - octmnist_cnn_hp_ready.py (your model)
# - octmnist_cnn_hp_ready_evaluation.py (your evaluation)
# - octmnist_hybrid_config.json (configuration)

# 2. Start evolution
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@octmnist_cnn_hp_ready.py" \
  -F "task_evaluation=@octmnist_cnn_hp_ready_evaluation.py" \
  -F "config=@octmnist_hybrid_config.json" \
  -F "use_standard_eval=false"

# 3. Get task ID from response
# {"task_id": "abc-123-def-456", ...}

# 4. Monitor progress
watch -n 3 'curl -s http://localhost:8000/api/v1/hybrid/status/abc-123-def-456 | jq'
```

### Option 3: Via Python Script

```python
import requests
import json
import time

# Configuration
config = {
    "generations": 20,
    "population_size": 15,
    "model_class": "OCTMNIST_CNN",
    "use_fuzzy": True,
    "fuzzy_num_inputs": 2,
    "fuzzy_num_outputs": 1,
    "evolvable_hyperparams": {
        "dropout_rate": {"range": [0.1, 0.5], "type": "float"},
        "conv1_filters": {"range": [16, 64], "type": "int"},
        "conv2_filters": {"range": [32, 128], "type": "int"}
    }
}

# Start evolution
with open('octmnist_cnn_hp_ready.py', 'rb') as model_file, \
     open('octmnist_cnn_hp_ready_evaluation.py', 'rb') as eval_file:
    
    response = requests.post(
        'http://localhost:8000/api/v1/hybrid/start',
        files={
            'model_definition': model_file,
            'task_evaluation': eval_file
        },
        data={
            'config': json.dumps(config),
            'use_standard_eval': 'false'
        }
    )

task_id = response.json()['task_id']
print(f"Task started: {task_id}")

# Monitor
while True:
    status = requests.get(f'http://localhost:8000/api/v1/hybrid/status/{task_id}').json()
    print(f"[{status['state']}] {status.get('message', '')}")
    
    if status['state'] in ['SUCCESS', 'FAILURE']:
        break
    
    time.sleep(5)
```

---

## Expected Results

### Performance Expectations

**OCTMNIST Dataset:**
- 4 classes (retinal OCT images)
- Training: 97,477 images
- Test: 1,000 images
- Image size: 28×28 grayscale

**Evolution Performance:**
- **Generation 0:** ~40-50% accuracy (random initialization)
- **Generation 10:** ~70-80% accuracy (learning patterns)
- **Generation 20:** ~85-90% accuracy (optimized)

**With Fuzzy System:**
- Fuzzy rules can adapt learning dynamics
- May converge faster than pure NN evolution
- Provides interpretable decision logic

### Timeline Estimates

**Small Run (5 gen, 5 pop):**
- With GPU: ~3-5 minutes
- Without GPU: ~10-15 minutes

**Medium Run (20 gen, 15 pop):**
- With GPU: ~15-20 minutes
- Without GPU: ~45-60 minutes

**Large Run (50 gen, 30 pop):**
- With GPU: ~60-90 minutes
- Without GPU: ~3-4 hours

---

## Fuzzy System Use Cases for OCTMNIST

### Use Case 1: Adaptive Learning Rate

```python
# Fuzzy inputs
inputs = {
    "input_0": current_loss,        # 0.0 to 1.0
    "input_1": validation_accuracy  # 0.0 to 1.0
}

# Fuzzy inference
outputs = fis.infer(inputs)
lr_multiplier = outputs["output_0"]

# Evolved rules might learn:
# IF loss is High AND accuracy is Low THEN increase LR (explore more)
# IF loss is Low AND accuracy is High THEN decrease LR (fine-tune)
```

### Use Case 2: Confidence Estimation

```python
# Fuzzy inputs
inputs = {
    "input_0": prediction_entropy,  # Model uncertainty
    "input_1": max_softmax_score    # Prediction confidence
}

# Fuzzy output
outputs = fis.infer(inputs)
confidence_level = outputs["output_0"]

# Use for:
# - Rejecting low-confidence predictions
# - Requesting human review
# - Ensemble weighting
```

### Use Case 3: Class Imbalance Handling

```python
# Fuzzy inputs
inputs = {
    "input_0": class_frequency,     # How common is this class
    "input_1": prediction_score     # Model's confidence
}

# Fuzzy output
outputs = fis.infer(inputs)
weight_adjustment = outputs["output_0"]

# Evolved rules might learn:
# IF class is Rare AND score is Low THEN increase weight
# IF class is Common AND score is High THEN decrease weight
```

---

## Interpreting Results

### After Evolution Completes

```python
# 1. Load best model
best_model = OCTMNIST_CNN(
    input_channels=1,
    num_classes=4,
    dropout_rate=best_hyperparams['dropout_rate'],
    conv1_filters=best_hyperparams['conv1_filters'],
    conv2_filters=best_hyperparams['conv2_filters']
)
best_model.load_state_dict(torch.load('results/hybrid_evolved_TASK_ID.pth'))

# 2. Load fuzzy system
fuzzy_params = np.load('results/fuzzy_system_TASK_ID.npy')
fis = create_default_fis(2, 1)
fis.decode_from_chromosome(fuzzy_params)

# 3. Inspect evolved fuzzy rules
print("\nEvolved Fuzzy Rules:")
for i, rule in enumerate(fis.rules):
    print(f"Rule {i+1}: {rule.antecedents} → {rule.consequent}")
    print(f"  Weight: {rule.weight:.3f}")
    if rule.weight > 1.5:
        print(f"  ⭐ High importance!")

# 4. Visualize membership functions
import matplotlib.pyplot as plt

for var_name, var in fis.input_variables.items():
    x = np.linspace(var.range[0], var.range[1], 100)
    plt.figure(figsize=(10, 6))
    
    for mf in var.membership_functions:
        y = [mf.evaluate(xi) for xi in x]
        plt.plot(x, y, label=mf.name, linewidth=2)
    
    plt.xlabel('Input Value')
    plt.ylabel('Membership Degree')
    plt.title(f'Evolved Membership Functions: {var_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'evolved_mf_{var_name}.png')
    plt.show()
```

---

## Troubleshooting

### Issue: "Invalid device in config"

**Cause:** Device not properly set in config

**Fix:**
```python
# In your config
config = {
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # ... other params
}
```

### Issue: "OCTMNIST dataset download failed"

**Cause:** Network issues or permissions

**Fix:**
```bash
# Pre-download dataset
docker exec -it neuroforge_v22-backend-1 python -c "
from medmnist import OCTMNIST
import os
os.makedirs('/code/medmnist_data', exist_ok=True)
OCTMNIST(split='test', download=True, root='/code/medmnist_data')
"
```

### Issue: "Out of memory"

**Cause:** GPU VRAM exhausted

**Fix:**
```python
# Reduce batch size in evaluation file
BATCH_SIZE = 128  # Instead of 256

# Or reduce population size
config['population_size'] = 10  # Instead of 15
```

### Issue: "Model architecture mismatch"

**Cause:** Evolved hyperparams create incompatible architecture

**Fix:**
```python
# Ensure model can handle evolved hyperparams
class OCTMNIST_CNN(nn.Module):
    def __init__(self, dropout_rate=0.2, conv1_filters=32, conv2_filters=64):
        # Validate parameters
        assert 16 <= conv1_filters <= 64, "conv1_filters out of range"
        assert 32 <= conv2_filters <= 128, "conv2_filters out of range"
        # ... rest of init
```

---

## Summary

✅ **Your OCTMNIST evaluation file is READY for fuzzy evolution!**

**What works:**
- ✅ Function signature matches
- ✅ Config handling is correct
- ✅ Returns proper fitness value
- ✅ Error handling is robust
- ✅ Dataset caching is efficient

**What was fixed:**
- ✅ Removed undefined `task_id` reference

**Next steps:**
1. Test with small run (5 gen, 5 pop)
2. Monitor fitness progression
3. Analyze evolved fuzzy rules
4. Scale up to full run

**Ready to evolve! 🧬🔮✨**
