# Phase 1: Hybrid Intelligence Core

## Overview

Phase 1 implements the foundational hybrid neuro-fuzzy evolution system, enabling co-evolution of neural network weights alongside fuzzy inference system parameters. This creates interpretable AI models that combine the learning power of neural networks with the explainability of fuzzy logic.

## Architecture

### Core Components

```
app/core/fuzzy.py              # Fuzzy Inference System (FIS)
├── MembershipFunction         # Triangular, Gaussian, Trapezoidal
├── FuzzyVariable             # Input/Output variables with MFs
├── FuzzyRule                 # IF-THEN rules with weights
└── FuzzyInferenceSystem      # Complete FIS with encoding/decoding

app/models/hybrid_individual.py  # Hybrid chromosome structure
├── HybridChromosome          # [Hyperparams | NN Weights | Fuzzy Params]
├── HybridPopulation          # Population management
└── create_hybrid_population  # Initialization utilities

app/utils/hybrid_operators.py   # Genetic operators
├── hybrid_crossover_segmented    # Segment-aware crossover
├── hybrid_mutation_adaptive      # Adaptive mutation
├── tournament_selection_hybrid   # Selection operators
└── memetic_local_search         # Local refinement

tasks/hybrid_evolution_tasks.py  # Celery evolution task
└── run_hybrid_evolution_task    # Main evolution loop
```

### Hybrid Chromosome Structure

```
┌─────────────┬──────────────┬────────────────┐
│ Hyperparams │  NN Weights  │ Fuzzy Params   │
│   (H dims)  │   (W dims)   │   (F dims)     │
└─────────────┴──────────────┴────────────────┘
```

**Segments:**
1. **Hyperparameters** (H): Learning rate, dropout, etc.
2. **NN Weights** (W): Flattened neural network parameters
3. **Fuzzy Parameters** (F): Membership function params + rule weights

## Features Implemented

### ✅ Fuzzy Inference System (FIS)

- **Membership Functions:**
  - Triangular: `[a, b, c]` - Simple, interpretable
  - Gaussian: `[mean, sigma]` - Smooth transitions
  - Trapezoidal: `[a, b, c, d]` - Plateau regions

- **Fuzzy Rules:**
  - IF-THEN structure with evolvable weights
  - Product T-norm for antecedent aggregation
  - Max aggregation for consequent combination

- **Inference:**
  - Mamdani-style inference
  - Center of Gravity (CoG) defuzzification
  - Efficient chromosome encoding/decoding

### ✅ Co-Evolution Engine

- **Unified Evolution:**
  - Single population with hybrid chromosomes
  - Simultaneous optimization of all components
  - Fitness-driven co-adaptation

- **Segmented Operators:**
  - Independent crossover rates per segment
  - Adaptive mutation strengths
  - Segment-aware genetic operations

### ✅ Hybrid Serialization

- **Encoding:**
  ```python
  chromosome = np.concatenate([
      hyperparams,           # Shape: (H,)
      nn_weights,           # Shape: (W,)
      fuzzy_params          # Shape: (F,)
  ])
  ```

- **Decoding:**
  ```python
  hyperparams = chromosome[:H]
  nn_weights = chromosome[H:H+W]
  fuzzy_params = chromosome[H+W:]
  ```

## API Usage

### Start Hybrid Evolution

```bash
POST /api/v1/hybrid/start
```

**Form Data:**
- `model_definition`: Python file with model class
- `task_evaluation`: Optional custom evaluation script
- `initial_weights`: Optional pretrained weights
- `config`: JSON configuration
- `use_standard_eval`: Boolean flag

**Config Example:**
```json
{
  "generations": 50,
  "population_size": 30,
  "model_class": "SimpleNN",
  "use_fuzzy": true,
  "fuzzy_num_inputs": 2,
  "fuzzy_num_outputs": 1,
  "hyperparam_mutation_rate": 0.1,
  "weight_mutation_rate": 0.1,
  "fuzzy_mutation_rate": 0.05,
  "hyperparam_mutation_strength": 0.02,
  "weight_mutation_strength": 0.05,
  "fuzzy_mutation_strength": 0.01,
  "elitism_count": 2,
  "tournament_size": 3,
  "evolvable_hyperparams": {
    "learning_rate": {
      "range": [0.0001, 0.1],
      "type": "float"
    },
    "dropout": {
      "range": [0.0, 0.5],
      "type": "float"
    }
  },
  "eval_config": {
    "dataset": "mnist",
    "batch_size": 64,
    "num_epochs": 1
  }
}
```

### Check Status

```bash
GET /api/v1/hybrid/status/{task_id}
```

**Response:**
```json
{
  "task_id": "abc-123",
  "state": "PROGRESS",
  "progress": 0.6,
  "message": "Gen 30/50 | Best: 0.9234",
  "fitness_history": [0.85, 0.87, 0.89, ...],
  "avg_fitness_history": [0.75, 0.78, 0.81, ...],
  "diversity_history": [12.3, 11.8, 10.5, ...],
  "hybrid_info": {
    "num_hyperparams": 2,
    "num_nn_weights": 7850,
    "num_fuzzy_params": 27,
    "used_fuzzy": true
  }
}
```

### Get System Info

```bash
GET /api/v1/hybrid/info
```

## Example: MNIST Classification

### 1. Define Model

```python
# model_definition.py
import torch.nn as nn

class MNISTClassifier(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(784, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
```

### 2. Configure Evolution

```python
config = {
    "generations": 30,
    "population_size": 20,
    "model_class": "MNISTClassifier",
    "use_fuzzy": True,
    "fuzzy_num_inputs": 2,  # e.g., loss and accuracy
    "fuzzy_num_outputs": 1,  # e.g., learning rate adjustment
    "evolvable_hyperparams": {
        "dropout": {
            "range": [0.1, 0.5],
            "type": "float"
        }
    }
}
```

### 3. Start Evolution

```bash
curl -X POST http://localhost:8000/api/v1/hybrid/start \
  -F "model_definition=@model_definition.py" \
  -F "config=$(cat config.json)" \
  -F "use_standard_eval=true"
```

## Fuzzy System Example

### Default FIS Structure

For 2 inputs, 1 output:

**Input Variables:**
- `input_0`: [Low, Medium, High]
- `input_1`: [Low, Medium, High]

**Output Variable:**
- `output_0`: [Low, Medium, High]

**Rule Base (9 rules):**
```
IF input_0 is Low AND input_1 is Low THEN output_0 is Low
IF input_0 is Low AND input_1 is Medium THEN output_0 is Low
IF input_0 is Low AND input_1 is High THEN output_0 is Medium
...
IF input_0 is High AND input_1 is High THEN output_0 is High
```

### Evolution Process

1. **Initialization:**
   - MF parameters initialized with default shapes
   - Rule weights initialized to 1.0
   - Small random perturbations for diversity

2. **Mutation:**
   - MF parameters: Gaussian noise
   - Rule weights: Bounded positive values
   - Adaptive rates based on fitness

3. **Crossover:**
   - Average crossover for smooth MF transitions
   - Preserves rule structure
   - Independent from NN crossover

## Performance Characteristics

### Computational Complexity

- **Chromosome Size:** `O(H + W + F)`
  - H: Typically 2-10 hyperparameters
  - W: Model-dependent (100s to millions)
  - F: `3 * num_MFs + num_rules` (typically 20-100)

- **Evaluation:** `O(forward_pass + fuzzy_inference)`
  - NN forward pass: Dominant cost
  - Fuzzy inference: Negligible (<1% overhead)

### Memory Usage

- **Per Individual:** `4 * (H + W + F)` bytes (float32)
- **Population:** `population_size * individual_size`
- **Example:** 20 individuals × 10K params = ~800KB

## Advantages of Hybrid Approach

### 1. Interpretability
- Fuzzy rules provide human-readable logic
- Membership functions visualize decision boundaries
- Rule weights show importance

### 2. Flexibility
- Can disable fuzzy component (pure NN mode)
- Backward compatible with existing evolution
- Gradual integration path

### 3. Performance
- Fuzzy system can guide NN training
- Adaptive learning rate control
- Multi-objective optimization support

### 4. Robustness
- Fuzzy logic handles uncertainty
- Smooth decision boundaries
- Graceful degradation

## Testing

### Unit Tests

```bash
# Test fuzzy system
pytest tests/test_fuzzy.py

# Test hybrid chromosome
pytest tests/test_hybrid_individual.py

# Test operators
pytest tests/test_hybrid_operators.py
```

### Integration Test

```python
# tests/test_hybrid_evolution.py
def test_hybrid_evolution_mnist():
    config = {
        "generations": 5,
        "population_size": 10,
        "use_fuzzy": True,
        # ... minimal config
    }
    
    task = run_hybrid_evolution_task.apply_async(args=[...])
    result = task.get(timeout=300)
    
    assert result['best_fitness'] > 0.8
    assert 'fuzzy_system_path' in result
```

## Troubleshooting

### Issue: Fuzzy parameters not evolving

**Solution:** Increase `fuzzy_mutation_rate` and `fuzzy_mutation_strength`

```json
{
  "fuzzy_mutation_rate": 0.15,
  "fuzzy_mutation_strength": 0.05
}
```

### Issue: High memory usage

**Solution:** Reduce population size or use gradient checkpointing

```json
{
  "population_size": 10,
  "eval_config": {
    "use_gradient_checkpointing": true
  }
}
```

### Issue: Slow convergence

**Solution:** Enable elitism and increase tournament size

```json
{
  "elitism_count": 3,
  "tournament_size": 5
}
```

## Next Steps: Phase 2

Phase 2 will introduce:

1. **NSGA-II Multi-Objective Optimization**
   - Pareto front calculation
   - Crowding distance
   - Non-dominated sorting

2. **Advanced Evolution**
   - Self-adaptive mutation rates
   - Memetic local search
   - Coevolutionary strategies

3. **Visualization**
   - Pareto front plots
   - Fuzzy rule visualization
   - Evolution progress tracking

## References

- Mamdani, E. H. (1974). "Application of fuzzy algorithms for control of simple dynamic plant"
- Deb, K. et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- Jang, J.-S. R. (1993). "ANFIS: Adaptive-Network-Based Fuzzy Inference System"

## Contributing

To extend Phase 1:

1. Add new membership function types in `fuzzy.py`
2. Implement new crossover operators in `hybrid_operators.py`
3. Add custom evaluation metrics in task evaluation scripts
4. Create visualization tools for fuzzy systems

---

**Status:** ✅ Phase 1 Complete
**Next:** Phase 2 - Advanced Evolution & Multi-Objective Optimization
