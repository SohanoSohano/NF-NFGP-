# app/models/hybrid_individual.py
"""
Hybrid Neuro-Fuzzy Individual for Co-Evolution
Combines neural network weights with fuzzy system parameters.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

from app.core.fuzzy import FuzzyInferenceSystem

logger = logging.getLogger(__name__)


@dataclass
class HybridChromosome:
    """
    Represents a hybrid individual combining:
    - Hyperparameters (learning rate, dropout, etc.)
    - Neural network weights
    - Fuzzy system parameters (membership functions + rule weights)
    """
    
    # Segment sizes
    num_hyperparams: int
    num_nn_weights: int
    num_fuzzy_params: int
    
    # The actual chromosome data
    data: np.ndarray
    
    # Fitness tracking
    fitness: float = -np.inf
    objectives: Dict[str, float] = None  # For multi-objective optimization
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = {}
        
        # Validate chromosome size
        expected_size = self.num_hyperparams + self.num_nn_weights + self.num_fuzzy_params
        if len(self.data) != expected_size:
            raise ValueError(
                f"Chromosome size mismatch: expected {expected_size}, "
                f"got {len(self.data)}"
            )
    
    @property
    def hyperparams(self) -> np.ndarray:
        """Extract hyperparameter segment."""
        return self.data[:self.num_hyperparams]
    
    @property
    def nn_weights(self) -> np.ndarray:
        """Extract neural network weights segment."""
        start = self.num_hyperparams
        end = start + self.num_nn_weights
        return self.data[start:end]
    
    @property
    def fuzzy_params(self) -> np.ndarray:
        """Extract fuzzy system parameters segment."""
        start = self.num_hyperparams + self.num_nn_weights
        return self.data[start:]
    
    def set_hyperparams(self, values: np.ndarray):
        """Update hyperparameter segment."""
        self.data[:self.num_hyperparams] = values
    
    def set_nn_weights(self, values: np.ndarray):
        """Update neural network weights segment."""
        start = self.num_hyperparams
        end = start + self.num_nn_weights
        self.data[start:end] = values
    
    def set_fuzzy_params(self, values: np.ndarray):
        """Update fuzzy system parameters segment."""
        start = self.num_hyperparams + self.num_nn_weights
        self.data[start:] = values
    
    def copy(self) -> 'HybridChromosome':
        """Create a deep copy of this chromosome."""
        return HybridChromosome(
            num_hyperparams=self.num_hyperparams,
            num_nn_weights=self.num_nn_weights,
            num_fuzzy_params=self.num_fuzzy_params,
            data=self.data.copy(),
            fitness=self.fitness,
            objectives=self.objectives.copy()
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'num_hyperparams': self.num_hyperparams,
            'num_nn_weights': self.num_nn_weights,
            'num_fuzzy_params': self.num_fuzzy_params,
            'fitness': float(self.fitness),
            'objectives': self.objectives,
            'data_shape': self.data.shape,
        }
    
    @classmethod
    def from_legacy_chromosome(
        cls,
        legacy_data: np.ndarray,
        num_hyperparams: int,
        fis: Optional[FuzzyInferenceSystem] = None
    ) -> 'HybridChromosome':
        """
        Convert legacy chromosome (hyperparams + weights) to hybrid format.
        
        Args:
            legacy_data: Original chromosome array
            num_hyperparams: Number of hyperparameters
            fis: Optional FIS to initialize fuzzy parameters
            
        Returns:
            HybridChromosome instance
        """
        num_nn_weights = len(legacy_data) - num_hyperparams
        
        # Initialize fuzzy parameters
        if fis is not None:
            fuzzy_params = fis.encode_to_chromosome()
            num_fuzzy_params = len(fuzzy_params)
        else:
            # No fuzzy system - use empty segment
            fuzzy_params = np.array([], dtype=np.float32)
            num_fuzzy_params = 0
        
        # Combine all segments
        hybrid_data = np.concatenate([
            legacy_data[:num_hyperparams],  # hyperparams
            legacy_data[num_hyperparams:],  # nn weights
            fuzzy_params                     # fuzzy params
        ])
        
        return cls(
            num_hyperparams=num_hyperparams,
            num_nn_weights=num_nn_weights,
            num_fuzzy_params=num_fuzzy_params,
            data=hybrid_data
        )


class HybridPopulation:
    """Manages a population of hybrid individuals."""
    
    def __init__(self, individuals: list[HybridChromosome]):
        self.individuals = individuals
    
    def __len__(self) -> int:
        return len(self.individuals)
    
    def __getitem__(self, idx: int) -> HybridChromosome:
        return self.individuals[idx]
    
    def __iter__(self):
        return iter(self.individuals)
    
    def get_best(self, n: int = 1) -> list[HybridChromosome]:
        """Get top n individuals by fitness."""
        sorted_pop = sorted(self.individuals, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]
    
    def get_fitness_stats(self) -> Dict[str, float]:
        """Calculate population fitness statistics."""
        fitnesses = [ind.fitness for ind in self.individuals if np.isfinite(ind.fitness)]
        
        if not fitnesses:
            return {
                'max': -np.inf,
                'min': -np.inf,
                'mean': -np.inf,
                'std': 0.0
            }
        
        return {
            'max': float(np.max(fitnesses)),
            'min': float(np.min(fitnesses)),
            'mean': float(np.mean(fitnesses)),
            'std': float(np.std(fitnesses))
        }
    
    def calculate_diversity(self) -> float:
        """
        Calculate population diversity using average pairwise distance.
        Focuses on neural network weights for efficiency.
        """
        if len(self.individuals) < 2:
            return 0.0
        
        # Extract weight vectors
        weight_vectors = [ind.nn_weights for ind in self.individuals]
        
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(weight_vectors)):
            for j in range(i + 1, len(weight_vectors)):
                try:
                    distance = np.linalg.norm(
                        weight_vectors[i].astype(np.float32) - 
                        weight_vectors[j].astype(np.float32)
                    )
                    total_distance += distance
                    num_pairs += 1
                except Exception as e:
                    logger.warning(f"Error calculating distance: {e}")
        
        return float(total_distance / num_pairs) if num_pairs > 0 else 0.0
    
    def to_legacy_format(self) -> list[np.ndarray]:
        """
        Convert hybrid population to legacy format (for backward compatibility).
        Returns list of concatenated [hyperparams + nn_weights] arrays.
        """
        legacy_pop = []
        for ind in self.individuals:
            legacy_chromosome = np.concatenate([ind.hyperparams, ind.nn_weights])
            legacy_pop.append(legacy_chromosome)
        return legacy_pop


def create_hybrid_population(
    population_size: int,
    num_hyperparams: int,
    num_nn_weights: int,
    fis_template: FuzzyInferenceSystem,
    initial_nn_weights: Optional[np.ndarray] = None,
    hyperparam_ranges: Optional[Dict[str, Tuple[float, float]]] = None
) -> HybridPopulation:
    """
    Create an initial hybrid population.
    
    Args:
        population_size: Number of individuals
        num_hyperparams: Number of hyperparameters
        num_nn_weights: Size of neural network weight vector
        fis_template: Template FIS for initialization
        initial_nn_weights: Optional initial weights (will be mutated for diversity)
        hyperparam_ranges: Optional ranges for hyperparameters
        
    Returns:
        HybridPopulation instance
    """
    individuals = []
    num_fuzzy_params = fis_template.get_chromosome_size()
    
    # Get initial fuzzy encoding
    initial_fuzzy = fis_template.encode_to_chromosome()
    
    # Initialize weights if not provided
    if initial_nn_weights is None:
        initial_nn_weights = np.random.randn(num_nn_weights).astype(np.float32) * 0.01
    
    for i in range(population_size):
        # Generate hyperparameters
        if hyperparam_ranges:
            hyperparams = np.array([
                np.random.uniform(low, high)
                for low, high in hyperparam_ranges.values()
            ], dtype=np.float32)
        else:
            hyperparams = np.random.rand(num_hyperparams).astype(np.float32)
        
        # Generate weights (first individual uses initial, others are mutated)
        if i == 0:
            nn_weights = initial_nn_weights.copy()
        else:
            # Add Gaussian noise for diversity
            nn_weights = initial_nn_weights + np.random.randn(num_nn_weights).astype(np.float32) * 0.05
        
        # Generate fuzzy parameters (mutate template for diversity)
        if i == 0:
            fuzzy_params = initial_fuzzy.copy()
        else:
            fuzzy_params = initial_fuzzy + np.random.randn(num_fuzzy_params).astype(np.float32) * 0.02
        
        # Combine into hybrid chromosome
        data = np.concatenate([hyperparams, nn_weights, fuzzy_params])
        
        individual = HybridChromosome(
            num_hyperparams=num_hyperparams,
            num_nn_weights=num_nn_weights,
            num_fuzzy_params=num_fuzzy_params,
            data=data
        )
        
        individuals.append(individual)
    
    return HybridPopulation(individuals)
