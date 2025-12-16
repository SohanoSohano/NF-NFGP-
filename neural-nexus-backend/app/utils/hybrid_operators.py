# app/utils/hybrid_operators.py
"""
Genetic operators for hybrid neuro-fuzzy evolution.
Specialized operators that respect the structure of hybrid chromosomes.
"""

import numpy as np
import random
from typing import List, Tuple
import logging

from app.models.hybrid_individual import HybridChromosome

logger = logging.getLogger(__name__)


def hybrid_crossover_segmented(
    parent1: HybridChromosome,
    parent2: HybridChromosome,
    crossover_probs: Tuple[float, float, float] = (0.5, 0.5, 0.5)
) -> Tuple[HybridChromosome, HybridChromosome]:
    """
    Segmented crossover that treats each chromosome segment independently.
    
    Args:
        parent1, parent2: Parent chromosomes
        crossover_probs: Probabilities for (hyperparams, nn_weights, fuzzy_params)
        
    Returns:
        Two offspring chromosomes
    """
    child1 = parent1.copy()
    child2 = parent2.copy()
    
    # Crossover hyperparameters
    if random.random() < crossover_probs[0]:
        hp1 = parent1.hyperparams.copy()
        hp2 = parent2.hyperparams.copy()
        
        # Uniform crossover
        for i in range(len(hp1)):
            if random.random() < 0.5:
                hp1[i], hp2[i] = hp2[i], hp1[i]
        
        child1.set_hyperparams(hp1)
        child2.set_hyperparams(hp2)
    
    # Crossover NN weights
    if random.random() < crossover_probs[1]:
        w1 = parent1.nn_weights.copy()
        w2 = parent2.nn_weights.copy()
        
        # One-point crossover
        if len(w1) > 1:
            point = random.randint(1, len(w1) - 1)
            w1[point:], w2[point:] = w2[point:].copy(), w1[point:].copy()
        
        child1.set_nn_weights(w1)
        child2.set_nn_weights(w2)
    
    # Crossover fuzzy parameters
    if random.random() < crossover_probs[2] and parent1.num_fuzzy_params > 0:
        f1 = parent1.fuzzy_params.copy()
        f2 = parent2.fuzzy_params.copy()
        
        # Average crossover for fuzzy params (smoother)
        alpha = random.random()
        f1_new = alpha * f1 + (1 - alpha) * f2
        f2_new = alpha * f2 + (1 - alpha) * f1
        
        child1.set_fuzzy_params(f1_new)
        child2.set_fuzzy_params(f2_new)
    
    return child1, child2


def hybrid_mutation_adaptive(
    individual: HybridChromosome,
    mutation_rates: Tuple[float, float, float],
    mutation_strengths: Tuple[float, float, float],
    bounds: dict = None
) -> HybridChromosome:
    """
    Adaptive mutation for hybrid chromosomes with segment-specific rates.
    
    Args:
        individual: Chromosome to mutate
        mutation_rates: Rates for (hyperparams, nn_weights, fuzzy_params)
        mutation_strengths: Strengths for (hyperparams, nn_weights, fuzzy_params)
        bounds: Optional bounds for hyperparameters
        
    Returns:
        Mutated chromosome
    """
    mutated = individual.copy()
    
    # Mutate hyperparameters
    hp = mutated.hyperparams.copy()
    for i in range(len(hp)):
        if random.random() < mutation_rates[0]:
            hp[i] += np.random.randn() * mutation_strengths[0]
            
            # Apply bounds if provided
            if bounds and i < len(bounds):
                hp[i] = np.clip(hp[i], bounds[i][0], bounds[i][1])
    
    mutated.set_hyperparams(hp)
    
    # Mutate NN weights
    weights = mutated.nn_weights.copy()
    mutation_mask = np.random.rand(len(weights)) < mutation_rates[1]
    weights[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * mutation_strengths[1]
    mutated.set_nn_weights(weights)
    
    # Mutate fuzzy parameters
    if mutated.num_fuzzy_params > 0:
        fuzzy = mutated.fuzzy_params.copy()
        mutation_mask = np.random.rand(len(fuzzy)) < mutation_rates[2]
        fuzzy[mutation_mask] += np.random.randn(np.sum(mutation_mask)) * mutation_strengths[2]
        
        # Ensure rule weights stay positive
        # Assuming last N params are rule weights (this is a simplification)
        # In practice, you'd track which indices are rule weights
        fuzzy = np.maximum(fuzzy, 0.01)  # Keep positive
        
        mutated.set_fuzzy_params(fuzzy)
    
    return mutated


def hybrid_mutation_self_adaptive(
    individual: HybridChromosome,
    tau: float = 0.1,
    tau_prime: float = 0.05
) -> HybridChromosome:
    """
    Self-adaptive mutation where mutation parameters evolve with the individual.
    
    Args:
        individual: Chromosome to mutate
        tau: Learning rate for global adaptation
        tau_prime: Learning rate for local adaptation
        
    Returns:
        Mutated chromosome with adapted mutation parameters
    """
    mutated = individual.copy()
    
    # Global mutation strength adaptation
    global_factor = np.exp(tau * np.random.randn())
    
    # Mutate each segment with self-adaptive strength
    for segment_name in ['hyperparams', 'nn_weights', 'fuzzy_params']:
        if segment_name == 'hyperparams':
            segment = mutated.hyperparams.copy()
        elif segment_name == 'nn_weights':
            segment = mutated.nn_weights.copy()
        else:
            if mutated.num_fuzzy_params == 0:
                continue
            segment = mutated.fuzzy_params.copy()
        
        # Local adaptation per parameter
        for i in range(len(segment)):
            local_factor = np.exp(tau_prime * np.random.randn())
            mutation_strength = global_factor * local_factor * 0.01
            segment[i] += np.random.randn() * mutation_strength
        
        # Update segment
        if segment_name == 'hyperparams':
            mutated.set_hyperparams(segment)
        elif segment_name == 'nn_weights':
            mutated.set_nn_weights(segment)
        else:
            mutated.set_fuzzy_params(segment)
    
    return mutated


def tournament_selection_hybrid(
    population: List[HybridChromosome],
    tournament_size: int = 3,
    num_parents: int = None
) -> List[HybridChromosome]:
    """
    Tournament selection for hybrid chromosomes.
    
    Args:
        population: List of hybrid individuals
        tournament_size: Number of individuals per tournament
        num_parents: Number of parents to select (default: len(population))
        
    Returns:
        Selected parents
    """
    if num_parents is None:
        num_parents = len(population)
    
    parents = []
    for _ in range(num_parents):
        # Select random individuals for tournament
        tournament = random.sample(population, min(tournament_size, len(population)))
        
        # Select best from tournament
        winner = max(tournament, key=lambda x: x.fitness if np.isfinite(x.fitness) else -np.inf)
        parents.append(winner.copy())
    
    return parents


def elitism_selection_hybrid(
    population: List[HybridChromosome],
    num_elite: int
) -> List[HybridChromosome]:
    """
    Select top individuals for elitism.
    
    Args:
        population: List of hybrid individuals
        num_elite: Number of elite individuals to preserve
        
    Returns:
        Elite individuals
    """
    # Sort by fitness (descending)
    sorted_pop = sorted(
        population,
        key=lambda x: x.fitness if np.isfinite(x.fitness) else -np.inf,
        reverse=True
    )
    
    return [ind.copy() for ind in sorted_pop[:num_elite]]


def memetic_local_search(
    individual: HybridChromosome,
    eval_func,
    max_iterations: int = 10,
    step_size: float = 0.01
) -> HybridChromosome:
    """
    Local search refinement for memetic algorithms.
    Performs gradient-free hill climbing on the individual.
    
    Args:
        individual: Chromosome to refine
        eval_func: Function to evaluate fitness
        max_iterations: Maximum local search iterations
        step_size: Step size for perturbations
        
    Returns:
        Refined chromosome
    """
    current = individual.copy()
    current_fitness = current.fitness
    
    for iteration in range(max_iterations):
        # Try small perturbations
        candidate = current.copy()
        
        # Perturb NN weights (most impactful)
        weights = candidate.nn_weights.copy()
        perturbation = np.random.randn(len(weights)) * step_size
        weights += perturbation
        candidate.set_nn_weights(weights)
        
        # Evaluate candidate
        try:
            candidate_fitness = eval_func(candidate)
            
            # Accept if better
            if candidate_fitness > current_fitness:
                current = candidate
                current_fitness = candidate_fitness
            else:
                # Reduce step size
                step_size *= 0.9
        except Exception as e:
            logger.warning(f"Local search evaluation failed: {e}")
            break
    
    current.fitness = current_fitness
    return current


def co_evolution_step(
    nn_population: List[HybridChromosome],
    fuzzy_population: List[HybridChromosome],
    eval_func,
    cooperation_rate: float = 0.3
) -> Tuple[List[HybridChromosome], List[HybridChromosome]]:
    """
    Co-evolutionary step where NN and fuzzy components evolve cooperatively.
    
    Args:
        nn_population: Population focused on NN optimization
        fuzzy_population: Population focused on fuzzy optimization
        eval_func: Evaluation function
        cooperation_rate: Rate of cross-population cooperation
        
    Returns:
        Updated (nn_population, fuzzy_population)
    """
    # Evaluate cooperative combinations
    for i in range(len(nn_population)):
        if random.random() < cooperation_rate:
            # Swap fuzzy components between populations
            j = random.randint(0, len(fuzzy_population) - 1)
            
            # Create hybrid by combining NN from one pop with fuzzy from another
            hybrid = nn_population[i].copy()
            hybrid.set_fuzzy_params(fuzzy_population[j].fuzzy_params.copy())
            
            # Evaluate
            try:
                hybrid.fitness = eval_func(hybrid)
                
                # Update if better
                if hybrid.fitness > nn_population[i].fitness:
                    nn_population[i] = hybrid
            except Exception as e:
                logger.warning(f"Co-evolution evaluation failed: {e}")
    
    return nn_population, fuzzy_population
