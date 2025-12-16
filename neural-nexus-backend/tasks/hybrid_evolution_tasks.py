# tasks/hybrid_evolution_tasks.py
"""
Hybrid Neuro-Fuzzy Evolution Tasks
Phase 1: Core hybrid intelligence with co-evolution of NN weights and fuzzy rules.
"""

from app.core.celery_app import celery_app
from celery import current_task, Task
import time
import os
import numpy as np
import torch
import logging
from typing import Dict, Any, List

from app.core.config import settings
from app.core.fuzzy import FuzzyInferenceSystem, create_default_fis
from app.models.hybrid_individual import (
    HybridChromosome, HybridPopulation, create_hybrid_population
)
from app.utils.hybrid_operators import (
    hybrid_crossover_segmented,
    hybrid_mutation_adaptive,
    tournament_selection_hybrid,
    elitism_selection_hybrid
)
from app.utils.evolution_helpers import (
    load_pytorch_model,
    flatten_weights,
    load_task_eval_function,
    load_weights_from_flat,
    decode_hyperparameters
)

logger = logging.getLogger(__name__)

RESULT_DIR = settings.RESULT_DIR
os.makedirs(RESULT_DIR, exist_ok=True)


class HybridEvolutionTask(Task):
    """Base task for hybrid evolution with retry logic."""
    autoretry_for = (Exception,)
    retry_kwargs = {'max_retries': 2}
    retry_backoff = True
    retry_backoff_max = 7000
    retry_jitter = False
    acks_late = True
    reject_on_worker_lost = True


@celery_app.task(bind=True, base=HybridEvolutionTask)
def run_hybrid_evolution_task(
    self: Task,
    model_definition_path: str,
    task_evaluation_path: str | None,
    use_standard_eval: bool,
    initial_weights_path: str | None,
    config: Dict[str, Any]
):
    """
    Hybrid neuro-fuzzy evolution task.
    
    Phase 1 Features:
    - Co-evolution of NN weights + fuzzy rules
    - Unified hybrid serialization
    - Segmented genetic operators
    """
    task_id = self.request.id
    logger.info(f"[Hybrid Task {task_id}] Starting hybrid evolution...")
    
    try:
        # --- Configuration ---
        generations = config.get('generations', 10)
        population_size = config.get('population_size', 20)
        model_class = config.get('model_class')
        model_args = config.get('model_args', [])
        model_kwargs_static = config.get('model_kwargs', {})
        eval_config = config.get('eval_config', {})
        
        # Hybrid-specific config
        use_fuzzy = config.get('use_fuzzy', True)
        fuzzy_num_inputs = config.get('fuzzy_num_inputs', 2)
        fuzzy_num_outputs = config.get('fuzzy_num_outputs', 1)
        
        # Evolution parameters
        elitism_count = min(config.get('elitism_count', 2), population_size - 1)
        tournament_size = config.get('tournament_size', 3)
        
        # Mutation rates: (hyperparams, nn_weights, fuzzy_params)
        mutation_rates = (
            config.get('hyperparam_mutation_rate', 0.1),
            config.get('weight_mutation_rate', 0.1),
            config.get('fuzzy_mutation_rate', 0.05)
        )
        
        # Mutation strengths
        mutation_strengths = (
            config.get('hyperparam_mutation_strength', 0.02),
            config.get('weight_mutation_strength', 0.05),
            config.get('fuzzy_mutation_strength', 0.01)
        )
        
        # Hyperparameter evolution
        evolvable_hyperparams_config = config.get('evolvable_hyperparams', {})
        hyperparam_keys = list(evolvable_hyperparams_config.keys())
        num_hyperparams = len(hyperparam_keys)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"[Hybrid Task {task_id}] Device: {device}, Fuzzy: {use_fuzzy}")
        
        # --- Load Evaluation Function ---
        if use_standard_eval:
            eval_script_path = settings.STANDARD_EVAL_SCRIPT_PATH
        else:
            eval_script_path = task_evaluation_path
        
        task_eval_func = load_task_eval_function(eval_script_path)
        
        # --- Initialize Fuzzy System ---
        if use_fuzzy:
            fis_template = create_default_fis(
                num_inputs=fuzzy_num_inputs,
                num_outputs=fuzzy_num_outputs
            )
            logger.info(
                f"[Hybrid Task {task_id}] FIS initialized: "
                f"{fuzzy_num_inputs} inputs, {fuzzy_num_outputs} outputs, "
                f"{len(fis_template.rules)} rules"
            )
        else:
            fis_template = FuzzyInferenceSystem()  # Empty FIS
            logger.info(f"[Hybrid Task {task_id}] Running without fuzzy component")
        
        # --- Get Initial NN Weights ---
        placeholder_hparams = {
            k: (evolvable_hyperparams_config[k]['range'][0] + 
                evolvable_hyperparams_config[k]['range'][1]) / 2
            for k in hyperparam_keys
            if 'range' in evolvable_hyperparams_config[k]
        }
        
        ref_model = load_pytorch_model(
            model_definition_path, model_class, initial_weights_path, device,
            *model_args, **model_kwargs_static, **placeholder_hparams
        )
        initial_weights = flatten_weights(ref_model)
        num_nn_weights = len(initial_weights)
        logger.info(f"[Hybrid Task {task_id}] NN weight vector size: {num_nn_weights}")
        del ref_model
        
        # --- Initialize Hybrid Population ---
        self.update_state(
            state='PROGRESS',
            meta={'progress': 0.0, 'message': 'Initializing hybrid population...'}
        )
        
        hyperparam_ranges = {
            k: tuple(evolvable_hyperparams_config[k]['range'])
            for k in hyperparam_keys
            if 'range' in evolvable_hyperparams_config[k]
        }
        
        population = create_hybrid_population(
            population_size=population_size,
            num_hyperparams=num_hyperparams,
            num_nn_weights=num_nn_weights,
            fis_template=fis_template,
            initial_nn_weights=initial_weights,
            hyperparam_ranges=hyperparam_ranges
        )
        
        logger.info(f"[Hybrid Task {task_id}] Population initialized: {len(population)} individuals")
        
        # --- Evolution Loop ---
        best_individual = None
        best_fitness = -np.inf
        fitness_history = []
        avg_fitness_history = []
        diversity_history = []
        
        for gen in range(generations):
            gen_num = gen + 1
            gen_start = time.time()
            logger.info(f"[Hybrid Task {task_id}] --- Generation {gen_num}/{generations} ---")
            
            # --- Evaluation ---
            for individual in population:
                try:
                    # Decode hyperparameters
                    hparams_dict = decode_hyperparameters(
                        individual.hyperparams,
                        hyperparam_keys,
                        evolvable_hyperparams_config
                    )
                    
                    # Load model with evolved hyperparameters
                    model = load_pytorch_model(
                        model_definition_path, model_class, None, device,
                        *model_args, **model_kwargs_static, **hparams_dict
                    )
                    
                    # Load evolved weights
                    load_weights_from_flat(model, individual.nn_weights)
                    
                    # Evaluate (fuzzy system can be used in eval_config if needed)
                    if use_fuzzy:
                        # Decode fuzzy system for potential use in evaluation
                        fis = create_default_fis(fuzzy_num_inputs, fuzzy_num_outputs)
                        fis.decode_from_chromosome(individual.fuzzy_params)
                        eval_config['fis'] = fis
                    
                    fitness = task_eval_func(model, eval_config)
                    individual.fitness = float(fitness) if np.isfinite(fitness) else -np.inf
                    
                    del model
                    
                except Exception as e:
                    logger.warning(f"Evaluation failed for individual: {e}")
                    individual.fitness = -np.inf
            
            # --- Statistics ---
            stats = population.get_fitness_stats()
            diversity = population.calculate_diversity()
            
            fitness_history.append(stats['max'])
            avg_fitness_history.append(stats['mean'])
            diversity_history.append(diversity)
            
            logger.info(
                f"[Hybrid Task {task_id}] Gen {gen_num}: "
                f"Max={stats['max']:.4f}, Avg={stats['mean']:.4f}, "
                f"Diversity={diversity:.4f}"
            )
            
            # Update best
            current_best = population.get_best(1)[0]
            if current_best.fitness > best_fitness:
                best_fitness = current_best.fitness
                best_individual = current_best.copy()
                logger.info(f"[Hybrid Task {task_id}] *** New best: {best_fitness:.4f} ***")
            
            # --- Update Progress ---
            progress = gen_num / generations
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'message': f'Gen {gen_num}/{generations} | Best: {best_fitness:.4f}',
                    'fitness_history': fitness_history,
                    'avg_fitness_history': avg_fitness_history,
                    'diversity_history': diversity_history
                }
            )
            
            # --- Selection ---
            elite = elitism_selection_hybrid(population.individuals, elitism_count)
            parents = tournament_selection_hybrid(
                population.individuals,
                tournament_size=tournament_size,
                num_parents=population_size
            )
            
            # --- Reproduction ---
            offspring = []
            
            # Add elite
            offspring.extend(elite)
            
            # Generate offspring
            while len(offspring) < population_size:
                # Select two parents
                p1, p2 = np.random.choice(parents, size=2, replace=False)
                
                # Crossover
                child1, child2 = hybrid_crossover_segmented(p1, p2)
                
                # Mutation
                child1 = hybrid_mutation_adaptive(
                    child1, mutation_rates, mutation_strengths
                )
                child2 = hybrid_mutation_adaptive(
                    child2, mutation_rates, mutation_strengths
                )
                
                offspring.append(child1)
                if len(offspring) < population_size:
                    offspring.append(child2)
            
            # Update population
            population = HybridPopulation(offspring[:population_size])
            
            gen_time = time.time() - gen_start
            logger.info(f"[Hybrid Task {task_id}] Gen {gen_num} completed in {gen_time:.2f}s")
        
        # --- Save Best Model ---
        if best_individual is None:
            raise RuntimeError("No valid individuals found during evolution")
        
        final_model_filename = f"hybrid_evolved_{task_id}.pth"
        final_model_path = os.path.join(RESULT_DIR, final_model_filename)
        
        # Decode best hyperparameters
        best_hparams = decode_hyperparameters(
            best_individual.hyperparams,
            hyperparam_keys,
            evolvable_hyperparams_config
        )
        
        # Create and save final model
        final_model = load_pytorch_model(
            model_definition_path, model_class, None, device,
            *model_args, **model_kwargs_static, **best_hparams
        )
        load_weights_from_flat(final_model, best_individual.nn_weights)
        torch.save(final_model.state_dict(), final_model_path)
        
        # Save fuzzy system separately if used
        fuzzy_path = None
        if use_fuzzy:
            fuzzy_path = os.path.join(RESULT_DIR, f"fuzzy_system_{task_id}.npy")
            np.save(fuzzy_path, best_individual.fuzzy_params)
            logger.info(f"[Hybrid Task {task_id}] Fuzzy system saved to {fuzzy_path}")
        
        logger.info(f"[Hybrid Task {task_id}] Best model saved to {final_model_path}")
        
        # --- Cleanup ---
        for f_path in [model_definition_path, task_evaluation_path, initial_weights_path]:
            if f_path and os.path.exists(f_path):
                try:
                    os.remove(f_path)
                except OSError as e:
                    logger.warning(f"Could not remove {f_path}: {e}")
        
        # --- Return Results ---
        result = {
            'message': f'Hybrid evolution completed. Best fitness: {best_fitness:.4f}',
            'final_model_path': final_model_path,
            'fuzzy_system_path': fuzzy_path,
            'best_fitness': float(best_fitness),
            'best_hyperparameters': best_hparams,
            'fitness_history': fitness_history,
            'avg_fitness_history': avg_fitness_history,
            'diversity_history': diversity_history,
            'hybrid_info': {
                'num_hyperparams': num_hyperparams,
                'num_nn_weights': num_nn_weights,
                'num_fuzzy_params': best_individual.num_fuzzy_params,
                'used_fuzzy': use_fuzzy
            }
        }
        
        self.update_state(state='SUCCESS', meta=result)
        logger.info(f"[Hybrid Task {task_id}] Task completed successfully")
        
        return result
        
    except Exception as e:
        error_msg = f'Hybrid evolution task failed: {str(e)}'
        logger.error(f"[Hybrid Task {task_id}] {error_msg}", exc_info=True)
        
        self.update_state(
            state='FAILURE',
            meta={'message': error_msg, 'error': str(e)}
        )
        raise
