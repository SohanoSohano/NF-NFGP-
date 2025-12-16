# app/api/endpoints/hybrid_evolver.py
"""
API endpoints for hybrid neuro-fuzzy evolution.
Phase 1: Core hybrid intelligence with co-evolution.
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import json
import os
import logging

from app.core.config import settings
from tasks.hybrid_evolution_tasks import run_hybrid_evolution_task

logger = logging.getLogger(__name__)

router = APIRouter()

UPLOAD_DIR = settings.UPLOAD_DIR
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/start")
async def start_hybrid_evolution(
    model_definition: UploadFile = File(...),
    task_evaluation: Optional[UploadFile] = File(None),
    initial_weights: Optional[UploadFile] = File(None),
    config: str = Form(...),
    use_standard_eval: bool = Form(False),
):
    """
    Start a hybrid neuro-fuzzy evolution task.
    
    **Phase 1 Features:**
    - Co-evolution of neural network weights and fuzzy rules
    - Evolvable membership functions
    - Unified hybrid serialization
    - Segmented genetic operators
    
    **Config Parameters:**
    - `generations`: Number of evolution generations
    - `population_size`: Population size
    - `model_class`: Model class name
    - `use_fuzzy`: Enable fuzzy component (default: true)
    - `fuzzy_num_inputs`: Number of fuzzy inputs (default: 2)
    - `fuzzy_num_outputs`: Number of fuzzy outputs (default: 1)
    - `hyperparam_mutation_rate`: Mutation rate for hyperparameters
    - `weight_mutation_rate`: Mutation rate for NN weights
    - `fuzzy_mutation_rate`: Mutation rate for fuzzy parameters
    - `evolvable_hyperparams`: Dictionary of hyperparameters to evolve
    
    **Returns:**
    - `task_id`: Celery task ID for tracking
    - `message`: Status message
    """
    try:
        # Parse config
        try:
            config_dict = json.loads(config)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON config: {e}")
        
        # Save uploaded files
        model_def_path = os.path.join(UPLOAD_DIR, f"model_{model_definition.filename}")
        with open(model_def_path, "wb") as f:
            content = await model_definition.read()
            f.write(content)
        
        task_eval_path = None
        if task_evaluation and not use_standard_eval:
            task_eval_path = os.path.join(UPLOAD_DIR, f"eval_{task_evaluation.filename}")
            with open(task_eval_path, "wb") as f:
                content = await task_evaluation.read()
                f.write(content)
        
        initial_weights_path = None
        if initial_weights:
            initial_weights_path = os.path.join(UPLOAD_DIR, f"weights_{initial_weights.filename}")
            with open(initial_weights_path, "wb") as f:
                content = await initial_weights.read()
                f.write(content)
        
        # Validate hybrid-specific config
        use_fuzzy = config_dict.get('use_fuzzy', True)
        if use_fuzzy:
            logger.info("Hybrid evolution with fuzzy component enabled")
        else:
            logger.info("Hybrid evolution without fuzzy component (NN-only mode)")
        
        # Start Celery task
        task = run_hybrid_evolution_task.apply_async(
            args=[
                model_def_path,
                task_eval_path,
                use_standard_eval,
                initial_weights_path,
                config_dict
            ]
        )
        
        logger.info(f"Hybrid evolution task started: {task.id}")
        
        return JSONResponse(
            status_code=202,
            content={
                "task_id": task.id,
                "message": "Hybrid evolution task started successfully",
                "hybrid_mode": "neuro-fuzzy" if use_fuzzy else "neural-only"
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to start hybrid evolution: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{task_id}")
async def get_hybrid_task_status(task_id: str):
    """
    Get the status of a hybrid evolution task.
    
    **Returns:**
    - `state`: Task state (PENDING, PROGRESS, SUCCESS, FAILURE, HALTED)
    - `progress`: Progress percentage (0-1)
    - `message`: Status message
    - `fitness_history`: Best fitness per generation
    - `avg_fitness_history`: Average fitness per generation
    - `diversity_history`: Population diversity per generation
    - `hybrid_info`: Information about hybrid chromosome structure
    """
    from celery.result import AsyncResult
    
    try:
        result = AsyncResult(task_id)
        
        response = {
            "task_id": task_id,
            "state": result.state,
        }
        
        if result.state == 'PENDING':
            response["message"] = "Task is waiting to start"
            response["progress"] = 0.0
        elif result.state == 'PROGRESS':
            info = result.info or {}
            response.update({
                "progress": info.get('progress', 0.0),
                "message": info.get('message', 'Processing...'),
                "fitness_history": info.get('fitness_history', []),
                "avg_fitness_history": info.get('avg_fitness_history', []),
                "diversity_history": info.get('diversity_history', []),
            })
        elif result.state == 'SUCCESS':
            info = result.info or {}
            response.update({
                "progress": 1.0,
                "message": info.get('message', 'Task completed'),
                "final_model_path": info.get('final_model_path'),
                "fuzzy_system_path": info.get('fuzzy_system_path'),
                "best_fitness": info.get('best_fitness'),
                "best_hyperparameters": info.get('best_hyperparameters'),
                "fitness_history": info.get('fitness_history', []),
                "avg_fitness_history": info.get('avg_fitness_history', []),
                "diversity_history": info.get('diversity_history', []),
                "hybrid_info": info.get('hybrid_info', {}),
            })
        elif result.state == 'FAILURE':
            # result.info might be an exception object, not a dict
            if isinstance(result.info, dict):
                info = result.info
                response.update({
                    "message": info.get('message', 'Task failed'),
                    "error": str(info.get('error', 'Unknown error')),
                })
            else:
                # Handle exception objects
                response.update({
                    "message": "Task failed",
                    "error": str(result.info) if result.info else 'Unknown error',
                })
        elif result.state == 'PENDING' and result.info is None:
            # Task not found in Celery
            response.update({
                "message": "Task not found or not yet registered",
                "error": "Task may not exist or Celery worker is not running"
            })
        else:
            response["message"] = f"Task state: {result.state}"
        
        return JSONResponse(content=response)
        
    except Exception as e:
        logger.error(f"Failed to get task status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/info")
async def get_hybrid_info():
    """
    Get information about the hybrid evolution system.
    
    **Returns:**
    - System capabilities
    - Phase 1 features
    - Configuration options
    """
    return {
        "system": "NeuroForge Hybrid Intelligence",
        "phase": "Phase 1 - Hybrid Intelligence Core",
        "features": [
            "Co-evolution of neural network weights and fuzzy rules",
            "Evolvable membership functions (triangular, gaussian, trapezoidal)",
            "Evolvable fuzzy rule weights",
            "Unified hybrid chromosome serialization",
            "Segmented genetic operators (separate rates for NN/fuzzy)",
            "Backward compatible with pure NN evolution"
        ],
        "fuzzy_capabilities": {
            "membership_functions": ["triangular", "gaussian", "trapezoidal"],
            "inference_method": "Mamdani",
            "defuzzification": "Center of Gravity",
            "rule_encoding": "Evolvable weights"
        },
        "evolution_operators": {
            "crossover": ["segmented", "uniform", "average"],
            "mutation": ["adaptive", "self-adaptive", "gaussian"],
            "selection": ["tournament", "elitism"]
        },
        "next_phases": [
            "Phase 2: NSGA-II multi-objective optimization",
            "Phase 3: AI-guided research with explainability",
            "Phase 4: Collaboration & lineage tracking",
            "Phase 5: Production reliability"
        ]
    }
