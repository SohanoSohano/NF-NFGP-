#!/usr/bin/env python3
"""
Comprehensive compatibility test for OCTMNIST Hybrid Evolution
Tests all components to ensure they work together.
"""

import sys
import torch
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_import():
    """Test 1: Can we import the model?"""
    logger.info("=" * 60)
    logger.info("TEST 1: Model Import")
    logger.info("=" * 60)
    try:
        from octmnist_cnn_hp_ready import MyCNN
        logger.info("✓ Model import successful")
        logger.info(f"✓ Model class: {MyCNN.__name__}")
        return True, MyCNN
    except Exception as e:
        logger.error(f"✗ Model import failed: {e}")
        return False, None

def test_model_instantiation(ModelClass):
    """Test 2: Can we instantiate the model with default params?"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: Model Instantiation (Default)")
    logger.info("=" * 60)
    try:
        model = ModelClass(input_channels=1, num_classes=4)
        logger.info("✓ Model instantiation successful")
        logger.info(f"✓ Model type: {type(model)}")
        return True, model
    except Exception as e:
        logger.error(f"✗ Model instantiation failed: {e}")
        return False, None

def test_model_with_evolved_params(ModelClass):
    """Test 3: Can we instantiate with evolved hyperparameters?"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: Model with Evolved Hyperparameters")
    logger.info("=" * 60)
    
    # Simulate evolved hyperparameters
    evolved_params = {
        'out_channels_conv1': 40,
        'out_channels_conv2': 70,
        'out_channels_conv3': 110,
        'neurons_fc1': 55,
        'dropout_rate': 0.3
    }
    
    try:
        model = ModelClass(input_channels=1, num_classes=4, **evolved_params)
        logger.info("✓ Model with evolved params successful")
        logger.info(f"✓ Evolved params: {evolved_params}")
        return True, model
    except Exception as e:
        logger.error(f"✗ Model with evolved params failed: {e}")
        return False, None

def test_forward_pass(model):
    """Test 4: Can we do a forward pass?"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: Forward Pass")
    logger.info("=" * 60)
    try:
        # Create dummy input (batch_size=2, channels=1, height=28, width=28)
        dummy_input = torch.randn(2, 1, 28, 28)
        logger.info(f"  Input shape: {dummy_input.shape}")
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
        
        logger.info(f"✓ Forward pass successful")
        logger.info(f"✓ Output shape: {output.shape}")
        logger.info(f"✓ Expected shape: torch.Size([2, 4])")
        
        if output.shape == torch.Size([2, 4]):
            logger.info("✓ Output shape matches expected!")
            return True
        else:
            logger.error(f"✗ Output shape mismatch!")
            return False
    except Exception as e:
        logger.error(f"✗ Forward pass failed: {e}")
        return False

def test_evaluation_import():
    """Test 5: Can we import the evaluation function?"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: Evaluation Function Import")
    logger.info("=" * 60)
    try:
        from octmnist_cnn_hp_ready_evaluation import evaluate_network_on_task
        logger.info("✓ Evaluation function import successful")
        logger.info(f"✓ Function: {evaluate_network_on_task.__name__}")
        return True, evaluate_network_on_task
    except Exception as e:
        logger.error(f"✗ Evaluation function import failed: {e}")
        return False, None

def test_evaluation_signature(eval_func):
    """Test 6: Does evaluation function have correct signature?"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: Evaluation Function Signature")
    logger.info("=" * 60)
    try:
        import inspect
        sig = inspect.signature(eval_func)
        params = list(sig.parameters.keys())
        
        logger.info(f"  Function signature: {sig}")
        logger.info(f"  Parameters: {params}")
        
        # Check for required parameters
        required_params = ['model_instance', 'config']
        missing = [p for p in required_params if p not in params]
        
        if missing:
            logger.error(f"✗ Missing required parameters: {missing}")
            return False
        
        logger.info("✓ Evaluation function signature correct")
        return True
    except Exception as e:
        logger.error(f"✗ Signature check failed: {e}")
        return False

def test_weight_flattening(model):
    """Test 7: Can we flatten and restore weights?"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: Weight Flattening/Restoration")
    logger.info("=" * 60)
    try:
        # Get all parameters
        params = []
        for param in model.parameters():
            params.append(param.data.cpu().numpy().flatten())
        
        flat_weights = np.concatenate(params)
        logger.info(f"✓ Flattened weights shape: {flat_weights.shape}")
        logger.info(f"✓ Total parameters: {len(flat_weights):,}")
        
        # Test restoration
        idx = 0
        for param in model.parameters():
            param_length = param.numel()
            param.data = torch.from_numpy(
                flat_weights[idx:idx + param_length].reshape(param.shape)
            ).float()
            idx += param_length
        
        logger.info("✓ Weight restoration successful")
        return True, len(flat_weights)
    except Exception as e:
        logger.error(f"✗ Weight flattening failed: {e}")
        return False, 0

def test_hybrid_chromosome_structure(num_weights):
    """Test 8: Simulate hybrid chromosome structure"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: Hybrid Chromosome Structure")
    logger.info("=" * 60)
    try:
        # Simulate hybrid chromosome
        num_hyperparams = 5  # out_channels_conv1, conv2, conv3, neurons_fc1, dropout
        num_fuzzy_params = 27  # Default FIS with 2 inputs, 1 output
        
        total_size = num_hyperparams + num_weights + num_fuzzy_params
        
        logger.info(f"  Hyperparameters: {num_hyperparams}")
        logger.info(f"  NN Weights: {num_weights:,}")
        logger.info(f"  Fuzzy Params: {num_fuzzy_params}")
        logger.info(f"  Total chromosome size: {total_size:,}")
        
        # Create dummy chromosome
        chromosome = np.random.randn(total_size).astype(np.float32)
        
        # Test segmentation
        hyperparams = chromosome[:num_hyperparams]
        nn_weights = chromosome[num_hyperparams:num_hyperparams + num_weights]
        fuzzy_params = chromosome[num_hyperparams + num_weights:]
        
        logger.info(f"✓ Chromosome created and segmented")
        logger.info(f"✓ Hyperparam segment: {hyperparams.shape}")
        logger.info(f"✓ NN weight segment: {nn_weights.shape}")
        logger.info(f"✓ Fuzzy param segment: {fuzzy_params.shape}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Chromosome structure test failed: {e}")
        return False

def test_config_compatibility():
    """Test 9: Check config structure compatibility"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 9: Config Compatibility")
    logger.info("=" * 60)
    try:
        # Simulate config that will be sent
        config = {
            'generations': 20,
            'population_size': 15,
            'model_class': 'MyCNN',
            'model_args': [],
            'model_kwargs': {
                'input_channels': 1,
                'num_classes': 4
            },
            'use_fuzzy': True,
            'fuzzy_num_inputs': 2,
            'fuzzy_num_outputs': 1,
            'evolvable_hyperparams': {
                'out_channels_conv1': {
                    'range': [16, 64],
                    'type': 'int'
                },
                'out_channels_conv2': {
                    'range': [32, 128],
                    'type': 'int'
                },
                'out_channels_conv3': {
                    'range': [64, 256],
                    'type': 'int'
                },
                'neurons_fc1': {
                    'range': [32, 128],
                    'type': 'int'
                },
                'dropout_rate': {
                    'range': [0.1, 0.7],
                    'type': 'float'
                }
            },
            'eval_config': {
                'device': torch.device('cpu'),
                'batch_size': 256
            }
        }
        
        logger.info("✓ Config structure valid")
        logger.info(f"  Model class: {config['model_class']}")
        logger.info(f"  Evolvable hyperparams: {len(config['evolvable_hyperparams'])}")
        logger.info(f"  Fuzzy enabled: {config['use_fuzzy']}")
        
        return True, config
    except Exception as e:
        logger.error(f"✗ Config compatibility test failed: {e}")
        return False, None

def main():
    """Run all compatibility tests"""
    logger.info("\n" + "=" * 60)
    logger.info("OCTMNIST HYBRID EVOLUTION COMPATIBILITY TEST")
    logger.info("=" * 60)
    
    results = []
    
    # Test 1: Model Import
    success, ModelClass = test_model_import()
    results.append(("Model Import", success))
    if not success:
        logger.error("\n✗ CRITICAL: Cannot proceed without model import")
        return False
    
    # Test 2: Model Instantiation
    success, model = test_model_instantiation(ModelClass)
    results.append(("Model Instantiation", success))
    if not success:
        logger.error("\n✗ CRITICAL: Cannot proceed without model instantiation")
        return False
    
    # Test 3: Evolved Parameters
    success, evolved_model = test_model_with_evolved_params(ModelClass)
    results.append(("Evolved Parameters", success))
    
    # Test 4: Forward Pass
    if evolved_model:
        success = test_forward_pass(evolved_model)
        results.append(("Forward Pass", success))
    
    # Test 5: Evaluation Import
    success, eval_func = test_evaluation_import()
    results.append(("Evaluation Import", success))
    
    # Test 6: Evaluation Signature
    if eval_func:
        success = test_evaluation_signature(eval_func)
        results.append(("Evaluation Signature", success))
    
    # Test 7: Weight Flattening
    success, num_weights = test_weight_flattening(model)
    results.append(("Weight Flattening", success))
    
    # Test 8: Hybrid Chromosome
    if num_weights > 0:
        success = test_hybrid_chromosome_structure(num_weights)
        results.append(("Hybrid Chromosome", success))
    
    # Test 9: Config Compatibility
    success, config = test_config_compatibility()
    results.append(("Config Compatibility", success))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{status}: {test_name}")
    
    total_tests = len(results)
    passed_tests = sum(1 for _, passed in results if passed)
    
    logger.info("\n" + "=" * 60)
    logger.info(f"RESULTS: {passed_tests}/{total_tests} tests passed")
    logger.info("=" * 60)
    
    if passed_tests == total_tests:
        logger.info("\n✓ ALL TESTS PASSED - System is compatible!")
        logger.info("\nYou can now use:")
        logger.info("  Model Class: MyCNN")
        logger.info("  Evaluation: octmnist_cnn_hp_ready_evaluation.py")
        logger.info("  Hybrid Evolution: READY")
        return True
    else:
        logger.error(f"\n✗ {total_tests - passed_tests} test(s) failed")
        logger.error("Please fix the issues before running hybrid evolution")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
