# app/core/fuzzy.py
"""
Fuzzy Inference System (FIS) for Hybrid Neuro-Fuzzy Evolution
Implements fuzzy logic with evolvable membership functions and rules.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class MembershipFunction:
    """Represents a fuzzy membership function (triangular or gaussian)."""
    name: str
    mf_type: str  # 'triangular', 'gaussian', 'trapezoidal'
    params: np.ndarray  # Parameters defining the shape
    
    def evaluate(self, x: float) -> float:
        """Evaluate membership degree for input x."""
        if self.mf_type == 'triangular':
            # params: [a, b, c] where b is peak
            a, b, c = self.params
            if x <= a or x >= c:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 0.0
            else:  # b < x < c
                return (c - x) / (c - b) if c != b else 0.0
                
        elif self.mf_type == 'gaussian':
            # params: [mean, sigma]
            mean, sigma = self.params
            return np.exp(-0.5 * ((x - mean) / sigma) ** 2)
            
        elif self.mf_type == 'trapezoidal':
            # params: [a, b, c, d]
            a, b, c, d = self.params
            if x <= a or x >= d:
                return 0.0
            elif a < x <= b:
                return (x - a) / (b - a) if b != a else 0.0
            elif b < x <= c:
                return 1.0
            else:  # c < x < d
                return (d - x) / (d - c) if d != c else 0.0
        
        return 0.0


@dataclass
class FuzzyVariable:
    """Represents a fuzzy variable with multiple membership functions."""
    name: str
    range: Tuple[float, float]
    membership_functions: List[MembershipFunction] = field(default_factory=list)
    
    def fuzzify(self, value: float) -> Dict[str, float]:
        """Convert crisp value to fuzzy memberships."""
        memberships = {}
        for mf in self.membership_functions:
            memberships[mf.name] = mf.evaluate(value)
        return memberships


@dataclass
class FuzzyRule:
    """Represents a fuzzy IF-THEN rule."""
    antecedents: Dict[str, str]  # {variable_name: membership_name}
    consequent: Dict[str, str]   # {variable_name: membership_name}
    weight: float = 1.0  # Rule strength (evolvable)
    
    def evaluate(self, input_memberships: Dict[str, Dict[str, float]]) -> float:
        """Evaluate rule activation strength using product T-norm."""
        activation = self.weight
        for var_name, mf_name in self.antecedents.items():
            if var_name in input_memberships and mf_name in input_memberships[var_name]:
                activation *= input_memberships[var_name][mf_name]
            else:
                return 0.0  # Rule doesn't fire if antecedent not found
        return activation


class FuzzyInferenceSystem:
    """
    Fuzzy Inference System with evolvable rules and membership functions.
    Supports Mamdani-style inference.
    """
    
    def __init__(self):
        self.input_variables: Dict[str, FuzzyVariable] = {}
        self.output_variables: Dict[str, FuzzyVariable] = {}
        self.rules: List[FuzzyRule] = []
        
    def add_input_variable(self, variable: FuzzyVariable):
        """Add an input fuzzy variable."""
        self.input_variables[variable.name] = variable
        
    def add_output_variable(self, variable: FuzzyVariable):
        """Add an output fuzzy variable."""
        self.output_variables[variable.name] = variable
        
    def add_rule(self, rule: FuzzyRule):
        """Add a fuzzy rule."""
        self.rules.append(rule)
        
    def infer(self, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Perform fuzzy inference.
        
        Args:
            inputs: Dictionary of {variable_name: crisp_value}
            
        Returns:
            Dictionary of {variable_name: defuzzified_value}
        """
        # Step 1: Fuzzification
        input_memberships = {}
        for var_name, value in inputs.items():
            if var_name in self.input_variables:
                input_memberships[var_name] = self.input_variables[var_name].fuzzify(value)
        
        # Step 2: Rule Evaluation
        outputs = {}
        for out_var_name, out_var in self.output_variables.items():
            # Accumulate weighted outputs for each membership function
            mf_activations = {mf.name: 0.0 for mf in out_var.membership_functions}
            
            for rule in self.rules:
                # Check if rule applies to this output variable
                if out_var_name not in rule.consequent:
                    continue
                    
                # Calculate rule activation
                activation = rule.evaluate(input_memberships)
                
                # Apply to consequent membership function
                consequent_mf = rule.consequent[out_var_name]
                if consequent_mf in mf_activations:
                    mf_activations[consequent_mf] = max(mf_activations[consequent_mf], activation)
            
            # Step 3: Defuzzification (Center of Gravity)
            outputs[out_var_name] = self._defuzzify(out_var, mf_activations)
        
        return outputs
    
    def _defuzzify(self, variable: FuzzyVariable, mf_activations: Dict[str, float]) -> float:
        """Defuzzify using center of gravity method."""
        numerator = 0.0
        denominator = 0.0
        
        # Sample the output space
        min_val, max_val = variable.range
        num_samples = 100
        step = (max_val - min_val) / num_samples
        
        for i in range(num_samples):
            x = min_val + i * step
            # Calculate aggregated membership at this point
            membership = 0.0
            for mf in variable.membership_functions:
                if mf.name in mf_activations:
                    membership = max(membership, min(mf.evaluate(x), mf_activations[mf.name]))
            
            numerator += x * membership
            denominator += membership
        
        return numerator / denominator if denominator > 0 else (min_val + max_val) / 2
    
    def encode_to_chromosome(self) -> np.ndarray:
        """
        Encode FIS parameters into a flat chromosome for evolution.
        
        Returns:
            Flat numpy array containing all evolvable parameters
        """
        params = []
        
        # Encode membership function parameters
        for var in list(self.input_variables.values()) + list(self.output_variables.values()):
            for mf in var.membership_functions:
                params.extend(mf.params.flatten())
        
        # Encode rule weights
        for rule in self.rules:
            params.append(rule.weight)
        
        return np.array(params, dtype=np.float32)
    
    def decode_from_chromosome(self, chromosome: np.ndarray):
        """
        Decode chromosome back into FIS parameters.
        
        Args:
            chromosome: Flat array of parameters
        """
        idx = 0
        
        # Decode membership function parameters
        for var in list(self.input_variables.values()) + list(self.output_variables.values()):
            for mf in var.membership_functions:
                param_len = len(mf.params)
                mf.params = chromosome[idx:idx + param_len].copy()
                idx += param_len
        
        # Decode rule weights
        for rule in self.rules:
            rule.weight = float(chromosome[idx])
            idx += 1
    
    def get_chromosome_size(self) -> int:
        """Calculate the size of the chromosome encoding."""
        size = 0
        
        # Count MF parameters
        for var in list(self.input_variables.values()) + list(self.output_variables.values()):
            for mf in var.membership_functions:
                size += len(mf.params)
        
        # Count rules
        size += len(self.rules)
        
        return size


def create_default_fis(num_inputs: int = 2, num_outputs: int = 1) -> FuzzyInferenceSystem:
    """
    Create a default FIS with triangular membership functions.
    
    Args:
        num_inputs: Number of input variables
        num_outputs: Number of output variables
        
    Returns:
        Initialized FuzzyInferenceSystem
    """
    fis = FuzzyInferenceSystem()
    
    # Create input variables with Low, Medium, High memberships
    for i in range(num_inputs):
        var = FuzzyVariable(name=f"input_{i}", range=(0.0, 1.0))
        var.membership_functions = [
            MembershipFunction(name="low", mf_type="triangular", params=np.array([0.0, 0.0, 0.5])),
            MembershipFunction(name="medium", mf_type="triangular", params=np.array([0.0, 0.5, 1.0])),
            MembershipFunction(name="high", mf_type="triangular", params=np.array([0.5, 1.0, 1.0]))
        ]
        fis.add_input_variable(var)
    
    # Create output variables
    for i in range(num_outputs):
        var = FuzzyVariable(name=f"output_{i}", range=(0.0, 1.0))
        var.membership_functions = [
            MembershipFunction(name="low", mf_type="triangular", params=np.array([0.0, 0.0, 0.5])),
            MembershipFunction(name="medium", mf_type="triangular", params=np.array([0.0, 0.5, 1.0])),
            MembershipFunction(name="high", mf_type="triangular", params=np.array([0.5, 1.0, 1.0]))
        ]
        fis.add_output_variable(var)
    
    # Create basic rules (all combinations for 2 inputs)
    if num_inputs == 2 and num_outputs == 1:
        rule_base = [
            ({"input_0": "low", "input_1": "low"}, {"output_0": "low"}),
            ({"input_0": "low", "input_1": "medium"}, {"output_0": "low"}),
            ({"input_0": "low", "input_1": "high"}, {"output_0": "medium"}),
            ({"input_0": "medium", "input_1": "low"}, {"output_0": "low"}),
            ({"input_0": "medium", "input_1": "medium"}, {"output_0": "medium"}),
            ({"input_0": "medium", "input_1": "high"}, {"output_0": "high"}),
            ({"input_0": "high", "input_1": "low"}, {"output_0": "medium"}),
            ({"input_0": "high", "input_1": "medium"}, {"output_0": "high"}),
            ({"input_0": "high", "input_1": "high"}, {"output_0": "high"}),
        ]
        
        for antecedents, consequent in rule_base:
            fis.add_rule(FuzzyRule(antecedents=antecedents, consequent=consequent, weight=1.0))
    
    return fis
