"""
Neural Architecture Search (NAS) Search Space Implementation

This module provides a comprehensive SearchSpace class for defining and managing
neural architecture search spaces, with integration to all relevant utilities
from the src/utils/ directory.

This module now uses the consolidated search space implementation from src/utils/nas_tas/.
"""

# Import the consolidated search space implementation
from src.utils.nas_tas.search_space import (
    SearchSpace, SearchSpaceConfig, ParameterRange, SearchSpaceType, OptimizationStrategy,
    create_default_nas_search_space, create_tree_search_space
)

# Re-export all the important classes and functions for backward compatibility
__all__ = [
    'SearchSpace', 'SearchSpaceConfig', 'ParameterRange', 'SearchSpaceType', 'OptimizationStrategy',
    'create_default_nas_search_space', 'create_tree_search_space'
]

# Legacy function for backward compatibility
def get_default_search_space() -> SearchSpace:
    """
    Create a default search space with common neural architecture parameters.
    
    Returns:
        SearchSpace instance with default parameters
    """
    return create_default_nas_search_space()


# Example usage and testing
if __name__ == "__main__":
    # Create default search space
    search_space = get_default_search_space()
    
    # Print summary
    summary = search_space.get_summary()
    print("Search Space Summary:")
    print(summary)
    
    # Example objective function
    def example_objective(params):
        """Example objective function for testing."""
        # Simple scoring based on parameter values
        score = 0.0
        
        # Learning rate scoring (prefer moderate values)
        lr = params.get('learning_rate', 1e-3)
        if 1e-4 <= lr <= 1e-2:
            score += 0.3
        
        # Hidden layers scoring (prefer moderate complexity)
        layers = params.get('hidden_layers', 3)
        if 2 <= layers <= 5:
            score += 0.2
        
        # Activation function scoring
        activation = params.get('activation', 'relu')
        if activation == 'relu':
            score += 0.2
        
        # Add some randomness
        import numpy as np
        score += np.random.normal(0, 0.1)
        
        return score
    
    # Run optimization with Grid + TPE strategy
    print("Running example optimization with Grid + TPE strategy...")
    results = search_space.optimize(example_objective)
    
    print("Optimization Results:")
    print(results)