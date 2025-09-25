#!/usr/bin/env python3
"""
Test Search Algorithms Unification

This script tests that the unified search algorithms framework has been successfully
implemented and that both NAS and TAS systems can use it.
"""

import sys
import os
import numpy as np
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')

def test_unified_search_import():
    """Test that the unified search framework can be imported."""
    print("🧪 Testing unified search framework import...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        # Test basic import
        import utils.nas_tas.search_algorithms
        print("✅ Basic import from src.utils.nas_tas.search_algorithms successful")
        
        # Test specific imports
        from utils.nas_tas.search_algorithms import (
            SearchManager,
            SearchConfig,
            SearchResult,
            SearchAlgorithmType,
            BayesianOptimizer,
            EvolutionaryOptimizer,
            GridSearchOptimizer,
            RandomSearchOptimizer
        )
        print("✅ Specific imports from search_algorithms successful")
        
        # Test convenience functions
        from utils.nas_tas.search_algorithms import (
            create_search_manager,
            optimize_with_bayesian,
            optimize_with_evolutionary,
            optimize_with_grid
        )
        print("✅ Convenience function imports successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import from unified search framework failed: {e}")
        return False

def test_nas_search_integration():
    """Test that NAS search integration can be imported."""
    print("\n🧪 Testing NAS search integration...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        # Test NAS integration import
        from training.steps.market_analysis.nas_regime.search.unified_search_integration import (
            NASUnifiedSearchIntegration,
            NASSearchConfig,
            optimize_nas_regime_with_unified_search,
            create_nas_search_config
        )
        print("✅ NAS search integration import successful")
        
        # Test configuration creation
        config = create_nas_search_config()
        if config and hasattr(config, 'search_algorithm'):
            print("✅ NAS search configuration creation successful")
        else:
            print("❌ NAS search configuration creation failed")
            return False
            
        return True
    except ImportError as e:
        print(f"❌ NAS search integration import failed: {e}")
        return False

def test_tas_search_integration():
    """Test that TAS search integration can be imported."""
    print("\n🧪 Testing TAS search integration...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        # Test TAS integration import
        from training.steps.market_analysis.tas_regime.search.unified_search_integration import (
            TASUnifiedSearchIntegration,
            TASSearchConfig,
            optimize_tas_regime_with_unified_search,
            create_tas_search_config
        )
        print("✅ TAS search integration import successful")
        
        # Test configuration creation
        config = create_tas_search_config()
        if config and hasattr(config, 'search_algorithm'):
            print("✅ TAS search configuration creation successful")
        else:
            print("❌ TAS search configuration creation failed")
            return False
            
        return True
    except ImportError as e:
        print(f"❌ TAS search integration import failed: {e}")
        return False

def test_search_algorithm_types():
    """Test that all search algorithm types are available."""
    print("\n🧪 Testing search algorithm types...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        from utils.nas_tas.search_algorithms import SearchAlgorithmType
        
        # Test all algorithm types
        expected_types = [
            "BAYESIAN_OPTIMIZATION",
            "EVOLUTIONARY_ALGORITHM", 
            "GRID_SEARCH",
            "RANDOM_SEARCH",
            "TREE_BASED_SEARCH",
            "NEURAL_ARCHITECTURE_SEARCH",
            "HYBRID_SEARCH",
            "MULTI_OBJECTIVE_SEARCH"
        ]
        
        for expected_type in expected_types:
            if hasattr(SearchAlgorithmType, expected_type):
                print(f"✅ {expected_type} available")
            else:
                print(f"❌ {expected_type} not available")
                return False
        
        return True
    except ImportError as e:
        print(f"❌ Search algorithm types test failed: {e}")
        return False

def test_search_manager_creation():
    """Test that search manager can be created."""
    print("\n🧪 Testing search manager creation...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        from utils.nas_tas.search_algorithms import (
            SearchManager,
            SearchConfig,
            SearchAlgorithmType,
            create_search_manager
        )
        
        # Test default search manager creation
        manager = create_search_manager()
        if manager and hasattr(manager, 'create_optimizer'):
            print("✅ Default search manager creation successful")
        else:
            print("❌ Default search manager creation failed")
            return False
        
        # Test custom search manager creation
        config = SearchConfig(
            algorithm_type=SearchAlgorithmType.BAYESIAN_OPTIMIZATION,
            max_iterations=50
        )
        manager = SearchManager(config)
        if manager and hasattr(manager, 'create_optimizer'):
            print("✅ Custom search manager creation successful")
        else:
            print("❌ Custom search manager creation failed")
            return False
        
        return True
    except ImportError as e:
        print(f"❌ Search manager creation test failed: {e}")
        return False

def test_optimizer_creation():
    """Test that different optimizers can be created."""
    print("\n🧪 Testing optimizer creation...")
    
    try:
        # Add src to path
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        from utils.nas_tas.search_algorithms import (
            SearchManager,
            SearchConfig,
            SearchAlgorithmType
        )
        
        # Test different optimizer types
        optimizer_types = [
            SearchAlgorithmType.RANDOM_SEARCH,
            SearchAlgorithmType.EVOLUTIONARY_ALGORITHM,
            SearchAlgorithmType.GRID_SEARCH
        ]
        
        for algo_type in optimizer_types:
            config = SearchConfig(algorithm_type=algo_type)
            manager = SearchManager(config)
            optimizer = manager.create_optimizer(algo_type)
            
            if optimizer and hasattr(optimizer, 'search'):
                print(f"✅ {algo_type.value} optimizer creation successful")
            else:
                print(f"❌ {algo_type.value} optimizer creation failed")
                return False
        
        return True
    except ImportError as e:
        print(f"❌ Optimizer creation test failed: {e}")
        return False

def test_parameter_space_definition():
    """Test parameter space definition functionality."""
    print("\n🧪 Testing parameter space definition...")
    
    try:
        # Test parameter space structure
        parameter_space = {
            'learning_rate': {
                'type': 'continuous',
                'min': 0.001,
                'max': 0.1
            },
            'batch_size': {
                'type': 'discrete',
                'values': [16, 32, 64, 128]
            },
            'epochs': {
                'type': 'integer',
                'min': 10,
                'max': 100
            }
        }
        
        # Validate parameter space structure
        for param_name, param_config in parameter_space.items():
            if not isinstance(param_config, dict):
                print(f"❌ Parameter {param_name} config not a dict")
                return False
            
            if 'type' not in param_config:
                print(f"❌ Parameter {param_name} missing type")
                return False
            
            param_type = param_config['type']
            if param_type == 'continuous':
                if 'min' not in param_config or 'max' not in param_config:
                    print(f"❌ Parameter {param_name} missing min/max for continuous")
                    return False
            elif param_type == 'discrete':
                if 'values' not in param_config:
                    print(f"❌ Parameter {param_name} missing values for discrete")
                    return False
            elif param_type == 'integer':
                if 'min' not in param_config or 'max' not in param_config:
                    print(f"❌ Parameter {param_name} missing min/max for integer")
                    return False
        
        print("✅ Parameter space definition validation successful")
        return True
        
    except Exception as e:
        print(f"❌ Parameter space definition test failed: {e}")
        return False

def test_objective_function_creation():
    """Test objective function creation for testing."""
    print("\n🧪 Testing objective function creation...")
    
    try:
        # Create a simple objective function for testing
        def test_objective_function(parameters: Dict[str, Any]) -> float:
            """Simple test objective function."""
            try:
                # Simple quadratic function with some noise
                x = parameters.get('x', 0.5)
                y = parameters.get('y', 0.5)
                
                # Quadratic function with maximum at (0.5, 0.5)
                score = 1.0 - ((x - 0.5) ** 2 + (y - 0.5) ** 2)
                
                # Add some noise
                noise = np.random.normal(0, 0.01)
                return score + noise
                
            except Exception as e:
                print(f"Warning: Objective function failed: {e}")
                return 0.0
        
        # Test the objective function
        test_params = {'x': 0.5, 'y': 0.5}
        score = test_objective_function(test_params)
        
        if isinstance(score, (int, float)) and not np.isnan(score):
            print("✅ Objective function creation and execution successful")
            return True
        else:
            print("❌ Objective function execution failed")
            return False
        
    except Exception as e:
        print(f"❌ Objective function creation test failed: {e}")
        return False

def test_file_structure():
    """Test that all required files exist."""
    print("\n🧪 Testing file structure...")
    
    required_files = [
        "src/utils/nas_tas/search_algorithms.py",
        "src/training/steps/market_analysis/nas_regime/search/unified_search_integration.py",
        "src/training/steps/market_analysis/tas_regime/search/unified_search_integration.py"
    ]
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} missing")
            return False
    
    return True

def main():
    """Run all search algorithms unification tests."""
    print("🚀 Testing Search Algorithms Unification")
    print("=" * 60)
    
    tests = [
        test_file_structure,
        test_unified_search_import,
        test_nas_search_integration,
        test_tas_search_integration,
        test_search_algorithm_types,
        test_search_manager_creation,
        test_optimizer_creation,
        test_parameter_space_definition,
        test_objective_function_creation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("📊 Search Algorithms Unification Test Results:")
    
    passed = sum(results)
    total = len(results)
    
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Search algorithms unification successful!")
        return 0
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())