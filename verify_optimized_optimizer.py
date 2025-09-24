#!/usr/bin/env python3
"""
Verification Script for Optimized Multi-Horizon Optimizer

This script verifies that the optimized multi-horizon optimizer is correctly
implemented with ml_commons integration and all components are working.
"""

import sys
import os
import logging
from pathlib import Path

# Add workspace to path
workspace_path = Path(__file__).parent
sys.path.insert(0, str(workspace_path))

def setup_logging():
    """Setup logging for verification."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger('Verification')

def verify_imports():
    """Verify all imports are working."""
    logger = logging.getLogger('Verification')
    logger.info('🔍 Verifying imports...')
    
    try:
        # Test optimized multi-horizon optimizer imports
        from src.training.steps.market_analysis.optimized_multi_horizon_optimizer import (
            OptimizedTimeframeOptimizer,
            GridBayesianOptimizer,
            EnhancedValidationFramework,
            OptimizationConfig,
            ModelType,
            OptimizationMethod
        )
        logger.info('✅ Optimized multi-horizon optimizer imports successful')
        
        # Test ml_commons imports
        from src.utils.ml_common.utils.hpo_utils import HyperparameterOptimization
        from src.utils.ml_common.optimization.grid_utils import (
            build_coarse_grid_from_search_space,
            build_fine_grid_around_best
        )
        from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem
        from src.utils.ml_common.validation.temporal_cross_validation import TemporalCrossValidator
        from src.utils.ml_common.validation.cv_utils import CrossValidationUtilities
        logger.info('✅ ml_commons imports successful')
        
        return True
        
    except ImportError as e:
        logger.error(f'❌ Import failed: {e}')
        return False

def verify_configuration():
    """Verify configuration classes are working."""
    logger = logging.getLogger('Verification')
    logger.info('🔍 Verifying configuration...')
    
    try:
        from src.training.steps.market_analysis.optimized_multi_horizon_optimizer import (
            OptimizationConfig, ModelType, OptimizationMethod, ValidationLevel
        )
        
        # Test configuration creation
        config = OptimizationConfig(
            model_type=ModelType.BOTH,
            optimization_method=OptimizationMethod.GRID_BAYESIAN,
            base_timeframe_analyst=15,
            base_timeframe_tactician=5,
            horizon_range=(1, 16)
        )
        
        logger.info(f'✅ Configuration created: {config.model_type.value}')
        logger.info(f'   → Optimization method: {config.optimization_method.value}')
        logger.info(f'   → Analyst timeframe: {config.base_timeframe_analyst}m')
        logger.info(f'   → Tactician timeframe: {config.base_timeframe_tactician}m')
        logger.info(f'   → Horizon range: {config.horizon_range}')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ Configuration verification failed: {e}')
        return False

def verify_ml_commons_integration():
    """Verify ml_commons integration is working."""
    logger = logging.getLogger('Verification')
    logger.info('🔍 Verifying ml_commons integration...')
    
    try:
        # Test ml_commons utilities
        from src.utils.ml_common.utils.hpo_utils import HyperparameterOptimization
        from src.utils.ml_common.optimization.grid_utils import (
            build_coarse_grid_from_search_space,
            build_fine_grid_around_best
        )
        from src.utils.ml_common.validation.unified_validation_system import UnifiedValidationSystem
        
        # Test HPO utility
        hpo_config = {
            'enable_parallel': True,
            'max_workers': 4,
            'enable_monitoring': True
        }
        hpo_optimizer = HyperparameterOptimization(hpo_config)
        logger.info('✅ HPO utility initialized')
        
        # Test grid utilities
        search_space = {
            'horizon_immediate': {'type': 'int', 'low': 1, 'high': 16},
            'horizon_short': {'type': 'int', 'low': 1, 'high': 16},
            'target_micro': {'type': 'float', 'low': 0.001, 'high': 0.010}
        }
        
        coarse_grid = build_coarse_grid_from_search_space(search_space, 5)
        logger.info(f'✅ Coarse grid generated: {len(coarse_grid)} points')
        
        if coarse_grid:
            fine_grid = build_fine_grid_around_best(search_space, coarse_grid[0], 10)
            logger.info(f'✅ Fine grid generated: {len(fine_grid)} points')
        
        # Test validation system
        validation_system = UnifiedValidationSystem()
        logger.info('✅ Validation system initialized')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ ml_commons integration verification failed: {e}')
        return False

def verify_optimizer_initialization():
    """Verify optimizer can be initialized."""
    logger = logging.getLogger('Verification')
    logger.info('🔍 Verifying optimizer initialization...')
    
    try:
        from src.training.steps.market_analysis.optimized_multi_horizon_optimizer import (
            OptimizedTimeframeOptimizer,
            OptimizationConfig,
            ModelType,
            OptimizationMethod
        )
        
        # Create configuration
        config = OptimizationConfig(
            model_type=ModelType.ANALYST,
            optimization_method=OptimizationMethod.GRID_BAYESIAN,
            base_timeframe_analyst=15,
            base_timeframe_tactician=5,
            horizon_range=(1, 16)
        )
        
        # Initialize optimizer
        optimizer = OptimizedTimeframeOptimizer(config)
        logger.info('✅ Optimized timeframe optimizer initialized')
        
        # Test optimizer methods
        summary = optimizer.get_optimization_summary()
        logger.info(f'✅ Optimization summary: {summary}')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ Optimizer initialization verification failed: {e}')
        return False

def verify_directory_structure():
    """Verify directory structure is correct."""
    logger = logging.getLogger('Verification')
    logger.info('🔍 Verifying directory structure...')
    
    try:
        optimizer_dir = Path('src/training/steps/market_analysis/optimized_multi_horizon_optimizer')
        
        required_files = [
            '__init__.py',
            'optimization_config.py',
            'grid_bayesian_optimizer.py',
            'enhanced_validation.py',
            'optimized_timeframe_optimizer.py',
            'integration_example.py',
            'README.md'
        ]
        
        for file in required_files:
            file_path = optimizer_dir / file
            if file_path.exists():
                logger.info(f'✅ {file} exists')
            else:
                logger.error(f'❌ {file} missing')
                return False
        
        logger.info('✅ Directory structure verification passed')
        return True
        
    except Exception as e:
        logger.error(f'❌ Directory structure verification failed: {e}')
        return False

def verify_integration_example():
    """Verify integration example can be imported."""
    logger = logging.getLogger('Verification')
    logger.info('🔍 Verifying integration example...')
    
    try:
        from src.training.steps.market_analysis.optimized_multi_horizon_optimizer.integration_example import (
            create_sample_market_data,
            create_optimization_config,
            run_optimization_example
        )
        
        # Test sample data creation
        market_data = create_sample_market_data()
        logger.info(f'✅ Sample market data created: {len(market_data)} samples')
        
        # Test configuration creation
        config = create_optimization_config()
        logger.info(f'✅ Optimization config created: {config.model_type.value}')
        
        logger.info('✅ Integration example verification passed')
        return True
        
    except Exception as e:
        logger.error(f'❌ Integration example verification failed: {e}')
        return False

def main():
    """Run all verification tests."""
    logger = setup_logging()
    logger.info('🚀 Starting optimized multi-horizon optimizer verification')
    
    tests = [
        ("Directory Structure", verify_directory_structure),
        ("Imports", verify_imports),
        ("Configuration", verify_configuration),
        ("ml_commons Integration", verify_ml_commons_integration),
        ("Optimizer Initialization", verify_optimizer_initialization),
        ("Integration Example", verify_integration_example)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.info(f'\n🔍 Running {test_name} test...')
        try:
            if test_func():
                logger.info(f'✅ {test_name} test passed')
                passed += 1
            else:
                logger.error(f'❌ {test_name} test failed')
        except Exception as e:
            logger.error(f'❌ {test_name} test failed with exception: {e}')
    
    logger.info(f'\n📊 Verification Results: {passed}/{total} tests passed')
    
    if passed == total:
        logger.info('🎉 All verification tests passed!')
        logger.info('✅ Optimized multi-horizon optimizer is ready for use')
        return True
    else:
        logger.error(f'❌ {total - passed} tests failed')
        logger.error('❌ Optimized multi-horizon optimizer needs fixes')
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)