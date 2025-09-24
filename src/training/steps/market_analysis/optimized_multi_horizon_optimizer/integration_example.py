"""
Integration Example for Optimized Multi-Horizon Optimizer

This module demonstrates how to use the optimized multi-horizon optimizer
with ml_commons utilities extensively integrated.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime
import logging

# Import optimized multi-horizon optimizer components
from .optimized_timeframe_optimizer import OptimizedTimeframeOptimizer
from .optimization_config import (
    OptimizationConfig, ModelType, OptimizationMethod, 
    ValidationConfig, ValidationLevel, GridSearchConfig, BayesianTPEConfig
)

# Import multi-horizon components
from src.training.steps.market_analysis.multi_horizon_profit_labeler import MultiHorizonConfig

logger = logging.getLogger(__name__)


def create_sample_market_data() -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate sample OHLCV data
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='5T')
    
    # Generate price data with trend and volatility
    base_price = 100.0
    returns = np.random.normal(0.0001, 0.02, n_samples)  # 0.01% mean return, 2% volatility
    prices = base_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, n_samples))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, n_samples)
    })
    
    return data


def create_optimization_config() -> OptimizationConfig:
    """Create optimization configuration."""
    # Grid search configuration
    grid_config = GridSearchConfig(
        coarse_grid_points=5,
        coarse_enabled=True,
        fine_grid_points=10,
        fine_enabled=True,
        fine_range_percentage=0.2,
        enable_parallel=True,
        max_workers=4,
        timeout_seconds=300
    )
    
    # Bayesian TPE configuration
    bayesian_config = BayesianTPEConfig(
        n_trials=50,
        n_startup_trials=10,
        n_ei_candidates=24,
        enable_parallel=True,
        max_workers=4,
        timeout_seconds=600,
        enable_pruning=True,
        pruning_patience=5,
        pruning_min_trials=10
    )
    
    # Validation configuration
    validation_config = ValidationConfig(
        validation_level=ValidationLevel.COMPREHENSIVE,
        cv_folds=5,
        cv_strategy="time_series",
        enable_statistical_validation=True,
        min_information_coefficient=0.05,
        min_signal_to_noise_ratio=1.0,
        min_hit_rate=0.55,
        enable_economic_validation=True,
        max_transaction_cost_ratio=0.1,
        min_sharpe_ratio=0.5,
        max_drawdown_threshold=0.2,
        enable_microstructure_validation=True,
        min_liquidity_score=0.7,
        min_volatility_stability=0.6
    )
    
    # Main optimization configuration
    config = OptimizationConfig(
        model_type=ModelType.BOTH,
        base_timeframe_analyst=15,  # 15 minutes
        base_timeframe_tactician=5,  # 5 minutes
        horizon_range=(1, 16),  # 1-16 periods
        optimization_method=OptimizationMethod.GRID_BAYESIAN,
        grid_search_config=grid_config,
        bayesian_tpe_config=bayesian_config,
        validation_config=validation_config,
        enable_caching=True,
        cache_ttl_hours=24,
        enable_monitoring=True,
        fast_fail_on_optimization_failure=True,
        min_optimization_score=0.3,
        min_validation_score=0.5,
        enable_detailed_logging=True,
        log_optimization_progress=True
    )
    
    return config


def run_optimization_example():
    """Run complete optimization example."""
    logger.info('🚀 Starting optimized multi-horizon optimization example')
    
    try:
        # Create sample market data
        market_data = create_sample_market_data()
        logger.info(f'📊 Created sample market data with {len(market_data)} samples')
        
        # Create optimization configuration
        config = create_optimization_config()
        logger.info(f'⚙️ Created optimization configuration for {config.model_type.value} model')
        
        # Initialize optimized timeframe optimizer
        optimizer = OptimizedTimeframeOptimizer(config)
        logger.info('🔧 Initialized optimized timeframe optimizer')
        
        # Run optimization for both models
        logger.info('🎯 Starting optimization for both Analyst and Tactician models')
        results = optimizer.get_optimal_timeframes_for_models(
            market_data=market_data,
            model_type=ModelType.BOTH,
            force_optimization=True
        )
        
        # Display results
        logger.info('📊 Optimization Results:')
        for model_type, result in results.items():
            logger.info(f'   {model_type.upper()}:')
            logger.info(f'     → Optimization Score: {result.optimization_score:.3f}')
            logger.info(f'     → Validation Score: {result.validation_score:.3f}')
            logger.info(f'     → Optimal Horizons: {result.optimal_horizons}')
            logger.info(f'     → Optimal Targets: {result.optimal_targets}')
            logger.info(f'     → Optimization Time: {result.optimization_time:.2f}s')
            logger.info(f'     → Method: {result.optimization_method}')
        
        # Get optimization summary
        summary = optimizer.get_optimization_summary()
        logger.info(f'📈 Optimization Summary:')
        logger.info(f'   → Total Optimizations: {summary["total_optimizations"]}')
        logger.info(f'   → History Count: {summary["optimization_history_count"]}')
        logger.info(f'   → Cache Size: {summary["performance_cache_size"]}')
        
        # Export results
        output_file = f"optimization_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        optimizer.export_optimization_results(output_file)
        logger.info(f'💾 Exported optimization results to {output_file}')
        
        logger.info('✅ Optimization example completed successfully')
        
        return results
        
    except Exception as e:
        logger.error(f'❌ Optimization example failed: {e}')
        raise


def run_analyst_optimization_example():
    """Run optimization example for Analyst model only."""
    logger.info('🎯 Starting Analyst model optimization example')
    
    try:
        # Create sample market data
        market_data = create_sample_market_data()
        
        # Create configuration for Analyst model
        config = create_optimization_config()
        config.model_type = ModelType.ANALYST
        
        # Initialize optimizer
        optimizer = OptimizedTimeframeOptimizer(config)
        
        # Run optimization for Analyst model
        result = optimizer.optimize_for_model(
            model_type=ModelType.ANALYST,
            market_data=market_data,
            force_optimization=True
        )
        
        logger.info(f'✅ Analyst optimization completed:')
        logger.info(f'   → Score: {result.optimization_score:.3f}')
        logger.info(f'   → Horizons: {result.optimal_horizons}')
        logger.info(f'   → Targets: {result.optimal_targets}')
        
        return result
        
    except Exception as e:
        logger.error(f'❌ Analyst optimization failed: {e}')
        raise


def run_tactician_optimization_example():
    """Run optimization example for Tactician model only."""
    logger.info('🎯 Starting Tactician model optimization example')
    
    try:
        # Create sample market data
        market_data = create_sample_market_data()
        
        # Create configuration for Tactician model
        config = create_optimization_config()
        config.model_type = ModelType.TACTICIAN
        
        # Initialize optimizer
        optimizer = OptimizedTimeframeOptimizer(config)
        
        # Run optimization for Tactician model
        result = optimizer.optimize_for_model(
            model_type=ModelType.TACTICIAN,
            market_data=market_data,
            force_optimization=True
        )
        
        logger.info(f'✅ Tactician optimization completed:')
        logger.info(f'   → Score: {result.optimization_score:.3f}')
        logger.info(f'   → Horizons: {result.optimal_horizons}')
        logger.info(f'   → Targets: {result.optimal_targets}')
        
        return result
        
    except Exception as e:
        logger.error(f'❌ Tactician optimization failed: {e}')
        raise


def demonstrate_ml_commons_integration():
    """Demonstrate extensive ml_commons integration."""
    logger.info('🔧 Demonstrating ml_commons integration')
    
    try:
        # Create sample data
        market_data = create_sample_market_data()
        
        # Create configuration
        config = create_optimization_config()
        
        # Initialize optimizer
        optimizer = OptimizedTimeframeOptimizer(config)
        
        # Demonstrate ml_commons utilities
        logger.info('   → ml_commons utilities integrated:')
        logger.info('     • Grid search utilities (coarse + fine)')
        logger.info('     • Bayesian TPE optimization')
        logger.info('     • Cross-validation utilities')
        logger.info('     • Unified validation system')
        logger.info('     • Temporal cross-validation')
        logger.info('     • Overfitting detection')
        logger.info('     • Data leakage prevention')
        logger.info('     • Stability validation')
        logger.info('     • Memory optimization')
        logger.info('     • Lookahead protection')
        
        # Run a quick optimization to demonstrate
        result = optimizer.optimize_for_model(
            model_type=ModelType.ANALYST,
            market_data=market_data,
            force_optimization=True
        )
        
        logger.info(f'✅ ml_commons integration demonstrated successfully')
        logger.info(f'   → Result score: {result.optimization_score:.3f}')
        
        return result
        
    except Exception as e:
        logger.error(f'❌ ml_commons integration demonstration failed: {e}')
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run examples
    try:
        # Run complete optimization example
        run_optimization_example()
        
        # Run individual model examples
        run_analyst_optimization_example()
        run_tactician_optimization_example()
        
        # Demonstrate ml_commons integration
        demonstrate_ml_commons_integration()
        
        logger.info('🎉 All examples completed successfully!')
        
    except Exception as e:
        logger.error(f'❌ Example execution failed: {e}')
        raise