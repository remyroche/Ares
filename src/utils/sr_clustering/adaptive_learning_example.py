"""
Example: Adaptive Learning for Small Sample SR Level Analysis

This example demonstrates how the adaptive learning system works with different sample sizes,
automatically adjusting the learning strategy and feature selection to prevent overfitting
while still learning useful rules from small datasets.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_sample_backtest_results(n_samples: int) -> List[Any]:
    """Create sample backtest results for testing."""
    from src.utils.sr_clustering.sr_backtesting_engine import BacktestResult, SRLevel
    
    results = []
    np.random.seed(42)  # For reproducible results
    
    for i in range(n_samples):
        # Create a mock SR level
        level = SRLevel(
            price=100.0 + np.random.normal(0, 5),
            level_type='support' if i % 2 == 0 else 'resistance',
            strength=0.5 + np.random.random() * 0.5,
            detection_time=pd.Timestamp.now(),
            touches=2 + np.random.randint(0, 5)
        )
        
        # Create backtest result with realistic metrics
        success_rate = 0.3 + np.random.random() * 0.6  # 30-90% success rate
        avg_bounce_strength = 0.001 + np.random.random() * 0.01  # 0.1-1.1% bounce
        total_touches = 2 + np.random.randint(0, 8)
        time_persistence = 0.1 + np.random.random() * 0.9
        total_volume_at_level = 1000 + np.random.randint(0, 9000)
        avg_hold_time = 1 + np.random.randint(0, 24)
        
        # Calculate quality score based on these metrics
        quality_score = (
            success_rate * 0.3 +
            min(avg_bounce_strength * 100, 1.0) * 0.25 +
            min(total_touches / 10, 1.0) * 0.2 +
            time_persistence * 0.15 +
            min(total_volume_at_level / 10000, 1.0) * 0.1
        )
        
        result = BacktestResult(
            level=level,
            total_touches=total_touches,
            successful_touches=int(total_touches * success_rate),
            failed_touches=int(total_touches * (1 - success_rate)),
            success_rate=success_rate,
            avg_bounce_strength=avg_bounce_strength,
            max_bounce_strength=avg_bounce_strength * (1 + np.random.random()),
            avg_hold_time=avg_hold_time,
            total_volume_at_level=total_volume_at_level,
            price_deviation=0.01 + np.random.random() * 0.05,
            time_persistence=time_persistence,
            quality_score=quality_score
        )
        
        results.append(result)
    
    return results

def create_sample_market_data(n_samples: int) -> pd.DataFrame:
    """Create sample market data for testing."""
    np.random.seed(42)
    
    dates = pd.date_range(start='2024-01-01', periods=n_samples, freq='H')
    
    data = pd.DataFrame({
        'timestamp': dates,
        'open': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
        'high': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)) + np.random.random(n_samples) * 2,
        'low': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)) - np.random.random(n_samples) * 2,
        'close': 100 + np.cumsum(np.random.normal(0, 0.5, n_samples)),
        'volume': 1000 + np.random.randint(0, 9000, n_samples)
    })
    
    return data

def test_adaptive_learning():
    """Test adaptive learning with different sample sizes."""
    from src.utils.sr_clustering.sr_backtesting_engine import SRBacktestingEngine, BacktestConfig
    
    # Test different sample sizes
    sample_sizes = [10, 20, 30, 50, 100]
    
    for n_samples in sample_sizes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing with {n_samples} samples")
        logger.info(f"{'='*60}")
        
        # Create sample data
        results = create_sample_backtest_results(n_samples)
        market_data = create_sample_market_data(n_samples)
        
        # Configure adaptive learning
        config = BacktestConfig(
            enable_adaptive_learning=True,
            use_feature_selection=True,
            min_samples_for_learning=10,
            max_features_per_sample_ratio=0.3,
            conservative_learning_threshold=50,
            minimal_learning_threshold=20
        )
        
        # Create backtesting engine
        engine = SRBacktestingEngine(config)
        
        # Learn quality rules
        try:
            rules = engine.learn_quality_rules(results, optimize_weights=True, market_data=market_data)
            
            logger.info(f"✅ Learning completed for {n_samples} samples")
            logger.info(f"Learning strategy: {rules.get('learning_strategy', 'unknown')}")
            logger.info(f"Selected features: {rules.get('selected_features', [])}")
            logger.info(f"Overfitting protection: {rules.get('overfitting_protection', 'unknown')}")
            
            if 'overfitting_metrics' in rules:
                metrics = rules['overfitting_metrics']
                logger.info(f"Overfitting risk: {metrics.get('overfitting_risk', 'unknown')}")
                logger.info(f"Sample-to-feature ratio: {metrics.get('sample_to_feature_ratio', 0):.2f}")
            
            # Show learned weights
            learned_weights = rules.get('learned_weights', {})
            if learned_weights:
                logger.info("Learned weights:")
                for feature, weight in learned_weights.items():
                    logger.info(f"  {feature}: {weight:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Learning failed for {n_samples} samples: {e}")

def test_adaptive_feature_selection():
    """Test adaptive feature selection with different sample sizes."""
    from src.feature_selection.specialized.adaptive_selector import AdaptiveFeatureSelector, AdaptiveFeatureSelectionConfig
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing Adaptive Feature Selection")
    logger.info(f"{'='*60}")
    
    # Test different sample sizes
    sample_sizes = [10, 20, 30, 50, 100]
    
    for n_samples in sample_sizes:
        logger.info(f"\n--- Testing with {n_samples} samples ---")
        
        # Create sample data
        np.random.seed(42)
        X = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, n_samples),
            'feature_2': np.random.normal(0, 1, n_samples),
            'feature_3': np.random.normal(0, 1, n_samples),
            'feature_4': np.random.normal(0, 1, n_samples),
            'feature_5': np.random.normal(0, 1, n_samples),
            'feature_6': np.random.normal(0, 1, n_samples),
            'feature_7': np.random.normal(0, 1, n_samples),
            'feature_8': np.random.normal(0, 1, n_samples),
            'feature_9': np.random.normal(0, 1, n_samples),
            'feature_10': np.random.normal(0, 1, n_samples),
        })
        
        # Create target with some correlation to features
        y = (X['feature_1'] * 0.3 + 
             X['feature_2'] * 0.2 + 
             X['feature_3'] * 0.1 + 
             np.random.normal(0, 0.5, n_samples))
        
        # Configure adaptive feature selection
        config = AdaptiveFeatureSelectionConfig(
            min_samples_absolute=10,
            min_samples_per_feature=2.0,
            max_features_absolute=8,
            conservative_mode_threshold=30
        )
        
        # Use adaptive feature selection
        selector = get_adaptive_feature_selector(config)
        result = selector.select_features(X, y)
        
        logger.info(f"Selected {len(result.selected_features)} features: {result.selected_features}")
        logger.info(f"Selection method: {result.selection_method}")
        logger.info(f"Overfitting risk: {result.overfitting_risk}")
        logger.info(f"Selection confidence: {result.selection_confidence:.2f}")

def test_weight_optimization():
    """Test adaptive weight optimization with different sample sizes."""
    from src.utils.sr_clustering.weight_optimization_engine import get_weight_optimization_engine, WeightOptimizationConfig
    
    logger.info(f"\n{'='*60}")
    logger.info("Testing Adaptive Weight Optimization")
    logger.info(f"{'='*60}")
    
    # Test different sample sizes
    sample_sizes = [10, 20, 30, 50, 100]
    
    for n_samples in sample_sizes:
        logger.info(f"\n--- Testing with {n_samples} samples ---")
        
        # Create sample data
        results = create_sample_backtest_results(n_samples)
        market_data = create_sample_market_data(n_samples)
        
        # Configure adaptive weight optimization
        config = WeightOptimizationConfig(
            enable_adaptive_optimization=True,
            min_samples_for_optimization=10,
            max_features_per_sample_ratio=0.5,
            small_sample_mode_threshold=30,
            minimal_optimization_threshold=15
        )
        
        # Use adaptive weight optimization
        optimizer = get_weight_optimization_engine(config)
        result = optimizer.optimize_weights(results, market_data)
        
        logger.info(f"Optimization success: {result.get('optimization_success', False)}")
        logger.info(f"Optimization method: {result.get('method', 'unknown')}")
        logger.info(f"Best score: {result.get('best_score', 0.0):.4f}")
        
        best_weights = result.get('best_weights', {})
        if best_weights:
            logger.info("Best weights:")
            for feature, weight in best_weights.items():
                logger.info(f"  {feature}: {weight:.3f}")

if __name__ == "__main__":
    logger.info("Starting Adaptive Learning Tests")
    
    # Test adaptive learning
    test_adaptive_learning()
    
    # Test adaptive feature selection
    test_adaptive_feature_selection()
    
    # Test weight optimization
    test_weight_optimization()
    
    logger.info("\n✅ All tests completed!")