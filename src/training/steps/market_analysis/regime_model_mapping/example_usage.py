"""
Example Usage of Data-Driven Regime-to-Model Mapping System

This example demonstrates how to use the new data-driven model selection system
with both NAS and TAS regime detection systems.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any
import logging
import time

# Import the new components
from .data_driven_model_selector import DataDrivenModelSelector, ModelSelectorConfig
from .nas_integration import NASModelSelector
from .tas_integration import TASModelSelector

# Import regime detection systems
from ..nas_regime.core.perfect_nas_config import PerfectNASConfig
from ..tas_regime.core.tas_regime_config import TASRegimeConfig

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_sample_market_data(n_samples: int = 1000, n_features: int = 10) -> np.ndarray:
    """Generate sample market data for testing."""
    np.random.seed(42)
    
    # Generate realistic market data with multiple regimes
    data = []
    for i in range(n_samples):
        # Simulate different market regimes
        if i < n_samples // 3:
            # Bull market regime
            regime_data = np.random.normal(0.02, 0.1, n_features)
        elif i < 2 * n_samples // 3:
            # Bear market regime
            regime_data = np.random.normal(-0.01, 0.15, n_features)
        else:
            # Sideways market regime
            regime_data = np.random.normal(0.0, 0.05, n_features)
        
        data.append(regime_data)
    
    return np.array(data)


def example_nas_model_selection():
    """Example of using NAS model selection system."""
    logger.info("🚀 Starting NAS Model Selection Example")
    
    try:
        # Generate sample data
        market_data = generate_sample_market_data(1000, 10)
        timestamps = np.arange(len(market_data))
        
        # Create NAS configuration
        nas_config = PerfectNASConfig.create_short_term_trading_config()
        
        # Create model selector configuration
        selector_config = ModelSelectorConfig(
            min_samples_for_evaluation=50,
            confidence_threshold=0.7,
            enable_ensemble=True,
            enable_continuous_learning=True,
            primary_metric="f1_score"
        )
        
        # Initialize NAS model selector
        nas_selector = NASModelSelector(nas_config, selector_config)
        
        # Detect regimes and select models
        logger.info("🔍 Detecting regimes and selecting models...")
        result = nas_selector.detect_regimes_and_select_models(
            market_data=market_data,
            timestamps=timestamps
        )
        
        if result['success']:
            logger.info("✅ NAS regime detection and model selection completed successfully")
            logger.info(f"   Execution time: {result['execution_time']:.2f}s")
            logger.info(f"   Regimes detected: {result['metadata']['n_regimes_detected']}")
            logger.info(f"   Models available: {result['metadata']['models_available']}")
            
            # Display regime-model mappings
            for regime_id, selection in result['regime_model_selections'].items():
                logger.info(f"   Regime {regime_id}: {selection['selected_model']} "
                          f"(confidence: {selection['regime_confidence']:.3f})")
            
            # Get system summary
            summary = nas_selector.get_system_summary()
            logger.info(f"   System health: {summary['system_health']}")
            logger.info(f"   Total regimes: {summary['total_regimes']}")
            logger.info(f"   Total model-regime pairs: {summary['total_model_regime_pairs']}")
            
        else:
            logger.error(f"❌ NAS model selection failed: {result['error_message']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ NAS model selection example failed: {e}")
        return None


def example_tas_model_selection():
    """Example of using TAS model selection system."""
    logger.info("🌲 Starting TAS Model Selection Example")
    
    try:
        # Generate sample data
        market_data = generate_sample_market_data(1000, 10)
        timestamps = np.arange(len(market_data))
        
        # Create TAS configuration
        tas_config = TASRegimeConfig(
            primary_architecture=TASArchitectureType.HYBRID,
            n_regimes=3,
            enable_statistical_methods=True,
            enable_economic_evaluation=True,
            enable_meta_learning=True
        )
        
        # Create model selector configuration
        selector_config = ModelSelectorConfig(
            min_samples_for_evaluation=50,
            confidence_threshold=0.7,
            enable_ensemble=True,
            enable_continuous_learning=True,
            primary_metric="f1_score"
        )
        
        # Initialize TAS model selector
        tas_selector = TASModelSelector(tas_config, selector_config)
        
        # Detect regimes and select models
        logger.info("🔍 Detecting regimes and selecting models...")
        result = tas_selector.detect_regimes_and_select_models(
            market_data=market_data,
            timestamps=timestamps
        )
        
        if result['success']:
            logger.info("✅ TAS regime detection and model selection completed successfully")
            logger.info(f"   Execution time: {result['execution_time']:.2f}s")
            logger.info(f"   Regimes detected: {result['metadata']['n_regimes_detected']}")
            logger.info(f"   Models available: {result['metadata']['models_available']}")
            
            # Display regime-model mappings
            for regime_id, selection in result['regime_model_selections'].items():
                logger.info(f"   Regime {regime_id}: {selection['selected_model']} "
                          f"(confidence: {selection['regime_confidence']:.3f})")
            
            # Get system summary
            summary = tas_selector.get_system_summary()
            logger.info(f"   System health: {summary['system_health']}")
            logger.info(f"   Total regimes: {summary['total_regimes']}")
            logger.info(f"   Total model-regime pairs: {summary['total_model_regime_pairs']}")
            
        else:
            logger.error(f"❌ TAS model selection failed: {result['error_message']}")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ TAS model selection example failed: {e}")
        return None


def example_continuous_learning():
    """Example of continuous learning with model performance updates."""
    logger.info("🧠 Starting Continuous Learning Example")
    
    try:
        # Generate initial data
        initial_data = generate_sample_market_data(500, 10)
        initial_timestamps = np.arange(len(initial_data))
        
        # Create model selector
        selector_config = ModelSelectorConfig(
            min_samples_for_evaluation=50,
            enable_continuous_learning=True,
            retraining_frequency=100
        )
        
        # Initialize selector
        model_selector = DataDrivenModelSelector(selector_config)
        
        # Simulate initial regime detection (simplified)
        regime_predictions = np.random.randint(0, 3, len(initial_data))
        
        # Simulate model performance updates
        logger.info("📊 Simulating model performance updates...")
        
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = initial_data[regime_mask]
            
            # Simulate different models performing differently in different regimes
            models = ['neural_ode', 'vision_transformer', 'tree_based_clustering', 'statistical_validation']
            
            for model_name in models:
                # Simulate predictions and actual values
                predictions = np.random.randint(0, 3, len(regime_data))
                actual_values = regime_predictions[regime_mask]
                
                # Simulate execution time
                execution_time = np.random.uniform(0.1, 1.0)
                
                # Update model performance
                metrics = model_selector.register_model_performance(
                    regime_id=regime_id,
                    model_name=model_name,
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=execution_time,
                    regime_characteristics={
                        'volatility': np.std(regime_data),
                        'data_size': len(regime_data),
                        'complexity_score': np.random.uniform(0.3, 0.8)
                    }
                )
                
                logger.info(f"   Updated {model_name} in regime {regime_id}: "
                          f"F1={metrics.f1_score:.3f}, Accuracy={metrics.accuracy:.3f}")
        
        # Get regime insights
        logger.info("🔍 Getting regime insights...")
        for regime_id in np.unique(regime_predictions):
            insights = model_selector.get_regime_insights(regime_id)
            logger.info(f"   Regime {regime_id} insights:")
            logger.info(f"     Performance trend: {insights['performance_trend']}")
            logger.info(f"     Recommendations: {insights['recommendations']}")
        
        # Get system summary
        summary = model_selector.get_system_summary()
        logger.info(f"✅ Continuous learning example completed")
        logger.info(f"   Total regimes: {summary['total_regimes']}")
        logger.info(f"   Total model-regime pairs: {summary['total_model_regime_pairs']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"❌ Continuous learning example failed: {e}")
        return None


def example_ensemble_selection():
    """Example of ensemble model selection."""
    logger.info("🎯 Starting Ensemble Model Selection Example")
    
    try:
        # Generate sample data
        market_data = generate_sample_market_data(800, 10)
        
        # Create model selector with ensemble enabled
        selector_config = ModelSelectorConfig(
            enable_ensemble=True,
            max_ensemble_models=3,
            ensemble_weight_threshold=0.1,
            primary_metric="f1_score"
        )
        
        model_selector = DataDrivenModelSelector(selector_config)
        
        # Simulate regime detection
        regime_predictions = np.random.randint(0, 3, len(market_data))
        
        # Simulate model performance for ensemble
        models = ['model_a', 'model_b', 'model_c', 'model_d']
        
        for regime_id in np.unique(regime_predictions):
            regime_mask = regime_predictions == regime_id
            regime_data = market_data[regime_mask]
            
            for model_name in models:
                # Simulate different performance levels
                base_performance = np.random.uniform(0.6, 0.9)
                predictions = np.random.randint(0, 3, len(regime_data))
                actual_values = regime_predictions[regime_mask]
                
                # Update performance
                model_selector.register_model_performance(
                    regime_id=regime_id,
                    model_name=model_name,
                    predictions=predictions,
                    actual_values=actual_values,
                    execution_time=np.random.uniform(0.1, 0.5)
                )
        
        # Get ensemble weights for each regime
        logger.info("🎯 Getting ensemble weights...")
        for regime_id in np.unique(regime_predictions):
            ensemble_weights = model_selector.get_ensemble_weights(
                regime_id=regime_id,
                available_models=models
            )
            
            logger.info(f"   Regime {regime_id} ensemble weights:")
            for model, weight in ensemble_weights.items():
                logger.info(f"     {model}: {weight:.3f}")
        
        logger.info("✅ Ensemble selection example completed")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Ensemble selection example failed: {e}")
        return False


def main():
    """Run all examples."""
    logger.info("🚀 Starting Data-Driven Regime-to-Model Mapping Examples")
    
    # Run examples
    examples = [
        ("NAS Model Selection", example_nas_model_selection),
        ("TAS Model Selection", example_tas_model_selection),
        ("Continuous Learning", example_continuous_learning),
        ("Ensemble Selection", example_ensemble_selection)
    ]
    
    results = {}
    
    for name, example_func in examples:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running: {name}")
        logger.info(f"{'='*60}")
        
        try:
            result = example_func()
            results[name] = result
            logger.info(f"✅ {name} completed successfully")
        except Exception as e:
            logger.error(f"❌ {name} failed: {e}")
            results[name] = None
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result is not None else "❌ FAILED"
        logger.info(f"{name}: {status}")
    
    logger.info("\n🎉 All examples completed!")


if __name__ == "__main__":
    main()