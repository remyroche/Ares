"""
Enhanced Utility Integration Example

This example demonstrates how to use the upgraded hybrid NAS-TAS regime system
with comprehensive utility integrations from src/utils/.
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path
from datetime import datetime, timedelta
import time

# Import enhanced utility integrations
from ..shared_utils.enhanced_utility_integration import (
    EnhancedUtilityIntegration, UtilityIntegrationConfig,
    create_enhanced_utility_integration
)
from ..shared_utils.enhanced_data_integration import (
    EnhancedDataIntegration, DataIntegrationConfig,
    create_enhanced_data_integration
)
from ..shared_utils.enhanced_ml_integration import (
    EnhancedMLIntegration, MLIntegrationConfig,
    create_enhanced_ml_integration
)

# Import the enhanced orchestrator
from ..enhanced_hybrid_orchestrator import EnhancedHybridOrchestrator
from ..config.hybrid_regime_config import HybridRegimeConfig, RegimeCombinationStrategy

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_market_data(n_samples: int = 1000) -> pd.DataFrame:
    """Create sample market data for demonstration."""
    np.random.seed(42)
    
    # Generate realistic market data
    dates = pd.date_range(start='2023-01-01', periods=n_samples, freq='15min')
    
    # Generate price data with trends and volatility
    base_price = 50000
    returns = np.random.normal(0, 0.02, n_samples)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # Create OHLC data
    data = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, n_samples)
    })
    
    # Ensure high >= low
    data['high'] = np.maximum(data['high'], data['low'])
    data['high'] = np.maximum(data['high'], data['open'])
    data['high'] = np.maximum(data['high'], data['close'])
    data['low'] = np.minimum(data['low'], data['open'])
    data['low'] = np.minimum(data['low'], data['close'])
    
    return data


def demonstrate_utility_integration():
    """Demonstrate enhanced utility integration capabilities."""
    logger.info("🚀 Starting Enhanced Utility Integration Demonstration")
    
    # Create sample data
    market_data = create_sample_market_data(1000)
    logger.info(f"📊 Created sample market data: {market_data.shape}")
    
    # Initialize utility integrations
    utility_config = UtilityIntegrationConfig(
        enable_data_validation=True,
        enable_math_validation=True,
        enable_serialization=True,
        enable_m1_optimizations=True,
        enable_ml_common=True,
        enable_matrix_operations=True
    )
    
    data_config = DataIntegrationConfig(
        enable_klines_parquet=True,
        enable_feature_engineering=True,
        enable_returns_engineering=True,
        enable_data_quality=True,
        enable_optimized_storage=True
    )
    
    ml_config = MLIntegrationConfig(
        enable_feature_selection=True,
        enable_cross_validation=True,
        enable_confidence_metrics=True,
        enable_hmm_regime_detection=True,
        enable_lookahead_bias_detection=True,
        enable_overfitting_detection=True,
        enable_data_leakage_detection=True
    )
    
    # Create integrations
    utility_integration = create_enhanced_utility_integration(utility_config)
    data_integration = create_enhanced_data_integration(data_config, utility_integration)
    ml_integration = create_enhanced_ml_integration(ml_config, utility_integration)
    
    logger.info("✅ Utility integrations initialized")
    
    # Demonstrate data processing
    logger.info("🔄 Demonstrating enhanced data processing...")
    
    # Process market data
    processed_data = data_integration.process_market_data(market_data, "BTCUSDT", "15m")
    logger.info(f"✅ Market data processed: {processed_data.shape}")
    
    # Engineer features
    features = data_integration.engineer_features(processed_data, ['momentum', 'volatility', 'volume'])
    logger.info(f"✅ Features engineered: {features.shape}")
    
    # Engineer returns
    returns = data_integration.engineer_returns(processed_data, ['simple', 'log'])
    logger.info(f"✅ Returns engineered: {returns.shape}")
    
    # Calculate data quality metrics
    quality_metrics = data_integration.calculate_data_quality_metrics(processed_data)
    logger.info(f"✅ Data quality score: {quality_metrics.get('quality_score', 0):.3f}")
    
    # Demonstrate ML capabilities
    logger.info("🔄 Demonstrating enhanced ML capabilities...")
    
    # Prepare data for ML
    X = processed_data.select_dtypes(include=[np.number]).values
    y = np.random.randint(0, 3, len(X))  # Mock target for demonstration
    
    # Feature selection
    X_selected, selected_features = ml_integration.select_features(X, y, method="mutual_info", n_features=10)
    logger.info(f"✅ Feature selection completed: {len(selected_features)} features selected")
    
    # Cross-validation
    from sklearn.ensemble import RandomForestClassifier
    estimator = RandomForestClassifier(n_estimators=100, random_state=42)
    cv_results = ml_integration.cross_validate_model(estimator, X_selected, y, cv=5, scoring="accuracy")
    logger.info(f"✅ Cross-validation completed: score={cv_results.get('mean', 0):.3f}")
    
    # Bias detection
    bias_results = ml_integration.detect_lookahead_bias(X_selected, y)
    logger.info(f"✅ Lookahead bias detection: {bias_results.get('bias_detected', False)}")
    
    # Overfitting detection
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    estimator.fit(X_train, y_train)
    overfitting_results = ml_integration.detect_overfitting(estimator, X_train, y_train, X_val, y_val)
    logger.info(f"✅ Overfitting detection: {overfitting_results.get('overfitting_detected', False)}")
    
    # Data leakage detection
    leakage_results = ml_integration.detect_data_leakage(X_selected, y)
    logger.info(f"✅ Data leakage detection: {leakage_results.get('leakage_detected', False)}")
    
    # Regime detection
    regime_results = ml_integration.detect_regimes_hmm(processed_data, n_regimes=3)
    logger.info(f"✅ HMM regime detection: {regime_results.get('n_regimes', 0)} regimes detected")
    
    # Feature importance analysis
    importance_results = ml_integration.analyze_feature_importance(estimator, X_selected, y)
    logger.info(f"✅ Feature importance analysis: {len(importance_results.get('importances', []))} features analyzed")
    
    # Demonstrate mathematical utilities
    logger.info("🔄 Demonstrating mathematical utilities...")
    
    # Safe mathematical operations
    safe_divide_result = utility_integration.safe_divide(10, 2, default=0)
    safe_log_result = utility_integration.safe_log(10, default=0)
    safe_sqrt_result = utility_integration.safe_sqrt(16, default=0)
    
    logger.info(f"✅ Safe math operations: divide={safe_divide_result}, log={safe_log_result:.3f}, sqrt={safe_sqrt_result}")
    
    # Correlation analysis
    x_array = np.random.randn(100)
    y_array = np.random.randn(100)
    correlation = utility_integration.safe_correlation(x_array, y_array)
    logger.info(f"✅ Correlation analysis: {correlation:.3f}")
    
    # Demonstrate serialization
    logger.info("🔄 Demonstrating serialization capabilities...")
    
    # Save data
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    # JSON serialization
    json_success = utility_integration.save_json(quality_metrics, "output/quality_metrics.json")
    logger.info(f"✅ JSON serialization: {json_success}")
    
    # Parquet serialization
    parquet_success = utility_integration.safe_to_parquet(processed_data, "output/processed_data.parquet")
    logger.info(f"✅ Parquet serialization: {parquet_success}")
    
    # Demonstrate M1 optimizations
    logger.info("🔄 Demonstrating M1 optimizations...")
    
    # Memory optimization
    memory_result = utility_integration.optimize_memory()
    logger.info(f"✅ Memory optimization: {memory_result.get('method', 'unknown')}")
    
    # GPU context
    with utility_integration.gpu_context("demo_operation"):
        # Simulate GPU operation
        gpu_result = np.random.rand(1000, 1000)
        logger.info(f"✅ GPU context operation completed: {gpu_result.shape}")
    
    # Memory checkpoint
    with utility_integration.memory_checkpoint("demo_checkpoint"):
        # Simulate memory-intensive operation
        large_array = np.random.rand(10000, 1000)
        logger.info(f"✅ Memory checkpoint operation completed: {large_array.shape}")
    
    # Get system status
    logger.info("🔄 Getting system status...")
    
    utility_status = utility_integration.get_system_status()
    data_status = data_integration.get_system_status()
    ml_status = ml_integration.get_system_status()
    
    logger.info(f"✅ Utility integration status: {len(utility_status.get('available_utilities', []))} utilities available")
    logger.info(f"✅ Data integration status: {len(data_status.get('available_utilities', []))} utilities available")
    logger.info(f"✅ ML integration status: {len(ml_status.get('available_utilities', []))} utilities available")
    
    # Performance metrics
    data_performance = data_integration.get_performance_metrics()
    ml_performance = ml_integration.get_performance_metrics()
    
    logger.info(f"✅ Data processing performance: {data_performance.get('processing_times', {}).get('mean', 0):.3f}s average")
    logger.info(f"✅ ML processing performance: {ml_performance.get('training_times', {}).get('mean', 0):.3f}s average")
    
    logger.info("🎉 Enhanced Utility Integration Demonstration Completed Successfully!")
    
    return {
        'utility_integration': utility_integration,
        'data_integration': data_integration,
        'ml_integration': ml_integration,
        'processed_data': processed_data,
        'quality_metrics': quality_metrics,
        'performance_metrics': {
            'data': data_performance,
            'ml': ml_performance
        }
    }


def demonstrate_enhanced_orchestrator():
    """Demonstrate the enhanced hybrid orchestrator with utility integrations."""
    logger.info("🚀 Starting Enhanced Hybrid Orchestrator Demonstration")
    
    # Create sample data
    market_data = create_sample_market_data(1000)
    logger.info(f"📊 Created sample market data: {market_data.shape}")
    
    # Create hybrid regime config
    config = HybridRegimeConfig(
        symbol="BTCUSDT",
        timeframe="15m",
        n_regimes=3,
        combination_strategy=RegimeCombinationStrategy.WEIGHTED_AVERAGE,
        enable_multi_timeframe=True,
        use_unified_search=True,
        use_signal_generation=True
    )
    
    # Create enhanced orchestrator
    orchestrator = EnhancedHybridOrchestrator(config)
    logger.info("✅ Enhanced hybrid orchestrator initialized")
    
    # Get system status
    status = orchestrator.get_system_status()
    logger.info(f"✅ System status: {status.get('orchestrator_version', 'unknown')}")
    
    # Analyze market regimes
    logger.info("🔄 Analyzing market regimes with enhanced utilities...")
    
    start_time = time.time()
    regime_result = orchestrator.analyze_market_regimes(market_data, enable_multi_timeframe=True)
    analysis_time = time.time() - start_time
    
    logger.info(f"✅ Market regime analysis completed in {analysis_time:.2f}s")
    
    # Display results
    if hasattr(regime_result, 'regime_predictions'):
        logger.info(f"✅ Regime predictions: {len(regime_result.regime_predictions)} samples")
        logger.info(f"✅ Regime probabilities: {regime_result.regime_probabilities.shape}")
        logger.info(f"✅ Economic significance scores: {len(regime_result.economic_significance_scores)}")
        logger.info(f"✅ Trading viability scores: {len(regime_result.trading_viability_scores)}")
        logger.info(f"✅ Regime stability scores: {len(regime_result.regime_stability_scores)}")
        logger.info(f"✅ Execution time: {regime_result.execution_time:.2f}s")
        
        # Display metadata
        metadata = regime_result.metadata
        logger.info(f"✅ Enhanced utilities used: {metadata.get('enhanced_utilities_used', False)}")
        logger.info(f"✅ Utility integration used: {metadata.get('utility_integration_used', False)}")
        logger.info(f"✅ Data integration used: {metadata.get('data_integration_used', False)}")
        logger.info(f"✅ ML integration used: {metadata.get('ml_integration_used', False)}")
        logger.info(f"✅ Cross-validation performed: {metadata.get('cross_validation_performed', False)}")
        logger.info(f"✅ Bias detection performed: {metadata.get('bias_detection_performed', False)}")
        logger.info(f"✅ Overfitting detection performed: {metadata.get('overfitting_detection_performed', False)}")
        logger.info(f"✅ Data leakage detection performed: {metadata.get('data_leakage_detection_performed', False)}")
        logger.info(f"✅ Feature selection performed: {metadata.get('feature_selection_performed', False)}")
        logger.info(f"✅ Hyperparameter optimization performed: {metadata.get('hyperparameter_optimization_performed', False)}")
        logger.info(f"✅ M1 optimizations enabled: {metadata.get('m1_optimizations_enabled', False)}")
        logger.info(f"✅ Memory optimization enabled: {metadata.get('memory_optimization_enabled', False)}")
        logger.info(f"✅ GPU acceleration enabled: {metadata.get('gpu_acceleration_enabled', False)}")
    
    # Generate architecture signals if available
    if hasattr(orchestrator, 'generate_architecture_signals'):
        logger.info("🔄 Generating architecture signals...")
        signals = orchestrator.generate_architecture_signals(market_data, regime_result.__dict__)
        logger.info(f"✅ Generated {len(signals)} architecture signals")
    
    # Get signal quality metrics if available
    if hasattr(orchestrator, 'get_signal_quality_metrics'):
        logger.info("🔄 Getting signal quality metrics...")
        quality_metrics = orchestrator.get_signal_quality_metrics(market_data)
        logger.info(f"✅ Signal quality metrics: {quality_metrics}")
    
    logger.info("🎉 Enhanced Hybrid Orchestrator Demonstration Completed Successfully!")
    
    return {
        'orchestrator': orchestrator,
        'regime_result': regime_result,
        'analysis_time': analysis_time,
        'system_status': status
    }


def main():
    """Main demonstration function."""
    logger.info("🚀 Starting Comprehensive Enhanced Utility Integration Demonstration")
    
    try:
        # Demonstrate utility integration
        utility_results = demonstrate_utility_integration()
        
        # Demonstrate enhanced orchestrator
        orchestrator_results = demonstrate_enhanced_orchestrator()
        
        # Summary
        logger.info("📊 Demonstration Summary:")
        logger.info(f"   - Utility integration: ✅ {len(utility_results['utility_integration'].get_available_utilities())} utilities")
        logger.info(f"   - Data integration: ✅ {len(utility_results['data_integration'].get_available_data_utilities())} utilities")
        logger.info(f"   - ML integration: ✅ {len(utility_results['ml_integration'].get_available_ml_utilities())} utilities")
        logger.info(f"   - Regime analysis: ✅ {orchestrator_results['analysis_time']:.2f}s")
        logger.info(f"   - Enhanced utilities: ✅ All integrations working")
        
        logger.info("🎉 All demonstrations completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()