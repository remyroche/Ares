"""
Example of using the new unified utilities for TAS and NAS architectures.

This example demonstrates how to use the consolidated shared utilities
for configuration management, performance monitoring, meta-learning,
hardware optimization, and evaluation.
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime

# Import the new unified utilities
from ..shared_utils import (
    # Configuration
    create_tas_config, create_nas_config, create_hybrid_config,
    ArchitectureType, SearchStrategy, OptimizationObjective,
    
    # Performance Monitoring
    create_performance_monitor, MonitoringLevel, PerformanceMetric,
    
    # Meta-Learning
    create_meta_learner, MetaLearningMethod, AdaptationType,
    
    # Hardware Management
    create_hardware_manager, OptimizationLevel, WorkloadType,
    
    # Evaluation Framework
    create_evaluation_framework, EvaluationType, EvaluationMetric
)

# Import ML Common integration
from src.utils.ml_common.integration import (
    create_ml_common_integration, create_tas_ml_common_integration,
    create_nas_ml_common_integration, create_hybrid_ml_common_integration,
    MLCommonIntegrationConfig
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_unified_configuration():
    """Example of using unified configuration management."""
    logger.info("🔧 Example: Unified Configuration Management")
    
    # Create TAS configuration
    tas_config = create_tas_config(
        search_strategy=SearchStrategy.BAYESIAN,
        enable_micro_regime_detection=True,
        economic_significance_threshold=0.8
    )
    
    # Create NAS configuration
    nas_config = create_nas_config(
        search_strategy=SearchStrategy.EVOLUTIONARY,
        enable_meta_learning=True,
        meta_learning_rate=1e-3
    )
    
    # Create hybrid configuration
    hybrid_config = create_hybrid_config(
        search_strategy=SearchStrategy.HYBRID,
        tas_weight=0.6,
        nas_weight=0.4,
        adaptive_weighting=True
    )
    
    # Save configurations
    tas_config.save_config("tas_config.json")
    nas_config.save_config("nas_config.json")
    hybrid_config.save_config("hybrid_config.json")
    
    logger.info("✅ Configurations created and saved")
    return tas_config, nas_config, hybrid_config


def example_performance_monitoring():
    """Example of using unified performance monitoring."""
    logger.info("📊 Example: Unified Performance Monitoring")
    
    # Create performance monitors for different architectures
    tas_monitor = create_performance_monitor(
        architecture_type=ArchitectureType.TAS,
        monitoring_level=MonitoringLevel.STANDARD,
        enable_real_time=True
    )
    
    nas_monitor = create_performance_monitor(
        architecture_type=ArchitectureType.NAS,
        monitoring_level=MonitoringLevel.COMPREHENSIVE,
        enable_real_time=True
    )
    
    # Start monitoring
    tas_monitor.start_monitoring()
    nas_monitor.start_monitoring()
    
    # Simulate performance recording
    for i in range(10):
        # Record TAS performance
        tas_metrics = {
            PerformanceMetric.ACCURACY: 0.8 + np.random.normal(0, 0.05),
            PerformanceMetric.SHARPE_RATIO: 1.5 + np.random.normal(0, 0.2),
            PerformanceMetric.ECONOMIC_SIGNIFICANCE: 0.7 + np.random.normal(0, 0.1)
        }
        tas_monitor.record_performance(tas_metrics, iteration=i, regime_id=f"regime_{i%3}")
        
        # Record NAS performance
        nas_metrics = {
            PerformanceMetric.ACCURACY: 0.85 + np.random.normal(0, 0.03),
            PerformanceMetric.SHARPE_RATIO: 1.8 + np.random.normal(0, 0.15),
            PerformanceMetric.ECONOMIC_SIGNIFICANCE: 0.75 + np.random.normal(0, 0.08)
        }
        nas_monitor.record_performance(nas_metrics, iteration=i, regime_id=f"regime_{i%3}")
        
        # Simulate some time passing
        import time
        time.sleep(0.1)
    
    # Get performance summaries
    tas_summary = tas_monitor.get_performance_summary()
    nas_summary = nas_monitor.get_performance_summary()
    
    logger.info(f"TAS Performance Summary: {tas_summary['total_iterations']} iterations")
    logger.info(f"NAS Performance Summary: {nas_summary['total_iterations']} iterations")
    
    # Stop monitoring
    tas_monitor.stop_monitoring()
    nas_monitor.stop_monitoring()
    
    # Export performance data
    tas_monitor.export_performance_data("tas_performance.json")
    nas_monitor.export_performance_data("nas_performance.json")
    
    logger.info("✅ Performance monitoring completed")
    return tas_monitor, nas_monitor


def example_meta_learning():
    """Example of using unified meta-learning."""
    logger.info("🧠 Example: Unified Meta-Learning")
    
    # Create meta-learners for different architectures
    tas_meta_learner = create_meta_learner(
        architecture_type=ArchitectureType.TAS,
        method=MetaLearningMethod.MAML,
        adaptation_type=AdaptationType.REGIME_ADAPTATION
    )
    
    nas_meta_learner = create_meta_learner(
        architecture_type=ArchitectureType.NAS,
        method=MetaLearningMethod.PROTONET,
        adaptation_type=AdaptationType.FEW_SHOT
    )
    
    # Generate sample data for meta-learning
    np.random.seed(42)
    X_data = np.random.randn(1000, 50)
    y_data = np.random.randint(0, 2, 1000)
    
    # Create meta-tasks
    tas_tasks = tas_meta_learner.create_meta_tasks(
        data={'regime_1': (X_data[:500], y_data[:500])},
        regime_labels=np.array(['regime_1'] * 500)
    )
    
    nas_tasks = nas_meta_learner.create_meta_tasks(
        data={'regime_1': (X_data[:500], y_data[:500])}
    )
    
    logger.info(f"Created {len(tas_tasks)} TAS meta-tasks")
    logger.info(f"Created {len(nas_tasks)} NAS meta-tasks")
    
    # Simulate meta-training (simplified)
    logger.info("Performing meta-training...")
    
    # Get adaptation statistics
    tas_stats = tas_meta_learner.get_adaptation_statistics()
    nas_stats = nas_meta_learner.get_adaptation_statistics()
    
    logger.info(f"TAS Meta-learning Stats: {tas_stats}")
    logger.info(f"NAS Meta-learning Stats: {nas_stats}")
    
    logger.info("✅ Meta-learning example completed")
    return tas_meta_learner, nas_meta_learner


def example_hardware_optimization():
    """Example of using unified hardware optimization."""
    logger.info("🖥️ Example: Unified Hardware Optimization")
    
    # Create hardware managers for different architectures
    tas_hardware = create_hardware_manager(
        architecture_type=ArchitectureType.TAS,
        optimization_level=OptimizationLevel.BALANCED,
        enable_gpu_acceleration=False  # Trees don't benefit much from GPU
    )
    
    nas_hardware = create_hardware_manager(
        architecture_type=ArchitectureType.NAS,
        optimization_level=OptimizationLevel.AGGRESSIVE,
        enable_gpu_acceleration=True
    )
    
    # Optimize for specific workloads
    tas_optimization = tas_hardware.optimize_for_workload(
        WorkloadType.TREE_TRAINING,
        parameters={'tree_count': 100, 'max_depth': 10}
    )
    
    nas_optimization = nas_hardware.optimize_for_workload(
        WorkloadType.NEURAL_TRAINING,
        parameters={'batch_size': 64, 'learning_rate': 0.001}
    )
    
    logger.info(f"TAS Hardware Optimization: {tas_optimization.success}")
    logger.info(f"NAS Hardware Optimization: {nas_optimization.success}")
    
    # Start performance monitoring
    tas_hardware.start_performance_monitoring()
    nas_hardware.start_performance_monitoring()
    
    # Simulate some work
    time.sleep(2)
    
    # Get hardware status
    tas_status = tas_hardware.get_hardware_status()
    nas_status = nas_hardware.get_hardware_status()
    
    logger.info(f"TAS Hardware Status: {tas_status['device']}")
    logger.info(f"NAS Hardware Status: {nas_status['device']}")
    
    # Get optimization recommendations
    tas_recommendations = tas_hardware.get_optimization_recommendations()
    nas_recommendations = nas_hardware.get_optimization_recommendations()
    
    logger.info(f"TAS Recommendations: {len(tas_recommendations['general_recommendations'])} items")
    logger.info(f"NAS Recommendations: {len(nas_recommendations['general_recommendations'])} items")
    
    # Stop monitoring
    tas_hardware.stop_performance_monitoring()
    nas_hardware.stop_performance_monitoring()
    
    # Export hardware data
    tas_hardware.export_hardware_data("tas_hardware.json")
    nas_hardware.export_hardware_data("nas_hardware.json")
    
    logger.info("✅ Hardware optimization example completed")
    return tas_hardware, nas_hardware


def example_evaluation_framework():
    """Example of using unified evaluation framework."""
    logger.info("🔬 Example: Unified Evaluation Framework")
    
    # Create evaluation frameworks for different architectures
    tas_evaluator = create_evaluation_framework(
        architecture_type=ArchitectureType.TAS,
        evaluation_type=EvaluationType.COMPREHENSIVE
    )
    
    nas_evaluator = create_evaluation_framework(
        architecture_type=ArchitectureType.NAS,
        evaluation_type=EvaluationType.TRADING
    )
    
    # Generate sample data
    np.random.seed(42)
    X_test = np.random.randn(200, 20)
    y_test = np.random.randint(0, 2, 200)
    
    # Create sample market data
    market_data = pd.DataFrame({
        'open': np.random.randn(200).cumsum() + 100,
        'high': np.random.randn(200).cumsum() + 102,
        'low': np.random.randn(200).cumsum() + 98,
        'close': np.random.randn(200).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 200)
    })
    
    # Create sample regime labels
    regime_labels = np.random.randint(0, 3, 200)
    
    # Create mock models
    class MockTASModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        
        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            return proba / proba.sum(axis=1, keepdims=True)
    
    class MockNASModel:
        def predict(self, X):
            return np.random.randint(0, 2, len(X))
        
        def predict_proba(self, X):
            proba = np.random.rand(len(X), 2)
            return proba / proba.sum(axis=1, keepdims=True)
    
    # Evaluate models
    tas_model = MockTASModel()
    nas_model = MockNASModel()
    
    tas_result = tas_evaluator.evaluate_model(
        model=tas_model,
        X_test=X_test,
        y_test=y_test,
        market_data=market_data,
        regime_labels=regime_labels,
        model_name="Mock_TAS_Model"
    )
    
    nas_result = nas_evaluator.evaluate_model(
        model=nas_model,
        X_test=X_test,
        y_test=y_test,
        market_data=market_data,
        regime_labels=regime_labels,
        model_name="Mock_NAS_Model"
    )
    
    logger.info(f"TAS Evaluation: Accuracy = {tas_result.basic_metrics.get(EvaluationMetric.ACCURACY, 0):.3f}")
    logger.info(f"NAS Evaluation: Accuracy = {nas_result.basic_metrics.get(EvaluationMetric.ACCURACY, 0):.3f}")
    
    # Compare models
    models = [(tas_model, "TAS_Model"), (nas_model, "NAS_Model")]
    comparison_results = tas_evaluator.compare_models(
        models=models,
        X_test=X_test,
        y_test=y_test,
        market_data=market_data,
        regime_labels=regime_labels
    )
    
    logger.info(f"Model comparison completed: {len(comparison_results)} models evaluated")
    
    # Get evaluation summary
    tas_summary = tas_evaluator.get_evaluation_summary()
    nas_summary = nas_evaluator.get_evaluation_summary()
    
    logger.info(f"TAS Evaluations: {tas_summary['total_evaluations']}")
    logger.info(f"NAS Evaluations: {nas_summary['total_evaluations']}")
    
    logger.info("✅ Evaluation framework example completed")
    return tas_evaluator, nas_evaluator


def example_ml_common_integration():
    """Example of using ML Common integration."""
    logger.info("🔗 Example: ML Common Integration")
    
    # Create ML Common integrations
    tas_integration = create_tas_ml_common_integration()
    nas_integration = create_nas_ml_common_integration()
    hybrid_integration = create_hybrid_ml_common_integration()
    
    # Generate sample data with timestamps
    np.random.seed(42)
    X_train = np.random.randn(100, 10)
    y_train = np.random.randint(0, 2, 100)
    X_val = np.random.randn(50, 10)
    y_val = np.random.randint(0, 2, 50)
    
    # Create timestamps
    timestamps_train = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    timestamps_val = pd.date_range(start='2023-01-05', periods=50, freq='1H')
    
    # Mock model for testing
    class MockModel:
        def fit(self, X, y): pass
        def predict(self, X): return np.random.randint(0, 2, len(X))
    
    mock_model = MockModel()
    
    # Comprehensive validation for TAS
    tas_validation = tas_integration.comprehensive_validation(
        model=mock_model,
        X_train=X_train,
        X_test=X_val,
        y_train=y_train,
        y_test=y_val,
        timestamps_train=timestamps_train,
        timestamps_test=timestamps_val
    )
    
    # Comprehensive validation for NAS
    nas_validation = nas_integration.comprehensive_validation(
        model=mock_model,
        X_train=X_train,
        X_test=X_val,
        y_train=y_train,
        y_test=y_val,
        timestamps_train=timestamps_train,
        timestamps_test=timestamps_val
    )
    
    # Get integration status
    tas_status = tas_integration.get_integration_status()
    nas_status = nas_integration.get_integration_status()
    hybrid_status = hybrid_integration.get_integration_status()
    
    logger.info(f"TAS Integration Status: {tas_status['integrations']}")
    logger.info(f"NAS Integration Status: {nas_status['integrations']}")
    logger.info(f"Hybrid Integration Status: {hybrid_status['integrations']}")
    
    # Test lookahead bias prevention
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randint(0, 2, 100),
        'timestamp': timestamps_train[:100]
    })
    
    tas_bias_result = tas_integration.prevent_lookahead_bias(
        data=sample_data,
        timestamp_col='timestamp',
        target_col='target'
    )
    
    logger.info(f"TAS Lookahead Bias Result: {tas_bias_result.get('leakage_detected', False)}")
    
    # Test overfitting detection
    tas_overfitting_result = tas_integration.detect_overfitting(
        model=mock_model,
        X_train=X_train,
        X_val=X_val,
        y_train=y_train,
        y_val=y_val
    )
    
    logger.info(f"TAS Overfitting Detection: {tas_overfitting_result}")
    
    logger.info("✅ ML Common integration example completed")
    return tas_integration, nas_integration, hybrid_integration


def main():
    """Run all examples."""
    logger.info("🚀 Starting Unified Utilities Examples")
    
    try:
        # Run all examples
        example_unified_configuration()
        example_performance_monitoring()
        example_meta_learning()
        example_hardware_optimization()
        example_evaluation_framework()
        example_ml_common_integration()
        
        logger.info("✅ All examples completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    main()