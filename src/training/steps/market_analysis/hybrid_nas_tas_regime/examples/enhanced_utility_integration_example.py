"""
Enhanced Utility Integration Example for Hybrid NAS-TAS Regime System

This example demonstrates how to use the enhanced utility integrations
for comprehensive data processing, ML operations, and regime detection.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedUtilityIntegrationExample:
    """
    Example class demonstrating enhanced utility integration usage.
    """
    
    def __init__(self):
        """Initialize the example with enhanced utility integrations."""
        self.logger = logger.getChild('EnhancedUtilityIntegrationExample')
        
        # Initialize utility integrations
        self._initialize_integrations()
        
        self.logger.info("🚀 Enhanced Utility Integration Example initialized")
    
    def _initialize_integrations(self):
        """Initialize all enhanced utility integrations."""
        # Utility integration configuration
        utility_config = UtilityIntegrationConfig(
            enable_data_validation=True,
            enable_data_quality_checks=True,
            enable_safe_operations=True,
            enable_math_validation=True,
            enable_safe_math=True,
            enable_serialization=True,
            enable_m1_optimizations=True,
            enable_gpu_acceleration=True,
            enable_memory_optimization=True,
            enable_cpu_optimization=True,
            enable_ml_common=True,
            enable_feature_selection=True,
            enable_cross_validation=True,
            enable_confidence_metrics=True,
            enable_matrix_operations=True,
            enable_vectorized_operations=True,
            enable_performance_monitoring=True,
            enable_memory_monitoring=True
        )
        self.utility_integration = create_enhanced_utility_integration(utility_config)
        
        # Data integration configuration
        data_config = DataIntegrationConfig(
            enable_klines_parquet=True,
            enable_unified_data_utils=True,
            enable_historical_downloader=True,
            enable_feature_engineering=True,
            enable_returns_engineering=True,
            enable_gap_detection=True,
            enable_data_quality=True,
            enable_advanced_quality_metrics=True,
            enable_comprehensive_quality_scoring=True,
            enable_optimized_storage=True,
            enable_parquet_optimization=True,
            enable_parallel_processing=True,
            enable_memory_optimization=True,
            enable_schema_validation=True,
            enable_data_consistency_checks=True
        )
        self.data_integration = create_enhanced_data_integration(data_config, utility_config)
        
        # ML integration configuration
        ml_config = MLIntegrationConfig(
            enable_ml_common=True,
            enable_feature_selection=True,
            enable_cross_validation=True,
            enable_confidence_metrics=True,
            enable_hmm_regime_detection=True,
            enable_regime_analysis=True,
            enable_grid_search=True,
            enable_bayesian_optimization=True,
            enable_tpe_optimization=True,
            enable_ensemble_management=True,
            enable_model_ensembles=True,
            enable_model_evaluation=True,
            enable_performance_metrics=True,
            enable_parallel_processing=True,
            enable_vectorization=True,
            enable_lookahead_bias_detection=True,
            enable_overfitting_detection=True,
            enable_data_leakage_detection=True
        )
        self.ml_integration = create_enhanced_ml_integration(ml_config, utility_config)
        
        self.logger.info("✅ All enhanced utility integrations initialized")
    
    def demonstrate_data_operations(self):
        """Demonstrate enhanced data operations."""
        self.logger.info("📊 Demonstrating enhanced data operations...")
        
        # Create sample market data
        dates = pd.date_range('2023-01-01', periods=1000, freq='15T')
        np.random.seed(42)
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'high': 100 + np.cumsum(np.random.randn(1000) * 0.01) + np.random.rand(1000) * 2,
            'low': 100 + np.cumsum(np.random.randn(1000) * 0.01) - np.random.rand(1000) * 2,
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.01),
            'volume': np.random.randint(1000, 10000, 1000)
        })
        
        # Ensure high >= low
        data['high'] = np.maximum(data['high'], data['low'])
        data['low'] = np.minimum(data['high'], data['low'])
        
        self.logger.info(f"📈 Created sample data with {len(data)} rows")
        
        # Demonstrate data validation
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        is_valid = self.utility_integration.validate_dataframe_columns(data, required_columns)
        self.logger.info(f"✅ Data validation: {'PASSED' if is_valid else 'FAILED'}")
        
        # Demonstrate data quality metrics
        quality_metrics = self.utility_integration.calculate_data_quality_metrics(data)
        self.logger.info(f"📊 Data quality metrics: {quality_metrics}")
        
        # Demonstrate data optimization
        optimized_data = self.utility_integration.optimize_dataframe_dtypes(data)
        self.logger.info(f"🔧 Data optimization completed")
        
        # Demonstrate data processing
        processed_data = self.data_integration.process_market_data(optimized_data, "BTCUSDT", "15m")
        self.logger.info(f"⚙️ Data processing completed")
        
        return processed_data
    
    def demonstrate_feature_engineering(self, data: pd.DataFrame):
        """Demonstrate enhanced feature engineering."""
        self.logger.info("🔧 Demonstrating enhanced feature engineering...")
        
        # Engineer features
        features = self.data_integration.engineer_features(data, ['momentum', 'volatility', 'volume'])
        self.logger.info(f"✅ Engineered {len(features.columns)} features")
        
        # Engineer returns
        returns = self.data_integration.engineer_returns(data, ['simple', 'log'])
        self.logger.info(f"📈 Engineered {len(returns.columns)} return features")
        
        # Detect gaps
        gaps = self.data_integration.detect_gaps(data)
        self.logger.info(f"🔍 Detected {len(gaps)} gaps in data")
        
        return features, returns
    
    def demonstrate_ml_operations(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate enhanced ML operations."""
        self.logger.info("🤖 Demonstrating enhanced ML operations...")
        
        # Feature selection
        X_selected, selected_features = self.ml_integration.select_features(
            X, y, method="mutual_info", n_features=10
        )
        self.logger.info(f"🎯 Selected {len(selected_features)} features")
        
        # Cross-validation
        from sklearn.ensemble import RandomForestClassifier
        estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        
        cv_results = self.ml_integration.cross_validate_model(
            estimator, X_selected, y, cv=5, scoring="accuracy"
        )
        self.logger.info(f"📊 Cross-validation results: {cv_results}")
        
        # Hyperparameter optimization
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        optimization_results = self.ml_integration.optimize_hyperparameters(
            estimator, X_selected, y, param_grid, method="grid_search"
        )
        self.logger.info(f"🔍 Hyperparameter optimization completed")
        
        return X_selected, cv_results, optimization_results
    
    def demonstrate_regime_detection(self, data: pd.DataFrame):
        """Demonstrate enhanced regime detection."""
        self.logger.info("🎯 Demonstrating enhanced regime detection...")
        
        # HMM regime detection
        regime_results = self.ml_integration.detect_regimes_hmm(
            data, n_regimes=3, features=['open', 'high', 'low', 'close', 'volume']
        )
        self.logger.info(f"🔍 HMM regime detection completed")
        
        # Regime transition analysis
        if 'regime_sequence' in regime_results:
            transition_analysis = self.ml_integration.analyze_regime_transitions(
                regime_results['regime_sequence']
            )
            self.logger.info(f"📊 Regime transition analysis completed")
        
        return regime_results
    
    def demonstrate_ensemble_methods(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate enhanced ensemble methods."""
        self.logger.info("🎭 Demonstrating enhanced ensemble methods...")
        
        # Create multiple models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.linear_model import LogisticRegression
        
        models = [
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42)),
            ('lr', LogisticRegression(random_state=42))
        ]
        
        # Create ensemble
        ensemble = self.ml_integration.create_ensemble(models, method="voting")
        self.logger.info(f"🎭 Created ensemble with {len(models)} models")
        
        # Train ensemble
        ensemble.fit(X, y)
        
        # Evaluate ensemble
        y_pred = ensemble.predict(X)
        y_proba = ensemble.predict_proba(X)
        
        evaluation_results = self.ml_integration.evaluate_model(
            ensemble, X, y, y_pred, y_proba
        )
        self.logger.info(f"📊 Ensemble evaluation completed")
        
        return ensemble, evaluation_results
    
    def demonstrate_bias_detection(self, X: np.ndarray, y: np.ndarray):
        """Demonstrate enhanced bias detection."""
        self.logger.info("🔍 Demonstrating enhanced bias detection...")
        
        # Lookahead bias detection
        lookahead_results = self.ml_integration.detect_lookahead_bias(X, y)
        self.logger.info(f"👁️ Lookahead bias detection completed")
        
        # Overfitting detection
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        overfitting_results = self.ml_integration.detect_overfitting(
            model, X_train, y_train, X_val, y_val
        )
        self.logger.info(f"📈 Overfitting detection completed")
        
        # Data leakage detection
        leakage_results = self.ml_integration.detect_data_leakage(X, y)
        self.logger.info(f"🔒 Data leakage detection completed")
        
        return lookahead_results, overfitting_results, leakage_results
    
    def demonstrate_hardware_optimization(self):
        """Demonstrate enhanced hardware optimization."""
        self.logger.info("⚡ Demonstrating enhanced hardware optimization...")
        
        # M1 optimization
        m1_results = self.utility_integration.optimize_for_m1()
        self.logger.info(f"🍎 M1 optimization: {m1_results}")
        
        # Memory optimization
        memory_results = self.utility_integration.optimize_memory()
        self.logger.info(f"🧠 Memory optimization: {memory_results}")
        
        # Memory usage
        memory_usage = self.utility_integration.get_memory_usage()
        self.logger.info(f"📊 Current memory usage: {memory_usage / (1024**3):.2f} GB")
        
        return m1_results, memory_results
    
    def demonstrate_serialization(self, data: pd.DataFrame):
        """Demonstrate enhanced serialization."""
        self.logger.info("💾 Demonstrating enhanced serialization...")
        
        # Save data
        filepath = "sample_data.parquet"
        success = self.utility_integration.save_parquet(data, filepath)
        self.logger.info(f"💾 Data saved: {'SUCCESS' if success else 'FAILED'}")
        
        # Load data
        loaded_data = self.utility_integration.load_parquet(filepath)
        self.logger.info(f"📂 Data loaded: {'SUCCESS' if loaded_data is not None else 'FAILED'}")
        
        # Clean up
        if Path(filepath).exists():
            Path(filepath).unlink()
        
        return success, loaded_data
    
    def demonstrate_performance_monitoring(self):
        """Demonstrate enhanced performance monitoring."""
        self.logger.info("⏱️ Demonstrating enhanced performance monitoring...")
        
        # Timed operation
        @self.utility_integration.timed_operation
        def sample_operation():
            time.sleep(0.1)
            return "Operation completed"
        
        result = sample_operation()
        self.logger.info(f"⏱️ Timed operation: {result}")
        
        # Memory checkpoint
        with self.utility_integration.memory_checkpoint("sample_checkpoint"):
            # Simulate some work
            data = np.random.randn(1000, 100)
            result = np.sum(data)
        
        self.logger.info(f"🧠 Memory checkpoint completed")
        
        return result
    
    def run_comprehensive_example(self):
        """Run comprehensive example demonstrating all enhanced utilities."""
        self.logger.info("🚀 Running comprehensive enhanced utility integration example...")
        
        try:
            # 1. Data operations
            data = self.demonstrate_data_operations()
            
            # 2. Feature engineering
            features, returns = self.demonstrate_feature_engineering(data)
            
            # 3. Prepare ML data
            X = features.select_dtypes(include=[np.number]).values
            y = np.random.randint(0, 3, len(X))  # Random labels for demonstration
            
            # 4. ML operations
            X_selected, cv_results, optimization_results = self.demonstrate_ml_operations(X, y)
            
            # 5. Regime detection
            regime_results = self.demonstrate_regime_detection(data)
            
            # 6. Ensemble methods
            ensemble, evaluation_results = self.demonstrate_ensemble_methods(X_selected, y)
            
            # 7. Bias detection
            bias_results = self.demonstrate_bias_detection(X_selected, y)
            
            # 8. Hardware optimization
            hardware_results = self.demonstrate_hardware_optimization()
            
            # 9. Serialization
            serialization_results = self.demonstrate_serialization(data)
            
            # 10. Performance monitoring
            performance_results = self.demonstrate_performance_monitoring()
            
            self.logger.info("✅ Comprehensive example completed successfully!")
            
            # Summary
            summary = {
                'data_shape': data.shape,
                'features_shape': features.shape,
                'returns_shape': returns.shape,
                'selected_features': len(X_selected[0]) if len(X_selected) > 0 else 0,
                'cv_mean_score': cv_results.get('mean', 0),
                'regime_detection': 'completed' if regime_results else 'failed',
                'ensemble_accuracy': evaluation_results.get('accuracy', 0),
                'hardware_optimization': hardware_results[0].get('integration_status', 'unknown'),
                'serialization': serialization_results[0],
                'performance_monitoring': 'completed'
            }
            
            self.logger.info(f"📊 Example Summary: {summary}")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"❌ Error in comprehensive example: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.utility_integration.cleanup_resources()
            self.data_integration.cleanup_data_resources()
            self.ml_integration.cleanup_ml_resources()
            self.logger.info("🧹 Resources cleaned up successfully")
        except Exception as e:
            self.logger.error(f"❌ Error during cleanup: {e}")


def main():
    """Main function to run the enhanced utility integration example."""
    example = EnhancedUtilityIntegrationExample()
    
    try:
        # Run comprehensive example
        results = example.run_comprehensive_example()
        
        print("\n" + "="*80)
        print("🎉 ENHANCED UTILITY INTEGRATION EXAMPLE COMPLETED SUCCESSFULLY!")
        print("="*80)
        print(f"📊 Results Summary:")
        for key, value in results.items():
            print(f"   {key}: {value}")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error running example: {e}")
        raise
    finally:
        # Cleanup
        example.cleanup()


if __name__ == "__main__":
    main()