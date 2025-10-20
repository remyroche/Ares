"""
Enhanced Vectorized Validation Usage Example

This example demonstrates how to use the comprehensive validation framework
with all integrated tools: VectorBTRollingOptimizer, UnifiedVectorizationManager,
ML commons utilities, and features_commons.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any

# Import the enhanced validation framework
from .comprehensive_validator import ComprehensiveValidator, ComprehensiveValidationConfig
from .enhanced_vectorized_validator import EnhancedVectorizedValidator, EnhancedVectorizedConfig

def create_sample_data(n_samples: int = 10000, n_features: int = 50) -> tuple:
    """Create sample data for validation testing."""
    
    # Generate time series data
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='1H')
    
    # Generate features with different patterns
    np.random.seed(42)
    features = {}
    
    # Price-based features
    price = 100 + np.cumsum(np.random.randn(n_samples) * 0.01)
    features['close'] = price
    features['high'] = price * (1 + np.abs(np.random.randn(n_samples)) * 0.02)
    features['low'] = price * (1 - np.abs(np.random.randn(n_samples)) * 0.02)
    features['volume'] = np.random.lognormal(10, 1, n_samples)
    
    # Technical indicators
    for i in range(5, 21, 5):
        features[f'sma_{i}'] = pd.Series(price).rolling(i).mean()
        features[f'std_{i}'] = pd.Series(price).rolling(i).std()
        features[f'rsi_{i}'] = np.random.uniform(0, 100, n_samples)
    
    # Momentum features
    for i in range(5, 21, 5):
        features[f'momentum_{i}'] = price / pd.Series(price).shift(i) - 1
        features[f'roc_{i}'] = (price - pd.Series(price).shift(i)) / pd.Series(price).shift(i)
    
    # Volatility features
    for i in range(5, 21, 5):
        features[f'volatility_{i}'] = pd.Series(price).rolling(i).std()
        features[f'atr_{i}'] = np.random.exponential(0.5, n_samples)
    
    # Create DataFrame
    data = pd.DataFrame(features, index=dates)
    data = data.fillna(method='bfill')
    
    # Generate targets (returns)
    targets = data['close'].pct_change().shift(-1)  # Next period returns
    targets = targets.fillna(0)
    
    # Add some signal
    targets = targets + np.random.normal(0, 0.001, len(targets))
    
    return data, targets

def create_sample_pipeline():
    """Create a sample pipeline for testing."""
    
    class SamplePipeline:
        def __init__(self):
            self.is_fitted = False
        
        def fit(self, X, y):
            """Fit the pipeline."""
            self.is_fitted = True
            return self
        
        def predict(self, X):
            """Make predictions."""
            if not self.is_fitted:
                raise ValueError("Pipeline not fitted")
            
            # Simple prediction based on rolling mean
            predictions = X['close'].rolling(20).mean()
            return predictions.fillna(method='bfill')
        
        def score(self, X, y):
            """Calculate score."""
            predictions = self.predict(X)
            return predictions.corr(y)
    
    return SamplePipeline()

def demonstrate_basic_validation():
    """Demonstrate basic comprehensive validation."""
    
    print("🔧 Basic Comprehensive Validation Demo")
    print("=" * 50)
    
    # Create sample data
    data, targets = create_sample_data(n_samples=5000, n_features=30)
    pipeline = create_sample_pipeline()
    
    # Configure validation
    config = ComprehensiveValidationConfig(
        enable_enhanced_vectorized=True,
        enable_label_leakage_prevention=True,
        enable_hierarchical_validation=True,
        enable_anchored_optimization=True,
        enable_interpretability_feedback=True,
        enable_vector_integrity=True,
        enable_forward_validation=True,
        verbose=True
    )
    
    # Initialize validator
    validator = ComprehensiveValidator(config)
    
    # Perform validation
    print("🚀 Starting comprehensive validation...")
    result = validator.validate_pipeline(
        data=data,
        targets=targets,
        pipeline=pipeline,
        metadata={
            'asset_ids': ['AAPL'] * len(data),
            'timestamps': data.index.tolist()
        }
    )
    
    # Display results
    print(f"\n📊 Validation Results:")
    print(f"✅ Passed: {result.passed_validation}")
    print(f"📈 Overall Score: {result.overall_score:.4f}")
    print(f"⚡ Performance Improvement: {result.performance_improvement:.2f}x")
    print(f"💾 Memory Efficiency: {result.memory_efficiency:.2f}")
    print(f"⏱️ Time Efficiency: {result.time_efficiency:.2f}")
    
    if result.critical_issues:
        print(f"\n❌ Critical Issues: {len(result.critical_issues)}")
        for issue in result.critical_issues[:3]:  # Show first 3
            print(f"  - {issue}")
    
    if result.recommendations:
        print(f"\n💡 Recommendations: {len(result.recommendations)}")
        for rec in result.recommendations[:3]:  # Show first 3
            print(f"  - {rec}")
    
    return result

def demonstrate_enhanced_vectorized_validation():
    """Demonstrate enhanced vectorized validation with all tools."""
    
    print("\n🚀 Enhanced Vectorized Validation Demo")
    print("=" * 50)
    
    # Create larger dataset for performance testing
    data, targets = create_sample_data(n_samples=20000, n_features=100)
    pipeline = create_sample_pipeline()
    
    # Configure enhanced validation
    config = EnhancedVectorizedConfig(
        # VectorBT optimization
        enable_vectorbt_optimization=True,
        vectorbt_gpu_acceleration=False,  # Set to True if GPU available
        vectorbt_parallel_processing=True,
        vectorbt_memory_optimization=True,
        
        # Unified vectorization
        enable_unified_vectorization=True,
        auto_strategy_selection=True,
        memory_budget_mb=4096.0,
        time_budget_seconds=1200.0,
        
        # ML Commons integration
        enable_ml_commons=True,
        enable_data_leakage_detection=True,
        enable_lookahead_detection=True,
        enable_hpo_optimization=True,
        enable_shap_lime=True,
        
        # Features commons integration
        enable_features_commons=True,
        enable_vectorbt_scaling=True,
        enable_advanced_cv=True,
        
        # Performance optimization
        chunk_size=5000,
        max_workers=8,
        enable_caching=True,
        cache_size_mb=1024,
        
        verbose=True
    )
    
    # Initialize enhanced validator
    validator = EnhancedVectorizedValidator(config)
    
    # Perform enhanced validation
    print("🚀 Starting enhanced vectorized validation...")
    result = validator.validate_pipeline_enhanced(
        data=data,
        targets=targets,
        pipeline=pipeline,
        metadata={
            'asset_ids': ['AAPL'] * len(data),
            'timestamps': data.index.tolist()
        }
    )
    
    # Display enhanced results
    print(f"\n📊 Enhanced Validation Results:")
    print(f"✅ Passed: {result.passed_validation}")
    print(f"📈 Overall Score: {result.overall_score:.4f}")
    print(f"⚡ Performance Improvement: {result.performance_improvement:.2f}x")
    print(f"💾 Memory Efficiency: {result.memory_efficiency:.2f}")
    print(f"⏱️ Time Efficiency: {result.time_efficiency:.2f}")
    
    # VectorBT optimization results
    if result.vectorbt_optimization_stats:
        print(f"\n🔧 VectorBT Optimization:")
        for stat, value in result.vectorbt_optimization_stats.items():
            if isinstance(value, (int, float)):
                print(f"  - {stat}: {value:.4f}")
    
    # Vectorization strategy
    if result.vectorization_strategy:
        print(f"\n🎯 Vectorization Strategy: {result.vectorization_strategy}")
    
    # ML Commons insights
    if result.data_leakage_detected:
        print(f"\n⚠️ Data Leakage Detected: {result.data_leakage_detected}")
    
    if result.lookahead_detected:
        print(f"\n⚠️ Lookahead Detected: {result.lookahead_detected}")
    
    if result.shap_lime_insights:
        print(f"\n🔍 SHAP/LIME Insights Available: {len(result.shap_lime_insights)} metrics")
    
    # Performance metrics
    if result.vectorbt_performance_gains:
        print(f"\n⚡ VectorBT Performance Gains:")
        for metric, gain in result.vectorbt_performance_gains.items():
            print(f"  - {metric}: {gain:.2f}x")
    
    return result

def demonstrate_performance_comparison():
    """Demonstrate performance comparison between basic and enhanced validation."""
    
    print("\n📊 Performance Comparison Demo")
    print("=" * 50)
    
    # Create test data
    data, targets = create_sample_data(n_samples=10000, n_features=50)
    pipeline = create_sample_pipeline()
    
    # Test basic validation
    print("🔧 Testing basic validation...")
    basic_config = ComprehensiveValidationConfig(
        enable_enhanced_vectorized=False,
        verbose=False
    )
    basic_validator = ComprehensiveValidator(basic_config)
    
    import time
    start_time = time.time()
    basic_result = basic_validator.validate_pipeline(data, targets, pipeline)
    basic_time = time.time() - start_time
    
    # Test enhanced validation
    print("🚀 Testing enhanced validation...")
    enhanced_config = ComprehensiveValidationConfig(
        enable_enhanced_vectorized=True,
        verbose=False
    )
    enhanced_validator = ComprehensiveValidator(enhanced_config)
    
    start_time = time.time()
    enhanced_result = enhanced_validator.validate_pipeline(data, targets, pipeline)
    enhanced_time = time.time() - start_time
    
    # Compare results
    print(f"\n📈 Performance Comparison:")
    print(f"Basic Validation Time: {basic_time:.2f}s")
    print(f"Enhanced Validation Time: {enhanced_time:.2f}s")
    print(f"Speed Improvement: {basic_time / enhanced_time:.2f}x")
    
    print(f"\nBasic Score: {basic_result.overall_score:.4f}")
    print(f"Enhanced Score: {enhanced_result.overall_score:.4f}")
    print(f"Score Improvement: {enhanced_result.overall_score / basic_result.overall_score:.2f}x")
    
    return basic_result, enhanced_result

def main():
    """Main demonstration function."""
    
    print("🎯 Enhanced Vectorized Validation Framework Demo")
    print("=" * 60)
    print("This demo showcases the integration of:")
    print("- VectorBTRollingOptimizer for efficient computations")
    print("- UnifiedVectorizationManager for optimal strategy selection")
    print("- ML commons utilities (CV, OOF, data leakage, lookahead, HPO, SHAP/LIME)")
    print("- features_commons (scalers, transforms, CV)")
    print("=" * 60)
    
    try:
        # Demonstrate basic validation
        basic_result = demonstrate_basic_validation()
        
        # Demonstrate enhanced validation
        enhanced_result = demonstrate_enhanced_vectorized_validation()
        
        # Demonstrate performance comparison
        basic_result, enhanced_result = demonstrate_performance_comparison()
        
        print("\n✅ Demo completed successfully!")
        print("The enhanced validation framework provides:")
        print("- 🚀 Significant performance improvements through VectorBT optimization")
        print("- 🎯 Optimal vectorization strategy selection")
        print("- 🔍 Advanced ML analysis with data leakage and lookahead detection")
        print("- 📊 Comprehensive validation with interpretability insights")
        print("- ⚡ Efficient memory and time usage")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        print("This might be due to missing dependencies or configuration issues.")
        print("Please ensure all required packages are installed.")

if __name__ == "__main__":
    main()
