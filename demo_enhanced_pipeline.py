"""
Demo script for Enhanced Unified Data-Driven Pipeline

This script demonstrates the enhanced pipeline functionality without requiring
external dependencies, showing the comprehensive logging and validation features.
"""

import sys
import os
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Mock data structures for demonstration
class MockDataFrame:
    def __init__(self, data: Dict[str, list]):
        self.data = data
        self.columns = list(data.keys())
        self.shape = (len(list(data.values())[0]), len(data))
    
    def __len__(self):
        return self.shape[0]
    
    def isna(self):
        return MockDataFrame({col: [False] * len(self.data[col]) for col in self.columns})
    
    def sum(self):
        if hasattr(self, '_sum_result'):
            return self._sum_result
        return 0

# Mock tprint functions for demonstration
def tprint(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}]", *args)

def tprint_info(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] INFO:", *args)

def tprint_success(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] SUCCESS:", *args)

def tprint_warning(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] WARNING:", *args)

def tprint_error(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] ERROR:", *args)

def tprint_debug(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] DEBUG:", *args)

def tprint_performance(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] PERFORMANCE:", *args)

def tprint_exception(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] EXCEPTION:", *args)

def tprint_structured(*args, **kwargs):
    print(f"[{datetime.now().strftime('%H:%M:%S.%f')[:-3]}] STRUCTURED:", *args)

class LogLevel:
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    SUCCESS = "SUCCESS"
    PERFORMANCE = "PERFORMANCE"

# Mock configuration
class EnhancedPipelineConfig:
    def __init__(self):
        self.enable_vectorbt_optimization = True
        self.enable_unified_vectorization = True
        self.enable_comprehensive_validation = True
        self.enable_performance_monitoring = True
        self.enable_caching = True
        self.enable_data_quality_checks = True
        self.fail_fast = True
        self.strict_validation = True
        self.validate_inputs = True
        self.validate_outputs = True
        self.validate_intermediates = True
        self.memory_limit_mb = 4096.0
        self.max_workers = None
        self.log_level = LogLevel.INFO
        self.min_data_quality_score = 0.8
        self.max_missing_ratio = 0.1
        self.max_outlier_ratio = 0.05

# Mock validation class
class FastFailingValidation:
    def __init__(self, config):
        self.config = config
        self.validation_results = []
        self.error_count = 0
        self.warning_count = 0
        tprint_info("🔍 Initializing FastFailingValidation")
    
    def validate_input(self, data, name, expected_type=None, expected_shape=None, allow_nan=False):
        tprint_debug(f"🔍 Validating input: {name}")
        
        try:
            # Type validation
            if expected_type and not isinstance(data, expected_type):
                error_msg = f"Input '{name}' has wrong type. Expected {expected_type}, got {type(data)}"
                tprint_error(error_msg)
                if self.config.fail_fast:
                    raise TypeError(error_msg)
                self.error_count += 1
                return False
            
            # Shape validation
            if hasattr(data, 'shape') and expected_shape:
                if data.shape != expected_shape:
                    error_msg = f"Input '{name}' has wrong shape. Expected {expected_shape}, got {data.shape}"
                    tprint_error(error_msg)
                    if self.config.fail_fast:
                        raise ValueError(error_msg)
                    self.error_count += 1
                    return False
            
            # Empty data validation
            if hasattr(data, '__len__') and len(data) == 0:
                error_msg = f"Input '{name}' is empty"
                tprint_error(error_msg)
                if self.config.fail_fast:
                    raise ValueError(error_msg)
                self.error_count += 1
                return False
            
            tprint_success(f"✅ Input '{name}' validation passed")
            return True
            
        except Exception as e:
            error_msg = f"Validation failed for input '{name}': {str(e)}"
            tprint_exception(e, error_msg)
            if self.config.fail_fast:
                raise
            self.error_count += 1
            return False
    
    def validate_output(self, data, name, expected_type=None):
        tprint_debug(f"🔍 Validating output: {name}")
        
        try:
            if expected_type and not isinstance(data, expected_type):
                error_msg = f"Output '{name}' has wrong type. Expected {expected_type}, got {type(data)}"
                tprint_error(error_msg)
                if self.config.fail_fast:
                    raise TypeError(error_msg)
                self.error_count += 1
                return False
            
            tprint_success(f"✅ Output '{name}' validation passed")
            return True
            
        except Exception as e:
            error_msg = f"Output validation failed for '{name}': {str(e)}"
            tprint_exception(e, error_msg)
            if self.config.fail_fast:
                raise
            self.error_count += 1
            return False
    
    def get_validation_summary(self):
        return {
            'total_validations': len(self.validation_results),
            'error_count': self.error_count,
            'warning_count': self.warning_count,
            'success_rate': (len(self.validation_results) - self.error_count) / max(len(self.validation_results), 1)
        }

# Mock pipeline class
class EnhancedUnifiedDataDrivenPipeline:
    def __init__(self, config=None):
        tprint_info("🚀 Initializing Enhanced Unified Data-Driven Pipeline")
        
        self.config = config or EnhancedPipelineConfig()
        self.validation = FastFailingValidation(self.config)
        
        # Mock component availability
        self.vectorbt_optimizer = True
        self.vectorization_manager = True
        self.validation_system = True
        self.error_handler = True
        self.performance_monitor = True
        self.cache = True
        self.data_quality_validator = True
        
        # Performance tracking
        self.performance_metrics = {}
        self.operation_times = {}
        self.error_log = []
        
        tprint_success("✅ Enhanced pipeline initialized successfully")
    
    def process_data(self, data, operation_type="feature_engineering", **kwargs):
        tprint_info(f"🔄 Starting data processing: {operation_type}")
        
        # Validate input
        if self.config.validate_inputs:
            self.validation.validate_input(data, "input_data", MockDataFrame)
        
        # Check data quality
        if self.data_quality_validator and self.config.enable_data_quality_checks:
            tprint_info("🔍 Performing data quality checks...")
            quality_score = 0.95  # Mock quality score
            if quality_score < self.config.min_data_quality_score:
                error_msg = f"Data quality score {quality_score} below threshold {self.config.min_data_quality_score}"
                tprint_error(error_msg)
                if self.config.fail_fast:
                    raise ValueError(error_msg)
        
        # Start performance monitoring
        if self.performance_monitor:
            start_time = datetime.now()
            tprint_info("⏱️ Starting performance monitoring...")
        
        try:
            # Process based on operation type
            if operation_type == "feature_engineering":
                result = self._process_feature_engineering(data, **kwargs)
            elif operation_type == "backtesting":
                result = self._process_backtesting(data, **kwargs)
            elif operation_type == "cross_validation":
                result = self._process_cross_validation(data, **kwargs)
            elif operation_type == "vectorbt_optimization":
                result = self._process_vectorbt_optimization(data, **kwargs)
            else:
                result = self._process_generic(data, operation_type, **kwargs)
            
            # Validate output
            if self.config.validate_outputs:
                self.validation.validate_output(result, "processed_result", dict)
            
            # Log performance metrics
            if self.performance_monitor:
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()
                tprint_performance(f"Operation {operation_type}", duration)
                self.performance_metrics[operation_type] = {'duration': duration}
            
            tprint_success(f"✅ Data processing completed: {operation_type}")
            return result
            
        except Exception as e:
            tprint_exception(e, f"Data processing failed: {operation_type}")
            if self.error_handler:
                tprint_error("🔧 Error handler processing error...")
            raise
    
    def _process_feature_engineering(self, data, **kwargs):
        tprint_info("🔧 Processing feature engineering...")
        
        if self.vectorbt_optimizer and self.config.enable_vectorbt_optimization:
            tprint_info("🚀 Using VectorBT rolling optimizer for feature engineering")
            
            # Mock VectorBT processing
            features = MockDataFrame({
                'feature_1': [1, 2, 3, 4, 5],
                'feature_2': [2, 4, 6, 8, 10],
                'feature_3': [3, 6, 9, 12, 15]
            })
            
            tprint_success("✅ VectorBT feature engineering completed")
            return {
                'features': features,
                'metadata': {'method': 'vectorbt', 'features_added': 3},
                'optimization_used': 'vectorbt_rolling'
            }
        
        tprint_warning("⚠️ Using fallback feature engineering (VectorBT not available)")
        return self._fallback_feature_engineering(data, **kwargs)
    
    def _process_backtesting(self, data, **kwargs):
        tprint_info("📊 Processing backtesting...")
        
        if self.vectorization_manager and self.config.enable_unified_vectorization:
            tprint_info("🚀 Using unified vectorization for backtesting")
            
            # Mock backtesting results
            tprint_success("✅ Unified vectorization backtesting completed")
            return {
                'backtest_results': {
                    'total_return': 0.15,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': -0.05
                },
                'strategy_used': 'unified_vectorization',
                'performance_gain': 2.5,
                'metadata': {'method': 'unified_vectorization'}
            }
        
        tprint_warning("⚠️ Using fallback backtesting (unified vectorization not available)")
        return self._fallback_backtesting(data, **kwargs)
    
    def _process_cross_validation(self, data, **kwargs):
        tprint_info("🔄 Processing cross-validation...")
        
        if self.validation_system and self.config.enable_comprehensive_validation:
            tprint_info("🚀 Using comprehensive validation for cross-validation")
            
            # Mock validation results
            tprint_success("✅ Comprehensive validation cross-validation completed")
            return {
                'cv_results': {'mean_score': 0.85, 'std_score': 0.05},
                'validation_metrics': {'accuracy': 0.85, 'precision': 0.82, 'recall': 0.88},
                'metadata': {'method': 'comprehensive_validation', 'folds': 5}
            }
        
        tprint_warning("⚠️ Using fallback cross-validation (validation system not available)")
        return self._fallback_cross_validation(data, **kwargs)
    
    def _process_vectorbt_optimization(self, data, **kwargs):
        tprint_info("⚡ Processing VectorBT optimization...")
        
        if not self.vectorbt_optimizer:
            error_msg = "VectorBT optimizer not available"
            tprint_error(error_msg)
            raise RuntimeError(error_msg)
        
        # Mock VectorBT optimization
        tprint_success("✅ VectorBT optimization completed")
        return {
            'optimized_features': MockDataFrame({'opt_feature': [1, 2, 3, 4, 5]}),
            'performance_metrics': {'speedup': 3.2, 'memory_usage': 0.5},
            'optimization_strategy': 'vectorbt_parallel',
            'metadata': {'method': 'vectorbt_optimization'}
        }
    
    def _process_generic(self, data, operation_type, **kwargs):
        tprint_info(f"🔧 Processing generic operation: {operation_type}")
        
        result = {
            'operation_type': operation_type,
            'data_shape': data.shape,
            'processed_at': datetime.now().isoformat(),
            'metadata': kwargs
        }
        
        tprint_success(f"✅ Generic operation completed: {operation_type}")
        return result
    
    def _fallback_feature_engineering(self, data, **kwargs):
        tprint_warning("⚠️ Using fallback feature engineering")
        return {
            'features': MockDataFrame({'fallback_feature': [1, 2, 3]}),
            'metadata': {'method': 'fallback', 'indicators_added': 1},
            'optimization_used': 'fallback'
        }
    
    def _fallback_backtesting(self, data, **kwargs):
        tprint_warning("⚠️ Using fallback backtesting")
        return {
            'backtest_results': {'total_return': 0.05},
            'strategy_used': 'fallback',
            'performance_gain': 1.0,
            'metadata': {'method': 'fallback'}
        }
    
    def _fallback_cross_validation(self, data, **kwargs):
        tprint_warning("⚠️ Using fallback cross-validation")
        return {
            'cv_results': {'mean_score': 0.5},
            'validation_metrics': {'accuracy': 0.5},
            'metadata': {'method': 'fallback', 'folds': 5}
        }
    
    def get_pipeline_status(self):
        tprint_info("📊 Getting pipeline status...")
        
        status = {
            'components_available': {
                'vectorbt_rolling_optimizer': self.vectorbt_optimizer is not None,
                'unified_vectorization_manager': self.vectorization_manager is not None,
                'validation_system': self.validation_system is not None,
                'error_handler': self.error_handler is not None,
                'performance_monitor': self.performance_monitor is not None,
                'cache': self.cache is not None,
                'data_quality_validator': self.data_quality_validator is not None
            },
            'validation_summary': self.validation.get_validation_summary(),
            'performance_metrics': self.performance_metrics,
            'error_count': len(self.error_log),
            'config': {
                'enable_vectorbt_optimization': self.config.enable_vectorbt_optimization,
                'enable_unified_vectorization': self.config.enable_unified_vectorization,
                'enable_comprehensive_validation': self.config.enable_comprehensive_validation,
                'fail_fast': self.config.fail_fast
            }
        }
        
        tprint_success("✅ Pipeline status retrieved")
        return status
    
    def cleanup(self):
        tprint_info("🧹 Cleaning up pipeline resources...")
        tprint_structured({
            'total_operations': len(self.performance_metrics),
            'total_errors': len(self.error_log),
            'validation_summary': self.validation.get_validation_summary()
        })
        tprint_success("✅ Pipeline cleanup completed")

def create_enhanced_pipeline(config=None):
    """Create an enhanced unified data-driven pipeline."""
    tprint_info("🏗️ Creating enhanced unified data-driven pipeline")
    return EnhancedUnifiedDataDrivenPipeline(config)

def process_data_with_enhanced_pipeline(data, operation_type="feature_engineering", config=None, **kwargs):
    """Process data with the enhanced pipeline."""
    pipeline = create_enhanced_pipeline(config)
    try:
        result = pipeline.process_data(data, operation_type, **kwargs)
        return result
    finally:
        pipeline.cleanup()

def main():
    """Run the demo."""
    print("🚀 Enhanced Unified Data-Driven Pipeline Demo")
    print("=" * 60)
    
    # Create sample data
    tprint_info("📊 Creating sample data...")
    data = MockDataFrame({
        'timestamp': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'],
        'open': [100, 101, 102, 103, 104],
        'high': [105, 106, 107, 108, 109],
        'low': [95, 96, 97, 98, 99],
        'close': [101, 102, 103, 104, 105],
        'volume': [1000, 1100, 1200, 1300, 1400]
    })
    tprint_success(f"✅ Created sample data with shape {data.shape}")
    
    # Create pipeline
    tprint_info("🏗️ Creating enhanced pipeline...")
    config = EnhancedPipelineConfig()
    pipeline = create_enhanced_pipeline(config)
    
    try:
        # Test different operations
        operations = [
            "feature_engineering",
            "backtesting",
            "cross_validation",
            "vectorbt_optimization"
        ]
        
        for operation in operations:
            print(f"\n{'='*50}")
            tprint_info(f"🔄 Testing {operation}...")
            
            try:
                result = pipeline.process_data(data, operation_type=operation)
                tprint_success(f"✅ {operation} completed successfully")
                tprint_structured({
                    'operation': operation,
                    'result_keys': list(result.keys()),
                    'optimization_used': result.get('optimization_used', 'N/A')
                })
            except Exception as e:
                tprint_exception(e, f"❌ {operation} failed")
        
        # Test error handling
        print(f"\n{'='*50}")
        tprint_info("🚨 Testing error handling...")
        
        try:
            # Test with empty data
            empty_data = MockDataFrame({})
            pipeline.process_data(empty_data, operation_type="feature_engineering")
            tprint_error("❌ Should have failed with empty data")
        except Exception as e:
            tprint_success(f"✅ Correctly caught error: {type(e).__name__}")
        
        # Get pipeline status
        print(f"\n{'='*50}")
        tprint_info("📊 Getting pipeline status...")
        status = pipeline.get_pipeline_status()
        tprint_structured(status)
        
    finally:
        pipeline.cleanup()
    
    print(f"\n{'='*60}")
    tprint_success("🎉 Enhanced Unified Data-Driven Pipeline Demo completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Comprehensive tprint logging throughout")
    print("✅ Fast failing validation with detailed error reporting")
    print("✅ Integration with VectorBTRollingOptimizer")
    print("✅ Integration with UnifiedVectorizationManager")
    print("✅ No silent failures - all operations logged")
    print("✅ Performance monitoring and metrics")
    print("✅ Error handling and recovery")
    print("✅ Structured logging and status reporting")

if __name__ == "__main__":
    main()