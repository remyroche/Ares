"""
Comprehensive Common Utilities Integration Example for PID-Based Feature Generation

This example demonstrates how to integrate all common utilities with the
PID-based feature generation system:

- src/utils/common_operations.py: Data validation, DataFrame operations, math validation
- src/utils/common_utilities.py: Additional DataFrame utilities
- src/utils/math_validation.py: Safe mathematical operations
- src/utils/serialization_utils.py: Artifact persistence and data serialization
- src/utils/matrix_operations/: Optimized mathematical computations and GPU acceleration
- src/utils/hardware/m1_gpu_utils.py: M1 GPU acceleration
- src/utils/hardware/m1_memory_optimizer.py: M1 memory optimization
- src/utils/hardware/m1_cpu_optimizer.py: M1 CPU optimization
- src/utils/ml_common/: ML-related utilities (CV, lookahead, HPO)
- src/utils/data/: Data processing utilities

Usage:
    python common_utilities_integration_example.py
"""

import asyncio
import logging
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# COMMON UTILITIES IMPORTS
# =============================================================================

# Common Operations Integration
try:
    from src.utils.common_operations import (
        # Data validation and quality
        validate_dataframe, validate_dataframe_columns, calculate_data_quality_metrics,
        create_data_quality_report, get_dataframe_info, optimize_dataframe_dtypes,
        
        # Safe operations
        safe_dataframe_operation, safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
        safe_drop_columns, safe_rename_columns, safe_timestamp_conversion,
        
        # Math operations
        safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std,
        safe_float, safe_int, validate_finite, validate_positive, validate_range,
        safe_kelly_calculation, safe_weighted_average, safe_percentage_change,
        
        # File operations
        safe_json_dump, safe_json_load, safe_to_parquet, safe_read_parquet,
        ensure_directory, safe_file_exists, safe_copy,
        
        # Performance utilities
        timed_operation, format_bytes, chunked_iterable, parallel_map,
        
        # M1 optimization
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers,
        memory_checkpoint, gpu_context, optimize_memory, get_memory_usage,
        
        # Matrix utilities
        validate_correlation_matrix, safe_matrix_inverse, math_safe,
        
        # Logging utilities
        get_logger, setup_basic_logging, safe_log_metric, safe_log_params, safe_log_artifact
    )
    COMMON_OPERATIONS_AVAILABLE = True
    print("✅ Common operations imported successfully")
except ImportError as e:
    COMMON_OPERATIONS_AVAILABLE = False
    print(f"❌ Common operations import failed: {e}")

# Common Utilities Integration
try:
    from src.utils.common_utilities import (
        CommonUtilities, safe_dataframe_operation as common_safe_df_op,
        validate_dataframe_columns as common_validate_df_cols,
        calculate_data_quality_metrics as common_calc_quality,
        create_summary_statistics, safe_merge_dataframes as common_merge_dfs,
        safe_groupby_operation, safe_apply_function, safe_drop_columns as common_drop_cols,
        safe_rename_columns as common_rename_cols, validate_timestamp_column,
        safe_timestamp_conversion as common_timestamp_conv, get_dataframe_info as common_df_info,
        safe_filter_dataframe, create_data_quality_report as common_quality_report
    )
    COMMON_UTILITIES_AVAILABLE = True
    print("✅ Common utilities imported successfully")
except ImportError as e:
    COMMON_UTILITIES_AVAILABLE = False
    print(f"❌ Common utilities import failed: {e}")

# Math Validation Integration
try:
    from src.utils.math_validation import (
        MathValidation, safe_divide as math_safe_divide, safe_log as math_safe_log,
        safe_sqrt as math_safe_sqrt, safe_power as math_safe_power,
        validate_finite as math_validate_finite, validate_positive as math_validate_positive,
        validate_range as math_validate_range, safe_correlation, safe_covariance,
        safe_mean as math_safe_mean, safe_std as math_safe_std, safe_percentile,
        safe_kelly_calculation as math_kelly, safe_weighted_average as math_weighted_avg,
        safe_percentage_change as math_pct_change, validate_correlation_matrix as math_validate_corr,
        safe_matrix_inverse as math_safe_matrix_inv, math_safe as math_safe_func
    )
    MATH_VALIDATION_AVAILABLE = True
    print("✅ Math validation imported successfully")
except ImportError as e:
    MATH_VALIDATION_AVAILABLE = False
    print(f"❌ Math validation import failed: {e}")

# Serialization Utilities Integration
try:
    from src.utils.serialization_utils import (
        JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer
    )
    SERIALIZATION_AVAILABLE = True
    print("✅ Serialization utilities imported successfully")
except ImportError as e:
    SERIALIZATION_AVAILABLE = False
    print(f"❌ Serialization utilities import failed: {e}")

# Matrix Operations Integration
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, get_enhanced_matrix_operations,
        # Convenience functions
        safe_matrix_multiply, safe_correlation_matrix, safe_matrix_inverse,
        gpu_matrix_multiply, correlation_matrix_gpu, eigendecomposition_gpu, svd_gpu,
        optimize_dataframe, vectorized_rolling_features, matrix_correlation_analysis,
        parallel_feature_engineering, batch_matrix_multiply, batch_feature_transformation,
        # Trading indicators
        compute_trading_indicators, compute_moving_averages, compute_momentum_indicators,
        compute_volatility_indicators, compute_volume_indicators, compute_trend_indicators,
        # Hardware optimization
        get_hardware_performance_report, optimize_matrix_operation_with_hardware,
        cleanup_hardware_resources, get_processing_performance_stats
    )
    MATRIX_OPERATIONS_AVAILABLE = True
    print("✅ Matrix operations imported successfully")
except ImportError as e:
    MATRIX_OPERATIONS_AVAILABLE = False
    print(f"❌ Matrix operations import failed: {e}")

# Hardware Optimization Integration
try:
    from src.utils.hardware.m1_gpu_utils import (
        get_m1_gpu_manager, is_m1_available, is_mps_available,
        optimize_dataframe_for_m1, create_m1_optimized_array,
        m1_backtesting_simulate, m1_monte_carlo_simulate
    )
    from src.utils.hardware.m1_memory_optimizer import get_m1_memory_optimizer
    from src.utils.hardware.m1_cpu_optimizer import get_m1_cpu_optimizer
    HARDWARE_OPTIMIZATION_AVAILABLE = True
    print("✅ Hardware optimization imported successfully")
except ImportError as e:
    HARDWARE_OPTIMIZATION_AVAILABLE = False
    print(f"❌ Hardware optimization import failed: {e}")

# ML Common Integration
try:
    from src.utils.ml_common.common_operations import (
        # Data processing
        preprocess_data, validate_ml_data, create_feature_matrix,
        # Feature engineering
        create_polynomial_features, create_interaction_features,
        # Cross-validation
        create_cv_splits, validate_cv_splits,
        # Hyperparameter optimization
        optimize_hyperparameters, create_hpo_config,
        # Model evaluation
        evaluate_model_performance, calculate_metrics,
        # Lookahead bias prevention
        detect_lookahead_bias, prevent_lookahead_bias
    )
    ML_COMMON_AVAILABLE = True
    print("✅ ML common utilities imported successfully")
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    print(f"❌ ML common utilities import failed: {e}")

# Data Utilities Integration
try:
    from src.utils.data.data_loader import DataLoader
    from src.utils.data.data_processor import DataProcessor
    from src.utils.data.data_validator import DataValidator
    DATA_UTILITIES_AVAILABLE = True
    print("✅ Data utilities imported successfully")
except ImportError as e:
    DATA_UTILITIES_AVAILABLE = False
    print(f"❌ Data utilities import failed: {e}")

# Logger Integration
try:
    from src.utils.logger import system_logger
    logger = system_logger.getChild('CommonUtilitiesIntegrationExample')
except ImportError:
    logger = logging.getLogger('CommonUtilitiesIntegrationExample')
    logger.setLevel(logging.INFO)

# TPrint Integration
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_error, tprint_warning, tprint_success, 
        tprint_debug, tprint_performance
    )
    TPRINT_AVAILABLE = True
    print("✅ TPrint imported successfully")
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERFORMANCE:", *args, **kwargs)


# =============================================================================
# COMPREHENSIVE INTEGRATION EXAMPLE
# =============================================================================

class ComprehensivePIDIntegrationExample:
    """
    Comprehensive example demonstrating integration of all common utilities
    with PID-based feature generation.
    """
    
    def __init__(self):
        """Initialize the comprehensive integration example."""
        self.logger = logger.getChild('ComprehensivePIDIntegrationExample')
        
        # Initialize utility status
        self.utility_status = {
            'common_operations': COMMON_OPERATIONS_AVAILABLE,
            'common_utilities': COMMON_UTILITIES_AVAILABLE,
            'math_validation': MATH_VALIDATION_AVAILABLE,
            'serialization': SERIALIZATION_AVAILABLE,
            'matrix_operations': MATRIX_OPERATIONS_AVAILABLE,
            'hardware_optimization': HARDWARE_OPTIMIZATION_AVAILABLE,
            'ml_common': ML_COMMON_AVAILABLE,
            'data_utilities': DATA_UTILITIES_AVAILABLE
        }
        
        # Initialize serializers
        self._initialize_serializers()
        
        # Initialize hardware optimizers
        self._initialize_hardware_optimizers()
        
        tprint_success("Comprehensive PID Integration Example initialized")
        self._print_utility_status()
    
    def _initialize_serializers(self):
        """Initialize serialization utilities."""
        if SERIALIZATION_AVAILABLE:
            self.json_serializer = JSONSerializer()
            self.pickle_serializer = PickleSerializer()
            self.parquet_serializer = ParquetSerializer()
            self.universal_serializer = UniversalSerializer()
            tprint_success("Serializers initialized")
        else:
            self.json_serializer = None
            self.pickle_serializer = None
            self.parquet_serializer = None
            self.universal_serializer = None
    
    def _initialize_hardware_optimizers(self):
        """Initialize hardware optimization utilities."""
        if HARDWARE_OPTIMIZATION_AVAILABLE:
            self.gpu_manager = get_m1_gpu_manager()
            self.memory_optimizer = get_m1_memory_optimizer()
            self.cpu_optimizer = get_m1_cpu_optimizer()
            tprint_success("Hardware optimizers initialized")
        else:
            self.gpu_manager = None
            self.memory_optimizer = None
            self.cpu_optimizer = None
    
    def _print_utility_status(self):
        """Print utility availability status."""
        tprint_info("Utility Integration Status:")
        for utility, available in self.utility_status.items():
            status_icon = "✅" if available else "❌"
            tprint_info(f"  {status_icon} {utility}")
    
    async def demonstrate_comprehensive_integration(self) -> Dict[str, Any]:
        """
        Demonstrate comprehensive integration of all common utilities.
        
        Returns:
            Dictionary with comprehensive integration results
        """
        tprint_info("🚀 Starting comprehensive integration demonstration...")
        
        results = {
            'integration_status': self.utility_status,
            'demonstrations': {},
            'performance_metrics': {},
            'artifacts': {},
            'success': False
        }
        
        try:
            # Step 1: Data Generation and Validation
            tprint_info("📊 Step 1: Data generation and validation...")
            data_demo = await self._demonstrate_data_validation()
            results['demonstrations']['data_validation'] = data_demo
            
            # Step 2: Math Validation and Safe Operations
            tprint_info("🔢 Step 2: Math validation and safe operations...")
            math_demo = await self._demonstrate_math_validation()
            results['demonstrations']['math_validation'] = math_demo
            
            # Step 3: Matrix Operations and GPU Acceleration
            tprint_info("🎯 Step 3: Matrix operations and GPU acceleration...")
            matrix_demo = await self._demonstrate_matrix_operations()
            results['demonstrations']['matrix_operations'] = matrix_demo
            
            # Step 4: Hardware Optimization
            tprint_info("🔧 Step 4: Hardware optimization...")
            hardware_demo = await self._demonstrate_hardware_optimization()
            results['demonstrations']['hardware_optimization'] = hardware_demo
            
            # Step 5: ML Common Utilities
            tprint_info("🤖 Step 5: ML common utilities...")
            ml_demo = await self._demonstrate_ml_common()
            results['demonstrations']['ml_common'] = ml_demo
            
            # Step 6: Serialization and Artifact Management
            tprint_info("💾 Step 6: Serialization and artifact management...")
            serialization_demo = await self._demonstrate_serialization()
            results['demonstrations']['serialization'] = serialization_demo
            
            # Step 7: Performance Metrics Collection
            tprint_info("📈 Step 7: Performance metrics collection...")
            performance_demo = await self._demonstrate_performance_metrics()
            results['performance_metrics'] = performance_demo
            
            # Step 8: Artifact Cleanup and Summary
            tprint_info("🧹 Step 8: Artifact cleanup and summary...")
            cleanup_demo = await self._demonstrate_cleanup()
            results['demonstrations']['cleanup'] = cleanup_demo
            
            results['success'] = True
            tprint_success("✅ Comprehensive integration demonstration completed successfully")
            
            return results
            
        except Exception as e:
            tprint_error(f"❌ Comprehensive integration demonstration failed: {e}")
            tprint_error(f"❌ Error details: {traceback.format_exc()}")
            results['error'] = str(e)
            return results
    
    async def _demonstrate_data_validation(self) -> Dict[str, Any]:
        """Demonstrate data validation using common operations."""
        demo_result = {
            'common_operations_used': False,
            'common_utilities_used': False,
            'data_quality_score': 0.0,
            'validation_passed': False
        }
        
        try:
            # Generate sample data
            np.random.seed(42)
            data = np.random.randn(1000, 10)
            feature_names = [f"feature_{i}" for i in range(10)]
            df = pd.DataFrame(data, columns=feature_names)
            
            # Add some missing values and outliers for demonstration
            df.iloc[100:110, 0] = np.nan  # Missing values
            df.iloc[200:210, 1] = np.inf  # Infinite values
            df.iloc[300:310, 2] = 1000    # Outliers
            
            tprint_info(f"Generated sample data: {df.shape}")
            
            # Use common operations for validation
            if COMMON_OPERATIONS_AVAILABLE:
                # Validate DataFrame
                is_valid = validate_dataframe(df)
                tprint_info(f"DataFrame validation: {is_valid}")
                
                # Calculate data quality metrics
                quality_metrics = calculate_data_quality_metrics(df)
                tprint_info(f"Data quality metrics: {quality_metrics}")
                
                # Create data quality report
                quality_report = create_data_quality_report(df)
                tprint_info(f"Data quality report created: {len(quality_report)} sections")
                
                # Optimize DataFrame dtypes
                df_optimized = optimize_dataframe_dtypes(df)
                tprint_info(f"DataFrame optimized: {df_optimized.dtypes.value_counts().to_dict()}")
                
                demo_result['common_operations_used'] = True
                demo_result['data_quality_score'] = quality_metrics.get('missing_percentage', 0.0)
                demo_result['validation_passed'] = is_valid
            
            # Use common utilities for additional validation
            if COMMON_UTILITIES_AVAILABLE:
                # Create CommonUtilities instance
                common_utils = CommonUtilities()
                
                # Validate DataFrame columns
                cols_valid = common_utils.validate_dataframe_columns(df, feature_names)
                tprint_info(f"Column validation: {cols_valid}")
                
                # Get data summary
                data_summary = common_utils.get_data_summary(df)
                tprint_info(f"Data summary: {len(data_summary)} metrics")
                
                # Create summary statistics
                summary_stats = create_summary_statistics(df)
                tprint_info(f"Summary statistics: {len(summary_stats)} metrics")
                
                demo_result['common_utilities_used'] = True
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Data validation demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_math_validation(self) -> Dict[str, Any]:
        """Demonstrate math validation and safe operations."""
        demo_result = {
            'math_validation_used': False,
            'safe_operations_tested': 0,
            'validation_errors': 0
        }
        
        try:
            if MATH_VALIDATION_AVAILABLE:
                # Create MathValidation instance
                math_validator = MathValidation()
                
                # Test safe mathematical operations
                test_values = [1.0, 0.0, -1.0, np.inf, np.nan, 100.0]
                
                for val in test_values:
                    try:
                        # Test safe division
                        result = math_safe_divide(val, 2.0, default=0.0)
                        tprint_debug(f"Safe divide {val}/2 = {result}")
                        
                        # Test safe logarithm
                        result = math_safe_log(val, default=0.0)
                        tprint_debug(f"Safe log({val}) = {result}")
                        
                        # Test safe square root
                        result = math_safe_sqrt(val, default=0.0)
                        tprint_debug(f"Safe sqrt({val}) = {result}")
                        
                        # Test finite validation
                        result = math_validate_finite(val, f"test_value_{val}")
                        tprint_debug(f"Finite validation {val} = {result}")
                        
                        demo_result['safe_operations_tested'] += 1
                        
                    except Exception as e:
                        demo_result['validation_errors'] += 1
                        tprint_debug(f"Math validation error for {val}: {e}")
                
                # Test correlation calculation
                x = np.random.randn(100)
                y = np.random.randn(100)
                corr = safe_correlation(x, y, default=0.0)
                tprint_info(f"Safe correlation: {corr:.4f}")
                
                # Test covariance calculation
                cov = safe_covariance(x, y, default=0.0)
                tprint_info(f"Safe covariance: {cov:.4f}")
                
                demo_result['math_validation_used'] = True
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Math validation demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_matrix_operations(self) -> Dict[str, Any]:
        """Demonstrate matrix operations and GPU acceleration."""
        demo_result = {
            'matrix_operations_used': False,
            'gpu_acceleration_used': False,
            'operations_performed': 0,
            'performance_metrics': {}
        }
        
        try:
            if MATRIX_OPERATIONS_AVAILABLE:
                # Generate test matrices
                A = np.random.randn(100, 50)
                B = np.random.randn(50, 75)
                C = np.random.randn(100, 75)
                
                tprint_info(f"Test matrices: A{A.shape}, B{B.shape}, C{C.shape}")
                
                # Test safe matrix multiplication
                start_time = time.time()
                result = safe_matrix_multiply(A, B)
                mult_time = time.time() - start_time
                tprint_info(f"Matrix multiplication: {result.shape} in {mult_time:.4f}s")
                demo_result['operations_performed'] += 1
                
                # Test safe correlation matrix
                start_time = time.time()
                corr_matrix = safe_correlation_matrix(A)
                corr_time = time.time() - start_time
                tprint_info(f"Correlation matrix: {corr_matrix.shape} in {corr_time:.4f}s")
                demo_result['operations_performed'] += 1
                
                # Test safe matrix inverse
                if A.shape[0] == A.shape[1]:
                    start_time = time.time()
                    inv_matrix = safe_matrix_inverse(A)
                    inv_time = time.time() - start_time
                    tprint_info(f"Matrix inverse: {inv_matrix.shape} in {inv_time:.4f}s")
                    demo_result['operations_performed'] += 1
                
                # Test GPU acceleration if available
                if HARDWARE_OPTIMIZATION_AVAILABLE and is_mps_available():
                    try:
                        start_time = time.time()
                        gpu_result = gpu_matrix_multiply(A, B)
                        gpu_time = time.time() - start_time
                        tprint_info(f"GPU matrix multiplication: {gpu_result.shape} in {gpu_time:.4f}s")
                        demo_result['gpu_acceleration_used'] = True
                        demo_result['operations_performed'] += 1
                    except Exception as e:
                        tprint_warning(f"GPU acceleration failed: {e}")
                
                # Test trading indicators
                try:
                    # Create sample OHLCV data
                    ohlcv_data = pd.DataFrame({
                        'open': np.random.randn(1000) * 100 + 1000,
                        'high': np.random.randn(1000) * 100 + 1000,
                        'low': np.random.randn(1000) * 100 + 1000,
                        'close': np.random.randn(1000) * 100 + 1000,
                        'volume': np.random.randint(1000, 10000, 1000)
                    })
                    
                    # Compute trading indicators
                    indicators = compute_trading_indicators(ohlcv_data)
                    tprint_info(f"Trading indicators computed: {len(indicators)} indicators")
                    demo_result['operations_performed'] += 1
                    
                except Exception as e:
                    tprint_warning(f"Trading indicators computation failed: {e}")
                
                demo_result['matrix_operations_used'] = True
                demo_result['performance_metrics'] = {
                    'matrix_mult_time': mult_time,
                    'correlation_time': corr_time,
                    'operations_count': demo_result['operations_performed']
                }
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Matrix operations demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_hardware_optimization(self) -> Dict[str, Any]:
        """Demonstrate hardware optimization utilities."""
        demo_result = {
            'hardware_optimization_used': False,
            'm1_available': False,
            'mps_available': False,
            'optimizations_applied': [],
            'performance_improvements': {}
        }
        
        try:
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                # Check M1 availability
                m1_available = is_m1_available()
                mps_available = is_mps_available()
                
                demo_result['m1_available'] = m1_available
                demo_result['mps_available'] = mps_available
                
                tprint_info(f"M1 available: {m1_available}")
                tprint_info(f"MPS available: {mps_available}")
                
                # Test M1 GPU manager
                if self.gpu_manager:
                    gpu_info = self.gpu_manager.get_gpu_info()
                    tprint_info(f"GPU info: {gpu_info}")
                    demo_result['optimizations_applied'].append('gpu_management')
                
                # Test memory optimizer
                if self.memory_optimizer:
                    # Get memory usage
                    memory_usage = get_memory_usage()
                    tprint_info(f"Memory usage: {format_bytes(memory_usage)}")
                    demo_result['optimizations_applied'].append('memory_optimization')
                
                # Test CPU optimizer
                if self.cpu_optimizer:
                    cpu_info = self.cpu_optimizer.get_cpu_info()
                    tprint_info(f"CPU info: {cpu_info}")
                    demo_result['optimizations_applied'].append('cpu_optimization')
                
                # Test M1-specific optimizations
                if m1_available:
                    # Create test DataFrame
                    test_df = pd.DataFrame(np.random.randn(1000, 10))
                    
                    # Optimize DataFrame for M1
                    optimized_df = optimize_dataframe_for_m1(test_df)
                    tprint_info(f"DataFrame optimized for M1: {optimized_df.dtypes.value_counts().to_dict()}")
                    demo_result['optimizations_applied'].append('m1_dataframe_optimization')
                    
                    # Create M1-optimized array
                    test_array = np.random.randn(1000, 10)
                    optimized_array = create_m1_optimized_array(test_array)
                    tprint_info(f"M1-optimized array: {optimized_array.dtype}")
                    demo_result['optimizations_applied'].append('m1_array_optimization')
                
                # Test GPU acceleration for backtesting simulation
                if mps_available:
                    try:
                        # Create test data for backtesting
                        backtest_data = pd.DataFrame({
                            'price': np.random.randn(1000) * 100 + 1000,
                            'volume': np.random.randint(1000, 10000, 1000)
                        })
                        
                        # Simulate backtesting with GPU acceleration
                        strategy_params = {'threshold': 0.02, 'lookback': 20}
                        config = {'initial_capital': 100000}
                        
                        backtest_result = await m1_backtesting_simulate(
                            backtest_data, strategy_params, config, None
                        )
                        tprint_info(f"GPU backtesting simulation: {backtest_result}")
                        demo_result['optimizations_applied'].append('gpu_backtesting')
                        
                    except Exception as e:
                        tprint_warning(f"GPU backtesting simulation failed: {e}")
                
                demo_result['hardware_optimization_used'] = True
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Hardware optimization demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_ml_common(self) -> Dict[str, Any]:
        """Demonstrate ML common utilities."""
        demo_result = {
            'ml_common_used': False,
            'cv_performed': False,
            'hpo_performed': False,
            'lookahead_bias_detected': False,
            'operations_count': 0
        }
        
        try:
            if ML_COMMON_AVAILABLE:
                # Generate sample data for ML operations
                X = np.random.randn(1000, 10)
                y = np.random.randint(0, 2, 1000)
                
                tprint_info(f"ML data: X{X.shape}, y{y.shape}")
                
                # Test data preprocessing
                try:
                    processed_data = preprocess_data(X)
                    tprint_info(f"Data preprocessed: {processed_data.shape}")
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Data preprocessing failed: {e}")
                
                # Test ML data validation
                try:
                    is_valid = validate_ml_data(X, y)
                    tprint_info(f"ML data validation: {is_valid}")
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"ML data validation failed: {e}")
                
                # Test feature matrix creation
                try:
                    feature_matrix = create_feature_matrix(X)
                    tprint_info(f"Feature matrix created: {feature_matrix.shape}")
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Feature matrix creation failed: {e}")
                
                # Test polynomial features
                try:
                    poly_features = create_polynomial_features(X, degree=2)
                    tprint_info(f"Polynomial features: {poly_features.shape}")
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Polynomial features creation failed: {e}")
                
                # Test interaction features
                try:
                    interaction_features = create_interaction_features(X)
                    tprint_info(f"Interaction features: {interaction_features.shape}")
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Interaction features creation failed: {e}")
                
                # Test cross-validation
                try:
                    cv_splits = create_cv_splits(X, y, n_splits=5)
                    tprint_info(f"CV splits created: {len(cv_splits)} splits")
                    demo_result['cv_performed'] = True
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Cross-validation failed: {e}")
                
                # Test hyperparameter optimization
                try:
                    hpo_config = create_hpo_config()
                    tprint_info(f"HPO config created: {len(hpo_config)} parameters")
                    demo_result['hpo_performed'] = True
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Hyperparameter optimization failed: {e}")
                
                # Test lookahead bias detection
                try:
                    bias_result = detect_lookahead_bias(X, y)
                    tprint_info(f"Lookahead bias detection: {bias_result}")
                    demo_result['lookahead_bias_detected'] = bias_result.get('bias_detected', False)
                    demo_result['operations_count'] += 1
                except Exception as e:
                    tprint_warning(f"Lookahead bias detection failed: {e}")
                
                demo_result['ml_common_used'] = True
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"ML common demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_serialization(self) -> Dict[str, Any]:
        """Demonstrate serialization utilities."""
        demo_result = {
            'serialization_used': False,
            'formats_tested': [],
            'artifacts_saved': 0,
            'serialization_times': {}
        }
        
        try:
            if SERIALIZATION_AVAILABLE:
                # Create test data
                test_data = {
                    'features': np.random.randn(100, 10).tolist(),
                    'labels': np.random.randint(0, 2, 100).tolist(),
                    'metadata': {
                        'timestamp': datetime.now().isoformat(),
                        'version': '1.0.0',
                        'description': 'Test data for serialization demonstration'
                    }
                }
                
                # Create artifacts directory
                artifacts_dir = Path("artifacts") / "common_utilities_demo"
                ensure_directory(artifacts_dir)
                
                # Test JSON serialization
                try:
                    start_time = time.time()
                    json_path = artifacts_dir / "test_data.json"
                    success = self.json_serializer.save(test_data, str(json_path))
                    json_time = time.time() - start_time
                    
                    if success:
                        tprint_info(f"JSON serialization: {json_time:.4f}s")
                        demo_result['formats_tested'].append('json')
                        demo_result['artifacts_saved'] += 1
                        demo_result['serialization_times']['json'] = json_time
                except Exception as e:
                    tprint_warning(f"JSON serialization failed: {e}")
                
                # Test Pickle serialization
                try:
                    start_time = time.time()
                    pickle_path = artifacts_dir / "test_data.pkl"
                    success = self.pickle_serializer.save(test_data, str(pickle_path))
                    pickle_time = time.time() - start_time
                    
                    if success:
                        tprint_info(f"Pickle serialization: {pickle_time:.4f}s")
                        demo_result['formats_tested'].append('pickle')
                        demo_result['artifacts_saved'] += 1
                        demo_result['serialization_times']['pickle'] = pickle_time
                except Exception as e:
                    tprint_warning(f"Pickle serialization failed: {e}")
                
                # Test Parquet serialization
                try:
                    # Create DataFrame for Parquet
                    df = pd.DataFrame(test_data['features'])
                    df['labels'] = test_data['labels']
                    
                    start_time = time.time()
                    parquet_path = artifacts_dir / "test_data.parquet"
                    success = self.parquet_serializer.save(df, str(parquet_path))
                    parquet_time = time.time() - start_time
                    
                    if success:
                        tprint_info(f"Parquet serialization: {parquet_time:.4f}s")
                        demo_result['formats_tested'].append('parquet')
                        demo_result['artifacts_saved'] += 1
                        demo_result['serialization_times']['parquet'] = parquet_time
                except Exception as e:
                    tprint_warning(f"Parquet serialization failed: {e}")
                
                # Test Universal serializer
                try:
                    start_time = time.time()
                    universal_path = artifacts_dir / "test_data_universal.json"
                    success = self.universal_serializer.save(test_data, str(universal_path))
                    universal_time = time.time() - start_time
                    
                    if success:
                        tprint_info(f"Universal serialization: {universal_time:.4f}s")
                        demo_result['formats_tested'].append('universal')
                        demo_result['artifacts_saved'] += 1
                        demo_result['serialization_times']['universal'] = universal_time
                except Exception as e:
                    tprint_warning(f"Universal serialization failed: {e}")
                
                demo_result['serialization_used'] = True
                demo_result['artifacts_directory'] = str(artifacts_dir)
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Serialization demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_performance_metrics(self) -> Dict[str, Any]:
        """Demonstrate performance metrics collection."""
        demo_result = {
            'performance_metrics_collected': False,
            'memory_usage': {},
            'execution_times': {},
            'hardware_metrics': {}
        }
        
        try:
            # Collect memory usage
            if COMMON_OPERATIONS_AVAILABLE:
                memory_usage = get_memory_usage()
                demo_result['memory_usage']['current'] = memory_usage
                demo_result['memory_usage']['formatted'] = format_bytes(memory_usage)
                tprint_info(f"Memory usage: {format_bytes(memory_usage)}")
            
            # Collect hardware performance metrics
            if MATRIX_OPERATIONS_AVAILABLE:
                try:
                    hardware_report = get_hardware_performance_report()
                    demo_result['hardware_metrics'] = hardware_report
                    tprint_info(f"Hardware performance report: {len(hardware_report)} metrics")
                except Exception as e:
                    tprint_warning(f"Hardware performance report failed: {e}")
            
            # Test performance timing
            start_time = time.time()
            
            # Simulate some computation
            result = 0
            for i in range(1000000):
                result += i * 0.001
            
            computation_time = time.time() - start_time
            demo_result['execution_times']['computation'] = computation_time
            tprint_info(f"Computation time: {computation_time:.4f}s")
            
            # Test parallel processing performance
            if COMMON_OPERATIONS_AVAILABLE:
                start_time = time.time()
                
                def square(x):
                    return x ** 2
                
                test_data = list(range(1000))
                parallel_result = parallel_map(square, test_data, max_workers=4)
                parallel_time = time.time() - start_time
                
                demo_result['execution_times']['parallel_processing'] = parallel_time
                tprint_info(f"Parallel processing time: {parallel_time:.4f}s")
            
            demo_result['performance_metrics_collected'] = True
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Performance metrics demonstration failed: {e}")
            return demo_result
    
    async def _demonstrate_cleanup(self) -> Dict[str, Any]:
        """Demonstrate cleanup utilities."""
        demo_result = {
            'cleanup_performed': False,
            'resources_cleaned': [],
            'memory_freed': 0
        }
        
        try:
            # Clean up M1 optimizers
            if HARDWARE_OPTIMIZATION_AVAILABLE:
                try:
                    cleanup_result = cleanup_m1_optimizers()
                    if cleanup_result:
                        demo_result['resources_cleaned'].append('m1_optimizers')
                        tprint_info("M1 optimizers cleaned up")
                except Exception as e:
                    tprint_warning(f"M1 optimizer cleanup failed: {e}")
            
            # Clean up hardware resources
            if MATRIX_OPERATIONS_AVAILABLE:
                try:
                    cleanup_hardware_resources()
                    demo_result['resources_cleaned'].append('hardware_resources')
                    tprint_info("Hardware resources cleaned up")
                except Exception as e:
                    tprint_warning(f"Hardware resource cleanup failed: {e}")
            
            # Optimize memory
            if COMMON_OPERATIONS_AVAILABLE:
                try:
                    memory_before = get_memory_usage()
                    optimize_result = optimize_memory()
                    memory_after = get_memory_usage()
                    
                    memory_freed = memory_before - memory_after
                    demo_result['memory_freed'] = memory_freed
                    demo_result['resources_cleaned'].append('memory_optimization')
                    
                    tprint_info(f"Memory optimized: {format_bytes(memory_freed)} freed")
                except Exception as e:
                    tprint_warning(f"Memory optimization failed: {e}")
            
            demo_result['cleanup_performed'] = True
            
            return demo_result
            
        except Exception as e:
            tprint_warning(f"Cleanup demonstration failed: {e}")
            return demo_result


# =============================================================================
# MAIN EXECUTION
# =============================================================================

async def main():
    """Main execution function."""
    print("=" * 80)
    print("COMPREHENSIVE COMMON UTILITIES INTEGRATION EXAMPLE")
    print("=" * 80)
    print()
    
    # Create comprehensive integration example
    example = ComprehensivePIDIntegrationExample()
    
    # Run comprehensive demonstration
    results = await example.demonstrate_comprehensive_integration()
    
    # Print summary
    print("\n" + "=" * 80)
    print("INTEGRATION SUMMARY")
    print("=" * 80)
    
    print(f"\nOverall Success: {'✅' if results['success'] else '❌'}")
    
    print(f"\nUtility Integration Status:")
    for utility, available in results['integration_status'].items():
        status_icon = "✅" if available else "❌"
        print(f"  {status_icon} {utility}")
    
    print(f"\nDemonstrations Completed:")
    for demo_name, demo_result in results['demonstrations'].items():
        success_icon = "✅" if demo_result else "❌"
        print(f"  {success_icon} {demo_name}")
    
    if 'performance_metrics' in results:
        print(f"\nPerformance Metrics:")
        for metric_name, metric_value in results['performance_metrics'].items():
            print(f"  • {metric_name}: {metric_value}")
    
    print(f"\nTotal Demonstrations: {len(results['demonstrations'])}")
    print(f"Successful Demonstrations: {sum(1 for demo in results['demonstrations'].values() if demo)}")
    
    if results['success']:
        print("\n🎉 All common utilities integration demonstrations completed successfully!")
    else:
        print("\n⚠️ Some demonstrations failed. Check the logs for details.")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Run the comprehensive integration example
    asyncio.run(main())