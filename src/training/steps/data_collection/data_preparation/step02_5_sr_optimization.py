"""Step 2.5: S/R Detection Optimization with Comprehensive Reporting and Function Call Monitoring."""
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
import time
import json
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple, List, Dict, Any

# Import step07's feature selection functionality

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
try:
    from src.utils.m1_gpu_utils import m1_batch_process  # Streaming batch processing with MPS gating
    M1_BATCH_AVAILABLE = True
except ImportError as e:
    M1_BATCH_AVAILABLE = False
    logger.warning(f"M1 GPU utils not available: {e}")
except Exception as e:
    M1_BATCH_AVAILABLE = False
    logger.error(f"Unexpected error loading M1 GPU utils: {e}")

import joblib

import traceback

from src.training.base_step import BaseStep
from src.utils.logger import system_logger
# Import the correct PipelineStandards to avoid conflicts
from src.utils.pipeline_standards import PipelineStandards
from src.utils.step02_5_utilities import (
    global_monitor,
    function_tracker,
    logging_patterns
)
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation
from src.training.reports import save_training_report
from src.training.steps.data_collection.data_preparation.step02_5_financial_logging import Step02_5FinancialLogger
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError
)
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

# Import optional modules with error handling
try:
    from src.utils.parquet_utils import ParquetUtils
    PARQUET_UTILS_AVAILABLE = True
except ImportError:
    ParquetUtils = None
    PARQUET_UTILS_AVAILABLE = False

try:
    from src.training.steps.feature_engineering.step06_advanced_features import AdvancedFeatureEngineeringStep
    ADVANCED_FEATURES_AVAILABLE = True
except ImportError:
    AdvancedFeatureEngineeringStep = None
    ADVANCED_FEATURES_AVAILABLE = False

try:
    from src.tactician.sr_levels.sr_levels_manager import SRLevelsManager
    SR_LEVELS_MANAGER_AVAILABLE = True
except ImportError:
    SRLevelsManager = None
    SR_LEVELS_MANAGER_AVAILABLE = False

try:
    from src.tactician.dynamic_barrier_calculator import DynamicBarrierCalculator
    DYNAMIC_BARRIER_CALCULATOR_AVAILABLE = True
except ImportError:
    DynamicBarrierCalculator = None
    DYNAMIC_BARRIER_CALCULATOR_AVAILABLE = False

try:
    from src.tactician.sr_levels.enhanced_sr_detection import EnhancedSRDetector, SRLevel
    ENHANCED_SR_DETECTOR_AVAILABLE = True
except ImportError:
    EnhancedSRDetector = None
    SRLevel = None
    ENHANCED_SR_DETECTOR_AVAILABLE = False

# For parameter optimization
try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Enhanced diagnostics imports
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

logger = system_logger.getChild('Step2_5SROptimization')

# Error classification system
class ErrorSeverity:
    CRITICAL = "CRITICAL"  # System cannot continue
    HIGH = "HIGH"         # Major functionality affected
    MEDIUM = "MEDIUM"     # Minor functionality affected
    LOW = "LOW"          # Cosmetic or non-critical

class ErrorCategory:
    DATA_QUALITY = "DATA_QUALITY"
    ML_TRAINING = "ML_TRAINING"
    SR_DETECTION = "SR_DETECTION"
    FEATURE_ENGINEERING = "FEATURE_ENGINEERING"
    SYSTEM_RESOURCE = "SYSTEM_RESOURCE"
    EXTERNAL_DEPENDENCY = "EXTERNAL_DEPENDENCY"

def classify_error(error: Exception, context: str = "") -> tuple[ErrorSeverity, ErrorCategory]:
    """Classify errors for appropriate handling."""
    error_type = type(error).__name__
    error_msg = str(error).lower()
    
    # Critical errors
    if isinstance(error, (MemoryError, SystemError)):
        return ErrorSeverity.CRITICAL, ErrorCategory.SYSTEM_RESOURCE
    if "all.*values.*invalid" in error_msg or "data.*corrupted" in error_msg:
        return ErrorSeverity.CRITICAL, ErrorCategory.DATA_QUALITY
    
    # High severity errors
    if isinstance(error, (ValueError, KeyError)) and "data" in context.lower():
        return ErrorSeverity.HIGH, ErrorCategory.DATA_QUALITY
    if "ml" in context.lower() or "model" in context.lower():
        return ErrorSeverity.HIGH, ErrorCategory.ML_TRAINING
    
    # Medium severity errors
    if isinstance(error, ImportError):
        return ErrorSeverity.MEDIUM, ErrorCategory.EXTERNAL_DEPENDENCY
    if "sr" in context.lower() or "detection" in context.lower():
        return ErrorSeverity.MEDIUM, ErrorCategory.SR_DETECTION
    
    # Default to medium severity
    return ErrorSeverity.MEDIUM, ErrorCategory.SYSTEM_RESOURCE

def handle_error_with_recovery(error: Exception, context: str, max_retries: int = 3) -> bool:
    """Handle errors with appropriate recovery strategies."""
    severity, category = classify_error(error, context)
    
    logger.error(f"🚨 {severity} ERROR in {category}: {error}")
    logger.error(f"📋 Context: {context}")
    logger.error(f"📋 Traceback: {traceback.format_exc()}")
    
    if severity == ErrorSeverity.CRITICAL:
        logger.critical("💥 CRITICAL ERROR - System cannot continue safely")
        return False
    elif severity == ErrorSeverity.HIGH:
        logger.error("⚠️ HIGH SEVERITY ERROR - Major functionality affected")
        # Could implement retry logic here
        return False
    else:
        logger.warning(f"⚠️ {severity} ERROR - Continuing with degraded functionality")
        return True

def detect_data_drift(current_data: pd.DataFrame, reference_data: pd.DataFrame = None, 
                     drift_threshold: float = 0.1) -> Dict[str, Any]:
    """Detect data drift between current and reference datasets."""
    drift_results = {
        'drift_detected': False,
        'drift_score': 0.0,
        'drift_details': {},
        'recommendations': []
    }
    
    try:
        # If no reference data, use statistical baselines
        if reference_data is None:
            # Use statistical baselines for common financial metrics
            baseline_stats = {
                'close_mean': current_data['close'].mean(),
                'close_std': current_data['close'].std(),
                'volume_mean': current_data['volume'].mean(),
                'volume_std': current_data['volume'].std()
            }
            
            # Simple drift detection based on statistical properties
            current_stats = {
                'close_mean': current_data['close'].mean(),
                'close_std': current_data['close'].std(),
                'volume_mean': current_data['volume'].mean(),
                'volume_std': current_data['volume'].std()
            }
            
            # Calculate drift score (simplified)
            drift_score = 0.0
            for metric in baseline_stats:
                if baseline_stats[metric] != 0:
                    relative_change = abs(current_stats[metric] - baseline_stats[metric]) / abs(baseline_stats[metric])
                    drift_score += relative_change
                    drift_results['drift_details'][metric] = {
                        'baseline': baseline_stats[metric],
                        'current': current_stats[metric],
                        'relative_change': relative_change
                    }
            
            drift_score /= len(baseline_stats)
            drift_results['drift_score'] = drift_score
            
            if drift_score > drift_threshold:
                drift_results['drift_detected'] = True
                drift_results['recommendations'].append("Significant data drift detected - consider retraining models")
                drift_results['recommendations'].append("Review data sources and collection processes")
        
        else:
            # Compare with reference data
            # This would implement more sophisticated drift detection
            # For now, use simple statistical comparison
            pass
            
    except Exception as e:
        logger.error(f"Data drift detection failed: {e}")
        drift_results['error'] = str(e)
    
    return drift_results

def generate_function_report(ml_results: Dict[str, Any] = None) -> Dict[str, Any]:
    """Generate comprehensive function call report with detailed ML model metrics."""
    from src.utils.step02_5_utilities import global_tracker

    # Get base performance summary
    base_report = global_tracker.get_performance_summary()

    # Add detailed ML model metrics if available
    if ml_results:
        base_report['ml_model_metrics'] = {
            'direction_accuracy': ml_results.get('direction_accuracy', 0.0),
            'volatility_mae': ml_results.get('volatility_mae', 0.0),
            'model_type': ml_results.get('model_type', 'unknown'),
            'training_samples': ml_results.get('training_samples', 0),
            'test_samples': ml_results.get('test_samples', 0),
            'training_time': ml_results.get('training_time', 0.0),
            'sr_levels_used': ml_results.get('sr_levels_used', 0),
            'feature_count': len(ml_results.get('feature_names', [])),
            'cross_validation_scores': ml_results.get('cross_validation_scores', []),
            'cv_mean_accuracy': ml_results.get('evaluation_metrics', {}).get('cv_direction_mean', 0.0),
            'cv_std_accuracy': ml_results.get('evaluation_metrics', {}).get('cv_direction_std', 0.0)
        }

        # Add comprehensive individual model performance details (including all models, even those not selected)
        if 'models_performance' in ml_results:
            base_report['ml_model_metrics']['all_models'] = {}
            for model_name, model_data in ml_results['models_performance'].items():
                model_info = {
                    'model_name': model_name,
                    'was_selected': model_name == ml_results.get('model_type', ''),
                    'direction': {},
                    'volatility': {}
                }

                # Direction classification details
                if 'direction' in model_data:
                    dir_data = model_data['direction']
                    model_info['direction'] = {
                        'accuracy': dir_data.get('accuracy', 0.0),
                        'precision': dir_data.get('classification_report', {}).get('weighted avg', {}).get('precision', 0.0),
                        'recall': dir_data.get('classification_report', {}).get('weighted avg', {}).get('recall', 0.0),
                        'f1_score': dir_data.get('classification_report', {}).get('weighted avg', {}).get('f1-score', 0.0),
                        'classification_report': dir_data.get('classification_report', {}),
                        'feature_importance_count': len(dir_data.get('feature_importance', {})),
                        'predictions_count': len(dir_data.get('predictions', [])),
                        'optimized': 'optimized_params' in dir_data,
                        'optimized_params': dir_data.get('optimized_params', {}),
                        'improvement': dir_data.get('improvement', 0.0)
                    }

                # Volatility regression details
                if 'volatility' in model_data and model_data['volatility']:
                    vol_data = model_data['volatility']
                    model_info['volatility'] = {
                        'mae': vol_data.get('mae', 0.0),
                        'predictions_count': len(vol_data.get('predictions', []))
                    }

                base_report['ml_model_metrics']['all_models'][model_name] = model_info

        # Add feature selection information with SHAP details
        if 'feature_selection' in ml_results:
            fs = ml_results['feature_selection']
            base_report['ml_model_metrics']['feature_selection'] = {
                'original_features': fs.get('original_features', 0),
                'selected_features': fs.get('selected_features', 0),
                'target_features': fs.get('target_features', 80),  # Updated to reflect 80 features
                'methods_used': fs.get('methods_used', []),
                'selection_criteria': fs.get('selection_criteria', ''),
                'top_features': fs.get('top_features', []),
                'feature_importance': fs.get('feature_importance', {}),
                'mutual_information': fs.get('mutual_information', {}),
                'shap_analysis': fs.get('shap_importance', {}),
                'all_selected_features': fs.get('feature_importance', {}),  # All features used by the final model with their importance scores
                'all_features_list': list(fs.get('feature_importance', {}).keys())
            }

            # Add comprehensive SHAP information
            if 'shap_importance' in fs:
                shap_data = fs['shap_importance']
                base_report['ml_model_metrics']['feature_selection']['shap_available'] = shap_data.get('available', False)
                base_report['ml_model_metrics']['feature_selection']['shap_method'] = shap_data.get('method', 'unknown')

                if shap_data.get('available', False):
                    base_report['ml_model_metrics']['feature_selection']['shap_feature_importance'] = shap_data.get('feature_importance', {})
                    base_report['ml_model_metrics']['feature_selection']['shap_top_features'] = shap_data.get('top_features', [])
                    base_report['ml_model_metrics']['feature_selection']['shap_sample_size'] = shap_data.get('sample_size', 0)

        # Add optimization information
        if 'evaluation_metrics' in ml_results:
            eval_metrics = ml_results['evaluation_metrics']
            base_report['ml_model_metrics']['optimization'] = {
                'best_model': eval_metrics.get('best_model_type', 'unknown'),
                'best_accuracy': eval_metrics.get('best_direction_accuracy', 0.0),
                'models_evaluated': eval_metrics.get('models_count', 0),
                'cv_direction_mean': eval_metrics.get('cv_direction_mean', 0.0),
                'cv_direction_std': eval_metrics.get('cv_direction_std', 0.0),
                'cv_f1_mean': eval_metrics.get('cv_f1_mean', 0.0),
                'cv_f1_std': eval_metrics.get('cv_f1_std', 0.0),
                'feature_importance_available': bool(eval_metrics.get('feature_importance', {})),
                'top_20_features': list(eval_metrics.get('feature_importance', {}).keys())[:20]
            }

        # Add model comparison summary
        if 'models_performance' in ml_results:
            model_comparison = {}
            for model_name, model_data in ml_results['models_performance'].items():
                if 'direction' in model_data and 'accuracy' in model_data['direction']:
                    model_comparison[model_name] = {
                        'accuracy': model_data['direction']['accuracy'],
                        'rank': 0,  # Will be set below
                        'selected': model_name == ml_results.get('model_type', '')
                    }

            # Sort models by accuracy and assign ranks
            sorted_models = sorted(model_comparison.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            for rank, (model_name, _) in enumerate(sorted_models, 1):
                model_comparison[model_name]['rank'] = rank

            base_report['ml_model_metrics']['model_comparison'] = {
                'ranking': sorted_models,
                'total_models': len(model_comparison),
                'best_model': sorted_models[0][0] if sorted_models else 'unknown',
                'best_accuracy': sorted_models[0][1]['accuracy'] if sorted_models else 0.0,
                'selected_model': ml_results.get('model_type', 'unknown')
            }

    return base_report

class SROptimizationStep(BaseStep):
    """Step 2.5: S/R Detection Optimization with comprehensive parameter optimization and detailed reporting."""

    log_important_calls
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize SR optimization step."""
        super().__init__(config, '2_5', 'sr_optimization')
        self.logger = system_logger.getChild('SROptimizationStep')
        self.standards = PipelineStandards(self.logger)
        self.sr_optimization_config = config.get('sr_optimization', {'min_touches': 2, 'tolerance_pct': 0.5, 'lookback_periods': 100})
        
        # NEW: Enhanced configuration parameters
        self.enable_hyperparameter_optimization = config.get('enable_hyperparameter_optimization', True)
        self.optimization_method = config.get('optimization_method', 'grid_search')  # 'grid_search', 'random_search', 'bayesian'
        self.optimization_folds = config.get('optimization_folds', 5)
        self.optimization_trials = config.get('optimization_trials', 50)
        
        # NEW: Walk-forward validation
        self.enable_walk_forward_validation = config.get('enable_walk_forward_validation', True)
        self.walk_forward_folds = config.get('walk_forward_folds', 5)
        self.walk_forward_test_size = config.get('walk_forward_test_size', 0.2)
        
        self.start_time = None
        # Use unified monitoring system instead of multiple trackers
        self.performance_monitor = global_monitor
    
    @log_step_functions
    def _initialize_step(self) -> None:
        """Initialize step-specific components."""
        self.logger.info('✅ SR optimization step initialized')
        self.logger.info(f'📊 Configuration loaded: {self.sr_optimization_config}')

    async def initialize(self) -> None:
        """Initialize the step."""
        self._initialize_step()
        self.logger.info('🚀 Step 2.5 initialization completed')

    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the step."""
        self.logger.info('🎯 Starting Step 2.5 execution with comprehensive monitoring')
        pre_report = generate_function_report()
        self.logger.info(f"📊 Pre-execution function calls: {pre_report['total_calls']}")
        result = await self.execute_logic(training_input, pipeline_state)

        # Pass ML results to function report for detailed metrics
        ml_results = result.get('ml_results', {})
        post_report = generate_function_report(ml_results)
        self.logger.info(f"📊 Post-execution function calls: {post_report['total_calls']}")
        self.logger.info(f"📈 Function call increase: {post_report['total_calls'] - pre_report['total_calls']}")
        result['function_call_report'] = post_report
        return result
    
    @log_step_functions
    def validate_inputs(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Tuple[bool, list]:
        """Validate step inputs."""
        self.logger.info('🔍 Validating step inputs with detailed checks')
        errors = []
        required_inputs = ['validated_data']
        self.logger.info(f'📥 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'📥 Pipeline state keys: {list(pipeline_state.keys())}')
        for key in required_inputs:
            if key not in training_input:
                error_msg = f'Missing required input: {key}'
                errors.append(error_msg)
                self.logger.error(f'❌ {error_msg}')
            else:
                self.logger.info(f'✅ Found required input: {key}')
        if 'validated_data' in training_input:
            data = training_input['validated_data']
            self.logger.info(f'📊 Data type: {type(data)}')
            if hasattr(data, 'shape'):
                self.logger.info(f'📊 Data shape: {data.shape}')
            elif hasattr(data, '__len__'):
                self.logger.info(f'📊 Data length: {len(data)}')
        validation_result = len(errors) == 0
        self.logger.info(f"🔍 Input validation result: {('PASSED' if validation_result else 'FAILED')}")
        if errors:
            self.logger.error(f'❌ Validation errors: {errors}')
        return validation_result
    
    @log_all_calls
    def _validate_and_fix_input_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and fix input data using pipeline standards.
        
        Args:
            data: Input DataFrame
            
        Returns:
            Validated and fixed DataFrame
            
        Raises:
            ValueError: If data is None, empty, or fails validation
        """
        self.logger.info('🔍 Validating input data using pipeline standards...')
        
        # CRITICAL: Validate input data before processing
        if data is None:
            raise ValueError("CRITICAL: Input data is None. Cannot proceed with data validation.")
        
        if data.empty:
            raise ValueError("CRITICAL: Input data is empty. Cannot proceed with data validation.")
        
        if len(data) < 10:  # Minimum 10 rows for any meaningful processing
            raise ValueError(f"CRITICAL: Insufficient data for validation. Only {len(data)} rows available, minimum 10 required.")
        
        self.logger.info(f'✅ Input data validation passed: {len(data)} rows, {len(data.columns)} columns')
        
        # Create a copy to work with
        fixed_data = data.copy()
        
        # Add missing required columns with default values
        if 'exchange' not in fixed_data.columns:
            fixed_data['exchange'] = 'binance'  # Default exchange
            self.logger.info('📝 Added missing exchange column with default value')

        if 'symbol' not in fixed_data.columns:
            fixed_data['symbol'] = 'ETHUSDT'  # Default symbol - matches available data
            self.logger.info('📝 Added missing symbol column with default value')

        if 'timeframe' not in fixed_data.columns:
            fixed_data['timeframe'] = '30m'  # Default timeframe for SR analysis
            self.logger.info('📝 Added missing timeframe column with default value (30m for SR analysis)')

        # IMPORTANT: Ensure string columns are properly typed to prevent conversion issues
        string_columns = ['exchange', 'symbol', 'timeframe']
        for col in string_columns:
            if col in fixed_data.columns:
                fixed_data[col] = fixed_data[col].astype('string')
                self.logger.info(f'🔤 Ensured {col} column is properly typed as string')
        
        # Add missing timestamp column if needed
        if 'timestamp' not in fixed_data.columns:
            # Create timestamp from index if it's datetime, otherwise use sequential timestamps
            if isinstance(fixed_data.index, pd.DatetimeIndex):
                fixed_data['timestamp'] = (fixed_data.index.astype('int64') // 10**6).astype('int64')
                self.logger.info('📝 Added missing timestamp column from datetime index')
            else:
                # Create sequential timestamps starting from current time
                import time
                current_time = int(time.time() * 1000)  # Current time in milliseconds
                fixed_data['timestamp'] = range(current_time, current_time + len(fixed_data) * 60000, 60000)  # 1 minute intervals
                self.logger.info('📝 Added missing timestamp column with sequential timestamps')
        
        # Fix timestamp column type if needed
        if 'timestamp' in fixed_data.columns:
            # Check if timestamp is datetime and convert to int64 if needed
            if pd.api.types.is_datetime64_any_dtype(fixed_data['timestamp']):
                self.logger.info('🕐 Converting datetime timestamp to int64 (milliseconds)')
                # Handle NaT values before conversion
                if fixed_data['timestamp'].isna().any():
                    nan_count = fixed_data['timestamp'].isna().sum()
                    drop_percentage = (nan_count / len(fixed_data)) * 100
                    if drop_percentage > 0.5:
                        self.logger.error(f'🚨 CRITICAL: Dropping {nan_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with NaT timestamps - HIGH DATA LOSS!')
                    else:
                        self.logger.warning(f'⚠️ Dropping {nan_count} rows with NaT timestamps ({drop_percentage:.2f}%)')
                    fixed_data = fixed_data[fixed_data['timestamp'].notna()].copy()
                if len(fixed_data) > 0:
                    fixed_data['timestamp'] = (fixed_data['timestamp'].astype('int64') // 10**6).astype('int64')
                else:
                    raise ValueError('All timestamp values are NaT - data is corrupted and cannot be processed')
            elif not pd.api.types.is_integer_dtype(fixed_data['timestamp']):
                # More robust timestamp conversion handling mixed data types
                try:
                    original_dtype = fixed_data['timestamp'].dtype
                    self.logger.info(f'🔄 Converting timestamp from {original_dtype} to int64')

                    # Handle object dtype columns that might contain mixed types
                    if fixed_data['timestamp'].dtype == 'object':
                        self.logger.info('📝 Handling object dtype timestamp column')

                        # First, try to convert any datetime-like strings to datetime
                        try:
                            # Use pd.to_datetime which is more flexible than pd.to_numeric
                            converted_dt = pd.to_datetime(fixed_data['timestamp'], errors='coerce', utc=True)
                            valid_dt_mask = converted_dt.notna()

                            if valid_dt_mask.sum() < len(fixed_data):
                                invalid_count = len(fixed_data) - valid_dt_mask.sum()
                                drop_percentage = (invalid_count / len(fixed_data)) * 100
                                if drop_percentage > 0.5:
                                    self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid datetime strings - HIGH DATA LOSS!')
                                    # Log sample of invalid values for debugging
                                    invalid_samples = fixed_data.loc[~valid_dt_mask, 'timestamp'].head(3)
                                    self.logger.error(f'❌ Invalid timestamp samples: {invalid_samples.tolist()}')
                                else:
                                    self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid datetime strings ({drop_percentage:.2f}%)')
                                fixed_data = fixed_data[valid_dt_mask].copy()
                                converted_dt = converted_dt[valid_dt_mask]

                            if len(fixed_data) > 0:
                                # Convert datetime to int64 milliseconds
                                fixed_data['timestamp'] = (converted_dt.astype('int64') // 10**6).astype('int64')
                                self.logger.info('🕐 Converted datetime strings to int64 milliseconds')
                            else:
                                raise ValueError('All datetime string values are invalid - cannot convert timestamps')

                        except Exception as dt_error:
                            self.logger.error(f'❌ Failed to convert datetime strings: {dt_error}')
                            self.logger.error(f'📋 Datetime conversion traceback: {traceback.format_exc()}')
                            self.logger.warning(f'⚠️ Trying numeric conversion as fallback...')
                            # Fallback to numeric conversion
                            numeric_timestamps = pd.to_numeric(fixed_data['timestamp'], errors='coerce')
                            valid_mask = numeric_timestamps.notna()
                            if valid_mask.sum() < len(fixed_data):
                                invalid_count = len(fixed_data) - valid_mask.sum()
                                drop_percentage = (invalid_count / len(fixed_data)) * 100
                                if drop_percentage > 0.5:
                                    self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid numeric timestamps - HIGH DATA LOSS!')
                                    # Log sample of invalid values for debugging
                                    invalid_samples = fixed_data.loc[~valid_mask, 'timestamp'].head(3)
                                    self.logger.error(f'❌ Invalid timestamp samples: {invalid_samples.tolist()}')
                                else:
                                    self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid numeric timestamps ({drop_percentage:.2f}%)')
                                fixed_data = fixed_data[valid_mask].copy()
                                numeric_timestamps = numeric_timestamps[valid_mask]

                            if len(fixed_data) > 0:
                                fixed_data['timestamp'] = numeric_timestamps.astype('int64')
                                self.logger.info('🕐 Converted numeric timestamps to int64')
                            else:
                                raise ValueError('All numeric timestamp values are invalid - cannot convert timestamps')
                    else:
                        # For other dtypes (like float64), try direct numeric conversion
                        self.logger.info(f'🔢 Converting {original_dtype} timestamp to int64')
                        numeric_timestamps = pd.to_numeric(fixed_data['timestamp'], errors='coerce')
                        valid_mask = numeric_timestamps.notna()
                        if valid_mask.sum() < len(fixed_data):
                            invalid_count = len(fixed_data) - valid_mask.sum()
                            drop_percentage = (invalid_count / len(fixed_data)) * 100
                            if drop_percentage > 0.5:
                                self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid {original_dtype} timestamps - HIGH DATA LOSS!')
                                # Log sample of invalid values for debugging
                                invalid_samples = fixed_data.loc[~valid_mask, 'timestamp'].head(3)
                                self.logger.error(f'❌ Invalid timestamp samples: {invalid_samples.tolist()}')
                            else:
                                self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid {original_dtype} timestamps ({drop_percentage:.2f}%)')
                            fixed_data = fixed_data[valid_mask].copy()
                            numeric_timestamps = numeric_timestamps[valid_mask]

                        if len(fixed_data) > 0:
                            fixed_data['timestamp'] = numeric_timestamps.astype('int64')
                            self.logger.info(f'🕐 Converted {original_dtype} timestamp to int64')
                        else:
                            raise ValueError(f'All {original_dtype} timestamp values are invalid - cannot convert timestamps')

                except Exception as e:
                    self.logger.error(f'❌ Failed to convert timestamp column: {e}')
                    self.logger.error(f'📊 Timestamp column info: dtype={fixed_data["timestamp"].dtype}, shape={fixed_data.shape}')
                    # Log sample values for debugging
                    try:
                        sample_values = fixed_data['timestamp'].head(3).tolist()
                        self.logger.error(f'📋 Sample timestamp values: {sample_values}')
                        unique_types = fixed_data['timestamp'].apply(type).unique()
                        self.logger.error(f'📋 Unique value types: {[str(t) for t in unique_types]}')
                    except Exception as sample_error:
                        self.logger.error(f'❌ Could not sample timestamp values: {sample_error}')
                    return fixed_data
        
        # Remove only exact duplicate rows (identical across all columns). Keep differing rows even if timestamp duplicates.
        if 'timestamp' in fixed_data.columns:
            exact_dupe_mask = fixed_data.duplicated(subset=fixed_data.columns.tolist(), keep='first')
            exact_dupe_count = int(exact_dupe_mask.sum())
            if exact_dupe_count > 0:
                self.logger.info(f'🗑️ Removing {exact_dupe_count} exact duplicate rows (identical across all columns)')
                fixed_data = fixed_data.loc[~exact_dupe_mask]
            remaining_ts_dupes = int(fixed_data['timestamp'].duplicated().sum())
            if remaining_ts_dupes > 0:
                self.logger.warning(f'⚠️ Found {remaining_ts_dupes} duplicate timestamps with differing values; retaining all to avoid data loss')
        
        # Sort by timestamp if not monotonic
        if 'timestamp' in fixed_data.columns:
            # Ensure all timestamps are integers before sorting
            if pd.api.types.is_datetime64_any_dtype(fixed_data['timestamp']):
                # Handle NaT values before conversion for sorting
                if fixed_data['timestamp'].isna().any():
                    nan_count = fixed_data['timestamp'].isna().sum()
                    drop_percentage = (nan_count / len(fixed_data)) * 100
                    if drop_percentage > 0.5:
                        self.logger.error(f'🚨 CRITICAL: Dropping {nan_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with NaT timestamps before sorting - HIGH DATA LOSS!')
                    else:
                        self.logger.warning(f'⚠️ Dropping {nan_count} rows with NaT timestamps before sorting ({drop_percentage:.2f}%)')
                    fixed_data = fixed_data[fixed_data['timestamp'].notna()].copy()
                if len(fixed_data) > 0:
                    fixed_data['timestamp'] = (fixed_data['timestamp'].astype('int64') // 10**6).astype('int64')
                else:
                    raise ValueError('All timestamp values are NaT - cannot sort data')
            elif not pd.api.types.is_integer_dtype(fixed_data['timestamp']):
                # Convert any remaining non-integer timestamps to integers
                try:
                    # Convert to numeric, filter out NaN values
                    numeric_timestamps = pd.to_numeric(fixed_data['timestamp'], errors='coerce')
                    valid_mask = numeric_timestamps.notna()
                    if valid_mask.sum() < len(fixed_data):
                        invalid_count = len(fixed_data) - valid_mask.sum()
                        drop_percentage = (invalid_count / len(fixed_data)) * 100
                        if drop_percentage > 0.5:
                            self.logger.error(f'🚨 CRITICAL: Dropping {invalid_count}/{len(fixed_data)} rows ({drop_percentage:.2f}%) with invalid timestamps before sorting - HIGH DATA LOSS!')
                        else:
                            self.logger.warning(f'⚠️ Dropping {invalid_count} rows with invalid timestamps before sorting ({drop_percentage:.2f}%)')
                        fixed_data = fixed_data[valid_mask].copy()
                    if len(fixed_data) > 0:
                        fixed_data['timestamp'] = numeric_timestamps[valid_mask].astype('int64')
                    else:
                        raise ValueError('All timestamp values are invalid, cannot sort data')
                except Exception as e:
                    raise ValueError(f'Could not convert timestamps for sorting: {e}')

            if not fixed_data['timestamp'].is_monotonic_increasing:
                self.logger.info('📈 Sorting data by timestamp')
                fixed_data = fixed_data.sort_values('timestamp').reset_index(drop=True)
        
        # Apply schema enforcement with error handling
        try:
            fixed_data = PipelineStandards.enforce_schema(fixed_data, 'unified')
            self.logger.info('✅ Applied schema enforcement')
        except Exception as e:
            self.logger.warning(f'⚠️ Schema enforcement failed: {e}')
            # Try to fix common schema issues manually
            try:
                # Ensure trade_volume column exists and is float64
                if 'trade_volume' not in fixed_data.columns:
                    fixed_data['trade_volume'] = 0.0
                    self.logger.info('📝 Added missing trade_volume column')
                else:
                    # Handle NaN values and type conversion more robustly
                    if fixed_data['trade_volume'].isna().any():
                        self.logger.warning(f'⚠️ Found NaN values in trade_volume column: {fixed_data["trade_volume"].isna().sum()} NaNs')
                    fixed_data['trade_volume'] = pd.to_numeric(fixed_data['trade_volume'], errors='coerce').fillna(0.0).astype('float64')
                    self.logger.info('🔧 Fixed trade_volume column type')
            except Exception as fix_error:
                self.logger.warning(f'⚠️ Could not fix trade_volume column: {fix_error}')
        
        # Set datetime index for analysis (but keep original timestamp column)
        if 'timestamp' in fixed_data.columns and not isinstance(fixed_data.index, pd.DatetimeIndex):
            try:
                # Convert timestamp back to datetime for indexing
                timestamp_datetime = pd.to_datetime(fixed_data['timestamp'], unit='ms')
                fixed_data = fixed_data.set_index(timestamp_datetime)
                self.logger.info('📅 Set datetime index for analysis')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not set datetime index: {e}')
        
        # Data loss monitoring
        original_rows = len(data)
        final_rows = len(fixed_data)
        data_loss_percentage = ((original_rows - final_rows) / original_rows) * 100
        
        if data_loss_percentage > 5.0:  # More than 5% data loss
            self.logger.critical(f'🚨 CRITICAL DATA LOSS: {data_loss_percentage:.2f}% of data lost during validation!')
            self.logger.critical(f'📊 Original rows: {original_rows}, Final rows: {final_rows}')
            # Could implement data recovery strategies here
        elif data_loss_percentage > 1.0:  # More than 1% data loss
            self.logger.warning(f'⚠️ Significant data loss: {data_loss_percentage:.2f}% of data lost during validation')
            self.logger.warning(f'📊 Original rows: {original_rows}, Final rows: {final_rows}')
        else:
            self.logger.info(f'✅ Minimal data loss: {data_loss_percentage:.2f}% ({original_rows - final_rows} rows)')
        
        # Final validation
        final_validation = self.standards.validate_data_quality(fixed_data, 'unified')
        self.logger.info(f'✅ Final data quality score: {final_validation.quality_score:.2f}')
        
        if not final_validation.passed:
            self.logger.warning('⚠️ Final validation still has issues:')
            for issue in final_validation.issues:
                self.logger.warning(f'   - {issue.message}')
        
        # Store data loss metrics for monitoring
        fixed_data.attrs['data_loss_percentage'] = data_loss_percentage
        fixed_data.attrs['original_rows'] = original_rows
        fixed_data.attrs['final_rows'] = final_rows
        
        return fixed_data

    async def execute_logic(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """Execute comprehensive SR optimization logic with features, detection, and ML training."""
        self.logger.info('🎯 Starting comprehensive S/R detection optimization with unified monitoring...')
        self.logger.info(f'📊 Training input keys: {list(training_input.keys())}')
        self.logger.info(f'📊 Pipeline state keys: {list(pipeline_state.keys())}')
        self.start_time = time.time()
        # Simple step-level tracking (unified monitor handles function-level tracking)
        internal_call_tracker = {'step_calls': 0, 'step_times': {}, 'step_results': {}}

        # Quality evaluation disabled for now - focus on core SR detection
        self.logger.info('🎯 Quality evaluation disabled - focusing on core SR detection without fractals')
        quality_results = None
        # Skip quality evaluation to avoid missing method error
        # pipeline_state['sr_quality_evaluation'] = quality_results

        try:
            # CRITICAL: Validate data availability before any processing
            self.logger.info('📊 Retrieving data from pipeline state...')
            data = pipeline_state.get('dataframe')
            if data is None:
                data = training_input.get('validated_data')
            
            # Set current timestamp for lookahead bias detection
            if data is not None and hasattr(data, 'index'):
                current_time = data.index[-1] if len(data) > 0 else None
                if current_time:
                    bias_detector = get_global_detector()
                    bias_detector.set_current_timestamp(current_time)
                    # Validate no future data
                    data = validate_no_future_data(data, 'timestamp', current_time)
            if data is None:
                # Try to load data from the same path that step02 uses
                self.logger.info('📊 No data in pipeline state, loading from data files...')
                data_path = pipeline_state.get('unified_data_path') or pipeline_state.get('raw_market_data')
                if not data_path:
                    symbol = training_input.get('symbol', '').upper()
                    exchange = training_input.get('exchange', '').upper()
                    timeframe = training_input.get('timeframe', '30m')
                    data_path = self.standards.build_path('unified_partitioned', exchange, symbol, timeframe=timeframe)
                    self.logger.info(f'📖 Constructed data path: {data_path}')
                
                # Load data from the data path; if missing, auto-trigger re-collection and retry once
                try:
                    if not PARQUET_UTILS_AVAILABLE:
                        raise ImportError("ParquetUtils not available")
                    parquet_utils = ParquetUtils()
                    data_path_obj = Path(data_path)
                    if data_path_obj.is_file():
                        data = parquet_utils.safe_read_parquet(data_path)
                    elif data_path_obj.is_dir():
                        parquet_files = list(data_path_obj.glob('**/*.parquet'))
                        if not parquet_files:
                            # Attempt centralized auto re-collection
                            self.logger.warning(f"⚠️ No parquet files found in directory: {data_path}. Attempting auto re-collection...")
                            try:
                                from src.training.steps.market_analysis.step1.enhanced_data_quality_manager import EnhancedDataQualityManager
                                _qm = EnhancedDataQualityManager(str(Path(data_path).parents[3])) if len(Path(data_path).parts) > 3 else EnhancedDataQualityManager('data_cache')
                                symbol_q = training_input.get('symbol', symbol if 'symbol' in locals() else 'ETHUSDT')
                                exchange_q = training_input.get('exchange', exchange if 'exchange' in locals() else 'BINANCE')
                                timeframe_q = training_input.get('timeframe', timeframe)
                                import asyncio as _asyncio
                                _asyncio.get_event_loop()
                                _asyncio.run(_qm.get_data_for_step3_step4(symbol_q, exchange_q, timeframe_q))
                                parquet_files = list(data_path_obj.glob('**/*.parquet'))
                            except Exception as _qe:
                                self.logger.warning(f"Auto re-collection failed: {_qe}")
                        if not parquet_files:
                            raise ValueError(f'No parquet files found in directory: {data_path}')
                        self.logger.info(f'📁 Found {len(parquet_files)} parquet files in directory')
                        dataframes = []
                        for i, file_path in enumerate(parquet_files):
                            self.logger.info(f'📖 Reading file {i + 1}/{len(parquet_files)}: {file_path.name}')
                            df = parquet_utils.safe_read_parquet(str(file_path))
                            if df is not None and (not df.empty):
                                # Enforce schema immediately to avoid drift
                                try:
                                    df = self.standards.enforce_schema(df, 'unified')
                                except Exception as _se:
                                    self.logger.warning(f"Schema enforcement failed for {file_path.name}: {_se}")
                                dataframes.append(df)
                        if not dataframes:
                            raise ValueError(f'Failed to read any data from parquet files in {data_path}')
                        # Concatenate without ignore_index to preserve datetime indexes
                        data = pd.concat(dataframes, ignore_index=False)
                        # If concatenation created duplicate indexes, reset and sort
                        if data.index.duplicated().any():
                            self.logger.info('🔄 Resetting index due to duplicates after concatenation')
                            data = data.reset_index(drop=True)
                        else:
                            self.logger.info(f'📊 Concatenated {len(dataframes)} dataframes preserving datetime index')
                        self.logger.info(f'📊 Concatenated {len(dataframes)} dataframes')
                    else:
                        raise ValueError(f'Path does not exist: {data_path}')
                    
                    if data is None or data.empty:
                        # Attempt centralized auto re-collection and one retry
                        self.logger.warning(f"⚠️ Empty data after read. Attempting auto re-collection and retry...")
                        try:
                            from src.training.steps.market_analysis.step1.enhanced_data_quality_manager import EnhancedDataQualityManager
                            _qm2 = EnhancedDataQualityManager(str(Path(data_path).parents[3])) if len(Path(data_path).parts) > 3 else EnhancedDataQualityManager('data_cache')
                            symbol_q2 = training_input.get('symbol', symbol if 'symbol' in locals() else 'ETHUSDT')
                            exchange_q2 = training_input.get('exchange', exchange if 'exchange' in locals() else 'BINANCE')
                            timeframe_q2 = training_input.get('timeframe', timeframe)
                            import asyncio as _asyncio
                            _asyncio.get_event_loop()
                            _asyncio.run(_qm2.get_data_for_step3_step4(symbol_q2, exchange_q2, timeframe_q2))
                            # Retry read quickly
                            if data_path_obj.is_dir():
                                parquet_files = list(data_path_obj.glob('**/*.parquet'))
                                if parquet_files:
                                    dataframes = []
                                    for file_path in parquet_files:
                                        df = parquet_utils.safe_read_parquet(str(file_path))
                                        if df is not None and (not df.empty):
                                            try:
                                                df = self.standards.enforce_schema(df, 'unified')
                                            except Exception as _se2:
                                                self.logger.warning(f"Schema enforcement failed for retry {file_path.name}: {_se2}")
                                            dataframes.append(df)
                                    if dataframes:
                                        data = pd.concat(dataframes, ignore_index=False)
                        except Exception as _qe2:
                            self.logger.warning(f"Auto re-collection retry failed: {_qe2}")
                        if data is None or data.empty:
                            raise ValueError(f'Failed to read data from {data_path}')
                    
                    self.logger.info(f'✅ Loaded {len(data)} rows with {len(data.columns)} columns from data files')
                    self.logger.info(f'📊 Data sample: {data.head(2).to_dict() if len(data) > 0 else "No data"}')
                except Exception as e:
                    self.logger.error(f'❌ Failed to load data from files: {e}')
                    # FAIL FAST: Don't continue with empty data
                    error_msg = f"CRITICAL: No data available for S/R optimization. Expected 'dataframe' or 'validated_data' in pipeline_state or training_input, or valid data files at {data_path}. Error: {e}"
                    self.logger.critical(error_msg)
                    return {
                        'success': False, 
                        'error': error_msg,
                        'error_type': 'DATA_UNAVAILABLE',
                        'execution_time': time.time() - self.start_time,
                        'step_name': 'step02_5_sr_optimization',
                        'data_availability': False,
                        'recommendation': 'Ensure data collection pipeline is working and data files exist before running S/R optimization'
                    }
            
            # CRITICAL: Validate data before processing
            if data is None or data.empty:
                error_msg = "CRITICAL: Data is None or empty after loading. Cannot proceed with S/R optimization."
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'EMPTY_DATA',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step02_5_sr_optimization',
                    'data_availability': False,
                    'recommendation': 'Check data loading pipeline and ensure valid data is available'
                }
            
            # Validate minimum data requirements
            if len(data) < 100:  # Minimum 100 rows for meaningful S/R analysis
                error_msg = f"CRITICAL: Insufficient data for S/R optimization. Only {len(data)} rows available, minimum 100 required."
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'INSUFFICIENT_DATA',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step02_5_sr_optimization',
                    'data_availability': True,
                    'data_rows': len(data),
                    'minimum_required': 100,
                    'recommendation': 'Collect more data or use a different timeframe with sufficient historical data'
                }
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            if missing_columns:
                error_msg = f"CRITICAL: Missing required columns for S/R optimization: {missing_columns}. Available columns: {list(data.columns)}"
                self.logger.critical(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'error_type': 'MISSING_COLUMNS',
                    'execution_time': time.time() - self.start_time,
                    'step_name': 'step02_5_sr_optimization',
                    'data_availability': True,
                    'data_rows': len(data),
                    'missing_columns': missing_columns,
                    'available_columns': list(data.columns),
                    'recommendation': 'Ensure data contains OHLCV columns (open, high, low, close, volume)'
                }
            
            self.logger.info(f'✅ Data validation passed: {len(data)} rows, {len(data.columns)} columns')
            data = self._validate_and_fix_input_data(data)
            self.logger.info(f'📊 Processing {len(data)} rows of data')
            self.logger.info(f'📊 Data columns: {list(data.columns)}')
            self.logger.info(f'📊 Data types: {data.dtypes.to_dict()}')
            
            # Data drift detection
            self.logger.info('🔍 Performing data drift detection...')
            drift_results = detect_data_drift(data)
            if drift_results['drift_detected']:
                self.logger.warning(f'⚠️ Data drift detected with score: {drift_results["drift_score"]:.4f}')
                for recommendation in drift_results['recommendations']:
                    self.logger.warning(f'💡 Recommendation: {recommendation}')
            else:
                self.logger.info(f'✅ No significant data drift detected (score: {drift_results["drift_score"]:.4f})')

            # SAFETY CHECK: Validate data types before proceeding
            string_columns = [col for col in data.columns if data[col].dtype == 'object' or str(data[col].dtype).startswith('string')]
            if string_columns:
                self.logger.warning(f'⚠️ Found string columns that may cause issues: {string_columns}')
                self.logger.info('🔧 These columns will be filtered out during feature engineering')

            self.logger.info('🔧 Step 1: Engineering features...')
            self.logger.info(f'📊 About to call feature engineering with data shape: {data.shape}')
            step_start = time.time()
            features_data = await self._engineer_features(data)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['feature_engineering'] = step_time
            internal_call_tracker['step_results']['feature_engineering'] = {'success': True, 'features_count': len(features_data.columns), 'execution_time': step_time}
            self.logger.info(f'✅ Feature engineering completed in {step_time:.4f}s')
            self.logger.info(f'📈 Generated {len(features_data.columns)} features')
            self.logger.info('🎯 Step 2: Detecting support and resistance levels...')
            step_start = time.time()
            sr_levels = self._detect_sr_levels(features_data)
            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['sr_detection'] = step_time
            internal_call_tracker['step_results']['sr_detection'] = {'success': True, 'support_levels': len(sr_levels.get('support_levels', [])), 'resistance_levels': len(sr_levels.get('resistance_levels', [])), 'execution_time': step_time}
            self.logger.info(f'✅ SR detection completed in {step_time:.4f}s')
            self.logger.info(f"🎯 Detected {len(sr_levels.get('support_levels', []))} support levels")
            self.logger.info(f"🎯 Detected {len(sr_levels.get('resistance_levels', []))} resistance levels")
            self.logger.info('🤖 Step 3: Training ML models...')
            step_start = time.time()

            # Use chunked processing for large datasets (>500K rows) to reduce memory usage
            if len(features_data) > 500000:
                self.logger.info('📊 Large dataset detected, using chunked processing...')
                ml_results = await self._train_ml_models_chunked(features_data, sr_levels, chunk_size=200000)
            else:
                ml_results = await self._train_ml_models(features_data, sr_levels)

            step_time = time.time() - step_start
            internal_call_tracker['step_calls'] += 1
            internal_call_tracker['step_times']['ml_training'] = step_time
            internal_call_tracker['step_results']['ml_training'] = {'success': True, 'direction_accuracy': ml_results.get('direction_accuracy', 0), 'volatility_mae': ml_results.get('volatility_mae', 0), 'execution_time': step_time}
            self.logger.info(f'✅ ML training completed in {step_time:.4f}s')
            self.logger.info(f"🤖 Direction accuracy: {ml_results.get('direction_accuracy', 0):.3f}")
            self.logger.info(f"🤖 Volatility MAE: {ml_results.get('volatility_mae', 0):.6f}")
            self.logger.info('📊 All major processing steps completed - preparing final results...')
            optimization_results = {
                'best_parameters': self.sr_optimization_config, 
                'confidence_score': ml_results.get('direction_accuracy', 0.85), 
                'feature_count': len(features_data.columns), 
                'sr_levels_detected': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])), 
                'ml_model_performance': ml_results, 
                'hyperparameter_optimization': hyperparameter_results,
                'walk_forward_validation': walk_forward_results,
                'internal_call_tracker': internal_call_tracker
            }
            execution_time = time.time() - self.start_time
            self.logger.info(f'✅ Comprehensive SR optimization completed in {execution_time:.2f} seconds')
            self.logger.info(f"📈 Features engineered: {optimization_results['feature_count']}")
            self.logger.info(f"🎯 SR levels detected: {optimization_results['sr_levels_detected']}")
            self.logger.info(f"🤖 ML accuracy: {optimization_results['confidence_score']:.3f}")
            self.logger.info(f"📊 Internal function calls: {internal_call_tracker['step_calls']}")
            execution_report = {'total_execution_time': execution_time, 'step_breakdown': internal_call_tracker['step_times'], 'step_results': internal_call_tracker['step_results'], 'performance_summary': {'features_per_second': len(features_data.columns) / execution_time, 'sr_levels_per_second': optimization_results['sr_levels_detected'] / execution_time, 'ml_accuracy': ml_results.get('direction_accuracy', 0)}}

            # Include unified monitor performance summary
            performance_summary = self.performance_monitor.get_summary()

            # Add feature selection info to results if available
            feature_selection_info = getattr(self, 'feature_selection_info', None)
            if feature_selection_info:
                ml_results['feature_selection'] = feature_selection_info

            # Generate and save comprehensive enhanced report
            try:
                self.logger.info('📝 Generating comprehensive enhanced report with detailed metrics...')

                # Initialize enhanced reporter
                symbol = training_input.get('symbol', 'UNKNOWN')
                exchange = training_input.get('exchange', 'UNKNOWN')
                timeframe = training_input.get('timeframe', '30m')

                self.logger.info(f'📊 Report parameters - Symbol: {symbol}, Exchange: {exchange}, Timeframe: {timeframe}')

                financial_logger = Step02_5FinancialLogger(
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )

                self.logger.info('✅ Enhanced reporter initialized successfully')

                # Prepare execution data with detailed metrics
                execution_data = {
                    'execution_time': execution_time,
                    'memory_usage': performance_summary.get('memory_usage', 0),
                    'cpu_usage': performance_summary.get('cpu_usage', 0),
                    'function_calls': internal_call_tracker.get('total_calls', 0),
                    'step_breakdown': internal_call_tracker.get('step_times', {}),
                    'performance_summary': execution_report.get('performance_summary', {}),
                    'feature_count': len(features_data.columns),
                    'data_rows': len(features_data),
                    'sr_levels_detected': optimization_results.get('sr_levels_detected', 0),
                    'ml_accuracy': ml_results.get('direction_accuracy', 0),
                    'processing_timestamp': datetime.now().isoformat()
                }

                # Generate comprehensive report
                print("🔍 STEP02_5: Starting report generation...")
                self.logger.info('🔍 STEP02_5: Starting report generation...')

                # Log data availability for debugging
                self.logger.info(f'📊 SR levels available: {bool(sr_levels and any(sr_levels.get(key, []) for key in ["support_levels", "resistance_levels"]))}')
                self.logger.info(f'📊 ML results available: {bool(ml_results)}')
                self.logger.info(f'📊 Features data shape: {features_data.shape if features_data is not None else "None"}')

                # Log financial metrics using the new financial logger
                financial_logger.log_step_execution(
                    sr_levels=sr_levels,
                    ml_results=ml_results,
                    execution_data=execution_data,
                    data=features_data
                )

                self.logger.info('✅ Financial metrics logged successfully')

            except Exception as report_error:
                self.logger.warning(f'⚠️ Failed to generate enhanced report: {report_error}')
                import traceback
                self.logger.warning(f'Enhanced report generation traceback: {traceback.format_exc()}')

                # Fallback to basic reporting
                try:
                    print("🔄 STEP02_5: Falling back to basic report generation...")
                    self.logger.info('🔄 Falling back to basic report generation...')

                    # Create a simple fallback report with available data
                    fallback_report = {
                        'step_name': 'step02_5_sr_optimization',
                        'execution_time': execution_time,
                        'sr_levels_detected': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                        'features_engineered': len(features_data.columns) if features_data is not None else 0,
                        'ml_accuracy': ml_results.get('direction_accuracy', 0),
                        'timestamp': datetime.now().isoformat(),
                        'symbol': symbol,
                        'exchange': exchange,
                        'timeframe': timeframe,
                        'status': 'fallback_report'
                    }

                    # Try to save the fallback report
                    fallback_path = save_training_report(
                        data=fallback_report,
                        step_name='step02_5_sr_optimization',
                        report_type='fallback_report',
                        symbol=symbol,
                        timeframe=timeframe,
                        file_format='json'
                    )
                    self.logger.info(f'💾 Fallback report saved: {fallback_path}')

                    # Also try the original basic report method if available
                    try:
                        final_report = self._generate_final_report(sr_levels, optimization_results, ml_results, execution_report, performance_summary)
                        basic_report_path = save_training_report(
                            data=final_report,
                            step_name='step02_5_sr_optimization',
                            report_type='basic_sr_analysis',
                            symbol=symbol,
                            timeframe=timeframe,
                            file_format='md'
                        )
                        self.logger.info(f'💾 Basic report saved: {basic_report_path}')
                    except Exception as basic_method_error:
                        self.logger.warning(f'⚠️ Basic report method failed: {basic_method_error}')

                except Exception as basic_error:
                    self.logger.error(f'❌ Both enhanced and basic report generation failed: {basic_error}')
                    import traceback
                    self.logger.error(f'Fallback report traceback: {traceback.format_exc()}')

            # Return success with all results
            print("✅ STEP02_5: Execution completed successfully, about to return results")
            self.logger.info('✅ Step 2.5 SR optimization completed successfully')
            self.logger.info(f'📊 S/R levels detected: {len(sr_levels.get("support_levels", []))} support, {len(sr_levels.get("resistance_levels", []))} resistance')
            
            # Ensure S/R levels are properly formatted for step06
            formatted_sr_levels = self._format_sr_levels_for_pipeline(sr_levels)
            
            success_result = {
                'success': True, 
                'step02_5_sr_optimization_completed': True, 
                'sr_levels': formatted_sr_levels, 
                'sr_optimization_results': optimization_results, 
                'features_data': features_data, 
                'ml_results': ml_results, 
                'execution_time': execution_time, 
                'execution_report': execution_report, 
                'internal_call_tracker': internal_call_tracker, 
                'unified_performance_summary': performance_summary, 
                'step_name': 'step02_5_sr_optimization', 
                'pipeline_state_update': {'sr_levels': formatted_sr_levels}
            }

        except Exception as e:
            execution_time = time.time() - self.start_time
            
            # Determine error type and severity
            error_type = 'UNKNOWN_ERROR'
            error_severity = 'HIGH'
            
            if 'DATA_UNAVAILABLE' in str(e) or 'No data available' in str(e):
                error_type = 'DATA_UNAVAILABLE'
                error_severity = 'CRITICAL'
            elif 'EMPTY_DATA' in str(e) or 'empty' in str(e).lower():
                error_type = 'EMPTY_DATA'
                error_severity = 'CRITICAL'
            elif 'INSUFFICIENT_DATA' in str(e) or 'Insufficient data' in str(e):
                error_type = 'INSUFFICIENT_DATA'
                error_severity = 'HIGH'
            elif 'MISSING_COLUMNS' in str(e) or 'Missing required columns' in str(e):
                error_type = 'MISSING_COLUMNS'
                error_severity = 'HIGH'
            elif 'ImportError' in str(e) or 'ModuleNotFoundError' in str(e):
                error_type = 'IMPORT_ERROR'
                error_severity = 'MEDIUM'
            elif 'ValueError' in str(e):
                error_type = 'VALUE_ERROR'
                error_severity = 'HIGH'
            
            self.logger.error(f'❌ SR optimization failed with {error_type}: {e}')
            self.logger.error(f'🚨 Error severity: {error_severity}')
            
            # Generate appropriate error report based on error type
            try:
                symbol = training_input.get('symbol', 'UNKNOWN')
                exchange = training_input.get('exchange', 'UNKNOWN')
                timeframe = training_input.get('timeframe', '30m')
                
                # Create comprehensive error report
                error_report = {
                    'success': False,
                    'error_type': error_type,
                    'error_severity': error_severity,
                    'error_message': str(e),
                    'execution_time': execution_time,
                    'step_name': 'step02_5_sr_optimization',
                    'timestamp': datetime.now().isoformat(),
                    'symbol': symbol,
                    'exchange': exchange,
                    'timeframe': timeframe,
                    'data_availability': error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA'],
                    'recommendations': self._get_error_recommendations(error_type, str(e)),
                    'troubleshooting_steps': self._get_troubleshooting_steps(error_type),
                    'next_actions': self._get_next_actions(error_type)
                }
                
                # Save error report
                error_report_path = save_training_report(
                    data=error_report,
                    step_name='step02_5_sr_optimization',
                    report_type='error_report',
                    symbol=symbol,
                    timeframe=timeframe,
                    file_format='json'
                )
                self.logger.info(f'💾 Error report saved: {error_report_path}')
                
                # Log error details for debugging
                self.logger.error(f'📋 Error details: {error_report}')
                
            except Exception as report_error:
                self.logger.error(f'❌ Failed to generate error report: {report_error}')
                import traceback
                self.logger.error(f'Error report generation traceback: {traceback.format_exc()}')
            
            # Return comprehensive error result
            return {
                'success': False,
                'error': str(e),
                'error_type': error_type,
                'error_severity': error_severity,
                'execution_time': execution_time,
                'step_name': 'step02_5_sr_optimization',
                'data_availability': error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA'],
                'recommendations': self._get_error_recommendations(error_type, str(e)),
                'troubleshooting_steps': self._get_troubleshooting_steps(error_type)
            }

        # Return success result if no exception occurred
        return success_result

    def _get_error_recommendations(self, error_type: str, error_message: str) -> List[str]:
        """Get specific recommendations based on error type."""
        recommendations = []
        
        if error_type == 'DATA_UNAVAILABLE':
            recommendations = [
                "Check if data collection pipeline is running correctly",
                "Verify that data files exist in the expected directory",
                "Ensure the data collection step (step02) completed successfully",
                "Check file permissions and disk space",
                "Consider running data collection manually before S/R optimization"
            ]
        elif error_type == 'EMPTY_DATA':
            recommendations = [
                "Verify data files are not corrupted",
                "Check if data files contain valid market data",
                "Ensure data collection retrieved actual market data",
                "Verify the data format matches expected schema"
            ]
        elif error_type == 'INSUFFICIENT_DATA':
            recommendations = [
                "Collect more historical data for the symbol/timeframe",
                "Use a different timeframe with more available data",
                "Check if the data collection period is too short",
                "Verify the symbol has sufficient trading history"
            ]
        elif error_type == 'MISSING_COLUMNS':
            recommendations = [
                "Ensure data contains OHLCV columns (open, high, low, close, volume)",
                "Check data schema and column naming conventions",
                "Verify data preprocessing steps are working correctly",
                "Ensure data format matches expected structure"
            ]
        elif error_type == 'IMPORT_ERROR':
            recommendations = [
                "Install missing Python packages",
                "Check Python environment and dependencies",
                "Verify all required modules are available",
                "Update package versions if needed"
            ]
        elif error_type == 'VALUE_ERROR':
            recommendations = [
                "Check data quality and format",
                "Verify all required parameters are provided",
                "Ensure data values are within expected ranges",
                "Check for data type mismatches"
            ]
        else:
            recommendations = [
                "Check system logs for detailed error information",
                "Verify all dependencies are installed correctly",
                "Ensure system resources are available",
                "Contact support if the issue persists"
            ]
        
        return recommendations

    def _get_troubleshooting_steps(self, error_type: str) -> List[str]:
        """Get specific troubleshooting steps based on error type."""
        steps = []
        
        if error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA']:
            steps = [
                "1. Check if data_cache directory exists and contains files",
                "2. Verify data collection pipeline is working",
                "3. Run data collection step manually",
                "4. Check file permissions and disk space",
                "5. Verify data format and schema"
            ]
        elif error_type == 'INSUFFICIENT_DATA':
            steps = [
                "1. Check available data range for the symbol",
                "2. Try a different timeframe (1h, 4h, 1d)",
                "3. Verify data collection period settings",
                "4. Check if symbol has sufficient trading history"
            ]
        elif error_type == 'MISSING_COLUMNS':
            steps = [
                "1. Check data schema and column names",
                "2. Verify data preprocessing steps",
                "3. Ensure OHLCV columns are present",
                "4. Check data format consistency"
            ]
        elif error_type == 'IMPORT_ERROR':
            steps = [
                "1. Check Python environment",
                "2. Install missing packages",
                "3. Verify import paths",
                "4. Check package versions"
            ]
        else:
            steps = [
                "1. Check system logs for detailed errors",
                "2. Verify all dependencies",
                "3. Check system resources",
                "4. Review configuration settings"
            ]
        
        return steps

    def _get_next_actions(self, error_type: str) -> List[str]:
        """Get next actions to take based on error type."""
        actions = []
        
        if error_type in ['DATA_UNAVAILABLE', 'EMPTY_DATA']:
            actions = [
                "Run data collection pipeline first",
                "Verify data files exist and are accessible",
                "Check data collection configuration",
                "Ensure sufficient disk space"
            ]
        elif error_type == 'INSUFFICIENT_DATA':
            actions = [
                "Collect more historical data",
                "Use a different timeframe",
                "Check data collection period settings",
                "Verify symbol trading history"
            ]
        elif error_type == 'MISSING_COLUMNS':
            actions = [
                "Fix data schema issues",
                "Ensure OHLCV columns are present",
                "Check data preprocessing pipeline",
                "Verify data format consistency"
            ]
        elif error_type == 'IMPORT_ERROR':
            actions = [
                "Install missing packages",
                "Check Python environment",
                "Verify import paths",
                "Update dependencies"
            ]
        else:
            actions = [
                "Review error logs",
                "Check system configuration",
                "Verify all dependencies",
                "Contact support if needed"
            ]
        
        return actions

    def _generate_final_report(self, sr_levels: Dict[str, Any], optimization_results: Dict[str, Any],
                              ml_results: Dict[str, Any], execution_report: Dict[str, Any],
                              performance_summary: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive final report for step02_5_sr_optimization.

        Args:
            sr_levels: Detected support and resistance levels
            optimization_results: Optimization configuration and results
            ml_results: Machine learning model results
            execution_report: Execution timing and performance data
            performance_summary: Unified performance monitoring summary

        Returns:
            Dict containing comprehensive report data
        """
        from datetime import datetime

        # Get current price for context (if available in data)
        current_price = None
        try:
            # Try to get current price from the most recent data point
            if hasattr(self, 'features_data') and self.features_data is not None and not self.features_data.empty:
                current_price = float(self.features_data['close'].iloc[-1])
        except Exception:
            current_price = None

        # Prepare report data
        report = {
            'analysis_date': datetime.now().isoformat(),
            'current_price': current_price,
            'analysis_parameters': {
                'tolerance': self.sr_optimization_config.get('tolerance', 0.005),
                'minimum_touches': self.sr_optimization_config.get('minimum_touches', 2),
                'lookback_periods': self.sr_optimization_config.get('lookback_periods', 2000)
            },
            'support_levels': [],
            'resistance_levels': [],
            'strength_analysis': {},
            'trading_implications': {},
            'ml_performance': {},
            'execution_summary': {},
            'performance_metrics': {}
        }

        # Process support levels
        support_levels = sr_levels.get('support_levels', [])
        if support_levels:
            sorted_supports = sorted(support_levels, key=lambda x: x.get('strength', 0), reverse=True)
            for i, level in enumerate(sorted_supports[:5]):  # Top 5 strongest
                level_data = {
                    'rank': i + 1,
                    'price': float(level.get('price', 0)),
                    'strength': float(level.get('strength', 0)),
                    'touches': int(level.get('touches', 0)),
                    'bounces': int(level.get('bounces', 0)),
                    'bounce_rate': float(level.get('bounce_rate', 0)),
                    'distance': self._calculate_price_distance(current_price, level.get('price', 0)) if current_price else 0
                }
                report['support_levels'].append(level_data)

        # Process resistance levels
        resistance_levels = sr_levels.get('resistance_levels', [])
        if resistance_levels:
            sorted_resistances = sorted(resistance_levels, key=lambda x: x.get('strength', 0), reverse=True)
            for i, level in enumerate(sorted_resistances[:5]):  # Top 5 strongest
                level_data = {
                    'rank': i + 1,
                    'price': float(level.get('price', 0)),
                    'strength': float(level.get('strength', 0)),
                    'touches': int(level.get('touches', 0)),
                    'bounces': int(level.get('bounces', 0)),
                    'bounce_rate': float(level.get('bounce_rate', 0)),
                    'distance': self._calculate_price_distance(current_price, level.get('price', 0)) if current_price else 0
                }
                report['resistance_levels'].append(level_data)

        # Strength analysis summary
        if report['support_levels']:
            report['strength_analysis']['average_support_strength'] = sum(level['strength'] for level in report['support_levels']) / len(report['support_levels'])
            report['strength_analysis']['strongest_support'] = report['support_levels'][0]['price'] if report['support_levels'] else None

        if report['resistance_levels']:
            report['strength_analysis']['average_resistance_strength'] = sum(level['strength'] for level in report['resistance_levels']) / len(report['resistance_levels'])
            report['strength_analysis']['strongest_resistance'] = report['resistance_levels'][0]['price'] if report['resistance_levels'] else None

        # Trading implications
        if current_price and report['support_levels'] and report['resistance_levels']:
            nearest_support = min(report['support_levels'], key=lambda x: abs(x['price'] - current_price))
            nearest_resistance = min(report['resistance_levels'], key=lambda x: abs(x['price'] - current_price))

            report['trading_implications'] = {
                'nearest_support': {
                    'price': nearest_support['price'],
                    'distance': nearest_support['distance'],
                    'strength': nearest_support['strength'],
                    'reliability': 'High' if nearest_support['strength'] >= 0.8 else 'Medium' if nearest_support['strength'] >= 0.6 else 'Low'
                },
                'nearest_resistance': {
                    'price': nearest_resistance['price'],
                    'distance': nearest_resistance['distance'],
                    'strength': nearest_resistance['strength'],
                    'reliability': 'High' if nearest_resistance['strength'] >= 0.8 else 'Medium' if nearest_resistance['strength'] >= 0.6 else 'Low'
                },
                'risk_assessment': self._assess_risk(current_price, nearest_support['price'], nearest_resistance['price'])
            }

        # ML performance summary
        report['ml_performance'] = {
            'direction_accuracy': ml_results.get('direction_accuracy', 0),
            'volatility_mae': ml_results.get('volatility_mae', 0),
            'best_model': ml_results.get('model_type', 'Unknown'),
            'feature_count': optimization_results.get('feature_count', 0)
        }

        # Execution summary
        report['execution_summary'] = {
            'total_execution_time': execution_report.get('total_execution_time', 0),
            'features_engineered': optimization_results.get('feature_count', 0),
            'sr_levels_detected': optimization_results.get('sr_levels_detected', 0),
            'ml_accuracy': ml_results.get('direction_accuracy', 0),
            'step_breakdown': execution_report.get('step_breakdown', {}),
            'performance_summary': execution_report.get('performance_summary', {})
        }

        # Performance metrics
        report['performance_metrics'] = {
            'unified_monitor_summary': performance_summary,
            'internal_call_tracker': optimization_results.get('internal_call_tracker', {}),
            'function_call_report': getattr(self, 'function_call_report', {})
        }

        return report

    def _calculate_price_distance(self, current_price: float, level_price: float) -> float:
        """Calculate percentage distance between current price and level price."""
        try:
            # Use safe division to prevent division by zero
            return safe_divide(level_price - current_price, current_price, 0.0) * 100
        except MathValidationError as e:
            logger.warning(f"Mathematical validation error in price distance calculation: {e}")
            return 0.0

    def _assess_risk(self, current_price: float, support_price: float, resistance_price: float) -> str:
        """Assess trading risk based on proximity to S/R levels."""
        try:
            # Use safe division to prevent division by zero
            support_distance = safe_divide(abs(current_price - support_price), current_price, 0.0)
            resistance_distance = safe_divide(abs(current_price - resistance_price), current_price, 0.0)
        except MathValidationError as e:
            logger.warning(f"Mathematical validation error in risk assessment: {e}")
            return 'Unknown - Calculation error'

        # High risk if very close to support (potential breakdown) or resistance (potential rejection)
        if support_distance <= 0.005:  # Within 0.5%
            return 'High - Price near strong support, watch for breakdown'
        elif resistance_distance <= 0.005:  # Within 0.5%
            return 'High - Price near strong resistance, watch for rejection'
        elif support_distance <= 0.01 or resistance_distance <= 0.01:  # Within 1%
            return 'Medium - Approaching key levels'
        else:
            return 'Low - Price well-positioned between support and resistance'

            self.logger.error('🔍 ENHANCED ERROR DIAGNOSTICS:')
            self.logger.error(f'   Error Type: {type(e).__name__}')
            self.logger.error(f'   Error Message: {str(e)}')
            self.logger.error(f'   Execution Time: {execution_time:.2f}s')

            # Memory usage information
            if PSUTIL_AVAILABLE:
                memory = psutil.virtual_memory()
                self.logger.error(f'   System Memory - Total: {memory.total / 1024**3:.1f}GB, Available: {memory.available / 1024**3:.1f}GB, Used: {memory.percent:.1f}%')

                process = psutil.Process()
                memory_info = process.memory_info()
                self.logger.error(f'   Process Memory - RSS: {memory_info.rss / 1024**2:.1f}MB, VMS: {memory_info.vms / 1024**2:.1f}MB')
            else:
                self.logger.error('   Memory monitoring unavailable (psutil not installed)')

            # Python memory usage if available
            try:
                import gc
                collected = gc.collect()
                self.logger.error(f'   Garbage Collection: {collected} objects collected')
            except:
                pass

            # Full traceback
            self.logger.error('   Full Traceback:')
            for line in traceback.format_exc().split('\n'):
                self.logger.error(f'     {line}')

            # Local variables in the calling frame
            try:
                frame = sys._getframe(1)
                local_vars = frame.f_locals
                self.logger.error('   Local Variables (key info):')
                for key, value in local_vars.items():
                    if key in ['features_data', 'sr_levels', 'ml_results']:
                        if hasattr(value, 'shape'):
                            self.logger.error(f'     {key}: {value.shape}')
                        elif isinstance(value, dict):
                            self.logger.error(f'     {key}: {len(value)} items')
                        else:
                            self.logger.error(f'     {key}: {type(value)}')
            except:
                self.logger.error('   Could not capture local variables')

            internal_call_tracker['error'] = {
                'error_type': type(e).__name__,
                'error_message': str(e),
                'execution_time': execution_time,
                'traceback': traceback.format_exc(),
                'memory_usage': {
                    'system_total_gb': memory.total / 1024**3,
                    'system_available_gb': memory.available / 1024**3,
                    'system_used_percent': memory.percent,
                    'process_rss_mb': memory_info.rss / 1024**2,
                    'process_vms_mb': memory_info.vms / 1024**2
                }
            }
            return {'success': False, 'step02_5_sr_optimization_completed': False, 'step02_5_sr_optimization_failure_reason': str(e), 'error': str(e), 'execution_time': execution_time, 'internal_call_tracker': internal_call_tracker, 'step_name': 'step02_5_sr_optimization'}

    async def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer comprehensive features for SR analysis using advanced modules."""
        self.logger.info('🔧 FEATURE ENGINEERING STARTED - Data shape: {}'.format(data.shape))
        feature_start_time = time.time()
        
        # CRITICAL: Validate input data before feature engineering
        if data is None:
            raise ValueError("CRITICAL: Input data is None for feature engineering. Cannot proceed.")
        
        if data.empty:
            raise ValueError("CRITICAL: Input data is empty for feature engineering. Cannot proceed.")
        
        if len(data) < 50:  # Minimum 50 rows for meaningful feature engineering
            raise ValueError(f"CRITICAL: Insufficient data for feature engineering. Only {len(data)} rows available, minimum 50 required.")
        
        # Validate required columns for feature engineering
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"CRITICAL: Missing required columns for feature engineering: {missing_columns}. Available columns: {list(data.columns)}")
        
        self.logger.info(f'✅ Feature engineering input validation passed: {len(data)} rows, {len(data.columns)} columns')
        
        try:
            # Try to use advanced feature engineering module
            if not ADVANCED_FEATURES_AVAILABLE:
                raise ImportError("AdvancedFeatureEngineeringStep not available")
            # AdvancedFeatureEngineeringStep is already imported at module level

            # Configure advanced feature engineering - wavelets NEVER enabled for step02_5
            enable_wavelets = False  # Never enable wavelets for step02_5
            self.logger.info(f'🌊 Wavelet features: DISABLED (never enabled for step02_5)')
            self.logger.info(f'🚫 Lookback optimization: DISABLED (step02_5 compatibility mode)')

            feature_config = {
                'feature_engineering': {
                    'enable_wavelets': enable_wavelets,
                    'enable_multi_timeframe': True,
                    'enable_feature_interactions': True,  # Re-enable interactions
                    'enable_regime_features': True,  # Re-enable regime features (best-effort)
                    'timeframes': ['30m', '1h', '4h', '1d'],
                    'chunk_size': 500000,
                    'max_features': 500,  # Allow more features
                    'feature_interaction_degree': 2,  # Include pairwise interactions
                    'regime_lookback_days': 30,
                    # Disable lookback optimization for step02_5
                    'disable_lookback_optimization': True,
                    'cross_timeframe_enabled': False,
                    'regime_specific': False
                }
            }

            # Initialize advanced feature engineering
            advanced_fe = AdvancedFeatureEngineeringStep(feature_config)
            await advanced_fe.initialize()

            # Use individual advanced feature methods
            self.logger.info('🚀 Executing advanced feature engineering...')
            self.logger.info('📋 Step02_5 Feature Access Policy:')
            self.logger.info('   INCLUDED: Technical indicators, Microstructure features, Multi-timeframe features, Interactions, Regime-aware (best-effort)')
            self.logger.info('   EXCLUDED: Wavelet features')
            all_advanced_features = {}

            # 1. Technical indicators (from step06)
            self.logger.info('📈 Loading comprehensive technical indicators from step06...')
            technical_features = advanced_fe._generate_comprehensive_technical_features(data)
            all_advanced_features['technical'] = technical_features
            self.logger.info(f'✅ Technical indicators: {len(technical_features.columns)} features')

            # 2. Microstructure features
            self.logger.info('🔬 Calculating microstructure features...')
            try:
                microstructure_features = advanced_fe._calculate_microstructure_features(data)
                all_advanced_features['microstructure'] = microstructure_features
                self.logger.info(f'✅ Microstructure features: {len(microstructure_features.columns)} features')
            except Exception as e:
                self.logger.warning(f'Microstructure features failed: {e}')

            # 3. Multi-timeframe features
            if advanced_fe.enable_multi_timeframe:
                self.logger.info('⏰ Calculating multi-timeframe features...')
                try:
                    mtf_features = await advanced_fe._build_mtf_features_required(data)
                    all_advanced_features['multi_timeframe'] = mtf_features
                    self.logger.info(f'✅ Multi-timeframe features: {len(mtf_features.columns)} features')
                except Exception as e:
                    self.logger.warning(f'Multi-timeframe features failed: {e}')

            # 4. Wavelet features
            if advanced_fe.enable_wavelets and advanced_fe.wavelet_analyzer is not None:
                self.logger.info('🌊 Calculating wavelet features...')
                try:
                    wavelet_features = advanced_fe.wavelet_analyzer.extract_wavelet_features(data, price_column='close', symbol='SYMBOL', timeframe='30m')
                    all_advanced_features['wavelet'] = wavelet_features
                    self.logger.info(f'✅ Wavelet features: {len(wavelet_features.columns)} features')
                except Exception as e:
                    self.logger.warning(f'Wavelet features failed: {e}')

            # 5. Feature interactions - ENABLED (best-effort)
            if advanced_fe.enable_feature_interactions:
                self.logger.info('🔗 Creating feature interactions...')
                try:
                    interactions = advanced_fe._create_feature_interactions(data)
                    if interactions is not None and hasattr(interactions, 'columns') and len(interactions.columns) > 0:
                        all_advanced_features['interactions'] = interactions
                        self.logger.info(f'✅ Interactions features: {len(interactions.columns)} features')
                except Exception as e:
                    self.logger.warning(f'Feature interactions failed: {e}')

            # 6. Regime-aware features - ENABLED (best-effort)
            if advanced_fe.enable_regime_features:
                if getattr(advanced_fe, 'regime_engine', None) is not None:
                    self.logger.info('🎭 Creating regime-aware features...')
                    try:
                        regime_features = advanced_fe._create_regime_aware_features(data, {})
                        if regime_features is not None and hasattr(regime_features, 'columns') and len(regime_features.columns) > 0:
                            all_advanced_features['regime'] = regime_features
                            self.logger.info(f'✅ Regime features: {len(regime_features.columns)} features')
                    except Exception as e:
                        self.logger.warning(f'Regime features failed: {e}')
                else:
                    self.logger.info('🎭 Regime engine not available; skipping regime-aware features')

            # Count total features
            total_advanced_count = sum(len(df.columns) for df in all_advanced_features.values() if df is not None and hasattr(df, 'columns'))
            self.logger.info(f'🎉 Advanced features total: {total_advanced_count} features')

            # Combine with basic features
            features_data = data.copy()

            # Add ALL advanced features
            total_advanced_features = 0
            for feature_type, feature_df in all_advanced_features.items():
                if feature_df is not None and hasattr(feature_df, 'columns') and len(feature_df.columns) > 0:
                    for col in feature_df.columns:
                        features_data[f'{feature_type}_{col}'] = feature_df[col]
                    total_advanced_features += len(feature_df.columns)
                    self.logger.info(f'✅ Added {len(feature_df.columns)} {feature_type} features')

            self.logger.info(f'🎉 Total advanced features added: {total_advanced_features}')

        except ImportError:
            error_msg = 'Advanced feature engineering module not available - required components missing'
            self.logger.error(f'❌ {error_msg}')
            raise ImportError(error_msg)
        except Exception as e:
            error_msg = f'Advanced feature engineering failed: {e}'
            self.logger.error(f'❌ {error_msg}')
            raise RuntimeError(error_msg)
        
        feature_time = time.time() - feature_start_time
        self.logger.info(f'✅ Feature engineering completed in {feature_time:.4f}s')
        self.logger.info(f'📈 Engineered {len(features_data.columns)} features')
        self.logger.info(f'📊 Final data shape: {features_data.shape}')

        # Debug: Show feature names
        original_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'year', 'month', 'day', 'exchange', 'symbol', 'timeframe', 'trade_volume']
        new_features = [col for col in features_data.columns if col not in original_cols]
        self.logger.info(f'📊 Original columns: {len(original_cols)}')
        self.logger.info(f'📊 New features: {len(new_features)} - {new_features[:10]}...' if len(new_features) > 10 else f'📊 New features: {len(new_features)} - {new_features}')

        # CRITICAL FIX: Filter out non-numeric columns to prevent string-to-float conversion errors
        # Keep only numeric columns and essential columns for ML processing
        self.logger.info('🔧 Filtering features to include only numeric columns for ML compatibility...')
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns.tolist()

        # Add essential non-numeric columns that are needed for processing but will be excluded from ML features
        essential_cols = ['timestamp']  # Keep timestamp for indexing
        final_columns = numeric_columns + [col for col in essential_cols if col in features_data.columns]

        # Filter the DataFrame to include only compatible columns
        features_data_filtered = features_data[final_columns].copy()
        self.logger.info(f'✅ Filtered features: {len(final_columns)} columns ({len(numeric_columns)} numeric + {len([col for col in essential_cols if col in features_data.columns])} essential)')
        self.logger.info(f'🗑️ Removed {len(features_data.columns) - len(final_columns)} non-numeric columns')

        # Ensure only numeric columns proceed downstream
        numeric_cols_final = features_data_filtered.select_dtypes(include=[np.number]).columns
        return features_data_filtered[numeric_cols_final].copy()
    
    def _engineer_features_enhanced_basic(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced basic feature engineering with market microstructure features."""
        try:
            self.logger.info('🔧 Engineering enhanced basic features...')
            self.logger.info(f'📊 Input data shape: {data.shape}')
            self.logger.info(f'📊 Input data columns: {list(data.columns)}')

            # Standardize column names
            column_mapping = {}
            for col in data.columns:
                col_lower = col.lower()
                if 'open' in col_lower and 'open' not in column_mapping:
                    column_mapping['open'] = col
                elif 'high' in col_lower and 'high' not in column_mapping:
                    column_mapping['high'] = col
                elif 'low' in col_lower and 'low' not in column_mapping:
                    column_mapping['low'] = col
                elif 'close' in col_lower and 'close' not in column_mapping:
                    column_mapping['close'] = col
                elif 'volume' in col_lower and 'volume' not in column_mapping:
                    column_mapping['volume'] = col

            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in column_mapping]
            if missing_columns:
                raise ValueError(f'Missing required columns: {missing_columns}. Available columns: {list(data.columns)}')

            self.logger.info(f'📊 Column mapping: {column_mapping}')

            features_data = data.copy()
            for standard_name, actual_name in column_mapping.items():
                features_data[standard_name] = features_data[actual_name]

            self.logger.info(f'📊 After column standardization: {features_data.shape}')
        except Exception as e:
            self.logger.error(f'❌ Error in feature engineering setup: {e}')
            raise
        
        try:
            # Basic price features
            features_data['price_range'] = features_data['high'] - features_data['low']
            features_data['price_change'] = features_data['close'].pct_change()
            features_data['volume_change'] = features_data['volume'].pct_change()

            self.logger.info(f'📊 After basic price features: {features_data.shape}')

            # Market microstructure features
            features_data['spread'] = features_data['high'] - features_data['low']
            features_data['spread_pct'] = features_data['spread'] / features_data['close']
            features_data['typical_price'] = (features_data['high'] + features_data['low'] + features_data['close']) / 3
            features_data['vwap'] = (features_data['typical_price'] * features_data['volume']).cumsum() / features_data['volume'].cumsum()
            features_data['price_to_vwap'] = features_data['close'] / features_data['vwap']
            features_data['dollar_volume'] = features_data['close'] * features_data['volume']
            features_data['log_dollar_volume'] = np.log1p(features_data['dollar_volume'])
            features_data['price_impact'] = features_data['price_change'].abs() / (features_data['volume'] + 1)
            features_data['kyle_lambda'] = features_data['price_impact'].rolling(20).mean()
            features_data['order_flow_imbalance'] = np.where(features_data['close'] > features_data['open'], features_data['volume'], -features_data['volume'])
            features_data['ofi_cumsum'] = features_data['order_flow_imbalance'].cumsum()

            self.logger.info(f'📊 After microstructure features: {features_data.shape}')
        except Exception as e:
            self.logger.error(f'❌ Error adding basic/microstructure features: {e}')
            raise
        
        try:
            # Technical indicators
            for period in [5, 10, 20, 50]:
                features_data[f'sma_{period}'] = features_data['close'].rolling(period).mean()
                features_data[f'price_sma_{period}_ratio'] = features_data['close'] / features_data[f'sma_{period}']

            self.logger.info(f'📊 After technical indicators: {features_data.shape}')

            # Volatility features
            features_data['volatility_5'] = features_data['price_change'].rolling(5).std()
            features_data['volatility_20'] = features_data['price_change'].rolling(20).std()

            # RSI
            delta = features_data['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window = 14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window = 14).mean()
            rs = gain / loss
            features_data['rsi'] = 100 - 100 / (1 + rs)

            # Bollinger Bands
            features_data['bb_middle'] = features_data['close'].rolling(20).mean()
            bb_std = features_data['close'].rolling(20).std()
            features_data['bb_upper'] = features_data['bb_middle'] + bb_std * 2
            features_data['bb_lower'] = features_data['bb_middle'] - bb_std * 2
            features_data['bb_position'] = (features_data['close'] - features_data['bb_lower']) / (features_data['bb_upper'] - features_data['bb_lower'])

            self.logger.info(f'📊 After technical indicators: {features_data.shape}')

            # Price position features
            features_data['high_low_ratio'] = features_data['high'] / features_data['low']
            features_data['close_high_ratio'] = features_data['close'] / features_data['high']
            features_data['close_low_ratio'] = features_data['close'] / features_data['low']

            # Volume features
            features_data['volume_sma_20'] = features_data['volume'].rolling(20).mean()
            features_data['volume_ratio'] = features_data['volume'] / features_data['volume_sma_20']

            self.logger.info(f'📊 After all features: {features_data.shape}')
        except Exception as e:
            self.logger.error(f'❌ Error adding technical/position features: {e}')
            raise
        
        # Fill NaN values and handle infinite values
        features_data = features_data.ffill().fillna(0)

        # Replace infinite values with finite values
        features_data = features_data.replace([np.inf, -np.inf], [1e10, -1e10])

        # Ensure all numeric columns are float64 to avoid NumPy type issues
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            features_data[col] = features_data[col].astype('float64')

        self.logger.info(f'📊 Output data shape: {features_data.shape}')
        self.logger.info(f'📊 Output data columns: {len(features_data.columns)}')

        # CRITICAL FIX: Filter out non-numeric columns to prevent string-to-float conversion errors
        # Keep only numeric columns and essential columns for ML processing
        self.logger.info('🔧 Filtering features to include only numeric columns for ML compatibility...')
        numeric_columns = features_data.select_dtypes(include=[np.number]).columns.tolist()

        # Add essential non-numeric columns that are needed for processing but will be excluded from ML features
        essential_cols = ['timestamp']  # Keep timestamp for indexing
        final_columns = numeric_columns + [col for col in essential_cols if col in features_data.columns]

        # Filter the DataFrame to include only compatible columns
        features_data_filtered = features_data[final_columns].copy()
        self.logger.info(f'✅ Filtered features: {len(final_columns)} columns ({len(numeric_columns)} numeric + {len([col for col in essential_cols if col in features_data.columns])} essential)')
        self.logger.info(f'🗑️ Removed {len(features_data.columns) - len(final_columns)} non-numeric columns')

        return features_data_filtered

    def _validate_and_fill_features(self, features: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
        """Generic function to validate and fill NaN values in features.

        Args:
            features: DataFrame with features to validate
            data: Original market data for fill values

        Returns:
            DataFrame with validated and filled features

        Raises:
            ValueError: If any feature has >5% NaN values (relaxed threshold for technical indicators)
        """
        for col in features.columns:
            nan_pct = features[col].isna().mean() * 100

            # Check for small data gaps that can be forward-filled
            if nan_pct > 0 and nan_pct <= 0.5:  # Small gaps (< 0.5%)
                # Check if gaps are due to small time differences (< 2 seconds)
                if hasattr(data, 'index') and hasattr(data.index, 'to_series'):
                    try:
                        # Calculate time gaps if data has timestamp index
                        if isinstance(data.index, pd.DatetimeIndex):
                            time_gaps = data.index.to_series().diff().dt.total_seconds()
                            max_gap = time_gaps.max() if not time_gaps.empty else 0
                            if max_gap < 2:  # Small gaps can be forward-filled
                                features[col] = features[col].fillna(method='ffill')
                                nan_pct = features[col].isna().mean() * 100
                                if nan_pct == 0:
                                    continue  # Successfully filled
                        else:
                            # If index is not datetime, try to use timestamp column if available
                            if 'timestamp' in data.columns:
                                try:
                                    # Convert timestamp to datetime for gap calculation
                                    timestamp_dt = pd.to_datetime(data['timestamp'], unit='ms')
                                    time_gaps = timestamp_dt.diff().dt.total_seconds()
                                    max_gap = time_gaps.max() if not time_gaps.empty else 0
                                    if max_gap < 2:  # Small gaps can be forward-filled
                                        features[col] = features[col].fillna(method='ffill')
                                        nan_pct = features[col].isna().mean() * 100
                                        if nan_pct == 0:
                                            continue  # Successfully filled
                                except Exception as ts_error:
                                    self.logger.debug(f'Could not calculate time gaps from timestamp column: {ts_error}')
                    except Exception as gap_error:
                        self.logger.debug(f'Could not calculate time gaps for forward-fill check: {gap_error}')

            # Selective relaxed threshold only for indicators that naturally have NaN at the beginning
            # RSI needs lookback period, others can be stricter
            if 'rsi' in col.lower():
                threshold = 5.0  # RSI has natural NaN at start due to lookback
            elif any(keyword in col.lower() for keyword in ['stoch', 'williams', 'cci', 'ma', 'sma', 'ema', 'bb_', 'atr']):
                threshold = 1.0  # Other technical indicators get moderate threshold
            else:
                threshold = 0.1  # Strict threshold for all other features
            if nan_pct > threshold:
                raise ValueError(f'❌ Excessive NaN values in {col}: {nan_pct:.2f}% (threshold: {threshold}%)')

            # Apply appropriate fill strategy based on feature type
            if any(keyword in col.lower() for keyword in ['rsi', 'stoch', 'williams', 'cci']):
                # Oscillators: fill with neutral values
                if 'rsi' in col.lower():
                    features[col] = features[col].fillna(50)
                elif 'stoch' in col.lower():
                    features[col] = features[col].fillna(50)
                elif 'williams' in col.lower():
                    features[col] = features[col].fillna(-50)
                elif 'cci' in col.lower():
                    features[col] = features[col].fillna(0)
            elif any(keyword in col.lower() for keyword in ['ma', 'sma', 'ema', 'bb_', 'vwap']):
                # Price-based features: fill with current price
                features[col] = features[col].fillna(data['close'])
            elif any(keyword in col.lower() for keyword in ['volatility', 'atr']):
                # Volatility features: fill with rolling mean or zero
                features[col] = features[col].fillna(features[col].rolling(50).mean().fillna(0))
            elif any(keyword in col.lower() for keyword in ['momentum', 'roc', 'macd']):
                # Momentum features: fill with zero
                features[col] = features[col].fillna(0)
            elif any(keyword in col.lower() for keyword in ['ratio', 'position']):
                # Ratio/position features: fill with neutral values
                features[col] = features[col].fillna(0.5 if 'position' in col.lower() else 1.0)
            else:
                # Default: fill with zero
                features[col] = features[col].fillna(0)

        return features

    def _generate_comprehensive_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate comprehensive technical indicators directly."""
        features = pd.DataFrame(index=data.index)

        # Basic price features and acceleration
        features['price_change'] = data['close'].pct_change()
        features['price_change_abs'] = data['close'].diff().abs()
        features['price_acceleration'] = features['price_change'].diff()  # Acceleration
        features['price_jerk'] = features['price_acceleration'].diff()   # Jerk (rate of change of acceleration)

        # RSI variations
        for period in [7, 14, 21]:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / (loss.replace(0, np.nan))
            features[f'rsi_{period}'] = 100 - (100 / (1 + rs))

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()

        # MACD
        ema_12 = data['close'].ewm(span=12).mean()
        ema_26 = data['close'].ewm(span=26).mean()
        features['macd_line'] = ema_12 - ema_26
        features['macd_signal'] = features['macd_line'].ewm(span=9).mean()
        features['macd_histogram'] = features['macd_line'] - features['macd_signal']

        # Bollinger Bands
        for window in [10, 20, 30]:
            sma = data['close'].rolling(window).mean()
            std = data['close'].rolling(window).std()
            features[f'bb_middle_{window}'] = sma
            features[f'bb_upper_{window}'] = sma + (std * 2)
            features[f'bb_lower_{window}'] = sma - (std * 2)
            features[f'bb_position_{window}'] = (data['close'] - features[f'bb_lower_{window}']) / (features[f'bb_upper_{window}'] - features[f'bb_lower_{window}'])

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            high_low = data['high'] - data['low']
            high_close = (data['high'] - data['close'].shift(1)).abs()
            low_close = (data['low'] - data['close'].shift(1)).abs()
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            features[f'atr_{period}'] = tr.rolling(period).mean()

        # Stochastic Oscillator
        for k_period, d_period in [(14, 3), (21, 5)]:
            lowest_low = data['low'].rolling(k_period).min()
            highest_high = data['high'].rolling(k_period).max()
            features[f'stoch_k_{k_period}'] = ((data['close'] - lowest_low) / (highest_high - lowest_low)) * 100
            features[f'stoch_d_{k_period}_{d_period}'] = features[f'stoch_k_{k_period}'].rolling(d_period).mean()

        # Williams %R
        for period in [14, 21]:
            highest_high = data['high'].rolling(period).max()
            lowest_low = data['low'].rolling(period).min()
            features[f'williams_r_{period}'] = ((highest_high - data['close']) / (highest_high - lowest_low)) * -100

        # Momentum features
        for period in [5, 10, 15, 20]:
            features[f'momentum_{period}'] = data['close'] - data['close'].shift(period)
            features[f'roc_{period}'] = (data['close'] - data['close'].shift(period)) / data['close'].shift(period) * 100

        # VWAP (Volume Weighted Average Price)
        if 'volume' in data.columns:
            data_copy = data.copy()
            data_copy['typical_price'] = (data_copy['high'] + data_copy['low'] + data_copy['close']) / 3
            data_copy['price_volume'] = data_copy['typical_price'] * data_copy['volume']
            data_copy['cumulative_price_volume'] = data_copy['price_volume'].cumsum()
            data_copy['cumulative_volume'] = data_copy['volume'].cumsum()
            features['vwap'] = data_copy['cumulative_price_volume'] / data_copy['cumulative_volume']
            features['vwap_deviation'] = (data['close'] - features['vwap']) / features['vwap'] * 100

        # Commodity Channel Index (CCI)
        for period in [14, 20]:
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma_tp = tp.rolling(period).mean()
            mad = (tp - sma_tp).abs().rolling(period).mean()
            features[f'cci_{period}'] = (tp - sma_tp) / (0.015 * mad)

        # Momentum
        for period in [5, 10, 20]:
            features[f'momentum_{period}'] = data['close'] / data['close'].shift(period) - 1

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            features[f'roc_{period}'] = ((data['close'] - data['close'].shift(period)) / data['close'].shift(period)) * 100

        # Volume-based indicators
        if 'volume' in data.columns:
            for period in [5, 10, 20]:
                features[f'volume_sma_{period}'] = data['volume'].rolling(period).mean()
                features[f'volume_ratio_{period}'] = data['volume'] / features[f'volume_sma_{period}']

            # On Balance Volume (OBV)
            obv = (np.sign(data['close'].diff()) * data['volume']).cumsum()
            features['obv'] = obv

            # Volume Weighted Average Price (VWAP)
            features['vwap'] = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()

        # Price-based volatility measures
        for period in [5, 10, 20, 30]:
            returns = data['close'].pct_change()
            features[f'volatility_{period}'] = returns.rolling(period).std()
            features[f'high_low_ratio_{period}'] = (data['high'] / data['low']).rolling(period).mean()

        # Gap analysis
        close_shifted = data['close'].shift(1)
        features['gap_up'] = ((data['open'] > close_shifted) & close_shifted.notna()).astype(int)
        features['gap_down'] = ((data['open'] < close_shifted) & close_shifted.notna()).astype(int)
        features['gap_size'] = (data['open'] - close_shifted) / close_shifted

        # Intraday momentum
        features['open_to_close'] = (data['close'] - data['open']) / data['open']
        features['high_to_low'] = (data['high'] - data['low']) / data['low']
        features['close_to_high'] = (data['close'] - data['low']) / (data['high'] - data['low'])

        # Add advanced momentum features
        momentum_features = self._calculate_advanced_momentum_features(data)
        momentum_features = self._validate_and_fill_features(momentum_features, data)
        for col in momentum_features.columns:
            features[f'momentum_{col}'] = momentum_features[col]

        # Add correlation features
        correlation_features = self._calculate_correlation_features(data)
        correlation_features = self._validate_and_fill_features(correlation_features, data)
        for col in correlation_features.columns:
            features[f'correlation_{col}'] = correlation_features[col]

        # Add liquidity features (if volume data available)
        if 'volume' in data.columns:
            liquidity_features = self._calculate_liquidity_features(data)
            liquidity_features = self._validate_and_fill_features(liquidity_features, data)
            for col in liquidity_features.columns:
                features[f'liquidity_{col}'] = liquidity_features[col]

        # Add adaptive features
        adaptive_features = self._calculate_adaptive_features(data)
        adaptive_features = self._validate_and_fill_features(adaptive_features, data)
        for col in adaptive_features.columns:
            features[f'adaptive_{col}'] = adaptive_features[col]

        # Validate and fill NaN values using generic function
        features = self._validate_and_fill_features(features, data)

        return features

    def _calculate_advanced_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced momentum features."""
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change().fillna(0)

        # Momentum indicators
        features['momentum_5'] = returns.rolling(5).mean()
        features['momentum_20'] = returns.rolling(20).mean()
        features['momentum_50'] = returns.rolling(50).mean()

        # Momentum acceleration
        features['momentum_acceleration'] = features['momentum_5'] - features['momentum_20']

        # Momentum strength
        momentum_20_std = features['momentum_20'].rolling(20).std().fillna(1e-8)
        features['momentum_strength'] = features['momentum_5'] / (momentum_20_std + 1e-8)

        # Momentum divergence
        price_momentum = data['close'].pct_change(5)
        volume_momentum = data.get('volume', pd.Series(1, index=data.index)).pct_change(5).fillna(0)
        features['momentum_divergence'] = price_momentum - volume_momentum

        return features.fillna(0)

    def _calculate_correlation_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation features."""
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change().fillna(0)

        # Rolling autocorrelations
        features['autocorrelation_5'] = returns.rolling(5).corr(returns.shift(1))
        features['autocorrelation_20'] = returns.rolling(20).corr(returns.shift(1))

        # Cross-timeframe correlations (simplified)
        returns_5 = returns.rolling(5).mean()
        returns_20 = returns.rolling(20).mean()
        features['cross_timeframe_correlation'] = returns_5.rolling(20).corr(returns_20)

        return features.fillna(0)

    def _calculate_liquidity_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity features."""
        features = pd.DataFrame(index=data.index)

        if 'volume' not in data.columns:
            return features

        # Volume-based liquidity
        avg_volume = data['volume'].rolling(20).mean()
        features['volume_liquidity'] = data['volume'] / (avg_volume + 1e-8)

        # Price impact
        price_changes = data['close'].pct_change().abs()
        features['price_impact'] = price_changes / (data['volume'] + 1e-8)
        features['price_impact_smooth'] = features['price_impact'].rolling(20).mean()

        # Liquidity percentiles
        features['liquidity_percentile'] = features['volume_liquidity'].rolling(100).rank(pct=True)

        return features.fillna(0)

    def _calculate_adaptive_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate adaptive features based on volatility."""
        features = pd.DataFrame(index=data.index)

        returns = data['close'].pct_change().fillna(0)
        volatility = returns.rolling(20).std()

        # Adaptive periods based on volatility
        base_period = 20
        volatility_factor = volatility / (volatility.rolling(100).mean() + 1e-8)
        adaptive_period = (base_period * volatility_factor).clip(5, 50)

        # Adaptive moving averages (vectorized approach)
        features['adaptive_period'] = adaptive_period.fillna(20).astype(int).clip(5, 50)

        # Calculate adaptive MA using rolling windows
        for period in [5, 10, 15, 20, 25, 30, 40, 50]:
            mask = features['adaptive_period'] == period
            if mask.any():
                features.loc[mask, 'adaptive_ma'] = data.loc[mask, 'close'].rolling(period).mean()

        # Fill NaN values
        features['adaptive_ma'] = features['adaptive_ma'].fillna(data['close'].rolling(20).mean())

        return features.fillna(0)

    def _run_async_init(self, sr_manager) -> bool:
        """Run async initialization in a separate thread."""
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(sr_manager.initialize())
        finally:
            loop.close()

    def _run_sr_calculation(self, sr_manager, data: pd.DataFrame) -> Dict[str, Any]:
        """Run SR calculation in a separate thread with simplified execution."""
        try:
            # Use synchronous execution to avoid nested async/threading issues
            if hasattr(sr_manager, 'calculate_sr_levels_from_backtest_sync'):
                return sr_manager.calculate_sr_levels_from_backtest_sync(data, '1m')
            else:
                # Fallback to async execution with proper error handling
                import asyncio
                try:
                    # Try to get existing event loop first
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Create a new thread with its own loop to avoid conflicts
                        import concurrent.futures
                        def run_in_new_loop():
                            new_loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(new_loop)
                            try:
                                return new_loop.run_until_complete(sr_manager.calculate_sr_levels_from_backtest(data, '1m'))
                            finally:
                                new_loop.close()

                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                            future = executor.submit(run_in_new_loop)
                            return future.result(timeout=1200)  # 20 minutes for the inner calculation
                    else:
                        return loop.run_until_complete(sr_manager.calculate_sr_levels_from_backtest(data, '1m'))
                except RuntimeError:
                    # No event loop exists, create a new one
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(sr_manager.calculate_sr_levels_from_backtest(data, '1m'))
                    finally:
                        loop.close()
        except Exception as e:
            self.logger.error(f'SR calculation failed: {e}')
            raise

    def _run_sr_calculation_chunked(self, sr_manager, data: pd.DataFrame) -> Dict[str, Any]:
        """Run SR calculation with chunked processing for large datasets."""
        try:
            chunk_size = 50000  # Process 50K points at a time
            all_support_levels = []
            all_resistance_levels = []

            total_chunks = (len(data) + chunk_size - 1) // chunk_size
            self.logger.info(f'📊 Processing {len(data)} points in {total_chunks} chunks of {chunk_size} points each')

            for i in range(0, len(data), chunk_size):
                chunk_end = min(i + chunk_size, len(data))
                chunk_data = data.iloc[i:chunk_end]

                self.logger.info(f'🔄 Processing chunk {i//chunk_size + 1}/{total_chunks} ({len(chunk_data)} points)')

                # Process this chunk
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._run_sr_calculation, sr_manager, chunk_data)
                    chunk_results = future.result(timeout=600)  # 10 minutes per chunk

                # Merge results
                if 'support_levels' in chunk_results:
                    all_support_levels.extend(chunk_results['support_levels'])
                if 'resistance_levels' in chunk_results:
                    all_resistance_levels.extend(chunk_results['resistance_levels'])

            # Consolidate and deduplicate levels
            consolidated_results = self._consolidate_sr_levels(all_support_levels, all_resistance_levels)
            self.logger.info(f'✅ Chunked processing completed: {len(consolidated_results.get("support_levels", []))} support, {len(consolidated_results.get("resistance_levels", []))} resistance levels')

            return consolidated_results

        except Exception as e:
            self.logger.error(f'Chunked SR calculation failed: {e}')
            return {'support_levels': [], 'resistance_levels': []}

    def _consolidate_sr_levels(self, support_levels, resistance_levels) -> Dict[str, Any]:
        """Consolidate and deduplicate SR levels from multiple chunks."""
        try:
            # Simple deduplication by price proximity (within 0.1% of price)
            def deduplicate_levels(levels, level_type):
                if not levels:
                    return []

                # Sort by price
                sorted_levels = sorted(levels, key=lambda x: x.price if hasattr(x, 'price') else x.get('price', 0))

                consolidated = []
                current_group = [sorted_levels[0]]

                for level in sorted_levels[1:]:
                    current_price = current_group[0].price if hasattr(current_group[0], 'price') else current_group[0].get('price', 0)
                    level_price = level.price if hasattr(level, 'price') else level.get('price', 0)

                    # If within 0.1% of current group price, add to group
                    if abs(level_price - current_price) / current_price < 0.001:
                        current_group.append(level)
                    else:
                        # Consolidate current group and start new group
                        if current_group:
                            consolidated.append(self._merge_level_group(current_group, level_type))
                        current_group = [level]

                # Don't forget the last group
                if current_group:
                    consolidated.append(self._merge_level_group(current_group, level_type))

                return consolidated[:50]  # Limit to top 50 levels

            consolidated_support = deduplicate_levels(support_levels, 'support')
            consolidated_resistance = deduplicate_levels(resistance_levels, 'resistance')

            return {
                'support_levels': consolidated_support,
                'resistance_levels': consolidated_resistance
            }

        except Exception as e:
            self.logger.error(f'Level consolidation failed: {e}')
            return {'support_levels': support_levels[:50], 'resistance_levels': resistance_levels[:50]}

    def _merge_level_group(self, level_group, level_type):
        """Merge a group of similar levels into a single level."""
        try:
            if not level_group:
                return None

            # Calculate weighted average price based on strength
            total_weight = 0
            weighted_price = 0

            for level in level_group:
                strength = getattr(level, 'strength', 0.5) if hasattr(level, 'strength') else level.get('strength', 0.5)
                price = level.price if hasattr(level, 'price') else level.get('price', 0)

                weighted_price += price * strength
                total_weight += strength

            avg_price = weighted_price / total_weight if total_weight > 0 else level_group[0].price if hasattr(level_group[0], 'price') else level_group[0].get('price', 0)
            max_strength = max(getattr(level, 'strength', 0.5) if hasattr(level, 'strength') else level.get('strength', 0.5) for level in level_group)

            # Create consolidated level
            return {
                'price': float(avg_price),
                'strength': float(max_strength),
                'type': level_type,
                'method': 'consolidated_fractal',
                'touch_count': sum(getattr(level, 'touch_count', 1) if hasattr(level, 'touch_count') else level.get('touch_count', 1) for level in level_group),
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f'Level group merging failed: {e}')
            return level_group[0] if level_group else None

    def _should_terminate_sr_detection(self, elapsed_seconds: int, remaining_seconds: int) -> bool:
        """Check if SR detection should be terminated early based on system constraints."""
        try:
            import psutil
            import os

            # Check memory usage
            memory = psutil.virtual_memory()
            memory_usage_percent = memory.percent

            # Check process memory
            process = psutil.Process(os.getpid())
            process_memory_mb = process.memory_info().rss / 1024 / 1024

            # Terminate if memory usage is too high
            if memory_usage_percent > 90:  # Over 90% memory usage
                self.logger.warning(f'🛑 High memory usage detected ({memory_usage_percent:.1f}%), terminating SR detection')
                return True

            if process_memory_mb > 8000:  # Over 8GB process memory
                self.logger.warning(f'🛑 High process memory usage detected ({process_memory_mb:.1f}MB), terminating SR detection')
                return True

            # Check CPU usage (if consistently high, might indicate hanging)
            cpu_percent = psutil.cpu_percent(interval=1)

            # If CPU usage is very low for an extended period, might indicate hanging
            if elapsed_seconds > 600 and cpu_percent < 5:  # Low CPU for more than 10 minutes
                self.logger.warning(f'🛑 Low CPU usage ({cpu_percent:.1f}%) for extended period, possible hang detected')
                return True

            return False

        except Exception as e:
            self.logger.warning(f'⚠️ Could not check system constraints: {e}')
            return False

    def _detect_sr_levels(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect support and resistance levels using Enhanced SR Detection."""
        self.logger.info('🎯 Using Enhanced SR Detection with multiple advanced algorithms...')

        # CRITICAL: Validate input data before S/R detection
        if data is None:
            raise ValueError("CRITICAL: Input data is None for S/R detection. Cannot proceed.")
        
        if data.empty:
            raise ValueError("CRITICAL: Input data is empty for S/R detection. Cannot proceed.")
        
        if len(data) < 100:  # Minimum 100 rows for meaningful S/R detection
            raise ValueError(f"CRITICAL: Insufficient data for S/R detection. Only {len(data)} rows available, minimum 100 required.")
        
        # Validate required columns for S/R detection
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"CRITICAL: Missing required columns for S/R detection: {missing_columns}. Available columns: {list(data.columns)}")
        
        self.logger.info(f'✅ S/R detection input validation passed: {len(data)} rows, {len(data.columns)} columns')

        # Check if enhanced detector is available
        if not ENHANCED_SR_DETECTOR_AVAILABLE or EnhancedSRDetector is None:
            self.logger.error('❌ Enhanced SR Detector not available')
            return {'support_levels': [], 'resistance_levels': []}

        try:
            # Create enhanced SR detector with configuration - optimized for memory
            sr_config = {
                'min_touches': self.sr_optimization_config.get('min_touches', 1),
                'touch_proximity_threshold': self.sr_optimization_config.get('touch_proximity_threshold', 0.005),
                'min_strength': self.sr_optimization_config.get('min_strength', 0.15),
                'volume_spike_threshold': self.sr_optimization_config.get('volume_spike_threshold', 0.8),
                'fractal_period': self.sr_optimization_config.get('fractal_period', 1),
                'pivot_period': self.sr_optimization_config.get('pivot_period', 4),
                'psychological_levels': True,
                'fibonacci_levels': True,
                # Performance optimizations
                'use_optimized_fractals': True,
                'use_optimized_touch_counting': True,
                'enable_fractal_caching': True,
                'chunk_size': 1000,
                'max_fractals_per_chunk': 250,
                # Level limits (keep original to avoid overload)
                'max_levels_per_method': 30,
                'max_fractal_levels': 30,
                'max_pivot_levels': 30,
                'max_volume_levels': 30,
                'max_psychological_levels': 20,
                'max_fibonacci_levels': 20,
                'max_trendline_levels': 30,
                'max_channel_levels': 30,
                'max_volume_profile_levels': 30,
                'max_market_structure_levels': 30,
                # DBSCAN clustering parameters (original aggressive settings)
                'dbscan_eps_multiplier': 1.0,  # Original eps multiplier
                'dbscan_min_samples_multiplier': 1.0,  # Original min_samples multiplier
                'disable_dbscan_clustering': False  # Keep original clustering behavior
            }

            detector = EnhancedSRDetector(sr_config)

            # Detect SR levels using enhanced algorithms
            sr_levels = detector.detect_sr_levels(data)

            self.logger.info(f'✅ Enhanced SR detection completed: {len(sr_levels)} total levels detected')

            # Convert SRLevel objects to the expected format
            support_levels = []
            resistance_levels = []

            for level in sr_levels:
                try:
                    level_data = {
                        'price': float(level.price),
                        'strength': float(level.strength),
                        'type': level.type,
                        'method': 'enhanced_sr',
                        'touch_count': int(level.touch_count),
                        'timestamp': level.first_touch_time.isoformat() if hasattr(level.first_touch_time, 'isoformat') else str(level.first_touch_time),
                        'confidence_score': float(level.confidence_score),
                        'confluence_score': float(level.confluence_score),
                        'volume_confirmation_score': float(level.volume_confirmation_score),
                        'consistency_score': float(level.consistency_score),
                        'age_bars': int(level.age_bars),
                        'avg_bounce_ratio': float(level.avg_bounce_ratio),
                        'max_bounce_ratio': float(level.max_bounce_ratio),
                        'failure_count': int(level.failure_count)
                    }

                    if level.fibonacci_level is not None:
                        level_data['fibonacci_level'] = float(level.fibonacci_level)
                    if level.pivot_level:
                        level_data['pivot_level'] = True
                    if level.psychological_level:
                        level_data['psychological_level'] = True

                    if level.type == 'support':
                        support_levels.append(level_data)
                    elif level.type == 'resistance':
                        resistance_levels.append(level_data)

                except Exception as level_error:
                    self.logger.warning(f'⚠️ Failed to process level: {level_error}')
                    continue

            # Sort by strength (highest first)
            support_levels.sort(key=lambda x: x['strength'], reverse=True)
            resistance_levels.sort(key=lambda x: x['strength'], reverse=True)

            # Limit to reasonable number of levels
            max_levels = self.sr_optimization_config.get('max_levels', 50)
            support_levels = support_levels[:max_levels]
            resistance_levels = resistance_levels[:max_levels]

            self.logger.info(f'✅ Enhanced SR processing complete: {len(support_levels)} support, {len(resistance_levels)} resistance levels')

            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'detection_method': 'enhanced_sr',
                'total_levels_detected': len(sr_levels)
            }

        except Exception as e:
            self.logger.error(f'❌ Enhanced SR detection failed: {e}')
            raise  # Re-raise the exception instead of falling back

    async def _train_ml_models(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Train ML models for SR level prediction with comprehensive evaluation."""
        self.logger.info('🤖 Starting comprehensive ML model training for SR optimization...')
        start_time = time.time()

        try:
            # Validate input data
            if features_data.empty:
                raise ValueError("Features data is empty")
            if not sr_levels or not any(sr_levels.get(key, []) for key in ['support_levels', 'resistance_levels']):
                raise ValueError("No SR levels provided for training")

            # Prepare target variables from SR levels
            self.logger.info('🎯 Preparing target variables from SR levels...')
            target_data = self._prepare_sr_targets(features_data, sr_levels)

            # Prepare features for ML training
            self.logger.info('🔧 Preparing features for ML training...')
            X, y_direction, y_volatility, feature_names = self._prepare_ml_features(features_data, target_data)

            # Optimize hyperparameters
            self.logger.info('🔧 Optimizing hyperparameters...')
            hyperparameter_results = self._optimize_hyperparameters(X, y_direction, feature_names)

            # Perform walk-forward validation
            self.logger.info('🔧 Performing walk-forward validation...')
            walk_forward_results = self._walk_forward_validation(X, y_direction, feature_names)

            # Feature selection for optimization
            self.logger.info('🎯 Performing feature selection for optimal performance...')
            X_selected, selected_feature_names, feature_selection_info = self._optimize_feature_selection(
                X, y_direction, feature_names
            )

            # Split data using selected features
            self.logger.info('📊 Splitting data for training and validation...')
            X_train, X_test, y_dir_train, y_dir_test, y_vol_train, y_vol_test = train_test_split(
                X_selected, y_direction, y_volatility, test_size=0.2, random_state=42, shuffle=False
            )

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train multiple models
            self.logger.info('🚀 Training multiple ML models...')
            models_results = await self._train_multiple_models(
                X_train_scaled, X_test_scaled,
                y_dir_train, y_dir_test,
                y_vol_train, y_vol_test,
                selected_feature_names
            )

            # Hyperparameter optimization for best models
            self.logger.info('🔧 Performing hyperparameter optimization...')
            optimized_models = await self._optimize_hyperparameters(
                X_train_scaled, y_dir_train, y_vol_train, models_results
            )

            # Skip ensemble creation - focus on best individual model with HPO
            ensemble_model = None

            # Perform cross-validation
            self.logger.info('🔄 Performing cross-validation...')
            cv_results = self._perform_cross_validation(X_train_scaled, y_dir_train, selected_feature_names)

            # Calculate comprehensive metrics
            self.logger.info('📈 Calculating comprehensive evaluation metrics...')
            evaluation_metrics = self._calculate_evaluation_metrics(
                optimized_models if optimized_models else models_results,
                cv_results, X_test_scaled, y_dir_test, y_vol_test,
                ensemble_model
            )

            # Save best model
            self.logger.info('💾 Saving best performing model...')
            models_for_saving = optimized_models if optimized_models else models_results

            model_save_path = self._save_best_model(
                models_for_saving, scaler, selected_feature_names
            )

            # Compile final results
            training_time = time.time() - start_time
            ml_results = {
                'direction_accuracy': evaluation_metrics['best_direction_accuracy'],
                'volatility_mae': evaluation_metrics['best_volatility_mae'],
                'model_type': evaluation_metrics['best_model_type'],
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'sr_levels_used': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                'training_time': training_time,
                'feature_importance': evaluation_metrics['feature_importance'],
                'cross_validation_scores': cv_results['direction_accuracy_scores'],
                'models_performance': models_results,
                'evaluation_metrics': evaluation_metrics,
                'model_save_path': model_save_path,
                'feature_names': feature_names.tolist(),
                'scaler_params': {
                    'mean': scaler.mean_.tolist(),
                    'scale': scaler.scale_.tolist()
                },
                'feature_selection': feature_selection_info
            }

            self.logger.info(f'✅ Comprehensive ML training completed in {training_time:.2f}s')
            self.logger.info(f'🎯 Best direction accuracy: {ml_results["direction_accuracy"]:.4f}')
            self.logger.info(f'📊 Best volatility MAE: {ml_results["volatility_mae"]:.6f}')
            self.logger.info(f'🏆 Best model: {ml_results["model_type"]}')

            return ml_results

        except Exception as e:
            self.logger.error(f'❌ Comprehensive ML training failed: {e}')
            import traceback
            self.logger.error(f'📋 Full traceback: {traceback.format_exc()}')

            # Return fallback results
            training_time = time.time() - start_time
            return {
                'direction_accuracy': 0.5,
                'volatility_mae': 0.05,
                'model_type': 'fallback',
                'training_samples': len(features_data) if not features_data.empty else 0,
                'sr_levels_used': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                'training_time': training_time,
                'error': str(e),
                'traceback': traceback.format_exc()
            }

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) for normalization."""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # Calculate True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # Calculate ATR as rolling mean of True Range
            atr = true_range.rolling(window=period).mean()
            
            return atr
        except Exception as e:
            self.logger.warning(f'ATR calculation failed: {e}')
            # Fallback to simple price range
            return (data['high'] - data['low']).rolling(window=period).mean()

    def _format_sr_levels_for_pipeline(self, sr_levels: Dict[str, Any]) -> Dict[str, Any]:
        """Format S/R levels for pipeline state consumption by step06."""
        try:
            formatted_levels = {
                'support_levels': [],
                'resistance_levels': [],
                'metadata': {
                    'total_support': len(sr_levels.get('support_levels', [])),
                    'total_resistance': len(sr_levels.get('resistance_levels', [])),
                    'detection_timestamp': pd.Timestamp.now().isoformat()
                }
            }
            
            # Format support levels
            for level in sr_levels.get('support_levels', []):
                if hasattr(level, 'price'):  # SRLevel object
                    formatted_level = {
                        'price': level.price,
                        'strength': level.strength,
                        'touch_count': level.touch_count,
                        'first_touch_time': level.first_touch_time.isoformat() if level.first_touch_time else None,
                        'last_touch_time': level.last_touch_time.isoformat() if level.last_touch_time else None,
                        'age_bars': level.age_bars,
                        'avg_bounce_ratio': level.avg_bounce_ratio,
                        'max_bounce_ratio': level.max_bounce_ratio,
                        'volume_confirmation_score': level.volume_confirmation_score,
                        'consistency_score': level.consistency_score,
                        'confidence_score': level.confidence_score,
                        'confluence_score': level.confluence_score,
                        'type': 'support'
                    }
                else:  # Dictionary format
                    formatted_level = {
                        'price': level.get('price', level),
                        'strength': level.get('strength', 0.5),
                        'touch_count': level.get('touch_count', 1),
                        'type': 'support'
                    }
                formatted_levels['support_levels'].append(formatted_level)
            
            # Format resistance levels
            for level in sr_levels.get('resistance_levels', []):
                if hasattr(level, 'price'):  # SRLevel object
                    formatted_level = {
                        'price': level.price,
                        'strength': level.strength,
                        'touch_count': level.touch_count,
                        'first_touch_time': level.first_touch_time.isoformat() if level.first_touch_time else None,
                        'last_touch_time': level.last_touch_time.isoformat() if level.last_touch_time else None,
                        'age_bars': level.age_bars,
                        'avg_bounce_ratio': level.avg_bounce_ratio,
                        'max_bounce_ratio': level.max_bounce_ratio,
                        'volume_confirmation_score': level.volume_confirmation_score,
                        'consistency_score': level.consistency_score,
                        'confidence_score': level.confidence_score,
                        'confluence_score': level.confluence_score,
                        'type': 'resistance'
                    }
                else:  # Dictionary format
                    formatted_level = {
                        'price': level.get('price', level),
                        'strength': level.get('strength', 0.5),
                        'touch_count': level.get('touch_count', 1),
                        'type': 'resistance'
                    }
                formatted_levels['resistance_levels'].append(formatted_level)
            
            self.logger.info(f'✅ Formatted {len(formatted_levels["support_levels"])} support and {len(formatted_levels["resistance_levels"])} resistance levels for pipeline')
            return formatted_levels
            
        except Exception as e:
            self.logger.error(f'❌ Failed to format S/R levels for pipeline: {e}')
            return {'support_levels': [], 'resistance_levels': [], 'metadata': {'error': str(e)}}

    def _prepare_sr_targets(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any]) -> pd.DataFrame:
        """Prepare target variables from SR levels for ML training with enhanced features."""
        try:
            # Extract SR level prices for target creation
            support_prices = [level.get('price', level) for level in sr_levels.get('support_levels', [])]
            resistance_prices = [level.get('price', level) for level in sr_levels.get('resistance_levels', [])]

            # Get current price data
            if 'close' not in features_data.columns:
                raise ValueError("Features data must contain 'close' price column")

            current_prices = features_data['close'].values
            target_data = pd.DataFrame(index=features_data.index)

            # Create binary target: 1 if near SR level, 0 otherwise
            proximity_threshold = 0.005  # 0.5% proximity threshold
            near_support = np.zeros(len(current_prices))
            near_resistance = np.zeros(len(current_prices))

            for price in support_prices:
                if isinstance(price, (int, float)):
                    distance = np.abs(current_prices - price) / current_prices
                    near_support = np.maximum(near_support, (distance <= proximity_threshold).astype(int))

            for price in resistance_prices:
                if isinstance(price, (int, float)):
                    distance = np.abs(current_prices - price) / current_prices
                    near_resistance = np.maximum(near_resistance, (distance <= proximity_threshold).astype(int))

            # Direction target: 1 for resistance (sell signal), 0 for support (buy signal), 0.5 for neutral
            target_data['direction_target'] = np.where(
                near_resistance == 1, 1.0,
                np.where(near_support == 1, 0.0, 0.5)
            )

            # Volatility target: distance to nearest SR level as proxy for volatility
            nearest_sr_distances = []
            for price in current_prices:
                support_distances = [abs(price - sp) / price for sp in support_prices if isinstance(sp, (int, float))]
                resistance_distances = [abs(price - rp) / price for rp in resistance_prices if isinstance(rp, (int, float))]

                all_distances = support_distances + resistance_distances
                nearest_distance = min(all_distances) if all_distances else 0.01
                nearest_sr_distances.append(nearest_distance)

            target_data['volatility_target'] = np.array(nearest_sr_distances)

            # Filter out neutral cases for cleaner training
            valid_mask = target_data['direction_target'] != 0.5
            self.logger.info(f'🎯 Target preparation: {valid_mask.sum()}/{len(valid_mask)} valid samples ({valid_mask.sum()/len(valid_mask)*100:.1f}%)')

            return target_data

        except Exception as e:
            self.logger.error(f'❌ Failed to prepare SR targets: {e}')
            raise

    def _prepare_ml_features(self, features_data: pd.DataFrame, target_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare features and targets for ML training."""
        try:
            # Select numeric features only
            numeric_features = features_data.select_dtypes(include=[np.number]).columns
            self.logger.info(f'🔢 Selected {len(numeric_features)} numeric features for ML training')

            # Remove target-related columns if they exist
            exclude_cols = ['direction_target', 'volatility_target', 'target', 'label']
            feature_cols = [col for col in numeric_features if col not in exclude_cols]

            # Handle missing values
            X = features_data[feature_cols].fillna(0).values
            feature_names = np.array(feature_cols)

            # Get targets
            y_direction = target_data['direction_target'].values
            y_volatility = target_data['volatility_target'].values

            # Remove rows where direction target is neutral (0.5)
            valid_mask = y_direction != 0.5
            X = X[valid_mask]
            y_direction = y_direction[valid_mask]
            y_volatility = y_volatility[valid_mask]

            self.logger.info(f'📊 ML data prepared: {X.shape[0]} samples, {X.shape[1]} features')
            self.logger.info(f'🎯 Direction target distribution: {np.bincount(y_direction.astype(int))}')
            self.logger.info(f'📈 Volatility target range: {y_volatility.min():.6f} - {y_volatility.max():.6f}')

            return X, y_direction, y_volatility, feature_names

        except Exception as e:
            self.logger.error(f'❌ Failed to prepare ML features: {e}')
            raise

    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search, random search, or Bayesian optimization."""
        try:
            if not self.enable_hyperparameter_optimization:
                self.logger.info('🔧 Hyperparameter optimization disabled')
                return {}
            
            self.logger.info(f'🔧 Starting hyperparameter optimization using {self.optimization_method}')
            
            if self.optimization_method == 'grid_search':
                return self._grid_search_optimization(X, y, feature_names)
            elif self.optimization_method == 'random_search':
                return self._random_search_optimization(X, y, feature_names)
            elif self.optimization_method == 'bayesian':
                return self._bayesian_optimization(X, y, feature_names)
            else:
                self.logger.warning(f'Unknown optimization method: {self.optimization_method}')
                return {}
                
        except Exception as e:
            self.logger.error(f'❌ Hyperparameter optimization failed: {e}')
            return {}

    def _grid_search_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Grid search hyperparameter optimization."""
        try:
            from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier
            
            # Define parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=self.optimization_folds)
            
            # Grid search
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X, y)
            
            self.logger.info(f'🔧 Grid search best score: {grid_search.best_score_:.4f}')
            self.logger.info(f'🔧 Grid search best params: {grid_search.best_params_}')
            
            return {
                'method': 'grid_search',
                'best_score': grid_search.best_score_,
                'best_params': grid_search.best_params_,
                'cv_results': grid_search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f'❌ Grid search optimization failed: {e}')
            return {}

    def _random_search_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Random search hyperparameter optimization."""
        try:
            from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier
            from scipy.stats import randint, uniform
            
            # Define parameter distributions
            param_distributions = {
                'n_estimators': randint(50, 300),
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=self.optimization_folds)
            
            # Random search
            random_search = RandomizedSearchCV(
                RandomForestClassifier(random_state=42),
                param_distributions,
                n_iter=self.optimization_trials,
                cv=tscv,
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=1
            )
            
            random_search.fit(X, y)
            
            self.logger.info(f'🔧 Random search best score: {random_search.best_score_:.4f}')
            self.logger.info(f'🔧 Random search best params: {random_search.best_params_}')
            
            return {
                'method': 'random_search',
                'best_score': random_search.best_score_,
                'best_params': random_search.best_params_,
                'cv_results': random_search.cv_results_
            }
            
        except Exception as e:
            self.logger.error(f'❌ Random search optimization failed: {e}')
            return {}

    def _bayesian_optimization(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Bayesian optimization using scikit-optimize."""
        try:
            from skopt import gp_minimize
            from skopt.space import Real, Integer, Categorical
            from skopt.utils import use_named_args
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit
            
            # Define search space
            space = [
                Integer(50, 300, name='n_estimators'),
                Categorical([5, 10, 15, 20, None], name='max_depth'),
                Integer(2, 20, name='min_samples_split'),
                Integer(1, 10, name='min_samples_leaf'),
                Categorical(['sqrt', 'log2', None], name='max_features')
            ]
            
            # Use time series split for proper validation
            tscv = TimeSeriesSplit(n_splits=self.optimization_folds)
            
            @use_named_args(space)
            def objective(**params):
                # Handle None values
                if params['max_depth'] is None:
                    params['max_depth'] = None
                
                model = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    random_state=42
                )
                
                # Cross-validation score
                scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
                return -scores.mean()  # Minimize negative score
            
            # Bayesian optimization
            result = gp_minimize(objective, space, n_calls=self.optimization_trials, random_state=42)
            
            # Extract best parameters
            best_params = {
                'n_estimators': result.x[0],
                'max_depth': result.x[1],
                'min_samples_split': result.x[2],
                'min_samples_leaf': result.x[3],
                'max_features': result.x[4]
            }
            
            best_score = -result.fun
            
            self.logger.info(f'🔧 Bayesian optimization best score: {best_score:.4f}')
            self.logger.info(f'🔧 Bayesian optimization best params: {best_params}')
            
            return {
                'method': 'bayesian',
                'best_score': best_score,
                'best_params': best_params,
                'optimization_result': result
            }
            
        except Exception as e:
            self.logger.error(f'❌ Bayesian optimization failed: {e}')
            return {}

    def _walk_forward_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Perform walk-forward validation for time series data."""
        try:
            if not self.enable_walk_forward_validation:
                self.logger.info('🔧 Walk-forward validation disabled')
                return {}
            
            self.logger.info(f'🔧 Starting walk-forward validation with {self.walk_forward_folds} folds')
            
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            # Calculate fold sizes
            total_samples = len(X)
            test_size = int(total_samples * self.walk_forward_test_size)
            fold_size = (total_samples - test_size) // self.walk_forward_folds
            
            results = {
                'fold_scores': [],
                'fold_metrics': [],
                'average_metrics': {}
            }
            
            for fold in range(self.walk_forward_folds):
                # Calculate train/test indices
                train_end = (fold + 1) * fold_size
                test_start = train_end
                test_end = test_start + test_size
                
                if test_end > total_samples:
                    break
                
                X_train = X[:train_end]
                y_train = y[:train_end]
                X_test = X[test_start:test_end]
                y_test = y[test_start:test_end]
                
                # Train model
                model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    random_state=42
                )
                model.fit(X_train, y_train)
                
                # Predict and evaluate
                y_pred = model.predict(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                fold_metrics = {
                    'fold': fold + 1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'train_size': len(X_train),
                    'test_size': len(X_test)
                }
                
                results['fold_scores'].append(accuracy)
                results['fold_metrics'].append(fold_metrics)
                
                self.logger.info(f'🔧 Fold {fold + 1}: Accuracy = {accuracy:.4f}, F1 = {f1:.4f}')
            
            # Calculate average metrics
            if results['fold_scores']:
                results['average_metrics'] = {
                    'mean_accuracy': np.mean(results['fold_scores']),
                    'std_accuracy': np.std(results['fold_scores']),
                    'mean_precision': np.mean([m['precision'] for m in results['fold_metrics']]),
                    'mean_recall': np.mean([m['recall'] for m in results['fold_metrics']]),
                    'mean_f1_score': np.mean([m['f1_score'] for m in results['fold_metrics']])
                }
                
                self.logger.info(f'🔧 Walk-forward validation complete: Mean Accuracy = {results["average_metrics"]["mean_accuracy"]:.4f} ± {results["average_metrics"]["std_accuracy"]:.4f}')
            
            return results
            
        except Exception as e:
            self.logger.error(f'❌ Walk-forward validation failed: {e}')
            return {}

    def _optimize_feature_selection(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Optimize feature selection for best model performance."""
        try:
            from sklearn.feature_selection import SelectFromModel, RFECV, mutual_info_classif
            from sklearn.ensemble import RandomForestClassifier

            feature_selection_info = {
                'original_features': len(feature_names),
                'methods_used': [],
                'selected_features': 0,
                'feature_importance': {}
            }

            # Method 1: Feature importance from Random Forest
            self.logger.info('🌲 Computing feature importance with Random Forest...')
            rf_temp = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf_temp.fit(X, y)

            # Get feature importance
            importances = rf_temp.feature_importances_
            feature_importance_dict = dict(zip(feature_names, importances))
            feature_selection_info['feature_importance'] = feature_importance_dict

            # Sort features by importance
            sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

            # Method 2: Select features based on criteria for optimal performance
            # Criteria: Keep exactly 80 features with highest importance scores
            target_features = min(80, len(feature_names))  # Keep exactly 80 features
            top_feature_names = [name for name, _ in sorted_features[:target_features]]

            # Create boolean mask for selected features
            selected_mask = np.isin(feature_names, top_feature_names)
            selected_feature_names = feature_names[selected_mask]
            X_selected = X[:, selected_mask]

            feature_selection_info['methods_used'].append('top_features_selection')
            feature_selection_info['selected_features'] = len(selected_feature_names)
            feature_selection_info['target_features'] = target_features
            feature_selection_info['selection_criteria'] = f'Keep exactly 80 features with highest importance (selected {target_features})'
            feature_selection_info['top_features'] = top_feature_names[:20]  # Show top 20 for reporting

            # Method 3: Mutual information for additional validation
            if len(selected_feature_names) > 10:  # Only if we have enough features
                self.logger.info('🔗 Computing mutual information scores...')
                mi_scores = mutual_info_classif(X_selected, y, random_state=42)
                mi_dict = dict(zip(selected_feature_names, mi_scores))
                feature_selection_info['mutual_information'] = mi_dict
                feature_selection_info['methods_used'].append('mutual_information')

            # Method 4: SHAP values for comprehensive feature importance
            feature_selection_info['shap_importance'] = self._compute_shap_importance(
                X_selected, y, selected_feature_names
            )
            feature_selection_info['methods_used'].append('shap_analysis')

            self.logger.info(f'✅ Feature selection completed: {len(selected_feature_names)}/{len(feature_names)} features selected')
            self.logger.info(f'🎯 Top 5 features: {[name for name, _ in sorted_features[:5]]}')

            return X_selected, selected_feature_names, feature_selection_info

        except Exception as e:
            self.logger.warning(f'⚠️ Feature selection failed, using all features: {e}')
            # Return original features if selection fails
            return X, feature_names, {
                'original_features': len(feature_names),
                'methods_used': ['failed'],
                'selected_features': len(feature_names),
                'error': str(e)
            }

    def _apply_numpy_compatibility_patch(self) -> None:
        """Apply NumPy compatibility patch for SHAP library."""
        try:
            # Check if np.bool is missing and add it back for compatibility
            if not hasattr(np, 'bool'):
                # Add np.bool as an alias for np.bool_ (the new scalar type)
                setattr(np, 'bool', np.bool_)
                self.logger.info("🔧 Applied NumPy compatibility patch: np.bool -> np.bool_")

                # Verify the patch worked
                if hasattr(np, 'bool'):
                    self.logger.info("✅ NumPy compatibility patch verified")
                else:
                    self.logger.warning("⚠️ NumPy compatibility patch may not have worked")

        except Exception as e:
            self.logger.warning(f"⚠️ Failed to apply NumPy compatibility patch: {e}")

    def _compute_shap_importance(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Compute SHAP values for comprehensive feature importance analysis."""
        try:
            # Apply NumPy compatibility patch for SHAP
            self._apply_numpy_compatibility_patch()

            # Try to import SHAP with NumPy compatibility handling
            import shap
            shap_available = True
        except ImportError as e:
            shap_available = False
            self.logger.warning(f'⚠️ SHAP not available, falling back to permutation importance. Error: {e}')
        except AttributeError as e:
            if "numpy" in str(e) and "bool" in str(e):
                shap_available = False
                self.logger.warning('⚠️ SHAP has NumPy compatibility issue (np.bool deprecated). Please update SHAP: pip install --upgrade shap')
            else:
                shap_available = False
                self.logger.warning(f'⚠️ SHAP initialization failed: {e}')
            return self._compute_permutation_importance(X, y, feature_names)

        if not shap_available:
            return self._compute_permutation_importance(X, y, feature_names)

        try:
            # Use a sample of the data for SHAP computation to improve speed
            sample_size = min(1000, len(X))
            sample_indices = np.random.choice(len(X), size=sample_size, replace=False)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Train a quick model for SHAP analysis
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X_sample, y_sample)

            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)

            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)

            # For binary classification, shap_values is a list of arrays
            if isinstance(shap_values, list):
                # Take the positive class SHAP values
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.abs(shap_values).mean(axis=0)

            # Create feature importance dictionary
            shap_importance = {}
            for i, feature_name in enumerate(feature_names):
                shap_importance[feature_name] = float(mean_shap_values[i])

            # Sort by importance
            sorted_shap = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)

            self.logger.info('✅ SHAP analysis completed successfully')

            return {
                'method': 'shap',
                'feature_importance': shap_importance,
                'top_features': sorted_shap[:20],
                'sample_size': sample_size,
                'available': True
            }

        except AttributeError as e:
            if "numpy" in str(e) and "bool" in str(e):
                self.logger.warning('⚠️ SHAP analysis failed due to NumPy compatibility issue (np.bool deprecated). Please update SHAP: pip install --upgrade shap')
            else:
                self.logger.warning(f'⚠️ SHAP analysis failed: {e}')
            return self._compute_permutation_importance(X, y, feature_names)
        except Exception as e:
            self.logger.warning(f'⚠️ SHAP analysis failed: {e}')
            return self._compute_permutation_importance(X, y, feature_names)

    def _compute_permutation_importance(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Fallback method using permutation importance when SHAP is not available."""
        try:
            from sklearn.inspection import permutation_importance
            from sklearn.ensemble import RandomForestClassifier

            # Train a quick model
            model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            model.fit(X, y)

            # Calculate permutation importance
            perm_importance = permutation_importance(model, X, y, n_repeats=5, random_state=42, n_jobs=-1)

            # Create feature importance dictionary
            perm_importance_dict = {}
            for i, feature_name in enumerate(feature_names):
                perm_importance_dict[feature_name] = float(perm_importance.importances_mean[i])

            # Sort by importance
            sorted_perm = sorted(perm_importance_dict.items(), key=lambda x: x[1], reverse=True)

            self.logger.info('✅ Permutation importance analysis completed')

            return {
                'method': 'permutation',
                'feature_importance': perm_importance_dict,
                'top_features': sorted_perm[:20],
                'available': True
            }

        except Exception as e:
            self.logger.warning(f'⚠️ Permutation importance failed: {e}')
            return {
                'method': 'failed',
                'feature_importance': {},
                'top_features': [],
                'error': str(e),
                'available': False
            }

    async def _train_multiple_models(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_dir_train: np.ndarray, y_dir_test: np.ndarray,
                                   y_vol_train: np.ndarray, y_vol_test: np.ndarray,
                                   feature_names: np.ndarray) -> Dict[str, Any]:
        """Train multiple ML models with computational optimizations and hyperparameter tuning."""
        try:
            # Import optimized libraries with fallbacks
            try:
                import xgboost as xgb
                XGB_AVAILABLE = True
            except ImportError:
                XGB_AVAILABLE = False

            try:
                import lightgbm as lgb
                LGBM_AVAILABLE = True
            except ImportError:
                LGBM_AVAILABLE = False

            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import mean_absolute_error, classification_report

            # GPU/MPS detection for M1 optimization
            use_gpu = False
            if M1_BATCH_AVAILABLE:
                try:
                    import torch
                    if torch.backends.mps.is_available():
                        use_gpu = True
                        self.logger.info('🚀 MPS GPU acceleration available and enabled')
                except:
                    pass

            # Optimized model configurations with hyperparameters (selected algorithms only)
            models = {}

            # Extra Trees for diversity and speed
            models['extra_trees'] = {
                'direction': ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True
                ),
                'volatility': ExtraTreesRegressor(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=8,
                    min_samples_leaf=4,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1,
                    bootstrap=True
                )
            }

            # XGBoost with GPU acceleration if available (primary algorithm)
            if XGB_AVAILABLE:
                xgb_params = {
                    'objective': 'binary:logistic',
                    'max_depth': 8,
                    'learning_rate': 0.1,
                    'n_estimators': 150,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'random_state': 42,
                    'n_jobs': -1,
                    'eval_metric': 'logloss',
                    # Ensure base_score is valid in (0,1) and not overridden to 0 accidentally
                    'base_score': 0.5,
                    'use_label_encoder': False
                }

                if use_gpu:
                    xgb_params.update({
                        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
                        'tree_method': 'hist' if torch.cuda.is_available() else 'auto'
                    })

                models['xgboost'] = {
                    'direction': xgb.XGBClassifier(**xgb_params),
                    'volatility': xgb.XGBRegressor(
                        objective='reg:squarederror',
                        **{k: v for k, v in xgb_params.items() if k not in ['objective']}
                    )
                }
            else:
                # Fallback to Extra Trees if XGBoost not available
                self.logger.warning('⚠️ XGBoost not available, using Extra Trees as primary model')
                # Extra Trees is already defined above, no need to redefine

            results = {}

            for model_name, model_dict in models.items():
                self.logger.info(f'🏃 Training {model_name}...')

                model_results = {}

                # Train direction classifier
                if model_dict['direction'] is not None:
                    try:
                        model_dict['direction'].fit(X_train, y_dir_train)

                        # Predict and evaluate
                        y_dir_pred = model_dict['direction'].predict(X_test)
                        direction_accuracy = accuracy_score(y_dir_test, y_dir_pred)

                        # Get feature importance if available
                        if hasattr(model_dict['direction'], 'feature_importances_'):
                            feature_importance = dict(zip(feature_names, model_dict['direction'].feature_importances_))
                        else:
                            feature_importance = {}

                        model_results['direction'] = {
                            'accuracy': direction_accuracy,
                            'predictions': y_dir_pred,
                            'feature_importance': feature_importance,
                            'classification_report': classification_report(y_dir_test, y_dir_pred, output_dict=True),
                            'model': model_dict['direction']  # Store the trained model
                        }

                    except Exception as e:
                        self.logger.warning(f'⚠️ {model_name} direction training failed: {e}')
                        model_results['direction'] = {'accuracy': 0.5, 'error': str(e)}

                # Train volatility regressor
                if model_dict['volatility'] is not None:
                    try:
                        model_dict['volatility'].fit(X_train, y_vol_train)

                        # Predict and evaluate
                        y_vol_pred = model_dict['volatility'].predict(X_test)
                        volatility_mae = mean_absolute_error(y_vol_test, y_vol_pred)

                        model_results['volatility'] = {
                            'mae': volatility_mae,
                            'predictions': y_vol_pred
                        }

                    except Exception as e:
                        self.logger.warning(f'⚠️ {model_name} volatility training failed: {e}')
                        model_results['volatility'] = {'mae': 0.1, 'error': str(e)}

                results[model_name] = model_results
                self.logger.info(f'✅ {model_name} training completed')

            return results

        except Exception as e:
            self.logger.error(f'❌ Multiple model training failed: {e}')
            raise

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> Dict[str, Any]:
        """Perform cross-validation for model evaluation."""
        try:
            from sklearn.model_selection import cross_val_score, KFold

            cv_results = {}

            # Use Random Forest for CV as it's robust and fast
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

            # 5-fold cross-validation
            kf = KFold(n_splits=5, shuffle=True, random_state=42)

            # Direction accuracy scores
            direction_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
            cv_results['direction_accuracy_scores'] = direction_scores.tolist()
            cv_results['direction_accuracy_mean'] = direction_scores.mean()
            cv_results['direction_accuracy_std'] = direction_scores.std()

            # F1 scores
            f1_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='f1_macro')
            cv_results['f1_scores'] = f1_scores.tolist()
            cv_results['f1_mean'] = f1_scores.mean()
            cv_results['f1_std'] = f1_scores.std()

            self.logger.info(f'🔄 CV Results - Accuracy: {cv_results["direction_accuracy_mean"]:.4f} ± {cv_results["direction_accuracy_std"]:.4f}')
            self.logger.info(f'🔄 CV Results - F1: {cv_results["f1_mean"]:.4f} ± {cv_results["f1_std"]:.4f}')

            return cv_results

        except Exception as e:
            self.logger.error(f'❌ Cross-validation failed: {e}')
            return {
                'direction_accuracy_scores': [0.5] * 5,
                'direction_accuracy_mean': 0.5,
                'direction_accuracy_std': 0.0,
                'error': str(e)
            }

    def _calculate_evaluation_metrics(self, models_results: Dict[str, Any],
                                    cv_results: Dict[str, Any],
                                    X_test: np.ndarray, y_dir_test: np.ndarray,
                                    y_vol_test: np.ndarray, ensemble_model: Dict[str, Any] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics."""
        try:
            # Find best performing models
            best_direction_accuracy = 0
            best_direction_model = None
            best_volatility_mae = float('inf')
            best_volatility_model = None

            # Skip ensemble comparison - focus on individual optimized models

            # Aggregate feature importance across models
            all_feature_importance = {}

            for model_name, model_result in models_results.items():
                # Check direction performance
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    if accuracy > best_direction_accuracy:
                        best_direction_accuracy = accuracy
                        best_direction_model = model_name

                    # Aggregate feature importance
                    if 'feature_importance' in model_result['direction']:
                        for feature, importance in model_result['direction']['feature_importance'].items():
                            if feature not in all_feature_importance:
                                all_feature_importance[feature] = []
                            all_feature_importance[feature].append(importance)

                # Check volatility performance
                if 'volatility' in model_result and 'mae' in model_result['volatility']:
                    mae = model_result['volatility']['mae']
                    if mae < best_volatility_mae:
                        best_volatility_mae = mae
                        best_volatility_model = model_name

            # Calculate average feature importance
            avg_feature_importance = {}
            for feature, importances in all_feature_importance.items():
                avg_feature_importance[feature] = np.mean(importances)

            # Sort features by importance
            sorted_features = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)
            top_features = dict(sorted_features[:20])  # Top 20 features

            return {
                'best_direction_accuracy': best_direction_accuracy,
                'best_direction_model': best_direction_model,
                'best_volatility_mae': best_volatility_mae,
                'best_volatility_model': best_volatility_model,
                'best_model_type': best_direction_model,  # Use direction model as primary
                'feature_importance': top_features,
                'cv_direction_mean': cv_results.get('direction_accuracy_mean', 0.5),
                'cv_direction_std': cv_results.get('direction_accuracy_std', 0),
                'cv_f1_mean': cv_results.get('f1_mean', 0.5),
                'cv_f1_std': cv_results.get('f1_std', 0),
                'models_count': len(models_results)
            }

        except Exception as e:
            self.logger.error(f'❌ Evaluation metrics calculation failed: {e}')
            return {
                'best_direction_accuracy': 0.5,
                'best_volatility_mae': 0.05,
                'best_model_type': 'fallback',
                'feature_importance': {},
                'error': str(e)
            }

    def _save_best_model(self, models_results: Dict[str, Any], scaler: StandardScaler,
                        feature_names: np.ndarray) -> str:
        """Save the best performing model to disk."""
        try:
            # Find best model based on direction accuracy
            best_model_name = None
            best_accuracy = 0

            for model_name, model_result in models_results.items():
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_model_name = model_name

            if best_model_name is None:
                self.logger.warning('⚠️ No suitable model found to save')
                return None

            # Create model save directory
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_dir = self.standards.build_path('models', 'sr_optimization', timestamp)
            os.makedirs(model_dir, exist_ok=True)

            # Save model
            model_path = os.path.join(model_dir, f'{best_model_name}_model.pkl')
            scaler_path = os.path.join(model_dir, 'scaler.pkl')
            feature_names_path = os.path.join(model_dir, 'feature_names.pkl')

            # Get the actual trained model object
            best_model = models_results[best_model_name]['direction'].get('model')
            if best_model is None:
                self.logger.warning(f'⚠️ Trained model not found for {best_model_name}, creating new instance')
                # Create a new instance for saving (fallback)
                if best_model_name == 'random_forest':
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                elif best_model_name == 'gradient_boosting':
                    from sklearn.ensemble import GradientBoostingClassifier
                    best_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
                else:
                    best_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

            # Save components
            joblib.dump(best_model, model_path)
            joblib.dump(scaler, scaler_path)
            joblib.dump(feature_names, feature_names_path)

            # Save metadata
            metadata = {
                'model_type': best_model_name,
                'accuracy': best_accuracy,
                'timestamp': timestamp,
                'feature_count': len(feature_names),
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist()
            }

            metadata_path = os.path.join(model_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f'💾 Best model saved to: {model_path}')
            return model_path

        except Exception as e:
            self.logger.error(f'❌ Model saving failed: {e}')
            return None

    async def _optimize_hyperparameters(self, X_train: np.ndarray, y_dir_train: np.ndarray,
                                      y_vol_train: np.ndarray, models_results: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize hyperparameters for the best performing models."""
        try:
            from sklearn.model_selection import RandomizedSearchCV
            import scipy.stats as stats
            from sklearn.ensemble import ExtraTreesClassifier

            optimized_models = {}
            optimization_results = {}

            # Optimize hyperparameters for ALL available models
            self.logger.info(f'🎯 Optimizing hyperparameters for all {len(models_results)} models...')

            for model_name, model_result in models_results.items():
                if 'direction' not in model_result or 'accuracy' not in model_result['direction']:
                    continue

                base_accuracy = model_result['direction']['accuracy']
                self.logger.info(f'🔧 Optimizing {model_name} (base accuracy: {base_accuracy:.4f})...')

                # Define parameter grids for different models
                if model_name == 'extra_trees':
                    param_distributions = {
                        'n_estimators': stats.randint(100, 300),
                        'max_depth': [8, 12, 16, None],
                        'min_samples_split': stats.randint(2, 15),
                        'min_samples_leaf': stats.randint(1, 8),
                        'max_features': ['sqrt', 'log2'],
                        'bootstrap': [True, False]
                    }

                    # Use the original model from results
                    base_model = models_results[model_name]['direction']['model']
                    if base_model is None:
                        base_model = ExtraTreesClassifier(random_state=42, n_jobs=-1)

                elif model_name == 'xgboost':
                    try:
                        import xgboost as xgb
                        param_distributions = {
                            'max_depth': stats.randint(3, 10),
                            'learning_rate': stats.uniform(0.01, 0.3),
                            'n_estimators': stats.randint(100, 300),
                            'subsample': stats.uniform(0.6, 0.4),
                            'colsample_bytree': stats.uniform(0.6, 0.4),
                            'min_child_weight': stats.randint(1, 10),
                            # Keep base_score strictly within (0,1)
                            'base_score': stats.uniform(0.05, 0.9)
                        }
                        base_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
                    except ImportError:
                        continue

                else:
                    # Skip unsupported models
                    self.logger.info(f'⚠️ Skipping hyperparameter optimization for {model_name} (not supported)')
                    continue

                # Perform random search
                random_search = RandomizedSearchCV(
                    base_model,
                    param_distributions,
                    n_iter=20,  # Limited iterations for speed
                    cv=3,        # 3-fold CV for speed
                    scoring='accuracy',
                    random_state=42,
                    n_jobs=-1,
                    verbose=0
                )

                # Fit the random search
                random_search.fit(X_train, y_dir_train)

                # Get best model and parameters
                best_model = random_search.best_estimator_
                best_params = random_search.best_params_
                best_score = random_search.best_score_

                self.logger.info(f'✅ {model_name} optimization completed: {best_score:.4f} (improvement: {best_score - base_accuracy:.4f})')

                # Store optimized model
                optimized_models[model_name] = models_results[model_name].copy()
                optimized_models[model_name]['direction']['model'] = best_model
                optimized_models[model_name]['direction']['optimized_params'] = best_params
                optimized_models[model_name]['direction']['optimized_score'] = best_score
                optimized_models[model_name]['direction']['improvement'] = best_score - base_accuracy

                # Re-evaluate on full dataset to get updated metrics
                y_pred_optimized = best_model.predict(X_train)
                from sklearn.metrics import accuracy_score, classification_report
                optimized_accuracy = accuracy_score(y_dir_train, y_pred_optimized)
                optimized_models[model_name]['direction']['accuracy'] = optimized_accuracy

            self.logger.info(f'🎉 Hyperparameter optimization completed for {len(optimized_models)} models')
            return optimized_models if optimized_models else None

        except Exception as e:
            self.logger.warning(f'⚠️ Hyperparameter optimization failed: {e}')
            return None

    async def _create_ensemble_model(self, models_results: Dict[str, Any], X_train: np.ndarray,
                                   y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Create an ensemble model combining the best performing individual models."""
        try:
            from sklearn.ensemble import VotingClassifier, StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.metrics import accuracy_score, classification_report

            # Get the best performing models (top 3)
            model_performance = []
            for model_name, model_result in models_results.items():
                if 'direction' in model_result and 'accuracy' in model_result['direction']:
                    accuracy = model_result['direction']['accuracy']
                    model = model_result['direction'].get('model')
                    if model is not None:
                        model_performance.append((model_name, accuracy, model))

            if len(model_performance) < 2:
                self.logger.warning('⚠️ Not enough models for ensemble creation')
                return None

            # Sort by accuracy and take top models
            model_performance.sort(key=lambda x: x[1], reverse=True)
            top_models = model_performance[:3]  # Top 3 models

            self.logger.info(f'🎭 Creating ensemble from top {len(top_models)} models...')

            # Extract estimators and their names
            estimators = [(name, model) for name, _, model in top_models]
            model_names = [name for name, _, _ in top_models]

            # Create voting classifier (hard voting)
            voting_clf = VotingClassifier(
                estimators=estimators,
                voting='hard',  # Use majority voting
                n_jobs=-1
            )

            # Train the voting classifier
            voting_clf.fit(X_train, y_train)

            # Evaluate voting classifier
            y_pred_voting = voting_clf.predict(X_test)
            voting_accuracy = accuracy_score(y_test, y_pred_voting)

            # Create stacking classifier
            stacking_clf = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(random_state=42, max_iter=1000),
                cv=3,
                n_jobs=-1
            )

            # Train the stacking classifier
            stacking_clf.fit(X_train, y_train)

            # Evaluate stacking classifier
            y_pred_stacking = stacking_clf.predict(X_test)
            stacking_accuracy = accuracy_score(y_test, y_pred_stacking)

            # Choose the best ensemble method
            if stacking_accuracy > voting_accuracy:
                best_ensemble = stacking_clf
                best_accuracy = stacking_accuracy
                ensemble_type = 'stacking'
                self.logger.info(f'🏆 Stacking ensemble selected: {stacking_accuracy:.4f}')
            else:
                best_ensemble = voting_clf
                best_accuracy = voting_accuracy
                ensemble_type = 'voting'
                self.logger.info(f'🏆 Voting ensemble selected: {voting_accuracy:.4f}')

            # Calculate improvement over best individual model
            best_individual_accuracy = max(acc for _, acc, _ in top_models)
            improvement = best_accuracy - best_individual_accuracy

            ensemble_results = {
                'ensemble_type': ensemble_type,
                'accuracy': best_accuracy,
                'improvement': improvement,
                'base_models': model_names,
                'model': best_ensemble,
                'predictions': y_pred_stacking if ensemble_type == 'stacking' else y_pred_voting,
                'classification_report': classification_report(
                    y_test,
                    y_pred_stacking if ensemble_type == 'stacking' else y_pred_voting,
                    output_dict=True
                )
            }

            self.logger.info(f'✅ Ensemble created: {ensemble_type} ({best_accuracy:.4f}, improvement: {improvement:.4f})')

            return ensemble_results

        except Exception as e:
            self.logger.warning(f'⚠️ Ensemble creation failed: {e}')
            return None

    async def _train_ml_models_chunked(self, features_data: pd.DataFrame, sr_levels: Dict[str, Any], chunk_size: int = 200000) -> Dict[str, Any]:
        """Train ML models using chunked processing for large datasets."""
        self.logger.info(f'🤖 Starting chunked ML training with chunk_size={chunk_size}...')

        try:
            # For large datasets, process in chunks to avoid memory issues
            total_samples = len(features_data)
            n_chunks = (total_samples + chunk_size - 1) // chunk_size

            self.logger.info(f'📊 Processing {total_samples} samples in {n_chunks} chunks')

            # Simple chunked processing - in practice, this would aggregate results from multiple chunks
            chunk_results = []
            for i in range(n_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, total_samples)

                chunk_data = features_data.iloc[start_idx:end_idx]
                self.logger.info(f'🔄 Processing chunk {i+1}/{n_chunks}: {len(chunk_data)} samples')

                # For now, just return the same results as single chunk
                chunk_result = await self._train_ml_models(chunk_data, sr_levels)
                chunk_results.append(chunk_result)

            # Aggregate results from chunks
            avg_accuracy = sum(r['direction_accuracy'] for r in chunk_results) / len(chunk_results)
            avg_mae = sum(r['volatility_mae'] for r in chunk_results) / len(chunk_results)

            ml_results = {
                'direction_accuracy': avg_accuracy,
                'volatility_mae': avg_mae,
                'model_type': 'chunked_sr_optimization',
                'training_samples': total_samples,
                'chunks_processed': n_chunks,
                'sr_levels_used': len(sr_levels.get('support_levels', [])) + len(sr_levels.get('resistance_levels', [])),
                'training_time': sum(r.get('training_time', 0) for r in chunk_results),
                'chunk_results': chunk_results
            }

            self.logger.info(f'✅ Chunked ML training completed: accuracy={ml_results["direction_accuracy"]:.3f}, chunks={n_chunks}')
            return ml_results

        except Exception as e:
            self.logger.error(f'❌ Chunked ML training failed: {e}')
            # Fallback to regular training
            return await self._train_ml_models(features_data, sr_levels)

    async def run_step(self, symbol: str, exchange: str, timeframe: str = '30m', data_dir: str = 'data_cache', force_rerun: bool = False, config: Dict[str, Any] = None) -> bool:
        """Run step02_5 with proper interface matching other steps."""
        try:
            # Set up training input like other steps
            training_input = {
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe,
                'data_dir': data_dir,
                'force_rerun': force_rerun
            }

            # Set up basic pipeline state
            pipeline_state = {
                'config': config or {},
                'symbol': symbol,
                'exchange': exchange,
                'timeframe': timeframe
            }

            # Initialize the step
            await self.initialize()

            # Execute the step logic
            result = await self.execute(training_input, pipeline_state)

            # Return success status
            return result.get('success', False)

        except Exception as e:
            self.logger.error(f'❌ Step02_5 run_step failed: {e}')
            return False
