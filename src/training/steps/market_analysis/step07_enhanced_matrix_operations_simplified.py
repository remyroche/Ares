from ...core.decorators import handles_errors
from src.utils.comprehensive_function_logger import log_step_functions, log_important_calls, log_all_calls, log_internal_call, log_step_progress, log_data_operation

"""Step 7: Enhanced Matrix Operations - Simplified Version.
from src.utils.logger import system_logger

This is the simplified version of step07_enhanced_matrix_operations.py with
reduced complexity through modular design. All functionality is preserved
but organized into separate, focused modules.
"""
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# Core imports
import numpy as np
import pandas as pd

# Project imports
project_root = Path(__file__).parent.parent.parent
import sys
sys.path.insert(0, str(project_root))

from .utils.common_operations import ensure_directory, safe_json_dump
from .utils.pipeline_standards import PipelineStandards, pipeline_standards

# Import our new modular components
from .utils.function_call_tracker import FunctionCallTracker, comprehensive_function_tracker
from .utils.enhanced_error_handler import EnhancedErrorHandler
from .utils.comprehensive_validator import ComprehensiveValidator
from .utils.performance_monitor import PerformanceMonitor
from .utils.matrix_operations import MatrixOperations
from .utils.quality_metrics import QualityMetricsCalculator
from .utils.feature_filtering import FeatureFiltering
import collections
import json

# Optional dependencies with fallback handling
REQUIRED_MODULES = [
    'pandas', 'numpy', 'psutil', 'sklearn', 'scipy', 'lightgbm',
    'src.training.enhanced_matrix_operations', 'src.utils.error_handler', 
    'src.utils.logger', 'src.training.feature_engineering_optimizer', 
    'src.training.timeframe_relevance_analyzer', 'src.utils.training_pipeline_decorators', 
    'src.utils.enhanced_mlflow_integration'
]

dependency_status = PipelineStandards.validate_environment_dependencies(REQUIRED_MODULES)

# Safe imports with fallbacks
enhanced_matrix_operations = PipelineStandards.safe_import('src.training.enhanced_matrix_operations', None)
error_handler = PipelineStandards.safe_import('src.utils.error_handler', None)
system_logger = PipelineStandards.safe_import('src.utils.logger', None)
feature_engineering_optimizer = PipelineStandards.safe_import('src.training.feature_engineering_optimizer', None)
timeframe_relevance_analyzer = PipelineStandards.safe_import('src.training.timeframe_relevance_analyzer', None)
training_pipeline_decorators = PipelineStandards.safe_import('src.utils.training_pipeline_decorators', None)
enhanced_mlflow = PipelineStandards.safe_import('src.utils.enhanced_mlflow_integration', None)

# Fallback logger and decorators
def create_fallback_logger():
    import logging
    logging.basicConfig(level = logging.INFO)
    return logging.getLogger(__name__)

def create_fallback_decorator():
    def decorator(func):
        return func
    return decorator

# Initialize fallbacks
if system_logger is None:
    system_logger = create_fallback_logger()

if training_pipeline_decorators is None:
    circuit_breaker_protection = create_fallback_decorator()
    debug_training_step = create_fallback_decorator()
    memory_efficient = create_fallback_decorator()
    prevent_data_leakage = create_fallback_decorator()
    quality_gate = create_fallback_decorator()
    resource_monitor = create_fallback_decorator()
    secure_data_processing = create_fallback_decorator()
    validate_step_output = create_fallback_decorator()
else:
    circuit_breaker_protection = training_pipeline_decorators.circuit_breaker_protection
    debug_training_step = training_pipeline_decorators.debug_training_step
    memory_efficient = training_pipeline_decorators.memory_efficient
    prevent_data_leakage = training_pipeline_decorators.prevent_data_leakage
    quality_gate = training_pipeline_decorators.quality_gate
    resource_monitor = training_pipeline_decorators.resource_monitor
    secure_data_processing = training_pipeline_decorators.secure_data_processing
    validate_step_output = training_pipeline_decorators.validate_step_output

if enhanced_mlflow is None:
    with_enhanced_mlflow_logging = create_fallback_decorator()
    log_step_report = lambda *args, **kwargs: 'fallback_report'
    create_detailed_step_report = lambda *args, **kwargs: {}
    log_step_metrics = lambda *args, **kwargs: None
    log_step_dataframe_with_standardized_name = lambda *args, **kwargs: 'fallback_dataframe'
    log_step_artifact_with_standardized_name = lambda *args, **kwargs: 'fallback_artifact'
else:
    with_enhanced_mlflow_logging = enhanced_mlflow.with_enhanced_mlflow_logging
    log_step_report = enhanced_mlflow.log_step_report
    create_detailed_step_report = enhanced_mlflow.create_detailed_step_report
    log_step_metrics = enhanced_mlflow.log_step_metrics
    log_step_dataframe_with_standardized_name = enhanced_mlflow.log_step_dataframe_with_standardized_name
    log_step_artifact_with_standardized_name = enhanced_mlflow.log_step_artifact_with_standardized_name


class Step7EnhancedMatrixOperations:
    """Step 7: Enhanced Matrix Operations - Simplified with modular design."""
    @log_important_calls

    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize Step 7 Enhanced Matrix Operations with modular components."""
        self.config = config
        self.logger = system_logger.getChild('Step7EnhancedMatrixOperations')
        self.standards = pipeline_standards
        
        # Initialize modular components
        self.call_tracker = FunctionCallTracker(self.logger)
        self.error_handler = EnhancedErrorHandler(self.logger)
        self.validator = ComprehensiveValidator(self.logger)
        self.performance_monitor = PerformanceMonitor(self.logger)
        self.matrix_operations = MatrixOperations(self.logger)
        self.quality_calculator = QualityMetricsCalculator(self.logger)
        self.feature_filtering = FeatureFiltering(self.logger, config.get("step07_enhanced_matrix_operations", {}))
        
        self.logger.info("🔧 Initialized Step 7 with modular components")
        
        # Validate environment
        self._validate_environment()
        
        # Initialize enhanced matrix operations if available
        if enhanced_matrix_operations is not None:
            self.matrix_ops = enhanced_matrix_operations.EnhancedMatrixOperations(config)
        else:
            self.logger.warning('⚠️ EnhancedMatrixOperations not available')
            self.matrix_ops = None
        
        # Step-specific configuration
        self.step_config = config.get("step07_enhanced_matrix_operations", {})
        self.output_dir = ensure_directory(self.step_config.get("output_dir", "data/matrix_operations"))
    @log_all_calls

    def _validate_environment(self) -> None:
        """Validate environment dependencies."""
        self.logger.info('🔍 Validating environment dependencies...')
        missing_modules = [module for module, available in dependency_status.items() if not available]
        if missing_modules:
            self.logger.warning(f'⚠️ Missing optional modules: {missing_modules}')
            self.logger.info('📝 Pipeline will continue with fallback implementations')
        else:
            self.logger.info("✅ All required dependencies available")

    @comprehensive_function_tracker(system_logger)
    @log_execution_time(threshold_ms = 30000)
    @cached(policy = CachePolicy.PER_REQUEST, ttl = 3600)
    @log_call()
    @circuit_breaker(failure_threshold = 3, recovery_timeout = 300.0)
    @validates()
    @handles_errors(exceptions=(ValueError, RuntimeError), default_return = False)
    async def execute(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute Step 7: Enhanced Matrix Operations with simplified modular design.
        
        Args:
            training_input: Input data from previous steps
            pipeline_state: Current pipeline state
            
        Returns:
            Updated pipeline state with matrix operations results
        """
        try:
            start_time = datetime.now()
            self.logger.info('🚀 Starting Step 7: Enhanced Matrix Operations (Simplified)...')
            
            # Extract parameters
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            
            # Load and prepare data
            df = await self._load_and_prepare_data(symbol, exchange, timeframe)
            
            # Run feature engineering optimization
            feature_optimization_results = await self._run_feature_optimization(df, symbol, exchange, timeframe)
            
            # Run timeframe relevance analysis
            timeframe_analysis_results = await self._run_timeframe_analysis(symbol, exchange)
            
            # Apply feature filtering
            df_filtered, filtering_metadata = await self._apply_feature_filtering(df, symbol, exchange, timeframe)
            
            # Execute matrix operations
            matrix_results = await self._execute_matrix_operations(df_filtered)
            
            # Calculate quality metrics
            quality_metrics = self.quality_calculator.calculate_quality_metrics(df, matrix_results)
            
            # Save results
            output_files = await self._save_results(matrix_results, quality_metrics, symbol, exchange, timeframe)
            
            # Update pipeline state
            pipeline_state = self._update_pipeline_state(
                pipeline_state, start_time, output_files, matrix_results, 
                quality_metrics, feature_optimization_results, 
                timeframe_analysis_results, filtering_metadata, symbol, exchange, timeframe
            )
            
            # Log comprehensive summaries
            self._log_comprehensive_summaries(pipeline_state)
            
            # Log artifacts and create detailed report
            await self._log_step7_artifacts_and_report(training_input, pipeline_state, matrix_results, output_files, quality_metrics)
            
            self.logger.info("✅ Step 7: Enhanced Matrix Operations completed successfully")
            return pipeline_state
            
        except Exception as e:
            self.logger.error(f'❌ Step 7 failed: {str(e)}')
            pipeline_state['step07_enhanced_matrix_operations'] = {
                'status': 'failed', 
                'error': str(e), 
                'timestamp': datetime.now().isoformat()
            }
            return pipeline_state

    async def _load_and_prepare_data(self, symbol: str, exchange: str, timeframe: str) -> pd.DataFrame:
        """Load and prepare data for processing."""
        features_train_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet'
        features_val_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_val.parquet'
        
        if not os.path.exists(features_train_path):
            raise ValueError(f'Features train file not found: {features_train_path}')
        if not os.path.exists(features_val_path):
            raise ValueError(f'Features validation file not found: {features_val_path}')
        
        self.logger.info(f'📊 Loading engineered features from: {features_train_path}')
        df_train = pd.read_parquet(features_train_path)
        df_val = pd.read_parquet(features_val_path)
        
        # Optimize data types
        for d in (df_train, df_val):
            for c in d.select_dtypes(include=['float64']).columns:
                d[c] = d[c].astype('float32')
        
        df = pd.concat([df_train, df_val], ignore_index = True)
        self.logger.info(f'📈 Loaded {len(df)} rows of engineered features')
        self.logger.info(f'🔢 Features: {len(df.columns)} columns')
        
        return df

    async def _run_feature_optimization(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Run feature engineering optimization."""
        if feature_engineering_optimizer is not None:
            feature_optimizer = feature_engineering_optimizer.FeatureEngineeringOptimizer(self.config)
            
            # Extract target variable
            target = None
            if 'returns' in df.columns:
                target = df['returns']
            elif 'close' in df.columns:
                target = df['close'].pct_change().dropna()
                df = df.loc[target.index]
            
            if target is not None:
                self.logger.info('🔧 Starting feature engineering parameter optimization...')
                # Load HMM regimes if available
                hmm_regimes = self._load_hmm_regimes(symbol, exchange, timeframe)
                
                feature_optimization_results = await feature_optimizer.optimize_feature_parameters(
                    data = df, target = target, regimes = hmm_regimes, 
                    symbol = symbol, exchange = exchange, timeframe = timeframe
                )
                self.logger.info('✅ Feature engineering parameter optimization completed')
                return feature_optimization_results
            else:
                self.logger.warning('⚠️ No target variable found for feature optimization')
        else:
            self.logger.warning('⚠️ Skipping feature engineering optimization - optimizer not available')
        
        return {}

    async def _run_timeframe_analysis(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """Run timeframe relevance analysis."""
        if timeframe_relevance_analyzer is not None:
            timeframe_analyzer = timeframe_relevance_analyzer.TimeframeRelevanceAnalyzer(self.config)
            
            self.logger.info('⏰ Starting timeframe relevance analysis...')
            timeframe_data = {}
            
            for tf in ['1m', '5m', '15m', '30m', '1h']:
                tf_path = f'data/training/{exchange}_{symbol}_{tf}_features_train.parquet'
                if os.path.exists(tf_path):
                    tf_data = pd.read_parquet(tf_path)
                    timeframe_data[tf] = tf_data
            
            if timeframe_data:
                timeframe_analysis_results = await timeframe_analyzer.analyze_timeframe_relevance(
                    data_dict = timeframe_data, symbol = symbol, exchange = exchange, leverage_range=(10, 100)
                )
                self.logger.info('✅ Timeframe relevance analysis completed')
                return timeframe_analysis_results
            else:
                self.logger.warning('⚠️ Skipping timeframe analysis - insufficient multi-timeframe data')
        else:
            self.logger.warning('⚠️ Skipping timeframe analysis - analyzer not available')
        
        return {}
    @log_all_calls

    def _load_hmm_regimes(self, symbol: str, exchange: str, timeframe: str) -> Optional[pd.Series]:
        """Load HMM regime data if available."""
        hmm_primary = f'data/hmm_regimes/{exchange}_{symbol}_{timeframe}_composite_clusters.parquet'
        hmm_alias = f'data/hmm_regimes/{exchange}_{symbol}_{timeframe}_hmm_regimes.parquet'
        hmm_path = hmm_primary if os.path.exists(hmm_primary) else hmm_alias if os.path.exists(hmm_alias) else None
        
        if hmm_path:
            self.logger.info(f'🎭 Loading HMM regimes from: {hmm_path}')
            hmm_data = pd.read_parquet(hmm_path)
            if 'composite_cluster_id' in hmm_data.columns:
                return hmm_data['composite_cluster_id']
            elif 'hmm_regime' in hmm_data.columns:
                return hmm_data['hmm_regime']
        
        return None

    async def _apply_feature_filtering(self, df: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Apply feature filtering using the modular feature filtering component."""
        self.logger.info("🎯 Applying regime-aware feature filtering...")
        
        # Separate features from labels
        label_columns = ['target', 'direction', 'profit', 'outcome', 'returns', 'timestamp', 
                        'open', 'high', 'low', 'close', 'volume']
        feature_columns = [col for col in df.columns if col not in label_columns]
        
        # Create features and labels dataframes
        features_df = df[feature_columns]
        labels_df = df[[col for col in label_columns if col in df.columns]]
        
        # Load regime labels if available
        regime_labels = self._load_hmm_regimes(symbol, exchange, timeframe)
        
        # Apply feature filtering using the modular component
        filtered_features_df, filtering_metadata = self.feature_filtering.regime_aware_initial_filtering(
            features_df = features_df,
            labels_df = labels_df,
            regime_labels = regime_labels
        )
        
        # Reconstruct full dataframe with filtered features
        df_filtered = pd.concat([filtered_features_df, labels_df], axis = 1)
        
        self.logger.info(f"✅ Feature filtering applied: {len(feature_columns)} → {len(filtered_features_df.columns)} features")
        
        # Save filtered features
        await self._save_filtered_features(df_filtered, symbol, exchange, timeframe)
        
        return df_filtered, filtering_metadata

    async def _save_filtered_features(self, df_filtered: pd.DataFrame, symbol: str, exchange: str, timeframe: str) -> None:
        """Save filtered features to files."""
        # Split back to train/val based on original sizes
        features_train_path = f'data/training/{exchange}_{symbol}_{timeframe}_features_train.parquet'
        if os.path.exists(features_train_path):
            df_train_original = pd.read_parquet(features_train_path)
            train_size = len(df_train_original)
        else:
            train_size = len(df_filtered) // 2  # Fallback to 50/50 split
        
        df_filtered_train = df_filtered.iloc[:train_size]
        df_filtered_val = df_filtered.iloc[train_size:]
        
        filtered_train_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_train.parquet"
        filtered_val_path = f"data/training/{exchange}_{symbol}_{timeframe}_features_filtered_val.parquet"
        
        df_filtered_train.to_parquet(filtered_train_path)
        df_filtered_val.to_parquet(filtered_val_path)
        
        self.logger.info(f"💾 Saved filtered features to {filtered_train_path} and {filtered_val_path}")

    async def _execute_matrix_operations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute matrix operations using the modular matrix operations component."""
        # Prepare matrix operations configuration
        matrix_config = self._prepare_matrix_operations_config(df)
        
        # Execute matrix operations on filtered features
        results = {}
        numeric_df = df.select_dtypes(include=[np.number])
        
        if len(numeric_df.columns) == 0:
            self.logger.warning('⚠️ No numeric columns found for matrix operations')
            return {'error': 'No numeric columns available'}
        
        self.logger.info(f'🔢 Performing matrix operations on {len(numeric_df.columns)} numeric columns')
        
        # Execute standard matrix operations
        results.update(await self.matrix_operations.execute_standard_matrix_operations(numeric_df, matrix_config))
        
        # Execute SR-specific operations if SR features are available
        if matrix_config.get('enable_sr_analysis', False) and matrix_config.get('sr_features'):
            self.logger.info('🎯 Performing SR-specific matrix operations...')
            results['sr_analysis'] = await self.matrix_operations.execute_sr_matrix_operations(df, matrix_config)
            results['sr_enhanced_analysis'] = await self.matrix_operations.execute_enhanced_sr_analysis(df, matrix_config)
            results['sr_optimization_analysis'] = await self.matrix_operations.execute_sr_optimization_analysis(df, matrix_config)
        
        return results
    @log_all_calls

    def _prepare_matrix_operations_config(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Prepare configuration for matrix operations."""
        sr_features = [col for col in df.columns if any(
            keyword in col.lower() for keyword in [
                'sr_', 'support', 'resistance', 'proximity', 'sr_distance', 'sr_proximity', 
                'sr_outcome', 'normalized_distance', 'sr_proximity_score', 'strength_score', 
                'clarity_factor', 'directional_pressure', 'sr_score', 'delta_sr_score', 
                'isolation_score', 'sr_level', 'sr_multi_timeframe', 'support_', 'resistance_'
            ]
        )]
        
        config = {
            'enable_gpu_acceleration': self.step_config.get('enable_gpu_acceleration', False),
            'enable_sparse_optimizations': self.step_config.get('enable_sparse_optimizations', True),
            'enable_memory_optimization': self.step_config.get('enable_memory_optimization', True),
            'enable_parallel_processing': self.step_config.get('enable_parallel_processing', True),
            'condition_number_threshold': self.step_config.get('condition_number_threshold', 1000000000000.0),
            'min_eigenvalue_threshold': self.step_config.get('min_eigenvalue_threshold', 1e-10),
            'correlation_threshold': self.step_config.get('correlation_threshold', 0.8),
            'memory_threshold_gb': self.step_config.get('memory_threshold_gb', 8.0),
            'batch_size': self.step_config.get('batch_size', 1000),
            'max_iterations': self.step_config.get('max_iterations', 1000),
            'tolerance': self.step_config.get('tolerance', 1e-06),
            'data_shape': df.shape,
            'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
            'sr_features': sr_features,
            'sr_feature_count': len(sr_features),
            'enable_sr_analysis': len(sr_features) > 0,
            'sr_correlation_threshold': self.step_config.get('sr_correlation_threshold', 0.7),
            'sr_condition_number_threshold': self.step_config.get('sr_condition_number_threshold', 10000000000.0)
        }
        
        self.logger.info(f'🔧 Matrix operations configuration prepared:')
        self.logger.info(f'   - Total features: {len(df.columns)}')
        self.logger.info(f'   - SR features: {len(sr_features)}')
        self.logger.info(f"   - Numeric features: {len(config['numeric_columns'])}")
        
        return config

    async def _save_results(self, matrix_results: Dict[str, Any], quality_metrics: Dict[str, Any], 
                          symbol: str, exchange: str, timeframe: str) -> Dict[str, str]:
        """Save matrix operations results to files."""
        output_files = {}
        
        # Save configuration
        config_file = self.output_dir / f'{exchange}_{symbol}_{timeframe}_matrix_operations_config.json'
        safe_json_dump(self.step_config, config_file, indent = 2, default = str)
        output_files['config'] = str(config_file)
        
        # Save results
        results_file = self.output_dir / f'{exchange}_{symbol}_{timeframe}_matrix_operations_results.json'
        safe_json_dump(matrix_results, results_file, indent = 2, default = str)
        output_files['results'] = str(results_file)
        
        # Save quality metrics
        quality_file = self.output_dir / f'{exchange}_{symbol}_{timeframe}_quality_metrics.json'
        safe_json_dump(quality_metrics, quality_file, indent = 2, default = str)
        output_files['quality_metrics'] = str(quality_file)
        
        # Generate and save detailed report
        detailed_report = self.quality_calculator.generate_detailed_quality_report(quality_metrics, matrix_results)
        report_file = self.output_dir / f'{exchange}_{symbol}_{timeframe}_quality_report.txt'
        with open(report_file, 'w') as f:
            f.write(detailed_report)
        output_files['quality_report'] = str(report_file)
        
        self.logger.info('\n' + detailed_report)
        
        # Save summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'exchange': exchange,
            'timeframe': timeframe,
            'operations_performed': list(matrix_results.keys()),
            'overall_quality_score': quality_metrics.get('overall_score', 0.0),
            'quality_summary': {
                'completeness_ratio': quality_metrics.get('completeness', {}).get('missing_ratio', 1.0),
                'zero_variance_features': quality_metrics.get('variance', {}).get('zero_variance_features', 0),
                'high_correlations': quality_metrics.get('correlation', {}).get('high_correlation_pairs', 0),
                'is_well_conditioned': quality_metrics.get('numerical_stability', {}).get('is_well_conditioned', False)
            }
        }
        
        summary_file = self.output_dir / f'{exchange}_{symbol}_{timeframe}_matrix_operations_summary.json'
        safe_json_dump(summary, summary_file, indent = 2, default = str)
        output_files['summary'] = str(summary_file)
        
        self.logger.info(f'💾 Saved matrix operations results to {self.output_dir}')
        return output_files
    @log_all_calls

    def _update_pipeline_state(self, pipeline_state: Dict[str, Any], start_time: datetime, 
                             output_files: Dict[str, str], matrix_results: Dict[str, Any],
                             quality_metrics: Dict[str, Any], feature_optimization_results: Dict[str, Any],
                             timeframe_analysis_results: Dict[str, Any], filtering_metadata: Dict[str, Any],
                             symbol: str, exchange: str, timeframe: str) -> Dict[str, Any]:
        """Update pipeline state with all results."""
        pipeline_state["step07_enhanced_matrix_operations"] = {
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "output_files": output_files,
            "matrix_results": matrix_results,
            "quality_metrics": quality_metrics,
            "symbol": symbol,
            "exchange": exchange,
            "timeframe": timeframe,
            "feature_engineering_optimization": feature_optimization_results,
            "timeframe_relevance_analysis": timeframe_analysis_results,
            "feature_filtering_metadata": filtering_metadata
        }
        
        return pipeline_state
    @log_all_calls

    def _log_comprehensive_summaries(self, pipeline_state: Dict[str, Any]) -> None:
        """Log comprehensive summaries from all monitoring components."""
        # Function call summary
        call_summary = self.call_tracker.get_call_summary()
        self.logger.info("📊 COMPREHENSIVE FUNCTION CALL SUMMARY:")
        self.logger.info(f"   Total function calls: {call_summary['total_function_calls']}")
        self.logger.info(f"   Successful calls: {call_summary['successful_calls']}")
        self.logger.info(f"   Failed calls: {call_summary['failed_calls']}")
        self.logger.info(f"   Success rate: {call_summary['success_rate']:.2%}")
        self.logger.info(f"   Total duration: {call_summary['total_duration_seconds']:.3f}s")
        self.logger.info(f"   Average duration: {call_summary['average_duration_seconds']:.3f}s")
        
        # Performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        self.logger.info("📊 PERFORMANCE MONITORING SUMMARY:")
        self.logger.info(f"   Functions monitored: {performance_summary['total_functions_monitored']}")
        self.logger.info(f"   Total duration: {performance_summary['total_duration_seconds']:.3f}s")
        self.logger.info(f"   Total memory delta: {performance_summary['total_memory_delta_mb']:.1f} MB")
        
        # Error summary
        error_summary = self.error_handler.get_error_summary()
        if error_summary['total_errors'] > 0:
            self.logger.warning(f"⚠️ ERROR HANDLING SUMMARY:")
            self.logger.warning(f"   Total errors: {error_summary['total_errors']}")
        else:
            self.logger.info("✅ No errors encountered during execution")
        
        # Validation summary
        validation_summary = self.validator.get_validation_summary()
        self.logger.info(f"🔍 VALIDATION SUMMARY:")
        self.logger.info(f"   Total validations: {validation_summary['total_validations']}")
        
        # Add summaries to pipeline state
        step_state = pipeline_state["step07_enhanced_matrix_operations"]
        step_state["function_call_summary"] = call_summary
        step_state["performance_summary"] = performance_summary
        step_state["error_summary"] = error_summary
        step_state["validation_summary"] = validation_summary

    async def _log_step7_artifacts_and_report(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any], 
                                            matrix_results: Dict[str, Any], output_files: Dict[str, str], 
                                            quality_metrics: Dict[str, Any]) -> None:
        """Log step 7 artifacts and create detailed report."""
        try:
            symbol = training_input.get('symbol', 'UNKNOWN')
            exchange = training_input.get('exchange', 'UNKNOWN')
            timeframe = training_input.get('timeframe', '1m')
            
            execution_metadata = {
                'start_time': datetime.now().isoformat(),
                'end_time': datetime.now().isoformat(),
                'duration_seconds': 0.0,
                'memory_usage_mb': 0.0,
                'cpu_usage_percent': 0.0,
                'data_quality_score': quality_metrics.get('overall_quality', 0.0),
                'processing_efficiency': 1.0 if pipeline_state.get('step07_enhanced_matrix_operations', {}).get('status') == 'completed' else 0.0
            }
            
            artifacts_generated = list(output_files.values()) if output_files else []
            
            metrics_calculated = {
                'matrix_operations_success': 1.0 if pipeline_state.get('step07_enhanced_matrix_operations', {}).get('status') == 'completed' else 0.0,
                'matrix_operations_count': len(matrix_results) if matrix_results else 0,
                'output_files_count': len(output_files) if output_files else 0,
                'overall_quality_score': quality_metrics.get('overall_quality', 0.0),
                'data_completeness': quality_metrics.get('data_completeness', 0.0),
                'feature_quality': quality_metrics.get('feature_quality', 0.0)
            }
            
            step_data = {
                'matrix_results': matrix_results,
                'output_files': output_files,
                'quality_metrics': quality_metrics
            }
            
            report_data = create_detailed_step_report(
                step_name='step07_enhanced_matrix_operations',
                step_data = step_data,
                training_input = training_input,
                execution_metadata = execution_metadata,
                artifacts_generated = artifacts_generated,
                metrics_calculated = metrics_calculated,
                errors_encountered=[] if pipeline_state.get('step07_enhanced_matrix_operations', {}).get('status') == 'completed' else ['Matrix operations failed']
            )
            
            report_name = log_step_report(
                config = self.config,
                step_name='step07_enhanced_matrix_operations',
                report_data = report_data,
                report_type='matrix_operations_report',
                additional_metadata={
                    'matrix_operations_success': pipeline_state.get('step07_enhanced_matrix_operations', {}).get('status') == 'completed',
                    'matrix_operations_count': len(matrix_results) if matrix_results else 0,
                    'asset': symbol,
                    'lookback_period': self.config.get('lookback_days', 1095),
                    'project_version': self.config.get('project_version', '1.0.0'),
                    'timeframe': timeframe
                }
            )
            
            self.logger.info(f'✅ Logged matrix operations report: {report_name}')
            
            # Log additional reports
            if matrix_results:
                matrix_report_name = log_step_report(
                    config = self.config,
                    step_name='step07_enhanced_matrix_operations',
                    report_data = matrix_results,
                    report_type='matrix_results',
                    additional_metadata={
                        'matrix_operations_count': len(matrix_results),
                        'timeframe': timeframe,
                        'asset': symbol,
                        'lookback_period': self.config.get('lookback_days', 1095),
                        'project_version': self.config.get('project_version', '1.0.0')
                    }
                )
                self.logger.info(f'✅ Logged matrix results: {matrix_report_name}')
            
            if quality_metrics:
                quality_report_name = log_step_report(
                    config = self.config,
                    step_name='step07_enhanced_matrix_operations',
                    report_data = quality_metrics,
                    report_type='quality_metrics',
                    additional_metadata={
                        'overall_quality_score': quality_metrics.get('overall_quality', 0.0),
                        'timeframe': timeframe,
                        'asset': symbol,
                        'lookback_period': self.config.get('lookback_days', 1095),
                        'project_version': self.config.get('project_version', '1.0.0')
                    }
                )
                self.logger.info(f'✅ Logged quality metrics: {quality_report_name}')
            
            log_step_metrics(
                config = self.config,
                step_name='step07_enhanced_matrix_operations',
                metrics = metrics_calculated,
                additional_metadata={
                    'metrics_type': 'matrix_operations_performance',
                    'timeframe': timeframe,
                    'asset': symbol,
                    'lookback_period': self.config.get('lookback_days', 1095),
                    'project_version': self.config.get('project_version', '1.0.0')
                }
            )
            
            self.logger.info('✅ Step 7 artifacts and reports logged successfully')
            
        except Exception as e:
            self.logger.error(f'❌ Failed to log step 7 artifacts and reports: {e}')


async def run_step(symbol: str, exchange: str, timeframe: str = '1m', data_dir: str = None, 
                  force_rerun: bool = False, **kwargs: Any) -> bool:
    """
    Run Step 7: Enhanced Matrix Operations with simplified modular design.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        timeframe: Timeframe
        data_dir: Data directory (will use standardized path if None)
        force_rerun: Force rerun the step
        **kwargs: Additional arguments
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if data_dir is None:
            data_dir = pipeline_standards.build_path('processed_data', exchange, symbol)
        
        from .config.training import get_training_config
        config = get_training_config()
        step = Step7EnhancedMatrixOperations(config)
        
        training_input = {
            'symbol': symbol, 
            'exchange': exchange, 
            'timeframe': timeframe, 
            'data_dir': data_dir, 
            'force_rerun': force_rerun, 
            'asset': symbol, 
            'lookback_period': config.get('lookback_days', 1095), 
            'project_version': config.get('project_version', '1.0.0'), 
            **kwargs
        }
        
        pipeline_state = {}
        result = await step.execute(training_input, pipeline_state)
        step_result = result.get('step07_enhanced_matrix_operations', {})
        return step_result.get('status') == 'completed'
        
    except Exception as e:
        system_logger.error(f'❌ Step 7 failed: {str(e)}')
        return False


__all__ = ['Step7EnhancedMatrixOperations', 'run_step']