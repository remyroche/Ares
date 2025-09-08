from ..standardized_parquet_handler import standardized_parquet_handler
"""
Unified Step08 Class Implementation - Part 2
"""

# Comprehensive utility imports for extensive integration
from src.utils.math_validation import (
    safe_divide, safe_log, safe_sqrt, safe_kelly_calculation,
    validate_positive, validate_range, MathValidationError,
    safe_power, safe_weighted_average, safe_percentage_change,
    validate_correlation_matrix, safe_matrix_inverse, math_safe
)
from src.utils.common_operations import (
    get_current_datetime, get_today, format_datetime, parse_datetime,
    create_empty_dataframe, safe_fillna, safe_rolling, safe_copy, safe_deepcopy,
    safe_mean, safe_std, ensure_directory, safe_file_exists,
    safe_json_dump, safe_json_load, safe_sleep, safe_gather,
    create_async_task, safe_append, safe_extend, safe_dict_get,
    safe_dict_items, safe_lower, safe_upper, safe_join,
    get_logger, setup_basic_logging, safe_exception_handler,
    safe_float, safe_int, suggest_float_uniform, suggest_int_uniform,
    validate_dataframe, validate_numeric_range, optimize_dataframe_dtypes,
    timed_operation, format_bytes, chunked_iterable, parallel_map,
    safe_log_metric, safe_log_params, safe_log_artifact,
    safe_read_parquet, safe_to_parquet, list_parquet_files,
    generate_hash, generate_cache_key, standardize_price_action_probabilities
)
from src.utils.common_utilities import (
    safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
    calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
    safe_apply_function, create_summary_statistics, safe_drop_columns,
    safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    get_dataframe_info, safe_filter_dataframe, create_data_quality_report
)
from src.utils.parquet_utils import ParquetUtils, get_parquet_utils
from src.utils.serialization_utils import (
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,
    save_json, load_json, save_pickle, load_pickle, save_parquet, load_parquet,
    save_data, load_data, SerializationError
)
from src.utils.data_processing_utils import (
    DataFrameValidator, DataFrameCleaner, DataFrameTransformer,
    DataQualityLevel, DataQualityIssue, DataQualityReport,
    validate_dataframe as validate_df_comprehensive, clean_dataframe,
    transform_dataframe, get_dataframe_info as get_df_info_comprehensive
)
from src.utils.m1_gpu_utils import (
    M1GPUManager, M1PerformanceOptimizer, initialize_m1_gpu,
    get_m1_gpu_manager, m1_tensor_multiply, m1_batch_process,
    m1_monte_carlo_simulate, create_m1_optimized_config
)
from src.utils.m1_memory_optimizer import (
    M1MemoryOptimizer, M1DataManager, get_m1_memory_optimizer,
    create_memory_efficient_dataframe, memory_efficient_groupby
)
from src.utils.m1_cpu_optimizer import (
    M1CPUOptimizer, M1BatchProcessor, get_m1_cpu_optimizer,
    initialize_m1_cpu_optimizer, parallel_map, parallel_dataframe_operation,
    parallel_monte_carlo_simulation, optimized_monte_carlo_worker
)
from src.utils.lookahead_bias_detector import (
    get_global_detector, validate_no_future_data, LookaheadBiasError
)

import datetime
import logging
import numpy as np
import os
import pandas as pd
import pathlib as Path
import time
import typing
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from enum import Enum
import warnings

class UnifiedStep08:
    """
    Unified Step08: Advanced Feature Selection with Regime Data Splitting and Financial Risk Assessment
    
    This class consolidates all Step08 functionality into a single, comprehensive module:
    - Regime data splitting with HMM composite clusters
    - Advanced feature selection with bias prevention
    - Financial metrics calculation (returns, volatility, Sharpe ratio, VaR)
    - Regime balance handling for imbalanced distributions
    - Comprehensive risk assessment with explicit risk metrics
    """

    def __init__(self, config: Dict[str, Any], 
                 parquet_utils: Optional[ParquetUtils] = None,
                 memory_optimizer: Optional[M1MemoryOptimizer] = None,
                 gpu_manager: Optional[M1GPUManager] = None,
                 cpu_optimizer: Optional[M1CPUOptimizer] = None,
                 data_validator: Optional[DataFrameValidator] = None,
                 data_cleaner: Optional[DataFrameCleaner] = None,
                 data_transformer: Optional[DataFrameTransformer] = None) -> None:
        """Initialize unified Step08 with comprehensive configuration and dependency injection."""
        self.config = config
        self.logger = get_logger('UnifiedStep08')
        
        # Dependency injection for utilities
        self.parquet_utils = parquet_utils or get_parquet_utils()
        self.memory_optimizer = memory_optimizer or get_m1_memory_optimizer()
        self.gpu_manager = gpu_manager or get_m1_gpu_manager()
        self.cpu_optimizer = cpu_optimizer or get_m1_cpu_optimizer()
        self.data_validator = data_validator or DataFrameValidator()
        self.data_cleaner = data_cleaner or DataFrameCleaner()
        self.data_transformer = data_transformer or DataFrameTransformer()
        
        # Initialize serialization utilities
        self.json_serializer = JSONSerializer()
        self.pickle_serializer = PickleSerializer()
        self.parquet_serializer = ParquetSerializer()
        self.universal_serializer = UniversalSerializer()
        
        # Initialize components
        self._initialize_optimizations()
        self._initialize_configuration()
        self._initialize_metrics()
        self._initialize_utility_integration()
        
        self.logger.info('🚀 Unified Step08 initialized successfully with extensive utility integration')

    def _initialize_optimizations(self) -> None:
        """Initialize enhanced optimization components."""
        self.logger.info("🔧 Initializing enhanced optimization components...")
        
        # Initialize M1 optimizations if available
        if ENHANCED_OPTIMIZATIONS_AVAILABLE:
            try:
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                self.pipeline_executor = OptimizedPipelineExecutor(max_concurrent_stages=6)
                self.matrix_operations = EnhancedMatrixOperations(
                    enable_gpu_acceleration=True,
                    enable_memory_optimization=True
                )
                self.optimization_selector = IntelligentOptimizationSelector()
                self.data_manager = OptimizedDataManager(
                    base_path=Path("data_cache"),
                    enable_compression=True,
                    enable_caching=True
                )
                self.error_handler = ErrorHandler(enable_recovery=True)
                self.logger.info("✅ Enhanced optimizations initialized")
            except Exception as e:
                self.logger.warning(f"⚠️ Enhanced optimizations failed: {e}")
                self._initialize_fallback_optimizations()
        else:
            self._initialize_fallback_optimizations()

    def _initialize_fallback_optimizations(self) -> None:
        """Initialize fallback optimization components."""
        self.m1_gpu_manager = None
        self.m1_memory_optimizer = None
        self.m1_cpu_optimizer = None
        self.pipeline_executor = None
        self.matrix_operations = None
        self.optimization_selector = None
        self.data_manager = None
        self.error_handler = None
        self.logger.info("✅ Fallback optimizations initialized")

    def _initialize_configuration(self) -> None:
        """Initialize configuration parameters."""
        self.step_config = self.config.get('step08_unified', {})
        
        # Feature selection parameters
        self.phase1_target_features = self.step_config.get('phase1_target_features', 150)
        self.phase2_targets = self.step_config.get('phase2_targets', [100, 80, 60])
        self.enable_mrmr = self.step_config.get('enable_mrmr', True)
        self.enable_rf_importance = self.step_config.get('enable_rf_importance', True)
        self.boruta_max_iter = self.step_config.get('boruta_max_iter', 100)
        self.boruta_alpha = self.step_config.get('boruta_alpha', 0.05)
        
        # Regime balance parameters
        self.min_regime_samples = self.step_config.get('min_regime_samples', 100)
        self.target_balance_ratio = self.step_config.get('target_balance_ratio', 0.8)
        self.enable_regime_rebalancing = self.step_config.get('enable_regime_rebalancing', True)
        self.rebalancing_method = self.step_config.get('rebalancing_method', 'oversample')
        
        # Financial metrics parameters
        self.risk_free_rate = self.step_config.get('risk_free_rate', 0.02)
        self.var_confidence_levels = self.step_config.get('var_confidence_levels', [0.95, 0.99])
        self.lookback_periods = self.step_config.get('lookback_periods', [30, 90, 252])
        
        # Risk assessment parameters
        self.model_risk_threshold = self.step_config.get('model_risk_threshold', 0.3)
        self.overfitting_threshold = self.step_config.get('overfitting_threshold', 0.1)
        self.feature_stability_threshold = self.step_config.get('feature_stability_threshold', 0.8)
        
        # Output directories
        self.output_dir = ensure_directory(self.step_config.get('output_dir', 'data/step08_unified'))
        self.reports_dir = ensure_directory(os.path.join(self.output_dir, 'reports'))
        self.artifacts_dir = ensure_directory(os.path.join(self.output_dir, 'artifacts'))
        self.metrics_dir = ensure_directory(os.path.join(self.output_dir, 'metrics'))

    def _initialize_metrics(self) -> None:
        """Initialize metrics tracking."""
        self.financial_metrics = FinancialMetrics()
        self.risk_metrics = RiskMetrics()
        self.regime_balance = RegimeBalanceMetrics()
        self.feature_validation = FeatureSelectionValidation()
        self.results = Step08Results()

    def _initialize_utility_integration(self) -> None:
        """Initialize comprehensive utility integration."""
        self.logger.info("🔧 Initializing comprehensive utility integration...")
        
        # Initialize data quality monitoring
        self.data_quality_monitor = {
            'validation_reports': [],
            'cleaning_reports': [],
            'transformation_reports': [],
            'memory_usage_history': [],
            'performance_metrics': {}
        }
        
        # Initialize utility health status
        self.utility_health = {
            'common_operations': get_common_operations_health_status(),
            'memory_optimizer': self.memory_optimizer.get_memory_report(),
            'gpu_manager': {'device': str(self.gpu_manager.device), 'memory_info': self.gpu_manager.memory_info},
            'cpu_optimizer': self.cpu_optimizer.get_cpu_usage_report(),
            'parquet_utils': {'status': 'initialized'},
            'serialization_utils': {'status': 'initialized'}
        }
        
        # Initialize performance tracking
        self.performance_tracker = {
            'operation_times': {},
            'memory_usage': {},
            'gpu_utilization': {},
            'cpu_utilization': {}
        }
        
        # Initialize cache for utility operations
        self.utility_cache = {
            'dataframe_validations': {},
            'parquet_operations': {},
            'serialization_operations': {},
            'memory_optimizations': {}
        }
        
        self.logger.info("✅ Comprehensive utility integration initialized")

    @with_tracing_span('step08_unified.execute', log_args=False)
    @handle_errors(exceptions=(Exception,), default_return={'success': False, 'error': 'Execution failed'}, context='step08_unified_execution')
    async def execute(self, training_input: Dict[str, Any] = None, pipeline_state: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute unified Step08 with comprehensive analysis."""
        try:
            start_time = datetime.now()
            self.logger.info('🚀 Starting Unified Step08 execution...')
            
            # Step 1: Load and validate data
            self.logger.info('📊 Step 1: Loading and validating data...')
            unified_data = await self._load_and_validate_data(training_input, pipeline_state)
            if unified_data is None:
                return {'success': False, 'error': 'Failed to load or validate data'}
            
            # Step 2: Regime balance analysis and handling
            self.logger.info('⚖️ Step 2: Analyzing and handling regime balance...')
            balanced_data = await self._handle_regime_balance(unified_data)
            
            # Step 3: Advanced feature selection with bias prevention
            self.logger.info('🔍 Step 3: Advanced feature selection with bias prevention...')
            selected_features = await self._advanced_feature_selection(balanced_data)
            
            # Step 4: Financial metrics calculation
            self.logger.info('💰 Step 4: Calculating financial metrics...')
            financial_metrics = await self._calculate_financial_metrics(balanced_data, selected_features)
            
            # Step 5: Risk assessment
            self.logger.info('⚠️ Step 5: Comprehensive risk assessment...')
            risk_metrics = await self._comprehensive_risk_assessment(balanced_data, selected_features, financial_metrics)
            
            # Step 6: Feature selection validation
            self.logger.info('✅ Step 6: Feature selection validation...')
            feature_validation = await self._validate_feature_selection(balanced_data, selected_features)
            
            # Step 7: Generate comprehensive results
            self.logger.info('📋 Step 7: Generating comprehensive results...')
            results = await self._generate_comprehensive_results(
                balanced_data, selected_features, financial_metrics, 
                risk_metrics, feature_validation, start_time
            )
            
            # Step 8: Save artifacts and reports
            self.logger.info('💾 Step 8: Saving artifacts and reports...')
            await self._save_artifacts_and_reports(results)
            
            self.logger.info('✅ Unified Step08 execution completed successfully')
            return {
                'success': True,
                'results': results,
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
            
        except Exception as e:
            self.logger.exception(f'❌ Unified Step08 execution failed: {e}')
            return {'success': False, 'error': str(e)}

    @timed_operation("data_loading_and_validation")
    async def _load_and_validate_data(self, training_input: Dict[str, Any], pipeline_state: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load and validate unified data with comprehensive utility integration."""
        try:
            self.logger.info("📊 Loading and validating data with extensive utility integration...")
            
            # Memory checkpoint for data loading
            with self.memory_optimizer.memory_checkpoint("data_loading_start"):
                # Load data using unified data loader with utility integration
                if UNIFIED_DATA_LOADER_AVAILABLE:
                    data_loader = UnifiedDataLoader(self.config)
                    unified_data = await data_loader.load_unified_data(
                        symbol=self.config.get('symbol', 'ETHUSDT'),
                        exchange=self.config.get('exchange', 'BINANCE'),
                        timeframe=self.config.get('timeframe', '1m'),
                        data_dir=self.config.get('data_dir', 'data_cache')
                    )
                else:
                    # Fallback to pipeline state data with utility integration
                    if pipeline_state and 'dataframe' in pipeline_state:
                        unified_data = pipeline_state['dataframe']
                    else:
                        self.logger.error('No data available and unified data loader not available')
                        return None
                
                # Comprehensive data validation using utility integration
                self.logger.info("🔍 Performing comprehensive data validation...")
                
                # Basic DataFrame validation
                if not validate_dataframe(unified_data, ['timestamp', 'composite_cluster_id']):
                    self.logger.error("DataFrame validation failed - missing required columns")
                    return None
                
                # Comprehensive data quality validation
                quality_report = self.data_validator.validate_dataframe(unified_data)
                self.data_quality_monitor['validation_reports'].append(quality_report)
                
                # Log data quality issues
                if quality_report.issues:
                    self.logger.warning(f"Data quality issues found: {len(quality_report.issues)}")
                    for issue in quality_report.issues:
                        if issue.level == DataQualityLevel.CRITICAL:
                            self.logger.error(f"Critical issue: {issue.description}")
                        elif issue.level == DataQualityLevel.WARNING:
                            self.logger.warning(f"Warning: {issue.description}")
                
                # Data cleaning with utility integration
                if quality_report.issues:
                    self.logger.info("🧹 Applying data cleaning...")
                    unified_data = self.data_cleaner.clean_dataframe(unified_data)
                    cleaning_report = create_data_quality_report(unified_data)
                    self.data_quality_monitor['cleaning_reports'].append(cleaning_report)
                
                # Memory optimization for large datasets
                data_size_mb = unified_data.memory_usage(deep=True).sum() / (1024**2)
                if self.memory_optimizer.should_chunk_data(data_size_mb, "general"):
                    self.logger.info(f"📦 Large dataset detected ({data_size_mb:.1f}MB), applying memory optimization...")
                    unified_data = optimize_dataframe_dtypes(unified_data)
                    unified_data = create_memory_efficient_dataframe(unified_data)
                
                # Timestamp validation and conversion
                timestamp_valid, timestamp_error = validate_timestamp_column(unified_data, 'timestamp')
                if not timestamp_valid:
                    self.logger.warning(f"Timestamp validation issue: {timestamp_error}")
                    unified_data = safe_timestamp_conversion(unified_data, 'timestamp')
                
                # Generate comprehensive DataFrame info
                df_info = get_dataframe_info(unified_data)
                self.logger.info(f"📊 Data loaded successfully: {df_info['shape']} shape, {df_info['memory_usage_mb']:.1f}MB memory")
                
                # Cache validation results
                cache_key = generate_cache_key("data_validation", str(unified_data.shape), str(unified_data.columns.tolist()))
                self.utility_cache['dataframe_validations'][cache_key] = {
                    'quality_report': quality_report,
                    'df_info': df_info,
                    'timestamp': get_current_datetime().isoformat()
                }
                
                return unified_data
                
        except Exception as e:
            self.logger.error(f"Failed to load and validate data: {e}")
            return None
        finally:
            # Memory cleanup after data loading
            self.memory_optimizer.optimize_memory()
            
            if unified_data is None or len(unified_data) == 0:
                self.logger.error('Unified data is empty or None')
                return None

    @timed_operation("financial_metrics_calculation")
    async def _calculate_financial_metrics(self, data: pd.DataFrame, selected_features: Dict[str, List[str]]) -> FinancialMetrics:
        """Calculate financial metrics with extensive utility integration."""
        try:
            self.logger.info("💰 Calculating financial metrics with utility integration...")
            
            # Memory checkpoint for financial calculations
            with self.memory_optimizer.memory_checkpoint("financial_metrics_start"):
                metrics = FinancialMetrics()
                
                # Extract price data with safe operations
                price_columns = [col for col in data.columns if 'close' in col.lower() or 'price' in col.lower()]
                if not price_columns:
                    self.logger.warning("No price columns found, using first numeric column")
                    numeric_cols = data.select_dtypes(include=[np.number]).columns
                    price_columns = [numeric_cols[0]] if len(numeric_cols) > 0 else []
                
                if not price_columns:
                    self.logger.error("No numeric columns found for financial calculations")
                    return metrics
                
                price_data = data[price_columns[0]].dropna()
                
                # Calculate returns using safe mathematical operations
                returns = price_data.pct_change().dropna()
                
                # Daily returns
                daily_returns = safe_mean(returns.tolist()) if len(returns) > 0 else 0.0
                metrics.returns['daily'] = safe_float(daily_returns)
                
                # Annualized returns using safe power operation
                if len(returns) > 0:
                    annualized_return = safe_power(1 + daily_returns, 252) - 1
                    metrics.returns['annualized'] = safe_float(annualized_return)
                
                # Volatility calculations using safe mathematical operations
                if len(returns) > 1:
                    daily_volatility = safe_std(returns.tolist())
                    metrics.volatility['daily'] = safe_float(daily_volatility)
                    
                    # Annualized volatility using safe square root
                    annualized_volatility = safe_sqrt(252) * daily_volatility
                    metrics.volatility['annualized'] = safe_float(annualized_volatility)
                
                # Sharpe ratio calculation using safe division
                if metrics.volatility.get('annualized', 0) > 0:
                    excess_return = metrics.returns.get('annualized', 0) - self.risk_free_rate
                    sharpe_ratio = safe_divide(excess_return, metrics.volatility['annualized'])
                    metrics.sharpe_ratio['overall'] = safe_float(sharpe_ratio)
                
                # Maximum drawdown calculation
                if len(price_data) > 1:
                    cumulative_returns = (1 + returns).cumprod()
                    running_max = cumulative_returns.expanding().max()
                    drawdown = (cumulative_returns - running_max) / running_max
                    max_drawdown = drawdown.min()
                    metrics.max_drawdown['overall'] = safe_float(max_drawdown)
                
                # VaR calculations using safe mathematical operations
                if len(returns) > 0:
                    for confidence_level in self.var_confidence_levels:
                        var_percentile = (1 - confidence_level) * 100
                        var_value = np.percentile(returns, var_percentile)
                        metrics.var_95[f'{confidence_level}'] = safe_float(var_value)
                
                # Kelly criterion calculation for position sizing
                if len(returns) > 0:
                    win_rate = len(returns[returns > 0]) / len(returns)
                    avg_win = safe_mean(returns[returns > 0].tolist()) if len(returns[returns > 0]) > 0 else 0
                    avg_loss = abs(safe_mean(returns[returns < 0].tolist())) if len(returns[returns < 0]) > 0 else 0
                    
                    if avg_loss > 0:
                        kelly_fraction = safe_kelly_calculation(win_rate, avg_win, avg_loss)
                        metrics.kelly_criterion = safe_float(kelly_fraction)
                
                # Monte Carlo simulation using M1 GPU optimization
                if self.gpu_manager and len(returns) > 100:
                    self.logger.info("🎲 Running Monte Carlo simulation with M1 GPU optimization...")
                    try:
                        mc_results = m1_monte_carlo_simulate(
                            returns.values, 
                            n_simulations=1000,
                            trading_days=252,
                            use_mps=True
                        )
                        
                        # Store Monte Carlo results
                        metrics.monte_carlo = {
                            'mean_return': safe_mean(mc_results.get('returns', [])),
                            'mean_sharpe': safe_mean(mc_results.get('sharpe_ratios', [])),
                            'mean_max_drawdown': safe_mean(mc_results.get('max_drawdowns', [])),
                            'var_95_mc': safe_mean(mc_results.get('var_95', [])),
                            'convergence_history': mc_results.get('convergence_history', [])
                        }
                    except Exception as e:
                        self.logger.warning(f"Monte Carlo simulation failed: {e}")
                
                # Log metrics using safe operations
                self.logger.info(f"✅ Financial metrics calculated:")
                self.logger.info(f"   Daily return: {metrics.returns.get('daily', 0):.4f}")
                self.logger.info(f"   Annualized return: {metrics.returns.get('annualized', 0):.4f}")
                self.logger.info(f"   Annualized volatility: {metrics.volatility.get('annualized', 0):.4f}")
                self.logger.info(f"   Sharpe ratio: {metrics.sharpe_ratio.get('overall', 0):.4f}")
                self.logger.info(f"   Max drawdown: {metrics.max_drawdown.get('overall', 0):.4f}")
                
                return metrics
                
        except Exception as e:
            self.logger.error(f"Failed to calculate financial metrics: {e}")
            return FinancialMetrics()
        finally:
            # Memory cleanup after financial calculations
            self.memory_optimizer.optimize_memory()

    @timed_operation("parallel_feature_processing")
    async def _parallel_feature_processing(self, data: pd.DataFrame, feature_groups: List[List[str]]) -> Dict[str, Any]:
        """Process feature groups in parallel using M1 CPU optimization."""
        try:
            self.logger.info("⚡ Processing feature groups in parallel with M1 CPU optimization...")
            
            # Memory checkpoint for parallel processing
            with self.memory_optimizer.memory_checkpoint("parallel_processing_start"):
                
                # Define processing function for each feature group
                def process_feature_group(feature_group: List[str]) -> Dict[str, Any]:
                    """Process a single feature group."""
                    try:
                        group_data = data[feature_group].dropna()
                        
                        # Calculate basic statistics
                        stats = {
                            'group_name': '_'.join(feature_group[:3]),  # Use first 3 features as name
                            'feature_count': len(feature_group),
                            'data_points': len(group_data),
                            'mean_values': group_data.mean().to_dict(),
                            'std_values': group_data.std().to_dict(),
                            'correlation_matrix': group_data.corr().to_dict() if len(feature_group) > 1 else {},
                            'memory_usage_mb': group_data.memory_usage(deep=True).sum() / (1024**2)
                        }
                        
                        # Calculate feature importance using safe operations
                        if len(group_data) > 0:
                            # Simple variance-based importance
                            variances = group_data.var()
                            total_variance = variances.sum()
                            if total_variance > 0:
                                importance_scores = (variances / total_variance).to_dict()
                                stats['importance_scores'] = importance_scores
                        
                        return stats
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to process feature group {feature_group}: {e}")
                        return {'error': str(e), 'group_name': '_'.join(feature_group[:3])}
                
                # Process feature groups in parallel using M1 CPU optimizer
                if len(feature_groups) > 1:
                    # Use parallel processing for multiple groups
                    results = self.cpu_optimizer.parallel_process(
                        feature_groups,
                        process_feature_group,
                        task_type="cpu_bound",
                        timeout=300.0  # 5 minute timeout
                    )
                else:
                    # Single group processing
                    results = [process_feature_group(feature_groups[0])] if feature_groups else []
                
                # Combine results
                combined_results = {
                    'total_groups_processed': len(results),
                    'successful_groups': len([r for r in results if 'error' not in r]),
                    'failed_groups': len([r for r in results if 'error' in r]),
                    'group_results': results,
                    'total_memory_usage_mb': sum(r.get('memory_usage_mb', 0) for r in results if 'memory_usage_mb' in r),
                    'processing_timestamp': get_current_datetime().isoformat()
                }
                
                # Log processing results
                self.logger.info(f"✅ Parallel feature processing completed:")
                self.logger.info(f"   Groups processed: {combined_results['total_groups_processed']}")
                self.logger.info(f"   Successful: {combined_results['successful_groups']}")
                self.logger.info(f"   Failed: {combined_results['failed_groups']}")
                self.logger.info(f"   Total memory usage: {combined_results['total_memory_usage_mb']:.1f}MB")
                
                return combined_results
                
        except Exception as e:
            self.logger.error(f"Failed to process feature groups in parallel: {e}")
            return {'error': str(e), 'total_groups_processed': 0}
        finally:
            # Memory cleanup after parallel processing
            self.memory_optimizer.optimize_memory()

    @timed_operation("monte_carlo_parallel_simulation")
    async def _parallel_monte_carlo_simulation(self, returns_data: np.ndarray, n_simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation in parallel using M1 CPU optimization."""
        try:
            self.logger.info("🎲 Running parallel Monte Carlo simulation with M1 CPU optimization...")
            
            # Memory checkpoint for Monte Carlo simulation
            with self.memory_optimizer.memory_checkpoint("monte_carlo_start"):
                
                # Use parallel Monte Carlo simulation
                mc_results = parallel_monte_carlo_simulation(
                    returns_data,
                    n_simulations=n_simulations,
                    simulation_func=optimized_monte_carlo_worker,
                    trading_days=252,
                    max_workers=self.cpu_optimizer.max_workers
                )
                
                # Process results using safe mathematical operations
                processed_results = {
                    'n_simulations': n_simulations,
                    'mean_return': safe_mean(mc_results.get('returns', [])),
                    'std_return': safe_std(mc_results.get('returns', [])),
                    'mean_sharpe': safe_mean(mc_results.get('sharpe_ratios', [])),
                    'mean_max_drawdown': safe_mean(mc_results.get('max_drawdowns', [])),
                    'mean_win_rate': safe_mean(mc_results.get('win_rates', [])),
                    'mean_volatility': safe_mean(mc_results.get('volatilities', [])),
                    'var_95': safe_mean(mc_results.get('var_95', [])),
                    'cvar_95': safe_mean(mc_results.get('cvar_95', [])),
                    'convergence_history': mc_results.get('convergence_history', []),
                    'simulation_timestamp': get_current_datetime().isoformat()
                }
                
                # Calculate confidence intervals using safe operations
                returns_array = np.array(mc_results.get('returns', []))
                if len(returns_array) > 0:
                    processed_results['confidence_intervals'] = {
                        'return_95_ci': [np.percentile(returns_array, 2.5), np.percentile(returns_array, 97.5)],
                        'return_99_ci': [np.percentile(returns_array, 0.5), np.percentile(returns_array, 99.5)]
                    }
                
                # Log simulation results
                self.logger.info(f"✅ Parallel Monte Carlo simulation completed:")
                self.logger.info(f"   Simulations: {n_simulations}")
                self.logger.info(f"   Mean return: {processed_results['mean_return']:.4f}")
                self.logger.info(f"   Mean Sharpe: {processed_results['mean_sharpe']:.4f}")
                self.logger.info(f"   Mean max drawdown: {processed_results['mean_max_drawdown']:.4f}")
                
                return processed_results
                
        except Exception as e:
            self.logger.error(f"Failed to run parallel Monte Carlo simulation: {e}")
            return {'error': str(e), 'n_simulations': 0}
        finally:
            # Memory cleanup after Monte Carlo simulation
            self.memory_optimizer.optimize_memory()
            
            # Validate required columns
            required_columns = ['timestamp', 'composite_cluster_id']
            missing_columns = [col for col in required_columns if col not in unified_data.columns]
            if missing_columns:
                self.logger.error(f'Missing required columns: {missing_columns}')
                return None
            
            # Validate regime data
            regime_data = unified_data['composite_cluster_id'].dropna()
            if regime_data.empty:
                self.logger.error('No valid regime data found')
                return None
            
            # Data quality validation
            unified_data = self._validate_and_fix_data_quality(unified_data)
            
            self.logger.info(f'✅ Loaded and validated data: {len(unified_data)} rows, {len(unified_data.columns)} columns')
            return unified_data
            
        except Exception as e:
            self.logger.error(f'Failed to load and validate data: {e}')
            return None

    def _validate_and_fix_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix data quality issues."""
        self.logger.info('🔍 Validating and fixing data quality...')
        
        # Remove duplicates
        if 'timestamp' in data.columns:
            duplicate_count = data['timestamp'].duplicated().sum()
            if duplicate_count > 0:
                self.logger.info(f'🗑️ Removing {duplicate_count} duplicate timestamps')
                data = data.drop_duplicates(subset=['timestamp'], keep='last')
        
        # Sort by timestamp
        if 'timestamp' in data.columns:
            if not data['timestamp'].is_monotonic_increasing:
                self.logger.info('📈 Sorting data by timestamp')
                data = data.sort_values('timestamp').reset_index(drop=True)
        
        # Set datetime index
        if 'timestamp' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            try:
                data['timestamp'] = pd.to_datetime(data['timestamp'])
                data = data.set_index('timestamp')
                self.logger.info('📅 Set datetime index')
            except Exception as e:
                self.logger.warning(f'⚠️ Could not set datetime index: {e}')
        
        # Handle missing values
        missing_before = data.isnull().sum().sum()
        if missing_before > 0:
            # Forward fill for regime data, drop for other columns
            if 'composite_cluster_id' in data.columns:
                data['composite_cluster_id'] = data['composite_cluster_id'].fillna(method='ffill')
            
            # Drop rows with missing values in critical columns
            critical_columns = ['open', 'high', 'low', 'close', 'volume']
            available_critical = [col for col in critical_columns if col in data.columns]
            if available_critical:
                data = data.dropna(subset=available_critical)
            
            missing_after = data.isnull().sum().sum()
            self.logger.info(f'🔧 Fixed missing values: {missing_before} → {missing_after}')
        
        return data

    async def _handle_regime_balance(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle regime balance for imbalanced distributions."""
        try:
            self.logger.info('⚖️ Analyzing regime balance...')
            
            # Calculate regime distribution
            regime_counts = data['composite_cluster_id'].value_counts().to_dict()
            total_samples = len(data)
            regime_percentages = {str(k): v/total_samples for k, v in regime_counts.items()}
            
            # Calculate balance score
            balance_score = self._calculate_balance_score(regime_percentages)
            
            # Determine imbalance severity
            imbalance_severity = self._assess_imbalance_severity(regime_percentages)
            
            # Update regime balance metrics
            self.regime_balance.regime_counts = {str(k): v for k, v in regime_counts.items()}
            self.regime_balance.regime_percentages = regime_percentages
            self.regime_balance.balance_score = balance_score
            self.regime_balance.imbalance_severity = imbalance_severity
            
            self.logger.info(f'📊 Regime balance analysis:')
            self.logger.info(f'   Balance score: {balance_score:.3f}')
            self.logger.info(f'   Imbalance severity: {imbalance_severity}')
            self.logger.info(f'   Regime distribution: {regime_percentages}')
            
            # Apply rebalancing if needed
            if self.enable_regime_rebalancing and imbalance_severity in ['moderate', 'severe']:
                self.logger.info('🔄 Applying regime rebalancing...')
                balanced_data = await self._apply_regime_rebalancing(data, regime_counts)
                self.regime_balance.rebalancing_applied = True
                self.regime_balance.rebalancing_method = self.rebalancing_method
                return balanced_data
            else:
                self.logger.info('✅ Regime balance is acceptable, no rebalancing needed')
                return data
                
        except Exception as e:
            self.logger.error(f'Failed to handle regime balance: {e}')
            return data

    def _calculate_balance_score(self, regime_percentages: Dict[str, float]) -> float:
        """Calculate regime balance score (0-1, higher is better)."""
        if not regime_percentages:
            return 0.0
        
        # Calculate Gini coefficient for balance assessment
        percentages = list(regime_percentages.values())
        n = len(percentages)
        if n <= 1:
            return 1.0
        
        # Sort percentages
        sorted_percentages = sorted(percentages)
        
        # Calculate Gini coefficient
        cumsum = np.cumsum(sorted_percentages)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n if cumsum[-1] > 0 else 0
        
        # Convert to balance score (1 - gini)
        balance_score = 1 - gini
        return max(0.0, min(1.0, balance_score))

    def _assess_imbalance_severity(self, regime_percentages: Dict[str, float]) -> str:
        """Assess the severity of regime imbalance."""
        if not regime_percentages:
            return 'none'
        
        percentages = list(regime_percentages.values())
        max_pct = max(percentages)
        min_pct = min(percentages)
        
        # Calculate imbalance ratio
        imbalance_ratio = max_pct / min_pct if min_pct > 0 else float('inf')
        
        if imbalance_ratio <= 2.0:
            return 'none'
        elif imbalance_ratio <= 5.0:
            return 'mild'
        elif imbalance_ratio <= 10.0:
            return 'moderate'
        else:
            return 'severe'

    async def _apply_regime_rebalancing(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Apply regime rebalancing using specified method."""
        try:
            if self.rebalancing_method == 'oversample':
                return self._oversample_minority_regimes(data, regime_counts)
            elif self.rebalancing_method == 'undersample':
                return self._undersample_majority_regimes(data, regime_counts)
            elif self.rebalancing_method == 'smote':
                return await self._apply_smote_rebalancing(data, regime_counts)
            else:
                self.logger.warning(f'Unknown rebalancing method: {self.rebalancing_method}, using oversample')
                return self._oversample_minority_regimes(data, regime_counts)
                
        except Exception as e:
            self.logger.error(f'Failed to apply regime rebalancing: {e}')
            return data

    def _oversample_minority_regimes(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Oversample minority regimes to balance the dataset."""
        # Find target sample size (median of regime counts)
        target_size = int(np.median(list(regime_counts.values())))
        
        balanced_data = []
        for regime_id, count in regime_counts.items():
            regime_data = data[data['composite_cluster_id'] == regime_id]
            
            if count < target_size:
                # Oversample minority regime
                n_samples = target_size - count
                oversampled = regime_data.sample(n=n_samples, replace=True, random_state=42)
                balanced_data.append(pd.concat([regime_data, oversampled]))
                self.logger.info(f'📈 Oversampled regime {regime_id}: {count} → {target_size}')
            else:
                balanced_data.append(regime_data)
        
        result = pd.concat(balanced_data, ignore_index=True)
        result = result.sort_values('timestamp' if 'timestamp' in result.columns else result.index.name or 'index')
        
        self.logger.info(f'✅ Regime rebalancing completed: {len(data)} → {len(result)} samples')
        return result

    def _undersample_majority_regimes(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Undersample majority regimes to balance the dataset."""
        # Find target sample size (minimum regime count above threshold)
        min_count = min(regime_counts.values())
        target_size = max(min_count, self.min_regime_samples)
        
        balanced_data = []
        for regime_id, count in regime_counts.items():
            regime_data = data[data['composite_cluster_id'] == regime_id]
            
            if count > target_size:
                # Undersample majority regime
                undersampled = regime_data.sample(n=target_size, random_state=42)
                balanced_data.append(undersampled)
                self.logger.info(f'📉 Undersampled regime {regime_id}: {count} → {target_size}')
            else:
                balanced_data.append(regime_data)
        
        result = pd.concat(balanced_data, ignore_index=True)
        result = result.sort_values('timestamp' if 'timestamp' in result.columns else result.index.name or 'index')
        
        self.logger.info(f'✅ Regime rebalancing completed: {len(data)} → {len(result)} samples')
        return result

    async def _apply_smote_rebalancing(self, data: pd.DataFrame, regime_counts: Dict[int, int]) -> pd.DataFrame:
        """Apply SMOTE (Synthetic Minority Oversampling Technique) for regime rebalancing."""
        # This is a placeholder for SMOTE implementation
        # In practice, you would use imbalanced-learn library
        self.logger.warning('SMOTE rebalancing not implemented, using oversample instead')
        return self._oversample_minority_regimes(data, regime_counts)