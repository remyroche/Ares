from src.training.steps.standardized_parquet_handler import standardized_parquet_handler
"""
Unified Step08 Class Implementation - Part 2
"""

# Import required classes and functions
try:
    from src.utils.m1_gpu_utils import get_m1_gpu_manager, M1GPUManager
    from src.utils.m1_memory_optimizer import get_m1_memory_optimizer, M1MemoryOptimizer
    from src.utils.m1_cpu_optimizer import get_m1_cpu_optimizer, M1CPUOptimizer
    from src.utils.vectorized_processing_core import OptimizedPipelineExecutor, PipelineStage, PipelineExecutionMode
    from src.utils.enhanced_matrix_operations import EnhancedMatrixOperations, ErrorHandler
    from src.utils.enhanced_step_optimizations import IntelligentOptimizationSelector, OptimizationStrategy, WorkloadType, OptimizationProfile
    from src.utils.optimized_data_manager import OptimizedDataManager, DataMetadata
    ENHANCED_OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    ENHANCED_OPTIMIZATIONS_AVAILABLE = False

# Import data classes
try:
    from .step08_unified_complete import (
        FinancialMetrics, RiskMetrics, RegimeBalanceMetrics,
        FeatureSelectionValidation, Step08Results
    )
except ImportError:
    # Fallback definitions
    from typing import Dict, List, Any
    from dataclasses import dataclass, field

    @dataclass
    class FinancialMetrics:
        returns: Dict[str, float] = field(default_factory=dict)
        volatility: Dict[str, float] = field(default_factory=dict)
        sharpe_ratio: Dict[str, float] = field(default_factory=dict)
        var_95: Dict[str, float] = field(default_factory=dict)
        var_99: Dict[str, float] = field(default_factory=dict)
        max_drawdown: Dict[str, float] = field(default_factory=dict)
        calmar_ratio: Dict[str, float] = field(default_factory=dict)
        sortino_ratio: Dict[str, float] = field(default_factory=dict)
        information_ratio: Dict[str, float] = field(default_factory=dict)
        beta: Dict[str, float] = field(default_factory=dict)
        alpha: Dict[str, float] = field(default_factory=dict)

    @dataclass
    class RiskMetrics:
        portfolio_var: float = 0.0
        portfolio_es: float = 0.0
        concentration_risk: float = 0.0
        liquidity_risk: float = 0.0
        model_risk: float = 0.0
        regime_risk: float = 0.0
        feature_stability_risk: float = 0.0
        overfitting_risk: float = 0.0
        data_quality_risk: float = 0.0
        operational_risk: float = 0.0
        overall_risk_score: float = 0.0

    @dataclass
    class RegimeBalanceMetrics:
        regime_counts: Dict[str, int] = field(default_factory=dict)
        regime_percentages: Dict[str, float] = field(default_factory=dict)
        balance_score: float = 0.0
        imbalance_severity: str = "none"
        rebalancing_applied: bool = False
        rebalancing_method: str = ""
        min_samples_per_regime: int = 100
        target_balance_ratio: float = 0.8

    @dataclass
    class FeatureSelectionValidation:
        selection_bias_score: float = 0.0
        temporal_stability: float = 0.0
        regime_consistency: float = 0.0
        correlation_stability: float = 0.0
        importance_stability: float = 0.0
        overfitting_indicators: Dict[str, float] = field(default_factory=dict)
        validation_passed: bool = False
        warnings: List[str] = field(default_factory=list)

    @dataclass
    class Step08Results:
        regime_data: Any = None
        selected_features: Dict[str, List[str]] = field(default_factory=dict)
        financial_metrics: FinancialMetrics = field(default_factory=FinancialMetrics)
        risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
        regime_balance: RegimeBalanceMetrics = field(default_factory=RegimeBalanceMetrics)
        feature_validation: FeatureSelectionValidation = field(default_factory=FeatureSelectionValidation)
        execution_metadata: Dict[str, Any] = field(default_factory=dict)
        artifacts_generated: List[str] = field(default_factory=list)
        success: bool = False
        errors: List[str] = field(default_factory=list)
        warnings: List[str] = field(default_factory=list)

# Import decorators and utilities
try:
    from src.utils.common_operations import get_common_operations_health_status
except ImportError:
    def get_common_operations_health_status():
        return {'status': 'fallback'}

try:
    from src.core.decorators import with_tracing_span, handle_errors
except ImportError:
    def with_tracing_span(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def handle_errors(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Import unified data loader
try:
    from src.training.steps.data_collection.unified_data_loader import UnifiedDataLoader
    UNIFIED_DATA_LOADER_AVAILABLE = True
except ImportError:
    UNIFIED_DATA_LOADER_AVAILABLE = False
    UnifiedDataLoader = None

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

# Optional ml_common utilities
try:
    from src.utils.ml_common import (
        LookaheadProtection,
        DataQualityUtilities,
        FeatureSelectionFramework
    )
    ML_COMMON_AVAILABLE = True
except Exception:
    ML_COMMON_AVAILABLE = False

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
        # Initialize ml_common utilities (optional)
        self.ml_data_quality = DataQualityUtilities() if ML_COMMON_AVAILABLE else None
        self.ml_lookahead = LookaheadProtection() if ML_COMMON_AVAILABLE else None
        self.ml_feature_selection = FeatureSelectionFramework() if ML_COMMON_AVAILABLE else None
        
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

            # ml_common: data quality + lookahead checks
            if ML_COMMON_AVAILABLE and isinstance(unified_data, pd.DataFrame):
                try:
                    symbol = (training_input or {}).get('symbol', '') if training_input else ''
                    exchange = (training_input or {}).get('exchange', '') if training_input else ''
                    dq = await self.ml_data_quality.perform_comprehensive_validation(
                        unified_data, symbol=symbol, exchange=exchange, context='step08_unified_load'
                    ) if self.ml_data_quality else None
                    if dq and dq.get('has_critical_issues'):
                        self.logger.warning(f"⚠️ Data quality issues detected: {dq.get('critical_issues', [])}")
                    if self.ml_lookahead:
                        lr = await self.ml_lookahead.detect_and_prevent_leakage(
                            unified_data, symbol=symbol, exchange=exchange, context='step08_unified_load'
                        )
                        if lr.get('has_leakage'):
                            self.logger.error(f"🚨 Lookahead leakage indications: {lr.get('leakage_details', [])}")
                except Exception as _e:
                    self.logger.warning(f"ml_common validation skipped: {_e}")
            
            # Step 2: Regime balance analysis and handling
            self.logger.info('⚖️ Step 2: Analyzing and handling regime balance...')
            balanced_data = await self._handle_regime_balance(unified_data)
            
            # Step 3: Advanced feature selection with bias prevention
            self.logger.info('🔍 Step 3: Advanced feature selection with bias prevention...')
            selected_features = await self._advanced_feature_selection(balanced_data)

            # ml_common: post-selection feature importance audit
            if ML_COMMON_AVAILABLE and self.ml_feature_selection and isinstance(balanced_data, pd.DataFrame):
                try:
                    label_col = 'label' if 'label' in balanced_data.columns else None
                    labels = balanced_data[label_col] if label_col else None
                    symbol = (training_input or {}).get('symbol', '') if training_input else ''
                    exchange = (training_input or {}).get('exchange', '') if training_input else ''
                    imp = await self.ml_feature_selection.analyze_feature_importance(
                        balanced_data.drop(columns=['timestamp'], errors='ignore'),
                        labels=labels, symbol=symbol, exchange=exchange, context='step08_unified_post_select'
                    )
                    if imp.get('recommendations'):
                        self.logger.info(f"🎯 ML recommendations: {imp['recommendations']}")
                except Exception as _e:
                    self.logger.warning(f"ml_common feature analysis skipped: {_e}")
            
            # Step 4: Financial metrics calculation
            self.logger.info('💰 Step 4: Calculating financial metrics...')
            financial_metrics = await self._calculate_financial_metrics(balanced_data, selected_features)
            
            # Step 5: Risk assessment
            self.logger.info('⚠️ Step 5: Comprehensive risk assessment...')
            risk_metrics = await self._comprehensive_risk_assessment(balanced_data, selected_features, financial_metrics)
            
            # Step 6: Feature selection validation
            self.logger.info('✅ Step 6: Feature selection validation...')
            feature_validation = await self._validate_feature_selection(balanced_data, selected_features)

            # Step 6.5: Model interpretability analysis (SHAP/LIME integration)
            self.logger.info('🧠 Step 6.5: Model interpretability analysis...')
            interpretability_results = await self._perform_interpretability_analysis(
                balanced_data, selected_features
            )

            # Step 6.6: Walk-forward validation (step18 integration)
            self.logger.info('🔄 Step 6.6: Walk-forward validation...')
            validation_results = await self._perform_walk_forward_validation(
                balanced_data, selected_features
            )

            # Step 7: Generate comprehensive results
            self.logger.info('📋 Step 7: Generating comprehensive results...')
            results = await self._generate_comprehensive_results(
                balanced_data, selected_features, financial_metrics,
                risk_metrics, feature_validation, interpretability_results, validation_results, start_time
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
        try:
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

    # ============================================================================
    # ROBUST ML TRAINING METHODS (PROTECTED FROM STEP02_5 ISSUES)
    # ============================================================================

    def _perform_cross_validation(self, X: np.ndarray, y: np.ndarray, feature_names: np.ndarray) -> dict[str, Any]:
        """Perform cross-validation for model evaluation with temporal integrity and class imbalance handling."""
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.utils.class_weight import compute_sample_weight
            from sklearn.metrics import balanced_accuracy_score, f1_score

            cv_results = {}

            # Use Random Forest for CV as it's robust and fast
            rf_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)

            # Ensure minimum samples per fold with class balance considerations
            min_samples_per_fold = max(100, len(X) // 20)  # At least 100 samples or 5% of total
            max_splits = min(5, max(2, len(X) // 1000))

            # Calculate appropriate test size
            test_size = max(min_samples_per_fold, len(X) // (max_splits + 1))
            n_splits = min(max_splits, max(2, (len(X) - test_size) // test_size))

            tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
            self.logger.info(f'🔄 Using TimeSeriesSplit CV: {n_splits} splits, test_size={test_size}')

            # Initialize metrics arrays
            direction_scores = []
            balanced_accuracy_scores = []
            f1_macro_scores = []

            for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
                try:
                    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
                    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

                    # Check for single-class folds
                    if len(np.unique(y_train_fold)) < 2 or len(np.unique(y_test_fold)) < 2:
                        self.logger.warning(f'⚠️ Skipping fold {fold_idx}: single-class detected (train: {len(np.unique(y_train_fold))}, test: {len(np.unique(y_test_fold))})')
                        continue

                    # Compute class weights for imbalanced data
                    sample_weight = compute_sample_weight('balanced', y_train_fold)

                    # Fit model with class weights
                    rf_model.fit(X_train_fold, y_train_fold, sample_weight=sample_weight)

                    # Make predictions
                    y_pred = rf_model.predict(X_test_fold)

                    # Calculate balanced metrics
                    direction_scores.append(rf_model.score(X_test_fold, y_test_fold))
                    balanced_accuracy_scores.append(balanced_accuracy_score(y_test_fold, y_pred))
                    f1_macro_scores.append(f1_score(y_test_fold, y_pred, average='macro'))

                except Exception as fold_e:
                    self.logger.warning(f'⚠️ Fold {fold_idx} failed: {fold_e}')
                    continue

            # Store results only if we have valid folds
            if direction_scores:
                cv_results['direction_accuracy_scores'] = direction_scores
                cv_results['direction_accuracy_mean'] = np.mean(direction_scores)
                cv_results['direction_accuracy_std'] = np.std(direction_scores)

                cv_results['balanced_accuracy_scores'] = balanced_accuracy_scores
                cv_results['balanced_accuracy_mean'] = np.mean(balanced_accuracy_scores)
                cv_results['balanced_accuracy_std'] = np.std(balanced_accuracy_scores)

                cv_results['f1_macro_scores'] = f1_macro_scores
                cv_results['f1_macro_mean'] = np.mean(f1_macro_scores)
                cv_results['f1_macro_std'] = np.std(f1_macro_scores)

                cv_results['n_folds_completed'] = len(direction_scores)
                cv_results['total_folds'] = n_splits

                self.logger.info(f'🔄 CV Results - Accuracy: {cv_results["direction_accuracy_mean"]:.4f} ± {cv_results["direction_accuracy_std"]:.4f}')
                self.logger.info(f'🔄 CV Results - Balanced Accuracy: {cv_results["balanced_accuracy_mean"]:.4f} ± {cv_results["balanced_accuracy_std"]:.4f}')
                self.logger.info(f'🔄 CV Results - F1 Macro: {cv_results["f1_macro_mean"]:.4f} ± {cv_results["f1_macro_std"]:.4f}')
            else:
                self.logger.warning('⚠️ No valid CV folds completed')
                cv_results = self._get_fallback_cv_results()

            return cv_results

        except Exception as e:
            self.logger.error(f'❌ Cross-validation failed: {e}')
            return self._get_fallback_cv_results()

    def _get_fallback_cv_results(self) -> dict[str, Any]:
        """Get fallback cross-validation results."""
        return {
            'direction_accuracy_scores': [0.5] * 5,
            'direction_accuracy_mean': 0.5,
            'direction_accuracy_std': 0.0,
            'balanced_accuracy_scores': [0.5] * 5,
            'balanced_accuracy_mean': 0.5,
            'balanced_accuracy_std': 0.0,
            'f1_macro_scores': [0.5] * 5,
            'f1_macro_mean': 0.5,
            'f1_macro_std': 0.0,
            'n_folds_completed': 0,
            'total_folds': 5,
            'error': 'CV failed - using fallback results'
        }

    def _calculate_evaluation_metrics(self, models_results: dict[str, Any],
                                    cv_results: dict[str, Any],
                                    X_test: np.ndarray, y_dir_test: np.ndarray,
                                    y_vol_test: np.ndarray, ensemble_model: dict[str, Any] = None) -> dict[str, Any]:
        """Calculate comprehensive evaluation metrics with class imbalance awareness."""
        try:
            from sklearn.metrics import balanced_accuracy_score, f1_score, matthews_corrcoef
            from sklearn.utils.class_weight import compute_sample_weight

            # Find best performing models using balanced metrics
            best_balanced_accuracy = 0
            best_direction_model = None
            best_volatility_mae = float('inf')
            best_volatility_model = None

            # Aggregate feature importance across models
            all_feature_importance = {}

            for model_name, model_result in models_results.items():
                # Check direction performance with balanced metrics
                if 'direction' in model_result and 'predictions' in model_result['direction']:
                    try:
                        y_pred = model_result['direction']['predictions']

                        # Calculate balanced metrics
                        balanced_acc = balanced_accuracy_score(y_dir_test, y_pred)
                        f1_macro = f1_score(y_dir_test, y_pred, average='macro')
                        mcc = matthews_corrcoef(y_dir_test, y_pred)

                        # Store balanced metrics
                        model_result['direction']['balanced_accuracy'] = balanced_acc
                        model_result['direction']['f1_macro'] = f1_macro
                        model_result['direction']['matthews_corrcoef'] = mcc

                        # Update best model
                        if balanced_acc > best_balanced_accuracy:
                            best_balanced_accuracy = balanced_acc
                            best_direction_model = model_name

                        # Aggregate feature importance
                        if 'feature_importance' in model_result['direction']:
                            for feature, importance in model_result['direction']['feature_importance'].items():
                                if feature not in all_feature_importance:
                                    all_feature_importance[feature] = []
                                all_feature_importance[feature].append(importance)

                    except Exception as metric_e:
                        self.logger.warning(f'⚠️ Could not calculate balanced metrics for {model_name}: {metric_e}')
                        continue

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

            # Class distribution analysis
            class_distribution = {}
            if len(y_dir_test) > 0:
                unique_classes, class_counts = np.unique(y_dir_test, return_counts=True)
                class_distribution = {
                    f'class_{int(cls)}': int(count) for cls, count in zip(unique_classes, class_counts)
                }
                class_distribution['total_samples'] = len(y_dir_test)
                class_distribution['num_classes'] = len(unique_classes)

            return {
                'best_balanced_accuracy': best_balanced_accuracy,
                'best_direction_model': best_direction_model,
                'best_volatility_mae': best_volatility_mae,
                'best_volatility_model': best_volatility_model,
                'top_features': top_features,
                'avg_feature_importance': avg_feature_importance,
                'class_distribution': class_distribution,
                'cv_results_summary': {
                    'direction_accuracy_mean': cv_results.get('direction_accuracy_mean', 0.5),
                    'balanced_accuracy_mean': cv_results.get('balanced_accuracy_mean', 0.5),
                    'f1_macro_mean': cv_results.get('f1_macro_mean', 0.5),
                    'n_folds_completed': cv_results.get('n_folds_completed', 0),
                    'total_folds': cv_results.get('total_folds', 5)
                }
            }

        except Exception as e:
            self.logger.error(f'❌ Evaluation metrics calculation failed: {e}')
            return {
                'best_balanced_accuracy': 0.5,
                'best_direction_model': 'fallback',
                'error': str(e)
            }

    def _handle_ml_failure(self, error_message: str, error_type: str = "UNKNOWN_ERROR") -> dict[str, Any]:
        """Handle ML training failures with intelligent fast fail mechanism and proper error classification."""
        # Initialize failure tracking if not exists
        if not hasattr(self, 'ml_failure_count'):
            self.ml_failure_count = 0
            self.ml_failure_reasons = []

        self.ml_failure_count += 1
        self.ml_failure_reasons.append({
            'timestamp': datetime.now().isoformat(),
            'error_type': error_type,
            'error_message': error_message,
            'failure_count': self.ml_failure_count
        })

        # Classify failure severity with better granularity
        critical_errors = ["FORWARD_BIAS_ERROR", "DATA_UNAVAILABLE", "EMPTY_DATA", "NO_VALID_CHUNKS"]
        recoverable_errors = ["OPTUNA_ERROR", "CV_ERROR", "MODEL_FIT_ERROR", "ML_TRAINING_ERROR", "METHOD_VALIDATION_ERROR"]
        data_related_errors = ["SINGLE_CLASS_ERROR", "EXTREME_IMBALANCE_ERROR", "INSUFFICIENT_DATA_ERROR"]

        is_critical = error_type in critical_errors
        is_recoverable = error_type in recoverable_errors
        is_data_related = error_type in data_related_errors

        # Log with appropriate emoji and context
        if is_critical:
            self.logger.error(f'❌ CRITICAL ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.error(f'🚨 Critical Error Type: {error_type}')
        elif is_data_related:
            self.logger.warning(f'⚠️ DATA-RELATED ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.warning(f'📊 Data Error Type: {error_type} - may be expected in some chunks')
        elif is_recoverable:
            self.logger.warning(f'⚠️ RECOVERABLE ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.warning(f'📊 Recoverable Error Type: {error_type}')
        else:
            self.logger.warning(f'⚠️ ML Training Failure #{self.ml_failure_count}: {error_message}')
            self.logger.warning(f'📊 Error Type: {error_type}')

        # Intelligent fast fail logic with differentiated thresholds
        if hasattr(self, 'enable_fast_fail') and self.enable_fast_fail:
            if is_critical and self.ml_failure_count >= 2:  # Fail faster on critical errors
                self.logger.critical(f'🚨 FAST FAIL: {self.ml_failure_count} critical ML failures detected, aborting training')
                raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} critical ML training failures")
            elif is_data_related and self.ml_failure_count >= 10:  # More tolerant of data issues
                self.logger.warning(f'🚨 FAST FAIL: {self.ml_failure_count} data-related ML failures detected, aborting training')
                raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} data-related ML training failures")
            elif self.ml_failure_count >= getattr(self, 'max_ml_failures', 5):  # Original threshold for other errors
                self.logger.critical(f'🚨 FAST FAIL: {self.ml_failure_count} ML failures detected, aborting training')
                raise RuntimeError(f"Fast fail triggered after {self.ml_failure_count} ML training failures")

        # Return fallback result with failure information
        return self._get_fallback_ml_result_with_failure_info(error_message, error_type)

    def _get_fallback_ml_result_with_failure_info(self, error_message: str, error_type: str) -> dict[str, Any]:
        """Get fallback ML result with detailed failure information."""
        return {
            'direction_accuracy': 0.5,
            'balanced_accuracy': 0.5,
            'volatility_mae': 0.1,
            'model_type': 'fallback_due_to_failure',
            'training_samples': 0,
            'failure_info': {
                'error_message': error_message,
                'error_type': error_type,
                'timestamp': datetime.now().isoformat()
            }
        }

    def _detect_class_imbalance(self, y: np.ndarray, threshold: float = 0.95) -> dict[str, Any]:
        """Detect and analyze class imbalance in target variable."""
        try:
            unique_classes, class_counts = np.unique(y, return_counts=True)
            total_samples = len(y)

            # Calculate class ratios
            class_ratios = class_counts / total_samples
            max_class_ratio = np.max(class_ratios)
            min_class_ratio = np.min(class_ratios)

            # Identify dominant class
            dominant_class_idx = np.argmax(class_counts)
            dominant_class = unique_classes[dominant_class_idx]

            imbalance_info = {
                'num_classes': len(unique_classes),
                'total_samples': total_samples,
                'class_distribution': {f'class_{int(cls)}': int(count) for cls, count in zip(unique_classes, class_counts)},
                'class_ratios': {f'class_{int(cls)}': float(ratio) for cls, ratio in zip(unique_classes, class_ratios)},
                'max_class_ratio': float(max_class_ratio),
                'min_class_ratio': float(min_class_ratio),
                'dominant_class': int(dominant_class),
                'is_single_class': len(unique_classes) < 2,
                'is_extreme_imbalance': max_class_ratio > threshold,
                'imbalance_severity': 'extreme' if max_class_ratio > 0.95 else 'severe' if max_class_ratio > 0.85 else 'moderate' if max_class_ratio > 0.75 else 'balanced'
            }

            # Log imbalance information
            if imbalance_info['is_single_class']:
                self.logger.warning(f'🚨 Single-class dataset detected: only class {dominant_class} present ({total_samples} samples)')
            elif imbalance_info['is_extreme_imbalance']:
                self.logger.warning(f'⚠️ Extreme class imbalance: {max_class_ratio:.2%} of samples are class {dominant_class} ({imbalance_info["imbalance_severity"]} imbalance)')
            elif imbalance_info['max_class_ratio'] > 0.75:
                self.logger.info(f'ℹ️ Class imbalance detected: {max_class_ratio:.2%} of samples are class {dominant_class} ({imbalance_info["imbalance_severity"]} imbalance)')

            return imbalance_info

        except Exception as e:
            self.logger.error(f'❌ Class imbalance detection failed: {e}')
            return {
                'error': str(e),
                'is_single_class': False,
                'is_extreme_imbalance': False
            }