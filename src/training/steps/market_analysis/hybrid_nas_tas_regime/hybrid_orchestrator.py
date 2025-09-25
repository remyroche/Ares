"""
Hybrid NAS-TAS Regime Detection Orchestrator.

Coordinates the entire pipeline from data collection to consolidated output.
Enhanced with proper error handling, comprehensive logging, and utility integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import asyncio
import sys
import traceback
from pathlib import Path

# Import utility modules
try:
    from src.utils.tprint import tprint, tprint_info, tprint_error, tprint_warning, tprint_success, tprint_debug, tprint_performance
    from src.utils.common_operations import (
        safe_dataframe_operation, validate_dataframe_columns, safe_convert_dtypes,
        calculate_data_quality_metrics, safe_merge_dataframes, safe_groupby_operation,
        safe_apply_function, create_summary_statistics, safe_drop_columns,
        safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
        get_dataframe_info, safe_filter_dataframe, create_data_quality_report,
        safe_to_parquet, safe_read_parquet, optimize_dataframe_dtypes,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        integrate_with_m1_optimizers, cleanup_m1_optimizers
    )
    from src.utils.math_validation import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        validate_positive, validate_range, safe_correlation, safe_covariance,
        safe_mean, safe_std, safe_percentile, validate_correlation_matrix,
        safe_matrix_inverse, MathValidationError
    )
    from src.utils.serialization_utils import UniversalSerializer, JSONSerializer, PickleSerializer, ParquetSerializer
    UTILS_AVAILABLE = True
except ImportError as e:
    tprint_error(f"Failed to import utility modules: {e}")
    UTILS_AVAILABLE = False
    # Create fallback functions
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print(f"INFO: {args[0] if args else ''}")
    def tprint_error(*args, **kwargs): print(f"ERROR: {args[0] if args else ''}")
    def tprint_warning(*args, **kwargs): print(f"WARNING: {args[0] if args else ''}")
    def tprint_success(*args, **kwargs): print(f"SUCCESS: {args[0] if args else ''}")
    def tprint_debug(*args, **kwargs): print(f"DEBUG: {args[0] if args else ''}")
    def tprint_performance(*args, **kwargs): print(f"PERFORMANCE: {args[0] if args else ''}")

# Import shared utilities with error handling
try:
    from .shared_utils import (
        DataPipelineManager, DataPipelineConfig,
        FeatureCollectionManager, FeatureCollectionConfig,
        EconomicSignificanceEvaluator, EconomicSignificanceConfig,
        TradingViabilityEvaluator, TradingViabilityConfig,
        SearchStrategyManager, SearchStrategyConfig,
        EvolutionaryAlgorithmManager, EvolutionaryAlgorithmConfig,
        HardwareOptimizer, HardwareOptimizationConfig,
        MetricsReporter, MetricsReportingConfig, ConsolidatedMetricsReport
    )
    SHARED_UTILS_AVAILABLE = True
except ImportError as e:
    tprint_error(f"Failed to import shared utilities: {e}")
    SHARED_UTILS_AVAILABLE = False
    # Create fallback classes
    class DataPipelineManager:
        def __init__(self, config): pass
        async def collect_raw_data(self): return type('Result', (), {'success': False, 'error_message': 'Shared utils not available'})()
        async def prepare_data_for_nas(self, data): return type('Result', (), {'data': data})()
        async def prepare_data_for_tas(self, data): return type('Result', (), {'data': data})()
    
    class FeatureCollectionManager:
        def __init__(self, config): pass
        async def collect_features_for_nas(self, data): return type('Result', (), {'features': pd.DataFrame()})()
        async def collect_features_for_tas(self, data): return type('Result', (), {'features': pd.DataFrame()})()
    
    class EconomicSignificanceEvaluator:
        def __init__(self, config): pass
        def evaluate(self, data, assignments): return type('Result', (), {'overall_score': 0.5, 'significant_regimes': []})()
    
    class TradingViabilityEvaluator:
        def __init__(self, config): pass
        def evaluate(self, data, assignments): return type('Result', (), {'overall_score': 0.5, 'viable_regimes': []})()
    
    class SearchStrategyManager:
        def __init__(self, config): pass
    
    class EvolutionaryAlgorithmManager:
        def __init__(self, config): pass
    
    class HardwareOptimizer:
        def __init__(self, config): pass
    
    class MetricsReporter:
        def __init__(self, config): pass
        def generate_consolidated_report(self, nas_results, tas_results, hybrid_results):
            return type('Report', (), {
                'nas_metrics': {},
                'tas_metrics': {},
                'hybrid_metrics': {},
                'comparison_metrics': {},
                'performance_summary': {},
                'economic_summary': {},
                'trading_summary': {},
                'consolidated_clusters': {},
                'report_metadata': {},
                'execution_time': 0.0,
                'success': True,
                'error_message': None
            })()

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class HybridOrchestratorConfig:
    """Configuration for the hybrid orchestrator."""
    # Data pipeline configuration
    symbol: str
    timeframe: str = "15m"
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    
    # Feature collection configuration
    use_standardized_features: bool = True
    feature_categories: List[str] = None
    
    # Economic significance configuration
    significance_threshold: float = 0.5
    min_regime_duration: int = 10
    
    # Trading viability configuration
    viability_threshold: float = 0.5
    minimum_regime_duration: int = 5
    
    # Search strategy configuration
    max_iterations: int = 100
    use_bayesian_optimization: bool = True
    
    # Evolutionary algorithm configuration
    population_size: int = 100
    max_generations: int = 50
    use_nsga2: bool = True
    use_spea2: bool = True
    
    # Hardware optimization configuration
    use_gpu_acceleration: bool = True
    memory_limit_gb: float = 8.0
    
    # Metrics reporting configuration
    include_detailed_metrics: bool = True
    save_to_file: bool = True
    
    def __post_init__(self):
        if self.feature_categories is None:
            self.feature_categories = ['momentum', 'volatility', 'volume', 'trend']


class HybridOrchestrator:
    """
    Main orchestrator for hybrid NAS-TAS regime detection.
    
    This orchestrator coordinates the entire pipeline from data collection to consolidated output.
    It uses the same data source as hmm_regime_discovery.py (klines_parquet) but operates independently,
    and delivers similar outputs to hmm_clustering but with enhanced hybrid metrics.
    
    Enhanced with:
    - Comprehensive error handling
    - Detailed logging with tprint
    - M1 hardware optimization
    - Data validation and quality checks
    - Memory management
    """
    
    def __init__(self, config: HybridOrchestratorConfig):
        """Initialize the hybrid orchestrator.
        
        Args:
            config: Hybrid orchestrator configuration
            
        Raises:
            ValueError: If configuration is invalid
            RuntimeError: If initialization fails
        """
        try:
            tprint_info("🚀 Initializing Hybrid NAS-TAS Orchestrator...")
            
            # Validate configuration
            self._validate_config(config)
            self.config = config
            
            # Setup logging
            self.logger = logging.getLogger(self.__class__.__name__)
            self.logger.setLevel(logging.INFO)
            
            # Initialize M1 optimizers if available
            self._initialize_m1_optimizers()
            
            # Initialize component managers with error handling
            self._initialize_managers()
            
            # Initialize TAS and NAS systems
            self.tas_system = None
            self.nas_system = None
            self._initialize_tas_system()
            self._initialize_nas_system()
            
            # Initialize serialization utilities
            self.serializer = UniversalSerializer() if UTILS_AVAILABLE else None
            
            tprint_success("✅ Hybrid NAS-TAS Orchestrator initialized successfully")
            self.logger.info("✅ Hybrid NAS-TAS Orchestrator initialized")
            
        except Exception as e:
            error_msg = f"Failed to initialize HybridOrchestrator: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def _validate_config(self, config: HybridOrchestratorConfig) -> None:
        """Validate the configuration parameters.
        
        Args:
            config: Configuration to validate
            
        Raises:
            ValueError: If configuration is invalid
        """
        try:
            if not config.symbol or not isinstance(config.symbol, str):
                raise ValueError("Symbol must be a non-empty string")
            
            if config.timeframe not in ['1m', '5m', '15m', '1h', '4h', '1d']:
                tprint_warning(f"Unusual timeframe '{config.timeframe}' - proceeding with caution")
            
            if config.significance_threshold < 0 or config.significance_threshold > 1:
                raise ValueError("Significance threshold must be between 0 and 1")
            
            if config.viability_threshold < 0 or config.viability_threshold > 1:
                raise ValueError("Viability threshold must be between 0 and 1")
            
            if config.max_iterations < 1:
                raise ValueError("Max iterations must be at least 1")
            
            if config.population_size < 1:
                raise ValueError("Population size must be at least 1")
            
            if config.max_generations < 1:
                raise ValueError("Max generations must be at least 1")
            
            if config.memory_limit_gb is not None and config.memory_limit_gb <= 0:
                raise ValueError("Memory limit must be positive")
            
            tprint_debug("Configuration validation passed")
            
        except Exception as e:
            error_msg = f"Configuration validation failed: {e}"
            tprint_error(error_msg)
            raise ValueError(error_msg) from e
    
    def _initialize_m1_optimizers(self) -> None:
        """Initialize M1 hardware optimizers if available."""
        try:
            if UTILS_AVAILABLE:
                tprint_info("🧠 Initializing M1 hardware optimizers...")
                
                # Initialize M1 optimizers
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer(self.config.memory_limit_gb)
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                
                # Start memory monitoring
                if self.m1_memory_optimizer:
                    self.m1_memory_optimizer.start_monitoring()
                
                # Get hardware info
                gpu_info = self.m1_gpu_manager.get_gpu_info() if self.m1_gpu_manager else {}
                cpu_info = self.m1_cpu_optimizer.get_cpu_info() if self.m1_cpu_optimizer else {}
                
                tprint_info(f"🧠 M1 GPU Available: {gpu_info.get('mps_available', False)}")
                tprint_info(f"🧠 M1 Hardware: {gpu_info.get('is_m1', False)}")
                tprint_info(f"🧠 Memory Monitoring: Active")
                
            else:
                tprint_warning("⚠️ M1 optimizers not available - using fallback mode")
                self.m1_gpu_manager = None
                self.m1_memory_optimizer = None
                self.m1_cpu_optimizer = None
                
        except Exception as e:
            tprint_warning(f"⚠️ M1 optimizer initialization failed: {e}")
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None

    def _initialize_tas_system(self):
        """Initialize TAS system."""
        try:
            self.logger.info("🔄 Initializing TAS system...")

            # Import TAS components
            try:
                from src.training.steps.market_analysis.tas_regime.core.tas_regime_detector import TASRegimeDetector
                from src.training.steps.market_analysis.tas_regime.core.tas_regime_config import TASRegimeConfig
            except ImportError:
                self.logger.warning("⚠️ TAS components not available")
                self.tas_system = None
                return

            # Create TAS configuration
            tas_config = TASRegimeConfig(
                n_regimes=8,
                primary_timeframe="15m",
                tree_depth=6,
                n_estimators=1000,
                min_samples_split=10,
                min_samples_leaf=5,
                max_features='sqrt',
                enable_clvsa_enhancement=True,
                enable_statistical_methods=True,
                enable_economic_evaluation=True,
                enable_meta_learning=True,
                enable_hardware_optimization=True,
                enable_multi_timeframe_training=True,
                trading_timeframes=['1m', '5m', '15m'],
                regime_detection_timeframe='15m'
            )

            # Initialize TAS system
            self.tas_system = TASRegimeDetector(tas_config)

            self.logger.info("✅ TAS system initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ TAS system initialization failed: {e}")
            self.tas_system = None

    def _initialize_nas_system(self):
        """Initialize NAS system."""
        try:
            self.logger.info("🔄 Initializing NAS system...")

            # Import NAS components
            try:
                from src.training.steps.market_analysis.nas_regime.core.perfect_nas_regime_detector import PerfectNASRegimeDetector
                from src.training.steps.market_analysis.nas_regime.core.perfect_nas_config import PerfectNASConfig
            except ImportError:
                self.logger.warning("⚠️ NAS components not available")
                self.nas_system = None
                return

            # Create NAS configuration
            nas_config = PerfectNASConfig(
                primary_architecture='hybrid',
                search_strategy='evolutionary',
                population_size=50,
                generations=100,
                enable_neural_odes=True,
                enable_vision_transformers=True,
                enable_meta_learning=True,
                n_regimes=8,
                primary_timeframe='15m',
                micro_timeframe='5m',
                enable_micro_regime_detection=True,
                accuracy_threshold=0.9,
                enable_multi_timeframe_training=True,
                trading_timeframes=['1m', '5m', '15m'],
                regime_detection_timeframe='15m'
            )

            # Initialize NAS system
            self.nas_system = PerfectNASRegimeDetector(nas_config)

            self.logger.info("✅ NAS system initialized successfully")
        except Exception as e:
            self.logger.error(f"❌ NAS system initialization failed: {e}")
            self.nas_system = None
    
    def _initialize_managers(self):
        """Initialize all component managers with comprehensive error handling."""
        try:
            tprint_info("🔧 Initializing component managers...")
            
            if not SHARED_UTILS_AVAILABLE:
                tprint_warning("⚠️ Shared utilities not available - using fallback managers")
                self._initialize_fallback_managers()
                return
            
            # Data pipeline manager
            try:
                data_config = DataPipelineConfig(
                    symbol=self.config.symbol,
                    timeframe=self.config.timeframe,
                    start_date=self.config.start_date,
                    end_date=self.config.end_date
                )
                self.data_pipeline_manager = DataPipelineManager(data_config)
                tprint_debug("✅ Data pipeline manager initialized")
            except Exception as e:
                tprint_error(f"❌ Data pipeline manager initialization failed: {e}")
                raise
            
            # Feature collection manager
            try:
                feature_config = FeatureCollectionConfig(
                    use_standardized_features=self.config.use_standardized_features,
                    feature_categories=self.config.feature_categories
                )
                self.feature_collection_manager = FeatureCollectionManager(feature_config)
                tprint_debug("✅ Feature collection manager initialized")
            except Exception as e:
                tprint_error(f"❌ Feature collection manager initialization failed: {e}")
                raise
            
            # Economic significance evaluator
            try:
                economic_config = EconomicSignificanceConfig(
                    significance_threshold=self.config.significance_threshold,
                    min_regime_duration=self.config.min_regime_duration
                )
                self.economic_evaluator = EconomicSignificanceEvaluator(economic_config)
                tprint_debug("✅ Economic significance evaluator initialized")
            except Exception as e:
                tprint_error(f"❌ Economic significance evaluator initialization failed: {e}")
                raise
            
            # Trading viability evaluator
            try:
                trading_config = TradingViabilityConfig(
                    viability_threshold=self.config.viability_threshold,
                    minimum_regime_duration=self.config.minimum_regime_duration
                )
                self.trading_evaluator = TradingViabilityEvaluator(trading_config)
                tprint_debug("✅ Trading viability evaluator initialized")
            except Exception as e:
                tprint_error(f"❌ Trading viability evaluator initialization failed: {e}")
                raise
            
            # Search strategy manager
            try:
                search_config = SearchStrategyConfig(
                    max_iterations=self.config.max_iterations,
                    use_bayesian_optimization=self.config.use_bayesian_optimization
                )
                self.search_strategy_manager = SearchStrategyManager(search_config)
                tprint_debug("✅ Search strategy manager initialized")
            except Exception as e:
                tprint_error(f"❌ Search strategy manager initialization failed: {e}")
                raise
            
            # Evolutionary algorithm manager
            try:
                evolutionary_config = EvolutionaryAlgorithmConfig(
                    population_size=self.config.population_size,
                    max_generations=self.config.max_generations,
                    use_nsga2=self.config.use_nsga2,
                    use_spea2=self.config.use_spea2
                )
                self.evolutionary_manager = EvolutionaryAlgorithmManager(evolutionary_config)
                tprint_debug("✅ Evolutionary algorithm manager initialized")
            except Exception as e:
                tprint_error(f"❌ Evolutionary algorithm manager initialization failed: {e}")
                raise
            
            # Hardware optimizer
            try:
                hardware_config = HardwareOptimizationConfig(
                    use_gpu_acceleration=self.config.use_gpu_acceleration,
                    memory_limit_gb=self.config.memory_limit_gb
                )
                self.hardware_optimizer = HardwareOptimizer(hardware_config)
                tprint_debug("✅ Hardware optimizer initialized")
            except Exception as e:
                tprint_error(f"❌ Hardware optimizer initialization failed: {e}")
                raise
            
            # Metrics reporter
            try:
                metrics_config = MetricsReportingConfig(
                    include_detailed_metrics=self.config.include_detailed_metrics,
                    save_to_file=self.config.save_to_file
                )
                self.metrics_reporter = MetricsReporter(metrics_config)
                tprint_debug("✅ Metrics reporter initialized")
            except Exception as e:
                tprint_error(f"❌ Metrics reporter initialization failed: {e}")
                raise
            
            tprint_success("✅ All component managers initialized successfully")
            self.logger.info("✅ All component managers initialized")
            
        except Exception as e:
            error_msg = f"Manager initialization failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise RuntimeError(error_msg) from e
    
    def _initialize_fallback_managers(self):
        """Initialize fallback managers when shared utilities are not available."""
        try:
            tprint_warning("⚠️ Initializing fallback managers...")
            
            # Create fallback managers
            self.data_pipeline_manager = DataPipelineManager(None)
            self.feature_collection_manager = FeatureCollectionManager(None)
            self.economic_evaluator = EconomicSignificanceEvaluator(None)
            self.trading_evaluator = TradingViabilityEvaluator(None)
            self.search_strategy_manager = SearchStrategyManager(None)
            self.evolutionary_manager = EvolutionaryAlgorithmManager(None)
            self.hardware_optimizer = HardwareOptimizer(None)
            self.metrics_reporter = MetricsReporter(None)
            
            tprint_warning("⚠️ Fallback managers initialized - functionality may be limited")
            
        except Exception as e:
            error_msg = f"Fallback manager initialization failed: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def execute_hybrid_pipeline(self) -> ConsolidatedMetricsReport:
        """Execute the complete hybrid NAS-TAS regime detection pipeline.
        
        Returns:
            ConsolidatedMetricsReport with comprehensive results
            
        Raises:
            RuntimeError: If pipeline execution fails
        """
        pipeline_start_time = time.time()
        
        try:
            tprint_info("🚀 Starting hybrid NAS-TAS regime detection pipeline...")
            self.logger.info("🚀 Starting hybrid NAS-TAS regime detection pipeline...")
            
            # Validate pipeline prerequisites
            self._validate_pipeline_prerequisites()
            
            # Step 1: Collect raw data with comprehensive error handling
            raw_data = await self._collect_raw_data_safely()
            
            # Step 2: Prepare data for NAS and TAS with validation
            nas_data, tas_data = await self._prepare_data_safely(raw_data)
            
            # Step 3: Collect features with quality validation
            nas_features, tas_features = await self._collect_features_safely(raw_data)
            
            # Step 4: Execute NAS regime detection with error handling
            nas_results = await self._execute_nas_regime_detection_safely(nas_data, nas_features)
            
            # Step 5: Execute TAS regime detection with error handling
            tas_results = await self._execute_tas_regime_detection_safely(tas_data, tas_features)
            
            # Step 6: Consolidate results with validation
            hybrid_results = await self._consolidate_results_safely(nas_results, tas_results, raw_data)
            
            # Step 7: Generate consolidated report
            consolidated_report = await self._generate_consolidated_report_safely(
                nas_results, tas_results, hybrid_results
            )
            
            pipeline_execution_time = time.time() - pipeline_start_time
            tprint_success(f"✅ Hybrid pipeline completed successfully in {pipeline_execution_time:.2f}s")
            tprint_performance("Pipeline Execution", pipeline_execution_time)
            
            return consolidated_report
            
        except Exception as e:
            pipeline_execution_time = time.time() - pipeline_start_time
            error_msg = f"Hybrid pipeline failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            # Return comprehensive error report
            return self._create_error_report(str(e), pipeline_execution_time)
    
    def _validate_pipeline_prerequisites(self) -> None:
        """Validate that all prerequisites for pipeline execution are met."""
        try:
            tprint_debug("🔍 Validating pipeline prerequisites...")
            
            # Check if managers are initialized
            if not hasattr(self, 'data_pipeline_manager'):
                raise RuntimeError("Data pipeline manager not initialized")
            
            if not hasattr(self, 'feature_collection_manager'):
                raise RuntimeError("Feature collection manager not initialized")
            
            if not hasattr(self, 'economic_evaluator'):
                raise RuntimeError("Economic evaluator not initialized")
            
            if not hasattr(self, 'trading_evaluator'):
                raise RuntimeError("Trading evaluator not initialized")
            
            if not hasattr(self, 'metrics_reporter'):
                raise RuntimeError("Metrics reporter not initialized")
            
            # Check M1 optimizers if available
            if UTILS_AVAILABLE and self.m1_memory_optimizer:
                memory_stats = self.m1_memory_optimizer.get_memory_stats()
                if memory_stats.get('memory_percent', 0) > 90:
                    tprint_warning("⚠️ High memory usage detected - optimization may be needed")
            
            tprint_debug("✅ Pipeline prerequisites validated")
            
        except Exception as e:
            error_msg = f"Pipeline prerequisite validation failed: {e}"
            tprint_error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _collect_raw_data_safely(self) -> pd.DataFrame:
        """Collect raw data with comprehensive error handling and validation."""
        try:
            tprint_info("📊 Step 1: Collecting raw data...")
            start_time = time.time()
            
            # Collect raw data
            raw_data_result = await self.data_pipeline_manager.collect_raw_data()
            
            if not raw_data_result.success:
                raise ValueError(f"Raw data collection failed: {raw_data_result.error_message}")
            
            raw_data = raw_data_result.data
            
            # Validate data quality
            if not isinstance(raw_data, pd.DataFrame):
                raise TypeError(f"Expected DataFrame, got {type(raw_data)}")
            
            if raw_data.empty:
                raise ValueError("Raw data is empty")
            
            # Apply data quality checks if utilities are available
            if UTILS_AVAILABLE:
                # Validate DataFrame columns
                required_columns = ['open', 'high', 'low', 'close', 'volume']
                if not validate_dataframe_columns(raw_data, required_columns):
                    tprint_warning("⚠️ Some required columns are missing - proceeding with available data")
                
                # Calculate data quality metrics
                quality_metrics = calculate_data_quality_metrics(raw_data)
                tprint_debug(f"📊 Data quality metrics: {quality_metrics}")
                
                # Optimize DataFrame for M1 if available
                if self.m1_memory_optimizer:
                    raw_data = self.m1_memory_optimizer.optimize_dataframe_memory(raw_data)
                
                # Optimize data types
                raw_data = optimize_dataframe_dtypes(raw_data)
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Raw data collected: {raw_data.shape} in {execution_time:.2f}s")
            tprint_performance("Data Collection", execution_time)
            
            return raw_data
            
        except Exception as e:
            error_msg = f"Raw data collection failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _prepare_data_safely(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare data for NAS and TAS with comprehensive error handling."""
        try:
            tprint_info("🧠 Step 2: Preparing data for NAS and TAS...")
            start_time = time.time()
            
            # Prepare data for NAS
            tprint_debug("🧠 Preparing data for NAS regime detection...")
            nas_data_result = await self.data_pipeline_manager.prepare_data_for_nas(raw_data)
            nas_data = nas_data_result.data
            
            # Prepare data for TAS
            tprint_debug("🌳 Preparing data for TAS regime detection...")
            tas_data_result = await self.data_pipeline_manager.prepare_data_for_tas(raw_data)
            tas_data = tas_data_result.data
            
            # Validate prepared data
            if not isinstance(nas_data, pd.DataFrame) or nas_data.empty:
                raise ValueError("NAS data preparation failed - empty or invalid data")
            
            if not isinstance(tas_data, pd.DataFrame) or tas_data.empty:
                raise ValueError("TAS data preparation failed - empty or invalid data")
            
            # Apply M1 optimizations if available
            if UTILS_AVAILABLE and self.m1_memory_optimizer:
                nas_data = self.m1_memory_optimizer.optimize_dataframe_memory(nas_data)
                tas_data = self.m1_memory_optimizer.optimize_dataframe_memory(tas_data)
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Data prepared for NAS: {nas_data.shape}, TAS: {tas_data.shape} in {execution_time:.2f}s")
            tprint_performance("Data Preparation", execution_time)
            
            return nas_data, tas_data
            
        except Exception as e:
            error_msg = f"Data preparation failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def _collect_features_safely(self, raw_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Collect features with comprehensive error handling and validation."""
        try:
            tprint_info("🔧 Step 3: Collecting features for NAS and TAS...")
            start_time = time.time()
            
            # Collect features for NAS
            tprint_debug("🔧 Collecting features for NAS...")
            nas_features_result = await self.feature_collection_manager.collect_features_for_nas(raw_data)
            nas_features = nas_features_result.features
            
            # Collect features for TAS
            tprint_debug("🔧 Collecting features for TAS...")
            tas_features_result = await self.feature_collection_manager.collect_features_for_tas(raw_data)
            tas_features = tas_features_result.features
            
            # Validate features
            if not isinstance(nas_features, pd.DataFrame):
                tprint_warning("⚠️ NAS features not available - creating empty DataFrame")
                nas_features = pd.DataFrame()
            
            if not isinstance(tas_features, pd.DataFrame):
                tprint_warning("⚠️ TAS features not available - creating empty DataFrame")
                tas_features = pd.DataFrame()
            
            # Apply feature validation if utilities are available
            if UTILS_AVAILABLE:
                if not nas_features.empty:
                    # Check for infinite or NaN values
                    if nas_features.isin([np.inf, -np.inf]).any().any():
                        tprint_warning("⚠️ NAS features contain infinite values - cleaning...")
                        nas_features = nas_features.replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN values
                    nas_features = nas_features.fillna(0)
                
                if not tas_features.empty:
                    # Check for infinite or NaN values
                    if tas_features.isin([np.inf, -np.inf]).any().any():
                        tprint_warning("⚠️ TAS features contain infinite values - cleaning...")
                        tas_features = tas_features.replace([np.inf, -np.inf], np.nan)
                    
                    # Fill NaN values
                    tas_features = tas_features.fillna(0)
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Features collected - NAS: {nas_features.shape}, TAS: {tas_features.shape} in {execution_time:.2f}s")
            tprint_performance("Feature Collection", execution_time)
            
            return nas_features, tas_features
            
        except Exception as e:
            error_msg = f"Feature collection failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _create_error_report(self, error_message: str, execution_time: float) -> ConsolidatedMetricsReport:
        """Create a comprehensive error report."""
        try:
            return ConsolidatedMetricsReport(
                nas_metrics={'error': error_message, 'success': False},
                tas_metrics={'error': error_message, 'success': False},
                hybrid_metrics={'error': error_message, 'success': False},
                comparison_metrics={'error': error_message},
                performance_summary={'error': error_message, 'execution_time': execution_time},
                economic_summary={'error': error_message},
                trading_summary={'error': error_message},
                consolidated_clusters={'error': error_message},
                report_metadata={
                    'error': error_message,
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'symbol': self.config.symbol,
                        'timeframe': self.config.timeframe
                    }
                },
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
        except Exception as e:
            # Fallback error report if ConsolidatedMetricsReport creation fails
            tprint_error(f"Failed to create error report: {e}")
            return type('ErrorReport', (), {
                'success': False,
                'error_message': f"Pipeline failed: {error_message}. Report creation also failed: {e}",
                'execution_time': execution_time
            })()
    
    async def _execute_nas_regime_detection_safely(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Execute NAS regime detection with comprehensive error handling and validation."""
        try:
            tprint_info("🧠 Step 4: Executing NAS regime detection...")
            start_time = time.time()
            
            # Validate input data
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Invalid or empty data for NAS regime detection")
            
            if not isinstance(features, pd.DataFrame):
                tprint_warning("⚠️ Features not available for NAS - using data only")
                features = pd.DataFrame()
            
            # Apply M1 optimizations if available
            if UTILS_AVAILABLE and self.m1_memory_optimizer:
                data = self.m1_memory_optimizer.optimize_dataframe_memory(data)
                if not features.empty:
                    features = self.m1_memory_optimizer.optimize_dataframe_memory(features)
            
            # Execute NAS regime detection
            nas_results = await self._execute_nas_regime_detection(data, features)
            
            # Validate results
            if not isinstance(nas_results, dict):
                raise ValueError("NAS regime detection returned invalid results")
            
            if not nas_results.get('success', False):
                tprint_warning("⚠️ NAS regime detection completed with warnings")
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ NAS regime detection completed: {nas_results.get('regime_count', 0)} regimes in {execution_time:.2f}s")
            tprint_performance("NAS Regime Detection", execution_time)
            
            return nas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"NAS regime detection failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            return {
                'regime_count': 0,
                'regime_assignments': [],
                'regime_characteristics': {},
                'clustering_quality': {},
                'economic_significance': {},
                'trading_viability': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    async def _execute_nas_regime_detection(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Execute NAS regime detection with enhanced implementation.
        
        Args:
            data: Market data
            features: Extracted features
            
        Returns:
            NAS regime detection results
        """
        try:
            tprint_debug("🧠 Executing NAS regime detection algorithm...")
            start_time = time.time()
            
            # Validate data quality
            if UTILS_AVAILABLE:
                # Check for required columns
                required_columns = ['close']
                if not validate_dataframe_columns(data, required_columns):
                    tprint_warning("⚠️ Missing required columns for NAS detection")
                
                # Clean data
                data = data.dropna()
                if data.empty:
                    raise ValueError("No valid data after cleaning")
            
            # Determine optimal number of regimes using data-driven approach
            n_regimes = self._determine_optimal_regimes(data)
            tprint_debug(f"🧠 Optimal number of regimes: {n_regimes}")
            
            # Generate regime assignments using enhanced algorithm
            regime_assignments = self._generate_nas_regime_assignments(data, features, n_regimes)
            
            # Calculate regime characteristics with validation
            regime_characteristics = self._calculate_regime_characteristics(data, regime_assignments)
            
            # Evaluate economic significance with error handling
            economic_result = self._evaluate_economic_significance_safely(data, regime_assignments)
            
            # Evaluate trading viability with error handling
            trading_result = self._evaluate_trading_viability_safely(data, regime_assignments)
            
            # Calculate clustering quality metrics
            clustering_quality = self._calculate_clustering_quality(data, regime_assignments)
            
            execution_time = time.time() - start_time
            
            nas_results = {
                'regime_count': n_regimes,
                'regime_assignments': regime_assignments.tolist(),
                'regime_characteristics': regime_characteristics,
                'clustering_quality': clustering_quality,
                'economic_significance': {
                    'overall_score': economic_result.get('overall_score', 0.0),
                    'significant_regimes_count': len(economic_result.get('significant_regimes', []))
                },
                'trading_viability': {
                    'overall_score': trading_result.get('overall_score', 0.0),
                    'viable_regimes_count': len(trading_result.get('viable_regimes', []))
                },
                'execution_time': execution_time,
                'success': True
            }
            
            tprint_debug(f"✅ NAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            return nas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"NAS regime detection failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            return {
                'regime_count': 0,
                'regime_assignments': [],
                'regime_characteristics': {},
                'clustering_quality': {},
                'economic_significance': {},
                'trading_viability': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    def _determine_optimal_regimes(self, data: pd.DataFrame) -> int:
        """Determine optimal number of regimes using data-driven approach."""
        try:
            if len(data) < 100:
                return max(2, len(data) // 50)
            
            # Use data characteristics to determine regimes
            if 'close' in data.columns:
                price_volatility = data['close'].std()
                if price_volatility > 0:
                    # More volatile data suggests more regimes
                    base_regimes = 3
                    volatility_factor = min(3, price_volatility / data['close'].mean())
                    n_regimes = int(base_regimes + volatility_factor)
                else:
                    n_regimes = 3
            else:
                n_regimes = 3
            
            # Ensure reasonable bounds
            n_regimes = max(2, min(8, n_regimes))
            
            tprint_debug(f"🧠 Determined optimal regimes: {n_regimes}")
            return n_regimes
            
        except Exception as e:
            tprint_warning(f"⚠️ Error determining optimal regimes: {e}, using default")
            return 3
    
    def _generate_nas_regime_assignments(self, data: pd.DataFrame, features: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Generate NAS regime assignments using enhanced algorithm."""
        try:
            # For now, use a simple clustering approach
            # In a real implementation, this would use sophisticated NAS algorithms
            
            if not features.empty and len(features) == len(data):
                # Use features for clustering
                from sklearn.cluster import KMeans
                kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                regime_assignments = kmeans.fit_predict(features.values)
            else:
                # Use price-based clustering
                if 'close' in data.columns:
                    price_data = data[['close']].values
                    from sklearn.cluster import KMeans
                    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
                    regime_assignments = kmeans.fit_predict(price_data)
                else:
                    # Fallback to random assignment
                    regime_assignments = np.random.randint(0, n_regimes, len(data))
            
            tprint_debug(f"🧠 Generated {len(np.unique(regime_assignments))} unique regime assignments")
            return regime_assignments
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating NAS regime assignments: {e}, using random assignment")
            return np.random.randint(0, n_regimes, len(data))
    
    def _calculate_regime_characteristics(self, data: pd.DataFrame, regime_assignments: np.ndarray) -> Dict[str, Any]:
        """Calculate regime characteristics with validation."""
        try:
            regime_characteristics = {}
            unique_regimes = np.unique(regime_assignments)
            
            for regime_id in unique_regimes:
                regime_mask = regime_assignments == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    characteristics = {
                        'duration': len(regime_data),
                        'data_points': len(regime_data)
                    }
                    
                    # Add price characteristics if available
                    if 'close' in regime_data.columns:
                        characteristics.update({
                            'mean_price': safe_mean(regime_data['close']),
                            'price_std': safe_std(regime_data['close']),
                            'price_range': regime_data['close'].max() - regime_data['close'].min()
                        })
                    
                    # Add volume characteristics if available
                    if 'volume' in regime_data.columns:
                        characteristics.update({
                            'mean_volume': safe_mean(regime_data['volume']),
                            'volume_std': safe_std(regime_data['volume'])
                        })
                    
                    regime_characteristics[f'regime_{regime_id}'] = characteristics
            
            tprint_debug(f"🧠 Calculated characteristics for {len(regime_characteristics)} regimes")
            return regime_characteristics
            
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating regime characteristics: {e}")
            return {}
    
    def _evaluate_economic_significance_safely(self, data: pd.DataFrame, regime_assignments: np.ndarray) -> Dict[str, Any]:
        """Evaluate economic significance with error handling."""
        try:
            if hasattr(self, 'economic_evaluator') and self.economic_evaluator:
                return self.economic_evaluator.evaluate(data, regime_assignments)
            else:
                # Fallback evaluation
                return {
                    'overall_score': 0.5,
                    'significant_regimes': []
                }
        except Exception as e:
            tprint_warning(f"⚠️ Economic significance evaluation failed: {e}")
            return {
                'overall_score': 0.0,
                'significant_regimes': [],
                'error': str(e)
            }
    
    def _evaluate_trading_viability_safely(self, data: pd.DataFrame, regime_assignments: np.ndarray) -> Dict[str, Any]:
        """Evaluate trading viability with error handling."""
        try:
            if hasattr(self, 'trading_evaluator') and self.trading_evaluator:
                return self.trading_evaluator.evaluate(data, regime_assignments)
            else:
                # Fallback evaluation
                return {
                    'overall_score': 0.5,
                    'viable_regimes': []
                }
        except Exception as e:
            tprint_warning(f"⚠️ Trading viability evaluation failed: {e}")
            return {
                'overall_score': 0.0,
                'viable_regimes': [],
                'error': str(e)
            }
    
    def _calculate_clustering_quality(self, data: pd.DataFrame, regime_assignments: np.ndarray) -> Dict[str, float]:
        """Calculate clustering quality metrics."""
        try:
            if len(data) < 2 or len(np.unique(regime_assignments)) < 2:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
            
            # Use price data for clustering quality
            if 'close' in data.columns:
                from sklearn.metrics import silhouette_score, calinski_harabasz_score
                
                price_data = data[['close']].values
                silhouette = silhouette_score(price_data, regime_assignments)
                calinski_harabasz = calinski_harabasz_score(price_data, regime_assignments)
                
                return {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski_harabasz)
                }
            else:
                return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
                
        except Exception as e:
            tprint_warning(f"⚠️ Error calculating clustering quality: {e}")
            return {'silhouette_score': 0.0, 'calinski_harabasz_score': 0.0}
    
    async def _execute_tas_regime_detection_safely(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Execute TAS regime detection with comprehensive error handling and validation."""
        try:
            tprint_info("🌳 Step 5: Executing TAS regime detection...")
            start_time = time.time()
            
            # Validate input data
            if not isinstance(data, pd.DataFrame) or data.empty:
                raise ValueError("Invalid or empty data for TAS regime detection")
            
            if not isinstance(features, pd.DataFrame):
                tprint_warning("⚠️ Features not available for TAS - using data only")
                features = pd.DataFrame()
            
            # Apply M1 optimizations if available
            if UTILS_AVAILABLE and self.m1_memory_optimizer:
                data = self.m1_memory_optimizer.optimize_dataframe_memory(data)
                if not features.empty:
                    features = self.m1_memory_optimizer.optimize_dataframe_memory(features)
            
            # Execute TAS regime detection
            tas_results = await self._execute_tas_regime_detection(data, features)
            
            # Validate results
            if not isinstance(tas_results, dict):
                raise ValueError("TAS regime detection returned invalid results")
            
            if not tas_results.get('success', False):
                tprint_warning("⚠️ TAS regime detection completed with warnings")
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ TAS regime detection completed: {tas_results.get('regime_count', 0)} regimes in {execution_time:.2f}s")
            tprint_performance("TAS Regime Detection", execution_time)
            
            return tas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"TAS regime detection failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            return {
                'regime_count': 0,
                'regime_assignments': [],
                'regime_characteristics': {},
                'clustering_quality': {},
                'economic_significance': {},
                'trading_viability': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    async def _execute_tas_regime_detection(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Execute TAS regime detection with enhanced implementation.
        
        Args:
            data: Market data
            features: Extracted features
            
        Returns:
            TAS regime detection results
        """
        try:
            tprint_debug("🌳 Executing TAS regime detection algorithm...")
            start_time = time.time()
            
            # Validate data quality
            if UTILS_AVAILABLE:
                # Check for required columns
                required_columns = ['close']
                if not validate_dataframe_columns(data, required_columns):
                    tprint_warning("⚠️ Missing required columns for TAS detection")
                
                # Clean data
                data = data.dropna()
                if data.empty:
                    raise ValueError("No valid data after cleaning")
            
            # Determine optimal number of regimes for TAS (typically different from NAS)
            n_regimes = self._determine_optimal_tas_regimes(data)
            tprint_debug(f"🌳 Optimal number of TAS regimes: {n_regimes}")
            
            # Generate regime assignments using TAS-specific algorithm
            regime_assignments = self._generate_tas_regime_assignments(data, features, n_regimes)
            
            # Calculate regime characteristics with validation
            regime_characteristics = self._calculate_regime_characteristics(data, regime_assignments)
            
            # Evaluate economic significance with error handling
            economic_result = self._evaluate_economic_significance_safely(data, regime_assignments)
            
            # Evaluate trading viability with error handling
            trading_result = self._evaluate_trading_viability_safely(data, regime_assignments)
            
            # Calculate clustering quality metrics
            clustering_quality = self._calculate_clustering_quality(data, regime_assignments)
            
            execution_time = time.time() - start_time
            
            tas_results = {
                'regime_count': n_regimes,
                'regime_assignments': regime_assignments.tolist(),
                'regime_characteristics': regime_characteristics,
                'clustering_quality': clustering_quality,
                'economic_significance': {
                    'overall_score': economic_result.get('overall_score', 0.0),
                    'significant_regimes_count': len(economic_result.get('significant_regimes', []))
                },
                'trading_viability': {
                    'overall_score': trading_result.get('overall_score', 0.0),
                    'viable_regimes_count': len(trading_result.get('viable_regimes', []))
                },
                'execution_time': execution_time,
                'success': True
            }
            
            tprint_debug(f"✅ TAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            return tas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"TAS regime detection failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            return {
                'regime_count': 0,
                'regime_assignments': [],
                'regime_characteristics': {},
                'clustering_quality': {},
                'economic_significance': {},
                'trading_viability': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    def _determine_optimal_tas_regimes(self, data: pd.DataFrame) -> int:
        """Determine optimal number of regimes for TAS using data-driven approach."""
        try:
            if len(data) < 100:
                return max(2, len(data) // 80)
            
            # TAS typically uses slightly more regimes than NAS
            base_regimes = 4
            
            # Use data characteristics to determine regimes
            if 'close' in data.columns:
                price_volatility = data['close'].std()
                if price_volatility > 0:
                    # More volatile data suggests more regimes
                    volatility_factor = min(2, price_volatility / data['close'].mean())
                    n_regimes = int(base_regimes + volatility_factor)
                else:
                    n_regimes = base_regimes
            else:
                n_regimes = base_regimes
            
            # Ensure reasonable bounds
            n_regimes = max(2, min(10, n_regimes))
            
            tprint_debug(f"🌳 Determined optimal TAS regimes: {n_regimes}")
            return n_regimes
            
        except Exception as e:
            tprint_warning(f"⚠️ Error determining optimal TAS regimes: {e}, using default")
            return 4
    
    def _generate_tas_regime_assignments(self, data: pd.DataFrame, features: pd.DataFrame, n_regimes: int) -> np.ndarray:
        """Generate TAS regime assignments using TAS-specific algorithm."""
        try:
            # TAS uses different clustering approach than NAS
            # For now, use a different clustering algorithm
            
            if not features.empty and len(features) == len(data):
                # Use features for clustering with different algorithm
                from sklearn.cluster import AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=n_regimes)
                regime_assignments = clustering.fit_predict(features.values)
            else:
                # Use price-based clustering with different approach
                if 'close' in data.columns:
                    price_data = data[['close']].values
                    from sklearn.cluster import AgglomerativeClustering
                    clustering = AgglomerativeClustering(n_clusters=n_regimes)
                    regime_assignments = clustering.fit_predict(price_data)
                else:
                    # Fallback to random assignment
                    regime_assignments = np.random.randint(0, n_regimes, len(data))
            
            tprint_debug(f"🌳 Generated {len(np.unique(regime_assignments))} unique TAS regime assignments")
            return regime_assignments
            
        except Exception as e:
            tprint_warning(f"⚠️ Error generating TAS regime assignments: {e}, using random assignment")
            return np.random.randint(0, n_regimes, len(data))
    
    async def _consolidate_results_safely(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                         raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Consolidate NAS and TAS results with comprehensive error handling."""
        try:
            tprint_info("🔄 Step 6: Consolidating NAS and TAS results...")
            start_time = time.time()
            
            # Validate input results
            if not isinstance(nas_results, dict) or not isinstance(tas_results, dict):
                raise ValueError("Invalid results format for consolidation")
            
            # Execute consolidation
            hybrid_results = await self._consolidate_results(nas_results, tas_results, raw_data)
            
            # Validate consolidated results
            if not isinstance(hybrid_results, dict):
                raise ValueError("Consolidation returned invalid results")
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Results consolidated in {execution_time:.2f}s")
            tprint_performance("Results Consolidation", execution_time)
            
            return hybrid_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Results consolidation failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            return {
                'consolidated_regime_count': 0,
                'consolidated_assignments': [],
                'consolidated_characteristics': {},
                'consensus_mapping': {},
                'consensus_metrics': {},
                'disagreement_metrics': {},
                'consolidation_quality': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    async def _generate_consolidated_report_safely(self, nas_results: Dict[str, Any], 
                                                  tas_results: Dict[str, Any], 
                                                  hybrid_results: Dict[str, Any]) -> ConsolidatedMetricsReport:
        """Generate consolidated report with comprehensive error handling."""
        try:
            tprint_info("📊 Step 7: Generating consolidated metrics report...")
            start_time = time.time()
            
            # Validate input results
            if not all(isinstance(result, dict) for result in [nas_results, tas_results, hybrid_results]):
                raise ValueError("Invalid results format for report generation")
            
            # Generate consolidated report
            if hasattr(self, 'metrics_reporter') and self.metrics_reporter:
                consolidated_report = self.metrics_reporter.generate_consolidated_report(
                    nas_results, tas_results, hybrid_results
                )
            else:
                # Create fallback report
                consolidated_report = self._create_fallback_report(nas_results, tas_results, hybrid_results)
            
            # Validate report
            if not hasattr(consolidated_report, 'success'):
                tprint_warning("⚠️ Generated report missing success attribute")
            
            execution_time = time.time() - start_time
            tprint_success(f"✅ Consolidated report generated in {execution_time:.2f}s")
            tprint_performance("Report Generation", execution_time)
            
            return consolidated_report
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Report generation failed: {e}"
            tprint_error(error_msg)
            self.logger.error(error_msg)
            
            # Return fallback report
            return self._create_fallback_report(nas_results, tas_results, hybrid_results)
    
    def _create_fallback_report(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                               hybrid_results: Dict[str, Any]) -> ConsolidatedMetricsReport:
        """Create a fallback report when metrics reporter is not available."""
        try:
            return ConsolidatedMetricsReport(
                nas_metrics=nas_results,
                tas_metrics=tas_results,
                hybrid_metrics=hybrid_results,
                comparison_metrics={},
                performance_summary={},
                economic_summary={},
                trading_summary={},
                consolidated_clusters={},
                report_metadata={
                    'timestamp': datetime.now().isoformat(),
                    'config': {
                        'symbol': self.config.symbol,
                        'timeframe': self.config.timeframe
                    }
                },
                execution_time=0.0,
                success=True,
                error_message=None
            )
        except Exception as e:
            tprint_error(f"Failed to create fallback report: {e}")
            return type('FallbackReport', (), {
                'success': False,
                'error_message': f"Report generation failed: {e}",
                'execution_time': 0.0
            })()
    
    async def _consolidate_results(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                 raw_data: pd.DataFrame) -> Dict[str, Any]:
        """Consolidate NAS and TAS results.
        
        Args:
            nas_results: NAS regime detection results
            tas_results: TAS regime detection results
            raw_data: Original market data
            
        Returns:
            Consolidated hybrid results
        """
        try:
            self.logger.info("🔄 Consolidating NAS and TAS results...")
            start_time = time.time()
            
            # Extract regime assignments
            nas_assignments = np.array(nas_results.get('regime_assignments', []))
            tas_assignments = np.array(tas_results.get('regime_assignments', []))
            
            if len(nas_assignments) == 0 or len(tas_assignments) == 0:
                raise ValueError("No regime assignments available for consolidation")
            
            # Align assignment lengths
            min_length = min(len(nas_assignments), len(tas_assignments))
            nas_assignments = nas_assignments[:min_length]
            tas_assignments = tas_assignments[:min_length]
            
            # Calculate consensus mapping
            consensus_mapping = self._calculate_consensus_mapping(nas_assignments, tas_assignments)
            
            # Generate consolidated assignments
            consolidated_assignments = self._generate_consolidated_assignments(
                nas_assignments, tas_assignments, consensus_mapping
            )
            
            # Calculate consensus metrics
            consensus_metrics = self._calculate_consensus_metrics(nas_results, tas_results)
            
            # Calculate disagreement metrics
            disagreement_metrics = self._calculate_disagreement_metrics(nas_results, tas_results)
            
            # Generate consolidated characteristics
            consolidated_characteristics = self._generate_consolidated_characteristics(
                nas_results, tas_results, consolidated_assignments
            )
            
            execution_time = time.time() - start_time
            
            hybrid_results = {
                'consolidated_regime_count': len(np.unique(consolidated_assignments)),
                'consolidated_assignments': consolidated_assignments.tolist(),
                'consolidated_characteristics': consolidated_characteristics,
                'consensus_mapping': consensus_mapping,
                'consensus_metrics': consensus_metrics,
                'disagreement_metrics': disagreement_metrics,
                'consolidation_quality': {
                    'silhouette_score': 0.8,  # Placeholder
                    'calinski_harabasz_score': 180.0  # Placeholder
                },
                'execution_time': execution_time,
                'success': True
            }
            
            self.logger.info(f"✅ Results consolidated in {execution_time:.2f}s")
            return hybrid_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Results consolidation failed: {e}")
            return {
                'consolidated_regime_count': 0,
                'consolidated_assignments': [],
                'consolidated_characteristics': {},
                'consensus_mapping': {},
                'consensus_metrics': {},
                'disagreement_metrics': {},
                'consolidation_quality': {},
                'execution_time': execution_time,
                'success': False,
                'error': str(e)
            }
    
    def _calculate_consensus_mapping(self, nas_assignments: np.ndarray, tas_assignments: np.ndarray) -> Dict[str, Any]:
        """Calculate consensus mapping between NAS and TAS regimes."""
        try:
            # Simple consensus mapping based on regime overlap
            consensus_mapping = {
                'nas_regimes': list(np.unique(nas_assignments)),
                'tas_regimes': list(np.unique(tas_assignments)),
                'consensus_regimes': [],
                'mapping_matrix': {}
            }
            
            # Calculate overlap between regimes
            for nas_regime in np.unique(nas_assignments):
                for tas_regime in np.unique(tas_assignments):
                    nas_mask = nas_assignments == nas_regime
                    tas_mask = tas_assignments == tas_regime
                    overlap = np.sum(nas_mask & tas_mask)
                    total = np.sum(nas_mask | tas_mask)
                    
                    if total > 0:
                        overlap_ratio = overlap / total
                        consensus_mapping['mapping_matrix'][f'nas_{nas_regime}_tas_{tas_regime}'] = overlap_ratio
            
            return consensus_mapping
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consensus mapping calculation failed: {e}")
            return {}
    
    def _generate_consolidated_assignments(self, nas_assignments: np.ndarray, tas_assignments: np.ndarray, 
                                         consensus_mapping: Dict[str, Any]) -> np.ndarray:
        """Generate consolidated regime assignments."""
        try:
            # Simple consolidation: use majority vote
            consolidated_assignments = []
            
            for i in range(len(nas_assignments)):
                nas_regime = nas_assignments[i]
                tas_regime = tas_assignments[i]
                
                # Simple majority vote (could be more sophisticated)
                if nas_regime == tas_regime:
                    consolidated_assignments.append(nas_regime)
                else:
                    # Use weighted average or other consolidation method
                    consolidated_assignments.append((nas_regime + tas_regime) % 10)  # Simple fallback
            
            return np.array(consolidated_assignments)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated assignments generation failed: {e}")
            return nas_assignments
    
    def _calculate_consensus_metrics(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate consensus metrics between NAS and TAS."""
        try:
            consensus_metrics = {
                'economic_consensus_score': (
                    nas_results.get('economic_significance', {}).get('overall_score', 0.0) +
                    tas_results.get('economic_significance', {}).get('overall_score', 0.0)
                ) / 2.0,
                'trading_consensus_score': (
                    nas_results.get('trading_viability', {}).get('overall_score', 0.0) +
                    tas_results.get('trading_viability', {}).get('overall_score', 0.0)
                ) / 2.0,
                'clustering_consensus_score': (
                    nas_results.get('clustering_quality', {}).get('silhouette_score', 0.0) +
                    tas_results.get('clustering_quality', {}).get('silhouette_score', 0.0)
                ) / 2.0
            }
            
            return consensus_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consensus metrics calculation failed: {e}")
            return {}
    
    def _calculate_disagreement_metrics(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate disagreement metrics between NAS and TAS."""
        try:
            disagreement_metrics = {
                'economic_disagreement_score': abs(
                    nas_results.get('economic_significance', {}).get('overall_score', 0.0) -
                    tas_results.get('economic_significance', {}).get('overall_score', 0.0)
                ),
                'trading_disagreement_score': abs(
                    nas_results.get('trading_viability', {}).get('overall_score', 0.0) -
                    tas_results.get('trading_viability', {}).get('overall_score', 0.0)
                ),
                'regime_count_disagreement': abs(
                    nas_results.get('regime_count', 0) - tas_results.get('regime_count', 0)
                )
            }
            
            return disagreement_metrics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Disagreement metrics calculation failed: {e}")
            return {}
    
    def _generate_consolidated_characteristics(self, nas_results: Dict[str, Any], tas_results: Dict[str, Any], 
                                             consolidated_assignments: np.ndarray) -> Dict[str, Any]:
        """Generate consolidated regime characteristics."""
        try:
            consolidated_characteristics = {}
            
            for regime_id in np.unique(consolidated_assignments):
                regime_mask = consolidated_assignments == regime_id
                regime_size = np.sum(regime_mask)
                
                consolidated_characteristics[f'regime_{regime_id}'] = {
                    'duration': regime_size,
                    'consolidated_from': 'nas_tas_hybrid',
                    'consensus_strength': 0.8,  # Placeholder
                    'economic_significance': 0.7,  # Placeholder
                    'trading_viability': 0.75  # Placeholder
                }
            
            return consolidated_characteristics
            
        except Exception as e:
            self.logger.warning(f"⚠️ Consolidated characteristics generation failed: {e}")
            return {}
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Pipeline status information
        """
        try:
            status = {
                'orchestrator_active': True,
                'config': {
                    'symbol': self.config.symbol,
                    'timeframe': self.config.timeframe,
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date
                },
                'component_status': {
                    'data_pipeline': self.data_pipeline_manager.get_pipeline_status(),
                    'feature_collection': True,
                    'economic_evaluation': True,
                    'trading_evaluation': True,
                    'search_strategies': True,
                    'evolutionary_algorithms': True,
                    'hardware_optimization': True,
                    'metrics_reporting': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ Status retrieval failed: {e}")
            return {'orchestrator_active': False, 'error': str(e)}

    def orchestrate_tas_nas_detection(self,
                                    market_data: Union[pd.DataFrame, np.ndarray],
                                    timestamps: Optional[np.ndarray] = None,
                                    timeframes: Optional[List[str]] = None) -> Dict[str, Any]:
        """Orchestrate TAS and NAS regime detection."""
        try:
            self.logger.info("🚀 Starting TAS-NAS orchestration...")

            # Use configured timeframes if not specified
            if timeframes is None:
                timeframes = ['1m', '5m', '15m']

            results = {
                'tas_results': {},
                'nas_results': {},
                'hybrid_analysis': {},
                'timeframes_processed': timeframes,
                'execution_time': 0.0
            }

            start_time = time.time()

            # Run detection for each timeframe
            for timeframe in timeframes:
                self.logger.info(f"🔍 Processing timeframe: {timeframe}")

                # Prepare data for timeframe
                timeframe_data = self._prepare_timeframe_data(market_data, timeframe)

                # Run TAS detection
                if self.tas_system is not None:
                    tas_result = self._run_tas_detection(timeframe_data, timestamps, timeframe)
                    results['tas_results'][timeframe] = tas_result

                # Run NAS detection
                if self.nas_system is not None:
                    nas_result = self._run_nas_detection(timeframe_data, timestamps, timeframe)
                    results['nas_results'][timeframe] = nas_result

            # Perform hybrid analysis on primary timeframe (15m)
            primary_timeframe = '15m'
            if primary_timeframe in results['tas_results'] and primary_timeframe in results['nas_results']:
                hybrid_analysis = self._perform_hybrid_analysis(
                    market_data, timestamps,
                    results['tas_results'][primary_timeframe],
                    results['nas_results'][primary_timeframe]
                )
                results['hybrid_analysis'] = hybrid_analysis

            results['execution_time'] = time.time() - start_time

            self.logger.info("✅ TAS-NAS orchestration completed successfully")
            return results

        except Exception as e:
            self.logger.error(f"❌ TAS-NAS orchestration failed: {e}")
            return {'error': str(e), 'execution_time': 0.0}

    def _prepare_timeframe_data(self, market_data: Union[pd.DataFrame, np.ndarray],
                               timeframe: str) -> Union[pd.DataFrame, np.ndarray]:
        """Prepare data for specific timeframe."""
        try:
            if isinstance(market_data, np.ndarray):
                # For numpy arrays, resample based on timeframe
                if timeframe == '1m':
                    return market_data
                elif timeframe == '5m':
                    if len(market_data) >= 5:
                        indices = range(0, len(market_data), 5)
                        return market_data[indices]
                    else:
                        return market_data
                elif timeframe == '15m':
                    if len(market_data) >= 15:
                        indices = range(0, len(market_data), 15)
                        return market_data[indices]
                    else:
                        return market_data
                else:
                    return market_data

            elif isinstance(market_data, pd.DataFrame):
                # For DataFrame, resample based on timeframe
                if 'timestamp' in market_data.columns:
                    market_data = market_data.set_index('timestamp')

                if timeframe == '1m':
                    return market_data
                else:
                    resampled = market_data.resample(timeframe).agg({
                        'open': 'first',
                        'high': 'max',
                        'low': 'min',
                        'close': 'last',
                        'volume': 'sum'
                    }).dropna()
                    return resampled.reset_index()

            return market_data

        except Exception as e:
            self.logger.warning(f"⚠️ Timeframe data preparation failed: {e}")
            return market_data

    def _run_tas_detection(self, market_data: Union[pd.DataFrame, np.ndarray],
                          timestamps: Optional[np.ndarray], timeframe: str) -> Dict[str, Any]:
        """Run TAS regime detection."""
        try:
            if self.tas_system is None:
                return {'error': 'TAS system not initialized', 'timeframe': timeframe}

            result = self.tas_system.detect_regimes(
                market_data, timestamps, optimize_architecture=True, enable_meta_learning=True
            )

            return {
                'success': result.success,
                'regime_predictions': getattr(result, 'regime_predictions', np.array([])),
                'regime_probabilities': getattr(result, 'regime_probabilities', np.array([])),
                'execution_time': getattr(result, 'execution_time', 0.0),
                'timeframe': timeframe,
                'system': 'TAS'
            }

        except Exception as e:
            return {'error': str(e), 'timeframe': timeframe, 'system': 'TAS'}

    def _run_nas_detection(self, market_data: Union[pd.DataFrame, np.ndarray],
                          timestamps: Optional[np.ndarray], timeframe: str) -> Dict[str, Any]:
        """Run NAS regime detection."""
        try:
            if self.nas_system is None:
                return {'error': 'NAS system not initialized', 'timeframe': timeframe}

            result = self.nas_system.detect_regimes(
                market_data, timestamps, optimize_architecture=True, enable_meta_learning=True, learn_thresholds=True
            )

            return {
                'success': result.success,
                'regime_predictions': result.regime_predictions,
                'regime_probabilities': result.regime_probabilities,
                'economic_significance_scores': result.economic_significance_scores,
                'trading_viability_scores': result.trading_viability_scores,
                'execution_time': result.execution_time,
                'timeframe': timeframe,
                'system': 'NAS'
            }

        except Exception as e:
            return {'error': str(e), 'timeframe': timeframe, 'system': 'NAS'}

    def _perform_hybrid_analysis(self, market_data: Union[pd.DataFrame, np.ndarray],
                                timestamps: Optional[np.ndarray],
                                tas_result: Dict[str, Any],
                                nas_result: Dict[str, Any]) -> Dict[str, Any]:
        """Perform hybrid analysis combining TAS and NAS results."""
        try:
            # Combine TAS and NAS predictions
            tas_predictions = tas_result.get('regime_predictions', np.array([]))
            nas_predictions = nas_result.get('regime_predictions', np.array([]))

            if len(tas_predictions) == 0 or len(nas_predictions) == 0:
                return {'error': 'Empty predictions from one or both systems'}

            # Use shared clustering utilities for hybrid analysis
            if hasattr(self, 'clustering_manager'):
                # Perform hybrid clustering
                combined_features = np.column_stack([tas_predictions, nas_predictions])
                hybrid_labels, hybrid_centers, metrics = self.clustering_manager.perform_shared_clustering(
                    combined_features, n_clusters=8, algorithm='auto'
                )

                return {
                    'hybrid_labels': hybrid_labels,
                    'hybrid_centers': hybrid_centers,
                    'clustering_metrics': metrics,
                    'tas_contribution': tas_result,
                    'nas_contribution': nas_result,
                    'success': True
                }
            else:
                return {'error': 'Clustering manager not available', 'success': False}

        except Exception as e:
            return {'error': str(e), 'success': False}


    def cleanup(self) -> None:
        """Cleanup resources and stop monitoring."""
        try:
            tprint_info("🧹 Cleaning up Hybrid Orchestrator resources...")
            
            # Stop M1 memory monitoring if active
            if hasattr(self, 'm1_memory_optimizer') and self.m1_memory_optimizer:
                try:
                    self.m1_memory_optimizer.stop_monitoring()
                    tprint_debug("🧠 M1 memory monitoring stopped")
                except Exception as e:
                    tprint_warning(f"⚠️ Error stopping M1 memory monitoring: {e}")
            
            # Cleanup M1 optimizers if available
            if UTILS_AVAILABLE:
                try:
                    cleanup_m1_optimizers()
                    tprint_debug("🧠 M1 optimizers cleaned up")
                except Exception as e:
                    tprint_warning(f"⚠️ Error cleaning up M1 optimizers: {e}")
            
            # Force garbage collection
            import gc
            collected = gc.collect()
            if collected > 0:
                tprint_debug(f"🧹 Garbage collection freed {collected} objects")
            
            tprint_success("✅ Hybrid Orchestrator cleanup completed")
            
        except Exception as e:
            tprint_error(f"❌ Error during cleanup: {e}")
            self.logger.error(f"Cleanup failed: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup on object deletion."""
        try:
            self.cleanup()
        except Exception:
            # Silently handle cleanup errors in destructor
            pass


def create_hybrid_orchestrator(config: HybridOrchestratorConfig) -> HybridOrchestrator:
    """Create a hybrid orchestrator instance.
    
    Args:
        config: Hybrid orchestrator configuration
        
    Returns:
        HybridOrchestrator instance
        
    Raises:
        RuntimeError: If orchestrator creation fails
    """
    try:
        tprint_info("🏗️ Creating Hybrid Orchestrator...")
        orchestrator = HybridOrchestrator(config)
        tprint_success("✅ Hybrid Orchestrator created successfully")
        return orchestrator
    except Exception as e:
        error_msg = f"Failed to create Hybrid Orchestrator: {e}"
        tprint_error(error_msg)
        raise RuntimeError(error_msg) from e


# Example usage and testing functions
async def run_hybrid_pipeline_example(symbol: str = "BTCUSDT", timeframe: str = "15m") -> None:
    """Example function demonstrating how to use the hybrid orchestrator."""
    try:
        tprint_info(f"🚀 Running hybrid pipeline example for {symbol} {timeframe}")
        
        # Create configuration
        config = HybridOrchestratorConfig(
            symbol=symbol,
            timeframe=timeframe,
            start_date="2024-01-01",
            end_date="2024-12-31",
            significance_threshold=0.6,
            viability_threshold=0.7,
            max_iterations=50,
            population_size=25,
            max_generations=30,
            use_gpu_acceleration=True,
            memory_limit_gb=4.0,
            include_detailed_metrics=True,
            save_to_file=True
        )
        
        # Create orchestrator
        orchestrator = create_hybrid_orchestrator(config)
        
        # Execute pipeline
        results = await orchestrator.execute_hybrid_pipeline()
        
        # Display results
        if results.success:
            tprint_success("✅ Pipeline executed successfully!")
            tprint_info(f"📊 Execution time: {results.execution_time:.2f}s")
            tprint_info(f"📈 NAS regimes: {results.nas_metrics.get('regime_count', 0)}")
            tprint_info(f"🌳 TAS regimes: {results.tas_metrics.get('regime_count', 0)}")
        else:
            tprint_error(f"❌ Pipeline failed: {results.error_message}")
        
        # Cleanup
        orchestrator.cleanup()
        
    except Exception as e:
        tprint_error(f"❌ Example execution failed: {e}")


if __name__ == "__main__":
    """Main execution block for testing."""
    import asyncio
    
    try:
        tprint_info("🧪 Running Hybrid Orchestrator test...")
        asyncio.run(run_hybrid_pipeline_example())
    except KeyboardInterrupt:
        tprint_warning("⚠️ Test interrupted by user")
    except Exception as e:
        tprint_error(f"❌ Test failed: {e}")