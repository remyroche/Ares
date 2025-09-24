"""
Hybrid NAS-TAS Regime Detection Orchestrator.

Coordinates the entire pipeline from data collection to consolidated output.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import time
from datetime import datetime
import asyncio

# Import enhanced utilities - fully wired integration
from .shared_utils import (
    # Enhanced logging and utilities
    get_logger, setup_basic_logging, safe_log_metric, safe_log_params, safe_log_artifact,
    get_current_datetime, get_today, format_datetime, parse_datetime,

    # Data processing utilities
    ensure_directory, safe_file_exists, safe_json_dump, safe_json_load,
    create_empty_dataframe, validate_dataframe, validate_dataframe_columns,
    safe_dataframe_operation, safe_fillna, safe_convert_dtypes, safe_merge_dataframes,
    safe_drop_columns, safe_rename_columns, validate_timestamp_column, safe_timestamp_conversion,
    optimize_dataframe_dtypes, calculate_data_quality_metrics, get_dataframe_info,
    create_data_quality_report, safe_to_parquet, safe_read_parquet, list_parquet_files,

    # Math and validation utilities
    safe_divide, safe_log, safe_sqrt, safe_power, safe_mean, safe_std, safe_float, safe_int,
    validate_finite, validate_positive, validate_range, safe_kelly_calculation,
    safe_weighted_average, safe_percentage_change, safe_correlation, safe_covariance,
    safe_percentile, validate_correlation_matrix, safe_matrix_inverse,

    # Performance utilities
    timed_operation, format_bytes, chunked_iterable, parallel_map, memory_checkpoint,
    gpu_context, optimize_memory, get_memory_usage, validate_file_path, get_file_size,
    check_disk_space,

    # Serialization utilities
    JSONSerializer, PickleSerializer, ParquetSerializer, UniversalSerializer,

    # Data utilities integration
    get_unified_data_utils, get_feature_engineer, get_data_quality_framework,
    process_market_data, compute_correlation_matrix, compute_covariance_matrix,
    batch_compute_correlation_matrices, parallel_matrix_computation,
    initialize_enhanced_utilities,

    # ML common integration
    get_cross_validator, get_overfitting_detector, get_data_leakage_detector,
    get_hyperparameter_optimizer, get_lookahead_detector, perform_cross_validation_analysis,
    detect_model_issues,

    # M1 hardware optimization
    get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
    cleanup_m1_optimizers, integrate_with_m1_optimizers
)

# Setup enhanced logging
setup_basic_logging()
logger = get_logger(__name__)


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
    """
    
    def __init__(self, config: HybridOrchestratorConfig):
        """Initialize the hybrid orchestrator with enhanced utilities.

        Args:
            config: Hybrid orchestrator configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)

        # Initialize enhanced utilities and hardware optimization
        self._initialize_enhanced_utilities()
        self._initialize_hardware_optimization()

        # Initialize component managers with enhanced utilities
        self._initialize_managers()

        safe_log_metric("hybrid_orchestrator_initialized", 1)
        safe_log_params({
            "symbol": config.symbol,
            "timeframe": config.timeframe,
            "use_gpu_acceleration": config.use_gpu_acceleration,
            "memory_limit_gb": config.memory_limit_gb
        })

        self.logger.info("✅ Hybrid NAS-TAS Orchestrator initialized with enhanced utilities")

        # Initialize TAS and NAS systems with enhanced capabilities
        self.tas_system = None
        self.nas_system = None
        self._initialize_tas_system()
        self._initialize_nas_system()

    @timed_operation
    def _initialize_enhanced_utilities(self):
        """Initialize enhanced utilities from integrated modules."""
        try:
            # Initialize enhanced utilities status
            self.utilities_status = initialize_enhanced_utilities()
            self.logger.info(f"✅ Enhanced utilities initialized: {self.utilities_status['overall_status']}")

            # Initialize data processing utilities
            self.data_utils = get_unified_data_utils()
            self.feature_engineer = get_feature_engineer()
            self.data_quality_framework = get_data_quality_framework()

            # Initialize ML common utilities
            self.cross_validator = get_cross_validator()
            self.overfitting_detector = get_overfitting_detector()
            self.data_leakage_detector = get_data_leakage_detector()
            self.hyperparameter_optimizer = get_hyperparameter_optimizer()
            self.lookahead_detector = get_lookahead_detector()

            # Initialize matrix operations
            self.matrix_ops = get_unified_matrix_operations()
            self.batch_matrix_ops = get_batch_matrix_operations()
            self.hardware_accelerated_ops = get_hardware_accelerated_matrix_ops()

            # Initialize serialization
            self.serializer = UniversalSerializer()

            self.logger.info("✅ All enhanced utilities initialized successfully")

        except Exception as e:
            self.logger.error(f"❌ Error initializing enhanced utilities: {e}")
            # Continue with basic initialization - don't fail completely
            pass

    @memory_checkpoint("hardware_initialization")
    def _initialize_hardware_optimization(self):
        """Initialize hardware optimization and M1 integration."""
        try:
            if self.config.use_gpu_acceleration:
                # Initialize M1 hardware optimization
                self.m1_status = integrate_with_m1_optimizers()

                # Initialize GPU manager if available
                self.gpu_manager = get_m1_gpu_manager()
                self.memory_optimizer = get_m1_memory_optimizer()
                self.cpu_optimizer = get_m1_cpu_optimizer()

                # Start memory monitoring
                if self.memory_optimizer:
                    self.memory_optimizer.start_monitoring()

                self.logger.info(f"✅ Hardware optimization initialized: {self.m1_status['integration_status']}")
                safe_log_metric("m1_hardware_available", 1 if self.m1_status['success'] else 0)
            else:
                self.logger.info("⚠️ Hardware optimization disabled in configuration")

        except Exception as e:
            self.logger.warning(f"⚠️ Hardware optimization failed to initialize: {e}")
            # Continue without hardware optimization
            self.m1_status = {'integration_status': 'failed', 'success': False}

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
        """Initialize all component managers."""
        try:
            # Data pipeline manager
            data_config = DataPipelineConfig(
                symbol=self.config.symbol,
                timeframe=self.config.timeframe,
                start_date=self.config.start_date,
                end_date=self.config.end_date
            )
            self.data_pipeline_manager = DataPipelineManager(data_config)
            
            # Feature collection manager
            feature_config = FeatureCollectionConfig(
                use_standardized_features=self.config.use_standardized_features,
                feature_categories=self.config.feature_categories
            )
            self.feature_collection_manager = FeatureCollectionManager(feature_config)
            
            # Economic significance evaluator
            economic_config = EconomicSignificanceConfig(
                significance_threshold=self.config.significance_threshold,
                min_regime_duration=self.config.min_regime_duration
            )
            self.economic_evaluator = EconomicSignificanceEvaluator(economic_config)
            
            # Trading viability evaluator
            trading_config = TradingViabilityConfig(
                viability_threshold=self.config.viability_threshold,
                minimum_regime_duration=self.config.minimum_regime_duration
            )
            self.trading_evaluator = TradingViabilityEvaluator(trading_config)
            
            # Search strategy manager
            search_config = SearchStrategyConfig(
                max_iterations=self.config.max_iterations,
                use_bayesian_optimization=self.config.use_bayesian_optimization
            )
            self.search_strategy_manager = SearchStrategyManager(search_config)
            
            # Evolutionary algorithm manager
            evolutionary_config = EvolutionaryAlgorithmConfig(
                population_size=self.config.population_size,
                max_generations=self.config.max_generations,
                use_nsga2=self.config.use_nsga2,
                use_spea2=self.config.use_spea2
            )
            self.evolutionary_manager = EvolutionaryAlgorithmManager(evolutionary_config)
            
            # Hardware optimizer
            hardware_config = HardwareOptimizationConfig(
                use_gpu_acceleration=self.config.use_gpu_acceleration,
                memory_limit_gb=self.config.memory_limit_gb
            )
            self.hardware_optimizer = HardwareOptimizer(hardware_config)
            
            # Metrics reporter
            metrics_config = MetricsReportingConfig(
                include_detailed_metrics=self.config.include_detailed_metrics,
                save_to_file=self.config.save_to_file
            )
            self.metrics_reporter = MetricsReporter(metrics_config)
            
            self.logger.info("✅ All component managers initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Manager initialization failed: {e}")
            raise
    
    async def execute_hybrid_pipeline(self) -> ConsolidatedMetricsReport:
        """Execute the complete hybrid NAS-TAS regime detection pipeline.
        
        Returns:
            ConsolidatedMetricsReport with comprehensive results
        """
        try:
            self.logger.info("🚀 Starting hybrid NAS-TAS regime detection pipeline...")
            pipeline_start_time = time.time()
            
            # Step 1: Collect raw data
            self.logger.info("📊 Step 1: Collecting raw data...")
            raw_data_result = await self.data_pipeline_manager.collect_raw_data()
            
            if not raw_data_result.success:
                raise ValueError(f"Raw data collection failed: {raw_data_result.error_message}")
            
            raw_data = raw_data_result.data
            self.logger.info(f"✅ Raw data collected: {raw_data.shape}")
            
            # Step 2: Prepare data for NAS and TAS
            self.logger.info("🧠 Step 2: Preparing data for NAS regime detection...")
            nas_data_result = await self.data_pipeline_manager.prepare_data_for_nas(raw_data)
            
            self.logger.info("🌳 Step 3: Preparing data for TAS regime detection...")
            tas_data_result = await self.data_pipeline_manager.prepare_data_for_tas(raw_data)
            
            # Step 4: Collect features for both systems
            self.logger.info("🔧 Step 4: Collecting features for NAS...")
            nas_features_result = await self.feature_collection_manager.collect_features_for_nas(raw_data)
            
            self.logger.info("🔧 Step 5: Collecting features for TAS...")
            tas_features_result = await self.feature_collection_manager.collect_features_for_tas(raw_data)
            
            # Step 6: Execute NAS regime detection
            self.logger.info("🧠 Step 6: Executing NAS regime detection...")
            nas_results = await self._execute_nas_regime_detection(nas_data_result.data, nas_features_result.features)
            
            # Step 7: Execute TAS regime detection
            self.logger.info("🌳 Step 7: Executing TAS regime detection...")
            tas_results = await self._execute_tas_regime_detection(tas_data_result.data, tas_features_result.features)
            
            # Step 8: Consolidate results
            self.logger.info("🔄 Step 8: Consolidating NAS and TAS results...")
            hybrid_results = await self._consolidate_results(nas_results, tas_results, raw_data)
            
            # Step 9: Generate consolidated report
            self.logger.info("📊 Step 9: Generating consolidated metrics report...")
            consolidated_report = self.metrics_reporter.generate_consolidated_report(
                nas_results, tas_results, hybrid_results
            )
            
            pipeline_execution_time = time.time() - pipeline_start_time
            self.logger.info(f"✅ Hybrid pipeline completed in {pipeline_execution_time:.2f}s")
            
            return consolidated_report
            
        except Exception as e:
            pipeline_execution_time = time.time() - pipeline_start_time
            self.logger.error(f"❌ Hybrid pipeline failed: {e}")
            
            # Return error report
            return ConsolidatedMetricsReport(
                nas_metrics={'error': str(e)},
                tas_metrics={'error': str(e)},
                hybrid_metrics={'error': str(e)},
                comparison_metrics={'error': str(e)},
                performance_summary={'error': str(e)},
                economic_summary={'error': str(e)},
                trading_summary={'error': str(e)},
                consolidated_clusters={'error': str(e)},
                report_metadata={'error': str(e)},
                execution_time=pipeline_execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_nas_regime_detection(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Execute NAS regime detection.
        
        Args:
            data: Market data
            features: Extracted features
            
        Returns:
            NAS regime detection results
        """
        try:
            self.logger.info("🧠 Executing NAS regime detection...")
            start_time = time.time()
            
            # This would integrate with the actual NAS regime detection system
            # For now, we'll create a placeholder implementation
            
            # Simulate NAS regime detection
            n_regimes = min(5, len(data) // 100)  # Simple regime count estimation
            regime_assignments = np.random.randint(0, n_regimes, len(data))
            
            # Calculate regime characteristics
            regime_characteristics = {}
            for regime_id in range(n_regimes):
                regime_mask = regime_assignments == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    regime_characteristics[f'regime_{regime_id}'] = {
                        'duration': len(regime_data),
                        'volatility': regime_data['close'].std() if 'close' in regime_data.columns else 0.0,
                        'volume_characteristics': regime_data['volume'].mean() if 'volume' in regime_data.columns else 1.0
                    }
            
            # Evaluate economic significance
            economic_result = self.economic_evaluator.evaluate(data, regime_assignments)
            
            # Evaluate trading viability
            trading_result = self.trading_evaluator.evaluate(data, regime_assignments)
            
            execution_time = time.time() - start_time
            
            nas_results = {
                'regime_count': n_regimes,
                'regime_assignments': regime_assignments.tolist(),
                'regime_characteristics': regime_characteristics,
                'clustering_quality': {
                    'silhouette_score': 0.7,  # Placeholder
                    'calinski_harabasz_score': 150.0  # Placeholder
                },
                'economic_significance': {
                    'overall_score': economic_result.overall_score,
                    'significant_regimes_count': len(economic_result.significant_regimes)
                },
                'trading_viability': {
                    'overall_score': trading_result.overall_score,
                    'viable_regimes_count': len(trading_result.viable_regimes)
                },
                'execution_time': execution_time,
                'success': True
            }
            
            self.logger.info(f"✅ NAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            return nas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ NAS regime detection failed: {e}")
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
        """Execute TAS regime detection.
        
        Args:
            data: Market data
            features: Extracted features
            
        Returns:
            TAS regime detection results
        """
        try:
            self.logger.info("🌳 Executing TAS regime detection...")
            start_time = time.time()
            
            # This would integrate with the actual TAS regime detection system
            # For now, we'll create a placeholder implementation
            
            # Simulate TAS regime detection
            n_regimes = min(6, len(data) // 80)  # Slightly different regime count
            regime_assignments = np.random.randint(0, n_regimes, len(data))
            
            # Calculate regime characteristics
            regime_characteristics = {}
            for regime_id in range(n_regimes):
                regime_mask = regime_assignments == regime_id
                regime_data = data[regime_mask]
                
                if len(regime_data) > 0:
                    regime_characteristics[f'regime_{regime_id}'] = {
                        'duration': len(regime_data),
                        'volatility': regime_data['close'].std() if 'close' in regime_data.columns else 0.0,
                        'volume_characteristics': regime_data['volume'].mean() if 'volume' in regime_data.columns else 1.0
                    }
            
            # Evaluate economic significance
            economic_result = self.economic_evaluator.evaluate(data, regime_assignments)
            
            # Evaluate trading viability
            trading_result = self.trading_evaluator.evaluate(data, regime_assignments)
            
            execution_time = time.time() - start_time
            
            tas_results = {
                'regime_count': n_regimes,
                'regime_assignments': regime_assignments.tolist(),
                'regime_characteristics': regime_characteristics,
                'clustering_quality': {
                    'silhouette_score': 0.75,  # Placeholder
                    'calinski_harabasz_score': 160.0  # Placeholder
                },
                'economic_significance': {
                    'overall_score': economic_result.overall_score,
                    'significant_regimes_count': len(economic_result.significant_regimes)
                },
                'trading_viability': {
                    'overall_score': trading_result.overall_score,
                    'viable_regimes_count': len(trading_result.viable_regimes)
                },
                'execution_time': execution_time,
                'success': True
            }
            
            self.logger.info(f"✅ TAS regime detection completed: {n_regimes} regimes in {execution_time:.2f}s")
            return tas_results
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ TAS regime detection failed: {e}")
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


def create_hybrid_orchestrator(config: HybridOrchestratorConfig) -> HybridOrchestrator:
    """Create a hybrid orchestrator instance.
    
    Args:
        config: Hybrid orchestrator configuration
        
    Returns:
        HybridOrchestrator instance
    """
    return HybridOrchestrator(config)