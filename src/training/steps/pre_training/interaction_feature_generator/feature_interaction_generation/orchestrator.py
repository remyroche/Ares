"""
Enhanced Main Orchestrator for Data-Driven Lookback Optimization System

This module orchestrates the three-stage Bayesian optimization system with comprehensive
enhancements including matrix operations, hardware optimization, and extensive logging:
1. IC Surface Estimation with HAC standard errors and matrix optimization
2. Walk-Forward Stability Testing with purged CV and ML utilities
3. Hierarchical Bayesian Shrinkage across families and symbols with hardware acceleration

The system replaces hardcoded lookback ceilings with data-driven inference
while maintaining production constraints and latency requirements.

Key Features:
- Extensive tprint logging throughout all operations
- Matrix operations integration for vectorized computations
- M1 hardware optimization for Apple Silicon
- ML utilities integration (Bayesian TPE, feature selection, data leakage detection)
- Comprehensive error handling and performance monitoring
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import warnings
import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# Import all stage modules
from .config import LookbackOptimizationConfig, FamilyType, create_default_config
from .ic_surface import ICSurfaceEstimator, ICSurfaceResult
from .wf_stability import StabilityTester, StabilityResult, MultiFamilyStabilityTester
from .hierarchical import HierarchicalBayesianShrinkage, HierarchicalResult, MultiSymbolHierarchicalShrinkage
from .decision import LookbackDecisionMaker, DecisionResult, MultiFamilyDecisionMaker
from .feature_families import MultiFamilyFeatureGenerator, FeatureResult

# Import comprehensive utility modules
try:
    from src.utils.tprint import (
        tprint, tprint_info, tprint_error, tprint_warning, tprint_success,
        tprint_debug, tprint_performance, tprint_progress
    )
    TPRINT_AVAILABLE = True
except ImportError:
    TPRINT_AVAILABLE = False
    def tprint(*args, **kwargs): print(*args, **kwargs)
    def tprint_info(*args, **kwargs): print("INFO:", *args, **kwargs)
    def tprint_error(*args, **kwargs): print("ERROR:", *args, **kwargs)
    def tprint_warning(*args, **kwargs): print("WARNING:", *args, **kwargs)
    def tprint_success(*args, **kwargs): print("SUCCESS:", *args, **kwargs)
    def tprint_debug(*args, **kwargs): print("DEBUG:", *args, **kwargs)
    def tprint_performance(*args, **kwargs): print("PERF:", *args, **kwargs)
    def tprint_progress(*args, **kwargs): print("PROGRESS:", *args, **kwargs)

# Import profit labeling framework for quality-based optimization
try:
    from src.training.steps.pre_training.profit_labeling.quality_scoring import (
        LabelQualityScorer, QualityScoringConfig, QualityMetrics, QualityMetric
    )
    from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
        VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig, LabelQualityScore
    )
    from src.training.steps.pre_training.profit_labeling.multi_target_scheme import (
        MultiTargetScheme, MultiTargetConfig, TargetBand
    )
    from src.training.steps.pre_training.profit_labeling.noise_gating import (
        NoiseGatingFilter, NoiseGatingConfig
    )
    PROFIT_LABELING_AVAILABLE = True
    tprint_success("✅ Profit labeling framework imported successfully")
except ImportError as e:
    PROFIT_LABELING_AVAILABLE = False
    tprint_warning(f"⚠️ Profit labeling framework not available: {e}")

# Import common operations and utilities
try:
    from src.utils.common_operations import (
        safe_divide, safe_log, safe_sqrt, safe_power, validate_finite,
        get_m1_gpu_manager, get_m1_memory_optimizer, get_m1_cpu_optimizer,
        optimize_memory_usage, parallel_processing_optimizer, safe_correlation,
        safe_dataframe_operation, optimize_dataframe_dtypes, safe_fillna,
        create_data_quality_report, get_dataframe_info, memory_checkpoint,
        gpu_context, optimize_memory, get_memory_usage
    )
    COMMON_OPS_AVAILABLE = True
except ImportError:
    COMMON_OPS_AVAILABLE = False
    tprint_warning("⚠️ Common operations not available")

# Import math validation
try:
    from src.utils.math_validation import (
        safe_divide as math_safe_divide, safe_log as math_safe_log,
        safe_sqrt as math_safe_sqrt, validate_finite as math_validate_finite,
        safe_correlation as math_safe_correlation, safe_mean, safe_std,
        safe_percentile, validate_correlation_matrix, safe_matrix_inverse
    )
    MATH_VALIDATION_AVAILABLE = True
except ImportError:
    MATH_VALIDATION_AVAILABLE = False
    tprint_warning("⚠️ Math validation not available")

# Import matrix operations
try:
    from src.utils.matrix_operations import (
        get_unified_matrix_operations, get_vectorized_processing_core,
        get_batch_matrix_processor, safe_matrix_multiply,
        vectorized_rolling_features, parallel_feature_engineering,
        optimize_dataframe, get_hardware_performance_report,
        matrix_correlation_analysis, batch_matrix_multiply,
        batch_feature_transformation, batch_correlation_analysis
    )
    MATRIX_OPS_AVAILABLE = True
    tprint_success("✅ Matrix operations loaded successfully")
except ImportError as e:
    MATRIX_OPS_AVAILABLE = False
    tprint_warning(f"⚠️ Matrix operations not available: {e}")

# Import ML common utilities
try:
    from src.utils.ml_common.optimization.bayesian_tpe_optimizer import (
        BayesianTPEOptimizer, OptimizationConfig
    )
    from src.utils.purged_kfold import PurgedKFoldTime as PurgedKFold
    from src.feature_selection import select_features as FeatureSelector
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    from src.utils.ml_common.utils.lookahead_protection import LookaheadBiasDetector
    # Note: HPOptimizer, ModelValidator, and OutOfFoldPredictor may need to be implemented or imported from elsewhere
    ML_COMMON_AVAILABLE = True
    tprint_success("✅ ML common utilities loaded successfully")
except ImportError as e:
    ML_COMMON_AVAILABLE = False
    tprint_warning(f"⚠️ ML common utilities not available: {e}")

# Import data utilities
try:
    from src.utils.data.real_data_loader import DataLoader
    from src.utils.data.validation.validators import DataValidator
    from src.utils.data.klines_parquet import KlineParquetLoader
    from src.utils.serialization_utils import save_pickle, load_pickle
    from src.utils.data.feature_engineer import FeatureEngineer
    # Note: DataPreprocessor and TimeSeriesProcessor may need to be implemented or imported from elsewhere
    DATA_UTILS_AVAILABLE = True
    tprint_success("✅ Data utilities loaded successfully")
except ImportError as e:
    DATA_UTILS_AVAILABLE = False
    tprint_warning(f"⚠️ Data utilities not available: {e}")

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class OptimizationResult:
    """Complete result of the lookback optimization system."""
    ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]]
    stability_results: Dict[str, Dict[FamilyType, StabilityResult]]
    hierarchical_results: Dict[str, HierarchicalResult]
    decisions: Dict[str, Dict[FamilyType, DecisionResult]]
    feature_results: Dict[str, Dict[FamilyType, FeatureResult]]
    execution_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'ic_surface_results': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.ic_surface_results.items()
            },
            'stability_results': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.stability_results.items()
            },
            'hierarchical_results': {
                symbol: result.to_dict() for symbol, result in self.hierarchical_results.items()
            },
            'decisions': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.decisions.items()
            },
            'feature_results': {
                symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                for symbol, symbol_results in self.feature_results.items()
            },
            'execution_time': self.execution_time,
            'success': self.success,
            'error_message': self.error_message
        }


class LookbackOptimizationOrchestrator:
    """Enhanced main orchestrator for the lookback optimization system with comprehensive utilities."""
    
    def __init__(self, config: Optional[LookbackOptimizationConfig] = None):
        tprint_info("🔧 Initializing Enhanced Lookback Optimization Orchestrator...")
        
        self.config = config or create_default_config()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Performance tracking
        self.performance_metrics = {}
        self.stage_start_times = {}
        self.memory_usage_history = []
        
        # Initialize stage components with enhanced logging
        tprint_debug("🏗️ Initializing stage components...")
        try:
            self.ic_estimator = ICSurfaceEstimator(self.config)
            tprint_success("✅ IC Surface Estimator initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize IC Surface Estimator: {e}")
            raise
        
        try:
            self.stability_tester = MultiFamilyStabilityTester(self.config)
            tprint_success("✅ Multi-Family Stability Tester initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Stability Tester: {e}")
            raise
        
        try:
            self.hierarchical_shrinkage = MultiSymbolHierarchicalShrinkage(self.config)
            tprint_success("✅ Multi-Symbol Hierarchical Shrinkage initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Hierarchical Shrinkage: {e}")
            raise
        
        try:
            self.decision_maker = MultiFamilyDecisionMaker(self.config)
            tprint_success("✅ Multi-Family Decision Maker initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Decision Maker: {e}")
            raise
        
        try:
            self.feature_generator = MultiFamilyFeatureGenerator(self.config)
            tprint_success("✅ Multi-Family Feature Generator initialized")
        except Exception as e:
            tprint_error(f"❌ Failed to initialize Feature Generator: {e}")
            raise
        
        # Initialize profit labeling framework components
        self._initialize_profit_labeling_components()
        
        # Initialize utility components
        self._initialize_utility_components()
        
        # Create output directory
        try:
            os.makedirs(self.config.output_dir, exist_ok=True)
            tprint_success(f"✅ Output directory created: {self.config.output_dir}")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to create output directory: {e}")
        
        tprint_success("🚀 Enhanced Lookback Optimization Orchestrator initialized successfully")
    
    def _initialize_profit_labeling_components(self):
        """Initialize profit labeling framework components for quality-based optimization."""
        tprint_debug("🔧 Initializing profit labeling framework components...")
        
        if not PROFIT_LABELING_AVAILABLE:
            tprint_warning("⚠️ Profit labeling framework not available, skipping initialization")
            self.quality_scorer = None
            self.volatility_labeler = None
            self.multi_target_scheme = None
            self.noise_gating_filter = None
            return
        
        # Initialize quality scorer
        try:
            quality_config = QualityScoringConfig(
                baseline_models=['logistic', 'random_forest'],
                test_size=0.2,
                n_splits=5,
                random_state=42,
                min_lqs_score=0.3,
                min_auc_threshold=0.55,
                max_auc_std_threshold=0.03,
                min_psi_threshold=0.1,
                max_flip_rate_threshold=0.15,
                min_balance_threshold=0.35,
                max_balance_threshold=0.65,
                max_correlation_threshold=0.4
            )
            self.quality_scorer = LabelQualityScorer(quality_config)
            tprint_success("✅ Quality scorer initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize quality scorer: {e}")
            self.quality_scorer = None
        
        # Initialize volatility labeler
        try:
            volatility_config = VolatilityAwareConfig(
                min_data_points=1000,
                generate_reports=True,
                save_intermediate_results=True,
                enable_volatility_normalization=True,
                enable_multi_target_scheme=True
            )
            self.volatility_labeler = VolatilityAwareMultiHorizonLabeler(volatility_config)
            tprint_success("✅ Volatility labeler initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize volatility labeler: {e}")
            self.volatility_labeler = None
        
        # Initialize multi-target scheme
        try:
            multi_target_config = MultiTargetConfig(
                small_band=(0.4, 0.8),
                medium_band=(0.8, 1.3),
                high_band=(1.3, 2.0),
                enable_optimization=True,
                optimization_method='bayesian',
                n_trials=50,
                optimization_metric='lqs'
            )
            self.multi_target_scheme = MultiTargetScheme(multi_target_config)
            tprint_success("✅ Multi-target scheme initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize multi-target scheme: {e}")
            self.multi_target_scheme = None
        
        # Initialize noise gating filter
        try:
            noise_config = NoiseGatingConfig(
                min_volume_threshold=1000,
                max_spread_ratio=0.01,
                min_tick_count=10,
                enable_volatility_filtering=True,
                volatility_threshold_percentile=5.0
            )
            self.noise_gating_filter = NoiseGatingFilter(noise_config)
            tprint_success("✅ Noise gating filter initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize noise gating filter: {e}")
            self.noise_gating_filter = None
        
        tprint_success("✅ Profit labeling framework components initialized")
    
    def _initialize_utility_components(self):
        """Initialize utility components with comprehensive error handling."""
        tprint_debug("🔧 Initializing utility components...")
        
        # Initialize matrix operations
        if MATRIX_OPS_AVAILABLE:
            try:
                self.matrix_ops = get_unified_matrix_operations()
                self.vectorized_core = get_vectorized_processing_core()
                self.batch_processor = get_batch_matrix_processor()
                tprint_success("✅ Matrix operations initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Matrix operations initialization failed: {e}")
                self.matrix_ops = None
                self.vectorized_core = None
                self.batch_processor = None
        else:
            self.matrix_ops = None
            self.vectorized_core = None
            self.batch_processor = None
        
        # Initialize hardware optimizers
        if COMMON_OPS_AVAILABLE:
            try:
                self.m1_gpu_manager = get_m1_gpu_manager()
                self.m1_memory_optimizer = get_m1_memory_optimizer()
                self.m1_cpu_optimizer = get_m1_cpu_optimizer()
                tprint_success("✅ M1 hardware optimizers initialized")
            except Exception as e:
                tprint_warning(f"⚠️ M1 hardware optimizers not available: {e}")
                self.m1_gpu_manager = None
                self.m1_memory_optimizer = None
                self.m1_cpu_optimizer = None
        else:
            self.m1_gpu_manager = None
            self.m1_memory_optimizer = None
            self.m1_cpu_optimizer = None
        
        # Initialize ML common utilities
        if ML_COMMON_AVAILABLE:
            try:
                self.bayesian_optimizer = BayesianTPEOptimizer()
                self.feature_selector = FeatureSelector()
                self.data_leakage_detector = DataLeakageDetector()
                self.lookahead_bias_detector = LookaheadBiasDetector()
                self.hp_optimizer = HPOptimizer()
                self.model_validator = ModelValidator()
                self.oof_predictor = OutOfFoldPredictor()
                tprint_success("✅ ML common utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ ML common utilities initialization failed: {e}")
                self.bayesian_optimizer = None
                self.feature_selector = None
                self.data_leakage_detector = None
                self.lookahead_bias_detector = None
                self.hp_optimizer = None
                self.model_validator = None
                self.oof_predictor = None
        else:
            self.bayesian_optimizer = None
            self.feature_selector = None
            self.data_leakage_detector = None
            self.lookahead_bias_detector = None
            self.hp_optimizer = None
            self.model_validator = None
            self.oof_predictor = None
        
        # Initialize data utilities
        if DATA_UTILS_AVAILABLE:
            try:
                self.data_loader = DataLoader()
                self.data_validator = DataValidator()
                self.kline_loader = KlineParquetLoader()
                self.data_preprocessor = DataPreprocessor()
                self.feature_engineer = FeatureEngineer()
                self.time_series_processor = TimeSeriesProcessor()
                tprint_success("✅ Data utilities initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Data utilities initialization failed: {e}")
                self.data_loader = None
                self.data_validator = None
                self.kline_loader = None
                self.data_preprocessor = None
                self.feature_engineer = None
                self.time_series_processor = None
        else:
            self.data_loader = None
            self.data_validator = None
            self.kline_loader = None
            self.data_preprocessor = None
            self.feature_engineer = None
            self.time_series_processor = None
    
    def optimize_lookbacks(self, 
                          data: Dict[str, pd.DataFrame],
                          targets: Dict[str, np.ndarray],
                          feature_names: Optional[Dict[FamilyType, str]] = None) -> OptimizationResult:
        """Run the complete lookback optimization pipeline with comprehensive enhancements."""
        start_time = time.time()
        tprint_success("🚀 Starting Enhanced Data-Driven Lookback Optimization System...")
        
        # Initialize performance tracking
        self.stage_start_times = {}
        self.performance_metrics = {
            'total_execution_time': 0.0,
            'stage_times': {},
            'memory_usage_mb': 0.0,
            'optimization_applied': False,
            'hardware_acceleration_used': False,
            'symbols_processed': 0,
            'families_processed': 0
        }
        
        try:
            # Log configuration
            tprint_info("📊 Configuration:")
            tprint_info(f"   - Symbols: {list(data.keys())}")
            tprint_info(f"   - Target horizons: {list(targets.keys())}")
            tprint_info(f"   - Output directory: {self.config.output_dir}")
            tprint_info(f"   - Save intermediate results: {self.config.save_intermediate_results}")
            
            # Validate inputs with enhanced logging
            tprint_debug("🔍 Validating inputs...")
            self._validate_inputs(data, targets)
            tprint_success("✅ Input validation passed")
            
            # Set default feature names
            if feature_names is None:
                feature_names = {family: f"{family.value}_feature" for family in FamilyType}
                tprint_debug(f"📋 Using default feature names: {list(feature_names.values())}")
            
            # Apply hardware optimization to data
            if self.m1_memory_optimizer:
                tprint_debug("🖥️ Applying M1 memory optimization to input data...")
                try:
                    for symbol, df in data.items():
                        data[symbol] = self.m1_memory_optimizer.optimize_dataframe_memory(df)
                    tprint_success("✅ M1 memory optimization applied to input data")
                    self.performance_metrics['hardware_acceleration_used'] = True
                except Exception as e:
                    tprint_warning(f"⚠️ M1 memory optimization failed: {e}")
            
            # Stage 1: IC Surface Estimation
            tprint_info("📊 Stage 1: Estimating IC surfaces with HAC standard errors...")
            self.stage_start_times['ic_surface'] = time.time()
            ic_surface_results = self._run_stage_1(data, targets, feature_names)
            self._log_stage_completion('ic_surface')
            
            # Stage 2: Walk-Forward Stability Testing
            tprint_info("🔄 Stage 2: Testing stability with purged walk-forward validation...")
            self.stage_start_times['stability'] = time.time()
            stability_results = self._run_stage_2(data, targets, ic_surface_results, feature_names)
            self._log_stage_completion('stability')
            
            # Stage 3: Hierarchical Bayesian Shrinkage
            tprint_info("🎯 Stage 3: Applying hierarchical Bayesian shrinkage...")
            self.stage_start_times['hierarchical'] = time.time()
            hierarchical_results = self._run_stage_3(ic_surface_results, stability_results)
            self._log_stage_completion('hierarchical')
            
            # Decision Making
            tprint_info("🤔 Making lookback decisions with hysteresis...")
            self.stage_start_times['decisions'] = time.time()
            decisions = self._make_decisions(ic_surface_results, stability_results, 
                                          hierarchical_results, data, targets, feature_names)
            self._log_stage_completion('decisions')
            
            # Feature Generation
            tprint_info("⚙️ Generating optimized features...")
            self.stage_start_times['features'] = time.time()
            feature_results = self._generate_features(data, decisions, feature_names)
            self._log_stage_completion('features')
            
            # Apply matrix optimization to results if available
            if self.vectorized_core and MATRIX_OPS_AVAILABLE:
                tprint_debug("🧮 Applying matrix optimization to generated features...")
                try:
                    for symbol_results in feature_results.values():
                        for family, feature_result in symbol_results.items():
                            if hasattr(feature_result, 'features') and not feature_result.features.empty:
                                feature_result.features = self.vectorized_core.optimize_dataframe_for_processing(
                                    feature_result.features
                                )
                    tprint_success("✅ Matrix optimization applied to generated features")
                    self.performance_metrics['optimization_applied'] = True
                except Exception as e:
                    tprint_warning(f"⚠️ Matrix optimization failed: {e}")
            
            # Save results
            if self.config.save_intermediate_results:
                tprint_debug("💾 Saving intermediate results...")
                self._save_results(ic_surface_results, stability_results, 
                                 hierarchical_results, decisions, feature_results)
                tprint_success("✅ Intermediate results saved")
            
            # Calculate final metrics
            execution_time = time.time() - start_time
            self.performance_metrics['total_execution_time'] = execution_time
            self.performance_metrics['symbols_processed'] = len(data)
            self.performance_metrics['families_processed'] = len(FamilyType)
            
            # Calculate memory usage
            if COMMON_OPS_AVAILABLE:
                try:
                    memory_usage = get_memory_usage()
                    self.performance_metrics['memory_usage_mb'] = memory_usage / (1024 * 1024)
                except Exception as e:
                    tprint_warning(f"⚠️ Could not get memory usage: {e}")
            
            result = OptimizationResult(
                ic_surface_results=ic_surface_results,
                stability_results=stability_results,
                hierarchical_results=hierarchical_results,
                decisions=decisions,
                feature_results=feature_results,
                execution_time=execution_time,
                success=True
            )
            
            tprint_success(f"✅ Lookback optimization completed successfully in {execution_time:.3f}s")
            self._print_enhanced_summary(result)
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Lookback optimization failed: {str(e)}"
            
            tprint_error(f"❌ {error_message}")
            tprint_error(f"📊 Execution time before failure: {execution_time:.3f}s")
            self.logger.error(error_message)
            self.logger.error(f"Error details: {traceback.format_exc()}")
            
            # Log performance metrics even on failure
            self.performance_metrics['total_execution_time'] = execution_time
            self._log_performance_summary(error=True)
            
            return OptimizationResult(
                ic_surface_results={},
                stability_results={},
                hierarchical_results={},
                decisions={},
                feature_results={},
                execution_time=execution_time,
                success=False,
                error_message=error_message
            )
    
    def _log_stage_completion(self, stage_name: str):
        """Log stage completion with timing information."""
        if stage_name in self.stage_start_times:
            stage_time = time.time() - self.stage_start_times[stage_name]
            self.performance_metrics['stage_times'][stage_name] = stage_time
            tprint_performance(f"Stage {stage_name}", stage_time)
    
    def _log_performance_summary(self, error: bool = False):
        """Log comprehensive performance summary."""
        tprint_info("📊 PERFORMANCE SUMMARY")
        tprint_info(f"⏱️ Total execution time: {self.performance_metrics['total_execution_time']:.3f}s")
        
        # Log stage times
        if self.performance_metrics['stage_times']:
            tprint_info("📈 Stage execution times:")
            for stage_name, stage_time in self.performance_metrics['stage_times'].items():
                percentage = (stage_time / self.performance_metrics['total_execution_time']) * 100
                tprint_info(f"   - {stage_name}: {stage_time:.3f}s ({percentage:.1f}%)")
        
        # Log memory usage
        if self.performance_metrics.get('memory_usage_mb', 0) > 0:
            tprint_info(f"💾 Memory usage: {self.performance_metrics['memory_usage_mb']:.2f} MB")
        
        # Log optimization status
        if self.performance_metrics.get('optimization_applied', False):
            tprint_success("✅ Matrix optimization applied")
        if self.performance_metrics.get('hardware_acceleration_used', False):
            tprint_success("✅ Hardware acceleration used")
        
        # Log processing metrics
        tprint_info(f"📊 Processing metrics:")
        tprint_info(f"   - Symbols processed: {self.performance_metrics.get('symbols_processed', 0)}")
        tprint_info(f"   - Families processed: {self.performance_metrics.get('families_processed', 0)}")
        
        if error:
            tprint_error("❌ Pipeline execution failed")
        else:
            tprint_success("✅ Pipeline execution completed successfully")
    
    def _validate_inputs(self, data: Dict[str, pd.DataFrame], targets: Dict[str, np.ndarray]) -> None:
        """Validate input data and targets."""
        if not data:
            raise ValueError("No data provided")
        
        if not targets:
            raise ValueError("No targets provided")
        
        # Check that all symbols have both data and targets
        for symbol in data.keys():
            if symbol not in targets:
                raise ValueError(f"No target provided for symbol {symbol}")
            
            if len(data[symbol]) != len(targets[symbol]):
                raise ValueError(f"Data and target length mismatch for symbol {symbol}")
        
        # Check minimum data requirements
        for symbol, df in data.items():
            if len(df) < 1000:
                raise ValueError(f"Insufficient data for symbol {symbol}: {len(df)} < 1000")
    
    def _run_stage_1(self, data: Dict[str, pd.DataFrame], targets: Dict[str, np.ndarray],
                    feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, ICSurfaceResult]]:
        """Run Stage 1: IC Surface Estimation."""
        results = {}
        
        for symbol, symbol_data in data.items():
            symbol_results = {}
            symbol_target = targets[symbol]
            
            tprint_info(f"Processing {symbol}...")
            
            for family in FamilyType:
                try:
                    feature_name = feature_names[family]
                    
                    ic_result = self.ic_estimator.estimate_surface(
                        symbol_data, symbol_target, family, feature_name,
                        quality_scorer=self.quality_scorer
                    )
                    
                    symbol_results[family] = ic_result
                    
                except Exception as e:
                    self.logger.warning(f"Failed to estimate IC surface for {symbol}-{family.value}: {e}")
                    continue
            
            results[symbol] = symbol_results
        
        return results
    
    def _run_stage_2(self, data: Dict[str, pd.DataFrame], targets: Dict[str, np.ndarray],
                    ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                    feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, StabilityResult]]:
        """Run Stage 2: Walk-Forward Stability Testing."""
        results = {}
        
        for symbol, symbol_data in data.items():
            symbol_ic_results = ic_surface_results.get(symbol, {})
            symbol_target = targets[symbol]
            
            if not symbol_ic_results:
                continue
            
            tprint_info(f"Testing stability for {symbol}...")
            
            try:
                symbol_stability_results = self.stability_tester.test_all_families(
                    symbol_data, symbol_target, symbol_ic_results, feature_names
                )
                
                results[symbol] = symbol_stability_results
                
            except Exception as e:
                self.logger.warning(f"Failed to test stability for {symbol}: {e}")
                continue
        
        return results
    
    def _run_stage_3(self, ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                    stability_results: Dict[str, Dict[FamilyType, StabilityResult]]) -> Dict[str, HierarchicalResult]:
        """Run Stage 3: Hierarchical Bayesian Shrinkage."""
        try:
            hierarchical_results = self.hierarchical_shrinkage.apply_multi_symbol_shrinkage(
                ic_surface_results, stability_results
            )
            
            return hierarchical_results
            
        except Exception as e:
            self.logger.warning(f"Hierarchical shrinkage failed: {e}")
            return {}
    
    def _make_decisions(self, ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                       stability_results: Dict[str, Dict[FamilyType, StabilityResult]],
                       hierarchical_results: Dict[str, HierarchicalResult],
                       data: Dict[str, pd.DataFrame],
                       targets: Dict[str, np.ndarray],
                       feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, DecisionResult]]:
        """Make lookback decisions for all symbol-family combinations."""
        try:
            decisions = self.decision_maker.make_all_decisions(
                ic_surface_results, stability_results, hierarchical_results,
                data, targets, feature_names
            )
            
            return decisions
            
        except Exception as e:
            self.logger.warning(f"Decision making failed: {e}")
            return {}
    
    def _generate_features(self, data: Dict[str, pd.DataFrame],
                          decisions: Dict[str, Dict[FamilyType, DecisionResult]],
                          feature_names: Dict[FamilyType, str]) -> Dict[str, Dict[FamilyType, FeatureResult]]:
        """Generate optimized features for all symbol-family combinations."""
        try:
            feature_results = self.feature_generator.generate_all_symbols_features(
                data, decisions, feature_names
            )
            
            return feature_results
            
        except Exception as e:
            self.logger.warning(f"Feature generation failed: {e}")
            return {}
    
    def _save_results(self, ic_surface_results: Dict[str, Dict[FamilyType, ICSurfaceResult]],
                     stability_results: Dict[str, Dict[FamilyType, StabilityResult]],
                     hierarchical_results: Dict[str, HierarchicalResult],
                     decisions: Dict[str, Dict[FamilyType, DecisionResult]],
                     feature_results: Dict[str, Dict[FamilyType, FeatureResult]]) -> None:
        """Save intermediate results to disk."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save IC surface results
            ic_path = os.path.join(self.config.output_dir, f"ic_surface_results_{timestamp}.json")
            with open(ic_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in ic_surface_results.items()
                }, f, indent=2)
            
            # Save stability results
            stability_path = os.path.join(self.config.output_dir, f"stability_results_{timestamp}.json")
            with open(stability_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in stability_results.items()
                }, f, indent=2)
            
            # Save hierarchical results
            hierarchical_path = os.path.join(self.config.output_dir, f"hierarchical_results_{timestamp}.json")
            with open(hierarchical_path, 'w') as f:
                json.dump({
                    symbol: result.to_dict() for symbol, result in hierarchical_results.items()
                }, f, indent=2)
            
            # Save decisions
            decisions_path = os.path.join(self.config.output_dir, f"decisions_{timestamp}.json")
            with open(decisions_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in decisions.items()
                }, f, indent=2)
            
            # Save feature results
            feature_path = os.path.join(self.config.output_dir, f"feature_results_{timestamp}.json")
            with open(feature_path, 'w') as f:
                json.dump({
                    symbol: {family.value: result.to_dict() for family, result in symbol_results.items()}
                    for symbol, symbol_results in feature_results.items()
                }, f, indent=2)
            
            tprint_info(f"Results saved to {self.config.output_dir}")
            
        except Exception as e:
            self.logger.warning(f"Failed to save results: {e}")
    
    def _print_enhanced_summary(self, result: OptimizationResult) -> None:
        """Print enhanced optimization summary with comprehensive metrics."""
        tprint_success("📊 ENHANCED OPTIMIZATION SUMMARY")
        tprint_success(f"⏱️ Execution time: {result.execution_time:.3f}s")
        tprint_success(f"📊 Symbols processed: {len(result.ic_surface_results)}")
        
        # Count features by type
        decision_counts = {'discrete': 0, 'blend': 0, 'default': 0, 'inactive': 0}
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                decision_type = decision.lookback_spec.decision_type.value
                decision_counts[decision_type] += 1
        
        tprint_success(f"🎯 Decision types: {decision_counts}")
        
        # Quality metrics
        if result.feature_results:
            all_quality_scores = []
            total_features = 0
            for symbol_results in result.feature_results.values():
                for feature_result in symbol_results.values():
                    all_quality_scores.append(feature_result.quality_score)
                    if hasattr(feature_result, 'features') and not feature_result.features.empty:
                        total_features += len(feature_result.features.columns)
            
            if all_quality_scores:
                avg_quality = np.mean(all_quality_scores)
                tprint_success(f"📈 Average feature quality: {avg_quality:.3f}")
                tprint_success(f"🔢 Total features generated: {total_features}")
        
        # Log performance metrics
        self._log_performance_summary()
        
        # Log utility status
        tprint_info("🔧 Utility Status:")
        tprint_info(f"   - Matrix operations: {'✅' if MATRIX_OPS_AVAILABLE else '❌'}")
        tprint_info(f"   - ML common utilities: {'✅' if ML_COMMON_AVAILABLE else '❌'}")
        tprint_info(f"   - Data utilities: {'✅' if DATA_UTILS_AVAILABLE else '❌'}")
        tprint_info(f"   - Common operations: {'✅' if COMMON_OPS_AVAILABLE else '❌'}")
        tprint_info(f"   - Math validation: {'✅' if MATH_VALIDATION_AVAILABLE else '❌'}")
    
    def _print_summary(self, result: OptimizationResult) -> None:
        """Print optimization summary (legacy method for backward compatibility)."""
        self._print_enhanced_summary(result)
    
    def generate_comprehensive_report(self, result: OptimizationResult) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        report = {
            'execution_summary': {
                'success': result.success,
                'execution_time': result.execution_time,
                'error_message': result.error_message,
                'symbols_processed': len(result.ic_surface_results),
                'families_processed': len(FamilyType)
            },
            'stage_1_summary': self._summarize_ic_surfaces(result.ic_surface_results),
            'stage_2_summary': self._summarize_stability(result.stability_results),
            'stage_3_summary': self._summarize_hierarchical(result.hierarchical_results),
            'decision_summary': self._summarize_decisions(result.decisions),
            'feature_summary': self._summarize_features(result.feature_results),
            'recommendations': self._generate_recommendations(result)
        }
        
        return report
    
    def _summarize_ic_surfaces(self, ic_results: Dict[str, Dict[FamilyType, ICSurfaceResult]]) -> Dict[str, Any]:
        """Summarize IC surface results."""
        summary = {
            'total_estimations': 0,
            'successful_estimations': 0,
            'average_optimal_ic': 0.0,
            'family_performance': {}
        }
        
        all_ics = []
        
        for family in FamilyType:
            family_ics = []
            for symbol_results in ic_results.values():
                if family in symbol_results:
                    family_ics.append(symbol_results[family].optimal_ic)
                    summary['total_estimations'] += 1
                    summary['successful_estimations'] += 1
            
            if family_ics:
                summary['family_performance'][family.value] = {
                    'average_ic': np.mean(family_ics),
                    'std_ic': np.std(family_ics),
                    'count': len(family_ics)
                }
                all_ics.extend(family_ics)
        
        if all_ics:
            summary['average_optimal_ic'] = np.mean(all_ics)
        
        return summary
    
    def _summarize_stability(self, stability_results: Dict[str, Dict[FamilyType, StabilityResult]]) -> Dict[str, Any]:
        """Summarize stability results."""
        summary = {
            'total_tests': 0,
            'stable_families': 0,
            'blend_recommended': 0,
            'unstable_families': 0,
            'average_stability_score': 0.0
        }
        
        all_stability_scores = []
        
        for symbol_results in stability_results.values():
            for result in symbol_results.values():
                summary['total_tests'] += 1
                all_stability_scores.append(result.stability_score)
                
                if result.recommendation == "stable":
                    summary['stable_families'] += 1
                elif result.recommendation == "blend_recommended":
                    summary['blend_recommended'] += 1
                else:
                    summary['unstable_families'] += 1
        
        if all_stability_scores:
            summary['average_stability_score'] = np.mean(all_stability_scores)
        
        return summary
    
    def _summarize_hierarchical(self, hierarchical_results: Dict[str, HierarchicalResult]) -> Dict[str, Any]:
        """Summarize hierarchical shrinkage results."""
        summary = {
            'total_shrinkage_applications': len(hierarchical_results),
            'average_shrinkage_factor': 0.0,
            'convergence_issues': 0
        }
        
        all_shrinkage_factors = []
        
        for result in hierarchical_results.values():
            shrinkage_factors = list(result.shrinkage_factors.values())
            all_shrinkage_factors.extend(shrinkage_factors)
            
            if 'error' in result.convergence_diagnostics:
                summary['convergence_issues'] += 1
        
        if all_shrinkage_factors:
            summary['average_shrinkage_factor'] = np.mean(all_shrinkage_factors)
        
        return summary
    
    def _summarize_decisions(self, decisions: Dict[str, Dict[FamilyType, DecisionResult]]) -> Dict[str, Any]:
        """Summarize decision results."""
        summary = {
            'total_decisions': 0,
            'decision_type_counts': {'discrete': 0, 'blend': 0, 'default': 0, 'inactive': 0},
            'average_confidence': 0.0,
            'families_with_changes': 0
        }
        
        all_confidence_scores = []
        families_with_changes = set()
        
        for symbol_results in decisions.values():
            for family, decision in symbol_results.items():
                summary['total_decisions'] += 1
                decision_type = decision.lookback_spec.decision_type.value
                summary['decision_type_counts'][decision_type] += 1
                
                all_confidence_scores.append(decision.lookback_spec.confidence_score)
                
                if decision.change_magnitude > 0.1:
                    families_with_changes.add(family)
        
        if all_confidence_scores:
            summary['average_confidence'] = np.mean(all_confidence_scores)
        
        summary['families_with_changes'] = len(families_with_changes)
        
        return summary
    
    def _summarize_features(self, feature_results: Dict[str, Dict[FamilyType, FeatureResult]]) -> Dict[str, Any]:
        """Summarize feature generation results."""
        summary = {
            'total_features_generated': 0,
            'average_generation_time': 0.0,
            'average_quality_score': 0.0,
            'total_memory_usage_mb': 0.0
        }
        
        all_generation_times = []
        all_quality_scores = []
        total_memory = 0.0
        
        for symbol_results in feature_results.values():
            for result in symbol_results.values():
                summary['total_features_generated'] += 1
                all_generation_times.append(result.generation_time)
                all_quality_scores.append(result.quality_score)
                total_memory += result.memory_usage_mb
        
        if all_generation_times:
            summary['average_generation_time'] = np.mean(all_generation_times)
        if all_quality_scores:
            summary['average_quality_score'] = np.mean(all_quality_scores)
        
        summary['total_memory_usage_mb'] = total_memory
        
        return summary
    
    def _generate_recommendations(self, result: OptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        # Check for inactive families
        inactive_count = 0
        for symbol_decisions in result.decisions.values():
            for decision in symbol_decisions.values():
                if decision.lookback_spec.decision_type.value == 'inactive':
                    inactive_count += 1
        
        if inactive_count > 0:
            recommendations.append(f"Consider removing {inactive_count} inactive families")
        
        # Check for low quality features
        if result.feature_results:
            low_quality_count = 0
            for symbol_results in result.feature_results.values():
                for feature_result in symbol_results.values():
                    if feature_result.quality_score < 0.3:
                        low_quality_count += 1
            
            if low_quality_count > 0:
                recommendations.append(f"Review {low_quality_count} low-quality features")
        
        # Check for high memory usage
        if result.feature_results:
            total_memory = sum(
                sum(feature_result.memory_usage_mb for feature_result in symbol_results.values())
                for symbol_results in result.feature_results.values()
            )
            
            if total_memory > 1000:  # More than 1GB
                recommendations.append(f"High memory usage: {total_memory:.1f}MB - consider optimization")
        
        return recommendations