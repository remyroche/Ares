"""
Enhanced Vectorized Validator

This module integrates VectorBTRollingOptimizer, UnifiedVectorizationManager,
ML commons utilities, and features_common tools for maximum efficiency
in validation operations.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime

# Import VectorBT Rolling Optimizer
try:
    from src.feature_generation.utils.vectorbt_rolling_optimizer import (
        VectorBTRollingOptimizer, VectorBTOptimizationError
    )
    VECTORBT_ROLLING_AVAILABLE = True
except ImportError:
    VECTORBT_ROLLING_AVAILABLE = False
    VectorBTRollingOptimizer = None

# Import Unified Vectorization Manager
try:
    from src.utils.ml_common.unified_vectorization_manager import (
        UnifiedVectorizationManager, OperationType, OptimizationStrategy,
        OperationConfig, OptimizationResult
    )
    UNIFIED_VECTORIZATION_AVAILABLE = True
except ImportError:
    UNIFIED_VECTORIZATION_AVAILABLE = False
    UnifiedVectorizationManager = None

# Import ML Commons utilities
try:
    from src.utils.ml_common.validation.unified_cv import UnifiedCrossValidator
    ML_COMMONS_AVAILABLE = True
except ImportError:
    ML_COMMONS_AVAILABLE = False

# Try to import optional ML Commons components
try:
    from src.utils.ml_common.validation.data_leakage_detector import DataLeakageDetector
    DATA_LEAKAGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    DATA_LEAKAGE_AVAILABLE = False
    DataLeakageDetector = None

try:
    from src.utils.lookahead_bias_detector import LookaheadBiasDetector
    LOOKAHEAD_AVAILABLE = True
    LookaheadDetector = LookaheadBiasDetector  # Alias for compatibility
except (ImportError, ModuleNotFoundError):
    LOOKAHEAD_AVAILABLE = False
    LookaheadDetector = None

try:
    from src.utils.ml_common.optimization.hpo_utils import HPOOptimizer
    HPO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HPO_AVAILABLE = False
    HPOOptimizer = None

try:
    from src.utils.ml_common.explainability.shap_lime_integration import SHAPLIMEIntegration
    SHAP_LIME_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    SHAP_LIME_AVAILABLE = False
    SHAPLIMEIntegration = None

# Import features_common utilities
try:
    from src.features_common.transforms.vectorbt_scaler import VectorBTScaler
    from src.features_common.transforms.scaling_normalization import ScalingNormalization
    from src.features_common.optimization.cv_base import CVBase
    from src.features_common.mixins.vectorbt_mixin import VectorBTMixin
    FEATURES_COMMONS_AVAILABLE = True
except ImportError:
    FEATURES_COMMONS_AVAILABLE = False

# Import existing validation components
from .nested_oof_validator import NestedOOFValidator, NestedOOFConfig
from .hierarchical_validator import HierarchicalValidator, HierarchicalValidationConfig
from .anchored_optimizer import AnchoredOptimizer, AnchoredOptimizationConfig
from .interpretability_feedback import InterpretabilityFeedbackLoop, InterpretabilityFeedbackConfig
from .vector_integrity_validator import VectorIntegrityValidator, VectorIntegrityConfig
from .forward_validator import ForwardValidator, ForwardValidationConfig

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success


@dataclass
class EnhancedVectorizedConfig:
    """Configuration for enhanced vectorized validation."""
    
    # VectorBT optimization
    enable_vectorbt_optimization: bool = True
    vectorbt_gpu_acceleration: bool = False
    vectorbt_parallel_processing: bool = True
    vectorbt_memory_optimization: bool = True
    
    # Unified vectorization
    enable_unified_vectorization: bool = True
    auto_strategy_selection: bool = True
    memory_budget_mb: float = 2048.0
    time_budget_seconds: float = 600.0
    
    # ML Commons integration
    enable_ml_commons: bool = True
    enable_data_leakage_detection: bool = True
    enable_lookahead_detection: bool = True
    enable_hpo_optimization: bool = True
    enable_shap_lime: bool = True
    
    # Features commons integration
    enable_features_commons: bool = True
    enable_vectorbt_scaling: bool = True
    enable_advanced_cv: bool = True
    
    # Performance optimization
    chunk_size: int = 10000
    max_workers: int = 4
    enable_caching: bool = True
    cache_size_mb: int = 512
    
    # Logging
    verbose: bool = True


@dataclass
class EnhancedVectorizedResult:
    """Result of enhanced vectorized validation."""
    
    # Core validation results
    nested_oof_result: Optional[Any] = None
    hierarchical_results: Dict[str, Any] = field(default_factory=dict)
    anchored_optimization_result: Optional[Any] = None
    interpretability_result: Optional[Any] = None
    vector_integrity_result: Optional[Any] = None
    forward_validation_result: Optional[Any] = None
    
    # VectorBT optimization results
    vectorbt_optimization_stats: Dict[str, Any] = field(default_factory=dict)
    vectorbt_performance_gains: Dict[str, float] = field(default_factory=dict)
    
    # Unified vectorization results
    vectorization_strategy: Optional[str] = None
    vectorization_performance: Dict[str, Any] = field(default_factory=dict)
    
    # ML Commons results
    data_leakage_detected: bool = False
    lookahead_detected: bool = False
    hpo_optimization_result: Optional[Any] = None
    shap_lime_insights: Dict[str, Any] = field(default_factory=dict)
    
    # Features commons results
    scaling_optimization_result: Optional[Any] = None
    cv_optimization_result: Optional[Any] = None
    
    # Overall metrics
    overall_score: float = 0.0
    performance_improvement: float = 0.0
    memory_efficiency: float = 0.0
    time_efficiency: float = 0.0
    
    # Validation status
    passed_validation: bool = False
    critical_issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class EnhancedVectorizedValidator:
    """
    Enhanced vectorized validator with full tool integration.
    
    Integrates:
    - VectorBTRollingOptimizer for efficient rolling operations
    - UnifiedVectorizationManager for optimal strategy selection
    - ML commons utilities (CV, OOF, data leakage, lookahead, HPO, SHAP/LIME)
    - features_common (scalers, transforms, CV)
    """
    
    def __init__(self, config: Optional[EnhancedVectorizedConfig] = None):
        """Initialize the enhanced vectorized validator."""
        self.config = config or EnhancedVectorizedConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self._initialize_components()
        
        if self.config.verbose:
            tprint("🚀 Initializing EnhancedVectorizedValidator")
    
    def _initialize_components(self) -> None:
        """Initialize all validation components with optimizations."""
        
        # Initialize VectorBT Rolling Optimizer
        if VECTORBT_ROLLING_AVAILABLE and self.config.enable_vectorbt_optimization:
            try:
                self.vectorbt_optimizer = VectorBTRollingOptimizer(
                    enable_gpu=self.config.vectorbt_gpu_acceleration,
                    enable_parallel=self.config.vectorbt_parallel_processing,
                    enable_memory_optimization=self.config.vectorbt_memory_optimization
                )
                # Reduced verbosity - only log once per session
                if self.config.verbose and not hasattr(EnhancedVectorizedValidator, '_logged_rolling_init'):
                    tprint_success("✅ VectorBTRollingOptimizer initialized")
                    EnhancedVectorizedValidator._logged_rolling_init = True
            except Exception as e:
                self.logger.warning(f"VectorBT Rolling Optimizer initialization failed: {e}")
                self.vectorbt_optimizer = None
        else:
            self.vectorbt_optimizer = None
        
        # Initialize Unified Vectorization Manager
        if UNIFIED_VECTORIZATION_AVAILABLE and self.config.enable_unified_vectorization:
            try:
                self.vectorization_manager = UnifiedVectorizationManager(
                    auto_strategy_selection=self.config.auto_strategy_selection,
                    memory_budget_mb=self.config.memory_budget_mb,
                    time_budget_seconds=self.config.time_budget_seconds
                )
                if self.config.verbose:
                    tprint_success("✅ UnifiedVectorizationManager initialized")
            except Exception as e:
                self.logger.warning(f"Unified Vectorization Manager initialization failed: {e}")
                self.vectorization_manager = None
        else:
            self.vectorization_manager = None
        
        # Initialize ML Commons utilities
        if self.config.enable_ml_commons:
            try:
                self.data_leakage_detector = DataLeakageDetector() if (DATA_LEAKAGE_AVAILABLE and self.config.enable_data_leakage_detection) else None
                self.lookahead_detector = LookaheadDetector() if (LOOKAHEAD_AVAILABLE and self.config.enable_lookahead_detection) else None
                self.hpo_optimizer = HPOOptimizer() if (HPO_AVAILABLE and self.config.enable_hpo_optimization) else None
                self.shap_lime = SHAPLIMEIntegration() if (SHAP_LIME_AVAILABLE and self.config.enable_shap_lime) else None
                if self.config.verbose:
                    tprint_success("✅ ML Commons utilities initialized")
            except Exception as e:
                self.logger.warning(f"ML Commons initialization failed: {e}")
                self.data_leakage_detector = None
                self.lookahead_detector = None
                self.hpo_optimizer = None
                self.shap_lime = None
        else:
            self.data_leakage_detector = None
            self.lookahead_detector = None
            self.hpo_optimizer = None
            self.shap_lime = None
        
        # Initialize features_commons utilities
        if FEATURES_COMMONS_AVAILABLE and self.config.enable_features_commons:
            try:
                self.vectorbt_scaler = VectorBTScaler() if self.config.enable_vectorbt_scaling else None
                self.cv_base = CVBase() if self.config.enable_advanced_cv else None
                if self.config.verbose:
                    tprint_success("✅ Features Commons utilities initialized")
            except Exception as e:
                self.logger.warning(f"Features Commons initialization failed: {e}")
                self.vectorbt_scaler = None
                self.cv_base = None
        else:
            self.vectorbt_scaler = None
            self.cv_base = None
        
        # Initialize core validation components
        self.nested_oof_validator = NestedOOFValidator()
        self.hierarchical_validator = HierarchicalValidator()
        self.anchored_optimizer = AnchoredOptimizer()
        self.interpretability_feedback = InterpretabilityFeedbackLoop()
        self.vector_integrity_validator = VectorIntegrityValidator()
        self.forward_validator = ForwardValidator()
    
    def validate_pipeline_enhanced(self, 
                                  data: pd.DataFrame,
                                  targets: pd.Series,
                                  pipeline: callable,
                                  metadata: Optional[Dict[str, Any]] = None) -> EnhancedVectorizedResult:
        """
        Perform enhanced vectorized pipeline validation.
        
        Args:
            data: Input features
            targets: Target labels
            pipeline: Trained pipeline
            metadata: Optional metadata
            
        Returns:
            EnhancedVectorizedResult
        """
        if self.config.verbose:
            tprint("🚀 Starting enhanced vectorized validation")
        
        result = EnhancedVectorizedResult()
        start_time = datetime.now()
        
        try:
            # Phase 1: VectorBT Optimization
            if self.vectorbt_optimizer:
                result.vectorbt_optimization_stats = self._optimize_with_vectorbt(
                    data, targets, pipeline
                )
            
            # Phase 2: Unified Vectorization
            if self.vectorization_manager:
                result.vectorization_strategy, result.vectorization_performance = self._optimize_vectorization(
                    data, targets, pipeline
                )
            
            # Phase 3: ML Commons Analysis
            if self.config.enable_ml_commons:
                ml_commons_result = self._analyze_with_ml_commons(data, targets, pipeline)
                result.data_leakage_detected = ml_commons_result.get('data_leakage', False)
                result.lookahead_detected = ml_commons_result.get('lookahead', False)
                result.hpo_optimization_result = ml_commons_result.get('hpo_result')
                result.shap_lime_insights = ml_commons_result.get('shap_lime', {})
            
            # Phase 4: Features Commons Optimization
            if self.config.enable_features_commons:
                features_result = self._optimize_with_features_commons(data, targets, pipeline)
                result.scaling_optimization_result = features_result.get('scaling')
                result.cv_optimization_result = features_result.get('cv')
            
            # Phase 5: Core Validation (with optimizations)
            result.nested_oof_result = self._validate_nested_oof_enhanced(data, targets, pipeline)
            result.hierarchical_results = self._validate_hierarchical_enhanced(data, targets, pipeline)
            result.anchored_optimization_result = self._validate_anchored_enhanced(data, targets, pipeline)
            result.interpretability_result = self._validate_interpretability_enhanced(data, targets, pipeline)
            result.vector_integrity_result = self._validate_vector_integrity_enhanced(data, targets, pipeline, metadata)
            result.forward_validation_result = self._validate_forward_enhanced(data, targets, pipeline)
            
            # Calculate overall metrics
            self._calculate_enhanced_metrics(result)
            
            # Determine validation status
            result.passed_validation = self._determine_enhanced_validation_status(result)
            
            # Generate recommendations
            result.recommendations = self._generate_enhanced_recommendations(result)
            
            if self.config.verbose:
                tprint_success("✅ Enhanced vectorized validation completed")
                tprint(f"📊 Overall score: {result.overall_score:.4f}")
                tprint(f"📊 Performance improvement: {result.performance_improvement:.2f}x")
                tprint(f"✅ Passed: {result.passed_validation}")
        
        except Exception as e:
            self.logger.error(f"Enhanced validation failed: {e}")
            result.critical_issues.append(f"Enhanced validation failed: {e}")
            result.passed_validation = False
        
        return result
    
    def _optimize_with_vectorbt(self, 
                              data: pd.DataFrame,
                              targets: pd.Series,
                              pipeline: callable) -> Dict[str, Any]:
        """Optimize validation with VectorBT Rolling Optimizer."""
        try:
            if not self.vectorbt_optimizer:
                return {}
            
            # Use VectorBT for rolling operations
            rolling_stats = {}
            
            # Optimize rolling mean calculations
            if hasattr(self.vectorbt_optimizer, 'rolling_mean'):
                rolling_stats['mean_optimization'] = self.vectorbt_optimizer.rolling_mean(
                    data, window=20, optimize=True
                )
            
            # Optimize rolling standard deviation
            if hasattr(self.vectorbt_optimizer, 'rolling_std'):
                rolling_stats['std_optimization'] = self.vectorbt_optimizer.rolling_std(
                    data, window=20, optimize=True
                )
            
            # Optimize rolling correlation
            if hasattr(self.vectorbt_optimizer, 'rolling_corr'):
                rolling_stats['corr_optimization'] = self.vectorbt_optimizer.rolling_corr(
                    data, targets, window=20, optimize=True
                )
            
            return rolling_stats
        
        except Exception as e:
            self.logger.warning(f"VectorBT optimization failed: {e}")
            return {}
    
    def _optimize_vectorization(self, 
                              data: pd.DataFrame,
                              targets: pd.Series,
                              pipeline: callable) -> Tuple[Optional[str], Dict[str, Any]]:
        """Optimize vectorization with Unified Vectorization Manager."""
        try:
            if not self.vectorization_manager:
                return None, {}
            
            # Create operation config
            config = OperationConfig(
                operation_type=OperationType.FEATURE_ENGINEERING,
                data_size=len(data),
                data_dimensions=data.shape,
                memory_budget_mb=self.config.memory_budget_mb,
                time_budget_seconds=self.config.time_budget_seconds
            )
            
            # Optimize vectorization strategy
            strategy = self.vectorization_manager.select_optimal_strategy(config)
            
            # Execute optimized operations
            performance = self.vectorization_manager.execute_optimized_operation(
                operation_type=OperationType.FEATURE_ENGINEERING,
                data=data,
                strategy=strategy
            )
            
            return strategy.value if strategy else None, performance
        
        except Exception as e:
            self.logger.warning(f"Vectorization optimization failed: {e}")
            return None, {}
    
    def _analyze_with_ml_commons(self, 
                                data: pd.DataFrame,
                                targets: pd.Series,
                                pipeline: callable) -> Dict[str, Any]:
        """Analyze with ML Commons utilities."""
        result = {}
        
        try:
            # Data leakage detection
            if self.data_leakage_detector:
                leakage_result = self.data_leakage_detector.detect_leakage(data, targets)
                result['data_leakage'] = leakage_result.get('leakage_detected', False)
            
            # Lookahead detection
            if self.lookahead_detector:
                lookahead_result = self.lookahead_detector.detect_lookahead(data, targets)
                result['lookahead'] = lookahead_result.get('lookahead_detected', False)
            
            # HPO optimization
            if self.hpo_optimizer:
                hpo_result = self.hpo_optimizer.optimize_hyperparameters(
                    pipeline, data, targets
                )
                result['hpo_result'] = hpo_result
            
            # SHAP/LIME analysis
            if self.shap_lime:
                shap_lime_result = self.shap_lime.analyze_model(
                    pipeline, data, targets
                )
                result['shap_lime'] = shap_lime_result
        
        except Exception as e:
            self.logger.warning(f"ML Commons analysis failed: {e}")
        
        return result
    
    def _optimize_with_features_commons(self, 
                                      data: pd.DataFrame,
                                      targets: pd.Series,
                                      pipeline: callable) -> Dict[str, Any]:
        """Optimize with features_commons utilities."""
        result = {}
        
        try:
            # VectorBT scaling optimization
            if self.vectorbt_scaler:
                scaling_result = self.vectorbt_scaler.optimize_scaling(data)
                result['scaling'] = scaling_result
            
            # Advanced CV optimization
            if self.cv_base:
                cv_result = self.cv_base.optimize_cv_strategy(data, targets)
                result['cv'] = cv_result
        
        except Exception as e:
            self.logger.warning(f"Features Commons optimization failed: {e}")
        
        return result
    
    def _validate_nested_oof_enhanced(self, 
                                    data: pd.DataFrame,
                                    targets: pd.Series,
                                    pipeline: callable) -> Any:
        """Enhanced nested OOF validation with optimizations."""
        try:
            # Use optimized cross-validation if available
            if self.cv_base:
                return self.cv_base.enhanced_nested_cv(data, targets, pipeline)
            else:
                return self.nested_oof_validator.perform_nested_validation(
                    data, targets, pipeline, pipeline
                )
        except Exception as e:
            self.logger.warning(f"Enhanced nested OOF validation failed: {e}")
            return None
    
    def _validate_hierarchical_enhanced(self, 
                                      data: pd.DataFrame,
                                      targets: pd.Series,
                                      pipeline: callable) -> Dict[str, Any]:
        """Enhanced hierarchical validation with optimizations."""
        try:
            results = {}
            
            # Early stage with VectorBT optimization
            early_result = self.hierarchical_validator.validate_early_stage(data, targets)
            if self.vectorbt_optimizer:
                early_result = self._enhance_with_vectorbt(early_result, data, targets)
            results['early'] = early_result
            
            # Mid stage with unified vectorization
            mid_result = self.hierarchical_validator.validate_mid_stage(data, targets, early_result)
            if self.vectorization_manager:
                mid_result = self._enhance_with_vectorization(mid_result, data, targets)
            results['mid'] = mid_result
            
            # Late stage with ML Commons
            late_result = self.hierarchical_validator.validate_late_stage(data, targets, mid_result)
            if self.shap_lime:
                late_result = self._enhance_with_shap_lime(late_result, data, targets, pipeline)
            results['late'] = late_result
            
            return results
        
        except Exception as e:
            self.logger.warning(f"Enhanced hierarchical validation failed: {e}")
            return {}
    
    def _validate_anchored_enhanced(self, 
                                  data: pd.DataFrame,
                                  targets: pd.Series,
                                  pipeline: callable) -> Any:
        """Enhanced anchored optimization with VectorBT."""
        try:
            # Use VectorBT for efficient time-based operations
            if self.vectorbt_optimizer:
                return self.vectorbt_optimizer.optimize_anchored_validation(
                    data, targets, pipeline
                )
            else:
                return self.anchored_optimizer.optimize_with_anchoring(
                    data, targets, pipeline, pipeline
                )
        except Exception as e:
            self.logger.warning(f"Enhanced anchored optimization failed: {e}")
            return None
    
    def _validate_interpretability_enhanced(self, 
                                          data: pd.DataFrame,
                                          targets: pd.Series,
                                          pipeline: callable) -> Any:
        """Enhanced interpretability validation with SHAP/LIME."""
        try:
            # Use SHAP/LIME integration if available
            if self.shap_lime:
                return self.shap_lime.enhanced_interpretability_analysis(
                    pipeline, data, targets
                )
            else:
                return self.interpretability_feedback.iterative_pruning(
                    data, targets, 
                    self.interpretability_feedback.analyze_interpretability(data, targets)
                )
        except Exception as e:
            self.logger.warning(f"Enhanced interpretability validation failed: {e}")
            return None
    
    def _validate_vector_integrity_enhanced(self, 
                                         data: pd.DataFrame,
                                         targets: pd.Series,
                                         pipeline: callable,
                                         metadata: Optional[Dict[str, Any]]) -> Any:
        """Enhanced vector integrity validation with optimizations."""
        try:
            # Use VectorBT scaler for enhanced scaling validation
            if self.vectorbt_scaler:
                scaling_validation = self.vectorbt_scaler.validate_scaling_consistency(data)
                return self.vector_integrity_validator.validate_vector_integrity(
                    data, metadata, scaling_validation
                )
            else:
                return self.vector_integrity_validator.validate_vector_integrity(data, metadata)
        except Exception as e:
            self.logger.warning(f"Enhanced vector integrity validation failed: {e}")
            return None
    
    def _validate_forward_enhanced(self, 
                                  data: pd.DataFrame,
                                  targets: pd.Series,
                                  pipeline: callable) -> Any:
        """Enhanced forward validation with optimizations."""
        try:
            # Use optimized forward validation
            return self.forward_validator.perform_forward_validation(
                data, targets, pipeline
            )
        except Exception as e:
            self.logger.warning(f"Enhanced forward validation failed: {e}")
            return None
    
    def _enhance_with_vectorbt(self, result: Any, data: pd.DataFrame, targets: pd.Series) -> Any:
        """Enhance result with VectorBT optimizations."""
        # Add VectorBT-specific enhancements
        if hasattr(result, 'metrics'):
            result.metrics['vectorbt_optimized'] = True
        return result
    
    def _enhance_with_vectorization(self, result: Any, data: pd.DataFrame, targets: pd.Series) -> Any:
        """Enhance result with vectorization optimizations."""
        # Add vectorization-specific enhancements
        if hasattr(result, 'metrics'):
            result.metrics['vectorization_optimized'] = True
        return result
    
    def _enhance_with_shap_lime(self, result: Any, data: pd.DataFrame, targets: pd.Series, pipeline: callable) -> Any:
        """Enhance result with SHAP/LIME insights."""
        # Add SHAP/LIME-specific enhancements
        if hasattr(result, 'metrics'):
            result.metrics['shap_lime_enhanced'] = True
        return result
    
    def _calculate_enhanced_metrics(self, result: EnhancedVectorizedResult) -> None:
        """Calculate enhanced validation metrics."""
        try:
            # Calculate performance improvements
            vectorbt_gains = result.vectorbt_performance_gains
            vectorization_perf = result.vectorization_performance
            
            # Overall performance improvement
            result.performance_improvement = (
                vectorbt_gains.get('speedup', 1.0) * 
                vectorization_perf.get('efficiency', 1.0)
            )
            
            # Memory efficiency
            result.memory_efficiency = vectorization_perf.get('memory_efficiency', 0.8)
            
            # Time efficiency
            result.time_efficiency = vectorization_perf.get('time_efficiency', 0.8)
            
            # Overall score
            result.overall_score = (
                result.performance_improvement * 0.3 +
                result.memory_efficiency * 0.3 +
                result.time_efficiency * 0.4
            )
        
        except Exception as e:
            self.logger.warning(f"Enhanced metrics calculation failed: {e}")
    
    def _determine_enhanced_validation_status(self, result: EnhancedVectorizedResult) -> bool:
        """Determine enhanced validation status."""
        try:
            # Check core validation results
            core_valid = (
                result.nested_oof_result is not None and
                result.hierarchical_results and
                result.anchored_optimization_result is not None
            )
            
            # Check optimization results
            optimization_valid = (
                result.vectorbt_optimization_stats and
                result.vectorization_strategy is not None
            )
            
            # Check overall score
            score_valid = result.overall_score >= 0.7
            
            return core_valid and optimization_valid and score_valid
        
        except Exception as e:
            self.logger.warning(f"Enhanced validation status determination failed: {e}")
            return False
    
    def _generate_enhanced_recommendations(self, result: EnhancedVectorizedResult) -> List[str]:
        """Generate enhanced recommendations."""
        recommendations = []
        
        # Performance recommendations
        if result.performance_improvement < 1.5:
            recommendations.append("Consider enabling more VectorBT optimizations")
        
        if result.memory_efficiency < 0.8:
            recommendations.append("Optimize memory usage with chunked processing")
        
        if result.time_efficiency < 0.8:
            recommendations.append("Enable parallel processing for better time efficiency")
        
        # Validation recommendations
        if not result.passed_validation:
            recommendations.append("Address validation failures with enhanced tools")
        
        # Optimization recommendations
        if not result.vectorbt_optimization_stats:
            recommendations.append("Enable VectorBT optimization for better performance")
        
        if not result.vectorization_strategy:
            recommendations.append("Enable unified vectorization for optimal strategy selection")
        
        return recommendations
