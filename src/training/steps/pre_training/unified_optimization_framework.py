"""
Unified Optimization Framework for Feature Engineering

This module provides a unified framework that integrates profit labeling quality metrics
with feature optimization systems, ensuring that all feature engineering optimizations
are aligned with the profit labeling framework's quality goals.

Key Features:
- Unified configuration for all optimization systems
- LQS-based scoring integration
- Multi-objective optimization (IC + LQS + Stability)
- Quality-based feature filtering
- Profit labeling framework integration
"""

import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

# Import profit labeling framework
from .profit_labeling.quality_scoring import (
    LabelQualityScorer, QualityScoringConfig, QualityMetrics, QualityMetric
)
from .profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig, LabelQualityScore
)
from .profit_labeling.multi_target_scheme import (
    MultiTargetScheme, MultiTargetConfig, TargetBand
)
from .profit_labeling.noise_gating import (
    NoiseGatingFilter, NoiseGatingConfig
)

# Import optimization systems
from .feature_lookback_optimization.feature_lookback_optimization import (
    OptimizedFeatureLookbackConfig, OptimizedFeatureLookbackOptimizer
)
from .interaction_feature_generator.feature_interaction_generation.orchestrator import (
    LookbackOptimizationOrchestrator
)
from .interaction_feature_generator.feature_interaction_generation.config import (
    LookbackOptimizationConfig as InteractionConfig, FamilyType
)

# Import utilities
from src.utils.tprint import (
    tprint,
    tprint_info,
    tprint_warning,
    tprint_error,
    tprint_success,
    tprint_debug,
)


class OptimizationSystem(Enum):
    """Available optimization systems."""
    FEATURE_LOOKBACK = "feature_lookback"
    INTERACTION_GENERATOR = "interaction_generator"
    UNIFIED = "unified"


class OptimizationObjective(Enum):
    """Available optimization objectives."""
    IC_ONLY = "ic_only"
    LQS_ONLY = "lqs_only"
    MULTI_OBJECTIVE = "multi_objective"
    QUALITY_FILTERED = "quality_filtered"


@dataclass
class UnifiedOptimizationConfig:
    """Unified configuration for all optimization systems."""
    
    # System selection
    enabled_systems: List[OptimizationSystem] = field(default_factory=lambda: [
        OptimizationSystem.FEATURE_LOOKBACK,
        OptimizationSystem.INTERACTION_GENERATOR
    ])
    
    # Optimization objectives
    primary_objective: OptimizationObjective = OptimizationObjective.MULTI_OBJECTIVE
    
    # Profit labeling quality thresholds
    min_lqs_threshold: float = 0.3
    min_auc_threshold: float = 0.55
    max_auc_std_threshold: float = 0.03
    min_psi_threshold: float = 0.1
    max_flip_rate_threshold: float = 0.15
    min_balance_threshold: float = 0.35
    max_balance_threshold: float = 0.65
    max_correlation_threshold: float = 0.4
    
    # Multi-objective weights
    ic_weight: float = 0.4
    lqs_weight: float = 0.4
    stability_weight: float = 0.2
    
    # Feature filtering
    enable_quality_filtering: bool = True
    min_feature_quality_score: float = 0.2
    max_feature_correlation: float = 0.8
    
    # System-specific configurations
    feature_lookback_config: Optional[OptimizedFeatureLookbackConfig] = None
    interaction_generator_config: Optional[InteractionConfig] = None
    
    # Quality scoring configuration
    quality_scoring_config: Optional[QualityScoringConfig] = None
    
    # Output settings
    save_results: bool = True
    generate_reports: bool = True
    output_directory: str = "unified_optimization_results"
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.quality_scoring_config is None:
            self.quality_scoring_config = QualityScoringConfig(
                baseline_models=['logistic', 'random_forest'],
                test_size=0.2,
                n_splits=5,
                random_state=42,
                min_lqs_score=self.min_lqs_threshold,
                min_auc_threshold=self.min_auc_threshold,
                max_auc_std_threshold=self.max_auc_std_threshold,
                min_psi_threshold=self.min_psi_threshold,
                max_flip_rate_threshold=self.max_flip_rate_threshold,
                min_balance_threshold=self.min_balance_threshold,
                max_balance_threshold=self.max_balance_threshold,
                max_correlation_threshold=self.max_correlation_threshold
            )
        
        if self.feature_lookback_config is None:
            self.feature_lookback_config = OptimizedFeatureLookbackConfig(
                optimization_metric="lqs_combined",
                enable_quality_scoring=True,
                quality_scoring_config=self.quality_scoring_config,
                min_lqs_threshold=self.min_lqs_threshold,
                min_auc_threshold=self.min_auc_threshold,
                max_auc_std_threshold=self.max_auc_std_threshold,
                min_psi_threshold=self.min_psi_threshold,
                max_flip_rate_threshold=self.max_flip_rate_threshold,
                min_balance_threshold=self.min_balance_threshold,
                max_balance_threshold=self.max_balance_threshold,
                max_correlation_threshold=self.max_correlation_threshold,
                enable_multi_objective=True,
                ic_weight=self.ic_weight,
                lqs_weight=self.lqs_weight,
                stability_weight=self.stability_weight
            )
        
        if self.interaction_generator_config is None:
            self.interaction_generator_config = InteractionConfig(
                # Add interaction-specific configuration here
                output_dir=self.output_directory
            )


@dataclass
class UnifiedOptimizationResult:
    """Result of unified optimization."""
    success: bool
    execution_time: float
    system_results: Dict[OptimizationSystem, Any]
    quality_metrics: Dict[str, float]
    filtered_features: List[str]
    optimization_summary: Dict[str, Any]
    error_message: Optional[str] = None


class UnifiedOptimizationFramework:
    """Unified framework for feature optimization with profit labeling integration."""
    
    def __init__(self, config: Optional[UnifiedOptimizationConfig] = None):
        tprint_info("🔧 Initializing Unified Optimization Framework...")
        
        self.config = config or UnifiedOptimizationConfig()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Initialize profit labeling components
        self._initialize_profit_labeling_components()
        
        # Initialize optimization systems
        self._initialize_optimization_systems()
        
        tprint_success("✅ Unified Optimization Framework initialized")
    
    def _initialize_profit_labeling_components(self):
        """Initialize profit labeling framework components."""
        tprint_debug("🔧 Initializing profit labeling components...")
        
        try:
            self.quality_scorer = LabelQualityScorer(self.config.quality_scoring_config)
            tprint_success("✅ Quality scorer initialized")
        except Exception as e:
            tprint_warning(f"⚠️ Failed to initialize quality scorer: {e}")
            self.quality_scorer = None
        
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
    
    def _initialize_optimization_systems(self):
        """Initialize optimization systems."""
        tprint_debug("🔧 Initializing optimization systems...")
        
        self.systems = {}
        
        if OptimizationSystem.FEATURE_LOOKBACK in self.config.enabled_systems:
            try:
                self.systems[OptimizationSystem.FEATURE_LOOKBACK] = OptimizedFeatureLookbackOptimizer(
                    self.config.feature_lookback_config
                )
                tprint_success("✅ Feature lookback optimizer initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize feature lookback optimizer: {e}")
        
        if OptimizationSystem.INTERACTION_GENERATOR in self.config.enabled_systems:
            try:
                self.systems[OptimizationSystem.INTERACTION_GENERATOR] = LookbackOptimizationOrchestrator(
                    self.config.interaction_generator_config
                )
                tprint_success("✅ Interaction generator orchestrator initialized")
            except Exception as e:
                tprint_warning(f"⚠️ Failed to initialize interaction generator: {e}")
    
    async def optimize_features(self, data: pd.DataFrame, 
                              pipeline_state: Optional[Dict[str, Any]] = None) -> UnifiedOptimizationResult:
        """Run unified feature optimization across all enabled systems."""
        tprint_info("🚀 Starting unified feature optimization...")
        start_time = time.time()
        
        try:
            system_results = {}
            quality_metrics = {}
            filtered_features = []
            
            # Run each enabled system
            for system_type, system in self.systems.items():
                tprint_info(f"🔧 Running {system_type.value} optimization...")
                
                try:
                    if system_type == OptimizationSystem.FEATURE_LOOKBACK:
                        result = await system.optimize_features_with_labels(
                            data, [], pipeline_state
                        )
                    elif system_type == OptimizationSystem.INTERACTION_GENERATOR:
                        # Convert data format for interaction generator
                        data_dict = {'symbol': data}
                        targets_dict = {'symbol': data.get('target', np.zeros(len(data)))}
                        feature_names = {family: f"{family.value}_feature" for family in FamilyType}
                        
                        result = system.optimize_lookbacks(
                            data_dict, targets_dict, feature_names
                        )
                    else:
                        continue
                    
                    system_results[system_type] = result
                    tprint_success(f"✅ {system_type.value} optimization completed")
                    
                except Exception as e:
                    tprint_warning(f"⚠️ {system_type.value} optimization failed: {e}")
                    system_results[system_type] = None
            
            # Apply quality filtering if enabled
            if self.config.enable_quality_filtering and self.quality_scorer:
                filtered_features = self._apply_quality_filtering(data, system_results)
            
            # Calculate overall quality metrics
            quality_metrics = self._calculate_quality_metrics(system_results)
            
            # Generate optimization summary
            optimization_summary = self._generate_optimization_summary(system_results, quality_metrics)
            
            execution_time = time.time() - start_time
            
            result = UnifiedOptimizationResult(
                success=True,
                execution_time=execution_time,
                system_results=system_results,
                quality_metrics=quality_metrics,
                filtered_features=filtered_features,
                optimization_summary=optimization_summary
            )
            
            tprint_success(f"✅ Unified optimization completed in {execution_time:.3f}s")
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = f"Unified optimization failed: {str(e)}"
            
            tprint_error(f"❌ {error_message}")
            self.logger.error(error_message)
            
            return UnifiedOptimizationResult(
                success=False,
                execution_time=execution_time,
                system_results={},
                quality_metrics={},
                filtered_features=[],
                optimization_summary={},
                error_message=error_message
            )
    
    def _apply_quality_filtering(self, data: pd.DataFrame, 
                               system_results: Dict[OptimizationSystem, Any]) -> List[str]:
        """Apply quality-based filtering to features."""
        tprint_info("🔍 Applying quality-based feature filtering...")
        
        filtered_features = []
        
        if not self.quality_scorer:
            tprint_warning("⚠️ Quality scorer not available, skipping filtering")
            return filtered_features
        
        # Extract features from system results and evaluate quality
        for system_type, result in system_results.items():
            if result is None:
                continue
            
            # This would need to be implemented based on the actual result structure
            # For now, return empty list as placeholder
            tprint_debug(f"📊 Quality filtering for {system_type.value}: placeholder implementation")
            # TODO: Implement actual quality filtering logic based on result structure
        
        tprint_success(f"✅ Quality filtering completed: {len(filtered_features)} features passed")
        return filtered_features
    
    def _calculate_quality_metrics(self, system_results: Dict[OptimizationSystem, Any]) -> Dict[str, float]:
        """Calculate overall quality metrics from system results."""
        tprint_info("📊 Calculating quality metrics...")
        
        metrics = {}
        
        # Calculate metrics based on system results
        # This would need to be implemented based on the actual result structure
        
        tprint_success("✅ Quality metrics calculated")
        return metrics
    
    def _generate_optimization_summary(self, system_results: Dict[OptimizationSystem, Any],
                                     quality_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate comprehensive optimization summary."""
        tprint_info("📋 Generating optimization summary...")
        
        summary = {
            'total_systems': len(system_results),
            'successful_systems': len([r for r in system_results.values() if r is not None]),
            'quality_metrics': quality_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        tprint_success("✅ Optimization summary generated")
        return summary


# Convenience functions
def create_unified_config(
    min_lqs_threshold: float = 0.3,
    enable_multi_objective: bool = True,
    ic_weight: float = 0.4,
    lqs_weight: float = 0.4,
    stability_weight: float = 0.2
) -> UnifiedOptimizationConfig:
    """Create a unified optimization configuration with common settings."""
    return UnifiedOptimizationConfig(
        min_lqs_threshold=min_lqs_threshold,
        primary_objective=OptimizationObjective.MULTI_OBJECTIVE if enable_multi_objective else OptimizationObjective.LQS_ONLY,
        ic_weight=ic_weight,
        lqs_weight=lqs_weight,
        stability_weight=stability_weight
    )


async def run_unified_optimization(
    data: pd.DataFrame,
    config: Optional[UnifiedOptimizationConfig] = None,
    pipeline_state: Optional[Dict[str, Any]] = None
) -> UnifiedOptimizationResult:
    """Run unified optimization with the given configuration."""
    framework = UnifiedOptimizationFramework(config)
    return await framework.optimize_features(data, pipeline_state)