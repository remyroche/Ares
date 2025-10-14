"""
Enhanced Multi-Horizon Profit Labeler Integration

This module provides the enhanced version of the multi-horizon profit labeler that
integrates all the advanced research components for a fully data-driven approach.

Key Enhancements Integrated:
1. ML-based Label Quality Assessment
2. Adaptive Market Regime-Aware Labeling  
3. Advanced Statistical Validation
4. Ensemble Labeling Approaches
5. Dynamic Target and Horizon Optimization
6. Contextual Feature Engineering
7. Backtesting-Integrated Validation
8. Real-time Performance Monitoring
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
from pathlib import Path
from datetime import datetime
import warnings

from src.utils.logger import get_logger
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, 
    MultiHorizonConfig
)

# Import all enhancement components
from .ml_label_quality_assessor import (
    MLLabelQualityAssessor,
    MLQualityAssessmentConfig,
    enhance_labels_with_ml
)
from .adaptive_labeling_strategy import (
    AdaptiveLabelingStrategy,
    AdaptiveLabelingConfig,
    get_regime_adaptive_config
)
from .advanced_statistical_validator import (
    AdvancedStatisticalValidator,
    AdvancedValidationConfig,
    validate_labels_advanced
)
from .ensemble_labeling_system import (
    EnsembleLabelingSystem,
    EnsembleLabelingConfig,
    generate_ensemble_labels
)
from .dynamic_target_optimizer import (
    JointTargetHorizonOptimizer,
    DynamicOptimizationConfig,
    create_optimized_multi_horizon_config
)
from .contextual_feature_labeling import (
    ContextualFeatureEngineer,
    ContextualFeatureConfig,
    create_feature_enhanced_labels
)
from .backtesting_integrated_validator import (
    BacktestingIntegratedValidator,
    BacktestingConfig,
    validate_labels_through_backtesting
)
from .bonus_penalty_optimizer import (
    BonusPenaltyOptimizer,
    BonusPenaltyOptimizationConfig,
    ModifiedMultiHorizonLabeler,
    optimize_bonus_penalty_parameters
)


class EnhancementLevel(Enum):
    """Enumeration of enhancement levels."""
    BASIC = "basic"                    # Original labeler only
    ML_ENHANCED = "ml_enhanced"        # + ML quality assessment
    ADAPTIVE = "adaptive"              # + Adaptive strategies
    ENSEMBLE = "ensemble"              # + Ensemble methods
    FULLY_OPTIMIZED = "fully_optimized"  # All enhancements


@dataclass
class EnhancedLabelingConfig:
    """Configuration for enhanced labeling system."""
    # Enhancement level
    enhancement_level: EnhancementLevel = EnhancementLevel.FULLY_OPTIMIZED
    
    # Base configuration
    base_config: Optional[MultiHorizonConfig] = None
    
    # Component configurations
    ml_config: Optional[MLQualityAssessmentConfig] = None
    adaptive_config: Optional[AdaptiveLabelingConfig] = None
    validation_config: Optional[AdvancedValidationConfig] = None
    ensemble_config: Optional[EnsembleLabelingConfig] = None
    optimization_config: Optional[DynamicOptimizationConfig] = None
    feature_config: Optional[ContextualFeatureConfig] = None
    backtesting_config: Optional[BacktestingConfig] = None
    bonus_penalty_config: Optional[BonusPenaltyOptimizationConfig] = None
    
    # Integration settings
    enable_ml_enhancement: bool = True
    enable_adaptive_labeling: bool = True
    enable_ensemble_methods: bool = True
    enable_feature_enhancement: bool = True
    enable_dynamic_optimization: bool = True
    enable_advanced_validation: bool = True
    enable_backtesting_validation: bool = True
    enable_bonus_penalty_optimization: bool = True
    
    # Performance monitoring
    enable_performance_monitoring: bool = True
    monitoring_window: int = 1000
    performance_update_frequency: int = 100
    
    # Caching and optimization
    enable_caching: bool = True
    cache_duration_minutes: int = 60
    parallel_processing: bool = True
    
    # Output settings
    save_intermediate_results: bool = True
    output_directory: str = "enhanced_labeling_results"
    generate_reports: bool = True


@dataclass
class EnhancedLabelingResult:
    """Result container for enhanced labeling."""
    # Core labeling results
    enhanced_labels: pd.DataFrame
    base_labels: pd.DataFrame
    
    # Component results
    ml_assessment_result: Optional[Any] = None
    adaptive_result: Optional[Any] = None
    ensemble_result: Optional[Any] = None
    feature_result: Optional[Any] = None
    validation_results: Optional[Dict[str, Any]] = None
    backtesting_result: Optional[Any] = None
    bonus_penalty_result: Optional[Any] = None
    
    # Performance metrics
    enhancement_metrics: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    config_used: EnhancedLabelingConfig = None
    processing_time: float = 0.0
    enhancement_level: EnhancementLevel = EnhancementLevel.BASIC
    timestamp: datetime = field(default_factory=datetime.now)


class EnhancedMultiHorizonProfitLabeler:
    """
    Enhanced Multi-Horizon Profit Labeler with full data-driven capabilities.
    
    This class integrates all research enhancements to provide a comprehensive,
    adaptive, and statistically rigorous profit labeling system.
    
    Key Features:
    1. **ML-Enhanced Quality Assessment**: Uses machine learning to assess and improve label quality
    2. **Adaptive Market Regime Awareness**: Automatically adjusts parameters based on market conditions  
    3. **Ensemble Labeling**: Combines multiple labeling strategies for robustness
    4. **Dynamic Optimization**: Data-driven discovery of optimal targets and horizons
    5. **Advanced Statistical Validation**: Rigorous statistical testing of label quality
    6. **Backtesting Integration**: Validates labels through actual trading performance
    7. **Contextual Feature Engineering**: Rich market context for better labeling decisions
    8. **Real-time Performance Monitoring**: Continuous monitoring and adaptation
    """
    
    def __init__(self, config: Optional[EnhancedLabelingConfig] = None):
        """Initialize enhanced multi-horizon profit labeler."""
        self.config = config or EnhancedLabelingConfig()
        self.logger = get_logger('EnhancedMultiHorizonProfitLabeler')
        
        # Initialize base labeler
        self.base_labeler = MultiHorizonProfitLabeler(self.config.base_config)
        
        # Initialize enhancement components based on configuration
        self._initialize_enhancement_components()
        
        # State tracking
        self.labeling_history: List[EnhancedLabelingResult] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        self.cache: Dict[str, Any] = {}
        
        self.logger.info('🚀 Enhanced Multi-Horizon Profit Labeler initialized')
        self.logger.info(f'   → Enhancement level: {self.config.enhancement_level.value}')
        self.logger.info(f'   → Components enabled: {self._get_enabled_components()}')
    
    def _initialize_enhancement_components(self):
        """Initialize enhancement components based on configuration."""
        self.components = {}
        
        # ML Quality Assessor
        if (self.config.enable_ml_enhancement and 
            self.config.enhancement_level.value in ['ml_enhanced', 'adaptive', 'ensemble', 'fully_optimized']):
            self.components['ml_assessor'] = MLLabelQualityAssessor(self.config.ml_config)
            self.logger.info('   ✓ ML Quality Assessor initialized')
        
        # Adaptive Labeling Strategy
        if (self.config.enable_adaptive_labeling and 
            self.config.enhancement_level.value in ['adaptive', 'ensemble', 'fully_optimized']):
            self.components['adaptive_strategy'] = AdaptiveLabelingStrategy(self.config.adaptive_config)
            self.logger.info('   ✓ Adaptive Labeling Strategy initialized')
        
        # Ensemble Labeling System
        if (self.config.enable_ensemble_methods and 
            self.config.enhancement_level.value in ['ensemble', 'fully_optimized']):
            self.components['ensemble_system'] = EnsembleLabelingSystem(self.config.ensemble_config)
            self.logger.info('   ✓ Ensemble Labeling System initialized')
        
        # Feature Engineer
        if (self.config.enable_feature_enhancement and 
            self.config.enhancement_level == EnhancementLevel.FULLY_OPTIMIZED):
            self.components['feature_engineer'] = ContextualFeatureEngineer(self.config.feature_config)
            self.logger.info('   ✓ Contextual Feature Engineer initialized')
        
        # Dynamic Optimizer
        if (self.config.enable_dynamic_optimization and 
            self.config.enhancement_level == EnhancementLevel.FULLY_OPTIMIZED):
            self.components['dynamic_optimizer'] = JointTargetHorizonOptimizer(self.config.optimization_config)
            self.logger.info('   ✓ Dynamic Target Optimizer initialized')
        
        # Advanced Validator
        if (self.config.enable_advanced_validation and 
            self.config.enhancement_level == EnhancementLevel.FULLY_OPTIMIZED):
            self.components['advanced_validator'] = AdvancedStatisticalValidator(self.config.validation_config)
            self.logger.info('   ✓ Advanced Statistical Validator initialized')
        
        # Backtesting Validator
        if (self.config.enable_backtesting_validation and 
            self.config.enhancement_level == EnhancementLevel.FULLY_OPTIMIZED):
            self.components['backtesting_validator'] = BacktestingIntegratedValidator(self.config.backtesting_config)
            self.logger.info('   ✓ Backtesting Validator initialized')
        
        # Bonus/Penalty Optimizer
        if (self.config.enable_bonus_penalty_optimization and 
            self.config.enhancement_level == EnhancementLevel.FULLY_OPTIMIZED):
            self.components['bonus_penalty_optimizer'] = BonusPenaltyOptimizer(self.config.bonus_penalty_config)
            self.logger.info('   ✓ Bonus/Penalty Optimizer initialized')
    
    def generate_enhanced_labels(self, market_data: pd.DataFrame) -> EnhancedLabelingResult:
        """
        Generate enhanced profit labels using all enabled components.
        
        Args:
            market_data: OHLCV market data
            
        Returns:
            EnhancedLabelingResult with comprehensive labeling and analysis
        """
        start_time = datetime.now()
        self.logger.info('🔍 Generating enhanced profit labels')
        
        if len(market_data) < 100:
            self.logger.warning('⚠️ Insufficient data for enhanced labeling')
            return self._create_fallback_result(market_data)
        
        # Check cache
        cache_key = self._generate_cache_key(market_data)
        if self.config.enable_caching and cache_key in self.cache:
            cached_result = self.cache[cache_key]
            if self._is_cache_valid(cached_result):
                self.logger.info('📋 Using cached labeling result')
                return cached_result
        
        # Initialize result container
        result = EnhancedLabelingResult(
            enhanced_labels=pd.DataFrame(),
            base_labels=pd.DataFrame(),
            config_used=self.config,
            enhancement_level=self.config.enhancement_level
        )
        
        try:
            # Step 1: Bonus/penalty optimization (if enabled)
            optimized_labeler = self._apply_bonus_penalty_optimization(market_data)
            if optimized_labeler:
                self.base_labeler = optimized_labeler
                self.logger.info('   ✓ Applied bonus/penalty optimization')
            
            # Step 2: Dynamic optimization (if enabled)
            optimized_config = self._apply_dynamic_optimization(market_data)
            if optimized_config:
                self.base_labeler.config = optimized_config
                self.logger.info('   ✓ Applied dynamic optimization')
            
            # Step 3: Adaptive configuration (if enabled)
            adaptive_config = self._apply_adaptive_configuration(market_data)
            if adaptive_config:
                self.base_labeler.config = adaptive_config.config
                result.adaptive_result = adaptive_config
                self.logger.info('   ✓ Applied adaptive configuration')
            
            # Step 4: Generate base labels or ensemble labels
            if 'ensemble_system' in self.components:
                # Use ensemble approach
                ensemble_result = self.components['ensemble_system'].generate_ensemble_labels(market_data)
                result.base_labels = pd.DataFrame()  # Ensemble doesn't use base labeler
                result.enhanced_labels = ensemble_result.ensemble_labels
                result.ensemble_result = ensemble_result
                self.logger.info('   ✓ Generated ensemble labels')
            else:
                # Use base labeler
                result.base_labels = self.base_labeler.generate_labels(market_data.copy())
                result.enhanced_labels = result.base_labels.copy()
                self.logger.info('   ✓ Generated base labels')
            
            # Step 4: Feature enhancement (if enabled)
            if 'feature_engineer' in self.components and not result.enhanced_labels.empty:
                enhanced_labels = self._apply_feature_enhancement(
                    result.enhanced_labels, market_data
                )
                result.enhanced_labels = enhanced_labels
                self.logger.info('   ✓ Applied feature enhancement')
            
            # Step 5: ML quality assessment and enhancement (if enabled)
            if 'ml_assessor' in self.components and not result.enhanced_labels.empty:
                ml_result = self._apply_ml_enhancement(result.enhanced_labels, market_data)
                result.enhanced_labels = ml_result['enhanced_labels']
                result.ml_assessment_result = ml_result['assessment']
                self.logger.info('   ✓ Applied ML enhancement')
            
            # Step 6: Advanced statistical validation (if enabled)
            if 'advanced_validator' in self.components and not result.enhanced_labels.empty:
                validation_results = self._apply_advanced_validation(
                    result.enhanced_labels, market_data
                )
                result.validation_results = validation_results
                self.logger.info('   ✓ Applied advanced validation')
            
            # Step 7: Backtesting validation (if enabled and sufficient data)
            if ('backtesting_validator' in self.components and 
                not result.enhanced_labels.empty and len(market_data) > 1000):
                backtesting_result = self._apply_backtesting_validation(
                    result.enhanced_labels, market_data
                )
                result.backtesting_result = backtesting_result
                self.logger.info('   ✓ Applied backtesting validation')
            
            # Step 8: Calculate enhancement metrics
            result.enhancement_metrics = self._calculate_enhancement_metrics(result, market_data)
            result.quality_scores = self._calculate_quality_scores(result)
            
            # Step 9: Performance monitoring update
            if self.config.enable_performance_monitoring:
                self._update_performance_monitoring(result)
            
        except Exception as e:
            self.logger.error(f'❌ Enhanced labeling failed: {e}')
            return self._create_fallback_result(market_data)
        
        # Calculate processing time
        result.processing_time = (datetime.now() - start_time).total_seconds()
        
        # Store in history and cache
        self.labeling_history.append(result)
        if self.config.enable_caching:
            self.cache[cache_key] = result
        
        # Clean up old history
        if len(self.labeling_history) > 100:
            self.labeling_history = self.labeling_history[-100:]
        
        self.logger.info('✅ Enhanced labeling completed')
        self.logger.info(f'   → Processing time: {result.processing_time:.2f}s')
        self.logger.info(f'   → Enhancement score: {result.quality_scores.get("overall_quality", 0):.3f}')
        
        return result
    
    def _apply_dynamic_optimization(self, market_data: pd.DataFrame) -> Optional[MultiHorizonConfig]:
        """Apply dynamic target and horizon optimization."""
        if 'dynamic_optimizer' not in self.components:
            return None
        
        try:
            optimization_result = self.components['dynamic_optimizer'].optimize_target_horizon_combinations(market_data)
            
            if optimization_result.objective_score > 0.5:  # Threshold for accepting optimization
                return create_optimized_multi_horizon_config(market_data, self.config.optimization_config)
            
        except Exception as e:
            self.logger.warning(f'Dynamic optimization failed: {e}')
        
        return None
    
    def _apply_bonus_penalty_optimization(self, market_data: pd.DataFrame) -> Optional[MultiHorizonProfitLabeler]:
        """Apply data-driven bonus/penalty optimization."""
        if 'bonus_penalty_optimizer' not in self.components:
            return None
        
        try:
            # Optimize bonus/penalty parameters
            optimization_result = self.components['bonus_penalty_optimizer'].optimize_bonus_penalty_parameters(market_data)
            
            if optimization_result.objective_score > 0.4:  # Threshold for accepting optimization
                # Create modified labeler with optimized parameters
                modified_labeler = ModifiedMultiHorizonLabeler(optimization_result.parameters)
                modified_labeler.config = self.base_labeler.config  # Keep existing config
                
                self.logger.info(f'   → Bonus/penalty optimization score: {optimization_result.objective_score:.3f}')
                return modified_labeler
            
        except Exception as e:
            self.logger.warning(f'Bonus/penalty optimization failed: {e}')
        
        return None
    
    def _apply_adaptive_configuration(self, market_data: pd.DataFrame) -> Optional[Any]:
        """Apply adaptive market regime-aware configuration."""
        if 'adaptive_strategy' not in self.components:
            return None
        
        try:
            return self.components['adaptive_strategy'].get_adaptive_config(market_data)
        except Exception as e:
            self.logger.warning(f'Adaptive configuration failed: {e}')
            return None
    
    def _apply_feature_enhancement(self, 
                                 labels: pd.DataFrame, 
                                 market_data: pd.DataFrame) -> pd.DataFrame:
        """Apply contextual feature enhancement."""
        try:
            feature_result = self.components['feature_engineer'].engineer_features(market_data)
            
            # Apply feature-based adjustments to labels
            enhanced_labels = self.components['feature_engineer'].apply_labeling_adjustments(
                labels, feature_result.features_df, self.base_labeler.config
            )
            
            return enhanced_labels
            
        except Exception as e:
            self.logger.warning(f'Feature enhancement failed: {e}')
            return labels
    
    def _apply_ml_enhancement(self, 
                            labels: pd.DataFrame, 
                            market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply ML-based quality assessment and enhancement."""
        try:
            # Assess label quality
            assessment_result = self.components['ml_assessor'].assess_label_quality(
                labels, market_data
            )
            
            # Enhance labels with ML
            enhanced_labels = self.components['ml_assessor'].enhance_label_quality(
                labels, market_data
            )
            
            return {
                'enhanced_labels': enhanced_labels,
                'assessment': assessment_result
            }
            
        except Exception as e:
            self.logger.warning(f'ML enhancement failed: {e}')
            return {'enhanced_labels': labels, 'assessment': None}
    
    def _apply_advanced_validation(self, 
                                 labels: pd.DataFrame, 
                                 market_data: pd.DataFrame) -> Dict[str, Any]:
        """Apply advanced statistical validation."""
        try:
            return self.components['advanced_validator'].comprehensive_validate(
                labels, market_data
            )
        except Exception as e:
            self.logger.warning(f'Advanced validation failed: {e}')
            return {}
    
    def _apply_backtesting_validation(self, 
                                    labels: pd.DataFrame, 
                                    market_data: pd.DataFrame) -> Optional[Any]:
        """Apply backtesting-integrated validation."""
        try:
            return self.components['backtesting_validator'].validate_through_backtesting(
                labels, market_data
            )
        except Exception as e:
            self.logger.warning(f'Backtesting validation failed: {e}')
            return None
    
    def _calculate_enhancement_metrics(self, 
                                     result: EnhancedLabelingResult, 
                                     market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate enhancement metrics."""
        metrics = {}
        
        # Basic metrics
        metrics['n_features'] = len(result.enhanced_labels.columns)
        metrics['n_samples'] = len(result.enhanced_labels)
        
        # ML assessment metrics
        if result.ml_assessment_result:
            ml_scores = result.ml_assessment_result.quality_scores
            metrics['ml_predictive_power'] = ml_scores.get('PREDICTIVE_POWER', 0)
            metrics['ml_stability'] = ml_scores.get('STABILITY_SCORE', 0)
        
        # Validation metrics
        if result.validation_results:
            significant_count = sum(1 for r in result.validation_results.values() if r.is_significant)
            metrics['validation_significance_ratio'] = significant_count / len(result.validation_results)
        
        # Backtesting metrics
        if result.backtesting_result:
            summary = result.backtesting_result.validation_summary
            metrics['backtesting_score'] = summary.get('overall_score', 0)
            metrics['profitable_strategies'] = summary.get('profitability_score', 0)
        
        # Ensemble metrics
        if result.ensemble_result:
            metrics['ensemble_diversity'] = result.ensemble_result.diversity_score
            metrics['ensemble_performance'] = result.ensemble_result.performance_metrics.get('correlation', 0)
        
        return metrics
    
    def _calculate_quality_scores(self, result: EnhancedLabelingResult) -> Dict[str, float]:
        """Calculate overall quality scores."""
        scores = {}
        
        # Component quality scores
        component_scores = []
        
        if result.ml_assessment_result:
            ml_quality = np.mean(list(result.ml_assessment_result.quality_scores.values()))
            component_scores.append(ml_quality)
            scores['ml_quality'] = ml_quality
        
        if result.validation_results:
            validation_quality = np.mean([
                r.summary_statistic for r in result.validation_results.values()
            ])
            component_scores.append(validation_quality)
            scores['validation_quality'] = validation_quality
        
        if result.backtesting_result:
            backtesting_quality = result.backtesting_result.validation_summary.get('overall_score', 0)
            component_scores.append(backtesting_quality)
            scores['backtesting_quality'] = backtesting_quality
        
        if result.ensemble_result:
            ensemble_quality = result.ensemble_result.diversity_score * 0.5 + \
                             result.ensemble_result.performance_metrics.get('correlation', 0) * 0.5
            component_scores.append(ensemble_quality)
            scores['ensemble_quality'] = ensemble_quality
        
        # Overall quality score
        if component_scores:
            scores['overall_quality'] = np.mean(component_scores)
        else:
            scores['overall_quality'] = 0.5  # Neutral score
        
        return scores
    
    def _update_performance_monitoring(self, result: EnhancedLabelingResult):
        """Update performance monitoring metrics."""
        # Track key metrics over time
        overall_quality = result.quality_scores.get('overall_quality', 0)
        
        if 'overall_quality' not in self.performance_metrics:
            self.performance_metrics['overall_quality'] = []
        
        self.performance_metrics['overall_quality'].append(overall_quality)
        
        # Keep limited history
        if len(self.performance_metrics['overall_quality']) > self.config.monitoring_window:
            self.performance_metrics['overall_quality'] = \
                self.performance_metrics['overall_quality'][-self.config.monitoring_window:]
        
        # Track other metrics
        for metric_name, value in result.enhancement_metrics.items():
            if metric_name not in self.performance_metrics:
                self.performance_metrics[metric_name] = []
            
            self.performance_metrics[metric_name].append(value)
            
            if len(self.performance_metrics[metric_name]) > self.config.monitoring_window:
                self.performance_metrics[metric_name] = \
                    self.performance_metrics[metric_name][-self.config.monitoring_window:]
    
    def _generate_cache_key(self, market_data: pd.DataFrame) -> str:
        """Generate cache key for market data."""
        # Simple cache key based on data shape and last timestamp
        data_hash = hash(str(market_data.shape) + str(market_data.index[-1]))
        config_hash = hash(str(self.config.enhancement_level.value))
        return f"enhanced_labels_{data_hash}_{config_hash}"
    
    def _is_cache_valid(self, cached_result: EnhancedLabelingResult) -> bool:
        """Check if cached result is still valid."""
        if not self.config.enable_caching:
            return False
        
        cache_age = datetime.now() - cached_result.timestamp
        return cache_age.total_seconds() < (self.config.cache_duration_minutes * 60)
    
    def _get_enabled_components(self) -> str:
        """Get string representation of enabled components."""
        enabled = []
        if 'ml_assessor' in self.components:
            enabled.append('ML Assessment')
        if 'adaptive_strategy' in self.components:
            enabled.append('Adaptive Strategy')
        if 'ensemble_system' in self.components:
            enabled.append('Ensemble System')
        if 'feature_engineer' in self.components:
            enabled.append('Feature Engineering')
        if 'dynamic_optimizer' in self.components:
            enabled.append('Dynamic Optimization')
        if 'advanced_validator' in self.components:
            enabled.append('Advanced Validation')
        if 'backtesting_validator' in self.components:
            enabled.append('Backtesting Validation')
        
        return ', '.join(enabled) if enabled else 'Base Labeler Only'
    
    def _create_fallback_result(self, market_data: pd.DataFrame) -> EnhancedLabelingResult:
        """Create fallback result when enhancement fails."""
        try:
            # Use base labeler as fallback
            base_labels = self.base_labeler.generate_labels(market_data.copy())
            
            return EnhancedLabelingResult(
                enhanced_labels=base_labels,
                base_labels=base_labels,
                config_used=self.config,
                enhancement_level=EnhancementLevel.BASIC,
                enhancement_metrics={'fallback': True},
                quality_scores={'overall_quality': 0.5}
            )
        except Exception as e:
            self.logger.error(f'Even fallback labeling failed: {e}')
            
            # Return empty result
            empty_labels = pd.DataFrame(index=market_data.index)
            empty_labels['overall_opportunity'] = 0.0
            
            return EnhancedLabelingResult(
                enhanced_labels=empty_labels,
                base_labels=empty_labels,
                config_used=self.config,
                enhancement_level=EnhancementLevel.BASIC,
                enhancement_metrics={'error': True},
                quality_scores={'overall_quality': 0.0}
            )
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance monitoring summary."""
        summary = {}
        
        if not self.performance_metrics:
            return summary
        
        for metric_name, values in self.performance_metrics.items():
            if values:
                summary[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'trend': np.polyfit(range(len(values)), values, 1)[0] if len(values) > 1 else 0,
                    'count': len(values)
                }
        
        return summary
    
    def generate_comprehensive_report(self) -> str:
        """Generate comprehensive enhancement report."""
        if not self.labeling_history:
            return "No labeling history available."
        
        latest_result = self.labeling_history[-1]
        
        report_lines = [
            "# Enhanced Multi-Horizon Profit Labeling Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration Summary",
            f"**Enhancement Level**: {latest_result.enhancement_level.value}",
            f"**Components Enabled**: {self._get_enabled_components()}",
            f"**Processing Time**: {latest_result.processing_time:.2f} seconds",
            "",
            "## Quality Assessment",
            f"**Overall Quality Score**: {latest_result.quality_scores.get('overall_quality', 0):.3f}",
        ]
        
        # Add component-specific results
        if latest_result.ml_assessment_result:
            ml_scores = latest_result.ml_assessment_result.quality_scores
            report_lines.extend([
                "",
                "### ML Quality Assessment",
                f"- Predictive Power: {ml_scores.get('PREDICTIVE_POWER', 0):.3f}",
                f"- Stability Score: {ml_scores.get('STABILITY_SCORE', 0):.3f}",
                f"- Information Content: {ml_scores.get('INFORMATION_CONTENT', 0):.3f}"
            ])
        
        if latest_result.validation_results:
            significant_count = sum(1 for r in latest_result.validation_results.values() if r.is_significant)
            report_lines.extend([
                "",
                "### Advanced Statistical Validation",
                f"- Total Tests: {len(latest_result.validation_results)}",
                f"- Significant Results: {significant_count}",
                f"- Significance Ratio: {significant_count / len(latest_result.validation_results):.2%}"
            ])
        
        if latest_result.backtesting_result:
            bt_summary = latest_result.backtesting_result.validation_summary
            report_lines.extend([
                "",
                "### Backtesting Validation",
                f"- Validation Result: {bt_summary.get('validation_result', 'N/A')}",
                f"- Overall Score: {bt_summary.get('overall_score', 0):.3f}",
                f"- Successful Strategies: {bt_summary.get('successful_strategies', 0)}"
            ])
        
        if latest_result.ensemble_result:
            report_lines.extend([
                "",
                "### Ensemble Analysis",
                f"- Diversity Score: {latest_result.ensemble_result.diversity_score:.3f}",
                f"- Strategy Count: {len(latest_result.ensemble_result.strategy_results)}",
                f"- Performance Correlation: {latest_result.ensemble_result.performance_metrics.get('correlation', 0):.3f}"
            ])
        
        # Performance monitoring summary
        perf_summary = self.get_performance_summary()
        if perf_summary:
            report_lines.extend([
                "",
                "## Performance Monitoring",
                f"**Monitoring Window**: {len(self.performance_metrics.get('overall_quality', []))} samples",
            ])
            
            for metric, stats in perf_summary.items():
                trend_direction = "↗️" if stats['trend'] > 0 else "↘️" if stats['trend'] < 0 else "➡️"
                report_lines.append(f"- {metric}: {stats['current']:.3f} {trend_direction} (μ={stats['mean']:.3f}, σ={stats['std']:.3f})")
        
        # Recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            *self._generate_recommendations(latest_result)
        ])
        
        return "\n".join(report_lines)
    
    def _generate_recommendations(self, result: EnhancedLabelingResult) -> List[str]:
        """Generate recommendations based on results."""
        recommendations = []
        
        overall_quality = result.quality_scores.get('overall_quality', 0)
        
        if overall_quality < 0.3:
            recommendations.extend([
                "⚠️ Low overall quality detected",
                "- Consider adjusting base labeling parameters",
                "- Review market data quality and completeness",
                "- Increase enhancement level if using basic mode"
            ])
        elif overall_quality < 0.6:
            recommendations.extend([
                "📊 Moderate quality - room for improvement",
                "- Fine-tune ML assessment parameters",
                "- Consider enabling additional enhancement components",
                "- Monitor performance trends closely"
            ])
        else:
            recommendations.extend([
                "✅ Good quality labeling achieved",
                "- Current configuration is performing well",
                "- Continue monitoring for any degradation",
                "- Consider periodic recalibration"
            ])
        
        # Component-specific recommendations
        if result.backtesting_result:
            bt_score = result.backtesting_result.validation_summary.get('overall_score', 0)
            if bt_score < 0.5:
                recommendations.append("- Backtesting shows poor performance - review labeling strategy")
        
        if result.ensemble_result:
            diversity = result.ensemble_result.diversity_score
            if diversity < 0.5:
                recommendations.append("- Low ensemble diversity - consider adding more varied strategies")
        
        # Performance monitoring recommendations
        perf_summary = self.get_performance_summary()
        if perf_summary and 'overall_quality' in perf_summary:
            trend = perf_summary['overall_quality']['trend']
            if trend < -0.01:  # Declining trend
                recommendations.append("- Quality trend is declining - investigate potential causes")
        
        if not recommendations:
            recommendations.append("- System is operating within normal parameters")
        
        return recommendations
    
    def save_results(self, output_directory: Optional[str] = None):
        """Save enhanced labeling results."""
        if not self.labeling_history:
            self.logger.warning('No results to save')
            return
        
        output_dir = Path(output_directory or self.config.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        latest_result = self.labeling_history[-1]
        
        # Save enhanced labels
        labels_path = output_dir / 'enhanced_labels.csv'
        latest_result.enhanced_labels.to_csv(labels_path)
        
        # Save comprehensive report
        if self.config.generate_reports:
            report_path = output_dir / 'enhancement_report.md'
            with open(report_path, 'w') as f:
                f.write(self.generate_comprehensive_report())
        
        # Save performance metrics
        perf_path = output_dir / 'performance_metrics.json'
        import json
        with open(perf_path, 'w') as f:
            json.dump(self.get_performance_summary(), f, indent=2)
        
        self.logger.info(f'💾 Results saved to {output_dir}')


# Convenience functions
def create_enhanced_labeler(enhancement_level: EnhancementLevel = EnhancementLevel.FULLY_OPTIMIZED,
                          custom_config: Optional[EnhancedLabelingConfig] = None) -> EnhancedMultiHorizonProfitLabeler:
    """Create enhanced labeler with specified enhancement level."""
    if custom_config is None:
        custom_config = EnhancedLabelingConfig(enhancement_level=enhancement_level)
    else:
        custom_config.enhancement_level = enhancement_level
    
    return EnhancedMultiHorizonProfitLabeler(custom_config)


def generate_fully_enhanced_labels(market_data: pd.DataFrame,
                                  config: Optional[EnhancedLabelingConfig] = None) -> EnhancedLabelingResult:
    """Generate fully enhanced labels with all components enabled."""
    labeler = EnhancedMultiHorizonProfitLabeler(config)
    return labeler.generate_enhanced_labels(market_data)


# Example usage for integration with existing multi_horizon_profit_labeler.py
def enhance_existing_labeler(existing_labeler: MultiHorizonProfitLabeler,
                           enhancement_level: EnhancementLevel = EnhancementLevel.ML_ENHANCED) -> EnhancedMultiHorizonProfitLabeler:
    """Enhance an existing multi-horizon profit labeler."""
    config = EnhancedLabelingConfig(
        enhancement_level=enhancement_level,
        base_config=existing_labeler.config
    )
    
    enhanced_labeler = EnhancedMultiHorizonProfitLabeler(config)
    return enhanced_labeler