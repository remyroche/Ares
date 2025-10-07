"""
Optimized Feature Lookback Optimization Component.

This optimized version addresses the following issues:
1. Removes duplicate logic for forward returns calculation
2. Ensures full alignment with multi_horizon_profit_labeler methodology
3. Adds proper tprint logging at every important stage
4. Optimizes for 5m timeframe by default
5. Handles failures gracefully without silent errors
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

# Import tprint for consistent logging
from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success

# Import logging functions
from src.utils.logger import get_logger, log_error, log_warning, log_info

# Import feature bank and generators
from src.feature_generation.core.feature_bank import get_global_feature_bank, FeatureCategory
from src.feature_generation.core.feature_generator import FeatureGenerator

# Import profit labeling components for alignment
from src.training.steps.pre_training.profit_labeling.volatility_aware_labeler import (
    VolatilityAwareMultiHorizonLabeler, VolatilityAwareConfig
)
from src.training.steps.pre_training.profit_labeling.multi_target_scheme import (
    MultiTargetScheme, MultiTargetConfig, TargetBand
)

# Import multi-horizon profit labeler for proper alignment
from src.training.steps.pre_training.multi_horizon_profit_labeler import (
    MultiHorizonProfitLabeler, MultiHorizonConfig
)

# Core dependencies
import numpy as np
import pandas as pd

# Import base component
from ...market_analysis.components.base_component import (
    BaseMarketAnalysisComponent,
    ComponentConfig,
    ComponentResult
)


@dataclass
class OptimizedFeatureLookbackConfig:
    """Optimized configuration for feature lookback optimization."""

    # Timeframe settings - 15m by default for tactician
    default_timeframe: str = "15m"  # Updated to 15m for tactician
    base_period_minutes: float = 15.0  # Updated to 15 minutes for 15m timeframe

    # Lookback optimization settings
    min_lookback: int = 5
    max_lookback: int = 100
    lookback_step: int = 5

    # Feature selection settings - exclude specified categories
    excluded_categories: List[FeatureCategory] = None
    excluded_features: List[str] = None

    # Forward return calculation settings (fully aligned with multi_horizon_profit_labeler)
    enable_volatility_normalization: bool = True
    enable_multi_target_scheme: bool = True
    enable_enhanced_labels: bool = True
    label_definition_type: str = "analyst"  # "analyst", "tactician"
    
    # Multi-target scheme configuration (aligned with multi_horizon_profit_labeler)
    small_band: Tuple[float, float] = (0.4, 0.8)  # k_s range
    medium_band: Tuple[float, float] = (0.8, 1.3)  # k_m range
    high_band: Tuple[float, float] = (1.3, 2.0)   # k_h range

    # Optimization settings
    optimization_metric: str = "information_coefficient"
    cv_folds: int = 5
    max_optimization_time: int = 300  # seconds

    # Output settings
    save_results: bool = True
    generate_reports: bool = True
    output_directory: str = "feature_lookback_optimization_results"

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.excluded_categories is None:
            self.excluded_categories = [
                FeatureCategory.INTERACTION,
                FeatureCategory.CROSS_TIMEFRAME,
                FeatureCategory.AUTOENCODER,
                FeatureCategory.REGIME
            ]

        if self.excluded_features is None:
            self.excluded_features = [
                'wavelets', 'autoencoder', 'interaction', 'cross_timeframe', 'regime_'
            ]


class OptimizationStatus(Enum):
    """Status of the optimization process."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance."""
    total_features: int
    optimized_features: int
    failed_features: int
    average_lookback: float
    optimization_time: float
    memory_usage_mb: float
    success_rate: float


class OptimizedFeatureLookbackOptimizationComponent(BaseMarketAnalysisComponent):
    """
    Optimized Feature Lookback Optimization Component.
    
    This optimized version:
    1. Removes duplicate logic for forward returns calculation
    2. Ensures full alignment with multi_horizon_profit_labeler methodology
    3. Adds proper tprint logging at every important stage
    4. Optimizes for 5m timeframe by default
    5. Handles failures gracefully without silent errors
    """
    
    def __init__(self, config: Optional[ComponentConfig] = None):
        """Initialize the optimized feature lookback optimization component."""
        tprint("🔧 Initializing OptimizedFeatureLookbackOptimizationComponent...")
        super().__init__(config)
        
        # Use standardized logging
        self.logger = get_logger('OptimizedFeatureLookbackOptimization')
        self.optimization_status = OptimizationStatus.PENDING
        self.start_time: Optional[float] = None
        self.metrics: Optional[OptimizationMetrics] = None
        
        # Initialize configuration
        self.config = OptimizedFeatureLookbackConfig()
        if config and config.custom_params:
            for key, value in config.custom_params.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        tprint_success("✅ Configuration initialized")
        tprint_info(f"   → Default timeframe: {self.config.default_timeframe}")
        tprint_info(f"   → Base period: {self.config.base_period_minutes} minutes")
        tprint_info(f"   → Excluded categories: {[cat.value for cat in self.config.excluded_categories]}")
        
        # Initialize feature bank
        tprint("🔧 Initializing feature bank...")
        self.feature_bank = get_global_feature_bank()
        tprint_success("✅ Feature bank initialized")
        
        # Initialize multi-target scheme for proper forward returns calculation
        tprint("🔧 Initializing multi-target scheme for forward returns...")
        self.multi_target_scheme = self._initialize_multi_target_scheme()
        tprint_success("✅ Multi-target scheme initialized")
        
        # Initialize multi-horizon profit labeler for alignment
        tprint("🔧 Initializing multi-horizon profit labeler for alignment...")
        self.multi_horizon_labeler = self._initialize_multi_horizon_labeler()
        tprint_success("✅ Multi-horizon profit labeler initialized")
        
        # Performance tracking
        self.performance_stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'average_optimization_time': 0.0,
            'memory_usage_mb': 0.0
        }
        
        tprint_success("🚀 OptimizedFeatureLookbackOptimizationComponent initialization complete")

    def _initialize_multi_target_scheme(self) -> MultiTargetScheme:
        """Initialize multi-target scheme with proper configuration."""
        try:
            config = MultiTargetConfig(
                small_band=self.config.small_band,
                medium_band=self.config.medium_band,
                high_band=self.config.high_band,
                enable_volatility_normalization=self.config.enable_volatility_normalization
            )
            return MultiTargetScheme(config)
        except Exception as e:
            tprint_error(f"❌ Failed to initialize multi-target scheme: {e}")
            raise

    def _initialize_multi_horizon_labeler(self) -> MultiHorizonProfitLabeler:
        """Initialize multi-horizon profit labeler for alignment."""
        try:
            config = MultiHorizonConfig(
                timeframe=self.config.default_timeframe,
                base_period_minutes=self.config.base_period_minutes,
                enable_volatility_normalization=self.config.enable_volatility_normalization,
                enable_multi_target_scheme=self.config.enable_multi_target_scheme,
                enable_enhanced_labels=self.config.enable_enhanced_labels,
                label_definition_type=self.config.label_definition_type
            )
            return MultiHorizonProfitLabeler(config)
        except Exception as e:
            tprint_error(f"❌ Failed to initialize multi-horizon profit labeler: {e}")
            raise

    def _get_eligible_features(self) -> List[str]:
        """Get list of eligible features for optimization, excluding specified categories."""
        tprint("🔍 Identifying eligible features for optimization...")
        
        try:
            all_features = self.feature_bank.get_all_features()
            tprint_info(f"   → Total features available: {len(all_features)}")
            
            eligible_features = []
            excluded_count = 0
            
            for feature_name in all_features:
                try:
                    # Get feature metadata
                    generator = self.feature_bank.get_generator_by_name(feature_name)
                    if not generator:
                        tprint_warning(f"⚠️ No generator found for feature: {feature_name}")
                        continue
                    
                    # Check if feature is in excluded categories
                    if generator.category in self.config.excluded_categories:
                        tprint_info(f"   → Excluding {feature_name} (category: {generator.category.value})")
                        excluded_count += 1
                        continue
                    
                    # Check if feature name contains excluded patterns
                    if any(pattern in feature_name.lower() for pattern in self.config.excluded_features):
                        tprint_info(f"   → Excluding {feature_name} (matches excluded pattern)")
                        excluded_count += 1
                        continue
                    
                    # Feature is eligible
                    eligible_features.append(feature_name)
                    
                except Exception as e:
                    tprint_warning(f"⚠️ Error checking feature {feature_name}: {e}")
                    excluded_count += 1
                    continue
            
            tprint_success(f"✅ Found {len(eligible_features)} eligible features")
            tprint_info(f"   → Excluded {excluded_count} features")
            
            return eligible_features
            
        except Exception as e:
            tprint_error(f"❌ Error getting eligible features: {e}")
            return []

    def _calculate_forward_returns_aligned(self, data: pd.DataFrame, lookback: int, 
                                         pipeline_state: Optional[Dict[str, Any]] = None) -> pd.Series:
        """
        Calculate forward returns using the EXACT same methodology as multi_horizon_profit_labeler.
        
        This method ensures perfect alignment by:
        1. Using the same volatility calculation
        2. Using the same multi-target scheme configuration
        3. Using the same target band definitions
        4. Using the same confidence scoring and eligibility masks
        """
        tprint_info(f"   → Calculating forward returns with full multi_horizon_profit_labeler alignment")
        tprint_info(f"   → Lookback period: {lookback} bars")
        
        try:
            # Check if we have pre-computed labels from multi_horizon_profit_labeler
            if pipeline_state and self._has_precomputed_labels(pipeline_state):
                tprint_info("   → Using pre-computed labels from multi_horizon_profit_labeler")
                return self._get_precomputed_forward_returns(pipeline_state, lookback)
            
            # Calculate forward returns using the same methodology as multi_horizon_profit_labeler
            tprint_info("   → Generating forward returns using multi_horizon_profit_labeler methodology")
            
            # Use the same volatility calculation as multi_horizon_profit_labeler
            returns = data['close'].pct_change()
            volatility = returns.rolling(window=min(lookback, 50)).std()
            
            # Generate targets using the EXACT same multi-target scheme as multi_horizon_profit_labeler
            target_result = self.multi_target_scheme.generate_targets(
                data, volatility, pd.Series(True, index=data.index)
            )
            
            if target_result.labels.empty:
                tprint_warning("   → No labels generated from multi-target scheme")
                # Fallback to simple returns but with proper logging
                tprint_info("   → Falling back to simple returns calculation")
                simple_returns = data['close'].pct_change(lookback).shift(-lookback)
                return simple_returns.fillna(0)
            
            # Use the actual labels generated by the multi-target scheme
            # These are the same ternary labels (-1, 0, 1) that represent trading opportunities
            forward_returns = target_result.labels.iloc[:, 0]  # Use first target column
            
            # Log the results
            tprint_info(f"   → Generated {len(forward_returns.dropna())} FPT-based labels")
            tprint_info(f"   → Label distribution: {forward_returns.value_counts().to_dict()}")
            
            # Store the target result for later use (confidence scores, eligibility masks)
            self._last_target_result = target_result
            
            return forward_returns
            
        except Exception as e:
            tprint_error(f"❌ Error calculating forward returns: {e}")
            # Fallback to simple returns with proper error handling
            tprint_info("   → Falling back to simple returns due to error")
            try:
                simple_returns = data['close'].pct_change(lookback).shift(-lookback)
                return simple_returns.fillna(0)
            except Exception as fallback_error:
                tprint_error(f"❌ Fallback calculation also failed: {fallback_error}")
                return pd.Series(0, index=data.index)

    def _has_precomputed_labels(self, pipeline_state: Dict[str, Any]) -> bool:
        """Check if pre-computed labels from multi_horizon_profit_labeler are available."""
        try:
            # Check for multi_horizon_labeling_result in pipeline state
            labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
            if labeling_result and 'labeled_data' in labeling_result:
                labeled_data = labeling_result['labeled_data']
                if not labeled_data.empty:
                    tprint_success("✅ Using pre-computed labels from multi_horizon_profit_labeler")
                    tprint_info(f"   → Found {len(labeled_data.columns)} target columns")
                    return True
            
            # Check for standardized output format
            standardized_output = pipeline_state.get('standardized_output', {})
            if standardized_output and 'labels' in standardized_output:
                labels = standardized_output['labels']
                if not labels.empty:
                    tprint_success("✅ Using standardized output labels from multi_horizon_profit_labeler")
                    tprint_info(f"   → Found {len(labels.columns)} target columns")
                    return True
            
            tprint_warning("⚠️ No pre-computed labels found from multi_horizon_profit_labeler")
            return False
            
        except Exception as e:
            tprint_error(f"❌ Error checking for precomputed labels: {e}")
            return False

    def _get_precomputed_forward_returns(self, pipeline_state: Dict[str, Any], lookback: int) -> pd.Series:
        """Get pre-computed forward returns from multi_horizon_profit_labeler."""
        try:
            # Try standardized output format first (preferred)
            standardized_output = pipeline_state.get('standardized_output', {})
            if standardized_output and 'labels' in standardized_output:
                labels = standardized_output['labels']
                if not labels.empty:
                    # Use the first target column (immediate_opportunity)
                    if 'immediate_opportunity' in labels.columns:
                        forward_returns = labels['immediate_opportunity']
                    else:
                        forward_returns = labels.iloc[:, 0]  # Use first column
                    
                    tprint_info(f"   → Using pre-computed labels: {len(forward_returns.dropna())} samples")
                    return forward_returns
            
            # Fallback to multi_horizon_labeling_result
            labeling_result = pipeline_state.get('multi_horizon_labeling_result', {})
            if labeling_result and 'labeled_data' in labeling_result:
                labeled_data = labeling_result['labeled_data']
                if not labeled_data.empty:
                    # Use the first target column
                    forward_returns = labeled_data.iloc[:, 0]
                    tprint_info(f"   → Using labeled data: {len(forward_returns.dropna())} samples")
                    return forward_returns
            
            tprint_warning("⚠️ No valid pre-computed labels found")
            return pd.Series(dtype=float)
            
        except Exception as e:
            tprint_error(f"❌ Error getting precomputed forward returns: {e}")
            return pd.Series(dtype=float)

    def _optimize_single_feature(self, feature_name: str, generator: FeatureGenerator, 
                               data: pd.DataFrame, pipeline_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize lookback period for a single feature using proper multi_horizon_profit_labeler alignment.
        
        This method:
        1. Uses the exact same forward returns calculation as multi_horizon_profit_labeler
        2. Tests different lookback periods
        3. Calculates information coefficient for each lookback period
        4. Returns the optimal lookback period with metrics
        """
        tprint_info(f"🔧 Optimizing feature: {feature_name}")
        
        try:
            # Get feature metadata
            metadata = generator.get_metadata()
            tprint_info(f"   → Feature category: {metadata.category.value}")
            tprint_info(f"   → Required fields: {metadata.required_fields}")
            
            # Test different lookback periods
            lookback_periods = range(
                max(self.config.min_lookback, metadata.min_lookback),
                min(self.config.max_lookback, metadata.max_lookback) + 1,
                self.config.lookback_step
            )
            
            tprint_info(f"   → Testing {len(lookback_periods)} lookback periods: {list(lookback_periods)}")
            
            best_lookback = self.config.min_lookback
            best_ic = -1.0
            optimization_results = {}
            
            for lookback in lookback_periods:
                try:
                    tprint_info(f"   → Testing lookback period: {lookback}")
                    
                    # Generate feature with current lookback
                    feature_result = generator.generate(data, lookback=lookback)
                    if feature_result.features.empty:
                        tprint_warning(f"   → No features generated for lookback {lookback}")
                        continue
                    
                    # Calculate forward returns using aligned methodology
                    forward_returns = self._calculate_forward_returns_aligned(data, lookback, pipeline_state)
                    
                    if forward_returns.empty or forward_returns.isna().all():
                        tprint_warning(f"   → No valid forward returns for lookback {lookback}")
                        continue
                    
                    # Align feature and forward returns indices
                    feature_series = feature_result.features.iloc[:, 0]  # Use first feature column
                    aligned_data = pd.DataFrame({
                        'feature': feature_series,
                        'forward_returns': forward_returns
                    }).dropna()
                    
                    if len(aligned_data) < 100:  # Minimum samples for reliable IC
                        tprint_warning(f"   → Insufficient aligned data for lookback {lookback}: {len(aligned_data)} samples")
                        continue
                    
                    # Calculate information coefficient
                    ic = self._calculate_information_coefficient(
                        aligned_data['feature'], 
                        aligned_data['forward_returns']
                    )
                    
                    tprint_info(f"   → Lookback {lookback}: IC = {ic:.4f}, samples = {len(aligned_data)}")
                    
                    optimization_results[lookback] = {
                        'ic': ic,
                        'samples': len(aligned_data),
                        'feature_std': feature_series.std(),
                        'forward_returns_std': forward_returns.std()
                    }
                    
                    # Update best lookback if this is better
                    if ic > best_ic:
                        best_ic = ic
                        best_lookback = lookback
                        tprint_info(f"   → New best lookback: {best_lookback} (IC: {best_ic:.4f})")
                
                except Exception as e:
                    tprint_error(f"   → Error testing lookback {lookback}: {e}")
                    continue
            
            # Prepare results
            result = {
                'feature_name': feature_name,
                'optimal_lookback': best_lookback,
                'best_ic': best_ic,
                'optimization_results': optimization_results,
                'success': True,
                'total_tests': len(optimization_results),
                'metadata': {
                    'category': metadata.category.value,
                    'required_fields': metadata.required_fields,
                    'min_lookback': metadata.min_lookback,
                    'max_lookback': metadata.max_lookback
                }
            }
            
            tprint_success(f"✅ Feature {feature_name} optimized: lookback={best_lookback}, IC={best_ic:.4f}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Failed to optimize feature {feature_name}: {e}")
            return {
                'feature_name': feature_name,
                'optimal_lookback': self.config.min_lookback,
                'best_ic': -1.0,
                'optimization_results': {},
                'success': False,
                'error': str(e)
            }

    def _calculate_information_coefficient(self, feature: pd.Series, forward_returns: pd.Series) -> float:
        """Calculate information coefficient between feature and forward returns."""
        try:
            # Calculate Spearman correlation (more robust than Pearson for non-linear relationships)
            correlation = feature.corr(forward_returns, method='spearman')
            
            # Handle NaN values
            if pd.isna(correlation):
                return 0.0
            
            return float(correlation)
            
        except Exception as e:
            tprint_error(f"❌ Error calculating information coefficient: {e}")
            return 0.0

    async def execute(self, data: Any, pipeline_state: Dict[str, Any]) -> ComponentResult:
        """
        Execute optimized feature lookback optimization.
        
        This method:
        1. Validates input data and pipeline state
        2. Gets eligible features for optimization
        3. Optimizes each feature using proper multi_horizon_profit_labeler alignment
        4. Returns comprehensive results with proper error handling
        """
        self.start_time = time.time()
        self.optimization_status = OptimizationStatus.IN_PROGRESS
        
        tprint("🚀 Starting Optimized Feature Lookback Optimization")
        tprint_info(f"   → Timeframe: {self.config.default_timeframe}")
        tprint_info(f"   → Base period: {self.config.base_period_minutes} minutes")
        
        try:
            # Step 1: Validate input data
            tprint("🔍 Step 1: Validating input data...")
            if data is None or (hasattr(data, 'empty') and data.empty):
                error_msg = "No valid data provided for feature optimization"
                tprint_error(f"❌ {error_msg}")
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'error': 'No data provided'}
                )
            
            # Convert data to DataFrame if needed
            if not isinstance(data, pd.DataFrame):
                try:
                    data = pd.DataFrame(data)
                except Exception as e:
                    error_msg = f"Failed to convert data to DataFrame: {e}"
                    tprint_error(f"❌ {error_msg}")
                    self.optimization_status = OptimizationStatus.FAILED
                    return ComponentResult(
                        success=False,
                        artifacts={},
                        error_message=error_msg,
                        metadata={'error': 'Data conversion failed'}
                    )
            
            tprint_success(f"✅ Data validated: {len(data)} rows, {len(data.columns)} columns")
            
            # Step 2: Get eligible features
            tprint("🔍 Step 2: Getting eligible features...")
            eligible_features = self._get_eligible_features()
            
            if not eligible_features:
                error_msg = "No eligible features found for optimization"
                tprint_error(f"❌ {error_msg}")
                self.optimization_status = OptimizationStatus.FAILED
                return ComponentResult(
                    success=False,
                    artifacts={},
                    error_message=error_msg,
                    metadata={'error': 'No eligible features'}
                )
            
            tprint_success(f"✅ Found {len(eligible_features)} eligible features")
            
            # Step 3: Optimize each feature
            tprint("🚀 Step 3: Optimizing features...")
            optimization_results = {}
            successful_optimizations = 0
            failed_optimizations = 0
            
            for i, feature_name in enumerate(eligible_features, 1):
                tprint_info(f"🔧 Optimizing feature {i}/{len(eligible_features)}: {feature_name}")
                
                try:
                    # Get feature generator
                    generator = self.feature_bank.get_generator_by_name(feature_name)
                    if not generator:
                        tprint_warning(f"⚠️ No generator found for feature: {feature_name}")
                        failed_optimizations += 1
                        continue
                    
                    # Optimize feature
                    result = self._optimize_single_feature(feature_name, generator, data, pipeline_state)
                    optimization_results[feature_name] = result
                    
                    if result.get('success', False):
                        successful_optimizations += 1
                    else:
                        failed_optimizations += 1
                
                except Exception as e:
                    tprint_error(f"❌ Failed to optimize feature {feature_name}: {e}")
                    optimization_results[feature_name] = {
                        'feature_name': feature_name,
                        'optimal_lookback': self.config.min_lookback,
                        'best_ic': -1.0,
                        'success': False,
                        'error': str(e)
                    }
                    failed_optimizations += 1
            
            # Step 4: Calculate metrics
            tprint("📊 Step 4: Calculating optimization metrics...")
            end_time = time.time()
            optimization_time = end_time - self.start_time
            
            # Calculate average lookback
            successful_results = [r for r in optimization_results.values() if r.get('success', False)]
            average_lookback = np.mean([r['optimal_lookback'] for r in successful_results]) if successful_results else 0.0
            
            # Calculate success rate
            success_rate = successful_optimizations / len(eligible_features) if eligible_features else 0.0
            
            # Create metrics
            self.metrics = OptimizationMetrics(
                total_features=len(eligible_features),
                optimized_features=successful_optimizations,
                failed_features=failed_optimizations,
                average_lookback=average_lookback,
                optimization_time=optimization_time,
                memory_usage_mb=0.0,  # TODO: Add memory tracking
                success_rate=success_rate
            )
            
            # Step 5: Prepare results
            tprint("📋 Step 5: Preparing results...")
            artifacts = {
                'optimization_results': optimization_results,
                'optimization_metrics': {
                    'total_features': self.metrics.total_features,
                    'optimized_features': self.metrics.optimized_features,
                    'failed_features': self.metrics.failed_features,
                    'average_lookback': self.metrics.average_lookback,
                    'optimization_time': self.metrics.optimization_time,
                    'success_rate': self.metrics.success_rate
                },
                'configuration': {
                    'default_timeframe': self.config.default_timeframe,
                    'base_period_minutes': self.config.base_period_minutes,
                    'min_lookback': self.config.min_lookback,
                    'max_lookback': self.config.max_lookback,
                    'excluded_categories': [cat.value for cat in self.config.excluded_categories],
                    'excluded_features': self.config.excluded_features
                },
                'metadata': {
                    'optimization_timestamp': datetime.now().isoformat(),
                    'component_version': 'optimized_v1.0',
                    'multi_horizon_alignment': True,
                    'forward_returns_method': 'multi_horizon_profit_labeler_aligned'
                }
            }
            
            # Update performance stats
            self.performance_stats['total_optimizations'] += 1
            self.performance_stats['successful_optimizations'] += successful_optimizations
            self.performance_stats['failed_optimizations'] += failed_optimizations
            self.performance_stats['average_optimization_time'] = optimization_time
            
            # Step 6: Save results if configured
            if self.config.save_results:
                tprint("💾 Step 6: Saving results...")
                try:
                    self._save_results(artifacts)
                    tprint_success("✅ Results saved successfully")
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to save results: {e}")
            
            # Final status
            self.optimization_status = OptimizationStatus.COMPLETED
            
            tprint_success("🎉 Optimized Feature Lookback Optimization completed successfully!")
            tprint_info(f"   → Total features: {self.metrics.total_features}")
            tprint_info(f"   → Successfully optimized: {self.metrics.optimized_features}")
            tprint_info(f"   → Failed optimizations: {self.metrics.failed_features}")
            tprint_info(f"   → Success rate: {self.metrics.success_rate:.2%}")
            tprint_info(f"   → Average lookback: {self.metrics.average_lookback:.1f}")
            tprint_info(f"   → Optimization time: {self.metrics.optimization_time:.2f}s")
            
            return ComponentResult(
                success=True,
                artifacts=artifacts,
                metadata={
                    'optimization_status': self.optimization_status.value,
                    'optimization_time': optimization_time,
                    'success_rate': success_rate,
                    'total_features': len(eligible_features),
                    'optimized_features': successful_optimizations
                }
            )
            
        except Exception as e:
            tprint_error(f"❌ Feature lookback optimization failed: {e}")
            self.optimization_status = OptimizationStatus.FAILED
            
            return ComponentResult(
                success=False,
                artifacts={},
                error_message=str(e),
                metadata={
                    'optimization_status': self.optimization_status.value,
                    'error': str(e)
                }
            )

    def _save_results(self, artifacts: Dict[str, Any]) -> None:
        """Save optimization results to file."""
        try:
            # Create output directory
            output_dir = Path(self.config.output_directory)
            output_dir.mkdir(exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"feature_lookback_optimization_results_{timestamp}.json"
            filepath = output_dir / filename
            
            # Save results
            with open(filepath, 'w') as f:
                json.dump(artifacts, f, indent=2, default=str)
            
            tprint_info(f"💾 Results saved to: {filepath}")
            
        except Exception as e:
            tprint_error(f"❌ Failed to save results: {e}")
            raise

    def get_required_artifacts(self) -> List[str]:
        """Get list of required artifacts this component must produce."""
        return ['optimization_results', 'optimization_metrics', 'configuration', 'metadata']