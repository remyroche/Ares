"""
Directional Feature Lookback Optimization

This module implements directional feature lookback optimization that generates
1 optimized lookback period per feature per direction (long/short) instead of 
2 periods per feature, maintaining the 60-100 feature limit for ML models.

Key Changes:
- Single lookback period per feature per direction
- Directional data splitting based on target signals
- Intelligent feature selection to manage total feature count
- Integration with existing MRMR optimization framework
"""

import asyncio
import logging
import time
import json
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

# Import existing optimization components
from .mrmr_lookback_optimizer import (
    MRMRLookbackOptimizer, LookbackOptimizationConfig, LookbackOptimizationResult
)

# Import tprint for consistent logging
from src.utils.tprint import tprint

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class DirectionalLookbackConfig:
    """Configuration for directional lookbook optimization."""
    
    # Base optimization settings
    min_lookback: int = 5
    max_lookback: int = 100
    optimization_method: str = "bayesian"
    
    # Directional settings
    enable_directional: bool = True
    min_samples_per_direction: int = 50
    direction_balance_threshold: float = 0.3  # Min ratio of smaller/larger direction
    
    # Period consolidation settings
    enable_period_consolidation: bool = True
    consolidation_variance_threshold: float = 0.20  # If variance < 20%, consolidate periods
    consolidation_method: str = "average"  # "average", "best_performance", "weighted_average"
    
    # Integration with existing pipeline
    use_existing_feature_pipeline: bool = True  # Use existing 100→80→60 pipeline
    generate_features_for_pipeline: bool = True  # Generate features for the pipeline to select from
    
    # Quality thresholds
    min_mutual_info_score: float = 0.01
    max_correlation_threshold: float = 0.95
    min_sample_size: int = 100
    
    # Optimization performance
    parallel_optimization: bool = True
    max_workers: int = 4
    timeout_per_feature: int = 300  # 5 minutes per feature
    
    # Advanced settings
    cross_directional_analysis: bool = True  # Compare long vs short performance
    adaptive_feature_selection: bool = True  # Adapt selection based on performance
    save_intermediate_results: bool = True

@dataclass
class DirectionalFeatureResult:
    """Result for a single feature's directional optimization."""
    
    feature_name: str
    direction: str  # "long", "short", or "consolidated"
    
    # Optimization results
    optimal_lookback_period: int
    mutual_info_score: float
    optimization_method: str
    optimization_time: float
    
    # Consolidation information
    is_consolidated: bool = False
    original_long_period: Optional[int] = None
    original_short_period: Optional[int] = None
    consolidation_variance: Optional[float] = None
    consolidation_reason: Optional[str] = None
    
    # Quality metrics
    sample_count: int
    data_quality_score: float
    convergence_achieved: bool
    
    # Performance metrics
    cross_validation_score: float
    stability_score: float
    
    # Metadata
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'feature_name': self.feature_name,
            'direction': self.direction,
            'optimal_lookback_period': self.optimal_lookback_period,
            'mutual_info_score': self.mutual_info_score,
            'optimization_method': self.optimization_method,
            'optimization_time': self.optimization_time,
            'sample_count': self.sample_count,
            'data_quality_score': self.data_quality_score,
            'convergence_achieved': self.convergence_achieved,
            'cross_validation_score': self.cross_validation_score,
            'stability_score': self.stability_score,
            'optimization_metadata': self.optimization_metadata
        }

@dataclass
class DirectionalOptimizationResult:
    """Complete result of directional lookback optimization."""
    
    # Optimization results by direction
    long_features: Dict[str, DirectionalFeatureResult] = field(default_factory=dict)
    short_features: Dict[str, DirectionalFeatureResult] = field(default_factory=dict)
    consolidated_features: Dict[str, DirectionalFeatureResult] = field(default_factory=dict)  # Features with <20% variance
    
    # Selected features (after feature selection)
    selected_long_features: List[str] = field(default_factory=list)
    selected_short_features: List[str] = field(default_factory=list)
    final_feature_count: int = 0
    
    # Performance metrics
    total_optimization_time: float = 0.0
    average_mutual_info_score: float = 0.0
    directional_balance_ratio: float = 0.0
    
    # Quality metrics
    convergence_rate: float = 0.0
    average_stability_score: float = 0.0
    feature_selection_quality: float = 0.0
    
    # Cross-directional analysis
    directional_differences: Dict[str, Dict[str, float]] = field(default_factory=dict)
    complementary_features: List[Tuple[str, str]] = field(default_factory=list)
    
    # Metadata
    config_used: Optional[DirectionalLookbackConfig] = None
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_all_selected_features(self) -> Dict[str, DirectionalFeatureResult]:
        """Get all selected features from both directions and consolidated."""
        all_features = {}
        
        for feature_name in self.selected_long_features:
            if feature_name in self.long_features:
                all_features[f"{feature_name}_long"] = self.long_features[feature_name]
        
        for feature_name in self.selected_short_features:
            if feature_name in self.short_features:
                all_features[f"{feature_name}_short"] = self.short_features[feature_name]
        
        # Add consolidated features
        for feature_name, feature_result in self.consolidated_features.items():
            all_features[f"{feature_name}_consolidated"] = feature_result
        
        return all_features
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'long_features': {k: v.to_dict() for k, v in self.long_features.items()},
            'short_features': {k: v.to_dict() for k, v in self.short_features.items()},
            'consolidated_features': {k: v.to_dict() for k, v in self.consolidated_features.items()},
            'selected_long_features': self.selected_long_features,
            'selected_short_features': self.selected_short_features,
            'final_feature_count': self.final_feature_count,
            'total_optimization_time': self.total_optimization_time,
            'average_mutual_info_score': self.average_mutual_info_score,
            'directional_balance_ratio': self.directional_balance_ratio,
            'convergence_rate': self.convergence_rate,
            'average_stability_score': self.average_stability_score,
            'feature_selection_quality': self.feature_selection_quality,
            'directional_differences': self.directional_differences,
            'complementary_features': self.complementary_features,
            'optimization_metadata': self.optimization_metadata,
            'consolidation_summary': {
                'consolidated_count': len(self.consolidated_features),
                'consolidation_enabled': True,
                'consolidation_method': getattr(self.config_used, 'consolidation_method', 'average') if self.config_used else 'unknown'
            }
        }

class DirectionalLookbackOptimizer:
    """
    Directional Feature Lookback Optimizer
    
    Optimizes lookback periods for features with directional differentiation,
    generating 1 period per feature per direction instead of 2 periods per feature.
    """
    
    def __init__(self, config: Optional[DirectionalLookbackConfig] = None):
        """Initialize the directional optimizer."""
        self.config = config or DirectionalLookbackConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize MRMR optimizer for individual feature optimization
        mrmr_config = LookbackOptimizationConfig(
            min_lookback=self.config.min_lookback,
            max_lookback=self.config.max_lookback,
            optimization_method="two_step_grid_tpe"  # Use existing method but adapt results
        )
        self.mrmr_optimizer = MRMRLookbackOptimizer(config=mrmr_config)
        
        # Track optimization state
        self.optimization_history: List[DirectionalOptimizationResult] = []
        self.feature_cache: Dict[str, DirectionalFeatureResult] = {}
        
        tprint("🎯 DirectionalLookbackOptimizer initialized")
    
    def optimize_features_directional(self,
                                    data: pd.DataFrame,
                                    feature_columns: List[str],
                                    target_column: str = 'returns',
                                    **kwargs) -> DirectionalOptimizationResult:
        """
        Optimize features with directional differentiation.
        
        Args:
            data: Input data with features and targets
            feature_columns: List of feature columns to optimize
            target_column: Target column for optimization
            **kwargs: Additional optimization parameters
            
        Returns:
            DirectionalOptimizationResult with optimized features
        """
        tprint("🚀 Starting directional feature lookback optimization...")
        start_time = time.time()
        
        # Split data by direction
        long_data, short_data = self._split_data_by_direction(data, target_column)
        
        # Validate data splits
        if not self._validate_directional_data(long_data, short_data):
            raise ValueError("Insufficient or imbalanced directional data")
        
        # Initialize result
        result = DirectionalOptimizationResult(config_used=self.config)
        
        # Optimize features for each direction
        tprint("📊 Optimizing features for LONG signals...")
        result.long_features = self._optimize_direction_features(
            long_data, feature_columns, target_column, "long"
        )
        
        tprint("📊 Optimizing features for SHORT signals...")
        result.short_features = self._optimize_direction_features(
            short_data, feature_columns, target_column, "short"
        )
        
        # Perform cross-directional analysis
        if self.config.cross_directional_analysis:
            tprint("🔄 Analyzing cross-directional differences...")
            result.directional_differences = self._analyze_directional_differences(
                result.long_features, result.short_features
            )
            result.complementary_features = self._find_complementary_features(
                result.long_features, result.short_features
            )
        
        # Perform period consolidation if enabled
        if self.config.enable_period_consolidation:
            tprint("🔀 Consolidating similar periods...")
            result = self._consolidate_similar_periods(result)
        
        # Select final features
        tprint("🎯 Selecting optimal feature subset...")
        result = self._select_final_features(result)
        
        # Calculate final metrics
        result.total_optimization_time = time.time() - start_time
        result = self._calculate_final_metrics(result)
        
        # Store result in history
        self.optimization_history.append(result)
        
        tprint(f"✅ Directional optimization completed in {result.total_optimization_time:.2f}s")
        tprint(f"📈 Final features: {result.final_feature_count} "
               f"({len(result.selected_long_features)} long + {len(result.selected_short_features)} short)")
        
        return result
    
    def _split_data_by_direction(self, 
                               data: pd.DataFrame, 
                               target_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into long and short direction subsets."""
        try:
            # Determine direction based on target values
            if target_column in data.columns:
                target_values = data[target_column]
            else:
                # Fallback to returns-based splitting
                target_values = data.get('returns', data.get('close_return', data.get('target', 0)))
            
            # Create direction masks
            long_mask = target_values > 0
            short_mask = target_values < 0
            
            long_data = data[long_mask].copy()
            short_data = data[short_mask].copy()
            
            tprint(f"📊 Data split: {len(long_data)} long samples, {len(short_data)} short samples")
            
            return long_data, short_data
            
        except Exception as e:
            tprint(f"❌ Error splitting data by direction: {e}")
            # Return equal splits as fallback
            mid_point = len(data) // 2
            return data.iloc[:mid_point].copy(), data.iloc[mid_point:].copy()
    
    def _validate_directional_data(self, 
                                 long_data: pd.DataFrame, 
                                 short_data: pd.DataFrame) -> bool:
        """Validate that directional data splits are sufficient for optimization."""
        
        # Check minimum sample sizes
        if len(long_data) < self.config.min_samples_per_direction:
            tprint(f"❌ Insufficient long samples: {len(long_data)} < {self.config.min_samples_per_direction}")
            return False
        
        if len(short_data) < self.config.min_samples_per_direction:
            tprint(f"❌ Insufficient short samples: {len(short_data)} < {self.config.min_samples_per_direction}")
            return False
        
        # Check balance ratio
        smaller_count = min(len(long_data), len(short_data))
        larger_count = max(len(long_data), len(short_data))
        balance_ratio = smaller_count / larger_count
        
        if balance_ratio < self.config.direction_balance_threshold:
            tprint(f"⚠️ Imbalanced directional data: ratio = {balance_ratio:.3f}")
            # Continue but warn about potential issues
        
        tprint(f"✅ Directional data validation passed (balance ratio: {balance_ratio:.3f})")
        return True
    
    def _optimize_direction_features(self,
                                   data: pd.DataFrame,
                                   feature_columns: List[str],
                                   target_column: str,
                                   direction: str) -> Dict[str, DirectionalFeatureResult]:
        """Optimize features for a specific direction."""
        direction_results = {}
        
        for i, feature_name in enumerate(feature_columns):
            tprint(f"🔧 Optimizing {feature_name} for {direction} ({i+1}/{len(feature_columns)})...")
            
            try:
                # Check cache first
                cache_key = f"{feature_name}_{direction}_{len(data)}"
                if cache_key in self.feature_cache:
                    direction_results[feature_name] = self.feature_cache[cache_key]
                    continue
                
                # Run single-period optimization (modify MRMR result)
                optimization_start = time.time()
                
                # Use MRMR optimizer but only take the first (best) lookback period
                mrmr_result = self.mrmr_optimizer.optimize_lookback_periods(
                    data=data,
                    feature_name=feature_name,
                    target_column=target_column,
                    parameter_type="technical_indicator"
                )
                
                # Create directional result using only the first (best) lookback period
                directional_result = DirectionalFeatureResult(
                    feature_name=feature_name,
                    direction=direction,
                    optimal_lookback_period=mrmr_result.first_lookback_period,
                    mutual_info_score=mrmr_result.first_mi_score,
                    optimization_method="directional_mrmr",
                    optimization_time=time.time() - optimization_start,
                    sample_count=len(data),
                    data_quality_score=self._calculate_data_quality_score(data, feature_name),
                    convergence_achieved=mrmr_result.convergence_rate > 0.8,
                    cross_validation_score=mrmr_result.cross_validation_score,
                    stability_score=mrmr_result.stability_score,
                    optimization_metadata={
                        'original_mrmr_result': mrmr_result.to_dict(),
                        'direction': direction,
                        'feature_generation_method': 'single_period_directional'
                    }
                )
                
                direction_results[feature_name] = directional_result
                
                # Cache result
                self.feature_cache[cache_key] = directional_result
                
                tprint(f"✅ {feature_name} ({direction}): "
                      f"lookback={directional_result.optimal_lookback_period}, "
                      f"MI={directional_result.mutual_info_score:.4f}")
                
            except Exception as e:
                tprint(f"❌ Failed to optimize {feature_name} for {direction}: {e}")
                
                # Create error result
                direction_results[feature_name] = DirectionalFeatureResult(
                    feature_name=feature_name,
                    direction=direction,
                    optimal_lookback_period=20,  # Default fallback
                    mutual_info_score=0.0,
                    optimization_method="error_fallback",
                    optimization_time=0.0,
                    sample_count=len(data),
                    data_quality_score=0.0,
                    convergence_achieved=False,
                    cross_validation_score=0.0,
                    stability_score=0.0,
                    optimization_metadata={'error': str(e)}
                )
        
        return direction_results
    
    def _calculate_data_quality_score(self, 
                                    data: pd.DataFrame, 
                                    feature_name: str) -> float:
        """Calculate data quality score for a feature."""
        try:
            if feature_name in data.columns:
                feature_data = data[feature_name]
            else:
                # Use close price as fallback
                feature_data = data.get('close', pd.Series([0]))
            
            # Calculate quality metrics
            null_ratio = feature_data.isnull().sum() / len(feature_data)
            inf_ratio = np.isinf(feature_data).sum() / len(feature_data)
            zero_ratio = (feature_data == 0).sum() / len(feature_data)
            
            # Variance check
            variance = feature_data.var()
            variance_score = min(1.0, variance / (variance + 0.01)) if variance > 0 else 0.0
            
            # Combined quality score
            quality_score = (1 - null_ratio) * (1 - inf_ratio) * (1 - zero_ratio) * variance_score
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception:
            return 0.5  # Neutral score on error
    
    def _analyze_directional_differences(self,
                                       long_features: Dict[str, DirectionalFeatureResult],
                                       short_features: Dict[str, DirectionalFeatureResult]) -> Dict[str, Dict[str, float]]:
        """Analyze differences between long and short feature optimization results."""
        differences = {}
        
        # Find common features
        common_features = set(long_features.keys()) & set(short_features.keys())
        
        for feature_name in common_features:
            long_result = long_features[feature_name]
            short_result = short_features[feature_name]
            
            differences[feature_name] = {
                'lookback_difference': abs(long_result.optimal_lookback_period - short_result.optimal_lookback_period),
                'mi_score_difference': abs(long_result.mutual_info_score - short_result.mutual_info_score),
                'quality_difference': abs(long_result.data_quality_score - short_result.data_quality_score),
                'stability_difference': abs(long_result.stability_score - short_result.stability_score),
                'long_mi_score': long_result.mutual_info_score,
                'short_mi_score': short_result.mutual_info_score,
                'directional_preference': 'long' if long_result.mutual_info_score > short_result.mutual_info_score else 'short'
            }
        
        return differences
    
    def _find_complementary_features(self,
                                   long_features: Dict[str, DirectionalFeatureResult],
                                   short_features: Dict[str, DirectionalFeatureResult]) -> List[Tuple[str, str]]:
        """Find features that are complementary across directions."""
        complementary = []
        
        # Find features where one direction significantly outperforms the other
        for feature_name in set(long_features.keys()) & set(short_features.keys()):
            long_score = long_features[feature_name].mutual_info_score
            short_score = short_features[feature_name].mutual_info_score
            
            # Check if there's a significant difference (> 50% better)
            if long_score > short_score * 1.5:
                complementary.append((feature_name, 'long_dominant'))
            elif short_score > long_score * 1.5:
                complementary.append((feature_name, 'short_dominant'))
        
        return complementary
    
    def _consolidate_similar_periods(self, result: DirectionalOptimizationResult) -> DirectionalOptimizationResult:
        """Consolidate long/short features with similar periods (< 20% variance)."""
        
        # Find common features between long and short
        common_features = set(result.long_features.keys()) & set(result.short_features.keys())
        
        consolidated_count = 0
        
        for feature_name in common_features:
            long_result = result.long_features[feature_name]
            short_result = result.short_features[feature_name]
            
            # Calculate variance between periods
            long_period = long_result.optimal_lookback_period
            short_period = short_result.optimal_lookback_period
            
            # Calculate relative variance
            avg_period = (long_period + short_period) / 2
            if avg_period == 0:
                continue
                
            variance = abs(long_period - short_period) / avg_period
            
            # Check if variance is below threshold
            if variance < self.config.consolidation_variance_threshold:
                tprint(f"🔀 Consolidating {feature_name}: long={long_period}, short={short_period}, variance={variance:.3f}")
                
                # Create consolidated feature
                consolidated_result = self._create_consolidated_feature(
                    feature_name, long_result, short_result, variance
                )
                
                # Add to consolidated features
                result.consolidated_features[feature_name] = consolidated_result
                
                # Remove from individual directions (they'll be replaced by consolidated)
                if feature_name in result.long_features:
                    del result.long_features[feature_name]
                if feature_name in result.short_features:
                    del result.short_features[feature_name]
                
                consolidated_count += 1
        
        tprint(f"✅ Consolidated {consolidated_count} features with similar periods")
        
        return result
    
    def _create_consolidated_feature(self,
                                   feature_name: str,
                                   long_result: DirectionalFeatureResult,
                                   short_result: DirectionalFeatureResult,
                                   variance: float) -> DirectionalFeatureResult:
        """Create a consolidated feature from long and short variants."""
        
        # Calculate consolidated values based on method
        if self.config.consolidation_method == "average":
            consolidated_period = int((long_result.optimal_lookback_period + short_result.optimal_lookback_period) / 2)
            consolidated_score = (long_result.mutual_info_score + short_result.mutual_info_score) / 2
            consolidation_reason = "averaged_periods"
            
        elif self.config.consolidation_method == "best_performance":
            if long_result.mutual_info_score >= short_result.mutual_info_score:
                consolidated_period = long_result.optimal_lookback_period
                consolidated_score = long_result.mutual_info_score
                consolidation_reason = "best_performance_long"
            else:
                consolidated_period = short_result.optimal_lookback_period
                consolidated_score = short_result.mutual_info_score
                consolidation_reason = "best_performance_short"
                
        elif self.config.consolidation_method == "weighted_average":
            # Weight by mutual information scores
            total_score = long_result.mutual_info_score + short_result.mutual_info_score
            if total_score > 0:
                long_weight = long_result.mutual_info_score / total_score
                short_weight = short_result.mutual_info_score / total_score
                
                consolidated_period = int(
                    long_result.optimal_lookback_period * long_weight +
                    short_result.optimal_lookback_period * short_weight
                )
                consolidated_score = (long_result.mutual_info_score + short_result.mutual_info_score) / 2
                consolidation_reason = "weighted_by_performance"
            else:
                # Fallback to simple average
                consolidated_period = int((long_result.optimal_lookback_period + short_result.optimal_lookback_period) / 2)
                consolidated_score = 0.0
                consolidation_reason = "fallback_average"
        else:
            # Default to average
            consolidated_period = int((long_result.optimal_lookback_period + short_result.optimal_lookback_period) / 2)
            consolidated_score = (long_result.mutual_info_score + short_result.mutual_info_score) / 2
            consolidation_reason = "default_average"
        
        # Create consolidated result
        consolidated_result = DirectionalFeatureResult(
            feature_name=feature_name,
            direction="consolidated",
            optimal_lookback_period=consolidated_period,
            mutual_info_score=consolidated_score,
            optimization_method="consolidated_directional",
            optimization_time=long_result.optimization_time + short_result.optimization_time,
            
            # Consolidation info
            is_consolidated=True,
            original_long_period=long_result.optimal_lookback_period,
            original_short_period=short_result.optimal_lookback_period,
            consolidation_variance=variance,
            consolidation_reason=consolidation_reason,
            
            # Quality metrics (average of both)
            sample_count=long_result.sample_count + short_result.sample_count,
            data_quality_score=(long_result.data_quality_score + short_result.data_quality_score) / 2,
            convergence_achieved=long_result.convergence_achieved and short_result.convergence_achieved,
            cross_validation_score=(long_result.cross_validation_score + short_result.cross_validation_score) / 2,
            stability_score=(long_result.stability_score + short_result.stability_score) / 2,
            
            # Metadata
            optimization_metadata={
                'consolidation_method': self.config.consolidation_method,
                'variance_threshold': self.config.consolidation_variance_threshold,
                'actual_variance': variance,
                'long_original': long_result.to_dict(),
                'short_original': short_result.to_dict()
            }
        )
        
        return consolidated_result
    
    def _select_final_features(self, result: DirectionalOptimizationResult) -> DirectionalOptimizationResult:
        """Select final features for the existing 100→80→60 pipeline."""
        
        if self.config.use_existing_feature_pipeline:
            # Generate features for the existing pipeline to select from
            # Don't limit here - let the existing pipeline handle the selection
            tprint("🎯 Generating features for existing 100→80→60 pipeline...")
            
            # Include all optimized features (long, short, consolidated)
            result.selected_long_features = list(result.long_features.keys())
            result.selected_short_features = list(result.short_features.keys())
            
            # Add consolidated features to a separate list for tracking
            consolidated_features = list(result.consolidated_features.keys())
            
            total_generated = (len(result.selected_long_features) + 
                             len(result.selected_short_features) + 
                             len(consolidated_features))
            
            result.final_feature_count = total_generated
            
            tprint(f"🎯 Generated features for pipeline: {len(result.selected_long_features)} long + "
                   f"{len(result.selected_short_features)} short + {len(consolidated_features)} consolidated = "
                   f"{total_generated} total")
            tprint("📋 Existing pipeline will select 100→80→60 features from these candidates")
            
        else:
            # Legacy selection logic (if not using existing pipeline)
            target_per_direction = self.config.target_total_features // 2
            max_per_direction = min(self.config.max_features_per_direction, target_per_direction)
            
            # Select long features
            long_candidates = [(name, res.mutual_info_score) for name, res in result.long_features.items()]
            long_candidates.sort(key=lambda x: x[1], reverse=True)
            result.selected_long_features = [name for name, _ in long_candidates[:max_per_direction]]
            
            # Select short features
            short_candidates = [(name, res.mutual_info_score) for name, res in result.short_features.items()]
            short_candidates.sort(key=lambda x: x[1], reverse=True)
            result.selected_short_features = [name for name, _ in short_candidates[:max_per_direction]]
            
            # Update final count
            result.final_feature_count = len(result.selected_long_features) + len(result.selected_short_features) + len(result.consolidated_features)
            
            tprint(f"🎯 Feature selection completed: {len(result.selected_long_features)} long + "
                   f"{len(result.selected_short_features)} short + {len(result.consolidated_features)} consolidated = "
                   f"{result.final_feature_count} total")
        
        return result
    
    def _calculate_final_metrics(self, result: DirectionalOptimizationResult) -> DirectionalOptimizationResult:
        """Calculate final performance metrics."""
        
        all_features = (list(result.long_features.values()) + 
                       list(result.short_features.values()) + 
                       list(result.consolidated_features.values()))
        
        if all_features:
            # Average mutual information score
            result.average_mutual_info_score = np.mean([f.mutual_info_score for f in all_features])
            
            # Convergence rate
            converged_count = sum(1 for f in all_features if f.convergence_achieved)
            result.convergence_rate = converged_count / len(all_features)
            
            # Average stability score
            result.average_stability_score = np.mean([f.stability_score for f in all_features])
            
            # Directional balance (considering consolidated as neutral)
            long_count = len(result.selected_long_features)
            short_count = len(result.selected_short_features)
            consolidated_count = len(result.consolidated_features)
            
            if long_count + short_count > 0:
                result.directional_balance_ratio = min(long_count, short_count) / max(long_count, short_count)
            elif consolidated_count > 0:
                # All features are consolidated, perfect balance
                result.directional_balance_ratio = 1.0
            
            # Feature selection quality (based on selected vs total)
            total_available = len(result.long_features) + len(result.short_features) + len(result.consolidated_features)
            if total_available > 0:
                result.feature_selection_quality = result.final_feature_count / total_available
        
        return result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of all optimization runs."""
        if not self.optimization_history:
            return {"message": "No optimization runs completed yet"}
        
        latest_result = self.optimization_history[-1]
        
        return {
            "total_runs": len(self.optimization_history),
            "latest_run": {
                "final_feature_count": latest_result.final_feature_count,
                "long_features": len(latest_result.selected_long_features),
                "short_features": len(latest_result.selected_short_features),
                "optimization_time": latest_result.total_optimization_time,
                "average_mi_score": latest_result.average_mutual_info_score,
                "convergence_rate": latest_result.convergence_rate,
                "directional_balance": latest_result.directional_balance_ratio
            },
            "config": {
                "target_total_features": self.config.target_total_features,
                "max_features_per_direction": self.config.max_features_per_direction,
                "min_lookback": self.config.min_lookback,
                "max_lookback": self.config.max_lookback
            }
        }

# Convenience function for easy integration
def optimize_features_directional(data: pd.DataFrame,
                                feature_columns: List[str],
                                target_column: str = 'returns',
                                config: Optional[DirectionalLookbackConfig] = None,
                                **kwargs) -> DirectionalOptimizationResult:
    """
    Convenience function to optimize features with directional differentiation.
    
    Args:
        data: Input data with features and targets
        feature_columns: List of feature columns to optimize
        target_column: Target column for optimization
        config: Optional configuration
        **kwargs: Additional parameters
        
    Returns:
        DirectionalOptimizationResult with optimized features
    """
    optimizer = DirectionalLookbackOptimizer(config=config)
    return optimizer.optimize_features_directional(
        data=data,
        feature_columns=feature_columns,
        target_column=target_column,
        **kwargs
    )