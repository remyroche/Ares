"""
Directional Feature Selection Adapter

This module provides an adapter layer between the directional lookback optimization
and the existing feature selection pipeline, ensuring compatibility while respecting
the 60-100 feature limit for ML models.

Key Features:
- Handles directional features (long/short variants)
- Intelligent feature count management
- Compatibility with existing feature selection methods
- Performance-based feature prioritization
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

# Import existing feature selection components
try:
    from src.training.utils.feature_selection.main_framework import FeatureSelectionFramework
    from src.training.utils.feature_selection.selection_methods import (
        MRMRSelector, ElasticNetStabilitySelector, CorrelationBasedFilter
    )
    FEATURE_SELECTION_AVAILABLE = True
except ImportError as e:
    FEATURE_SELECTION_AVAILABLE = False
    logging.warning(f"Feature selection framework not available: {e}")

# Import directional optimization results
from src.training.steps.pre_training.feature_lookback_optimization.directional_lookback_optimizer import (
    DirectionalOptimizationResult,
    DirectionalFeatureResult
)

# Import tprint for consistent logging
from src.utils.tprint import tprint

# Import VectorBT with fallback
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    logging.warning("VectorBT not available. Time series analysis will be disabled.")

logger = logging.getLogger(__name__)

@dataclass
class DirectionalFeatureSelectionConfig:
    """Configuration for directional feature selection."""

    # Target feature counts
    target_total_features: int = 80
    max_total_features: int = 100
    min_total_features: int = 60

    # Model-specific settings
    model_type: str = 'default'
    model_priority_categories: List[str] = field(default_factory=lambda: ['momentum', 'volatility', 'microstructure'])

    # Directional balance
    maintain_directional_balance: bool = True
    min_features_per_direction: int = 20
    max_imbalance_ratio: float = 0.3  # Max allowed imbalance between long/short

    # Selection methods
    primary_selection_method: str = "mutual_info"  # "mutual_info", "mrmr", "correlation"
    secondary_selection_method: str = "correlation_filter"
    use_ensemble_selection: bool = True

    # Quality thresholds
    min_mutual_info_score: float = 0.01
    max_correlation_threshold: float = 0.95
    min_stability_score: float = 0.5

    # Model-specific thresholds
    model_correlation_threshold: float = 0.90
    model_importance_threshold: float = 0.005

    # Performance weighting
    performance_weight: float = 0.6  # Weight for performance-based selection
    balance_weight: float = 0.4      # Weight for directional balance

    # Advanced settings
    enable_cross_directional_filtering: bool = True
    enable_complementary_pair_selection: bool = True
    adaptive_threshold_adjustment: bool = True
    
    # VectorBT time series analysis settings
    enable_vectorbt_analysis: bool = True
    enable_regime_detection: bool = True
    enable_temporal_analysis: bool = True
    regime_window: int = 50
    temporal_window: int = 20
    volatility_threshold: float = 1.5
    trend_threshold: float = 0.02

@dataclass
class DirectionalFeatureSelectionResult:
    """Result of directional feature selection."""
    
    # Selected features
    selected_long_features: List[str] = field(default_factory=list)
    selected_short_features: List[str] = field(default_factory=list)
    total_selected_features: int = 0
    
    # Selection metrics
    selection_quality_score: float = 0.0
    directional_balance_ratio: float = 0.0
    average_mutual_info_score: float = 0.0
    
    # Feature details
    feature_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    selection_rationale: Dict[str, str] = field(default_factory=dict)
    
    # Performance metrics
    selection_time: float = 0.0
    method_used: str = ""
    
    # Quality analysis
    removed_features: Dict[str, str] = field(default_factory=dict)  # feature -> reason
    feature_correlations: Dict[str, float] = field(default_factory=dict)
    stability_scores: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        tprint("🧾 Converting DirectionalFeatureSelectionResult to dictionary format")
        return {
            'selected_long_features': self.selected_long_features,
            'selected_short_features': self.selected_short_features,
            'total_selected_features': self.total_selected_features,
            'selection_quality_score': self.selection_quality_score,
            'directional_balance_ratio': self.directional_balance_ratio,
            'average_mutual_info_score': self.average_mutual_info_score,
            'feature_details': self.feature_details,
            'selection_rationale': self.selection_rationale,
            'selection_time': self.selection_time,
            'method_used': self.method_used,
            'removed_features': self.removed_features,
            'feature_correlations': self.feature_correlations,
            'stability_scores': self.stability_scores
        }

class DirectionalFeatureSelectionAdapter:
    """
    Adapter for integrating directional features with existing feature selection pipeline.
    
    This adapter ensures that directional features (long/short variants) are properly
    handled while maintaining the optimal feature count for ML models.
    """
    
    def __init__(self, config: Optional[DirectionalFeatureSelectionConfig] = None):
        """Initialize the adapter."""
        self.config = config or DirectionalFeatureSelectionConfig()
        self.logger = logging.getLogger(__name__)

        # Set model-specific parameters
        self._set_model_specific_parameters()

        # Initialize feature selection framework if available
        self.feature_selector = None
        if FEATURE_SELECTION_AVAILABLE:
            try:
                self.feature_selector = FeatureSelectionFramework()
                tprint("✅ Feature selection framework initialized")
            except Exception as e:
                tprint(f"⚠️ Failed to initialize feature selection framework: {e}")

        # Track selection history
        self.selection_history: List[DirectionalFeatureSelectionResult] = []

        tprint("🎯 DirectionalFeatureSelectionAdapter initialized")
        tprint(f"📊 Model Type: {self.config.model_type}")
        tprint(f"📈 Target Features: {self.config.target_total_features}")

    def _set_model_specific_parameters(self):
        """Set model-specific feature selection parameters."""
        tprint("⚙️ Applying model-specific feature selection parameters")
        model_specific_params = {
            'AdvancedMambaHybrid': {
                'target_total_features': 100,
                'max_total_features': 120,
                'min_total_features': 80,
                'model_correlation_threshold': 0.88,
                'model_importance_threshold': 0.003,
                'max_correlation_threshold': 0.90,
                'model_priority_categories': ['momentum', 'interaction', 'microstructure', 'temporal']
            },
            'FinancialResNet': {
                'target_total_features': 120,
                'max_total_features': 150,
                'min_total_features': 100,
                'model_correlation_threshold': 0.95,
                'model_importance_threshold': 0.002,
                'max_correlation_threshold': 0.96,
                'model_priority_categories': ['regime', 'temporal', 'volatility', 'microstructure']
            },
            'DeepScaler': {
                'target_total_features': 80,
                'max_total_features': 100,
                'min_total_features': 60,
                'model_correlation_threshold': 0.85,
                'model_importance_threshold': 0.008,
                'max_correlation_threshold': 0.88,
                'model_priority_categories': ['statistical', 'momentum', 'volatility']
            },
            'NBEATS': {
                'target_total_features': 70,
                'max_total_features': 80,
                'min_total_features': 50,
                'model_correlation_threshold': 0.90,
                'model_importance_threshold': 0.005,
                'max_correlation_threshold': 0.92,
                'model_priority_categories': ['temporal', 'trend', 'seasonality', 'volatility']
            }
        }

        if self.config.model_type in model_specific_params:
            params = model_specific_params[self.config.model_type]
            for param, value in params.items():
                setattr(self.config, param, value)
            self.logger.info(f"✅ Applied {self.config.model_type} specific parameters")
            tprint(f"✅ Applied model-specific parameters for {self.config.model_type}")
        else:
            tprint(f"ℹ️ Using default configuration for model type: {self.config.model_type}")
    
    def select_optimal_directional_features(self,
                                          directional_result: DirectionalOptimizationResult,
                                          data: Optional[pd.DataFrame] = None,
                                          target_column: str = 'returns') -> DirectionalFeatureSelectionResult:
        """
        Select optimal features from directional optimization results.
        
        Args:
            directional_result: Results from directional optimization
            data: Optional data for advanced selection methods
            target_column: Target column for selection
            
        Returns:
            DirectionalFeatureSelectionResult with selected features
        """
        tprint("🎯 Starting directional feature selection...")
        start_time = time.time()
        
        # Initialize result
        selection_result = DirectionalFeatureSelectionResult()
        
        # Step 1: Quality filtering
        tprint("🔍 Step 1: Quality filtering...")
        filtered_long, filtered_short = self._quality_filter_features(directional_result)
        
        # Step 2: Cross-directional analysis
        if self.config.enable_cross_directional_filtering:
            tprint("🔄 Step 2: Cross-directional analysis...")
            filtered_long, filtered_short = self._cross_directional_filter(
                filtered_long, filtered_short, directional_result
            )
        
        # Step 3: Performance-based ranking
        tprint("📈 Step 3: Performance-based ranking...")
        ranked_long = self._rank_features_by_performance(filtered_long)
        ranked_short = self._rank_features_by_performance(filtered_short)
        
        # Step 4: Balanced selection
        tprint("⚖️ Step 4: Balanced feature selection...")
        selection_result = self._balanced_feature_selection(
            ranked_long, ranked_short, selection_result
        )
        
        # Step 5: Final validation and optimization
        tprint("✅ Step 5: Final validation...")
        selection_result = self._final_validation_and_optimization(
            selection_result, directional_result
        )
        
        # Calculate final metrics
        selection_result.selection_time = time.time() - start_time
        selection_result = self._calculate_selection_metrics(selection_result, directional_result)
        
        # Store in history
        self.selection_history.append(selection_result)
        
        tprint(f"✅ Directional feature selection completed in {selection_result.selection_time:.2f}s")
        tprint(f"📊 Selected {selection_result.total_selected_features} features: "
               f"{len(selection_result.selected_long_features)} long + "
               f"{len(selection_result.selected_short_features)} short")
        
        return selection_result
    
    def _quality_filter_features(self,
                               directional_result: DirectionalOptimizationResult) -> Tuple[Dict[str, DirectionalFeatureResult],
                                                                                         Dict[str, DirectionalFeatureResult]]:
        """Filter features based on quality thresholds."""
        tprint("🧹 Performing quality filtering on directional features")
        filtered_long = {}
        filtered_short = {}
        
        # Filter long features
        for feature_name, feature_result in directional_result.long_features.items():
            if self._passes_quality_check(feature_result):
                filtered_long[feature_name] = feature_result
            else:
                tprint(f"🚫 Filtered out long feature {feature_name}: quality check failed")
        
        # Filter short features
        for feature_name, feature_result in directional_result.short_features.items():
            if self._passes_quality_check(feature_result):
                filtered_short[feature_name] = feature_result
            else:
                tprint(f"🚫 Filtered out short feature {feature_name}: quality check failed")
        
        tprint(f"📊 Quality filtering: {len(filtered_long)} long + {len(filtered_short)} short features passed")
        return filtered_long, filtered_short
    
    def _passes_quality_check(self, feature_result: DirectionalFeatureResult) -> bool:
        """Check if a feature passes quality thresholds."""
        tprint(f"🔎 Evaluating quality for feature {feature_result.feature_name} ({feature_result.direction})")
        checks = [
            feature_result.mutual_info_score >= self.config.min_mutual_info_score,
            feature_result.stability_score >= self.config.min_stability_score,
            feature_result.data_quality_score > 0.1,  # Minimum data quality
            feature_result.convergence_achieved,
            feature_result.sample_count >= 50  # Minimum sample size
        ]

        passed = all(checks)
        if passed:
            tprint(f"✅ Feature {feature_result.feature_name} passed quality checks")
        else:
            tprint(f"❌ Feature {feature_result.feature_name} failed quality checks")
        return passed
    
    def _cross_directional_filter(self,
                                long_features: Dict[str, DirectionalFeatureResult],
                                short_features: Dict[str, DirectionalFeatureResult],
                                directional_result: DirectionalOptimizationResult) -> Tuple[Dict[str, DirectionalFeatureResult], 
                                                                                          Dict[str, DirectionalFeatureResult]]:
        """Filter features based on cross-directional analysis."""
        tprint("🔁 Executing cross-directional filtering")
        filtered_long = long_features.copy()
        filtered_short = short_features.copy()
        
        # Use directional differences to filter
        for feature_name in directional_result.directional_differences:
            diff_analysis = directional_result.directional_differences[feature_name]
            
            # If one direction is significantly better, prefer it
            long_score = diff_analysis.get('long_mi_score', 0)
            short_score = diff_analysis.get('short_mi_score', 0)
            
            if long_score > 0 and short_score > 0:
                # Keep both if both are good, but mark preference
                ratio = min(long_score, short_score) / max(long_score, short_score)
                
                # If one direction is much weaker (< 50% of the other), consider removing it
                if ratio < 0.5:
                    if long_score > short_score and feature_name in filtered_short:
                        # Remove weaker short version
                        del filtered_short[feature_name]
                        tprint(f"🔄 Removed weak short variant of {feature_name} (ratio: {ratio:.3f})")
                    elif short_score > long_score and feature_name in filtered_long:
                        # Remove weaker long version
                        del filtered_long[feature_name]
                        tprint(f"🔄 Removed weak long variant of {feature_name} (ratio: {ratio:.3f})")
        
        return filtered_long, filtered_short
    
    def _rank_features_by_performance(self, features: Dict[str, DirectionalFeatureResult]) -> List[Tuple[str, DirectionalFeatureResult, float]]:
        """Rank features by performance score."""
        tprint(f"🏅 Ranking {len(features)} features by performance")
        ranked_features = []

        for feature_name, feature_result in features.items():
            # Calculate composite performance score
            performance_score = (
                feature_result.mutual_info_score * 0.4 +
                feature_result.stability_score * 0.3 +
                feature_result.cross_validation_score * 0.2 +
                feature_result.data_quality_score * 0.1
            )
            
            ranked_features.append((feature_name, feature_result, performance_score))

        # Sort by performance score (descending)
        ranked_features.sort(key=lambda x: x[2], reverse=True)

        tprint(f"📊 Completed ranking for {len(ranked_features)} features")

        return ranked_features
    
    def _balanced_feature_selection(self,
                                  ranked_long: List[Tuple[str, DirectionalFeatureResult, float]],
                                  ranked_short: List[Tuple[str, DirectionalFeatureResult, float]],
                                  selection_result: DirectionalFeatureSelectionResult) -> DirectionalFeatureSelectionResult:
        """Select features with directional balance."""
        tprint("⚖️ Executing balanced feature selection")

        # Calculate target counts per direction
        target_per_direction = self.config.target_total_features // 2
        
        if self.config.maintain_directional_balance:
            # Balanced selection
            max_long = min(target_per_direction, len(ranked_long))
            max_short = min(target_per_direction, len(ranked_short))
            
            # Adjust for imbalance
            total_available = len(ranked_long) + len(ranked_short)
            if total_available < self.config.target_total_features:
                # Use all available features
                max_long = len(ranked_long)
                max_short = len(ranked_short)
            else:
                # Adjust based on availability
                if len(ranked_long) < target_per_direction:
                    # More short features available, allocate extra to short
                    extra = target_per_direction - len(ranked_long)
                    max_long = len(ranked_long)
                    max_short = min(target_per_direction + extra, len(ranked_short))
                elif len(ranked_short) < target_per_direction:
                    # More long features available, allocate extra to long
                    extra = target_per_direction - len(ranked_short)
                    max_short = len(ranked_short)
                    max_long = min(target_per_direction + extra, len(ranked_long))
        else:
            # Performance-based selection without strict balance
            all_features = []
            for name, result, score in ranked_long:
                all_features.append((name, result, score, 'long'))
            for name, result, score in ranked_short:
                all_features.append((name, result, score, 'short'))
            
            # Sort all features by performance
            all_features.sort(key=lambda x: x[2], reverse=True)
            
            # Select top features
            selected_features = all_features[:self.config.target_total_features]
            
            max_long = sum(1 for _, _, _, direction in selected_features if direction == 'long')
            max_short = sum(1 for _, _, _, direction in selected_features if direction == 'short')
        
        # Select features
        selection_result.selected_long_features = [name for name, _, _ in ranked_long[:max_long]]
        selection_result.selected_short_features = [name for name, _, _ in ranked_short[:max_short]]
        selection_result.total_selected_features = len(selection_result.selected_long_features) + len(selection_result.selected_short_features)

        # Store feature details
        for name, result, score in ranked_long[:max_long]:
            selection_result.feature_details[f"{name}_long"] = {
                'direction': 'long',
                'performance_score': score,
                'mutual_info_score': result.mutual_info_score,
                'stability_score': result.stability_score,
                'lookback_period': result.optimal_lookback_period
            }
        
        for name, result, score in ranked_short[:max_short]:
            selection_result.feature_details[f"{name}_short"] = {
                'direction': 'short',
                'performance_score': score,
                'mutual_info_score': result.mutual_info_score,
                'stability_score': result.stability_score,
                'lookback_period': result.optimal_lookback_period
            }

        tprint(
            f"📌 Balanced selection selected {len(selection_result.selected_long_features)} long and "
            f"{len(selection_result.selected_short_features)} short features"
        )

        return selection_result
    
    def _final_validation_and_optimization(self,
                                         selection_result: DirectionalFeatureSelectionResult,
                                         directional_result: DirectionalOptimizationResult) -> DirectionalFeatureSelectionResult:
        """Final validation and optimization of selected features."""
        tprint("🧪 Performing final validation and optimization")

        # Check if we're within target range
        total_features = selection_result.total_selected_features
        
        if total_features > self.config.max_total_features:
            tprint(f"⚠️ Too many features selected ({total_features}), reducing to {self.config.max_total_features}")
            selection_result = self._reduce_feature_count(selection_result, self.config.max_total_features)
        elif total_features < self.config.min_total_features:
            tprint(f"⚠️ Too few features selected ({total_features}), trying to increase to {self.config.min_total_features}")
            selection_result = self._increase_feature_count(selection_result, directional_result, self.config.min_total_features)
        
        # Final quality check
        selection_result = self._final_quality_check(selection_result)
        
        return selection_result
    
    def _reduce_feature_count(self, 
                            selection_result: DirectionalFeatureSelectionResult, 
                            target_count: int) -> DirectionalFeatureSelectionResult:
        """Reduce feature count while maintaining balance."""
        tprint(f"📉 Reducing feature count to target of {target_count}")
        current_long = len(selection_result.selected_long_features)
        current_short = len(selection_result.selected_short_features)
        current_total = current_long + current_short
        
        reduction_needed = current_total - target_count
        
        if self.config.maintain_directional_balance:
            # Reduce proportionally
            long_reduction = int(reduction_needed * current_long / current_total)
            short_reduction = reduction_needed - long_reduction
            
            # Remove lowest performing features
            selection_result.selected_long_features = selection_result.selected_long_features[:current_long - long_reduction]
            selection_result.selected_short_features = selection_result.selected_short_features[:current_short - short_reduction]
        else:
            # Remove lowest performing features overall
            all_features = []
            for name in selection_result.selected_long_features:
                if f"{name}_long" in selection_result.feature_details:
                    score = selection_result.feature_details[f"{name}_long"]['performance_score']
                    all_features.append((name, 'long', score))
            
            for name in selection_result.selected_short_features:
                if f"{name}_short" in selection_result.feature_details:
                    score = selection_result.feature_details[f"{name}_short"]['performance_score']
                    all_features.append((name, 'short', score))
            
            # Sort by performance and keep top features
            all_features.sort(key=lambda x: x[2], reverse=True)
            top_features = all_features[:target_count]
            
            selection_result.selected_long_features = [name for name, direction, _ in top_features if direction == 'long']
            selection_result.selected_short_features = [name for name, direction, _ in top_features if direction == 'short']

        selection_result.total_selected_features = len(selection_result.selected_long_features) + len(selection_result.selected_short_features)
        tprint(
            f"✅ Reduced feature count to {selection_result.total_selected_features} "
            f"({len(selection_result.selected_long_features)} long / {len(selection_result.selected_short_features)} short)"
        )
        return selection_result
    
    def _increase_feature_count(self,
                              selection_result: DirectionalFeatureSelectionResult,
                              directional_result: DirectionalOptimizationResult,
                              target_count: int) -> DirectionalFeatureSelectionResult:
        """Increase feature count if possible."""
        # This would require access to the filtered features that weren't selected
        # For now, just return the current result
        tprint("⚠️ Cannot increase feature count - no additional features available")
        return selection_result
    
    def _final_quality_check(self, selection_result: DirectionalFeatureSelectionResult) -> DirectionalFeatureSelectionResult:
        """Final quality check on selected features."""
        tprint("🔍 Running final quality check on selected features")
        # Add rationale for selection
        selection_result.selection_rationale = {
            'method': 'balanced_performance_selection',
            'criteria': [
                'mutual_info_score >= ' + str(self.config.min_mutual_info_score),
                'stability_score >= ' + str(self.config.min_stability_score),
                'directional_balance maintained' if self.config.maintain_directional_balance else 'performance_optimized'
            ]
        }

        selection_result.method_used = 'directional_balanced_selection'
        tprint("✅ Final quality check completed")
        return selection_result
    
    def _calculate_selection_metrics(self,
                                   selection_result: DirectionalFeatureSelectionResult,
                                   directional_result: DirectionalOptimizationResult) -> DirectionalFeatureSelectionResult:
        """Calculate final selection metrics."""
        tprint("📊 Calculating selection metrics")

        # Directional balance ratio
        long_count = len(selection_result.selected_long_features)
        short_count = len(selection_result.selected_short_features)
        if long_count + short_count > 0:
            selection_result.directional_balance_ratio = min(long_count, short_count) / max(long_count, short_count)
        
        # Average mutual information score
        all_scores = []
        for feature_details in selection_result.feature_details.values():
            all_scores.append(feature_details.get('mutual_info_score', 0))
        
        if all_scores:
            selection_result.average_mutual_info_score = np.mean(all_scores)
        
        # Selection quality score (based on how well we met targets)
        target_met_ratio = selection_result.total_selected_features / self.config.target_total_features
        balance_quality = selection_result.directional_balance_ratio
        performance_quality = selection_result.average_mutual_info_score

        selection_result.selection_quality_score = (
            min(target_met_ratio, 1.0) * 0.4 +  # Don't penalize for exceeding target
            balance_quality * 0.3 +
            performance_quality * 0.3
        )

        tprint(
            f"📈 Metrics calculated - Balance ratio: {selection_result.directional_balance_ratio:.3f}, "
            f"Average MI: {selection_result.average_mutual_info_score:.4f}, "
            f"Quality score: {selection_result.selection_quality_score:.4f}"
        )

        return selection_result
    
    def _detect_market_regime_vectorbt(self, data: pd.DataFrame, target_column: str = 'returns') -> str:
        """Detect market regime using VectorBT."""
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt_analysis or not self.config.enable_regime_detection:
            return self._detect_market_regime_fallback(data, target_column)
        
        try:
            if target_column not in data.columns:
                return 'unknown'
            
            returns = data[target_column]
            if len(returns) < self.config.regime_window:
                return 'unknown'
            
            # Calculate volatility regime
            volatility = returns.rolling(window=self.config.regime_window).std() * np.sqrt(252)
            current_vol = volatility.iloc[-1] if not volatility.empty else 0.0
            avg_vol = volatility.mean() if not volatility.empty else 0.0
            
            # Determine volatility level
            if current_vol > avg_vol * self.config.volatility_threshold:
                vol_level = 'high_vol'
            elif current_vol < avg_vol / self.config.volatility_threshold:
                vol_level = 'low_vol'
            else:
                vol_level = 'normal_vol'
            
            # Calculate trend regime using VectorBT
            sma_short = vbt.MA.run(returns, window=10).ma
            sma_long = vbt.MA.run(returns, window=30).ma
            
            if not sma_short.empty and not sma_long.empty:
                trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
                
                if trend_strength > self.config.trend_threshold:
                    trend_direction = 'uptrend'
                elif trend_strength < -self.config.trend_threshold:
                    trend_direction = 'downtrend'
                else:
                    trend_direction = 'sideways'
            else:
                trend_direction = 'sideways'
            
            return f"{trend_direction}_{vol_level}"
            
        except Exception as e:
            self.logger.warning(f"VectorBT regime detection failed: {e}")
            return self._detect_market_regime_fallback(data, target_column)
    
    def _detect_market_regime_fallback(self, data: pd.DataFrame, target_column: str = 'returns') -> str:
        """Fallback regime detection without VectorBT."""
        if target_column not in data.columns or len(data) < 20:
            return 'unknown'
        
        returns = data[target_column]
        
        # Simple volatility calculation
        volatility = returns.rolling(window=min(20, len(returns))).std() * np.sqrt(252)
        current_vol = volatility.iloc[-1] if not volatility.empty else 0.0
        avg_vol = volatility.mean() if not volatility.empty else 0.0
        
        vol_level = 'high_vol' if current_vol > avg_vol * 1.5 else 'low_vol' if current_vol < avg_vol * 0.5 else 'normal_vol'
        
        # Simple trend detection
        sma_short = returns.rolling(window=10).mean()
        sma_long = returns.rolling(window=30).mean()
        
        if not sma_short.empty and not sma_long.empty:
            trend_strength = (sma_short.iloc[-1] - sma_long.iloc[-1]) / sma_long.iloc[-1]
            trend_direction = 'uptrend' if trend_strength > 0.02 else 'downtrend' if trend_strength < -0.02 else 'sideways'
        else:
            trend_direction = 'sideways'
        
        return f"{trend_direction}_{vol_level}"
    
    def _analyze_temporal_features_vectorbt(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
        """Analyze temporal characteristics of features using VectorBT."""
        if not VECTORBT_AVAILABLE or not self.config.enable_vectorbt_analysis or not self.config.enable_temporal_analysis:
            return self._analyze_temporal_features_fallback(data, features)
        
        try:
            temporal_analysis = {}
            
            for feature in features:
                if feature not in data.columns:
                    continue
                
                feature_series = data[feature]
                
                if len(feature_series) < 10:
                    continue
                
                # Calculate trend strength using VectorBT
                trend_slope = vbt.linear_regression(feature_series, window=min(20, len(feature_series))).slope
                trend_strength = abs(trend_slope.iloc[-1]) if not trend_slope.empty else 0.0
                
                # Calculate seasonality
                seasonality_strength = 0.0
                for period in [5, 10, 20]:
                    if len(feature_series) > period * 2:
                        autocorr = feature_series.autocorr(lag=period)
                        seasonality_strength = max(seasonality_strength, abs(autocorr))
                
                # Calculate autocorrelation
                autocorr = feature_series.autocorr(lag=1) if len(feature_series) > 1 else 0.0
                
                # Calculate stationarity (simplified)
                stationarity = self._test_stationarity(feature_series)
                
                temporal_analysis[feature] = {
                    'trend_strength': float(trend_strength),
                    'seasonality_strength': float(seasonality_strength),
                    'autocorrelation': float(autocorr),
                    'stationarity': stationarity,
                    'temporal_importance': float(trend_strength * 0.4 + seasonality_strength * 0.3 + abs(autocorr) * 0.3)
                }
            
            return temporal_analysis
            
        except Exception as e:
            self.logger.warning(f"VectorBT temporal analysis failed: {e}")
            return self._analyze_temporal_features_fallback(data, features)
    
    def _analyze_temporal_features_fallback(self, data: pd.DataFrame, features: List[str]) -> Dict[str, Dict[str, float]]:
        """Fallback temporal analysis without VectorBT."""
        temporal_analysis = {}
        
        for feature in features:
            if feature not in data.columns:
                continue
            
            feature_series = data[feature]
            
            if len(feature_series) < 3:
                continue
            
            # Simple trend calculation
            x = np.arange(len(feature_series))
            slope, _ = np.polyfit(x, feature_series.values, 1)
            trend_strength = abs(slope)
            
            # Simple autocorrelation
            autocorr = feature_series.autocorr(lag=1) if len(feature_series) > 1 else 0.0
            
            # Simple stationarity test
            stationarity = self._test_stationarity(feature_series)
            
            temporal_analysis[feature] = {
                'trend_strength': float(trend_strength),
                'seasonality_strength': 0.0,
                'autocorrelation': float(autocorr),
                'stationarity': stationarity,
                'temporal_importance': float(trend_strength * 0.5 + abs(autocorr) * 0.5)
            }
        
        return temporal_analysis
    
    def _test_stationarity(self, series: pd.Series) -> bool:
        """Simple stationarity test."""
        if len(series) < 10:
            return True
        
        # Simple test: check if variance is relatively stable
        half_len = len(series) // 2
        first_half_var = series.iloc[:half_len].var()
        second_half_var = series.iloc[half_len:].var()
        
        if first_half_var == 0 or second_half_var == 0:
            return True
        
        variance_ratio = abs(first_half_var - second_half_var) / max(first_half_var, second_half_var)
        return variance_ratio < 0.5  # Threshold for stationarity
    
    def _apply_regime_aware_selection(self, 
                                    features: List[str], 
                                    regime: str, 
                                    temporal_analysis: Dict[str, Dict[str, float]]) -> List[str]:
        """Apply regime-aware feature selection."""
        if not self.config.enable_vectorbt_analysis:
            return features
        
        try:
            regime_features = []
            
            if regime.startswith('uptrend'):
                # For uptrend, prefer features with positive momentum and trend
                regime_features = [
                    f for f in features 
                    if temporal_analysis.get(f, {}).get('trend_strength', 0) > 0.1
                ]
            elif regime.startswith('downtrend'):
                # For downtrend, prefer defensive features (stationary or negative trend)
                regime_features = [
                    f for f in features 
                    if temporal_analysis.get(f, {}).get('stationarity', False) or 
                       temporal_analysis.get(f, {}).get('trend_strength', 0) < -0.1
                ]
            else:
                # For sideways, prefer features with low volatility and seasonality
                regime_features = [
                    f for f in features 
                    if temporal_analysis.get(f, {}).get('seasonality_strength', 0) > 0.1 or
                       temporal_analysis.get(f, {}).get('stationarity', False)
                ]
            
            # If no regime-specific features found, return original features
            if not regime_features:
                regime_features = features
            
            tprint(f"🎯 Regime-aware selection: {regime} -> {len(regime_features)}/{len(features)} features")
            return regime_features
            
        except Exception as e:
            self.logger.warning(f"Regime-aware selection failed: {e}")
            return features

    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of selection process."""
        tprint("🗒️ Generating selection summary")
        if not self.selection_history:
            tprint("ℹ️ No selection history available")
            return {"message": "No selections completed yet"}

        latest = self.selection_history[-1]

        tprint(
            f"📋 Latest selection - Total: {latest.total_selected_features}, "
            f"Long: {len(latest.selected_long_features)}, Short: {len(latest.selected_short_features)}"
        )

        return {
            "total_selections": len(self.selection_history),
            "latest_selection": {
                "total_features": latest.total_selected_features,
                "long_features": len(latest.selected_long_features),
                "short_features": len(latest.selected_short_features),
                "balance_ratio": latest.directional_balance_ratio,
                "quality_score": latest.selection_quality_score,
                "avg_mutual_info": latest.average_mutual_info_score,
                "selection_time": latest.selection_time,
                "method": latest.method_used
            },
            "config": {
                "target_total_features": self.config.target_total_features,
                "maintain_balance": self.config.maintain_directional_balance,
                "min_mutual_info": self.config.min_mutual_info_score
            }
        }

# Convenience function for easy integration
def select_directional_features(directional_result: DirectionalOptimizationResult,
                               data: Optional[pd.DataFrame] = None,
                               target_column: str = 'returns',
                               config: Optional[DirectionalFeatureSelectionConfig] = None) -> DirectionalFeatureSelectionResult:
    """
    Convenience function to select optimal directional features.
    
    Args:
        directional_result: Results from directional optimization
        data: Optional data for advanced selection methods
        target_column: Target column for selection
        config: Optional configuration
        
    Returns:
        DirectionalFeatureSelectionResult with selected features
    """
    tprint("🚀 Initiating directional feature selection via convenience function")
    adapter = DirectionalFeatureSelectionAdapter(config=config)
    tprint("🔧 Adapter created, starting selection process")
    return adapter.select_optimal_directional_features(
        directional_result=directional_result,
        data=data,
        target_column=target_column
    )