"""
Regime Validator for Market Analysis Components.

This module provides validation capabilities specifically for regime detection
and clustering results, including regime persistence, economic validity,
and temporal consistency validation.
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
from datetime import datetime

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error
from src.utils.common_utilities import safe_dataframe_operation, validate_dataframe_columns
from src.utils.math_validation import validate_finite, safe_divide, safe_log
from src.training.steps.market_analysis.components.base_component import BaseMarketAnalysisComponent, ComponentConfig

class RegimeValidationLevel(Enum):
    """Regime validation severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class RegimeValidationConfig:
    """Configuration for regime validation."""
    # Regime persistence validation
    min_regime_duration: int = 5
    min_regime_persistence: float = 0.1
    max_regime_volatility: float = 0.5
    
    # Economic validation
    min_economic_significance: float = 0.05
    max_correlation_threshold: float = 0.95
    min_regime_separation: float = 0.1
    
    # Temporal validation
    check_temporal_consistency: bool = True
    min_temporal_stability: float = 0.5
    max_regime_switches: float = 0.3
    
    # Clustering validation
    min_cluster_size: int = 10
    max_cluster_size_ratio: float = 0.8
    min_silhouette_score: float = 0.1

@dataclass
class RegimeValidationResult:
    """Result of regime validation."""
    passed: bool
    score: float
    level: RegimeValidationLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    issues: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class RegimeValidator(BaseMarketAnalysisComponent):
    """
    Validator for regime detection and clustering results.
    
    Provides validation for:
    - Regime persistence and stability
    - Economic significance of regimes
    - Temporal consistency
    - Clustering quality metrics
    """
    
    def __init__(self, config: Optional[RegimeValidationConfig] = None):
        """Initialize the regime validator."""
        super().__init__(ComponentConfig())
        self.validation_config = config or RegimeValidationConfig()
        self.logger = logging.getLogger(__name__)
        
    async def validate_regimes(self, 
                             data: pd.DataFrame,
                             regime_assignments: np.ndarray,
                             context: str = "regime_validation") -> RegimeValidationResult:
        """
        Validate regime detection results.
        
        Args:
            data: Market data DataFrame
            regime_assignments: Array of regime assignments
            context: Validation context for logging
            
        Returns:
            RegimeValidationResult with validation details
        """
        try:
            tprint_info(f"🔍 Starting regime validation for {context}")
            
            # Initialize result
            result = RegimeValidationResult(
                passed=True,
                score=1.0,
                level=RegimeValidationLevel.INFO,
                message="Regime validation completed successfully"
            )
            
            # Perform validation checks
            await self._validate_regime_persistence(data, regime_assignments, result)
            await self._validate_economic_significance(data, regime_assignments, result)
            await self._validate_temporal_consistency(regime_assignments, result)
            await self._validate_clustering_quality(data, regime_assignments, result)
            
            # Calculate overall score
            result.score = self._calculate_regime_score(result)
            result.passed = result.score >= 0.7 and result.level != RegimeValidationLevel.CRITICAL
            
            # Generate recommendations
            result.recommendations = self._generate_regime_recommendations(result)
            
            tprint_info(f"✅ Regime validation completed: score={result.score:.3f}, passed={result.passed}")
            return result
            
        except Exception as e:
            tprint_error(f"❌ Regime validation failed: {str(e)}")
            return RegimeValidationResult(
                passed=False,
                score=0.0,
                level=RegimeValidationLevel.CRITICAL,
                message=f"Regime validation failed with error: {str(e)}",
                issues=[str(e)]
            )
    
    async def _validate_regime_persistence(self, 
                                         data: pd.DataFrame, 
                                         regime_assignments: np.ndarray, 
                                         result: RegimeValidationResult):
        """Validate regime persistence and stability."""
        try:
            unique_regimes = np.unique(regime_assignments)
            regime_stats = {}
            
            for regime in unique_regimes:
                regime_mask = regime_assignments == regime
                regime_data = data[regime_mask]
                
                # Calculate regime duration
                regime_duration = len(regime_data)
                
                # Calculate regime persistence (consecutive periods)
                regime_changes = np.diff(regime_assignments)
                regime_starts = np.where(regime_changes != 0)[0] + 1
                regime_ends = np.concatenate([regime_starts[1:], [len(regime_assignments)]])
                
                regime_periods = []
                for i, start in enumerate(regime_starts):
                    if i < len(regime_ends):
                        period_length = regime_ends[i] - start
                        regime_periods.append(period_length)
                
                avg_persistence = np.mean(regime_periods) if regime_periods else 0
                
                # Calculate regime volatility (if price data available)
                volatility = 0.0
                if 'close' in data.columns:
                    regime_prices = data.loc[regime_mask, 'close'].dropna()
                    if len(regime_prices) > 1:
                        returns = regime_prices.pct_change().dropna()
                        volatility = returns.std() if len(returns) > 0 else 0.0
                
                regime_stats[regime] = {
                    'duration': regime_duration,
                    'avg_persistence': avg_persistence,
                    'volatility': volatility,
                    'n_periods': len(regime_periods)
                }
                
                # Check minimum duration
                if regime_duration < self.validation_config.min_regime_duration:
                    result.issues.append(f"Regime {regime} too short: {regime_duration} < {self.validation_config.min_regime_duration}")
                    result.level = RegimeValidationLevel.ERROR
                
                # Check persistence
                if avg_persistence < self.validation_config.min_regime_persistence:
                    result.warnings.append(f"Regime {regime} low persistence: {avg_persistence:.3f}")
                
                # Check volatility
                if volatility > self.validation_config.max_regime_volatility:
                    result.warnings.append(f"Regime {regime} high volatility: {volatility:.3f}")
            
            result.details['regime_persistence'] = regime_stats
            
        except Exception as e:
            result.issues.append(f"Regime persistence validation error: {str(e)}")
            result.level = RegimeValidationLevel.ERROR
    
    async def _validate_economic_significance(self, 
                                            data: pd.DataFrame, 
                                            regime_assignments: np.ndarray, 
                                            result: RegimeValidationResult):
        """Validate economic significance of regimes."""
        try:
            unique_regimes = np.unique(regime_assignments)
            
            if len(unique_regimes) < 2:
                result.warnings.append("Only one regime detected - cannot validate economic significance")
                return
            
            # Calculate regime statistics
            regime_stats = {}
            for regime in unique_regimes:
                regime_mask = regime_assignments == regime
                regime_data = data[regime_mask]
                
                stats = {}
                if 'close' in data.columns:
                    prices = regime_data['close'].dropna()
                    if len(prices) > 1:
                        returns = prices.pct_change().dropna()
                        stats['mean_return'] = returns.mean()
                        stats['volatility'] = returns.std()
                        stats['sharpe_ratio'] = safe_divide(returns.mean(), returns.std())
                
                regime_stats[regime] = stats
            
            # Check regime separation
            regime_means = [stats.get('mean_return', 0) for stats in regime_stats.values()]
            if len(regime_means) > 1:
                mean_separation = np.std(regime_means)
                if mean_separation < self.validation_config.min_regime_separation:
                    result.warnings.append(f"Low regime separation: {mean_separation:.3f}")
            
            # Check economic significance
            significant_regimes = 0
            for regime, stats in regime_stats.items():
                if 'mean_return' in stats and abs(stats['mean_return']) > self.validation_config.min_economic_significance:
                    significant_regimes += 1
            
            if significant_regimes < len(unique_regimes) * 0.5:
                result.warnings.append(f"Only {significant_regimes}/{len(unique_regimes)} regimes economically significant")
            
            result.details['economic_significance'] = {
                'regime_stats': regime_stats,
                'mean_separation': mean_separation if len(regime_means) > 1 else 0.0,
                'significant_regimes': significant_regimes
            }
            
        except Exception as e:
            result.issues.append(f"Economic significance validation error: {str(e)}")
            result.level = RegimeValidationLevel.ERROR
    
    async def _validate_temporal_consistency(self, 
                                           regime_assignments: np.ndarray, 
                                           result: RegimeValidationResult):
        """Validate temporal consistency of regime assignments."""
        try:
            if not self.validation_config.check_temporal_consistency:
                return
            
            # Calculate regime switches
            regime_changes = np.diff(regime_assignments)
            n_switches = np.sum(regime_changes != 0)
            switch_ratio = n_switches / len(regime_assignments) if len(regime_assignments) > 0 else 0
            
            if switch_ratio > self.validation_config.max_regime_switches:
                result.warnings.append(f"High regime switching: {switch_ratio:.3f}")
            
            # Calculate temporal stability
            unique_regimes = np.unique(regime_assignments)
            stability_scores = []
            
            for regime in unique_regimes:
                regime_mask = regime_assignments == regime
                regime_positions = np.where(regime_mask)[0]
                
                if len(regime_positions) > 1:
                    # Calculate how clustered the regime periods are
                    position_diffs = np.diff(regime_positions)
                    avg_gap = np.mean(position_diffs) if len(position_diffs) > 0 else 0
                    stability = 1.0 / (1.0 + avg_gap) if avg_gap > 0 else 1.0
                    stability_scores.append(stability)
            
            avg_stability = np.mean(stability_scores) if stability_scores else 0.0
            
            if avg_stability < self.validation_config.min_temporal_stability:
                result.warnings.append(f"Low temporal stability: {avg_stability:.3f}")
            
            result.details['temporal_consistency'] = {
                'n_switches': n_switches,
                'switch_ratio': switch_ratio,
                'avg_stability': avg_stability,
                'stability_scores': stability_scores
            }
            
        except Exception as e:
            result.issues.append(f"Temporal consistency validation error: {str(e)}")
            result.level = RegimeValidationLevel.ERROR
    
    async def _validate_clustering_quality(self, 
                                         data: pd.DataFrame, 
                                         regime_assignments: np.ndarray, 
                                         result: RegimeValidationResult):
        """Validate clustering quality metrics."""
        try:
            unique_regimes = np.unique(regime_assignments)
            n_regimes = len(unique_regimes)
            
            # Check minimum cluster size
            regime_sizes = [np.sum(regime_assignments == regime) for regime in unique_regimes]
            min_size = min(regime_sizes) if regime_sizes else 0
            
            if min_size < self.validation_config.min_cluster_size:
                result.issues.append(f"Minimum cluster size {min_size} < {self.validation_config.min_cluster_size}")
                result.level = RegimeValidationLevel.ERROR
            
            # Check cluster size balance
            if regime_sizes:
                max_size = max(regime_sizes)
                size_ratio = max_size / len(regime_assignments) if len(regime_assignments) > 0 else 0
                
                if size_ratio > self.validation_config.max_cluster_size_ratio:
                    result.warnings.append(f"Unbalanced clusters: max size ratio {size_ratio:.3f}")
            
            # Calculate silhouette score if possible
            silhouette_score = 0.0
            if len(unique_regimes) > 1 and len(data) > 0:
                try:
                    from sklearn.metrics import silhouette_score
                    # Use only numeric columns for silhouette calculation
                    numeric_data = data.select_dtypes(include=[np.number]).dropna()
                    if len(numeric_data) > 0 and len(numeric_data) == len(regime_assignments):
                        silhouette_score = silhouette_score(numeric_data, regime_assignments)
                except Exception:
                    pass
            
            if silhouette_score < self.validation_config.min_silhouette_score:
                result.warnings.append(f"Low silhouette score: {silhouette_score:.3f}")
            
            result.details['clustering_quality'] = {
                'n_regimes': n_regimes,
                'regime_sizes': regime_sizes,
                'min_size': min_size,
                'max_size_ratio': size_ratio if regime_sizes else 0.0,
                'silhouette_score': silhouette_score
            }
            
        except Exception as e:
            result.issues.append(f"Clustering quality validation error: {str(e)}")
            result.level = RegimeValidationLevel.ERROR
    
    def _calculate_regime_score(self, result: RegimeValidationResult) -> float:
        """Calculate overall regime validation score."""
        base_score = 1.0
        
        # Deduct for issues
        issue_penalty = len(result.issues) * 0.15
        warning_penalty = len(result.warnings) * 0.08
        
        # Level-based penalties
        level_penalty = {
            RegimeValidationLevel.INFO: 0.0,
            RegimeValidationLevel.WARNING: 0.1,
            RegimeValidationLevel.ERROR: 0.3,
            RegimeValidationLevel.CRITICAL: 0.5
        }.get(result.level, 0.0)
        
        score = max(0.0, base_score - issue_penalty - warning_penalty - level_penalty)
        return min(1.0, score)
    
    def _generate_regime_recommendations(self, result: RegimeValidationResult) -> List[str]:
        """Generate recommendations based on regime validation results."""
        recommendations = []
        
        if result.issues:
            recommendations.append("Address critical regime validation issues before proceeding")
        
        if result.warnings:
            recommendations.append("Review regime validation warnings and consider parameter tuning")
        
        if result.score < 0.7:
            recommendations.append("Consider adjusting regime detection parameters for better results")
        
        if len(result.issues) > 3:
            recommendations.append("Multiple regime validation issues detected - consider data preprocessing")
        
        return recommendations