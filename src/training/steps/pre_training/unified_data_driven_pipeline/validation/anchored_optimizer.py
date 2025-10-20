"""
Anchored Optimization Windows

This module implements anchored optimization to prevent recency bias
by enforcing time-based embargo and trailing window optimization.
"""

import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from src.utils.tprint import tprint, tprint_info, tprint_warning, tprint_error, tprint_success
from src.utils.common_operations import (
    safe_divide, safe_correlation, safe_mean, safe_std,
    validate_finite, validate_positive, memory_checkpoint
)


@dataclass
class AnchoredOptimizationConfig:
    """Configuration for anchored optimization."""
    
    # Time-based constraints
    max_optimization_window_days: int = 252  # 1 year max
    min_optimization_window_days: int = 63   # 3 months min
    embargo_days: int = 7                    # Days to embargo after optimization
    gap_days: int = 1                        # Gap between train/test
    
    # Anchoring parameters
    anchor_to_latest: bool = True           # Anchor to latest available data
    trailing_window_only: bool = True       # Only use trailing windows
    forbid_future_lookback: bool = True     # Prevent future information
    
    # Validation parameters
    min_periods_per_feature: int = 2        # Minimum periods per feature
    max_correlation_threshold: float = 0.85 # Max correlation between periods
    
    # Regime detection
    enable_regime_detection: bool = True    # Detect regime shifts
    regime_stability_days: int = 30         # Days for regime stability
    regime_change_threshold: float = 0.1    # Threshold for regime change
    
    # Logging
    verbose: bool = True


@dataclass
class OptimizationWindow:
    """Represents an anchored optimization window."""
    window_id: str
    anchor_date: datetime
    start_date: datetime
    end_date: datetime
    embargo_start: datetime
    embargo_end: datetime
    is_valid: bool = True
    regime_stable: bool = True
    violations: List[str] = field(default_factory=list)


@dataclass
class AnchoredOptimizationResult:
    """Result of anchored optimization."""
    
    # Window information
    windows: List[OptimizationWindow] = field(default_factory=list)
    valid_windows: int = 0
    invalid_windows: int = 0
    
    # Optimization results
    optimized_periods: Dict[str, int] = field(default_factory=dict)
    optimized_lookbacks: Dict[str, int] = field(default_factory=dict)
    
    # Validation metrics
    recency_bias_detected: bool = False
    regime_instability_detected: bool = False
    future_leakage_detected: bool = False
    
    # Performance metrics
    mean_ic: float = 0.0
    mean_sharpe: float = 0.0
    stability_score: float = 0.0
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)


class AnchoredOptimizer:
    """
    Anchored optimization to prevent recency bias.
    
    Enforces:
    1. Time-based embargo periods
    2. Trailing window optimization only
    3. No future information leakage
    4. Regime stability validation
    """
    
    def __init__(self, config: Optional[AnchoredOptimizationConfig] = None):
        """Initialize the anchored optimizer."""
        self.config = config or AnchoredOptimizationConfig()
        self.logger = logging.getLogger(__name__)
        
        if self.config.verbose:
            tprint("⚓ Initializing AnchoredOptimizer")
    
    def create_anchored_windows(self, 
                              data: pd.DataFrame,
                              start_date: Optional[datetime] = None,
                              end_date: Optional[datetime] = None) -> List[OptimizationWindow]:
        """
        Create anchored optimization windows.
        
        Args:
            data: Input data with datetime index
            start_date: Start date for optimization (default: data start)
            end_date: End date for optimization (default: data end)
            
        Returns:
            List of OptimizationWindow objects
        """
        if self.config.verbose:
            tprint("⚓ Creating anchored optimization windows")
        
        if start_date is None:
            start_date = data.index[0]
        if end_date is None:
            end_date = data.index[-1]
        
        windows = []
        current_date = start_date
        
        while current_date < end_date:
            # Calculate window end date
            window_end = current_date + timedelta(days=self.config.max_optimization_window_days)
            if window_end > end_date:
                window_end = end_date
            
            # Check minimum window size
            window_days = (window_end - current_date).days
            if window_days < self.config.min_optimization_window_days:
                break
            
            # Create optimization window
            window = OptimizationWindow(
                window_id=f"window_{len(windows)}",
                anchor_date=current_date,
                start_date=current_date,
                end_date=window_end,
                embargo_start=window_end + timedelta(days=self.config.gap_days),
                embargo_end=window_end + timedelta(days=self.config.gap_days + self.config.embargo_days)
            )
            
            # Validate window
            self._validate_window(window, data)
            windows.append(window)
            
            # Move to next window (with overlap prevention)
            current_date = window_end + timedelta(days=self.config.embargo_days)
        
        if self.config.verbose:
            tprint(f"✅ Created {len(windows)} anchored windows")
        
        return windows
    
    def _validate_window(self, window: OptimizationWindow, data: pd.DataFrame) -> None:
        """Validate an optimization window for violations."""
        violations = []
        
        # Check for future information leakage
        if self.config.forbid_future_lookback:
            future_data = data[data.index > window.end_date]
            if len(future_data) > 0:
                violations.append("Future information leakage detected")
                window.is_valid = False
        
        # Check for regime stability
        if self.config.enable_regime_detection:
            regime_stable = self._check_regime_stability(window, data)
            if not regime_stable:
                violations.append("Regime instability detected")
                window.regime_stable = False
        
        # Check for recency bias
        if self.config.trailing_window_only:
            recency_bias = self._check_recency_bias(window, data)
            if recency_bias:
                violations.append("Recency bias detected")
                window.is_valid = False
        
        # Check window size constraints
        window_days = (window.end_date - window.start_date).days
        if window_days < self.config.min_optimization_window_days:
            violations.append("Window too small")
            window.is_valid = False
        elif window_days > self.config.max_optimization_window_days:
            violations.append("Window too large")
            window.is_valid = False
        
        window.violations = violations
    
    def _check_regime_stability(self, window: OptimizationWindow, data: pd.DataFrame) -> bool:
        """Check for regime stability within the window."""
        try:
            # Extract window data
            window_data = data[
                (data.index >= window.start_date) & 
                (data.index <= window.end_date)
            ]
            
            if len(window_data) < self.config.regime_stability_days:
                return False
            
            # Calculate regime indicators (volatility, trend, etc.)
            volatility = window_data['close'].pct_change().rolling(20).std()
            trend = window_data['close'].rolling(20).mean()
            
            # Check for regime changes
            vol_changes = volatility.pct_change().abs()
            trend_changes = trend.pct_change().abs()
            
            # Regime change if volatility or trend changes exceed threshold
            regime_changes = (
                (vol_changes > self.config.regime_change_threshold).sum() +
                (trend_changes > self.config.regime_change_threshold).sum()
            )
            
            # Regime is stable if changes are minimal
            return regime_changes < len(window_data) * 0.1
        except:
            return False
    
    def _check_recency_bias(self, window: OptimizationWindow, data: pd.DataFrame) -> bool:
        """Check for recency bias in the window."""
        try:
            # Extract window data
            window_data = data[
                (data.index >= window.start_date) & 
                (data.index <= window.end_date)
            ]
            
            if len(window_data) < 20:
                return False
            
            # Check for recency bias by comparing early vs late performance
            mid_point = len(window_data) // 2
            early_data = window_data.iloc[:mid_point]
            late_data = window_data.iloc[mid_point:]
            
            # Calculate performance metrics
            early_vol = early_data['close'].pct_change().std()
            late_vol = late_data['close'].pct_change().std()
            
            # Recency bias if late volatility is significantly different
            vol_ratio = late_vol / early_vol if early_vol > 0 else 1.0
            return vol_ratio > 2.0 or vol_ratio < 0.5
        except:
            return False
    
    def optimize_with_anchoring(self, 
                              data: pd.DataFrame,
                              targets: pd.Series,
                              feature_optimizer: callable,
                              lookback_optimizer: callable) -> AnchoredOptimizationResult:
        """
        Perform anchored optimization.
        
        Args:
            data: Input features
            targets: Target labels
            feature_optimizer: Function to optimize features
            lookback_optimizer: Function to optimize lookbacks
            
        Returns:
            AnchoredOptimizationResult
        """
        if self.config.verbose:
            tprint("⚓ Starting anchored optimization")
        
        result = AnchoredOptimizationResult()
        
        # Create anchored windows
        windows = self.create_anchored_windows(data)
        result.windows = windows
        
        # Count valid/invalid windows
        result.valid_windows = sum(1 for w in windows if w.is_valid)
        result.invalid_windows = sum(1 for w in windows if not w.is_valid)
        
        # Optimize on valid windows only
        valid_windows = [w for w in windows if w.is_valid]
        
        if not valid_windows:
            tprint_error("❌ No valid optimization windows found")
            return result
        
        # Perform optimization on each valid window
        optimization_results = []
        for window in valid_windows:
            if self.config.verbose:
                tprint(f"🔄 Optimizing window {window.window_id}")
            
            # Extract window data
            window_data = data[
                (data.index >= window.start_date) & 
                (data.index <= window.end_date)
            ]
            window_targets = targets[
                (targets.index >= window.start_date) & 
                (targets.index <= window.end_date)
            ]
            
            # Optimize features and lookbacks
            try:
                feature_result = feature_optimizer(window_data, window_targets)
                lookback_result = lookback_optimizer(window_data, window_targets)
                
                optimization_results.append({
                    'window_id': window.window_id,
                    'feature_result': feature_result,
                    'lookback_result': lookback_result,
                    'ic': self._calculate_ic(feature_result, window_targets),
                    'sharpe': self._calculate_sharpe(feature_result, window_targets)
                })
            except Exception as e:
                self.logger.warning(f"Optimization failed for window {window.window_id}: {e}")
                continue
        
        # Aggregate results
        if optimization_results:
            result.mean_ic = np.mean([r['ic'] for r in optimization_results])
            result.mean_sharpe = np.mean([r['sharpe'] for r in optimization_results])
            result.stability_score = self._calculate_stability_score(optimization_results)
            
            # Detect issues
            result.recency_bias_detected = self._detect_recency_bias(optimization_results)
            result.regime_instability_detected = self._detect_regime_instability(windows)
            result.future_leakage_detected = self._detect_future_leakage(windows)
            
            # Generate recommendations
            result.recommendations = self._generate_recommendations(result)
        
        if self.config.verbose:
            tprint_success(f"✅ Anchored optimization completed")
            tprint(f"📊 Valid windows: {result.valid_windows}")
            tprint(f"📊 Mean IC: {result.mean_ic:.4f}")
            tprint(f"📊 Mean Sharpe: {result.mean_sharpe:.4f}")
        
        return result
    
    def _calculate_ic(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate Information Coefficient."""
        try:
            correlation = predictions.corr(actual)
            return correlation if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_sharpe(self, predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate Sharpe ratio."""
        try:
            returns = predictions.pct_change().dropna()
            if len(returns) == 0:
                return 0.0
            return returns.mean() / returns.std() if returns.std() > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_stability_score(self, optimization_results: List[Dict]) -> float:
        """Calculate stability score across optimization windows."""
        try:
            if len(optimization_results) < 2:
                return 0.0
            
            # Calculate stability as inverse of variance
            ic_scores = [r['ic'] for r in optimization_results]
            sharpe_scores = [r['sharpe'] for r in optimization_results]
            
            ic_stability = 1.0 / (1.0 + np.var(ic_scores))
            sharpe_stability = 1.0 / (1.0 + np.var(sharpe_scores))
            
            return (ic_stability + sharpe_stability) / 2.0
        except:
            return 0.0
    
    def _detect_recency_bias(self, optimization_results: List[Dict]) -> bool:
        """Detect recency bias in optimization results."""
        try:
            if len(optimization_results) < 3:
                return False
            
            # Check if later windows perform significantly better
            ic_scores = [r['ic'] for r in optimization_results]
            sharpe_scores = [r['sharpe'] for r in optimization_results]
            
            # Recency bias if last half performs much better than first half
            mid_point = len(ic_scores) // 2
            early_ic = np.mean(ic_scores[:mid_point])
            late_ic = np.mean(ic_scores[mid_point:])
            
            early_sharpe = np.mean(sharpe_scores[:mid_point])
            late_sharpe = np.mean(sharpe_scores[mid_point:])
            
            return (late_ic > early_ic * 1.5) or (late_sharpe > early_sharpe * 1.5)
        except:
            return False
    
    def _detect_regime_instability(self, windows: List[OptimizationWindow]) -> bool:
        """Detect regime instability across windows."""
        try:
            unstable_windows = sum(1 for w in windows if not w.regime_stable)
            return unstable_windows > len(windows) * 0.3  # More than 30% unstable
        except:
            return False
    
    def _detect_future_leakage(self, windows: List[OptimizationWindow]) -> bool:
        """Detect future information leakage."""
        try:
            leaked_windows = sum(1 for w in windows if not w.is_valid and "Future information leakage" in w.violations)
            return leaked_windows > 0
        except:
            return False
    
    def _generate_recommendations(self, result: AnchoredOptimizationResult) -> List[str]:
        """Generate recommendations based on optimization results."""
        recommendations = []
        
        if result.recency_bias_detected:
            recommendations.append("Reduce recency bias by increasing embargo periods")
        
        if result.regime_instability_detected:
            recommendations.append("Improve regime detection and stability validation")
        
        if result.future_leakage_detected:
            recommendations.append("Strengthen future information leakage prevention")
        
        if result.stability_score < 0.5:
            recommendations.append("Improve optimization stability across windows")
        
        if result.valid_windows < len(result.windows) * 0.5:
            recommendations.append("Increase valid window count by relaxing constraints")
        
        return recommendations
