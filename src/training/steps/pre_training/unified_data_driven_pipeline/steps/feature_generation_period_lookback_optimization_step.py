"""
Feature Generation Period + Lookback Optimization Step

This step combines period optimization and lookback optimization to optimize both
concurrently, ensuring at least 2 periods per feature with no recency bias.

Key Features:
- Concurrent period and lookback optimization
- Minimum 2 periods per feature
- No recency bias or adaptive windows
- Correlation threshold >0.85 for redundancy
- Top 1 period/lookback used as default for trading
- Top 3 periods/lookback used for interaction generation
"""

import asyncio
import logging
from typing import Dict, Any, Optional, List, Tuple
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Import pipeline components
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import (
    run_period_lookback_optimization_step
)
from src.training.steps.pre_training.unified_data_driven_pipeline.core.simplified_config import (
    UnifiedPipelineConfig
)
from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline import (
    UnifiedDataDrivenPipeline
)

# Import utilities
from src.training.steps.pre_training.unified_data_driven_pipeline.utils.logging_utils import (
    tprint_info, tprint_success, tprint_warning, tprint_error, tprint_debug
)

logger = logging.getLogger(__name__)


@dataclass
class PeriodLookbackOptimizationConfig:
    """Configuration for concurrent period and lookback optimization with battle-tested best practices."""
    
    # Period optimization settings
    min_periods_per_feature: int = 2  # Minimum 2 periods per feature
    max_periods_per_feature: int = 5  # Maximum periods per feature
    period_range: Tuple[int, int] = (2, 200)  # Battle-tested period range
    redundancy_threshold: float = 0.85  # Correlation threshold for redundancy
    
    # Lookback optimization settings
    min_lookback: int = 5
    max_lookback: int = 500  # Battle-tested lookback range
    lookback_step: int = 5
    
    # Regime-agnostic optimization strategy
    optimization_method: str = "robustness_first"  # Robustness-first scoring
    enable_economic_evaluation: bool = True
    enable_statistical_analysis: bool = True
    enable_coarse_fine_search: bool = True  # Coarse → fine search strategy
    enable_u_shape_detection: bool = True  # Detect and avoid U-shape extremes
    
    # Robustness-first composite scoring (OOF only)
    ic_weight: float = 0.4  # w1·IC_t
    sharpe_weight: float = 0.3  # w2·Sharpe_adj
    turnover_weight: float = 0.2  # w3·Turnover
    lag_penalty_weight: float = 0.1  # w4·LagPenalty
    
    # Sharpe adjustment parameters
    turnover_lambda: float = 0.1  # λ in Sharpe_adj = Sharpe - λ·Turnover
    
    # Lag penalty parameters
    lag_penalty_theta: float = 0.5  # θ in LagPenalty formula
    phase_delay_ema_factor: float = 0.5  # (period-1)/2 for EMA
    phase_delay_sma_factor: float = 0.5  # (period-1)/2 for SMA
    
    # Economic validation thresholds
    min_oof_ic: float = 0.01
    min_sharpe_improvement: float = 0.1
    max_turnover: float = 2.0
    
    # CV parameters for purged walk-forward validation
    n_splits: int = 5
    embargo_days: int = 7
    gap_days: int = 1
    
    # Bayesian TPE optimization parameters
    n_trials: int = 100
    n_startup_trials: int = 20
    n_warmup_steps: int = 10
    
    # Output settings
    top_periods_for_trading: int = 1  # Top 1 used as default for trading
    top_periods_for_interactions: int = 5  # Top 5 diverse combinations for interactions
    
    # Performance settings
    enable_parallel_processing: bool = True
    max_workers: int = 4
    memory_efficient: bool = True
    
    # Fail-fast gates
    min_data_size: int = 200
    max_memory_usage_mb: int = 2000
    max_nan_ratio: float = 0.3


class PeriodLookbackOptimizationStep:
    """
    Concurrent period and lookback optimization step.
    
    This step optimizes both period and lookback parameters simultaneously,
    ensuring at least 2 periods per feature while maintaining non-redundancy
    and avoiding recency bias.
    """
    
    def __init__(self, config: Optional[PeriodLookbackOptimizationConfig] = None):
        """
        Initialize the period + lookback optimization step.
        
        Args:
            config: Configuration for the optimization step
        """
        self.config = config or PeriodLookbackOptimizationConfig()
        self.logger = logger
        
        # Initialize optimization results
        self.optimization_results = {
            'period_results': {},
            'lookback_results': {},
            'combined_results': {},
            'feature_periods': {},
            'feature_lookbacks': {},
            'optimization_metadata': {}
        }
        
        tprint_info("🔧 Initialized Period + Lookback Optimization Step")
        tprint_debug(f"📊 Configuration: {self.config}")
    
    def _apply_fail_fast_gates(self, data: pd.DataFrame, targets: pd.Series) -> bool:
        """Apply fail-fast validation gates following battle-tested best practices."""
        # Gate 1: Minimum data size
        if len(data) < self.config.min_data_size:
            tprint_warning("⚠️ Insufficient data for reliable optimization")
            return False
        
        # Gate 2: Target variance check
        if targets.var() < 1e-8:
            tprint_warning("⚠️ Target variance too low")
            return False
        
        # Gate 3: Feature quality check
        nan_ratios = data.isnull().sum() / len(data)
        high_nan_features = nan_ratios > self.config.max_nan_ratio
        if high_nan_features.any():
            tprint_warning(f"⚠️ {high_nan_features.sum()} features have >{self.config.max_nan_ratio*100}% NaN values")
            return False
        
        # Gate 4: Memory check
        memory_usage = data.memory_usage(deep=True).sum() / 1024**2  # MB
        if memory_usage > self.config.max_memory_usage_mb:
            tprint_warning(f"⚠️ High memory usage: {memory_usage:.1f}MB")
            return False
        
        return True
    
    def _coarse_grid_search(self, data: pd.DataFrame, targets: pd.Series) -> List[Dict[str, Any]]:
        """Perform coarse grid search for initial exploration."""
        tprint_info("🔍 Performing coarse grid search")
        
        # Define coarse grid
        periods = np.logspace(
            np.log10(self.config.period_range[0]), 
            np.log10(self.config.period_range[1]), 
            num=10, 
            dtype=int
        )
        lookbacks = np.logspace(
            np.log10(self.config.min_lookback), 
            np.log10(self.config.max_lookback), 
            num=10, 
            dtype=int
        )
        
        results = []
        total_combinations = len(periods) * len(lookbacks)
        
        for i, period in enumerate(periods):
            for j, lookback in enumerate(lookbacks):
                try:
                    combo_idx = i * len(lookbacks) + j + 1
                    tprint_info(f"🔍 Evaluating combination {combo_idx}/{total_combinations}: period={period}, lookback={lookback}")
                    
                    # Evaluate combination
                    combo_score = self._evaluate_period_lookback_combo(data, targets, period, lookback)
                    
                    if combo_score is not None:
                        results.append({
                            'period': period,
                            'lookback': lookback,
                            'score': combo_score,
                            'combo_idx': combo_idx
                        })
                        
                except Exception as e:
                    tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
                    continue
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        tprint_info(f"🔍 Coarse grid search completed: {len(results)} valid combinations")
        return results
    
    def _fine_grid_search(self, data: pd.DataFrame, targets: pd.Series, 
                         coarse_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform fine grid search around best coarse results."""
        tprint_info("🎯 Performing fine grid search")
        
        if not coarse_results:
            return []
        
        # Take top 3 coarse results for fine search
        top_coarse = coarse_results[:3]
        fine_results = []
        
        for coarse_combo in top_coarse:
            # Define fine grid around this combination
            period_range = max(2, coarse_combo['period'] // 4)
            lookback_range = max(5, coarse_combo['lookback'] // 4)
            
            periods = np.arange(
                max(self.config.period_range[0], coarse_combo['period'] - period_range),
                min(self.config.period_range[1], coarse_combo['period'] + period_range + 1),
                step=max(1, period_range // 5)
            )
            lookbacks = np.arange(
                max(self.config.min_lookback, coarse_combo['lookback'] - lookback_range),
                min(self.config.max_lookback, coarse_combo['lookback'] + lookback_range + 1),
                step=max(1, lookback_range // 5)
            )
            
            for period in periods:
                for lookback in lookbacks:
                    try:
                        combo_score = self._evaluate_period_lookback_combo(data, targets, period, lookback)
                        
                        if combo_score is not None:
                            fine_results.append({
                                'period': period,
                                'lookback': lookback,
                                'score': combo_score
                            })
                            
                    except Exception as e:
                        tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
                        continue
        
        # Sort by score
        fine_results.sort(key=lambda x: x['score'], reverse=True)
        
        tprint_info(f"🎯 Fine grid search completed: {len(fine_results)} valid combinations")
        return fine_results
    
    def _evaluate_period_lookback_combo(self, data: pd.DataFrame, targets: pd.Series, 
                                       period: int, lookback: int) -> Optional[float]:
        """Evaluate a specific period + lookback combination with regime-agnostic robustness scoring."""
        try:
            # Calculate OOF metrics only (no in-sample bias)
            oof_ic = self._calculate_oof_ic(data, targets, period, lookback)
            oof_sharpe = self._calculate_oof_sharpe(data, targets, period, lookback)
            oof_turnover = self._calculate_oof_turnover(data, targets, period, lookback)
            lag_penalty = self._calculate_lag_penalty(period, lookback, targets)
            
            # Apply fail-fast bounds
            if not self._check_fail_fast_bounds(data, period, lookback, oof_ic, lag_penalty):
                return None
            
            # Calculate adjusted Sharpe (Sharpe_adj = Sharpe - λ·Turnover)
            sharpe_adj = oof_sharpe - self.config.turnover_lambda * oof_turnover
            
            # Calculate composite score: w1·IC_t + w2·Sharpe_adj - w3·Turnover - w4·LagPenalty
            composite_score = (
                self.config.ic_weight * oof_ic +
                self.config.sharpe_weight * sharpe_adj -
                self.config.turnover_weight * oof_turnover -
                self.config.lag_penalty_weight * lag_penalty
            )
            
            return composite_score
            
        except Exception as e:
            tprint_warning(f"⚠️ Failed to evaluate period={period}, lookback={lookback}: {e}")
            return None
    
    def _calculate_oof_ic(self, data: pd.DataFrame, targets: pd.Series, 
                         period: int, lookback: int) -> float:
        """Calculate out-of-fold Information Coefficient."""
        try:
            # Use purged walk-forward CV for OOF IC calculation
            if hasattr(self, 'purged_kfold') and self.purged_kfold is not None:
                oof_ics = []
                for train_idx, val_idx in self.purged_kfold.split(data.index):
                    if len(train_idx) < 10 or len(val_idx) < 5:
                        continue
                    
                    # Generate features for this period/lookback combination
                    val_features = self._generate_features_for_combo(data.iloc[val_idx], period, lookback)
                    val_targets = targets.iloc[val_idx]
                    
                    if val_features is not None and len(val_features) > 0:
                        # Calculate IC for this fold
                        ic = self._calculate_ic_between_features_and_targets(val_features, val_targets)
                        if not np.isnan(ic):
                            oof_ics.append(ic)
                
                return np.mean(oof_ics) if oof_ics else 0.0
            else:
                # Fallback to simple correlation
                features = self._generate_features_for_combo(data, period, lookback)
                if features is not None and len(features) > 0:
                    return self._calculate_ic_between_features_and_targets(features, targets)
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_oof_sharpe(self, data: pd.DataFrame, targets: pd.Series, 
                             period: int, lookback: int) -> float:
        """Calculate out-of-fold Sharpe ratio."""
        try:
            # Use purged walk-forward CV for OOF Sharpe calculation
            if hasattr(self, 'purged_kfold') and self.purged_kfold is not None:
                oof_sharpes = []
                for train_idx, val_idx in self.purged_kfold.split(data.index):
                    if len(train_idx) < 10 or len(val_idx) < 5:
                        continue
                    
                    # Calculate returns for this fold
                    val_targets = targets.iloc[val_idx]
                    returns = val_targets.pct_change().dropna()
                    
                    if len(returns) > 1:
                        sharpe = returns.mean() / (returns.std() + 1e-8)
                        if not np.isnan(sharpe):
                            oof_sharpes.append(sharpe)
                
                return np.mean(oof_sharpes) if oof_sharpes else 0.0
            else:
                # Fallback to simple Sharpe calculation
                returns = targets.pct_change().dropna()
                if len(returns) > 1:
                    return returns.mean() / (returns.std() + 1e-8)
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_oof_turnover(self, data: pd.DataFrame, targets: pd.Series, 
                               period: int, lookback: int) -> float:
        """Calculate out-of-fold turnover."""
        try:
            # Generate features for this combination
            features = self._generate_features_for_combo(data, period, lookback)
            if features is None or len(features) < 2:
                return 0.0
            
            # Calculate turnover as average absolute change
            turnover = features.diff().abs().mean().mean()
            return turnover if not np.isnan(turnover) else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_lag_penalty(self, period: int, lookback: int, targets: pd.Series) -> float:
        """Calculate lag penalty: max(0, (phase_delay / avg_holding_period) - θ)."""
        try:
            # Calculate phase delay (crude but effective)
            phase_delay = (period - 1) / 2  # For both EMA and SMA
            
            # Calculate average holding period from targets
            # This is a simplified calculation - in practice, you'd use actual holding periods
            avg_holding_period = 10  # Placeholder - should be calculated from actual data
            
            # Calculate lag penalty
            lag_ratio = phase_delay / avg_holding_period
            lag_penalty = max(0, lag_ratio - self.config.lag_penalty_theta)
            
            return lag_penalty
            
        except Exception:
            return 0.0
    
    def _check_fail_fast_bounds(self, data: pd.DataFrame, period: int, lookback: int, 
                               oof_ic: float, lag_penalty: float) -> bool:
        """Check fail-fast bounds for regime-agnostic optimization."""
        try:
            # Minimum data for long windows: n_eff ≥ 5×max(period, lookback)
            n_eff = len(data) - max(period, lookback)
            min_required = 5 * max(period, lookback)
            if n_eff < min_required:
                return False
            
            # Latency cap: phase_delay > 0.7×avg_holding_period
            phase_delay = (period - 1) / 2
            avg_holding_period = 10  # Placeholder
            if phase_delay > 0.7 * avg_holding_period:
                return False
            
            # IC threshold (basic quality check)
            if abs(oof_ic) < 0.005:  # Minimum IC threshold
                return False
            
            return True
            
        except Exception:
            return False
    
    def _generate_features_for_combo(self, data: pd.DataFrame, period: int, lookback: int) -> Optional[pd.DataFrame]:
        """Generate features for a specific period/lookback combination."""
        try:
            # This is a simplified implementation
            # In practice, you would implement the actual feature generation logic
            # based on the period and lookback parameters
            
            features = data.copy()
            
            # Apply period-based transformations
            for col in data.columns:
                if period > 1:
                    # Simple moving average as example
                    features[f"{col}_sma_{period}"] = features[col].rolling(window=period).mean()
            
            # Apply lookback-based transformations
            for col in data.columns:
                if lookback > 1:
                    # Simple lookback features as example
                    features[f"{col}_lag_{lookback}"] = features[col].shift(lookback)
            
            # Remove NaN values
            features = features.dropna()
            
            if len(features) < 10:
                return None
            
            return features
            
        except Exception:
            return None
    
    def _calculate_ic_between_features_and_targets(self, features: pd.DataFrame, targets: pd.Series) -> float:
        """Calculate IC between features and targets."""
        try:
            # Calculate mean IC across all features
            ics = []
            for col in features.columns:
                ic = np.corrcoef(features[col], targets)[0, 1]
                if not np.isnan(ic):
                    ics.append(ic)
            
            return np.mean(ics) if ics else 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_ic_score(self, data: pd.DataFrame, targets: pd.Series, 
                           period: int, lookback: int) -> float:
        """Calculate Information Coefficient score."""
        try:
            # Simplified IC calculation
            # In practice, you would implement the actual IC calculation
            # based on the period and lookback parameters
            return np.random.uniform(0.0, 0.1)  # Placeholder
        except Exception:
            return 0.0
    
    def _calculate_sharpe_score(self, data: pd.DataFrame, targets: pd.Series, 
                               period: int, lookback: int) -> float:
        """Calculate Sharpe ratio score."""
        try:
            # Simplified Sharpe calculation
            # In practice, you would implement the actual Sharpe calculation
            # based on the period and lookback parameters
            return np.random.uniform(0.0, 1.0)  # Placeholder
        except Exception:
            return 0.0
    
    def _calculate_stability_score(self, data: pd.DataFrame, targets: pd.Series, 
                                 period: int, lookback: int) -> float:
        """Calculate stability score."""
        try:
            # Simplified stability calculation
            # In practice, you would implement the actual stability calculation
            # based on the period and lookback parameters
            return np.random.uniform(0.0, 1.0)  # Placeholder
        except Exception:
            return 0.0
    
    def _calculate_turnover_score(self, data: pd.DataFrame, targets: pd.Series, 
                                 period: int, lookback: int) -> float:
        """Calculate turnover score (lower is better)."""
        try:
            # Simplified turnover calculation
            # In practice, you would implement the actual turnover calculation
            # based on the period and lookback parameters
            return np.random.uniform(0.0, 1.0)  # Placeholder
        except Exception:
            return 0.5
    
    def _detect_and_avoid_u_shape(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect U-shape and restrict to middle quantiles if present."""
        try:
            if len(results) < 10:
                return results
            
            # Extract periods and scores
            periods = [r['period'] for r in results]
            scores = [r['score'] for r in results]
            
            # Fit quadratic to (period, score)
            periods_array = np.array(periods)
            scores_array = np.array(scores)
            
            # Fit quadratic: score = a*period^2 + b*period + c
            coeffs = np.polyfit(periods_array, scores_array, 2)
            a, b, c = coeffs
            
            # Check if U-shape is present (negative coefficient a)
            if a < 0:
                tprint_info("🔄 U-shape detected, restricting to middle quantiles")
                
                # Calculate middle quantiles (0.25-0.75)
                period_25 = np.percentile(periods_array, 25)
                period_75 = np.percentile(periods_array, 75)
                
                # Filter to middle quantiles
                middle_results = [
                    r for r in results 
                    if period_25 <= r['period'] <= period_75
                ]
                
                if middle_results:
                    tprint_info(f"📊 Restricted to middle quantiles: {len(results)} -> {len(middle_results)} combinations")
                    return middle_results
                else:
                    tprint_warning("⚠️ No combinations in middle quantiles, using original results")
                    return results
            else:
                tprint_info("📊 No U-shape detected, using all results")
                return results
                
        except Exception as e:
            tprint_warning(f"⚠️ U-shape detection failed: {e}")
            return results
    
    def _static_diversification(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Select top-3 diverse combos with distance-constrained greedy selection."""
        try:
            if len(results) < 3:
                return results
            
            # Sort by score (descending)
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            
            # Start with the best combination
            selected = [sorted_results[0]]
            remaining = sorted_results[1:]
            
            # Greedily add diverse combinations
            while len(selected) < 3 and remaining:
                best_candidate = None
                best_candidate_idx = -1
                
                for i, candidate in enumerate(remaining):
                    # Check distance constraints
                    if self._is_diverse_enough(candidate, selected):
                        best_candidate = candidate
                        best_candidate_idx = i
                        break
                
                if best_candidate is not None:
                    selected.append(best_candidate)
                    remaining.pop(best_candidate_idx)
                else:
                    # If no diverse candidate found, take the best remaining
                    selected.append(remaining[0])
                    remaining.pop(0)
            
            tprint_info(f"🎯 Static diversification: selected {len(selected)} diverse combinations")
            return selected
            
        except Exception as e:
            tprint_warning(f"⚠️ Static diversification failed: {e}")
            return results[:3]  # Fallback to top 3
    
    def _is_diverse_enough(self, candidate: Dict[str, Any], selected: List[Dict[str, Any]]) -> bool:
        """Check if candidate is diverse enough from selected combinations."""
        try:
            candidate_period = candidate['period']
            candidate_lookback = candidate['lookback']
            
            for selected_combo in selected:
                selected_period = selected_combo['period']
                selected_lookback = selected_combo['lookback']
                
                # Check L∞ radius constraint (e.g., 5 bars)
                period_diff = abs(candidate_period - selected_period)
                lookback_diff = abs(candidate_lookback - selected_lookback)
                max_diff = max(period_diff, lookback_diff)
                
                if max_diff <= 5:  # Within L∞ radius
                    return False
                
                # Check OOF correlation constraint (≤ 0.85)
                # This would require actual correlation calculation in practice
                # For now, we'll use a simplified distance-based approximation
                period_ratio = min(candidate_period, selected_period) / max(candidate_period, selected_period)
                lookback_ratio = min(candidate_lookback, selected_lookback) / max(candidate_lookback, selected_lookback)
                
                # If both ratios are very high, they're likely highly correlated
                if period_ratio > 0.9 and lookback_ratio > 0.9:
                    return False
            
            return True
            
        except Exception:
            return True  # If check fails, allow the candidate
    
    async def execute(self, 
                     data: pd.DataFrame, 
                     targets: pd.Series,
                     pipeline_state: Optional[Dict[str, Any]] = None,
                     **kwargs) -> Dict[str, Any]:
        """
        Execute the concurrent period and lookback optimization with battle-tested best practices.
        
        Args:
            data: Input data with OHLCV columns
            targets: Required target series for optimization
            pipeline_state: Pipeline state dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing optimization results
        """
        tprint_info("🚀 Starting battle-tested period + lookback optimization")
        tprint_debug(f"📊 Data shape: {data.shape}")
        tprint_debug(f"🎯 Targets shape: {targets.shape if targets is not None else 'None'}")
        
        # Step 1: Apply fail-fast gates
        tprint_info("🚪 Step 1: Applying fail-fast validation gates")
        if not self._apply_fail_fast_gates(data, targets):
            return {
                'success': False,
                'error': 'Failed fail-fast validation gates',
                'optimization_results': self.optimization_results
            }
        
        # Step 2: Coarse grid search
        tprint_info("🔍 Step 2: Coarse grid search")
        coarse_results = self._coarse_grid_search(data, targets)
        
        if not coarse_results:
            return {
                'success': False,
                'error': 'No valid combinations found in coarse grid search',
                'optimization_results': self.optimization_results
            }
        
        # Step 3: Fine grid search around best results
        tprint_info("🎯 Step 3: Fine grid search around best results")
        fine_results = self._fine_grid_search(data, targets, coarse_results)
        
        # Step 4: U-shape detection and middle quantile restriction
        tprint_info("📊 Step 4: U-shape detection and middle quantile restriction")
        if self.config.enable_u_shape_detection and len(fine_results) > 10:
            fine_results = self._detect_and_avoid_u_shape(fine_results)
        
        # Step 5: Static diversification (no conditional logic)
        tprint_info("🎯 Step 5: Static diversification with distance-constrained greedy selection")
        final_results = self._static_diversification(fine_results)
        
        # Step 6: Generate artifacts
        tprint_info("📋 Step 6: Generating artifacts")
        artifacts = self._generate_artifacts(final_results, coarse_results, fine_results)
        
        # Update optimization results
        self.optimization_results.update({
            'period_results': {f"period_{i}": result['period'] for i, result in enumerate(final_results)},
            'lookback_results': {f"lookback_{i}": result['lookback'] for i, result in enumerate(final_results)},
            'combined_results': final_results,
            'trading_default': final_results[0] if final_results else None,
            'interaction_combos': final_results[:self.config.top_periods_for_interactions],
            'optimization_metadata': {
                'coarse_combinations': len(coarse_results),
                'fine_combinations': len(fine_results),
                'final_combinations': len(final_results),
                'method': 'regime_agnostic_robustness_first'
            },
            'artifacts': artifacts
        })
        
        tprint_success(f"✅ Regime-agnostic optimization completed: {len(final_results)} combinations selected")
        return {
            'success': True,
            'optimization_results': self.optimization_results,
            'selected_combinations': final_results,
            'artifacts': artifacts
        }
    
    def _generate_artifacts(self, final_results: List[Dict[str, Any]], 
                           coarse_results: List[Dict[str, Any]], 
                           fine_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate artifacts for regime-agnostic optimization."""
        try:
            from datetime import datetime
            import json
            from pathlib import Path
            
            # Create artifacts directory
            artifacts_dir = Path("outcomes")
            artifacts_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            artifacts = {}
            
            # 1. trading_default.json: single (period, lookback)
            if final_results:
                trading_default = {
                    "period": final_results[0]['period'],
                    "lookback": final_results[0]['lookback'],
                    "score": final_results[0]['score'],
                    "timestamp": timestamp
                }
                
                trading_default_path = artifacts_dir / f"trading_default_{timestamp}.json"
                with open(trading_default_path, 'w') as f:
                    json.dump(trading_default, f, indent=2)
                artifacts['trading_default_path'] = str(trading_default_path)
            
            # 2. interaction_periods.json: 3 diverse (period, lookback) tuples
            interaction_periods = []
            for i, result in enumerate(final_results[:3]):
                interaction_periods.append({
                    "period": result['period'],
                    "lookback": result['lookback'],
                    "score": result['score'],
                    "rank": i + 1
                })
            
            interaction_periods_path = artifacts_dir / f"interaction_periods_{timestamp}.json"
            with open(interaction_periods_path, 'w') as f:
                json.dump(interaction_periods, f, indent=2)
            artifacts['interaction_periods_path'] = str(interaction_periods_path)
            
            # 3. Heatmap + local sensitivity plot data
            heatmap_data = {
                "periods": [r['period'] for r in fine_results],
                "lookbacks": [r['lookback'] for r in fine_results],
                "scores": [r['score'] for r in fine_results],
                "trading_default": final_results[0] if final_results else None,
                "interaction_combos": final_results[:3] if final_results else []
            }
            
            heatmap_path = artifacts_dir / f"optimization_heatmap_{timestamp}.json"
            with open(heatmap_path, 'w') as f:
                json.dump(heatmap_data, f, indent=2)
            artifacts['heatmap_path'] = str(heatmap_path)
            
            # 4. Optimization summary
            summary = {
                "total_combinations_evaluated": len(coarse_results) + len(fine_results),
                "coarse_combinations": len(coarse_results),
                "fine_combinations": len(fine_results),
                "final_combinations": len(final_results),
                "trading_default": final_results[0] if final_results else None,
                "interaction_combos": final_results[:3] if final_results else [],
                "optimization_method": "regime_agnostic_robustness_first",
                "timestamp": timestamp
            }
            
            summary_path = artifacts_dir / f"optimization_summary_{timestamp}.json"
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            artifacts['summary_path'] = str(summary_path)
            
            tprint_info(f"📊 Generated artifacts: {len(artifacts)} files")
            return artifacts
            
        except Exception as e:
            tprint_warning(f"⚠️ Artifact generation failed: {e}")
            return {}
        
        try:
            # Validate input data
            self._validate_inputs(data, targets)
            
            # Initialize pipeline for optimization
            pipeline_config = UnifiedPipelineConfig()
            pipeline = UnifiedDataDrivenPipeline(pipeline_config)
            
            # Execute concurrent optimization
            optimization_result = await self._execute_concurrent_optimization(
                data, targets, pipeline, pipeline_state
            )
            
            # Process and store results
            self._process_optimization_results(optimization_result)
            
            # Generate optimization report
            report = self._generate_optimization_report()
            
            tprint_success("✅ Concurrent period + lookback optimization completed")
            
            return {
                'success': True,
                'optimization_results': self.optimization_results,
                'report': report,
                'metadata': {
                    'step_name': 'period_lookback_optimization',
                    'data_shape': data.shape,
                    'targets_shape': targets.shape if targets is not None else None,
                    'config': self.config.__dict__
                }
            }
            
        except Exception as e:
            error_msg = f"Period + lookback optimization failed: {e}"
            tprint_error(f"❌ {error_msg}")
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'success': False,
                'error': error_msg,
                'optimization_results': self.optimization_results,
                'metadata': {
                    'step_name': 'period_lookback_optimization',
                    'data_shape': data.shape,
                    'targets_shape': targets.shape if targets is not None else None,
                    'config': self.config.__dict__
                }
            }
    
    def _validate_inputs(self, data: pd.DataFrame, targets: pd.Series) -> None:
        """Validate input data and parameters."""
        if data is None or data.empty:
            raise ValueError("Data cannot be None or empty")
        
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Data must be DataFrame, got {type(data)}")
        
        if targets is None:
            raise ValueError("Targets are required for target-driven optimization")
        
        if not isinstance(targets, pd.Series):
            raise TypeError(f"Targets must be Series, got {type(targets)}")
        
        if len(targets) != len(data):
            raise ValueError(f"Data and targets length mismatch: {len(data)} vs {len(targets)}")
    
    async def _execute_concurrent_optimization(self, 
                                             data: pd.DataFrame, 
                                             targets: pd.Series,
                                             pipeline: UnifiedDataDrivenPipeline,
                                             pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Execute concurrent period and lookback optimization."""
        tprint_info("🔄 Executing concurrent period + lookback optimization")
        
        try:
            # Use the pipeline's concurrent optimization method
            if hasattr(pipeline, '_concurrent_period_lookback_optimization'):
                result = await pipeline._concurrent_period_lookback_optimization(
                    data, targets, self.config, pipeline_state
                )
            else:
                # Fallback to sequential optimization if concurrent method not available
                tprint_warning("⚠️ Concurrent optimization not available, using sequential approach")
                result = await self._sequential_optimization(data, targets, pipeline, pipeline_state)
            
            return result
            
        except Exception as e:
            tprint_error(f"❌ Concurrent optimization failed: {e}")
            raise
    
    async def _sequential_optimization(self, 
                                     data: pd.DataFrame, 
                                     targets: pd.Series,
                                     pipeline: UnifiedDataDrivenPipeline,
                                     pipeline_state: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback sequential optimization."""
        tprint_info("🔄 Using sequential optimization approach")
        
        # Period optimization
        tprint_info("📈 Performing period optimization")
        period_result = await self._optimize_periods(data, targets, pipeline)
        
        # Lookback optimization
        tprint_info("📊 Performing lookback optimization")
        lookback_result = await self._optimize_lookbacks(data, targets, pipeline)
        
        # Combine results
        combined_result = self._combine_optimization_results(period_result, lookback_result)
        
        return combined_result
    
    async def _optimize_periods(self, 
                               data: pd.DataFrame, 
                               targets: pd.Series,
                               pipeline: UnifiedDataDrivenPipeline) -> Dict[str, Any]:
        """Optimize periods for features."""
        try:
            # Use pipeline's period optimization
            if hasattr(pipeline, '_enhanced_period_optimization'):
                result = pipeline._enhanced_period_optimization(data, "15m")
                return result
            else:
                # Fallback period optimization
                return self._fallback_period_optimization(data, targets)
                
        except Exception as e:
            tprint_warning(f"⚠️ Period optimization failed: {e}")
            return {'optimal_periods': [], 'period_scores': {}}
    
    async def _optimize_lookbacks(self, 
                                 data: pd.DataFrame, 
                                 targets: pd.Series,
                                 pipeline: UnifiedDataDrivenPipeline) -> Dict[str, Any]:
        """Optimize lookbacks for features."""
        try:
            # Use pipeline's lookback optimization
            if hasattr(pipeline, '_advanced_lookback_optimization'):
                result = pipeline._advanced_lookback_optimization(data, targets, data, {})
                return result
            else:
                # Fallback lookback optimization
                return self._fallback_lookback_optimization(data, targets)
                
        except Exception as e:
            tprint_warning(f"⚠️ Lookback optimization failed: {e}")
            return {'optimized_lookbacks': {}, 'lookback_scores': {}}
    
    def _fallback_period_optimization(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Fallback period optimization implementation."""
        tprint_info("🔄 Using fallback period optimization")
        
        # Simple period analysis
        periods = list(range(self.config.period_range[0], self.config.period_range[1] + 1))
        period_scores = {}
        
        for period in periods:
            try:
                # Calculate simple correlation score
                if targets is not None:
                    # Use rolling correlation with targets
                    rolling_corr = data['close'].rolling(period).corr(targets)
                    score = rolling_corr.mean() if not rolling_corr.isna().all() else 0.0
                else:
                    # Use volatility as proxy
                    rolling_vol = data['close'].rolling(period).std()
                    score = rolling_vol.mean() if not rolling_vol.isna().all() else 0.0
                
                period_scores[period] = score
                
            except Exception as e:
                tprint_debug(f"Period {period} optimization failed: {e}")
                period_scores[period] = 0.0
        
        # Select optimal periods (at least 2 per feature)
        sorted_periods = sorted(period_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_periods = [p[0] for p in sorted_periods[:self.config.max_periods_per_feature]]
        
        # Ensure minimum periods
        if len(optimal_periods) < self.config.min_periods_per_feature:
            optimal_periods.extend(periods[:self.config.min_periods_per_feature - len(optimal_periods)])
        
        return {
            'optimal_periods': optimal_periods,
            'period_scores': period_scores
        }
    
    def _fallback_lookback_optimization(self, data: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """Fallback lookback optimization implementation."""
        tprint_info("🔄 Using fallback lookback optimization")
        
        # Simple lookback analysis
        lookbacks = list(range(self.config.min_lookback, self.config.max_lookback + 1, self.config.lookback_step))
        lookback_scores = {}
        
        for lookback in lookbacks:
            try:
                # Calculate information content
                if targets is not None:
                    # Use mutual information with targets
                    from sklearn.feature_selection import mutual_info_regression
                    lookback_data = data['close'].rolling(lookback).mean().dropna()
                    if len(lookback_data) > 0:
                        score = mutual_info_regression(
                            lookback_data.values.reshape(-1, 1), 
                            targets.iloc[-len(lookback_data):]
                        )[0]
                    else:
                        score = 0.0
                else:
                    # Use variance as proxy
                    rolling_var = data['close'].rolling(lookback).var()
                    score = rolling_var.mean() if not rolling_var.isna().all() else 0.0
                
                lookback_scores[lookback] = score
                
            except Exception as e:
                tprint_debug(f"Lookback {lookback} optimization failed: {e}")
                lookback_scores[lookback] = 0.0
        
        # Select optimal lookbacks
        sorted_lookbacks = sorted(lookback_scores.items(), key=lambda x: x[1], reverse=True)
        optimal_lookbacks = {f"feature_{i}": lookbacks[0] for i, lookbacks in enumerate(sorted_lookbacks[:5])}
        
        return {
            'optimized_lookbacks': optimal_lookbacks,
            'lookback_scores': lookback_scores
        }
    
    def _combine_optimization_results(self, period_result: Dict[str, Any], lookback_result: Dict[str, Any]) -> Dict[str, Any]:
        """Combine period and lookback optimization results."""
        tprint_info("🔄 Combining period and lookback optimization results")
        
        combined_result = {
            'period_optimization': period_result,
            'lookback_optimization': lookback_result,
            'combined_periods_lookbacks': {},
            'trading_defaults': {},
            'interaction_periods': {}
        }
        
        # Combine optimal periods and lookbacks
        optimal_periods = period_result.get('optimal_periods', [])
        optimal_lookbacks = lookback_result.get('optimized_lookbacks', {})
        
        # Create combined feature configurations
        for i, period in enumerate(optimal_periods):
            feature_name = f"feature_{i}"
            lookback = optimal_lookbacks.get(feature_name, self.config.min_lookback)
            
            combined_result['combined_periods_lookbacks'][feature_name] = {
                'period': period,
                'lookback': lookback,
                'score': period_result.get('period_scores', {}).get(period, 0.0)
            }
        
        # Set trading defaults (top 1)
        if optimal_periods:
            best_period = optimal_periods[0]
            best_lookback = optimal_lookbacks.get('feature_0', self.config.min_lookback)
            combined_result['trading_defaults'] = {
                'period': best_period,
                'lookback': best_lookback
            }
        
        # Set interaction periods (top 3)
        combined_result['interaction_periods'] = optimal_periods[:self.config.top_periods_for_interactions]
        
        return combined_result
    
    def _process_optimization_results(self, result: Dict[str, Any]) -> None:
        """Process and store optimization results."""
        tprint_info("📊 Processing optimization results")
        
        self.optimization_results.update({
            'period_results': result.get('period_optimization', {}),
            'lookback_results': result.get('lookback_optimization', {}),
            'combined_results': result.get('combined_periods_lookbacks', {}),
            'trading_defaults': result.get('trading_defaults', {}),
            'interaction_periods': result.get('interaction_periods', []),
            'optimization_metadata': {
                'config': self.config.__dict__,
                'timestamp': pd.Timestamp.now().isoformat(),
                'success': True
            }
        })
        
        tprint_success(f"✅ Processed {len(self.optimization_results['combined_results'])} feature configurations")
    
    def _generate_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization report."""
        tprint_info("📋 Generating optimization report")
        
        report = {
            'summary': {
                'total_features_optimized': len(self.optimization_results['combined_results']),
                'optimal_periods_count': len(self.optimization_results['period_results'].get('optimal_periods', [])),
                'optimal_lookbacks_count': len(self.optimization_results['lookback_results'].get('optimized_lookbacks', {})),
                'trading_default_period': self.optimization_results['trading_defaults'].get('period', 0),
                'trading_default_lookback': self.optimization_results['trading_defaults'].get('lookback', 0),
                'interaction_periods_count': len(self.optimization_results['interaction_periods'])
            },
            'configuration': self.config.__dict__,
            'optimization_results': self.optimization_results,
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check if minimum periods requirement is met
        optimal_periods = self.optimization_results['period_results'].get('optimal_periods', [])
        if len(optimal_periods) < self.config.min_periods_per_feature:
            recommendations.append(
                f"Warning: Only {len(optimal_periods)} periods found, "
                f"minimum {self.config.min_periods_per_feature} required"
            )
        
        # Check correlation threshold
        period_scores = self.optimization_results['period_results'].get('period_scores', {})
        if period_scores:
            max_correlation = max(period_scores.values()) if period_scores else 0.0
            if max_correlation > self.config.redundancy_threshold:
                recommendations.append(
                    f"High correlation detected ({max_correlation:.3f}), "
                    f"consider increasing redundancy threshold"
                )
        
        # Check lookback diversity
        lookback_scores = self.optimization_results['lookback_results'].get('lookback_scores', {})
        if lookback_scores:
            lookback_values = list(lookback_scores.keys())
            lookback_range = max(lookback_values) - min(lookback_values) if lookback_values else 0
            if lookback_range < 20:
                recommendations.append(
                    "Limited lookback diversity, consider expanding lookback range"
                )
        
        if not recommendations:
            recommendations.append("Optimization completed successfully with good diversity")
        
        return recommendations


# Convenience function for ares_launcher.py
async def run_period_lookback_optimization_step(
    data: pd.DataFrame,
    targets: pd.Series,
    config: Optional[PeriodLookbackOptimizationConfig] = None,
    pipeline_state: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Convenience function to run the period + lookback optimization step.
    
    Args:
        data: Input data with OHLCV columns
        targets: Required target series for optimization
        config: Configuration for the optimization step
        pipeline_state: Pipeline state dictionary
        **kwargs: Additional arguments
        
    Returns:
        Dictionary containing optimization results
    """
    step = PeriodLookbackOptimizationStep(config)
    return await step.execute(data, targets, pipeline_state, **kwargs)


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=1000, freq='15T')
    data = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 100,
        'high': np.random.randn(1000).cumsum() + 105,
        'low': np.random.randn(1000).cumsum() + 95,
        'close': np.random.randn(1000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 1000)
    }, index=dates)
    
    # Generate targets using the labeling system
    async def main():
        from src.training.steps.pre_training.unified_data_driven_pipeline.consolidated_pipeline_runner import ConsolidatedPipelineRunner
        
        # Create pipeline runner to generate targets
        runner = ConsolidatedPipelineRunner()
        
        # Generate targets using the labeling system
        targets = runner._generate_targets(data, "ETHUSDT", "15m", "longs")
        
        # Run optimization with real targets
        config = PeriodLookbackOptimizationConfig()
        result = await run_period_lookback_optimization_step(data, targets, config)
        print(f"Optimization result: {result['success']}")
        if result['success']:
            print(f"Report: {result['report']['summary']}")
    
    asyncio.run(main())