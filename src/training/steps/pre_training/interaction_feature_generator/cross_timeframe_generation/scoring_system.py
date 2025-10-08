"""
Adaptive Penalized OOS Scoring System

Implements adaptive penalized out-of-sample scoring with:
- Regime-aware scoring with uncertainty estimation
- Wild/bootstrap standard errors for heavy tails
- Cost-aware scoring with CPU and staleness penalties
- Meta-learning of penalty parameters
- Fold aggregation with regime weighting
"""

from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from scipy import stats
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

# Import tprint utilities for structured logging while remaining compatible with
# environments where the enhanced printer is unavailable.
try:
    from src.utils.tprint import (
        tprint,
        tprint_debug,
        tprint_info,
        tprint_warning,
        tprint_error,
    )
    TPRINT_AVAILABLE = True
except ImportError:  # pragma: no cover - fallback for isolated test execution
    TPRINT_AVAILABLE = False

    def tprint(*args, **kwargs):  # type: ignore
        print(*args, **kwargs)

    def tprint_debug(*args, **kwargs):  # type: ignore
        print("DEBUG:", *args, **kwargs)

    def tprint_info(*args, **kwargs):  # type: ignore
        print("INFO:", *args, **kwargs)

    def tprint_warning(*args, **kwargs):  # type: ignore
        print("WARNING:", *args, **kwargs)

    def tprint_error(*args, **kwargs):  # type: ignore
        print("ERROR:", *args, **kwargs)

from .staleness_curve import StalenessCurveCalculator
from .config import ScoringConfig, SessionConfig


@dataclass
class ScoringResult:
    """Result of scoring a feature candidate."""
    feature_name: str
    lookback: int
    regime: str
    ic_oos: float
    se_wild_bootstrap: float
    se_stationary_bootstrap: float
    cpu_p95: float
    staleness: float
    utility_score: float
    fold_pass_rate: float
    regime_weight: float
    metadata: Dict[str, Any]


class UncertaintyEstimator:
    """Estimates uncertainty using wild/bootstrap methods."""
    
    def __init__(self, n_bootstrap: int = 100, block_size: Optional[int] = None):
        self.n_bootstrap = n_bootstrap
        self.block_size = block_size
        self.logger = logging.getLogger(__name__)
        tprint_debug(
            "Initialized UncertaintyEstimator",
            {
                "n_bootstrap": self.n_bootstrap,
                "block_size": self.block_size,
            },
        )
    
    def estimate_wild_bootstrap_se(self, 
                                 feature: pd.Series, 
                                 target: pd.Series,
                                 regime_segments: Optional[List[Any]] = None) -> float:
        """
        Estimate standard error using wild bootstrap.
        
        Args:
            feature: Feature series
            target: Target series
            regime_segments: Regime segments for regime-aware bootstrapping
            
        Returns:
            Wild bootstrap standard error
        """
        if len(feature) < 10:
            tprint_warning(
                "[UncertaintyEstimator] Insufficient samples for wild bootstrap; using default.",
                {"length": len(feature)},
            )
            return 1.0

        # Align data
        aligned_data = pd.DataFrame({
            'feature': feature,
            'target': target
        }).dropna()

        if len(aligned_data) < 10:
            tprint_warning(
                "[UncertaintyEstimator] Insufficient aligned samples for wild bootstrap; using default.",
                {"length": len(aligned_data)},
            )
            return 1.0

        feature_vals = aligned_data['feature'].values
        target_vals = aligned_data['target'].values

        # Calculate original IC
        original_ic = np.corrcoef(feature_vals, target_vals)[0, 1]

        # Wild bootstrap with Rademacher weights
        bootstrap_ics = []

        tprint_debug(
            "[UncertaintyEstimator] Starting wild bootstrap estimation",
            {
                "samples": len(feature_vals),
                "regime_segments": bool(regime_segments),
                "original_ic": float(original_ic) if not np.isnan(original_ic) else None,
            },
        )

        for _ in range(self.n_bootstrap):
            # Generate Rademacher weights
            weights = np.random.choice([-1, 1], size=len(feature_vals))
            
            # Apply weights to residuals
            if regime_segments:
                # Regime-aware wild bootstrap
                weighted_feature = self._apply_regime_aware_weights(
                    feature_vals, weights, regime_segments
                )
            else:
                # Standard wild bootstrap
                weighted_feature = feature_vals * weights
            
            # Calculate IC with weighted feature
            try:
                ic = np.corrcoef(weighted_feature, target_vals)[0, 1]
                if not np.isnan(ic):
                    bootstrap_ics.append(ic)
            except (ValueError, TypeError, RuntimeWarning):
                # Handle numerical computation issues (e.g., insufficient data, NaN values)
                continue
        
        if not bootstrap_ics:
            tprint_warning(
                "[UncertaintyEstimator] Wild bootstrap produced no valid samples; using default.",
                {"iterations": self.n_bootstrap},
            )
            return 1.0

        se = float(np.std(bootstrap_ics))
        tprint_debug(
            "[UncertaintyEstimator] Wild bootstrap SE computed",
            {"se": se, "valid_iterations": len(bootstrap_ics)},
        )
        return se
    
    def estimate_stationary_bootstrap_se(self, 
                                       feature: pd.Series, 
                                       target: pd.Series,
                                       block_size: Optional[int] = None) -> float:
        """
        Estimate standard error using stationary block bootstrap.
        
        Args:
            feature: Feature series
            target: Target series
            block_size: Block size for bootstrap (auto if None)
            
        Returns:
            Stationary bootstrap standard error
        """
        if len(feature) < 20:
            tprint_warning(
                "[UncertaintyEstimator] Insufficient samples for stationary bootstrap; using default.",
                {"length": len(feature)},
            )
            return 1.0
        
        # Align data
        aligned_data = pd.DataFrame({
            'feature': feature,
            'target': target
        }).dropna()
        
        if len(aligned_data) < 20:
            tprint_warning(
                "[UncertaintyEstimator] Insufficient aligned samples for stationary bootstrap; using default.",
                {"length": len(aligned_data)},
            )
            return 1.0
        
        feature_vals = aligned_data['feature'].values
        target_vals = aligned_data['target'].values
        
        # Auto-determine block size if not provided
        if block_size is None:
            block_size = max(5, int(np.sqrt(len(feature_vals))))

        # Calculate original IC
        original_ic = np.corrcoef(feature_vals, target_vals)[0, 1]

        # Stationary block bootstrap
        bootstrap_ics = []

        tprint_debug(
            "[UncertaintyEstimator] Starting stationary bootstrap estimation",
            {
                "samples": len(feature_vals),
                "block_size": block_size,
                "original_ic": float(original_ic) if not np.isnan(original_ic) else None,
            },
        )

        for _ in range(self.n_bootstrap):
            # Generate bootstrap sample
            bootstrap_feature, bootstrap_target = self._stationary_bootstrap(
                feature_vals, target_vals, block_size
            )
            
            # Calculate IC
            try:
                ic = np.corrcoef(bootstrap_feature, bootstrap_target)[0, 1]
                if not np.isnan(ic):
                    bootstrap_ics.append(ic)
            except (ValueError, TypeError, RuntimeWarning):
                # Handle numerical computation issues (e.g., insufficient data, NaN values)
                continue
        
        if not bootstrap_ics:
            tprint_warning(
                "[UncertaintyEstimator] Stationary bootstrap produced no valid samples; using default.",
                {"iterations": self.n_bootstrap},
            )
            return 1.0

        se = float(np.std(bootstrap_ics))
        tprint_debug(
            "[UncertaintyEstimator] Stationary bootstrap SE computed",
            {"se": se, "valid_iterations": len(bootstrap_ics)},
        )
        return se
    
    def _apply_regime_aware_weights(self, 
                                  feature_vals: np.ndarray,
                                  weights: np.ndarray,
                                  regime_segments: List[Any]) -> np.ndarray:
        """Apply regime-aware weights for wild bootstrap."""
        # For now, apply weights uniformly
        # In practice, you'd weight differently based on regime
        return feature_vals * weights
    
    def _stationary_bootstrap(self, 
                            feature_vals: np.ndarray,
                            target_vals: np.ndarray,
                            block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Perform stationary block bootstrap."""
        n = len(feature_vals)
        bootstrap_feature = []
        bootstrap_target = []
        
        while len(bootstrap_feature) < n:
            # Random start point
            start_idx = np.random.randint(0, n)
            
            # Random block length (geometric distribution)
            block_length = np.random.geometric(1.0 / block_size)
            block_length = min(block_length, n - start_idx)
            
            # Add block
            end_idx = start_idx + block_length
            bootstrap_feature.extend(feature_vals[start_idx:end_idx])
            bootstrap_target.extend(target_vals[start_idx:end_idx])
        
        # Truncate to original length
        return (np.array(bootstrap_feature[:n]), 
                np.array(bootstrap_target[:n]))


class CostEstimator:
    """Estimates computational costs for features."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        tprint_debug("Initialized CostEstimator")

    def estimate_cpu_cost(self,
                         lookback: int,
                         family: str,
                         feature_type: str = 'htf') -> float:
        """
        Estimate CPU cost in milliseconds.
        
        Args:
            lookback: Lookback period in minutes
            family: Feature family
            feature_type: Type of feature ('htf', 'base', 'interaction')
            
        Returns:
            Estimated CPU cost in milliseconds
        """
        # Base costs per minute of lookback
        base_costs = {
            'htf': 0.01,  # ms per minute
            'base': 0.005,
            'interaction': 0.02
        }
        
        # Family-specific multipliers
        family_multipliers = {
            'trend_level_vol': 1.0,
            'oscillators': 1.2,
            'anchors': 0.8,
            'liquidity_micro': 1.1,
            'context': 1.3
        }
        
        base_cost = base_costs.get(feature_type, 0.01)
        multiplier = family_multipliers.get(family, 1.0)
        
        estimated_cost = float(base_cost * lookback * multiplier)
        tprint_debug(
            "[CostEstimator] Estimated CPU cost",
            {
                "lookback": lookback,
                "family": family,
                "feature_type": feature_type,
                "cost_ms": estimated_cost,
            },
        )
        return estimated_cost

    def estimate_memory_cost(self,
                           lookback: int,
                           family: str) -> float:
        """Estimate memory cost in MB."""
        # Simplified memory estimation
        base_memory = 0.001  # MB per minute
        estimated_memory = float(base_memory * lookback)
        tprint_debug(
            "[CostEstimator] Estimated memory cost",
            {
                "lookback": lookback,
                "family": family,
                "memory_mb": estimated_memory,
            },
        )
        return estimated_memory


class StalenessCalculator:
    """Calculates staleness scores for features."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.curve_calculator = StalenessCurveCalculator()
        tprint_debug("Initialized StalenessCalculator")

    def calculate_staleness(self,
                          lookback: int,
                          family: str,
                          base_timeframe: int = 5) -> float:
        """
        Calculate staleness score for a feature.
        
        Args:
            lookback: Lookback period in minutes
            family: Feature family
            base_timeframe: Base timeframe in minutes
            
        Returns:
            Staleness score (0-1, higher = more stale)
        """
        tprint_debug(
            "[StalenessCalculator] Calculating staleness",
            {
                "lookback": lookback,
                "family": family,
                "base_timeframe": base_timeframe,
            },
        )

        summary = self.curve_calculator.get_summary(
            feature_name=f"family::{family}",
            family=family,
            lookback=lookback,
            base_timeframe=base_timeframe,
        )
        tprint_debug(
            "[StalenessCalculator] Staleness summary retrieved",
            {"at_base": summary.at_base},
        )
        return summary.at_base


class MetaLearner:
    """Meta-learns penalty parameters based on recent performance."""

    def __init__(self,
                 learning_rate: float = 0.01,
                 adaptation_range: float = 0.05):
        self.learning_rate = learning_rate
        self.adaptation_range = adaptation_range
        self.logger = logging.getLogger(__name__)
        
        # Initialize penalty parameters
        self.lambda_unc = 0.10
        self.lambda_cost = 0.05
        self.lambda_stale = 0.05
        
        # Performance history
        self.performance_history = []
        self.penalty_history = []
        tprint_info(
            "Initialized MetaLearner",
            {
                "lambda_unc": self.lambda_unc,
                "lambda_cost": self.lambda_cost,
                "lambda_stale": self.lambda_stale,
                "learning_rate": self.learning_rate,
                "adaptation_range": self.adaptation_range,
            },
        )

    def update_penalties(self,
                        recent_performance: List[Dict[str, Any]],
                        market_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Update penalty parameters based on recent performance.
        
        Args:
            recent_performance: List of recent scoring results
            market_state: Current market state (volatility, etc.)
            
        Returns:
            Updated penalty parameters
        """
        if not recent_performance:
            tprint_warning(
                "[MetaLearner] No recent performance provided; retaining current penalties.",
                self._get_current_penalties(),
            )
            return self._get_current_penalties()

        # Analyze recent performance
        avg_ic = np.mean([p.get('ic_oos', 0) for p in recent_performance])
        avg_uncertainty = np.mean([p.get('se_wild_bootstrap', 1) for p in recent_performance])
        avg_cost = np.mean([p.get('cpu_p95', 0) for p in recent_performance])
        avg_staleness = np.mean([p.get('staleness', 0) for p in recent_performance])

        tprint_debug(
            "[MetaLearner] Aggregated recent performance",
            {
                "avg_ic": float(avg_ic),
                "avg_uncertainty": float(avg_uncertainty),
                "avg_cost": float(avg_cost),
                "avg_staleness": float(avg_staleness),
            },
        )

        # Market state adjustments
        vol_level = market_state.get('volatility_level', 0.5)
        news_proximity = market_state.get('news_proximity', 0.0)

        tprint_debug(
            "[MetaLearner] Market state inputs",
            {
                "volatility_level": vol_level,
                "news_proximity": news_proximity,
            },
        )
        
        # Adjust penalties based on performance and market state
        if avg_ic < 0.05:  # Low IC, increase uncertainty penalty
            self.lambda_unc = min(0.20, self.lambda_unc + self.learning_rate)
        elif avg_ic > 0.15:  # High IC, decrease uncertainty penalty
            self.lambda_unc = max(0.05, self.lambda_unc - self.learning_rate)
        
        if vol_level > 0.7:  # High volatility, increase uncertainty penalty
            self.lambda_unc = min(0.20, self.lambda_unc + self.learning_rate * 0.5)
        
        if news_proximity > 0.5:  # Near news events, increase staleness penalty
            self.lambda_stale = min(0.15, self.lambda_stale + self.learning_rate)
        
        # Record history
        self.performance_history.append({
            'avg_ic': avg_ic,
            'avg_uncertainty': avg_uncertainty,
            'avg_cost': avg_cost,
            'avg_staleness': avg_staleness
        })
        
        self.penalty_history.append(self._get_current_penalties())

        updated_penalties = self._get_current_penalties()
        tprint_info(
            "[MetaLearner] Updated penalties",
            updated_penalties,
        )

        return updated_penalties

    def _get_current_penalties(self) -> Dict[str, float]:
        """Get current penalty parameters."""
        return {
            'lambda_unc': self.lambda_unc,
            'lambda_cost': self.lambda_cost,
            'lambda_stale': self.lambda_stale
        }

    def set_penalties(self, penalties: Dict[str, float]):
        """Force-set penalty parameters from external adaptive learners."""
        if 'lambda_unc' in penalties:
            self.lambda_unc = float(penalties['lambda_unc'])
        if 'lambda_cost' in penalties:
            self.lambda_cost = float(penalties['lambda_cost'])
        if 'lambda_stale' in penalties:
            self.lambda_stale = float(penalties['lambda_stale'])

        self.penalty_history.append(self._get_current_penalties())
        tprint_info(
            "[MetaLearner] Penalties set externally",
            self._get_current_penalties(),
        )


class AdaptiveScoringSystem:
    """Main adaptive scoring system."""

    def __init__(self, scoring_config: ScoringConfig, session_config: SessionConfig):
        self.scoring_config = scoring_config
        self.session_config = session_config
        self.logger = logging.getLogger(__name__)

        self.uncertainty_estimator = UncertaintyEstimator()
        self.cost_estimator = CostEstimator()
        self.staleness_calculator = StalenessCalculator()
        self.meta_learner = MetaLearner(
            adaptation_range=scoring_config.meta_learning_range
        )
        self.lambda_unc = scoring_config.lambda_unc
        self.lambda_cost = scoring_config.lambda_cost
        self.lambda_stale = scoring_config.lambda_stale
        self.meta_learner.lambda_unc = scoring_config.lambda_unc
        self.meta_learner.lambda_cost = scoring_config.lambda_cost
        self.meta_learner.lambda_stale = scoring_config.lambda_stale
        tprint_info(
            "Initialized AdaptiveScoringSystem",
            {
                "lambda_unc": self.lambda_unc,
                "lambda_cost": self.lambda_cost,
                "lambda_stale": self.lambda_stale,
                "meta_learning_range": scoring_config.meta_learning_range,
            },
        )
    
    def score_feature_candidate(self, 
                              feature: pd.Series,
                              target: pd.Series,
                              lookback: int,
                              family: str,
                              regime: str,
                              regime_segments: Optional[List[Any]] = None) -> ScoringResult:
        """
        Score a feature candidate with adaptive penalties.
        
        Args:
            feature: Feature series
            target: Target series
            lookback: Lookback period in minutes
            family: Feature family
            regime: Regime type
            regime_segments: Regime segments for regime-aware scoring
            
        Returns:
            Scoring result with utility score
        """
        # Align data
        aligned_data = pd.DataFrame({
            'feature': feature,
            'target': target
        }).dropna()
        
        if len(aligned_data) < 10:
            tprint_warning(
                "[AdaptiveScoringSystem] Not enough aligned samples to score feature; returning empty result.",
                {
                    "feature": feature.name if hasattr(feature, 'name') else 'unknown',
                    "lookback": lookback,
                    "regime": regime,
                    "length": len(aligned_data),
                },
            )
            return self._create_empty_result(feature.name if hasattr(feature, 'name') else 'unknown',
                                           lookback, regime)

        feature_vals = aligned_data['feature'].values
        target_vals = aligned_data['target'].values

        tprint_info(
            "[AdaptiveScoringSystem] Scoring feature candidate",
            {
                "feature": feature.name if hasattr(feature, 'name') else 'unknown',
                "lookback": lookback,
                "family": family,
                "regime": regime,
                "samples": len(aligned_data),
            },
        )
        
        # Calculate IC
        ic_oos = np.corrcoef(feature_vals, target_vals)[0, 1]
        if np.isnan(ic_oos):
            ic_oos = 0.0
        
        # Estimate uncertainties
        se_wild_bootstrap = self.uncertainty_estimator.estimate_wild_bootstrap_se(
            feature, target, regime_segments
        )
        se_stationary_bootstrap = self.uncertainty_estimator.estimate_stationary_bootstrap_se(
            feature, target
        )
        
        # Estimate costs
        cpu_p95 = self.cost_estimator.estimate_cpu_cost(lookback, family)
        
        # Calculate staleness
        staleness = self.staleness_calculator.calculate_staleness(
            lookback,
            family,
            self.session_config.base_timeframe_minutes,
        )
        
        # Calculate fold pass rate
        fold_pass_rate = self._calculate_fold_pass_rate(feature, target)
        
        # Get current penalties (potentially updated by meta-learner)
        penalties = self.meta_learner._get_current_penalties()
        
        # Calculate utility score
        utility_score = self._calculate_utility_score(
            ic_oos, se_wild_bootstrap, cpu_p95, staleness, penalties
        )

        # Calculate regime weight
        regime_weight = self._calculate_regime_weight(regime, regime_segments)

        tprint_debug(
            "[AdaptiveScoringSystem] Computed scoring components",
            {
                "ic_oos": float(ic_oos),
                "se_wild": float(se_wild_bootstrap),
                "cpu_p95": float(cpu_p95),
                "staleness": float(staleness),
                "utility_score": float(utility_score),
                "fold_pass_rate": float(fold_pass_rate),
                "regime_weight": float(regime_weight),
            },
        )

        return ScoringResult(
            feature_name=feature.name if hasattr(feature, 'name') else 'unknown',
            lookback=lookback,
            regime=regime,
            ic_oos=ic_oos,
            se_wild_bootstrap=se_wild_bootstrap,
            se_stationary_bootstrap=se_stationary_bootstrap,
            cpu_p95=cpu_p95,
            staleness=staleness,
            utility_score=utility_score,
            fold_pass_rate=fold_pass_rate,
            regime_weight=regime_weight,
            metadata={
                'penalties': penalties,
                'data_length': len(aligned_data)
            }
        )
    
    def _calculate_fold_pass_rate(self, feature: pd.Series, target: pd.Series) -> float:
        """Calculate fold pass rate using time series cross-validation."""
        if len(feature) < 50:
            return 0.0
        
        # Use 3-fold time series split
        tscv = TimeSeriesSplit(n_splits=3)
        pass_count = 0
        total_folds = 0
        
        for train_idx, val_idx in tscv.split(feature):
            if len(val_idx) < 10:
                continue
            
            val_feature = feature.iloc[val_idx]
            val_target = target.iloc[val_idx]
            
            # Calculate IC on validation set
            val_ic = val_feature.corr(val_target)
            
            # Pass if IC > 0.05
            if not pd.isna(val_ic) and val_ic > 0.05:
                pass_count += 1
            
            total_folds += 1
        
        return pass_count / total_folds if total_folds > 0 else 0.0
    
    def _calculate_utility_score(self,
                               ic_oos: float,
                               se_wild_bootstrap: float,
                               cpu_p95: float,
                               staleness: float,
                               penalties: Dict[str, float]) -> float:
        """Calculate utility score with penalties."""
        utility = (
            ic_oos -
            penalties['lambda_unc'] * se_wild_bootstrap -
            penalties['lambda_cost'] * cpu_p95 -
            penalties['lambda_stale'] * staleness
        )
        tprint_debug(
            "[AdaptiveScoringSystem] Utility score computed",
            {
                "ic_oos": float(ic_oos),
                "se_wild_bootstrap": float(se_wild_bootstrap),
                "cpu_p95": float(cpu_p95),
                "staleness": float(staleness),
                "penalties": penalties,
                "utility": float(utility),
            },
        )
        return utility

    def calculate_utility_score(self,
                                ic_oos: float,
                                se_wild_bootstrap: float,
                                cpu_p95: float,
                                staleness: float) -> float:
        """Public helper to calculate utility using current adaptive penalties."""
        penalties = self.get_current_penalties()
        utility = self._calculate_utility_score(
            ic_oos=ic_oos,
            se_wild_bootstrap=se_wild_bootstrap,
            cpu_p95=cpu_p95,
            staleness=staleness,
            penalties=penalties,
        )
        tprint_info(
            "[AdaptiveScoringSystem] Calculated utility score with current penalties",
            {
                "utility": float(utility),
                "penalties": penalties,
            },
        )
        return utility
    
    def _calculate_regime_weight(self, 
                               regime: str, 
                               regime_segments: Optional[List[Any]]) -> float:
        """Calculate regime weight for fold aggregation."""
        if not regime_segments:
            return 1.0
        
        # Count segments by regime type
        regime_counts = {}
        for segment in regime_segments:
            regime_type = getattr(segment, 'regime_type', 'unknown')
            regime_counts[regime_type] = regime_counts.get(regime_type, 0) + 1
        
        total_segments = sum(regime_counts.values())
        if total_segments == 0:
            return 1.0
        
        # Weight by segment length and current regime posterior
        current_regime_count = regime_counts.get(regime, 0)
        return current_regime_count / total_segments
    
    def _create_empty_result(self, feature_name: str, lookback: int, regime: str) -> ScoringResult:
        """Create empty scoring result for failed cases."""
        return ScoringResult(
            feature_name=feature_name,
            lookback=lookback,
            regime=regime,
            ic_oos=0.0,
            se_wild_bootstrap=1.0,
            se_stationary_bootstrap=1.0,
            cpu_p95=0.0,
            staleness=1.0,
            utility_score=-1.0,
            fold_pass_rate=0.0,
            regime_weight=0.0,
            metadata={}
        )
    
    def update_meta_learning(self,
                           recent_performance: List[Dict[str, Any]],
                           market_state: Dict[str, Any]):
        """Update meta-learning with recent performance."""
        tprint_info(
            "[AdaptiveScoringSystem] Updating meta learning",
            {
                "recent_samples": len(recent_performance),
                "market_state": market_state,
            },
        )
        self.meta_learner.update_penalties(recent_performance, market_state)

    def apply_penalty_parameters(self, penalty_parameters: Dict[str, float]):
        """Apply externally provided penalty parameters to the scorer."""
        if not penalty_parameters:
            return

        self.meta_learner.set_penalties(penalty_parameters)
        self.logger.info(
            "Adaptive scoring penalties updated from monitoring system: %s",
            penalty_parameters,
        )
        tprint_info(
            "[AdaptiveScoringSystem] Penalty parameters applied",
            penalty_parameters,
        )

    def get_current_penalties(self) -> Dict[str, float]:
        """Get current penalty parameters."""
        penalties = self.meta_learner._get_current_penalties()
        tprint_debug(
            "[AdaptiveScoringSystem] Retrieved current penalties",
            penalties,
        )
        return penalties
