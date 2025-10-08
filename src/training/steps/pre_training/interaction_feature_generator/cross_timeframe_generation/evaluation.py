"""
Walk-Forward Evaluation System

Implements comprehensive evaluation with:
- Walk-forward validation with purging and embargo
- Block bootstrap confidence intervals
- SPA (Superior Predictive Ability) test
- Regime-aware evaluation
- Ablation studies
- Performance metrics by regime
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import warnings

from src.utils.tprint import tprint
warnings.filterwarnings('ignore')

from .config import EvaluationConfig

# Try to import SPA test implementation
try:
    from scipy.stats import norm
    SPA_AVAILABLE = True
except ImportError:
    SPA_AVAILABLE = False
    logging.warning("SPA test not available, using simplified version")


@dataclass
class EvaluationResult:
    """Result of walk-forward evaluation."""
    overall_ic: float
    overall_ic_std: float
    overall_ic_ci: Tuple[float, float]
    regime_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    ablation_results: Dict[str, Dict[str, float]] = field(default_factory=dict)
    spa_test_result: Dict[str, Any] = field(default_factory=dict)
    walk_forward_results: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    overall_pnl: float = 0.0
    overall_pnl_std: float = 0.0
    overall_pnl_ci: Tuple[float, float] = (0.0, 0.0)
    robust_statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    block_bootstrap_cis: Dict[str, Tuple[float, float]] = field(default_factory=dict)


@dataclass
class WalkForwardFold:
    """Single walk-forward fold."""
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    ic: float
    pnl: float
    mse: float
    mae: float
    r2: float
    sharpe: float
    max_drawdown: float
    metadata: Dict[str, Any]


def _newey_west_statistics(
    series: List[float],
    *,
    max_lag: Optional[int] = None,
) -> Dict[str, float]:
    """Compute Newey-West/HAC standard error, t-statistic, and p-value for a series.

    Args:
        series: Sequence of fold-level statistics (IC or PnL).
        max_lag: Optional manual lag length. If ``None`` a Bartlett kernel
            with the Newey-West rule-of-thumb lag (⌊1.1447 * n^(1/3)⌋) is used.

    Returns:
        Dictionary with ``mean``, ``std_error``, ``t_stat`` and ``p_value``.
    """

    values = np.asarray(series, dtype=float)
    values = values[~np.isnan(values)]

    if values.size == 0:
        return {
            'mean': 0.0,
            'std_error': 0.0,
            't_stat': 0.0,
            'p_value': 1.0,
        }

    n_obs = values.size
    mean_value = float(np.mean(values))

    if n_obs == 1:
        # With a single observation we cannot estimate autocovariances.
        std_error = 0.0
        t_stat = np.inf if mean_value > 0 else (-np.inf if mean_value < 0 else 0.0)
        p_value = 0.0 if np.isinf(t_stat) else 1.0
        return {
            'mean': mean_value,
            'std_error': std_error,
            't_stat': t_stat,
            'p_value': p_value,
        }

    if max_lag is None:
        max_lag = int(np.floor(1.1447 * n_obs ** (1 / 3)))
        max_lag = max(1, min(max_lag, n_obs - 1))
    else:
        max_lag = max(1, min(int(max_lag), n_obs - 1))

    demeaned = values - mean_value
    gamma_zero = float(np.dot(demeaned, demeaned) / n_obs)
    hac_variance = gamma_zero

    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1)
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n_obs)
        hac_variance += 2.0 * weight * cov

    std_error = float(np.sqrt(max(hac_variance / n_obs, 0.0)))

    if std_error == 0.0:
        t_stat = np.inf if mean_value > 0 else (-np.inf if mean_value < 0 else 0.0)
        p_value = 0.0 if np.isinf(t_stat) else 1.0
    else:
        t_stat = mean_value / std_error
        degrees_of_freedom = max(n_obs - 1, 1)
        p_value = float(2 * stats.t.sf(np.abs(t_stat), df=degrees_of_freedom))

    return {
        'mean': mean_value,
        'std_error': std_error,
        't_stat': t_stat,
        'p_value': p_value,
    }


class WalkForwardValidator:
    """Implements walk-forward validation with purging and embargo."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.n_folds = config.walk_forward_folds
        self.embargo_minutes = config.embargo_minutes
    
    def validate_features(self, 
                         features: pd.DataFrame,
                         targets: pd.Series,
                         regime_segments: Optional[Dict[str, Any]] = None) -> List[WalkForwardFold]:
        """
        Perform walk-forward validation.
        
        Args:
            features: Feature matrix
            targets: Target series
            regime_segments: Regime segmentation results
            
        Returns:
            List of walk-forward fold results
        """
        self.logger.info("Starting walk-forward validation")
        
        # Create walk-forward splits
        splits = self._create_walk_forward_splits(features, targets)
        tprint(
            f"📆 Generated {len(splits)} walk-forward splits for evaluation",
            "INFO"
        )

        fold_results = []

        for i, (train_idx, test_idx) in enumerate(splits):
            try:
                # Get train/test data
                train_features = features.iloc[train_idx]
                train_targets = targets.iloc[train_idx]
                test_features = features.iloc[test_idx]
                test_targets = targets.iloc[test_idx]
                
                # Apply purging and embargo
                train_features, train_targets = self._apply_purging_embargo(
                    train_features, train_targets, test_features.index[0]
                )
                
                if len(train_features) < 50:  # Need sufficient training data
                    continue
                
                # Train model
                model = self._train_model(train_features, train_targets)
                
                # Make predictions
                predictions = self._make_predictions(model, test_features)
                
                # Calculate metrics
                fold_result = self._calculate_fold_metrics(
                    predictions, test_targets, train_idx, test_idx, i
                )
                
                fold_results.append(fold_result)

            except Exception as e:
                self.logger.warning(f"Walk-forward fold {i} failed: {e}")
                tprint(f"⚠️ Walk-forward fold {i} failed: {e}", "WARNING")
                continue

        self.logger.info(f"Walk-forward validation completed: {len(fold_results)} folds")
        tprint(
            f"✅ Walk-forward validation complete with {len(fold_results)} successful folds",
            "SUCCESS"
        )
        return fold_results

    def _create_walk_forward_splits(self,
                                   features: pd.DataFrame,
                                   targets: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create walk-forward splits."""
        # Use TimeSeriesSplit with custom parameters
        tscv = TimeSeriesSplit(n_splits=self.n_folds)
        splits = list(tscv.split(features))

        tprint(
            f"🔀 Created {len(splits)} walk-forward splits (folds={self.n_folds})",
            "DEBUG"
        )

        return splits

    def _apply_purging_embargo(self,
                              train_features: pd.DataFrame,
                              train_targets: pd.Series,
                              test_start: datetime) -> Tuple[pd.DataFrame, pd.Series]:
        """Apply purging and embargo to training data."""
        # Calculate embargo period
        embargo_end = test_start - timedelta(minutes=self.embargo_minutes)

        # Filter training data
        train_mask = train_features.index < embargo_end
        purged_train_features = train_features[train_mask]
        purged_train_targets = train_targets[train_mask]

        tprint(
            "🧹 Applied purging & embargo: "
            f"kept {len(purged_train_features)} of {len(train_features)} rows (embargo {self.embargo_minutes}m)",
            "DEBUG"
        )

        return purged_train_features, purged_train_targets

    def _train_model(self,
                    train_features: pd.DataFrame,
                    train_targets: pd.Series) -> Any:
        """Train a model on training data."""
        # Use simple linear regression for now
        # In practice, you'd use more sophisticated models
        model = LinearRegression()
        model.fit(train_features, train_targets)
        tprint(
            f"🎓 Trained linear regression on {len(train_features)} samples with {train_features.shape[1]} features",
            "INFO"
        )
        return model

    def _make_predictions(self,
                         model: Any,
                         test_features: pd.DataFrame) -> pd.Series:
        """Make predictions on test data."""
        predictions = model.predict(test_features)
        tprint(
            f"🧮 Generated predictions for {len(test_features)} test samples",
            "DEBUG"
        )
        return pd.Series(predictions, index=test_features.index)

    def _calculate_fold_metrics(self,
                              predictions: pd.Series,
                              test_targets: pd.Series,
                              train_idx: np.ndarray,
                              test_idx: np.ndarray,
                              fold_number: int) -> WalkForwardFold:
        """Calculate metrics for a single fold."""
        # Align predictions and targets
        aligned_data = pd.DataFrame({
            'predictions': predictions,
            'targets': test_targets
        }).dropna()
        
        if len(aligned_data) == 0:
            return WalkForwardFold(
                train_start=datetime.now(),
                train_end=datetime.now(),
                test_start=datetime.now(),
                test_end=datetime.now(),
                ic=0.0,
                pnl=0.0,
                mse=0.0,
                mae=0.0,
                r2=0.0,
                sharpe=0.0,
                max_drawdown=0.0,
                metadata={'fold': fold_number, 'error': 'No aligned data'}
            )
        
        pred_vals = aligned_data['predictions'].values
        target_vals = aligned_data['targets'].values
        
        # Calculate IC
        ic = np.corrcoef(pred_vals, target_vals)[0, 1] if len(pred_vals) > 1 else 0.0
        if pd.isna(ic):
            ic = 0.0
        
        # Calculate regression metrics
        mse = mean_squared_error(target_vals, pred_vals)
        mae = mean_absolute_error(target_vals, pred_vals)
        r2 = r2_score(target_vals, pred_vals)
        
        # Calculate Sharpe ratio (simplified)
        returns = np.diff(target_vals)
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0.0

        # Aggregate fold-level PnL as cumulative returns
        pnl = float(np.sum(returns)) if len(returns) > 0 else 0.0
        
        # Calculate maximum drawdown (simplified)
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
        
        return WalkForwardFold(
            train_start=datetime.now(),  # Would be actual timestamps
            train_end=datetime.now(),
            test_start=datetime.now(),
            test_end=datetime.now(),
            ic=ic,
            pnl=pnl,
            mse=mse,
            mae=mae,
            r2=r2,
            sharpe=sharpe,
            max_drawdown=max_drawdown,
            metadata={'fold': fold_number, 'n_samples': len(aligned_data)}
        )

        tprint(
            "📊 Fold metrics "
            f"(fold={fold_number}): IC={ic:.4f}, MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}, "
            f"Sharpe={sharpe:.4f}, MaxDD={max_drawdown:.4f}",
            "INFO"
        )


class BootstrapEvaluator:
    """Implements block bootstrap evaluation."""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def calculate_confidence_intervals(self,
                                     fold_results: List[WalkForwardFold],
                                     metric: str = 'ic') -> Tuple[float, float, Tuple[float, float]]:
        """
        Calculate confidence intervals using a moving block bootstrap.

        Args:
            fold_results: List of walk-forward fold results
            metric: Metric to calculate CI for

        Returns:
            Tuple of (mean, std_error, (lower_ci, upper_ci))
        """
        if not fold_results:
            return 0.0, 0.0, (0.0, 0.0)

        # Extract metric values
        metric_values = []
        for fold in fold_results:
            if hasattr(fold, metric):
                value = getattr(fold, metric)
                if not pd.isna(value):
                    metric_values.append(value)

        if not metric_values:
            return 0.0, 0.0, (0.0, 0.0)

        metric_array = np.array(metric_values)

        # Calculate mean
        mean_value = np.mean(metric_array)

        n_obs = len(metric_array)
        if n_obs == 0:
            return 0.0, 0.0, (0.0, 0.0)

        if n_obs == 1:
            std_error = 0.0
            ci = (mean_value, mean_value)
            return mean_value, std_error, ci

        std_error = np.std(metric_array, ddof=1) / np.sqrt(n_obs)

        block_length = self._resolve_block_length(n_obs)
        n_resamples = int(getattr(self.config, 'bootstrap_resamples', 1000))
        confidence_level = float(getattr(self.config, 'bootstrap_confidence_level', 0.95))

        bootstrap_means = self._moving_block_bootstrap(metric_array, block_length, n_resamples)

        alpha = 1.0 - confidence_level
        lower_quantile = alpha / 2.0
        upper_quantile = 1.0 - alpha / 2.0
        ci_lower = float(np.quantile(bootstrap_means, lower_quantile))
        ci_upper = float(np.quantile(bootstrap_means, upper_quantile))

        tprint(
            f"🎯 Bootstrap CI for {metric}: mean={mean_value:.4f}, "
            f"CI=({ci_lower:.4f}, {ci_upper:.4f}) from {len(metric_array)} folds",
            "INFO"
        )

        return mean_value, std_error, (ci_lower, ci_upper)

    def _resolve_block_length(self, n_obs: int) -> int:
        """Determine the block length for the bootstrap procedure."""
        configured_block = getattr(self.config, 'bootstrap_block_size', None)
        if configured_block is not None:
            block_length = int(configured_block)
        else:
            block_length = int(np.sqrt(n_obs))

        block_length = max(1, min(block_length, n_obs))
        return block_length

    def _moving_block_bootstrap(
        self,
        data: np.ndarray,
        block_length: int,
        n_resamples: int,
    ) -> np.ndarray:
        """Generate bootstrap resamples using the moving block bootstrap."""
        n_obs = len(data)
        if n_obs == 0:
            return np.zeros(n_resamples)

        if block_length >= n_obs:
            mean_value = float(np.mean(data))
            return np.full(n_resamples, mean_value)

        block_starts = np.arange(0, n_obs - block_length + 1)
        rng_seed = getattr(self.config, 'bootstrap_random_seed', None)
        rng = np.random.default_rng(rng_seed)

        bootstrap_means = np.empty(n_resamples, dtype=float)

        for i in range(n_resamples):
            sampled_indices: List[int] = []
            while len(sampled_indices) < n_obs:
                start = int(rng.choice(block_starts))
                block_indices = list(range(start, start + block_length))
                sampled_indices.extend(block_indices)

            sampled_indices = sampled_indices[:n_obs]
            resampled = data[sampled_indices]
            bootstrap_means[i] = float(np.mean(resampled))

        return bootstrap_means


class SPATester:
    """Implements Superior Predictive Ability (SPA) test."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def test_spa(self, 
                fold_results: List[WalkForwardFold],
                benchmark_ic: float = 0.0) -> Dict[str, Any]:
        """
        Perform SPA test.
        
        Args:
            fold_results: List of walk-forward fold results
            benchmark_ic: Benchmark IC for comparison
            
        Returns:
            SPA test results
        """
        if not fold_results:
            return {'spa_statistic': 0.0, 'p_value': 1.0, 'reject_null': False}
        
        # Extract IC values
        ic_values = []
        for fold in fold_results:
            if hasattr(fold, 'ic') and not pd.isna(fold.ic):
                ic_values.append(fold.ic)
        
        if not ic_values:
            return {'spa_statistic': 0.0, 'p_value': 1.0, 'reject_null': False}
        
        ic_array = np.array(ic_values)
        
        # Calculate SPA statistic
        spa_statistic = self._calculate_spa_statistic(ic_array, benchmark_ic)
        
        # Calculate p-value using bootstrap
        p_value = self._calculate_spa_p_value(ic_array, benchmark_ic, spa_statistic)
        
        # Decision
        reject_null = p_value < 0.05

        tprint(
            "🧪 SPA test completed: "
            f"stat={spa_statistic:.4f}, p_value={p_value:.4f}, reject_null={reject_null}",
            "INFO"
        )

        return {
            'spa_statistic': spa_statistic,
            'p_value': p_value,
            'reject_null': reject_null,
            'benchmark_ic': benchmark_ic,
            'mean_ic': np.mean(ic_array),
            'n_folds': len(ic_values)
        }
    
    def _calculate_spa_statistic(self, 
                               ic_values: np.ndarray,
                               benchmark_ic: float) -> float:
        """Calculate SPA statistic."""
        # SPA statistic: max of standardized excess returns
        excess_ics = ic_values - benchmark_ic
        
        if len(excess_ics) == 0:
            return 0.0

        # Standardize
        mean_excess = np.mean(excess_ics)
        std_excess = np.std(excess_ics)

        if std_excess == 0:
            return 0.0

        standardized_excess = excess_ics / std_excess

        # SPA statistic is the maximum of standardized excess returns
        spa_statistic = np.max(standardized_excess)

        tprint(
            f"🧮 Calculated SPA statistic={spa_statistic:.4f} (mean_excess={mean_excess:.4f}, std={std_excess:.4f})",
            "DEBUG"
        )

        return spa_statistic

    def _calculate_spa_p_value(self,
                             ic_values: np.ndarray,
                             benchmark_ic: float,
                             spa_statistic: float) -> float:
        """Calculate SPA p-value using bootstrap."""
        n_bootstrap = 1000
        bootstrap_stats = []

        for _ in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_ics = np.random.choice(ic_values, size=len(ic_values), replace=True)

            # Calculate bootstrap SPA statistic
            bootstrap_spa = self._calculate_spa_statistic(bootstrap_ics, benchmark_ic)
            bootstrap_stats.append(bootstrap_spa)

        # Calculate p-value
        p_value = np.mean(np.array(bootstrap_stats) >= spa_statistic)

        tprint(
            f"📈 SPA bootstrap completed with {n_bootstrap} samples; p_value={p_value:.4f}",
            "DEBUG"
        )

        return p_value


class RegimeEvaluator:
    """Implements regime-aware evaluation."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def evaluate_by_regime(self, 
                          fold_results: List[WalkForwardFold],
                          regime_segments: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate performance by regime.
        
        Args:
            fold_results: List of walk-forward fold results
            regime_segments: Regime segmentation results
            
        Returns:
            Dictionary of regime-specific results
        """
        if not regime_segments or 'segments' not in regime_segments:
            return {}
        
        regime_results = {}
        segments = regime_segments['segments']
        
        # Group folds by regime
        for segment in segments:
            regime_type = getattr(segment, 'regime_type', 'unknown')
            
            if regime_type not in regime_results:
                regime_results[regime_type] = {
                    'ic': [],
                    'mse': [],
                    'mae': [],
                    'r2': [],
                    'sharpe': [],
                    'max_drawdown': []
                }
            
            # Find folds in this regime (simplified)
            # In practice, you'd match fold timestamps with regime segments
            for fold in fold_results:
                if hasattr(fold, 'ic') and not pd.isna(fold.ic):
                    regime_results[regime_type]['ic'].append(fold.ic)
                    regime_results[regime_type]['mse'].append(fold.mse)
                    regime_results[regime_type]['mae'].append(fold.mae)
                    regime_results[regime_type]['r2'].append(fold.r2)
                    regime_results[regime_type]['sharpe'].append(fold.sharpe)
                    regime_results[regime_type]['max_drawdown'].append(fold.max_drawdown)
        
        # Calculate summary statistics for each regime
        summary_results = {}
        for regime_type, metrics in regime_results.items():
            summary_results[regime_type] = {}
            for metric_name, values in metrics.items():
                if values:
                    summary_results[regime_type][f'{metric_name}_mean'] = np.mean(values)
                    summary_results[regime_type][f'{metric_name}_std'] = np.std(values)
                    summary_results[regime_type][f'{metric_name}_count'] = len(values)
                else:
                    summary_results[regime_type][f'{metric_name}_mean'] = 0.0
                    summary_results[regime_type][f'{metric_name}_std'] = 0.0
                    summary_results[regime_type][f'{metric_name}_count'] = 0

            tprint(
                f"📊 Regime '{regime_type}' metrics computed with {summary_results[regime_type].get('ic_count', 0)} folds",
                "INFO"
            )

        tprint(
            f"🗂️ Regime evaluation completed for {len(summary_results)} regimes",
            "SUCCESS"
        )

        return summary_results


class AblationEvaluator:
    """Implements ablation studies."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def perform_ablation_study(self, 
                             features: pd.DataFrame,
                             targets: pd.Series,
                             feature_groups: Dict[str, List[str]]) -> Dict[str, Dict[str, float]]:
        """
        Perform ablation study by removing feature groups.
        
        Args:
            features: Feature matrix
            targets: Target series
            feature_groups: Dictionary of group_name -> feature_list
            
        Returns:
            Dictionary of ablation results
        """
        self.logger.info("Starting ablation study")

        if features is None or features.empty or not list(features.columns):
            self.logger.warning("Feature matrix empty; skipping ablation study")
            return {}

        ablation_results = {}

        # Baseline (all features)
        baseline_ic = self._calculate_group_ic(features, targets, list(features.columns))
        ablation_results['baseline'] = {'ic': baseline_ic, 'n_features': len(features.columns)}
        tprint(
            f"🧪 Ablation baseline IC={baseline_ic:.4f} across {len(features.columns)} features",
            "INFO"
        )

        # Ablate each group
        for group_name, group_features in feature_groups.items():
            # Remove group features
            remaining_features = [f for f in features.columns if f not in group_features]

            if remaining_features:
                ablated_ic = self._calculate_group_ic(features, targets, remaining_features)
                ablation_results[f'without_{group_name}'] = {
                    'ic': ablated_ic,
                    'n_features': len(remaining_features),
                    'removed_features': group_features
                }
                tprint(
                    "🧪 Ablation result "
                    f"without '{group_name}': IC={ablated_ic:.4f} ({len(remaining_features)} features)",
                    "DEBUG"
                )

        self.logger.info(f"Ablation study completed: {len(ablation_results)} configurations")
        tprint(
            f"✅ Ablation study completed with {len(ablation_results)} configurations",
            "SUCCESS"
        )
        return ablation_results
    
    def _calculate_group_ic(self, 
                          features: pd.DataFrame,
                          targets: pd.Series,
                          feature_list: List[str]) -> float:
        """Calculate IC for a specific group of features."""
        if not feature_list:
            return 0.0
        
        try:
            # Use simple correlation as IC proxy
            # In practice, you'd train a model and calculate IC
            group_features = features[feature_list]

            # Calculate average correlation
            correlations = []
            for feature_name in feature_list:
                if feature_name in group_features.columns:
                    corr = group_features[feature_name].corr(targets)
                    if not pd.isna(corr):
                        correlations.append(corr)

            if correlations:
                tprint(
                    f"📐 Group IC calculated for {len(correlations)} features: mean={np.mean(correlations):.4f}",
                    "DEBUG"
                )
                return np.mean(correlations)
            else:
                tprint(
                    "📐 Group IC calculation yielded no valid correlations",
                    "WARNING"
                )
                return 0.0

        except Exception as e:
            self.logger.warning(f"Failed to calculate group IC: {e}")
            tprint(f"⚠️ Failed to calculate group IC: {e}", "WARNING")
            return 0.0


class WalkForwardEvaluation:
    """Main walk-forward evaluation system."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        self.walk_forward_validator = WalkForwardValidator(config)
        self.bootstrap_evaluator = BootstrapEvaluator(config)
        self.spa_tester = SPATester(config)
        self.regime_evaluator = RegimeEvaluator(config)
        self.ablation_evaluator = AblationEvaluator(config)

    def _prepare_feature_matrix(self,
                                materialized_htfs: Optional[Dict[str, Any]],
                                interactions: Optional[List[Any]]) -> pd.DataFrame:
        """Reconstruct the feature matrix from materialized HTFs and interactions."""
        feature_data: Dict[str, pd.Series] = {}

        if isinstance(materialized_htfs, dict):
            for name, htf in materialized_htfs.items():
                series = getattr(htf, 'feature_series', None)
                if isinstance(series, pd.Series):
                    feature_data[name] = series

        if isinstance(interactions, list):
            for interaction in interactions:
                interaction_name = getattr(interaction, 'name', None)
                series = getattr(interaction, 'feature_series', None)
                if interaction_name and isinstance(series, pd.Series):
                    feature_data[interaction_name] = series

        if not feature_data:
            tprint("ℹ️ No feature data available to build matrix", "INFO")
            return pd.DataFrame()

        feature_matrix = pd.DataFrame(feature_data)
        feature_matrix = feature_matrix.dropna()
        tprint(
            f"🧱 Prepared feature matrix with shape {feature_matrix.shape}",
            "INFO"
        )
        return feature_matrix

    def _create_default_result(self,
                               reason: str,
                               extra_metadata: Optional[Dict[str, Any]] = None) -> EvaluationResult:
        """Create an empty evaluation result with metadata about the reason."""
        metadata = {
            'n_folds': 0,
            'n_features': 0,
            'evaluation_method': 'walk_forward',
            'embargo_minutes': self.config.embargo_minutes,
            'reason': reason
        }

        if extra_metadata:
            metadata.update(extra_metadata)

        tprint(
            f"ℹ️ Returning default evaluation result due to: {reason}",
            "WARNING"
        )

        return EvaluationResult(
            overall_ic=0.0,
            overall_ic_std=0.0,
            overall_ic_ci=(0.0, 0.0),
            overall_pnl=0.0,
            overall_pnl_std=0.0,
            overall_pnl_ci=(0.0, 0.0),
            robust_statistics={},
            block_bootstrap_cis={},
            regime_results={},
            ablation_results={},
            spa_test_result={},
            walk_forward_results=[],
            metadata=metadata
        )

    def evaluate_features(self,
                        final_features: List[str],
                        targets: pd.Series,
                        regime_segments: Optional[Dict[str, Any]] = None,
                        materialized_htfs: Optional[Dict[str, Any]] = None,
                        interactions: Optional[List[Any]] = None) -> EvaluationResult:
        """
        Evaluate final features using walk-forward validation.
        
        Args:
            final_features: List of selected feature names
            targets: Target series
            regime_segments: Regime segmentation results
            materialized_htfs: Materialized HTF containers keyed by feature name
            interactions: Interaction feature containers
            
        Returns:
            Evaluation result
        """
        self.logger.info("Starting walk-forward evaluation")
        tprint("🚀 Starting walk-forward evaluation", "INFO")

        if not final_features:
            self.logger.warning("No final features provided for evaluation")
            tprint("⚠️ No final features provided for evaluation", "WARNING")
            return self._create_default_result('no_features_provided')

        feature_matrix = self._prepare_feature_matrix(materialized_htfs, interactions)

        if feature_matrix.empty:
            self.logger.warning("Constructed feature matrix is empty; skipping evaluation")
            tprint("⚠️ Feature matrix empty after preparation", "WARNING")
            return self._create_default_result('empty_feature_matrix')

        available_features = [f for f in final_features if f in feature_matrix.columns]
        missing_features = sorted(set(final_features) - set(available_features))

        if missing_features:
            self.logger.warning(
                "Missing features from materialized data: %s",
                missing_features
            )
            tprint(
                f"⚠️ Missing {len(missing_features)} features from materialized data",
                "WARNING"
            )

        if not available_features:
            return self._create_default_result(
                'no_available_features',
                {'missing_features': missing_features}
            )

        feature_matrix = feature_matrix.loc[:, available_features]
        tprint(
            f"🧱 Using {len(available_features)} available features for evaluation",
            "INFO"
        )

        aligned_index = feature_matrix.index.intersection(targets.index)
        if aligned_index.empty:
            self.logger.warning("No overlapping timestamps between features and targets")
            tprint(
                "⚠️ No overlapping timestamps between features and targets",
                "WARNING"
            )
            return self._create_default_result('no_overlapping_timestamps')

        feature_matrix = feature_matrix.loc[aligned_index]
        aligned_targets = targets.loc[aligned_index]

        feature_matrix = feature_matrix.dropna()
        aligned_targets = aligned_targets.loc[feature_matrix.index]

        if feature_matrix.empty:
            self.logger.warning("Feature matrix empty after alignment and NaN removal")
            tprint(
                "⚠️ Feature matrix empty after alignment and NaN filtering",
                "WARNING"
            )
            return self._create_default_result(
                'empty_after_alignment',
                {'missing_features': missing_features}
            )

        # Walk-forward validation
        fold_results = self.walk_forward_validator.validate_features(
            feature_matrix, aligned_targets, regime_segments
        )

        tprint(
            f"📆 Obtained {len(fold_results)} fold results from walk-forward validation",
            "INFO"
        )

        # Calculate overall metrics
        overall_ic, overall_ic_std, overall_ic_ci = self.bootstrap_evaluator.calculate_confidence_intervals(
            fold_results, 'ic'
        )
        tprint(
            f"📈 Overall IC={overall_ic:.4f} (std={overall_ic_std:.4f}, CI=({overall_ic_ci[0]:.4f}, {overall_ic_ci[1]:.4f}))",
            "INFO"
        )

        overall_pnl, overall_pnl_std, overall_pnl_ci = self.bootstrap_evaluator.calculate_confidence_intervals(
            fold_results, 'pnl'
        )
        tprint(
            f"💰 Overall PnL={overall_pnl:.4f} (std={overall_pnl_std:.4f}, CI=({overall_pnl_ci[0]:.4f}, {overall_pnl_ci[1]:.4f}))",
            "INFO"
        )

        hac_lag = getattr(self.config, 'hac_max_lag', None)
        ic_series = [getattr(fold, 'ic', np.nan) for fold in fold_results]
        ic_series = [value for value in ic_series if not pd.isna(value)]
        pnl_series = [getattr(fold, 'pnl', np.nan) for fold in fold_results]
        pnl_series = [value for value in pnl_series if not pd.isna(value)]

        ic_hac_stats = _newey_west_statistics(ic_series, max_lag=hac_lag)
        pnl_hac_stats = _newey_west_statistics(pnl_series, max_lag=hac_lag)

        robust_statistics = {
            'ic': ic_hac_stats,
            'pnl': pnl_hac_stats,
        }

        block_bootstrap_cis = {
            'ic': overall_ic_ci,
            'pnl': overall_pnl_ci,
        }

        tprint(
            "🛡️ Robust stats: "
            f"IC t={ic_hac_stats['t_stat']:.2f} (p={ic_hac_stats['p_value']:.4f}), "
            f"PnL t={pnl_hac_stats['t_stat']:.2f} (p={pnl_hac_stats['p_value']:.4f})",
            "INFO"
        )

        # Regime-aware evaluation
        regime_results = {}
        if regime_segments:
            regime_results = self.regime_evaluator.evaluate_by_regime(
                fold_results, regime_segments
            )

        tprint(
            f"🗂️ Regime results computed: {len(regime_results)} regimes",
            "DEBUG"
        )

        # Ablation study
        feature_groups = self._create_ablation_groups(available_features)
        ablation_results = self.ablation_evaluator.perform_ablation_study(
            feature_matrix, aligned_targets, feature_groups
        )

        tprint(
            f"🧪 Ablation produced {len(ablation_results)} configurations",
            "DEBUG"
        )

        # SPA test
        spa_test_result = {}
        if self.config.spa_test:
            spa_test_result = self.spa_tester.test_spa(fold_results)

        if spa_test_result:
            tprint(
                f"🧪 SPA test p-value={spa_test_result.get('p_value', 0.0):.4f}",
                "INFO"
            )

        # Create evaluation result
        result = EvaluationResult(
            overall_ic=overall_ic,
            overall_ic_std=overall_ic_std,
            overall_ic_ci=overall_ic_ci,
            overall_pnl=overall_pnl,
            overall_pnl_std=overall_pnl_std,
            overall_pnl_ci=overall_pnl_ci,
            robust_statistics=robust_statistics,
            block_bootstrap_cis=block_bootstrap_cis,
            regime_results=regime_results,
            ablation_results=ablation_results,
            spa_test_result=spa_test_result,
            walk_forward_results=[self._fold_to_dict(f) for f in fold_results],
            metadata={
                'n_folds': len(fold_results),
                'n_features': len(available_features),
                'evaluation_method': 'walk_forward',
                'embargo_minutes': self.config.embargo_minutes,
                'bootstrap_block_size': getattr(
                    self.config,
                    'bootstrap_block_size',
                    self.bootstrap_evaluator._resolve_block_length(max(len(ic_series), 1))
                ),
                'hac_max_lag': hac_lag,
            }
        )

        self.logger.info(f"Walk-forward evaluation completed: IC={overall_ic:.4f}")
        tprint(
            f"✅ Walk-forward evaluation completed successfully (IC={overall_ic:.4f})",
            "SUCCESS"
        )
        return result

    def _create_ablation_groups(self, final_features: List[str]) -> Dict[str, List[str]]:
        """Create feature groups for ablation study."""
        groups = {
            'htf_features': [],
            'base_features': [],
            'interactions': []
        }

        for feature_name in final_features:
            if 'htf' in feature_name.lower():
                groups['htf_features'].append(feature_name)
            elif 'int_' in feature_name.lower():
                groups['interactions'].append(feature_name)
            else:
                groups['base_features'].append(feature_name)

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}

        tprint(
            "🧪 Ablation groups prepared: "
            + ", ".join(f"{k}={len(v)}" for k, v in groups.items()),
            "DEBUG"
        )

        return groups

    def _fold_to_dict(self, fold: WalkForwardFold) -> Dict[str, Any]:
        """Convert WalkForwardFold to dictionary."""
        tprint(
            f"🗂️ Serializing fold (IC={fold.ic:.4f}, samples={fold.metadata.get('n_samples', 'n/a')})",
            "DEBUG"
        )
        return {
            'train_start': fold.train_start.isoformat(),
            'train_end': fold.train_end.isoformat(),
            'test_start': fold.test_start.isoformat(),
            'test_end': fold.test_end.isoformat(),
            'ic': fold.ic,
            'pnl': getattr(fold, 'pnl', 0.0),
            'mse': fold.mse,
            'mae': fold.mae,
            'r2': fold.r2,
            'sharpe': fold.sharpe,
            'max_drawdown': fold.max_drawdown,
            'metadata': fold.metadata
        }
    
    def get_evaluation_summary(self, result: EvaluationResult) -> Dict[str, Any]:
        """Get summary of evaluation results."""
        tprint(
            f"🧾 Generating evaluation summary for {len(result.walk_forward_results)} folds",
            "INFO"
        )
        return {
            'overall_ic': result.overall_ic,
            'overall_ic_std': result.overall_ic_std,
            'overall_ic_ci': result.overall_ic_ci,
            'overall_pnl': result.overall_pnl,
            'overall_pnl_std': result.overall_pnl_std,
            'overall_pnl_ci': result.overall_pnl_ci,
            'robust_statistics': result.robust_statistics,
            'block_bootstrap_cis': result.block_bootstrap_cis,
            'regime_results': result.regime_results,
            'ablation_results': result.ablation_results,
            'spa_test': result.spa_test_result,
            'n_folds': len(result.walk_forward_results),
            'metadata': result.metadata
        }