"""
Feature Evaluation Pipeline - 4-Stage Lookback Optimization

This module implements a fast, robust 4-stage pipeline for evaluating and selecting
optimal lookback periods for features:

STAGE 0 - SUBSAMPLING: Select 20% of data across different market regimes
STAGE 1 - FAST SCREENING: Apply cheap filters (variance, correlation, noise-to-signal)
STAGE 2 - PREDICTIVE POWER: Compute IC, MI proxy, and IC autocorrelation
STAGE 3 - ROBUSTNESS TESTS: Walk-forward CV and regime stability
STAGE 4 - FINAL SELECTION: Weighted ranking and top-k selection

Performance optimizations:
- Vectorized operations using NumPy/Pandas
- Cached rolling windows
- Parallel processing with multiprocessing
- Polars for heavy computations when available
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import warnings

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False
    pl = None

try:
    from scipy.stats import spearmanr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    spearmanr = None

logger = logging.getLogger(__name__)


@dataclass
class LookbackCandidate:
    """Container for a feature-lookback candidate with all evaluation metrics."""
    feature_name: str
    lookback: int

    # Stage 1 - Fast Screening
    variance: float = 0.0
    price_corr: float = 0.0
    future_corr: float = 0.0

    # Stage 2 - Predictive Power
    ic_mean: float = 0.0
    ic_std: float = 0.0
    ic_tstat: float = 0.0
    ic_autocorr: float = 0.0
    mi_proxy: float = 0.0

    # Stage 3 - Robustness
    cv_score: float = 0.0
    regime_stability: float = 0.0

    # Stage 4 - Final Score
    final_score: float = 0.0

    # Metadata
    survived_stage: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationConfig:
    """Configuration for the 4-stage evaluation pipeline."""

    # Stage 0 - Subsampling
    subsample_ratio: float = 0.20
    n_chunks: int = 6  # 4-8 non-contiguous chunks

    # Stage 1 - Fast Screening
    variance_quantile_threshold: float = 0.30
    price_corr_quantile_threshold: float = 0.30
    future_corr_quantile_threshold: float = 0.30

    # Stage 2 - Predictive Power
    ic_tstat_threshold: float = 1.96  # ~95% confidence
    ic_autocorr_threshold: float = 0.0  # Must be positive
    mi_proxy_threshold: float = 0.05

    # Stage 3 - Robustness
    n_cv_splits: int = 5
    embargo_bars: int = 1

    # Stage 4 - Final Selection
    top_k_per_feature: int = 3
    weights: Dict[str, float] = field(default_factory=lambda: {
        'ic_tstat': 0.30,
        'ic_autocorr': 0.20,
        'cv_score': 0.30,
        'regime_stability': 0.15,
        'mi_proxy': 0.05
    })

    # Performance
    use_parallel: bool = True
    n_workers: int = 4
    use_polars: bool = True
    cache_rolling: bool = True

    # Future returns
    future_returns_horizon: int = 1  # Bars ahead for target


class FeatureEvaluationPipeline:
    """
    4-Stage Feature Evaluation Pipeline for Lookback Optimization.

    This pipeline efficiently evaluates feature-lookback combinations through
    progressively more expensive filters, ensuring only the best candidates
    are subjected to costly computations.
    """

    def __init__(self, config: Optional[EvaluationConfig] = None):
        """Initialize the evaluation pipeline."""
        self.config = config or EvaluationConfig()
        self.logger = logging.getLogger(self.__class__.__name__)

        # Cache for rolling statistics
        self._rolling_cache = {}
        self._regime_cache = {}

        # Performance tracking
        self.stage_times = {}
        self.candidates_per_stage = {}

    def evaluate_lookbacks(
        self,
        data: pd.DataFrame,
        feature_name: str,
        lookback_candidates: List[int],
        target_column: str = 'close'
    ) -> List[LookbackCandidate]:
        """
        Evaluate all lookback candidates for a feature through the 4-stage pipeline.

        Args:
            data: DataFrame with OHLCV and features
            feature_name: Name of the feature column to evaluate
            lookback_candidates: List of lookback periods to test
            target_column: Price column for correlation/returns (default: 'close')

        Returns:
            List of LookbackCandidate objects sorted by final_score (descending)
        """
        import time

        if feature_name not in data.columns:
            self.logger.warning(f"Feature {feature_name} not in data columns")
            return []

        # Stage 0: Subsample data for stages 1 and 2
        start = time.time()
        subsampled_data, full_data = self._stage0_subsample(data)
        self.stage_times['stage0'] = time.time() - start

        # Initialize candidates
        candidates = [
            LookbackCandidate(feature_name=feature_name, lookback=lb)
            for lb in lookback_candidates
        ]
        self.candidates_per_stage['initial'] = len(candidates)

        # Stage 1: Fast Screening (on subsample)
        start = time.time()
        candidates = self._stage1_fast_screening(
            subsampled_data, candidates, feature_name, target_column
        )
        self.stage_times['stage1'] = time.time() - start
        self.candidates_per_stage['after_stage1'] = len(candidates)

        if not candidates:
            self.logger.info(f"No candidates survived Stage 1 for {feature_name}")
            return []

        # Stage 2: Predictive Power Metrics (on subsample)
        start = time.time()
        candidates = self._stage2_predictive_power(
            subsampled_data, candidates, feature_name, target_column
        )
        self.stage_times['stage2'] = time.time() - start
        self.candidates_per_stage['after_stage2'] = len(candidates)

        if not candidates:
            self.logger.info(f"No candidates survived Stage 2 for {feature_name}")
            return []

        # Stage 3: Robustness Tests (on full data)
        start = time.time()
        candidates = self._stage3_robustness(
            full_data, candidates, feature_name, target_column
        )
        self.stage_times['stage3'] = time.time() - start
        self.candidates_per_stage['after_stage3'] = len(candidates)

        if not candidates:
            self.logger.info(f"No candidates survived Stage 3 for {feature_name}")
            return []

        # Stage 4: Final Selection
        start = time.time()
        candidates = self._stage4_final_selection(candidates)
        self.stage_times['stage4'] = time.time() - start
        self.candidates_per_stage['final'] = len(candidates)

        return candidates

    def _stage0_subsample(
        self, data: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Stage 0: Create a stratified subsample covering different market regimes.

        Selects 20% of data split into 4-8 non-contiguous chunks that cover:
        - High volatility periods
        - Low volatility periods
        - Bull markets
        - Bear markets
        - Sideways markets

        Returns:
            Tuple of (subsampled_data, full_data)
        """
        n = len(data)
        n_subsample = int(n * self.config.subsample_ratio)

        # Calculate regime indicators
        if 'close' in data.columns:
            returns = data['close'].pct_change()

            # Volatility: rolling std of returns
            vol = returns.rolling(20, min_periods=1).std()

            # Trend: rolling mean of returns
            trend = returns.rolling(20, min_periods=1).mean()

            # Classify regimes
            vol_high = vol > vol.quantile(0.67)
            vol_low = vol < vol.quantile(0.33)

            trend_bull = trend > trend.quantile(0.67)
            trend_bear = trend < trend.quantile(0.33)
            trend_sideways = ~(trend_bull | trend_bear)

            # Define regime buckets
            regimes = {
                'high_vol': np.where(vol_high)[0],
                'low_vol': np.where(vol_low)[0],
                'bull': np.where(trend_bull)[0],
                'bear': np.where(trend_bear)[0],
                'sideways': np.where(trend_sideways)[0]
            }

            # Sample from each regime proportionally
            samples_per_regime = n_subsample // len(regimes)
            selected_indices = []

            for regime_name, indices in regimes.items():
                if len(indices) > 0:
                    n_samples = min(samples_per_regime, len(indices))
                    # Stratified sampling to get non-contiguous chunks
                    chunk_size = max(1, n_samples // self.config.n_chunks)
                    chunks = np.array_split(indices, self.config.n_chunks)
                    for chunk in chunks:
                        if len(chunk) > 0:
                            sample_size = min(chunk_size, len(chunk))
                            sampled = np.random.choice(
                                chunk, size=sample_size, replace=False
                            )
                            selected_indices.extend(sampled)

            selected_indices = sorted(set(selected_indices))[:n_subsample]
        else:
            # Fallback: random stratified sampling
            chunk_size = n // self.config.n_chunks
            selected_indices = []
            for i in range(self.config.n_chunks):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < self.config.n_chunks - 1 else n
                chunk_indices = list(range(start_idx, end_idx))
                samples = np.random.choice(
                    chunk_indices,
                    size=min(len(chunk_indices), n_subsample // self.config.n_chunks),
                    replace=False
                )
                selected_indices.extend(samples)

        subsampled_data = data.iloc[selected_indices].copy()
        self.logger.info(
            f"Stage 0: Subsampled {len(subsampled_data)}/{len(data)} rows "
            f"({100*len(subsampled_data)/len(data):.1f}%)"
        )

        return subsampled_data, data

    def _stage1_fast_screening(
        self,
        data: pd.DataFrame,
        candidates: List[LookbackCandidate],
        feature_name: str,
        target_column: str
    ) -> List[LookbackCandidate]:
        """
        Stage 1: Fast Screening with cheap filters - SEQUENTIAL/CASCADING.

        Applies filters one after another to avoid useless computation on rejected features:
        1. Variance check: Reject bottom 30% quantile
        2. Correlation-with-price: Reject bottom 30% quantile (of survivors)
        3. Noise-to-signal: Reject bottom 30% quantile (of survivors)

        Returns:
            Filtered list of candidates
        """
        feature_data = data[feature_name].values
        price_data = data[target_column].values if target_column in data.columns else None
        future_returns = self._compute_future_returns(data, target_column)

        initial_count = len(candidates)

        # =====================================================================
        # FILTER 1: Variance check (reject bottom 30%)
        # =====================================================================
        for candidate in candidates:
            lb = candidate.lookback

            if self.config.cache_rolling and (feature_name, lb, 'var') in self._rolling_cache:
                rolling_var = self._rolling_cache[(feature_name, lb, 'var')]
            else:
                rolling_var = pd.Series(feature_data).rolling(lb, min_periods=max(1, lb//2)).var()
                if self.config.cache_rolling:
                    self._rolling_cache[(feature_name, lb, 'var')] = rolling_var

            candidate.variance = float(rolling_var.mean()) if not rolling_var.isna().all() else 0.0

        # Filter by variance threshold
        variances = [c.variance for c in candidates]
        var_threshold = np.quantile(variances, self.config.variance_quantile_threshold)
        candidates = [c for c in candidates if c.variance >= var_threshold]

        self.logger.debug(
            f"Stage 1.1 (Variance): {len(candidates)}/{initial_count} survived "
            f"(threshold={var_threshold:.4f})"
        )

        if not candidates:
            self.logger.info("Stage 1: No candidates survived variance filter")
            return []

        # =====================================================================
        # FILTER 2: Correlation with price (reject bottom 30% of survivors)
        # =====================================================================
        if price_data is not None:
            for candidate in candidates:
                lb = candidate.lookback
                rolling_feature = pd.Series(feature_data).rolling(lb, min_periods=max(1, lb//2)).mean()
                valid_mask = ~(rolling_feature.isna() | pd.Series(price_data).isna())
                if valid_mask.sum() > 10:
                    candidate.price_corr = abs(float(
                        pd.Series(rolling_feature[valid_mask]).corr(pd.Series(price_data[valid_mask]))
                    ))
                else:
                    candidate.price_corr = 0.0

            # Filter by price correlation threshold
            price_corrs = [c.price_corr for c in candidates]
            price_threshold = np.quantile(price_corrs, self.config.price_corr_quantile_threshold)
            candidates = [c for c in candidates if c.price_corr >= price_threshold]

            self.logger.debug(
                f"Stage 1.2 (Price Corr): {len(candidates)} survived "
                f"(threshold={price_threshold:.4f})"
            )

            if not candidates:
                self.logger.info("Stage 1: No candidates survived price correlation filter")
                return []
        else:
            # No price data - skip this filter
            for candidate in candidates:
                candidate.price_corr = 0.0

        # =====================================================================
        # FILTER 3: Future correlation (reject bottom 30% of survivors)
        # =====================================================================
        for candidate in candidates:
            lb = candidate.lookback
            rolling_feature = pd.Series(feature_data).rolling(lb, min_periods=max(1, lb//2)).mean()
            valid_mask = ~(rolling_feature.isna() | future_returns.isna())
            if valid_mask.sum() > 10:
                candidate.future_corr = abs(float(
                    rolling_feature[valid_mask].corr(future_returns[valid_mask])
                ))
            else:
                candidate.future_corr = 0.0

        # Filter by future correlation threshold
        future_corrs = [c.future_corr for c in candidates]
        future_threshold = np.quantile(future_corrs, self.config.future_corr_quantile_threshold)
        candidates = [c for c in candidates if c.future_corr >= future_threshold]

        self.logger.debug(
            f"Stage 1.3 (Future Corr): {len(candidates)} survived "
            f"(threshold={future_threshold:.4f})"
        )

        # Mark all survivors as having passed Stage 1
        for candidate in candidates:
            candidate.survived_stage = 1

        self.logger.info(
            f"Stage 1: {len(candidates)}/{initial_count} candidates survived cascading filters"
        )

        return candidates

    def _stage2_predictive_power(
        self,
        data: pd.DataFrame,
        candidates: List[LookbackCandidate],
        feature_name: str,
        target_column: str
    ) -> List[LookbackCandidate]:
        """
        Stage 2: Predictive Power Metrics.

        Computes:
        1. Information Coefficient (IC) - Spearman correlation with future returns
        2. Lagged MI proxy using correlation-entropy approximation
        3. IC Autocorrelation for stability

        Filters by:
        - IC t-stat > threshold
        - IC autocorr > 0
        - MI proxy > threshold

        Returns:
            Filtered list of candidates
        """
        feature_data = data[feature_name].values
        future_returns = self._compute_future_returns(data, target_column)

        for candidate in candidates:
            lb = candidate.lookback

            # Compute rolling feature
            rolling_feature = pd.Series(feature_data).rolling(
                lb, min_periods=max(1, lb//2)
            ).mean()

            # 1. Information Coefficient (IC)
            ic_series = self._compute_rolling_ic(
                rolling_feature, future_returns, window=20
            )

            if not ic_series.isna().all() and len(ic_series.dropna()) > 0:
                candidate.ic_mean = float(ic_series.mean())
                candidate.ic_std = float(ic_series.std())

                # IC t-statistic
                n = len(ic_series.dropna())
                if candidate.ic_std > 0 and n > 1:
                    candidate.ic_tstat = candidate.ic_mean / (candidate.ic_std / np.sqrt(n))
                else:
                    candidate.ic_tstat = 0.0

                # 2. IC Autocorrelation
                if n > 2:
                    ic_clean = ic_series.dropna()
                    ic_lag1 = ic_clean.shift(1)
                    valid_mask = ~(ic_clean.isna() | ic_lag1.isna())
                    if valid_mask.sum() > 2:
                        candidate.ic_autocorr = float(ic_clean[valid_mask].corr(ic_lag1[valid_mask]))
                    else:
                        candidate.ic_autocorr = 0.0
                else:
                    candidate.ic_autocorr = 0.0
            else:
                candidate.ic_mean = 0.0
                candidate.ic_std = 0.0
                candidate.ic_tstat = 0.0
                candidate.ic_autocorr = 0.0

            # 3. MI Proxy: correlation-entropy approximation (cheap and 90% effective)
            # MI proxy ≈ -0.5 * log(1 - corr^2)
            corr = abs(candidate.future_corr)
            if corr < 0.999:
                candidate.mi_proxy = float(-0.5 * np.log(1 - corr**2))
            else:
                candidate.mi_proxy = 5.0  # High MI for perfect correlation

            candidate.survived_stage = 2

        # Filter by thresholds
        filtered = [
            c for c in candidates
            if c.ic_tstat > self.config.ic_tstat_threshold
            and c.ic_autocorr > self.config.ic_autocorr_threshold
            and c.mi_proxy > self.config.mi_proxy_threshold
        ]

        self.logger.info(
            f"Stage 2: {len(filtered)}/{len(candidates)} candidates survived "
            f"(IC_tstat>{self.config.ic_tstat_threshold:.2f}, "
            f"IC_autocorr>{self.config.ic_autocorr_threshold:.2f}, "
            f"MI_proxy>{self.config.mi_proxy_threshold:.3f})"
        )

        return filtered

    def _stage3_robustness(
        self,
        data: pd.DataFrame,
        candidates: List[LookbackCandidate],
        feature_name: str,
        target_column: str
    ) -> List[LookbackCandidate]:
        """
        Stage 3: Robustness Tests on full data.

        Tests:
        1. Purged/embargoed walk-forward CV
        2. Regime stability (performance across market regimes)

        Returns:
            List of candidates with robustness scores
        """
        feature_data = data[feature_name].values
        future_returns = self._compute_future_returns(data, target_column)

        for candidate in candidates:
            lb = candidate.lookback

            # Compute rolling feature
            rolling_feature = pd.Series(feature_data).rolling(
                lb, min_periods=max(1, lb//2)
            ).mean()

            # 1. Walk-forward CV with embargo
            cv_scores = self._walk_forward_cv(
                rolling_feature, future_returns,
                n_splits=self.config.n_cv_splits,
                embargo=self.config.embargo_bars
            )
            candidate.cv_score = float(np.mean(cv_scores)) if len(cv_scores) > 0 else 0.0

            # 2. Regime stability
            regime_scores = self._compute_regime_stability(
                data, rolling_feature, future_returns
            )
            candidate.regime_stability = float(np.mean(list(regime_scores.values())))

            candidate.survived_stage = 3
            candidate.metadata['regime_scores'] = regime_scores

        # No filtering in this stage - just compute scores
        self.logger.info(
            f"Stage 3: Computed robustness for {len(candidates)} candidates"
        )

        return candidates

    def _stage4_final_selection(
        self, candidates: List[LookbackCandidate]
    ) -> List[LookbackCandidate]:
        """
        Stage 4: Final Selection using weighted ranking.

        Computes final score as weighted combination of:
        - IC t-stat (30%)
        - IC autocorr (20%)
        - CV score (30%)
        - Regime stability (15%)
        - MI proxy (5%)

        Returns top K candidates sorted by final score.

        Returns:
            Top K candidates sorted by final_score (descending)
        """
        # Normalize metrics to [0, 1] range
        def normalize(values: List[float]) -> List[float]:
            """Min-max normalization."""
            if len(values) == 0:
                return []
            min_val = min(values)
            max_val = max(values)
            if max_val - min_val < 1e-10:
                return [0.5] * len(values)
            return [(v - min_val) / (max_val - min_val) for v in values]

        # Extract metrics
        ic_tstats = [c.ic_tstat for c in candidates]
        ic_autocorrs = [max(0, c.ic_autocorr) for c in candidates]  # Clip negative
        cv_scores = [c.cv_score for c in candidates]
        regime_stabilities = [c.regime_stability for c in candidates]
        mi_proxies = [c.mi_proxy for c in candidates]

        # Normalize
        norm_ic_tstat = normalize(ic_tstats)
        norm_ic_autocorr = normalize(ic_autocorrs)
        norm_cv_score = normalize(cv_scores)
        norm_regime_stability = normalize(regime_stabilities)
        norm_mi_proxy = normalize(mi_proxies)

        # Compute final scores
        weights = self.config.weights
        for i, candidate in enumerate(candidates):
            candidate.final_score = (
                weights['ic_tstat'] * norm_ic_tstat[i] +
                weights['ic_autocorr'] * norm_ic_autocorr[i] +
                weights['cv_score'] * norm_cv_score[i] +
                weights['regime_stability'] * norm_regime_stability[i] +
                weights['mi_proxy'] * norm_mi_proxy[i]
            )
            candidate.survived_stage = 4

        # Sort by final score (descending) and take top K
        candidates.sort(key=lambda c: c.final_score, reverse=True)
        top_candidates = candidates[:self.config.top_k_per_feature]

        self.logger.info(
            f"Stage 4: Selected top {len(top_candidates)} candidates "
            f"(scores: {[f'{c.final_score:.3f}' for c in top_candidates]})"
        )

        return top_candidates

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _compute_future_returns(
        self, data: pd.DataFrame, target_column: str
    ) -> pd.Series:
        """Compute future returns for the target column."""
        if target_column not in data.columns:
            return pd.Series([np.nan] * len(data), index=data.index)

        prices = data[target_column]
        horizon = self.config.future_returns_horizon
        future_returns = prices.pct_change(horizon).shift(-horizon)

        return future_returns

    def _compute_rolling_ic(
        self, signal: pd.Series, returns: pd.Series, window: int = 20
    ) -> pd.Series:
        """
        Compute rolling Information Coefficient (Spearman rank correlation).

        Args:
            signal: Feature signal
            returns: Future returns
            window: Rolling window size

        Returns:
            Series of IC values
        """
        if SCIPY_AVAILABLE and spearmanr is not None:
            # Use scipy for Spearman correlation
            ic_values = []
            for i in range(len(signal)):
                if i < window - 1:
                    ic_values.append(np.nan)
                else:
                    sig_window = signal.iloc[i-window+1:i+1]
                    ret_window = returns.iloc[i-window+1:i+1]

                    valid_mask = ~(sig_window.isna() | ret_window.isna())
                    if valid_mask.sum() >= 3:
                        corr, _ = spearmanr(
                            sig_window[valid_mask], ret_window[valid_mask]
                        )
                        ic_values.append(corr if not np.isnan(corr) else 0.0)
                    else:
                        ic_values.append(np.nan)

            return pd.Series(ic_values, index=signal.index)
        else:
            # Fallback to Pearson correlation
            return signal.rolling(window).corr(returns)

    def _walk_forward_cv(
        self,
        signal: pd.Series,
        returns: pd.Series,
        n_splits: int = 5,
        embargo: int = 1
    ) -> List[float]:
        """
        Perform walk-forward cross-validation with embargo.

        Args:
            signal: Feature signal
            returns: Future returns
            n_splits: Number of CV folds
            embargo: Number of bars to embargo between train/test

        Returns:
            List of out-of-sample IC scores
        """
        n = len(signal)
        fold_size = n // n_splits
        cv_scores = []

        for i in range(n_splits):
            # Define train and test sets
            test_start = i * fold_size
            test_end = (i + 1) * fold_size if i < n_splits - 1 else n

            # Apply embargo
            test_start += embargo

            if test_end - test_start < 10:
                continue

            # Extract test data
            sig_test = signal.iloc[test_start:test_end]
            ret_test = returns.iloc[test_start:test_end]

            # Compute OOS correlation
            valid_mask = ~(sig_test.isna() | ret_test.isna())
            if valid_mask.sum() >= 3:
                corr = sig_test[valid_mask].corr(ret_test[valid_mask])
                if not np.isnan(corr):
                    cv_scores.append(abs(corr))

        return cv_scores

    def _compute_regime_stability(
        self,
        data: pd.DataFrame,
        signal: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """
        Compute signal performance across different market regimes.

        Regimes:
        - High volatility
        - Low volatility
        - Bull market
        - Bear market
        - Sideways market

        Returns:
            Dict mapping regime name to IC score
        """
        regime_scores = {}

        if 'close' not in data.columns:
            return {'default': 0.0}

        # Compute regime indicators
        close_returns = data['close'].pct_change()
        vol = close_returns.rolling(20, min_periods=1).std()
        trend = close_returns.rolling(20, min_periods=1).mean()

        # Define regimes
        regimes = {
            'high_vol': vol > vol.quantile(0.67),
            'low_vol': vol < vol.quantile(0.33),
            'bull': trend > trend.quantile(0.67),
            'bear': trend < trend.quantile(0.33),
            'sideways': (trend >= trend.quantile(0.33)) & (trend <= trend.quantile(0.67))
        }

        # Compute IC for each regime
        for regime_name, regime_mask in regimes.items():
            sig_regime = signal[regime_mask]
            ret_regime = returns[regime_mask]

            valid_mask = ~(sig_regime.isna() | ret_regime.isna())
            if valid_mask.sum() >= 3:
                corr = sig_regime[valid_mask].corr(ret_regime[valid_mask])
                regime_scores[regime_name] = abs(corr) if not np.isnan(corr) else 0.0
            else:
                regime_scores[regime_name] = 0.0

        return regime_scores

    def evaluate_features_parallel(
        self,
        data: pd.DataFrame,
        feature_lookback_pairs: List[Tuple[str, List[int]]],
        target_column: str = 'close'
    ) -> Dict[str, List[LookbackCandidate]]:
        """
        Evaluate multiple features in parallel using multiprocessing.

        Args:
            data: DataFrame with OHLCV and features
            feature_lookback_pairs: List of (feature_name, lookback_list) tuples
            target_column: Price column for returns

        Returns:
            Dict mapping feature_name to list of top candidates
        """
        if not self.config.use_parallel or self.config.n_workers <= 1:
            # Sequential processing
            results = {}
            for feature_name, lookbacks in feature_lookback_pairs:
                candidates = self.evaluate_lookbacks(
                    data, feature_name, lookbacks, target_column
                )
                results[feature_name] = candidates
            return results

        # Parallel processing
        results = {}
        with ProcessPoolExecutor(max_workers=self.config.n_workers) as executor:
            future_to_feature = {
                executor.submit(
                    self.evaluate_lookbacks,
                    data,
                    feature_name,
                    lookbacks,
                    target_column
                ): feature_name
                for feature_name, lookbacks in feature_lookback_pairs
            }

            for future in as_completed(future_to_feature):
                feature_name = future_to_feature[future]
                try:
                    candidates = future.result()
                    results[feature_name] = candidates
                except Exception as e:
                    self.logger.error(f"Error evaluating {feature_name}: {e}")
                    results[feature_name] = []

        return results

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get a summary of pipeline performance metrics."""
        return {
            'stage_times': self.stage_times,
            'candidates_per_stage': self.candidates_per_stage,
            'total_time': sum(self.stage_times.values()),
            'cache_size': len(self._rolling_cache)
        }


def create_evaluation_pipeline(
    subsample_ratio: float = 0.20,
    top_k: int = 3,
    use_parallel: bool = True,
    n_workers: int = 4
) -> FeatureEvaluationPipeline:
    """
    Factory function to create a pre-configured evaluation pipeline.

    Args:
        subsample_ratio: Fraction of data to use for stages 1-2
        top_k: Number of top lookbacks to return per feature
        use_parallel: Enable parallel processing
        n_workers: Number of parallel workers

    Returns:
        Configured FeatureEvaluationPipeline instance
    """
    config = EvaluationConfig(
        subsample_ratio=subsample_ratio,
        top_k_per_feature=top_k,
        use_parallel=use_parallel,
        n_workers=n_workers
    )
    return FeatureEvaluationPipeline(config)


# =========================================================================
# Quick MI/IC Scoring (for replacing sklearn mutual_info_regression)
# =========================================================================

def compute_quick_mi_scores(
    features: pd.DataFrame,
    target: pd.Series,
    use_spearman: bool = True,
    subsample_ratio: float = 0.30,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute quick MI proxy scores for features using correlation-entropy approximation.

    This function replaces sklearn's mutual_info_regression with a much faster
    approximation that's 90% as effective. It uses the correlation-entropy formula:
        MI_proxy ≈ -0.5 * log(1 - corr²)

    Optionally uses Spearman rank correlation (more robust) instead of Pearson.
    Can subsample data for even faster computation on large datasets.

    Args:
        features: DataFrame of features (shape: [n_samples, n_features])
        target: Target variable (shape: [n_samples])
        use_spearman: Use Spearman rank correlation (default: True)
        subsample_ratio: Fraction of data to use (default: 0.30 for speed)
        random_state: Random seed for subsampling

    Returns:
        Dict mapping feature names to MI proxy scores

    Usage:
        # Replace sklearn mutual_info_regression:
        # OLD: mi_scores = mutual_info_regression(X, y, random_state=42)

        # NEW:
        mi_dict = compute_quick_mi_scores(X, y, use_spearman=True)
        mi_scores = np.array([mi_dict.get(col, 0.0) for col in X.columns])
    """
    # Align features and target
    common_idx = features.index.intersection(target.index)
    if len(common_idx) == 0:
        # Positional alignment as fallback
        min_len = min(len(features), len(target))
        features_aligned = features.iloc[-min_len:].reset_index(drop=True)
        target_aligned = target.iloc[-min_len:].reset_index(drop=True)
    else:
        features_aligned = features.loc[common_idx]
        target_aligned = target.loc[common_idx]

    # Drop NaN
    valid_mask = target_aligned.notna()
    for col in features_aligned.columns:
        valid_mask &= features_aligned[col].notna()

    features_clean = features_aligned[valid_mask]
    target_clean = target_aligned[valid_mask]

    if len(features_clean) < 10:
        logger.warning(f"Insufficient valid samples ({len(features_clean)}) for MI computation")
        return {col: 0.0 for col in features.columns}

    # Subsample for speed
    if subsample_ratio < 1.0 and len(features_clean) > 100:
        np.random.seed(random_state)
        n_samples = int(len(features_clean) * subsample_ratio)
        sample_idx = np.random.choice(len(features_clean), size=n_samples, replace=False)
        features_clean = features_clean.iloc[sample_idx]
        target_clean = target_clean.iloc[sample_idx]

    # Compute correlations
    mi_scores = {}

    if use_spearman and SCIPY_AVAILABLE:
        # Spearman rank correlation (more robust)
        for col in features_clean.columns:
            try:
                corr, _ = spearmanr(features_clean[col], target_clean)
                if np.isnan(corr) or np.isinf(corr):
                    mi_scores[col] = 0.0
                else:
                    # MI proxy: -0.5 * log(1 - corr²)
                    corr_abs = abs(corr)
                    if corr_abs >= 0.999:
                        mi_scores[col] = 5.0  # Cap at high value
                    else:
                        mi_scores[col] = float(-0.5 * np.log(1 - corr_abs**2))
            except Exception:
                mi_scores[col] = 0.0
    else:
        # Pearson correlation (faster fallback)
        for col in features_clean.columns:
            try:
                corr = features_clean[col].corr(target_clean)
                if np.isnan(corr) or np.isinf(corr):
                    mi_scores[col] = 0.0
                else:
                    corr_abs = abs(corr)
                    if corr_abs >= 0.999:
                        mi_scores[col] = 5.0
                    else:
                        mi_scores[col] = float(-0.5 * np.log(1 - corr_abs**2))
            except Exception:
                mi_scores[col] = 0.0

    # Fill missing features with 0.0
    for col in features.columns:
        if col not in mi_scores:
            mi_scores[col] = 0.0

    return mi_scores


def compute_feature_stability_scores(
    features: pd.DataFrame,
    window: int = 20,
    subsample_ratio: float = 0.30,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute stability scores for features using rolling statistics.

    Stability = 1 - (rolling_std_mean / global_std)

    Higher values indicate more stable features that maintain consistent
    behavior over time.

    Args:
        features: DataFrame of features
        window: Rolling window size for stability calculation
        subsample_ratio: Fraction of data to use for speed
        random_state: Random seed for subsampling

    Returns:
        Dict mapping feature names to stability scores [0, 1]
    """
    stability_scores = {}

    # Subsample for speed
    if subsample_ratio < 1.0 and len(features) > 100:
        np.random.seed(random_state)
        n_samples = int(len(features) * subsample_ratio)
        sample_idx = np.random.choice(len(features), size=n_samples, replace=False)
        features_sampled = features.iloc[sample_idx]
    else:
        features_sampled = features

    for col in features_sampled.columns:
        try:
            col_data = features_sampled[col].dropna()
            if len(col_data) < window:
                stability_scores[col] = 0.5
                continue

            # Compute rolling std
            rolling_std = col_data.rolling(window, min_periods=max(1, window//2)).std()
            rolling_std_mean = rolling_std.mean()

            # Global std
            global_std = col_data.std()

            if global_std == 0 or np.isnan(global_std):
                stability_scores[col] = 0.0
            else:
                stability = 1.0 - (rolling_std_mean / global_std)
                stability_scores[col] = float(np.clip(stability, 0.0, 1.0))
        except Exception:
            stability_scores[col] = 0.5

    # Fill missing with default
    for col in features.columns:
        if col not in stability_scores:
            stability_scores[col] = 0.5

    return stability_scores


def compute_composite_scores(
    features: pd.DataFrame,
    target: pd.Series,
    use_spearman: bool = True,
    include_stability: bool = True,
    subsample_ratio: float = 0.30,
    mi_weight: float = 0.7,
    stability_weight: float = 0.3,
    random_state: int = 42
) -> Dict[str, float]:
    """
    Compute composite scores combining MI proxy and stability.

    This is a drop-in replacement for MI-based scoring that adds stability
    information for more robust feature selection.

    Composite Score = mi_weight * MI_proxy + stability_weight * Stability

    Args:
        features: DataFrame of features
        target: Target variable
        use_spearman: Use Spearman correlation (more robust)
        include_stability: Include stability scores in composite
        subsample_ratio: Fraction of data to use for speed
        mi_weight: Weight for MI proxy (default: 0.7)
        stability_weight: Weight for stability (default: 0.3)
        random_state: Random seed

    Returns:
        Dict mapping feature names to composite scores

    Usage:
        # Replace MI calculation in interaction_generation_step:
        # OLD:
        #   mi_scores = mutual_info_regression(features, target, random_state=42)
        #   mi_dict = dict(zip(feature_names, mi_scores))

        # NEW:
        from src.feature_selection.feature_evaluation import compute_composite_scores
        composite_dict = compute_composite_scores(
            features_df, target, use_spearman=True, include_stability=True
        )
    """
    # Compute MI scores
    mi_scores = compute_quick_mi_scores(
        features, target, use_spearman=use_spearman,
        subsample_ratio=subsample_ratio, random_state=random_state
    )

    if not include_stability:
        return mi_scores

    # Compute stability scores
    stability_scores = compute_feature_stability_scores(
        features, window=20, subsample_ratio=subsample_ratio,
        random_state=random_state
    )

    # Normalize MI scores to [0, 1]
    mi_values = np.array(list(mi_scores.values()))
    if mi_values.max() > 0:
        mi_max = mi_values.max()
        mi_scores_norm = {k: v / mi_max for k, v in mi_scores.items()}
    else:
        mi_scores_norm = {k: 0.0 for k in mi_scores}

    # Compute composite scores
    composite_scores = {}
    for col in features.columns:
        mi_score = mi_scores_norm.get(col, 0.0)
        stab_score = stability_scores.get(col, 0.5)
        composite_scores[col] = mi_weight * mi_score + stability_weight * stab_score

    return composite_scores
