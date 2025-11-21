"""
Liquidity Cluster Quality Assessor

This module assesses the quality of liquidity-based regimes, focusing on:
- Effort vs Result separation (volume vs price range)
- Trap/Ghost behavior (fake moves on low participation)
- Absorption behavior (high volume + low move)
- Trend confirmation and apathy/noise characterization

Returns are used only as secondary diagnostics (e.g., 1h forward return),
while the primary metrics operate directly on liquidity and microstructure
features.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

import numpy as np
import pandas as pd

try:
    from src.utils.tprint import (
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success,
    )
except ImportError:
    tprint_info = logging.info
    tprint_warning = logging.warning
    tprint_error = logging.error
    tprint_success = logging.info

logger = logging.getLogger(__name__)


@dataclass
class LiquidityClusterQualityMetrics:
    """Liquidity-specific cluster quality metrics."""

    # Effort vs Result separation
    effort_result_separation_score: float = 0.0
    ghost_vs_valid_contrast: float = 0.0
    absorption_vs_valid_contrast: float = 0.0

    # CoV-based separation diagnostics
    effort_result_cov_separation_score: float = 0.0
    returns_cov_separation_score: float = 0.0

    # Trap / ghost quality (using secondary 1h forward returns if available)
    ghost_reversal_rate: float = 0.0
    ghost_false_trend_rate: float = 0.0

    # Absorption quality
    absorption_reversal_rate: float = 0.0
    absorption_follow_through_rate: float = 0.0

    # Trend confirmation & apathy
    valid_trend_follow_through: float = 0.0
    apathy_noise_fraction: float = 0.0

    # Class balance & coverage
    class_balance_score: float = 0.0
    n_regimes: int = 0
    n_samples: int = 0

    # Per-regime detailed metrics
    per_regime_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Feature distinctiveness analysis (winsorized CoV ratios)
    distinctiveness_analysis: Dict[str, Any] = field(default_factory=dict)

    # Overall quality score
    overall_quality_score: float = 0.0

    # Metadata
    assessment_timestamp: str = ""


class LiquidityClusterQualityAssessor:
    """Assess quality of liquidity-based regime clusters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logger

    def assess_liquidity_clusters(
        self,
        liquidity_df: pd.DataFrame,
        regime_labels: pd.Series,
        forward_returns_1h: Optional[pd.Series] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> LiquidityClusterQualityMetrics:
        """Assess quality of liquidity-based regimes.

        Args:
            liquidity_df: DataFrame with liquidity/microstructure features.
            regime_labels: Series with regime assignments (0=Apathy,1=Valid,2=Absorption,3=Ghost).
            forward_returns_1h: Optional 1h forward log returns (secondary diagnostics).
            config: Optional configuration dict.

        Returns:
            LiquidityClusterQualityMetrics with assessment results.
        """
        cfg = dict(self.config)
        if config:
            cfg.update(config)

        common_idx = liquidity_df.index.intersection(regime_labels.index)
        df = liquidity_df.loc[common_idx].copy()
        labels = regime_labels.loc[common_idx].astype(int)

        if len(df) == 0 or len(labels) == 0:
            tprint_warning("Empty data for liquidity cluster assessment")
            return LiquidityClusterQualityMetrics()

        if forward_returns_1h is not None:
            fwd = forward_returns_1h.reindex(common_idx)
        else:
            fwd = None

        n_regimes = int(labels.nunique())
        n_samples = len(df)
        tprint_info(f"📊 Assessing {n_regimes} liquidity regimes on {n_samples} samples")

        # 1. Effort vs Result separation
        effort_sep, ghost_valid_contrast, absorption_valid_contrast = self._assess_effort_result_separation(
            df,
            labels,
        )

        # 2. Trap / ghost quality
        ghost_rev_rate, ghost_false_trend = self._assess_ghost_behavior(
            df,
            labels,
            forward_returns=fwd,
        )

        # 3. Absorption quality
        absorption_rev_rate, absorption_follow = self._assess_absorption_behavior(
            df,
            labels,
            forward_returns=fwd,
        )

        # 4. Trend confirmation & apathy
        valid_follow, apathy_noise = self._assess_trend_and_apathy(
            df,
            labels,
            forward_returns=fwd,
        )

        # 5. Class balance
        class_balance_score = self._assess_class_balance(labels)

        # 6. Per-regime metrics
        per_regime_metrics = self._calculate_per_regime_metrics(
            df,
            labels,
            forward_returns=fwd,
        )

        # 6b. CoV-based separation diagnostics (feature and returns patterns)
        cov_effort_sep, cov_returns_sep = self._compute_cov_based_scores(per_regime_metrics)

        # 6c. Feature distinctiveness analysis (winsorized CoV ratios)
        distinctiveness_analysis = self._compute_feature_distinctiveness(
            df,
            labels,
            per_regime_metrics,
            top_n=10,
        )

        # 7. Overall quality score (weighted combination)
        overall_quality_score = self._calculate_overall_quality_score(
            effort_sep,
            ghost_valid_contrast,
            absorption_valid_contrast,
            ghost_rev_rate,
            ghost_false_trend,
            absorption_rev_rate,
            absorption_follow,
            valid_follow,
            apathy_noise,
            class_balance_score,
            cov_effort_sep,
            cov_returns_sep,
        )

        metrics = LiquidityClusterQualityMetrics(
            effort_result_separation_score=effort_sep,
            ghost_vs_valid_contrast=ghost_valid_contrast,
            absorption_vs_valid_contrast=absorption_valid_contrast,
            effort_result_cov_separation_score=cov_effort_sep,
            returns_cov_separation_score=cov_returns_sep,
            ghost_reversal_rate=ghost_rev_rate,
            ghost_false_trend_rate=ghost_false_trend,
            absorption_reversal_rate=absorption_rev_rate,
            absorption_follow_through_rate=absorption_follow,
            valid_trend_follow_through=valid_follow,
            apathy_noise_fraction=apathy_noise,
            class_balance_score=class_balance_score,
            n_regimes=n_regimes,
            n_samples=n_samples,
            per_regime_metrics=per_regime_metrics,
            distinctiveness_analysis=distinctiveness_analysis,
            overall_quality_score=overall_quality_score,
            assessment_timestamp=datetime.now().isoformat(),
        )

        tprint_success(f"✅ Liquidity cluster assessment complete: quality={overall_quality_score:.3f}")
        return metrics

    # ------------------------------------------------------------------
    # Metric components
    # ------------------------------------------------------------------
    def _assess_effort_result_separation(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
    ) -> Tuple[float, float, float]:
        """Measure separation of Ghost/Absorption vs Valid using key ratios."""
        ghost_ratio = df.get("ghost_ratio")
        absorption_ratio = df.get("absorption_ratio")
        normalized_range = df.get("normalized_range")

        if ghost_ratio is None or absorption_ratio is None or normalized_range is None:
            return 0.0, 0.0, 0.0

        stats: Dict[int, Dict[str, float]] = {}
        for regime in sorted(labels.unique()):
            mask = labels == regime
            if mask.sum() < 5:
                continue
            stats[int(regime)] = {
                "ghost_ratio_mean": float(ghost_ratio[mask].mean()),
                "absorption_ratio_mean": float(absorption_ratio[mask].mean()),
                "range_mean": float(normalized_range[mask].mean()),
            }

        def _contrast(regime_a: int, regime_b: int, key: str) -> float:
            if regime_a not in stats or regime_b not in stats:
                return 0.0
            a = stats[regime_a][key]
            b = stats[regime_b][key]
            denom = abs(a) + abs(b) + 1e-9
            return float((a - b) / denom)

        # semantic mapping: 1=Valid, 2=Absorption, 3=Ghost
        ghost_valid_contrast = _contrast(3, 1, "ghost_ratio_mean")
        absorption_valid_contrast = _contrast(2, 1, "absorption_ratio_mean")

        # Aggregate effort/result separation
        effort_components = [abs(ghost_valid_contrast), abs(absorption_valid_contrast)]
        effort_result_separation_score = float(np.tanh(np.nanmean(effort_components))) if effort_components else 0.0

        return effort_result_separation_score, ghost_valid_contrast, absorption_valid_contrast

    def _assess_ghost_behavior(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        forward_returns: Optional[pd.Series] = None,
    ) -> Tuple[float, float]:
        """Assess how often Ghost regimes behave as traps vs real trends."""
        if forward_returns is None or forward_returns.isna().all():
            return 0.0, 0.0

        mask_ghost = labels == 3
        if mask_ghost.sum() < 10:
            return 0.0, 0.0

        ghost_fwd = forward_returns[mask_ghost].dropna()
        if len(ghost_fwd) == 0:
            return 0.0, 0.0

        # Define small band around zero as "reversal/failed move"
        thresh_small = float(self.config.get("liquidity_ghost_small_return_threshold", 0.001))
        thresh_trend = float(self.config.get("liquidity_ghost_trend_threshold", 0.003))

        reversal_mask = ghost_fwd.abs() <= thresh_small
        false_trend_mask = ghost_fwd.abs() >= thresh_trend

        ghost_reversal_rate = float(reversal_mask.mean()) if len(ghost_fwd) > 0 else 0.0
        ghost_false_trend_rate = float(false_trend_mask.mean()) if len(ghost_fwd) > 0 else 0.0

        return ghost_reversal_rate, ghost_false_trend_rate

    def _assess_absorption_behavior(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        forward_returns: Optional[pd.Series] = None,
    ) -> Tuple[float, float]:
        """Assess how often Absorption regimes precede reversals vs follow-through."""
        if forward_returns is None or forward_returns.isna().all():
            return 0.0, 0.0

        mask_abs = labels == 2
        if mask_abs.sum() < 10:
            return 0.0, 0.0

        abs_fwd = forward_returns[mask_abs].dropna()
        if len(abs_fwd) == 0:
            return 0.0, 0.0

        # Assume absorption should more often lead to reversal than strong continuation
        thresh_rev = float(self.config.get("liquidity_absorption_reversal_threshold", 0.0))
        thresh_follow = float(self.config.get("liquidity_absorption_follow_threshold", 0.003))

        # For now, treat sign opposite to recent move as reversal proxy
        # Use zero-based: negative return for longs => reversal
        reversal_mask = abs_fwd <= thresh_rev
        follow_mask = abs_fwd >= thresh_follow

        absorption_reversal_rate = float(reversal_mask.mean()) if len(abs_fwd) > 0 else 0.0
        absorption_follow_through_rate = float(follow_mask.mean()) if len(abs_fwd) > 0 else 0.0

        return absorption_reversal_rate, absorption_follow_through_rate

    def _assess_trend_and_apathy(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        forward_returns: Optional[pd.Series] = None,
    ) -> Tuple[float, float]:
        """Assess Valid trend follow-through and Apathy noise fraction."""
        if forward_returns is None or forward_returns.isna().all():
            return 0.0, 0.0

        # Valid Trend
        mask_valid = labels == 1
        valid_fwd = forward_returns[mask_valid].dropna()
        if len(valid_fwd) > 0:
            valid_trend_follow = float(valid_fwd.mean())
        else:
            valid_trend_follow = 0.0

        # Apathy noise fraction (returns near zero)
        mask_apathy = labels == 0
        apathy_fwd = forward_returns[mask_apathy].dropna()
        if len(apathy_fwd) > 0:
            noise_band = float(self.config.get("liquidity_apathy_noise_band", 0.001))
            apathy_noise_fraction = float((apathy_fwd.abs() <= noise_band).mean())
        else:
            apathy_noise_fraction = 0.0

        return valid_trend_follow, apathy_noise_fraction

    def _assess_class_balance(self, labels: pd.Series) -> float:
        """Assess how balanced the classes are using entropy of class distribution."""
        counts = labels.value_counts()
        probs = counts / counts.sum()
        if len(probs) == 0:
            return 0.0
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_entropy = np.log(len(probs))
        if max_entropy <= 0:
            return 0.0
        return float(entropy / max_entropy)

    def _calculate_per_regime_metrics(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        forward_returns: Optional[pd.Series] = None,
    ) -> Dict[int, Dict[str, float]]:
        """Calculate per-regime liquidity metrics including std and CoV.

        Now includes directional orderflow, trend persistence, and vol-momentum metrics.
        """
        ghost_ratio = df.get("ghost_ratio")
        absorption_ratio = df.get("absorption_ratio")
        rvol_24 = df.get("rvol_24")
        intraday_close_ratio = df.get("intraday_close_ratio")

        # CATEGORY 1: Directional orderflow metrics
        volume_direction_conviction = df.get("volume_direction_conviction")
        volume_direction_imbalance = df.get("volume_direction_imbalance")

        # CATEGORY 2: Trend persistence metrics (3h and 6h windows)
        trend_confirmation_6h = df.get("trend_confirmation_6h")
        momentum_persistence_3h = df.get("momentum_persistence_3h")
        consecutive_direction_ratio_6h = df.get("consecutive_direction_ratio_6h")

        # CATEGORY 3: Volatility-momentum correlation metrics
        vol_momentum_sync = df.get("vol_momentum_sync")
        range_momentum_divergence = df.get("range_momentum_divergence")
        realized_vol_6h = df.get("realized_vol_6h")

        # CATEGORY 4: Orderbook Pressure Proxy
        volume_concentration_ratio_3h = df.get("volume_concentration_ratio_3h")
        pressure_ratio = df.get("pressure_ratio")
        kyle_lambda_proxy = df.get("kyle_lambda_proxy")

        # CATEGORY 5: Reversal Patterns
        reversal_intensity = df.get("reversal_intensity")
        whipsaw_count = df.get("whipsaw_count")
        reversal_conviction = df.get("reversal_conviction")

        # CATEGORY 6: Multi-Timeframe Volatility Alignment
        vol_clustering = df.get("vol_clustering")
        vol_regime_change = df.get("vol_regime_change")
        session_vol_percentile = df.get("session_vol_percentile")
        intra_bar_vol_estimate = df.get("intra_bar_vol_estimate")

        # CATEGORY 7: Information Efficiency Metrics
        efficiency_ratio = df.get("efficiency_ratio")
        return_autocorr_lag6 = df.get("return_autocorr_lag6")
        volume_price_trend_sync = df.get("volume_price_trend_sync")
        price_impact_ratio = df.get("price_impact_ratio")

        eps = 1e-9

        per_regime: Dict[int, Dict[str, float]] = {}
        for regime in sorted(labels.unique()):
            mask = labels == regime
            n = int(mask.sum())
            regime_metrics: Dict[str, float] = {"n_samples": n}

            if n < 3:
                # Not enough samples for reliable std/CoV
                regime_metrics.update(
                    {
                        "ghost_ratio_mean": 0.0,
                        "ghost_ratio_std": 0.0,
                        "ghost_ratio_cov": 0.0,
                        "absorption_ratio_mean": 0.0,
                        "absorption_ratio_std": 0.0,
                        "absorption_ratio_cov": 0.0,
                        "rvol_24_mean": 0.0,
                        "rvol_24_std": 0.0,
                        "rvol_24_cov": 0.0,
                        "intraday_close_ratio_mean": 0.0,
                        "intraday_close_ratio_std": 0.0,
                        "intraday_close_ratio_cov": 0.0,
                        "volume_direction_conviction_mean": 0.0,
                        "volume_direction_conviction_std": 0.0,
                        "volume_direction_conviction_cov": 0.0,
                        "trend_confirmation_mean": 0.0,
                        "trend_confirmation_std": 0.0,
                        "trend_confirmation_cov": 0.0,
                        "range_momentum_divergence_mean": 0.0,
                        "range_momentum_divergence_std": 0.0,
                        "range_momentum_divergence_cov": 0.0,
                        "forward_return_mean": 0.0,
                        "forward_return_std": 0.0,
                        "forward_return_cov": 0.0,
                    }
                )
                per_regime[int(regime)] = regime_metrics
                continue

            def _mean_std_cov(series: Optional[pd.Series]) -> Tuple[float, float, float]:
                if series is None:
                    return 0.0, 0.0, 0.0
                vals = series[mask].dropna()
                if len(vals) == 0:
                    return 0.0, 0.0, 0.0
                mean_val = float(vals.mean())
                std_val = float(vals.std())
                cov_val = float(std_val / (abs(mean_val) + eps)) if mean_val != 0.0 else 0.0
                return mean_val, std_val, cov_val

            # Original effort metrics
            gr_mean, gr_std, gr_cov = _mean_std_cov(ghost_ratio)
            ar_mean, ar_std, ar_cov = _mean_std_cov(absorption_ratio)
            rv_mean, rv_std, rv_cov = _mean_std_cov(rvol_24)
            ic_mean, ic_std, ic_cov = _mean_std_cov(intraday_close_ratio)

            # CATEGORY 1: Directional orderflow
            vdc_mean, vdc_std, vdc_cov = _mean_std_cov(volume_direction_conviction)
            vdi_mean, vdi_std, vdi_cov = _mean_std_cov(volume_direction_imbalance)

            # CATEGORY 2: Trend persistence (6h window primary)
            tc6_mean, tc6_std, tc6_cov = _mean_std_cov(trend_confirmation_6h)
            mp3_mean, mp3_std, mp3_cov = _mean_std_cov(momentum_persistence_3h)
            cdr6_mean, cdr6_std, cdr6_cov = _mean_std_cov(consecutive_direction_ratio_6h)

            # CATEGORY 3: Volatility-momentum correlation
            vms_mean, vms_std, vms_cov = _mean_std_cov(vol_momentum_sync)
            rmd_mean, rmd_std, rmd_cov = _mean_std_cov(range_momentum_divergence)

            # CATEGORY 4: Orderbook Pressure Proxy
            vcr_mean, vcr_std, vcr_cov = _mean_std_cov(volume_concentration_ratio_3h)
            pr_mean, pr_std, pr_cov = _mean_std_cov(pressure_ratio)
            kl_mean, kl_std, kl_cov = _mean_std_cov(kyle_lambda_proxy)

            # CATEGORY 5: Reversal Patterns
            ri_mean, ri_std, ri_cov = _mean_std_cov(reversal_intensity)
            wc_mean, wc_std, wc_cov = _mean_std_cov(whipsaw_count)
            rc_mean, rc_std, rc_cov = _mean_std_cov(reversal_conviction)

            # CATEGORY 6: Multi-Timeframe Volatility
            vol_c_mean, vol_c_std, vol_c_cov = _mean_std_cov(vol_clustering)
            vol_r_mean, vol_r_std, vol_r_cov = _mean_std_cov(vol_regime_change)
            svp_mean, svp_std, svp_cov = _mean_std_cov(session_vol_percentile)
            ibv_mean, ibv_std, ibv_cov = _mean_std_cov(intra_bar_vol_estimate)

            # CATEGORY 7: Information Efficiency
            er_mean, er_std, er_cov = _mean_std_cov(efficiency_ratio)
            ral_mean, ral_std, ral_cov = _mean_std_cov(return_autocorr_lag6)
            vpts_mean, vpts_std, vpts_cov = _mean_std_cov(volume_price_trend_sync)
            pir_mean, pir_std, pir_cov = _mean_std_cov(price_impact_ratio)

            if forward_returns is not None:
                fr_mean, fr_std, fr_cov = _mean_std_cov(forward_returns)
            else:
                fr_mean, fr_std, fr_cov = 0.0, 0.0, 0.0

            regime_metrics.update(
                {
                    # Original metrics
                    "ghost_ratio_mean": gr_mean,
                    "ghost_ratio_std": gr_std,
                    "ghost_ratio_cov": gr_cov,
                    "absorption_ratio_mean": ar_mean,
                    "absorption_ratio_std": ar_std,
                    "absorption_ratio_cov": ar_cov,
                    "rvol_24_mean": rv_mean,
                    "rvol_24_std": rv_std,
                    "rvol_24_cov": rv_cov,
                    "intraday_close_ratio_mean": ic_mean,
                    "intraday_close_ratio_std": ic_std,
                    "intraday_close_ratio_cov": ic_cov,
                    # CATEGORY 1: Directional orderflow
                    "volume_direction_conviction_mean": vdc_mean,
                    "volume_direction_conviction_std": vdc_std,
                    "volume_direction_conviction_cov": vdc_cov,
                    "volume_direction_imbalance_mean": vdi_mean,
                    "volume_direction_imbalance_std": vdi_std,
                    "volume_direction_imbalance_cov": vdi_cov,
                    # CATEGORY 2: Trend persistence
                    "trend_confirmation_6h_mean": tc6_mean,
                    "trend_confirmation_6h_std": tc6_std,
                    "trend_confirmation_6h_cov": tc6_cov,
                    "momentum_persistence_3h_mean": mp3_mean,
                    "momentum_persistence_3h_std": mp3_std,
                    "momentum_persistence_3h_cov": mp3_cov,
                    # CATEGORY 3: Volatility-momentum correlation
                    "vol_momentum_sync_mean": vms_mean,
                    "vol_momentum_sync_std": vms_std,
                    "vol_momentum_sync_cov": vms_cov,
                    "range_momentum_divergence_mean": rmd_mean,
                    "range_momentum_divergence_std": rmd_std,
                    "range_momentum_divergence_cov": rmd_cov,
                    # CATEGORY 4: Orderbook Pressure Proxy
                    "volume_concentration_ratio_3h_mean": vcr_mean,
                    "volume_concentration_ratio_3h_std": vcr_std,
                    "volume_concentration_ratio_3h_cov": vcr_cov,
                    "pressure_ratio_mean": pr_mean,
                    "pressure_ratio_std": pr_std,
                    "pressure_ratio_cov": pr_cov,
                    "kyle_lambda_proxy_mean": kl_mean,
                    "kyle_lambda_proxy_std": kl_std,
                    "kyle_lambda_proxy_cov": kl_cov,
                    # CATEGORY 5: Reversal Patterns
                    "reversal_intensity_mean": ri_mean,
                    "reversal_intensity_std": ri_std,
                    "reversal_intensity_cov": ri_cov,
                    "whipsaw_count_mean": wc_mean,
                    "whipsaw_count_std": wc_std,
                    "whipsaw_count_cov": wc_cov,
                    # CATEGORY 6: Multi-Timeframe Volatility
                    "vol_clustering_mean": vol_c_mean,
                    "vol_clustering_std": vol_c_std,
                    "vol_clustering_cov": vol_c_cov,
                    "vol_regime_change_mean": vol_r_mean,
                    "vol_regime_change_std": vol_r_std,
                    "vol_regime_change_cov": vol_r_cov,
                    # CATEGORY 7: Information Efficiency
                    "efficiency_ratio_mean": er_mean,
                    "efficiency_ratio_std": er_std,
                    "efficiency_ratio_cov": er_cov,
                    "return_autocorr_lag6_mean": ral_mean,
                    "return_autocorr_lag6_std": ral_std,
                    "return_autocorr_lag6_cov": ral_cov,
                    # Forward returns
                    "forward_return_mean": fr_mean,
                    "forward_return_std": fr_std,
                    "forward_return_cov": fr_cov,
                }
            )

            per_regime[int(regime)] = regime_metrics

        return per_regime

    def _compute_cov_based_scores(
        self,
        per_regime_metrics: Dict[int, Dict[str, float]],
    ) -> Tuple[float, float]:
        """Compute CoV-based separation scores across regimes.

        Now includes all 7 liquidity feature categories:
        1. Directional orderflow
        2. Trend persistence
        3. Volatility-momentum correlation
        4. Orderbook pressure proxy
        5. Reversal patterns
        6. Multi-timeframe volatility alignment
        7. Information efficiency metrics
        """
        effort_covs: List[float] = []
        returns_covs: List[float] = []

        for regime_id, metrics in per_regime_metrics.items():
            eff_components: List[float] = []

            # Original effort metrics (absorption, volume, price range)
            for key in [
                "ghost_ratio_cov",
                "absorption_ratio_cov",
                "rvol_24_cov",
                "intraday_close_ratio_cov",
            ]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 1: Directional orderflow
            # Skip if not available yet; will be populated on next run
            for key in ["volume_direction_conviction_cov", "volume_direction_imbalance_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 2: Trend persistence (use 6h window as primary)
            for key in ["trend_confirmation_6h_cov", "momentum_persistence_3h_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 3: Volatility-momentum correlation
            for key in ["vol_momentum_sync_cov", "range_momentum_divergence_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 4: Orderbook pressure proxy
            for key in ["pressure_ratio_cov", "kyle_lambda_proxy_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 5: Reversal patterns (key indicator of Ghost vs Absorption)
            for key in ["whipsaw_count_cov", "reversal_intensity_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 6: Multi-timeframe volatility alignment
            for key in ["vol_clustering_cov", "vol_regime_change_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            # CATEGORY 7: Information efficiency metrics
            for key in ["efficiency_ratio_cov", "return_autocorr_lag6_cov"]:
                val = metrics.get(key)
                if isinstance(val, (int, float)):
                    eff_components.append(float(val))

            if eff_components:
                effort_covs.append(float(np.mean(eff_components)))

            fr_cov = metrics.get("forward_return_cov")
            if isinstance(fr_cov, (int, float)):
                returns_covs.append(float(fr_cov))

        def _spread(vals: List[float]) -> float:
            if len(vals) < 2:
                return 0.0
            arr = np.asarray(vals, dtype=float)
            mean_val = float(np.mean(arr))
            std_val = float(np.std(arr))
            if mean_val <= 0.0:
                return 0.0
            return float(np.tanh(std_val / (mean_val + 1e-9)))

        return _spread(effort_covs), _spread(returns_covs)

    def _calculate_overall_quality_score(
        self,
        effort_sep: float,
        ghost_valid_contrast: float,
        absorption_valid_contrast: float,
        ghost_rev_rate: float,
        ghost_false_trend: float,
        absorption_rev_rate: float,
        absorption_follow: float,
        valid_follow: float,
        apathy_noise: float,
        class_balance: float,
        cov_effort_sep: float,
        cov_returns_sep: float,
    ) -> float:
        """Aggregate component metrics into a single quality score."""
        # Normalize components roughly to [0, 1]
        eff = float(np.tanh(abs(effort_sep)))
        ghost_good = float(np.clip(ghost_rev_rate - ghost_false_trend, -1.0, 1.0) * 0.5 + 0.5)
        absorption_good = float(np.clip(absorption_rev_rate - absorption_follow, -1.0, 1.0) * 0.5 + 0.5)
        valid_good = float(np.tanh(max(valid_follow, 0.0) * 50.0))  # scale small returns
        apathy_good = float(apathy_noise)
        balance_good = float(class_balance)
        cov_effort_good = float(cov_effort_sep)
        cov_returns_good = float(cov_returns_sep)

        # Weights (sum to 1.0)
        w_effort = 0.20
        w_ghost = 0.15
        w_absorption = 0.10
        w_valid = 0.10
        w_apathy = 0.10
        w_balance = 0.15
        w_cov_effort = 0.10
        w_cov_returns = 0.10

        overall = (
            w_effort * eff
            + w_ghost * ghost_good
            + w_absorption * absorption_good
            + w_valid * valid_good
            + w_apathy * apathy_good
            + w_balance * balance_good
            + w_cov_effort * cov_effort_good
            + w_cov_returns * cov_returns_good
        )

        return float(overall)

    # ------------------------------------------------------------------
    # Reporting helpers
    # ------------------------------------------------------------------
    def save_markdown_report(
        self,
        metrics: LiquidityClusterQualityMetrics,
        symbol: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Save a human-readable markdown report of liquidity cluster quality."""
        if output_dir is None:
            output_dir = "outcomes"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"liquidity_cluster_quality_{symbol}_{timestamp}.md"
        filepath = output_path / filename

        with open(filepath, "w") as f:
            f.write(f"# Liquidity Cluster Quality Report\n\n")
            f.write(f"**Symbol:** {symbol}  \\n")
            f.write(f"**Assessment time:** {metrics.assessment_timestamp}\n\n")

            f.write(f"## Overall Quality\n\n")
            f.write(f"- Overall quality score: **{metrics.overall_quality_score:.4f}**\n\n")

            f.write("## CoV-based Separation\n\n")
            f.write(f"- Effort/Result CoV separation score: {metrics.effort_result_cov_separation_score:.4f}\n")
            f.write(f"- Returns CoV separation score: {metrics.returns_cov_separation_score:.4f}\n\n")

            f.write("## Effort vs Result Separation\n\n")
            f.write(f"- Effort/Result separation score: {metrics.effort_result_separation_score:.4f}\n")
            f.write(f"- Ghost vs Valid contrast: {metrics.ghost_vs_valid_contrast:.4f}\n")
            f.write(f"- Absorption vs Valid contrast: {metrics.absorption_vs_valid_contrast:.4f}\n\n")

            f.write("## Trap / Ghost Behavior\n\n")
            f.write(f"- Ghost reversal rate: {metrics.ghost_reversal_rate:.4f}\n")
            f.write(f"- Ghost false-trend rate: {metrics.ghost_false_trend_rate:.4f}\n\n")

            f.write("## Absorption Behavior\n\n")
            f.write(f"- Absorption reversal rate: {metrics.absorption_reversal_rate:.4f}\n")
            f.write(f"- Absorption follow-through rate: {metrics.absorption_follow_through_rate:.4f}\n\n")

            f.write("## Trend Confirmation & Apathy\n\n")
            f.write(f"- Valid trend follow-through (mean fwd return): {metrics.valid_trend_follow_through:.6f}\n")
            f.write(f"- Apathy noise fraction: {metrics.apathy_noise_fraction:.4f}\n\n")

            f.write("## Class Balance\n\n")
            f.write(f"- Class balance score: {metrics.class_balance_score:.4f}\n")
            f.write(f"- Number of regimes: {metrics.n_regimes}\n")
            f.write(f"- Number of samples: {metrics.n_samples}\n\n")

            f.write("## Per-Regime Metrics\n\n")
            for regime_id, regime_data in sorted(metrics.per_regime_metrics.items()):
                f.write(f"### Regime {regime_id}\n\n")
                for key, value in regime_data.items():
                    f.write(f"- {key}: {value:.6f}\n")
                f.write("\n")

        tprint_info(f"📄 Liquidity cluster quality markdown report saved to: {filepath}")
        return str(filepath)

    def _compute_feature_distinctiveness(
        self,
        liquidity_df: pd.DataFrame,
        regime_labels: pd.Series,
        per_regime_metrics: Dict[int, Dict[str, float]],
        top_n: int = 10,
    ) -> Dict[str, Any]:
        """Compute feature distinctiveness scores using winsorized CV ratios.

        For each feature, compute:
        - Between-regime CoV: variance of feature means across regimes
        - Within-regime CoV: average of feature CoVs within each regime
        - Distinctiveness = Between / Within (higher = better separation)

        Returns dict with top N features for overall distinctiveness and per-regime-pair.
        """
        common_idx = liquidity_df.index.intersection(regime_labels.index)
        df = liquidity_df.loc[common_idx].copy()
        labels = regime_labels.loc[common_idx].astype(int)

        # Get all numeric features (60 liquidity regime features)
        liquidity_features = [
            col for col in df.columns
            if any(x in col for x in [
                'volume_direction', 'consecutive_direction', 'return_autocorr',
                'momentum_persistence', 'trend_confirmation', 'realized_vol',
                'vol_ratio', 'vol_momentum', 'range_momentum', 'momentum_vol',
                'reversal', 'whipsaw', 'volume_concentration', 'pressure_ratio',
                'kyle_lambda', 'intra_bar_vol', 'wick_vol', 'session_vol',
                'vol_clustering', 'vol_regime', 'efficiency_ratio', 'volume_price',
                'price_impact', 'momentum_volume'
            ])
        ]

        eps = 1e-9
        regime_ids = sorted(labels.unique())
        n_regimes = len(regime_ids)

        distinctiveness_scores = {}

        for feature in liquidity_features:
            if feature not in df.columns:
                continue

            feature_data = df[feature].dropna()
            if len(feature_data) < 3:
                continue

            # Between-regime variance: how much does mean differ across regimes?
            regime_means = []
            within_covs = []

            for regime_id in regime_ids:
                mask = labels == regime_id
                regime_vals = feature_data[mask]

                if len(regime_vals) >= 2:
                    regime_means.append(float(regime_vals.mean()))
                    # Compute CoV within this regime
                    mean_val = float(regime_vals.mean())
                    std_val = float(regime_vals.std())
                    cov_val = float(std_val / (abs(mean_val) + eps)) if mean_val != 0.0 else 0.0
                    within_covs.append(cov_val)

            if len(regime_means) < 2:
                continue

            # Winsorize between-regime means (cap extreme outliers at 1st/99th percentile)
            regime_means_arr = np.array(regime_means)
            q01 = np.percentile(regime_means_arr, 1)
            q99 = np.percentile(regime_means_arr, 99)
            regime_means_winsorized = np.clip(regime_means_arr, q01, q99)

            # Between-regime CoV: std of regime means / mean of regime means
            between_mean = float(np.mean(regime_means_winsorized))
            between_std = float(np.std(regime_means_winsorized))
            between_cov = float(between_std / (abs(between_mean) + eps)) if between_mean != 0.0 else 0.0

            # Within-regime CoV: average of per-regime CoVs
            within_cov = float(np.mean(within_covs)) if within_covs else 0.0

            # Distinctiveness = between / within (higher = better)
            distinctiveness = float(between_cov / (within_cov + eps))

            distinctiveness_scores[feature] = {
                'between_cov': between_cov,
                'within_cov': within_cov,
                'distinctiveness': distinctiveness,
                'regime_means': regime_means,
            }

        # Sort by distinctiveness
        sorted_features = sorted(
            distinctiveness_scores.items(),
            key=lambda x: x[1]['distinctiveness'],
            reverse=True
        )

        # Top overall features
        top_overall = sorted_features[:top_n]

        # Compute top features for each regime pair
        regime_pair_features = {}

        for i, regime_a in enumerate(regime_ids):
            for regime_b in regime_ids[i+1:]:
                pair_key = f"Regime{regime_a}_vs_Regime{regime_b}"

                # For this pair, compute which features best separate them
                pair_distinctiveness = {}

                for feature in liquidity_features:
                    if feature not in df.columns:
                        continue

                    feature_data = df[feature].dropna()

                    mask_a = labels == regime_a
                    mask_b = labels == regime_b

                    vals_a = feature_data[mask_a]
                    vals_b = feature_data[mask_b]

                    if len(vals_a) < 2 or len(vals_b) < 2:
                        continue

                    mean_a = float(vals_a.mean())
                    mean_b = float(vals_b.mean())

                    # Mean difference
                    mean_diff = abs(mean_a - mean_b)

                    # Pooled std
                    std_a = float(vals_a.std())
                    std_b = float(vals_b.std())
                    pooled_std = float(np.sqrt((std_a**2 + std_b**2) / 2.0))

                    # Cohen's d-like separation
                    separation = mean_diff / (pooled_std + eps)

                    pair_distinctiveness[feature] = separation

                # Sort by separation
                sorted_pair = sorted(
                    pair_distinctiveness.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:top_n]

                regime_pair_features[pair_key] = sorted_pair

        return {
            'top_overall_features': top_overall,
            'regime_pair_features': regime_pair_features,
            'all_distinctiveness_scores': distinctiveness_scores,
        }

    def _format_feature_distinctiveness_report(
        self,
        distinctiveness_analysis: Dict[str, Any],
        regime_names: Dict[int, str] = None,
    ) -> str:
        """Format feature distinctiveness analysis as readable text."""
        if regime_names is None:
            regime_names = {
                0: "Apathy",
                1: "Valid Trend",
                2: "Absorption",
                3: "Ghost",
            }

        lines = []
        lines.append("\n" + "=" * 100)
        lines.append("FEATURE DISTINCTIVENESS ANALYSIS (Winsorized CoV Ratios)")
        lines.append("=" * 100)

        lines.append("\n## Top Overall Features for Regime Distinction (Between/Within CoV)\n")
        lines.append(f"{'Rank':<6} {'Feature':<40} {'Between-CoV':<15} {'Within-CoV':<15} {'Distinctiveness':<15}")
        lines.append("-" * 95)

        for rank, (feature, scores) in enumerate(distinctiveness_analysis['top_overall_features'], 1):
            lines.append(
                f"{rank:<6} {feature:<40} {scores['between_cov']:<15.4f} "
                f"{scores['within_cov']:<15.4f} {scores['distinctiveness']:<15.4f}"
            )

        # Per-regime pair analysis
        lines.append("\n\n## Best Features for Each Regime Pair (Separation Score)\n")

        for pair_key, features in distinctiveness_analysis['regime_pair_features'].items():
            # Parse pair key: "Regime0_vs_Regime1"
            parts = pair_key.replace("Regime", "").split("_vs_")
            regime_a = int(parts[0])
            regime_b = int(parts[1])
            regime_a_name = regime_names.get(regime_a, f"Regime {regime_a}")
            regime_b_name = regime_names.get(regime_b, f"Regime {regime_b}")

            lines.append(f"\n### {regime_a_name} vs {regime_b_name}\n")
            lines.append(f"{'Rank':<6} {'Feature':<40} {'Separation Score':<15}")
            lines.append("-" * 65)

            for rank, (feature, separation) in enumerate(features, 1):
                lines.append(f"{rank:<6} {feature:<40} {separation:<15.4f}")

        return "\n".join(lines)

    def save_csv_report(
        self,
        metrics: LiquidityClusterQualityMetrics,
        symbol: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Save a CSV summary of liquidity cluster quality metrics."""
        if output_dir is None:
            output_dir = "outcomes"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"liquidity_cluster_quality_{symbol}_{timestamp}.csv"
        filepath = output_path / filename

        row: Dict[str, Any] = {
            "symbol": symbol,
            "assessment_timestamp": metrics.assessment_timestamp,
            "overall_quality_score": metrics.overall_quality_score,
            "effort_result_separation_score": metrics.effort_result_separation_score,
            "ghost_vs_valid_contrast": metrics.ghost_vs_valid_contrast,
            "absorption_vs_valid_contrast": metrics.absorption_vs_valid_contrast,
            "effort_result_cov_separation_score": metrics.effort_result_cov_separation_score,
            "returns_cov_separation_score": metrics.returns_cov_separation_score,
            "ghost_reversal_rate": metrics.ghost_reversal_rate,
            "ghost_false_trend_rate": metrics.ghost_false_trend_rate,
            "absorption_reversal_rate": metrics.absorption_reversal_rate,
            "absorption_follow_through_rate": metrics.absorption_follow_through_rate,
            "valid_trend_follow_through": metrics.valid_trend_follow_through,
            "apathy_noise_fraction": metrics.apathy_noise_fraction,
            "class_balance_score": metrics.class_balance_score,
            "n_regimes": metrics.n_regimes,
            "n_samples": metrics.n_samples,
        }

        # Flatten per-regime metrics
        for regime_id, regime_data in sorted(metrics.per_regime_metrics.items()):
            prefix = f"regime_{regime_id}_"
            for key, value in regime_data.items():
                row[prefix + key] = value

        df = pd.DataFrame([row])
        df.to_csv(filepath, index=False)

        tprint_info(f"📊 Liquidity cluster quality CSV report saved to: {filepath}")
        return str(filepath)

    def save_feature_distinctiveness_report(
        self,
        metrics: LiquidityClusterQualityMetrics,
        symbol: str,
        output_dir: Optional[str] = None,
    ) -> str:
        """Save feature distinctiveness analysis report as markdown.

        Shows which features best distinguish between regimes using winsorized CV ratios.
        """
        if output_dir is None:
            output_dir = "outcomes"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"liquidity_feature_distinctiveness_{symbol}_{timestamp}.md"
        filepath = output_path / filename

        # Generate the distinctiveness report
        distinctiveness_report = self._format_feature_distinctiveness_report(
            metrics.distinctiveness_analysis,
            regime_names={
                0: "Apathy",
                1: "Valid Trend",
                2: "Absorption",
                3: "Ghost",
            }
        )

        with open(filepath, "w") as f:
            f.write(f"# Feature Distinctiveness Report\n\n")
            f.write(f"**Symbol:** {symbol}\n")
            f.write(f"**Assessment time:** {metrics.assessment_timestamp}\n")
            f.write(f"**Number of regimes:** {metrics.n_regimes}\n")
            f.write(f"**Number of samples:** {metrics.n_samples}\n\n")
            f.write(distinctiveness_report)

        tprint_info(f"📊 Feature distinctiveness report saved to: {filepath}")
        return str(filepath)
