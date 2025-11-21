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

        # NEW: Directional orderflow metrics
        volume_direction_conviction = df.get("volume_direction_conviction")
        volume_direction_imbalance = df.get("volume_direction_imbalance")

        # NEW: Trend persistence metrics
        consecutive_direction_ratio = df.get("consecutive_direction_ratio")
        trend_confirmation = df.get("trend_confirmation")
        momentum_persistence = df.get("momentum_persistence")

        # NEW: Volatility-momentum correlation metrics
        vol_momentum_sync = df.get("vol_momentum_sync")
        range_momentum_divergence = df.get("range_momentum_divergence")
        realized_vol_6h = df.get("realized_vol_6h")

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

            gr_mean, gr_std, gr_cov = _mean_std_cov(ghost_ratio)
            ar_mean, ar_std, ar_cov = _mean_std_cov(absorption_ratio)
            rv_mean, rv_std, rv_cov = _mean_std_cov(rvol_24)
            ic_mean, ic_std, ic_cov = _mean_std_cov(intraday_close_ratio)

            # NEW metrics
            vdc_mean, vdc_std, vdc_cov = _mean_std_cov(volume_direction_conviction)
            tc_mean, tc_std, tc_cov = _mean_std_cov(trend_confirmation)
            rmd_mean, rmd_std, rmd_cov = _mean_std_cov(range_momentum_divergence)

            if forward_returns is not None:
                fr_mean, fr_std, fr_cov = _mean_std_cov(forward_returns)
            else:
                fr_mean, fr_std, fr_cov = 0.0, 0.0, 0.0

            regime_metrics.update(
                {
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
                    "volume_direction_conviction_mean": vdc_mean,
                    "volume_direction_conviction_std": vdc_std,
                    "volume_direction_conviction_cov": vdc_cov,
                    "trend_confirmation_mean": tc_mean,
                    "trend_confirmation_std": tc_std,
                    "trend_confirmation_cov": tc_cov,
                    "range_momentum_divergence_mean": rmd_mean,
                    "range_momentum_divergence_std": rmd_std,
                    "range_momentum_divergence_cov": rmd_cov,
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

        Now includes volume/orderflow metrics alongside effort (absorption, rvol, etc.)
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

            # NEW: Volume and orderflow metrics
            for key in [
                "volume_direction_conviction_cov",
                "trend_confirmation_cov",
                "range_momentum_divergence_cov",
            ]:
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
