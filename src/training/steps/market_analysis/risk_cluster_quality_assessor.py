"""
Risk Cluster Quality Assessor

This module assesses the quality of risk-based regime clusters, focusing on:
- VaR/CVaR stratification (monotonicity checks)
- Volatility clustering coefficient (distribution distinctness)
- Transition stability (safe->danger vs safe->crash paths)
- Risk-specific economic metrics (MDD, MRU, skewness, tail ratio)

Unlike return-based cluster assessment, this evaluates regimes based on their
risk characteristics rather than profitability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from src.utils.tprint import (
        tprint_info,
        tprint_warning,
        tprint_error,
        tprint_success
    )
except ImportError:
    tprint_info = logging.info
    tprint_warning = logging.warning
    tprint_error = logging.error
    tprint_success = logging.info

logger = logging.getLogger(__name__)


@dataclass
class RiskClusterQualityMetrics:
    """Risk-specific cluster quality metrics."""
    # VaR/CVaR Stratification
    var_stratification_score: float = 0.0
    cvar_stratification_score: float = 0.0
    var_monotonicity: bool = False
    cvar_monotonicity: bool = False

    # Volatility Clustering
    volatility_clustering_coeff: float = 0.0
    within_vol_cv: float = 0.0
    between_vol_cv: float = 0.0
    vol_separation_ratio: float = 0.0

    # Transition Stability
    calm_to_crash_direct_prob: float = 0.0
    calm_to_turbulent_to_crash_prob: float = 0.0
    transition_stability_score: float = 0.0

    # Per-regime risk metrics
    regime_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Overall quality score
    overall_quality_score: float = 0.0

    # Metadata
    n_regimes: int = 0
    n_samples: int = 0
    assessment_timestamp: str = ""


class RiskClusterQualityAssessor:
    """Assess quality of risk-based regime clusters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize risk cluster quality assessor."""
        self.config = config or {}
        self.logger = logger

    def assess_risk_clusters(
        self,
        risk_df: pd.DataFrame,
        regime_labels: pd.Series,
        config: Optional[Dict[str, Any]] = None
    ) -> RiskClusterQualityMetrics:
        """Assess quality of risk-based regime clusters.

        Args:
            risk_df: DataFrame with OHLCV and risk features
            regime_labels: Series with regime assignments
            config: Optional configuration dict

        Returns:
            RiskClusterQualityMetrics with assessment results
        """
        cfg = config or self.config

        # Align data
        common_idx = risk_df.index.intersection(regime_labels.index)
        df = risk_df.loc[common_idx].copy()
        labels = regime_labels.loc[common_idx]

        if len(df) == 0 or len(labels) == 0:
            tprint_warning("Empty data for risk cluster assessment")
            return RiskClusterQualityMetrics()

        # Calculate returns for economic metrics
        if 'close' in df.columns:
            returns = np.log(df['close'] / df['close'].shift(1))
        else:
            tprint_warning("No 'close' column for returns calculation")
            returns = pd.Series(0.0, index=df.index)

        # Calculate volatility if not present
        if 'volatility_1h' not in df.columns:
            df['volatility_1h'] = returns.rolling(window=20).std()

        n_regimes = int(labels.nunique())
        n_samples = len(df)

        tprint_info(f"📊 Assessing {n_regimes} risk regimes on {n_samples} samples")

        # 1. VaR/CVaR Stratification
        var_strat, cvar_strat, var_mono, cvar_mono = self._assess_var_cvar_stratification(
            df, labels, returns
        )

        # 2. Volatility Clustering Coefficient
        vol_cluster_coeff, within_vol_cv, between_vol_cv, vol_sep_ratio = self._assess_volatility_clustering(
            df, labels
        )

        # 3. Transition Stability
        calm_crash_direct, calm_crash_via_turb, trans_stability = self._assess_transition_stability(
            labels, n_regimes
        )

        # 4. Per-regime risk metrics
        regime_metrics = self._calculate_per_regime_metrics(df, labels, returns)

        # 5. Overall quality score (weighted combination)
        overall_score = self._calculate_overall_quality_score(
            var_strat, cvar_strat, vol_cluster_coeff, trans_stability
        )

        metrics = RiskClusterQualityMetrics(
            var_stratification_score=var_strat,
            cvar_stratification_score=cvar_strat,
            var_monotonicity=var_mono,
            cvar_monotonicity=cvar_mono,
            volatility_clustering_coeff=vol_cluster_coeff,
            within_vol_cv=within_vol_cv,
            between_vol_cv=between_vol_cv,
            vol_separation_ratio=vol_sep_ratio,
            calm_to_crash_direct_prob=calm_crash_direct,
            calm_to_turbulent_to_crash_prob=calm_crash_via_turb,
            transition_stability_score=trans_stability,
            regime_metrics=regime_metrics,
            overall_quality_score=overall_score,
            n_regimes=n_regimes,
            n_samples=n_samples,
            assessment_timestamp=datetime.now().isoformat()
        )

        tprint_success(f"✅ Risk cluster assessment complete: quality={overall_score:.3f}")

        return metrics

    def _assess_var_cvar_stratification(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        returns: pd.Series
    ) -> Tuple[float, float, bool, bool]:
        """Assess VaR/CVaR stratification across regimes."""
        unique_regimes = sorted(labels.unique())

        var_values = []
        cvar_values = []

        for regime in unique_regimes:
            regime_mask = labels == regime
            regime_returns = returns[regime_mask].dropna()

            if len(regime_returns) < 10:
                var_values.append(0.0)
                cvar_values.append(0.0)
                continue

            # VaR at 5% level
            var = regime_returns.quantile(0.05)
            var_values.append(abs(var))

            # CVaR (expected shortfall)
            cvar = regime_returns[regime_returns <= var].mean()
            cvar_values.append(abs(cvar))

        # Check monotonicity (risk should increase with regime number)
        var_mono = all(var_values[i] <= var_values[i+1] for i in range(len(var_values)-1))
        cvar_mono = all(cvar_values[i] <= cvar_values[i+1] for i in range(len(cvar_values)-1))

        # Stratification score: how well-separated are the risk levels?
        var_strat = np.std(var_values) / (np.mean(var_values) + 1e-9) if var_values else 0.0
        cvar_strat = np.std(cvar_values) / (np.mean(cvar_values) + 1e-9) if cvar_values else 0.0

        return float(var_strat), float(cvar_strat), var_mono, cvar_mono

    def _assess_volatility_clustering(
        self,
        df: pd.DataFrame,
        labels: pd.Series
    ) -> Tuple[float, float, float, float]:
        """Assess volatility clustering coefficient."""
        vol_col = 'volatility_1h' if 'volatility_1h' in df.columns else None

        if vol_col is None:
            return 0.0, 0.0, 0.0, 0.0

        volatility = df[vol_col].dropna()
        aligned_labels = labels.loc[volatility.index]

        unique_regimes = sorted(aligned_labels.unique())

        # Within-regime volatility CV (lower is better - tight clustering)
        within_cvs = []
        regime_mean_vols = []

        for regime in unique_regimes:
            regime_mask = aligned_labels == regime
            regime_vol = volatility[regime_mask]

            if len(regime_vol) > 1:
                # Winsorized CV
                vol_clean = regime_vol.clip(lower=regime_vol.quantile(0.01), upper=regime_vol.quantile(0.99))
                cv = vol_clean.std() / (vol_clean.mean() + 1e-9)
                within_cvs.append(cv)
                regime_mean_vols.append(regime_vol.mean())

        within_vol_cv = float(np.mean(within_cvs)) if within_cvs else 0.0

        # Between-regime volatility CV (higher is better - distinct regimes)
        between_vol_cv = float(np.std(regime_mean_vols) / (np.mean(regime_mean_vols) + 1e-9)) if regime_mean_vols else 0.0

        # Separation ratio
        vol_sep_ratio = between_vol_cv / (within_vol_cv + 1e-9) if within_vol_cv > 0 else 0.0

        # Clustering coefficient (higher = better separation)
        vol_cluster_coeff = float(np.tanh(vol_sep_ratio / 2.0))  # Normalize to [0, 1]

        return vol_cluster_coeff, within_vol_cv, between_vol_cv, vol_sep_ratio

    def _calculate_transition_matrix_and_stability(
        self,
        labels: pd.Series,
    ) -> Dict[str, Any]:
        """Calculate transition matrix and stability metrics (entropy + stickiness)."""
        labels_clean = labels.dropna()
        if len(labels_clean) < 2:
            return {
                "transition_matrix": np.array([[1.0]]),
                "transition_entropy": 0.0,
                "regime_stickiness": 1.0,
                "transition_stability_score": 1.0,
            }

        unique_regimes = np.sort(labels_clean.unique())
        n_regimes = len(unique_regimes)
        if n_regimes <= 1:
            return {
                "transition_matrix": np.array([[1.0]]),
                "transition_entropy": 0.0,
                "regime_stickiness": 1.0,
                "transition_stability_score": 1.0,
            }

        # Create transition count matrix
        transition_matrix = np.zeros((n_regimes, n_regimes))
        label_values = labels_clean.values
        for i in range(len(label_values) - 1):
            from_regime = label_values[i]
            to_regime = label_values[i + 1]
            from_idx = np.where(unique_regimes == from_regime)[0][0]
            to_idx = np.where(unique_regimes == to_regime)[0][0]
            transition_matrix[from_idx, to_idx] += 1.0

        # Convert counts to probabilities
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        transition_matrix = transition_matrix / row_sums

        # Transition entropy per row
        transition_entropies: List[float] = []
        for i in range(n_regimes):
            row_probs = transition_matrix[i, :]
            if np.sum(row_probs) > 0:
                row_probs = row_probs / np.sum(row_probs)
                entropy = -np.sum(row_probs * np.log(row_probs + 1e-10))
                transition_entropies.append(float(entropy))

        avg_transition_entropy = float(np.mean(transition_entropies)) if transition_entropies else 0.0
        max_entropy = float(np.log(n_regimes)) if n_regimes > 0 else 1.0

        # Regime stickiness: average probability of staying in same regime
        diagonal_sum = float(np.trace(transition_matrix))
        regime_stickiness = float(diagonal_sum / n_regimes)

        if max_entropy <= 0:
            entropy_score = 1.0
        else:
            entropy_score = 1.0 - (avg_transition_entropy / max_entropy)

        transition_stability_score = float((entropy_score + regime_stickiness) / 2.0)

        return {
            "transition_matrix": transition_matrix,
            "transition_entropy": avg_transition_entropy,
            "regime_stickiness": regime_stickiness,
            "transition_stability_score": transition_stability_score,
        }

    def _assess_transition_stability(
        self,
        labels: pd.Series,
        n_regimes: int
    ) -> Tuple[float, float, float]:
        """Assess transition stability (calm->crash vs calm->turbulent->crash)."""
        labels_clean = labels.dropna()
        if len(labels_clean) < 2:
            return 0.0, 0.0, 1.0

        # Always compute matrix-based transition stability
        tm_data = self._calculate_transition_matrix_and_stability(labels_clean)
        trans_stability = float(tm_data.get("transition_stability_score", 0.0))

        # Calm/crash path metrics require at least 3 ordered regimes
        n_unique = int(labels_clean.nunique())
        if n_unique < 3:
            return 0.0, 0.0, trans_stability

        calm_regime = labels_clean.min()
        crash_regime = labels_clean.max()

        # Count transitions
        transitions = pd.DataFrame({
            'from': labels_clean.iloc[:-1].values,
            'to': labels_clean.iloc[1:].values
        })

        # Direct calm -> crash
        calm_to_crash_direct = len(
            transitions[
                (transitions['from'] == calm_regime)
                & (transitions['to'] == crash_regime)
            ]
        )

        # Calm -> middle -> crash (any middle state)
        calm_to_middle = transitions[
            (transitions['from'] == calm_regime)
            & (transitions['to'] > calm_regime)
            & (transitions['to'] < crash_regime)
        ]
        middle_to_crash = transitions[
            (transitions['from'] > calm_regime)
            & (transitions['from'] < crash_regime)
            & (transitions['to'] == crash_regime)
        ]
        calm_to_middle_to_crash = min(len(calm_to_middle), len(middle_to_crash))

        total_calm_transitions = len(transitions[transitions['from'] == calm_regime])
        if total_calm_transitions == 0:
            return 0.0, 0.0, trans_stability

        calm_crash_direct_prob = float(calm_to_crash_direct / total_calm_transitions)
        calm_crash_via_turb_prob = float(calm_to_middle_to_crash / total_calm_transitions)

        return calm_crash_direct_prob, calm_crash_via_turb_prob, trans_stability

    def _calculate_per_regime_metrics(
        self,
        df: pd.DataFrame,
        labels: pd.Series,
        returns: pd.Series
    ) -> Dict[int, Dict[str, float]]:
        """Calculate risk metrics for each regime."""
        unique_regimes = sorted(labels.unique())
        regime_metrics = {}

        for regime in unique_regimes:
            regime_mask = labels == regime
            regime_returns = returns[regime_mask].dropna()

            if len(regime_returns) < 3:
                regime_metrics[int(regime)] = {
                    'avg_mdd': 0.0,
                    'avg_mru': 0.0,
                    'skewness': 0.0,
                    'tail_ratio': 0.0,
                    'n_samples': 0
                }
                continue

            # Max Drawdown (risk for longs)
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            avg_mdd = float(drawdown.min())

            # Max Run-Up (risk for shorts)
            running_min = cumulative.expanding().min()
            runup = (cumulative - running_min) / (running_min + 1e-9)
            avg_mru = float(runup.max())

            # Skewness (negative = crash risk, positive = pump risk)
            skewness = float(regime_returns.skew())

            # Tail Ratio (95th %ile / 5th %ile)
            p95 = regime_returns.quantile(0.95)
            p05 = regime_returns.quantile(0.05)
            tail_ratio = float(abs(p95 / (p05 + 1e-9)))

            regime_metrics[int(regime)] = {
                'avg_mdd': avg_mdd,
                'avg_mru': avg_mru,
                'skewness': skewness,
                'tail_ratio': tail_ratio,
                'n_samples': int(regime_mask.sum())
            }

        return regime_metrics

    def _calculate_overall_quality_score(
        self,
        var_strat: float,
        cvar_strat: float,
        vol_cluster_coeff: float,
        trans_stability: float
    ) -> float:
        """Calculate overall quality score (weighted combination)."""
        # Weights (must sum to 1.0)
        w_var = 0.25
        w_cvar = 0.25
        w_vol = 0.30
        w_trans = 0.20

        # Normalize components to [0, 1]
        var_norm = np.tanh(var_strat)
        cvar_norm = np.tanh(cvar_strat)
        vol_norm = vol_cluster_coeff  # Already normalized
        trans_norm = trans_stability  # Already normalized

        overall = w_var * var_norm + w_cvar * cvar_norm + w_vol * vol_norm + w_trans * trans_norm

        return float(overall)

    def save_assessment_report(
        self,
        metrics: RiskClusterQualityMetrics,
        symbol: str,
        output_dir: Optional[str] = None
    ) -> str:
        """Save assessment report to file.

        Returns:
            Path to saved report
        """
        if output_dir is None:
            output_dir = "outcomes"

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"risk_cluster_quality_{symbol}_{timestamp}.txt"
        filepath = output_path / filename

        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"Risk Cluster Quality Assessment Report\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Timestamp: {metrics.assessment_timestamp}\n")
            f.write(f"Number of Regimes: {metrics.n_regimes}\n")
            f.write(f"Total Samples: {metrics.n_samples}\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Overall Quality Score: {metrics.overall_quality_score:.4f}\n\n")

            f.write("VaR/CVaR Stratification:\n")
            f.write(f"  VaR Stratification: {metrics.var_stratification_score:.4f}\n")
            f.write(f"  CVaR Stratification: {metrics.cvar_stratification_score:.4f}\n")
            f.write(f"  VaR Monotonicity: {metrics.var_monotonicity}\n")
            f.write(f"  CVaR Monotonicity: {metrics.cvar_monotonicity}\n\n")

            f.write("Volatility Clustering:\n")
            f.write(f"  Clustering Coefficient: {metrics.volatility_clustering_coeff:.4f}\n")
            f.write(f"  Within-Regime Vol CV: {metrics.within_vol_cv:.4f}\n")
            f.write(f"  Between-Regime Vol CV: {metrics.between_vol_cv:.4f}\n")
            f.write(f"  Separation Ratio: {metrics.vol_separation_ratio:.4f}\n\n")

            f.write("Transition Stability:\n")
            f.write(f"  Calm->Crash Direct Prob: {metrics.calm_to_crash_direct_prob:.4f}\n")
            f.write(f"  Calm->Turbulent->Crash Prob: {metrics.calm_to_turbulent_to_crash_prob:.4f}\n")
            f.write(f"  Transition Stability Score: {metrics.transition_stability_score:.4f}\n\n")

            f.write("Per-Regime Risk Metrics:\n")
            for regime_id, regime_data in sorted(metrics.regime_metrics.items()):
                f.write(f"  Regime {regime_id} (n={regime_data['n_samples']}):\n")
                f.write(f"    Avg MDD: {regime_data['avg_mdd']:.4f}\n")
                f.write(f"    Avg MRU: {regime_data['avg_mru']:.4f}\n")
                f.write(f"    Skewness: {regime_data['skewness']:.4f}\n")
                f.write(f"    Tail Ratio: {regime_data['tail_ratio']:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")

        tprint_info(f"📄 Risk cluster quality report saved to: {filepath}")
        return str(filepath)
