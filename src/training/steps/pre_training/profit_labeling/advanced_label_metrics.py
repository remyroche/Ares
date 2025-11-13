"""
Advanced Label Quality Metrics and Export Module

This module provides comprehensive metrics for evaluating label quality,
including:
- AUC (ROC) for directional separability
- Precision, Recall, F1 for threshold sensitivity
- Rolling IC for time stability
- Information Ratio for risk-adjusted performance
- Cross-sectional correlation stability
- Parameter sensitivity testing
- Risk-adjusted labels and uncertainty scores
- Label provenance metadata
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from pathlib import Path
import json
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve


@dataclass
class LabelProvenance:
    """Metadata for label provenance and lineage tracking."""
    creation_timestamp: str
    volatility_threshold: float
    lookahead_periods: int
    volatility_multiplier_range: Tuple[float, float]
    instrument: str
    timeframe: str
    price_series_version: str
    local_extrema_weight: float
    triple_barrier_method: str
    n_samples: int
    n_opportunities: int
    config_hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class AdvancedMetrics:
    """Container for all advanced label quality metrics."""
    # Core metrics
    ic: float
    ic_pvalue: float
    ic_confidence_interval: Tuple[float, float]

    # Classification metrics
    auc_roc: float
    precision: float
    recall: float
    f1_score: float

    # Time stability
    rolling_ic_mean: float
    rolling_ic_std: float
    rolling_ic_stability: float

    # Risk-adjusted performance
    information_ratio: float
    sharpe_ratio: float

    # Trading simulation
    cumulative_pnl: float
    max_drawdown: float
    hit_rate: float
    mean_profit_per_trade: float
    mean_loss_per_trade: float
    profit_factor: float

    # Cross-sectional stability
    rank_correlation_stability: float

    # Parameter sensitivity
    parameter_sensitivity_score: float
    volatility_threshold_sensitivity: float
    lookahead_sensitivity: float

    # Uncertainty metrics
    mean_uncertainty: float
    max_uncertainty: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class AdvancedLabelMetricsCalculator:
    """
    Calculator for advanced label quality metrics.

    This class extends basic IC-based metrics with comprehensive evaluations:
    - Statistical significance testing
    - Classification performance metrics
    - Temporal stability analysis
    - Risk-adjusted performance
    - Parameter sensitivity
    - Uncertainty quantification
    """

    def __init__(self, n_bootstrap: int = 100, rolling_window: int = 50):
        """
        Initialize the calculator.

        Args:
            n_bootstrap: Number of bootstrap samples for confidence intervals
            rolling_window: Window size for rolling metrics
        """
        self.n_bootstrap = n_bootstrap
        self.rolling_window = rolling_window

    def calculate_all_metrics(
        self,
        labels: pd.Series,
        prices: pd.Series,
        lookahead_periods: int,
        volatility: Optional[pd.Series] = None,
    ) -> AdvancedMetrics:
        """
        Calculate all advanced metrics.

        Args:
            labels: Trading labels (1=long, -1=short, 0=no position)
            prices: Price series
            lookahead_periods: Forward-looking periods for return calculation
            volatility: Optional volatility series for risk adjustment

        Returns:
            AdvancedMetrics object with all metrics
        """
        # Calculate forward returns
        forward_returns = prices.pct_change(lookahead_periods).shift(-lookahead_periods)

        # Align labels and returns
        mask = labels.notna() & forward_returns.notna()
        labels_clean = labels[mask]
        returns_clean = forward_returns[mask]

        # Only use non-zero labels for trading metrics
        trading_mask = labels_clean != 0
        labels_trading = labels_clean[trading_mask]
        returns_trading = returns_clean[trading_mask]

        if len(labels_trading) < 10:
            return self._create_fallback_metrics()

        # Calculate each metric category
        ic_metrics = self._calculate_ic_metrics(labels_clean, returns_clean)
        classification_metrics = self._calculate_classification_metrics(labels_trading, returns_trading)
        stability_metrics = self._calculate_stability_metrics(labels_clean, returns_clean)
        risk_metrics = self._calculate_risk_adjusted_metrics(labels_trading, returns_trading, volatility)
        trading_metrics = self._calculate_trading_simulation_metrics(labels_trading, returns_trading)
        sensitivity_metrics = self._calculate_parameter_sensitivity(labels, prices, lookahead_periods)
        uncertainty_metrics = self._calculate_uncertainty_metrics(labels_trading, returns_trading)

        return AdvancedMetrics(
            **ic_metrics,
            **classification_metrics,
            **stability_metrics,
            **risk_metrics,
            **trading_metrics,
            **sensitivity_metrics,
            **uncertainty_metrics
        )

    def _calculate_ic_metrics(self, labels: pd.Series, returns: pd.Series) -> Dict[str, float]:
        """Calculate Information Coefficient with statistical significance."""
        if len(labels) < 10:
            return {
                'ic': 0.0,
                'ic_pvalue': 1.0,
                'ic_confidence_interval': (0.0, 0.0)
            }

        # Calculate Spearman IC
        ic, pvalue = stats.spearmanr(labels, returns)
        ic = float(ic) if not np.isnan(ic) else 0.0
        pvalue = float(pvalue) if not np.isnan(pvalue) else 1.0

        # Bootstrap confidence interval
        ci_lower, ci_upper = self._bootstrap_ic_confidence_interval(labels, returns)

        return {
            'ic': ic,
            'ic_pvalue': pvalue,
            'ic_confidence_interval': (ci_lower, ci_upper)
        }

    def _bootstrap_ic_confidence_interval(
        self,
        labels: pd.Series,
        returns: pd.Series,
        alpha: float = 0.05
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval for IC."""
        ics = []
        n = len(labels)

        for _ in range(self.n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n, size=n, replace=True)
            labels_boot = labels.iloc[indices]
            returns_boot = returns.iloc[indices]

            try:
                ic_boot, _ = stats.spearmanr(labels_boot, returns_boot)
                if not np.isnan(ic_boot):
                    ics.append(ic_boot)
            except:
                continue

        if len(ics) < 10:
            return (0.0, 0.0)

        lower = np.percentile(ics, alpha / 2 * 100)
        upper = np.percentile(ics, (1 - alpha / 2) * 100)

        return (float(lower), float(upper))

    def _calculate_classification_metrics(
        self,
        labels: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate classification metrics (AUC, Precision, Recall, F1)."""
        if len(labels) < 10:
            return {
                'auc_roc': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

        # Convert to binary classification (hit target or not)
        y_true = (labels * returns > 0).astype(int)  # Correct direction prediction
        y_score = np.abs(labels)  # Use label magnitude as confidence score

        try:
            # Calculate AUC
            if len(np.unique(y_true)) > 1:
                auc = roc_auc_score(y_true, y_score)
            else:
                auc = 0.5

            # Calculate Precision, Recall, F1
            # Use median score as threshold
            threshold = y_score.median()
            y_pred = (y_score >= threshold).astype(int)

            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0
            )

            return {
                'auc_roc': float(auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1)
            }
        except Exception:
            return {
                'auc_roc': 0.5,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0
            }

    def _calculate_stability_metrics(
        self,
        labels: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate rolling IC to measure time stability."""
        if len(labels) < self.rolling_window * 2:
            return {
                'rolling_ic_mean': 0.0,
                'rolling_ic_std': 0.0,
                'rolling_ic_stability': 0.0,
                'rank_correlation_stability': 0.0
            }

        # Calculate rolling IC
        rolling_ics = []
        for i in range(len(labels) - self.rolling_window + 1):
            window_labels = labels.iloc[i:i + self.rolling_window]
            window_returns = returns.iloc[i:i + self.rolling_window]

            try:
                ic, _ = stats.spearmanr(window_labels, window_returns)
                if not np.isnan(ic):
                    rolling_ics.append(ic)
            except:
                continue

        if len(rolling_ics) < 5:
            return {
                'rolling_ic_mean': 0.0,
                'rolling_ic_std': 0.0,
                'rolling_ic_stability': 0.0,
                'rank_correlation_stability': 0.0
            }

        rolling_ic_series = pd.Series(rolling_ics)

        # Stability = 1 - coefficient of variation
        mean_ic = rolling_ic_series.mean()
        std_ic = rolling_ic_series.std()
        stability = 1.0 - (std_ic / abs(mean_ic)) if abs(mean_ic) > 0 else 0.0
        stability = max(0.0, min(1.0, stability))

        # Rank correlation stability: correlation between consecutive window ranks
        rank_corr_stability = self._calculate_rank_correlation_stability(labels, returns)

        return {
            'rolling_ic_mean': float(mean_ic),
            'rolling_ic_std': float(std_ic),
            'rolling_ic_stability': float(stability),
            'rank_correlation_stability': float(rank_corr_stability)
        }

    def _calculate_rank_correlation_stability(
        self,
        labels: pd.Series,
        returns: pd.Series
    ) -> float:
        """Calculate stability of signal rankings over time."""
        if len(labels) < self.rolling_window * 2:
            return 0.0

        # Split into two halves
        mid = len(labels) // 2
        labels_first = labels.iloc[:mid]
        labels_second = labels.iloc[mid:]
        returns_first = returns.iloc[:mid]
        returns_second = returns.iloc[mid:]

        # Calculate ranks in each half
        try:
            ranks_first = labels_first.rank()
            ranks_second = labels_second.rank()

            # Calculate correlation between rank positions
            # (This measures if high-ranked signals stay high-ranked)
            corr, _ = stats.spearmanr(ranks_first, ranks_second)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0

    def _calculate_risk_adjusted_metrics(
        self,
        labels: pd.Series,
        returns: pd.Series,
        volatility: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Calculate risk-adjusted performance metrics."""
        if len(labels) < 10:
            return {
                'information_ratio': 0.0,
                'sharpe_ratio': 0.0
            }

        # Calculate strategy returns (label * return)
        strategy_returns = labels * returns

        # Information Ratio: mean return / std of returns
        mean_return = strategy_returns.mean()
        std_return = strategy_returns.std()

        if std_return > 0:
            ir = mean_return / std_return * np.sqrt(252)  # Annualized
            sharpe = ir  # For simplicity, assuming zero risk-free rate
        else:
            ir = 0.0
            sharpe = 0.0

        return {
            'information_ratio': float(ir) if not np.isnan(ir) else 0.0,
            'sharpe_ratio': float(sharpe) if not np.isnan(sharpe) else 0.0
        }

    def _calculate_trading_simulation_metrics(
        self,
        labels: pd.Series,
        returns: pd.Series,
        transaction_cost: float = 0.001  # 0.1% per trade
    ) -> Dict[str, float]:
        """Simulate trading based on labels and calculate P&L metrics."""
        if len(labels) < 10:
            return {
                'cumulative_pnl': 0.0,
                'max_drawdown': 0.0,
                'hit_rate': 0.0,
                'mean_profit_per_trade': 0.0,
                'mean_loss_per_trade': 0.0,
                'profit_factor': 0.0
            }

        # Calculate P&L for each trade (label * return - transaction cost)
        gross_pnl = labels * returns
        net_pnl = gross_pnl - transaction_cost

        # Cumulative P&L
        cumulative_pnl = net_pnl.sum()

        # Max drawdown
        cum_returns = (1 + net_pnl).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())

        # Hit rate
        hit_rate = (gross_pnl > 0).mean()

        # Mean profit and loss
        profits = net_pnl[net_pnl > 0]
        losses = net_pnl[net_pnl < 0]

        mean_profit = profits.mean() if len(profits) > 0 else 0.0
        mean_loss = abs(losses.mean()) if len(losses) > 0 else 0.0

        # Profit factor
        total_profit = profits.sum() if len(profits) > 0 else 0.0
        total_loss = abs(losses.sum()) if len(losses) > 0 else 1e-10
        profit_factor = total_profit / total_loss if total_loss > 0 else 0.0

        return {
            'cumulative_pnl': float(cumulative_pnl),
            'max_drawdown': float(max_drawdown),
            'hit_rate': float(hit_rate),
            'mean_profit_per_trade': float(mean_profit),
            'mean_loss_per_trade': float(mean_loss),
            'profit_factor': float(profit_factor)
        }

    def _calculate_parameter_sensitivity(
        self,
        labels: pd.Series,
        prices: pd.Series,
        lookahead_periods: int
    ) -> Dict[str, float]:
        """Test sensitivity to parameter perturbations."""
        # Simplified sensitivity test: compare IC with perturbed lookahead
        forward_returns = prices.pct_change(lookahead_periods).shift(-lookahead_periods)

        mask = labels.notna() & forward_returns.notna()
        labels_clean = labels[mask]
        returns_clean = forward_returns[mask]

        if len(labels_clean) < 10:
            return {
                'parameter_sensitivity_score': 0.0,
                'volatility_threshold_sensitivity': 0.0,
                'lookahead_sensitivity': 0.0
            }

        # Base IC
        try:
            base_ic, _ = stats.spearmanr(labels_clean, returns_clean)
            base_ic = float(base_ic) if not np.isnan(base_ic) else 0.0
        except:
            base_ic = 0.0

        # Test lookahead sensitivity: +/- 1 period
        ics = []
        for perturb in [-1, 1]:
            new_lookahead = max(1, lookahead_periods + perturb)
            perturbed_returns = prices.pct_change(new_lookahead).shift(-new_lookahead)

            mask_p = labels.notna() & perturbed_returns.notna()
            labels_p = labels[mask_p]
            returns_p = perturbed_returns[mask_p]

            if len(labels_p) >= 10:
                try:
                    ic_p, _ = stats.spearmanr(labels_p, returns_p)
                    if not np.isnan(ic_p):
                        ics.append(ic_p)
                except:
                    pass

        # Sensitivity = std of ICs (lower is more robust)
        if len(ics) > 0:
            lookahead_sensitivity = np.std([base_ic] + ics)
            # Normalize: 1 - sensitivity (so higher = more robust)
            lookahead_sensitivity = max(0.0, 1.0 - lookahead_sensitivity)
        else:
            lookahead_sensitivity = 0.0

        # Overall sensitivity score (placeholder for now)
        sensitivity_score = lookahead_sensitivity

        return {
            'parameter_sensitivity_score': float(sensitivity_score),
            'volatility_threshold_sensitivity': 0.0,  # Placeholder
            'lookahead_sensitivity': float(lookahead_sensitivity)
        }

    def _calculate_uncertainty_metrics(
        self,
        labels: pd.Series,
        returns: pd.Series
    ) -> Dict[str, float]:
        """Calculate per-opportunity uncertainty scores."""
        if len(labels) < 10:
            return {
                'mean_uncertainty': 0.0,
                'max_uncertainty': 0.0
            }

        # Bootstrap variance as uncertainty measure
        n = len(labels)
        n_boot = min(50, self.n_bootstrap)  # Lighter bootstrap

        individual_vars = []
        for i in range(len(labels)):
            # Bootstrap estimate of return variance for this label
            boot_returns = []
            for _ in range(n_boot):
                # Sample with replacement from similar labels
                similar_labels = labels[labels == labels.iloc[i]]
                similar_returns = returns[labels == labels.iloc[i]]

                if len(similar_returns) > 1:
                    boot_sample = np.random.choice(similar_returns, size=min(10, len(similar_returns)), replace=True)
                    boot_returns.append(boot_sample.mean())

            if len(boot_returns) > 1:
                variance = np.var(boot_returns)
                individual_vars.append(variance)

        if len(individual_vars) > 0:
            mean_uncertainty = np.mean(individual_vars)
            max_uncertainty = np.max(individual_vars)
        else:
            mean_uncertainty = 0.0
            max_uncertainty = 0.0

        return {
            'mean_uncertainty': float(mean_uncertainty),
            'max_uncertainty': float(max_uncertainty)
        }

    def _create_fallback_metrics(self) -> AdvancedMetrics:
        """Create fallback metrics when calculations fail."""
        return AdvancedMetrics(
            ic=0.0, ic_pvalue=1.0, ic_confidence_interval=(0.0, 0.0),
            auc_roc=0.5, precision=0.0, recall=0.0, f1_score=0.0,
            rolling_ic_mean=0.0, rolling_ic_std=0.0, rolling_ic_stability=0.0,
            information_ratio=0.0, sharpe_ratio=0.0,
            cumulative_pnl=0.0, max_drawdown=0.0, hit_rate=0.0,
            mean_profit_per_trade=0.0, mean_loss_per_trade=0.0, profit_factor=0.0,
            rank_correlation_stability=0.0,
            parameter_sensitivity_score=0.0, volatility_threshold_sensitivity=0.0,
            lookahead_sensitivity=0.0,
            mean_uncertainty=0.0, max_uncertainty=0.0
        )


class LabelMetricsExporter:
    """
    Export label quality metrics to CSV and Markdown reports.

    Creates timestamped files in the outcomes/ directory with comprehensive
    metrics for label quality assessment.
    """

    def __init__(self, outcomes_dir: Path = Path("outcomes")):
        """
        Initialize the exporter.

        Args:
            outcomes_dir: Directory to save output files
        """
        self.outcomes_dir = outcomes_dir
        self.outcomes_dir.mkdir(parents=True, exist_ok=True)

    def export_metrics(
        self,
        metrics: AdvancedMetrics,
        provenance: LabelProvenance,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Tuple[Path, Path]:
        """
        Export metrics to CSV and Markdown files.

        Args:
            metrics: Advanced metrics to export
            provenance: Label provenance metadata
            additional_info: Additional information to include in report

        Returns:
            Tuple of (csv_path, md_path)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"label_quality_metrics_{timestamp}"

        csv_path = self.outcomes_dir / f"{base_name}.csv"
        md_path = self.outcomes_dir / f"{base_name}.md"

        # Export to CSV
        self._export_csv(csv_path, metrics, provenance, additional_info)

        # Export to Markdown
        self._export_markdown(md_path, metrics, provenance, additional_info)

        return csv_path, md_path

    def _export_csv(
        self,
        path: Path,
        metrics: AdvancedMetrics,
        provenance: LabelProvenance,
        additional_info: Optional[Dict[str, Any]]
    ):
        """Export metrics to CSV format."""
        # Combine all data
        data = {
            **provenance.to_dict(),
            **metrics.to_dict()
        }

        if additional_info:
            data.update(additional_info)

        # Convert to DataFrame and save
        df = pd.DataFrame([data])
        df.to_csv(path, index=False)

    def _export_markdown(
        self,
        path: Path,
        metrics: AdvancedMetrics,
        provenance: LabelProvenance,
        additional_info: Optional[Dict[str, Any]]
    ):
        """Export metrics to Markdown report format."""
        with open(path, 'w') as f:
            f.write("# Label Quality Metrics Report\n\n")
            f.write(f"**Generated:** {provenance.creation_timestamp}\n\n")

            # Provenance section
            f.write("## Label Provenance\n\n")
            f.write(f"- **Instrument:** {provenance.instrument}\n")
            f.write(f"- **Timeframe:** {provenance.timeframe}\n")
            f.write(f"- **Samples:** {provenance.n_samples:,}\n")
            f.write(f"- **Opportunities:** {provenance.n_opportunities:,}\n")
            f.write(f"- **Volatility Threshold:** {provenance.volatility_threshold:.4f}\n")
            f.write(f"- **Lookahead Periods:** {provenance.lookahead_periods}\n")
            f.write(f"- **Volatility Multiplier Range:** {provenance.volatility_multiplier_range[0]:.2f}x - {provenance.volatility_multiplier_range[1]:.2f}x\n")
            f.write(f"- **Local Extrema Weight:** {provenance.local_extrema_weight:.2f}\n")
            f.write(f"- **Triple Barrier Method:** {provenance.triple_barrier_method}\n")
            f.write(f"- **Price Series Version:** {provenance.price_series_version}\n")
            f.write(f"- **Config Hash:** {provenance.config_hash}\n\n")

            # Core metrics
            f.write("## Core Predictive Metrics\n\n")
            f.write("| Metric | Value | Interpretation |\n")
            f.write("|--------|-------|----------------|\n")
            f.write(f"| **Information Coefficient (IC)** | {metrics.ic:.4f} | Spearman correlation between labels and returns |\n")
            f.write(f"| **IC p-value** | {metrics.ic_pvalue:.4f} | Statistical significance (< 0.05 is significant) |\n")
            f.write(f"| **IC 95% CI** | [{metrics.ic_confidence_interval[0]:.4f}, {metrics.ic_confidence_interval[1]:.4f}] | Bootstrap confidence interval |\n\n")

            # Classification metrics
            f.write("## Classification Performance\n\n")
            f.write("| Metric | Value | What it measures |\n")
            f.write("|--------|-------|------------------|\n")
            f.write(f"| **AUC (ROC)** | {metrics.auc_roc:.4f} | Directional separability (prob. hit target vs miss) |\n")
            f.write(f"| **Precision** | {metrics.precision:.4f} | Ratio of correct positive predictions |\n")
            f.write(f"| **Recall** | {metrics.recall:.4f} | Ratio of detected opportunities |\n")
            f.write(f"| **F1 Score** | {metrics.f1_score:.4f} | Harmonic mean of precision and recall |\n\n")

            # Time stability
            f.write("## Temporal Stability\n\n")
            f.write("| Metric | Value | Robustness insight |\n")
            f.write("|--------|-------|--------------------|\n")
            f.write(f"| **Rolling IC (mean)** | {metrics.rolling_ic_mean:.4f} | Average IC over time windows |\n")
            f.write(f"| **Rolling IC (std)** | {metrics.rolling_ic_std:.4f} | Variability of IC over time |\n")
            f.write(f"| **Rolling IC Stability** | {metrics.rolling_ic_stability:.4f} | 1 - (std/mean), higher is more stable |\n")
            f.write(f"| **Rank Correlation Stability** | {metrics.rank_correlation_stability:.4f} | Consistency of signal rankings across assets/time |\n\n")

            # Risk-adjusted metrics
            f.write("## Risk-Adjusted Performance\n\n")
            f.write("| Metric | Value | What it measures |\n")
            f.write("|--------|-------|------------------|\n")
            f.write(f"| **Information Ratio (IR)** | {metrics.information_ratio:.4f} | Mean return / return volatility of labeled strategy |\n")
            f.write(f"| **Sharpe Ratio** | {metrics.sharpe_ratio:.4f} | Risk-adjusted return (annualized) |\n\n")

            # Trading simulation
            f.write("## Trading Simulation (Backtest)\n\n")
            f.write("| Metric | Value | What it measures |\n")
            f.write("|--------|-------|------------------|\n")
            f.write(f"| **Cumulative P&L** | {metrics.cumulative_pnl:.4f} | Total profit/loss from labeled signals |\n")
            f.write(f"| **Max Drawdown** | {metrics.max_drawdown:.4f} | Largest peak-to-trough decline |\n")
            f.write(f"| **Hit Rate** | {metrics.hit_rate:.4f} | Percentage of profitable trades |\n")
            f.write(f"| **Mean Profit/Trade** | {metrics.mean_profit_per_trade:.4f} | Average profit when winning |\n")
            f.write(f"| **Mean Loss/Trade** | {metrics.mean_loss_per_trade:.4f} | Average loss when losing |\n")
            f.write(f"| **Profit Factor** | {metrics.profit_factor:.4f} | Ratio of total profits to total losses |\n\n")

            # Parameter sensitivity
            f.write("## Parameter Sensitivity\n\n")
            f.write("| Metric | Value | Robustness insight |\n")
            f.write("|--------|-------|--------------------|\n")
            f.write(f"| **Overall Sensitivity Score** | {metrics.parameter_sensitivity_score:.4f} | How fragile the labeling logic is |\n")
            f.write(f"| **Lookahead Sensitivity** | {metrics.lookahead_sensitivity:.4f} | Stability under lookahead perturbations |\n")
            f.write(f"| **Volatility Threshold Sensitivity** | {metrics.volatility_threshold_sensitivity:.4f} | Stability under threshold changes |\n\n")

            # Uncertainty metrics
            f.write("## Uncertainty Quantification\n\n")
            f.write("| Metric | Value | What it measures |\n")
            f.write("|--------|-------|------------------|\n")
            f.write(f"| **Mean Uncertainty** | {metrics.mean_uncertainty:.4f} | Average confidence interval width |\n")
            f.write(f"| **Max Uncertainty** | {metrics.max_uncertainty:.4f} | Maximum uncertainty across all labels |\n\n")

            # Additional info
            if additional_info:
                f.write("## Additional Information\n\n")
                f.write("```json\n")
                f.write(json.dumps(additional_info, indent=2))
                f.write("\n```\n\n")

            # Summary interpretation
            f.write("## Summary Interpretation\n\n")
            self._write_interpretation(f, metrics)

    def _write_interpretation(self, f, metrics: AdvancedMetrics):
        """Write human-readable interpretation of metrics."""
        f.write("### Overall Assessment\n\n")

        # IC interpretation
        if abs(metrics.ic) > 0.05 and metrics.ic_pvalue < 0.05:
            f.write("✅ **Predictive Power:** Labels show statistically significant predictive power (|IC| > 0.05, p < 0.05)\n\n")
        else:
            f.write("⚠️ **Predictive Power:** Labels show weak or non-significant predictive power\n\n")

        # AUC interpretation
        if metrics.auc_roc > 0.6:
            f.write("✅ **Classification:** Good directional separability (AUC > 0.6)\n\n")
        elif metrics.auc_roc > 0.55:
            f.write("⚠️ **Classification:** Moderate directional separability (AUC > 0.55)\n\n")
        else:
            f.write("❌ **Classification:** Poor directional separability (AUC ≤ 0.55)\n\n")

        # Stability interpretation
        if metrics.rolling_ic_stability > 0.7:
            f.write("✅ **Stability:** Labels are temporally stable (stability > 0.7)\n\n")
        elif metrics.rolling_ic_stability > 0.5:
            f.write("⚠️ **Stability:** Moderate temporal stability (stability > 0.5)\n\n")
        else:
            f.write("❌ **Stability:** Poor temporal stability (stability ≤ 0.5)\n\n")

        # Trading simulation interpretation
        if metrics.profit_factor > 1.5 and metrics.hit_rate > 0.5:
            f.write("✅ **Trading:** Profitable simulated strategy (PF > 1.5, Hit Rate > 50%)\n\n")
        elif metrics.profit_factor > 1.0:
            f.write("⚠️ **Trading:** Marginally profitable strategy (PF > 1.0)\n\n")
        else:
            f.write("❌ **Trading:** Unprofitable simulated strategy (PF ≤ 1.0)\n\n")

        # Parameter sensitivity interpretation
        if metrics.parameter_sensitivity_score > 0.7:
            f.write("✅ **Robustness:** Labels are robust to parameter changes (sensitivity > 0.7)\n\n")
        elif metrics.parameter_sensitivity_score > 0.5:
            f.write("⚠️ **Robustness:** Moderate parameter sensitivity (sensitivity > 0.5)\n\n")
        else:
            f.write("❌ **Robustness:** High parameter sensitivity (sensitivity ≤ 0.5)\n\n")
