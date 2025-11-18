"""
Phase 5: Subgroup & Spatio-Temporal Diagnostics

This module identifies where and when models perform poorly:
1. Subgroup SNR scanning - Find subgroups with different performance
2. Residual autocorrelation - Detect temporal structure in errors
3. Spatio-temporal heatmaps - Visualize performance across dimensions

Use cases:
- Identify demographic/categorical groups where model fails
- Detect temporal patterns (seasonality, drift)
- Find regions/conditions with poor performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass
import json
import warnings

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, pacf


@dataclass
class SubgroupResults:
    """Results from subgroup analysis."""
    subgroup_metrics: Dict[str, Dict[str, float]]
    significant_differences: List[Tuple[str, str, float]]  # (feature, subgroup, delta_snr)
    worst_subgroups: List[Tuple[str, str, float]]  # (feature, subgroup, snr)


@dataclass
class TemporalResults:
    """Results from temporal analysis."""
    autocorrelation: np.ndarray
    partial_autocorrelation: np.ndarray
    durbin_watson_stat: float
    temporal_drift: Dict[str, float]
    temporal_heatmap_data: pd.DataFrame


class SubgroupDiagnostics:
    """
    Subgroup Performance Diagnostics.

    Scans categorical and binned continuous features to find subgroups
    with significantly different model performance.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        min_samples_per_group: int = 30,
        verbose: bool = True
    ):
        """
        Initialize subgroup diagnostics.

        Parameters
        ----------
        output_dir : str or Path
            Output directory
        min_samples_per_group : int
            Minimum samples required for subgroup analysis
        verbose : bool
            Print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.min_samples_per_group = min_samples_per_group
        self.verbose = verbose

    def scan_subgroups(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        y_pred: np.ndarray,
        categorical_features: Optional[List[str]] = None,
        continuous_features_to_bin: Optional[List[str]] = None,
        n_bins: int = 5
    ) -> SubgroupResults:
        """
        Scan all subgroups for performance differences.

        Parameters
        ----------
        X : pd.DataFrame
            Features with column names
        y : np.ndarray
            True values
        y_pred : np.ndarray
            Predictions
        categorical_features : list of str, optional
            Categorical feature names to analyze
        continuous_features_to_bin : list of str, optional
            Continuous features to bin and analyze
        n_bins : int
            Number of bins for continuous features

        Returns
        -------
        SubgroupResults
            Complete subgroup analysis
        """
        if self.verbose:
            print("\n" + "="*80)
            print("SUBGROUP SNR ANALYSIS")
            print("="*80)

        # Overall baseline
        from .snr_diagnostics import compute_snr_metrics
        baseline_metrics = compute_snr_metrics(y, y_pred)
        baseline_snr = baseline_metrics['snr']

        if self.verbose:
            print(f"  Baseline SNR: {baseline_snr:.4f}\n")

        subgroup_metrics = {}
        significant_differences = []
        worst_subgroups = []

        # Analyze categorical features
        if categorical_features is None:
            categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        for feature in categorical_features:
            if feature not in X.columns:
                continue

            if self.verbose:
                print(f"  Analyzing categorical: {feature}")

            feature_results = self._analyze_categorical_feature(
                X[feature], y, y_pred, feature, baseline_snr
            )

            subgroup_metrics[feature] = feature_results

            # Find significant differences
            for subgroup, metrics in feature_results.items():
                delta_snr = metrics['delta_snr']
                snr = metrics['snr']

                if abs(delta_snr) > 0.2:  # Significant threshold
                    significant_differences.append((feature, subgroup, delta_snr))

                if snr < baseline_snr * 0.5:  # Substantially worse
                    worst_subgroups.append((feature, subgroup, snr))

        # Analyze continuous features (binned)
        if continuous_features_to_bin:
            for feature in continuous_features_to_bin:
                if feature not in X.columns:
                    continue

                if self.verbose:
                    print(f"  Analyzing continuous (binned): {feature}")

                feature_results = self._analyze_continuous_feature(
                    X[feature], y, y_pred, feature, baseline_snr, n_bins
                )

                subgroup_metrics[f'{feature}_binned'] = feature_results

                for subgroup, metrics in feature_results.items():
                    delta_snr = metrics['delta_snr']
                    snr = metrics['snr']

                    if abs(delta_snr) > 0.2:
                        significant_differences.append((f'{feature}_binned', subgroup, delta_snr))

                    if snr < baseline_snr * 0.5:
                        worst_subgroups.append((f'{feature}_binned', subgroup, snr))

        # Sort by magnitude
        significant_differences.sort(key=lambda x: abs(x[2]), reverse=True)
        worst_subgroups.sort(key=lambda x: x[2])

        results = SubgroupResults(
            subgroup_metrics=subgroup_metrics,
            significant_differences=significant_differences,
            worst_subgroups=worst_subgroups
        )

        # Visualize
        self._plot_subgroup_analysis(results, baseline_snr)

        # Report
        self._generate_subgroup_report(results, baseline_snr)

        # Save
        self._save_subgroup_results(results)

        return results

    def _analyze_categorical_feature(
        self,
        feature_values: pd.Series,
        y: np.ndarray,
        y_pred: np.ndarray,
        feature_name: str,
        baseline_snr: float
    ) -> Dict[str, Dict[str, float]]:
        """Analyze SNR for each category."""
        from .snr_diagnostics import compute_snr_metrics

        results = {}
        unique_values = feature_values.dropna().unique()

        for value in unique_values:
            mask = (feature_values == value).values
            n_samples = np.sum(mask)

            if n_samples < self.min_samples_per_group:
                continue

            y_sub = y[mask]
            y_pred_sub = y_pred[mask]

            metrics = compute_snr_metrics(y_sub, y_pred_sub)

            results[str(value)] = {
                'snr': metrics['snr'],
                'r2': metrics['r2'],
                'n_samples': int(n_samples),
                'delta_snr': metrics['snr'] - baseline_snr
            }

            if self.verbose:
                print(f"    {value}: SNR={metrics['snr']:.4f} (Δ={metrics['snr'] - baseline_snr:+.4f}), n={n_samples}")

        return results

    def _analyze_continuous_feature(
        self,
        feature_values: pd.Series,
        y: np.ndarray,
        y_pred: np.ndarray,
        feature_name: str,
        baseline_snr: float,
        n_bins: int
    ) -> Dict[str, Dict[str, float]]:
        """Analyze SNR for binned continuous feature."""
        from .snr_diagnostics import compute_snr_metrics

        # Create bins
        try:
            binned, bin_edges = pd.cut(feature_values, bins=n_bins, retbins=True, duplicates='drop')
        except:
            warnings.warn(f"Failed to bin {feature_name}")
            return {}

        results = {}

        for i, interval in enumerate(binned.cat.categories):
            mask = (binned == interval).values
            n_samples = np.sum(mask)

            if n_samples < self.min_samples_per_group:
                continue

            y_sub = y[mask]
            y_pred_sub = y_pred[mask]

            metrics = compute_snr_metrics(y_sub, y_pred_sub)

            bin_label = f"[{interval.left:.2f}, {interval.right:.2f})"

            results[bin_label] = {
                'snr': metrics['snr'],
                'r2': metrics['r2'],
                'n_samples': int(n_samples),
                'delta_snr': metrics['snr'] - baseline_snr
            }

            if self.verbose:
                print(f"    {bin_label}: SNR={metrics['snr']:.4f} (Δ={metrics['snr'] - baseline_snr:+.4f}), n={n_samples}")

        return results

    def _plot_subgroup_analysis(self, results: SubgroupResults, baseline_snr: float):
        """Visualize subgroup analysis."""
        # Plot top significant differences
        if len(results.significant_differences) > 0:
            top_n = min(15, len(results.significant_differences))
            top_diffs = results.significant_differences[:top_n]

            fig, ax = plt.subplots(figsize=(12, 8))

            labels = [f"{feat}: {subgroup}" for feat, subgroup, _ in top_diffs]
            deltas = [delta for _, _, delta in top_diffs]
            colors = ['red' if d < 0 else 'green' for d in deltas]

            y_pos = np.arange(len(labels))
            ax.barh(y_pos, deltas, color=colors, alpha=0.7, edgecolor='black')

            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels, fontsize=9)
            ax.set_xlabel('Δ SNR vs Baseline', fontsize=12)
            ax.set_title('Subgroups with Largest SNR Differences', fontsize=14, fontweight='bold')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
            ax.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()
            plt.savefig(self.output_dir / 'subgroup_differences.png', dpi=150, bbox_inches='tight')
            plt.close()

    def _generate_subgroup_report(self, results: SubgroupResults, baseline_snr: float):
        """Generate subgroup analysis report."""
        report_path = self.output_dir / 'subgroup_analysis_report.md'

        with open(report_path, 'w') as f:
            f.write("# Subgroup Performance Analysis\n\n")
            f.write(f"**Baseline SNR**: {baseline_snr:.4f}\n\n")
            f.write("---\n\n")

            f.write("## Significant Differences\n\n")

            if len(results.significant_differences) > 0:
                f.write("Subgroups with |Δ SNR| > 0.2:\n\n")
                for feat, subgroup, delta in results.significant_differences[:10]:
                    sign = "📈" if delta > 0 else "📉"
                    f.write(f"- {sign} **{feat}={subgroup}**: Δ SNR = {delta:+.4f}\n")
                f.write("\n")
            else:
                f.write("No significant differences found.\n\n")

            f.write("## Worst Performing Subgroups\n\n")

            if len(results.worst_subgroups) > 0:
                f.write("Subgroups with SNR < 50% of baseline:\n\n")
                for feat, subgroup, snr in results.worst_subgroups[:10]:
                    f.write(f"- ❌ **{feat}={subgroup}**: SNR = {snr:.4f}\n")
                f.write("\n")
            else:
                f.write("No critically poor subgroups found.\n\n")

            f.write("## Recommendations\n\n")

            if results.worst_subgroups:
                f.write("- Investigate worst-performing subgroups for data quality issues\n")
                f.write("- Consider training separate models for different subgroups\n")
                f.write("- Add subgroup-specific features\n")

        if self.verbose:
            print(f"\n  Report saved to {report_path}")

    def _save_subgroup_results(self, results: SubgroupResults):
        """Save results to JSON."""
        output = {
            'subgroup_metrics': results.subgroup_metrics,
            'significant_differences': [
                {'feature': f, 'subgroup': s, 'delta_snr': float(d)}
                for f, s, d in results.significant_differences
            ],
            'worst_subgroups': [
                {'feature': f, 'subgroup': s, 'snr': float(snr)}
                for f, s, snr in results.worst_subgroups
            ]
        }

        with open(self.output_dir / 'subgroup_results.json', 'w') as f:
            json.dump(output, f, indent=2)


class TemporalDiagnostics:
    """
    Temporal Diagnostics for Time Series.

    Analyzes temporal structure in residuals, drift, and seasonality.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        verbose: bool = True
    ):
        """Initialize temporal diagnostics."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose

    def analyze_temporal_structure(
        self,
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex] = None,
        max_lags: int = 40
    ) -> TemporalResults:
        """
        Analyze temporal structure in residuals.

        Parameters
        ----------
        residuals : np.ndarray
            Model residuals (y - y_pred)
        timestamps : pd.DatetimeIndex, optional
            Timestamps for residuals
        max_lags : int
            Maximum lags for ACF/PACF

        Returns
        -------
        TemporalResults
            Complete temporal analysis
        """
        if self.verbose:
            print("\n" + "="*80)
            print("TEMPORAL STRUCTURE ANALYSIS")
            print("="*80)

        # 1. Autocorrelation
        if self.verbose:
            print("  Computing autocorrelation...")

        acf_values = acf(residuals, nlags=max_lags, fft=True)
        pacf_values = pacf(residuals, nlags=max_lags)

        # 2. Durbin-Watson statistic
        dw_stat = durbin_watson(residuals)

        if self.verbose:
            print(f"    Durbin-Watson: {dw_stat:.4f}")
            print(f"    ACF(1): {acf_values[1]:.4f}")

        # 3. Temporal drift (if timestamps provided)
        temporal_drift = {}
        temporal_heatmap_data = None

        if timestamps is not None:
            if self.verbose:
                print("  Analyzing temporal drift...")

            # Group by time periods and compute metrics
            df = pd.DataFrame({
                'timestamp': timestamps,
                'residual': residuals,
                'abs_residual': np.abs(residuals)
            })

            # Monthly drift
            df['year_month'] = df['timestamp'].dt.to_period('M')
            monthly_mae = df.groupby('year_month')['abs_residual'].mean()

            temporal_drift['monthly_trend'] = stats.linregress(
                range(len(monthly_mae)), monthly_mae.values
            ).slope

            # Create heatmap data
            df['year'] = df['timestamp'].dt.year
            df['month'] = df['timestamp'].dt.month
            df['hour'] = df['timestamp'].dt.hour

            temporal_heatmap_data = df.pivot_table(
                values='abs_residual',
                index='hour',
                columns='month',
                aggfunc='mean'
            )

        results = TemporalResults(
            autocorrelation=acf_values,
            partial_autocorrelation=pacf_values,
            durbin_watson_stat=float(dw_stat),
            temporal_drift=temporal_drift,
            temporal_heatmap_data=temporal_heatmap_data
        )

        # Visualize
        self._plot_temporal_analysis(results, residuals, timestamps)

        # Report
        self._generate_temporal_report(results)

        # Save
        self._save_temporal_results(results)

        return results

    def _plot_temporal_analysis(
        self,
        results: TemporalResults,
        residuals: np.ndarray,
        timestamps: Optional[pd.DatetimeIndex]
    ):
        """Visualize temporal analysis."""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. ACF plot
        ax1 = fig.add_subplot(gs[0, 0])
        lags = np.arange(len(results.autocorrelation))
        ax1.stem(lags, results.autocorrelation, basefmt=' ')
        ax1.axhline(y=0, color='black', linestyle='-')
        ax1.axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', label='95% CI')
        ax1.axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--')
        ax1.set_xlabel('Lag', fontsize=12)
        ax1.set_ylabel('ACF', fontsize=12)
        ax1.set_title('Autocorrelation Function', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. PACF plot
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.stem(lags, results.partial_autocorrelation, basefmt=' ')
        ax2.axhline(y=0, color='black', linestyle='-')
        ax2.axhline(y=1.96/np.sqrt(len(residuals)), color='red', linestyle='--', label='95% CI')
        ax2.axhline(y=-1.96/np.sqrt(len(residuals)), color='red', linestyle='--')
        ax2.set_xlabel('Lag', fontsize=12)
        ax2.set_ylabel('PACF', fontsize=12)
        ax2.set_title('Partial Autocorrelation Function', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Residual time series
        if timestamps is not None:
            ax3 = fig.add_subplot(gs[1, :])
            ax3.plot(timestamps, residuals, alpha=0.5, linewidth=0.5)
            ax3.axhline(y=0, color='red', linestyle='--', linewidth=2)
            ax3.set_xlabel('Time', fontsize=12)
            ax3.set_ylabel('Residuals', fontsize=12)
            ax3.set_title('Residuals Over Time', fontsize=14, fontweight='bold')
            ax3.grid(True, alpha=0.3)

        # 4. Temporal heatmap
        if results.temporal_heatmap_data is not None:
            ax4 = fig.add_subplot(gs[2, :])
            sns.heatmap(
                results.temporal_heatmap_data,
                cmap='YlOrRd',
                ax=ax4,
                cbar_kws={'label': 'Mean Absolute Error'}
            )
            ax4.set_xlabel('Month', fontsize=12)
            ax4.set_ylabel('Hour of Day', fontsize=12)
            ax4.set_title('Error Heatmap (Hour × Month)', fontsize=14, fontweight='bold')

        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_temporal_report(self, results: TemporalResults):
        """Generate temporal analysis report."""
        report_path = self.output_dir / 'temporal_analysis_report.md'

        with open(report_path, 'w') as f:
            f.write("# Temporal Structure Analysis\n\n")
            f.write("---\n\n")

            f.write("## Autocorrelation Analysis\n\n")
            f.write(f"- **Durbin-Watson Statistic**: {results.durbin_watson_stat:.4f}\n")
            f.write(f"- **Lag-1 Autocorrelation**: {results.autocorrelation[1]:.4f}\n\n")

            f.write("**Interpretation (Durbin-Watson):**\n")
            if results.durbin_watson_stat < 1.5:
                f.write("- ❌ Strong positive autocorrelation detected (DW < 1.5)\n")
                f.write("- Residuals are not independent\n")
                f.write("- Model is missing temporal structure\n\n")
            elif results.durbin_watson_stat > 2.5:
                f.write("- ⚠️ Negative autocorrelation detected (DW > 2.5)\n")
                f.write("- May indicate overcorrection or model issues\n\n")
            else:
                f.write("- ✓ No significant autocorrelation (1.5 < DW < 2.5)\n")
                f.write("- Residuals appear independent\n\n")

            f.write("**Interpretation (ACF):**\n")
            if abs(results.autocorrelation[1]) > 0.2:
                f.write(f"- ⚠️ High lag-1 autocorrelation ({results.autocorrelation[1]:.3f})\n")
                f.write("- Consider adding lagged features or AR terms\n\n")
            elif abs(results.autocorrelation[1]) > 0.1:
                f.write(f"- ⚠️ Moderate lag-1 autocorrelation ({results.autocorrelation[1]:.3f})\n")
                f.write("- Some temporal structure remains\n\n")
            else:
                f.write(f"- ✓ Low lag-1 autocorrelation ({results.autocorrelation[1]:.3f})\n")
                f.write("- Temporal structure well-captured\n\n")

            if results.temporal_drift:
                f.write("## Temporal Drift\n\n")
                for key, value in results.temporal_drift.items():
                    f.write(f"- **{key}**: {value:.6f}\n")
                f.write("\n")

            f.write("## Recommendations\n\n")

            if results.durbin_watson_stat < 1.5 or abs(results.autocorrelation[1]) > 0.2:
                f.write("**Add Temporal Features:**\n")
                f.write("- Lagged values (y_{t-1}, y_{t-2}, ...)\n")
                f.write("- Rolling statistics (moving averages, std)\n")
                f.write("- Autoregressive terms\n")
                f.write("- Seasonality indicators\n\n")

        if self.verbose:
            print(f"  Report saved to {report_path}")

    def _save_temporal_results(self, results: TemporalResults):
        """Save results to JSON."""
        output = {
            'durbin_watson': float(results.durbin_watson_stat),
            'acf': results.autocorrelation.tolist(),
            'pacf': results.partial_autocorrelation.tolist(),
            'temporal_drift': results.temporal_drift
        }

        with open(self.output_dir / 'temporal_results.json', 'w') as f:
            json.dump(output, f, indent=2)

        if results.temporal_heatmap_data is not None:
            results.temporal_heatmap_data.to_csv(
                self.output_dir / 'temporal_heatmap.csv'
            )

        if self.verbose:
            print(f"  Results saved to {self.output_dir}")
