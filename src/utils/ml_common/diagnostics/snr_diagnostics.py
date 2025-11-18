"""
Signal-to-Noise Ratio (SNR) Diagnostics Module

This module implements comprehensive diagnostics for assessing the predictability
of targets in machine learning models, including:
- SNR and R² computation
- Bootstrap confidence intervals
- Permutation testing for statistical significance
- Cross-validated predictions
- Visualization utilities

Phase 1 - Core Diagnostics MVP
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
from sklearn.model_selection import cross_val_predict, KFold, StratifiedKFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from tqdm import tqdm


@dataclass
class SNRMetrics:
    """Container for SNR-related metrics."""
    r2: float
    snr: float
    rmse: float
    nrmse: float
    bootstrap_ci_lower: float
    bootstrap_ci_upper: float
    permutation_pvalue: float
    n_samples: int
    model_name: str


class SNRDiagnostics:
    """
    Main class for computing Signal-to-Noise Ratio diagnostics.

    This class provides methods for:
    - Cross-validated predictions across multiple models
    - Computing SNR, R², and related metrics
    - Bootstrap confidence intervals
    - Permutation testing
    - Visualization and reporting
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        cv_folds: int = 5,
        bootstrap_iterations: int = 1000,
        permutation_iterations: int = 1000,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize SNR diagnostics.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save outputs (plots, data, reports)
        cv_folds : int, default=5
            Number of cross-validation folds
        bootstrap_iterations : int, default=1000
            Number of bootstrap iterations for confidence intervals
        permutation_iterations : int, default=1000
            Number of permutation iterations for statistical testing
        random_state : int, default=42
            Random seed for reproducibility
        verbose : bool, default=True
            Whether to print progress information
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cv_folds = cv_folds
        self.bootstrap_iterations = bootstrap_iterations
        self.permutation_iterations = permutation_iterations
        self.random_state = random_state
        self.verbose = verbose

        # Storage for results
        self.cv_predictions = None
        self.metrics = {}

    def cross_val_predictions(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        stratify: bool = False
    ) -> pd.DataFrame:
        """
        Generate cross-validated predictions for multiple models.

        Parameters
        ----------
        models : dict
            Dictionary of {model_name: model_instance} to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        groups : np.ndarray, optional
            Group labels for grouped cross-validation
        stratify : bool, default=False
            Whether to use stratified k-fold (for classification targets)

        Returns
        -------
        pd.DataFrame
            DataFrame with columns: id, y_true, y_pred, fold, model_name
        """
        if self.verbose:
            print(f"Generating cross-validated predictions for {len(models)} model(s)...")

        # Choose CV strategy
        if groups is not None:
            from sklearn.model_selection import GroupKFold
            cv = GroupKFold(n_splits=self.cv_folds)
            cv_splits = cv.split(X, y, groups)
        elif stratify:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_splits = cv.split(X, y)
        else:
            cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            cv_splits = cv.split(X, y)

        # Collect predictions
        predictions_list = []

        for model_name, model in models.items():
            if self.verbose:
                print(f"  Processing model: {model_name}")

            try:
                # Get cross-validated predictions
                y_pred = cross_val_predict(
                    model, X, y,
                    cv=cv,
                    n_jobs=-1,
                    verbose=0
                )

                # Get fold assignments
                fold_ids = np.zeros(len(y), dtype=int)
                if groups is not None:
                    cv_iterator = GroupKFold(n_splits=self.cv_folds).split(X, y, groups)
                elif stratify:
                    cv_iterator = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state).split(X, y)
                else:
                    cv_iterator = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state).split(X, y)

                for fold_idx, (_, test_idx) in enumerate(cv_iterator):
                    fold_ids[test_idx] = fold_idx

                # Create DataFrame for this model
                model_df = pd.DataFrame({
                    'id': np.arange(len(y)),
                    'y_true': y,
                    'y_pred': y_pred,
                    'fold': fold_ids,
                    'model_name': model_name
                })

                predictions_list.append(model_df)

            except Exception as e:
                warnings.warn(f"Failed to generate predictions for {model_name}: {str(e)}")
                continue

        # Combine all predictions
        self.cv_predictions = pd.concat(predictions_list, ignore_index=True)

        # Save to parquet
        output_file = self.output_dir / 'cv_predictions.parquet'
        self.cv_predictions.to_parquet(output_file, index=False)
        if self.verbose:
            print(f"  Saved cross-validated predictions to {output_file}")

        return self.cv_predictions

    def compute_metrics(
        self,
        cv_predictions: Optional[pd.DataFrame] = None
    ) -> Dict[str, SNRMetrics]:
        """
        Compute SNR metrics for each model from cross-validated predictions.

        Parameters
        ----------
        cv_predictions : pd.DataFrame, optional
            DataFrame with cross-validated predictions. If None, uses stored predictions.

        Returns
        -------
        dict
            Dictionary of {model_name: SNRMetrics}
        """
        if cv_predictions is None:
            cv_predictions = self.cv_predictions

        if cv_predictions is None:
            raise ValueError("No cross-validated predictions available. Run cross_val_predictions() first.")

        if self.verbose:
            print("Computing SNR metrics...")

        metrics_dict = {}

        for model_name in cv_predictions['model_name'].unique():
            model_data = cv_predictions[cv_predictions['model_name'] == model_name]
            y_true = model_data['y_true'].values
            y_pred = model_data['y_pred'].values

            # Compute base metrics
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            y_std = np.std(y_true)
            nrmse = rmse / y_std if y_std > 0 else np.inf

            # SNR = R² / (1 - R²)
            # Handle edge cases
            if r2 >= 1.0:
                snr = np.inf
            elif r2 <= 0:
                snr = 0.0
            else:
                snr = r2 / (1 - r2)

            # Bootstrap confidence intervals
            if self.verbose:
                print(f"  Computing bootstrap CI for {model_name}...")
            ci_lower, ci_upper = bootstrap_r2(
                y_true, y_pred,
                n_iterations=self.bootstrap_iterations,
                confidence_level=0.95,
                random_state=self.random_state
            )

            # Permutation test
            if self.verbose:
                print(f"  Running permutation test for {model_name}...")
            pvalue = permutation_test(
                y_true, y_pred,
                n_permutations=self.permutation_iterations,
                random_state=self.random_state
            )

            # Create metrics object
            metrics_dict[model_name] = SNRMetrics(
                r2=r2,
                snr=snr,
                rmse=rmse,
                nrmse=nrmse,
                bootstrap_ci_lower=ci_lower,
                bootstrap_ci_upper=ci_upper,
                permutation_pvalue=pvalue,
                n_samples=len(y_true),
                model_name=model_name
            )

            if self.verbose:
                print(f"  {model_name}: R²={r2:.4f}, SNR={snr:.4f}, p-value={pvalue:.4f}")

        self.metrics = metrics_dict

        # Save metrics to JSON
        metrics_json = {
            name: {
                'r2': float(m.r2),
                'snr': float(m.snr) if not np.isinf(m.snr) else 'inf',
                'rmse': float(m.rmse),
                'nrmse': float(m.nrmse),
                'bootstrap_ci_lower': float(m.bootstrap_ci_lower),
                'bootstrap_ci_upper': float(m.bootstrap_ci_upper),
                'permutation_pvalue': float(m.permutation_pvalue),
                'n_samples': int(m.n_samples)
            }
            for name, m in metrics_dict.items()
        }

        output_file = self.output_dir / 'snr_metrics.json'
        with open(output_file, 'w') as f:
            json.dump(metrics_json, f, indent=2)
        if self.verbose:
            print(f"  Saved metrics to {output_file}")

        return metrics_dict

    def create_visualizations(
        self,
        cv_predictions: Optional[pd.DataFrame] = None,
        metrics: Optional[Dict[str, SNRMetrics]] = None
    ) -> Dict[str, Path]:
        """
        Create diagnostic visualizations.

        Parameters
        ----------
        cv_predictions : pd.DataFrame, optional
            Cross-validated predictions
        metrics : dict, optional
            SNR metrics dictionary

        Returns
        -------
        dict
            Dictionary mapping plot names to file paths
        """
        if cv_predictions is None:
            cv_predictions = self.cv_predictions
        if metrics is None:
            metrics = self.metrics

        if cv_predictions is None or metrics is None:
            raise ValueError("Need predictions and metrics to create visualizations")

        if self.verbose:
            print("Creating visualizations...")

        plot_paths = {}

        # 1. Y vs Y_pred scatter plots (one per model)
        for model_name in cv_predictions['model_name'].unique():
            model_data = cv_predictions[cv_predictions['model_name'] == model_name]

            fig, ax = plt.subplots(figsize=(10, 10))

            # Scatter plot
            ax.scatter(
                model_data['y_true'],
                model_data['y_pred'],
                alpha=0.5,
                s=20,
                edgecolors='none'
            )

            # Identity line
            min_val = min(model_data['y_true'].min(), model_data['y_pred'].min())
            max_val = max(model_data['y_true'].max(), model_data['y_pred'].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')

            # Add density contours if enough points
            if len(model_data) > 100:
                from scipy.stats import gaussian_kde
                try:
                    xy = np.vstack([model_data['y_true'], model_data['y_pred']])
                    z = gaussian_kde(xy)(xy)
                    idx = z.argsort()
                    x, y, z = model_data['y_true'].values[idx], model_data['y_pred'].values[idx], z[idx]
                    scatter = ax.scatter(x, y, c=z, s=20, alpha=0.5, cmap='viridis', edgecolors='none')
                    plt.colorbar(scatter, ax=ax, label='Density')
                except:
                    pass

            # Add metrics to plot
            m = metrics[model_name]
            textstr = f'R² = {m.r2:.4f}\nSNR = {m.snr:.4f}\nRMSE = {m.rmse:.4f}\np-value = {m.permutation_pvalue:.4f}'
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
                   verticalalignment='top', bbox=props)

            ax.set_xlabel('True Values', fontsize=12)
            ax.set_ylabel('Predicted Values', fontsize=12)
            ax.set_title(f'Predicted vs True Values: {model_name}', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plot_path = self.output_dir / f'y_vs_ypred_{model_name.replace(" ", "_")}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            plot_paths[f'y_vs_ypred_{model_name}'] = plot_path
            if self.verbose:
                print(f"  Saved plot: {plot_path}")

        # 2. Residual plots
        for model_name in cv_predictions['model_name'].unique():
            model_data = cv_predictions[cv_predictions['model_name'] == model_name]
            residuals = model_data['y_true'] - model_data['y_pred']

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

            # Residuals vs predicted
            ax1.scatter(model_data['y_pred'], residuals, alpha=0.5, s=20, edgecolors='none')
            ax1.axhline(y=0, color='r', linestyle='--', lw=2)

            # Add LOWESS smoothing
            try:
                from statsmodels.nonparametric.smoothers_lowess import lowess
                smoothed = lowess(residuals.values, model_data['y_pred'].values, frac=0.3)
                ax1.plot(smoothed[:, 0], smoothed[:, 1], 'g-', lw=2, label='LOWESS smoothing')
                ax1.legend()
            except:
                pass

            ax1.set_xlabel('Predicted Values', fontsize=12)
            ax1.set_ylabel('Residuals', fontsize=12)
            ax1.set_title(f'Residual Plot: {model_name}', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)

            # Residual histogram
            ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
            ax2.axvline(x=0, color='r', linestyle='--', lw=2)
            ax2.set_xlabel('Residuals', fontsize=12)
            ax2.set_ylabel('Frequency', fontsize=12)
            ax2.set_title(f'Residual Distribution: {model_name}', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)

            plot_path = self.output_dir / f'residuals_{model_name.replace(" ", "_")}.png'
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()

            plot_paths[f'residuals_{model_name}'] = plot_path
            if self.verbose:
                print(f"  Saved plot: {plot_path}")

        # 3. SNR bar chart across models
        model_names = list(metrics.keys())
        snr_values = [metrics[name].snr for name in model_names]
        r2_values = [metrics[name].r2 for name in model_names]
        ci_lower = [metrics[name].bootstrap_ci_lower for name in model_names]
        ci_upper = [metrics[name].bootstrap_ci_upper for name in model_names]

        # Calculate error bars for R² (converted to SNR scale)
        snr_ci_lower = [r / (1 - r) if r < 1 and r > 0 else 0 for r in ci_lower]
        snr_ci_upper = [r / (1 - r) if r < 1 and r > 0 else max(snr_values) * 2 for r in ci_upper]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # SNR bar chart
        x_pos = np.arange(len(model_names))
        bars = ax1.bar(x_pos, snr_values, alpha=0.7, edgecolor='black', linewidth=1.5)

        # Add error bars (bootstrap CI converted to SNR)
        yerr_lower = [snr - ci_l for snr, ci_l in zip(snr_values, snr_ci_lower)]
        yerr_upper = [ci_u - snr for snr, ci_u in zip(snr_values, snr_ci_upper)]
        ax1.errorbar(x_pos, snr_values, yerr=[yerr_lower, yerr_upper], fmt='none',
                    ecolor='black', capsize=5, capthick=2)

        # Color bars by SNR thresholds
        colors = []
        for snr in snr_values:
            if snr > 1.0:
                colors.append('green')
            elif snr > 0.3:
                colors.append('orange')
            else:
                colors.append('red')

        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)

        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_ylabel('SNR', fontsize=12)
        ax1.set_title('Signal-to-Noise Ratio by Model (with 95% CI)', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.axhline(y=1.0, color='blue', linestyle='--', lw=2, label='SNR = 1.0 (signal = noise)')
        ax1.axhline(y=0.3, color='orange', linestyle='--', lw=2, label='SNR = 0.3 (weak signal)')
        ax1.legend()

        # R² bar chart
        bars2 = ax2.bar(x_pos, r2_values, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.errorbar(x_pos, r2_values,
                    yerr=[[r - ci_l for r, ci_l in zip(r2_values, ci_lower)],
                          [ci_u - r for r, ci_u in zip(r2_values, ci_upper)]],
                    fmt='none', ecolor='black', capsize=5, capthick=2)

        # Color bars by R² thresholds
        colors2 = []
        for r2 in r2_values:
            if r2 > 0.40:
                colors2.append('green')
            elif r2 > 0.10:
                colors2.append('orange')
            else:
                colors2.append('red')

        for bar, color in zip(bars2, colors2):
            bar.set_facecolor(color)

        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_title('R² Score by Model (with 95% CI)', fontsize=14, fontweight='bold')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(model_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.axhline(y=0.40, color='blue', linestyle='--', lw=2, label='R² = 0.40 (strong signal)')
        ax2.axhline(y=0.10, color='orange', linestyle='--', lw=2, label='R² = 0.10 (weak signal)')
        ax2.legend()

        plot_path = self.output_dir / 'snr_comparison.png'
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        plot_paths['snr_comparison'] = plot_path
        if self.verbose:
            print(f"  Saved plot: {plot_path}")

        return plot_paths

    def generate_report(
        self,
        metrics: Optional[Dict[str, SNRMetrics]] = None,
        plot_paths: Optional[Dict[str, Path]] = None
    ) -> Tuple[Path, Path]:
        """
        Generate comprehensive CSV and Markdown reports.

        Parameters
        ----------
        metrics : dict, optional
            SNR metrics dictionary
        plot_paths : dict, optional
            Dictionary of plot file paths

        Returns
        -------
        tuple
            Paths to (CSV report, Markdown report)
        """
        if metrics is None:
            metrics = self.metrics
        if metrics is None:
            raise ValueError("No metrics available for reporting")

        if self.verbose:
            print("Generating reports...")

        # Create CSV report
        csv_data = []
        for model_name, m in metrics.items():
            csv_data.append({
                'model_name': model_name,
                'r2': m.r2,
                'snr': m.snr if not np.isinf(m.snr) else 999.0,
                'rmse': m.rmse,
                'nrmse': m.nrmse,
                'bootstrap_ci_lower': m.bootstrap_ci_lower,
                'bootstrap_ci_upper': m.bootstrap_ci_upper,
                'permutation_pvalue': m.permutation_pvalue,
                'n_samples': m.n_samples,
                'signal_strength': _interpret_r2(m.r2),
                'snr_interpretation': _interpret_snr(m.snr),
                'statistical_significance': _interpret_pvalue(m.permutation_pvalue),
                'ci_interpretation': _interpret_ci(m.bootstrap_ci_lower, m.bootstrap_ci_upper)
            })

        csv_df = pd.DataFrame(csv_data)
        csv_path = self.output_dir / 'snr_diagnostics_report.csv'
        csv_df.to_csv(csv_path, index=False)
        if self.verbose:
            print(f"  Saved CSV report: {csv_path}")

        # Create Markdown report
        md_path = self.output_dir / 'snr_diagnostics_report.md'
        with open(md_path, 'w') as f:
            f.write("# Signal-to-Noise Ratio (SNR) Diagnostic Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            # Executive Summary
            f.write("## Executive Summary\n\n")
            best_model = max(metrics.items(), key=lambda x: x[1].snr)
            f.write(f"- **Total Models Evaluated**: {len(metrics)}\n")
            f.write(f"- **Best Performing Model**: {best_model[0]}\n")
            f.write(f"  - R² = {best_model[1].r2:.4f}\n")
            f.write(f"  - SNR = {best_model[1].snr:.4f}\n")
            f.write(f"  - p-value = {best_model[1].permutation_pvalue:.4f}\n\n")

            # Metrics Table
            f.write("## Model Performance Metrics\n\n")
            f.write("| Model | R² | SNR | RMSE | nRMSE | 95% CI Lower | 95% CI Upper | p-value | N |\n")
            f.write("|-------|----|----|------|-------|--------------|--------------|---------|---|\n")

            for model_name, m in sorted(metrics.items(), key=lambda x: x[1].snr, reverse=True):
                snr_str = f"{m.snr:.4f}" if not np.isinf(m.snr) else "∞"
                f.write(f"| {model_name} | {m.r2:.4f} | {snr_str} | {m.rmse:.4f} | {m.nrmse:.4f} | "
                       f"{m.bootstrap_ci_lower:.4f} | {m.bootstrap_ci_upper:.4f} | "
                       f"{m.permutation_pvalue:.4f} | {m.n_samples} |\n")

            f.write("\n")

            # Interpretation Guidelines
            f.write("## Interpretation Guidelines\n\n")
            f.write(_get_interpretation_guidelines())

            # Model-specific interpretations
            f.write("## Model-Specific Analysis\n\n")
            for model_name, m in sorted(metrics.items(), key=lambda x: x[1].snr, reverse=True):
                f.write(f"### {model_name}\n\n")
                f.write(f"**Performance Metrics:**\n")
                f.write(f"- R² = {m.r2:.4f} ({_interpret_r2(m.r2)})\n")
                snr_str = f"{m.snr:.4f}" if not np.isinf(m.snr) else "∞"
                f.write(f"- SNR = {snr_str} ({_interpret_snr(m.snr)})\n")
                f.write(f"- RMSE = {m.rmse:.4f}, nRMSE = {m.nrmse:.4f}\n")
                f.write(f"- 95% Bootstrap CI: [{m.bootstrap_ci_lower:.4f}, {m.bootstrap_ci_upper:.4f}]\n")
                f.write(f"- Permutation p-value = {m.permutation_pvalue:.4f} ({_interpret_pvalue(m.permutation_pvalue)})\n\n")

                f.write(f"**Interpretation:**\n")
                f.write(_get_model_interpretation(m))
                f.write("\n\n")

            # Visualizations
            if plot_paths:
                f.write("## Visualizations\n\n")
                for plot_name, plot_path in plot_paths.items():
                    rel_path = plot_path.relative_to(self.output_dir)
                    f.write(f"### {plot_name.replace('_', ' ').title()}\n\n")
                    f.write(f"![{plot_name}]({rel_path})\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")
            f.write(_get_recommendations(metrics))

        if self.verbose:
            print(f"  Saved Markdown report: {md_path}")

        return csv_path, md_path

    def run_full_diagnostics(
        self,
        models: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
        stratify: bool = False
    ) -> Tuple[pd.DataFrame, Dict[str, SNRMetrics], Dict[str, Path], Tuple[Path, Path]]:
        """
        Run the complete diagnostic pipeline.

        Parameters
        ----------
        models : dict
            Dictionary of models to evaluate
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        groups : np.ndarray, optional
            Group labels for grouped CV
        stratify : bool, default=False
            Whether to stratify folds

        Returns
        -------
        tuple
            (cv_predictions, metrics, plot_paths, report_paths)
        """
        # 1. Cross-validated predictions
        cv_preds = self.cross_val_predictions(models, X, y, groups, stratify)

        # 2. Compute metrics
        metrics = self.compute_metrics(cv_preds)

        # 3. Create visualizations
        plot_paths = self.create_visualizations(cv_preds, metrics)

        # 4. Generate reports
        report_paths = self.generate_report(metrics, plot_paths)

        if self.verbose:
            print("\n" + "="*80)
            print("SNR DIAGNOSTICS COMPLETE")
            print("="*80)
            print(f"Output directory: {self.output_dir}")
            print(f"  - Cross-validated predictions: cv_predictions.parquet")
            print(f"  - Metrics JSON: snr_metrics.json")
            print(f"  - CSV report: {report_paths[0].name}")
            print(f"  - Markdown report: {report_paths[1].name}")
            print(f"  - Visualizations: {len(plot_paths)} plots generated")
            print("="*80 + "\n")

        return cv_preds, metrics, plot_paths, report_paths


# Standalone utility functions

def compute_snr_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute basic SNR metrics for a single model.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values

    Returns
    -------
    dict
        Dictionary with r2, snr, rmse, nrmse
    """
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    y_std = np.std(y_true)
    nrmse = rmse / y_std if y_std > 0 else np.inf

    if r2 >= 1.0:
        snr = np.inf
    elif r2 <= 0:
        snr = 0.0
    else:
        snr = r2 / (1 - r2)

    return {
        'r2': r2,
        'snr': snr,
        'rmse': rmse,
        'nrmse': nrmse
    }


def bootstrap_r2(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float]:
    """
    Compute bootstrap confidence interval for R².

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    n_iterations : int, default=1000
        Number of bootstrap iterations
    confidence_level : float, default=0.95
        Confidence level (e.g., 0.95 for 95% CI)
    random_state : int, default=42
        Random seed

    Returns
    -------
    tuple
        (lower_bound, upper_bound) of confidence interval
    """
    np.random.seed(random_state)
    n_samples = len(y_true)

    r2_scores = np.zeros(n_iterations)

    for i in range(n_iterations):
        # Bootstrap resample with replacement
        indices = np.random.randint(0, n_samples, n_samples)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]

        # Compute R² for this bootstrap sample
        r2_scores[i] = r2_score(y_true_boot, y_pred_boot)

    # Compute confidence interval
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    ci_lower = np.percentile(r2_scores, lower_percentile)
    ci_upper = np.percentile(r2_scores, upper_percentile)

    return ci_lower, ci_upper


def permutation_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_permutations: int = 1000,
    random_state: int = 42,
    metric: str = 'r2'
) -> float:
    """
    Perform permutation test to assess statistical significance.

    Parameters
    ----------
    y_true : np.ndarray
        True target values
    y_pred : np.ndarray
        Predicted values
    n_permutations : int, default=1000
        Number of permutation iterations
    random_state : int, default=42
        Random seed
    metric : str, default='r2'
        Metric to use ('r2' or 'mse')

    Returns
    -------
    float
        p-value (fraction of permutations with score >= observed score)
    """
    np.random.seed(random_state)

    # Compute observed score
    if metric == 'r2':
        observed_score = r2_score(y_true, y_pred)
    elif metric == 'mse':
        observed_score = -mean_squared_error(y_true, y_pred)  # Negative for "higher is better"
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Permutation test
    permuted_scores = np.zeros(n_permutations)

    for i in range(n_permutations):
        # Shuffle target values
        y_permuted = np.random.permutation(y_true)

        # Compute score with permuted targets
        if metric == 'r2':
            permuted_scores[i] = r2_score(y_permuted, y_pred)
        elif metric == 'mse':
            permuted_scores[i] = -mean_squared_error(y_permuted, y_pred)

    # Compute p-value
    pvalue = np.mean(permuted_scores >= observed_score)

    return pvalue


# Interpretation helper functions

def _interpret_r2(r2: float) -> str:
    """Interpret R² value."""
    if r2 > 0.40:
        return "Strong predictable signal"
    elif r2 > 0.10:
        return "Weak-moderate signal"
    else:
        return "Barely predictable"


def _interpret_snr(snr: float) -> str:
    """Interpret SNR value."""
    if np.isinf(snr) or snr > 1.0:
        return "Signal > noise (learnable)"
    elif snr > 0.3:
        return "Weak but real signal"
    else:
        return "Noise dominates"


def _interpret_pvalue(pvalue: float) -> str:
    """Interpret permutation p-value."""
    if pvalue < 0.01:
        return "Statistically robust"
    elif pvalue <= 0.20:
        return "Weak/unstable signal"
    else:
        return "No better than chance"


def _interpret_ci(ci_lower: float, ci_upper: float) -> str:
    """Interpret bootstrap confidence interval."""
    if ci_lower > 0.05:
        return "Reliably above noise"
    elif ci_lower > 0:
        return "Signal present but fragile"
    else:
        return "May be indistinguishable from noise"


def _get_interpretation_guidelines() -> str:
    """Get the interpretation guidelines markdown text."""
    return """
### 1. R² (Coefficient of Determination)
- **R² > 0.40**: The target has a **strong predictable signal**; meaningful modeling gains are possible.
- **0.10 < R² ≤ 0.40**: The target has a **weak-moderate signal**; features matter more than model choice.
- **R² ≤ 0.10**: The target is **barely predictable**; noise likely dominates.

### 2. SNR (Signal-to-Noise Ratio)
- **SNR > 1**: **Signal is stronger than noise**; the target is learnable.
- **0.3 < SNR ≤ 1**: **Weak but real signal** exists; more features or nonlinear models may help.
- **SNR ≤ 0.3**: **Noise overwhelms signal**; predictability is fundamentally low.

**Note**: SNR depends on features & model. Improvements can come from:
- Trying stronger models (deeper trees, neural networks)
- Adding engineered features
- If SNR rises → features were missing structure
- If SNR stays low → target may be intrinsically noisy

### 3. Permutation p-value
- **p < 0.01**: The model captures a **real, statistically robust pattern**.
- **0.01 ≤ p ≤ 0.20**: There might be signal, but it's **weak or unstable**.
- **p > 0.20**: The model performs **no better than chance**; label likely noisy.

### 4. Bootstrap R² Confidence Interval
- **CI does NOT include 0**: Performance is **reliably above noise level**.
- **CI barely clears 0** (lower bound < 0.05): Signal is present but **fragile**.
- **CI spans below 0**: Model performance might be **indistinguishable from noise**.

### 5. Residual Structure
- **Residuals look random**: The model extracted essentially **all available signal**.
- **Residuals show patterns**: There is **remaining structure** the model/features are missing.
- **Residuals differ across subgroups**: **Predictability varies by segment** (not globally noisy).
"""


def _get_model_interpretation(m: SNRMetrics) -> str:
    """Get interpretation text for a specific model."""
    interpretation = []

    # Overall assessment
    if m.r2 > 0.40 and m.snr > 1.0 and m.permutation_pvalue < 0.01:
        interpretation.append("✅ **Strong Performance**: This model demonstrates excellent predictive capability with statistically robust signal detection.")
    elif m.r2 > 0.10 and m.snr > 0.3 and m.permutation_pvalue < 0.20:
        interpretation.append("⚠️ **Moderate Performance**: This model captures some real signal but there's room for improvement through feature engineering or model complexity.")
    else:
        interpretation.append("❌ **Weak Performance**: This model struggles to extract meaningful signal. Consider feature engineering, different model architectures, or investigating if the target is intrinsically noisy.")

    # Specific recommendations
    if m.snr < 0.3:
        interpretation.append("\n**Recommendation**: SNR is very low. Investigate whether the target has intrinsic noise or if critical features are missing.")

    if m.permutation_pvalue > 0.20:
        interpretation.append("\n**Warning**: Permutation test suggests performance may not be significantly better than random chance.")

    if m.bootstrap_ci_lower < 0:
        interpretation.append("\n**Warning**: Bootstrap CI includes negative values, suggesting performance may be unstable across different samples.")

    if 0.10 < m.r2 <= 0.40:
        interpretation.append("\n**Opportunity**: R² suggests moderate signal. Try:\n"
                            "  - Adding engineered features (interactions, transformations)\n"
                            "  - Trying nonlinear models (gradient boosting, neural networks)\n"
                            "  - Feature selection to reduce noise")

    return "\n".join(interpretation)


def _get_recommendations(metrics: Dict[str, SNRMetrics]) -> str:
    """Generate recommendations based on all model metrics."""
    recommendations = []

    best_model = max(metrics.items(), key=lambda x: x[1].snr)
    worst_model = min(metrics.items(), key=lambda x: x[1].snr)

    avg_snr = np.mean([m.snr for m in metrics.values() if not np.isinf(m.snr)])
    avg_r2 = np.mean([m.r2 for m in metrics.values()])

    # Overall assessment
    if avg_snr > 1.0:
        recommendations.append("### ✅ Strong Signal Detected\n")
        recommendations.append(f"The average SNR across models is {avg_snr:.3f}, indicating that the target contains substantial predictable signal. "
                             "Focus on model optimization and hyperparameter tuning to maximize performance.\n")
    elif avg_snr > 0.3:
        recommendations.append("### ⚠️ Moderate Signal Present\n")
        recommendations.append(f"The average SNR across models is {avg_snr:.3f}, indicating moderate predictability. "
                             "Consider the following improvements:\n"
                             "- Engineer additional features (interactions, domain-specific transforms)\n"
                             "- Try ensemble methods or more complex model architectures\n"
                             "- Investigate feature selection to reduce noise\n")
    else:
        recommendations.append("### ❌ Weak Signal - Fundamental Limitations\n")
        recommendations.append(f"The average SNR across models is {avg_snr:.3f}, suggesting the target may be inherently difficult to predict. "
                             "Recommended actions:\n"
                             "- Review data collection process for potential quality issues\n"
                             "- Investigate if target definition is appropriate\n"
                             "- Consider if the prediction task is fundamentally feasible\n"
                             "- Explore alternative target formulations\n")

    # Model comparison
    if len(metrics) > 1:
        snr_range = best_model[1].snr - worst_model[1].snr
        if snr_range > 0.5:
            recommendations.append(f"### Model Selection Impact\n")
            recommendations.append(f"There's significant variation in SNR across models (range: {snr_range:.3f}). "
                                 f"The best model ({best_model[0]}) substantially outperforms the worst ({worst_model[0]}), "
                                 f"indicating model architecture choice is important for this problem.\n")

    # Next steps
    recommendations.append("### Next Steps\n")
    recommendations.append("1. **Feature Engineering**: Conduct feature ablation studies to identify which feature groups contribute most to signal\n")
    recommendations.append("2. **Model Exploration**: Test additional model families (if not already done)\n")
    recommendations.append("3. **Residual Analysis**: Examine residuals for patterns that suggest missing features\n")
    recommendations.append("4. **Uncertainty Quantification**: Implement ensemble or heteroscedastic models to separate aleatoric vs epistemic uncertainty\n")
    recommendations.append("5. **Subgroup Analysis**: Check if SNR varies across data subgroups\n")

    return "\n".join(recommendations)


# Convenience function for quick analysis
def cross_val_predictions(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    groups: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Generate cross-validated predictions for a single model.

    Parameters
    ----------
    model : estimator
        Scikit-learn compatible model
    X : np.ndarray
        Feature matrix
    y : np.ndarray
        Target values
    cv : int, default=5
        Number of CV folds
    groups : np.ndarray, optional
        Group labels for GroupKFold

    Returns
    -------
    np.ndarray
        Cross-validated predictions
    """
    if groups is not None:
        from sklearn.model_selection import GroupKFold
        cv_splitter = GroupKFold(n_splits=cv)
    else:
        cv_splitter = cv

    return cross_val_predict(model, X, y, cv=cv_splitter, n_jobs=-1)
