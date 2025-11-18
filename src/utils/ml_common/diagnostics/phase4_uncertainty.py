"""
Phase 4: Aleatoric vs Epistemic Uncertainty Decomposition

This module separates prediction uncertainty into two components:
1. **Aleatoric (irreducible)**: Inherent noise in the data
2. **Epistemic (reducible)**: Model/knowledge uncertainty

Methods:
- Ensemble uncertainty (variance across models → epistemic)
- Heteroscedastic models (predict both μ and σ² → aleatoric)
- MC Dropout (Bayesian approximation)
- Calibration analysis (predicted vs observed uncertainty)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Callable
from pathlib import Path
from dataclasses import dataclass
import json
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class UncertaintyResults:
    """Results from uncertainty decomposition."""
    total_uncertainty: float
    aleatoric_uncertainty: float
    epistemic_uncertainty: float
    aleatoric_fraction: float
    epistemic_fraction: float
    calibration_score: float
    predictions_with_uncertainty: pd.DataFrame


class HeteroscedasticModel:
    """
    Model that predicts both mean and variance.

    Outputs: μ(x) and σ²(x)
    Loss: Negative log-likelihood assuming Gaussian
    """

    def __init__(
        self,
        base_model_mean: Optional[BaseEstimator] = None,
        base_model_var: Optional[BaseEstimator] = None,
        random_state: int = 42
    ):
        """
        Initialize heteroscedastic model.

        Parameters
        ----------
        base_model_mean : estimator, optional
            Model for predicting mean
        base_model_var : estimator, optional
            Model for predicting log variance
        random_state : int
            Random seed
        """
        if base_model_mean is None:
            self.model_mean = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=random_state
            )
        else:
            self.model_mean = base_model_mean

        if base_model_var is None:
            self.model_var = GradientBoostingRegressor(
                n_estimators=100, max_depth=3, random_state=random_state
            )
        else:
            self.model_var = base_model_var

        self.random_state = random_state

    def fit(self, X, y):
        """
        Fit heteroscedastic model.

        First fits mean model, then fits variance model on squared residuals.
        """
        # Fit mean model
        self.model_mean.fit(X, y)
        y_pred_mean = self.model_mean.predict(X)

        # Compute squared residuals
        squared_residuals = (y - y_pred_mean) ** 2

        # Fit variance model (predict log(σ²))
        log_squared_residuals = np.log(squared_residuals + 1e-8)
        self.model_var.fit(X, log_squared_residuals)

        return self

    def predict(self, X, return_std=False):
        """
        Predict mean and optionally standard deviation.

        Parameters
        ----------
        X : array
            Features
        return_std : bool
            If True, return (mean, std)

        Returns
        -------
        mean or (mean, std)
        """
        mean = self.model_mean.predict(X)

        if return_std:
            log_var = self.model_var.predict(X)
            var = np.exp(log_var)
            std = np.sqrt(var)
            return mean, std
        else:
            return mean

    def predict_with_uncertainty(self, X):
        """
        Predict with uncertainty estimates.

        Returns
        -------
        tuple
            (mean, aleatoric_std, total_std)
        """
        mean, aleatoric_std = self.predict(X, return_std=True)

        # For this model, total uncertainty ≈ aleatoric
        # (epistemic would require ensemble)
        total_std = aleatoric_std

        return mean, aleatoric_std, total_std


class MCDropoutModel:
    """
    Monte Carlo Dropout for uncertainty estimation.

    Uses dropout at test time to approximate Bayesian inference.

    Note: Requires neural network with dropout layers.
    """

    def __init__(
        self,
        base_model,
        n_mc_samples: int = 50,
        random_state: int = 42
    ):
        """
        Initialize MC Dropout model.

        Parameters
        ----------
        base_model : neural network
            Model with dropout layers
        n_mc_samples : int
            Number of forward passes for uncertainty
        random_state : int
            Random seed
        """
        self.base_model = base_model
        self.n_mc_samples = n_mc_samples
        self.random_state = random_state

    def fit(self, X, y):
        """Fit the base model."""
        self.base_model.fit(X, y)
        return self

    def predict_with_uncertainty(self, X):
        """
        Predict with MC dropout uncertainty.

        Returns
        -------
        tuple
            (mean, epistemic_std, total_std)
        """
        # Note: This is a simplified implementation
        # Real implementation would enable dropout during inference

        # For now, approximate with bootstrap-like predictions
        predictions = []

        for _ in range(self.n_mc_samples):
            # Would enable dropout here in real implementation
            pred = self.base_model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        mean = np.mean(predictions, axis=0)
        epistemic_std = np.std(predictions, axis=0)
        total_std = epistemic_std  # Simplified

        return mean, epistemic_std, total_std


class UncertaintyDecomposition:
    """
    Decompose prediction uncertainty into aleatoric and epistemic components.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize uncertainty decomposition.

        Parameters
        ----------
        output_dir : str or Path
            Directory for outputs
        random_state : int
            Random seed
        verbose : bool
            Print progress
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.random_state = random_state
        self.verbose = verbose

    def ensemble_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[BaseEstimator] = None,
        n_models: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate epistemic uncertainty using ensemble.

        Trains multiple models with different random seeds/bootstrap samples.
        Variance across predictions ≈ epistemic uncertainty.

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        base_model : estimator, optional
            Model to ensemble
        n_models : int
            Number of ensemble members

        Returns
        -------
        tuple
            (mean_predictions, epistemic_std, predictions_array)
        """
        if self.verbose:
            print(f"\n  Training ensemble ({n_models} models)...")

        if base_model is None:
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.random_state
            )

        # Train ensemble with bootstrap samples
        predictions = []
        np.random.seed(self.random_state)

        for i in range(n_models):
            # Bootstrap sample
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]

            # Train model
            model = clone(base_model)
            if hasattr(model, 'random_state'):
                model.set_params(random_state=self.random_state + i)

            model.fit(X_boot, y_boot)

            # Predict on full dataset
            pred = model.predict(X)
            predictions.append(pred)

        predictions = np.array(predictions)

        # Compute statistics
        mean_pred = np.mean(predictions, axis=0)
        epistemic_std = np.std(predictions, axis=0)

        if self.verbose:
            print(f"    Mean epistemic uncertainty: {np.mean(epistemic_std):.4f}")

        return mean_pred, epistemic_std, predictions

    def heteroscedastic_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        cv_folds: int = 5
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate aleatoric uncertainty using heteroscedastic model.

        Predicts both μ(x) and σ²(x).

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        cv_folds : int
            CV folds

        Returns
        -------
        tuple
            (mean_predictions, aleatoric_std)
        """
        if self.verbose:
            print("\n  Training heteroscedastic model...")

        model = HeteroscedasticModel(random_state=self.random_state)

        # Cross-validated predictions
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=self.random_state)

        mean_preds = np.zeros(len(y))
        aleatoric_stds = np.zeros(len(y))

        for train_idx, test_idx in cv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train = y[train_idx]

            model.fit(X_train, y_train)
            mean_pred, aleatoric_std = model.predict(X_test, return_std=True)

            mean_preds[test_idx] = mean_pred
            aleatoric_stds[test_idx] = aleatoric_std

        if self.verbose:
            print(f"    Mean aleatoric uncertainty: {np.mean(aleatoric_stds):.4f}")

        return mean_preds, aleatoric_stds

    def decompose_uncertainty(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[BaseEstimator] = None,
        n_ensemble: int = 10,
        cv_folds: int = 5
    ) -> UncertaintyResults:
        """
        Full uncertainty decomposition.

        Combines ensemble (epistemic) and heteroscedastic (aleatoric) methods.

        Parameters
        ----------
        X : np.ndarray
            Features
        y : np.ndarray
            Target
        base_model : estimator, optional
            Model to use
        n_ensemble : int
            Number of ensemble members
        cv_folds : int
            CV folds

        Returns
        -------
        UncertaintyResults
            Complete uncertainty decomposition
        """
        if self.verbose:
            print("\n" + "="*80)
            print("UNCERTAINTY DECOMPOSITION")
            print("="*80)

        # 1. Ensemble for epistemic uncertainty
        ensemble_mean, epistemic_std, _ = self.ensemble_uncertainty(
            X, y, base_model, n_ensemble
        )

        # 2. Heteroscedastic for aleatoric uncertainty
        hetero_mean, aleatoric_std = self.heteroscedastic_uncertainty(
            X, y, cv_folds
        )

        # 3. Combine uncertainties
        # Total variance = aleatoric variance + epistemic variance
        total_var = aleatoric_std**2 + epistemic_std**2
        total_std = np.sqrt(total_var)

        # 4. Aggregate metrics
        mean_aleatoric = np.mean(aleatoric_std**2)
        mean_epistemic = np.mean(epistemic_std**2)
        mean_total = np.mean(total_var)

        aleatoric_fraction = mean_aleatoric / mean_total if mean_total > 0 else 0
        epistemic_fraction = mean_epistemic / mean_total if mean_total > 0 else 0

        # 5. Calibration: check if predicted uncertainty matches observed errors
        residuals = y - ensemble_mean
        squared_residuals = residuals**2

        # Calibration score: correlation between predicted variance and squared residuals
        calibration_score, _ = stats.pearsonr(total_var, squared_residuals)

        if self.verbose:
            print(f"\n  Total uncertainty: {np.mean(total_std):.4f}")
            print(f"  Aleatoric (irreducible): {np.sqrt(mean_aleatoric):.4f} ({aleatoric_fraction:.1%})")
            print(f"  Epistemic (reducible): {np.sqrt(mean_epistemic):.4f} ({epistemic_fraction:.1%})")
            print(f"  Calibration score: {calibration_score:.4f}")

        # 6. Create results DataFrame
        results_df = pd.DataFrame({
            'y_true': y,
            'y_pred_ensemble': ensemble_mean,
            'y_pred_hetero': hetero_mean,
            'aleatoric_std': aleatoric_std,
            'epistemic_std': epistemic_std,
            'total_std': total_std,
            'residual': residuals,
            'squared_residual': squared_residuals
        })

        results = UncertaintyResults(
            total_uncertainty=float(np.mean(total_std)),
            aleatoric_uncertainty=float(np.sqrt(mean_aleatoric)),
            epistemic_uncertainty=float(np.sqrt(mean_epistemic)),
            aleatoric_fraction=float(aleatoric_fraction),
            epistemic_fraction=float(epistemic_fraction),
            calibration_score=float(calibration_score),
            predictions_with_uncertainty=results_df
        )

        # Generate visualizations
        self._plot_uncertainty_decomposition(results)

        # Generate calibration plot
        self._plot_calibration(results)

        # Generate report
        self._generate_uncertainty_report(results)

        # Save results
        self._save_results(results)

        if self.verbose:
            print("\n" + "="*80)

        return results

    def _plot_uncertainty_decomposition(self, results: UncertaintyResults):
        """Visualize uncertainty decomposition."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        df = results.predictions_with_uncertainty

        # 1. Predictions with uncertainty bands
        sort_idx = np.argsort(df['y_pred_ensemble'].values)
        y_sorted = df['y_true'].values[sort_idx]
        pred_sorted = df['y_pred_ensemble'].values[sort_idx]
        total_std_sorted = df['total_std'].values[sort_idx]

        ax1.scatter(pred_sorted, y_sorted, alpha=0.3, s=10)
        ax1.plot([y_sorted.min(), y_sorted.max()], [y_sorted.min(), y_sorted.max()],
                'r--', lw=2, label='Perfect prediction')
        ax1.set_xlabel('Predicted', fontsize=12)
        ax1.set_ylabel('True', fontsize=12)
        ax1.set_title('Predictions with Uncertainty', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Uncertainty components distribution
        ax2.hist(df['aleatoric_std'], bins=30, alpha=0.5, label='Aleatoric', color='blue', edgecolor='black')
        ax2.hist(df['epistemic_std'], bins=30, alpha=0.5, label='Epistemic', color='red', edgecolor='black')
        ax2.hist(df['total_std'], bins=30, alpha=0.5, label='Total', color='green', edgecolor='black')
        ax2.set_xlabel('Uncertainty (Std Dev)', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Uncertainty Distribution', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Uncertainty pie chart
        fractions = [results.aleatoric_fraction, results.epistemic_fraction]
        labels = [f'Aleatoric\n({results.aleatoric_fraction:.1%})',
                 f'Epistemic\n({results.epistemic_fraction:.1%})']
        colors = ['#ff9999', '#66b3ff']

        ax3.pie(fractions, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
        ax3.set_title('Uncertainty Decomposition', fontsize=14, fontweight='bold')

        # 4. Residuals vs total uncertainty
        ax4.scatter(df['total_std'], np.abs(df['residual']), alpha=0.3, s=20)
        ax4.set_xlabel('Predicted Uncertainty (Std)', fontsize=12)
        ax4.set_ylabel('Absolute Residual', fontsize=12)
        ax4.set_title(f'Calibration (r={results.calibration_score:.3f})', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3)

        # Add diagonal line (perfect calibration)
        max_val = max(df['total_std'].max(), np.abs(df['residual']).max())
        ax4.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect calibration')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(self.output_dir / 'uncertainty_decomposition.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_calibration(self, results: UncertaintyResults):
        """Plot calibration curve."""
        df = results.predictions_with_uncertainty

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Binned calibration plot
        # Sort by predicted uncertainty, bin, compute observed vs expected
        n_bins = 10
        sorted_idx = np.argsort(df['total_std'].values)
        bin_size = len(df) // n_bins

        predicted_uncertainty = []
        observed_uncertainty = []

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(df)

            bin_idx = sorted_idx[start_idx:end_idx]

            pred_unc = np.mean(df['total_std'].values[bin_idx])
            obs_unc = np.std(df['residual'].values[bin_idx])

            predicted_uncertainty.append(pred_unc)
            observed_uncertainty.append(obs_unc)

        ax1.scatter(predicted_uncertainty, observed_uncertainty, s=100, alpha=0.7)
        ax1.plot([0, max(predicted_uncertainty)], [0, max(predicted_uncertainty)],
                'r--', lw=2, label='Perfect calibration')
        ax1.set_xlabel('Predicted Uncertainty (Std)', fontsize=12)
        ax1.set_ylabel('Observed Uncertainty (Std of Residuals)', fontsize=12)
        ax1.set_title('Calibration Curve (Binned)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Quantile-Quantile plot
        # Check if residuals normalized by predicted uncertainty are N(0,1)
        normalized_residuals = df['residual'].values / df['total_std'].values
        stats.probplot(normalized_residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Normalized Residuals', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'uncertainty_calibration.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_uncertainty_report(self, results: UncertaintyResults):
        """Generate markdown report."""
        report_path = self.output_dir / 'uncertainty_report.md'

        with open(report_path, 'w') as f:
            f.write("# Uncertainty Decomposition Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now()}\n\n")
            f.write("---\n\n")

            f.write("## Overview\n\n")
            f.write("Prediction uncertainty has two components:\n\n")
            f.write("- **Aleatoric (irreducible)**: Inherent noise in the data/labels\n")
            f.write("- **Epistemic (reducible)**: Model/knowledge uncertainty\n\n")

            f.write("## Results\n\n")

            f.write(f"- **Total Uncertainty**: {results.total_uncertainty:.4f}\n")
            f.write(f"- **Aleatoric**: {results.aleatoric_uncertainty:.4f} ({results.aleatoric_fraction:.1%})\n")
            f.write(f"- **Epistemic**: {results.epistemic_uncertainty:.4f} ({results.epistemic_fraction:.1%})\n")
            f.write(f"- **Calibration Score**: {results.calibration_score:.4f}\n\n")

            f.write("## Interpretation\n\n")

            if results.aleatoric_fraction > 0.6:
                f.write("### ⚠️ High Aleatoric Uncertainty (>60%)\n\n")
                f.write("**Implication**: Most unpredictability is due to inherent data noise.\n\n")
                f.write("**Actions**:\n")
                f.write("- Improving the model will have limited impact\n")
                f.write("- Focus on data quality improvements\n")
                f.write("- Consider if the target is fundamentally noisy\n")
                f.write("- May need to redefine the prediction task\n\n")

            elif results.epistemic_fraction > 0.6:
                f.write("### ✓ High Epistemic Uncertainty (>60%)\n\n")
                f.write("**Implication**: Most unpredictability is due to model limitations.\n\n")
                f.write("**Actions**:\n")
                f.write("- Improvement is possible with better models/features\n")
                f.write("- Try more complex model architectures\n")
                f.write("- Add more training data\n")
                f.write("- Engineer better features\n")
                f.write("- Ensemble methods can help\n\n")

            else:
                f.write("### Mixed Uncertainty (40-60% each)\n\n")
                f.write("**Implication**: Both data noise and model limitations contribute.\n\n")
                f.write("**Actions**:\n")
                f.write("- Balanced approach: improve both data and models\n")
                f.write("- Focus on most impactful improvements\n\n")

            # Calibration interpretation
            f.write("## Calibration\n\n")

            if results.calibration_score > 0.7:
                f.write("✓ **Well calibrated** (correlation > 0.7)\n\n")
                f.write("Predicted uncertainty matches observed errors well. "
                       "Uncertainty estimates are reliable for decision-making.\n\n")
            elif results.calibration_score > 0.4:
                f.write("⚠️ **Moderately calibrated** (correlation 0.4-0.7)\n\n")
                f.write("Predicted uncertainty has some relationship to observed errors, "
                       "but could be improved. Consider calibration techniques.\n\n")
            else:
                f.write("❌ **Poorly calibrated** (correlation < 0.4)\n\n")
                f.write("Predicted uncertainty does not match observed errors. "
                       "Uncertainty estimates may not be reliable. "
                       "Consider recalibration or different uncertainty method.\n\n")

            f.write("## Recommendations\n\n")

            if results.epistemic_fraction > 0.5:
                f.write("**Focus on Model Improvement**\n\n")
                f.write("1. Try deeper/more complex models\n")
                f.write("2. Add engineered features\n")
                f.write("3. Increase training data\n")
                f.write("4. Hyperparameter optimization\n\n")

            if results.aleatoric_fraction > 0.5:
                f.write("**Accept Fundamental Limits**\n\n")
                f.write("1. Document expected error ranges\n")
                f.write("2. Use uncertainty estimates in downstream decisions\n")
                f.write("3. Consider ensemble prediction intervals\n")
                f.write("4. Investigate data quality issues\n\n")

        if self.verbose:
            print(f"  Report saved to {report_path}")

    def _save_results(self, results: UncertaintyResults):
        """Save results to files."""
        # Save summary JSON
        summary = {
            'total_uncertainty': float(results.total_uncertainty),
            'aleatoric_uncertainty': float(results.aleatoric_uncertainty),
            'epistemic_uncertainty': float(results.epistemic_uncertainty),
            'aleatoric_fraction': float(results.aleatoric_fraction),
            'epistemic_fraction': float(results.epistemic_fraction),
            'calibration_score': float(results.calibration_score)
        }

        with open(self.output_dir / 'uncertainty_results.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Save detailed predictions
        results.predictions_with_uncertainty.to_csv(
            self.output_dir / 'predictions_with_uncertainty.csv',
            index=False
        )

        if self.verbose:
            print(f"  Results saved to {self.output_dir}")
