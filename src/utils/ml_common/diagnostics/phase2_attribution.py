"""
Phase 2: Signal-vs-Noise Attribution Experiments

This module provides tools for diagnosing whether low SNR is due to:
- Feature limitations (missing important features)
- Inherent noise in the target
- Model limitations (wrong model family)

Experiments:
1. Model family sweep - Compare different model architectures
2. Feature ablation - Test impact of removing feature groups
3. Synthetic signal injection - Verify pipeline can detect known signals
4. Residual modeling - Check for missed patterns in residuals
5. Heteroscedastic analysis - Test if prediction errors are predictable
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
from dataclasses import dataclass
import json
import warnings

from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb

from .snr_diagnostics import compute_snr_metrics, SNRDiagnostics


@dataclass
class AttributionResults:
    """Results from attribution experiments."""
    model_family_results: Dict[str, Dict[str, float]]
    ablation_results: Dict[str, Dict[str, float]]
    synthetic_injection_results: Dict[str, Any]
    residual_model_results: Dict[str, float]
    heteroscedastic_results: Dict[str, float]


class SignalAttributionExperiments:
    """
    Signal-vs-Noise Attribution Experiment Runner.

    This class runs comprehensive experiments to determine the source
    of low predictability: features, model, or inherent noise.
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        cv_folds: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Initialize attribution experiments.

        Parameters
        ----------
        output_dir : str or Path
            Directory to save outputs
        cv_folds : int, default=5
            Number of CV folds
        random_state : int, default=42
            Random seed
        verbose : bool, default=True
            Print progress information
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.cv_folds = cv_folds
        self.random_state = random_state
        self.verbose = verbose

    def model_family_sweep(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task: str = 'regression'
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare SNR across different model families.

        Tests: Linear, Random Forest, Gradient Boosting, MLP

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        task : str, default='regression'
            'regression' or 'classification'

        Returns
        -------
        dict
            {model_name: {r2, snr, rmse, ...}}
        """
        if self.verbose:
            print("\n" + "="*80)
            print("MODEL FAMILY SWEEP")
            print("="*80)

        # Define model families
        if task == 'regression':
            models = {
                # LightGBM (PRIMARY - matching meta-labeling config)
                'LGBM_MetaLabeling': lgb.LGBMRegressor(
                    objective='regression',
                    metric='rmse',
                    n_estimators=800,
                    max_depth=8,
                    learning_rate=0.01,
                    num_leaves=63,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.2,
                    n_jobs=-1,
                    verbose=-1,
                    random_state=self.random_state
                ),
                # Linear models
                'Linear_Ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'Linear_Lasso': Lasso(alpha=0.1, random_state=self.random_state, max_iter=2000),
                'Linear_ElasticNet': ElasticNet(alpha=0.1, random_state=self.random_state, max_iter=2000),
                # Random Forest
                'RF_Shallow': RandomForestRegressor(
                    n_estimators=100, max_depth=5, random_state=self.random_state, n_jobs=-1
                ),
                'RF_Deep': RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1
                ),
                # Gradient Boosting (sklearn)
                'GBM_Shallow': GradientBoostingRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=self.random_state
                ),
                'GBM_Deep': GradientBoostingRegressor(
                    n_estimators=100, max_depth=7, learning_rate=0.1, random_state=self.random_state
                ),
                # Neural networks
                'MLP_Small': MLPRegressor(
                    hidden_layer_sizes=(50,), activation='relu',
                    random_state=self.random_state, max_iter=500
                ),
                'MLP_Large': MLPRegressor(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    random_state=self.random_state, max_iter=500
                )
            }
        else:  # classification
            models = {
                # LightGBM (PRIMARY - matching meta-labeling config)
                'LGBM_MetaLabeling': lgb.LGBMClassifier(
                    objective='binary',
                    metric='auc',
                    n_estimators=800,
                    max_depth=8,
                    learning_rate=0.01,
                    num_leaves=63,
                    min_child_samples=20,
                    subsample=0.8,
                    subsample_freq=1,
                    colsample_bytree=0.7,
                    reg_alpha=0.1,
                    reg_lambda=0.2,
                    class_weight='balanced',
                    n_jobs=-1,
                    verbose=-1,
                    random_state=self.random_state
                ),
                # Linear models
                'Logistic': LogisticRegression(max_iter=1000, random_state=self.random_state),
                # Random Forest
                'RF_Shallow': RandomForestClassifier(
                    n_estimators=100, max_depth=5, random_state=self.random_state, n_jobs=-1
                ),
                'RF_Deep': RandomForestClassifier(
                    n_estimators=100, max_depth=15, random_state=self.random_state, n_jobs=-1
                ),
                # Gradient Boosting (sklearn)
                'GBM_Shallow': GradientBoostingClassifier(
                    n_estimators=100, max_depth=3, learning_rate=0.1, random_state=self.random_state
                ),
                'GBM_Deep': GradientBoostingClassifier(
                    n_estimators=100, max_depth=7, learning_rate=0.1, random_state=self.random_state
                ),
                # Neural networks
                'MLP_Small': MLPClassifier(
                    hidden_layer_sizes=(50,), activation='relu',
                    random_state=self.random_state, max_iter=500
                ),
                'MLP_Large': MLPClassifier(
                    hidden_layer_sizes=(100, 50), activation='relu',
                    random_state=self.random_state, max_iter=500
                )
            }

        results = {}
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        for model_name, model in models.items():
            if self.verbose:
                print(f"  Testing {model_name}...")

            try:
                # Get CV predictions
                if task == 'classification':
                    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1, method='predict_proba')[:, 1]
                else:
                    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

                # Compute metrics
                metrics = compute_snr_metrics(y, y_pred)
                results[model_name] = metrics

                if self.verbose:
                    print(f"    R²={metrics['r2']:.4f}, SNR={metrics['snr']:.4f}")

            except Exception as e:
                warnings.warn(f"Failed for {model_name}: {str(e)}")
                continue

        # Create visualization
        self._plot_model_family_comparison(results)

        # Save results
        output_file = self.output_dir / 'model_family_sweep.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n  Results saved to {output_file}")

        return results

    def feature_ablation(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_groups: Dict[str, List[str]],
        base_model: Optional[BaseEstimator] = None,
        mode: str = 'drop_group'
    ) -> Dict[str, Dict[str, float]]:
        """
        Test impact of removing feature groups.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix with column names
        y : np.ndarray
            Target values
        feature_groups : dict
            {group_name: [feature_names]}
        base_model : estimator, optional
            Model to use (default: RandomForestRegressor)
        mode : str, default='drop_group'
            'drop_group' - remove each group independently
            'drop_all_except' - keep only one group at a time

        Returns
        -------
        dict
            {experiment_name: {r2, snr, delta_r2, delta_snr}}
        """
        if self.verbose:
            print("\n" + "="*80)
            print("FEATURE ABLATION")
            print("="*80)

        if base_model is None:
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.random_state, n_jobs=-1
            )

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Baseline with all features
        if self.verbose:
            print("  Computing baseline (all features)...")
        y_pred_baseline = cross_val_predict(clone(base_model), X.values, y, cv=cv, n_jobs=-1)
        baseline_metrics = compute_snr_metrics(y, y_pred_baseline)

        results = {
            'baseline_all_features': {
                **baseline_metrics,
                'delta_r2': 0.0,
                'delta_snr': 0.0,
                'n_features': X.shape[1]
            }
        }

        if self.verbose:
            print(f"    Baseline R²={baseline_metrics['r2']:.4f}, SNR={baseline_metrics['snr']:.4f}")

        # Ablation experiments
        for group_name, features in feature_groups.items():
            if mode == 'drop_group':
                # Drop this group
                experiment_name = f'drop_{group_name}'
                remaining_features = [col for col in X.columns if col not in features]

                if len(remaining_features) == 0:
                    warnings.warn(f"Dropping {group_name} would leave no features")
                    continue

                X_ablated = X[remaining_features]

            else:  # drop_all_except
                # Keep only this group
                experiment_name = f'only_{group_name}'
                X_ablated = X[features]

                if len(features) == 0:
                    warnings.warn(f"Group {group_name} has no features")
                    continue

            if self.verbose:
                print(f"  Testing {experiment_name} ({X_ablated.shape[1]} features)...")

            try:
                y_pred = cross_val_predict(clone(base_model), X_ablated.values, y, cv=cv, n_jobs=-1)
                metrics = compute_snr_metrics(y, y_pred)

                # Compute deltas
                delta_r2 = metrics['r2'] - baseline_metrics['r2']
                delta_snr = metrics['snr'] - baseline_metrics['snr']

                results[experiment_name] = {
                    **metrics,
                    'delta_r2': delta_r2,
                    'delta_snr': delta_snr,
                    'n_features': X_ablated.shape[1]
                }

                if self.verbose:
                    print(f"    R²={metrics['r2']:.4f} (Δ={delta_r2:+.4f}), "
                          f"SNR={metrics['snr']:.4f} (Δ={delta_snr:+.4f})")

            except Exception as e:
                warnings.warn(f"Failed for {experiment_name}: {str(e)}")
                continue

        # Visualize ablation impact
        self._plot_ablation_results(results, mode)

        # Save results
        output_file = self.output_dir / f'feature_ablation_{mode}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n  Results saved to {output_file}")

        return results

    def incremental_feature_addition(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        feature_groups_ordered: List[Tuple[str, List[str]]],
        base_model: Optional[BaseEstimator] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Add feature groups incrementally to see SNR growth curve.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix
        y : np.ndarray
            Target values
        feature_groups_ordered : list of tuples
            [(group_name, [features]), ...] in order of addition
        base_model : estimator, optional
            Model to use

        Returns
        -------
        dict
            {stage_name: {r2, snr, cumulative_features}}
        """
        if self.verbose:
            print("\n" + "="*80)
            print("INCREMENTAL FEATURE ADDITION")
            print("="*80)

        if base_model is None:
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.random_state, n_jobs=-1
            )

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        results = {}
        cumulative_features = []

        for i, (group_name, features) in enumerate(feature_groups_ordered):
            cumulative_features.extend(features)
            stage_name = f'stage_{i+1}_{group_name}'

            X_current = X[cumulative_features]

            if self.verbose:
                print(f"  Stage {i+1}: Adding {group_name} ({len(features)} features, "
                      f"total={len(cumulative_features)})...")

            try:
                y_pred = cross_val_predict(clone(base_model), X_current.values, y, cv=cv, n_jobs=-1)
                metrics = compute_snr_metrics(y, y_pred)

                results[stage_name] = {
                    **metrics,
                    'n_features': len(cumulative_features),
                    'added_group': group_name,
                    'added_count': len(features)
                }

                if self.verbose:
                    print(f"    R²={metrics['r2']:.4f}, SNR={metrics['snr']:.4f}")

            except Exception as e:
                warnings.warn(f"Failed for {stage_name}: {str(e)}")
                continue

        # Plot SNR growth curve
        self._plot_incremental_snr(results)

        # Save results
        output_file = self.output_dir / 'incremental_features.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        if self.verbose:
            print(f"\n  Results saved to {output_file}")

        return results

    def synthetic_signal_injection(
        self,
        X: np.ndarray,
        y: np.ndarray,
        signal_strength: float = 0.3,
        signal_type: str = 'linear',
        base_model: Optional[BaseEstimator] = None
    ) -> Dict[str, Any]:
        """
        Inject synthetic signal to verify pipeline can detect it.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        signal_strength : float, default=0.3
            Strength of injected signal (as fraction of y std)
        signal_type : str, default='linear'
            'linear', 'quadratic', or 'interaction'
        base_model : estimator, optional
            Model to test

        Returns
        -------
        dict
            Detection results and sensitivity metrics
        """
        if self.verbose:
            print("\n" + "="*80)
            print("SYNTHETIC SIGNAL INJECTION")
            print("="*80)
            print(f"  Signal type: {signal_type}, Strength: {signal_strength}")

        if base_model is None:
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.random_state, n_jobs=-1
            )

        np.random.seed(self.random_state)
        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Baseline: original data
        y_pred_baseline = cross_val_predict(clone(base_model), X, y, cv=cv, n_jobs=-1)
        baseline_metrics = compute_snr_metrics(y, y_pred_baseline)

        # Create synthetic signal
        n_samples = X.shape[0]
        y_std = np.std(y)

        if signal_type == 'linear':
            # Linear combination of first 3 features
            synthetic = (X[:, 0] + X[:, 1] + X[:, 2]) / 3
        elif signal_type == 'quadratic':
            # Quadratic of first feature
            synthetic = X[:, 0] ** 2
        elif signal_type == 'interaction':
            # Interaction of first two features
            synthetic = X[:, 0] * X[:, 1]
        else:
            raise ValueError(f"Unknown signal type: {signal_type}")

        # Normalize and scale
        synthetic = (synthetic - np.mean(synthetic)) / np.std(synthetic)
        synthetic = synthetic * signal_strength * y_std

        # Inject signal
        y_injected = y + synthetic

        # Test with injected signal
        y_pred_injected = cross_val_predict(clone(base_model), X, y_injected, cv=cv, n_jobs=-1)
        injected_metrics = compute_snr_metrics(y_injected, y_pred_injected)

        # Detection metrics
        snr_improvement = injected_metrics['snr'] - baseline_metrics['snr']
        r2_improvement = injected_metrics['r2'] - baseline_metrics['r2']

        detected = snr_improvement > 0.05  # Threshold for detection

        results = {
            'signal_type': signal_type,
            'signal_strength': signal_strength,
            'baseline_snr': baseline_metrics['snr'],
            'injected_snr': injected_metrics['snr'],
            'snr_improvement': snr_improvement,
            'baseline_r2': baseline_metrics['r2'],
            'injected_r2': injected_metrics['r2'],
            'r2_improvement': r2_improvement,
            'detected': detected,
            'sensitivity': snr_improvement / signal_strength if signal_strength > 0 else 0
        }

        if self.verbose:
            print(f"\n  Baseline SNR: {baseline_metrics['snr']:.4f}")
            print(f"  Injected SNR: {injected_metrics['snr']:.4f}")
            print(f"  Improvement: {snr_improvement:+.4f}")
            print(f"  Detection: {'✓ DETECTED' if detected else '✗ NOT DETECTED'}")

        # Save results
        output_file = self.output_dir / 'synthetic_injection.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def residual_modeling(
        self,
        X: np.ndarray,
        y: np.ndarray,
        primary_model: Optional[BaseEstimator] = None,
        residual_model: Optional[BaseEstimator] = None
    ) -> Dict[str, float]:
        """
        Train model on residuals to check for missed signal.

        Workflow:
        1. Fit primary model A → get residuals r
        2. Fit residual model B on (X → r)
        3. If B reduces residual variance, primary model missed signal

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        primary_model : estimator, optional
            Initial model
        residual_model : estimator, optional
            Model to fit on residuals

        Returns
        -------
        dict
            Metrics about residual predictability
        """
        if self.verbose:
            print("\n" + "="*80)
            print("RESIDUAL MODELING")
            print("="*80)

        if primary_model is None:
            primary_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.random_state, n_jobs=-1
            )

        if residual_model is None:
            residual_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=5, random_state=self.random_state
            )

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Get primary model predictions and residuals
        if self.verbose:
            print("  Fitting primary model...")
        y_pred_primary = cross_val_predict(clone(primary_model), X, y, cv=cv, n_jobs=-1)
        residuals = y - y_pred_primary
        primary_metrics = compute_snr_metrics(y, y_pred_primary)

        # Try to predict residuals
        if self.verbose:
            print("  Fitting residual model...")
        residual_pred = cross_val_predict(clone(residual_model), X, residuals, cv=cv, n_jobs=-1)
        residual_metrics = compute_snr_metrics(residuals, residual_pred)

        # If residual model has positive R², there's missed signal
        missed_signal = residual_metrics['r2'] > 0.05

        # Variance decomposition
        total_var = np.var(y)
        explained_var = np.var(y_pred_primary)
        residual_var = np.var(residuals)
        residual_explainable_var = residual_metrics['r2'] * residual_var if residual_metrics['r2'] > 0 else 0

        results = {
            'primary_model_r2': primary_metrics['r2'],
            'primary_model_snr': primary_metrics['snr'],
            'residual_model_r2': residual_metrics['r2'],
            'residual_model_snr': residual_metrics['snr'],
            'missed_signal_detected': missed_signal,
            'total_variance': total_var,
            'explained_variance': explained_var,
            'residual_variance': residual_var,
            'residual_explainable_variance': residual_explainable_var,
            'potential_r2_gain': residual_explainable_var / total_var
        }

        if self.verbose:
            print(f"\n  Primary model R²: {primary_metrics['r2']:.4f}")
            print(f"  Residual model R²: {residual_metrics['r2']:.4f}")
            print(f"  Missed signal: {'✓ YES' if missed_signal else '✗ NO'}")
            print(f"  Potential R² gain: {results['potential_r2_gain']:.4f}")

        # Save results
        output_file = self.output_dir / 'residual_modeling.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def heteroscedastic_analysis(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model: Optional[BaseEstimator] = None
    ) -> Dict[str, float]:
        """
        Test if residual magnitude is predictable (heteroscedasticity).

        Predicts log(residual²) to measure explainable error variance.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix
        y : np.ndarray
            Target values
        base_model : estimator, optional
            Model for prediction

        Returns
        -------
        dict
            Heteroscedasticity metrics
        """
        if self.verbose:
            print("\n" + "="*80)
            print("HETEROSCEDASTIC ANALYSIS")
            print("="*80)

        if base_model is None:
            base_model = RandomForestRegressor(
                n_estimators=100, max_depth=8, random_state=self.random_state, n_jobs=-1
            )

        cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Get predictions and residuals
        y_pred = cross_val_predict(clone(base_model), X, y, cv=cv, n_jobs=-1)
        residuals = y - y_pred

        # Predict log(residual²)
        log_squared_residuals = np.log(residuals**2 + 1e-8)  # Add small constant to avoid log(0)

        # Fit model to predict error magnitude
        error_predictor = GradientBoostingRegressor(
            n_estimators=100, max_depth=5, random_state=self.random_state
        )

        log_res_pred = cross_val_predict(clone(error_predictor), X, log_squared_residuals, cv=cv, n_jobs=-1)
        hetero_metrics = compute_snr_metrics(log_squared_residuals, log_res_pred)

        # Compute fraction of residual variance that is predictable
        heteroscedastic = hetero_metrics['r2'] > 0.1

        results = {
            'log_squared_residual_r2': hetero_metrics['r2'],
            'log_squared_residual_snr': hetero_metrics['snr'],
            'heteroscedastic': heteroscedastic,
            'residual_variance_explainable_fraction': max(0, hetero_metrics['r2'])
        }

        if self.verbose:
            print(f"\n  Log(residual²) R²: {hetero_metrics['r2']:.4f}")
            print(f"  Heteroscedastic: {'✓ YES' if heteroscedastic else '✗ NO'}")
            print(f"  Explainable error variance: {results['residual_variance_explainable_fraction']:.1%}")

        # Save results
        output_file = self.output_dir / 'heteroscedastic_analysis.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        return results

    def run_all_experiments(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        feature_groups: Optional[Dict[str, List[str]]] = None,
        task: str = 'regression'
    ) -> AttributionResults:
        """
        Run all Phase 2 experiments.

        Parameters
        ----------
        X : array or DataFrame
            Feature matrix
        y : np.ndarray
            Target values
        feature_groups : dict, optional
            Feature groupings for ablation
        task : str, default='regression'
            'regression' or 'classification'

        Returns
        -------
        AttributionResults
            Complete results from all experiments
        """
        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 2: SIGNAL-VS-NOISE ATTRIBUTION")
            print("Running all experiments...")
            print("="*80)

        # Convert to DataFrame if needed for ablation
        if not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        else:
            X_df = X

        X_array = X_df.values if isinstance(X, pd.DataFrame) else X

        # 1. Model family sweep
        model_results = self.model_family_sweep(X_array, y, task=task)

        # 2. Feature ablation (if groups provided)
        ablation_results = {}
        if feature_groups is not None:
            ablation_results = self.feature_ablation(X_df, y, feature_groups, mode='drop_group')

        # 3. Synthetic signal injection
        synthetic_results = self.synthetic_signal_injection(X_array, y, signal_strength=0.3)

        # 4. Residual modeling
        residual_results = self.residual_modeling(X_array, y)

        # 5. Heteroscedastic analysis
        hetero_results = self.heteroscedastic_analysis(X_array, y)

        # Compile results
        results = AttributionResults(
            model_family_results=model_results,
            ablation_results=ablation_results,
            synthetic_injection_results=synthetic_results,
            residual_model_results=residual_results,
            heteroscedastic_results=hetero_results
        )

        # Generate summary report
        self._generate_phase2_report(results)

        if self.verbose:
            print("\n" + "="*80)
            print("PHASE 2 COMPLETE")
            print(f"Results saved to: {self.output_dir}")
            print("="*80)

        return results

    # Visualization methods

    def _plot_model_family_comparison(self, results: Dict[str, Dict[str, float]]):
        """Create bar chart comparing model families."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        models = list(results.keys())
        snr_values = [results[m]['snr'] for m in models]
        r2_values = [results[m]['r2'] for m in models]

        # SNR comparison
        bars = ax1.barh(models, snr_values, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('SNR', fontsize=12)
        ax1.set_title('Model Family SNR Comparison', fontsize=14, fontweight='bold')
        ax1.axvline(x=1.0, color='red', linestyle='--', label='SNR=1.0')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')

        # R² comparison
        bars2 = ax2.barh(models, r2_values, alpha=0.7, edgecolor='black', color='green')
        ax2.set_xlabel('R²', fontsize=12)
        ax2.set_title('Model Family R² Comparison', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_family_comparison.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_ablation_results(self, results: Dict[str, Dict[str, float]], mode: str):
        """Visualize feature ablation impact."""
        # Extract delta metrics
        experiments = [k for k in results.keys() if k != 'baseline_all_features']
        delta_r2 = [results[k]['delta_r2'] for k in experiments]
        delta_snr = [results[k]['delta_snr'] for k in experiments]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Delta R²
        colors = ['red' if d < 0 else 'green' for d in delta_r2]
        ax1.barh(experiments, delta_r2, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Δ R²', fontsize=12)
        ax1.set_title(f'Feature Ablation Impact ({mode})', fontsize=14, fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax1.grid(True, alpha=0.3, axis='x')

        # Delta SNR
        colors2 = ['red' if d < 0 else 'green' for d in delta_snr]
        ax2.barh(experiments, delta_snr, color=colors2, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Δ SNR', fontsize=12)
        ax2.set_title(f'SNR Change ({mode})', fontsize=14, fontweight='bold')
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax2.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(self.output_dir / f'ablation_{mode}.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _plot_incremental_snr(self, results: Dict[str, Dict[str, float]]):
        """Plot SNR growth as features are added."""
        stages = list(results.keys())
        snr_values = [results[s]['snr'] for s in stages]
        r2_values = [results[s]['r2'] for s in stages]
        n_features = [results[s]['n_features'] for s in stages]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # SNR growth
        ax1.plot(range(len(stages)), snr_values, 'o-', linewidth=2, markersize=8)
        ax1.set_ylabel('SNR', fontsize=12)
        ax1.set_title('SNR Growth with Feature Addition', fontsize=14, fontweight='bold')
        ax1.axhline(y=1.0, color='red', linestyle='--', label='SNR=1.0')
        ax1.set_xticks(range(len(stages)))
        ax1.set_xticklabels(stages, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # R² growth
        ax2.plot(range(len(stages)), r2_values, 'o-', linewidth=2, markersize=8, color='green')
        ax2.set_xlabel('Stage', fontsize=12)
        ax2.set_ylabel('R²', fontsize=12)
        ax2.set_title('R² Growth with Feature Addition', fontsize=14, fontweight='bold')
        ax2.set_xticks(range(len(stages)))
        ax2.set_xticklabels(stages, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)

        # Add feature counts as text
        for i, (stage, n_feat) in enumerate(zip(stages, n_features)):
            ax1.text(i, snr_values[i], f'n={n_feat}', ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'incremental_features.png', dpi=150, bbox_inches='tight')
        plt.close()

    def _generate_phase2_report(self, results: AttributionResults):
        """Generate comprehensive Phase 2 report."""
        report_path = self.output_dir / 'phase2_attribution_report.md'

        with open(report_path, 'w') as f:
            f.write("# Phase 2: Signal-vs-Noise Attribution Report\n\n")
            f.write(f"**Generated**: {pd.Timestamp.now()}\n\n")
            f.write("---\n\n")

            # Model family results
            f.write("## 1. Model Family Comparison\n\n")
            f.write("Tests whether low SNR is due to model choice.\n\n")

            if results.model_family_results:
                best_model = max(results.model_family_results.items(), key=lambda x: x[1]['snr'])
                worst_model = min(results.model_family_results.items(), key=lambda x: x[1]['snr'])

                f.write(f"- **Best Model**: {best_model[0]} (SNR={best_model[1]['snr']:.4f})\n")
                f.write(f"- **Worst Model**: {worst_model[0]} (SNR={worst_model[1]['snr']:.4f})\n")
                f.write(f"- **SNR Range**: {best_model[1]['snr'] - worst_model[1]['snr']:.4f}\n\n")

                if best_model[1]['snr'] - worst_model[1]['snr'] > 0.5:
                    f.write("**Interpretation**: Large SNR variation across models → Model choice matters significantly\n\n")
                else:
                    f.write("**Interpretation**: Similar SNR across models → Problem may be feature-limited or inherently noisy\n\n")

            # Ablation results
            if results.ablation_results:
                f.write("## 2. Feature Ablation Analysis\n\n")
                f.write("Tests which feature groups contribute most to signal.\n\n")

                # Find most impactful group (largest negative delta when dropped)
                ablations = {k: v for k, v in results.ablation_results.items() if k.startswith('drop_')}
                if ablations:
                    most_impact = min(ablations.items(), key=lambda x: x[1]['delta_snr'])
                    f.write(f"- **Most Important Group**: {most_impact[0].replace('drop_', '')} "
                           f"(Δ SNR={most_impact[1]['delta_snr']:.4f})\n\n")

            # Synthetic injection
            f.write("## 3. Synthetic Signal Injection\n\n")
            f.write("Verifies pipeline can detect known signals.\n\n")

            synth = results.synthetic_injection_results
            f.write(f"- **Signal Detected**: {'✓ YES' if synth['detected'] else '✗ NO'}\n")
            f.write(f"- **SNR Improvement**: {synth['snr_improvement']:+.4f}\n")
            f.write(f"- **Sensitivity**: {synth['sensitivity']:.4f}\n\n")

            # Residual modeling
            f.write("## 4. Residual Modeling\n\n")
            f.write("Tests if primary model missed signal.\n\n")

            resid = results.residual_model_results
            f.write(f"- **Missed Signal**: {'✓ YES' if resid['missed_signal_detected'] else '✗ NO'}\n")
            f.write(f"- **Potential R² Gain**: {resid['potential_r2_gain']:.4f}\n\n")

            # Heteroscedastic
            f.write("## 5. Heteroscedastic Analysis\n\n")
            f.write("Tests if prediction errors are predictable.\n\n")

            hetero = results.heteroscedastic_results
            f.write(f"- **Heteroscedastic**: {'✓ YES' if hetero['heteroscedastic'] else '✗ NO'}\n")
            f.write(f"- **Explainable Error Variance**: {hetero['residual_variance_explainable_fraction']:.1%}\n\n")

            # Overall conclusions
            f.write("## Overall Conclusions\n\n")
            f.write("**Sources of Low Predictability:**\n\n")

            if resid['missed_signal_detected']:
                f.write("- ⚠️ Primary model is missing signal (try more complex models)\n")
            if hetero['heteroscedastic']:
                f.write("- ⚠️ Errors are predictable (consider heteroscedastic models)\n")
            if synth['detected']:
                f.write("- ✓ Pipeline is capable of detecting signals (features may be the issue)\n")
            else:
                f.write("- ❌ Pipeline struggling to detect even synthetic signals (check implementation)\n")

        if self.verbose:
            print(f"\n  Phase 2 report saved to {report_path}")
