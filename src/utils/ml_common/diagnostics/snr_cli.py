"""
CLI for SNR Diagnostics - Independent Analysis Commands

This module provides command-line interfaces for running diagnostic analyses
on trained models, features, and labels.

Commands:
- label-quality: Noise ceiling + aleatoric uncertainty
- label-learnability: R², SNR, permutation, naive baselines
- model-robustness: Bootstrap CI, residual analysis, model comparison

All commands use latest artifacts and save timestamped reports to outcomes/
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils.ml_common.diagnostics import (
    NoiseCeilingAnalysis,
    UncertaintyDecomposition,
    SNRDiagnostics,
    SignalAttributionExperiments,
    TemporalDiagnostics,
)
from src.utils.tprint import tprint


class DiagnosticsCLI:
    """
    CLI for running SNR diagnostics on latest trained models.
    """

    def __init__(self):
        """Initialize CLI."""
        self.project_root = Path(__file__).parent.parent.parent.parent.parent
        self.artifacts_dir = self.project_root / 'artifacts'
        self.outcomes_dir = self.project_root / 'outcomes'
        self.outcomes_dir.mkdir(exist_ok=True)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def load_latest_artifacts(
        self,
        symbol: str,
        timeframe: str,
        model_type: str = 'analyst'
    ) -> Tuple[Any, pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Load latest trained model, features, and labels.

        Parameters
        ----------
        symbol : str
            Trading symbol (e.g., 'BTCUSDT')
        timeframe : str
            Timeframe (e.g., '15m')
        model_type : str
            'analyst' or 'tactician'

        Returns
        -------
        tuple
            (model, X_df, y, y_pred)
        """
        tprint(f"Loading latest artifacts for {symbol}/{timeframe}...", "INFO")

        # Find latest labeled data
        labeled_pattern = f"labeled_data_{symbol}_{timeframe}_*.parquet"
        labeled_files = sorted(self.artifacts_dir.glob(labeled_pattern))

        if not labeled_files:
            raise FileNotFoundError(f"No labeled data found for {symbol}/{timeframe}")

        latest_labeled = labeled_files[-1]
        tprint(f"  Found labeled data: {latest_labeled.name}", "INFO")

        # Load labeled data
        df = pd.read_parquet(latest_labeled)

        # Extract labels
        if 'meta_label' in df.columns:
            y = df['meta_label'].values
        elif 'label' in df.columns:
            y = df['label'].values
        elif 'target' in df.columns:
            y = df['target'].values
        else:
            raise ValueError("No label column found in data")

        # Extract features (exclude label columns)
        exclude_cols = [
            'meta_label', 'label', 'target', 'timestamp', 'close', 'open', 'high', 'low', 'volume',
            'realized_return', 'smoothed_label', 'binary_label', 'exit_reason'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        X_df = df[feature_cols].copy()

        # Handle NaNs
        X_df = X_df.fillna(X_df.median())

        tprint(f"  Loaded {len(df)} samples, {len(feature_cols)} features", "INFO")

        # Try to load trained model
        model = None
        y_pred = None

        model_pattern = f"{model_type}_model_{symbol}_{timeframe}_*.pkl"
        model_files = sorted(self.artifacts_dir.glob(model_pattern))

        if model_files:
            latest_model_file = model_files[-1]
            tprint(f"  Found model: {latest_model_file.name}", "INFO")

            try:
                import pickle
                with open(latest_model_file, 'rb') as f:
                    model = pickle.load(f)

                # Generate predictions
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_df.values)[:, 1]
                else:
                    y_pred = model.predict(X_df.values)

                tprint(f"  Generated predictions from trained model", "SUCCESS")

            except Exception as e:
                tprint(f"  Warning: Could not load model: {e}", "WARNING")

        # If no predictions, train a simple model for diagnostics
        if y_pred is None:
            tprint("  Training simple model for diagnostics...", "INFO")
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import cross_val_predict

            model = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)

            # Remove NaN labels
            valid_mask = ~pd.isna(y)
            X_valid = X_df[valid_mask].values
            y_valid = y[valid_mask]

            # Cross-validated predictions to avoid overfitting
            y_pred_valid = cross_val_predict(model, X_valid, y_valid, cv=5, n_jobs=-1)

            # Create full prediction array
            y_pred = np.full(len(y), np.nan)
            y_pred[valid_mask] = y_pred_valid

            tprint(f"  Generated CV predictions", "SUCCESS")

        return model, X_df, y, y_pred

    def run_label_quality(self, symbol: str, timeframe: str, **kwargs):
        """
        Run label quality diagnostics.

        Analyzes:
        1. Noise ceiling (if multiple labelers available)
        2. Aleatoric uncertainty fraction

        Parameters
        ----------
        symbol : str
            Trading symbol
        timeframe : str
            Timeframe
        """
        tprint("\n" + "="*80, "INFO")
        tprint("LABEL QUALITY DIAGNOSTICS", "INFO")
        tprint("="*80 + "\n", "INFO")

        # Load data
        model, X_df, y, y_pred = self.load_latest_artifacts(symbol, timeframe)

        # Create output directory
        output_dir = self.outcomes_dir / f'label_quality_{symbol}_{timeframe}_{self.timestamp}'
        output_dir.mkdir(exist_ok=True)

        results = {}

        # 1. Noise Ceiling Analysis (if replicates available)
        tprint("\n1. Checking for noise ceiling data...", "INFO")

        # Check if we have multiple annotations/replicates
        # This would require loading multiple label versions
        # For now, we'll estimate from ensemble variance

        # Create pseudo-replicates using bootstrap ensemble
        tprint("  Creating pseudo-replicates via bootstrap ensemble...", "INFO")

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.base import clone

        valid_mask = ~pd.isna(y)
        X_valid = X_df[valid_mask].values
        y_valid = y[valid_mask]

        n_replicates = 5
        pseudo_labels = []

        np.random.seed(42)
        for i in range(n_replicates):
            # Bootstrap sample
            idx = np.random.choice(len(X_valid), size=len(X_valid), replace=True)
            X_boot = X_valid[idx]
            y_boot = y_valid[idx]

            # Train model
            boot_model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42+i, n_jobs=-1)
            boot_model.fit(X_boot, y_boot)

            # Predict on full valid set
            pseudo_label = boot_model.predict(X_valid)
            pseudo_labels.append(pseudo_label)

        # Stack as (n_samples, n_replicates)
        ratings = np.column_stack(pseudo_labels)

        # Run noise ceiling analysis
        noise_ceiling = NoiseCeilingAnalysis(output_dir=output_dir, verbose=True)

        # Compute R² of actual model
        from sklearn.metrics import r2_score
        model_r2 = r2_score(y_valid, y_pred[valid_mask]) if y_pred is not None else None

        ceiling_results = noise_ceiling.compute_noise_ceiling(ratings, model_r2=model_r2)

        results['noise_ceiling'] = {
            'icc_one_way': ceiling_results.icc_one_way,
            'icc_two_way': ceiling_results.icc_two_way,
            'krippendorff_alpha': ceiling_results.krippendorff_alpha,
            'expected_max_r2': ceiling_results.expected_max_r2,
            'label_variance_ratio': ceiling_results.label_variance_ratio,
            'model_r2': model_r2,
            'ceiling_exceeded': model_r2 > ceiling_results.expected_max_r2 + 0.05 if model_r2 else False
        }

        # 2. Aleatoric Uncertainty
        tprint("\n2. Computing aleatoric uncertainty...", "INFO")

        unc_decomp = UncertaintyDecomposition(output_dir=output_dir, verbose=True)

        unc_results = unc_decomp.decompose_uncertainty(
            X_valid, y_valid,
            base_model=RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42),
            n_ensemble=10,
            cv_folds=5
        )

        results['aleatoric_uncertainty'] = {
            'total_uncertainty': unc_results.total_uncertainty,
            'aleatoric_uncertainty': unc_results.aleatoric_uncertainty,
            'epistemic_uncertainty': unc_results.epistemic_uncertainty,
            'aleatoric_fraction': unc_results.aleatoric_fraction,
            'epistemic_fraction': unc_results.epistemic_fraction,
            'calibration_score': unc_results.calibration_score
        }

        # Generate summary report
        self._generate_label_quality_report(results, output_dir, symbol, timeframe)

        tprint(f"\n✅ Label quality diagnostics complete!", "SUCCESS")
        tprint(f"   Reports saved to: {output_dir}", "SUCCESS")

        return results

    def run_label_learnability(self, symbol: str, timeframe: str, **kwargs):
        """
        Run label learnability diagnostics.

        Analyzes:
        1. R²
        2. SNR
        3. Permutation p-value
        4. Naive baselines

        Parameters
        ----------
        symbol : str
            Trading symbol
        timeframe : str
            Timeframe
        """
        tprint("\n" + "="*80, "INFO")
        tprint("LABEL LEARNABILITY DIAGNOSTICS", "INFO")
        tprint("="*80 + "\n", "INFO")

        # Load data
        model, X_df, y, y_pred = self.load_latest_artifacts(symbol, timeframe)

        # Create output directory
        output_dir = self.outcomes_dir / f'label_learnability_{symbol}_{timeframe}_{self.timestamp}'
        output_dir.mkdir(exist_ok=True)

        valid_mask = ~pd.isna(y) & ~pd.isna(y_pred)
        X_valid = X_df[valid_mask].values
        y_valid = y[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        results = {}

        # 1. Core SNR metrics
        tprint("\n1. Computing core SNR metrics...", "INFO")

        from src.utils.ml_common.diagnostics import compute_snr_metrics, bootstrap_r2, permutation_test

        core_metrics = compute_snr_metrics(y_valid, y_pred_valid)

        tprint(f"   R² = {core_metrics['r2']:.4f}", "INFO")
        tprint(f"   SNR = {core_metrics['snr']:.4f}", "INFO")

        # Bootstrap CI
        ci_lower, ci_upper = bootstrap_r2(y_valid, y_pred_valid, n_iterations=1000, random_state=42)

        tprint(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]", "INFO")

        # Permutation test
        pvalue = permutation_test(y_valid, y_pred_valid, n_permutations=1000, random_state=42)

        tprint(f"   Permutation p-value = {pvalue:.4f}", "INFO")

        results['core_metrics'] = {
            'r2': core_metrics['r2'],
            'snr': core_metrics['snr'],
            'rmse': core_metrics['rmse'],
            'nrmse': core_metrics['nrmse'],
            'bootstrap_ci_lower': ci_lower,
            'bootstrap_ci_upper': ci_upper,
            'permutation_pvalue': pvalue
        }

        # 2. Naive Baselines
        tprint("\n2. Computing naive baselines...", "INFO")

        baselines = self._compute_naive_baselines(X_valid, y_valid)

        for baseline_name, baseline_metrics in baselines.items():
            tprint(f"   {baseline_name}: R²={baseline_metrics['r2']:.4f}, SNR={baseline_metrics['snr']:.4f}", "INFO")

        results['naive_baselines'] = baselines

        # 3. Comparison with baselines
        tprint("\n3. Comparing model with baselines...", "INFO")

        best_baseline = max(baselines.items(), key=lambda x: x[1]['r2'])
        improvement = core_metrics['r2'] - best_baseline[1]['r2']

        tprint(f"   Best baseline: {best_baseline[0]} (R²={best_baseline[1]['r2']:.4f})", "INFO")
        tprint(f"   Model improvement: {improvement:+.4f}", "INFO")

        if improvement < 0.05:
            tprint(f"   ⚠️  Model barely beats baselines - may be fitting noise", "WARNING")
        elif improvement > 0.2:
            tprint(f"   ✅ Model significantly outperforms baselines - real signal detected", "SUCCESS")

        results['baseline_comparison'] = {
            'best_baseline_name': best_baseline[0],
            'best_baseline_r2': best_baseline[1]['r2'],
            'model_improvement': improvement,
            'signal_detected': improvement > 0.05
        }

        # Generate summary report
        self._generate_learnability_report(results, output_dir, symbol, timeframe)

        tprint(f"\n✅ Label learnability diagnostics complete!", "SUCCESS")
        tprint(f"   Reports saved to: {output_dir}", "SUCCESS")

        return results

    def run_model_robustness(self, symbol: str, timeframe: str, **kwargs):
        """
        Run model & features robustness diagnostics.

        Analyzes:
        1. Bootstrap R² confidence interval
        2. Residual structure
        3. Residual autocorrelation
        4. Model family comparison

        Parameters
        ----------
        symbol : str
            Trading symbol
        timeframe : str
            Timeframe
        """
        tprint("\n" + "="*80, "INFO")
        tprint("MODEL & FEATURES ROBUSTNESS DIAGNOSTICS", "INFO")
        tprint("="*80 + "\n", "INFO")

        # Load data
        model, X_df, y, y_pred = self.load_latest_artifacts(symbol, timeframe)

        # Create output directory
        output_dir = self.outcomes_dir / f'model_robustness_{symbol}_{timeframe}_{self.timestamp}'
        output_dir.mkdir(exist_ok=True)

        valid_mask = ~pd.isna(y) & ~pd.isna(y_pred)
        X_valid = X_df[valid_mask].values
        y_valid = y[valid_mask]
        y_pred_valid = y_pred[valid_mask]

        results = {}

        # 1. Bootstrap CI (detailed)
        tprint("\n1. Computing bootstrap confidence intervals...", "INFO")

        from src.utils.ml_common.diagnostics import bootstrap_r2

        ci_lower, ci_upper = bootstrap_r2(y_valid, y_pred_valid, n_iterations=2000, random_state=42)
        ci_width = ci_upper - ci_lower

        tprint(f"   95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]", "INFO")
        tprint(f"   CI Width: {ci_width:.4f}", "INFO")

        if ci_width > 0.2:
            tprint(f"   ⚠️  Wide CI suggests unstable performance", "WARNING")
        elif ci_lower < 0:
            tprint(f"   ⚠️  CI includes negative values - performance may be unreliable", "WARNING")
        else:
            tprint(f"   ✅ Stable performance (narrow CI, positive)", "SUCCESS")

        results['bootstrap_ci'] = {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'ci_width': ci_width,
            'includes_zero': ci_lower < 0,
            'stable': ci_width < 0.2 and ci_lower > 0
        }

        # 2. Residual Structure Analysis
        tprint("\n2. Analyzing residual structure...", "INFO")

        residuals = y_valid - y_pred_valid

        # Check for patterns in residuals
        from scipy.stats import shapiro, anderson

        # Normality test
        shapiro_stat, shapiro_p = shapiro(residuals[:min(5000, len(residuals))])  # Shapiro max 5000

        # Anderson-Darling test
        anderson_result = anderson(residuals)

        tprint(f"   Shapiro-Wilk p-value: {shapiro_p:.4f}", "INFO")
        tprint(f"   Anderson-Darling stat: {anderson_result.statistic:.4f}", "INFO")

        # Test if residuals have structure by fitting a simple model
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score

        residual_predictor = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42)
        residual_r2_scores = cross_val_score(
            residual_predictor, X_valid, residuals, cv=5, scoring='r2', n_jobs=-1
        )
        residual_r2 = np.mean(residual_r2_scores)

        tprint(f"   Residual R² (predictability): {residual_r2:.4f}", "INFO")

        if residual_r2 > 0.1:
            tprint(f"   ⚠️  Residuals show structure - model missing patterns", "WARNING")
        else:
            tprint(f"   ✅ Residuals appear random - model captured available signal", "SUCCESS")

        results['residual_structure'] = {
            'shapiro_pvalue': shapiro_p,
            'anderson_darling_stat': anderson_result.statistic,
            'residual_r2': residual_r2,
            'has_structure': residual_r2 > 0.1,
            'approximately_normal': shapiro_p > 0.05
        }

        # 3. Residual Autocorrelation
        tprint("\n3. Computing residual autocorrelation...", "INFO")

        temporal_diag = TemporalDiagnostics(output_dir=output_dir, verbose=False)

        # Check if we have timestamps
        try:
            labeled_pattern = f"labeled_data_{symbol}_{timeframe}_*.parquet"
            labeled_files = sorted(self.artifacts_dir.glob(labeled_pattern))
            df_full = pd.read_parquet(labeled_files[-1])

            if 'timestamp' in df_full.columns:
                timestamps = pd.to_datetime(df_full[valid_mask]['timestamp'])
            else:
                timestamps = None
        except:
            timestamps = None

        temporal_results = temporal_diag.analyze_temporal_structure(
            residuals,
            timestamps=timestamps,
            max_lags=40
        )

        tprint(f"   Durbin-Watson: {temporal_results.durbin_watson_stat:.4f}", "INFO")
        tprint(f"   ACF(1): {temporal_results.autocorrelation[1]:.4f}", "INFO")

        if abs(temporal_results.autocorrelation[1]) > 0.2:
            tprint(f"   ⚠️  High autocorrelation - missing temporal features", "WARNING")
        elif temporal_results.durbin_watson_stat < 1.5:
            tprint(f"   ⚠️  Positive autocorrelation detected", "WARNING")
        else:
            tprint(f"   ✅ Residuals appear independent", "SUCCESS")

        results['residual_autocorrelation'] = {
            'durbin_watson': temporal_results.durbin_watson_stat,
            'acf_lag1': temporal_results.autocorrelation[1],
            'has_autocorrelation': abs(temporal_results.autocorrelation[1]) > 0.2 or temporal_results.durbin_watson_stat < 1.5
        }

        # 4. Model Family Comparison
        tprint("\n4. Running model family comparison...", "INFO")

        attrib = SignalAttributionExperiments(output_dir=output_dir, verbose=False)

        model_family_results = attrib.model_family_sweep(X_valid, y_valid, task='regression')

        # Find best and current model performance
        best_model = max(model_family_results.items(), key=lambda x: x[1]['snr'])
        current_r2 = results.get('bootstrap_ci', {}).get('ci_lower', 0)  # Use CI lower as conservative estimate

        from sklearn.metrics import r2_score
        actual_r2 = r2_score(y_valid, y_pred_valid)

        tprint(f"   Best model family: {best_model[0]} (SNR={best_model[1]['snr']:.4f})", "INFO")
        tprint(f"   Current model R²: {actual_r2:.4f}", "INFO")

        if best_model[1]['r2'] > actual_r2 + 0.1:
            tprint(f"   ⚠️  Other model families perform significantly better", "WARNING")
        else:
            tprint(f"   ✅ Current model competitive with best families", "SUCCESS")

        results['model_family_comparison'] = {
            'best_family_name': best_model[0],
            'best_family_r2': best_model[1]['r2'],
            'best_family_snr': best_model[1]['snr'],
            'current_model_r2': actual_r2,
            'improvement_possible': best_model[1]['r2'] > actual_r2 + 0.1,
            'all_families': model_family_results
        }

        # Generate summary report
        self._generate_robustness_report(results, output_dir, symbol, timeframe)

        tprint(f"\n✅ Model robustness diagnostics complete!", "SUCCESS")
        tprint(f"   Reports saved to: {output_dir}", "SUCCESS")

        return results

    def _compute_naive_baselines(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Compute naive baseline predictions."""
        from sklearn.model_selection import cross_val_predict
        from sklearn.dummy import DummyRegressor
        from sklearn.linear_model import Ridge
        from src.utils.ml_common.diagnostics import compute_snr_metrics

        baselines = {}

        # 1. Mean baseline
        mean_model = DummyRegressor(strategy='mean')
        y_pred_mean = cross_val_predict(mean_model, X, y, cv=5)
        baselines['mean'] = compute_snr_metrics(y, y_pred_mean)

        # 2. Median baseline
        median_model = DummyRegressor(strategy='median')
        y_pred_median = cross_val_predict(median_model, X, y, cv=5)
        baselines['median'] = compute_snr_metrics(y, y_pred_median)

        # 3. Simple linear (Ridge with high regularization)
        simple_linear = Ridge(alpha=10.0)
        y_pred_linear = cross_val_predict(simple_linear, X, y, cv=5)
        baselines['simple_linear'] = compute_snr_metrics(y, y_pred_linear)

        # 4. First feature only
        if X.shape[1] > 0:
            first_feature_model = Ridge(alpha=1.0)
            y_pred_first = cross_val_predict(first_feature_model, X[:, :1], y, cv=5)
            baselines['first_feature_only'] = compute_snr_metrics(y, y_pred_first)

        return baselines

    def _generate_label_quality_report(self, results: Dict, output_dir: Path, symbol: str, timeframe: str):
        """Generate label quality report."""
        # CSV
        csv_data = []

        # Noise ceiling
        nc = results.get('noise_ceiling', {})
        csv_data.append({
            'category': 'Noise Ceiling',
            'metric': 'ICC (One-Way)',
            'value': nc.get('icc_one_way', np.nan),
            'interpretation': self._interpret_icc(nc.get('icc_one_way', 0))
        })
        csv_data.append({
            'category': 'Noise Ceiling',
            'metric': 'ICC (Two-Way)',
            'value': nc.get('icc_two_way', np.nan),
            'interpretation': self._interpret_icc(nc.get('icc_two_way', 0))
        })
        csv_data.append({
            'category': 'Noise Ceiling',
            'metric': 'Expected Max R²',
            'value': nc.get('expected_max_r2', np.nan),
            'interpretation': 'Theoretical performance ceiling'
        })
        csv_data.append({
            'category': 'Noise Ceiling',
            'metric': 'Ceiling Exceeded',
            'value': nc.get('ceiling_exceeded', False),
            'interpretation': 'WARNING: Check for data leakage' if nc.get('ceiling_exceeded') else 'OK'
        })

        # Aleatoric uncertainty
        au = results.get('aleatoric_uncertainty', {})
        csv_data.append({
            'category': 'Aleatoric Uncertainty',
            'metric': 'Aleatoric Fraction',
            'value': au.get('aleatoric_fraction', np.nan),
            'interpretation': self._interpret_aleatoric(au.get('aleatoric_fraction', 0))
        })
        csv_data.append({
            'category': 'Aleatoric Uncertainty',
            'metric': 'Epistemic Fraction',
            'value': au.get('epistemic_fraction', np.nan),
            'interpretation': self._interpret_epistemic(au.get('epistemic_fraction', 0))
        })
        csv_data.append({
            'category': 'Aleatoric Uncertainty',
            'metric': 'Calibration Score',
            'value': au.get('calibration_score', np.nan),
            'interpretation': self._interpret_calibration(au.get('calibration_score', 0))
        })

        df = pd.DataFrame(csv_data)
        csv_path = output_dir / f'label_quality_report_{symbol}_{timeframe}_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)

        # Markdown
        md_path = output_dir / f'label_quality_report_{symbol}_{timeframe}_{self.timestamp}.md'
        with open(md_path, 'w') as f:
            f.write(f"# Label Quality Diagnostics Report\n\n")
            f.write(f"**Symbol**: {symbol}\n")
            f.write(f"**Timeframe**: {timeframe}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Noise Ceiling Analysis\n\n")
            f.write(f"- **ICC (Two-Way)**: {nc.get('icc_two_way', 'N/A'):.4f}\n")
            f.write(f"- **Expected Max R²**: {nc.get('expected_max_r2', 'N/A'):.4f}\n")
            f.write(f"- **Model R²**: {nc.get('model_r2', 'N/A'):.4f}\n")
            f.write(f"- **Ceiling Exceeded**: {'⚠️ YES' if nc.get('ceiling_exceeded') else '✓ NO'}\n\n")

            if nc.get('ceiling_exceeded'):
                f.write("### ⚠️ WARNING\n\n")
                f.write("Model performance exceeds the noise ceiling. This may indicate:\n")
                f.write("- Data leakage (features contain future information)\n")
                f.write("- Incorrect cross-validation\n")
                f.write("- Overfitting\n\n")

            f.write("## Aleatoric Uncertainty\n\n")
            f.write(f"- **Aleatoric (Irreducible)**: {au.get('aleatoric_fraction', 'N/A'):.1%}\n")
            f.write(f"- **Epistemic (Reducible)**: {au.get('epistemic_fraction', 'N/A'):.1%}\n")
            f.write(f"- **Calibration Score**: {au.get('calibration_score', 'N/A'):.4f}\n\n")

            if au.get('aleatoric_fraction', 0) > 0.6:
                f.write("### Interpretation\n\n")
                f.write("Aleatoric uncertainty dominates (>60%). This indicates:\n")
                f.write("- Most unpredictability is due to inherent label noise\n")
                f.write("- Improving the model will have limited impact\n")
                f.write("- Focus on improving label quality or redefining the task\n\n")

        tprint(f"  Reports saved: {csv_path.name}, {md_path.name}", "SUCCESS")

    def _generate_learnability_report(self, results: Dict, output_dir: Path, symbol: str, timeframe: str):
        """Generate learnability report."""
        # CSV
        csv_data = []

        cm = results.get('core_metrics', {})
        csv_data.append({
            'category': 'Core Metrics',
            'metric': 'R²',
            'value': cm.get('r2', np.nan),
            'interpretation': self._interpret_r2(cm.get('r2', 0))
        })
        csv_data.append({
            'category': 'Core Metrics',
            'metric': 'SNR',
            'value': cm.get('snr', np.nan),
            'interpretation': self._interpret_snr(cm.get('snr', 0))
        })
        csv_data.append({
            'category': 'Core Metrics',
            'metric': 'Permutation p-value',
            'value': cm.get('permutation_pvalue', np.nan),
            'interpretation': self._interpret_pvalue(cm.get('permutation_pvalue', 1))
        })

        # Baselines
        for name, metrics in results.get('naive_baselines', {}).items():
            csv_data.append({
                'category': 'Naive Baselines',
                'metric': f'{name} R²',
                'value': metrics['r2'],
                'interpretation': f"SNR={metrics['snr']:.4f}"
            })

        bc = results.get('baseline_comparison', {})
        csv_data.append({
            'category': 'Baseline Comparison',
            'metric': 'Model Improvement',
            'value': bc.get('model_improvement', np.nan),
            'interpretation': 'Real signal' if bc.get('signal_detected') else 'Weak signal'
        })

        df = pd.DataFrame(csv_data)
        csv_path = output_dir / f'label_learnability_report_{symbol}_{timeframe}_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)

        # Markdown
        md_path = output_dir / f'label_learnability_report_{symbol}_{timeframe}_{self.timestamp}.md'
        with open(md_path, 'w') as f:
            f.write(f"# Label Learnability Diagnostics Report\n\n")
            f.write(f"**Symbol**: {symbol}\n")
            f.write(f"**Timeframe**: {timeframe}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Core Metrics\n\n")
            f.write(f"- **R²**: {cm.get('r2', 'N/A'):.4f} ({self._interpret_r2(cm.get('r2', 0))})\n")
            f.write(f"- **SNR**: {cm.get('snr', 'N/A'):.4f} ({self._interpret_snr(cm.get('snr', 0))})\n")
            f.write(f"- **Permutation p-value**: {cm.get('permutation_pvalue', 'N/A'):.4f}\n")
            f.write(f"- **Bootstrap 95% CI**: [{cm.get('bootstrap_ci_lower', 'N/A'):.4f}, {cm.get('bootstrap_ci_upper', 'N/A'):.4f}]\n\n")

            f.write("## Naive Baselines\n\n")
            f.write("| Baseline | R² | SNR |\n")
            f.write("|----------|----|----|\ n")
            for name, metrics in results.get('naive_baselines', {}).items():
                f.write(f"| {name} | {metrics['r2']:.4f} | {metrics['snr']:.4f} |\n")
            f.write("\n")

            f.write("## Baseline Comparison\n\n")
            f.write(f"- **Best Baseline**: {bc.get('best_baseline_name', 'N/A')} (R²={bc.get('best_baseline_r2', 0):.4f})\n")
            f.write(f"- **Model Improvement**: {bc.get('model_improvement', 0):+.4f}\n")
            f.write(f"- **Signal Detected**: {'✓ YES' if bc.get('signal_detected') else '✗ NO'}\n\n")

            if not bc.get('signal_detected'):
                f.write("### ⚠️ Warning\n\n")
                f.write("Model barely outperforms naive baselines. This suggests:\n")
                f.write("- Weak learnable signal\n")
                f.write("- Model may be fitting noise\n")
                f.write("- Features may not contain relevant information\n\n")

        tprint(f"  Reports saved: {csv_path.name}, {md_path.name}", "SUCCESS")

    def _generate_robustness_report(self, results: Dict, output_dir: Path, symbol: str, timeframe: str):
        """Generate robustness report."""
        # CSV
        csv_data = []

        bc = results.get('bootstrap_ci', {})
        csv_data.append({
            'category': 'Bootstrap CI',
            'metric': 'CI Lower',
            'value': bc.get('ci_lower', np.nan),
            'interpretation': 'Includes zero' if bc.get('includes_zero') else 'Positive'
        })
        csv_data.append({
            'category': 'Bootstrap CI',
            'metric': 'CI Upper',
            'value': bc.get('ci_upper', np.nan),
            'interpretation': f"Width={bc.get('ci_width', 0):.4f}"
        })
        csv_data.append({
            'category': 'Bootstrap CI',
            'metric': 'Stable',
            'value': bc.get('stable', False),
            'interpretation': 'Stable performance' if bc.get('stable') else 'Unstable'
        })

        rs = results.get('residual_structure', {})
        csv_data.append({
            'category': 'Residual Structure',
            'metric': 'Residual R²',
            'value': rs.get('residual_r2', np.nan),
            'interpretation': 'Has structure' if rs.get('has_structure') else 'Random'
        })
        csv_data.append({
            'category': 'Residual Structure',
            'metric': 'Shapiro p-value',
            'value': rs.get('shapiro_pvalue', np.nan),
            'interpretation': 'Normal' if rs.get('approximately_normal') else 'Non-normal'
        })

        ra = results.get('residual_autocorrelation', {})
        csv_data.append({
            'category': 'Residual Autocorrelation',
            'metric': 'Durbin-Watson',
            'value': ra.get('durbin_watson', np.nan),
            'interpretation': self._interpret_dw(ra.get('durbin_watson', 2))
        })
        csv_data.append({
            'category': 'Residual Autocorrelation',
            'metric': 'ACF(1)',
            'value': ra.get('acf_lag1', np.nan),
            'interpretation': 'High' if ra.get('has_autocorrelation') else 'Low'
        })

        mf = results.get('model_family_comparison', {})
        csv_data.append({
            'category': 'Model Family',
            'metric': 'Best Family',
            'value': mf.get('best_family_name', 'N/A'),
            'interpretation': f"R²={mf.get('best_family_r2', 0):.4f}"
        })
        csv_data.append({
            'category': 'Model Family',
            'metric': 'Improvement Possible',
            'value': mf.get('improvement_possible', False),
            'interpretation': 'Try other models' if mf.get('improvement_possible') else 'Current model OK'
        })

        df = pd.DataFrame(csv_data)
        csv_path = output_dir / f'model_robustness_report_{symbol}_{timeframe}_{self.timestamp}.csv'
        df.to_csv(csv_path, index=False)

        # Markdown
        md_path = output_dir / f'model_robustness_report_{symbol}_{timeframe}_{self.timestamp}.md'
        with open(md_path, 'w') as f:
            f.write(f"# Model & Features Robustness Report\n\n")
            f.write(f"**Symbol**: {symbol}\n")
            f.write(f"**Timeframe**: {timeframe}\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write("---\n\n")

            f.write("## Bootstrap Confidence Interval\n\n")
            f.write(f"- **95% CI**: [{bc.get('ci_lower', 'N/A'):.4f}, {bc.get('ci_upper', 'N/A'):.4f}]\n")
            f.write(f"- **CI Width**: {bc.get('ci_width', 'N/A'):.4f}\n")
            f.write(f"- **Stable**: {'✓ YES' if bc.get('stable') else '✗ NO'}\n\n")

            f.write("## Residual Analysis\n\n")
            f.write(f"- **Residual R²**: {rs.get('residual_r2', 'N/A'):.4f}\n")
            f.write(f"- **Has Structure**: {'⚠️ YES' if rs.get('has_structure') else '✓ NO'}\n")
            f.write(f"- **Approximately Normal**: {'✓ YES' if rs.get('approximately_normal') else '✗ NO'}\n\n")

            if rs.get('has_structure'):
                f.write("### ⚠️ Warning: Residuals Show Structure\n\n")
                f.write("Residuals are predictable (R²>0.1). This indicates:\n")
                f.write("- Model is missing patterns in the data\n")
                f.write("- Try more complex models\n")
                f.write("- Add engineered features\n\n")

            f.write("## Residual Autocorrelation\n\n")
            f.write(f"- **Durbin-Watson**: {ra.get('durbin_watson', 'N/A'):.4f}\n")
            f.write(f"- **ACF(1)**: {ra.get('acf_lag1', 'N/A'):.4f}\n")
            f.write(f"- **Has Autocorrelation**: {'⚠️ YES' if ra.get('has_autocorrelation') else '✓ NO'}\n\n")

            if ra.get('has_autocorrelation'):
                f.write("### Recommendation\n\n")
                f.write("Add temporal features:\n")
                f.write("- Lagged values\n")
                f.write("- Rolling statistics\n")
                f.write("- Autoregressive terms\n\n")

            f.write("## Model Family Comparison\n\n")
            f.write(f"- **Best Family**: {mf.get('best_family_name', 'N/A')} (R²={mf.get('best_family_r2', 0):.4f}, SNR={mf.get('best_family_snr', 0):.4f})\n")
            f.write(f"- **Current Model**: R²={mf.get('current_model_r2', 0):.4f}\n")
            f.write(f"- **Improvement Possible**: {'✓ YES' if mf.get('improvement_possible') else '✗ NO'}\n\n")

        tprint(f"  Reports saved: {csv_path.name}, {md_path.name}", "SUCCESS")

    # Interpretation helpers
    def _interpret_r2(self, r2: float) -> str:
        if r2 > 0.40:
            return "Strong signal"
        elif r2 > 0.10:
            return "Moderate signal"
        else:
            return "Weak signal"

    def _interpret_snr(self, snr: float) -> str:
        if snr > 1.0:
            return "Signal > noise"
        elif snr > 0.3:
            return "Weak signal"
        else:
            return "Noise dominates"

    def _interpret_pvalue(self, p: float) -> str:
        if p < 0.01:
            return "Highly significant"
        elif p < 0.05:
            return "Significant"
        else:
            return "Not significant"

    def _interpret_icc(self, icc: float) -> str:
        if icc > 0.75:
            return "Excellent"
        elif icc > 0.60:
            return "Good"
        elif icc > 0.40:
            return "Fair"
        else:
            return "Poor"

    def _interpret_aleatoric(self, frac: float) -> str:
        if frac > 0.6:
            return "Noise-limited"
        elif frac > 0.4:
            return "Mixed"
        else:
            return "Model-limited"

    def _interpret_epistemic(self, frac: float) -> str:
        if frac > 0.6:
            return "Improvement possible"
        elif frac > 0.4:
            return "Mixed"
        else:
            return "Limited improvement"

    def _interpret_calibration(self, score: float) -> str:
        if score > 0.7:
            return "Well calibrated"
        elif score > 0.4:
            return "Moderately calibrated"
        else:
            return "Poorly calibrated"

    def _interpret_dw(self, dw: float) -> str:
        if dw < 1.5:
            return "Positive autocorrelation"
        elif dw > 2.5:
            return "Negative autocorrelation"
        else:
            return "Independent"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="SNR Diagnostics CLI - Run independent diagnostic analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run label quality diagnostics
  python snr_cli.py label-quality --symbol BTCUSDT --timeframe 15m

  # Run label learnability diagnostics
  python snr_cli.py label-learnability --symbol BTCUSDT --timeframe 15m

  # Run model robustness diagnostics
  python snr_cli.py model-robustness --symbol BTCUSDT --timeframe 15m

All commands use the latest trained model, configs, and features.
Reports are saved to outcomes/ with timestamps.
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Diagnostic command')

    # Label quality command
    lq_parser = subparsers.add_parser(
        'label-quality',
        help='Analyze label quality (noise ceiling + aleatoric uncertainty)'
    )
    lq_parser.add_argument('--symbol', required=True, help='Trading symbol (e.g., BTCUSDT)')
    lq_parser.add_argument('--timeframe', required=True, help='Timeframe (e.g., 15m)')

    # Label learnability command
    ll_parser = subparsers.add_parser(
        'label-learnability',
        help='Analyze label learnability (R², SNR, permutation, baselines)'
    )
    ll_parser.add_argument('--symbol', required=True, help='Trading symbol')
    ll_parser.add_argument('--timeframe', required=True, help='Timeframe')

    # Model robustness command
    mr_parser = subparsers.add_parser(
        'model-robustness',
        help='Analyze model robustness (bootstrap CI, residuals, model families)'
    )
    mr_parser.add_argument('--symbol', required=True, help='Trading symbol')
    mr_parser.add_argument('--timeframe', required=True, help='Timeframe')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    cli = DiagnosticsCLI()

    try:
        if args.command == 'label-quality':
            cli.run_label_quality(args.symbol, args.timeframe)
        elif args.command == 'label-learnability':
            cli.run_label_learnability(args.symbol, args.timeframe)
        elif args.command == 'model-robustness':
            cli.run_model_robustness(args.symbol, args.timeframe)
    except Exception as e:
        tprint(f"\n❌ Error: {str(e)}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
