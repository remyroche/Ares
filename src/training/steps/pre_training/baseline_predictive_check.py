"""
Baseline Predictive Check Module

This module provides baseline predictive checks using Linear Regression and LightGBM
to assess feature quality and predictive power before model training.

Features:
- Linear Regression baseline with residual analysis
- LightGBM with cross-validation and hyperparameter tuning
- Feature importance analysis
- Overfitting detection
- Comprehensive metrics and interpretations
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from pathlib import Path

logger = logging.getLogger(__name__)


class BaselinePredictiveCheck:
    """
    Performs baseline predictive checks using Linear Regression and LightGBM.

    This class provides a standardized way to:
    1. Train simple Linear Regression model
    2. Analyze residuals and estimate noise
    3. Train LightGBM with cross-validation
    4. Compare models and detect overfitting
    5. Extract feature importances
    """

    def __init__(
        self,
        max_features: Optional[int] = None,
        random_state: int = 42,
        n_cv_folds: int = 5,
        test_size: float = 0.2
    ):
        """
        Initialize the baseline predictive check.

        Args:
            max_features: Maximum number of features to use (for sampling)
            random_state: Random state for reproducibility
            n_cv_folds: Number of cross-validation folds
            test_size: Test set size for train/test split
        """
        self.max_features = max_features
        self.random_state = random_state
        self.n_cv_folds = n_cv_folds
        self.test_size = test_size

        # Models
        self.lr_model: Optional[LinearRegression] = None
        self.lgbm_model: Optional[lgb.LGBMRegressor] = None
        self.scaler: Optional[StandardScaler] = None

        # Results
        self.results: Dict[str, Any] = {}

    def run_check(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Run the complete baseline predictive check.

        Args:
            features: Feature dataframe
            target: Target series
            feature_names: Optional list of feature names (if not in dataframe)

        Returns:
            Dict containing all results and metrics
        """
        try:
            logger.info(f"Starting baseline predictive check with {len(features.columns)} features")

            # Prepare data
            X, y, selected_features = self._prepare_data(features, target)

            if X is None or y is None:
                return {
                    'success': False,
                    'error': 'Failed to prepare data',
                    'timestamp': datetime.now().isoformat()
                }

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )

            # Run Linear Regression check
            logger.info("Running Linear Regression baseline...")
            lr_results = self._run_linear_regression(X_train, X_test, y_train, y_test, selected_features)

            # Run LightGBM check
            logger.info("Running LightGBM baseline...")
            lgbm_results = self._run_lightgbm(X_train, X_test, y_train, y_test, X, y, selected_features)

            # Compare models and generate interpretation
            logger.info("Comparing models and generating interpretation...")
            comparison = self._compare_models(lr_results, lgbm_results)
            interpretation = self._generate_interpretation(lr_results, lgbm_results, comparison)

            # Compile results
            self.results = {
                'success': True,
                'timestamp': datetime.now().isoformat(),
                'data_info': {
                    'n_samples': len(X),
                    'n_features': X.shape[1],
                    'train_size': len(X_train),
                    'test_size': len(X_test),
                    'selected_features': selected_features,
                    'target_stats': {
                        'mean': float(y.mean()),
                        'std': float(y.std()),
                        'min': float(y.min()),
                        'max': float(y.max())
                    }
                },
                'linear_regression': lr_results,
                'lightgbm': lgbm_results,
                'comparison': comparison,
                'interpretation': interpretation
            }

            logger.info("Baseline predictive check completed successfully")
            return self.results

        except Exception as e:
            logger.error(f"Baseline predictive check failed: {e}", exc_info=True)
            return {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        """Prepare data for modeling."""
        try:
            # Align features and target
            common_idx = features.index.intersection(target.index)

            if len(common_idx) == 0:
                logger.error("No common index between features and target")
                return None, None, []

            X = features.loc[common_idx]
            y = target.loc[common_idx]

            # Remove rows with NaN in target
            valid_mask = y.notna()
            X = X[valid_mask]
            y = y[valid_mask]

            # Select numeric features only
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            X = X[numeric_cols]

            # Remove features with too many NaNs (>50%)
            nan_threshold = 0.5
            good_features = X.columns[X.isnull().mean() < nan_threshold].tolist()
            X = X[good_features]

            # Fill remaining NaNs with column means
            X = X.fillna(X.mean())

            # Remove constant features
            non_constant = X.std() > 1e-10
            X = X.loc[:, non_constant]

            # Sample features if needed
            selected_features = X.columns.tolist()
            if self.max_features and len(selected_features) > self.max_features:
                np.random.seed(self.random_state)
                selected_features = np.random.choice(
                    selected_features,
                    size=self.max_features,
                    replace=False
                ).tolist()
                X = X[selected_features]

            logger.info(f"Data prepared: {len(X)} samples, {len(selected_features)} features")
            return X, y, selected_features

        except Exception as e:
            logger.error(f"Failed to prepare data: {e}")
            return None, None, []

    def _run_linear_regression(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Run Linear Regression analysis."""
        try:
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model
            self.lr_model = LinearRegression()
            self.lr_model.fit(X_train_scaled, y_train)

            # Predictions
            y_train_pred = self.lr_model.predict(X_train_scaled)
            y_test_pred = self.lr_model.predict(X_test_scaled)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Residual analysis
            train_residuals = y_train - y_train_pred
            test_residuals = y_test - y_test_pred

            residual_stats = {
                'train_mean': float(train_residuals.mean()),
                'train_std': float(train_residuals.std()),
                'test_mean': float(test_residuals.mean()),
                'test_std': float(test_residuals.std()),
                'train_noise_estimate': float(train_residuals.std()),
                'test_noise_estimate': float(test_residuals.std())
            }

            # Feature coefficients analysis
            coefficients = pd.Series(
                self.lr_model.coef_,
                index=feature_names
            ).sort_values(key=abs, ascending=False)

            top_coefficients = {
                str(k): float(v)
                for k, v in coefficients.head(10).items()
            }

            # Signal detection
            significant_features = sum(abs(coefficients) > 0.1)
            signal_strength = float(abs(coefficients).mean())

            return {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_rmse': float(np.sqrt(train_mse)),
                'test_rmse': float(np.sqrt(test_mse)),
                'residual_analysis': residual_stats,
                'top_coefficients': top_coefficients,
                'significant_features': int(significant_features),
                'signal_strength': signal_strength,
                'overfitting_gap': float(train_r2 - test_r2)
            }

        except Exception as e:
            logger.error(f"Linear Regression failed: {e}")
            return {'error': str(e)}

    def _run_lightgbm(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        X_full: pd.DataFrame,
        y_full: pd.Series,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Run LightGBM analysis with cross-validation."""
        try:
            # Create LightGBM datasets
            train_data = lgb.Dataset(X_train, label=y_train)

            # Define parameters for tuning
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'learning_rate': 0.05,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.random_state
            }

            # Cross-validation
            cv_results = lgb.cv(
                params,
                train_data,
                num_boost_round=100,
                nfold=self.n_cv_folds,
                stratified=False,
                shuffle=False,
                metrics=['rmse', 'l2'],
                return_cvbooster=True
            )

            # Train final model
            self.lgbm_model = lgb.train(
                params,
                train_data,
                num_boost_round=len(cv_results['valid rmse-mean'])
            )

            # Predictions
            y_train_pred = self.lgbm_model.predict(X_train)
            y_test_pred = self.lgbm_model.predict(X_test)

            # Metrics
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            # Feature importance
            importance = pd.Series(
                self.lgbm_model.feature_importance(importance_type='gain'),
                index=feature_names
            ).sort_values(ascending=False)

            top_importance = {
                str(k): float(v)
                for k, v in importance.head(10).items()
            }

            # CV statistics
            cv_mean = float(cv_results['valid rmse-mean'][-1])
            cv_std = float(cv_results['valid rmse-stdv'][-1])

            return {
                'train_r2': float(train_r2),
                'test_r2': float(test_r2),
                'train_mse': float(train_mse),
                'test_mse': float(test_mse),
                'train_mae': float(train_mae),
                'test_mae': float(test_mae),
                'train_rmse': float(np.sqrt(train_mse)),
                'test_rmse': float(np.sqrt(test_mse)),
                'cv_rmse_mean': cv_mean,
                'cv_rmse_std': cv_std,
                'top_feature_importance': top_importance,
                'overfitting_gap': float(train_r2 - test_r2),
                'n_trees': len(cv_results['valid rmse-mean'])
            }

        except Exception as e:
            logger.error(f"LightGBM failed: {e}")
            return {'error': str(e)}

    def _compare_models(
        self,
        lr_results: Dict[str, Any],
        lgbm_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare Linear Regression and LightGBM results."""
        try:
            if 'error' in lr_results or 'error' in lgbm_results:
                return {'error': 'One or both models failed'}

            lr_test_r2 = lr_results['test_r2']
            lgbm_test_r2 = lgbm_results['test_r2']

            improvement = lgbm_test_r2 - lr_test_r2
            improvement_pct = (improvement / max(abs(lr_test_r2), 1e-10)) * 100

            lr_mse = lr_results['test_mse']
            lgbm_mse = lgbm_results['test_mse']
            mse_improvement = ((lr_mse - lgbm_mse) / max(lr_mse, 1e-10)) * 100

            return {
                'lr_test_r2': float(lr_test_r2),
                'lgbm_test_r2': float(lgbm_test_r2),
                'r2_improvement': float(improvement),
                'r2_improvement_pct': float(improvement_pct),
                'lr_test_mse': float(lr_mse),
                'lgbm_test_mse': float(lgbm_mse),
                'mse_improvement_pct': float(mse_improvement),
                'lgbm_better': lgbm_test_r2 > lr_test_r2,
                'both_poor': lr_test_r2 < 0.1 and lgbm_test_r2 < 0.1
            }

        except Exception as e:
            logger.error(f"Model comparison failed: {e}")
            return {'error': str(e)}

    def _generate_interpretation(
        self,
        lr_results: Dict[str, Any],
        lgbm_results: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate human-readable interpretation of results."""
        try:
            if 'error' in comparison:
                return {'interpretation': 'Analysis failed', 'recommendation': 'Check model errors'}

            interpretations = []
            recommendations = []
            quality_score = 0.0

            # Interpret Linear Regression performance
            lr_r2 = lr_results['test_r2']
            if lr_r2 > 0.5:
                interpretations.append("Linear Regression shows strong performance, indicating linear relationships in data")
                quality_score += 2
            elif lr_r2 > 0.2:
                interpretations.append("Linear Regression shows moderate performance, some linear signal present")
                quality_score += 1
            else:
                interpretations.append("Linear Regression shows poor performance, limited linear relationships")

            # Interpret LightGBM performance
            lgbm_r2 = lgbm_results['test_r2']
            if lgbm_r2 > 0.5:
                interpretations.append("LightGBM shows strong performance")
                quality_score += 2
            elif lgbm_r2 > 0.2:
                interpretations.append("LightGBM shows moderate performance")
                quality_score += 1
            else:
                interpretations.append("LightGBM shows poor performance")

            # Interpret comparison
            if comparison['lgbm_better'] and comparison['r2_improvement'] > 0.1:
                interpretations.append("LightGBM significantly outperforms Linear Regression → nonlinear relationships and feature interactions exist")
                recommendations.append("Consider using tree-based models or neural networks")
                quality_score += 1
            elif comparison['lgbm_better']:
                interpretations.append("LightGBM slightly outperforms Linear Regression → some nonlinearity present")
            else:
                interpretations.append("Linear Regression matches or exceeds LightGBM → relationships are primarily linear")
                recommendations.append("Linear models may be sufficient")

            # Check for overfitting
            lr_overfit = lr_results.get('overfitting_gap', 0)
            lgbm_overfit = lgbm_results.get('overfitting_gap', 0)

            if lgbm_overfit > 0.2:
                interpretations.append("LightGBM shows signs of overfitting (train-test gap > 0.2)")
                recommendations.append("Consider regularization or feature selection")

            # Overall assessment
            if comparison['both_poor']:
                interpretations.append("⚠️ Both models perform poorly → features or target are weak/noisy")
                recommendations.append("Consider going back to data diagnostics (target/features)")
                recommendations.append("Examine target variable definition and feature engineering")

            # Noise analysis
            noise_estimate = lr_results.get('residual_analysis', {}).get('test_noise_estimate', 0)
            target_std = abs(lr_results.get('test_mse', 1)) ** 0.5
            if noise_estimate > target_std * 0.8:
                interpretations.append(f"High noise level detected (noise ≈ {noise_estimate:.3f})")
                recommendations.append("Consider noise reduction or robust modeling techniques")

            # Normalize quality score to 0-1 range
            quality_score = min(quality_score / 5.0, 1.0)

            return {
                'interpretations': interpretations,
                'recommendations': recommendations,
                'quality_score': float(quality_score),
                'summary': self._generate_summary(comparison, quality_score)
            }

        except Exception as e:
            logger.error(f"Interpretation generation failed: {e}")
            return {
                'interpretations': ['Error generating interpretation'],
                'recommendations': ['Check logs for details'],
                'quality_score': 0.0,
                'summary': 'Analysis incomplete'
            }

    def _generate_summary(self, comparison: Dict[str, Any], quality_score: float) -> str:
        """Generate a one-line summary of the analysis."""
        if comparison.get('both_poor', False):
            return "⚠️ Poor predictive power - revisit features and target"
        elif comparison.get('lgbm_better', False) and comparison.get('r2_improvement', 0) > 0.1:
            return "✓ Strong nonlinear relationships detected - use advanced models"
        elif quality_score > 0.6:
            return "✓ Good predictive potential - proceed with confidence"
        else:
            return "⚠️ Moderate predictive power - consider feature engineering"

    def save_results_to_csv(self, output_dir: Path, filename_prefix: str = "baseline_check") -> str:
        """
        Save detailed results to CSV file.

        Args:
            output_dir: Output directory path
            filename_prefix: Prefix for the filename

        Returns:
            Path to saved CSV file
        """
        try:
            if not self.results or not self.results.get('success', False):
                logger.error("No valid results to save")
                return ""

            # Create output directory
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{filename_prefix}_{timestamp}.csv"
            filepath = output_dir / filename

            # Flatten results for CSV
            rows = []

            # Data info
            data_info = self.results.get('data_info', {})
            rows.append({
                'metric_category': 'data_info',
                'metric_name': 'n_samples',
                'value': data_info.get('n_samples', 0)
            })
            rows.append({
                'metric_category': 'data_info',
                'metric_name': 'n_features',
                'value': data_info.get('n_features', 0)
            })

            # Linear Regression metrics
            lr_results = self.results.get('linear_regression', {})
            for key, value in lr_results.items():
                if isinstance(value, (int, float)):
                    rows.append({
                        'metric_category': 'linear_regression',
                        'metric_name': key,
                        'value': value
                    })

            # LightGBM metrics
            lgbm_results = self.results.get('lightgbm', {})
            for key, value in lgbm_results.items():
                if isinstance(value, (int, float)):
                    rows.append({
                        'metric_category': 'lightgbm',
                        'metric_name': key,
                        'value': value
                    })

            # Comparison metrics
            comparison = self.results.get('comparison', {})
            for key, value in comparison.items():
                if isinstance(value, (int, float, bool)):
                    rows.append({
                        'metric_category': 'comparison',
                        'metric_name': key,
                        'value': value
                    })

            # Interpretation
            interpretation = self.results.get('interpretation', {})
            rows.append({
                'metric_category': 'interpretation',
                'metric_name': 'quality_score',
                'value': interpretation.get('quality_score', 0)
            })
            rows.append({
                'metric_category': 'interpretation',
                'metric_name': 'summary',
                'value': interpretation.get('summary', '')
            })

            # Create DataFrame and save
            df = pd.DataFrame(rows)
            df['timestamp'] = self.results.get('timestamp', datetime.now().isoformat())
            df.to_csv(filepath, index=False)

            logger.info(f"Results saved to {filepath}")
            return str(filepath)

        except Exception as e:
            logger.error(f"Failed to save results to CSV: {e}")
            return ""

    def format_for_markdown(self) -> str:
        """
        Format results for markdown report.

        Returns:
            Markdown-formatted string
        """
        try:
            if not self.results or not self.results.get('success', False):
                return "## Baseline Predictive Check\n\n❌ Check failed or not run\n"

            md = "## Baseline Predictive Check\n\n"

            # Overview
            data_info = self.results.get('data_info', {})
            md += f"**Dataset**: {data_info.get('n_samples', 0)} samples, {data_info.get('n_features', 0)} features\n\n"

            # Linear Regression section
            lr_results = self.results.get('linear_regression', {})
            if 'error' not in lr_results:
                md += "### Linear Regression Baseline\n\n"
                md += f"- **Test R²**: {lr_results.get('test_r2', 0):.4f}\n"
                md += f"- **Test RMSE**: {lr_results.get('test_rmse', 0):.4f}\n"
                md += f"- **Noise Estimate**: {lr_results.get('residual_analysis', {}).get('test_noise_estimate', 0):.4f}\n"
                md += f"- **Significant Features**: {lr_results.get('significant_features', 0)}\n"
                md += f"- **Signal Strength**: {lr_results.get('signal_strength', 0):.4f}\n\n"

                md += "**Top Coefficients**:\n"
                for feat, coef in list(lr_results.get('top_coefficients', {}).items())[:5]:
                    md += f"- {feat}: {coef:.4f}\n"
                md += "\n"

            # LightGBM section
            lgbm_results = self.results.get('lightgbm', {})
            if 'error' not in lgbm_results:
                md += "### LightGBM Baseline\n\n"
                md += f"- **Test R²**: {lgbm_results.get('test_r2', 0):.4f}\n"
                md += f"- **Test RMSE**: {lgbm_results.get('test_rmse', 0):.4f}\n"
                md += f"- **CV RMSE**: {lgbm_results.get('cv_rmse_mean', 0):.4f} ± {lgbm_results.get('cv_rmse_std', 0):.4f}\n"
                md += f"- **Overfitting Gap**: {lgbm_results.get('overfitting_gap', 0):.4f}\n\n"

                md += "**Top Feature Importance**:\n"
                for feat, imp in list(lgbm_results.get('top_feature_importance', {}).items())[:5]:
                    md += f"- {feat}: {imp:.1f}\n"
                md += "\n"

            # Comparison section
            comparison = self.results.get('comparison', {})
            if 'error' not in comparison:
                md += "### Model Comparison\n\n"
                md += f"- **LR Test R²**: {comparison.get('lr_test_r2', 0):.4f}\n"
                md += f"- **LGBM Test R²**: {comparison.get('lgbm_test_r2', 0):.4f}\n"
                md += f"- **Improvement**: {comparison.get('r2_improvement', 0):.4f} ({comparison.get('r2_improvement_pct', 0):.1f}%)\n\n"

            # Interpretation section
            interpretation = self.results.get('interpretation', {})
            md += "### Interpretation\n\n"
            md += f"**Quality Score**: {interpretation.get('quality_score', 0):.2f}/1.0\n\n"
            md += f"**Summary**: {interpretation.get('summary', 'N/A')}\n\n"

            md += "**Key Findings**:\n"
            for interp in interpretation.get('interpretations', [])[:5]:
                md += f"- {interp}\n"
            md += "\n"

            if interpretation.get('recommendations'):
                md += "**Recommendations**:\n"
                for rec in interpretation.get('recommendations', []):
                    md += f"- {rec}\n"
                md += "\n"

            return md

        except Exception as e:
            logger.error(f"Failed to format markdown: {e}")
            return "## Baseline Predictive Check\n\n❌ Error formatting results\n"


def run_baseline_check(
    features: pd.DataFrame,
    target: pd.Series,
    max_features: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_csv: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run baseline predictive check.

    Args:
        features: Feature dataframe
        target: Target series
        max_features: Maximum number of features to sample
        output_dir: Output directory for CSV (if save_csv=True)
        save_csv: Whether to save detailed results to CSV

    Returns:
        Dict containing all results
    """
    checker = BaselinePredictiveCheck(max_features=max_features)
    results = checker.run_check(features, target)

    if save_csv and output_dir and results.get('success', False):
        checker.save_results_to_csv(output_dir)

    return results
