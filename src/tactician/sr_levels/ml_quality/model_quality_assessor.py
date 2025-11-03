"""
Model Quality Assessor

Comprehensive evaluation of SR Quality Model performance.
Detects overfitting, checks calibration, analyzes predictions, and ensures production readiness.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import ks_2samp, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


class ModelQualityAssessor:
    """Comprehensive assessment of model quality and production readiness."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def detect_overfitting(self, train_metrics: Dict, val_metrics: Dict, 
                          fold_scores: List[Dict] = None) -> Dict:
        """Detect overfitting by comparing train vs validation performance.
        
        Args:
            train_metrics: Training set metrics {'rmse', 'r2', 'mae'}
            val_metrics: Validation set metrics
            fold_scores: Optional cross-validation scores
            
        Returns:
            Overfitting analysis report
        """
        self.logger.info("\n🔍 OVERFITTING DETECTION")
        self.logger.info("="*70)
        
        # Calculate gaps
        rmse_gap = train_metrics['rmse'] - val_metrics['rmse']
        r2_gap = train_metrics['r2'] - val_metrics['r2']
        mae_gap = train_metrics['mae'] - val_metrics['mae']
        
        # Relative gaps (percentage)
        rmse_gap_pct = (rmse_gap / val_metrics['rmse']) * 100 if val_metrics['rmse'] > 0 else 0
        r2_gap_pct = (r2_gap / max(abs(val_metrics['r2']), 0.01)) * 100
        
        report = {
            'rmse_gap': float(rmse_gap),
            'r2_gap': float(r2_gap),
            'mae_gap': float(mae_gap),
            'rmse_gap_pct': float(rmse_gap_pct),
            'r2_gap_pct': float(r2_gap_pct),
        }
        
        # Determine overfitting severity
        if rmse_gap > 0.10 or r2_gap > 0.15 or rmse_gap_pct > 30:
            severity = 'severe'
            status = '❌'
            recommendation = 'CRITICAL: Increase regularization, reduce complexity, or get more data'
        elif rmse_gap > 0.05 or r2_gap > 0.08 or rmse_gap_pct > 15:
            severity = 'moderate'
            status = '⚠️'
            recommendation = 'Monitor closely, consider increasing regularization'
        elif rmse_gap > 0.02 or r2_gap > 0.03:
            severity = 'mild'
            status = '🟡'
            recommendation = 'Acceptable - some overfitting is normal'
        else:
            severity = 'none'
            status = '✅'
            recommendation = 'Healthy - no significant overfitting detected'
        
        report['severity'] = severity
        report['status'] = status
        report['recommendation'] = recommendation
        
        # Cross-validation variability (if available)
        if fold_scores:
            val_r2s = [s['val_r2'] for s in fold_scores]
            val_rmses = [s['val_rmse'] for s in fold_scores]
            
            report['cv_r2_std'] = float(np.std(val_r2s))
            report['cv_rmse_std'] = float(np.std(val_rmses))
            report['cv_stable'] = np.std(val_r2s) < 0.05
        
        # Log results
        self.logger.info(f"\n{status} Overfitting Status: {severity.upper()}")
        self.logger.info(f"\n   Train vs Validation Gaps:")
        self.logger.info(f"      RMSE gap: {rmse_gap:+.4f} ({rmse_gap_pct:+.1f}%)")
        self.logger.info(f"      R² gap:   {r2_gap:+.4f} ({r2_gap_pct:+.1f}%)")
        self.logger.info(f"      MAE gap:  {mae_gap:+.4f}")
        
        if fold_scores:
            self.logger.info(f"\n   Cross-Validation Stability:")
            self.logger.info(f"      R² std:   {report['cv_r2_std']:.4f}")
            self.logger.info(f"      RMSE std: {report['cv_rmse_std']:.4f}")
            self.logger.info(f"      Stable:   {report['cv_stable']}")
        
        self.logger.info(f"\n   💡 {recommendation}")
        
        return report
    
    def assess_calibration(self, y_pred: np.ndarray, y_true: np.ndarray, 
                          n_bins: int = 10) -> Dict:
        """Check if predictions are well-calibrated.
        
        When model predicts 0.8, is actual quality really ~0.8?
        
        Args:
            y_pred: Predicted quality scores
            y_true: True quality scores
            n_bins: Number of bins for calibration analysis
            
        Returns:
            Calibration report
        """
        self.logger.info("\n🎯 CALIBRATION ANALYSIS")
        self.logger.info("="*70)
        
        calibration_data = []
        
        for i in range(n_bins):
            bin_min = i / n_bins
            bin_max = (i + 1) / n_bins
            
            # Get predictions in this bin
            in_bin = (y_pred >= bin_min) & (y_pred < bin_max)
            
            if in_bin.sum() > 0:
                predicted_mean = y_pred[in_bin].mean()
                actual_mean = y_true[in_bin].mean()
                calibration_error = abs(predicted_mean - actual_mean)
                
                calibration_data.append({
                    'bin': i,
                    'bin_range': (float(bin_min), float(bin_max)),
                    'count': int(in_bin.sum()),
                    'predicted_mean': float(predicted_mean),
                    'actual_mean': float(actual_mean),
                    'calibration_error': float(calibration_error),
                    'well_calibrated': calibration_error < 0.10
                })
        
        # Mean calibration error
        errors = [d['calibration_error'] for d in calibration_data]
        mce = np.mean(errors) if errors else 0
        
        # Expected Calibration Error (ECE)
        weighted_errors = []
        total_samples = sum(d['count'] for d in calibration_data)
        for d in calibration_data:
            weight = d['count'] / total_samples
            weighted_errors.append(d['calibration_error'] * weight)
        ece = sum(weighted_errors)
        
        report = {
            'calibration_by_bin': calibration_data,
            'mean_calibration_error': float(mce),
            'expected_calibration_error': float(ece),
            'well_calibrated': ece < 0.05,
            'n_bins': n_bins
        }
        
        # Log summary
        self.logger.info(f"\n   Mean Calibration Error: {mce:.4f}")
        self.logger.info(f"   Expected Calibration Error: {ece:.4f}")
        
        if ece < 0.05:
            self.logger.info(f"   ✅ Well calibrated (ECE < 0.05)")
        elif ece < 0.10:
            self.logger.info(f"   🟡 Moderately calibrated (ECE < 0.10)")
        else:
            self.logger.info(f"   ❌ Poorly calibrated (ECE ≥ 0.10)")
        
        # Show worst calibrated bins
        worst_bins = sorted(calibration_data, key=lambda x: x['calibration_error'], reverse=True)[:3]
        self.logger.info(f"\n   Worst calibrated bins:")
        for b in worst_bins:
            self.logger.info(f"      [{b['bin_range'][0]:.1f}-{b['bin_range'][1]:.1f}]: "
                           f"pred={b['predicted_mean']:.3f}, actual={b['actual_mean']:.3f}, "
                           f"error={b['calibration_error']:.3f}")
        
        return report
    
    def analyze_prediction_distribution(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
        """Analyze prediction distribution quality.
        
        Detects if model is collapsing to mean or has good variance.
        
        Args:
            y_pred: Predicted scores
            y_true: True scores
            
        Returns:
            Distribution analysis report
        """
        self.logger.info("\n📊 PREDICTION DISTRIBUTION ANALYSIS")
        self.logger.info("="*70)
        
        # Prediction statistics
        pred_stats = {
            'mean': float(y_pred.mean()),
            'median': float(np.median(y_pred)),
            'std': float(y_pred.std()),
            'min': float(y_pred.min()),
            'max': float(y_pred.max()),
            'range': float(y_pred.max() - y_pred.min())
        }
        
        # True statistics (for comparison)
        true_stats = {
            'mean': float(y_true.mean()),
            'median': float(np.median(y_true)),
            'std': float(y_true.std()),
            'min': float(y_true.min()),
            'max': float(y_true.max()),
            'range': float(y_true.max() - y_true.min())
        }
        
        # Collapse detection
        pred_at_mean = (np.abs(y_pred - y_pred.mean()) < 0.05).sum() / len(y_pred)
        pred_collapsed = y_pred.std() < 0.05
        
        # Distribution similarity (KS test)
        ks_stat, ks_pvalue = ks_2samp(y_pred, y_true)
        
        # Coverage (how much of the range is used)
        pred_coverage = (y_pred.max() - y_pred.min()) / (y_true.max() - y_true.min())
        
        report = {
            'prediction_stats': pred_stats,
            'true_stats': true_stats,
            'pred_at_mean_pct': float(pred_at_mean),
            'pred_collapsed': bool(pred_collapsed),
            'ks_statistic': float(ks_stat),
            'ks_pvalue': float(ks_pvalue),
            'distributions_similar': ks_pvalue > 0.05,
            'range_coverage': float(pred_coverage),
            'variance_ratio': float(pred_stats['std'] / true_stats['std']) if true_stats['std'] > 0 else 0
        }
        
        # Determine health
        issues = []
        if pred_collapsed:
            issues.append("Predictions collapsed (std < 0.05)")
        if pred_at_mean > 0.80:
            issues.append(f"{pred_at_mean*100:.0f}% predictions near mean")
        if pred_coverage < 0.50:
            issues.append(f"Limited range coverage ({pred_coverage*100:.0f}%)")
        if report['variance_ratio'] < 0.50:
            issues.append(f"Pred variance only {report['variance_ratio']*100:.0f}% of true")
        
        report['health_issues'] = issues
        report['healthy'] = len(issues) == 0
        
        # Log results
        self.logger.info(f"\n   Prediction Statistics:")
        self.logger.info(f"      Mean:   {pred_stats['mean']:.4f} (true: {true_stats['mean']:.4f})")
        self.logger.info(f"      Std:    {pred_stats['std']:.4f} (true: {true_stats['std']:.4f})")
        self.logger.info(f"      Range:  [{pred_stats['min']:.4f}, {pred_stats['max']:.4f}]")
        
        self.logger.info(f"\n   Distribution Health:")
        self.logger.info(f"      Collapsed: {pred_collapsed}")
        self.logger.info(f"      At mean: {pred_at_mean*100:.1f}%")
        self.logger.info(f"      Range coverage: {pred_coverage*100:.0f}%")
        self.logger.info(f"      Variance ratio: {report['variance_ratio']:.2f}")
        
        if issues:
            self.logger.warning(f"\n   ⚠️  Issues detected:")
            for issue in issues:
                self.logger.warning(f"      • {issue}")
        else:
            self.logger.info(f"\n   ✅ Distribution is healthy")
        
        return report
    
    def analyze_errors_by_quality_bin(self, y_pred: np.ndarray, y_true: np.ndarray) -> Dict:
        """Analyze model errors across different quality ranges.
        
        Shows where the model performs well and where it struggles.
        
        Args:
            y_pred: Predicted scores
            y_true: True scores
            
        Returns:
            Error analysis by quality bin
        """
        self.logger.info("\n📉 ERROR ANALYSIS BY QUALITY BIN")
        self.logger.info("="*70)
        
        bins = {
            'Low (0.0-0.3)': (0.0, 0.3),
            'Medium (0.3-0.6)': (0.3, 0.6),
            'High (0.6-0.8)': (0.6, 0.8),
            'Excellent (0.8-1.0)': (0.8, 1.0)
        }
        
        error_analysis = {}
        
        for bin_name, (min_q, max_q) in bins.items():
            mask = (y_true >= min_q) & (y_true < max_q)
            
            if mask.sum() > 0:
                errors = y_pred[mask] - y_true[mask]
                
                error_analysis[bin_name] = {
                    'count': int(mask.sum()),
                    'mae': float(mean_absolute_error(y_true[mask], y_pred[mask])),
                    'rmse': float(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))),
                    'bias': float(errors.mean()),  # Positive = over-predict, negative = under-predict
                    'r2': float(r2_score(y_true[mask], y_pred[mask])) if mask.sum() > 1 else 0.0,
                    'pred_mean': float(y_pred[mask].mean()),
                    'true_mean': float(y_true[mask].mean())
                }
        
        # Log results
        self.logger.info(f"\n   {'Bin':<20} {'Count':<8} {'MAE':<8} {'RMSE':<8} {'Bias':<8} {'R²':<8}")
        self.logger.info(f"   {'-'*68}")
        
        for bin_name, stats in error_analysis.items():
            self.logger.info(f"   {bin_name:<20} {stats['count']:<8} "
                           f"{stats['mae']:<8.4f} {stats['rmse']:<8.4f} "
                           f"{stats['bias']:<+8.4f} {stats['r2']:<8.3f}")
        
        # Identify problem areas
        problem_bins = [name for name, stats in error_analysis.items() 
                       if stats['mae'] > 0.15 or abs(stats['bias']) > 0.10]
        
        if problem_bins:
            self.logger.warning(f"\n   ⚠️  High errors in: {', '.join(problem_bins)}")
        else:
            self.logger.info(f"\n   ✅ Consistent performance across all quality ranges")
        
        return error_analysis
    
    def analyze_feature_importance_stability(self, fold_models: List, 
                                            feature_names: List[str]) -> Dict:
        """Check if feature importance is consistent across CV folds.
        
        Stable importance = reliable features.
        Unstable importance = model is unstable or data issues.
        
        Args:
            fold_models: List of trained models from each fold
            feature_names: List of feature names
            
        Returns:
            Feature stability analysis
        """
        self.logger.info("\n🔬 FEATURE IMPORTANCE STABILITY")
        self.logger.info("="*70)
        
        # Collect importance from each fold
        fold_importances = []
        for model in fold_models:
            importance = model.feature_importance(importance_type='gain')
            fold_importances.append(importance)
        
        # Stack into matrix (folds × features)
        importance_matrix = np.array(fold_importances)
        
        # Calculate statistics
        mean_importance = importance_matrix.mean(axis=0)
        std_importance = importance_matrix.std(axis=0)
        
        # Coefficient of variation (CV)
        cv_importance = std_importance / (mean_importance + 1e-8)
        
        # Create stability report
        stability_df = pd.DataFrame({
            'feature': feature_names,
            'mean_importance': mean_importance,
            'std_importance': std_importance,
            'cv': cv_importance,
            'stable': cv_importance < 0.3  # CV < 0.3 = stable
        }).sort_values('mean_importance', ascending=False)
        
        # Top 10 stability
        top_10 = stability_df.head(10)
        top_10_stable_count = top_10['stable'].sum()
        top_10_cv_mean = top_10['cv'].mean()
        
        # Unstable features
        unstable = stability_df[~stability_df['stable']]
        
        report = {
            'top_10_stable_count': int(top_10_stable_count),
            'top_10_cv_mean': float(top_10_cv_mean),
            'top_10_stable': top_10_stable_count >= 8,  # At least 8/10 stable
            'unstable_features': unstable['feature'].tolist(),
            'unstable_count': len(unstable),
            'stability_df': stability_df
        }
        
        # Log results
        self.logger.info(f"\n   Top 10 Features Stability:")
        self.logger.info(f"      Stable: {top_10_stable_count}/10")
        self.logger.info(f"      Mean CV: {top_10_cv_mean:.3f}")
        
        if top_10_stable_count >= 8:
            self.logger.info(f"      ✅ Top features are stable")
        else:
            self.logger.warning(f"      ⚠️  Top features unstable (only {top_10_stable_count}/10)")
        
        self.logger.info(f"\n   Top 10 Features:")
        for idx, row in top_10.iterrows():
            status = '✅' if row['stable'] else '❌'
            self.logger.info(f"      {status} {row['feature']:<35} "
                           f"importance={row['mean_importance']:.0f} ± {row['std_importance']:.0f} "
                           f"(CV={row['cv']:.2f})")
        
        if len(unstable) > 0:
            self.logger.warning(f"\n   ⚠️  {len(unstable)} unstable features (CV > 0.3)")
            self.logger.warning(f"      Consider removing or investigating these features")
        
        return report
    
    def comprehensive_assessment(self, model, training_data: pd.DataFrame,
                                train_metrics: Dict, val_metrics: Dict,
                                fold_scores: List[Dict],
                                fold_models: List) -> Dict:
        """Run all quality assessments in one go.
        
        Args:
            model: Trained model
            training_data: Full training dataset
            train_metrics: Training metrics
            val_metrics: Validation metrics
            fold_scores: CV fold scores
            fold_models: CV fold models
            
        Returns:
            Complete assessment report
        """
        self.logger.info("\n" + "="*70)
        self.logger.info("🔍 COMPREHENSIVE MODEL QUALITY ASSESSMENT")
        self.logger.info("="*70)
        
        # Get predictions for analysis
        feature_cols = [c for c in training_data.columns if c.startswith('feature_')]
        X = training_data[feature_cols].fillna(0.0)
        y_true = training_data['quality_score'].values
        
        # Use model to predict
        y_pred = model.predict(X)
        
        assessment = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_samples': len(training_data),
            'n_features': len(feature_cols)
        }
        
        # 1. Overfitting Detection
        assessment['overfitting'] = self.detect_overfitting(
            train_metrics, val_metrics, fold_scores
        )
        
        # 2. Calibration
        assessment['calibration'] = self.assess_calibration(y_pred, y_true)
        
        # 3. Prediction Distribution
        assessment['prediction_distribution'] = self.analyze_prediction_distribution(y_pred, y_true)
        
        # 4. Error Analysis by Bin
        assessment['error_by_bin'] = self.analyze_errors_by_quality_bin(y_pred, y_true)
        
        # 5. Feature Importance Stability
        if fold_models and len(fold_models) > 1:
            assessment['feature_stability'] = self.analyze_feature_importance_stability(
                fold_models, feature_cols
            )
        
        # Overall health score
        health_score = self._calculate_health_score(assessment)
        assessment['health_score'] = health_score
        assessment['production_ready'] = health_score >= 0.70
        
        # Summary
        self.logger.info("\n" + "="*70)
        self.logger.info("📊 ASSESSMENT SUMMARY")
        self.logger.info("="*70)
        self.logger.info(f"\n   Overall Health Score: {health_score:.2f}/1.00")
        
        if health_score >= 0.80:
            self.logger.info(f"   ✅ EXCELLENT - Production ready!")
        elif health_score >= 0.70:
            self.logger.info(f"   🟢 GOOD - Production ready with monitoring")
        elif health_score >= 0.60:
            self.logger.info(f"   🟡 FAIR - Needs improvement before production")
        else:
            self.logger.info(f"   ❌ POOR - Not ready for production")
        
        return assessment
    
    def _calculate_health_score(self, assessment: Dict) -> float:
        """Calculate overall model health score (0-1)."""
        
        scores = []
        
        # Overfitting (30% weight)
        if assessment['overfitting']['severity'] == 'none':
            scores.append(1.0 * 0.30)
        elif assessment['overfitting']['severity'] == 'mild':
            scores.append(0.8 * 0.30)
        elif assessment['overfitting']['severity'] == 'moderate':
            scores.append(0.5 * 0.30)
        else:
            scores.append(0.2 * 0.30)
        
        # Calibration (25% weight)
        ece = assessment['calibration']['expected_calibration_error']
        if ece < 0.05:
            scores.append(1.0 * 0.25)
        elif ece < 0.10:
            scores.append(0.7 * 0.25)
        else:
            scores.append(0.4 * 0.25)
        
        # Prediction distribution (20% weight)
        if assessment['prediction_distribution']['healthy']:
            scores.append(1.0 * 0.20)
        else:
            issue_count = len(assessment['prediction_distribution']['health_issues'])
            score = max(1.0 - issue_count * 0.25, 0.3)  # Penalize per issue
            scores.append(score * 0.20)
        
        # Feature stability (15% weight)
        if 'feature_stability' in assessment:
            if assessment['feature_stability']['top_10_stable']:
                scores.append(1.0 * 0.15)
            else:
                ratio = assessment['feature_stability']['top_10_stable_count'] / 10
                scores.append(ratio * 0.15)
        else:
            scores.append(0.7 * 0.15)  # No data = moderate score
        
        # Cross-validation stability (10% weight)
        if 'cv_stable' in assessment['overfitting']:
            if assessment['overfitting']['cv_stable']:
                scores.append(1.0 * 0.10)
            else:
                scores.append(0.6 * 0.10)
        else:
            scores.append(0.8 * 0.10)
        
        return sum(scores)


class FeatureImportanceAnalyzer:
    """Analyze feature importance using multiple methods."""
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def calculate_all_importances(self, model, X: pd.DataFrame, y: pd.Series,
                                  feature_names: List[str]) -> Dict:
        """Calculate feature importance using multiple methods.
        
        Args:
            model: Trained LightGBM model
            X: Features
            y: Target
            feature_names: Feature names
            
        Returns:
            Dictionary with all importance methods
        """
        self.logger.info("\n🎯 FEATURE IMPORTANCE ANALYSIS")
        self.logger.info("="*70)
        
        importances = {}
        
        # 1. LightGBM Gain-based (built-in)
        importances['lgbm_gain'] = self._get_lgbm_importance(model, feature_names, 'gain')
        
        # 2. LightGBM Split-based
        importances['lgbm_split'] = self._get_lgbm_importance(model, feature_names, 'split')
        
        # 3. Permutation Importance
        importances['permutation'] = self._calculate_permutation_importance(model, X, y, feature_names)
        
        # 4. SHAP values (if available)
        try:
            import shap
            importances['shap'] = self._calculate_shap_importance(model, X, feature_names)
            shap_available = True
        except ImportError:
            self.logger.warning("   ⚠️  SHAP not available, skipping SHAP importance")
            shap_available = False
        
        # Combine and rank
        combined_df = self._combine_importances(importances, feature_names)
        
        # Log top features
        self.logger.info(f"\n   Top 15 Features (Combined Ranking):")
        self.logger.info(f"   {'Rank':<6} {'Feature':<35} {'LGBM':<10} {'Perm':<10} {'SHAP':<10}")
        self.logger.info(f"   {'-'*70}")
        
        for i, row in combined_df.head(15).iterrows():
            lgbm_str = f"{row['lgbm_gain_rank']:.0f}"
            perm_str = f"{row['permutation_rank']:.0f}"
            shap_str = f"{row['shap_rank']:.0f}" if shap_available else 'N/A'
            
            self.logger.info(f"   {i+1:<6} {row['feature']:<35} {lgbm_str:<10} {perm_str:<10} {shap_str:<10}")
        
        return {
            'importances': importances,
            'combined_ranking': combined_df,
            'shap_available': shap_available
        }
    
    def _get_lgbm_importance(self, model, feature_names: List[str], 
                            importance_type: str) -> pd.DataFrame:
        """Get LightGBM feature importance."""
        importance = model.feature_importance(importance_type=importance_type)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance,
            'importance_pct': importance / importance.sum() * 100 if importance.sum() > 0 else 0
        }).sort_values('importance', ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_permutation_importance(self, model, X: pd.DataFrame, 
                                         y: pd.Series, feature_names: List[str]) -> pd.DataFrame:
        """Calculate permutation importance."""
        self.logger.info("   🔄 Calculating permutation importance...")
        
        from sklearn.inspection import permutation_importance
        
        # Get baseline predictions
        y_pred = model.predict(X)
        baseline_rmse = np.sqrt(mean_squared_error(y, y_pred))
        
        importances = []
        
        for i, feat in enumerate(feature_names):
            # Shuffle this feature
            X_permuted = X.copy()
            X_permuted[feat] = np.random.permutation(X_permuted[feat].values)
            
            # Predict with permuted feature
            y_pred_perm = model.predict(X_permuted)
            permuted_rmse = np.sqrt(mean_squared_error(y, y_pred_perm))
            
            # Importance = increase in error
            importance = permuted_rmse - baseline_rmse
            importances.append(importance)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances,
            'importance_pct': np.array(importances) / sum(importances) * 100 if sum(importances) > 0 else 0
        }).sort_values('importance', ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _calculate_shap_importance(self, model, X: pd.DataFrame, 
                                  feature_names: List[str]) -> pd.DataFrame:
        """Calculate SHAP-based importance."""
        self.logger.info("   🔄 Calculating SHAP values...")
        
        import shap
        
        # Create explainer (use sample for speed)
        sample_size = min(100, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        
        # Mean absolute SHAP value per feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': mean_abs_shap,
            'importance_pct': mean_abs_shap / mean_abs_shap.sum() * 100
        }).sort_values('importance', ascending=False)
        
        df['rank'] = range(1, len(df) + 1)
        
        return df
    
    def _combine_importances(self, importances: Dict, feature_names: List[str]) -> pd.DataFrame:
        """Combine all importance methods into unified ranking."""
        
        combined = pd.DataFrame({'feature': feature_names})
        
        # Add each method's rank
        if 'lgbm_gain' in importances:
            combined = combined.merge(
                importances['lgbm_gain'][['feature', 'rank']].rename(columns={'rank': 'lgbm_gain_rank'}),
                on='feature'
            )
        
        if 'permutation' in importances:
            combined = combined.merge(
                importances['permutation'][['feature', 'rank']].rename(columns={'rank': 'permutation_rank'}),
                on='feature'
            )
        
        if 'shap' in importances:
            combined = combined.merge(
                importances['shap'][['feature', 'rank']].rename(columns={'rank': 'shap_rank'}),
                on='feature'
            )
        
        # Calculate average rank
        rank_cols = [c for c in combined.columns if c.endswith('_rank')]
        combined['avg_rank'] = combined[rank_cols].mean(axis=1)
        combined = combined.sort_values('avg_rank')
        
        return combined

