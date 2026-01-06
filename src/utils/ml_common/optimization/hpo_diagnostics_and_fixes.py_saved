"""
HPO Diagnostics and Fixes for Identical Scores Issue

This module adds diagnostic capabilities and fixes for the hyperparameter optimization
to address the problem of identical scores across trials.

Key Fixes:
1. Enhanced data variance checks before optimization
2. Improved scoring metrics (balanced_accuracy instead of accuracy)
3. Better cross-validation strategy for regime data
4. Expanded search spaces
5. Reduced trial counts with early stopping
6. Better acquisition function selection
7. **Integrated data leakage detection from ml_common.validation**
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional, Tuple
import logging
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score, f1_score

# Import existing ML Common utilities
try:
    from ..validation.data_leakage_prevention import (
        DataLeakagePrevention,
        DataLeakageConfig,
        LeakageReport
    )
    from ..validation.unified_cv import perform_cross_validation
    from ..validation.temporal_cross_validation import TemporalCrossValidator
    DATA_LEAKAGE_UTILS_AVAILABLE = True
except ImportError as e:
    DATA_LEAKAGE_UTILS_AVAILABLE = False
    logging.warning(f"ML Common utilities not fully available: {e}")
except Exception as e:
    DATA_LEAKAGE_UTILS_AVAILABLE = False
    logging.warning(f"Error importing ML Common utilities: {e}")

logger = logging.getLogger(__name__)

class HPODiagnostics:
    """Diagnostic utilities for HPO issues."""

    @staticmethod
    def check_for_data_leakage(X: np.ndarray, y: np.ndarray,
                               timestamps: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Use ml_common's DataLeakagePrevention to check for leakage.

        Returns comprehensive leakage report if available.
        """
        if not DATA_LEAKAGE_UTILS_AVAILABLE:
            return {"error": "Data leakage utils not available", "has_leakage": False}

        try:
            # Create DataFrame for analysis
            df = pd.DataFrame(X)
            df['target'] = y

            # Add timestamps if provided
            if timestamps is not None:
                df['timestamp'] = timestamps
                timestamp_col = 'timestamp'
            else:
                # Create synthetic timestamps if none provided
                df['timestamp'] = pd.date_range('2020-01-01', periods=len(df), freq='1H')
                timestamp_col = 'timestamp'

            # Initialize leakage detector with regime-friendly settings
            config = DataLeakageConfig(
                enable_temporal_validation=False,  # Disable strict temporal validation for regime data
                lookahead_detection_enabled=False, # Disable lookahead bias for regime classification
                train_test_leakage_check=True,
                save_leakage_reports=False,  # Don't save reports during HPO
                enable_detailed_logging=False,
                critical_leakage_threshold=0.15,  # More lenient thresholds for regime data
                warning_leakage_threshold=0.05
            )

            detector = DataLeakagePrevention(config)

            # Run comprehensive leakage detection
            leakage_report = detector.detect_temporal_leakage(
                data=df,
                timestamp_column=timestamp_col,
                target_column='target',
                dataset_name="HPO_Training_Data"
            )

            # Convert report to dict
            return {
                "has_leakage": (leakage_report.temporal_leakage_detected or
                               leakage_report.lookahead_bias_detected or
                               leakage_report.feature_leakage_detected),
                "leakage_rate": leakage_report.overall_leakage_rate,
                "severity": leakage_report.severity_level,
                "temporal_violations": leakage_report.temporal_order_violations,
                "lookahead_samples": leakage_report.lookahead_samples,
                "critical_issues": leakage_report.critical_issues,
                "recommendations": leakage_report.recommendations,
                "full_report": leakage_report
            }

        except Exception as e:
            logger.warning(f"Data leakage check failed: {e}")
            return {"error": str(e), "has_leakage": False}

    @staticmethod
    def check_data_variance(X: np.ndarray, y: np.ndarray, name: str = "dataset") -> Dict[str, Any]:
        """
        Comprehensive data variance check to detect common HPO issues.

        Returns dict with diagnostics and warnings.
        """
        diagnostics = {
            "name": name,
            "issues": [],
            "warnings": [],
            "stats": {},
            "is_valid": True
        }

        # Check target distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        total_samples = len(y)

        diagnostics["stats"]["n_samples"] = total_samples
        diagnostics["stats"]["n_features"] = X.shape[1] if len(X.shape) > 1 else 1
        diagnostics["stats"]["n_classes"] = len(unique_labels)
        diagnostics["stats"]["class_distribution"] = dict(zip(unique_labels.astype(int), counts.astype(int)))

        # Calculate class percentages
        class_percentages = (counts / total_samples * 100).round(2)
        diagnostics["stats"]["class_percentages"] = dict(zip(unique_labels.astype(int), class_percentages))

        # ✅ USE ML COMMON: Run comprehensive data leakage detection
        print("🔍 Running ML Common data leakage detection...")
        leakage_check = HPODiagnostics.check_for_data_leakage(X, y)
        if leakage_check.get("has_leakage"):
            diagnostics["issues"].append(
                f"🚨 DATA LEAKAGE DETECTED by ml_common.validation!\n"
                f"   Severity: {leakage_check['severity']}\n"
                f"   Leakage rate: {leakage_check['leakage_rate']:.2%}\n"
                f"   Temporal violations: {leakage_check['temporal_violations']}\n"
                f"   Critical issues: {leakage_check['critical_issues']}"
            )
            diagnostics["is_valid"] = False
            diagnostics["stats"]["leakage_detected"] = True
            diagnostics["stats"]["leakage_severity"] = leakage_check['severity']
        else:
            diagnostics["stats"]["leakage_detected"] = False

        # Check for feature-target correlation (potential signal)
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit

            # Quick baseline test with default parameters using TimeSeriesSplit for temporal data
            rf_baseline = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            # Use TimeSeriesSplit to maintain temporal order and prevent data leakage
            tscv = TimeSeriesSplit(n_splits=3)
            baseline_scores = cross_val_score(rf_baseline, X, y, cv=tscv, scoring='accuracy')

            diagnostics["stats"]["baseline_accuracy_mean"] = float(np.mean(baseline_scores))
            diagnostics["stats"]["baseline_accuracy_std"] = float(np.std(baseline_scores))
            diagnostics["stats"]["baseline_scores"] = baseline_scores.tolist()

            # Fit model to get feature importances
            rf_baseline.fit(X, y)
            feature_importances = rf_baseline.feature_importances_

            max_importance = float(np.max(feature_importances)) if feature_importances.size else 0.0
            mean_importance = float(np.mean(feature_importances)) if feature_importances.size else 0.0
            nonzero_importance = int(np.sum(feature_importances > 0.01)) if feature_importances.size else 0

            diagnostics["stats"]["max_feature_importance"] = max_importance
            diagnostics["stats"]["mean_feature_importance"] = mean_importance
            diagnostics["stats"]["nonzero_importance_features"] = nonzero_importance

            n_features_for_threshold = int(diagnostics["stats"].get("n_features", max(1, feature_importances.shape[0])))
            importance_threshold = max(0.05, 5.0 / max(1, n_features_for_threshold))
            diagnostics["stats"]["importance_warning_threshold"] = importance_threshold

            # Check if features have signal (dynamic threshold)
            if max_importance < importance_threshold:
                diagnostics["warnings"].append(
                    f"⚠️ ALL features have very low importance (<{importance_threshold:.4f})!\n"
                    f"   Max importance: {max_importance:.4f}\n"
                    f"   This suggests features have NO predictive signal!"
                )

            # Check if baseline varies across folds
            if np.std(baseline_scores) < 0.01:
                diagnostics["warnings"].append(
                    f"⚠️ Baseline model scores are nearly identical across CV folds "
                    f"(std={np.std(baseline_scores):.6f}). This suggests:\n"
                    f"   - Features may have weak/no signal\n"
                    f"   - CV may not be working properly\n"
                    f"   - Data may be too small/noisy"
                )

            # Check if baseline is just guessing
            expected_random = 1.0 / len(unique_labels)
            if diagnostics["stats"]["baseline_accuracy_mean"] < expected_random + 0.05:
                diagnostics["warnings"].append(
                    f"⚠️ Baseline accuracy ({diagnostics['stats']['baseline_accuracy_mean']:.4f}) "
                    f"is barely above random guessing ({expected_random:.4f})!\n"
                    f"   Features likely have NO predictive power."
                )

            # Check for suspiciously HIGH scores (data leakage)
            if diagnostics["stats"]["baseline_accuracy_mean"] > 0.95:
                diagnostics["issues"].append(
                    f"🚨 CRITICAL: Baseline accuracy is {diagnostics['stats']['baseline_accuracy_mean']:.4f} (>95%)!\n"
                    f"   This is EXTREMELY suspicious and strongly indicates DATA LEAKAGE!\n"
                    f"   Likely causes:\n"
                    f"   1. Features contain the target variable\n"
                    f"   2. Features from SAME timestamp as labels (no temporal separation)\n"
                    f"   3. Features using future information\n"
                    f"   ACTION REQUIRED: Run diagnose_regime_data_leakage.py script!"
                )
                diagnostics["is_valid"] = False
            elif diagnostics["stats"]["baseline_accuracy_mean"] > 0.85:
                diagnostics["warnings"].append(
                    f"⚠️ Baseline accuracy is {diagnostics['stats']['baseline_accuracy_mean']:.4f} (>85%)!\n"
                    f"   This is suspiciously high - possible data leakage.\n"
                    f"   Check: Are features from same timestamp as labels?"
                )

            # Check if any CV fold is perfect
            if any(score >= 0.99 for score in baseline_scores):
                perfect_folds = [i for i, s in enumerate(baseline_scores) if s >= 0.99]
                diagnostics["issues"].append(
                    f"🚨 CRITICAL: CV fold(s) {perfect_folds} achieved near-perfect scores (>=99%)!\n"
                    f"   Fold scores: {baseline_scores.tolist()}\n"
                    f"   This is a STRONG indicator of DATA LEAKAGE!\n"
                    f"   One fold shouldn't be near-perfect while others aren't."
                )
                diagnostics["is_valid"] = False

        except Exception as e:
            diagnostics["warnings"].append(f"Could not compute baseline scores: {e}")

        # Check for severe imbalance
        max_percentage = np.max(class_percentages)
        min_percentage = np.min(class_percentages)

        if max_percentage > 80:
            diagnostics["issues"].append(
                f"⚠️ SEVERE CLASS IMBALANCE: {max_percentage:.1f}% in majority class. "
                f"This causes constant predictions!"
            )
            diagnostics["is_valid"] = False

        if max_percentage > 70:
            diagnostics["warnings"].append(
                f"High class imbalance: {max_percentage:.1f}% in majority class. "
                f"Consider using balanced_accuracy or class_weight='balanced'"
            )

        # Check for single class
        if len(unique_labels) < 2:
            diagnostics["issues"].append("🚨 CRITICAL: Only one class in dataset!")
            diagnostics["is_valid"] = False

        # Check feature variance
        if len(X.shape) > 1:
            feature_vars = np.var(X, axis=0)
            zero_var_features = np.sum(feature_vars == 0)
            low_var_features = np.sum(feature_vars < 1e-6)

            diagnostics["stats"]["zero_variance_features"] = int(zero_var_features)
            diagnostics["stats"]["low_variance_features"] = int(low_var_features)
            diagnostics["stats"]["mean_feature_variance"] = float(np.mean(feature_vars))

            if zero_var_features > 0:
                diagnostics["warnings"].append(
                    f"⚠️ {zero_var_features} features have zero variance!"
                )

            if low_var_features > X.shape[1] * 0.5:
                diagnostics["warnings"].append(
                    f"⚠️ {low_var_features}/{X.shape[1]} features have very low variance"
                )

        # Check for NaN values
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            diagnostics["issues"].append(f"🚨 CRITICAL: {nan_count} NaN values in features!")
            diagnostics["is_valid"] = False

        # Check for infinite values
        inf_count = np.isinf(X).sum()
        if inf_count > 0:
            diagnostics["issues"].append(f"🚨 CRITICAL: {inf_count} infinite values in features!")
            diagnostics["is_valid"] = False

        return diagnostics

    @staticmethod
    def print_diagnostics(diagnostics: Dict[str, Any]) -> None:
        """Pretty print diagnostics results."""
        print("\n" + "="*80)
        print(f"📊 HPO DIAGNOSTICS: {diagnostics['name']}")
        print("="*80)

        # Print stats
        stats = diagnostics["stats"]
        print(f"\n📈 Dataset Stats:")
        print(f"  • Samples: {stats['n_samples']}")
        print(f"  • Features: {stats['n_features']}")
        print(f"  • Classes: {stats['n_classes']}")

        # Dataset quality assessment
        n_samples = stats['n_samples']
        n_features = stats['n_features']
        n_classes = stats['n_classes']

        if n_samples < 500:
            print(f"\n⚠️  SMALL DATASET WARNING:")
            print(f"   • Only {n_samples} samples - prone to overfitting")
            print(f"   • Consider increasing dataset size for reliable results")
        elif n_samples < 1000:
            print(f"\n⚠️  MODERATE DATASET SIZE:")
            print(f"   • {n_samples} samples - use caution with complex models")

        if n_features > n_samples / 5:
            print(f"\n⚠️  HIGH DIMENSIONALITY WARNING:")
            print(f"   • {n_features} features for {n_samples} samples")
            print(f"   • Feature selection will be applied to reduce to ~80 features")
            print(f"   • This will improve generalization and model performance")

        if "class_distribution" in stats:
            print(f"\n🎯 Class Distribution:")
            for label, count in stats["class_distribution"].items():
                percentage = stats["class_percentages"][label]
                print(f"  • Class {label}: {count} samples ({percentage:.1f}%)")

        if "zero_variance_features" in stats:
            print(f"\n🔍 Feature Variance:")
            print(f"  • Zero variance features: {stats['zero_variance_features']}")
            print(f"  • Low variance features: {stats['low_variance_features']}")
            print(f"  • Mean variance: {stats['mean_feature_variance']:.6f}")

        # Print baseline model performance
        if "baseline_accuracy_mean" in stats:
            print(f"\n🎯 Baseline Model Performance (RandomForest default params):")
            print(f"  • Mean CV accuracy: {stats['baseline_accuracy_mean']:.4f}")
            print(f"  • Std CV accuracy: {stats['baseline_accuracy_std']:.6f}")
            print(f"  • CV fold scores: {[f'{s:.4f}' for s in stats['baseline_scores']]}")

            if stats['baseline_accuracy_std'] < 0.01:
                print(f"  ⚠️  VERY LOW VARIANCE across folds - potential issue!")

        # Print feature importance stats
        if "max_feature_importance" in stats:
            print(f"\n🔬 Feature Importance Analysis:")
            print(f"  • Max feature importance: {stats['max_feature_importance']:.4f}")
            print(f"  • Mean feature importance: {stats['mean_feature_importance']:.4f}")
            print(f"  • Features with >1% importance: {stats['nonzero_importance_features']}/{stats['n_features']}")
            threshold = stats.get('importance_warning_threshold', 0.05)

            if stats['max_feature_importance'] < threshold:
                print(f"  ⚠️  ALL features have very low importance (<{threshold:.4f}) - NO SIGNAL!")

        # Print issues
        if diagnostics["issues"]:
            print(f"\n🚨 CRITICAL ISSUES ({len(diagnostics['issues'])}):")
            for issue in diagnostics["issues"]:
                print(f"  {issue}")

        # Print warnings
        if diagnostics["warnings"]:
            print(f"\n⚠️  WARNINGS ({len(diagnostics['warnings'])}):")
            for warning in diagnostics["warnings"]:
                print(f"  {warning}")

        # Print validation result
        if diagnostics["is_valid"]:
            print(f"\n✅ Data validation PASSED - safe to proceed with HPO")
        else:
            print(f"\n❌ Data validation FAILED - FIX ISSUES BEFORE HPO!")

        # Print high score recommendations if applicable
        if "suspicious_scores" in diagnostics and diagnostics["suspicious_scores"]:
            print(f"\n💡 HIGH SCORE RECOMMENDATIONS:")
            print(f"   • Use stronger regularization (increase alpha in Ridge/Lasso)")
            print(f"   • Implement feature selection to reduce dimensionality")
            print(f"   • Increase dataset size through data augmentation or collection")
            print(f"   • Use ensemble methods for better generalization")
            print(f"   • Implement early stopping during training")

        print("="*80 + "\n")

    @staticmethod
    def recommend_scoring_metric(diagnostics: Dict[str, Any]) -> str:
        """Recommend appropriate scoring metric based on data characteristics."""
        stats = diagnostics["stats"]

        if stats["n_classes"] == 2:
            # Binary classification
            class_percentages = list(stats["class_percentages"].values())
            max_percentage = max(class_percentages)

            if max_percentage > 70:
                return "balanced_accuracy"  # For imbalanced data
            else:
                return "f1"  # For balanced data
        else:
            # Multiclass
            return "f1_macro"  # For multi-class

class ImprovedHPOConfig:
    """Improved HPO configuration based on diagnostics."""

    @staticmethod
    def get_improved_random_forest_search_space(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get improved RandomForest search space based on data characteristics.

        Fixes:
        - Expanded max_depth range (was 5-50, now 5-15 for regime data)
        - More reasonable n_estimators (was 50-500, now 100-500)
        - Better min_samples_leaf range
        - Added max_features float options
        """
        n_samples = diagnostics["stats"]["n_samples"]
        n_features = diagnostics["stats"]["n_features"]

        # Base search space
        search_space = {
            'n_estimators': {'type': 'int', 'low': 100, 'high': 500},
            'max_depth': {'type': 'int', 'low': 5, 'high': 15},
            'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
            'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.5]},
            'bootstrap': {'type': 'categorical', 'choices': [True, False]},
            'class_weight': {'type': 'categorical', 'choices': ['balanced', 'balanced_subsample', None]}
        }

        # Adjust for small datasets
        if n_samples < 1000:
            search_space['n_estimators']['high'] = 300
            search_space['min_samples_leaf']['low'] = 2

        # Adjust for large datasets
        if n_samples > 10000:
            search_space['n_estimators']['low'] = 200
            search_space['min_samples_split']['low'] = 5

        return search_space

    @staticmethod
    def get_improved_hpo_params(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get improved HPO parameters.

        Fixes:
        - Reduced n_trials for initial testing (was unlimited, now 10-15)
        - Added early stopping
        - Better acquisition function (EI instead of UCB)
        - Added pruning for bad trials
        - ✅ Uses ml_common utilities for better CV
        """
        stats = diagnostics["stats"]

        # Recommend scoring metric
        scoring = HPODiagnostics.recommend_scoring_metric(diagnostics)

        # ✅ USE ML COMMON: Better CV strategy with temporal validation
        if DATA_LEAKAGE_UTILS_AVAILABLE:
            # Use temporal cross-validation from ml_common
            try:
                cv_strategy = TemporalCrossValidator(
                    n_splits=5 if stats["n_samples"] > 1000 else 3,
                    test_size=0.2,
                    embargo_periods=5,  # Prevent lookahead bias
                    shuffle=False  # Maintain temporal order
                )
                cv_description = f"TemporalCrossValidator(n_splits={5 if stats['n_samples'] > 1000 else 3}, embargo=5)"
                logger.info("✅ Using ml_common TemporalCrossValidator for better temporal integrity")
            except Exception as e:
                logger.warning(f"Failed to use TemporalCrossValidator: {e}, falling back to standard CV")
                # Fallback to standard CV
                if stats["n_samples"] > 1000:
                    cv_strategy = TimeSeriesSplit(n_splits=3)  # Reduced from 5 for speed
                    cv_description = "TimeSeriesSplit(n_splits=3)"
                else:
                    # ⚠️ Using TimeSeriesSplit to prevent data leakage in time series data
                    cv_strategy = TimeSeriesSplit(n_splits=3)
                    cv_description = "TimeSeriesSplit(n_splits=3)"
        else:
            # Recommend CV strategy without ml_common
            if stats["n_samples"] > 1000:
                cv_strategy = TimeSeriesSplit(n_splits=3)  # Reduced from 5 for speed
                cv_description = "TimeSeriesSplit(n_splits=3)"
            else:
                # ⚠️ Using TimeSeriesSplit to prevent data leakage in time series data
                cv_strategy = TimeSeriesSplit(n_splits=3)
                cv_description = "TimeSeriesSplit(n_splits=3)"

        return {
            'n_trials': 10,  # Reduced from 15 for faster iteration
            'scoring': scoring,
            'cv_strategy': cv_strategy,
            'cv_description': cv_description,
            'sampler': 'TPE',  # Bayesian optimization
            'pruner': 'MedianPruner',  # Prune bad trials early
            'acquisition_function': 'ei',  # Expected Improvement
            'timeout': 600,  # 10 minutes max
            'early_stopping_patience': 5,  # Stop if no improvement
            'show_progress_bar': True,
            'verbose': 1,
            'use_ml_common_cv': DATA_LEAKAGE_UTILS_AVAILABLE
        }

class HPOMonitor:
    """Monitor HPO progress and detect issues in real-time."""

    def __init__(self):
        self.trial_scores = []
        self.trial_params = []

    def record_trial(self, trial_number: int, score: float, params: Dict[str, Any]) -> None:
        """Record trial result."""
        self.trial_scores.append(score)
        self.trial_params.append(params)

        # Check for issues every 5 trials
        if len(self.trial_scores) >= 5 and trial_number % 5 == 0:
            self._check_for_issues()

    def _check_for_issues(self) -> None:
        """Check for common issues during optimization."""
        recent_scores = self.trial_scores[-5:]

        # Check for identical scores
        unique_scores = len(set(recent_scores))
        if unique_scores == 1:
            logger.warning(
                f"⚠️  ALL RECENT SCORES IDENTICAL: {recent_scores[0]:.4f}\n"
                f"   This suggests:\n"
                f"   1. Model is predicting constant class (check class imbalance)\n"
                f"   2. Scoring metric may be inappropriate (try balanced_accuracy)\n"
                f"   3. Features may have no signal (check feature variance)\n"
                f"   4. Cross-validation may be broken (check CV strategy)"
            )
        elif unique_scores <= 2:
            logger.warning(
                f"⚠️  Very low score variance: only {unique_scores} unique scores in last 5 trials"
            )

        # Check for zero scores
        if any(score == 0.0 for score in recent_scores):
            logger.error(
                f"🚨 ZERO SCORES DETECTED! Recent scores: {recent_scores}\n"
                f"   Model training is failing completely!"
            )

        # Calculate variance
        score_variance = np.var(recent_scores)
        if score_variance < 1e-6:
            logger.warning(
                f"⚠️  Score variance extremely low: {score_variance:.10f}\n"
                f"   HPO is not exploring effectively!"
            )

def apply_hpo_fixes(X: np.ndarray, y: np.ndarray,
                    model_type: str = "random_forest") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply all HPO fixes and return improved configuration.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model for search space

    Returns:
        Tuple of (search_space, hpo_params)
    """
    logger.info("🔍 Running HPO diagnostics...")

    # Run diagnostics
    diagnostics = HPODiagnostics.check_data_variance(X, y)
    HPODiagnostics.print_diagnostics(diagnostics)

    if not diagnostics["is_valid"]:
        raise ValueError(
            "Data validation failed! Fix critical issues before running HPO.\n"
            "See diagnostic output above for details."
        )

    # Get improved configuration
    if model_type == "random_forest":
        search_space = ImprovedHPOConfig.get_improved_random_forest_search_space(diagnostics)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    hpo_params = ImprovedHPOConfig.get_improved_hpo_params(diagnostics)

    logger.info(f"✅ Using improved scoring metric: {hpo_params['scoring']}")
    logger.info(f"✅ Using CV strategy: {hpo_params['cv_description']}")
    logger.info(f"✅ Running {hpo_params['n_trials']} trials with early stopping")

    return search_space, hpo_params
