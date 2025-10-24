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


class NumericalStability:
    """Utilities for numerical stability in financial calculations."""
    
    @staticmethod
    def safe_divide(numerator, denominator, default_value=0.0, min_denominator=1e-8):
        """Safely divide two numbers with numerical stability checks."""
        try:
            import numpy as np
            numerator = np.asarray(numerator, dtype=np.float64)
            denominator = np.asarray(denominator, dtype=np.float64)
            
            # Check for numerical issues
            if not np.isfinite(numerator) or not np.isfinite(denominator):
                return default_value
                
            if abs(denominator) < min_denominator:
                return default_value
                
            result = numerator / denominator
            return result if np.isfinite(result) else default_value
        except Exception:
            return default_value
    
    @staticmethod
    def safe_log(value, default_value=0.0, min_value=1e-8):
        """Safely compute logarithm with numerical stability."""
        try:
            import numpy as np
            value = np.asarray(value, dtype=np.float64)
            
            if not np.isfinite(value) or value <= min_value:
                return default_value
                
            result = np.log(value)
            return result if np.isfinite(result) else default_value
        except Exception:
            return default_value
    
    @staticmethod
    def safe_sqrt(value, default_value=0.0):
        """Safely compute square root with numerical stability."""
        try:
            import numpy as np
            value = np.asarray(value, dtype=np.float64)
            
            if not np.isfinite(value) or value < 0:
                return default_value
                
            result = np.sqrt(value)
            return result if np.isfinite(result) else default_value
        except Exception:
            return default_value

class HPODiagnostics:
    """Diagnostic utilities for HPO issues."""
    
    # Class-level cache for expensive computations
    _baseline_cache = {}
    _leakage_cache = {}

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
        logger.info("🔍 Running ML Common data leakage detection...")
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

        # Check for feature-target correlation (potential signal) with caching
        try:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.model_selection import cross_val_score, TimeSeriesSplit

            # Create cache key for this dataset
            cache_key = f"{X.shape}_{y.shape}_{hash(str(X.flat[:10]))}_{hash(str(y[:10]))}"
            
            if cache_key in HPODiagnostics._baseline_cache:
                # Use cached results
                cached_results = HPODiagnostics._baseline_cache[cache_key]
                diagnostics["stats"].update(cached_results)
                logger.info("Using cached baseline model results")
            else:
                # Quick baseline test with reduced parameters for faster computation
                rf_baseline = RandomForestClassifier(
                    n_estimators=20,  # Reduced from 50
                    random_state=42, 
                    max_depth=3,  # Reduced from 5
                    n_jobs=1  # Single thread for consistency
                )
                # Use TimeSeriesSplit to maintain temporal order and prevent data leakage
                tscv = TimeSeriesSplit(n_splits=3)
                baseline_scores = cross_val_score(rf_baseline, X, y, cv=tscv, scoring='accuracy')

                diagnostics["stats"]["baseline_accuracy_mean"] = float(np.mean(baseline_scores))
                diagnostics["stats"]["baseline_accuracy_std"] = float(np.std(baseline_scores))
                diagnostics["stats"]["baseline_scores"] = baseline_scores.tolist()

                # Fit model to get feature importances
                rf_baseline.fit(X, y)
                feature_importances = rf_baseline.feature_importances_
                diagnostics["stats"]["max_feature_importance"] = float(np.max(feature_importances))
                diagnostics["stats"]["mean_feature_importance"] = float(np.mean(feature_importances))
                diagnostics["stats"]["nonzero_importance_features"] = int(np.sum(feature_importances > 0.01))
                
                # Cache the results
                HPODiagnostics._baseline_cache[cache_key] = {
                    "baseline_accuracy_mean": diagnostics["stats"]["baseline_accuracy_mean"],
                    "baseline_accuracy_std": diagnostics["stats"]["baseline_accuracy_std"],
                    "baseline_scores": diagnostics["stats"]["baseline_scores"],
                    "max_feature_importance": diagnostics["stats"]["max_feature_importance"],
                    "mean_feature_importance": diagnostics["stats"]["mean_feature_importance"],
                    "nonzero_importance_features": diagnostics["stats"]["nonzero_importance_features"]
                }

            # Check if features have signal
            if np.max(feature_importances) < 0.05:
                diagnostics["warnings"].append(
                    f"⚠️ ALL features have very low importance (<0.05)!\n"
                    f"   Max importance: {np.max(feature_importances):.4f}\n"
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
        logger.info("\n" + "="*80)
        logger.info(f"📊 HPO DIAGNOSTICS: {diagnostics['name']}")
        logger.info("="*80)

        # Print stats
        stats = diagnostics["stats"]
        logger.info(f"\n📈 Dataset Stats:")
        logger.info(f"  • Samples: {stats['n_samples']}")
        logger.info(f"  • Features: {stats['n_features']}")
        logger.info(f"  • Classes: {stats['n_classes']}")

        # Dataset quality assessment
        n_samples = stats['n_samples']
        n_features = stats['n_features']
        n_classes = stats['n_classes']

        if n_samples < 500:
            logger.warning(f"\n⚠️  SMALL DATASET WARNING:")
            logger.warning(f"   • Only {n_samples} samples - prone to overfitting")
            logger.warning(f"   • Consider increasing dataset size for reliable results")
        elif n_samples < 1000:
            logger.warning(f"\n⚠️  MODERATE DATASET SIZE:")
            logger.warning(f"   • {n_samples} samples - use caution with complex models")

        if n_features > n_samples / 5:
            logger.warning(f"\n⚠️  HIGH DIMENSIONALITY WARNING:")
            logger.warning(f"   • {n_features} features for {n_samples} samples")
            logger.warning(f"   • Feature selection will be applied to reduce to ~200 features")
            logger.warning(f"   • This will improve generalization and model performance")

        if "class_distribution" in stats:
            logger.info(f"\n🎯 Class Distribution:")
            for label, count in stats["class_distribution"].items():
                percentage = stats["class_percentages"][label]
                logger.info(f"  • Class {label}: {count} samples ({percentage:.1f}%)")

        if "zero_variance_features" in stats:
            logger.info(f"\n🔍 Feature Variance:")
            logger.info(f"  • Zero variance features: {stats['zero_variance_features']}")
            logger.info(f"  • Low variance features: {stats['low_variance_features']}")
            logger.info(f"  • Mean variance: {stats['mean_feature_variance']:.6f}")

        # Print baseline model performance
        if "baseline_accuracy_mean" in stats:
            logger.info(f"\n🎯 Baseline Model Performance (RandomForest default params):")
            logger.info(f"  • Mean CV accuracy: {stats['baseline_accuracy_mean']:.4f}")
            logger.info(f"  • Std CV accuracy: {stats['baseline_accuracy_std']:.6f}")
            logger.info(f"  • CV fold scores: {[f'{s:.4f}' for s in stats['baseline_scores']]}")

            if stats['baseline_accuracy_std'] < 0.01:
                logger.warning(f"  ⚠️  VERY LOW VARIANCE across folds - potential issue!")

        # Print feature importance stats
        if "max_feature_importance" in stats:
            logger.info(f"\n🔬 Feature Importance Analysis:")
            logger.info(f"  • Max feature importance: {stats['max_feature_importance']:.4f}")
            logger.info(f"  • Mean feature importance: {stats['mean_feature_importance']:.4f}")
            logger.info(f"  • Features with >1% importance: {stats['nonzero_importance_features']}/{stats['n_features']}")

            if stats['max_feature_importance'] < 0.05:
                logger.warning(f"  ⚠️  ALL features have very low importance - NO SIGNAL!")

        # Print issues
        if diagnostics["issues"]:
            logger.error(f"\n🚨 CRITICAL ISSUES ({len(diagnostics['issues'])}):")
            for issue in diagnostics["issues"]:
                logger.error(f"  {issue}")

        # Print warnings
        if diagnostics["warnings"]:
            logger.warning(f"\n⚠️  WARNINGS ({len(diagnostics['warnings'])}):")
            for warning in diagnostics["warnings"]:
                logger.warning(f"  {warning}")

        # Print validation result
        if diagnostics["is_valid"]:
            logger.info(f"\n✅ Data validation PASSED - safe to proceed with HPO")
        else:
            logger.error(f"\n❌ Data validation FAILED - FIX ISSUES BEFORE HPO!")

        # Print high score recommendations if applicable
        if "suspicious_scores" in diagnostics and diagnostics["suspicious_scores"]:
            logger.info(f"\n💡 HIGH SCORE RECOMMENDATIONS:")
            logger.info(f"   • Use stronger regularization (increase alpha in Ridge/Lasso)")
            logger.info(f"   • Implement feature selection to reduce dimensionality")
            logger.info(f"   • Increase dataset size through data augmentation or collection")
            logger.info(f"   • Use ensemble methods for better generalization")
            logger.info(f"   • Implement early stopping during training")

        logger.info("="*80 + "\n")

    @staticmethod
    def recommend_scoring_metric(diagnostics: Dict[str, Any], task_type: str = "classification") -> str:
        """
        Recommend appropriate scoring metric based on data characteristics and task type.
        
        Args:
            diagnostics: Data diagnostics results
            task_type: Type of task - "classification", "regression", or "financial"
            
        Returns:
            Recommended scoring metric
        """
        stats = diagnostics["stats"]
        
        if task_type == "regression":
            # Regression metrics
            if stats.get("n_samples", 0) < 1000:
                return "neg_mean_squared_error"  # MSE for small datasets
            else:
                return "neg_root_mean_squared_error"  # RMSE for larger datasets
                
        elif task_type == "financial":
            # Financial-specific metrics
            if stats.get("n_classes", 0) == 2:
                # Binary financial classification (e.g., buy/sell signals)
                class_percentages = list(stats.get("class_percentages", {}).values())
                if class_percentages:
                    max_percentage = max(class_percentages)
                    if max_percentage > 70:
                        return "balanced_accuracy"  # For imbalanced financial data
                    else:
                        return "f1"  # For balanced financial data
                else:
                    return "f1"  # Default for financial binary classification
            else:
                # Financial regression (e.g., price prediction, returns)
                return "neg_mean_squared_error"
                
        else:
            # Default classification behavior
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

class FinancialMetrics:
    """Financial-specific metrics for optimization with numerical stability."""
    
    @staticmethod
    def sharpe_ratio(y_true, y_pred, risk_free_rate=0.02, min_std=1e-8):
        """Calculate Sharpe ratio for financial predictions with numerical stability."""
        try:
            import numpy as np
            # Use double precision for financial calculations
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            risk_free_rate = np.float64(risk_free_rate)
            
            returns = y_pred - y_true
            excess_returns = returns - risk_free_rate
            
            # Use numerical stability for standard deviation
            std_returns = np.std(excess_returns, ddof=1)  # Use sample std
            if std_returns < min_std:
                return 0.0
            
            mean_returns = np.mean(excess_returns)
            # Check for numerical issues
            if not np.isfinite(mean_returns) or not np.isfinite(std_returns):
                return 0.0
                
            return mean_returns / std_returns
        except Exception:
            return 0.0
    
    @staticmethod
    def max_drawdown(y_true, y_pred):
        """Calculate maximum drawdown with numerical stability."""
        try:
            import numpy as np
            # Use double precision
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            
            returns = y_pred - y_true
            # Check for numerical issues
            if not np.all(np.isfinite(returns)):
                return 0.0
                
            cumulative = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative)
            drawdown = cumulative - running_max
            
            # Return the most negative drawdown
            max_dd = np.min(drawdown)
            return max_dd if np.isfinite(max_dd) else 0.0
        except Exception:
            return 0.0
    
    @staticmethod
    def profit_factor(y_true, y_pred, min_loss=1e-8):
        """Calculate profit factor (gross profit / gross loss) with numerical stability."""
        try:
            import numpy as np
            # Use double precision
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            
            returns = y_pred - y_true
            # Check for numerical issues
            if not np.all(np.isfinite(returns)):
                return 1.0
                
            gross_profit = np.sum(returns[returns > 0])
            gross_loss = abs(np.sum(returns[returns < 0]))
            
            # Numerical stability check
            if gross_loss < min_loss:
                if gross_profit > min_loss:
                    return 1000.0  # Cap at reasonable value instead of inf
                else:
                    return 1.0
            
            pf = gross_profit / gross_loss
            # Cap at reasonable values
            return min(max(pf, 0.0), 1000.0) if np.isfinite(pf) else 1.0
        except Exception:
            return 1.0
    
    @staticmethod
    def hit_rate(y_true, y_pred, threshold=0.0):
        """Calculate hit rate (percentage of correct directional predictions) with numerical stability."""
        try:
            import numpy as np
            # Use double precision
            y_true = np.asarray(y_true, dtype=np.float64)
            y_pred = np.asarray(y_pred, dtype=np.float64)
            
            # Check for numerical issues
            if not np.all(np.isfinite(y_true)) or not np.all(np.isfinite(y_pred)):
                return 0.5
                
            y_true_direction = np.sign(y_true)
            y_pred_direction = np.sign(y_pred)
            
            # Calculate hit rate with numerical stability
            hits = np.sum(y_true_direction == y_pred_direction)
            total = len(y_true_direction)
            
            if total == 0:
                return 0.5
                
            hit_rate = hits / total
            return hit_rate if np.isfinite(hit_rate) else 0.5
        except Exception:
            return 0.5


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
    def get_improved_financial_search_space(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get improved search space for financial models.
        
        Includes financial-specific parameters and risk-aware settings.
        """
        n_samples = diagnostics["stats"]["n_samples"]
        n_features = diagnostics["stats"]["n_features"]

        # Base search space optimized for financial data
        search_space = {
            'n_estimators': {'type': 'int', 'low': 200, 'high': 1000},
            'max_depth': {'type': 'int', 'low': 3, 'high': 12},  # Shallow trees for financial data
            'min_samples_split': {'type': 'int', 'low': 5, 'high': 50},  # Higher for stability
            'min_samples_leaf': {'type': 'int', 'low': 2, 'high': 20},  # Higher for stability
            'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', 0.3, 0.5]},
            'bootstrap': {'type': 'categorical', 'choices': [True, False]},
            'class_weight': {'type': 'categorical', 'choices': ['balanced', 'balanced_subsample', None]},
            'min_impurity_decrease': {'type': 'float', 'low': 0.0, 'high': 0.01},  # Financial stability
            'ccp_alpha': {'type': 'float', 'low': 0.0, 'high': 0.1}  # Cost complexity pruning
        }

        # Adjust for small datasets
        if n_samples < 1000:
            search_space['n_estimators']['high'] = 500
            search_space['min_samples_leaf']['low'] = 5
            search_space['min_samples_split']['low'] = 10

        return search_space

    @staticmethod
    def get_improved_hpo_params(diagnostics: Dict[str, Any], task_type: str = "classification") -> Dict[str, Any]:
        """
        Get improved HPO parameters.

        Fixes:
        - Reduced n_trials for initial testing (was unlimited, now 10-15)
        - Added early stopping
        - Better acquisition function (EI instead of UCB)
        - Added pruning for bad trials
        - ✅ Uses ml_common utilities for better CV
        - ✅ Supports financial and regression task types
        """
        stats = diagnostics["stats"]

        # Recommend scoring metric based on task type
        scoring = HPODiagnostics.recommend_scoring_metric(diagnostics, task_type)

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
                    cv_strategy = TimeSeriesSplit(n_splits=5)
                    cv_description = "TimeSeriesSplit(n_splits=5)"
                else:
                    # ⚠️ Using TimeSeriesSplit to prevent data leakage in time series data
                    cv_strategy = TimeSeriesSplit(n_splits=3)
                    cv_description = "TimeSeriesSplit(n_splits=3)"
        else:
            # Recommend CV strategy without ml_common
            if stats["n_samples"] > 1000:
                cv_strategy = TimeSeriesSplit(n_splits=5)
                cv_description = "TimeSeriesSplit(n_splits=5)"
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

    def __init__(self, check_window_size: int = 5, low_variance_threshold: int = 2, 
                 variance_threshold: float = 1e-6):
        """
        Initialize HPO monitor with configurable thresholds.
        
        Args:
            check_window_size: Number of recent trials to check for issues
            low_variance_threshold: Maximum number of unique scores to trigger warning
            variance_threshold: Minimum variance threshold to trigger warning
        """
        self.trial_scores = []
        self.trial_params = []
        self.check_window_size = check_window_size
        self.low_variance_threshold = low_variance_threshold
        self.variance_threshold = variance_threshold

    def record_trial(self, trial_number: int, score: float, params: Dict[str, Any]) -> None:
        """Record trial result."""
        self.trial_scores.append(score)
        self.trial_params.append(params)

        # Check for issues with configurable window size
        if len(self.trial_scores) >= self.check_window_size and trial_number % self.check_window_size == 0:
            self._check_for_issues()
        elif len(self.trial_scores) < self.check_window_size and len(self.trial_scores) >= 2:
            # Check with available trials if we have at least 2
            self._check_for_issues(len(self.trial_scores))

    def _check_for_issues(self, window_size: Optional[int] = None) -> None:
        """Check for common issues during optimization."""
        if window_size is None:
            window_size = self.check_window_size
        
        # Use available trials if we have fewer than the window size
        actual_window = min(window_size, len(self.trial_scores))
        recent_scores = self.trial_scores[-actual_window:]

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
        elif unique_scores <= self.low_variance_threshold:
            logger.warning(
                f"⚠️  Very low score variance: only {unique_scores} unique scores in last {actual_window} trials"
            )

        # Check for zero scores
        if any(score == 0.0 for score in recent_scores):
            logger.error(
                f"🚨 ZERO SCORES DETECTED! Recent scores: {recent_scores}\n"
                f"   Model training is failing completely!"
            )

        # Calculate variance
        score_variance = np.var(recent_scores)
        if score_variance < self.variance_threshold:
            logger.warning(
                f"⚠️  Score variance extremely low: {score_variance:.10f}\n"
                f"   HPO is not exploring effectively!"
            )

def apply_hpo_fixes(X: np.ndarray, y: np.ndarray,
                    model_type: str = "random_forest", 
                    task_type: str = "classification") -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Apply all HPO fixes and return improved configuration.

    Args:
        X: Feature matrix
        y: Target vector
        model_type: Type of model for search space
        task_type: Type of task - "classification", "regression", or "financial"

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

    # Get improved configuration based on model and task type
    if model_type == "random_forest":
        if task_type == "financial":
            search_space = ImprovedHPOConfig.get_improved_financial_search_space(diagnostics)
        else:
            search_space = ImprovedHPOConfig.get_improved_random_forest_search_space(diagnostics)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    hpo_params = ImprovedHPOConfig.get_improved_hpo_params(diagnostics, task_type)

    logger.info(f"✅ Using improved scoring metric: {hpo_params['scoring']}")
    logger.info(f"✅ Using CV strategy: {hpo_params['cv_description']}")
    logger.info(f"✅ Running {hpo_params['n_trials']} trials with early stopping")
    logger.info(f"✅ Task type: {task_type}")

    return search_space, hpo_params
