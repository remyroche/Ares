"""
Baseline Predictive Check Module

Evaluate each feature individually via a simple train/test split using a
univariate linear regression.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_auc_score, log_loss
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMRegressor
    LGBM_AVAILABLE = True
except Exception:
    LGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class BaselinePredictiveCheck:
    def __init__(
        self,
        max_features: Optional[int] = None,
        random_state: int = 42,
        test_size: float = 0.25,
        enable_lgbm: bool = True,
    ) -> None:
        self.max_features = max_features
        self.random_state = random_state
        self.test_size = test_size
        self.results: Dict[str, Any] = {}
        self.enable_lgbm = enable_lgbm and LGBM_AVAILABLE
        # Cache last split to reuse for multivariate diagnostics
        self._last_feature_df: Optional[pd.DataFrame] = None
        self._last_target_series: Optional[pd.Series] = None
        self._last_train_idx: Optional[np.ndarray] = None
        self._last_test_idx: Optional[np.ndarray] = None

    def run_check(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        feature_df, target_series, selected_features = self._prepare_data(features, target, feature_names)
        if feature_df is None or target_series is None or feature_df.empty:
            return {
                'success': False,
                'error': 'Failed to prepare data',
                'timestamp': datetime.now().isoformat(),
            }

        per_feature_metrics = self._evaluate_single_features(feature_df, target_series)
        if not per_feature_metrics:
            return {
                'success': False,
                'error': 'No valid numeric features to evaluate',
                'timestamp': datetime.now().isoformat(),
            }

        summary = self._summarize_feature_metrics(per_feature_metrics)
        interpretation = self._build_interpretation(summary)

        # Evaluate small multivariate (2-3 feature) LGBM models using the same split
        multivariate_metrics = self._evaluate_multivariate_combinations(per_feature_metrics)
        multivariate_summary = self._summarize_multivariate_metrics(multivariate_metrics)

        self.results = {
            'success': True,
            'timestamp': datetime.now().isoformat(),
            'data_info': {
                'n_samples': len(feature_df),
                'n_features': len(selected_features),
                'selected_features': selected_features,
                'target_stats': {
                    'mean': float(target_series.mean()),
                    'std': float(target_series.std()),
                    'min': float(target_series.min()),
                    'max': float(target_series.max()),
                },
            },
            'per_feature_metrics': per_feature_metrics,
            'summary': summary,
            'interpretation': interpretation,
            'multivariate_lgbm_metrics': multivariate_metrics,
            'multivariate_summary': multivariate_summary,
        }
        return self.results

    def _prepare_data(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        feature_names: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], List[str]]:
        # IMPROVED DIAGNOSTIC: Log input sizes to help debug sample size issues
        logger.info(f"📊 Baseline check input: features shape={features.shape}, target length={len(target)}")
        
        common_idx = features.index.intersection(target.index)
        if common_idx.empty:
            logger.warning("No overlapping indices between features and target")
            logger.warning(f"   Features index range: {features.index.min()} to {features.index.max()}")
            logger.warning(f"   Target index range: {target.index.min()} to {target.index.max()}")
            return None, None, []

        # DIAGNOSTIC: Log common index size
        logger.info(f"📊 Common index size: {len(common_idx)} (features: {len(features)}, target: {len(target)})")
        if len(common_idx) < len(features) * 0.5:
            logger.warning(f"⚠️ Only {len(common_idx)}/{len(features)} rows aligned - check data pipeline!")

        feature_df = features.loc[common_idx].copy()
        target_series = target.loc[common_idx].copy()

        valid_mask = target_series.notna()
        feature_df = feature_df.loc[valid_mask]
        target_series = target_series.loc[valid_mask]
        
        # DIAGNOSTIC: Log after filtering NaN targets
        nan_targets = (~valid_mask).sum()
        if nan_targets > 0:
            logger.info(f"📊 Removed {nan_targets} rows with NaN targets, {len(feature_df)} rows remain")
        
        # IMPROVED: Warn if sample size is too small for reliable evaluation
        MIN_RECOMMENDED_SAMPLES = 500
        if len(feature_df) < MIN_RECOMMENDED_SAMPLES:
            logger.warning(
                f"⚠️ SAMPLE SIZE WARNING: Only {len(feature_df)} samples available for baseline learnability. "
                f"Recommended minimum is {MIN_RECOMMENDED_SAMPLES}. Results may be unreliable. "
                f"Check data pipeline for: 1) target NaN filtering 2) index alignment 3) temporal windowing"
            )

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_df = feature_df[numeric_cols]
        if feature_df.empty:
            return None, None, []

        # Exclude known pseudo-features and target columns from baseline diagnostics.
        # Treat log_ret, primary_signal, and smoothed_label as non-trainable
        # pseudo-targets for this check.
        excluded_baseline_cols = {
            "adaptive_stop_threshold",
            "close_log_return",
            "close_return",
            "log_ret",
            "primary_signal",
            "smoothed_label",
        }
        remaining_cols = [
            c
            for c in feature_df.columns
            if c not in excluded_baseline_cols and not str(c).startswith("target_")
        ]
        if not remaining_cols:
            return None, None, []
        feature_df = feature_df[remaining_cols]

        if feature_names:
            missing = [col for col in feature_names if col not in feature_df.columns]
            if missing:
                logger.warning("Requested baseline features missing: %s", missing)
            available = [col for col in feature_names if col in feature_df.columns]
            if not available:
                return None, None, []
            feature_df = feature_df[available]

        min_non_na = max(int(len(feature_df) * 0.5), 1)
        feature_df = feature_df.dropna(axis=1, thresh=min_non_na)
        if feature_df.empty:
            return None, None, []

        feature_df = feature_df.fillna(feature_df.mean())

        # Drop near-constant features for baseline diagnostics
        varying_mask = feature_df.std() > 1e-5
        feature_df = feature_df.loc[:, varying_mask]
        if feature_df.empty:
            return None, None, []

        feature_list = feature_df.columns.tolist()
        if self.max_features and len(feature_list) > self.max_features:
            rng = np.random.default_rng(self.random_state)
            feature_list = rng.choice(feature_list, size=self.max_features, replace=False).tolist()
            feature_df = feature_df[feature_list]

        return feature_df, target_series, feature_list

    def _evaluate_single_features(
        self,
        feature_df: pd.DataFrame,
        target_series: pd.Series,
    ) -> List[Dict[str, Any]]:
        metrics: List[Dict[str, Any]] = []
        if feature_df.empty:
            return metrics

        indices = np.arange(len(feature_df))
        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.test_size,
            random_state=self.random_state,
        )
        y_train = target_series.iloc[train_idx]
        y_test = target_series.iloc[test_idx]

        target_is_binary = False
        try:
            unique_vals = target_series.dropna().unique()
            if len(unique_vals) == 2:
                vals = set(float(v) for v in unique_vals)
                if vals.issubset({0.0, 1.0}):
                    target_is_binary = True
        except Exception:
            target_is_binary = False

        # Cache split and data for multivariate diagnostics
        self._last_feature_df = feature_df
        self._last_target_series = target_series
        self._last_train_idx = train_idx
        self._last_test_idx = test_idx

        for column in feature_df.columns:
            series = feature_df[column]
            std = float(series.std())
            if std <= 1e-10:
                continue

            model = LinearRegression()
            X_train = series.iloc[train_idx].values.reshape(-1, 1)
            X_test = series.iloc[test_idx].values.reshape(-1, 1)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)
            # Older scikit-learn versions do not support the `squared` keyword, so compute
            # RMSE manually as the square root of MSE for compatibility.
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)
            train_rmse = float(np.sqrt(train_mse))
            test_rmse = float(np.sqrt(test_mse))
            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            clf_train_auc = None
            clf_test_auc = None
            clf_train_logloss = None
            clf_test_logloss = None
            if target_is_binary:
                try:
                    y_train_bin = y_train.astype(float)
                    y_test_bin = y_test.astype(float)
                    raw_train = series.iloc[train_idx].values.astype(float)
                    raw_test = series.iloc[test_idx].values.astype(float)
                    proba_train = 1.0 / (1.0 + np.exp(-raw_train))
                    proba_test = 1.0 / (1.0 + np.exp(-raw_test))
                    proba_train = np.clip(proba_train, 1e-6, 1.0 - 1e-6)
                    proba_test = np.clip(proba_test, 1e-6, 1.0 - 1e-6)
                    if np.unique(y_train_bin).size == 2:
                        clf_train_auc = float(roc_auc_score(y_train_bin, proba_train))
                        clf_train_logloss = float(log_loss(y_train_bin, proba_train, labels=[0.0, 1.0]))
                    if np.unique(y_test_bin).size == 2:
                        clf_test_auc = float(roc_auc_score(y_test_bin, proba_test))
                        clf_test_logloss = float(log_loss(y_test_bin, proba_test, labels=[0.0, 1.0]))
                except Exception:
                    clf_train_auc = None
                    clf_test_auc = None
                    clf_train_logloss = None
                    clf_test_logloss = None

            lgbm_train_r2 = None
            lgbm_test_r2 = None
            lgbm_train_rmse = None
            lgbm_test_rmse = None
            lgbm_train_mae = None
            lgbm_test_mae = None
            if self.enable_lgbm:
                try:
                    lgbm_model = LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.05,
                        max_depth=3,
                        num_leaves=15,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        min_child_samples=max(10, int(len(X_train) * 0.05)),
                        random_state=self.random_state,
                        n_jobs=1,
                    )
                    lgbm_model.fit(X_train, y_train)
                    y_train_pred_lgbm = lgbm_model.predict(X_train)
                    y_test_pred_lgbm = lgbm_model.predict(X_test)
                    lgbm_train_r2 = float(r2_score(y_train, y_train_pred_lgbm))
                    lgbm_test_r2 = float(r2_score(y_test, y_test_pred_lgbm))
                    lgbm_train_mse = mean_squared_error(y_train, y_train_pred_lgbm)
                    lgbm_test_mse = mean_squared_error(y_test, y_test_pred_lgbm)
                    lgbm_train_rmse = float(np.sqrt(lgbm_train_mse))
                    lgbm_test_rmse = float(np.sqrt(lgbm_test_mse))
                    lgbm_train_mae = float(mean_absolute_error(y_train, y_train_pred_lgbm))
                    lgbm_test_mae = float(mean_absolute_error(y_test, y_test_pred_lgbm))
                except Exception as exc:
                    logger.warning("LGBM evaluation failed for feature %s: %s", column, exc)

            pearson_corr = float(series.corr(target_series))
            spearman_corr = float(series.rank().corr(target_series.rank()))

            if target_is_binary and clf_test_auc is not None:
                quality_score = float(clf_test_auc)
            else:
                quality_score = max(test_r2, 0.0) * 0.6 + abs(pearson_corr) * 0.4

            metrics.append(
                {
                    'feature': column,
                    'train_r2': float(train_r2),
                    'test_r2': float(test_r2),
                    'train_rmse': float(train_rmse),
                    'test_rmse': float(test_rmse),
                    'train_mae': float(train_mae),
                    'test_mae': float(test_mae),
                    'lgbm_train_r2': lgbm_train_r2,
                    'lgbm_test_r2': lgbm_test_r2,
                    'lgbm_train_rmse': lgbm_train_rmse,
                    'lgbm_test_rmse': lgbm_test_rmse,
                    'lgbm_train_mae': lgbm_train_mae,
                    'lgbm_test_mae': lgbm_test_mae,
                    'pearson_corr': pearson_corr,
                    'spearman_corr': spearman_corr,
                    'clf_train_auc': clf_train_auc,
                    'clf_test_auc': clf_test_auc,
                    'clf_train_logloss': clf_train_logloss,
                    'clf_test_logloss': clf_test_logloss,
                    'quality_score': float(quality_score),
                }
            )

        metrics.sort(key=lambda item: item['quality_score'], reverse=True)
        return metrics

    def _evaluate_multivariate_combinations(
        self,
        per_feature_metrics: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Evaluate small multivariate (2-3 feature) LGBM models.

        Uses the same cached train/test split as the univariate baseline to
        assess the combined predictive power of top-ranked features.
        """
        if not (self.enable_lgbm and LGBM_AVAILABLE):
            return []

        if (
            self._last_feature_df is None
            or self._last_target_series is None
            or self._last_train_idx is None
            or self._last_test_idx is None
        ):
            return []

        feature_df = self._last_feature_df
        target_series = self._last_target_series
        train_idx = self._last_train_idx
        test_idx = self._last_test_idx

        # Defensive: ensure indices are numpy arrays for iloc
        train_idx = np.asarray(train_idx)
        test_idx = np.asarray(test_idx)

        y_train = target_series.iloc[train_idx]
        y_test = target_series.iloc[test_idx]

        # Use the top-K single features by quality score as candidates
        if not per_feature_metrics:
            return []

        top_k = min(8, len(per_feature_metrics))
        candidate_features = [m['feature'] for m in per_feature_metrics[:top_k]]

        results: List[Dict[str, Any]] = []
        for size in (2, 3):
            if len(candidate_features) < size:
                continue
            for combo in combinations(candidate_features, size):
                cols = list(combo)
                try:
                    X_train = feature_df.iloc[train_idx][cols].values
                    X_test = feature_df.iloc[test_idx][cols].values

                    model = LGBMRegressor(
                        n_estimators=150,
                        learning_rate=0.05,
                        max_depth=3,
                        num_leaves=15,
                        subsample=0.8,
                        colsample_bytree=1.0,
                        min_child_samples=max(10, int(len(X_train) * 0.05)),
                        random_state=self.random_state,
                        n_jobs=1,
                    )
                    model.fit(X_train, y_train)

                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    train_r2 = float(r2_score(y_train, y_train_pred))
                    test_r2 = float(r2_score(y_test, y_test_pred))
                    train_mse = mean_squared_error(y_train, y_train_pred)
                    test_mse = mean_squared_error(y_test, y_test_pred)
                    train_rmse = float(np.sqrt(train_mse))
                    test_rmse = float(np.sqrt(test_mse))
                    train_mae = float(mean_absolute_error(y_train, y_train_pred))
                    test_mae = float(mean_absolute_error(y_test, y_test_pred))

                    results.append(
                        {
                            'features': cols,
                            'size': size,
                            'train_r2': train_r2,
                            'test_r2': test_r2,
                            'train_rmse': train_rmse,
                            'test_rmse': test_rmse,
                            'train_mae': train_mae,
                            'test_mae': test_mae,
                        }
                    )
                except Exception as exc:
                    logger.warning("Multivariate LGBM evaluation failed for %s: %s", combo, exc)
                    continue

        results.sort(key=lambda m: m['test_r2'], reverse=True)
        return results

    @staticmethod
    def _summarize_feature_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        best = metrics[0]
        positive = [m for m in metrics if m['test_r2'] > 0]
        avg_test_r2 = float(np.mean([m['test_r2'] for m in metrics])) if metrics else 0.0
        median_test_r2 = float(np.median([m['test_r2'] for m in metrics])) if metrics else 0.0

        lgbm_best_feature = None
        lgbm_best_test_r2 = None
        lgbm_metrics = [m for m in metrics if m.get('lgbm_test_r2') is not None]
        if lgbm_metrics:
            best_lgbm = max(lgbm_metrics, key=lambda m: m['lgbm_test_r2'])
            lgbm_best_feature = best_lgbm['feature']
            lgbm_best_test_r2 = best_lgbm['lgbm_test_r2']

        clf_best_feature = None
        clf_best_test_auc = None
        clf_best_test_logloss = None
        clf_metrics = [m for m in metrics if m.get('clf_test_auc') is not None]
        if clf_metrics:
            best_clf = max(clf_metrics, key=lambda m: m['clf_test_auc'])
            clf_best_feature = best_clf['feature']
            clf_best_test_auc = best_clf['clf_test_auc']
            clf_best_test_logloss = best_clf.get('clf_test_logloss')

        summary: Dict[str, Any] = {
            'best_feature': best['feature'],
            'best_test_r2': best['test_r2'],
            'best_quality_score': best['quality_score'],
            'positive_features': len(positive),
            'positive_ratio': len(positive) / max(len(metrics), 1),
            'avg_test_r2': avg_test_r2,
            'median_test_r2': median_test_r2,
            'top_features': metrics[: min(10, len(metrics))],
        }
        if lgbm_best_feature is not None and lgbm_best_test_r2 is not None:
            summary['lgbm_best_feature'] = lgbm_best_feature
            summary['lgbm_best_test_r2'] = float(lgbm_best_test_r2)
        if clf_best_feature is not None and clf_best_test_auc is not None:
            summary['clf_best_feature'] = clf_best_feature
            summary['clf_best_test_auc'] = float(clf_best_test_auc)
            if clf_best_test_logloss is not None:
                summary['clf_best_test_logloss'] = float(clf_best_test_logloss)
        return summary

    @staticmethod
    def _summarize_multivariate_metrics(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Summarize multivariate LGBM diagnostics.

        Returns best pair and best triplet (by Test R²) if available.
        """
        summary: Dict[str, Any] = {}
        if not metrics:
            return summary

        pairs = [m for m in metrics if m.get('size') == 2]
        triplets = [m for m in metrics if m.get('size') == 3]

        def _best(items: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
            return max(items, key=lambda m: m['test_r2']) if items else None

        best_pair = _best(pairs)
        best_triplet = _best(triplets)

        if best_pair:
            summary['best_pair_features'] = best_pair['features']
            summary['best_pair_test_r2'] = float(best_pair['test_r2'])

        if best_triplet:
            summary['best_triplet_features'] = best_triplet['features']
            summary['best_triplet_test_r2'] = float(best_triplet['test_r2'])

        summary['num_pairs_evaluated'] = len(pairs)
        summary['num_triplets_evaluated'] = len(triplets)
        return summary

    @staticmethod
    def _build_interpretation(summary: Dict[str, Any]) -> Dict[str, Any]:
        quality_score = summary['best_quality_score']
        positive_ratio = summary['positive_ratio']

        if quality_score > 0.5:
            summary_text = "✅ Strong individual signals detected"
        elif positive_ratio > 0.25:
            summary_text = "⚠️ Moderate predictive signals detected"
        else:
            summary_text = "⚠️ Weak predictive signals"

        insights = [
            f"Best feature `{summary['best_feature']}` achieved Test R² = {summary['best_test_r2']:.3f}",
            f"Positive Test R² features: {summary['positive_features']} ({positive_ratio:.1%})",
            f"Median Test R² across evaluated features: {summary['median_test_r2']:.3f}",
        ]
        if 'lgbm_best_feature' in summary and 'lgbm_best_test_r2' in summary:
            insights.append(
                f"LGBM best feature `{summary['lgbm_best_feature']}` achieved Test R² = {summary['lgbm_best_test_r2']:.3f}"
            )

        recommendations: List[str] = []
        if positive_ratio < 0.2:
            recommendations.append("Consider revisiting labeling/target definitions; very few features carry signal")
        if summary['best_test_r2'] < 0:
            recommendations.append("Even the best single feature underperforms; investigate data leakage or excessive noise")
        if not recommendations:
            recommendations.append("Focus downstream modeling on the top-ranked features")

        return {
            'quality_score': float(min(max(quality_score, 0.0), 1.0)),
            'summary': summary_text,
            'interpretations': insights,
            'recommendations': recommendations,
        }

    def save_results_to_csv(self, output_dir: Path, filename_prefix: str = "baseline_check") -> str:
        if not self.results or not self.results.get('success', False):
            logger.error("No baseline results available to export")
            return ""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        rows = []
        data_info = self.results.get('data_info', {})
        rows.append({'metric_category': 'data_info', 'metric_name': 'n_samples', 'value': data_info.get('n_samples', 0)})
        rows.append({'metric_category': 'data_info', 'metric_name': 'n_features', 'value': data_info.get('n_features', 0)})

        per_feature_metrics = self.results.get('per_feature_metrics', [])
        for metric in per_feature_metrics:
            rows.append({
                'metric_category': 'per_feature',
                'metric_name': metric['feature'],
                'value': metric['test_r2'],
            })

        for metric in per_feature_metrics:
            lgbm_test_r2 = metric.get('lgbm_test_r2')
            if lgbm_test_r2 is not None:
                rows.append({
                    'metric_category': 'per_feature_lgbm',
                    'metric_name': metric['feature'],
                    'value': lgbm_test_r2,
                })

        for metric in per_feature_metrics:
            clf_auc = metric.get('clf_test_auc')
            if clf_auc is not None:
                rows.append({
                    'metric_category': 'per_feature_auc',
                    'metric_name': metric['feature'],
                    'value': clf_auc,
                })

        for metric in per_feature_metrics:
            clf_logloss = metric.get('clf_test_logloss')
            if clf_logloss is not None:
                rows.append({
                    'metric_category': 'per_feature_logloss',
                    'metric_name': metric['feature'],
                    'value': clf_logloss,
                })

        summary = self.results.get('summary', {})
        rows.append({'metric_category': 'summary', 'metric_name': 'best_feature', 'value': summary.get('best_feature', '')})
        rows.append({'metric_category': 'summary', 'metric_name': 'best_test_r2', 'value': summary.get('best_test_r2', 0)})
        if 'lgbm_best_feature' in summary:
            rows.append({'metric_category': 'summary', 'metric_name': 'lgbm_best_feature', 'value': summary.get('lgbm_best_feature', '')})
        if 'lgbm_best_test_r2' in summary:
            rows.append({'metric_category': 'summary', 'metric_name': 'lgbm_best_test_r2', 'value': summary.get('lgbm_best_test_r2', 0)})

        # Compact learnability summary
        rows.append({'metric_category': 'summary', 'metric_name': 'best_quality_score', 'value': summary.get('best_quality_score', 0)})
        rows.append({'metric_category': 'summary', 'metric_name': 'positive_features', 'value': summary.get('positive_features', 0)})
        rows.append({'metric_category': 'summary', 'metric_name': 'positive_ratio', 'value': summary.get('positive_ratio', 0.0)})
        rows.append({'metric_category': 'summary', 'metric_name': 'avg_test_r2', 'value': summary.get('avg_test_r2', 0.0)})
        rows.append({'metric_category': 'summary', 'metric_name': 'median_test_r2', 'value': summary.get('median_test_r2', 0.0)})

        multivariate_summary = self.results.get('multivariate_summary', {})
        if isinstance(multivariate_summary, dict) and multivariate_summary:
            if 'best_pair_test_r2' in multivariate_summary:
                rows.append({
                    'metric_category': 'multivariate_summary',
                    'metric_name': 'best_pair_test_r2',
                    'value': multivariate_summary.get('best_pair_test_r2', 0.0),
                })
            if 'best_triplet_test_r2' in multivariate_summary:
                rows.append({
                    'metric_category': 'multivariate_summary',
                    'metric_name': 'best_triplet_test_r2',
                    'value': multivariate_summary.get('best_triplet_test_r2', 0.0),
                })
            rows.append({
                'metric_category': 'multivariate_summary',
                'metric_name': 'num_pairs_evaluated',
                'value': multivariate_summary.get('num_pairs_evaluated', 0),
            })
            rows.append({
                'metric_category': 'multivariate_summary',
                'metric_name': 'num_triplets_evaluated',
                'value': multivariate_summary.get('num_triplets_evaluated', 0),
            })

        interpretation = self.results.get('interpretation', {})
        rows.append({'metric_category': 'interpretation', 'metric_name': 'quality_score', 'value': interpretation.get('quality_score', 0)})
        rows.append({'metric_category': 'interpretation', 'metric_name': 'summary', 'value': interpretation.get('summary', '')})

        df = pd.DataFrame(rows)
        df['timestamp'] = self.results.get('timestamp', datetime.now().isoformat())
        df.to_csv(filepath, index=False)
        return str(filepath)

    def save_multivariate_results_to_csv(self, output_dir: Path, filename_prefix: str = "multivariate_baseline") -> str:
        """Save small multivariate LGBM diagnostics to a dedicated CSV file."""
        if not self.results or not self.results.get('success', False):
            logger.error("No baseline results available to export (multivariate)")
            return ""

        metrics = self.results.get('multivariate_lgbm_metrics', [])
        if not metrics:
            return ""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        rows = []
        for item in metrics:
            rows.append(
                {
                    'model_size': item.get('size', 0),
                    'features': ",".join(item.get('features', [])),
                    'train_r2': item.get('train_r2', 0.0),
                    'test_r2': item.get('test_r2', 0.0),
                    'train_rmse': item.get('train_rmse', 0.0),
                    'test_rmse': item.get('test_rmse', 0.0),
                    'train_mae': item.get('train_mae', 0.0),
                    'test_mae': item.get('test_mae', 0.0),
                }
            )

        df = pd.DataFrame(rows)
        df['timestamp'] = self.results.get('timestamp', datetime.now().isoformat())
        df.to_csv(filepath, index=False)
        return str(filepath)

    def format_for_markdown(self) -> str:
        if not self.results or not self.results.get('success', False):
            return "## Baseline Predictive Check\n\n❌ Check failed or not run\n"

        md = "## Baseline Predictive Check\n\n"
        data_info = self.results.get('data_info', {})
        md += f"**Dataset:** {data_info.get('n_samples', 0)} samples, {data_info.get('n_features', 0)} features\n\n"

        md += "### Top Single-Feature Signals\n\n"
        md += "| Rank | Feature | Test R² | Pearson | AUC | Quality Score |\n"
        md += "|------|---------|---------|---------|-----|---------------|\n"
        top_features = self.results.get('summary', {}).get('top_features', [])
        for idx, metric in enumerate(top_features[:5], 1):
            auc_val = metric.get('clf_test_auc')
            auc_str = f"{auc_val:.3f}" if isinstance(auc_val, (int, float)) else "N/A"
            md += (
                f"| {idx} | `{metric['feature']}` | {metric['test_r2']:.3f} | "
                f"{metric['pearson_corr']:.3f} | {auc_str} | {metric['quality_score']:.3f} |\n"
            )
        md += "\n"

        # Small multivariate LGBM baseline (2-3 feature combinations)
        multivariate_summary = self.results.get('multivariate_summary', {})
        multivariate_metrics = self.results.get('multivariate_lgbm_metrics', [])
        if multivariate_metrics:
            md += "### Small Multivariate LGBM Baseline\n\n"
            md += "| Type | Features | Test R² |\n"
            md += "|------|----------|---------|\n"

            best_pair_features = multivariate_summary.get('best_pair_features')
            if best_pair_features:
                pair_r2 = multivariate_summary.get('best_pair_test_r2', 0.0)
                features_str = ", ".join(f"`{f}`" for f in best_pair_features)
                md += f"| Pair | {features_str} | {pair_r2:.3f} |\n"

            best_triplet_features = multivariate_summary.get('best_triplet_features')
            if best_triplet_features:
                triplet_r2 = multivariate_summary.get('best_triplet_test_r2', 0.0)
                features_str = ", ".join(f"`{f}`" for f in best_triplet_features)
                md += f"| Triplet | {features_str} | {triplet_r2:.3f} |\n"

            md += "\n"

        interpretation = self.results.get('interpretation', {})
        md += "### Interpretation\n\n"
        md += f"**Quality Score:** {interpretation.get('quality_score', 0):.2f}/1.0\n\n"
        md += f"**Summary:** {interpretation.get('summary', 'N/A')}\n\n"
        if interpretation.get('interpretations'):
            md += "**Insights:**\n"
            for insight in interpretation['interpretations']:
                md += f"- {insight}\n"
            md += "\n"
        if interpretation.get('recommendations'):
            md += "**Recommendations:**\n"
            for rec in interpretation['recommendations']:
                md += f"- {rec}\n"
            md += "\n"
        return md


def run_baseline_check(
    features: pd.DataFrame,
    target: pd.Series,
    max_features: Optional[int] = None,
    output_dir: Optional[Path] = None,
    save_csv: bool = True,
) -> Dict[str, Any]:
    checker = BaselinePredictiveCheck(max_features=max_features)
    results = checker.run_check(features, target)

    if save_csv and output_dir and results.get('success', False):
        checker.save_results_to_csv(output_dir)

    return results
