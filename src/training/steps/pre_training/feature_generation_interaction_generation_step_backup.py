"""Legacy backup helpers for FeatureGenerationInteractionGenerationStep.

This module reintroduces the historical implementation fragments that the
primary step relies on via ``__getattr__`` delegation.  Only the methods
that have been observed missing at runtime are included here; the main step
binds them dynamically when needed.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.metrics import r2_score
    from sklearn.model_selection import cross_val_score
    lgb_available = True
except ImportError:  # pragma: no cover - fallback environment
    lgb = None  # type: ignore
    MultiOutputRegressor = None  # type: ignore
    r2_score = None  # type: ignore
    cross_val_score = None  # type: ignore
    lgb_available = False

try:
    from src.utils.ml_common.validation.unified_cv import temporal_cross_validation
    overfitting_prevention_available = True
except ImportError:  # pragma: no cover - fallback environment
    temporal_cross_validation = None  # type: ignore
    overfitting_prevention_available = False

from src.utils.tprint import (
    tprint_info,
    tprint_success,
    tprint_warning,
    tprint_error,
)


class FeatureGenerationInteractionGenerationStep:  # pragma: no cover - legacy helper
    """Container for legacy helper implementations."""

    def _get_consistent_sample(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        max_samples: int = 8000,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if features is None or len(features) == 0:
            return features, targets

        if len(features) <= max_samples:
            return features, targets

        np.random.seed(42)
        sample_idx = np.random.choice(len(features), max_samples, replace=False)
        sampled_features = features.iloc[sample_idx]
        sampled_targets = targets.iloc[sample_idx]
        return sampled_features, sampled_targets

    def _fast_mi_proxy(
        self,
        feature: pd.Series,
        target: pd.Series,
        n_bins: int = 5,
    ) -> float:
        try:
            feature_clean = feature.fillna(0)
            target_clean = target.fillna(0)
            if feature_clean.std() == 0 or target_clean.std() == 0:
                return 0.0
            feature_bins = pd.cut(feature_clean, bins=n_bins, labels=False, duplicates="drop")
            target_bins = pd.cut(target_clean, bins=n_bins, labels=False, duplicates="drop")
            valid_mask = ~(pd.isna(feature_bins) | pd.isna(target_bins))
            feature_bins = feature_bins[valid_mask]
            target_bins = target_bins[valid_mask]
            if len(feature_bins) == 0:
                return 0.0
            contingency = pd.crosstab(feature_bins, target_bins, normalize=True)
            feature_marginal = contingency.sum(axis=1)
            target_marginal = contingency.sum(axis=0)
            mi_proxy = 0.0
            for i in range(len(feature_marginal)):
                for j in range(len(target_marginal)):
                    value = contingency.iloc[i, j]
                    if value > 0:
                        mi_proxy += value * np.log2(
                            value / (feature_marginal.iloc[i] * target_marginal.iloc[j] + 1e-8)
                        )
            return max(0.0, mi_proxy)
        except Exception:
            return 0.0

    async def _phase3_2_deeper_refinement(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        config: Dict[str, Any],
    ) -> pd.DataFrame:
        """Legacy implementation of the deeper refinement phase."""
        if not lgb_available:
            raise RuntimeError("LightGBM is required for Phase 3.2 refinement")

        tprint_info("  📊 Training deeper LGBM for refinement...")

        common_indices = features.index.intersection(targets.index)
        if len(common_indices) == 0:
            tprint_error("❌ CRITICAL ERROR: No common indices between features and targets!")
            raise ValueError("No common indices between features and targets")

        overlap_ratio = len(common_indices) / min(len(features), len(targets))
        if overlap_ratio < 0.5:
            tprint_warning(
                f"⚠️ Low index overlap: {overlap_ratio:.3f} - this may affect model performance"
            )

        features_aligned = features.loc[common_indices]
        targets_aligned = targets.loc[common_indices]

        if len(features_aligned) == 0 or len(targets_aligned) == 0:
            raise ValueError("Alignment resulted in empty datasets")

        features_cleaned = features_aligned.fillna(0)
        targets_cleaned = targets_aligned.fillna(0)

        valid_targets = []
        for col in targets_cleaned.columns:
            col_data = targets_cleaned[col]
            variance = col_data.var()
            non_zero_count = (col_data != 0).sum()
            if variance < 1e-10 or non_zero_count == 0:
                tprint_warning(
                    f"⚠️ Target '{col}' is nearly constant (var={variance:.2e}, non-zero={non_zero_count})"
                )
                continue
            valid_targets.append(col)

        if not valid_targets:
            raise ValueError("No valid targets available for model training")

        targets_cleaned = targets_cleaned[valid_targets]

        blank_mode = str(config.get("execution_mode", "")).lower() == "blank"
        max_samples = config.get("max_samples_phase3", 6000 if blank_mode else 8000)
        features_sample, targets_sample = self._get_consistent_sample(
            features_cleaned, targets_cleaned, max_samples=max_samples
        )

        if len(features_sample) > 5000 and hasattr(self, "_chunked_processing"):
            features_sample = self._chunked_processing(features_sample, targets_sample, chunk_size=2000)

        lgbm_params = {
            "max_depth": 3,
            "num_leaves": 10,
            "n_estimators": 80,
            "learning_rate": 0.05,
            "reg_alpha": 0.2,
            "reg_lambda": 0.2,
            "min_child_samples": 80,
            "min_split_gain": 0.02,
            "subsample": 0.6,
            "colsample_bytree": 0.6,
            "max_bin": 255,
            "min_data_per_group": 50,
            "random_state": 42,
            "verbose": -1,
        }

        model = MultiOutputRegressor(lgb.LGBMRegressor(**lgbm_params))
        model.fit(features_sample, targets_sample)

        try:
            if overfitting_prevention_available and temporal_cross_validation is not None:
                cv_results = temporal_cross_validation(
                    model,
                    features_sample,
                    targets_sample,
                    n_splits=5,
                    gap=1,
                    test_size=None,
                    scoring="r2",
                )
                cv_score = cv_results.get("mean", 0.0)
                cv_scores_std = cv_results.get("std", 0.0)
            else:
                scores = cross_val_score(model, features_sample, targets_sample, cv=3, scoring="r2")
                cv_score = float(scores.mean())
                cv_scores_std = float(scores.std())

            predictions = model.predict(features_sample)
            accuracy = r2_score(targets_sample, predictions)

            self._phase3_2_performance = {
                "accuracy": accuracy,
                "cv_score": cv_score,
                "importance_consistency": 0.0,
                "cv_scores_std": cv_scores_std,
                "mean_mi": 0.0,
                "mi_scores": {},
            }
        except Exception as exc:  # pragma: no cover - diagnostic surface
            tprint_error(f"❌ Phase 3.2 performance metrics calculation failed: {exc}")
            raise

        feature_importance = model.estimators_[0].feature_importances_
        mi_scores = []
        for col in features_sample.columns:
            mi_scores.append(
                self._fast_mi_proxy(features_sample[col], targets_sample.iloc[:, 0], n_bins=5)
            )
        mi_scores = np.array(mi_scores)

        stability = np.var(features_sample.values, axis=0)

        imp_scores = (feature_importance - np.min(feature_importance)) / (
            np.max(feature_importance) - np.min(feature_importance) + 1e-8
        )
        mi_scores = (mi_scores - np.min(mi_scores)) / (np.max(mi_scores) - np.min(mi_scores) + 1e-8)
        stab_scores = (stability - np.min(stability)) / (
            np.max(stability) - np.min(stability) + 1e-8
        )

        combined_scores = 0.6 * imp_scores + 0.3 * mi_scores + 0.1 * stab_scores
        feature_scores = pd.Series(combined_scores, index=features.columns).sort_values(ascending=False)
        n_select = min(80, len(features.columns))
        top_features = feature_scores.head(n_select).index.tolist()

        tprint_success(f"  ✅ Selected {len(top_features)} features (top 80) using backup implementation")
        return features[top_features]
