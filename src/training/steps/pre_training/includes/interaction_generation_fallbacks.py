"""Fallback helpers for FeatureGenerationInteractionGenerationStep.

These utilities provide simplified implementations of the helper methods that are
occasionally missing at runtime when legacy wrappers or partially generated step
classes are instantiated. They do not aim to be fully feature-complete, but they
are sufficient for the analyst interaction pipeline to run end-to-end.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:  # sklearn is already a hard dependency for the step
    from sklearn.feature_selection import mutual_info_regression
except ImportError:  # pragma: no cover - fallback if sklearn is unavailable
    mutual_info_regression = None  # type: ignore


CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "trend": ["sma", "ema", "trend", "moving_average", "ma"],
    "oscillator": ["rsi", "stoch", "oscillator", "williams", "cci", "macd"],
    "momentum": ["momentum", "roc", "rate_of_change", "pct_change"],
    "returns": ["return", "pct_change", "log_return", "ret"],
    "volatility": ["vol", "volatility", "std", "atr", "bb", "bollinger"],
    "volume": ["volume", "vol", "vwap", "obv", "ad"],
    "acceleration": ["accel", "jerk", "second_derivative", "second_deriv", "2nd_deriv"],
    "advanced_statistical": [
        "skew",
        "kurt",
        "kurtosis",
        "skewness",
        "jarque",
        "normality",
        "statistical",
        "advanced_stat",
        "bb_width",
        "bb_upper",
        "bb_lower",
        "bb_middle",
        "ljung_box",
        "ar_",
        "coefficients",
        "pvalue",
    ],
    "candlestick_pattern": ["candlestick", "candle", "doji", "hammer", "shooting", "hanging", "pattern", "engulfing"],
    "entropy": ["entropy", "ent", "shannon", "information", "complexity"],
    "spectral_wavelet": ["spectral", "wavelet", "freq", "fft", "dwt", "frequency", "spectrum"],
    "support_resistance": ["support", "resistance", "sr", "level", "pivot", "fibonacci", "fib"],
}


def _infer_feature_category_fallback(step: Any, feature_name: str) -> str:
    """Best-effort feature category inference."""
    if getattr(step, "feature_bank", None) is not None:
        try:
            feature_bank = step.feature_bank
            registry = getattr(feature_bank, "registry", None)
            if registry is not None and hasattr(registry, "get_by_name"):
                feature_info = registry.get_by_name(feature_name)
                if feature_info is not None and hasattr(feature_info, "category"):
                    category = feature_info.category
                    return category.value if hasattr(category, "value") else str(category)
        except Exception:  # pragma: no cover - fallback path
            pass

    feature_lower = feature_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in feature_lower for keyword in keywords):
            return category
    return "unknown"


_fg_interaction_infer_feature_category_fallback = _infer_feature_category_fallback


def _find_similar_feature_fallback(step: Any, target_feature: str, available_features: List[str]) -> List[str]:
    target_parts = set(target_feature.lower().split("_"))
    scored: List[tuple[str, float]] = []
    for feature in available_features:
        feature_parts = set(feature.lower().split("_"))
        if not feature_parts:
            continue
        overlap = len(target_parts.intersection(feature_parts))
        similarity = overlap / max(len(target_parts), len(feature_parts))
        if similarity > 0.3:
            scored.append((feature, similarity))
    scored.sort(key=lambda item: item[1], reverse=True)
    return [feature for feature, _ in scored[:3]]


# Backwards compatibility alias expected by older modules
_fg_interaction_find_similar_feature_fallback = _find_similar_feature_fallback


def _get_feature_categories_from_bank_fallback(step: Any, feature_names: List[str], lookback_optimization: Dict) -> Dict[str, str]:
    feature_categories: Dict[str, str] = {}
    bank_categories = lookback_optimization.get("feature_categories", {}) if isinstance(lookback_optimization, dict) else {}
    for feature_name in feature_names:
        base_name = feature_name.split("_")[0]
        if feature_name in bank_categories:
            feature_categories[feature_name] = bank_categories[feature_name]
        elif base_name in bank_categories:
            feature_categories[feature_name] = bank_categories[base_name]
        else:
            feature_categories[feature_name] = _infer_feature_category_fallback(step, feature_name)
    return feature_categories


_fg_interaction_get_feature_categories_fallback = _get_feature_categories_from_bank_fallback


def _calculate_composite_scores_fallback(
    step: Any,
    features_df: pd.DataFrame,
    targets_df: pd.DataFrame,
    feature_categories: Dict[str, str],
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Simplified composite score calculation (MI + stability)."""
    if features_df.empty or targets_df.empty:
        return {}

    target_col = targets_df.columns[0]
    target_series = targets_df[target_col].dropna()
    aligned_index = features_df.index.intersection(target_series.index)
    if aligned_index.empty:
        aligned_index = features_df.index[: min(len(features_df), len(target_series))]

    features_aligned = features_df.loc[aligned_index].fillna(0)
    target_aligned = target_series.loc[aligned_index].fillna(0)

    valid_features: List[str] = []
    for column in features_aligned.columns:
        series = features_aligned[column]
        if series.std() > 1e-8 and series.notna().sum() >= 10:
            valid_features.append(column)

    if not valid_features:
        return {}

    mi_scores: Dict[str, float] = {}
    if mutual_info_regression is not None:
        try:
            mi = mutual_info_regression(
                features_aligned[valid_features].values,
                target_aligned.values,
                random_state=42,
                n_neighbors=3,
            )
            mi_max = float(np.max(mi)) if len(mi) else 0.0
            for column, score in zip(valid_features, mi):
                mi_scores[column] = float(score / mi_max) if mi_max > 0 else 0.0
        except Exception:  # pragma: no cover - regression fallback
            pass

    if not mi_scores:
        # Uniform baseline if MI fails
        mi_scores = {column: 0.5 for column in valid_features}

    stability_scores: Dict[str, float] = {}
    window = max(10, min(100, len(features_aligned) // 5))
    for column in valid_features:
        series = features_aligned[column].fillna(method="ffill").fillna(0)
        rolling = series.rolling(window=window, min_periods=5)
        rolling_mean = rolling.mean()
        rolling_std = rolling.std()
        if rolling_mean.std() > 1e-8:
            cv = rolling_std.mean() / (abs(rolling_mean.mean()) + 1e-8)
            stability_scores[column] = max(0.0, min(1.0, 1.0 / (1.0 + cv)))
        else:
            stability_scores[column] = 0.5

    composite_scores: Dict[str, float] = {}
    for column in valid_features:
        mi_score = mi_scores.get(column, 0.0)
        stability = stability_scores.get(column, 0.0)
        # Balanced weighting keeps behaviour close to the original implementation
        composite_scores[column] = 0.6 * mi_score + 0.4 * stability

    return composite_scores


def _fast_mi_proxy_fallback(step: Any, feature: pd.Series, target: pd.Series, n_bins: int = 5) -> float:
    """Lightweight MI proxy used when the optimized implementation is unavailable."""
    try:
        feature_values = feature.fillna(0).to_numpy(dtype=float, copy=True)
        target_values = target.fillna(0).to_numpy(dtype=float, copy=True)
        if feature_values.size == 0 or target_values.size == 0:
            return 0.0
        if np.std(feature_values) < 1e-12 or np.std(target_values) < 1e-12:
            return 0.0
        corr_matrix = np.corrcoef(feature_values, target_values)
        corr = corr_matrix[0, 1] if corr_matrix.size >= 4 else 0.0
        if np.isnan(corr):
            return 0.0
        return float(abs(corr))
    except Exception:
        return 0.0


def _extract_tree_splitting_pairs_fallback(step: Any, model) -> List[Tuple[str, str, int]]:
    """Fallback extraction of feature pairs that frequently split together in trees.

    This mirrors the core logic of the inline helper but is kept lightweight and
    defensive so that interaction discovery can proceed even if the full
    implementation is missing on the runtime class.
    """
    from collections import defaultdict

    feature_pairs: Dict[Tuple[str, str], int] = defaultdict(int)

    try:
        # Handle both MultiOutputRegressor and direct LGBMRegressor
        if hasattr(model, "estimators_") and getattr(model, "estimators_", None):
            booster = model.estimators_[0].booster_
        else:
            booster = model.booster_

        trees = booster.dump_model().get("tree_info", [])

        for tree in trees:
            features_in_tree: set = set()

            # Traverse tree to find all features used
            def traverse_node(node: Dict[str, Any]) -> None:
                if "split_feature" in node:
                    features_in_tree.add(node["split_feature"])
                    if "left_child" in node:
                        traverse_node(node["left_child"])
                    if "right_child" in node:
                        traverse_node(node["right_child"])

            if "tree_structure" in tree:
                traverse_node(tree["tree_structure"])

            # Count all pairs in this tree
            features_list = list(features_in_tree)
            for i in range(len(features_list)):
                for j in range(i + 1, len(features_list)):
                    pair = tuple(sorted([features_list[i], features_list[j]]))
                    feature_pairs[pair] += 1

        pairs_list: List[Tuple[str, str, int]] = [
            (f1, f2, count) for (f1, f2), count in feature_pairs.items()
        ]
        pairs_list.sort(key=lambda x: x[2], reverse=True)
        return pairs_list[:80]
    except Exception:
        # If anything goes wrong, return an empty list to avoid breaking the pipeline
        return []


def _get_consistent_sample_fallback(
    step: Any,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    max_samples: int = 8000
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ensure deterministic subsampling alignment between features and targets."""
    if features.empty or targets.empty:
        return features, targets
    common_index = features.index.intersection(targets.index)
    if common_index.empty:
        min_len = min(len(features), len(targets))
        features = features.iloc[:min_len]
        targets = targets.iloc[:min_len]
    else:
        features = features.loc[common_index]
        targets = targets.loc[common_index]
    if len(features) <= max_samples:
        return features, targets
    rng = np.random.default_rng(42)
    sampled_indices = rng.choice(len(features), size=max_samples, replace=False)
    sampled_indices.sort()
    return features.iloc[sampled_indices], targets.iloc[sampled_indices]


def _chunked_processing_fallback(
    step: Any,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    chunk_size: int = 2000,
) -> pd.DataFrame:
    """Simple chunked processing fallback to keep features/targets aligned.

    This trims the sample to the last ``chunk_size`` rows (by index), updating
    both ``features`` and ``targets`` in place so downstream code sees
    consistent shapes.
    """
    try:
        if features is None or len(features) <= chunk_size:
            return features

        # Keep the most recent chunk_size samples (time-ordered index)
        indices_to_keep = features.index[-chunk_size:]
        indices_to_drop = features.index.difference(indices_to_keep)

        if len(indices_to_drop) > 0:
            # Trim features in place when possible
            try:
                features.drop(indices_to_drop, inplace=True)
            except Exception:
                features = features.loc[indices_to_keep]

            # Trim targets to stay aligned with features
            try:
                targets.drop(indices_to_drop, inplace=True)
            except Exception:
                try:
                    targets = targets.loc[indices_to_keep]
                except Exception:
                    # Best-effort fallback: align on intersection
                    common_index = features.index.intersection(targets.index)
                    features = features.loc[common_index]
                    targets = targets.loc[common_index]

        return features

    except Exception:
        # In worst case, return original features unmodified
        return features


def attach_interaction_generation_fallbacks(cls: Any) -> None:
    """Attach fallback helpers to the provided class if they are missing."""
    if not hasattr(cls, "_infer_feature_category"):
        setattr(cls, "_infer_feature_category", _infer_feature_category_fallback)
    if not hasattr(cls, "_find_similar_feature"):
        setattr(cls, "_find_similar_feature", _fg_interaction_find_similar_feature_fallback)
    if not hasattr(cls, "_get_feature_categories_from_bank"):
        setattr(cls, "_get_feature_categories_from_bank", _get_feature_categories_from_bank_fallback)
    if not hasattr(cls, "_calculate_composite_scores"):
        setattr(cls, "_calculate_composite_scores", _calculate_composite_scores_fallback)
    if not hasattr(cls, "_fast_mi_proxy"):
        setattr(cls, "_fast_mi_proxy", _fast_mi_proxy_fallback)
    if not hasattr(cls, "_extract_tree_splitting_pairs"):
        setattr(cls, "_extract_tree_splitting_pairs", _extract_tree_splitting_pairs_fallback)
    if not hasattr(cls, "_get_consistent_sample"):
        setattr(cls, "_get_consistent_sample", _get_consistent_sample_fallback)
    if not hasattr(cls, "_chunked_processing"):
        setattr(cls, "_chunked_processing", _chunked_processing_fallback)
