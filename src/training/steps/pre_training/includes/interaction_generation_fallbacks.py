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
    """Simplified composite score calculation (MI + stability).

    This fallback mirrors the quick per-feature correlation-based MI/stability
    proxy used in Phase 4, avoiding the strict "all-features non-NaN"
    requirement of mutual_info_regression.
    """

    if features_df.empty or targets_df.empty:
        return {}

    try:
        target_col = targets_df.columns[0]
        target_series = pd.to_numeric(targets_df[target_col], errors="coerce")
    except Exception:
        return {}

    # Align on index intersection; if none, fall back to positional alignment on
    # the most recent overlapping window.
    common_idx = features_df.index.intersection(target_series.index)
    if len(common_idx) == 0:
        common_len = min(len(features_df), len(target_series))
        if common_len == 0:
            return {}
        features_aligned = features_df.iloc[-common_len:].copy()
        target_aligned = target_series.iloc[-common_len:].copy()
        features_aligned.index = pd.RangeIndex(common_len)
        target_aligned.index = pd.RangeIndex(common_len)
    else:
        features_aligned = features_df.loc[common_idx].copy()
        target_aligned = target_series.loc[common_idx].copy()

    y_clean = pd.to_numeric(target_aligned, errors="coerce")

    cfg = config or {}
    min_samples = int(cfg.get("mi_quick_min_samples", 200))
    subsample_ratio = float(cfg.get("mi_quick_subsample_ratio", 0.30))
    rng = np.random.default_rng(int(cfg.get("mi_quick_random_state", 42)))

    # Per-feature correlation-based MI proxy
    mi_raw: Dict[str, float] = {}
    for column in features_aligned.columns:
        series = pd.to_numeric(features_aligned[column], errors="coerce")
        mask = series.notna() & y_clean.notna()
        n_valid = int(mask.sum())
        if n_valid < min_samples:
            continue

        s_vals = series[mask]
        t_vals = y_clean[mask]

        if 0.0 < subsample_ratio < 1.0 and n_valid > min_samples:
            n_sub = max(min_samples, int(n_valid * subsample_ratio))
            idx = rng.choice(np.arange(n_valid), size=n_sub, replace=False)
            s_vals = s_vals.iloc[idx]
            t_vals = t_vals.iloc[idx]

        corr = s_vals.corr(t_vals)
        if not np.isfinite(corr):
            continue
        corr_abs = abs(corr)
        if corr_abs >= 0.999:
            mi_raw[column] = 5.0
        else:
            mi_raw[column] = float(-0.5 * np.log(1.0 - corr_abs ** 2))

    if mi_raw:
        mi_vals = np.array(list(mi_raw.values()), dtype=float)
        max_val = float(np.nanmax(mi_vals)) if np.isfinite(mi_vals).any() else 0.0
        if max_val > 0.0:
            mi_scores: Dict[str, float] = {k: float(v) / max_val for k, v in mi_raw.items()}
        else:
            mi_scores = {k: 0.0 for k in mi_raw.keys()}
    else:
        # Uniform baseline when no feature passes the min_samples check
        mi_scores = {column: 0.5 for column in features_aligned.columns}

    # Stability scores using rolling coefficient of variation
    stability_scores: Dict[str, float] = {}
    try:
        window = min(100, max(10, len(features_aligned) // 5))
        for column in features_aligned.columns:
            series = features_aligned[column].ffill().fillna(0)
            rolling = series.rolling(window=window, min_periods=10)
            rolling_mean = rolling.mean()
            rolling_std = rolling.std()
            if rolling_mean.std() > 1e-8:
                cv = rolling_std.mean() / (abs(rolling_mean.mean()) + 1e-8)
                stability = 1.0 / (1.0 + cv)
            else:
                stability = 0.5
            stability_scores[column] = max(0.0, min(1.0, float(stability)))
    except Exception:
        stability_scores = {column: 0.5 for column in features_aligned.columns}

    composite_scores: Dict[str, float] = {}
    for column in features_df.columns:
        mi_score = float(mi_scores.get(column, 0.0))
        stability = float(stability_scores.get(column, 0.5))
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


def _extract_base_feature_name_fallback(step: Any, variant_col: str) -> str:
    """Best-effort extraction of base feature name from variant column name."""
    base_name = variant_col
    suffixes_to_remove = ["_base", "_volnorm", "_vwap", "_trend_adj"]

    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break

    return base_name


def _extract_variant_type_fallback(step: Any, variant_col: str) -> str:
    """Infer variant type (base/volnorm/vwap/trend_adj) from column name."""
    if variant_col.endswith("_volnorm"):
        return "volnorm"
    if variant_col.endswith("_vwap"):
        return "vwap"
    if variant_col.endswith("_trend_adj"):
        return "trend_adj"
    return "base"


async def _phase2_cheap_pruning_fallback(
    step: Any,
    variant_features: pd.DataFrame,
    labeled_data: pd.DataFrame,
    lookback_optimization: Dict,
    config: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, Any], pd.DataFrame]:
    """Minimal Phase 2 cheap pruning fallback.

    This implementation is intentionally lightweight: it selects a single
    target column from ``labeled_data`` (if available) and returns the input
    ``variant_features`` unchanged together with basic statistics. It is
    only used when the main step class does not provide its own
    ``_phase2_cheap_pruning`` implementation.
    """

    if isinstance(labeled_data, pd.DataFrame) and not labeled_data.empty:
        candidate_cols = [
            "smoothed_label",
            "binary_label",
            "realized_return",
        ]
        existing = [c for c in candidate_cols if c in labeled_data.columns]
        if existing:
            targets_df = labeled_data[[existing[0]]]
        else:
            targets_df = labeled_data.iloc[:, :1].copy()
    else:
        targets_df = pd.DataFrame(index=variant_features.index)
        targets_df["dummy_target"] = 0.0

    stats: Dict[str, Any] = {
        "fallback": True,
        "reason": "interaction_generation_fallbacks._phase2_cheap_pruning_fallback",
        "initial_features": len(variant_features.columns),
        "final_features": len(variant_features.columns),
        "reduction": 0.0,
    }

    return variant_features, stats, targets_df


async def _phase3_lgbm_shap_pipeline_fallback(
    step: Any,
    pruned_features: pd.DataFrame,
    targets: pd.DataFrame,
    config: Dict[str, Any],
    lookback_optimization: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """Minimal Phase 3 LGBM+SHAP pipeline fallback.

    This orchestrator treats ``pruned_features`` as the final base feature
    set and relies on the existing ``_phase3_3_label_guided_interaction_discovery``
    helper (or its legacy fallback) to generate interactions. It is only
    attached when the runtime class does not define its own
    ``_phase3_lgbm_shap_pipeline`` implementation.
    """

    # Use pruned_features directly as the base feature set
    final_features = pruned_features

    # Infer feature categories for compatibility with label-guided discovery
    feature_categories: Dict[str, str] = {}
    try:
        for col in final_features.columns:
            if hasattr(step, "_infer_feature_category"):
                feature_categories[col] = step._infer_feature_category(col)  # type: ignore[assignment]
            else:
                feature_categories[col] = _infer_feature_category_fallback(step, col)
    except Exception:
        # Best-effort fallback: leave feature_categories partially filled
        pass

    # Run label-guided interaction discovery when available
    try:
        if hasattr(step, "_phase3_3_label_guided_interaction_discovery"):
            interactions, shap_metadata = await step._phase3_3_label_guided_interaction_discovery(  # type: ignore[call-arg]
                final_features,
                targets,
                config,
                feature_categories,
            )
        elif hasattr(step, "_phase3_3_interaction_discovery_legacy"):
            interactions, shap_metadata = await step._phase3_3_interaction_discovery_legacy(  # type: ignore[call-arg]
                final_features,
                targets,
                config,
                feature_categories,
            )
        else:
            # No interaction discovery available; return empty interactions
            interactions = pd.DataFrame(index=final_features.index)
            shap_metadata = {
                "feature_categories": feature_categories,
                "interaction_discovery": {"selected_interactions": 0},
                "model_performance": {
                    "lgbm_training_successful": False,
                    "interaction_generation_successful": False,
                },
            }
    except Exception as exc:
        # On failure, return empty interactions but keep base features
        interactions = pd.DataFrame(index=final_features.index)
        shap_metadata = {
            "feature_categories": feature_categories,
            "interaction_discovery": {
                "selected_interactions": 0,
                "error": str(exc),
            },
            "model_performance": {
                "lgbm_training_successful": False,
                "interaction_generation_successful": False,
            },
        }

    # Optional: LightGBM gain + permutation-based feature selection
    #
    # Disabled for this fallback: we keep all features returned by the
    # interaction discovery helper and delegate any pruning/selection to
    # downstream FeatureSelection components.
    lgbm_fs_stats: Dict[str, Any] = {
        "method": "disabled",
        "reason": "Interaction generation step no longer performs LGBM-based pruning",
    }

    # Attach minimal LGBM FS metadata to shap_metadata, preserving any existing keys
    try:
        if isinstance(shap_metadata, dict):
            shap_metadata["lgbm_feature_selection"] = lgbm_fs_stats
    except Exception:
        # Best-effort; do not break the pipeline on metadata issues
        pass

    return final_features, interactions, shap_metadata


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
    if not hasattr(cls, "_extract_base_feature_name"):
        setattr(cls, "_extract_base_feature_name", _extract_base_feature_name_fallback)
    if not hasattr(cls, "_extract_variant_type"):
        setattr(cls, "_extract_variant_type", _extract_variant_type_fallback)
    if not hasattr(cls, "_phase2_cheap_pruning"):
        setattr(cls, "_phase2_cheap_pruning", _phase2_cheap_pruning_fallback)
    if not hasattr(cls, "_phase3_lgbm_shap_pipeline"):
        setattr(cls, "_phase3_lgbm_shap_pipeline", _phase3_lgbm_shap_pipeline_fallback)

