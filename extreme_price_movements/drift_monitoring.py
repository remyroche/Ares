"""Drift monitoring benchmarks, recaps, and regime-adaptor features."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


DRIFT_SCHEMA_VERSION = "expanded_v1"
DRIFT_WINDOWS_DAYS: dict[str, int] = {
    "1d": 1,
    "3d": 3,
    "7d": 7,
    "10d": 10,
    "14d": 14,
}
RECENT_BAR_WINDOWS_HOURS: dict[str, int] = {
    "1h": 1,
    "3h": 3,
    "6h": 6,
    "12h": 12,
    "24h": 24,
}
RECENT_BAR_TOP_FRACTIONS: dict[str, float] = {
    "top05": 0.05,
    "top10": 0.10,
    "top20": 0.20,
    "top30": 0.30,
}
RECENT_BAR_DEFAULT_OUTCOME_HORIZON_HOURS = 10
RECENT_BAR_MARKET_PRICE_COLUMNS: tuple[str, ...] = (
    "mid",
    "mark_close",
    "close",
    "vwap_1h",
    "best_bid",
    "best_ask",
)
REGIME_ADAPTOR_TIER1_WINDOWS = tuple(DRIFT_WINDOWS_DAYS.keys())
REGIME_ADAPTOR_EXPANDED_WINDOWS = ("1d", "7d")
REGIME_ADAPTOR_INDIVIDUAL_WINDOWS = REGIME_ADAPTOR_EXPANDED_WINDOWS
REGIME_ADAPTOR_FAMILY_WINDOWS = REGIME_ADAPTOR_TIER1_WINDOWS
LABEL_MATURITY_TS_COLUMNS = (
    "label_available_ts",
    "label_maturity_ts",
    "target_available_ts",
    "outcome_available_ts",
    "exit_available_ts",
    "exit_ts",
    "exit_time",
    "closed_at",
    "close_ts",
    "trade_close_ts",
)
ROW_TIMESTAMP_COLUMNS = (
    "_drift_ts",
    "timestamp",
    "signal_ts",
    "signal_bar_ts",
    "bar_ts",
    "entry_ts",
)


@dataclass(frozen=True)
class DriftMetricSpec:
    metric_name: str
    family: str
    tier: int
    severity_direction: str
    requires_matured_label: bool = False
    requires_trade: bool = False
    min_count: int = 3
    baseline_min_count: int = 12
    fallback_allowed: bool = True
    description: str = ""

    @property
    def higher_is_worse(self) -> bool:
        return self.severity_direction == "high"


def _spec(
    metric_name: str,
    family: str,
    severity_direction: str,
    *,
    tier: int = 1,
    requires_matured_label: bool = False,
    requires_trade: bool = False,
    min_count: int = 3,
    baseline_min_count: int = 12,
    description: str = "",
) -> DriftMetricSpec:
    if severity_direction not in {"high", "low", "two_sided"}:
        raise ValueError(f"Invalid severity_direction for {metric_name}: {severity_direction}")
    return DriftMetricSpec(
        metric_name=metric_name,
        family=family,
        tier=int(tier),
        severity_direction=severity_direction,
        requires_matured_label=bool(requires_matured_label),
        requires_trade=bool(requires_trade),
        min_count=int(min_count),
        baseline_min_count=int(baseline_min_count),
        description=description,
    )


TIER1_DRIFT_METRICS: tuple[DriftMetricSpec, ...] = (
    _spec("feature_psi_mean", "feature_regime_drift", "high"),
    _spec("feature_ks_mean", "feature_regime_drift", "high"),
    _spec("feature_embedding_distance", "feature_regime_drift", "high"),
    _spec("raw_state_reconstruction_error", "feature_regime_drift", "high"),
    _spec("target_return_mean", "target_drift", "two_sided", requires_matured_label=True),
    _spec("target_return_volatility", "target_drift", "high", requires_matured_label=True),
    _spec("target_positive_rate", "target_drift", "two_sided", requires_matured_label=True),
    _spec("target_top_bottom_spread", "target_drift", "low", requires_matured_label=True),
    _spec("base_prediction_mean", "prediction_drift", "two_sided"),
    _spec("base_prediction_std", "prediction_drift", "low"),
    _spec("meta_prediction_mean", "prediction_drift", "two_sided"),
    _spec("meta_prediction_std", "prediction_drift", "low"),
    _spec("high_confidence_share", "prediction_drift", "two_sided"),
    _spec("threshold_pass_share", "prediction_drift", "two_sided"),
    _spec("rare_leaf_share", "model_internal_drift", "high"),
    _spec("leaf_surprisal_mean", "model_internal_drift", "high"),
    _spec("leaf_centroid_distance", "model_internal_drift", "high"),
    _spec("contribution_drift", "model_internal_drift", "high"),
    _spec("path_instability", "model_internal_drift", "high"),
    _spec("uncertainty_score_mean", "uncertainty_drift", "high"),
    _spec("prob_uncertainty_mean", "uncertainty_drift", "high"),
    _spec("entropy_mean", "uncertainty_drift", "high"),
    _spec("disagreement_mean", "uncertainty_drift", "high"),
    _spec("net_return_per_accepted_trade", "performance_drift", "low", requires_matured_label=True, requires_trade=True),
    _spec("top_bottom_return_spread", "performance_drift", "low", requires_matured_label=True),
    _spec("rolling_rank_ic", "performance_drift", "low", requires_matured_label=True),
    _spec("score_bucket_monotonicity", "performance_drift", "low", requires_matured_label=True),
    _spec("meta_error_surprise", "residual_drift", "high", requires_matured_label=True),
    _spec("hit_rate_top20", "residual_drift", "low", requires_matured_label=True),
    _spec("accepted_trade_residual_return", "residual_drift", "two_sided", requires_matured_label=True, requires_trade=True),
)

EXPANDED_DRIFT_METRICS: tuple[DriftMetricSpec, ...] = (
    # Feature/regime drift: distribution shift, embedding distance, and raw-state manifold drift.
    _spec("feature_psi_p95", "feature_regime_drift", "high", tier=2),
    _spec("feature_psi_max", "feature_regime_drift", "high", tier=2),
    _spec("feature_ks_p95", "feature_regime_drift", "high", tier=2),
    _spec("feature_ks_max", "feature_regime_drift", "high", tier=2),
    _spec("feature_wasserstein_mean", "feature_regime_drift", "high", tier=2),
    _spec("feature_cov_shift", "feature_regime_drift", "high", tier=2),
    _spec("regime_centroid_similarity", "feature_regime_drift", "low", tier=2),
    _spec("regime_centroid_pc0_similarity", "feature_regime_drift", "low", tier=2),
    _spec("regime_centroid_pc1_similarity", "feature_regime_drift", "low", tier=2),
    _spec("regime_centroid_pc2_similarity", "feature_regime_drift", "low", tier=2),
    _spec("mahalanobis_mean", "feature_regime_drift", "high", tier=2),
    _spec("mahalanobis_p90", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_knn_distance_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_min_cluster_distance_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_transition_norm_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_transition_mahalanobis_mean", "feature_regime_drift", "high", tier=2),
    _spec("state_log_likelihood_mean", "feature_regime_drift", "low", tier=2),
    _spec("state_tod_mahalanobis_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_psi_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_psi_max", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_ks_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_ks_max", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_svd_psi_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_svd_psi_max", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_svd_ks_mean", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_svd_ks_max", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_svd_component_mean", "feature_regime_drift", "two_sided", tier=2),
    _spec("raw_state_svd_component_std", "feature_regime_drift", "high", tier=2),
    _spec("raw_state_svd_l2_norm", "feature_regime_drift", "high", tier=2),
    # Target/label drift: realized target distribution, ambiguity, label instability, and path shape.
    _spec("target_return_p10", "target_drift", "low", tier=2, requires_matured_label=True),
    _spec("target_return_p90", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("target_downside_tail", "target_drift", "low", tier=2, requires_matured_label=True),
    _spec("target_upside_tail", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("soft_label_mean", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("soft_label_std", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("hard_label_positive_rate", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("near_barrier_ambiguity_mean", "target_drift", "high", tier=2, requires_matured_label=True),
    _spec("near_barrier_share", "target_drift", "high", tier=2, requires_matured_label=True),
    _spec("unstable_label_score_mean", "target_drift", "high", tier=2, requires_matured_label=True),
    _spec("label_flip_under_cost_share", "target_drift", "high", tier=2, requires_matured_label=True),
    _spec("mfe_magnitude_mean", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("mae_magnitude_mean", "target_drift", "high", tier=2, requires_matured_label=True),
    _spec("mfe_mae_ratio_mean", "target_drift", "low", tier=2, requires_matured_label=True),
    _spec("mfe_time_frac_mean", "target_drift", "high", tier=2, requires_matured_label=True),
    _spec("mae_time_frac_mean", "target_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("quick_profit_share", "target_drift", "low", tier=2, requires_matured_label=True),
    _spec("mfe_before_mae_share", "target_drift", "low", tier=2, requires_matured_label=True),
    # Prediction drift: score distribution, rank issuance, confidence, and base/meta disagreement.
    _spec("base_prediction_p10", "prediction_drift", "two_sided", tier=2),
    _spec("base_prediction_p50", "prediction_drift", "two_sided", tier=2),
    _spec("base_prediction_p90", "prediction_drift", "two_sided", tier=2),
    _spec("meta_prediction_p10", "prediction_drift", "two_sided", tier=2),
    _spec("meta_prediction_p50", "prediction_drift", "two_sided", tier=2),
    _spec("meta_prediction_p90", "prediction_drift", "two_sided", tier=2),
    _spec("base_high_confidence_share", "prediction_drift", "two_sided", tier=2),
    _spec("meta_high_confidence_share", "prediction_drift", "two_sided", tier=2),
    _spec("base_neutral_share", "prediction_drift", "high", tier=2),
    _spec("meta_neutral_share", "prediction_drift", "high", tier=2),
    _spec("threshold_pass_share_top10", "prediction_drift", "two_sided", tier=2),
    _spec("threshold_pass_share_top20", "prediction_drift", "two_sided", tier=2),
    _spec("threshold_pass_share_top30", "prediction_drift", "two_sided", tier=2),
    _spec("rank_std", "prediction_drift", "low", tier=2),
    _spec("rank_p90_minus_p10", "prediction_drift", "low", tier=2),
    _spec("base_meta_disagreement_mean", "prediction_drift", "high", tier=2),
    _spec("base_meta_disagreement_p90", "prediction_drift", "high", tier=2),
    _spec("base_model_disagreement_mean", "prediction_drift", "high", tier=2),
    _spec("score_margin_top10", "prediction_drift", "low", tier=2),
    _spec("score_margin_top20", "prediction_drift", "low", tier=2),
    _spec("score_margin_top30", "prediction_drift", "low", tier=2),
    # Model internal drift: leaf rarity, target dispersion, centroid distance, path instability, contributions.
    _spec("rare_leaf_p90", "model_internal_drift", "high", tier=2),
    _spec("leaf_count_p10", "model_internal_drift", "low", tier=2),
    _spec("leaf_count_min", "model_internal_drift", "low", tier=2),
    _spec("leaf_train_freq_mean", "model_internal_drift", "low", tier=2),
    _spec("leaf_train_freq_p10", "model_internal_drift", "low", tier=2),
    _spec("leaf_train_freq_min", "model_internal_drift", "low", tier=2),
    _spec("leaf_train_freq_std", "model_internal_drift", "high", tier=2),
    _spec("leaf_surprisal_p90", "model_internal_drift", "high", tier=2),
    _spec("leaf_surprisal_max", "model_internal_drift", "high", tier=2),
    _spec("leaf_low_freq_fraction", "model_internal_drift", "high", tier=2),
    _spec("leaf_target_dispersion", "model_internal_drift", "high", tier=2),
    _spec("leaf_target_iqr_mean", "model_internal_drift", "high", tier=2),
    _spec("leaf_target_range_mean", "model_internal_drift", "high", tier=2),
    _spec("leaf_target_abs_mean", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_radius_mean", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_distance_p90", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_distance_max", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_distance_cv", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_distance_rel_mean", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_distance_norm_p90", "model_internal_drift", "high", tier=2),
    _spec("leaf_centroid_distance_norm_max", "model_internal_drift", "high", tier=2),
    _spec("contrib_abs_sum_mean", "model_internal_drift", "high", tier=2),
    _spec("contrib_l2_norm_mean", "model_internal_drift", "high", tier=2),
    _spec("contrib_entropy_mean", "model_internal_drift", "high", tier=2),
    _spec("contrib_top1_abs_share_mean", "model_internal_drift", "two_sided", tier=2),
    _spec("contrib_top3_abs_share_mean", "model_internal_drift", "two_sided", tier=2),
    _spec("contrib_balance_abs_mean", "model_internal_drift", "high", tier=2),
    _spec("num_material_contrib_features_mean", "model_internal_drift", "two_sided", tier=2),
    _spec("archetype_contrib_svd_l2_norm", "model_internal_drift", "high", tier=2),
    _spec("large_leaf_value_fraction", "model_internal_drift", "high", tier=2),
    _spec("leaf_depth_mean", "model_internal_drift", "two_sided", tier=2),
    _spec("leaf_depth_std", "model_internal_drift", "high", tier=2),
    _spec("score_path_range", "model_internal_drift", "high", tier=2),
    _spec("rank_path_range", "model_internal_drift", "high", tier=2),
    _spec("rank_bin_lift_oof", "model_internal_drift", "low", tier=2),
    _spec("rank_bin_net_ret_oof", "model_internal_drift", "low", tier=2),
    _spec("rank_bin_se_oof", "model_internal_drift", "high", tier=2),
    # Uncertainty drift: output uncertainty, entropy, ensemble spread, and neutral margins.
    _spec("uncertainty_score_p90", "uncertainty_drift", "high", tier=2),
    _spec("prob_uncertainty_p90", "uncertainty_drift", "high", tier=2),
    _spec("prob_uncertainty_max", "uncertainty_drift", "high", tier=2),
    _spec("entropy_p90", "uncertainty_drift", "high", tier=2),
    _spec("entropy_max", "uncertainty_drift", "high", tier=2),
    _spec("disagreement_p90", "uncertainty_drift", "high", tier=2),
    _spec("disagreement_max", "uncertainty_drift", "high", tier=2),
    _spec("prob_std_mean", "uncertainty_drift", "high", tier=2),
    _spec("raw_score_std_mean", "uncertainty_drift", "high", tier=2),
    _spec("margin_from_neutral_mean", "uncertainty_drift", "low", tier=2),
    _spec("variance_proxy_mean", "uncertainty_drift", "high", tier=2),
    # Performance drift: top-slice economics, IC, bucket monotonicity, and calibration by confidence.
    _spec("realized_return_top_decile", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("realized_return_top_quintile", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("realized_return_top_quarter", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("realized_return_bottom_decile", "performance_drift", "high", tier=2, requires_matured_label=True),
    _spec("top_decile_minus_bottom_decile_return", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("top_quarter_minus_bottom_quarter_return", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("rank_ic_top30", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("rank_ic_top20", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("rank_ic_top10", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("bps_weighted_hit_top30", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("bps_weighted_hit_top20", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("bps_weighted_hit_top10", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("meta_confidence_top_return", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("meta_confidence_bucket_monotonicity", "performance_drift", "low", tier=2, requires_matured_label=True),
    _spec("calibration_slope", "performance_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("calibration_abs_error", "performance_drift", "high", tier=2, requires_matured_label=True),
    # Residual/error drift: expected-vs-actual gaps and adaptive-window shift proxies.
    _spec("expected_hit_gap_mean", "residual_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("expected_hit_abs_error", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("brier_score", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("expected_ev_gap_mean", "residual_drift", "two_sided", tier=2, requires_matured_label=True),
    _spec("expected_ev_abs_error", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("residual_return_volatility", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("residual_return_p90_abs", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("adwin_net_return_shift", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("adwin_top_bottom_spread_shift", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("adwin_meta_error_shift", "residual_drift", "high", tier=2, requires_matured_label=True),
    _spec("adwin_hit_rate_top20_shift", "residual_drift", "high", tier=2, requires_matured_label=True),
)

ALL_DRIFT_METRICS: tuple[DriftMetricSpec, ...] = tuple(
    dict.fromkeys((*TIER1_DRIFT_METRICS, *EXPANDED_DRIFT_METRICS))
)
TIER1_DRIFT_METRIC_NAMES: frozenset[str] = frozenset(
    spec.metric_name for spec in TIER1_DRIFT_METRICS
)
DRIFT_METRIC_REGISTRY: dict[str, DriftMetricSpec] = {
    spec.metric_name: spec for spec in ALL_DRIFT_METRICS
}
DRIFT_FAMILIES: tuple[str, ...] = tuple(
    dict.fromkeys(spec.family for spec in ALL_DRIFT_METRICS)
)
REGIME_ADAPTOR_TIER1_MODEL_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "feature_psi_mean",
        "feature_embedding_distance",
        "base_prediction_std",
        "threshold_pass_share",
        "leaf_centroid_distance",
        "rolling_rank_ic",
        "meta_error_surprise",
    }
)
REGIME_ADAPTOR_EXPANDED_MODEL_METRIC_NAMES: frozenset[str] = frozenset(
    {
        "feature_cov_shift",
        "regime_centroid_similarity",
        "raw_state_knn_distance_mean",
        "raw_state_svd_component_mean",
        "raw_state_svd_component_std",
        "raw_state_svd_l2_norm",
        "base_meta_disagreement_mean",
        "rank_std",
        "score_margin_top20",
        "threshold_pass_share_top10",
        "rare_leaf_p90",
        "leaf_target_dispersion",
        "leaf_centroid_distance_norm_p90",
        "contrib_entropy_mean",
        "score_path_range",
        "realized_return_top_decile",
        "rank_ic_top20",
        "bps_weighted_hit_top20",
        "calibration_abs_error",
        "brier_score",
        "adwin_net_return_shift",
        "adwin_hit_rate_top20_shift",
    }
)


def _drift_specs_for_tier(tier: int | None = None) -> tuple[DriftMetricSpec, ...]:
    if tier is None:
        return ALL_DRIFT_METRICS
    return tuple(spec for spec in ALL_DRIFT_METRICS if spec.tier == int(tier))


def drift_regime_feature_names(*, include_expanded: bool = True) -> list[str]:
    names: list[str] = []
    for window in REGIME_ADAPTOR_TIER1_WINDOWS:
        for spec in TIER1_DRIFT_METRICS:
            if spec.metric_name in REGIME_ADAPTOR_TIER1_MODEL_METRIC_NAMES:
                names.append(f"drift_{spec.family}_{spec.metric_name}_{window}")
    if include_expanded:
        for window in REGIME_ADAPTOR_EXPANDED_WINDOWS:
            for spec in EXPANDED_DRIFT_METRICS:
                if spec.metric_name in REGIME_ADAPTOR_EXPANDED_MODEL_METRIC_NAMES:
                    names.append(f"drift_{spec.family}_{spec.metric_name}_{window}")
    for window in REGIME_ADAPTOR_TIER1_WINDOWS:
        for family in DRIFT_FAMILIES:
            names.append(f"drift_{family}_score_{window}")
    if include_expanded:
        for window in REGIME_ADAPTOR_EXPANDED_WINDOWS:
            for family in DRIFT_FAMILIES:
                names.append(f"drift_{family}_all_score_{window}")
    for family in DRIFT_FAMILIES:
        names.append(f"drift_{family}_score_7d_minus_3d")
        if include_expanded:
            names.append(f"drift_{family}_all_score_7d_minus_1d")
    return list(dict.fromkeys(names))


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _as_utc_timestamp(value: Any = None, *, default: Any = None) -> pd.Timestamp:
    if value is None:
        if default is None:
            return pd.Timestamp.now(tz="UTC")
        value = default
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        if default is None:
            return pd.Timestamp.now(tz="UTC")
        ts = pd.Timestamp(default)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _timestamp_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return pd.to_datetime(df[col], utc=True, errors="coerce")
    return None


def _matured_label_group(group: pd.DataFrame, meta: Mapping[str, Any]) -> pd.DataFrame:
    cutoff_raw = meta.get("label_maturity_cutoff_ts")
    if cutoff_raw is None:
        return group
    cutoff = _as_utc_timestamp(cutoff_raw, default=meta.get("asof_ts"))
    availability = _timestamp_series(group, LABEL_MATURITY_TS_COLUMNS)
    if availability is not None:
        mask = availability.notna() & (availability <= cutoff)
        return group.loc[mask.to_numpy(dtype=bool)]
    row_ts = _timestamp_series(group, ROW_TIMESTAMP_COLUMNS)
    if row_ts is None:
        return group.iloc[0:0]
    mask = row_ts.notna() & (row_ts <= cutoff)
    return group.loc[mask.to_numpy(dtype=bool)]


def _numeric_series(df: pd.DataFrame, candidates: Sequence[str]) -> pd.Series | None:
    for col in candidates:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce")
    return None


def _mean(df: pd.DataFrame, candidates: Sequence[str]) -> float:
    s = _numeric_series(df, candidates)
    if s is None:
        return float("nan")
    vals = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def _std(df: pd.DataFrame, candidates: Sequence[str]) -> float:
    s = _numeric_series(df, candidates)
    if s is None:
        return float("nan")
    vals = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.std(ddof=0)) if len(vals) else float("nan")


def _quantile(df: pd.DataFrame, candidates: Sequence[str], q: float) -> float:
    s = _numeric_series(df, candidates)
    if s is None:
        return float("nan")
    vals = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float(vals.quantile(float(q))) if len(vals) else float("nan")


def _numeric_matrix(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> pd.DataFrame:
    cols: list[str] = []
    seen: set[str] = set()
    for col in candidates:
        if col in df.columns and col not in seen:
            cols.append(col)
            seen.add(col)
    for prefix in prefixes:
        for col in df.columns:
            col_s = str(col)
            if col_s.startswith(prefix) and col_s not in seen:
                cols.append(col_s)
                seen.add(col_s)
    if not cols:
        return pd.DataFrame(index=df.index)
    return df.loc[:, cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _mean_any(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> float:
    mat = _numeric_matrix(df, candidates=candidates, prefixes=prefixes)
    if mat.empty:
        return float("nan")
    vals = mat.to_numpy(dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _quantile_any(
    df: pd.DataFrame,
    q: float,
    *,
    candidates: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> float:
    mat = _numeric_matrix(df, candidates=candidates, prefixes=prefixes)
    if mat.empty:
        return float("nan")
    vals = mat.to_numpy(dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(np.quantile(vals, float(q))) if vals.size else float("nan")


def _max_any(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> float:
    mat = _numeric_matrix(df, candidates=candidates, prefixes=prefixes)
    if mat.empty:
        return float("nan")
    vals = mat.to_numpy(dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    return float(vals.max()) if vals.size else float("nan")


def _mean_abs_any(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> float:
    mat = _numeric_matrix(df, candidates=candidates, prefixes=prefixes)
    if mat.empty:
        return float("nan")
    vals = np.abs(mat.to_numpy(dtype=float).reshape(-1))
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _row_l2_mean(
    df: pd.DataFrame,
    *,
    candidates: Sequence[str] = (),
    prefixes: Sequence[str] = (),
) -> float:
    mat = _numeric_matrix(df, candidates=candidates, prefixes=prefixes)
    if mat.empty:
        return float("nan")
    arr = mat.to_numpy(dtype=float)
    row_ok = np.isfinite(arr).any(axis=1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    vals = np.sqrt(np.sum(np.square(arr), axis=1))
    vals = vals[row_ok & np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def _share(df: pd.DataFrame, mask: pd.Series | np.ndarray | None) -> float:
    if mask is None or len(df) == 0:
        return float("nan")
    arr = np.asarray(mask, dtype=bool)
    if len(arr) != len(df):
        return float("nan")
    return float(np.mean(arr))


def _corr(a: pd.Series | None, b: pd.Series | None, *, method: str = "spearman") -> float:
    if a is None or b is None:
        return float("nan")
    tmp = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    if len(tmp) < 3 or tmp["a"].nunique() < 2 or tmp["b"].nunique() < 2:
        return float("nan")
    return float(tmp["a"].corr(tmp["b"], method=method))


def _rank_score(df: pd.DataFrame) -> pd.Series | None:
    return _numeric_series(
        df,
        (
            "auction_rank_score",
            "auction_rank_pct",
            "normalized_rank_score",
            "policy_rank_pct",
            "meta_train_rank_pct",
            "meta_pred",
            "calibrated_score",
        ),
    )


def _return_series(df: pd.DataFrame) -> pd.Series | None:
    return _numeric_series(
        df,
        (
            "net_return",
            "realized_net_return",
            "shadow_exit_return",
            "recent_forward_net_return",
            "recent_forward_return",
            "gross_return",
        ),
    )


def _recent_return_series(df: pd.DataFrame) -> pd.Series | None:
    ret = _return_series(df)
    if ret is not None:
        return ret
    bps = _numeric_series(
        df,
        (
            "fill_forward_net_bps",
            "decision_mid_forward_net_bps",
            "signal_forward_net_bps",
            "realized_trade_net_bps",
            "shadow_exit_net_bps",
            "recent_forward_net_bps",
            "recent_forward_bps",
        ),
    )
    if bps is None:
        return None
    return pd.to_numeric(bps, errors="coerce") / 10000.0


def _soft_label_series(df: pd.DataFrame) -> pd.Series | None:
    return _numeric_series(df, ("soft_label", "y_soft", "label_soft", "soft_y", "target_soft"))


def _hard_label_series(df: pd.DataFrame) -> pd.Series | None:
    return _numeric_series(df, ("hard_label", "y", "target", "label", "y_bin"))


def _bool_share_from_columns(df: pd.DataFrame, candidates: Sequence[str]) -> float:
    s = _numeric_series(df, candidates)
    if s is None:
        for col in candidates:
            if col in df.columns:
                raw = pd.Series(df[col])
                if raw.empty:
                    continue
                return float(raw.fillna(False).astype(bool).mean())
        return float("nan")
    vals = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float((vals > 0.0).mean()) if len(vals) else float("nan")


def _share_threshold(df: pd.DataFrame, candidates: Sequence[str], threshold: float) -> float:
    s = _numeric_series(df, candidates)
    if s is None:
        return float("nan")
    vals = s.replace([np.inf, -np.inf], np.nan).dropna()
    return float((vals >= float(threshold)).mean()) if len(vals) else float("nan")


def _metric_value_top_bottom_spread(df: pd.DataFrame) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_top_bottom_spread_from_rank_return(_rank_return_frame(score, ret))


def _rank_return_frame(score: pd.Series | None, ret: pd.Series | None) -> pd.DataFrame:
    if score is None or ret is None:
        return pd.DataFrame(columns=["score", "ret"])
    return (
        pd.DataFrame(
            {
                "score": pd.to_numeric(score, errors="coerce"),
                "ret": pd.to_numeric(ret, errors="coerce"),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )


def _metric_value_top_bottom_spread_from_rank_return(tmp: pd.DataFrame) -> float:
    if len(tmp) < 10 or tmp["score"].nunique() < 2:
        return float("nan")
    hi = tmp["score"] >= tmp["score"].quantile(0.80)
    lo = tmp["score"] <= tmp["score"].quantile(0.20)
    if hi.sum() == 0 or lo.sum() == 0:
        return float("nan")
    return float(tmp.loc[hi, "ret"].mean() - tmp.loc[lo, "ret"].mean())


def _metric_value_slice_return_from_rank_return(tmp: pd.DataFrame, top_fraction: float) -> float:
    if len(tmp) < 3 or tmp["score"].nunique() < 2:
        return float("nan")
    threshold = tmp["score"].quantile(1.0 - float(top_fraction))
    selected = tmp["score"] >= threshold
    return float(tmp.loc[selected, "ret"].mean()) if selected.any() else float("nan")


def _metric_value_bottom_slice_return_from_rank_return(tmp: pd.DataFrame, bottom_fraction: float) -> float:
    if len(tmp) < 3 or tmp["score"].nunique() < 2:
        return float("nan")
    threshold = tmp["score"].quantile(float(bottom_fraction))
    selected = tmp["score"] <= threshold
    return float(tmp.loc[selected, "ret"].mean()) if selected.any() else float("nan")


def _metric_value_top_minus_bottom_from_rank_return(tmp: pd.DataFrame, fraction: float) -> float:
    top = _metric_value_slice_return_from_rank_return(tmp, fraction)
    bottom = _metric_value_bottom_slice_return_from_rank_return(tmp, fraction)
    if not (np.isfinite(top) and np.isfinite(bottom)):
        return float("nan")
    return float(top - bottom)


def _metric_value_rank_ic_slice_from_rank_return(tmp: pd.DataFrame, top_fraction: float) -> float:
    if len(tmp) < 5 or tmp["score"].nunique() < 2:
        return float("nan")
    selected = tmp["score"] >= tmp["score"].quantile(1.0 - float(top_fraction))
    if selected.sum() < 3:
        return float("nan")
    return float(tmp.loc[selected, "score"].corr(tmp.loc[selected, "ret"], method="spearman"))


def _metric_value_bps_weighted_hit_slice_from_rank_return(tmp: pd.DataFrame, top_fraction: float) -> float:
    if len(tmp) < 3 or tmp["score"].nunique() < 2:
        return float("nan")
    selected = tmp["score"] >= tmp["score"].quantile(1.0 - float(top_fraction))
    vals = tmp.loc[selected, "ret"].to_numpy(dtype=float)
    weights = np.abs(vals)
    denom = float(weights.sum())
    if denom <= 1e-12:
        return float("nan")
    return float(weights[vals > 0.0].sum() / denom)


def _metric_value_monotonicity_from_rank_return(tmp: pd.DataFrame) -> float:
    if len(tmp) < 20 or tmp["score"].nunique() < 5:
        return float("nan")
    bins = min(10, max(3, int(np.sqrt(len(tmp)))))
    tmp = tmp.copy()
    try:
        tmp["bucket"] = pd.qcut(tmp["score"], bins, labels=False, duplicates="drop")
    except Exception:
        return float("nan")
    means = tmp.groupby("bucket", observed=True)["ret"].mean()
    if len(means) < 3:
        return float("nan")
    return float(pd.Series(range(len(means)), index=means.index).corr(means, method="spearman"))


def _corr_from_rank_return(tmp: pd.DataFrame, *, method: str = "spearman") -> float:
    if len(tmp) < 3 or tmp["score"].nunique() < 2 or tmp["ret"].nunique() < 2:
        return float("nan")
    return float(tmp["score"].corr(tmp["ret"], method=method))


def _metric_value_slice_return(df: pd.DataFrame, top_fraction: float) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_slice_return_from_rank_return(_rank_return_frame(score, ret), top_fraction)


def _metric_value_bottom_slice_return(df: pd.DataFrame, bottom_fraction: float) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_bottom_slice_return_from_rank_return(_rank_return_frame(score, ret), bottom_fraction)


def _metric_value_top_minus_bottom(df: pd.DataFrame, fraction: float) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_top_minus_bottom_from_rank_return(_rank_return_frame(score, ret), fraction)


def _metric_value_rank_ic_slice(df: pd.DataFrame, top_fraction: float) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_rank_ic_slice_from_rank_return(_rank_return_frame(score, ret), top_fraction)


def _metric_value_bps_weighted_hit_slice(df: pd.DataFrame, top_fraction: float) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_bps_weighted_hit_slice_from_rank_return(_rank_return_frame(score, ret), top_fraction)


def _metric_value_monotonicity(df: pd.DataFrame) -> float:
    score = _rank_score(df)
    ret = _return_series(df)
    return _metric_value_monotonicity_from_rank_return(_rank_return_frame(score, ret))


def _metric_value_confidence_monotonicity(df: pd.DataFrame) -> float:
    pred = _numeric_series(df, ("meta_pred", "calibrated_score", "estimated_hit_rate"))
    ret = _return_series(df)
    if pred is None or ret is None:
        return float("nan")
    conf = np.abs(pd.to_numeric(pred, errors="coerce") - 0.5)
    tmp = pd.DataFrame({"conf": conf, "ret": ret}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 20 or tmp["conf"].nunique() < 5:
        return float("nan")
    bins = min(10, max(3, int(np.sqrt(len(tmp)))))
    try:
        tmp["bucket"] = pd.qcut(tmp["conf"], bins, labels=False, duplicates="drop")
    except Exception:
        return float("nan")
    means = tmp.groupby("bucket", observed=True)["ret"].mean()
    if len(means) < 3:
        return float("nan")
    return float(pd.Series(range(len(means)), index=means.index).corr(means, method="spearman"))


def _metric_value_calibration(df: pd.DataFrame) -> tuple[float, float]:
    pred = _numeric_series(df, ("estimated_hit_rate", "meta_pred", "calibrated_score"))
    ret = _return_series(df)
    if pred is None or ret is None:
        return float("nan"), float("nan")
    tmp = pd.DataFrame({"pred": pred, "hit": (pd.to_numeric(ret, errors="coerce") > 0.0).astype(float)}).dropna()
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 10 or tmp["pred"].nunique() < 2:
        return float("nan"), float("nan")
    slope = float(tmp["pred"].corr(tmp["hit"], method="pearson"))
    abs_error = float(np.abs(tmp["hit"] - np.clip(tmp["pred"], 0.0, 1.0)).mean())
    return slope, abs_error


def _adwin_like_shift(values: pd.Series | np.ndarray | None, *, min_count: int = 8) -> float:
    if values is None:
        return float("nan")
    vals = pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) < min_count:
        return float("nan")
    arr = vals.to_numpy(dtype=float)
    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]
    if len(left) < 2 or len(right) < 2:
        return float("nan")
    scale = max(float(np.nanstd(arr, ddof=0)), 1e-12)
    return float(abs(float(np.nanmean(right)) - float(np.nanmean(left))) / scale)


def _top20_hit_series(df: pd.DataFrame) -> pd.Series | None:
    score = _rank_score(df)
    ret = _return_series(df)
    if score is None or ret is None:
        return None
    tmp = pd.DataFrame({"score": score, "ret": ret}).replace([np.inf, -np.inf], np.nan)
    if tmp["score"].notna().sum() < 5:
        return None
    threshold = tmp["score"].quantile(0.80)
    out = pd.Series(np.nan, index=df.index, dtype=float)
    mask = tmp["score"] >= threshold
    out.loc[mask] = (tmp.loc[mask, "ret"] > 0.0).astype(float)
    return out


def _safe_asset_class(symbol: Any) -> str:
    s = str(symbol or "")
    if "/" in s and ":USD" in s:
        return "crypto_perp"
    return "unknown"


def _volatility_regime(values: pd.Series | None) -> str:
    if values is None:
        return "unknown"
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(vals):
        return "unknown"
    med = float(vals.median())
    if med <= 0:
        return "low"
    q = float(vals.rank(pct=True).median())
    if q >= 0.67:
        return "high"
    if q <= 0.33:
        return "low"
    return "mid"


def _add_metric(rows: list[dict[str, Any]], meta: Mapping[str, Any], metric_name: str, value: float, count: int) -> None:
    spec = DRIFT_METRIC_REGISTRY.get(metric_name)
    if spec is None:
        return
    if not np.isfinite(value):
        return
    rows.append(
        {
            **dict(meta),
            "family": spec.family,
            "metric_name": spec.metric_name,
            "tier": int(spec.tier),
            "metric_value": float(value),
            "metric_count": int(count),
            "severity_direction": spec.severity_direction,
            "requires_matured_label": bool(spec.requires_matured_label),
            "requires_trade": bool(spec.requires_trade),
            "metric_min_count": int(spec.min_count),
            "baseline_min_count": int(spec.baseline_min_count),
            "metric_available": bool(count >= spec.min_count),
        }
    )


def _metric_rows_for_group(group: pd.DataFrame, meta: Mapping[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    n = int(len(group))
    if n == 0:
        return out
    if "_drift_ts" in group.columns:
        group = group.sort_values("_drift_ts")
    base = _numeric_series(group, ("base_pred", "raw_prediction_score", "lgbm_prob"))
    meta_pred = _numeric_series(group, ("meta_pred", "calibrated_score", "estimated_hit_rate"))
    rank = _rank_score(group)
    label_group = _matured_label_group(group, meta)
    label_n = int(len(label_group))
    label_rank = _rank_score(label_group)
    label_ret = _return_series(label_group)
    label_net = _numeric_series(label_group, ("net_return", "realized_net_return", "shadow_exit_return"))
    label_gross = _numeric_series(label_group, ("gross_return", "estimated_ev_gross_return"))
    label_meta_pred = _numeric_series(label_group, ("meta_pred", "calibrated_score", "estimated_hit_rate"))
    label_traded = label_group.get("was_traded")
    if label_traded is not None:
        label_traded_mask = pd.Series(label_traded).fillna(False).astype(bool).to_numpy()
    else:
        label_traded_mask = np.ones(label_n, dtype=bool)

    _add_metric(out, meta, "feature_psi_mean", _mean(group, ("feature_drift_psi_core", "feature_drift_psi_core_80", "feature_drift_psi_bin_mean", "base_lgbm_feature_drift_psi_core", "base_lgbm_feature_drift_psi_core_80", "meta_lgbm_feature_drift_psi_core", "meta_lgbm_feature_drift_psi_core_80")), n)
    _add_metric(out, meta, "feature_ks_mean", _mean(group, ("feature_drift_ks_core", "feature_drift_ks_bin_mean", "base_lgbm_feature_drift_ks_core", "meta_lgbm_feature_drift_ks_core")), n)
    embedding = _mean(group, ("mahalanobis_mean_shift", "raw_state_mahalanobis", "raw_state_knn_distance", "base_lgbm_raw_state_mahalanobis", "base_lgbm_raw_state_knn_distance", "meta_lgbm_raw_state_mahalanobis", "meta_lgbm_raw_state_knn_distance"))
    sim = _mean(group, ("regime_centroid_similarity_train", "base_lgbm_regime_centroid_similarity_train", "meta_lgbm_regime_centroid_similarity_train"))
    if not np.isfinite(embedding) and np.isfinite(sim):
        embedding = 1.0 - sim
    _add_metric(out, meta, "feature_embedding_distance", embedding, n)
    _add_metric(out, meta, "raw_state_reconstruction_error", _mean(group, ("raw_state_reconstruction_error", "base_lgbm_raw_state_reconstruction_error", "meta_lgbm_raw_state_reconstruction_error")), n)
    _add_metric(out, meta, "feature_psi_p95", _quantile_any(group, 0.95, candidates=("feature_drift_psi_core_80", "feature_drift_psi_bin_mean", "raw_state_psi_mean", "raw_state_svd_psi_mean", "base_lgbm_feature_drift_psi_core_80", "base_lgbm_raw_state_psi_mean", "base_lgbm_raw_state_svd_psi_mean", "meta_lgbm_feature_drift_psi_core_80", "meta_lgbm_raw_state_psi_mean", "meta_lgbm_raw_state_svd_psi_mean"), prefixes=("feature_drift_psi_", "base_lgbm_feature_drift_psi_", "meta_lgbm_feature_drift_psi_")), n)
    _add_metric(out, meta, "feature_psi_max", _max_any(group, candidates=("feature_drift_psi_bin_max", "raw_state_psi_max", "raw_state_svd_psi_max", "base_lgbm_raw_state_psi_max", "base_lgbm_raw_state_svd_psi_max", "meta_lgbm_raw_state_psi_max", "meta_lgbm_raw_state_svd_psi_max"), prefixes=("feature_drift_psi_", "base_lgbm_feature_drift_psi_", "meta_lgbm_feature_drift_psi_")), n)
    _add_metric(out, meta, "feature_ks_p95", _quantile_any(group, 0.95, candidates=("feature_drift_ks_core", "feature_drift_ks_bin_mean", "raw_state_ks_mean", "raw_state_svd_ks_mean", "base_lgbm_feature_drift_ks_core", "base_lgbm_raw_state_ks_mean", "base_lgbm_raw_state_svd_ks_mean", "meta_lgbm_feature_drift_ks_core", "meta_lgbm_raw_state_ks_mean", "meta_lgbm_raw_state_svd_ks_mean"), prefixes=("feature_drift_ks_", "base_lgbm_feature_drift_ks_", "meta_lgbm_feature_drift_ks_")), n)
    _add_metric(out, meta, "feature_ks_max", _max_any(group, candidates=("feature_drift_ks_bin_max", "raw_state_ks_max", "raw_state_svd_ks_max", "base_lgbm_raw_state_ks_max", "base_lgbm_raw_state_svd_ks_max", "meta_lgbm_raw_state_ks_max", "meta_lgbm_raw_state_svd_ks_max"), prefixes=("feature_drift_ks_", "base_lgbm_feature_drift_ks_", "meta_lgbm_feature_drift_ks_")), n)
    _add_metric(out, meta, "feature_wasserstein_mean", _mean_any(group, candidates=("feature_wasserstein_mean", "wasserstein_mean", "raw_state_wasserstein_mean", "base_lgbm_feature_wasserstein_mean", "base_lgbm_raw_state_wasserstein_mean", "meta_lgbm_feature_wasserstein_mean", "meta_lgbm_raw_state_wasserstein_mean")), n)
    _add_metric(out, meta, "feature_cov_shift", _mean(group, ("feature_drift_cov_shift", "frobenius_corr_shift", "base_lgbm_feature_drift_cov_shift", "base_lgbm_frobenius_corr_shift", "meta_lgbm_feature_drift_cov_shift", "meta_lgbm_frobenius_corr_shift")), n)
    _add_metric(out, meta, "regime_centroid_similarity", _mean(group, ("regime_centroid_similarity_train", "regime_centroid_similarity_train_window_mean", "base_lgbm_regime_centroid_similarity_train", "base_lgbm_regime_centroid_similarity_train_window_mean", "meta_lgbm_regime_centroid_similarity_train", "meta_lgbm_regime_centroid_similarity_train_window_mean")), n)
    _add_metric(out, meta, "regime_centroid_pc0_similarity", _mean(group, ("regime_centroid_similarity_train_pc0", "base_lgbm_regime_centroid_similarity_train_pc0", "meta_lgbm_regime_centroid_similarity_train_pc0")), n)
    _add_metric(out, meta, "regime_centroid_pc1_similarity", _mean(group, ("regime_centroid_similarity_train_pc1", "base_lgbm_regime_centroid_similarity_train_pc1", "meta_lgbm_regime_centroid_similarity_train_pc1")), n)
    _add_metric(out, meta, "regime_centroid_pc2_similarity", _mean(group, ("regime_centroid_similarity_train_pc2", "base_lgbm_regime_centroid_similarity_train_pc2", "meta_lgbm_regime_centroid_similarity_train_pc2")), n)
    _add_metric(out, meta, "mahalanobis_mean", _mean(group, ("mahalanobis_mean_shift", "raw_state_mahalanobis", "base_lgbm_raw_state_mahalanobis", "meta_lgbm_raw_state_mahalanobis")), n)
    _add_metric(out, meta, "mahalanobis_p90", _quantile(group, ("mahalanobis_mean_shift", "raw_state_mahalanobis", "base_lgbm_raw_state_mahalanobis", "meta_lgbm_raw_state_mahalanobis"), 0.90), n)
    _add_metric(out, meta, "raw_state_knn_distance_mean", _mean(group, ("raw_state_knn_distance",)), n)
    _add_metric(out, meta, "raw_state_min_cluster_distance_mean", _mean(group, ("raw_state_min_cluster_distance",)), n)
    _add_metric(out, meta, "raw_state_transition_norm_mean", _mean(group, ("raw_state_transition_norm",)), n)
    _add_metric(out, meta, "raw_state_transition_mahalanobis_mean", _mean(group, ("raw_state_transition_mahalanobis",)), n)
    _add_metric(out, meta, "state_log_likelihood_mean", _mean(group, ("state_log_likelihood",)), n)
    _add_metric(out, meta, "state_tod_mahalanobis_mean", _mean(group, ("state_tod_mahalanobis",)), n)
    _add_metric(out, meta, "raw_state_psi_mean", _mean(group, ("raw_state_psi_mean",)), n)
    _add_metric(out, meta, "raw_state_psi_max", _mean(group, ("raw_state_psi_max",)), n)
    _add_metric(out, meta, "raw_state_ks_mean", _mean(group, ("raw_state_ks_mean",)), n)
    _add_metric(out, meta, "raw_state_ks_max", _mean(group, ("raw_state_ks_max",)), n)
    _add_metric(out, meta, "raw_state_svd_psi_mean", _mean(group, ("raw_state_svd_psi_mean",)), n)
    _add_metric(out, meta, "raw_state_svd_psi_max", _mean(group, ("raw_state_svd_psi_max",)), n)
    _add_metric(out, meta, "raw_state_svd_ks_mean", _mean(group, ("raw_state_svd_ks_mean",)), n)
    _add_metric(out, meta, "raw_state_svd_ks_max", _mean(group, ("raw_state_svd_ks_max",)), n)
    _add_metric(out, meta, "raw_state_svd_component_mean", _mean(group, ("raw_state_svd_mean",)), n)
    _add_metric(out, meta, "raw_state_svd_component_std", _mean(group, ("raw_state_svd_std",)), n)
    _add_metric(
        out,
        meta,
        "raw_state_svd_l2_norm",
        _row_l2_mean(group, candidates=tuple(f"raw_state_svd_{i:02d}" for i in range(16))),
        n,
    )

    if label_ret is not None:
        ret_clean = pd.to_numeric(label_ret, errors="coerce").replace([np.inf, -np.inf], np.nan)
        target_rank_ret = _rank_return_frame(label_rank, ret_clean)
        _add_metric(out, meta, "target_return_mean", float(ret_clean.mean()), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_return_volatility", float(ret_clean.std(ddof=0)), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_positive_rate", float((ret_clean > 0.0).mean()), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_top_bottom_spread", _metric_value_top_bottom_spread_from_rank_return(target_rank_ret), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_return_p10", float(ret_clean.quantile(0.10)), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_return_p90", float(ret_clean.quantile(0.90)), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_downside_tail", float(ret_clean.quantile(0.05)), int(ret_clean.notna().sum()))
        _add_metric(out, meta, "target_upside_tail", float(ret_clean.quantile(0.95)), int(ret_clean.notna().sum()))
    soft = _soft_label_series(label_group)
    if soft is not None:
        soft_clean = pd.to_numeric(soft, errors="coerce").replace([np.inf, -np.inf], np.nan)
        _add_metric(out, meta, "soft_label_mean", float(soft_clean.mean()), int(soft_clean.notna().sum()))
        _add_metric(out, meta, "soft_label_std", float(soft_clean.std(ddof=0)), int(soft_clean.notna().sum()))
    hard = _hard_label_series(label_group)
    if hard is not None:
        hard_clean = pd.to_numeric(hard, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        _add_metric(out, meta, "hard_label_positive_rate", float((hard_clean > 0.0).mean()), int(len(hard_clean)))
    _add_metric(out, meta, "near_barrier_ambiguity_mean", _mean(label_group, ("near_barrier_ambiguity", "barrier_ambiguity_score", "outcome_ambiguity", "near_barrier_score")), label_n)
    _add_metric(out, meta, "near_barrier_share", _share_threshold(label_group, ("near_barrier_ambiguity", "barrier_ambiguity_score", "outcome_ambiguity", "near_barrier_score"), 0.5), label_n)
    _add_metric(out, meta, "unstable_label_score_mean", _mean(label_group, ("unstable_label_score", "label_instability_score", "cost_slippage_flip_score")), label_n)
    _add_metric(out, meta, "label_flip_under_cost_share", _share_threshold(label_group, ("label_flip_under_cost", "outcome_flip_under_cost_share", "cost_slippage_flip_score"), 0.5), label_n)
    _add_metric(out, meta, "mfe_magnitude_mean", _mean(label_group, ("mfe", "mfe_pct", "mfe_return", "path_mfe", "max_favorable_excursion")), label_n)
    _add_metric(out, meta, "mae_magnitude_mean", _mean_abs_any(label_group, candidates=("mae", "mae_pct", "mae_return", "path_mae", "max_adverse_excursion")), label_n)
    mfe = _numeric_series(label_group, ("mfe", "mfe_pct", "mfe_return", "path_mfe", "max_favorable_excursion"))
    mae = _numeric_series(label_group, ("mae", "mae_pct", "mae_return", "path_mae", "max_adverse_excursion"))
    if mfe is not None and mae is not None:
        ratio = pd.to_numeric(mfe, errors="coerce") / (pd.to_numeric(mae, errors="coerce").abs() + 1e-12)
        _add_metric(out, meta, "mfe_mae_ratio_mean", float(ratio.replace([np.inf, -np.inf], np.nan).mean()), int(ratio.notna().sum()))
    _add_metric(out, meta, "mfe_time_frac_mean", _mean(label_group, ("mfe_time_frac", "time_to_mfe_frac", "mfe_bar_frac")), label_n)
    _add_metric(out, meta, "mae_time_frac_mean", _mean(label_group, ("mae_time_frac", "time_to_mae_frac", "mae_bar_frac")), label_n)
    _add_metric(out, meta, "quick_profit_share", _share_threshold(label_group, ("quick_profit_score", "quick_mfe_score", "mfe_speed_score"), 0.5), label_n)
    _add_metric(out, meta, "mfe_before_mae_share", _share_threshold(label_group, ("mfe_before_mae", "profit_before_adverse", "mfe_first"), 0.5), label_n)

    if base is not None:
        base_clean = pd.to_numeric(base, errors="coerce").replace([np.inf, -np.inf], np.nan)
        _add_metric(out, meta, "base_prediction_mean", float(base_clean.mean()), int(base_clean.notna().sum()))
        _add_metric(out, meta, "base_prediction_std", float(base_clean.std(ddof=0)), int(base_clean.notna().sum()))
        _add_metric(out, meta, "base_prediction_p10", float(base_clean.quantile(0.10)), int(base_clean.notna().sum()))
        _add_metric(out, meta, "base_prediction_p50", float(base_clean.quantile(0.50)), int(base_clean.notna().sum()))
        _add_metric(out, meta, "base_prediction_p90", float(base_clean.quantile(0.90)), int(base_clean.notna().sum()))
        _add_metric(out, meta, "base_high_confidence_share", float((np.abs(base_clean - 0.5) >= 0.25).mean()), int(base_clean.notna().sum()))
        _add_metric(out, meta, "base_neutral_share", float((np.abs(base_clean - 0.5) <= 0.05).mean()), int(base_clean.notna().sum()))
    if meta_pred is not None:
        meta_clean = pd.to_numeric(meta_pred, errors="coerce").replace([np.inf, -np.inf], np.nan)
        _add_metric(out, meta, "meta_prediction_mean", float(meta_clean.mean()), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "meta_prediction_std", float(meta_clean.std(ddof=0)), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "high_confidence_share", float((np.abs(meta_clean - 0.5) >= 0.25).mean()), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "meta_prediction_p10", float(meta_clean.quantile(0.10)), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "meta_prediction_p50", float(meta_clean.quantile(0.50)), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "meta_prediction_p90", float(meta_clean.quantile(0.90)), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "meta_high_confidence_share", float((np.abs(meta_clean - 0.5) >= 0.25).mean()), int(meta_clean.notna().sum()))
        _add_metric(out, meta, "meta_neutral_share", float((np.abs(meta_clean - 0.5) <= 0.05).mean()), int(meta_clean.notna().sum()))
    if rank is not None:
        rank_clean = pd.to_numeric(rank, errors="coerce").replace([np.inf, -np.inf], np.nan)
        _add_metric(out, meta, "threshold_pass_share", float((rank_clean >= 0.80).mean()), int(rank_clean.notna().sum()))
        _add_metric(out, meta, "threshold_pass_share_top10", float((rank_clean >= 0.90).mean()), int(rank_clean.notna().sum()))
        _add_metric(out, meta, "threshold_pass_share_top20", float((rank_clean >= 0.80).mean()), int(rank_clean.notna().sum()))
        _add_metric(out, meta, "threshold_pass_share_top30", float((rank_clean >= 0.70).mean()), int(rank_clean.notna().sum()))
        _add_metric(out, meta, "rank_std", float(rank_clean.std(ddof=0)), int(rank_clean.notna().sum()))
        _add_metric(out, meta, "rank_p90_minus_p10", float(rank_clean.quantile(0.90) - rank_clean.quantile(0.10)), int(rank_clean.notna().sum()))
    if base is not None and meta_pred is not None:
        diff = (pd.to_numeric(base, errors="coerce") - pd.to_numeric(meta_pred, errors="coerce")).abs().replace([np.inf, -np.inf], np.nan)
        _add_metric(out, meta, "base_meta_disagreement_mean", float(diff.mean()), int(diff.notna().sum()))
        _add_metric(out, meta, "base_meta_disagreement_p90", float(diff.quantile(0.90)), int(diff.notna().sum()))
    _add_metric(out, meta, "base_model_disagreement_mean", _mean_any(group, candidates=("base_models_pred_std", "base_models_pred_range", "base_lgbm_prob_std", "base_lgbm_raw_score_std")), n)
    _add_metric(out, meta, "score_margin_top10", _mean_any(group, candidates=("score_margin_top10", "base_lgbm_score_margin_top10", "meta_lgbm_score_margin_top10")), n)
    _add_metric(out, meta, "score_margin_top20", _mean_any(group, candidates=("score_margin_top20", "base_lgbm_score_margin_top20", "meta_lgbm_score_margin_top20")), n)
    _add_metric(out, meta, "score_margin_top30", _mean_any(group, candidates=("score_margin_top30", "base_lgbm_score_margin_top30", "meta_lgbm_score_margin_top30")), n)

    _add_metric(out, meta, "rare_leaf_share", _mean(group, ("rare_leaf_fraction", "rare_leaf_low_support_score")), n)
    _add_metric(out, meta, "leaf_surprisal_mean", _mean(group, ("leaf_surprisal_mean", "base_lgbm_leaf_surprisal_mean", "meta_lgbm_leaf_surprisal_mean")), n)
    _add_metric(out, meta, "leaf_centroid_distance", _mean(group, ("leaf_centroid_dist_norm_mean", "leaf_centroid_dist_mean", "base_lgbm_leaf_centroid_dist_norm_mean", "meta_lgbm_leaf_centroid_dist_norm_mean")), n)
    _add_metric(out, meta, "contribution_drift", _mean(group, ("contribution_drift_score", "contrib_abs_sum", "base_lgbm_contrib_abs_sum", "meta_lgbm_contrib_abs_sum")), n)
    _add_metric(out, meta, "path_instability", _mean(group, ("score_path_std", "rank_path_std", "base_lgbm_score_path_std", "meta_lgbm_score_path_std")), n)
    _add_metric(out, meta, "rare_leaf_p90", _quantile_any(group, 0.90, candidates=("rare_leaf_fraction", "rare_leaf_low_support_score", "base_lgbm_rare_leaf_fraction", "meta_lgbm_rare_leaf_fraction")), n)
    _add_metric(out, meta, "leaf_count_p10", _mean_any(group, candidates=("leaf_count_p10", "base_lgbm_leaf_count_p10", "meta_lgbm_leaf_count_p10")), n)
    _add_metric(out, meta, "leaf_count_min", _mean_any(group, candidates=("leaf_count_min", "base_lgbm_leaf_count_min", "meta_lgbm_leaf_count_min")), n)
    _add_metric(out, meta, "leaf_train_freq_mean", _mean_any(group, candidates=("leaf_train_freq_mean", "base_lgbm_leaf_train_freq_mean", "meta_lgbm_leaf_train_freq_mean")), n)
    _add_metric(out, meta, "leaf_train_freq_p10", _mean_any(group, candidates=("leaf_train_freq_p10", "base_lgbm_leaf_train_freq_p10", "meta_lgbm_leaf_train_freq_p10")), n)
    _add_metric(out, meta, "leaf_train_freq_min", _mean_any(group, candidates=("leaf_train_freq_min", "base_lgbm_leaf_train_freq_min", "meta_lgbm_leaf_train_freq_min")), n)
    _add_metric(out, meta, "leaf_train_freq_std", _mean_any(group, candidates=("leaf_train_freq_std", "base_lgbm_leaf_train_freq_std", "meta_lgbm_leaf_train_freq_std")), n)
    _add_metric(out, meta, "leaf_surprisal_p90", _mean_any(group, candidates=("leaf_surprisal_p90", "base_lgbm_leaf_surprisal_p90", "meta_lgbm_leaf_surprisal_p90")), n)
    _add_metric(out, meta, "leaf_surprisal_max", _mean_any(group, candidates=("leaf_surprisal_max", "base_lgbm_leaf_surprisal_max", "meta_lgbm_leaf_surprisal_max")), n)
    _add_metric(out, meta, "leaf_low_freq_fraction", _mean_any(group, candidates=("leaf_low_freq_fraction", "base_lgbm_leaf_low_freq_fraction", "meta_lgbm_leaf_low_freq_fraction")), n)
    _add_metric(out, meta, "leaf_target_dispersion", _mean_any(group, candidates=("leaf_target_std_mean", "leaf_target_iqr_mean", "leaf_target_range_mean", "base_lgbm_leaf_target_std_mean", "meta_lgbm_leaf_target_std_mean")), n)
    _add_metric(out, meta, "leaf_target_iqr_mean", _mean_any(group, candidates=("leaf_target_iqr_mean", "base_lgbm_leaf_target_iqr_mean", "meta_lgbm_leaf_target_iqr_mean")), n)
    _add_metric(out, meta, "leaf_target_range_mean", _mean_any(group, candidates=("leaf_target_range_mean", "base_lgbm_leaf_target_range_mean", "meta_lgbm_leaf_target_range_mean")), n)
    _add_metric(out, meta, "leaf_target_abs_mean", _mean_any(group, candidates=("leaf_target_abs_mean", "base_lgbm_leaf_target_abs_mean", "meta_lgbm_leaf_target_abs_mean")), n)
    _add_metric(out, meta, "leaf_centroid_radius_mean", _mean_any(group, candidates=("leaf_centroid_radius_mean", "base_lgbm_leaf_centroid_radius_mean", "meta_lgbm_leaf_centroid_radius_mean")), n)
    _add_metric(out, meta, "leaf_centroid_distance_p90", _mean_any(group, candidates=("leaf_centroid_dist_p90", "leaf_centroid_dist_norm_p90", "base_lgbm_leaf_centroid_dist_p90", "meta_lgbm_leaf_centroid_dist_p90")), n)
    _add_metric(out, meta, "leaf_centroid_distance_max", _mean_any(group, candidates=("leaf_centroid_dist_max", "leaf_centroid_dist_norm_max", "base_lgbm_leaf_centroid_dist_max", "meta_lgbm_leaf_centroid_dist_max")), n)
    _add_metric(out, meta, "leaf_centroid_distance_cv", _mean_any(group, candidates=("leaf_centroid_dist_cv", "base_lgbm_leaf_centroid_dist_cv", "meta_lgbm_leaf_centroid_dist_cv")), n)
    _add_metric(out, meta, "leaf_centroid_distance_rel_mean", _mean_any(group, candidates=("leaf_centroid_dist_rel_mean", "base_lgbm_leaf_centroid_dist_rel_mean", "meta_lgbm_leaf_centroid_dist_rel_mean")), n)
    _add_metric(out, meta, "leaf_centroid_distance_norm_p90", _mean_any(group, candidates=("leaf_centroid_dist_norm_p90", "base_lgbm_leaf_centroid_dist_norm_p90", "meta_lgbm_leaf_centroid_dist_norm_p90")), n)
    _add_metric(out, meta, "leaf_centroid_distance_norm_max", _mean_any(group, candidates=("leaf_centroid_dist_norm_max", "base_lgbm_leaf_centroid_dist_norm_max", "meta_lgbm_leaf_centroid_dist_norm_max")), n)
    _add_metric(out, meta, "contrib_abs_sum_mean", _mean_any(group, candidates=("contrib_abs_sum", "base_lgbm_contrib_abs_sum", "meta_lgbm_contrib_abs_sum")), n)
    _add_metric(out, meta, "contrib_l2_norm_mean", _mean_any(group, candidates=("contrib_l2_norm", "base_lgbm_contrib_l2_norm", "meta_lgbm_contrib_l2_norm")), n)
    _add_metric(out, meta, "contrib_entropy_mean", _mean_any(group, candidates=("contrib_entropy", "base_lgbm_contrib_entropy", "meta_lgbm_contrib_entropy")), n)
    _add_metric(out, meta, "contrib_top1_abs_share_mean", _mean_any(group, candidates=("contrib_top1_abs_share", "base_lgbm_contrib_top1_abs_share", "meta_lgbm_contrib_top1_abs_share")), n)
    _add_metric(out, meta, "contrib_top3_abs_share_mean", _mean_any(group, candidates=("contrib_top3_abs_share", "base_lgbm_contrib_top3_abs_share", "meta_lgbm_contrib_top3_abs_share")), n)
    _add_metric(out, meta, "contrib_balance_abs_mean", _mean_abs_any(group, candidates=("contrib_balance", "base_lgbm_contrib_balance", "meta_lgbm_contrib_balance")), n)
    _add_metric(out, meta, "num_material_contrib_features_mean", _mean_any(group, candidates=("num_material_contrib_features", "base_lgbm_num_material_contrib_features", "meta_lgbm_num_material_contrib_features")), n)
    _add_metric(out, meta, "archetype_contrib_svd_l2_norm", _row_l2_mean(group, prefixes=("archetype_contrib_svd_", "base_lgbm_archetype_contrib_svd_", "meta_lgbm_archetype_contrib_svd_")), n)
    _add_metric(out, meta, "large_leaf_value_fraction", _mean_any(group, candidates=("large_leaf_value_fraction", "base_lgbm_large_leaf_value_fraction", "meta_lgbm_large_leaf_value_fraction")), n)
    _add_metric(out, meta, "leaf_depth_mean", _mean_any(group, candidates=("leaf_depth_mean", "base_lgbm_leaf_depth_mean", "meta_lgbm_leaf_depth_mean")), n)
    _add_metric(out, meta, "leaf_depth_std", _mean_any(group, candidates=("leaf_depth_std", "base_lgbm_leaf_depth_std", "meta_lgbm_leaf_depth_std")), n)
    _add_metric(out, meta, "score_path_range", _mean_any(group, candidates=("score_100_minus_50", "score_100_minus_75", "base_lgbm_score_100_minus_50", "meta_lgbm_score_100_minus_50")), n)
    _add_metric(out, meta, "rank_path_range", _mean_any(group, candidates=("rank_100_minus_50", "rank_path_std", "base_lgbm_rank_100_minus_50", "meta_lgbm_rank_100_minus_50")), n)
    _add_metric(out, meta, "rank_bin_lift_oof", _mean_any(group, candidates=("rank_bin_lift_oof", "base_lgbm_rank_bin_lift_oof", "meta_lgbm_rank_bin_lift_oof")), n)
    _add_metric(out, meta, "rank_bin_net_ret_oof", _mean_any(group, candidates=("rank_bin_net_ret_oof", "base_lgbm_rank_bin_net_ret_oof", "meta_lgbm_rank_bin_net_ret_oof")), n)
    _add_metric(out, meta, "rank_bin_se_oof", _mean_any(group, candidates=("rank_bin_se_oof", "base_lgbm_rank_bin_se_oof", "meta_lgbm_rank_bin_se_oof")), n)

    _add_metric(out, meta, "uncertainty_score_mean", _mean(group, ("uncertainty_score",)), n)
    _add_metric(out, meta, "prob_uncertainty_mean", _mean(group, ("prob_uncertainty", "base_lgbm_prob_uncertainty", "meta_lgbm_prob_uncertainty")), n)
    _add_metric(out, meta, "entropy_mean", _mean(group, ("entropy", "base_lgbm_entropy", "meta_lgbm_entropy")), n)
    disagreement = _mean(group, ("disagreement", "raw_score_std", "prob_std", "base_lgbm_raw_score_std", "meta_lgbm_raw_score_std"))
    _add_metric(out, meta, "disagreement_mean", disagreement, n)
    _add_metric(out, meta, "uncertainty_score_p90", _quantile(group, ("uncertainty_score",), 0.90), n)
    _add_metric(out, meta, "prob_uncertainty_p90", _quantile_any(group, 0.90, candidates=("prob_uncertainty", "base_lgbm_prob_uncertainty", "meta_lgbm_prob_uncertainty")), n)
    _add_metric(out, meta, "prob_uncertainty_max", _max_any(group, candidates=("prob_uncertainty", "base_lgbm_prob_uncertainty", "meta_lgbm_prob_uncertainty")), n)
    _add_metric(out, meta, "entropy_p90", _quantile_any(group, 0.90, candidates=("entropy", "base_lgbm_entropy", "meta_lgbm_entropy")), n)
    _add_metric(out, meta, "entropy_max", _max_any(group, candidates=("entropy", "base_lgbm_entropy", "meta_lgbm_entropy")), n)
    _add_metric(out, meta, "disagreement_p90", _quantile_any(group, 0.90, candidates=("disagreement", "raw_score_std", "prob_std", "base_lgbm_raw_score_std", "meta_lgbm_raw_score_std")), n)
    _add_metric(out, meta, "disagreement_max", _max_any(group, candidates=("disagreement", "raw_score_std", "prob_std", "base_lgbm_raw_score_std", "meta_lgbm_raw_score_std")), n)
    _add_metric(out, meta, "prob_std_mean", _mean_any(group, candidates=("prob_std", "base_lgbm_prob_std", "meta_lgbm_prob_std")), n)
    _add_metric(out, meta, "raw_score_std_mean", _mean_any(group, candidates=("raw_score_std", "base_lgbm_raw_score_std", "meta_lgbm_raw_score_std")), n)
    _add_metric(out, meta, "margin_from_neutral_mean", _mean_any(group, candidates=("margin_from_neutral", "base_lgbm_margin_from_neutral", "meta_lgbm_margin_from_neutral")), n)
    _add_metric(out, meta, "variance_proxy_mean", _mean_any(group, candidates=("variance_proxy", "base_lgbm_variance_proxy", "meta_lgbm_variance_proxy")), n)

    if label_net is not None:
        net_clean = pd.to_numeric(label_net, errors="coerce").replace([np.inf, -np.inf], np.nan)
        perf_rank_ret = _rank_return_frame(label_rank, net_clean)
        if label_traded_mask.any():
            _add_metric(out, meta, "net_return_per_accepted_trade", float(net_clean.loc[label_traded_mask].mean()), int(np.isfinite(net_clean.loc[label_traded_mask]).sum()))
        _add_metric(out, meta, "top_bottom_return_spread", _metric_value_top_bottom_spread_from_rank_return(perf_rank_ret), int(net_clean.notna().sum()))
        _add_metric(out, meta, "rolling_rank_ic", _corr_from_rank_return(perf_rank_ret), int(net_clean.notna().sum()))
        _add_metric(out, meta, "score_bucket_monotonicity", _metric_value_monotonicity_from_rank_return(perf_rank_ret), int(net_clean.notna().sum()))
        _add_metric(out, meta, "realized_return_top_decile", _metric_value_slice_return_from_rank_return(perf_rank_ret, 0.10), int(net_clean.notna().sum()))
        _add_metric(out, meta, "realized_return_top_quintile", _metric_value_slice_return_from_rank_return(perf_rank_ret, 0.20), int(net_clean.notna().sum()))
        _add_metric(out, meta, "realized_return_top_quarter", _metric_value_slice_return_from_rank_return(perf_rank_ret, 0.25), int(net_clean.notna().sum()))
        _add_metric(out, meta, "realized_return_bottom_decile", _metric_value_bottom_slice_return_from_rank_return(perf_rank_ret, 0.10), int(net_clean.notna().sum()))
        _add_metric(out, meta, "top_decile_minus_bottom_decile_return", _metric_value_top_minus_bottom_from_rank_return(perf_rank_ret, 0.10), int(net_clean.notna().sum()))
        _add_metric(out, meta, "top_quarter_minus_bottom_quarter_return", _metric_value_top_minus_bottom_from_rank_return(perf_rank_ret, 0.25), int(net_clean.notna().sum()))
        _add_metric(out, meta, "rank_ic_top30", _metric_value_rank_ic_slice_from_rank_return(perf_rank_ret, 0.30), int(net_clean.notna().sum()))
        _add_metric(out, meta, "rank_ic_top20", _metric_value_rank_ic_slice_from_rank_return(perf_rank_ret, 0.20), int(net_clean.notna().sum()))
        _add_metric(out, meta, "rank_ic_top10", _metric_value_rank_ic_slice_from_rank_return(perf_rank_ret, 0.10), int(net_clean.notna().sum()))
        _add_metric(out, meta, "bps_weighted_hit_top30", _metric_value_bps_weighted_hit_slice_from_rank_return(perf_rank_ret, 0.30), int(net_clean.notna().sum()))
        _add_metric(out, meta, "bps_weighted_hit_top20", _metric_value_bps_weighted_hit_slice_from_rank_return(perf_rank_ret, 0.20), int(net_clean.notna().sum()))
        _add_metric(out, meta, "bps_weighted_hit_top10", _metric_value_bps_weighted_hit_slice_from_rank_return(perf_rank_ret, 0.10), int(net_clean.notna().sum()))
        _add_metric(out, meta, "meta_confidence_top_return", _mean(label_group.loc[np.abs(pd.to_numeric(label_meta_pred, errors="coerce") - 0.5) >= np.nanquantile(np.abs(pd.to_numeric(label_meta_pred, errors="coerce") - 0.5), 0.75)] if label_meta_pred is not None and pd.to_numeric(label_meta_pred, errors="coerce").notna().sum() >= 4 else label_group.iloc[0:0], ("net_return", "realized_net_return", "shadow_exit_return")), int(net_clean.notna().sum()))
        _add_metric(out, meta, "meta_confidence_bucket_monotonicity", _metric_value_confidence_monotonicity(label_group), int(net_clean.notna().sum()))
        slope, abs_error = _metric_value_calibration(label_group)
        _add_metric(out, meta, "calibration_slope", slope, int(net_clean.notna().sum()))
        _add_metric(out, meta, "calibration_abs_error", abs_error, int(net_clean.notna().sum()))
        hit_top20 = float("nan")
        if len(perf_rank_ret) >= 5:
            top = perf_rank_ret["score"] >= perf_rank_ret["score"].quantile(0.80)
            if top.any():
                hit_top20 = float((perf_rank_ret.loc[top, "ret"] > 0.0).mean())
        _add_metric(out, meta, "hit_rate_top20", hit_top20, int(net_clean.notna().sum()))
        expected_hit = _numeric_series(label_group, ("estimated_hit_rate", "meta_pred", "calibrated_score"))
        if expected_hit is not None:
            tmp = pd.DataFrame({"net": net_clean, "expected": expected_hit}).dropna()
            if len(tmp):
                surprise = np.abs((tmp["net"] > 0.0).astype(float) - np.clip(tmp["expected"], 0.0, 1.0))
                _add_metric(out, meta, "meta_error_surprise", float(surprise.mean()), int(len(tmp)))
                hit = (tmp["net"] > 0.0).astype(float)
                expected = np.clip(tmp["expected"], 0.0, 1.0)
                _add_metric(out, meta, "expected_hit_gap_mean", float((hit - expected).mean()), int(len(tmp)))
                _add_metric(out, meta, "expected_hit_abs_error", float(np.abs(hit - expected).mean()), int(len(tmp)))
                _add_metric(out, meta, "brier_score", float(np.square(hit - expected).mean()), int(len(tmp)))
                _add_metric(out, meta, "adwin_meta_error_shift", _adwin_like_shift(np.abs(hit - expected)), int(len(tmp)))
        expected_ev = _numeric_series(label_group, ("estimated_ev_net_return", "ev_adjusted_net_return_after_friction"))
        if expected_ev is not None:
            tmp = pd.DataFrame({"net": net_clean, "expected": expected_ev}).dropna()
            if len(tmp):
                residual = tmp["net"] - tmp["expected"]
                _add_metric(out, meta, "accepted_trade_residual_return", float(residual.mean()), int(len(tmp)))
                _add_metric(out, meta, "expected_ev_gap_mean", float(residual.mean()), int(len(tmp)))
                _add_metric(out, meta, "expected_ev_abs_error", float(np.abs(residual).mean()), int(len(tmp)))
                _add_metric(out, meta, "residual_return_volatility", float(residual.std(ddof=0)), int(len(tmp)))
                _add_metric(out, meta, "residual_return_p90_abs", float(np.abs(residual).quantile(0.90)), int(len(tmp)))
        _add_metric(out, meta, "adwin_net_return_shift", _adwin_like_shift(net_clean), int(net_clean.notna().sum()))
        if label_rank is not None:
            top_bottom_series = pd.Series(np.nan, index=label_group.index, dtype=float)
            tmp_tb = pd.DataFrame({"rank": label_rank, "net": net_clean}).replace([np.inf, -np.inf], np.nan)
            if tmp_tb["rank"].notna().sum() >= 5:
                hi = tmp_tb["rank"] >= tmp_tb["rank"].quantile(0.80)
                lo = tmp_tb["rank"] <= tmp_tb["rank"].quantile(0.20)
                top_bottom_series.loc[hi] = tmp_tb.loc[hi, "net"]
                top_bottom_series.loc[lo] = -tmp_tb.loc[lo, "net"]
            _add_metric(out, meta, "adwin_top_bottom_spread_shift", _adwin_like_shift(top_bottom_series), int(net_clean.notna().sum()))
            _add_metric(out, meta, "adwin_hit_rate_top20_shift", _adwin_like_shift(_top20_hit_series(label_group)), int(net_clean.notna().sum()))

    if label_gross is not None and label_net is not None:
        tmp = pd.DataFrame({"gross": label_gross, "net": label_net}).dropna()
        if len(tmp):
            _add_metric(out, meta, "gross_to_net_drag_bps", float(((tmp["gross"] - tmp["net"]) * 10000.0).mean()), int(len(tmp)))
    _add_metric(out, meta, "spread_slippage_drag_bps", _mean(group, ("spread_bps", "ticker_spread_bps", "orderbook_slippage_bps", "slippage_bps", "expected_fill_slippage_bps")), n)
    _add_metric(out, meta, "entry_gap_bps", _mean(group, ("entry_gap_bps", "adverse_signal_gap_bps", "hourly_close_to_latest_decision_price_bps")), n)
    _add_metric(out, meta, "stop_exit_replay_gap_bps", _mean(group, ("shadow_exit_vs_live_stop_bps", "shadow_stop_gap_bps", "latest_stop_gap_bps")), n)
    _add_metric(out, meta, "spread_bps_mean", _mean(group, ("spread_bps", "ticker_spread_bps", "decision_spread_bps")), n)
    _add_metric(out, meta, "spread_bps_p90", _quantile(group, ("spread_bps", "ticker_spread_bps", "decision_spread_bps"), 0.90), n)
    _add_metric(out, meta, "slippage_bps_mean", _mean(group, ("orderbook_slippage_bps", "slippage_bps", "expected_fill_slippage_bps", "modeled_slippage_bps")), n)
    _add_metric(out, meta, "slippage_bps_p90", _quantile(group, ("orderbook_slippage_bps", "slippage_bps", "expected_fill_slippage_bps", "modeled_slippage_bps"), 0.90), n)
    _add_metric(out, meta, "entry_gap_bps_p90", _quantile(group, ("entry_gap_bps", "adverse_signal_gap_bps", "hourly_close_to_latest_decision_price_bps"), 0.90), n)
    _add_metric(out, meta, "adverse_rejection_share", _bool_share_from_columns(group, ("adverse_rejection", "adverse_gap_rejected", "blocked_by_adverse_gap")), n)
    _add_metric(out, meta, "fill_slippage_bps_mean", _mean(group, ("fill_slippage_bps", "live_fill_slippage_bps", "shadow_fill_slippage_bps")), n)
    _add_metric(out, meta, "shadow_live_entry_gap_bps", _mean(group, ("shadow_live_entry_gap_bps", "live_vs_shadow_entry_bps", "entry_replay_gap_bps")), n)
    _add_metric(out, meta, "shadow_live_exit_gap_bps", _mean(group, ("shadow_live_exit_gap_bps", "live_vs_shadow_exit_bps", "exit_replay_gap_bps")), n)
    _add_metric(out, meta, "exit_spread_gap_bps", _mean(group, ("exit_spread_gap_bps", "exit_spread_bps", "close_spread_bps")), n)
    _add_metric(out, meta, "stop_update_failure_share", _bool_share_from_columns(group, ("stop_update_failed", "stop_replace_failed", "protective_stop_update_failed")), n)
    _add_metric(out, meta, "stop_violation_share", _bool_share_from_columns(group, ("stop_violation", "software_stop_breach", "protective_stop_breached")), n)
    _add_metric(out, meta, "stop_replacement_skip_share", _bool_share_from_columns(group, ("stop_replacement_skipped", "stop_update_skipped", "protective_stop_skip")), n)
    expected_edge = _numeric_series(group, ("expected_edge_bps", "expected_net_edge_bps", "expected_ev_net_bps"))
    costs = _numeric_series(group, ("spread_bps", "ticker_spread_bps", "orderbook_slippage_bps", "expected_fill_slippage_bps"))
    if expected_edge is not None and costs is not None:
        ratio = pd.to_numeric(costs, errors="coerce") / (pd.to_numeric(expected_edge, errors="coerce").abs() + 1e-12)
        _add_metric(out, meta, "execution_cost_edge_ratio", float(ratio.replace([np.inf, -np.inf], np.nan).mean()), int(ratio.notna().sum()))
    return out


def _recent_bar_timestamp_series(df: pd.DataFrame) -> pd.Series | None:
    return _timestamp_series(
        df,
        (
            "signal_bar_ts",
            "timestamp",
            "signal_ts",
            "bar_ts",
            "entry_ts",
            "_drift_ts",
        ),
    )


def _recent_bar_market_data_root(path: str | Path | None = None) -> Path:
    raw = path or os.getenv("EPM_RECENT_BAR_MARKET_DATA_ROOT") or "data_perp/exchanges/krakenfutures/orderbook_hourly"
    return Path(raw)


def _recent_bar_symbol_file_stem(symbol: Any) -> str:
    raw = str(symbol or "").strip().upper()
    if not raw:
        return ""
    raw = raw.replace("-", "_")
    if "/" in raw:
        base, quote = raw.split("/", 1)
        quote = quote.replace(":", "_").replace("/", "_")
        return f"{base}_{quote}"
    return raw.replace(":", "_").replace("/", "_")


def _recent_bar_side(row: Mapping[str, Any]) -> float:
    side = str(row.get("side") or "").strip().lower()
    strategy = str(row.get("strategy_id") or "").strip().lower()
    if side.startswith("short") or strategy.startswith("short_") or "_short_" in strategy:
        return -1.0
    if side.startswith("long") or strategy.startswith("long_") or "_long_" in strategy:
        return 1.0
    return 1.0


def _recent_bar_price_series(path: Path) -> pd.Series | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return None
    if frame.empty:
        return None
    if not isinstance(frame.index, pd.DatetimeIndex):
        ts = _timestamp_series(frame, ("timestamp", "snapshot_ts", "bar_ts", "datetime"))
        if ts is None:
            return None
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(ts)
    if frame.index.tz is None:
        frame.index = frame.index.tz_localize("UTC")
    else:
        frame.index = frame.index.tz_convert("UTC")
    price: pd.Series | None = None
    if {"best_bid", "best_ask"}.issubset(frame.columns):
        bid = pd.to_numeric(frame["best_bid"], errors="coerce")
        ask = pd.to_numeric(frame["best_ask"], errors="coerce")
        mid = (bid + ask) / 2.0
        if mid.replace([np.inf, -np.inf], np.nan).notna().any():
            price = mid
    if price is None:
        for col in RECENT_BAR_MARKET_PRICE_COLUMNS:
            if col in frame.columns:
                candidate = pd.to_numeric(frame[col], errors="coerce")
                if candidate.replace([np.inf, -np.inf], np.nan).notna().any():
                    price = candidate
                    break
    if price is None:
        return None
    price = price.replace([np.inf, -np.inf], np.nan).sort_index()
    price = price.loc[price > 0.0].dropna()
    if price.empty:
        return None
    return price


def _recent_bar_price_at(series: pd.Series, ts: pd.Timestamp) -> float:
    ts = _as_utc_timestamp(ts).floor("h")
    if ts in series.index:
        val = float(series.loc[ts])
        return val if np.isfinite(val) and val > 0.0 else float("nan")
    try:
        loc = series.index.get_indexer([ts], method="nearest", tolerance=pd.Timedelta(minutes=1))[0]
    except Exception:
        loc = -1
    if loc < 0:
        return float("nan")
    val = float(series.iloc[loc])
    return val if np.isfinite(val) and val > 0.0 else float("nan")


def attach_recent_bar_forward_returns(
    rows: pd.DataFrame,
    *,
    asof_ts: Any = None,
    market_data_root: str | Path | None = None,
    horizon_hours: int = RECENT_BAR_DEFAULT_OUTCOME_HORIZON_HOURS,
) -> pd.DataFrame:
    """Attach side-aware close-to-close proxy outcomes from local hourly bars.

    The function is intentionally conservative: rows whose horizon has not
    elapsed get a future label_available_ts and no return, so IC/HR cannot be
    accidentally reported before the outcome can exist.
    """
    if rows is None or rows.empty:
        return rows
    existing = _recent_return_series(rows)
    if existing is not None and _finite_count(existing) > 0:
        return rows
    ts = _recent_bar_timestamp_series(rows)
    if ts is None:
        return rows
    root = _recent_bar_market_data_root(market_data_root)
    out = rows.copy()
    asof = _as_utc_timestamp(asof_ts)
    horizon = int(max(1, horizon_hours))
    returns = pd.Series(np.nan, index=out.index, dtype=float)
    bps = pd.Series(np.nan, index=out.index, dtype=float)
    availability = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns, UTC]")
    status = pd.Series("missing_market_data", index=out.index, dtype=object)
    cache: dict[str, pd.Series | None] = {}
    for idx, row in out.iterrows():
        bar_ts = ts.loc[idx]
        if pd.isna(bar_ts):
            status.loc[idx] = "missing_bar_ts"
            continue
        bar_ts = _as_utc_timestamp(bar_ts).floor("h")
        exit_ts = bar_ts + pd.Timedelta(hours=horizon)
        availability.loc[idx] = exit_ts
        if exit_ts > asof:
            status.loc[idx] = "awaiting_maturity"
            continue
        symbol = row.get("symbol", "")
        stem = _recent_bar_symbol_file_stem(symbol)
        if not stem:
            status.loc[idx] = "missing_symbol"
            continue
        path = root / f"{stem}.parquet"
        if stem not in cache:
            cache[stem] = _recent_bar_price_series(path)
        series = cache[stem]
        if series is None:
            status.loc[idx] = "missing_market_file"
            continue
        entry = _recent_bar_price_at(series, bar_ts)
        exit_ = _recent_bar_price_at(series, exit_ts)
        if not (np.isfinite(entry) and np.isfinite(exit_) and entry > 0.0 and exit_ > 0.0):
            status.loc[idx] = "missing_outcome_bar"
            continue
        ret = _recent_bar_side(row) * ((exit_ / entry) - 1.0)
        returns.loc[idx] = float(ret)
        bps.loc[idx] = float(ret * 10000.0)
        status.loc[idx] = "ok"
    if "label_available_ts" not in out.columns:
        out["label_available_ts"] = availability
    else:
        existing_avail = pd.to_datetime(out["label_available_ts"], utc=True, errors="coerce")
        out["label_available_ts"] = existing_avail.fillna(availability)
    out["recent_forward_return"] = returns
    out["recent_forward_net_return"] = returns
    out["recent_forward_bps"] = bps
    out["recent_forward_net_bps"] = bps
    out["recent_forward_horizon_hours"] = horizon
    out["recent_forward_return_source"] = "hourly_market_proxy"
    out["recent_bar_outcome_status"] = status
    return out


def _finite_count(series: pd.Series | None) -> int:
    if series is None:
        return 0
    vals = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return int(vals.notna().sum())


def _recent_top_slice(tmp: pd.DataFrame, fraction: float) -> pd.DataFrame:
    if tmp.empty or tmp["score"].nunique() < 2:
        return tmp.iloc[0:0]
    threshold = tmp["score"].quantile(1.0 - float(fraction))
    return tmp.loc[tmp["score"] >= threshold]


def _recent_bar_metric_row(
    group: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    maturity: pd.Timestamp,
    window: str,
    window_hours: int,
    scope: str,
    scope_value: str,
    model_run_id: str | None,
    policy_run_id: str | None,
    min_count: int,
) -> dict[str, Any]:
    candidate_score = _rank_score(group)
    candidate_score_rows = _finite_count(candidate_score)
    candidate_last = group["_recent_bar_ts"].max() if "_recent_bar_ts" in group.columns and len(group) else pd.NaT
    label_group = _matured_label_group(
        group,
        {
            "asof_ts": asof,
            "label_maturity_cutoff_ts": maturity,
        },
    )
    label_score = _rank_score(label_group)
    label_ret = _recent_return_series(label_group)
    rank_ret = _rank_return_frame(label_score, label_ret)
    matured_rows = int(len(rank_ret))
    availability = _timestamp_series(group, LABEL_MATURITY_TS_COLUMNS)
    pending_rows = int(((availability.notna()) & (availability > maturity)).sum()) if availability is not None else 0
    outcome_status = group.get("recent_bar_outcome_status")
    outcome_status_values = set(outcome_status.dropna().astype(str)) if outcome_status is not None else set()
    status = "ok"
    if len(group) == 0:
        status = "no_candidates"
    elif candidate_score_rows == 0:
        status = "no_scored_candidates"
    elif label_ret is None:
        status = "missing_return_column"
    elif matured_rows < int(min_count):
        if pending_rows and pending_rows >= max(1, len(group) - matured_rows):
            status = "awaiting_maturity"
        elif outcome_status_values and outcome_status_values <= {"missing_market_data", "missing_market_file", "missing_outcome_bar", "missing_symbol", "missing_bar_ts"}:
            status = "missing_outcome_bar"
        else:
            status = "insufficient_matured_labels"

    row: dict[str, Any] = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "asof_ts": asof,
        "label_maturity_cutoff_ts": maturity,
        "model_run_id": model_run_id or "",
        "policy_run_id": policy_run_id or "",
        "window": window,
        "window_hours": int(window_hours),
        "window_start_ts": asof - pd.Timedelta(hours=int(window_hours)),
        "window_end_ts": asof,
        "scope": scope,
        "scope_value": scope_value,
        "last_bar_ts": candidate_last,
        "candidate_rows": int(len(group)),
        "candidate_score_rows": int(candidate_score_rows),
        "label_available_rows": int(len(label_group)),
        "pending_label_rows": int(pending_rows),
        "matured_rows": int(matured_rows),
        "status": status,
        "rank_ic": float("nan"),
        "hit_rate": float("nan"),
        "mean_return": float("nan"),
        "median_return": float("nan"),
        "bps_weighted_hit": float("nan"),
    }
    if matured_rows >= int(min_count):
        row["rank_ic"] = _corr_from_rank_return(rank_ret)
        row["hit_rate"] = float((rank_ret["ret"] > 0.0).mean())
        row["mean_return"] = float(rank_ret["ret"].mean())
        row["median_return"] = float(rank_ret["ret"].median())
        weights = np.abs(rank_ret["ret"].to_numpy(dtype=float))
        denom = float(weights.sum())
        if denom > 1e-12:
            row["bps_weighted_hit"] = float(weights[rank_ret["ret"].to_numpy(dtype=float) > 0.0].sum() / denom)

    candidate_rank = pd.DataFrame({"score": pd.to_numeric(candidate_score, errors="coerce")}) if candidate_score is not None else pd.DataFrame(columns=["score"])
    candidate_rank = candidate_rank.replace([np.inf, -np.inf], np.nan).dropna()
    for label, fraction in RECENT_BAR_TOP_FRACTIONS.items():
        candidate_top = _recent_top_slice(
            pd.DataFrame({"score": candidate_rank["score"], "ret": 0.0}) if not candidate_rank.empty else pd.DataFrame(columns=["score", "ret"]),
            fraction,
        )
        top = _recent_top_slice(rank_ret, fraction)
        row[f"{label}_candidate_rows"] = int(len(candidate_top))
        row[f"{label}_matured_rows"] = int(len(top))
        row[f"{label}_rank_ic"] = _metric_value_rank_ic_slice_from_rank_return(rank_ret, fraction)
        row[f"{label}_hit_rate"] = float((top["ret"] > 0.0).mean()) if len(top) >= int(min_count) else float("nan")
        row[f"{label}_mean_return"] = float(top["ret"].mean()) if len(top) >= int(min_count) else float("nan")
        row[f"{label}_bps_weighted_hit"] = _metric_value_bps_weighted_hit_slice_from_rank_return(rank_ret, fraction)
    return row


def build_recent_bar_metrics(
    rows: pd.DataFrame,
    *,
    asof_ts: Any = None,
    label_maturity_cutoff_ts: Any = None,
    model_run_id: str | None = None,
    policy_run_id: str | None = None,
    windows_hours: Mapping[str, int] | None = None,
    min_count: int = 3,
    market_data_root: str | Path | None = None,
    outcome_horizon_hours: int = RECENT_BAR_DEFAULT_OUTCOME_HORIZON_HOURS,
    attach_market_outcomes: bool = True,
) -> pd.DataFrame:
    """Build bar-level live IC/HR metrics for the latest prediction rows."""
    if rows is None or rows.empty:
        return pd.DataFrame()
    asof = _as_utc_timestamp(asof_ts)
    maturity = _as_utc_timestamp(label_maturity_cutoff_ts, default=asof)
    if attach_market_outcomes:
        rows = attach_recent_bar_forward_returns(
            rows,
            asof_ts=asof,
            market_data_root=market_data_root,
            horizon_hours=int(outcome_horizon_hours),
        )
    ts = _recent_bar_timestamp_series(rows)
    if ts is None:
        return pd.DataFrame()
    df = rows.copy()
    valid_ts = ts.notna()
    df = df.loc[valid_ts.to_numpy(dtype=bool)].copy()
    if df.empty:
        return pd.DataFrame()
    df["_recent_bar_ts"] = ts.loc[df.index]
    df = df.loc[df["_recent_bar_ts"] <= asof].copy()
    if df.empty:
        return pd.DataFrame()
    windows = dict(windows_hours or RECENT_BAR_WINDOWS_HOURS)
    out: list[dict[str, Any]] = []
    for window, hours in windows.items():
        start = asof - pd.Timedelta(hours=int(hours))
        subset = df.loc[df["_recent_bar_ts"] >= start].copy()
        out.append(
            _recent_bar_metric_row(
                subset,
                asof=asof,
                maturity=maturity,
                window=str(window),
                window_hours=int(hours),
                scope="global",
                scope_value="all",
                model_run_id=model_run_id,
                policy_run_id=policy_run_id,
                min_count=int(min_count),
            )
        )
        for scope_col, scope_name in (("strategy_id", "strategy_id"), ("side", "side")):
            if scope_col not in subset.columns or subset.empty:
                continue
            for value, group in subset.groupby(scope_col, dropna=False, sort=True):
                if pd.isna(value):
                    continue
                out.append(
                    _recent_bar_metric_row(
                        group,
                        asof=asof,
                        maturity=maturity,
                        window=str(window),
                        window_hours=int(hours),
                        scope=scope_name,
                        scope_value=str(value),
                        model_run_id=model_run_id,
                        policy_run_id=policy_run_id,
                        min_count=int(min_count),
                    )
                )
    result = pd.DataFrame(out)
    if result.empty:
        return result
    for col in ("asof_ts", "label_maturity_cutoff_ts", "window_start_ts", "window_end_ts", "last_bar_ts"):
        result[col] = pd.to_datetime(result[col], utc=True, errors="coerce")
    return result


def build_metric_panel(
    rows: pd.DataFrame,
    *,
    freq: str = "D",
    asof_ts: Any = None,
    label_maturity_cutoff_ts: Any = None,
    model_run_id: str | None = None,
    policy_run_id: str | None = None,
    source: str = "unknown",
) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    df = rows.copy()
    ts_col = "timestamp" if "timestamp" in df.columns else "signal_bar_ts" if "signal_bar_ts" in df.columns else None
    if ts_col is None:
        return pd.DataFrame()
    ts = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
    df = df.loc[ts.notna()].copy()
    if df.empty:
        return pd.DataFrame()
    df["_drift_ts"] = ts.loc[df.index]
    freq_norm = freq.upper()
    if freq_norm == "D":
        df["_drift_period_start"] = df["_drift_ts"].dt.floor("D")
    else:
        ts_utc = df["_drift_ts"].dt.tz_convert("UTC")
        df["_drift_period_start"] = ts_utc.dt.floor("D") - pd.to_timedelta(ts_utc.dt.dayofweek, unit="D")
    asof = _as_utc_timestamp(asof_ts)
    maturity = _as_utc_timestamp(label_maturity_cutoff_ts, default=asof)
    out: list[dict[str, Any]] = []
    symbol_col = "symbol" if "symbol" in df.columns else None
    if symbol_col is None:
        df["symbol"] = "all"
        symbol_col = "symbol"
    group_cols = ["_drift_period_start", symbol_col]
    for (period_start, symbol), group in df.groupby(group_cols, dropna=False, sort=True):
        period_ts = pd.Timestamp(period_start)
        if period_ts.tzinfo is None:
            period_ts = period_ts.tz_localize("UTC")
        period_end = period_ts + (pd.Timedelta(days=1) if freq_norm == "D" else pd.Timedelta(days=7))
        vol_source = _numeric_series(group, ("rv_24h", "atr_pct", "asset_atr_30d", "ticker_spread_bps"))
        meta = {
            "schema_version": DRIFT_SCHEMA_VERSION,
            "source": source,
            "asof_ts": asof,
            "label_maturity_cutoff_ts": maturity,
            "model_run_id": model_run_id or "",
            "policy_run_id": policy_run_id or "",
            "period_freq": freq,
            "period_start_ts": period_ts,
            "period_end_ts": period_end,
            "symbol": str(symbol),
            "asset_class": str(group.get("asset_class", pd.Series([_safe_asset_class(symbol)])).dropna().astype(str).iloc[0] if "asset_class" in group.columns and group["asset_class"].notna().any() else _safe_asset_class(symbol)),
            "hour_of_day": int(pd.Series(pd.DatetimeIndex(group["_drift_ts"]).hour).mode().iloc[0]) if len(group) else -1,
            "day_of_week": int(pd.Series(pd.DatetimeIndex(group["_drift_ts"]).dayofweek).mode().iloc[0]) if len(group) else -1,
            "volatility_regime": _volatility_regime(vol_source),
            "strategy_id": str(group["strategy_id"].dropna().astype(str).iloc[0]) if "strategy_id" in group.columns and group["strategy_id"].notna().any() else "",
        }
        out.extend(_metric_rows_for_group(group, meta))
    panel = pd.DataFrame(out)
    if panel.empty:
        return panel
    for col in ("asof_ts", "label_maturity_cutoff_ts", "period_start_ts", "period_end_ts"):
        panel[col] = pd.to_datetime(panel[col], utc=True, errors="coerce")
    return panel


def _mad(values: pd.Series) -> float:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if not len(vals):
        return float("nan")
    med = float(vals.median())
    return float(np.median(np.abs(vals.to_numpy(dtype=float) - med)))


def _stats_for_group(values: pd.Series, baseline_min_count: int) -> dict[str, Any]:
    vals = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    count = int(len(vals))
    if count == 0:
        return {}
    mad = _mad(vals)
    std = float(vals.std(ddof=0)) if count else float("nan")
    reliable = count >= int(baseline_min_count) and (
        (np.isfinite(mad) and mad > 1e-12) or (np.isfinite(std) and std > 1e-12)
    )
    return {
        "baseline_count": count,
        "mean": float(vals.mean()),
        "median": float(vals.median()),
        "mad": mad,
        "std": std,
        "p10": float(vals.quantile(0.10)),
        "p25": float(vals.quantile(0.25)),
        "p50": float(vals.quantile(0.50)),
        "p75": float(vals.quantile(0.75)),
        "p90": float(vals.quantile(0.90)),
        "min": float(vals.min()),
        "max": float(vals.max()),
        "baseline_reliable": bool(reliable),
    }


def build_metric_baselines(metric_panel: pd.DataFrame) -> pd.DataFrame:
    if metric_panel is None or metric_panel.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    cohort_defs = [
        ("global", lambda r: "global"),
        ("asset", lambda r: str(r.get("symbol", ""))),
        ("asset_class", lambda r: str(r.get("asset_class", ""))),
        ("hour_of_day", lambda r: str(r.get("hour_of_day", ""))),
        ("day_of_week", lambda r: str(r.get("day_of_week", ""))),
        ("volatility_regime", lambda r: str(r.get("volatility_regime", ""))),
        ("asset_hour", lambda r: f"{r.get('symbol','')}|{r.get('hour_of_day','')}"),
        ("asset_class_hour", lambda r: f"{r.get('asset_class','')}|{r.get('hour_of_day','')}"),
        ("asset_class_volatility", lambda r: f"{r.get('asset_class','')}|{r.get('volatility_regime','')}"),
    ]
    work = metric_panel.copy()
    for cohort_name, fn in cohort_defs:
        work["_cohort_name"] = cohort_name
        work["_cohort_value"] = work.apply(fn, axis=1)
        for keys, group in work.groupby(["family", "metric_name", "_cohort_name", "_cohort_value"], dropna=False):
            family, metric, cohort, value = keys
            spec = DRIFT_METRIC_REGISTRY.get(str(metric))
            stats = _stats_for_group(group["metric_value"], spec.baseline_min_count if spec else 12)
            if not stats:
                continue
            rows.append(
                {
                    "schema_version": DRIFT_SCHEMA_VERSION,
                    "family": family,
                    "metric_name": metric,
                    "cohort_name": cohort,
                    "cohort_value": value,
                    "baseline_min_count": int(spec.baseline_min_count if spec else 12),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def _approx_percentile(value: float, baseline: Mapping[str, Any]) -> float:
    if not np.isfinite(value):
        return float("nan")
    points = [
        (float(baseline.get("min", value)), 0.0),
        (float(baseline.get("p10", value)), 0.10),
        (float(baseline.get("p25", value)), 0.25),
        (float(baseline.get("p50", value)), 0.50),
        (float(baseline.get("p75", value)), 0.75),
        (float(baseline.get("p90", value)), 0.90),
        (float(baseline.get("max", value)), 1.0),
    ]
    points = [(x, p) for x, p in points if np.isfinite(x)]
    points = sorted(points, key=lambda item: item[0])
    if not points:
        return float("nan")
    if value <= points[0][0]:
        return 0.0
    if value >= points[-1][0]:
        return 1.0
    for (x0, p0), (x1, p1) in zip(points[:-1], points[1:]):
        if x0 <= value <= x1:
            if abs(x1 - x0) <= 1e-12:
                return float((p0 + p1) / 2.0)
            return float(p0 + (value - x0) / (x1 - x0) * (p1 - p0))
    return 0.5


def _severity_from_percentile(percentile: float, direction: str) -> float:
    if not np.isfinite(percentile):
        return float("nan")
    p = float(np.clip(percentile, 0.0, 1.0))
    if direction == "high":
        return p
    if direction == "low":
        return 1.0 - p
    return float(min(1.0, 2.0 * abs(p - 0.5)))


def _baseline_lookup(baselines: pd.DataFrame) -> dict[tuple[str, str, str, str], Mapping[str, Any]]:
    if baselines is None or baselines.empty:
        return {}
    out = {}
    for row in baselines.to_dict("records"):
        out[(str(row["family"]), str(row["metric_name"]), str(row["cohort_name"]), str(row["cohort_value"]))] = row
    return out


def _cohort_candidates(row: Mapping[str, Any]) -> list[tuple[str, str]]:
    return [
        ("asset_hour", f"{row.get('symbol','')}|{row.get('hour_of_day','')}"),
        ("asset", str(row.get("symbol", ""))),
        ("asset_class_hour", f"{row.get('asset_class','')}|{row.get('hour_of_day','')}"),
        ("asset_class", str(row.get("asset_class", ""))),
        ("asset_class_volatility", f"{row.get('asset_class','')}|{row.get('volatility_regime','')}"),
        ("volatility_regime", str(row.get("volatility_regime", ""))),
        ("hour_of_day", str(row.get("hour_of_day", ""))),
        ("global", "global"),
    ]


def score_metric_panel(metric_panel: pd.DataFrame, baselines: pd.DataFrame | None) -> pd.DataFrame:
    if metric_panel is None or metric_panel.empty:
        return pd.DataFrame()
    lookup = _baseline_lookup(baselines if baselines is not None else pd.DataFrame())
    rows: list[dict[str, Any]] = []
    for row in metric_panel.to_dict("records"):
        spec = DRIFT_METRIC_REGISTRY.get(str(row.get("metric_name")))
        selected = None
        selected_name = ""
        selected_value = ""
        for cohort_name, cohort_value in _cohort_candidates(row):
            candidate = lookup.get((str(row.get("family")), str(row.get("metric_name")), cohort_name, cohort_value))
            if candidate and bool(candidate.get("baseline_reliable", False)):
                selected = candidate
                selected_name = cohort_name
                selected_value = cohort_value
                break
        value = float(row.get("metric_value", np.nan))
        pct = _approx_percentile(value, selected) if selected else float("nan")
        direction = spec.severity_direction if spec else str(row.get("severity_direction", "high"))
        med = float(selected.get("median", np.nan)) if selected else float("nan")
        mad = float(selected.get("mad", np.nan)) if selected else float("nan")
        std = float(selected.get("std", np.nan)) if selected else float("nan")
        robust_z = (value - med) / max(1.4826 * mad, 1e-12) if selected and np.isfinite(mad) else float("nan")
        z = (value - med) / max(std, 1e-12) if selected and np.isfinite(std) else float("nan")
        rows.append(
            {
                **row,
                "baseline_cohort_name": selected_name,
                "baseline_cohort_value": selected_value,
                "baseline_reliable": bool(selected is not None),
                "baseline_count": int(selected.get("baseline_count", 0)) if selected else 0,
                "baseline_percentile": pct,
                "severity_percentile": _severity_from_percentile(pct, direction),
                "robust_z": robust_z,
                "z_score": z,
            }
        )
    return pd.DataFrame(rows)


def build_daily_cross_section(scored_panel: pd.DataFrame) -> pd.DataFrame:
    if scored_panel is None or scored_panel.empty:
        return pd.DataFrame()
    rows: list[pd.DataFrame] = []
    work = scored_panel.copy()
    for _, group in work.groupby(["period_start_ts", "family", "metric_name"], dropna=False):
        vals = pd.to_numeric(group["metric_value"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().sum() == 0:
            continue
        med = float(vals.median())
        p10 = float(vals.quantile(0.10))
        p90 = float(vals.quantile(0.90))
        std = float(vals.std(ddof=0))
        rank = vals.rank(pct=True)
        z = (vals - med) / max(std, 1e-12)
        part = group.copy()
        part["cross_sectional_median"] = med
        part["cross_sectional_p10"] = p10
        part["cross_sectional_p90"] = p90
        part["cross_sectional_rank"] = rank
        part["cross_sectional_z_score"] = z
        part["distance_from_own_norm"] = part["robust_z"]
        part["distance_from_peer_norm"] = z
        rows.append(part)
    return pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()


def build_family_scores(
    scored_panel: pd.DataFrame,
    *,
    window_label: str,
    tier: int | None = 1,
) -> pd.DataFrame:
    if scored_panel is None or scored_panel.empty:
        return pd.DataFrame()
    work = scored_panel.copy()
    if tier is not None and "tier" in work.columns:
        work = work.loc[pd.to_numeric(work["tier"], errors="coerce") == int(tier)].copy()
    if work.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    specs = _drift_specs_for_tier(tier)
    expected_by_family = {family: sum(1 for spec in specs if spec.family == family) for family in DRIFT_FAMILIES}
    for (symbol, family), group in work.groupby(["symbol", "family"], dropna=False):
        sev = pd.to_numeric(group["severity_percentile"], errors="coerce")
        reliable = group["baseline_reliable"].fillna(False).astype(bool)
        available = group["metric_available"].fillna(False).astype(bool)
        requires_label = group["requires_matured_label"].fillna(False).astype(bool)
        label_available = available | ~requires_label
        expected = max(1, int(expected_by_family.get(str(family), len(group))))
        score = float(sev.dropna().mean()) if sev.notna().any() else float("nan")
        rows.append(
            {
                "symbol": str(symbol),
                "family": str(family),
                "window": window_label,
                "tier": int(tier) if tier is not None else 0,
                "family_score": score,
                "family_metric_coverage_ratio": float(min(1.0, available.sum() / expected)),
                "family_asset_coverage_ratio": 1.0,
                "family_matured_label_coverage_ratio": float(label_available.mean()) if len(label_available) else 0.0,
                "family_reliable_baseline_ratio": float(reliable.mean()) if len(reliable) else 0.0,
                "metric_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def build_regime_drift_features(scored_by_window: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    symbols: set[str] = set()
    for frame in scored_by_window.values():
        if frame is not None and not frame.empty and "symbol" in frame.columns:
            symbols.update(str(s) for s in frame["symbol"].dropna().astype(str))
    if not symbols:
        return pd.DataFrame()
    records: dict[str, dict[str, Any]] = {symbol: {"symbol": symbol} for symbol in sorted(symbols)}

    def _set(symbol: Any, column: str, value: Any) -> None:
        try:
            numeric = float(value)
        except Exception:
            numeric = float("nan")
        if np.isfinite(numeric):
            key = str(symbol)
            records.setdefault(key, {"symbol": key})[column] = numeric

    family_score_cache: dict[str, pd.DataFrame] = {}
    all_family_score_cache: dict[str, pd.DataFrame] = {}
    for window, scored in scored_by_window.items():
        if scored is None or scored.empty:
            continue
        if window in REGIME_ADAPTOR_TIER1_WINDOWS:
            tier1_scored = scored.loc[pd.to_numeric(scored.get("tier"), errors="coerce") == 1] if "tier" in scored.columns else scored
            for (symbol, family, metric), group in tier1_scored.groupby(["symbol", "family", "metric_name"], dropna=False):
                value = pd.to_numeric(group["severity_percentile"], errors="coerce").dropna()
                if len(value):
                    _set(symbol, f"drift_{family}_{metric}_{window}", value.mean())
        if window in REGIME_ADAPTOR_EXPANDED_WINDOWS:
            expanded_scored = scored.loc[pd.to_numeric(scored.get("tier"), errors="coerce") != 1] if "tier" in scored.columns else scored.iloc[0:0]
            for (symbol, family, metric), group in expanded_scored.groupby(["symbol", "family", "metric_name"], dropna=False):
                value = pd.to_numeric(group["severity_percentile"], errors="coerce").dropna()
                if len(value):
                    _set(symbol, f"drift_{family}_{metric}_{window}", value.mean())
        fam = build_family_scores(scored, window_label=window, tier=1)
        fam_all = build_family_scores(scored, window_label=window, tier=None)
        family_score_cache[window] = fam
        all_family_score_cache[window] = fam_all
        if window in REGIME_ADAPTOR_TIER1_WINDOWS and not fam.empty:
            for _, row in fam.iterrows():
                symbol = str(row["symbol"])
                family = str(row["family"])
                _set(symbol, f"drift_{family}_score_{window}", row["family_score"])
                _set(symbol, f"drift_{family}_coverage_ratio_{window}", row["family_metric_coverage_ratio"])
                _set(symbol, f"drift_{family}_reliable_baseline_ratio_{window}", row["family_reliable_baseline_ratio"])
                _set(symbol, f"drift_{family}_matured_label_coverage_ratio_{window}", row["family_matured_label_coverage_ratio"])
        if window in REGIME_ADAPTOR_EXPANDED_WINDOWS and not fam_all.empty:
            for _, row in fam_all.iterrows():
                symbol = str(row["symbol"])
                family = str(row["family"])
                _set(symbol, f"drift_{family}_all_score_{window}", row["family_score"])
                _set(symbol, f"drift_{family}_all_coverage_ratio_{window}", row["family_metric_coverage_ratio"])
                _set(symbol, f"drift_{family}_all_reliable_baseline_ratio_{window}", row["family_reliable_baseline_ratio"])
                _set(symbol, f"drift_{family}_all_matured_label_coverage_ratio_{window}", row["family_matured_label_coverage_ratio"])
    if "7d" in family_score_cache and "3d" in family_score_cache:
        f7 = family_score_cache["7d"].set_index(["symbol", "family"])
        f3 = family_score_cache["3d"].set_index(["symbol", "family"])
        idx = f7.index.intersection(f3.index)
        for symbol, family in idx:
            _set(
                symbol,
                f"drift_{family}_score_7d_minus_3d",
                f7.loc[(symbol, family), "family_score"] - f3.loc[(symbol, family), "family_score"],
            )
    if "7d" in all_family_score_cache and "1d" in all_family_score_cache:
        f7_all = all_family_score_cache["7d"].set_index(["symbol", "family"])
        f1_all = all_family_score_cache["1d"].set_index(["symbol", "family"])
        idx_all = f7_all.index.intersection(f1_all.index)
        for symbol, family in idx_all:
            _set(
                symbol,
                f"drift_{family}_all_score_7d_minus_1d",
                f7_all.loc[(symbol, family), "family_score"] - f1_all.loc[(symbol, family), "family_score"],
            )
    feature_names = drift_regime_feature_names()
    for record in records.values():
        for name in feature_names:
            record.setdefault(name, np.nan)
    return pd.DataFrame([records[symbol] for symbol in sorted(records)], columns=["symbol", *feature_names])


def build_correlation_reports(metric_panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if metric_panel is None or metric_panel.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = metric_panel.copy()
    fam = (
        work.groupby(["period_start_ts", "symbol", "family"], dropna=False)["metric_value"]
        .mean()
        .reset_index()
    )
    pivot = fam.pivot_table(index=["period_start_ts", "symbol"], columns="family", values="metric_value", aggfunc="mean").reset_index()
    corrs: list[dict[str, Any]] = []
    cond: list[dict[str, Any]] = []
    targets = [c for c in ("performance_drift", "residual_drift", "execution_drift", "target_drift") if c in pivot.columns]
    drivers = [c for c in DRIFT_FAMILIES if c in pivot.columns and c not in targets]
    pivot = pivot.sort_values(["symbol", "period_start_ts"])
    for driver in drivers:
        for target in targets:
            for lag in (0, 1, 3, 7, 14):
                shifted = pivot.groupby("symbol", sort=False)[driver].shift(lag)
                corr = _corr(shifted, pivot[target], method="spearman")
                corrs.append({"driver_family": driver, "target_family": target, "lag_periods": lag, "spearman_corr": corr, "n": int(pd.DataFrame({"a": shifted, "b": pivot[target]}).dropna().shape[0])})
            vals = pd.to_numeric(pivot[driver], errors="coerce")
            target_vals = pd.to_numeric(pivot[target], errors="coerce")
            degradation = target_vals >= target_vals.quantile(0.80) if target != "performance_drift" else target_vals <= target_vals.quantile(0.20)
            for q in (0.80, 0.90, 0.95):
                thr = vals.quantile(q)
                mask = vals >= thr
                cond.append(
                    {
                        "driver_family": driver,
                        "target_family": target,
                        "driver_threshold_quantile": q,
                        "driver_threshold_value": float(thr) if np.isfinite(thr) else np.nan,
                        "conditional_degradation_probability": float(degradation[mask].mean()) if mask.any() else np.nan,
                        "n": int(mask.sum()),
                    }
                )
    return pd.DataFrame(corrs), pd.DataFrame(cond)


def write_policy_drift_benchmarks(
    candidate_table: pd.DataFrame,
    *,
    output_dir: str | Path,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prov = dict(provenance or {})
    asof_ts = prov.get("asof_ts") or pd.Timestamp.now(tz="UTC")
    label_cutoff = prov.get("label_maturity_cutoff_ts") or asof_ts
    daily = build_metric_panel(
        candidate_table,
        freq="D",
        asof_ts=asof_ts,
        label_maturity_cutoff_ts=label_cutoff,
        model_run_id=str(prov.get("model_run_id", "")),
        policy_run_id=str(prov.get("policy_run_id", "")),
        source="policy_candidate_table",
    )
    weekly = build_metric_panel(
        candidate_table,
        freq="W",
        asof_ts=asof_ts,
        label_maturity_cutoff_ts=label_cutoff,
        model_run_id=str(prov.get("model_run_id", "")),
        policy_run_id=str(prov.get("policy_run_id", "")),
        source="policy_candidate_table",
    )
    baselines = build_metric_baselines(weekly if not weekly.empty else daily)
    scored_daily = score_metric_panel(daily, baselines)
    cross = build_daily_cross_section(scored_daily)
    corr, cond = build_correlation_reports(weekly if not weekly.empty else daily)
    registry_path = output / "metric_registry.json"
    schema_path = output / "schema.json"
    daily_path = output / "daily_cross_section.parquet"
    weekly_path = output / "weekly_metric_observations.parquet"
    baseline_path = output / "metric_baselines.parquet"
    corr_path = output / "correlation_report.parquet"
    cond_path = output / "conditional_degradation_report.parquet"
    registry_path.write_text(json.dumps([asdict(spec) for spec in ALL_DRIFT_METRICS], indent=2), encoding="utf-8")
    schema_path.write_text(
        json.dumps(
            _json_safe(
                {
                    "schema_version": DRIFT_SCHEMA_VERSION,
                    "asof_ts": asof_ts,
                    "label_maturity_cutoff_ts": label_cutoff,
                    "tier1_metric_count": len(TIER1_DRIFT_METRICS),
                    "expanded_metric_count": len(EXPANDED_DRIFT_METRICS),
                    "all_metric_count": len(ALL_DRIFT_METRICS),
                    **prov,
                }
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    if not daily.empty:
        daily.to_parquet(output / "daily_metric_observations.parquet", index=False)
    if not weekly.empty:
        weekly.to_parquet(weekly_path, index=False)
    if not baselines.empty:
        baselines.to_parquet(baseline_path, index=False)
        (output / "metric_baselines.json").write_text(json.dumps(_json_safe(baselines.head(200).to_dict("records")), indent=2), encoding="utf-8")
    if not cross.empty:
        cross.to_parquet(daily_path, index=False)
    if not corr.empty:
        corr.to_parquet(corr_path, index=False)
    if not cond.empty:
        cond.to_parquet(cond_path, index=False)
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "output_dir": str(output),
        "daily_metric_rows": int(len(daily)),
        "weekly_metric_rows": int(len(weekly)),
        "baseline_rows": int(len(baselines)),
        "cross_section_rows": int(len(cross)),
        "correlation_rows": int(len(corr)),
        "conditional_degradation_rows": int(len(cond)),
        "tier1_metric_count": int(len(TIER1_DRIFT_METRICS)),
        "expanded_metric_count": int(len(EXPANDED_DRIFT_METRICS)),
        "all_metric_count": int(len(ALL_DRIFT_METRICS)),
    }


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return raw if isinstance(raw, dict) else {}


def _normalise_lgbm_reference_sample(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()
    if frame.empty:
        return pd.DataFrame()
    manifest = _read_json_if_exists(path.parent / "manifest.json")
    out = frame.copy()
    if "symbol" not in out.columns and "asset" in out.columns:
        out["symbol"] = out["asset"]
    if "symbol" not in out.columns:
        out["symbol"] = "reference"
    if "timestamp" not in out.columns:
        return pd.DataFrame()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True, errors="coerce")
    out = out.loc[out["timestamp"].notna()].copy()
    if out.empty:
        return out
    strategy_id = str(manifest.get("strategy_id") or path.parent.name)
    model_scope = "meta" if "/meta/" in path.as_posix() else "base"
    score = _numeric_series(out, ("score", "prob", "lgbm_prob", "raw_prediction_score"))
    if score is not None:
        if "base_pred" not in out.columns:
            out["base_pred"] = score
        if "meta_pred" not in out.columns:
            out["meta_pred"] = score
        if "calibrated_score" not in out.columns:
            out["calibrated_score"] = score
    rank = _numeric_series(out, ("rank_pct", "auction_rank_pct", "normalized_rank_score", "policy_rank_pct"))
    if rank is not None:
        out["auction_rank_pct"] = rank
        out["normalized_rank_score"] = rank
    ret = _numeric_series(out, ("return", "net_return", "realized_net_return", "gross_return"))
    if ret is not None:
        out["net_return"] = ret
        out["gross_return"] = ret
    target = _numeric_series(out, ("target", "y_bin", "hard_label"))
    if target is not None:
        out["hard_label"] = target
    out["strategy_id"] = strategy_id
    out["reference_model_scope"] = model_scope
    out["asset_class"] = out.get("asset_class", "crypto_perp")
    out["was_traded"] = True
    out["label_available_ts"] = out["timestamp"]
    return out


def _reference_sample_metric_panel(rows: pd.DataFrame, *, model_run_id: str, policy_run_id: str) -> pd.DataFrame:
    if rows is None or rows.empty:
        return pd.DataFrame()
    ts = pd.to_datetime(rows.get("timestamp"), utc=True, errors="coerce")
    work = rows.loc[ts.notna()].copy()
    if work.empty:
        return pd.DataFrame()
    work["_drift_ts"] = ts.loc[work.index]
    symbol = work["symbol"].astype(str) if "symbol" in work.columns else pd.Series("reference", index=work.index)
    asset_class = work["asset_class"].astype(str) if "asset_class" in work.columns else pd.Series("crypto_perp", index=work.index)
    strategy = work["strategy_id"].astype(str) if "strategy_id" in work.columns else pd.Series("", index=work.index)
    base_meta = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "source": "lgbm_reference_samples",
        "asof_ts": pd.Timestamp.now(tz="UTC"),
        "label_maturity_cutoff_ts": pd.Timestamp.now(tz="UTC"),
        "model_run_id": model_run_id,
        "policy_run_id": policy_run_id,
        "period_freq": "reference_sample",
        "period_start_ts": work["_drift_ts"].dt.floor("D"),
        "period_end_ts": work["_drift_ts"].dt.floor("D") + pd.Timedelta(days=1),
        "symbol": symbol,
        "asset_class": asset_class,
        "hour_of_day": work["_drift_ts"].dt.hour.astype(int),
        "day_of_week": work["_drift_ts"].dt.dayofweek.astype(int),
        "volatility_regime": "unknown",
        "strategy_id": strategy,
        "metric_count": 1,
        "metric_available": True,
    }
    metric_sources: list[tuple[str, pd.Series]] = []

    def add(metric_name: str, values: pd.Series | np.ndarray | None) -> None:
        spec = DRIFT_METRIC_REGISTRY.get(metric_name)
        if spec is None or values is None:
            return
        vals = pd.to_numeric(pd.Series(values, index=work.index), errors="coerce").replace([np.inf, -np.inf], np.nan)
        if vals.notna().any():
            metric_sources.append((metric_name, vals))

    add("feature_psi_mean", _numeric_series(work, ("feature_drift_psi_core", "feature_drift_psi_core_80")))
    add("feature_psi_p95", _numeric_series(work, ("feature_drift_psi_core", "feature_drift_psi_core_80")))
    add("feature_psi_max", _numeric_series(work, ("feature_drift_psi_core", "feature_drift_psi_core_80")))
    add("feature_cov_shift", _numeric_series(work, ("feature_drift_cov_shift", "frobenius_corr_shift")))
    sim = _numeric_series(work, ("regime_centroid_similarity_train", "regime_centroid_similarity_train_window_mean"))
    add("regime_centroid_similarity", sim)
    if sim is not None:
        add("feature_embedding_distance", 1.0 - pd.to_numeric(sim, errors="coerce"))
    add("prob_uncertainty_mean", _numeric_series(work, ("prob_uncertainty",)))
    add("prob_uncertainty_p90", _numeric_series(work, ("prob_uncertainty",)))
    add("prob_uncertainty_max", _numeric_series(work, ("prob_uncertainty",)))
    add("rare_leaf_share", _numeric_series(work, ("rare_leaf_fraction", "rare_leaf_low_support_score")))
    add("rare_leaf_p90", _numeric_series(work, ("rare_leaf_fraction", "rare_leaf_low_support_score")))
    add("leaf_count_p10", _numeric_series(work, ("leaf_count_p10",)))
    add("leaf_count_min", _numeric_series(work, ("leaf_count_min",)))
    add("contrib_entropy_mean", _numeric_series(work, ("contrib_entropy",)))
    add("contrib_top1_abs_share_mean", _numeric_series(work, ("contrib_top1_abs_share",)))
    add("contrib_top3_abs_share_mean", _numeric_series(work, ("contrib_top3_abs_share",)))
    score = _numeric_series(work, ("score", "base_pred", "meta_pred", "calibrated_score"))
    add("base_prediction_mean", score)
    add("meta_prediction_mean", score)
    rank = _numeric_series(work, ("rank_pct", "auction_rank_pct", "normalized_rank_score"))
    add("threshold_pass_share", (pd.to_numeric(rank, errors="coerce") >= 0.80).astype(float) if rank is not None else None)
    add("threshold_pass_share_top10", (pd.to_numeric(rank, errors="coerce") >= 0.90).astype(float) if rank is not None else None)
    add("threshold_pass_share_top20", (pd.to_numeric(rank, errors="coerce") >= 0.80).astype(float) if rank is not None else None)
    add("threshold_pass_share_top30", (pd.to_numeric(rank, errors="coerce") >= 0.70).astype(float) if rank is not None else None)

    parts: list[pd.DataFrame] = []
    for metric_name, vals in metric_sources:
        spec = DRIFT_METRIC_REGISTRY[metric_name]
        part = pd.DataFrame(base_meta)
        part["family"] = spec.family
        part["metric_name"] = spec.metric_name
        part["tier"] = int(spec.tier)
        part["metric_value"] = vals.to_numpy(dtype=float)
        part["severity_direction"] = spec.severity_direction
        part["requires_matured_label"] = bool(spec.requires_matured_label)
        part["requires_trade"] = bool(spec.requires_trade)
        part["metric_min_count"] = int(spec.min_count)
        part["baseline_min_count"] = int(spec.baseline_min_count)
        part = part.loc[np.isfinite(part["metric_value"].to_numpy(dtype=float))]
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True, sort=False) if parts else pd.DataFrame()


def build_reference_sample_metric_baselines(reference_panel: pd.DataFrame) -> pd.DataFrame:
    """Build baseline stats for reference sample observations without row-wise cohort apply."""
    if reference_panel is None or reference_panel.empty:
        return pd.DataFrame()
    panel = reference_panel.copy()
    if "asset_class" not in panel.columns:
        panel["asset_class"] = "crypto_perp"
    if "hour_of_day" not in panel.columns:
        panel["hour_of_day"] = -1
    if "day_of_week" not in panel.columns:
        panel["day_of_week"] = -1
    if "volatility_regime" not in panel.columns:
        panel["volatility_regime"] = "unknown"
    cohort_values = {
        "global": pd.Series("global", index=panel.index, dtype=object),
        "asset": panel["symbol"].astype(str),
        "asset_class": panel["asset_class"].astype(str),
        "hour_of_day": panel["hour_of_day"].astype(str),
        "day_of_week": panel["day_of_week"].astype(str),
        "volatility_regime": panel["volatility_regime"].astype(str),
        "asset_hour": panel["symbol"].astype(str) + "|" + panel["hour_of_day"].astype(str),
        "asset_class_hour": panel["asset_class"].astype(str) + "|" + panel["hour_of_day"].astype(str),
        "asset_class_volatility": panel["asset_class"].astype(str) + "|" + panel["volatility_regime"].astype(str),
    }
    rows: list[dict[str, Any]] = []
    keys = ["family", "metric_name", "_cohort_name", "_cohort_value"]
    base_cols = ["family", "metric_name", "metric_value"]
    for cohort_name, values in cohort_values.items():
        work = panel[base_cols].copy()
        work["_cohort_name"] = cohort_name
        work["_cohort_value"] = values.to_numpy(dtype=object)
        for (family, metric, cohort, value), group in work.groupby(keys, dropna=False, sort=False):
            spec = DRIFT_METRIC_REGISTRY.get(str(metric))
            stats = _stats_for_group(group["metric_value"], spec.baseline_min_count if spec else 12)
            if not stats:
                continue
            rows.append(
                {
                    "schema_version": DRIFT_SCHEMA_VERSION,
                    "family": family,
                    "metric_name": metric,
                    "cohort_name": cohort,
                    "cohort_value": value,
                    "baseline_min_count": int(spec.baseline_min_count if spec else 12),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def write_lgbm_reference_drift_benchmarks(
    *,
    artifact_root: str | Path,
    output_dir: str | Path | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build lightweight drift baselines from saved LGBM reference samples."""
    root = Path(artifact_root)
    output = Path(output_dir) if output_dir is not None else root / "drift_benchmarks"
    output.mkdir(parents=True, exist_ok=True)
    samples = sorted((root / "lgbm_reference").glob("*/*/lgbm_reference_sample.parquet"))
    max_rows_per_file = int(os.getenv("EPM_LGBM_REFERENCE_BENCHMARK_MAX_ROWS_PER_FILE", "10000") or "10000")
    frames: list[pd.DataFrame] = []
    sample_info: list[dict[str, Any]] = []
    for path in samples:
        frame = _normalise_lgbm_reference_sample(path)
        if frame.empty:
            continue
        original_rows = int(len(frame))
        if max_rows_per_file > 0 and len(frame) > max_rows_per_file:
            frame = frame.sort_values("timestamp")
            keep = np.unique(np.linspace(0, len(frame) - 1, max_rows_per_file).round().astype(int))
            frame = frame.iloc[keep].copy()
        frames.append(frame)
        ts = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
        sample_info.append(
            {
                "path": str(path),
                "rows": int(len(frame)),
                "original_rows": original_rows,
                "max_rows_per_file": int(max_rows_per_file),
                "min_ts": ts.min().isoformat() if ts.notna().any() else "",
                "max_ts": ts.max().isoformat() if ts.notna().any() else "",
            }
        )
    prov = dict(provenance or {})
    asof_ts = _as_utc_timestamp(prov.get("asof_ts"))
    registry_path = output / "metric_registry.json"
    schema_path = output / "schema.json"
    baseline_path = output / "metric_baselines.parquet"
    registry_path.write_text(json.dumps([asdict(spec) for spec in ALL_DRIFT_METRICS], indent=2), encoding="utf-8")
    if not frames:
        schema_path.write_text(
            json.dumps(
                _json_safe(
                    {
                        "schema_version": DRIFT_SCHEMA_VERSION,
                        "asof_ts": asof_ts,
                        "benchmark_source": "lgbm_reference_samples",
                        "artifact_root": str(root),
                        "sample_files": [],
                        "reference_sample_rows": 0,
                        "baseline_rows": 0,
                        **prov,
                    }
                ),
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "schema_version": DRIFT_SCHEMA_VERSION,
            "output_dir": str(output),
            "reference_sample_rows": 0,
            "baseline_rows": 0,
            "sample_files": 0,
        }
    rows = pd.concat(frames, ignore_index=True, sort=False)
    reference_panel = _reference_sample_metric_panel(
        rows,
        model_run_id=str(prov.get("model_run_id", root.name)),
        policy_run_id=str(prov.get("policy_run_id", root.name)),
    )
    baselines = build_reference_sample_metric_baselines(reference_panel)
    if not reference_panel.empty:
        reference_panel.to_parquet(output / "reference_sample_metric_observations.parquet", index=False)
    if not baselines.empty:
        baselines.to_parquet(baseline_path, index=False)
        (output / "metric_baselines.json").write_text(json.dumps(_json_safe(baselines.head(200).to_dict("records")), indent=2), encoding="utf-8")
    schema = {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "asof_ts": asof_ts,
        "benchmark_source": "lgbm_reference_samples",
        "artifact_root": str(root),
        "sample_files": sample_info,
        "reference_sample_rows": int(len(rows)),
        "reference_metric_rows": int(len(reference_panel)),
        "baseline_rows": int(len(baselines)),
        "tier1_metric_count": len(TIER1_DRIFT_METRICS),
        "expanded_metric_count": len(EXPANDED_DRIFT_METRICS),
        "all_metric_count": len(ALL_DRIFT_METRICS),
        **prov,
    }
    schema_path.write_text(json.dumps(_json_safe(schema), indent=2), encoding="utf-8")
    return {
        "schema_version": DRIFT_SCHEMA_VERSION,
        "output_dir": str(output),
        "reference_sample_rows": int(len(rows)),
        "reference_metric_rows": int(len(reference_panel)),
        "baseline_rows": int(len(baselines)),
        "sample_files": int(len(sample_info)),
    }


def _maybe_build_reference_benchmarks(path: str | Path | None, *, model_run_id: str | None = None, policy_run_id: str | None = None) -> None:
    if not path:
        return
    p = Path(path)
    benchmark_file = p / "metric_baselines.parquet" if p.is_dir() or p.name == "drift_benchmarks" else p
    if benchmark_file.exists():
        return
    output_dir = benchmark_file.parent
    artifact_root = output_dir.parent if output_dir.name == "drift_benchmarks" else output_dir
    if not (artifact_root / "lgbm_reference").exists():
        return
    write_lgbm_reference_drift_benchmarks(
        artifact_root=artifact_root,
        output_dir=output_dir,
        provenance={
            "model_run_id": model_run_id or artifact_root.name,
            "policy_run_id": policy_run_id or artifact_root.name,
        },
    )


def _load_benchmarks(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    p = Path(path)
    if p.is_dir():
        p = p / "metric_baselines.parquet"
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(p)
    except Exception:
        return pd.DataFrame()


def write_live_drift_recap(
    *,
    ledger_path: str | Path,
    output_root: str | Path,
    benchmark_dir: str | Path | None = None,
    asof_ts: Any = None,
    model_run_id: str | None = None,
    policy_run_id: str | None = None,
    schema_version: str = DRIFT_SCHEMA_VERSION,
    recent_bar_market_data_root: str | Path | None = None,
    recent_bar_outcome_horizon_hours: int = RECENT_BAR_DEFAULT_OUTCOME_HORIZON_HOURS,
) -> dict[str, Any]:
    ledger = Path(ledger_path)
    output = Path(output_root)
    if not ledger.exists():
        return {"enabled": True, "reason": "prediction_ledger_missing", "ledger_path": str(ledger)}
    try:
        rows = pd.read_parquet(ledger)
    except Exception as exc:
        return {"enabled": True, "reason": "prediction_ledger_unreadable", "error": str(exc), "ledger_path": str(ledger)}
    if rows.empty:
        return {"enabled": True, "reason": "prediction_ledger_empty", "ledger_path": str(ledger)}
    asof = _as_utc_timestamp(asof_ts)
    label_cutoff = asof
    _maybe_build_reference_benchmarks(benchmark_dir, model_run_id=model_run_id, policy_run_id=policy_run_id)
    baselines = _load_benchmarks(benchmark_dir)
    benchmark_path = Path(benchmark_dir) if benchmark_dir else None
    benchmark_metric_path = benchmark_path / "metric_baselines.parquet" if benchmark_path and benchmark_path.is_dir() else benchmark_path
    scored_by_window: dict[str, pd.DataFrame] = {}
    for window, days in DRIFT_WINDOWS_DAYS.items():
        ts = _timestamp_series(rows, ("timestamp", "signal_bar_ts", "signal_ts", "bar_ts", "entry_ts"))
        if ts is None:
            subset = rows.iloc[0:0].copy()
        else:
            subset = rows.loc[(ts.notna()) & (ts <= asof) & (ts >= asof - pd.Timedelta(days=days))].copy()
        panel = build_metric_panel(
            subset,
            freq="D",
            asof_ts=asof,
            label_maturity_cutoff_ts=label_cutoff,
            model_run_id=model_run_id,
            policy_run_id=policy_run_id,
            source="prediction_ledger",
        )
        scored_by_window[window] = score_metric_panel(panel, baselines)
    regime_features = build_regime_drift_features(scored_by_window)
    recent_bar_metrics = build_recent_bar_metrics(
        rows,
        asof_ts=asof,
        label_maturity_cutoff_ts=label_cutoff,
        model_run_id=model_run_id,
        policy_run_id=policy_run_id,
        market_data_root=recent_bar_market_data_root,
        outcome_horizon_hours=int(recent_bar_outcome_horizon_hours),
    )
    day_dir = output / "daily" / asof.strftime("%Y-%m-%d")
    day_dir.mkdir(parents=True, exist_ok=True)
    all_scored = pd.concat([df.assign(window=w) for w, df in scored_by_window.items() if df is not None and not df.empty], ignore_index=True, sort=False) if any(df is not None and not df.empty for df in scored_by_window.values()) else pd.DataFrame()
    recap = {
        "schema_version": schema_version,
        "asof_ts": asof.isoformat(),
        "label_maturity_cutoff_ts": label_cutoff.isoformat(),
        "model_run_id": model_run_id or "",
        "policy_run_id": policy_run_id or "",
        "ledger_path": str(ledger),
        "benchmark_dir": str(benchmark_dir or ""),
        "benchmark_metric_path": str(benchmark_metric_path or ""),
        "benchmark_available": bool(not baselines.empty),
        "benchmark_rows": int(len(baselines)),
        "ledger_rows": int(len(rows)),
        "scored_metric_rows": int(len(all_scored)),
        "regime_feature_rows": int(len(regime_features)),
        "recent_bar_metric_rows": int(len(recent_bar_metrics)),
        "recent_bar_status_counts": recent_bar_metrics["status"].value_counts(dropna=False).to_dict() if not recent_bar_metrics.empty and "status" in recent_bar_metrics.columns else {},
        "recent_bar_outcome_horizon_hours": int(recent_bar_outcome_horizon_hours),
        "recent_bar_market_data_root": str(_recent_bar_market_data_root(recent_bar_market_data_root)),
        "family_scores": {},
        "all_family_scores": {},
    }
    for window, scored in scored_by_window.items():
        fam = build_family_scores(scored, window_label=window, tier=1)
        if fam.empty:
            fam_all = build_family_scores(scored, window_label=window, tier=None)
        else:
            recap["family_scores"][window] = fam.groupby("family")[
                [
                    "family_score",
                    "family_metric_coverage_ratio",
                    "family_reliable_baseline_ratio",
                    "family_matured_label_coverage_ratio",
                ]
            ].mean().to_dict("index")
            fam_all = build_family_scores(scored, window_label=window, tier=None)
        if fam_all.empty:
            continue
        recap["all_family_scores"][window] = fam_all.groupby("family")[
            [
                "family_score",
                "family_metric_coverage_ratio",
                "family_reliable_baseline_ratio",
                "family_matured_label_coverage_ratio",
            ]
        ].mean().to_dict("index")
    recap_json = json.dumps(_json_safe(recap), indent=2)
    (day_dir / "drift_recap.json").write_text(recap_json, encoding="utf-8")
    if not all_scored.empty:
        all_scored.to_parquet(day_dir / "drift_recap.parquet", index=False)
    if not recent_bar_metrics.empty:
        recent_bar_metrics.to_parquet(day_dir / "recent_bar_metrics.parquet", index=False)
    latest_dir = output / "latest"
    latest_dir.mkdir(parents=True, exist_ok=True)
    if not regime_features.empty:
        regime_features.to_parquet(day_dir / "regime_drift_features.parquet", index=False)
        regime_features.to_parquet(latest_dir / "regime_drift_features.parquet", index=False)
    if not recent_bar_metrics.empty:
        recent_bar_metrics.to_parquet(latest_dir / "recent_bar_metrics.parquet", index=False)
    md_lines = [
        f"# Drift Recap {asof.strftime('%Y-%m-%d')}",
        "",
        f"- Ledger rows: {len(rows)}",
        f"- Scored metric rows: {len(all_scored)}",
        f"- Regime feature rows: {len(regime_features)}",
        f"- Recent-bar metric rows: {len(recent_bar_metrics)}",
        f"- Benchmark rows: {len(baselines)}",
        f"- Benchmark available: {bool(not baselines.empty)}",
    ]
    if not recent_bar_metrics.empty:
        def _fmt_metric(value: Any, digits: int = 4) -> str:
            try:
                val = float(value)
            except Exception:
                return "n/a"
            return f"{val:.{digits}f}" if math.isfinite(val) else "n/a"

        md_lines.extend(
            [
                "",
                "## Recent Bar IC/HR",
                "",
                "| window | scope | candidates | matured | rank_ic | hit_rate | top20_hit | mean_return | status |",
                "|---|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        recent_summary = recent_bar_metrics.loc[
            recent_bar_metrics["scope"].eq("global")
            | (
                recent_bar_metrics["scope"].eq("strategy_id")
                & recent_bar_metrics["window"].isin(("3h", "6h", "12h", "24h"))
            )
        ].copy()
        for _, rec in recent_summary.head(40).iterrows():
            md_lines.append(
                "| "
                f"{rec.get('window', '')} | "
                f"{rec.get('scope', '')}:{rec.get('scope_value', '')} | "
                f"{int(rec.get('candidate_rows', 0) or 0)} | "
                f"{int(rec.get('matured_rows', 0) or 0)} | "
                f"{_fmt_metric(rec.get('rank_ic'))} | "
                f"{_fmt_metric(rec.get('hit_rate'))} | "
                f"{_fmt_metric(rec.get('top20_hit_rate'))} | "
                f"{_fmt_metric(rec.get('mean_return'))} | "
                f"{rec.get('status', '')} |"
            )
    for window, families in recap["family_scores"].items():
        md_lines.append("")
        md_lines.append(f"## {window}")
        for family, values in families.items():
            md_lines.append(
                f"- {family}: score={values.get('family_score')} "
                f"coverage={values.get('family_metric_coverage_ratio')}"
            )
    recap_md = "\n".join(md_lines) + "\n"
    (day_dir / "drift_recap.md").write_text(recap_md, encoding="utf-8")
    (latest_dir / "drift_recap.json").write_text(recap_json, encoding="utf-8")
    (latest_dir / "drift_recap.md").write_text(recap_md, encoding="utf-8")
    return recap


def load_latest_drift_regime_features(
    *,
    live_data_root: str | Path | None = None,
    data_root: str | Path | None = None,
) -> pd.DataFrame:
    roots = [p for p in (live_data_root, data_root) if p]
    for root in roots:
        path = Path(root) / "live_state" / "drift_monitoring" / "latest" / "regime_drift_features.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                continue
    return pd.DataFrame()
